"""Real-Waveform Signal Conditioning, Multi-Lead ECG & Coupled ECG+PPG+RESP Feature Extraction, and Personalized PAFs."""

from collections import Counter
import math
from typing import Dict, List, Optional, Tuple

from .segmentation import ExtractedBeatFeatures, detrend_baseline
from ..tokenizers.raspro import (
    PAFThreshold,
    RASPROTokenSequence,
    RASPROTokenizer,
    RASPROWindowMotif,
    SYMBOL_SEVERITY,
)


def detect_real_r_peaks(ecg: List[float], fs: int = 125, refractory_sec: float = 0.28) -> List[int]:
    """Robust derivative-energy (Pan-Tompkins style) R-peak detector for real clinical ECGs."""
    n = len(ecg)
    if n < fs:
        return []

    clean = detrend_baseline(ecg, window_samples=max(25, int(0.60 * fs)))
    diff_sq = [0.0] * n
    for i in range(1, n - 1):
        d = clean[i + 1] - clean[i - 1]
        diff_sq[i] = d * d

    win = max(3, int(0.10 * fs))
    energy = [0.0] * n
    running = 0.0
    for i in range(n):
        running += diff_sq[i]
        if i >= win:
            running -= diff_sq[i - win]
        energy[i] = running / min(i + 1, win)

    sorted_en = sorted(energy)
    p92 = sorted_en[int(0.92 * (n - 1))]
    p50 = sorted_en[int(0.50 * (n - 1))]
    thresh = max(p50 * 2.5, 0.32 * p92)
    if thresh <= 1e-9:
        return []

    min_dist = int(refractory_sec * fs)
    candidate_peaks: List[int] = []
    search_rad = max(2, int(0.06 * fs))

    for i in range(search_rad, n - search_rad):
        if energy[i] >= thresh and energy[i] >= energy[i - 1] and energy[i] >= energy[i + 1]:
            lo = max(0, i - search_rad)
            hi = min(n, i + search_rad + 1)
            r_idx = max(range(lo, hi), key=lambda k: abs(clean[k]))
            if not candidate_peaks or (r_idx - candidate_peaks[-1]) >= min_dist:
                candidate_peaks.append(r_idx)
            elif abs(clean[r_idx]) > abs(clean[candidate_peaks[-1]]):
                candidate_peaks[-1] = r_idx

    return candidate_peaks


def extract_bidmc_coupled_beats(
    ecg: List[float],
    ppg: List[float],
    resp: Optional[List[float]] = None,
    fs: int = 125,
    ref_ppg_amp: Optional[float] = None,
) -> List[ExtractedBeatFeatures]:
    """Extract beat-synchronous coupled ECG + PPG + Respiration features from real BIDMC ICU waveforms (125 Hz)."""
    clean_ecg = detrend_baseline(ecg, window_samples=int(0.60 * fs))
    clean_ppg = detrend_baseline(ppg, window_samples=int(1.20 * fs))
    n = min(len(clean_ecg), len(clean_ppg))

    r_peaks = detect_real_r_peaks(ecg, fs=fs, refractory_sec=0.30)
    if len(r_peaks) < 2:
        return []

    abs_r = sorted(abs(clean_ecg[r]) for r in r_peaks)
    median_r = max(0.05, abs_r[len(abs_r) // 2])
    ecg_scale = 1.0 / median_r

    if ref_ppg_amp is None or ref_ppg_amp <= 0.01:
        ppg_std = math.sqrt(sum(x * x for x in clean_ppg) / max(1, n))
        ref_ppg_amp = max(0.05, ppg_std * 2.4)

    beats: List[ExtractedBeatFeatures] = []
    for b_idx, r_idx in enumerate(r_peaks):
        if b_idx > 0:
            rr_ms = ((r_idx - r_peaks[b_idx - 1]) / fs) * 1000.0
        else:
            rr_ms = ((r_peaks[1] - r_peaks[0]) / fs) * 1000.0

        # Isoelectric PR baseline [-80 ms to -40 ms]
        pr_lo = max(0, r_idx - int(0.08 * fs))
        pr_hi = max(pr_lo + 1, r_idx - int(0.04 * fs))
        iso = sum(clean_ecg[pr_lo:pr_hi]) / (pr_hi - pr_lo)

        # QRS width at 30% peak height
        r_amp_raw = abs(clean_ecg[r_idx] - iso)
        thresh_qrs = 0.28 * max(0.02, r_amp_raw)
        l_ptr = r_idx
        while l_ptr > max(0, r_idx - int(0.09 * fs)) and abs(clean_ecg[l_ptr] - iso) > thresh_qrs:
            l_ptr -= 1
        r_ptr = r_idx
        while r_ptr < min(n - 1, r_idx + int(0.09 * fs)) and abs(clean_ecg[r_ptr] - iso) > thresh_qrs:
            r_ptr += 1
        qrs_ms = max(62.0, ((r_ptr - l_ptr) / fs) * 1000.0 * 1.55)

        # ST deviation [+70 ms to +120 ms]
        st_lo = min(n - 1, r_idx + int(0.07 * fs))
        st_hi = min(n, r_idx + int(0.12 * fs))
        st_mv = ((sum(clean_ecg[st_lo:st_hi]) / max(1, st_hi - st_lo)) - iso) * ecg_scale

        # T-wave amplitude [+160 ms to +280 ms]
        t_lo = min(n - 1, r_idx + int(0.16 * fs))
        t_hi = min(n, r_idx + int(0.28 * fs))
        if t_hi > t_lo:
            t_vals = [(v - iso) * ecg_scale for v in clean_ecg[t_lo:t_hi]]
            t_amp = max(t_vals, key=abs)
        else:
            t_amp = 0.20

        # Coupled PPG Pulse Arrival Time (steepest systolic upstroke dP/dt in [120 ms, 420 ms])
        up_lo = min(n - 2, r_idx + int(0.12 * fs))
        up_hi = min(n - 2, r_idx + int(0.42 * fs))
        if up_hi > up_lo:
            up_idx = max(range(up_lo, up_hi), key=lambda k: clean_ppg[k + 1] - clean_ppg[k - 1])
            pat_ms = ((up_idx - r_idx) / fs) * 1000.0
            foot_val = min(clean_ppg[max(0, up_idx - int(0.12 * fs)):up_idx + 1])
            pk_hi = min(n, up_idx + int(0.22 * fs))
            sys_idx = max(range(up_idx, pk_hi), key=lambda k: clean_ppg[k])
            sys_amp = max(0.01, clean_ppg[sys_idx] - foot_val)
            ppg_ratio = sys_amp / ref_ppg_amp

            dia_lo = min(n - 1, sys_idx + int(0.12 * fs))
            dia_hi = min(n, sys_idx + int(0.26 * fs))
            if dia_hi > dia_lo:
                dia_amp = max(clean_ppg[dia_lo:dia_hi]) - foot_val
                notch_idx = max(0.0, min(1.0, dia_amp / max(0.02, sys_amp)))
            else:
                notch_idx = 0.32
        else:
            pat_ms = 340.0
            ppg_ratio = 0.70
            notch_idx = 0.32

        beats.append(
            ExtractedBeatFeatures(
                beat_idx=b_idx,
                r_idx=r_idx,
                rr_interval_ms=rr_ms,
                qrs_duration_ms=qrs_ms,
                st_deviation_mv=st_mv,
                t_wave_amp_mv=t_amp,
                ppg_amp_ratio=ppg_ratio,
                ppg_notch_index=notch_idx,
                pat_ms=pat_ms,
            )
        )

    return beats
