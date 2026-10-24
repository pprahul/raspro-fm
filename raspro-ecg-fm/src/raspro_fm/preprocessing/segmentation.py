"""Beat-synchronous ECG and PPG waveform segmentation and clinical feature extraction."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExtractedBeatFeatures:
    """Measured physiological features extracted from raw coupled ECG + PPG for one cardiac cycle."""
    beat_idx: int
    r_idx: int
    rr_interval_ms: float
    qrs_duration_ms: float
    st_deviation_mv: float
    t_wave_amp_mv: float
    ppg_amp_ratio: float
    ppg_notch_index: float
    pat_ms: float


def detrend_baseline(signal: List[float], window_samples: int = 75) -> List[float]:
    """Remove low-frequency respiration baseline wander using a fast moving-median/mean filter."""
    n = len(signal)
    if n == 0 or window_samples <= 1:
        return list(signal)
    half = window_samples // 2
    out = [0.0] * n
    # Compute prefix sum for O(N) baseline estimation
    prefix = [0.0] * (n + 1)
    for i, val in enumerate(signal):
        prefix[i + 1] = prefix[i] + val
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        baseline = (prefix[hi] - prefix[lo]) / (hi - lo)
        out[i] = signal[i] - baseline
    return out


def detect_r_peaks(ecg: List[float], fs: int = 100, refractory_sec: float = 0.32) -> List[int]:
    """Detect ECG R-peaks using adaptive local maximum search above energy threshold."""
    clean_ecg = detrend_baseline(ecg, window_samples=max(31, int(0.65 * fs)))
    n = len(clean_ecg)
    if n < fs:
        return []

    sorted_vals = sorted(clean_ecg)
    p95 = sorted_vals[int(0.95 * (n - 1))]
    threshold = max(0.38, 0.52 * p95)
    min_dist = int(refractory_sec * fs)

    peaks: List[int] = []
    for i in range(2, n - 2):
        val = clean_ecg[i]
        if val >= threshold and val >= clean_ecg[i - 1] and val >= clean_ecg[i + 1]:
            if val >= clean_ecg[i - 2] and val >= clean_ecg[i + 2]:
                if not peaks or (i - peaks[-1]) >= min_dist:
                    peaks.append(i)
                elif val > clean_ecg[peaks[-1]]:
                    peaks[-1] = i
    return peaks


def extract_coupled_beat_features(
    ecg: List[float],
    ppg: List[float],
    fs: int = 100,
    r_peaks: Optional[List[int]] = None,
) -> List[ExtractedBeatFeatures]:
    """Extract beat-by-beat morphological and hemodynamic measurements from ECG + PPG."""
    clean_ecg = detrend_baseline(ecg, window_samples=max(31, int(0.60 * fs)))
    clean_ppg = detrend_baseline(ppg, window_samples=max(51, int(1.10 * fs)))
    n = len(clean_ecg)

    if r_peaks is None or len(r_peaks) < 2:
        r_peaks = detect_r_peaks(ecg, fs=fs)

    if not r_peaks:
        return []

    # Estimate patient median PPG systolic amplitude for relative perfusion normalization
    raw_ppg_amps: List[float] = []
    for r_idx in r_peaks:
        s_lo = min(n - 1, r_idx + int(0.10 * fs))
        s_hi = min(n, r_idx + int(0.38 * fs))
        if s_hi > s_lo:
            foot = min(clean_ppg[max(0, r_idx):s_hi])
            peak = max(clean_ppg[s_lo:s_hi])
            raw_ppg_amps.append(max(0.05, peak - foot))
    baseline_ppg_amp = sorted(raw_ppg_amps)[len(raw_ppg_amps) // 2] if raw_ppg_amps else 1.0
    baseline_ppg_amp = max(0.45, baseline_ppg_amp)

    extracted: List[ExtractedBeatFeatures] = []
    for b_idx, r_idx in enumerate(r_peaks):
        # 1. RR interval (ms)
        if b_idx > 0:
            rr_ms = ((r_idx - r_peaks[b_idx - 1]) / fs) * 1000.0
        elif len(r_peaks) > 1:
            rr_ms = ((r_peaks[1] - r_peaks[0]) / fs) * 1000.0
        else:
            rr_ms = 800.0

        # 2. Isoelectric PR reference level [-90 ms to -50 ms before R peak]
        pr_lo = max(0, r_idx - int(0.09 * fs))
        pr_hi = max(pr_lo + 1, r_idx - int(0.05 * fs))
        iso_level = sum(clean_ecg[pr_lo:pr_hi]) / (pr_hi - pr_lo)

        # 3. QRS duration estimation via half-prominence width around R peak
        r_height = max(0.2, clean_ecg[r_idx] - iso_level)
        half_thresh = iso_level + 0.35 * r_height
        left_idx = r_idx
        while left_idx > max(0, r_idx - int(0.08 * fs)) and clean_ecg[left_idx] > half_thresh:
            left_idx -= 1
        right_idx = r_idx
        while right_idx < min(n - 1, r_idx + int(0.08 * fs)) and clean_ecg[right_idx] > half_thresh:
            right_idx += 1
        qrs_ms = max(65.0, ((right_idx - left_idx) / fs) * 1000.0 * 1.85)

        # 4. ST-segment level [+80 ms to +130 ms after R peak]
        st_lo = min(n - 1, r_idx + int(0.08 * fs))
        st_hi = min(n, r_idx + int(0.14 * fs))
        if st_hi > st_lo:
            st_mv = (sum(clean_ecg[st_lo:st_hi]) / (st_hi - st_lo)) - iso_level
        else:
            st_mv = 0.0

        # 5. T-wave peak amplitude [+180 ms to +300 ms after R peak]
        t_lo = min(n - 1, r_idx + int(0.18 * fs))
        t_hi = min(n, r_idx + int(0.30 * fs))
        if t_hi > t_lo:
            t_slice = [v - iso_level for v in clean_ecg[t_lo:t_hi]]
            t_max = max(t_slice)
            t_min = min(t_slice)
            t_amp = t_max if abs(t_max) >= abs(t_min) else t_min
        else:
            t_amp = 0.25

        # 6. Coupled PPG pulse peak, Pulse Arrival Time (PAT), and Dicrotic Notch Reflection Index
        ppg_lo = min(n - 1, r_idx + int(0.11 * fs))
        ppg_hi = min(n, r_idx + int(0.36 * fs))
        if ppg_hi > ppg_lo:
            ppg_foot = min(clean_ppg[max(0, r_idx):ppg_hi])
            best_offset = 0
            best_val = clean_ppg[ppg_lo]
            for offset, val in enumerate(clean_ppg[ppg_lo:ppg_hi]):
                if val > best_val:
                    best_val = val
                    best_offset = offset
            sys_peak_idx = ppg_lo + best_offset
            pat_ms = ((sys_peak_idx - r_idx) / fs) * 1000.0
            sys_amp = max(0.05, best_val - ppg_foot)
            ppg_amp_ratio = sys_amp / 1.05

            # Measure diastolic reflected wave / dicrotic notch prominence [+150 ms to +240 ms after systolic peak]
            dia_lo = min(n - 1, sys_peak_idx + int(0.15 * fs))
            dia_hi = min(n, sys_peak_idx + int(0.24 * fs))
            if dia_hi > dia_lo:
                dia_val = max(clean_ppg[dia_lo:dia_hi]) - ppg_foot
                notch_idx = max(0.0, min(1.0, dia_val / max(0.15, sys_amp)))
            else:
                notch_idx = 0.30
        else:
            pat_ms = 200.0
            ppg_amp_ratio = 1.0
            notch_idx = 0.35

        extracted.append(
            ExtractedBeatFeatures(
                beat_idx=b_idx,
                r_idx=r_idx,
                rr_interval_ms=rr_ms,
                qrs_duration_ms=qrs_ms,
                st_deviation_mv=st_mv,
                t_wave_amp_mv=t_amp,
                ppg_amp_ratio=ppg_amp_ratio,
                ppg_notch_index=notch_idx,
                pat_ms=pat_ms,
            )
        )

    return extracted
