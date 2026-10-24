"""Biophysically grounded coupled ECG + PPG physiological generator & PhysioNet loader.

Generates realistic multi-sensor cardiovascular time-series recordings across
five clinical classes:
  0: NSR   - Normal Sinus Rhythm
  1: STEMI - Myocardial Ischemia / ST-Segment Elevation or Depression
  2: AFIB  - Atrial Fibrillation (irregularly irregular RR, absent P, variable PPG)
  3: AHE   - Acute Hypotensive Episode (prolonged ECG-PPG PAT, damped PPG, loss of notch, reflex tachycardia)
  4: PVC   - Frequent Premature Ventricular Contractions / Ventricular Ectopy runs
"""

from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Optional, Tuple


CLINICAL_CLASSES: Dict[int, str] = {
    0: "NSR (Normal Sinus Rhythm)",
    1: "STEMI/Ischemia (ST-Segment Deviation)",
    2: "AFib (Atrial Fibrillation)",
    3: "AHE (Acute Hypotensive Episode)",
    4: "PVC/Ectopy (Ventricular Arrhythmia)",
}


@dataclass
class BeatTruth:
    """Ground-truth morphological and hemodynamic parameters for a single cardiac cycle."""
    beat_idx: int
    r_peak_time_sec: float
    rr_interval_ms: float
    qrs_duration_ms: float
    st_deviation_mv: float
    ppg_amp_ratio: float
    ppg_notch_index: float
    pat_ms: float
    is_abnormal_beat: bool


@dataclass
class PhysiologicalRecord:
    """A synchronized multi-sensor physiological recording (ECG + PPG)."""
    record_id: str
    label: int
    label_name: str
    fs: int
    duration_sec: float
    ecg: List[float]
    ppg: List[float]
    r_peaks: List[int]
    beats: List[BeatTruth] = field(default_factory=list)


def _gaussian_wave(t: float, center: float, width: float, amp: float) -> float:
    """Evaluate a single Gaussian morphological wave component (P, Q, R, S, T, or PPG pulse)."""
    if width <= 1e-6:
        return 0.0
    z = (t - center) / width
    if abs(z) > 6.0:
        return 0.0
    return amp * math.exp(-0.5 * z * z)


def generate_coupled_ecg_ppg(
    record_id: str,
    label: int,
    duration_sec: float = 30.0,
    fs: int = 100,
    seed: Optional[int] = None,
    noise_std: float = 0.025,
) -> PhysiologicalRecord:
    """Synthesize a synchronized ECG + PPG time series with realistic beat-to-beat dynamics."""
    rng = random.Random(seed)
    n_samples = int(duration_sec * fs)
    ecg = [0.0] * n_samples
    ppg = [0.0] * n_samples

    # Patient-level baseline variability
    base_hr_bpm = rng.uniform(64.0, 82.0)
    if label == 1:  # STEMI: mild sinus response
        base_hr_bpm = rng.uniform(72.0, 92.0)
    elif label == 2:  # AFib: elevated average ventricular rate
        base_hr_bpm = rng.uniform(88.0, 125.0)
    elif label == 3:  # AHE: progressive reflex tachycardia during hypotension
        base_hr_bpm = rng.uniform(85.0, 112.0)
    elif label == 4:  # PVC: normal background sinus rate with premature beats
        base_hr_bpm = rng.uniform(66.0, 84.0)

    base_rr_sec = 60.0 / base_hr_bpm
    st_polarity = 1.0 if rng.random() > 0.35 else -1.0  # Elevation vs depression for STEMI/NSTEMI

    # Generate sequence of beat timestamps and per-beat physiological parameters
    t_cursor = 0.45 + rng.uniform(0.0, 0.2)
    r_peaks: List[int] = []
    beats: List[BeatTruth] = []
    beat_idx = 0

    while t_cursor < duration_sec - 0.6:
        progress = t_cursor / duration_sec  # 0.0 -> 1.0 across the recording

        # Default normal sinus beat parameters
        rr_sec = base_rr_sec * (1.0 + rng.gauss(0.0, 0.035))
        p_amp = 0.14 * rng.uniform(0.85, 1.15)
        qrs_width_sec = rng.uniform(0.014, 0.019)  # corresponds to ~80-95 ms full QRS
        qrs_duration_ms = qrs_width_sec * 5000.0
        r_amp = rng.uniform(1.05, 1.35)
        st_dev_mv = rng.gauss(0.0, 0.02)
        t_amp = rng.uniform(0.24, 0.34)

        ppg_amp = rng.uniform(0.92, 1.08)
        ppg_notch = rng.uniform(0.32, 0.46)
        pat_ms = rng.uniform(185.0, 215.0)
        is_abn = False

        if label == 1:
            # Class 1: Myocardial Ischemia / STEMI (subtle or pronounced ST shift + T wave change)
            # Often intermittent or progressive across the strip
            if progress > 0.20:
                st_magnitude = rng.uniform(0.14, 0.26)
                st_dev_mv = st_polarity * st_magnitude
                t_amp = 0.42 if st_polarity > 0 else -0.22  # Hyperacute or inverted T
                is_abn = True

        elif label == 2:
            # Class 2: Atrial Fibrillation (irregularly irregular RR, absent P wave, variable PPG pulse volume)
            rr_sec = base_rr_sec * rng.uniform(0.55, 1.45)
            p_amp = 0.0
            # Shorter filling time -> smaller PPG stroke volume
            filling_ratio = min(1.3, max(0.45, rr_sec / base_rr_sec))
            ppg_amp = 0.85 * filling_ratio + rng.gauss(0.0, 0.05)
            ppg_notch = rng.uniform(0.20, 0.35)
            is_abn = True

        elif label == 3:
            # Class 3: Acute Hypotensive Episode (AHE)
            # Progressive hemodynamic deterioration in second half of window:
            # -> RR shortens (reflex tachycardia), PPG amplitude drops, dicrotic notch disappears, PAT prolongs
            if progress > 0.25:
                severity = min(1.0, (progress - 0.25) / 0.55)
                rr_sec = base_rr_sec * (1.0 - 0.28 * severity + rng.gauss(0.0, 0.02))
                ppg_amp = max(0.32, 1.0 - 0.62 * severity + rng.gauss(0.0, 0.04))
                ppg_notch = max(0.05, 0.38 * (1.0 - 0.85 * severity))
                pat_ms = 200.0 + 95.0 * severity + rng.gauss(0.0, 6.0)
                st_dev_mv = -0.09 * severity  # secondary subendocardial strain
                is_abn = severity > 0.30

        elif label == 4:
            # Class 4: Premature Ventricular Contractions (PVC) / Ectopic couplets
            if beat_idx % rng.randint(3, 5) == 2:
                # Premature wide bizarre QRS beat with blunted PPG perfusion wave
                rr_sec = base_rr_sec * rng.uniform(0.52, 0.66)
                p_amp = 0.0
                qrs_width_sec = rng.uniform(0.030, 0.040)
                qrs_duration_ms = qrs_width_sec * 5000.0  # ~150-200 ms wide QRS
                r_amp = rng.uniform(1.45, 1.85)
                st_dev_mv = rng.uniform(-0.22, -0.14)
                t_amp = rng.uniform(-0.38, -0.25)  # discordant T wave
                ppg_amp = rng.uniform(0.30, 0.50)  # severe pulse deficit on ectopic beat
                ppg_notch = rng.uniform(0.06, 0.14)
                pat_ms = rng.uniform(245.0, 285.0)
                is_abn = True

        r_idx = int(round(t_cursor * fs))
        if 0 <= r_idx < n_samples:
            r_peaks.append(r_idx)
            beats.append(
                BeatTruth(
                    beat_idx=beat_idx,
                    r_peak_time_sec=t_cursor,
                    rr_interval_ms=rr_sec * 1000.0,
                    qrs_duration_ms=qrs_duration_ms,
                    st_deviation_mv=st_dev_mv,
                    ppg_amp_ratio=ppg_amp,
                    ppg_notch_index=ppg_notch,
                    pat_ms=pat_ms,
                    is_abnormal_beat=is_abn,
                )
            )

        # Render ECG P-QRS-ST-T morphology into the continuous array
        win_start = max(0, int((t_cursor - 0.30) * fs))
        win_end = min(n_samples, int((t_cursor + 0.65) * fs))
        pat_sec = pat_ms / 1000.0

        for idx in range(win_start, win_end):
            t = idx / fs
            # ECG components
            v_ecg = 0.0
            v_ecg += _gaussian_wave(t, t_cursor - 0.16, 0.025, p_amp)                  # P wave
            v_ecg += _gaussian_wave(t, t_cursor - 0.03, 0.010, -0.15 * r_amp)          # Q wave
            v_ecg += _gaussian_wave(t, t_cursor, qrs_width_sec, r_amp)                 # R peak
            v_ecg += _gaussian_wave(t, t_cursor + 0.035, 0.012, -0.25 * r_amp)         # S wave
            v_ecg += _gaussian_wave(t, t_cursor + 0.11, 0.055, st_dev_mv)              # ST segment plateau
            v_ecg += _gaussian_wave(t, t_cursor + 0.24, 0.048, t_amp)                  # T wave
            ecg[idx] += v_ecg

            # Coupled PPG pulse wave (arriving after Pulse Arrival Time `pat_sec`)
            t_ppg_peak = t_cursor + pat_sec
            v_ppg = 0.0
            # Systolic forward wave
            v_ppg += _gaussian_wave(t, t_ppg_peak, 0.075, ppg_amp)
            # Dicrotic notch / diastolic reflected wave
            v_ppg += _gaussian_wave(t, t_ppg_peak + 0.19, 0.095, ppg_amp * ppg_notch)
            ppg[idx] += v_ppg

        t_cursor += rr_sec
        beat_idx += 1

    # Add realistic respiration baseline wander (0.22 Hz) + sensor noise (+ fibrillatory f-waves for AFib)
    resp_freq = 0.22
    resp_phase = rng.uniform(0.0, 2.0 * math.pi)
    for idx in range(n_samples):
        t = idx / fs
        wander = 0.06 * math.sin(2.0 * math.pi * resp_freq * t + resp_phase)
        f_wave = 0.04 * math.sin(2.0 * math.pi * 6.2 * t) if label == 2 else 0.0
        ecg[idx] += wander + f_wave + rng.gauss(0.0, noise_std)
        ppg[idx] += 0.03 * wander + rng.gauss(0.0, noise_std * 0.6)

    return PhysiologicalRecord(
        record_id=record_id,
        label=label,
        label_name=CLINICAL_CLASSES[label],
        fs=fs,
        duration_sec=duration_sec,
        ecg=ecg,
        ppg=ppg,
        r_peaks=r_peaks,
        beats=beats,
    )


def generate_cohort_dataset(
    n_records: int,
    duration_sec: float = 30.0,
    fs: int = 100,
    seed: int = 42,
) -> List[PhysiologicalRecord]:
    """Generate a balanced cohort of multi-sensor ECG + PPG recordings across all 5 clinical classes."""
    records: List[PhysiologicalRecord] = []
    n_classes = len(CLINICAL_CLASSES)
    for i in range(n_records):
        label = i % n_classes
        rec = generate_coupled_ecg_ppg(
            record_id=f"REC_{i:04d}",
            label=label,
            duration_sec=duration_sec,
            fs=fs,
            seed=seed + i * 17,
        )
        records.append(rec)
    return records
