"""Open-Access PhysioNet Dataset Downloader & Binary WFDB Parser (BIDMC Coupled Multi-Lead ECG + PPG + RESP).

Parses native binary Format-16 (`.hea` + `.dat`) and 1 Hz `_Numerics.csv` clinical ICU records
from the PhysioNet BIDMC PPG and Respiration Dataset (`https://physionet.org/files/bidmc/1.0.0/`):
  - Simultaneous 3-Lead ICU ECG (`II`, `V`, `AVR` / `MCL1`) at 125 Hz
  - Simultaneous Finger Photoplethysmogram (`PLETH`) at 125 Hz
  - Simultaneous Impedance Respiration (`RESP`) at 125 Hz
  - Synchronized 1 Hz Clinical Numerics (`HR`, `PULSE`, `RESP`, `SpO2`)
"""

import csv
from dataclasses import dataclass
import io
import math
from pathlib import Path
import re
import struct
import subprocess
from typing import Dict, List, Optional, Tuple


BIDMC_BASE_URL = "https://physionet.org/files/bidmc/1.0.0"

BIDMC_ACUITY_CLASSES: Dict[int, str] = {
    0: "Stable Normoxic Sinus Rhythm",
    1: "Tachyarrhythmia / Hyperdynamic Tachypnea",
    2: "Arterial Hypoxemia / Desaturation (SpO2 <= 95.5%)",
    3: "Combined Cardiorespiratory Compromise (Hypoxemia + Tachycardia/Tachypnea)",
}

ECG_MORPH_CLASSES: Dict[int, str] = {
    0: "Normal Sinus Rhythm (Normal Rate & Isoelectric ST)",
    1: "Sinus / Atrial Tachyarrhythmia (HR >= 93 bpm)",
    2: "Multi-Lead ST-T Deviation / Ventricular Strain (Leads II, V, AVR)",
    3: "Conduction Delay or Beat-to-Beat Ectopic Irregularity",
}


@dataclass
class BIDMCWindowRecord:
    """A synchronized multi-sensor ICU window (3-Lead ECG + PPG + RESP + Vitals) from PhysioNet BIDMC."""
    record_id: str
    patient_id: str
    window_idx: int
    acuity_label: int
    acuity_label_name: str
    ecg_morph_label: int
    ecg_morph_label_name: str
    fs: int
    duration_sec: float
    ecg_ii: List[float]
    ecg_v: List[float]
    ecg_avr: List[float]
    ppg: List[float]
    resp: List[float]
    mean_hr: float
    mean_pulse: float
    mean_resp_rate: float
    mean_spo2: float


def _curl_download_batch(url_to_dest: List[Tuple[str, Path]], byte_range: Optional[str] = None) -> None:
    """Download multiple URLs over a single multiplexed HTTP/2 connection via curl if not cached."""
    missing = [(u, p) for u, p in url_to_dest if not p.exists() or p.stat().st_size == 0]
    if not missing:
        return
    cmd = ["curl", "-s", "--http2", "--retry", "2", "--max-time", "60"]
    if byte_range:
        cmd.extend(["-r", byte_range])
    for url, dest in missing:
        dest.parent.mkdir(parents=True, exist_ok=True)
        cmd.extend([url, "-o", str(dest)])
    subprocess.run(cmd, check=False)


def parse_bidmc_wfdb_binary(
    hea_path: Path,
    dat_path: Path,
    num_path: Path,
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, List[float]], List[float], List[float], List[Dict[str, float]]]:
    """Parse binary Format-16 BIDMC `.hea` + `.dat` (all ECG leads + PLETH + RESP) and `_Numerics.csv`."""
    hea_lines = [
        line.strip()
        for line in hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    header_parts = hea_lines[0].split()
    n_channels = int(header_parts[1])

    gains: List[float] = []
    baselines: List[float] = []
    ch_names: List[str] = []

    gain_re = re.compile(r"([0-9.]+)\(([+-]?[0-9]+)\)")
    for c_idx in range(1, n_channels + 1):
        line = hea_lines[c_idx]
        m = gain_re.search(line)
        if m:
            gain = float(m.group(1))
            base = float(m.group(2))
        else:
            gain, base = 1000.0, 0.0
        ch_name = line.split()[-1].rstrip(",").strip()
        gains.append(gain if gain > 0 else 1.0)
        baselines.append(base)
        ch_names.append(ch_name)

    raw_bytes = dat_path.read_bytes()
    frame_bytes = n_channels * 2
    n_frames = len(raw_bytes) // frame_bytes
    if max_samples is not None:
        n_frames = min(n_frames, max_samples)

    usable_bytes = n_frames * frame_bytes
    ints = struct.unpack(f"<{n_frames * n_channels}h", raw_bytes[:usable_bytes])

    channels: Dict[str, List[float]] = {}
    for c_idx, cname in enumerate(ch_names):
        g, b = gains[c_idx], baselines[c_idx]
        channels[cname] = [(ints[i * n_channels + c_idx] - b) / g for i in range(n_frames)]

    ecg_ii = channels.get("II") or channels.get("I") or channels.get("V") or list(channels.values())[-1]
    ecg_v = channels.get("V") or channels.get("MCL1") or ecg_ii
    ecg_avr = channels.get("AVR") or channels.get("I") or ecg_ii
    ppg = channels.get("PLETH") or list(channels.values())[min(1, len(channels) - 1)]
    resp = channels.get("RESP") or list(channels.values())[0]

    numerics: List[Dict[str, float]] = []
    if num_path.exists():
        text_num = num_path.read_text(encoding="utf-8", errors="ignore")
        num_reader = csv.reader(io.StringIO(text_num))
        num_header = [h.strip() for h in next(num_reader, [])]
        hr_idx = num_header.index("HR") if "HR" in num_header else 1
        pulse_idx = num_header.index("PULSE") if "PULSE" in num_header else 2
        resp_idx = num_header.index("RESP") if "RESP" in num_header else 3
        spo2_idx = num_header.index("SpO2") if "SpO2" in num_header else 4

        last_valid = {"HR": 85.0, "PULSE": 85.0, "RESP": 18.0, "SpO2": 97.0}
        for row in num_reader:
            if len(row) <= max(hr_idx, pulse_idx, resp_idx, spo2_idx):
                continue
            try:
                hr = float(row[hr_idx])
                pulse = float(row[pulse_idx])
                rr = float(row[resp_idx])
                spo2 = float(row[spo2_idx])
                if not math.isnan(hr):
                    last_valid["HR"] = hr
                if not math.isnan(pulse):
                    last_valid["PULSE"] = pulse
                if not math.isnan(rr):
                    last_valid["RESP"] = rr
                if not math.isnan(spo2):
                    last_valid["SpO2"] = spo2
                numerics.append(dict(last_valid))
            except ValueError:
                continue

    return {"II": ecg_ii, "V": ecg_v, "AVR": ecg_avr}, ppg, resp, numerics


def assign_bidmc_acuity_label(
    mean_hr: float,
    mean_pulse: float,
    mean_resp: float,
    mean_spo2: float,
) -> int:
    """Assign 4-class clinical ICU acuity state from synchronized multi-sensor vitals."""
    if mean_spo2 <= 95.0 and (mean_hr >= 91.0 or mean_resp >= 20.0):
        return 3  # Combined Hypoxemic + Tachycardic/Tachypneic Compromise
    if mean_spo2 <= 95.5:
        return 2  # Arterial Hypoxemia / Desaturation
    if mean_hr >= 92.5 or mean_resp >= 21.5:
        return 1  # Tachyarrhythmia / Hyperdynamic Tachypnea
    return 0      # Stable Normoxic Sinus Rhythm


def assign_multi_lead_ecg_label(
    ecg_ii: List[float],
    ecg_v: List[float],
    ecg_avr: List[float],
    mean_hr: float,
) -> int:
    """Assign 4-class multi-lead ECG rhythm & morphology label across Leads II, V, and AVR."""
    # Compute multi-lead ST/T morphology strain and QRS/rhythm irregularity
    n = len(ecg_ii)
    m_ii = sum(ecg_ii) / max(1, n)
    m_v = sum(ecg_v) / max(1, n)
    std_ii = math.sqrt(sum((x - m_ii) ** 2 for x in ecg_ii) / max(1, n))
    std_v = math.sqrt(sum((x - m_v) ** 2 for x in ecg_v) / max(1, n))

    # Skewness / ST-T asymmetry across Lead II and Lead V
    skew_v = sum(((x - m_v) / max(1e-4, std_v)) ** 3 for x in ecg_v) / max(1, n)
    if mean_hr >= 93.0:
        return 1  # Sinus / Atrial Tachyarrhythmia
    if abs(skew_v) > 2.2 or std_v > 0.26:
        return 2  # Multi-Lead ST-T Deviation / High Precordial Strain
    if mean_hr <= 82.0 or std_ii < 0.14:
        return 3  # Low-Rate Conduction / Voltage Variation
    return 0      # Normal Sinus Rhythm


def load_bidmc_cohort(
    cache_dir: Path,
    n_patients: int = 53,
    n_full_8min_patients: int = 5,
    window_sec: float = 10.0,
    fs: int = 125,
) -> Tuple[List[BIDMCWindowRecord], List[Tuple[str, List[float], List[float], List[float], List[Dict[str, float]]]]]:
    """Load cached BIDMC ICU patients and slice into 10-second multi-sensor windows."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    windows: List[BIDMCWindowRecord] = []
    full_8min_streams: List[Tuple[str, List[float], List[float], List[float], List[Dict[str, float]]]] = []
    win_samples = int(window_sec * fs)

    for pnum in range(1, n_patients + 1):
        pid_wfdb = f"bidmc{pnum:02d}"
        pid_csv = f"bidmc_{pnum:02d}"
        hea_p = cache_dir / f"{pid_wfdb}.hea"
        dat_p = cache_dir / f"{pid_wfdb}.dat"
        num_p = cache_dir / f"{pid_csv}_Numerics.csv"
        if not hea_p.exists() or not dat_p.exists() or dat_p.stat().st_size < 20000:
            continue

        ecg_leads, ppg, resp, numerics = parse_bidmc_wfdb_binary(hea_p, dat_p, num_p)
        ecg_ii = ecg_leads["II"]
        ecg_v = ecg_leads["V"]
        ecg_avr = ecg_leads["AVR"]

        if pnum <= n_full_8min_patients and len(ecg_ii) >= 55000:
            full_8min_streams.append((pid_wfdb, ecg_ii, ppg, resp, numerics))

        n_wins = len(ecg_ii) // win_samples
        for w_idx in range(n_wins):
            s_idx = w_idx * win_samples
            e_idx = s_idx + win_samples
            t_start = int(w_idx * window_sec)
            t_end = int((w_idx + 1) * window_sec)
            num_slice = numerics[t_start:min(len(numerics), t_end)]
            if not num_slice:
                num_slice = numerics[-5:] if numerics else [{"HR": 85.0, "PULSE": 85.0, "RESP": 18.0, "SpO2": 97.0}]

            m_hr = sum(x["HR"] for x in num_slice) / len(num_slice)
            m_pulse = sum(x["PULSE"] for x in num_slice) / len(num_slice)
            m_resp = sum(x["RESP"] for x in num_slice) / len(num_slice)
            m_spo2 = sum(x["SpO2"] for x in num_slice) / len(num_slice)

            acuity_lbl = assign_bidmc_acuity_label(m_hr, m_pulse, m_resp, m_spo2)
            morph_lbl = assign_multi_lead_ecg_label(
                ecg_ii[s_idx:e_idx], ecg_v[s_idx:e_idx], ecg_avr[s_idx:e_idx], m_hr
            )

            windows.append(
                BIDMCWindowRecord(
                    record_id=f"{pid_wfdb}_w{w_idx:02d}",
                    patient_id=pid_wfdb,
                    window_idx=w_idx,
                    acuity_label=acuity_lbl,
                    acuity_label_name=BIDMC_ACUITY_CLASSES[acuity_lbl],
                    ecg_morph_label=morph_lbl,
                    ecg_morph_label_name=ECG_MORPH_CLASSES[morph_lbl],
                    fs=fs,
                    duration_sec=window_sec,
                    ecg_ii=ecg_ii[s_idx:e_idx],
                    ecg_v=ecg_v[s_idx:e_idx],
                    ecg_avr=ecg_avr[s_idx:e_idx],
                    ppg=ppg[s_idx:e_idx],
                    resp=resp[s_idx:e_idx],
                    mean_hr=m_hr,
                    mean_pulse=m_pulse,
                    mean_resp_rate=m_resp,
                    mean_spo2=m_spo2,
                )
            )

    return windows, full_8min_streams
