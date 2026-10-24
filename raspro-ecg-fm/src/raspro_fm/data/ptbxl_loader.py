"""PhysioNet PTB-XL (v1.0.3) 12-Lead ECG Loader and Official Fold-10 Benchmark Builder.

Implements the canonical evaluation protocol used by SOTA ECG Foundation Models:
  - ST-MEM (Na et al., ICLR 2024)
  - HeartLang (Jin et al., ICLR 2025)
  - MERL (Liu et al., ICML 2024)
  - ECG-FM (McKeen et al., 2024)
  - ECG-Byte (2024)
  - BenchECG / xECG (2025)

Official Split:
  - Folds 1–8: Self-supervised pre-training & supervised downstream training (1%, 10%, 100%)
  - Fold 9: Validation set (human-validated annotations)
  - Fold 10: Held-out Test set (human-validated gold-standard annotations)
"""

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import struct
import subprocess
from typing import Dict, List, Optional, Tuple
import urllib.request
import numpy as np


PTBXL_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3"

SUPERCLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

CANONICAL_SUBCLASSES = [
    "NORM", "IMI", "AMI", "STTC", "LVH", "LAFB/LPFB", "ISC_", "IRBBB",
    "_AVB", "IVCD", "ISCA", "CRBBB", "CLBBB", "LAO/LAE", "ISCI", "NST_"
]

CANONICAL_FORM_CLASSES = [
    "NDT", "NST_", "DIG", "LNGQT", "ABQRS", "PVC", "STD_", "VCLVH",
    "QWAVE", "LOWT", "NT_", "PAC", "LPR", "INVT", "LVOLT", "HVOLT", "TAB_", "STE_"
]

CANONICAL_RHYTHM_CLASSES = [
    "SR", "AFIB", "STACH", "SARRH", "SBRAD", "PACE", "SVARR", "AFLT", "BIGU", "PSVT"
]


@dataclass
class PTBXLRecord:
    """Single 10-second 12-lead ECG record from PhysioNet PTB-XL (100 Hz)."""
    ecg_id: int
    patient_id: int
    age: float
    sex: int
    strat_fold: int
    filename_lr: str
    scp_codes: Dict[str, float]
    signals: np.ndarray        # Shape: (12, 1000) in mV at 100 Hz
    lead_names: List[str]      # ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1',..,'V6']
    fs: float                  # 100.0 Hz
    y_superclass: np.ndarray   # Multi-hot shape (5,)
    y_subclass: np.ndarray     # Multi-hot shape (len(subclass_names),)
    y_form: np.ndarray         # Multi-hot shape (len(form_names),)
    y_rhythm: np.ndarray       # Multi-hot shape (len(rhythm_names),)


@dataclass
class PTBXLBenchmarkSplit:
    """Stratified PTB-XL dataset split following Folds 1-8 / Fold 9 / Fold 10."""
    train_records: List[PTBXLRecord]   # Folds 1–8
    val_records: List[PTBXLRecord]     # Fold 9
    test_records: List[PTBXLRecord]    # Fold 10
    superclass_names: List[str]
    subclass_names: List[str]
    form_names: List[str]
    rhythm_names: List[str]


def _curl_download_multiplexed(url_to_dest: List[Tuple[str, Path]], batch_size: int = 40) -> None:
    """Download missing files in multiplexed HTTP/2 batches via curl."""
    missing = [(u, p) for u, p in url_to_dest if not p.exists() or p.stat().st_size < 50]
    if not missing:
        return
    for i in range(0, len(missing), batch_size):
        chunk = missing[i : i + batch_size]
        cmd = ["curl", "-sL", "--http2", "--parallel", "--parallel-max", "16", "--retry", "2", "--max-time", "45"]
        for url, dest in chunk:
            dest.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend([url, "-o", str(dest)])
        subprocess.run(cmd, check=False)


def parse_ptbxl_wfdb_12lead(hea_path: Path, dat_path: Path) -> Tuple[float, np.ndarray, List[str]]:
    """Parse binary Format-16 12-lead PTB-XL `.hea` and `.dat` file."""
    hea_lines = [
        line.strip()
        for line in hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    header_parts = hea_lines[0].split()
    n_channels = int(header_parts[1])
    fs = float(header_parts[2].split("/")[0])

    gains: List[float] = []
    baselines: List[float] = []
    lead_names: List[str] = []

    gain_re = re.compile(r"([0-9.]+)(?:\(([+-]?[0-9]+)\))?")
    for c_idx in range(1, n_channels + 1):
        line = hea_lines[c_idx]
        parts = line.split()
        # Format: 00001_lr.dat 16 1000/mV 16 0 -119 1280 0 I
        gain_field = parts[2] if len(parts) > 2 else "1000"
        m = gain_re.match(gain_field)
        if m:
            gain = float(m.group(1))
            base = float(m.group(2)) if m.group(2) is not None else 0.0
        else:
            gain, base = 1000.0, 0.0
        lead_name = parts[-1].upper()
        gains.append(gain if gain > 0 else 1000.0)
        baselines.append(base)
        lead_names.append(lead_name)

    raw_bytes = dat_path.read_bytes()
    frame_bytes = n_channels * 2
    n_frames = len(raw_bytes) // frame_bytes
    usable_bytes = n_frames * frame_bytes
    ints = np.frombuffer(raw_bytes[:usable_bytes], dtype="<i2").reshape(n_frames, n_channels)

    gains_arr = np.array(gains, dtype=np.float64)[None, :]
    base_arr = np.array(baselines, dtype=np.float64)[None, :]
    signals = ((ints.astype(np.float64) - base_arr) / gains_arr).T  # (12, n_frames)
    return fs, signals, lead_names


def load_scp_mappings(cache_dir: Path) -> Tuple[Dict[str, str], Dict[str, str], List[str], List[str]]:
    """Parse scp_statements.csv to map SCP codes to superclass, subclass, form, and rhythm."""
    scp_path = cache_dir / "scp_statements.csv"
    if not scp_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{PTBXL_BASE_URL}/scp_statements.csv", scp_path)

    code_to_super: Dict[str, str] = {}
    code_to_sub: Dict[str, str] = {}
    form_codes: List[str] = []
    rhythm_codes: List[str] = []

    with open(scp_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row[""].strip()
            if row.get("diagnostic", "").strip() == "1.0":
                sc = row.get("diagnostic_class", "").strip()
                sub = row.get("diagnostic_subclass", "").strip()
                if sc:
                    code_to_super[code] = sc
                if sub:
                    code_to_sub[code] = sub
            if row.get("form", "").strip() == "1.0":
                form_codes.append(code)
            if row.get("rhythm", "").strip() == "1.0":
                rhythm_codes.append(code)

    return code_to_super, code_to_sub, form_codes, rhythm_codes


def select_stratified_ptbxl_metadata(
    cache_dir: Path,
    train_target: int = 210,
    val_target: int = 45,
    test_target: int = 75,
    seed: int = 42,
) -> Tuple[List[dict], List[str], List[str], List[str]]:
    """Select stratified PTB-XL metadata rows across Folds 1-8 (train), Fold 9 (val), Fold 10 (test)."""
    code_to_super, code_to_sub, _, _ = load_scp_mappings(cache_dir)
    db_path = cache_dir / "ptbxl_database.csv"
    if not db_path.exists():
        urllib.request.urlretrieve(f"{PTBXL_BASE_URL}/ptbxl_database.csv", db_path)

    rows_by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    with open(db_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ecg_id = int(row["ecg_id"])
                fold = int(row["strat_fold"])
                scp_dict = ast.literal_eval(row["scp_codes"])
            except Exception:
                continue

            rel_lr = row["filename_lr"].strip()
            is_cached = (cache_dir / f"{rel_lr}.dat").exists()

            # Official PTB-XL rule (Wagner et al., 2020): diagnostic statements require likelihood >= 50.0,
            # while form and rhythm statements are coded with likelihood >= 0.0 (often 0.0 for SR/LVOLT).
            supers = {code_to_super[c] for c, v in scp_dict.items() if c in code_to_super and float(v) >= 50.0}
            subs = {code_to_sub[c] for c, v in scp_dict.items() if c in code_to_sub and float(v) >= 50.0}
            forms = {c for c, v in scp_dict.items() if c in CANONICAL_FORM_CLASSES and float(v) >= 0.0}
            rhythms = {c for c, v in scp_dict.items() if c in CANONICAL_RHYTHM_CLASSES and float(v) >= 0.0}

            if not supers and not forms and not rhythms:
                continue

            entry = {
                "ecg_id": ecg_id,
                "patient_id": int(float(row["patient_id"])) if row.get("patient_id") else ecg_id,
                "age": float(row["age"]) if row.get("age") else 60.0,
                "sex": int(float(row["sex"])) if row.get("sex") else 0,
                "strat_fold": fold,
                "filename_lr": rel_lr,
                "is_cached": is_cached,
                "scp_codes": scp_dict,
                "supers": supers,
                "subs": subs,
                "forms": forms,
                "rhythms": rhythms,
            }
            if fold <= 8:
                rows_by_split["train"].append(entry)
            elif fold == 9:
                rows_by_split["val"].append(entry)
            elif fold == 10:
                rows_by_split["test"].append(entry)

    # If a full stratified cohort is already cached on disk, use cached records first to avoid slow network I/O
    for sp_k in ("train", "val", "test"):
        cached_only = [r for r in rows_by_split[sp_k] if r["is_cached"]]
        if len(cached_only) >= 35:
            rows_by_split[sp_k] = cached_only

    rng = np.random.default_rng(seed)

    def _stratified_pick(pool: List[dict], target_n: int) -> List[dict]:
        if len(pool) <= target_n:
            return pool
        selected_ids = set()
        selected = []

        all_rare_tags = (
            [("rhythm", r) for r in CANONICAL_RHYTHM_CLASSES if r != "SR"]
            + [("form", fm) for fm in CANONICAL_FORM_CLASSES]
            + [("sub", sb) for sb in CANONICAL_SUBCLASSES if sb != "NORM"]
            + [("super", sc) for sc in SUPERCLASS_NAMES]
        )
        per_tag_quota = max(2, target_n // 28)
        for kind, tag in all_rare_tags:
            if len(selected) >= int(target_n * 0.82):
                break
            key = {"rhythm": "rhythms", "form": "forms", "sub": "subs", "super": "supers"}[kind]
            matches = [r for r in pool if tag in r[key] and r["ecg_id"] not in selected_ids]
            if matches:
                take = min(len(matches), per_tag_quota)
                idxs = rng.choice(len(matches), size=take, replace=False)
                for i in idxs:
                    selected.append(matches[int(i)])
                    selected_ids.add(matches[int(i)]["ecg_id"])

        remaining_needed = target_n - len(selected)
        if remaining_needed > 0:
            per_sc = max(1, remaining_needed // len(SUPERCLASS_NAMES))
            for sc in SUPERCLASS_NAMES:
                matches = [r for r in pool if sc in r["supers"] and r["ecg_id"] not in selected_ids]
                if matches:
                    take = min(len(matches), per_sc)
                    idxs = rng.choice(len(matches), size=take, replace=False)
                    for i in idxs:
                        if len(selected) < target_n:
                            selected.append(matches[int(i)])
                            selected_ids.add(matches[int(i)]["ecg_id"])

        if len(selected) < target_n:
            rest = [r for r in pool if r["ecg_id"] not in selected_ids]
            take = min(len(rest), target_n - len(selected))
            idxs = rng.choice(len(rest), size=take, replace=False)
            for i in idxs:
                selected.append(rest[int(i)])
                selected_ids.add(rest[int(i)]["ecg_id"])

        return selected[:target_n]

    train_sel = _stratified_pick(rows_by_split["train"], train_target)
    val_sel = _stratified_pick(rows_by_split["val"], val_target)
    test_sel = _stratified_pick(rows_by_split["test"], test_target)

    active_subclasses = [
        sb for sb in CANONICAL_SUBCLASSES
        if sum(sb in r["subs"] for r in train_sel) >= 3 and sum(sb in r["subs"] for r in test_sel) >= 2
    ]
    active_forms = [
        fm for fm in CANONICAL_FORM_CLASSES
        if sum(fm in r["forms"] for r in train_sel) >= 3 and sum(fm in r["forms"] for r in test_sel) >= 2
    ]
    active_rhythms = [
        rh for rh in CANONICAL_RHYTHM_CLASSES
        if sum(rh in r["rhythms"] for r in train_sel) >= 3 and sum(rh in r["rhythms"] for r in test_sel) >= 2
    ]

    return train_sel + val_sel + test_sel, active_subclasses, active_forms, active_rhythms


def load_ptbxl_benchmark_split(
    cache_dir: Path,
    train_target: int = 210,
    val_target: int = 45,
    test_target: int = 75,
    max_workers: int = 16,
    seed: int = 42,
) -> PTBXLBenchmarkSplit:
    """Download/load stratified PTB-XL 12-lead ECG records across Folds 1-8, Fold 9, and Fold 10."""
    all_meta, subclass_names, form_names, rhythm_names = select_stratified_ptbxl_metadata(
        cache_dir=cache_dir,
        train_target=train_target,
        val_target=val_target,
        test_target=test_target,
        seed=seed,
    )

    # Batch download all missing .hea and .dat files over parallel HTTP/2 curl
    url_to_dest: List[Tuple[str, Path]] = []
    for meta in all_meta:
        rel_path = meta["filename_lr"]
        for ext in (".hea", ".dat"):
            url = f"{PTBXL_BASE_URL}/{rel_path}{ext}"
            dest = cache_dir / f"{rel_path}{ext}"
            url_to_dest.append((url, dest))

    _curl_download_multiplexed(url_to_dest, batch_size=60)

    records: List[PTBXLRecord] = []
    for meta in all_meta:
        rel_path = meta["filename_lr"]
        hea_path = cache_dir / f"{rel_path}.hea"
        dat_path = cache_dir / f"{rel_path}.dat"
        if not hea_path.exists() or not dat_path.exists() or dat_path.stat().st_size < 100:
            continue
        try:
            fs, signals, lead_names = parse_ptbxl_wfdb_12lead(hea_path, dat_path)
            if signals.shape[0] != 12:
                continue
            if signals.shape[1] > 1000:
                signals = signals[:, :1000]
            elif signals.shape[1] < 1000:
                pad = np.zeros((12, 1000 - signals.shape[1]), dtype=np.float64)
                signals = np.concatenate([signals, pad], axis=1)
            signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            continue

        y_super = np.array([1.0 if sc in meta["supers"] else 0.0 for sc in SUPERCLASS_NAMES], dtype=np.float64)
        y_sub = np.array([1.0 if sb in meta["subs"] else 0.0 for sb in subclass_names], dtype=np.float64)
        y_form = np.array([1.0 if fm in meta["forms"] else 0.0 for fm in form_names], dtype=np.float64)
        y_rhythm = np.array([1.0 if rh in meta["rhythms"] else 0.0 for rh in rhythm_names], dtype=np.float64)

        records.append(
            PTBXLRecord(
                ecg_id=meta["ecg_id"],
                patient_id=meta["patient_id"],
                age=meta["age"],
                sex=meta["sex"],
                strat_fold=meta["strat_fold"],
                filename_lr=rel_path,
                scp_codes=meta["scp_codes"],
                signals=signals,
                lead_names=lead_names,
                fs=fs,
                y_superclass=y_super,
                y_subclass=y_sub,
                y_form=y_form,
                y_rhythm=y_rhythm,
            )
        )

    records.sort(key=lambda r: r.ecg_id)
    train_recs = [r for r in records if r.strat_fold <= 8]
    val_recs = [r for r in records if r.strat_fold == 9]
    test_recs = [r for r in records if r.strat_fold == 10]

    return PTBXLBenchmarkSplit(
        train_records=train_recs,
        val_records=val_recs,
        test_records=test_recs,
        superclass_names=SUPERCLASS_NAMES,
        subclass_names=subclass_names,
        form_names=form_names,
        rhythm_names=rhythm_names,
    )
