"""12-Lead Clinical ECG Tokenizers for PTB-XL Benchmark (Phase 3).

Implements apples-to-apples 12-lead tokenizers:
  1. RASPRO12LeadTokenizer: Clinical anatomical 12-lead PAF + MSM + CM/CAM + CMI
  2. HeartLangQRSTokenizer: Beat-synchronous R-peak slicing + learned VQ codebook ("ECG words")
"""

from collections import Counter
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from raspro_fm.preprocessing.real_signal import detect_real_r_peaks
from raspro_fm.tokenizers.raspro import (
    PAFThreshold,
    RASPROTokenSequence,
    RASPROWindowMotif,
    SYMBOL_SEVERITY,
)


# Default 12-Lead PTB-XL Anatomical PAF Thresholds (6/8/9-Slot Clinical Configuration)
PTBXL_12LEAD_PAFS: Dict[str, PAFThreshold] = {
    "RR": PAFThreshold(0.58, 0.67, 0.96, 1.12),
    "QRS": PAFThreshold(0.052, 0.064, 0.104, 0.122),
    "ST_INF": PAFThreshold(-0.11, -0.045, 0.055, 0.12),
    "ST_ANT": PAFThreshold(-0.12, -0.050, 0.075, 0.16),
    "ST_LAT": PAFThreshold(-0.11, -0.045, 0.055, 0.12),
    "VOLT_HYP": PAFThreshold(0.85, 1.25, 2.55, 3.15),
    "PR_COND": PAFThreshold(-0.08, -0.03, 0.06, 0.12),
    "QRS_AXIS": PAFThreshold(-0.42, -0.18, 0.16, 0.36),
    "QT_MORPH": PAFThreshold(-0.38, -0.16, 0.24, 0.52),
}


@dataclass
class Beat12LeadFeatures:
    """Anatomical 12-lead electrophysiological features for a single heartbeat."""
    beat_idx: int
    r_peak_idx: int
    rr_interval_s: float
    qrs_duration_s: float
    st_inf: float       # Leads II, III, aVF ST-T elevation/depression & Q-wave score
    st_ant: float       # Leads V1-V4 precordial ST-T & anteroseptal R-wave progression score
    st_lat: float       # Leads I, aVL, V5, V6 lateral ST-T & T-wave concordance score
    volt_hyp: float     # Sokolow-Lyon / Cornell ventricular & atrial voltage index (mV)
    pr_cond: float      # PR conduction interval & V1/II atrial P-wave morphology index
    qrs_axis: float     # Precordial V1-vs-V6 terminal polarity & frontal I/aVF axis index
    raw_beat_vec: np.ndarray  # Compact beat waveform descriptor for HeartLang VQ codebook
    qt_morph: float = 0.0  # Pathological Q-wave / QS ratio, precordial R-progression & T-wave inversion index


def _fast_detrend_12lead(signals: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Fast vectorized baseline wander removal across all 12 leads."""
    win = max(15, int(0.60 * fs))
    step = 4
    sub = signals[:, ::step]
    w_sub = max(3, win // step)
    pad = w_sub // 2
    padded = np.pad(sub, ((0, 0), (pad, pad)), mode="edge")
    shape = (sub.shape[0], sub.shape[1], w_sub)
    strides = (padded.strides[0], padded.strides[1], padded.strides[1])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    base_sub = np.median(windows[:, : sub.shape[1], :], axis=-1)
    x_sub = np.arange(sub.shape[1]) * step
    x_full = np.arange(signals.shape[1])
    cleaned = np.zeros_like(signals, dtype=np.float64)
    for c in range(signals.shape[0]):
        cleaned[c] = signals[c] - np.interp(x_full, x_sub, base_sub[c])
    return cleaned


def extract_12lead_beats_and_summary(
    signals: np.ndarray,
    fs: float = 100.0,
) -> Tuple[List[Beat12LeadFeatures], np.ndarray]:
    """Extract beat-synchronous 12-lead anatomical features.

    Standard PTB-XL lead order:
      0: I, 1: II, 2: III, 3: AVR, 4: AVL, 5: AVF,
      6: V1, 7: V2, 8: V3, 9: V4, 10: V5, 11: V6
    """
    cleaned = _fast_detrend_12lead(signals, fs)

    composite_lead = (0.5 * cleaned[1] + 0.3 * cleaned[10] - 0.2 * cleaned[3]).tolist()
    r_peaks = detect_real_r_peaks(composite_lead, fs=int(fs), refractory_sec=0.28)
    if len(r_peaks) < 2:
        r_peaks = list(range(int(0.4 * fs), signals.shape[1] - int(0.4 * fs), int(0.80 * fs)))

    beats: List[Beat12LeadFeatures] = []
    qrs_half = max(3, int(0.06 * fs))
    st_start = max(4, int(0.06 * fs))
    st_end = max(8, int(0.14 * fs))
    t_start = max(9, int(0.15 * fs))
    t_end = max(16, int(0.30 * fs))

    for b_idx, r_idx in enumerate(r_peaks):
        if b_idx == 0:
            rr_s = (r_peaks[1] - r_peaks[0]) / fs if len(r_peaks) > 1 else 0.80
        else:
            rr_s = (r_idx - r_peaks[b_idx - 1]) / fs
        rr_s = float(np.clip(rr_s, 0.30, 2.00))

        # QRS width & terminal conduction delay across Leads II(1), V1(6), V5(10), I(0)
        q_lo = max(0, r_idx - qrs_half)
        q_hi = min(cleaned.shape[1], r_idx + qrs_half + 2)
        qrs_widths = []
        for ld in [0, 1, 6, 10]:
            seg = np.abs(cleaned[ld, q_lo:q_hi])
            pk = float(np.max(seg)) if seg.size > 0 else 0.0
            if pk > 1e-4:
                qrs_widths.append(float(np.sum(seg >= 0.25 * pk)) / fs)
        qrs_dur_s = float(np.clip(np.mean(qrs_widths) if qrs_widths else 0.082, 0.040, 0.20))

        # PR / Isoelectric baseline window [-140 ms to -50 ms] before R-peak
        pr_lo = max(0, r_idx - int(0.14 * fs))
        pr_hi = max(pr_lo + 1, r_idx - int(0.05 * fs))
        iso = np.median(cleaned[:, pr_lo:pr_hi], axis=1) if pr_hi > pr_lo else np.zeros(12)

        # ST-segment and T-wave windows referenced to isoelectric PR baseline
        s_lo = min(cleaned.shape[1] - 2, r_idx + st_start)
        s_hi = min(cleaned.shape[1], r_idx + st_end)
        t_lo = min(cleaned.shape[1] - 2, r_idx + t_start)
        t_hi = min(cleaned.shape[1], r_idx + t_end)

        st_vals = np.mean(cleaned[:, s_lo:s_hi], axis=1) - iso if s_hi > s_lo else np.zeros(12)
        t_vals = np.mean(cleaned[:, t_lo:t_hi], axis=1) - iso if t_hi > t_lo else np.zeros(12)

        # Initial Q-wave window [-45 ms to -15 ms] before R-peak (avoids V1/V2 S-wave after R-peak)
        qw_lo = max(0, r_idx - int(0.045 * fs))
        qw_hi = max(qw_lo + 1, r_idx - int(0.015 * fs))
        q_init = np.min(cleaned[:, qw_lo:qw_hi] - iso[:, None], axis=1)

        # 1. Inferior coronary territory (Leads II=1, III=2, aVF=5): ST shift + T-wave + inferior Q-wave
        st_inf = float(np.mean(st_vals[[1, 2, 5]]) + 0.35 * np.mean(t_vals[[1, 2, 5]]) + 0.25 * np.mean(q_init[[1, 2, 5]]))

        # 2. Anterior/Anteroseptal territory (Leads V1=6, V2=7, V3=8, V4=9): ST shift + T-wave + anterior Q-wave
        st_ant = float(np.mean(st_vals[[6, 7, 8, 9]]) + 0.35 * np.mean(t_vals[[7, 8, 9]]) + 0.20 * np.mean(q_init[[6, 7, 8]]))

        # 3. Lateral territory (Leads I=0, aVL=4, V5=10, V6=11): ST shift + T-wave concordance + lateral Q-wave
        st_lat = float(np.mean(st_vals[[0, 4, 10, 11]]) + 0.45 * np.mean(t_vals[[0, 4, 10, 11]]) + 0.15 * np.mean(q_init[[0, 4, 10, 11]]))

        # 4. Sokolow-Lyon + Cornell Hypertrophy Voltage Index
        s_v1 = abs(float(np.min(cleaned[6, q_lo:q_hi] - iso[6]))) if q_hi > q_lo else 0.5
        s_v3 = abs(float(np.min(cleaned[8, q_lo:q_hi] - iso[8]))) if q_hi > q_lo else 0.5
        r_v5 = abs(float(np.max(cleaned[10, q_lo:q_hi] - iso[10]))) if q_hi > q_lo else 0.8
        r_v6 = abs(float(np.max(cleaned[11, q_lo:q_hi] - iso[11]))) if q_hi > q_lo else 0.8
        r_avl = abs(float(np.max(cleaned[4, q_lo:q_hi] - iso[4]))) if q_hi > q_lo else 0.3
        volt_hyp = float(0.65 * (s_v1 + max(r_v5, r_v6)) + 0.35 * (r_avl + s_v3))

        # 5. PR Conduction & P-Wave Atrial Morphology Index (_AVB, WPW/LPR, LAO/LAE)
        pe_lo = max(0, r_idx - int(0.21 * fs))
        pe_hi = max(pe_lo + 1, r_idx - int(0.13 * fs))
        pn_lo = max(0, r_idx - int(0.13 * fs))
        pn_hi = max(pn_lo + 1, r_idx - int(0.06 * fs))
        p_early = float(np.mean(np.abs(cleaned[[1, 6], pe_lo:pe_hi] - iso[[1, 6], None]))) if pe_hi > pe_lo else 0.0
        p_norm = float(np.mean(np.abs(cleaned[[1, 6], pn_lo:pn_hi] - iso[[1, 6], None]))) if pn_hi > pn_lo else 0.0
        v1_p_neg = float(np.min(cleaned[6, pn_lo:pn_hi] - iso[6])) if pn_hi > pn_lo else 0.0
        pr_cond = float((p_early - 0.55 * p_norm) - 0.45 * v1_p_neg)

        # 6. Precordial V1-vs-V6 Polarity & Frontal QRS Axis Index (CLBBB vs CRBBB vs LAFB vs RVH)
        net_qrs = np.mean(cleaned[:, q_lo:q_hi] - iso[:, None], axis=1) if q_hi > q_lo else np.zeros(12)
        precordial_pol = float(net_qrs[6] - 0.50 * net_qrs[11])  # Positive in RBBB/RVH, negative in LBBB
        frontal_axis = float(0.5 * (net_qrs[5] + net_qrs[2]) - 0.5 * (net_qrs[0] + net_qrs[4]))  # Negative in LAFB
        qrs_axis = float(0.55 * precordial_pol + 0.45 * frontal_axis)

        # 7. Pathological Q-Wave / QS Ratio, Precordial R-Wave Progression & T-Wave Inversion (QT_MORPH for Evolved MI)
        qrs_seg = cleaned[:, q_lo:q_hi] - iso[:, None] if q_hi > q_lo else np.zeros((12, 1))
        eq_lo = max(0, r_idx - int(0.045 * fs))
        eq_hi = max(eq_lo + 1, r_idx + 1)
        early_qrs = cleaned[:, eq_lo:eq_hi] - iso[:, None]
        r_pos = np.max(np.maximum(0.0, qrs_seg), axis=1)
        q_neg = np.max(np.maximum(0.0, -early_qrs), axis=1)
        s_neg = np.max(np.maximum(0.0, -qrs_seg), axis=1)
        t_min = np.min(cleaned[:, t_lo:t_hi] - iso[:, None], axis=1) if t_hi > t_lo else np.zeros(12)

        q_rat = q_neg / (q_neg + r_pos + 0.12)
        r_prog = 0.5 * (
            float(r_pos[7] / (r_pos[7] + s_neg[7] + 0.10))
            + float(r_pos[8] / (r_pos[8] + s_neg[8] + 0.10))
        )
        imi_q_idx = float(0.45 * (q_rat[2] + q_rat[5] + 0.5 * q_rat[1]) + 0.25 * max(0.0, -float(t_min[2] + t_min[5])))
        asmi_q_idx = float(
            0.45 * (q_rat[6] + q_rat[7] + q_rat[8])
            + 0.45 * (1.0 - r_prog)
            + 0.20 * max(0.0, -float(t_min[6] + t_min[7] + t_min[8]))
        )
        qt_morph = float(0.45 * imi_q_idx + 0.55 * asmi_q_idx - 0.68)

        raw_vec = np.array([
            rr_s, qrs_dur_s * 10.0,
            float(st_vals[1]), float(st_vals[7]), float(st_vals[10]),
            float(t_vals[1]), float(t_vals[7]), float(t_vals[10]),
            float(q_init[1]), float(q_init[7]),
            s_v1, max(r_v5, r_v6)
        ], dtype=np.float64)

        beats.append(
            Beat12LeadFeatures(
                beat_idx=b_idx,
                r_peak_idx=int(r_idx),
                rr_interval_s=rr_s,
                qrs_duration_s=qrs_dur_s,
                st_inf=st_inf,
                st_ant=st_ant,
                st_lat=st_lat,
                volt_hyp=volt_hyp,
                pr_cond=pr_cond,
                qrs_axis=qrs_axis,
                raw_beat_vec=raw_vec,
                qt_morph=qt_morph,
            )
        )

    summary_vec = np.zeros(12, dtype=np.float64)
    return beats, summary_vec


class RASPRO12LeadTokenizer:
    """12-Lead Clinical RASPRO Tokenizer (PAF + 6/8/9-Slot Anatomical MSM + CM/CAM + CMI)."""

    def __init__(
        self,
        pafs: Optional[Dict[str, PAFThreshold]] = None,
        msm_window_beats: int = 4,
        merge_consecutive_cm: bool = True,
        n_slots: int = 8,
        max_cm_run: int = 4,
    ):
        self.pafs = dict(pafs) if pafs is not None else dict(PTBXL_12LEAD_PAFS)
        full_order = ["RR", "QRS", "ST_INF", "ST_ANT", "ST_LAT", "VOLT_HYP", "PR_COND", "QRS_AXIS", "QT_MORPH"]
        if n_slots >= 9:
            self.n_slots = 9
        elif n_slots >= 8:
            self.n_slots = 8
        else:
            self.n_slots = 6
        self.param_order = full_order[: self.n_slots]
        self.msm_window_beats = msm_window_beats
        self.merge_consecutive_cm = merge_consecutive_cm
        self.max_cm_run = max_cm_run

    def calibrate_unsupervised_pafs(self, beat_lists: List[List[Beat12LeadFeatures]]) -> None:
        """Calibrate PAF boundaries by blending clinical textbook PAFs with unlabeled cohort quantiles."""
        all_beats = [b for bl in beat_lists for b in bl]
        if len(all_beats) < 20:
            return
        param_arrays = {
            "RR": np.array([b.rr_interval_s for b in all_beats], dtype=np.float64),
            "QRS": np.array([b.qrs_duration_s for b in all_beats], dtype=np.float64),
            "ST_INF": np.array([b.st_inf for b in all_beats], dtype=np.float64),
            "ST_ANT": np.array([b.st_ant for b in all_beats], dtype=np.float64),
            "ST_LAT": np.array([b.st_lat for b in all_beats], dtype=np.float64),
            "VOLT_HYP": np.array([b.volt_hyp for b in all_beats], dtype=np.float64),
            "PR_COND": np.array([b.pr_cond for b in all_beats], dtype=np.float64),
            "QRS_AXIS": np.array([b.qrs_axis for b in all_beats], dtype=np.float64),
            "QT_MORPH": np.array([b.qt_morph for b in all_beats], dtype=np.float64),
        }
        for p_name in self.param_order:
            arr = param_arrays[p_name]
            q05, q16, q84, q95 = np.percentile(arr, [5.0, 16.0, 84.0, 95.0])
            ref = PTBXL_12LEAD_PAFS[p_name]
            self.pafs[p_name] = PAFThreshold(
                critical_low=float(0.60 * ref.critical_low + 0.40 * q05),
                borderline_low=float(0.60 * ref.borderline_low + 0.40 * q16),
                borderline_high=float(0.60 * ref.borderline_high + 0.40 * q84),
                critical_high=float(0.60 * ref.critical_high + 0.40 * q95),
            )

    def quantize_beat(self, b: Beat12LeadFeatures) -> Tuple[str, float]:
        vals = {
            "RR": b.rr_interval_s,
            "QRS": b.qrs_duration_s,
            "ST_INF": b.st_inf,
            "ST_ANT": b.st_ant,
            "ST_LAT": b.st_lat,
            "VOLT_HYP": b.volt_hyp,
            "PR_COND": b.pr_cond,
            "QRS_AXIS": b.qrs_axis,
            "QT_MORPH": b.qt_morph,
        }
        chars = [self.pafs[p].quantize(vals[p]) for p in self.param_order]
        code = "".join(chars)
        cmi = float(np.mean([SYMBOL_SEVERITY[c] for c in chars]))
        return code, cmi

    def tokenize(self, beats: List[Beat12LeadFeatures]) -> RASPROTokenSequence:
        default_norm = "N" * self.n_slots
        if not beats:
            wm = RASPROWindowMotif(
                window_idx=0,
                msm_rows=[default_norm],
                consensus_normal_motif=default_norm,
                consensus_abnormality_motif=default_norm,
                cmi_score=0.0,
                rr_cv=0.0,
                active_token=f"CM:{default_norm}|REG",
            )
            return RASPROTokenSequence(
                beat_motifs=[default_norm],
                beat_cmi=[0.0],
                window_motifs=[wm],
                active_tokens=[f"CM:{default_norm}|REG"],
                token_ids=[3],
                token_cmi_weights=[0.0],
                overall_cmi=0.0,
                motif_entropy=0.0,
                n_raw_samples=12000,
                n_beat_tokens=1,
                n_active_tokens=1,
            )

        beat_codes = []
        beat_cmis = []
        for b in beats:
            code, cmi = self.quantize_beat(b)
            beat_codes.append(code)
            beat_cmis.append(cmi)

        window_motifs: List[RASPROWindowMotif] = []
        step = max(1, self.msm_window_beats)

        for w_idx, start in enumerate(range(0, len(beats), step)):
            w_codes = beat_codes[start : start + self.msm_window_beats]
            w_beats = beats[start : start + self.msm_window_beats]
            if not w_codes:
                continue

            n_cols = len(self.param_order)
            consensus_chars = []
            aberrant_chars = []
            is_aberrant = False
            total_sev = 0.0

            rr_vals = [wb.rr_interval_s for wb in w_beats]
            rr_cv = float(np.std(rr_vals) / (np.mean(rr_vals) + 1e-6)) if len(rr_vals) > 1 else 0.0
            rhythm_irreg = rr_cv > 0.11

            for col in range(n_cols):
                col_syms = [c[col] for c in w_codes]
                counts = Counter(col_syms)
                mode_sym = sorted(
                    counts.keys(),
                    key=lambda s: (counts[s], SYMBOL_SEVERITY[s]),
                    reverse=True,
                )[0]
                consensus_chars.append(mode_sym)

                worst_sym = max(col_syms, key=lambda s: SYMBOL_SEVERITY[s])
                min_req = 1 if (rhythm_irreg or len(w_codes) <= 2) else 2
                if worst_sym in ("D", "E") and counts[worst_sym] >= min_req:
                    aberrant_chars.append(worst_sym)
                    is_aberrant = True
                else:
                    aberrant_chars.append(mode_sym)
                total_sev += sum(SYMBOL_SEVERITY[s] for s in col_syms)

            cmi_w = total_sev / (len(w_codes) * n_cols)
            if rhythm_irreg:
                is_aberrant = True
                cmi_w = min(1.0, cmi_w + (0.20 if rr_cv > 0.20 else 0.12))
                if aberrant_chars[0] == "N":
                    aberrant_chars[0] = "D" if rr_cv > 0.20 else "L"

            cm_str = "".join(consensus_chars)
            cam_str = "".join(aberrant_chars)
            if rr_cv > 0.20:
                rhythm_tag = "IRR"
            elif rr_cv > 0.11:
                rhythm_tag = "ECT"
            else:
                rhythm_tag = "REG"
            if is_aberrant:
                tok = f"CAM:{cam_str}|{rhythm_tag}"
            else:
                tok = f"CM:{cm_str}|{rhythm_tag}"

            window_motifs.append(
                RASPROWindowMotif(
                    window_idx=w_idx,
                    msm_rows=w_codes,
                    consensus_normal_motif=cm_str,
                    consensus_abnormality_motif=cam_str,
                    cmi_score=float(cmi_w),
                    rr_cv=rr_cv,
                    active_token=tok,
                )
            )

        active_tokens: List[str] = []
        token_cmi_weights: List[float] = []
        i = 0
        while i < len(window_motifs):
            wm = window_motifs[i]
            if self.merge_consecutive_cm and self.max_cm_run > 1 and wm.active_token.startswith("CM:"):
                run_len = 1
                cmi_acc = [wm.cmi_score]
                while (
                    i + run_len < len(window_motifs)
                    and run_len < self.max_cm_run
                    and window_motifs[i + run_len].active_token == wm.active_token
                ):
                    cmi_acc.append(window_motifs[i + run_len].cmi_score)
                    run_len += 1
                dedup_tok = f"{wm.active_token}x{run_len}" if run_len > 1 else wm.active_token
                active_tokens.append(dedup_tok)
                token_cmi_weights.append(float(np.mean(cmi_acc)))
                i += run_len
            else:
                active_tokens.append(wm.active_token)
                token_cmi_weights.append(wm.cmi_score)
                i += 1

        counts = Counter(beat_codes)
        total_b = len(beat_codes)
        entropy = -sum((c / total_b) * math.log2(c / total_b) for c in counts.values())
        overall_cmi = float(np.mean([w.cmi_score for w in window_motifs])) if window_motifs else 0.0

        return RASPROTokenSequence(
            beat_motifs=beat_codes,
            beat_cmi=beat_cmis,
            window_motifs=window_motifs,
            active_tokens=active_tokens,
            token_ids=list(range(len(active_tokens))),
            token_cmi_weights=token_cmi_weights,
            overall_cmi=overall_cmi,
            motif_entropy=entropy,
            n_raw_samples=12000,
            n_beat_tokens=len(beat_codes),
            n_active_tokens=len(active_tokens),
        )


class HeartLangQRSTokenizer:
    """HeartLang-style QRS Heartbeat Tokenizer (ICLR 2025 baseline).

    Segments 12-lead ECGs into individual R-peak-centered heartbeats ("ECG words")
    and quantizes each heartbeat into a discrete codebook token via K-means VQ
    without clinical PAF thresholds or RASPRO CM/CAM consensus merging.
    """

    def __init__(self, codebook_size: int = 64, seed: int = 42):
        self.codebook_size = codebook_size
        self.seed = seed
        self.centroids: np.ndarray = np.zeros((codebook_size, 12), dtype=np.float64)
        self.mean: np.ndarray = np.zeros(12, dtype=np.float64)
        self.std: np.ndarray = np.ones(12, dtype=np.float64)

    def fit(self, beat_lists: List[List[Beat12LeadFeatures]], n_iters: int = 18) -> None:
        all_vecs = [b.raw_beat_vec for bl in beat_lists for b in bl]
        if not all_vecs:
            return
        mat = np.stack(all_vecs, axis=0)
        self.mean = np.mean(mat, axis=0)
        self.std = np.std(mat, axis=0) + 1e-6
        z = np.clip((mat - self.mean) / self.std, -4.0, 4.0)

        rng = np.random.default_rng(self.seed)
        k = min(self.codebook_size, len(z))
        init_idx = rng.choice(len(z), size=k, replace=False)
        centroids = z[init_idx].copy()

        for _ in range(n_iters):
            dists = np.sum((z[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
            assign = np.argmin(dists, axis=1)
            for c in range(k):
                mask = assign == c
                if np.any(mask):
                    centroids[c] = np.mean(z[mask], axis=0)
        self.centroids = centroids

    def tokenize(self, beats: List[Beat12LeadFeatures]) -> List[str]:
        if not beats:
            return ["HL_W00"]
        tokens = []
        for b in beats:
            z = np.clip((b.raw_beat_vec - self.mean) / self.std, -4.0, 4.0)
            dists = np.sum((self.centroids - z[None, :]) ** 2, axis=1)
            cid = int(np.argmin(dists))
            tokens.append(f"HL_W{cid:02d}")
        return tokens

    def get_codebook_anchors(self, embed_dim: int = 48) -> Dict[str, List[float]]:
        """Project learned VQ centroids into embed_dim anchor vectors."""
        rng = np.random.default_rng(self.seed + 7)
        proj = rng.normal(0.0, 1.0 / math.sqrt(12.0), size=(12, embed_dim))
        anchors: Dict[str, List[float]] = {}
        for cid in range(len(self.centroids)):
            vec = (self.centroids[cid] @ proj).tolist()
            anchors[f"HL_W{cid:02d}"] = vec
        return anchors
