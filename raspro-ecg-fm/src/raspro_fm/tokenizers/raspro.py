"""Proposed RASPRO Clinical Motif Tokenizer for Biosignal Foundation Models.

Implements the core components of RASPRO (Rapid Active Summarization for effective PROgnosis):
  1. Physician Assist Filters (PAFs): Domain-grounded clinical quantization of
     beat-synchronous ECG and PPG features into severity-graded symbols.
  2. Single-Sensor Matrix (SSM) & Multi-Sensor Matrix (MSM): Structured symbolic
     organization across synchronized physiological channels.
  3. Consensus Normal Motif (CM) & Consensus Abnormality Motif (CAM): Column-wise
     consensus motifs capturing baseline rhythm vs. dominant pathological deviations.
  4. Criticality Measure Index (CMI) & Active Summarization: Quantifies clinical
     acuity per window and deduplicates redundant normal motif runs while preserving
     full temporal resolution on pathological transitions.
"""

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

from ..preprocessing.segmentation import ExtractedBeatFeatures, extract_coupled_beat_features


# Severity weights for clinical symbols:
#   N = Normal (0.0)
#   L = Borderline Low (0.45), H = Borderline High (0.45)
#   D = Critical Low / Depressed (1.0), E = Critical High / Elevated (1.0)
SYMBOL_SEVERITY: Dict[str, float] = {
    "N": 0.0,
    "L": 0.45,
    "H": 0.45,
    "D": 1.0,
    "E": 1.0,
}


@dataclass
class PAFThreshold:
    """Physician Assist Filter threshold boundaries for a single physiological parameter."""
    critical_low: float
    borderline_low: float
    borderline_high: float
    critical_high: float

    def quantize(self, value: float) -> str:
        """Map a continuous physiological measurement into a severity-graded clinical symbol."""
        if value < self.critical_low:
            return "D"  # Critical Low / Depressed
        if value < self.borderline_low:
            return "L"  # Borderline Low
        if value <= self.borderline_high:
            return "N"  # Normal
        if value <= self.critical_high:
            return "H"  # Borderline High
        return "E"      # Critical High / Elevated


DEFAULT_PAF_CONFIG: Dict[str, PAFThreshold] = {
    "rr_interval_ms": PAFThreshold(520.0, 620.0, 1000.0, 1180.0),
    "qrs_duration_ms": PAFThreshold(60.0, 70.0, 112.0, 128.0),
    "st_deviation_mv": PAFThreshold(-0.12, -0.06, 0.06, 0.12),
    "ppg_amp_ratio": PAFThreshold(0.52, 0.76, 1.25, 1.55),
    "ppg_notch_index": PAFThreshold(0.14, 0.24, 0.56, 0.75),
    "pat_ms": PAFThreshold(145.0, 170.0, 228.0, 255.0),
}

PARAMETER_ORDER: List[str] = [
    "rr_interval_ms",
    "qrs_duration_ms",
    "st_deviation_mv",
    "ppg_amp_ratio",
    "ppg_notch_index",
    "pat_ms",
]


@dataclass
class RASPROWindowMotif:
    """RASPRO summary for a temporal window of Multi-Sensor Matrix (MSM) rows."""
    window_idx: int
    msm_rows: List[str]                # Each row is a 6-symbol string [RR, QRS, ST, PPG_AMP, NOTCH, PAT]
    consensus_normal_motif: str        # CM: Dominant normal/baseline symbol per sensor column
    consensus_abnormality_motif: str   # CAM: Dominant abnormal symbol per sensor column (or 'N')
    cmi_score: float                   # Criticality Measure Index in [0.0, 1.0]
    rr_cv: float                       # RR interval coefficient of variation inside window (captures AFib irregularity)
    active_token: str                  # Summarized token emitted to the Foundation Model


@dataclass
class RASPROTokenSequence:
    """Output of RASPRO tokenization on a coupled ECG + PPG recording."""
    beat_motifs: List[str]             # Per-beat 6-symbol MSM strings
    beat_cmi: List[float]              # Per-beat criticality score
    window_motifs: List[RASPROWindowMotif]
    active_tokens: List[str]           # Deduplicated / actively summarized token sequence
    token_ids: List[int]
    token_cmi_weights: List[float]     # CMI weight per active token for CMI-weighted attention/loss
    overall_cmi: float                 # Recording-level Criticality Measure Index
    motif_entropy: float               # Shannon entropy of beat motifs (measures rhythm/morphology complexity)
    n_raw_samples: int
    n_beat_tokens: int
    n_active_tokens: int


class RASPROTokenizer:
    """Hierarchical RASPRO Clinical Summarizer & Tokenizer for ECG + PPG."""

    def __init__(
        self,
        fs: int = 100,
        paf_config: Optional[Dict[str, PAFThreshold]] = None,
        window_beats: int = 4,
        active_dedup_max_run: int = 4,
    ):
        self.fs = fs
        self.paf_config = paf_config or DEFAULT_PAF_CONFIG
        self.window_beats = max(2, window_beats)
        self.active_dedup_max_run = max(1, active_dedup_max_run)
        self.vocab: Dict[str, int] = {"<PAD>": 0, "<MASK>": 1, "<UNK>": 2}

    def _get_or_add_vocab(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def quantize_beat(self, beat: ExtractedBeatFeatures) -> Tuple[str, float]:
        """Pass a single cardiac cycle through the Physician Assist Filters (PAFs)."""
        vals = {
            "rr_interval_ms": beat.rr_interval_ms,
            "qrs_duration_ms": beat.qrs_duration_ms,
            "st_deviation_mv": beat.st_deviation_mv,
            "ppg_amp_ratio": beat.ppg_amp_ratio,
            "ppg_notch_index": beat.ppg_notch_index,
            "pat_ms": beat.pat_ms,
        }
        syms: List[str] = []
        sev_sum = 0.0
        for param in PARAMETER_ORDER:
            sym = self.paf_config[param].quantize(vals[param])
            syms.append(sym)
            sev_sum += SYMBOL_SEVERITY[sym]

        motif_str = "".join(syms)
        cmi_beat = sev_sum / len(PARAMETER_ORDER)
        return motif_str, cmi_beat

    def _summarize_msm_window(
        self,
        window_idx: int,
        rows: List[str],
        beats_slice: List[ExtractedBeatFeatures],
    ) -> RASPROWindowMotif:
        """Compute Consensus Normal Motif (CM), Consensus Abnormality Motif (CAM), and CMI for an MSM window."""
        n_cols = len(PARAMETER_ORDER)
        cm_chars: List[str] = []
        cam_chars: List[str] = []
        abnormal_severity_total = 0.0

        for col in range(n_cols):
            col_syms = [r[col] for r in rows]
            normal_syms = [s for s in col_syms if s == "N"]
            abn_syms = [s for s in col_syms if s != "N"]

            # Consensus Normal Motif (CM) column entry
            if normal_syms:
                cm_chars.append("N")
            else:
                cm_chars.append(Counter(col_syms).most_common(1)[0][0])

            # Consensus Abnormality Motif (CAM) column entry
            if abn_syms:
                dominant_abn = Counter(abn_syms).most_common(1)[0][0]
                cam_chars.append(dominant_abn)
            else:
                cam_chars.append("N")

            for s in col_syms:
                abnormal_severity_total += SYMBOL_SEVERITY[s]

        # Calculate RR interval variability within window (critical for AFib / ectopy detection)
        rr_vals = [b.rr_interval_ms for b in beats_slice]
        if len(rr_vals) >= 2:
            rr_mean = sum(rr_vals) / len(rr_vals)
            rr_std = math.sqrt(sum((x - rr_mean) ** 2 for x in rr_vals) / len(rr_vals))
            rr_cv = rr_std / max(100.0, rr_mean)
        else:
            rr_cv = 0.0

        # If high beat-to-beat RR irregularity is present, reflect it in the rhythm slot of CAM
        if rr_cv > 0.14 and cam_chars[0] == "N":
            cam_chars[0] = "L" if rr_cv < 0.22 else "D"
            abnormal_severity_total += len(rows) * (0.55 if rr_cv < 0.22 else 1.0)

        cm_str = "".join(cm_chars)
        cam_str = "".join(cam_chars)
        cmi_score = min(1.0, abnormal_severity_total / max(1, len(rows) * n_cols))

        # Rhythm irregularity tag for the motif token
        rhythm_tag = "IRR" if rr_cv > 0.14 else "REG"
        active_tok = f"CAM:{cam_str}|{rhythm_tag}" if cam_str != "NNNNNN" else f"CM:{cm_str}|{rhythm_tag}"

        return RASPROWindowMotif(
            window_idx=window_idx,
            msm_rows=rows,
            consensus_normal_motif=cm_str,
            consensus_abnormality_motif=cam_str,
            cmi_score=cmi_score,
            rr_cv=rr_cv,
            active_token=active_tok,
        )

    def tokenize(
        self,
        ecg: List[float],
        ppg: List[float],
        r_peaks: Optional[List[int]] = None,
    ) -> RASPROTokenSequence:
        """Transform synchronized ECG + PPG waveforms into actively summarized RASPRO motif tokens."""
        beats = extract_coupled_beat_features(ecg, ppg, fs=self.fs, r_peaks=r_peaks)
        n_raw = len(ecg) + len(ppg)

        if not beats:
            tok = "CM:NNNNNN|REG"
            tid = self._get_or_add_vocab(tok)
            return RASPROTokenSequence(
                beat_motifs=[ "NNNNNN" ],
                beat_cmi=[0.0],
                window_motifs=[],
                active_tokens=[tok],
                token_ids=[tid],
                token_cmi_weights=[0.0],
                overall_cmi=0.0,
                motif_entropy=0.0,
                n_raw_samples=n_raw,
                n_beat_tokens=1,
                n_active_tokens=1,
            )

        beat_motifs: List[str] = []
        beat_cmi: List[float] = []
        for b in beats:
            m_str, cmi_val = self.quantize_beat(b)
            beat_motifs.append(m_str)
            beat_cmi.append(cmi_val)

        # Group beats into Multi-Sensor Matrix (MSM) windows
        window_motifs: List[RASPROWindowMotif] = []
        for w_idx, start in enumerate(range(0, len(beat_motifs), self.window_beats)):
            end = min(len(beat_motifs), start + self.window_beats)
            w_motif = self._summarize_msm_window(
                window_idx=w_idx,
                rows=beat_motifs[start:end],
                beats_slice=beats[start:end],
            )
            window_motifs.append(w_motif)

        # Active Summarization: Collapse consecutive stationary normal motif runs (CM),
        # while keeping full resolution on abnormal motifs (CAM) and transition boundaries
        active_tokens: List[str] = []
        token_cmi_weights: List[float] = []
        i = 0
        while i < len(window_motifs):
            wm = window_motifs[i]
            if wm.active_token.startswith("CM:NNNNNN|REG"):
                run_len = 1
                while (
                    i + run_len < len(window_motifs)
                    and window_motifs[i + run_len].active_token == wm.active_token
                    and run_len < self.active_dedup_max_run
                ):
                    run_len += 1
                dedup_tok = f"{wm.active_token}x{run_len}"
                active_tokens.append(dedup_tok)
                token_cmi_weights.append(wm.cmi_score)
                i += run_len
            else:
                active_tokens.append(wm.active_token)
                token_cmi_weights.append(wm.cmi_score)
                i += 1

        token_ids = [self._get_or_add_vocab(t) for t in active_tokens]

        # Compute overall Criticality Measure Index (CMI) and Shannon motif entropy
        overall_cmi = sum(w.cmi_score for w in window_motifs) / max(1, len(window_motifs))
        counts = Counter(beat_motifs)
        total_b = len(beat_motifs)
        entropy = -sum((c / total_b) * math.log2(c / total_b) for c in counts.values())

        return RASPROTokenSequence(
            beat_motifs=beat_motifs,
            beat_cmi=beat_cmi,
            window_motifs=window_motifs,
            active_tokens=active_tokens,
            token_ids=token_ids,
            token_cmi_weights=token_cmi_weights,
            overall_cmi=overall_cmi,
            motif_entropy=entropy,
            n_raw_samples=n_raw,
            n_beat_tokens=len(beat_motifs),
            n_active_tokens=len(active_tokens),
        )


def select_cmi_coreset(
    tokenized_records: List[Tuple[int, RASPROTokenSequence]],
    fraction: float,
) -> List[int]:
    """Select the top `fraction` most informative pre-training records via CMI-Weighted Submodular Coverage.

    Uses a greedy submodular maximum-coverage objective over both whole RASPRO active tokens
    and column-level Multi-Sensor Matrix (MSM) sub-motifs, weighted by corpus prevalence and
    Criticality Measure Index (CMI). This ensures the selected coreset maximizes downstream
    vocabulary coverage while prioritizing clinically critical pathological transitions.
    """
    n_total = len(tokenized_records)
    k = max(5, int(round(n_total * fraction)))
    if k >= n_total:
        return [idx for idx, _ in tokenized_records]

    # 1. Extract the set of whole tokens and factorized MSM slot features per record,
    #    and compute their corpus-wide utility weights.
    feature_freq: Counter = Counter()
    feature_cmi: Dict[str, float] = {}
    record_features: List[Tuple[int, set, float]] = []

    for rec_idx, seq in tokenized_records:
        feats = set()
        for tok, w_cmi in zip(seq.active_tokens, seq.token_cmi_weights):
            feats.add(tok)
            feature_freq[tok] += 1
            feature_cmi[tok] = max(feature_cmi.get(tok, 0.0), w_cmi)
            # Also add factorized column-pair sub-motifs if structured CM/CAM token
            if ":" in tok and "|" in tok:
                core = tok.split(":", 1)[1].split("|", 1)[0]
                if len(core) in (6, 8, 9):
                    for c_i in range(len(core)):
                        sf = f"C{c_i}:{core[c_i]}"
                        feats.add(sf)
                        feature_freq[sf] += 0.5
                        feature_cmi[sf] = max(feature_cmi.get(sf, 0.0), w_cmi)
        record_features.append((rec_idx, feats, seq.overall_cmi))

    # Utility of covering each motif/sub-motif combines corpus frequency and CMI criticality
    feature_utility: Dict[str, float] = {
        f: (freq ** 0.65) * (1.0 + 2.5 * feature_cmi.get(f, 0.0))
        for f, freq in feature_freq.items()
    }

    # 2. Greedy submodular maximum-coverage selection with diminishing returns
    covered_counts: Counter = Counter()
    selected: List[int] = []
    remaining = list(record_features)

    while len(selected) < k and remaining:
        best_pos = 0
        best_gain = -1.0
        for pos, (rec_idx, feats, rec_cmi) in enumerate(remaining):
            gain = 0.0
            for f in feats:
                # Diminishing marginal gain: 1.0 on first coverage, 0.15 on second, 0.03 thereafter
                c_cnt = covered_counts[f]
                if c_cnt == 0:
                    gain += feature_utility[f]
                elif c_cnt == 1:
                    gain += 0.15 * feature_utility[f]
                else:
                    gain += 0.02 * feature_utility[f]
            # Slight tie-breaker favoring higher CMI acuity
            gain += 0.05 * rec_cmi
            if gain > best_gain:
                best_gain = gain
                best_pos = pos

        chosen_idx, chosen_feats, _ = remaining.pop(best_pos)
        selected.append(chosen_idx)
        for f in chosen_feats:
            covered_counts[f] += 1

    return selected
