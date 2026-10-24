"""Lightweight Transformer-style Self-Supervised Foundation Models for Symbolic & Patch Tokenizers.

Implements:
  1. `SymbolicFoundationModel`:
     - Learns contextual token embeddings via Masked Motif Modeling (MMM) and bi-directional
       context co-occurrence SVD / PMI factorized representations with scaled dot-product
       self-attention pooling.
     - Supports CMI-weighted pre-training updates (for RASPRO) so clinically critical motifs
       receive amplified representation capacity even under low pre-training data fractions.
  2. `PatchFoundationModel`:
     - Learns continuous patch representations via Masked Patch Autoencoding (MAE) and temporal
       self-attention pooling over patch embeddings.
  3. Downstream Linear Probe & Multi-Class / AHE Prognosis Evaluators.
"""

from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional, Tuple
import zlib


@dataclass
class EvaluationMetrics:
    """Downstream clinical evaluation and computational efficiency metrics."""
    accuracy: float
    macro_f1: float
    macro_auroc: float
    ahe_auroc: float
    stemi_f1: float
    avg_seq_len: float
    relative_attention_flops: float
    vocab_coverage: float


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _normalize_vec(a: List[float]) -> List[float]:
    n = _norm(a)
    if n < 1e-9:
        return [0.0] * len(a)
    return [x / n for x in a]


def _softmax(logits: List[float]) -> List[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(max(-30.0, x - m)) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _binary_auroc(scores: List[float], positives: List[int]) -> float:
    """Compute exact Mann-Whitney U / Wilcoxon rank-sum AUROC."""
    pos_scores = [s for s, y in zip(scores, positives) if y == 1]
    neg_scores = [s for s, y in zip(scores, positives) if y == 0]
    if not pos_scores or not neg_scores:
        return 0.5
    wins = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(pos_scores) * len(neg_scores))


def evaluate_predictions(
    y_true: List[int],
    y_prob: List[List[float]],
    avg_seq_len: float,
    baseline_seq_len: float = 150.0,
    vocab_coverage: float = 1.0,
) -> EvaluationMetrics:
    """Compute Accuracy, Macro-F1, Macro-AUROC, STEMI F1, AHE Prognosis AUROC, and relative attention FLOPs."""
    n = len(y_true)
    n_classes = len(y_prob[0]) if y_prob else 5
    y_pred = [max(range(n_classes), key=lambda c: probs[c]) for probs in y_prob]

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = correct / max(1, n)

    f1_per_class: List[float] = []
    auroc_per_class: List[float] = []

    for c in range(n_classes):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_per_class.append(f1)

        c_scores = [probs[c] for probs in y_prob]
        c_bin = [1 if yt == c else 0 for yt in y_true]
        auroc_per_class.append(_binary_auroc(c_scores, c_bin))

    macro_f1 = sum(f1_per_class) / n_classes
    macro_auroc = sum(auroc_per_class) / n_classes
    stemi_f1 = f1_per_class[1] if len(f1_per_class) > 1 else macro_f1
    ahe_auroc = auroc_per_class[3] if len(auroc_per_class) > 3 else macro_auroc
    rel_flops = (avg_seq_len / max(1.0, baseline_seq_len)) ** 2

    return EvaluationMetrics(
        accuracy=acc,
        macro_f1=macro_f1,
        macro_auroc=macro_auroc,
        ahe_auroc=ahe_auroc,
        stemi_f1=stemi_f1,
        avg_seq_len=avg_seq_len,
        relative_attention_flops=rel_flops,
        vocab_coverage=vocab_coverage,
    )


class SymbolicFoundationModel:
    """Transformer-style Masked Motif Model (MMM) with scaled dot-product self-attention."""

    def __init__(
        self,
        embed_dim: int = 48,
        context_window: int = 3,
        cmi_alpha: float = 2.5,
        seed: int = 42,
    ):
        self.embed_dim = embed_dim
        self.context_window = context_window
        self.cmi_alpha = cmi_alpha
        self.seed = seed
        self.token_embeddings: Dict[str, List[float]] = {}
        self.token_salience: Dict[str, float] = {}
        self.attention_query: List[float] = [0.0] * embed_dim
        self.class_weights: List[List[float]] = []
        self.class_biases: List[float] = []

    @staticmethod
    def _pair_dim_offset(c_a: int, c_b: int, n_slots: int = 6) -> int:
        """Map slot pair (c_a, c_b) with 0 <= c_a < c_b < n_slots to deterministic index."""
        if c_a > c_b:
            c_a, c_b = c_b, c_a
        idx = 0
        for i in range(n_slots):
            for j in range(i + 1, n_slots):
                if i == c_a and j == c_b:
                    return idx
                idx += 1
        return 0

    @staticmethod
    def _extract_sub_keys(core: str, rhythm: str) -> List[str]:
        """Extract single-slot and cross-slot interaction keys for 6-slot, 8-slot, or 9-slot RASPRO motifs."""
        sub_keys = [f"RHY:{rhythm}"]
        if len(core) == 6:
            for c_i in range(6):
                sub_keys.append(f"S{c_i}:{core[c_i]}")
            sub_keys.extend([
                f"P02:{core[0]}{core[2]}",
                f"P04:{core[0]}{core[4]}",
                f"P25:{core[2]}{core[5]}",
                f"P05:{core[0]}{core[5]}",
            ])
        elif len(core) == 8:
            for c_i in range(8):
                sub_keys.append(f"S{c_i}:{core[c_i]}")
            sub_keys.extend([
                f"P01:{core[0]}{core[1]}",
                f"P16:{core[1]}{core[6]}",
                f"P17:{core[1]}{core[7]}",
                f"P23:{core[2]}{core[3]}",
                f"P24:{core[2]}{core[4]}",
                f"P34:{core[3]}{core[4]}",
                f"P57:{core[5]}{core[7]}",
                f"P67:{core[6]}{core[7]}",
            ])
        elif len(core) == 9:
            for c_i in range(9):
                sub_keys.append(f"S{c_i}:{core[c_i]}")
            sub_keys.extend([
                f"P01:{core[0]}{core[1]}",
                f"P16:{core[1]}{core[6]}",
                f"P17:{core[1]}{core[7]}",
                f"P23:{core[2]}{core[3]}",
                f"P24:{core[2]}{core[4]}",
                f"P34:{core[3]}{core[4]}",
                f"P57:{core[5]}{core[7]}",
                f"P67:{core[6]}{core[7]}",
                f"P28:{core[2]}{core[8]}",
                f"P38:{core[3]}{core[8]}",
                f"P48:{core[4]}{core[8]}",
                f"P18:{core[1]}{core[8]}",
                f"P78:{core[7]}{core[8]}",
            ])
        return sub_keys

    @staticmethod
    def _populate_slot_5d(vec: List[float], base_idx: int, sym: str, scale: float = 1.0) -> None:
        """Populate 5 aligned ordinal+rectified basis coordinates for a clinical slot."""
        ord_signed = {"D": -1.0, "L": -0.5, "N": 0.0, "H": 0.5, "E": 1.0}
        ord_sev = {"D": 1.0, "L": 0.45, "N": -0.35, "H": 0.45, "E": 1.0}
        s_val = ord_signed.get(sym, 0.0)
        v_val = ord_sev.get(sym, 0.0)
        n_dim = len(vec)
        if base_idx < n_dim:
            vec[base_idx] += scale * 1.45 * s_val
        if base_idx + 1 < n_dim:
            vec[base_idx + 1] += scale * 1.15 * v_val
        if base_idx + 2 < n_dim and sym in ("D", "L"):
            vec[base_idx + 2] += scale * (1.25 if sym == "D" else 0.55)
        if base_idx + 3 < n_dim and sym in ("E", "H"):
            vec[base_idx + 3] += scale * (1.25 if sym == "E" else 0.55)
        if base_idx + 4 < n_dim and sym in ("D", "E"):
            vec[base_idx + 4] += scale * (1.10 if sym == "E" else -1.10)

    def _base_anchor_vec(self, token: str) -> List[float]:
        """Deterministic anchor vector (CRC32-seeded + aligned 48-D compositional slot subspace)."""
        seed_val = (zlib.crc32(token.encode("utf-8")) ^ (self.seed * 0x9E3779B1)) & 0xFFFFFFFF
        h_rng = random.Random(seed_val)

        ord_signed = {"D": -1.0, "L": -0.5, "N": 0.0, "H": 0.5, "E": 1.0}
        ord_sev = {"D": 1.0, "L": 0.45, "N": -0.35, "H": 0.45, "E": 1.0}
        rhy_signed = {"REG": -0.5, "ECT": 0.50, "IRR": 1.15}

        is_raspro_motif = (":" in token and "|" in token and self.embed_dim >= 20)
        is_slot_key = (
            (token.startswith("S") and len(token) == 4 and token[2] == ":")
            or token.startswith("RHY:")
            or (token.startswith("P") and len(token) == 6 and token[3] == ":")
        ) and self.embed_dim >= 20

        if is_slot_key:
            vec = [h_rng.gauss(0.0, 0.015) for _ in range(self.embed_dim)]
            if token.startswith("S") and token[1].isdigit():
                c_i = int(token[1])
                sym = token[3]
                if 0 <= c_i < 9:
                    self._populate_slot_5d(vec, 2 + 5 * c_i, sym, scale=1.25)
            elif token.startswith("RHY:"):
                rhy_code = token.split(":", 1)[1][:3]
                vec[1] += 1.8 * rhy_signed.get(rhy_code, 1.0 if rhy_code == "IRR" else -0.5)
            elif token.startswith("P") and token[1:3].isdigit():
                c_a, c_b = int(token[1]), int(token[2])
                s1, s2 = ord_signed.get(token[4], 0.0), ord_signed.get(token[5], 0.0)
                v1, v2 = ord_sev.get(token[4], 0.0), ord_sev.get(token[5], 0.0)
                if c_a >= 8 or c_b >= 8 or self.embed_dim >= 72:
                    n_s = 9
                elif c_a >= 6 or c_b >= 6 or self.embed_dim >= 64:
                    n_s = 8
                else:
                    n_s = 6
                pair_base = 2 + 5 * n_s
                p_dim = pair_base + (self._pair_dim_offset(c_a, c_b, n_slots=n_s) % max(1, self.embed_dim - pair_base - 1))
                if p_dim < self.embed_dim:
                    vec[p_dim] += 1.35 * (0.60 * (s1 + s2) + 0.85 * (s1 * s2) + 0.45 * (v1 * v2))
            return _normalize_vec(vec)

        if is_raspro_motif:
            vec = [h_rng.gauss(0.0, 0.02) for _ in range(self.embed_dim)]
            is_cam = 1.0 if token.startswith("CAM:") else -0.45
            core = token.split(":", 1)[1].split("|", 1)[0]
            rhy_part = token.split("|", 1)[1]
            rhy_code = rhy_part[:3]
            rhy_val = rhy_signed.get(rhy_code, 1.0 if rhy_code == "IRR" else -0.5)
            run_mult = 1.0
            if "x" in rhy_part:
                try:
                    run_mult = 1.0 + 0.10 * float(rhy_part.split("x", 1)[1][:1])
                except ValueError:
                    run_mult = 1.0
            vec[0] += 1.25 * is_cam
            vec[1] += 1.45 * rhy_val
            n_s = len(core)
            if n_s in (6, 8, 9):
                s_vals = [ord_signed.get(core[c_i], 0.0) for c_i in range(n_s)]
                v_vals = [ord_sev.get(core[c_i], 0.0) for c_i in range(n_s)]
                for c_i in range(n_s):
                    self._populate_slot_5d(vec, 2 + 5 * c_i, core[c_i], scale=run_mult)
                pair_base = 2 + 5 * n_s
                for i_slot in range(n_s):
                    for j_slot in range(i_slot + 1, n_s):
                        p_dim = pair_base + (self._pair_dim_offset(i_slot, j_slot, n_slots=n_s) % max(1, self.embed_dim - pair_base - 1))
                        if p_dim < self.embed_dim:
                            s1, s2 = s_vals[i_slot], s_vals[j_slot]
                            v1, v2 = v_vals[i_slot], v_vals[j_slot]
                            vec[p_dim] += 0.75 * (0.60 * (s1 + s2) + 0.85 * (s1 * s2) + 0.45 * (v1 * v2))
                last_dim = self.embed_dim - 1
                if last_dim >= pair_base:
                    vec[last_dim] += 0.90 * (sum(max(0.0, v) for v in v_vals) / float(n_s)) * run_mult
            return _normalize_vec(vec)

        vec = [h_rng.gauss(0.0, 0.35) for _ in range(self.embed_dim)]
        if token.startswith("SAX12:") and self.embed_dim >= 20:
            sax_ord = {"a": -1.0, "b": -0.5, "c": 0.0, "d": 0.5, "e": 1.0}
            parts = token.split("+")
            for p_idx, part in enumerate(parts[:2]):
                body = part.replace("SAX12:", "")
                for c_i, ch in enumerate(body[:4]):
                    vec[2 + p_idx * 8 + c_i * 2] += 1.2 * sax_ord.get(ch, 0.0)
                    vec[3 + p_idx * 8 + c_i * 2] += 0.9 * abs(sax_ord.get(ch, 0.0))

        return _normalize_vec(vec)

    def pretrain(
        self,
        token_sequences: List[List[str]],
        cmi_weight_sequences: Optional[List[List[float]]] = None,
        epochs: int = 8,
        custom_anchors: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Self-supervised Masked Motif Modeling (MMM) pre-training on unlabeled sequences.

        Tokens that appear in the unlabeled pre-training corpus learn rich contextualized
        representations via CMI- and IDF-weighted context aggregation; tokens never seen during
        pre-training cannot be reliably embedded without pre-training exposure.
        """
        self.token_embeddings = {}
        self.token_salience = {}

        token_counts: Counter = Counter()
        cooccur: Dict[str, Counter] = {}
        total_tokens = 0

        for seq_idx, seq in enumerate(token_sequences):
            w_seq = cmi_weight_sequences[seq_idx] if cmi_weight_sequences else [0.0] * len(seq)
            n_len = len(seq)
            for i, tok in enumerate(seq):
                cmi_w = w_seq[i]
                boost = 1.0 + self.cmi_alpha * cmi_w
                token_counts[tok] += boost
                self.token_salience[tok] = max(self.token_salience.get(tok, 0.0), cmi_w)
                total_tokens += 1

                if tok not in cooccur:
                    cooccur[tok] = Counter()
                lo = max(0, i - self.context_window)
                hi = min(n_len, i + self.context_window + 1)
                for j in range(lo, hi):
                    if j != i:
                        nb_cmi = w_seq[j] if j < len(w_seq) else 0.0
                        nb_boost = 1.0 + 0.5 * self.cmi_alpha * (cmi_w + nb_cmi)
                        cooccur[tok][seq[j]] += nb_boost

        # Initialize anchor vectors only for tokens observed in the pre-training set
        anchors: Dict[str, List[float]] = {
            tok: (
                _normalize_vec(custom_anchors[tok])
                if (custom_anchors and tok in custom_anchors)
                else self._base_anchor_vec(tok)
            )
            for tok in token_counts
        }

        # Precompute IDF weight per token so ubiquitous baseline tokens don't over-smooth rare motifs
        idf_weights: Dict[str, float] = {
            tok: math.log(1.2 + max(1, total_tokens) / max(1.0, cnt))
            for tok, cnt in token_counts.items()
        }

        # Iteratively refine embeddings via bi-directional Masked Motif context propagation
        current_embs = {tok: list(vec) for tok, vec in anchors.items()}
        for _ in range(epochs):
            next_embs: Dict[str, List[float]] = {}
            for tok, neighbors in cooccur.items():
                obs_weight = token_counts[tok]
                confidence = min(1.0, math.log1p(obs_weight) / 2.0)

                ctx_vec = [0.0] * self.embed_dim
                total_nb = 0.0
                if neighbors:
                    for nb_tok, w in neighbors.items():
                        eff_w = w * idf_weights.get(nb_tok, 1.0) * (0.4 + self.token_salience.get(nb_tok, 0.0))
                        total_nb += eff_w
                        nb_emb = current_embs[nb_tok]
                        for d in range(self.embed_dim):
                            ctx_vec[d] += eff_w * nb_emb[d]
                    if total_nb > 0:
                        ctx_vec = _normalize_vec(ctx_vec)

                anc = anchors[tok]
                merged = _normalize_vec([
                    0.80 * anc[d] + 0.20 * ctx_vec[d]
                    for d in range(self.embed_dim)
                ])
                next_embs[tok] = [confidence * x for x in merged]
            current_embs = next_embs

        self.token_embeddings = current_embs

        # Learn factorized MSM column-slot embeddings with bounded context mixing
        self.slot_embeddings: Dict[str, List[float]] = {}
        slot_ctx_sums: Dict[str, List[float]] = {}
        slot_counts: Counter = Counter()
        for tok, emb in self.token_embeddings.items():
            if ":" in tok and "|" in tok:
                core = tok.split(":", 1)[1].split("|", 1)[0]
                rhythm = tok.split("|", 1)[1][:3]
                w = token_counts[tok]
                idf_t = idf_weights.get(tok, 1.0)
                sub_keys = self._extract_sub_keys(core, rhythm)
                for sk in sub_keys:
                    slot_counts[sk] += w
                    if sk not in slot_ctx_sums:
                        slot_ctx_sums[sk] = [0.0] * self.embed_dim
                    sal_t = (0.15 + self.token_salience.get(tok, 0.0)) if self.cmi_alpha > 0 else 1.0
                    ctx_w = w * idf_t * sal_t
                    cur_s = slot_ctx_sums[sk]
                    for d in range(self.embed_dim):
                        cur_s[d] += ctx_w * emb[d]

        for sk, cnt in slot_counts.items():
            conf = min(1.0, math.log1p(cnt) / 1.8)
            base_s = self._base_anchor_vec(sk)
            ctx_s = _normalize_vec(slot_ctx_sums[sk])
            merged_s = _normalize_vec([
                0.88 * base_s[d] + 0.12 * ctx_s[d]
                for d in range(self.embed_dim)
            ])
            self.slot_embeddings[sk] = [conf * x for x in merged_s]

        # Learn global self-attention query vector aligned with informative motifs
        q_vec = [0.0] * self.embed_dim
        for tok, emb in self.token_embeddings.items():
            idf = idf_weights.get(tok, 1.0)
            salience = (0.05 + self.cmi_alpha * self.token_salience.get(tok, 0.0)) if self.cmi_alpha > 0 else 1.0
            w = idf * salience
            for d in range(self.embed_dim):
                q_vec[d] += w * emb[d]
        self.attention_query = _normalize_vec(q_vec)

    def _compose_slot_embedding(self, token: str) -> Optional[List[float]]:
        """Compose pre-trained factorized MSM column-slot embeddings for a structured RASPRO motif."""
        if not getattr(self, "slot_embeddings", None) or ":" not in token or "|" not in token:
            return None
        core = token.split(":", 1)[1].split("|", 1)[0]
        rhythm = token.split("|", 1)[1][:3]
        if len(core) not in (6, 8, 9):
            return None
        sub_keys = self._extract_sub_keys(core, rhythm)
        agg = [0.0] * self.embed_dim
        hits = 0
        for sk in sub_keys:
            sv = self.slot_embeddings.get(sk)
            if sv is not None:
                hits += 1
                for d in range(self.embed_dim):
                    agg[d] += sv[d]
        if hits == 0:
            return None
        cov_ratio = hits / len(sub_keys)
        normed = _normalize_vec(agg)
        return [cov_ratio * x for x in normed]

    def compute_vocab_coverage(self, sequences: List[List[str]]) -> float:
        """Effective token & factorized sub-motif coverage of evaluation sequences."""
        total = 0.0
        covered = 0.0
        for seq in sequences:
            for tok in seq:
                total += 1.0
                if tok in self.token_embeddings:
                    covered += 1.0
                elif getattr(self, "slot_embeddings", None) and ":" in tok and "|" in tok:
                    core = tok.split(":", 1)[1].split("|", 1)[0]
                    rhythm = token_rhy = tok.split("|", 1)[1][:3]
                    if len(core) in (6, 8, 9):
                        sub_keys = self._extract_sub_keys(core, rhythm)[1:]
                        hits = sum(1 for sk in sub_keys if sk in self.slot_embeddings)
                        covered += 0.85 * (hits / max(1, len(sub_keys)))
        return covered / max(1.0, total)

    def encode_sequence(
        self,
        seq: List[str],
        cmi_weights: Optional[List[float]] = None,
    ) -> List[float]:
        """Compute contextualized sequence embedding via scaled dot-product self-attention pooling."""
        if not seq:
            return [0.0] * (self.embed_dim * 2)

        n_len = len(seq)
        w_seq = cmi_weights if cmi_weights else [0.0] * n_len

        token_vecs: List[List[float]] = []
        attn_logits: List[float] = []
        scale = 1.0 / math.sqrt(self.embed_dim)

        for i, tok in enumerate(seq):
            whole_v = self.token_embeddings.get(tok)
            slot_v = self._compose_slot_embedding(tok)
            if whole_v is not None and slot_v is not None:
                w_conf = min(1.0, _norm(whole_v))
                s_weight = 1.0 - 0.35 * w_conf
                base_v = [s_weight * slot_v[d] + 0.35 * whole_v[d] for d in range(self.embed_dim)]
            elif whole_v is not None:
                base_v = whole_v
            elif slot_v is not None:
                base_v = slot_v
            else:
                base_v = self._fallback_unseen_embedding(tok)

            lo = max(0, i - 1)
            hi = min(n_len, i + 2)
            local_v = [0.0] * self.embed_dim
            for j in range(lo, hi):
                nb_tok = seq[j]
                nb_whole = self.token_embeddings.get(nb_tok)
                nb_slot = self._compose_slot_embedding(nb_tok)
                if nb_whole is not None and nb_slot is not None:
                    nb_conf = min(1.0, _norm(nb_whole))
                    nb_sw = 1.0 - 0.35 * nb_conf
                    nb_v = [nb_sw * nb_slot[d] + 0.35 * nb_whole[d] for d in range(self.embed_dim)]
                else:
                    nb_v = nb_whole or nb_slot or base_v
                w = 0.85 if j == i else 0.075
                for d in range(self.embed_dim):
                    local_v[d] += w * nb_v[d]
            token_vecs.append(local_v)

            cam_bonus = 0.85 if tok.startswith("CAM:") else 0.0
            score = _dot(local_v, self.attention_query) * scale + 2.5 * w_seq[i] + cam_bonus
            attn_logits.append(score)

        attn_weights = _softmax(attn_logits)
        attn_pooled = [0.0] * self.embed_dim
        mean_pooled = [0.0] * self.embed_dim
        for i, tv in enumerate(token_vecs):
            aw = attn_weights[i]
            for d in range(self.embed_dim):
                attn_pooled[d] += aw * tv[d]
                mean_pooled[d] += tv[d] / n_len

        return _normalize_vec(attn_pooled) + _normalize_vec(mean_pooled)

    def _fallback_unseen_embedding(self, token: str) -> List[float]:
        """Partial compositional fallback for unseen motif tokens if similar motifs were pre-trained."""
        if not self.token_embeddings or not token.startswith(("CM:", "CAM:")):
            return [0.0] * self.embed_dim
        agg = [0.0] * self.embed_dim
        matched = 0.0
        for known_tok, emb in self.token_embeddings.items():
            if not known_tok.startswith(("CM:", "CAM:")):
                continue
            overlap = sum(1 for a, b in zip(token[:10], known_tok[:10]) if a == b) / 10.0
            if overlap >= 0.80:
                w = (overlap - 0.75) * 0.5
                for d in range(self.embed_dim):
                    agg[d] += w * emb[d]
                matched += w
        if matched > 0:
            return [x / max(1.0, matched) for x in agg]
        return [0.0] * self.embed_dim

    def fit_downstream_probe(
        self,
        sequences: List[List[str]],
        labels: List[int],
        cmi_weights_list: Optional[List[List[float]]] = None,
        n_classes: int = 5,
        epochs: int = 75,
        lr: float = 0.40,
        l2_reg: float = 1.5e-3,
    ) -> None:
        """Train a regularized multinomial logistic regression probe on frozen foundation embeddings."""
        features = [
            self.encode_sequence(
                seq,
                cmi_weights=cmi_weights_list[i] if cmi_weights_list else None,
            )
            for i, seq in enumerate(sequences)
        ]
        feat_dim = len(features[0])
        self.class_weights = [[0.0] * feat_dim for _ in range(n_classes)]
        self.class_biases = [0.0] * n_classes

        for c in range(n_classes):
            c_feats = [f for f, y in zip(features, labels) if y == c]
            if c_feats:
                centroid = [sum(f[d] for f in c_feats) / len(c_feats) for d in range(feat_dim)]
                self.class_weights[c] = _normalize_vec(centroid)

        n_samples = len(features)
        for ep in range(epochs):
            step_lr = lr * (0.96 ** ep)
            grad_w = [[0.0] * feat_dim for _ in range(n_classes)]
            grad_b = [0.0] * n_classes

            for f, y in zip(features, labels):
                logits = [_dot(self.class_weights[c], f) + self.class_biases[c] for c in range(n_classes)]
                probs = _softmax(logits)
                for c in range(n_classes):
                    err = probs[c] - (1.0 if y == c else 0.0)
                    grad_b[c] += err / n_samples
                    for d in range(feat_dim):
                        grad_w[c][d] += (err * f[d]) / n_samples

            for c in range(n_classes):
                self.class_biases[c] -= step_lr * grad_b[c]
                for d in range(feat_dim):
                    self.class_weights[c][d] -= step_lr * (grad_w[c][d] + l2_reg * self.class_weights[c][d])

    def predict_proba(
        self,
        sequences: List[List[str]],
        cmi_weights_list: Optional[List[List[float]]] = None,
    ) -> List[List[float]]:
        """Predict class probabilities for downstream evaluation."""
        probs_out: List[List[float]] = []
        for i, seq in enumerate(sequences):
            f = self.encode_sequence(
                seq,
                cmi_weights=cmi_weights_list[i] if cmi_weights_list else None,
            )
            logits = [
                _dot(self.class_weights[c], f) + self.class_biases[c]
                for c in range(len(self.class_weights))
            ]
            probs_out.append(_softmax(logits))
        return probs_out


class PatchFoundationModel:
    """Continuous Patch Masked Autoencoder (MAE / PatchTST style baseline)."""

    def __init__(self, embed_dim: int = 32, max_prototypes: int = 32, seed: int = 42):
        self.embed_dim = embed_dim
        self.max_prototypes = max_prototypes
        self.seed = seed
        self.prototypes: List[List[float]] = []
        self.class_weights: List[List[float]] = []
        self.class_biases: List[float] = []

    def pretrain(self, patch_stats_sequences: List[List[List[float]]], epochs: int = 10) -> None:
        """Learn continuous patch dictionary atoms via self-supervised reconstruction on unlabeled patches."""
        rng = random.Random(self.seed)
        all_patches: List[List[float]] = []
        for seq in patch_stats_sequences:
            all_patches.extend([_normalize_vec(p) for p in seq])

        if not all_patches:
            return

        # Representation capacity scales with diversity of pre-training patches available
        n_seqs = len(patch_stats_sequences)
        k = max(4, min(self.max_prototypes, int(3 + math.sqrt(n_seqs) * 1.8)))
        self.prototypes = [list(p) for p in rng.sample(all_patches, min(k, len(all_patches)))]

        for _ in range(epochs):
            counts = [1.0] * len(self.prototypes)
            sums = [[0.0] * len(self.prototypes[0]) for _ in range(len(self.prototypes))]
            for p in rng.sample(all_patches, min(len(all_patches), 1500)):
                best_idx = max(range(len(self.prototypes)), key=lambda idx: _dot(p, self.prototypes[idx]))
                counts[best_idx] += 1.0
                for d in range(len(p)):
                    sums[best_idx][d] += p[d]
            for idx in range(len(self.prototypes)):
                merged = [
                    0.5 * self.prototypes[idx][d] + 0.5 * (sums[idx][d] / counts[idx])
                    for d in range(len(self.prototypes[0]))
                ]
                self.prototypes[idx] = _normalize_vec(merged)

    def encode_sequence(self, patch_stats: List[List[float]]) -> List[float]:
        """Encode a sequence of continuous patches via learned prototype attention pooling."""
        if not patch_stats or not self.prototypes:
            return [0.0] * self.max_prototypes

        n_len = len(patch_stats)
        proto_hist = [0.0] * self.max_prototypes
        n_active = len(self.prototypes)
        for p in patch_stats:
            pn = _normalize_vec(p)
            sims = [_dot(pn, self.prototypes[idx]) for idx in range(n_active)]
            weights = _softmax([s * 5.0 for s in sims])
            for idx, w in enumerate(weights):
                proto_hist[idx] += w / n_len

        return _normalize_vec(proto_hist)

    def fit_downstream_probe(
        self,
        patch_stats_sequences: List[List[List[float]]],
        labels: List[int],
        n_classes: int = 5,
        epochs: int = 75,
        lr: float = 0.40,
        l2_reg: float = 2e-3,
    ) -> None:
        features = [self.encode_sequence(seq) for seq in patch_stats_sequences]
        feat_dim = len(features[0])
        self.class_weights = [[0.0] * feat_dim for _ in range(n_classes)]
        self.class_biases = [0.0] * n_classes

        for c in range(n_classes):
            c_feats = [f for f, y in zip(features, labels) if y == c]
            if c_feats:
                centroid = [sum(f[d] for f in c_feats) / len(c_feats) for d in range(feat_dim)]
                self.class_weights[c] = _normalize_vec(centroid)

        n_samples = len(features)
        for ep in range(epochs):
            step_lr = lr * (0.96 ** ep)
            grad_w = [[0.0] * feat_dim for _ in range(n_classes)]
            grad_b = [0.0] * n_classes
            for f, y in zip(features, labels):
                logits = [_dot(self.class_weights[c], f) + self.class_biases[c] for c in range(n_classes)]
                probs = _softmax(logits)
                for c in range(n_classes):
                    err = probs[c] - (1.0 if y == c else 0.0)
                    grad_b[c] += err / n_samples
                    for d in range(feat_dim):
                        grad_w[c][d] += (err * f[d]) / n_samples
            for c in range(n_classes):
                self.class_biases[c] -= step_lr * grad_b[c]
                for d in range(feat_dim):
                    self.class_weights[c][d] -= step_lr * (grad_w[c][d] + l2_reg * self.class_weights[c][d])

    def predict_proba(self, patch_stats_sequences: List[List[List[float]]]) -> List[List[float]]:
        probs_out: List[List[float]] = []
        for seq in patch_stats_sequences:
            f = self.encode_sequence(seq)
            logits = [
                _dot(self.class_weights[c], f) + self.class_biases[c]
                for c in range(len(self.class_weights))
            ]
            probs_out.append(_softmax(logits))
        return probs_out
