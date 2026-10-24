"""Baseline 2 & 3: Symbolic Aggregate approXimation (SAX) and SAX + Byte-Pair Encoding (ECG-Byte style)."""

from collections import Counter
from dataclasses import dataclass
import math
from typing import Dict, List, Tuple


# Standard Gaussian N(0,1) equiprobable breakpoints for alphabet sizes 4, 6, 8, 10, 12
GAUSSIAN_BREAKPOINTS: Dict[int, List[float]] = {
    4: [-0.674, 0.0, 0.674],
    6: [-0.967, -0.431, 0.0, 0.431, 0.967],
    8: [-1.150, -0.674, -0.319, 0.0, 0.319, 0.674, 1.150],
    10: [-1.282, -0.842, -0.524, -0.253, 0.0, 0.253, 0.524, 0.842, 1.282],
    12: [-1.383, -0.967, -0.674, -0.431, -0.210, 0.0, 0.210, 0.431, 0.674, 0.967, 1.383],
}


@dataclass
class SAXTokenSequence:
    """Output of SAX or SAX-BPE tokenization on synchronized ECG + PPG streams."""
    ecg_symbols: List[str]
    ppg_symbols: List[str]
    combined_tokens: List[str]
    token_ids: List[int]
    n_tokens: int
    paa_ecg: List[float]
    paa_ppg: List[float]
    reconstructed_ecg: List[float]
    reconstructed_ppg: List[float]


def _z_normalize(seq: List[float]) -> Tuple[List[float], float, float]:
    """Z-score normalize a sequence to zero mean and unit variance."""
    if not seq:
        return [], 0.0, 1.0
    mean = sum(seq) / len(seq)
    var = sum((x - mean) ** 2 for x in seq) / len(seq)
    std = math.sqrt(var) if var > 1e-8 else 1.0
    return [(x - mean) / std for x in seq], mean, std


class SAXTokenizer:
    """Classical Symbolic Aggregate approXimation (PAA + Gaussian quantile breakpoints)."""

    def __init__(self, fs: int = 100, paa_window_sec: float = 0.20, alphabet_size: int = 8):
        if alphabet_size not in GAUSSIAN_BREAKPOINTS:
            alphabet_size = 8
        self.fs = fs
        self.paa_samples = max(2, int(round(paa_window_sec * fs)))
        self.alphabet_size = alphabet_size
        self.breakpoints = GAUSSIAN_BREAKPOINTS[alphabet_size]
        self.alphabet = [chr(ord("a") + i) for i in range(alphabet_size)]
        # Centroids for reconstructing the continuous signal from Gaussian bins
        self.centroids = self._compute_bin_centroids()
        self.vocab: Dict[str, int] = {
            f"{e}{p}": i * alphabet_size + j
            for i, e in enumerate(self.alphabet)
            for j, p in enumerate(self.alphabet)
        }

    def _compute_bin_centroids(self) -> List[float]:
        """Representative z-score value for each Gaussian bin."""
        bps = [-1.85] + self.breakpoints + [1.85]
        return [0.5 * (bps[i] + bps[i + 1]) for i in range(len(bps) - 1)]

    def _value_to_symbol_idx(self, z_val: float) -> int:
        for idx, bp in enumerate(self.breakpoints):
            if z_val < bp:
                return idx
        return len(self.breakpoints)

    def _encode_single_channel(self, signal: List[float]) -> Tuple[List[str], List[float], List[float]]:
        z_sig, mean, std = _z_normalize(signal)
        n = len(z_sig)
        paa_vals: List[float] = []
        symbols: List[str] = []
        reconstructed: List[float] = [mean] * n

        for start in range(0, n - self.paa_samples + 1, self.paa_samples):
            end = start + self.paa_samples
            seg = z_sig[start:end]
            paa_mean = sum(seg) / len(seg)
            sym_idx = self._value_to_symbol_idx(paa_mean)
            paa_vals.append(paa_mean)
            symbols.append(self.alphabet[sym_idx])
            rec_val = self.centroids[sym_idx] * std + mean
            for k in range(start, end):
                reconstructed[k] = rec_val

        return symbols, paa_vals, reconstructed

    def tokenize(self, ecg: List[float], ppg: List[float]) -> SAXTokenSequence:
        """Convert ECG and PPG waveforms into coupled SAX token sequences."""
        e_syms, e_paa, e_rec = self._encode_single_channel(ecg)
        p_syms, p_paa, p_rec = self._encode_single_channel(ppg)
        m = min(len(e_syms), len(p_syms))
        combined = [f"{e_syms[i]}{p_syms[i]}" for i in range(m)]
        token_ids = [self.vocab[tok] for tok in combined]

        return SAXTokenSequence(
            ecg_symbols=e_syms[:m],
            ppg_symbols=p_syms[:m],
            combined_tokens=combined,
            token_ids=token_ids,
            n_tokens=m,
            paa_ecg=e_paa[:m],
            paa_ppg=p_paa[:m],
            reconstructed_ecg=e_rec,
            reconstructed_ppg=p_rec,
        )


class SAXBpeTokenizer:
    """SAX + Byte-Pair Encoding (BPE) subword tokenizer (inspired by ECG-Byte, arXiv:2412.14373)."""

    def __init__(self, fs: int = 100, paa_window_sec: float = 0.20, alphabet_size: int = 8, num_merges: int = 48):
        self.base_sax = SAXTokenizer(fs=fs, paa_window_sec=paa_window_sec, alphabet_size=alphabet_size)
        self.num_merges = num_merges
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = dict(self.base_sax.vocab)

    def fit(self, records_ecg_ppg: List[Tuple[List[float], List[float]]]) -> None:
        """Learn BPE merge rules over a corpus of coupled SAX token sequences."""
        corpus: List[List[str]] = []
        for ecg, ppg in records_ecg_ppg:
            sax_out = self.base_sax.tokenize(ecg, ppg)
            corpus.append(list(sax_out.combined_tokens))

        self.merges = []
        self.vocab = dict(self.base_sax.vocab)
        next_id = len(self.vocab)

        for _ in range(self.num_merges):
            pair_counts: Counter = Counter()
            for seq in corpus:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += 1
            if not pair_counts:
                break
            best_pair, count = pair_counts.most_common(1)[0]
            if count < 3:
                break
            merged_tok = f"{best_pair[0]}+{best_pair[1]}"
            self.merges.append(best_pair)
            if merged_tok not in self.vocab:
                self.vocab[merged_tok] = next_id
                next_id += 1

            # Apply merge to corpus
            new_corpus: List[List[str]] = []
            for seq in corpus:
                new_seq: List[str] = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                        new_seq.append(merged_tok)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_corpus.append(new_seq)
            corpus = new_corpus

    def tokenize(self, ecg: List[float], ppg: List[float]) -> SAXTokenSequence:
        """Tokenize ECG + PPG with base SAX and compress via learned BPE merges."""
        base_out = self.base_sax.tokenize(ecg, ppg)
        seq = list(base_out.combined_tokens)

        for pair in self.merges:
            merged_tok = f"{pair[0]}+{pair[1]}"
            new_seq: List[str] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                    new_seq.append(merged_tok)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq

        token_ids = [self.vocab.get(tok, 0) for tok in seq]
        return SAXTokenSequence(
            ecg_symbols=base_out.ecg_symbols,
            ppg_symbols=base_out.ppg_symbols,
            combined_tokens=seq,
            token_ids=token_ids,
            n_tokens=len(seq),
            paa_ecg=base_out.paa_ecg,
            paa_ppg=base_out.paa_ppg,
            reconstructed_ecg=base_out.reconstructed_ecg,
            reconstructed_ppg=base_out.reconstructed_ppg,
        )
