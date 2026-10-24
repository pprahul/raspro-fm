"""Baseline 1: Continuous Waveform Patch Tokenizer (PatchTST / ECGFounder style)."""

from dataclasses import dataclass
import math
from typing import List


@dataclass
class PatchTokenSequence:
    """Output of slicing multi-sensor continuous waveforms into fixed temporal patches."""
    patches: List[List[float]]       # Each patch is a concatenated [ECG_patch, PPG_patch] vector
    patch_stats: List[List[float]]   # Summary vector [ecg_mean, ecg_std, ecg_max, ppg_mean, ppg_std, ppg_max]
    n_tokens: int
    raw_samples: int
    patch_size_samples: int


class PatchTokenizer:
    """Slices continuous ECG and PPG waveforms into fixed-length numerical patches."""

    def __init__(self, fs: int = 100, patch_duration_sec: float = 0.20, stride_sec: float = 0.20):
        self.fs = fs
        self.patch_size = max(4, int(round(patch_duration_sec * fs)))
        self.stride = max(4, int(round(stride_sec * fs)))

    def tokenize(self, ecg: List[float], ppg: List[float]) -> PatchTokenSequence:
        """Convert synchronized ECG and PPG streams into a sequence of continuous patch vectors."""
        n = min(len(ecg), len(ppg))
        patches: List[List[float]] = []
        patch_stats: List[List[float]] = []

        for start in range(0, n - self.patch_size + 1, self.stride):
            end = start + self.patch_size
            e_seg = ecg[start:end]
            p_seg = ppg[start:end]
            patches.append(e_seg + p_seg)

            e_mean = sum(e_seg) / len(e_seg)
            e_var = sum((x - e_mean) ** 2 for x in e_seg) / len(e_seg)
            p_mean = sum(p_seg) / len(p_seg)
            p_var = sum((x - p_mean) ** 2 for x in p_seg) / len(p_seg)

            patch_stats.append([
                e_mean,
                math.sqrt(e_var),
                max(e_seg) - min(e_seg),
                p_mean,
                math.sqrt(p_var),
                max(p_seg) - min(p_seg),
            ])

        return PatchTokenSequence(
            patches=patches,
            patch_stats=patch_stats,
            n_tokens=len(patches),
            raw_samples=n * 2,
            patch_size_samples=self.patch_size,
        )
