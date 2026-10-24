"""Signal conditioning and beat-synchronous morphological feature extraction for ECG and PPG."""

from .segmentation import (
    ExtractedBeatFeatures,
    detrend_baseline,
    detect_r_peaks,
    extract_coupled_beat_features,
)

__all__ = [
    "ExtractedBeatFeatures",
    "detrend_baseline",
    "detect_r_peaks",
    "extract_coupled_beat_features",
]
