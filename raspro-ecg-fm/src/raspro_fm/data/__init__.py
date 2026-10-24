"""Data generation and loading utilities for coupled ECG and PPG waveforms."""

from .physiological_generator import (
    CLINICAL_CLASSES,
    PhysiologicalRecord,
    generate_coupled_ecg_ppg,
    generate_cohort_dataset,
)

__all__ = [
    "CLINICAL_CLASSES",
    "PhysiologicalRecord",
    "generate_coupled_ecg_ppg",
    "generate_cohort_dataset",
]
