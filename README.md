# RASPRO-FM: Rapid Active Summarization for Data and Compute Efficient Biosignal Foundation Models

[![Project Page](https://img.shields.io/badge/Project_Page-pprahul.github.io%2Fraspro--fm-1d4ed8)](https://pprahul.github.io/raspro-fm/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch MPS/CUDA](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-ee4c2c.svg)](https://pytorch.org/)
[![Manuscript](https://img.shields.io/badge/Manuscript-Under_Review_(IEEE_JBHI)-0f766e.svg)](https://pprahul.github.io/raspro-fm/)

**Authors:** Rahul Krishnan Pathinarupothi¹, Durga Padmavilochanan¹, Maneesha Vinodini Ramesh¹, and Saptarshi Purkayastha²  
*¹Amrita Vishwa Vidyapeetham, India · ²Indiana University, Indianapolis, IN, USA*

🌐 **Interactive Project Page:** [https://pprahul.github.io/raspro-fm/](https://pprahul.github.io/raspro-fm/)

---

## Sub-Projects in This Repository

### 1. [`raspro-ecg-fm/`](./raspro-ecg-fm) — 12-Lead ECG & Multi-Sensor ICU Foundation Model
Contains the complete implementation, pre-tokenized hospital caches (`PTB-XL` & `MIMIC-IV-ECG`), and benchmark suite for **RASPRO-ECG-FM**:
- **Beat-Synchronous 9-Slot 12-Lead (`RASPRO-12Lead`) & 6-Slot Multi-Sensor (`MSM`) Tokenizers** with Physician Assist Filters (`PAF`), Consensus / Consensus Abnormality Motifs (`CM` / `CAM`), and Active Run-Length Deduplication ($L = 3.54$ active tokens per 10s ECG; $50\times\text{–}1,797\times$ fewer $\mathcal{O}(L^2)$ attention FLOPs).
- **72-D Compositional Slot-and-Pair Subspace & CMI-Biased Multi-Layer `RASPRO-Transformer FM`** (`79.79%` Superclass AUROC and `79.08%` 4-Task Mean Macro-AUROC on PTB-XL Fold 10; `78.06%` zero-shot cross-hospital transfer from `MIMIC-IV-ECG`).
- **Two-Stage Hierarchical CMI Submodular Coreset Selection** (`0.98 s` pre-training on `5%` Coreset reaching `77.61%` Superclass AUROC).

<p align="center">
  <img src="raspro-ecg-fm/docs/assets/raspro_fm_tokenizer_comparison.png" width="96%" alt="RASPRO-FM Tokenizer Comparison" />
</p>

<p align="center">
  <img src="raspro-ecg-fm/docs/assets/raspro_fm_architecture_diagram.png" width="96%" alt="RASPRO-FM Architecture Diagram" />
</p>

---

## Quickstart (`raspro-ecg-fm`)

```bash
git clone https://github.com/pprahul/raspro-fm.git
cd raspro-fm/raspro-ecg-fm
pip install -e .

# 1. Run Multi-Layer GPU/MPS RASPRO-Transformer FM on PTB-XL Fold 10 + Zero-Shot MIMIC-IV-ECG Transfer
python scripts/run_phase4_full_hospital_transformer.py

# 2. Run Controlled 12-Lead Tokenizer Ablations (Patch, SAX, ECG-Byte, HeartLang-QRS, RASPRO 6/8/9-Slot)
python scripts/run_phase3_ptbxl_sota_benchmark.py

# 3. Run Real-World Multi-Sensor ICU Telemetry Benchmark (PhysioNet BIDMC ECG + PPG + RESP)
python scripts/run_phase2_physionet_benchmark.py

# 4. Run Coupled ECG + PPG Biophysical Data-Scaling Benchmark (5% to 100% Coreset Pruning)
python scripts/run_scaling_benchmark.py
```
