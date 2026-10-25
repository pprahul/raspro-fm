# Phase 3 Benchmark: Apples-to-Apples Comparison Against SOTA ECG Foundation Models (`PhysioNet PTB-XL`)

## 1. Benchmark Setup & Apples-to-Apples Evaluation Protocol
- **Dataset:** PhysioNet **PTB-XL (`v1.0.3`)** 12-lead clinical ECGs (`I, II, III, aVR, aVL, aVF, V1–V6`) at $f_s = 100\text{ Hz}$ ($10\text{ s} = 1,000\text{ samples} \times 12\text{ leads}$).
- **Official Patient-Stratified Split (`strat_fold`):** Folds 1–8 (`210` records) for self-supervised pre-training & downstream linear probe training; Fold 9 (`45` records) for validation; **Fold 10 (`75` human-verified gold-standard records)** for held-out test evaluation.
- **4 Canonical Multi-Label Diagnostic Hierarchies:** **Superclass (`5` classes: `NORM, MI, STTC, CD, HYP`)**, **Subclass (`13` classes)**, **Form (`16` classes)**, and **Rhythm (`10` classes)**.
- **Primary Metrics:** Unweighted **Macro-AUROC (%)** and **Macro-AUPRC (%)** on **Fold 10** under **1%, 10%, and 100%** labeled data linear probing (Protocol 1, $5$ seeds) and **5%–100%** unlabeled pre-training data pruning (Protocol 2, $5$ seeds).

---

## 2. Tokenization Compression & Transformer $O(L^2)$ Self-Attention Efficiency (`12-Lead PTB-XL`)

| Foundation Model Tokenizer | Tokenization Granularity | Avg Tokens ($L$) per 10s 12-Lead ECG | Sequence Compression vs. Patch | Relative $O(L^2)$ Attention FLOPs | Attention Compute Savings |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Patch-12Lead (`ST-MEM` / `PatchTST` style)** | Fixed $0.4\text{ s}$ 12-lead patches | `25.0` | `1.0x` | `100.00%` | `1.0x` |
| **Vanilla SAX-12Lead** | Fixed $0.4\text{ s}$ PAA + Gaussian symbols | `25.0` | `1.0x` | `100.00%` | `1.0x` |
| **ECG-Byte (`SAX + BPE`)** | Byte-Pair Encoded 12-lead SAX subwords | `20.0` | `1.2x` | `64.17%` | `1.6x` |
| **HeartLang-QRS (`ICLR 2025` style)** | R-peak heartbeat words (VQ Codebook, no CM merge) | `13.6` | `1.8x` | `29.55%` | `3.4x` |
| **RASPRO-12Lead (`Ours: 8-Slot PAF + MSM + CM/CAM`)** | 4-beat 12-lead anatomical `CM`/`CAM` + run-length merge | **`3.74`** | **`6.7x`** | **`2.23%`** | **`45x`** |

---

## 3. Protocol 1: Canonical PTB-XL Linear Probing Leaderboard (`1%`, `10%`, `100%` Labeled Data on Fold 10)

| Model / Framework | Venue / Type | Superclass (1% / 10% / 100%) | Subclass (1% / 10% / 100%) | Form (1% / 10% / 100%) | Rhythm (1% / 10% / 100%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| *SimCLR (Contrastive SSL)* | ICML 2020 / eSSL (Published) | `63.41` / `69.77` / `73.53` | `54.77` / `61.38` / `66.89` | `54.84` / `57.09` / `62.52` | `52.95` / `59.35` / `67.30` |
| *BYOL (Bootstrap SSL)* | NeurIPS 2020 / eSSL (Published) | `68.09` / `73.95` / `76.45` | `55.54` / `64.56` / `70.69` | `53.26` / `58.61` / `65.65` | `52.66` / `61.50` / `70.69` |
| *CLOCS (Cardiac Contrastive)* | ICML 2021 (Published) | `68.94` / `76.31` / `77.69` | `57.94` / `68.83` / `73.79` | `51.98` / `59.95` / `67.60` | `53.28` / `65.41` / `74.34` |
| *MTAE (MaeFE Masked Time)* | AAAI / Pattern Rec. (Published) | `62.89` / `71.93` / `76.98` | `53.49` / `63.60` / `71.13` | `52.10` / `56.92` / `62.94` | `51.83` / `61.96` / `69.25` |
| *MLAE (MaeFE Masked Lead)* | AAAI / Pattern Rec. (Published) | `66.15` / `73.69` / `77.10` | `54.92` / `64.94` / `71.68` | `52.85` / `57.88` / `64.12` | `52.40` / `62.11` / `70.18` |
| *ST-MEM (Spatio-Temporal MAE)* | ICLR 2024 (Published) | `67.15` / `76.12` / `81.90` | `56.36` / `66.84` / `74.85` | `55.71` / `60.86` / `70.20` | `55.12` / `65.48` / `77.26` |
| *MERL (Linear Probe Only)* | ICML 2024 (Published) | `69.80` / `72.95` / `74.29` | `58.20` / `69.10` / `74.32` | `54.90` / `60.15` / `67.40` | `56.40` / `66.80` / `75.80` |
| *HeartLang (QRS-Words + Transformer)* | ICLR 2025 (Published) | `78.94` / `85.59` / `87.52` | `64.68` / `79.34` / `88.91` | `58.70` / `63.99` / `80.23` | `62.08` / `76.22` / `90.34` |
| **--- Controlled Head-to-Head Evaluation on Identical PTB-XL Cohort ---** | **---** | **---** | **---** | **---** | **---** |
| Patch-12Lead FM (ST-MEM style, 100% Pretrain) | Controlled Fold-10 | 57.50±3.2 / 60.07±2.7 / 60.29 | 52.01±1.7 / 53.52±1.7 / 55.36 | 51.59±0.4 / 55.02±1.7 / 56.94 | 59.02±2.6 / 67.80±3.0 / 69.53 |
| Vanilla SAX-12Lead FM (100% Pretrain) | Controlled Fold-10 | 52.90±3.0 / 53.77±2.0 / 58.37 | 50.56±2.8 / 52.96±1.1 / 56.79 | 49.12±2.3 / 49.79±2.9 / 52.00 | 57.05±4.1 / 59.28±4.9 / 62.94 |
| ECG-Byte FM (SAX + BPE, 100% Pretrain) | Controlled Fold-10 | 53.11±4.9 / 53.95±2.5 / 57.66 | 51.32±2.0 / 54.10±2.3 / 55.77 | 49.34±2.4 / 50.24±4.2 / 50.11 | 53.80±2.9 / 51.61±4.2 / 56.16 |
| HeartLang-QRS FM (VQ Codebook, 100% Pretrain) | Controlled Fold-10 | 59.22±6.0 / 68.31±4.6 / 72.53 | 54.20±5.2 / 59.29±3.0 / 67.98 | 50.42±1.5 / 54.31±4.4 / 62.90 | 58.83±2.7 / 72.17±3.8 / 72.48 |
| **RASPRO-12Lead FM (6-Slot, 100% Pretrain)** | Controlled Fold-10 | **60.15±5.0 / 68.05±2.8 / 74.43** | **54.18±3.0 / 58.54±2.3 / 66.86** | **53.32±1.4 / 60.88±2.7 / 67.23** | **60.66±2.8 / 78.78±5.5 / 83.51** |
| **RASPRO-12Lead FM (8-Slot, 10% CMI Coreset Pretrain)** | Controlled Fold-10 | **60.36±5.2 / 69.36±3.5 / 73.98** | **54.11±4.2 / 60.58±2.4 / 68.84** | **54.76±2.1 / 62.97±2.6 / 70.87** | **58.60±3.7 / 78.75±5.2 / 87.25** |
| **RASPRO-12Lead FM (8-Slot, 100% Pretrain)** | Controlled Fold-10 | **60.49±5.2 / 69.33±3.5 / 74.56** | **53.96±3.9 / 60.02±2.6 / 68.32** | **54.22±1.9 / 62.56±3.0 / 71.35** | **58.53±3.9 / 78.63±5.2 / 86.34** |
| **RASPRO-12Lead FM (9-Slot + QT_MORPH, 10% Coreset)** | Controlled Fold-10 | **59.43±4.7 / 70.42±3.1 / 78.34** | **53.68±3.8 / 61.08±2.1 / 69.07** | **54.50±0.9 / 63.37±3.5 / 70.47** | **56.76±3.3 / 76.55±4.8 / 84.57** |
| **RASPRO-12Lead FM (9-Slot + QT_MORPH, 100% Pretrain)** | Controlled Fold-10 | **59.66±4.6 / 70.34±3.4 / 78.06** | **53.56±3.8 / 60.82±1.8 / 69.00** | **54.18±2.1 / 63.29±3.7 / 70.62** | **57.29±3.6 / 76.92±5.0 / 84.77** |

### 3.2 Per-Class Superclass AUROC (`NORM, MI, STTC, CD, HYP`) & Macro-AUPRC on Fold 10 (100% Labels)
| Model / Framework | `NORM` | `MI` | `STTC` | `CD` | `HYP` | Superclass Macro-AUROC | Superclass Macro-AUPRC | 4-Task Mean Macro-AUPRC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Patch-12Lead FM (ST-MEM style, 100% Pretrain) | 71.08% | 57.85% | 69.08% | 49.61% | 53.81% | 60.29% | 40.53% | 25.15% |
| Vanilla SAX-12Lead FM (100% Pretrain) | 54.03% | 52.58% | 62.48% | 54.78% | 68.01% | 58.37% | 33.62% | 23.70% |
| ECG-Byte FM (SAX + BPE, 100% Pretrain) | 48.20% | 58.90% | 56.53% | 58.38% | 66.31% | 57.66% | 33.42% | 23.34% |
| HeartLang-QRS FM (VQ Codebook, 100% Pretrain) | 87.08% | 62.65% | 81.92% | 62.87% | 68.11% | 72.53% | 54.96% | 34.50% |
| **RASPRO-12Lead FM (6-Slot, 100% Pretrain)** | 78.81% | 60.30% | 84.00% | 76.02% | 72.99% | **74.43%** | **56.00%** | **35.80%** |
| **RASPRO-12Lead FM (8-Slot, 10% CMI Coreset Pretrain)** | 87.39% | 53.51% | 82.14% | 69.40% | 77.44% | **73.98%** | **55.29%** | **39.37%** |
| **RASPRO-12Lead FM (8-Slot, 100% Pretrain)** | 85.81% | 56.67% | 82.21% | 70.76% | 77.33% | **74.56%** | **55.78%** | **38.92%** |
| **RASPRO-12Lead FM (9-Slot + QT_MORPH, 10% Coreset)** | 87.82% | 74.82% | 81.49% | 72.03% | 75.53% | **78.34%** | **57.99%** | **39.91%** |
| **RASPRO-12Lead FM (9-Slot + QT_MORPH, 100% Pretrain)** | 86.23% | 76.11% | 81.99% | 71.05% | 74.89% | **78.06%** | **58.21%** | **40.80%** |

---

## 4. Protocol 2: Unlabeled Pre-Training Data Pruning (`5%` to `100%` of Folds 1–8 Unlabeled Pool)

### 4.1 PTB-XL `Superclass` Macro-AUROC (%) vs. Pre-Training Data Budget
| Foundation Model Pipeline | Avg Seq Len ($L$) | Relative Attention FLOPs | 5% Unlabeled Data | 10% Unlabeled Data | 25% Unlabeled Data | 50% Unlabeled Data | 100% Unlabeled Data |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Continuous Patch (ST-MEM style) | 25.0 | `100.00%` | 55.99% | 63.14% | 62.23% | 60.13% | 60.91% |
| Vanilla SAX (12-Lead) | 25.0 | `100.00%` | 59.45% | 59.04% | 58.25% | 58.62% | 58.37% |
| ECG-Byte (SAX + BPE) | 20.0 | `64.17%` | 58.92% | 56.56% | 56.37% | 60.02% | 59.45% |
| HeartLang-QRS (Beat Codebook) | 13.6 | `29.55%` | 64.85% | 73.55% | 74.60% | 73.46% | 74.81% |
| **RASPRO-12Lead (Random Subset)** | 3.7 | `2.23%` | **72.45±0.65%** | **73.17±0.62%** | **73.75±0.25%** | **74.25±0.25%** | **74.58%** |
| **RASPRO-12Lead + CMI Coreset** | 3.7 | `2.23%` | **73.91%** | **73.98%** | **74.21%** | **74.53%** | **74.56%** |

### 4.2 Mean Macro-AUROC (%) Across All 4 PTB-XL Hierarchies (`Superclass + Subclass + Form + Rhythm`)
| Foundation Model Pipeline | 5% Data (Mean AUC / Cov) | 10% Data (Mean AUC / Cov) | 25% Data (Mean AUC / Cov) | 50% Data (Mean AUC / Cov) | 100% Data (Mean AUC / Cov) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Continuous Patch (ST-MEM style) | 57.57% (`100%`) | 61.30% (`100%`) | 61.45% (`100%`) | 60.75% (`100%`) | 60.48% (`100%`) |
| Vanilla SAX (12-Lead) | 56.79% (`76%`) | 56.77% (`87%`) | 56.91% (`93%`) | 57.39% (`96%`) | 57.53% (`98%`) |
| ECG-Byte (SAX + BPE) | 53.79% (`74%`) | 52.02% (`85%`) | 52.76% (`91%`) | 54.58% (`95%`) | 55.93% (`97%`) |
| HeartLang-QRS (Beat Codebook) | 60.54% (`100%`) | 69.03% (`99%`) | 69.57% (`100%`) | 68.33% (`100%`) | 70.31% (`100%`) |
| **RASPRO-12Lead (Random Subset)** | **73.87±0.50%** (`76%`) | **74.73±0.58%** (`83%`) | **75.20±0.29%** (`87%`) | **75.38±0.15%** (`89%`) | **75.15%** (`91%`) |
| **RASPRO-12Lead + CMI Coreset** | **75.33%** (`83%`) | **75.23%** (`86%`) | **75.35%** (`88%`) | **75.41%** (`90%`) | **75.14%** (`91%`) |

---

## 5. Protocol 3: Hyperparameter Pareto Sensitivity, Pre-Training Attribution & Noise Robustness

### 5.1 Hyperparameter Pareto Frontier (`W_b` Window Size & `r_max` Run-Length Deduplication Cap)
| RASPRO Configuration | Avg Seq Len ($L$) | Compression vs. Patch | Relative Attention FLOPs | CPU Latency (ms/rec) | Superclass AUROC | 4-Task Mean AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| W_b=1 beat, r_max=1 (Beat-Level, No Merge) | 13.59 | `1.8x` | `29.55%` | `3.16 ms` | 76.92% | 73.14% |
| W_b=2 beats, r_max=2 (2-Beat MSM) | 6.99 | `3.6x` | `7.81%` | `1.87 ms` | 74.97% | 75.72% |
| W_b=4 beats, r_max=1 (4-Beat MSM, No Merge) | 3.78 | `6.6x` | `2.28%` | `1.02 ms` | 74.83% | 75.45% |
| **W_b=4 beats, r_max=4 (RASPRO Default)** | 3.74 | `6.7x` | `2.23%` | `1.02 ms` | **74.56%** | **75.14%** |
| W_b=8 beats, r_max=4 (Coarse 8-Beat MSM) | 2.16 | `11.6x` | `0.74%` | `0.59 ms` | 73.50% | 75.16% |

### 5.2 Pre-Training Attribution (`0-Epoch Random Init` vs. `Pre-Trained`) & Test-Time R-Peak Jitter / Noise Stress-Test
| Evaluation Condition | Superclass AUROC | Subclass AUROC | Form AUROC | Rhythm AUROC | 4-Task Mean AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RASPRO-12Lead (`0-Epoch Random Token Hash`, No SSL)** | `52.45%` | `51.54%` | `53.79%` | `47.47%` | `51.31%` |
| **RASPRO-12Lead (`6-Slot Anatomical MSM`, 100% Pretrain)** | `74.43%` | `66.86%` | `67.23%` | `83.51%` | `73.01%` |
| **RASPRO-12Lead (`8-Slot + PR & QRS-Axis`, Clean Test)** | `74.56%` | `68.32%` | `71.35%` | `86.34%` | `75.14%` |
| **RASPRO-12Lead (`9-Slot + QT_MORPH Evolved MI`, Clean Test)** | **`78.06%`** | **`69.00%`** | **`70.62%`** | **`84.77%`** | **`75.61%`** |
| **RASPRO-12Lead (`8-Slot`, Test-Time $\pm 20\text{ ms}$ R-Peak Jitter & Noise)** | `74.02%` | `67.30%` | `70.16%` | `85.23%` | `74.18%` |
