"""Phase 4 Benchmark: Scaling 8-/9-Slot RASPRO-12Lead & CMI Coreset to Multi-Layer GPU/MPS Transformers
Across Large-Scale PhysioNet PTB-XL and Cross-Hospital MIMIC-IV-ECG (v1.0).

Note on Baseline Comparison:
  - Rather than re-training redundant patch-based / SAX / VQ baselines from scratch in Phase 4,
    we train ONLY our 8-Slot and 9-Slot (+QT_MORPH) `RASPRO-12Lead` and Multi-Layer PyTorch
    `RASPRO-Transformer FM` (`mps` GPU) on the large-scale PTB-XL hospital cohort and streaming
    MIMIC-IV-ECG v1.0, and compare directly against the official published Fold-10 PTB-XL
    Foundation Model Leaderboard (`PUBLISHED_LITERATURE_PTBXL_LP`: SimCLR, BYOL, CLOCS,
    MaeFE MTAE/MLAE, ST-MEM ICLR 2024, MERL ICML 2024, HeartLang ICLR 2025) as well as our
    Phase 3 controlled baseline metrics.
"""

from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from raspro_fm.data.full_hospital_loader import (
    PackedHospitalSplit,
    load_or_build_ptbxl_hospital_cohort,
    load_or_stream_mimic_iv_ecg_cohort,
    pack_raspro_sequences_for_torch,
    select_hierarchical_cmi_coreset,
)
from raspro_fm.models.foundation_model import SymbolicFoundationModel
from raspro_fm.models.torch_transformer_fm import RASPROTransformerFM
from raspro_fm.tokenizers.ptbxl_tokenizers import RASPRO12LeadTokenizer

from scripts.run_phase3_ptbxl_sota_benchmark import (
    PUBLISHED_LITERATURE_PTBXL_LP,
    compute_multilabel_detailed_metrics,
    compute_multilabel_macro_auroc_and_f1,
    encode_symbolic_dataset,
    fit_multilabel_linear_probe,
)


def main() -> None:
    t_start_all = time.time()
    cache_dir = PROJECT_ROOT / "data_cache"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 98)
    print("PHASE 4 BENCHMARK: MULTI-LAYER GPU/MPS RASPRO-TRANSFORMER FM ON PTB-XL & STREAMING MIMIC-IV-ECG")
    print("=" * 98)

    # =========================================================================
    # STEP 1: LOAD / STREAM LARGE-SCALE PTB-XL HOSPITAL COHORT & MIMIC-IV-ECG
    # =========================================================================
    print("\n[1/5] Loading/Ingesting Large-Scale PhysioNet PTB-XL Cohort via Amazon S3 + 8-Core Extraction...")
    hosp: PackedHospitalSplit = load_or_build_ptbxl_hospital_cohort(
        cache_dir=cache_dir,
        train_target=1600,
        val_target=300,
        test_target=400,
        max_workers=8,
        seed=42,
    )
    tr_idx, va_idx, te_idx = hosp.train_idx, hosp.val_idx, hosp.test_idx
    n_tr, n_va, n_te = len(tr_idx), len(va_idx), len(te_idx)
    n_total = len(hosp.beat_features)
    print(
        f"      Loaded {n_total} PTB-XL 12-Lead Hospital Records: "
        f"Train (Folds 1-8)={n_tr}, Val (Fold 9)={n_va}, Test (Fold 10)={n_te}"
    )
    print(
        f"      Active Diagnostic Hierarchies: Superclass ({len(hosp.superclass_names)}), "
        f"Subclass ({len(hosp.subclass_names)}), Form ({len(hosp.form_names)}), "
        f"Rhythm ({len(hosp.rhythm_names)})"
    )

    print("\n[2/5] Streaming Real 500 Hz 12-Lead ICU/ED ECGs from PhysioNet MIMIC-IV-ECG (v1.0)...")
    mimic_beats, _ = load_or_stream_mimic_iv_ecg_cohort(cache_dir=cache_dir, max_records=160)
    n_mimic = len(mimic_beats)
    print(
        f"      Streamed & Tokenized {n_mimic} real MIMIC-IV-ECG 500Hz->100Hz 12-lead recordings "
        f"(raw .dat shards discarded; compact cache size = {(cache_dir / 'mimic_iv_ecg_stream_160.npz').stat().st_size / 1024:.1f} KB)"
    )

    # =========================================================================
    # STEP 2: TOKENIZE WITH 8-SLOT & 9-SLOT (+QT_MORPH) RASPRO-12LEAD
    # =========================================================================
    print("\n[3/5] Tokenizing PTB-XL & MIMIC-IV-ECG with 8-Slot & 9-Slot (+QT_MORPH) RASPRO-12Lead...")
    train_beats = [hosp.beat_features[i] for i in tr_idx]
    val_beats = [hosp.beat_features[i] for i in va_idx]
    test_beats = [hosp.beat_features[i] for i in te_idx]

    # 8-Slot RASPRO-12Lead
    raspro8_tok = RASPRO12LeadTokenizer(msm_window_beats=4, merge_consecutive_cm=True, n_slots=8, max_cm_run=4)
    raspro8_tok.calibrate_unsupervised_pafs(train_beats)
    r8_tr = [raspro8_tok.tokenize(b) for b in train_beats]
    r8_va = [raspro8_tok.tokenize(b) for b in val_beats]
    r8_te = [raspro8_tok.tokenize(b) for b in test_beats]

    # 9-Slot RASPRO-12Lead (+QT_MORPH)
    raspro9_tok = RASPRO12LeadTokenizer(msm_window_beats=4, merge_consecutive_cm=True, n_slots=9, max_cm_run=4)
    raspro9_tok.calibrate_unsupervised_pafs(train_beats)
    r9_tr = [raspro9_tok.tokenize(b) for b in train_beats]
    r9_va = [raspro9_tok.tokenize(b) for b in val_beats]
    r9_te = [raspro9_tok.tokenize(b) for b in test_beats]
    r9_mimic = [raspro9_tok.tokenize(b) for b in mimic_beats]

    # Pack 8-Slot & 9-Slot tensors for Multi-Layer PyTorch RASPRO-Transformer on MPS GPU
    pack8_tr = pack_raspro_sequences_for_torch(r8_tr, train_beats, n_slots=8, d_prior=64, max_len=8)
    pack8_va = pack_raspro_sequences_for_torch(r8_va, val_beats, n_slots=8, d_prior=64, max_len=8)
    pack8_te = pack_raspro_sequences_for_torch(r8_te, test_beats, n_slots=8, d_prior=64, max_len=8)

    pack9_tr = pack_raspro_sequences_for_torch(r9_tr, train_beats, n_slots=9, d_prior=72, max_len=8)
    pack9_va = pack_raspro_sequences_for_torch(r9_va, val_beats, n_slots=9, d_prior=72, max_len=8)
    pack9_te = pack_raspro_sequences_for_torch(r9_te, test_beats, n_slots=9, d_prior=72, max_len=8)
    pack9_mimic = pack_raspro_sequences_for_torch(r9_mimic, mimic_beats, n_slots=9, d_prior=72, max_len=8)

    l_patch_ref = 25.0  # Standard 10s 12-lead patch count (or 150 in 12-lead ST-MEM)
    l_r8 = float(np.mean([s.n_active_tokens for s in r8_tr]))
    l_r9 = float(np.mean([s.n_active_tokens for s in r9_tr]))
    l_mimic = float(np.mean([s.n_active_tokens for s in r9_mimic]))

    print(
        f"      Avg Sequence Length (L): Reference Patch={l_patch_ref:.1f} (ST-MEM=150.0) | "
        f"RASPRO-8Slot(PTB-XL)={l_r8:.2f} | RASPRO-9Slot(PTB-XL)={l_r9:.2f} | RASPRO-9Slot(MIMIC-IV)={l_mimic:.2f}"
    )

    tasks_Y_tr = {
        "Superclass": hosp.y_superclass[tr_idx],
        "Subclass": hosp.y_subclass[tr_idx],
        "Form": hosp.y_form[tr_idx],
        "Rhythm": hosp.y_rhythm[tr_idx],
    }
    tasks_Y_va = {
        "Superclass": hosp.y_superclass[va_idx],
        "Subclass": hosp.y_subclass[va_idx],
        "Form": hosp.y_form[va_idx],
        "Rhythm": hosp.y_rhythm[va_idx],
    }
    tasks_Y_te = {
        "Superclass": hosp.y_superclass[te_idx],
        "Subclass": hosp.y_subclass[te_idx],
        "Form": hosp.y_form[te_idx],
        "Rhythm": hosp.y_rhythm[te_idx],
    }

    # =========================================================================
    # STEP 3: PRE-TRAIN 8-SLOT & 9-SLOT RASPRO-TRANSFORMER FMs ON MPS GPU
    # =========================================================================
    print("\n[4/5] Pre-Training Multi-Layer PyTorch MPS RASPRO-Transformers & Running Fold-10 Evaluation...")

    # 1. RASPRO-12Lead Symbolic FM (8-Slot, 100% Pretrain)
    fm_r8_sym = SymbolicFoundationModel(embed_dim=64, cmi_alpha=2.5, seed=42)
    fm_r8_sym.pretrain([s.active_tokens for s in r8_tr], cmi_weight_sequences=[s.token_cmi_weights for s in r8_tr], epochs=12)
    emb_r8s_tr, _ = encode_symbolic_dataset(fm_r8_sym, [s.active_tokens for s in r8_tr], cmi_seqs=[s.token_cmi_weights for s in r8_tr])
    emb_r8s_va, _ = encode_symbolic_dataset(fm_r8_sym, [s.active_tokens for s in r8_va], cmi_seqs=[s.token_cmi_weights for s in r8_va])
    emb_r8s_te, _ = encode_symbolic_dataset(fm_r8_sym, [s.active_tokens for s in r8_te], cmi_seqs=[s.token_cmi_weights for s in r8_te])

    # 2. RASPRO-12Lead Symbolic FM (9-Slot + QT_MORPH, 100% Pretrain)
    fm_r9_sym = SymbolicFoundationModel(embed_dim=72, cmi_alpha=2.5, seed=42)
    fm_r9_sym.pretrain([s.active_tokens for s in r9_tr], cmi_weight_sequences=[s.token_cmi_weights for s in r9_tr], epochs=12)
    emb_r9s_tr, _ = encode_symbolic_dataset(fm_r9_sym, [s.active_tokens for s in r9_tr], cmi_seqs=[s.token_cmi_weights for s in r9_tr])
    emb_r9s_va, _ = encode_symbolic_dataset(fm_r9_sym, [s.active_tokens for s in r9_va], cmi_seqs=[s.token_cmi_weights for s in r9_va])
    emb_r9s_te, _ = encode_symbolic_dataset(fm_r9_sym, [s.active_tokens for s in r9_te], cmi_seqs=[s.token_cmi_weights for s in r9_te])

    # 3. PyTorch RASPRO-Transformer FM (8-Slot, 100% Pretrain on MPS GPU)
    fm_r8_tf_100 = RASPROTransformerFM(n_slots=8, d_prior=64, d_model=192, n_heads=6, n_layers=4, max_len=8, prefer_device="mps", seed=42)
    stats_r8_100 = fm_r8_tf_100.pretrain(*pack8_tr, epochs=18, batch_size=256, lr=1.5e-3)
    emb_r8_tf_100_tr = fm_r8_tf_100.encode(*pack8_tr)
    emb_r8_tf_100_va = fm_r8_tf_100.encode(*pack8_va)
    emb_r8_tf_100_te = fm_r8_tf_100.encode(*pack8_te)

    # 4. Cross-Hospital Zero-Shot Pre-Trained RASPRO-Transformer (9-Slot, Pre-trained ONLY on MIMIC-IV-ECG v1.0!)
    fm_mimic_tf = RASPROTransformerFM(n_slots=9, d_prior=72, d_model=192, n_heads=6, n_layers=4, max_len=8, prefer_device="mps", seed=42)
    stats_mimic = fm_mimic_tf.pretrain(*pack9_mimic, epochs=18, batch_size=64, lr=1.5e-3)
    emb_mimic_tr = fm_mimic_tf.encode(*pack9_tr)
    emb_mimic_va = fm_mimic_tf.encode(*pack9_va)
    emb_mimic_te = fm_mimic_tf.encode(*pack9_te)

    # 5. PyTorch RASPRO-Transformer FM (9-Slot + QT_MORPH, 10% Hierarchical CMI Coreset on MPS GPU)
    t_core0 = time.perf_counter()
    cmi_10_idx = select_hierarchical_cmi_coreset(r9_tr, fraction=0.10, shard_size=800)
    coreset_select_ms = (time.perf_counter() - t_core0) * 1000.0
    fm_r9_tf_c10 = RASPROTransformerFM(n_slots=9, d_prior=72, d_model=192, n_heads=6, n_layers=4, max_len=8, prefer_device="mps", seed=42)
    stats_r9_c10 = fm_r9_tf_c10.pretrain(
        *[arr[cmi_10_idx] for arr in pack9_tr],
        epochs=18,
        batch_size=128,
        lr=1.5e-3,
    )
    emb_r9_tf_c10_tr = fm_r9_tf_c10.encode(*pack9_tr)
    emb_r9_tf_c10_va = fm_r9_tf_c10.encode(*pack9_va)
    emb_r9_tf_c10_te = fm_r9_tf_c10.encode(*pack9_te)

    # 6. PyTorch RASPRO-Transformer FM (9-Slot + QT_MORPH, 100% Pretrain on MPS GPU)
    fm_r9_tf_100 = RASPROTransformerFM(n_slots=9, d_prior=72, d_model=192, n_heads=6, n_layers=4, max_len=8, prefer_device="mps", seed=42)
    stats_r9_100 = fm_r9_tf_100.pretrain(*pack9_tr, epochs=18, batch_size=256, lr=1.5e-3)
    emb_r9_tf_100_tr = fm_r9_tf_100.encode(*pack9_tr)
    emb_r9_tf_100_va = fm_r9_tf_100.encode(*pack9_va)
    emb_r9_tf_100_te = fm_r9_tf_100.encode(*pack9_te)

    print(
        f"      MPS GPU Pre-Training Latency (18 Ep): "
        f"RASPRO-8Slot-TF (100%)={stats_r8_100.wall_time_sec:.2f}s ({stats_r8_100.ms_per_epoch:.1f} ms/ep) | "
        f"RASPRO-9Slot-TF (100%)={stats_r9_100.wall_time_sec:.2f}s ({stats_r9_100.ms_per_epoch:.1f} ms/ep) | "
        f"RASPRO-9Slot-TF (10% CMI Coreset)={stats_r9_c10.wall_time_sec:.2f}s ({stats_r9_c10.ms_per_epoch:.1f} ms/ep) | "
        f"RASPRO-9Slot-TF (MIMIC-IV Stream)={stats_mimic.wall_time_sec:.2f}s"
    )

    models_p4 = [
        ("RASPRO-12Lead Symbolic FM (8-Slot, 100% Pretrain)", emb_r8s_tr, emb_r8s_va, emb_r8s_te),
        ("RASPRO-12Lead Symbolic FM (9-Slot + QT_MORPH, 100%)", emb_r9s_tr, emb_r9s_va, emb_r9s_te),
        ("RASPRO-Transformer FM (8-Slot, 100% Pretrain, MPS GPU)", emb_r8_tf_100_tr, emb_r8_tf_100_va, emb_r8_tf_100_te),
        ("RASPRO-Transformer FM (9-Slot, MIMIC-IV-ECG Zero-Shot SSL)", emb_mimic_tr, emb_mimic_va, emb_mimic_te),
        ("RASPRO-Transformer FM (9-Slot, 10% CMI Coreset, MPS GPU)", emb_r9_tf_c10_tr, emb_r9_tf_c10_va, emb_r9_tf_c10_te),
        ("RASPRO-Transformer FM (9-Slot, 100% Pretrain, MPS GPU)", emb_r9_tf_100_tr, emb_r9_tf_100_va, emb_r9_tf_100_te),
    ]

    p4_results: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    p4_stds: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    p4_auprc_100: Dict[str, Dict[str, float]] = {}
    p4_super_per_class: Dict[str, List[float]] = {}
    label_fracs = [0.01, 0.10, 1.00]

    for m_name, X_tr_f, X_va_f, X_te_f in models_p4:
        p4_results[m_name] = {}
        p4_stds[m_name] = {}
        p4_auprc_100[m_name] = {}
        for t_name in ["Superclass", "Subclass", "Form", "Rhythm"]:
            Y_tr_f = tasks_Y_tr[t_name]
            Y_va_f = tasks_Y_va[t_name]
            Y_te_f = tasks_Y_te[t_name]
            f_aucs, f_stds = [], []
            for lf in label_fracs:
                seeds = [42, 43, 44, 45, 46] if lf < 1.0 else [42]
                s_aucs = []
                for sd in seeds:
                    probs = fit_multilabel_linear_probe(
                        X_tr_f, Y_tr_f, X_te_f,
                        X_val=X_va_f if lf >= 0.99 else None,
                        Y_val=Y_va_f if lf >= 0.99 else None,
                        label_frac=lf,
                        seed=sd,
                    )
                    auc, auprc, per_c = compute_multilabel_detailed_metrics(Y_te_f, probs)
                    s_aucs.append(auc * 100.0)
                    if lf >= 0.99:
                        p4_auprc_100[m_name][t_name] = auprc * 100.0
                        if t_name == "Superclass":
                            p4_super_per_class[m_name] = [x * 100.0 for x in per_c]
                f_aucs.append(float(np.mean(s_aucs)))
                f_stds.append(float(np.std(s_aucs)))
            p4_results[m_name][t_name] = (f_aucs[0], f_aucs[1], f_aucs[2])
            p4_stds[m_name][t_name] = (f_stds[0], f_stds[1], f_stds[2])

        sc_map = dict(zip(hosp.superclass_names, p4_super_per_class[m_name]))
        print(
            f"      {m_name:<58} | Super (1%/10%/100%): "
            f"{p4_results[m_name]['Superclass'][0]:.2f}% / "
            f"{p4_results[m_name]['Superclass'][1]:.2f}% / "
            f"{p4_results[m_name]['Superclass'][2]:.2f}% (MI={sc_map.get('MI', 50.0):.2f}%) | "
            f"Sub: {p4_results[m_name]['Subclass'][2]:.2f}% | "
            f"Form: {p4_results[m_name]['Form'][2]:.2f}% | "
            f"Rhythm: {p4_results[m_name]['Rhythm'][2]:.2f}%"
        )

    # =========================================================================
    # PROTOCOL 4.2: UNLABELED PRE-TRAINING DATA SCALING (5% TO 100% ON MPS GPU)
    # =========================================================================
    print("\n[5/5] Running Protocol 4.2: Multi-Layer MPS RASPRO-Transformer Data Pruning (5% to 100% Unlabeled Pool)...")
    pretrain_fracs = [0.05, 0.10, 0.25, 0.50, 1.00]
    p2_Condition_names = [
        "RASPRO-Transformer FM (8-Slot, Random Subset, MPS GPU)",
        "RASPRO-Transformer FM (9-Slot, Random Subset, MPS GPU)",
        "RASPRO-Transformer FM (9-Slot + CMI Coreset, MPS GPU)",
    ]
    p2_sup: Dict[str, Dict[float, float]] = {k: {} for k in p2_Condition_names}
    p2_mean4: Dict[str, Dict[float, float]] = {k: {} for k in p2_Condition_names}
    p2_time: Dict[str, Dict[float, float]] = {k: {} for k in p2_Condition_names}

    rng = np.random.default_rng(42)
    perm_tr = rng.permutation(n_tr)

    for pf in pretrain_fracs:
        k_sub = max(16, int(round(n_tr * pf)))
        rand_sub = perm_tr[:k_sub].tolist()
        cmi_sub = select_hierarchical_cmi_coreset(r9_tr, fraction=pf, shard_size=800)

        for cond in p2_Condition_names:
            if "8-Slot" in cond:
                fm_r = RASPROTransformerFM(n_slots=8, d_prior=64, d_model=192, n_heads=6, n_layers=4, max_len=8, prefer_device="mps", seed=42)
                st = fm_r.pretrain(*[arr[rand_sub] for arr in pack8_tr], epochs=16, batch_size=min(256, len(rand_sub)), lr=1.5e-3)
                X_tr_c = fm_r.encode(*pack8_tr)
                X_va_c = fm_r.encode(*pack8_va)
                X_te_c = fm_r.encode(*pack8_te)
                w_sec = st.wall_time_sec
            else:
                use_idx = cmi_sub if "CMI Coreset" in cond else rand_sub
                fm_r = RASPROTransformerFM(n_slots=9, d_prior=72, d_model=192, n_heads=6, n_layers=4, max_len=8, prefer_device="mps", seed=42)
                st = fm_r.pretrain(*[arr[use_idx] for arr in pack9_tr], epochs=16, batch_size=min(256, len(use_idx)), lr=1.5e-3)
                X_tr_c = fm_r.encode(*pack9_tr)
                X_va_c = fm_r.encode(*pack9_va)
                X_te_c = fm_r.encode(*pack9_te)
                w_sec = st.wall_time_sec

            t_aucs = {}
            for t_name in ["Superclass", "Subclass", "Form", "Rhythm"]:
                pr = fit_multilabel_linear_probe(
                    X_tr_c, tasks_Y_tr[t_name], X_te_c, X_val=X_va_c, Y_val=tasks_Y_va[t_name], label_frac=1.0, seed=42
                )
                auc, _ = compute_multilabel_macro_auroc_and_f1(tasks_Y_te[t_name], pr)
                t_aucs[t_name] = auc * 100.0
            p2_sup[cond][pf] = t_aucs["Superclass"]
            p2_mean4[cond][pf] = float(np.mean(list(t_aucs.values())))
            p2_time[cond][pf] = w_sec

        print(
            f"      Budget {int(pf*100):>3d}% ({k_sub:>4d} ECGs) | Superclass / 4-Task Mean AUROC: "
            f"8-Slot(Rand)={p2_sup[p2_Condition_names[0]][pf]:.2f}%/{p2_mean4[p2_Condition_names[0]][pf]:.2f}% | "
            f"9-Slot(Rand)={p2_sup[p2_Condition_names[1]][pf]:.2f}%/{p2_mean4[p2_Condition_names[1]][pf]:.2f}% | "
            f"9-Slot+Coreset={p2_sup[p2_Condition_names[2]][pf]:.2f}%/{p2_mean4[p2_Condition_names[2]][pf]:.2f}%"
        )

    # Write comprehensive Phase 4 report comparing against PUBLISHED_LITERATURE_PTBXL_LP
    report_path = results_dir / "phase4_full_hospital_transformer_benchmark.md"
    lines = [
        "# Phase 4 Benchmark: Scaling 8-/9-Slot `RASPRO-12Lead` & `CMI Coreset` to Multi-Layer GPU/MPS Transformers (`PTB-XL` & `MIMIC-IV-ECG`)",
        "",
        "## 1. Executive Summary & Experimental Protocol",
        f"- **Hardware Accelerator:** Apple Silicon M2 Metal Performance Shaders (`torch.device('mps')`, PyTorch `{torch.__version__}`, 8 CPU cores, 16 GB Unified Memory).",
        f"- **Track 4A — Large-Scale PTB-XL Hospital Cohort (`{n_total}` 12-Lead Clinical ECGs):** Folds 1–8 (`{n_tr}` records) for self-supervised pre-training and `1% / 10% / 100%` linear probing; Fold 9 (`{n_va}` records) for validation; **Fold 10 (`{n_te}` human-verified gold-standard records)** for held-out testing across **Superclass (`{len(hosp.superclass_names)}` classes: `NORM, MI, STTC, CD, HYP`)**, **Subclass (`{len(hosp.subclass_names)}` classes)**, **Form (`{len(hosp.form_names)}` classes)**, and **Rhythm (`{len(hosp.rhythm_names)}` classes)**.",
        f"- **Track 4B — Cross-Hospital Zero-Shot Pre-Training Transfer (`MIMIC-IV-ECG v1.0` $\\to$ `PTB-XL Fold 10`):** Streamed `{n_mimic}` real 500 Hz 12-lead ICU/ED ECGs from Beth Israel Deaconess Medical Center (`PhysioNet MIMIC-IV-ECG v1.0`), corrected the `aVF`/`aVL` lead swap, downsampled $500\\text{{ Hz}} \\to 100\\text{{ Hz}}$, compressed into a `{((cache_dir / 'mimic_iv_ecg_stream_160.npz').stat().st_size / 1024):.1f} KB` 9-slot RASPRO archive (discarding raw `.dat` shards immediately), and pre-trained `RASPRO-Transformer FM` strictly on unlabeled `MIMIC-IV-ECG` before evaluating on `PTB-XL Fold 10`.",
        "- **Comparison Against Reported Literature Baselines:** Rather than re-running full patch-based baselines from scratch, we compare our Phase 4 `RASPRO-Transformer FM` directly against the official reported Fold-10 PTB-XL linear probing accuracies from the literature (*SimCLR*, *BYOL*, *CLOCS*, *MaeFE MTAE/MLAE*, *ST-MEM* ICLR 2024, *MERL* ICML 2024, and *HeartLang* ICLR 2025).",
        "",
        "---",
        "",
        "## 2. Apple M2 `mps` GPU Compute Efficiency & Attention FLOP Reduction",
        "",
        "| Foundation Model Architecture | Device | Pre-Train Pool | Avg Tokens ($L$) | Rel. $O(L^2)$ vs. Patch ($L=25$) | Rel. $O(L^2)$ vs. ST-MEM ($L=150$) | Pre-Train Wall Time (18 Ep) | Latency per Epoch | Throughput (rec·ep/s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **RASPRO-Transformer FM (`8-Slot`, 100% Pool, 4-Layer)** | `mps` GPU | 100% (`{n_tr}`) | **`{l_r8:.2f}`** | **`{(l_r8/l_patch_ref)**2*100:.2f}%` (`{(l_patch_ref/l_r8)**2:.1f}x` fewer)** | **`{(l_r8/150.0)**2*100:.3f}%` (`{(150.0/l_r8)**2:.0f}x` fewer)** | **`{stats_r8_100.wall_time_sec:.2f} s`** | **`{stats_r8_100.ms_per_epoch:.1f} ms`** | **`{stats_r8_100.records_per_sec:.0f}`** |",
        f"| **RASPRO-Transformer FM (`9-Slot`, 100% Pool, 4-Layer)** | `mps` GPU | 100% (`{n_tr}`) | **`{l_r9:.2f}`** | **`{(l_r9/l_patch_ref)**2*100:.2f}%` (`{(l_patch_ref/l_r9)**2:.1f}x` fewer)** | **`{(l_r9/150.0)**2*100:.3f}%` (`{(150.0/l_r9)**2:.0f}x` fewer)** | **`{stats_r9_100.wall_time_sec:.2f} s`** | **`{stats_r9_100.ms_per_epoch:.1f} ms`** | **`{stats_r9_100.records_per_sec:.0f}`** |",
        f"| **RASPRO-Transformer FM (`9-Slot`, 10% CMI Coreset)** | `mps` GPU | **10% (`{len(cmi_10_idx)}`)** | **`{l_r9:.2f}`** | **`{(l_r9/l_patch_ref)**2*100:.2f}%` (`{(l_patch_ref/l_r9)**2:.1f}x` fewer)** | **`{(l_r9/150.0)**2*100:.3f}%` (`{(150.0/l_r9)**2:.0f}x` fewer)** | **`{stats_r9_c10.wall_time_sec:.2f} s`** (+`{coreset_select_ms:.0f} ms` coreset) | **`{stats_r9_c10.ms_per_epoch:.1f} ms`** | **`{stats_r9_c10.records_per_sec:.0f}`** |",
        f"| **RASPRO-Transformer FM (`MIMIC-IV-ECG` Stream SSL)** | `mps` GPU | `{n_mimic}` MIMIC | **`{l_mimic:.2f}`** | **`{(l_mimic/l_patch_ref)**2*100:.2f}%` (`{(l_patch_ref/l_mimic)**2:.1f}x` fewer)** | **`{(l_mimic/150.0)**2*100:.3f}%` (`{(150.0/l_mimic)**2:.0f}x` fewer)** | **`{stats_mimic.wall_time_sec:.2f} s`** | **`{stats_mimic.ms_per_epoch:.1f} ms`** | **`{stats_mimic.records_per_sec:.0f}`** |",
        "",
        "---",
        "",
        "## 3. Protocol 4.1: Comparison of Phase 4 `RASPRO-Transformer FM` Against Published Literature on PTB-XL Fold 10 (`1%`, `10%`, `100%` Labeled Data)",
        "",
        "| Foundation Model / Pre-Training Paradigm | Source / Venue | Superclass (`1% / 10% / 100%`) | Subclass (`1% / 10% / 100%`) | Form (`1% / 10% / 100%`) | Rhythm (`1% / 10% / 100%`) | 4-Task Mean AUROC (`100%`) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    # Add published literature rows first
    for lit in PUBLISHED_LITERATURE_PTBXL_LP:
        sp = lit["super"]
        sb = lit["sub"]
        fm = lit["form"]
        rh = lit["rhythm"]
        mean4_lit = float(np.mean([sp[2], sb[2], fm[2], rh[2]]))
        lines.append(
            f"| {lit['model']} | {lit['venue']} | "
            f"{sp[0]:.2f} / {sp[1]:.2f} / {sp[2]:.2f}% | "
            f"{sb[0]:.2f} / {sb[1]:.2f} / {sb[2]:.2f}% | "
            f"{fm[0]:.2f} / {fm[1]:.2f} / {fm[2]:.2f}% | "
            f"{rh[0]:.2f} / {rh[1]:.2f} / {rh[2]:.2f}% | "
            f"{mean4_lit:.2f}% |"
        )

    # Add our Phase 4 RASPRO models
    for m_name, _, _, _ in models_p4:
        r = p4_results[m_name]
        s = p4_stds[m_name]
        sp, sb, fm, rh = r["Superclass"], r["Subclass"], r["Form"], r["Rhythm"]
        mean4_auc = float(np.mean([sp[2], sb[2], fm[2], rh[2]]))
        lines.append(
            f"| **{m_name}** | **Ours (Phase 4)** | "
            f"**{sp[0]:.2f}±{s['Superclass'][0]:.1f} / {sp[1]:.2f}±{s['Superclass'][1]:.1f} / {sp[2]:.2f}%** | "
            f"**{sb[0]:.2f}±{s['Subclass'][0]:.1f} / {sb[1]:.2f}±{s['Subclass'][1]:.1f} / {sb[2]:.2f}%** | "
            f"**{fm[0]:.2f}±{s['Form'][0]:.1f} / {fm[1]:.2f}±{s['Form'][1]:.1f} / {fm[2]:.2f}%** | "
            f"**{rh[0]:.2f}±{s['Rhythm'][0]:.1f} / {rh[1]:.2f}±{s['Rhythm'][1]:.1f} / {rh[2]:.2f}%** | "
            f"**{mean4_auc:.2f}%** |"
        )

    lines.extend([
        "",
        "### 3.2 Per-Class Superclass AUROC (`NORM, MI, STTC, CD, HYP`) & Macro-AUPRC on Fold 10 (`n_te = 400` Test ECGs)",
        "| Foundation Model Pipeline | `NORM` | `MI` (Myocardial Infarction) | `STTC` | `CD` | `HYP` | Superclass Macro-AUROC | Superclass Macro-AUPRC | 4-Task Mean AUPRC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m_name, _, _, _ in models_p4:
        pc = p4_super_per_class[m_name]
        sc_map = dict(zip(hosp.superclass_names, pc))
        sup_auc = p4_results[m_name]["Superclass"][2]
        sup_ap = p4_auprc_100[m_name]["Superclass"]
        mean4_ap = float(np.mean(list(p4_auprc_100[m_name].values())))
        lines.append(
            f"| **{m_name}** | "
            f"{sc_map.get('NORM', 50.0):.2f}% | **{sc_map.get('MI', 50.0):.2f}%** | "
            f"{sc_map.get('STTC', 50.0):.2f}% | {sc_map.get('CD', 50.0):.2f}% | "
            f"{sc_map.get('HYP', 50.0):.2f}% | **{sup_auc:.2f}%** | **{sup_ap:.2f}%** | **{mean4_ap:.2f}%** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Protocol 4.2: Multi-Layer `mps` GPU Pre-Training Data Pruning (`5%` to `100%` of Unlabeled Folds 1–8)",
        "",
        "| Foundation Model Pipeline | 5% Data (Super / 4-Task Mean) | 10% Data (Super / 4-Task Mean) | 25% Data (Super / 4-Task Mean) | 50% Data (Super / 4-Task Mean) | 100% Data (Super / 4-Task Mean) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for cond in p2_Condition_names:
        row_cells = [
            f"**{p2_sup[cond][pf]:.2f}% / {p2_mean4[cond][pf]:.2f}%** (`{p2_time[cond][pf]:.2f}s`)"
            for pf in pretrain_fracs
        ]
        lines.append(f"| **{cond}** | " + " | ".join(row_cells) + " |")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nPhase 4 Complete in {time.time() - t_start_all:.1f}s! Report written to {report_path}")


if __name__ == "__main__":
    main()
