#!/usr/bin/env python3
"""Head-to-Head Tokenizer & Clinical Summarization Demo (Raw Patch vs. SAX vs. SAX-BPE vs. RASPRO).

Demonstrates:
  1. Sequence length compression & Transformer O(L^2) self-attention FLOP reduction.
  2. Why Vanilla SAX misses subtle ST-elevation (STEMI) and PPG dicrotic notch loss (AHE),
     whereas RASPRO Physician Assist Filters (PAFs) + Multi-Sensor Matrix (MSM) + CAM/CMI
     capture them explicitly.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raspro_fm.data import CLINICAL_CLASSES, generate_coupled_ecg_ppg
from raspro_fm.tokenizers import (
    PatchTokenizer,
    SAXTokenizer,
    SAXBpeTokenizer,
    RASPROTokenizer,
)


def main() -> None:
    fs = 100
    duration_sec = 30.0

    # Generate one representative recording for each of the 5 clinical classes
    records = [
        generate_coupled_ecg_ppg(
            record_id=f"DEMO_{label}",
            label=label,
            duration_sec=duration_sec,
            fs=fs,
            seed=100 + label * 13,
        )
        for label in sorted(CLINICAL_CLASSES.keys())
    ]

    patch_tok = PatchTokenizer(fs=fs, patch_duration_sec=0.20, stride_sec=0.20)
    sax_tok = SAXTokenizer(fs=fs, paa_window_sec=0.20, alphabet_size=8)
    sax_bpe_tok = SAXBpeTokenizer(fs=fs, paa_window_sec=0.20, alphabet_size=8, num_merges=48)
    sax_bpe_tok.fit([(r.ecg, r.ppg) for r in records])
    raspro_tok = RASPROTokenizer(fs=fs, window_beats=4, active_dedup_max_run=4)

    lines = [
        "# Tokenizer & Clinical Summarization Comparison Report",
        "",
        "Comparison across **30-second coupled ECG + PPG recordings** ($f_s = 100\\text{ Hz}$, $6,000$ raw samples per record):",
        "",
        "| Clinical Condition | Raw Samples | Patch Tokens | Vanilla SAX Tokens | SAX-BPE Tokens | RASPRO Beat MSM Rows | RASPRO Active Tokens | Sequence Compression (vs. Patch) | Attention $O(L^2)$ Compute Savings | Overall CMI |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    detail_sections = ["", "## Detailed RASPRO Motif & SAX Inspection by Clinical Class", ""]

    for rec in records:
        p_out = patch_tok.tokenize(rec.ecg, rec.ppg)
        s_out = sax_tok.tokenize(rec.ecg, rec.ppg)
        b_out = sax_bpe_tok.tokenize(rec.ecg, rec.ppg)
        r_out = raspro_tok.tokenize(rec.ecg, rec.ppg, r_peaks=rec.r_peaks)

        comp_vs_patch = p_out.n_tokens / max(1, r_out.n_active_tokens)
        attn_savings = (p_out.n_tokens / max(1, r_out.n_active_tokens)) ** 2

        lines.append(
            f"| **{rec.label_name}** | {r_out.n_raw_samples} | {p_out.n_tokens} | "
            f"{s_out.n_tokens} | {b_out.n_tokens} | {r_out.n_beat_tokens} | "
            f"**{r_out.n_active_tokens}** | **{comp_vs_patch:.1f}$\\times$** | "
            f"**{attn_savings:.0f}$\\times$** | `{r_out.overall_cmi:.3f}` |"
        )

        detail_sections.extend([
            f"### {rec.label}: {rec.label_name}",
            f"- **Vanilla SAX sample (first 15 tokens):** `{' '.join(s_out.combined_tokens[:15])} ...` ({s_out.n_tokens} tokens)",
            f"- **SAX-BPE sample (first 12 merged subwords):** `{' '.join(b_out.combined_tokens[:12])} ...` ({b_out.n_tokens} tokens)",
            f"- **RASPRO Beat-Level MSM Motifs (`[RR, QRS, ST, PPG_AMP, NOTCH, PAT]`, first 8 beats):** `{' '.join(r_out.beat_motifs[:8])}`",
            f"- **RASPRO Actively Summarized Sequence ({r_out.n_active_tokens} tokens, CMI = `{r_out.overall_cmi:.3f}`):**",
            f"  `{' -> '.join(r_out.active_tokens)}`",
            "",
        ])

    report_text = "\n".join(lines + detail_sections)
    print(report_text)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "tokenizer_comparison_demo.md"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\nSaved detailed tokenizer comparison report to: {out_path}")


if __name__ == "__main__":
    main()
