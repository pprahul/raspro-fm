"""Generate publication-grade IEEE JBHI architecture diagrams for RASPRO-FM.

Produces high-resolution PNG (300 DPI) and vector PDF figures with:
  - Header-band card design and high-contrast typography
  - Exact zero-pad bounding boxes (`pad=0.0`) so arrows never overlap box borders
  - Clean orthogonal routing corridors between stages (never crossing text or titles)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


def draw_card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    header_bg: str = "#1E3A8A",
    body_bg: str = "#F8FAFC",
    edge_color: str = "#1E3A8A",
    header_fg: str = "#FFFFFF",
    body_fg: str = "#0F172A",
    title_fs: float = 10.6,
    body_fs: float = 9.4,
    header_ratio: float = 0.26,
    lw: float = 1.8,
    radius: float = 0.012,
):
    """Draw a publication-style card with a solid colored header band and crisp body text."""
    card_bg = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=body_bg,
        edgecolor=edge_color,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(card_bg)

    hh = h * header_ratio
    hy = y + h - hh
    header_patch = patches.FancyBboxPatch(
        (x, hy),
        w,
        hh,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=header_bg,
        edgecolor=edge_color,
        linewidth=lw,
        zorder=4,
    )
    ax.add_patch(header_patch)
    rect_fill = patches.Rectangle(
        (x + lw * 0.0005, hy),
        w - lw * 0.001,
        hh * 0.45,
        facecolor=header_bg,
        edgecolor="none",
        zorder=4,
    )
    ax.add_patch(rect_fill)

    ax.text(
        x + w / 2.0,
        hy + hh / 2.0,
        title,
        ha="center",
        va="center",
        fontsize=title_fs,
        fontweight="bold",
        color=header_fg,
        linespacing=1.15,
        zorder=5,
    )

    body_h = h - hh
    body_text = "\n".join(lines)
    ax.text(
        x + w / 2.0,
        y + body_h / 2.0,
        body_text,
        ha="center",
        va="center",
        fontsize=body_fs,
        color=body_fg,
        linespacing=1.32,
        fontweight="medium",
        zorder=5,
    )


def draw_straight_arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#1E293B",
    lw: float = 2.4,
    label: str = "",
    label_pos: tuple[float, float] | None = None,
    label_fs: float = 9.5,
):
    """Draw a straight arrow with exact endpoints and optional pill label."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.65,head_width=0.38",
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=6,
    )
    if label:
        lx, ly = label_pos if label_pos is not None else ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=label_fs,
            fontweight="bold",
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.22,rounding_size=0.15",
                facecolor="#FFFFFF",
                edgecolor=color,
                linewidth=1.2,
            ),
            zorder=7,
        )


def draw_orthogonal_arrow(
    ax,
    points: list[tuple[float, float]],
    color: str = "#1E293B",
    lw: float = 2.4,
    label: str = "",
    label_pos: tuple[float, float] | None = None,
    label_fs: float = 9.5,
):
    """Draw a clean multi-segment orthogonal polyline with an arrowhead on the final segment."""
    xs = [p[0] for p in points[:-1]]
    ys = [p[1] for p in points[:-1]]
    if len(points) > 2:
        ax.plot(xs, ys, color=color, linewidth=lw, solid_capstyle="round", zorder=6)
    x_prev, y_prev = points[-2]
    x_end, y_end = points[-1]
    ax.annotate(
        "",
        xy=(x_end, y_end),
        xytext=(x_prev, y_prev),
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.65,head_width=0.38",
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=6,
    )
    if label and label_pos is not None:
        ax.text(
            label_pos[0],
            label_pos[1],
            label,
            ha="center",
            va="center",
            fontsize=label_fs,
            fontweight="bold",
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.26,rounding_size=0.15",
                facecolor="#FFFFFF",
                edgecolor=color,
                linewidth=1.4,
            ),
            zorder=7,
        )


def generate_tokenizer_comparison_diagram(out_paths: list[str]):
    """Figure 1: Clean, high-readability comparison of 5 biosignal tokenization paradigms."""
    fig, ax = plt.subplots(figsize=(15.6, 8.0), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    ax.text(
        0.5,
        0.962,
        "Comparison of Biosignal Tokenization Paradigms for Foundation Models",
        ha="center",
        va="center",
        fontsize=16.0,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        0.5,
        0.924,
        "Evaluating Sequence Length (L), Physiological Fidelity, and O(L²) Self-Attention Compute on 10s 12-Lead ECG & 8-Min ICU Streams",
        ha="center",
        va="center",
        fontsize=11.0,
        color="#334155",
    )

    col_headers = [
        (0.012, 0.205, "Tokenization Paradigm", "#1E293B"),
        (0.242, 0.220, "Step 1: Temporal Segmentation", "#1E293B"),
        (0.487, 0.255, "Step 2: Symbol / Token Representation", "#1E293B"),
        (0.767, 0.221, "Sequence Length (L) & Efficiency", "#1E293B"),
    ]
    for cx, cw, ctitle, ccol in col_headers:
        hdr = patches.FancyBboxPatch(
            (cx, 0.855),
            cw,
            0.042,
            boxstyle="round,pad=0.0,rounding_size=0.008",
            facecolor=ccol,
            edgecolor=ccol,
            linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(hdr)
        ax.text(
            cx + cw / 2.0,
            0.876,
            ctitle,
            ha="center",
            va="center",
            fontsize=10.6,
            fontweight="bold",
            color="#FFFFFF",
            zorder=4,
        )

    rows = [
        {
            "y": 0.690,
            "name": "1. Continuous Patch FM\n(ST-MEM / PatchTST)",
            "seg": "Fixed 0.2s–0.4s Clock Slices\nSplits P-QRS-T across boundaries",
            "rep": "Linear Patch Projection (x_p ∈ ℝ^d)\nNo beat lock; sensitive to phase shift",
            "eff": "12-Lead (10s): L = 25.0 tokens\n8-Min ICU: L = 2,400 tokens\nAttention FLOPs: 100% (1.0×)",
            "h_bg": "#475569",
            "b_bg": "#F8FAFC",
            "edge": "#475569",
            "txt": "#0F172A",
        },
        {
            "y": 0.520,
            "name": "2. Vanilla SAX FM\n(PAA + Gaussian Bins)",
            "seg": "Fixed 0.2s–0.4s Window +\nGlobal Z-Normalization",
            "rep": "Gaussian Quantiles {a, b, c, d, e}\nQRS spike dominates variance; blurs ST",
            "eff": "12-Lead (10s): L = 25.0 tokens\n8-Min ICU: L = 2,400 tokens\nAttention FLOPs: 100% (1.0×)",
            "h_bg": "#0369A1",
            "b_bg": "#F0F9FF",
            "edge": "#0284C7",
            "txt": "#0C4A6E",
        },
        {
            "y": 0.350,
            "name": "3. ECG-Byte FM\n(SAX + BPE Subwords)",
            "seg": "Fixed-Clock SAX Stream +\nByte-Pair Encoding (BPE)",
            "rep": "Merges frequent normal bigrams\nBlind to calibrated mV ST/PR shifts",
            "eff": "12-Lead (10s): L = 20.0 tokens\n8-Min ICU: L = 1,870 tokens\nAttention FLOPs: 64.2% (1.6× less)",
            "h_bg": "#4338CA",
            "b_bg": "#EEF2FF",
            "edge": "#4F46E5",
            "txt": "#1E1B4B",
        },
        {
            "y": 0.180,
            "name": "4. HeartLang-QRS FM\n(ICLR 2025 Beat VQ)",
            "seg": "R-Peak Heartbeat Slicing\n(1 Token per Heartbeat)",
            "rep": "Learned Vector-Quantized Codebook\nNo CM merge; degrades @ 5% data",
            "eff": "12-Lead (10s): L = 13.6 tokens\n8-Min ICU: L = 730 tokens\nAttention FLOPs: 29.6% (3.4× less)",
            "h_bg": "#0F766E",
            "b_bg": "#F0FDFA",
            "edge": "#0D9488",
            "txt": "#134E4A",
        },
        {
            "y": 0.010,
            "name": "5. RASPRO-FM (Ours)\n(PAF + MSM + CM/CAM)",
            "seg": "R-Peak Lock + 9-Slot 12-Lead\n& 6-Slot ICU PAF {A,B,N,Y,Z}",
            "rep": "4-Beat Consensus Motifs (CM & CAM)\n+ Active Run-Length Deduplication",
            "eff": "12-Lead: L = 3.54 (1,797× vs ST-MEM)\n8-Min ICU: L = 51–129 (2,215× less)\nFold-10: 79.8% Super / 79.1% Mean",
            "h_bg": "#B45309",
            "b_bg": "#FEF3C7",
            "edge": "#D97706",
            "txt": "#78350F",
        },
    ]

    bh = 0.145
    for r in rows:
        y = r["y"]
        ym = y + bh / 2.0
        is_ours = "RASPRO-FM" in r["name"]
        lw = 2.4 if is_ours else 1.6

        # Box 1
        p1 = patches.FancyBboxPatch(
            (0.012, y),
            0.205,
            bh,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            facecolor=r["h_bg"],
            edgecolor=r["edge"],
            linewidth=lw,
            zorder=3,
        )
        ax.add_patch(p1)
        ax.text(
            0.012 + 0.205 / 2.0,
            ym,
            r["name"],
            ha="center",
            va="center",
            fontsize=10.8,
            fontweight="bold",
            color="#FFFFFF",
            linespacing=1.25,
            zorder=4,
        )

        draw_straight_arrow(ax, 0.219, ym, 0.240, ym, color=r["edge"], lw=2.2)

        # Box 2
        p2 = patches.FancyBboxPatch(
            (0.242, y),
            0.220,
            bh,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            facecolor=r["b_bg"],
            edgecolor=r["edge"],
            linewidth=lw,
            zorder=3,
        )
        ax.add_patch(p2)
        ax.text(
            0.242 + 0.220 / 2.0,
            ym,
            r["seg"],
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold" if is_ours else "medium",
            color=r["txt"],
            linespacing=1.3,
            zorder=4,
        )

        draw_straight_arrow(ax, 0.464, ym, 0.485, ym, color=r["edge"], lw=2.2)

        # Box 3
        p3 = patches.FancyBboxPatch(
            (0.487, y),
            0.255,
            bh,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            facecolor=r["b_bg"],
            edgecolor=r["edge"],
            linewidth=lw,
            zorder=3,
        )
        ax.add_patch(p3)
        ax.text(
            0.487 + 0.255 / 2.0,
            ym,
            r["rep"],
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold" if is_ours else "medium",
            color=r["txt"],
            linespacing=1.3,
            zorder=4,
        )

        draw_straight_arrow(ax, 0.744, ym, 0.765, ym, color=r["edge"], lw=2.2)

        # Box 4
        p4 = patches.FancyBboxPatch(
            (0.767, y),
            0.221,
            bh,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            facecolor=r["b_bg"],
            edgecolor=r["edge"],
            linewidth=lw,
            zorder=3,
        )
        ax.add_patch(p4)
        ax.text(
            0.767 + 0.221 / 2.0,
            ym,
            r["eff"],
            ha="center",
            va="center",
            fontsize=9.8,
            fontweight="bold",
            color=r["txt"],
            linespacing=1.28,
            zorder=4,
        )

    plt.tight_layout(pad=0.3)
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_end_to_end_diagram(out_paths: list[str]):
    """Figure 2: End-to-end RASPRO-FM architecture with zero overlapping arrows and large crisp text."""
    fig, ax = plt.subplots(figsize=(16.0, 9.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Top Title Banner
    ax.text(
        0.5,
        0.968,
        "RASPRO-FM: End-to-End Clinical Symbolic Summarization & Foundation Model Architecture",
        ha="center",
        va="center",
        fontsize=16.0,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        0.5,
        0.935,
        "Extending RASPRO Edge Summarization (PAF + SSM/MSM + CM/CAM + CMI) to Compositional Self-Supervised Foundation Models",
        ha="center",
        va="center",
        fontsize=11.0,
        color="#334155",
    )

    # =========================================================================
    # STAGE 1: y = 0.675 .. 0.905 (Height = 0.230)
    # =========================================================================
    s1_panel = patches.FancyBboxPatch(
        (0.012, 0.675),
        0.976,
        0.230,
        boxstyle="round,pad=0.0,rounding_size=0.014",
        facecolor="#F1F5F9",
        edgecolor="#64748B",
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(s1_panel)
    ax.text(
        0.025,
        0.883,
        "STAGE 1: Beat-Synchronous Multi-Sensor & 12-Lead Anatomical Feature Extraction + PAF Quantization",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#0F172A",
        zorder=2,
    )

    y1 = 0.690
    h1 = 0.172
    ym1 = y1 + h1 / 2.0

    # Card 1.1: x=0.024, w=0.205 -> right=0.229
    draw_card(
        ax,
        0.024,
        y1,
        0.205,
        h1,
        "1. Raw Biosignal Streams",
        [
            "• ICU Multi-Sensor (125 Hz):",
            "  3-Lead ECG + PPG + RESP",
            "• Clinical 12-Lead ECG (100 Hz):",
            "  I, II, III, aVR, aVL, aVF, V1–V6",
        ],
        header_bg="#0369A1",
        body_bg="#FFFFFF",
        edge_color="#0284C7",
    )
    draw_straight_arrow(ax, 0.231, ym1, 0.256, ym1, color="#0284C7")

    # Card 1.2: x=0.258, w=0.210 -> right=0.468
    draw_card(
        ax,
        0.258,
        y1,
        0.210,
        h1,
        "2. R-Peak Lock & Isoelectric Ref",
        [
            "• 0.60s Moving-Median Detrend",
            "• Multi-Lead R-Peak Detection {r_b}",
            "• Local PR-Segment Isoelectric",
            "  Baseline Reference [r_b-90..40ms]",
        ],
        header_bg="#0369A1",
        body_bg="#FFFFFF",
        edge_color="#0284C7",
    )
    draw_straight_arrow(ax, 0.470, ym1, 0.495, ym1, color="#0284C7")

    # Card 1.3: x=0.497, w=0.235 -> right=0.732
    draw_card(
        ax,
        0.497,
        y1,
        0.235,
        h1,
        "3. Beat Anatomical Slots (f_b)",
        [
            "• 6-Slot ICU MSM: [RR, ST, PPG_PERF,",
            "  PPG_NOTCH, PAT_MS, RESP_DRIVE]",
            "• 8-Slot 12-Lead ECG: [RR, QRS, ST_INF,",
            "  ST_ANT, ST_LAT, V_AMP, PR, AXIS]",
        ],
        header_bg="#1D4ED8",
        body_bg="#FFFFFF",
        edge_color="#2563EB",
    )
    draw_straight_arrow(ax, 0.734, ym1, 0.759, ym1, color="#2563EB")

    # Card 1.4: x=0.761, w=0.215 -> right=0.976
    draw_card(
        ax,
        0.761,
        y1,
        0.215,
        h1,
        "4. Physician Assist Filters (PAF)",
        [
            "Quantizes each slot f_{b,d} into",
            "5 Clinical Severity Symbols:",
            "• A (Sev Low, w=1.0) • B (Mild, 0.5)",
            "• N (Normal, w=0.0)  • Y (Mild, 0.5)",
            "• Z (Sev High, w=1.0) -> Motif M_b",
        ],
        header_bg="#5B21B6",
        body_bg="#FFFFFF",
        edge_color="#7C3AED",
    )

    # =========================================================================
    # ROUTING CORRIDOR 1 -> 2 (y = 0.625 .. 0.675)
    # Drops at x=0.095 (left of Stage 2 header title starting at x=0.140!)
    # =========================================================================
    draw_orthogonal_arrow(
        ax,
        points=[
            (0.868, 0.690),  # Bottom center of Card 4 (PAF)
            (0.868, 0.650),  # Corridor between Stage 1 and Stage 2
            (0.095, 0.650),  # Travel left to x=0.095
            (0.095, 0.562),  # Enter top of Card 5 (left of Stage 2 title!)
        ],
        color="#6D28D9",
        lw=2.4,
        label="Beat Severity Motifs M_b ∈ {A,B,N,Y,Z}^D + Beat CMI s_b",
        label_pos=(0.50, 0.650),
        label_fs=9.6,
    )

    # =========================================================================
    # STAGE 2: y = 0.370 .. 0.625 (Height = 0.255)
    # =========================================================================
    s2_panel = patches.FancyBboxPatch(
        (0.012, 0.370),
        0.976,
        0.255,
        boxstyle="round,pad=0.0,rounding_size=0.014",
        facecolor="#FFFBEB",
        edgecolor="#D97706",
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(s2_panel)
    # Place Stage 2 title starting at x=0.135 so vertical arrow at x=0.095 never touches it!
    ax.text(
        0.135,
        0.598,
        "STAGE 2: RASPRO Hierarchical Summarization (SSM/MSM -> CM/CAM) & CMI Submodular Coreset Selection",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#78350F",
        zorder=2,
    )

    y2 = 0.385
    h2 = 0.177
    ym2 = y2 + h2 / 2.0

    # Card 2.1: x=0.024, w=0.220 -> right=0.244
    draw_card(
        ax,
        0.024,
        y2,
        0.220,
        h2,
        "5. Multi-Beat Matrix (SSM / MSM)",
        [
            "4-Beat Window W_k (D=6 or 8 slots):",
            "  Beat 1: [N, N, N, N, N, N, N, N]",
            "  Beat 2: [N, N, N, N, N, N, N, N]",
            "  Beat 3: [A, Z, Z, N, N, Y, N, A] (PVC)",
            "  Beat 4: [N, N, N, N, N, N, N, N]",
        ],
        header_bg="#B45309",
        body_bg="#FFFFFF",
        edge_color="#D97706",
    )
    draw_straight_arrow(ax, 0.246, ym2, 0.271, ym2, color="#D97706")

    # Card 2.2: x=0.273, w=0.235 -> right=0.508
    draw_card(
        ax,
        0.273,
        y2,
        0.235,
        h2,
        "6. Consensus Motifs (CM & CAM)",
        [
            "• Consensus Motif (CM_k):",
            "  CM:NNNNNNNNN (Homeostatic Mode)",
            "• Consensus Abnormality Motif (CAM_k):",
            "  CAM:AZZNNYNAN|IRR (Preserves PVC!)",
            "• Criticality Measure Index: CMI_k",
        ],
        header_bg="#C2410C",
        body_bg="#FFFFFF",
        edge_color="#EA580C",
    )
    draw_straight_arrow(ax, 0.510, ym2, 0.535, ym2, color="#EA580C")

    # Card 2.3: x=0.537, w=0.205 -> right=0.742
    draw_card(
        ax,
        0.537,
        y2,
        0.205,
        h2,
        "7. Active Run-Length Merge",
        [
            "• Merges stationary normal runs:",
            "  CM:NNNNNNNN|REGx4",
            "• Preserves transient CAMs intact!",
            "• Compresses L: 25 -> 3.74 (PTB-XL)",
            "  and 2,400 -> 51–129 (8m ICU)",
        ],
        header_bg="#B91C1C",
        body_bg="#FFFFFF",
        edge_color="#DC2626",
    )
    draw_straight_arrow(ax, 0.744, ym2, 0.769, ym2, color="#DC2626")

    # Card 2.4: x=0.771, w=0.205 -> right=0.976
    draw_card(
        ax,
        0.771,
        y2,
        0.205,
        h2,
        "8. CMI Submodular Coreset",
        [
            "• Prunes 90%–95% of redundant",
            "  unlabeled recordings pre-SSL",
            "• Maximizes Δ(S_i | C): CMI +",
            "  Novel CAM & Slot-Pair Cover",
            "• 10×–20× Data Efficiency!",
        ],
        header_bg="#9D174D",
        body_bg="#FFFFFF",
        edge_color="#DB2777",
    )

    # =========================================================================
    # ROUTING CORRIDOR 2 -> 3 (y = 0.315 .. 0.370)
    # Drops at x=0.095 (left of Stage 3 header title starting at x=0.135!)
    # =========================================================================
    draw_orthogonal_arrow(
        ax,
        points=[
            (0.640, 0.385),  # Bottom center of Card 7 (Active Run-Length Merge)
            (0.640, 0.342),  # Corridor between Stage 2 and Stage 3
            (0.095, 0.342),  # Travel left to x=0.095
            (0.095, 0.252),  # Enter top of Card 9 (left of Stage 3 title!)
        ],
        color="#15803D",
        lw=2.4,
        label="Active Motif Sequence {T_1, ..., T_L} + Salience Weights {CMI_1, ..., CMI_L}",
        label_pos=(0.40, 0.342),
        label_fs=9.6,
    )

    # =========================================================================
    # STAGE 3: y = 0.015 .. 0.315 (Height = 0.300)
    # =========================================================================
    s3_panel = patches.FancyBboxPatch(
        (0.012, 0.015),
        0.976,
        0.300,
        boxstyle="round,pad=0.0,rounding_size=0.014",
        facecolor="#F0FDF4",
        edgecolor="#16A34A",
        linewidth=1.5,
        zorder=1,
    )
    ax.add_patch(s3_panel)
    ax.text(
        0.135,
        0.288,
        "STAGE 3: Compositional 72-D Subspace, CMI-Biased Multi-Layer Transformer & Downstream Clinical Probes",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#14532D",
        zorder=2,
    )

    y3 = 0.032
    h3 = 0.220
    ym3 = y3 + h3 / 2.0

    # Card 3.1: x=0.024, w=0.220 -> right=0.244
    draw_card(
        ax,
        0.024,
        y3,
        0.220,
        h3,
        "9. 72-D Compositional Subspace",
        [
            "Decomposes PHM m ∈ {A,B,N,Y,Z}^9:",
            "• Dims 0–8: Signed Slot Severity φ(s_i)",
            "• Dims 9–17: Quadratic Magnitude ν(s_i)",
            "• Dims 18–53: 36 Cross-Lead Pairs",
            "  Ψ(s_i, s_j) = 0.6(φ_i+φ_j) + 0.8φ_iφ_j",
            "• Dims 54–71: IRR/REG + Run-Length",
        ],
        header_bg="#15803D",
        body_bg="#FFFFFF",
        edge_color="#16A34A",
        header_ratio=0.23,
        body_fs=9.3,
    )
    draw_straight_arrow(ax, 0.246, ym3, 0.271, ym3, color="#16A34A")

    # Card 3.2: x=0.273, w=0.235 -> right=0.508
    draw_card(
        ax,
        0.273,
        y3,
        0.235,
        h3,
        "10. CMI-Biased Transformer FM",
        [
            "• 4-Layer, 6-Head CMI-Attention:",
            "  Attn_ij ∝ q_i^T k_j + γ log(1+α·CMI_j)",
            "  Focuses on transient CAM bursts",
            "• Hybrid Symbolic-Continuous SSL:",
            "  9-Slot Masked Language Model (CE)",
            "  + Continuous Physiology Recon (MSE)",
        ],
        header_bg="#0F766E",
        body_bg="#FFFFFF",
        edge_color="#0D9488",
        header_ratio=0.23,
        body_fs=9.3,
    )
    draw_straight_arrow(ax, 0.510, ym3, 0.535, ym3, color="#0D9488")

    # Card 3.3: x=0.537, w=0.205 -> right=0.742
    draw_card(
        ax,
        0.537,
        y3,
        0.205,
        h3,
        "11. Frozen FM Embedding",
        [
            "• Recording Vector z_rec ∈ ℝ^{504}:",
            "  192-D CMI-Attention Pool",
            "  + 192-D Max Pool + 72-D Prior",
            "  + 48-D Clinical Beat Summary",
            "• MPS Pre-Train: 12.7s (2.4s @ 10%)",
            "• 50×–2,215× Less O(L²) FLOPs",
        ],
        header_bg="#3730A3",
        body_bg="#FFFFFF",
        edge_color="#4F46E5",
        header_ratio=0.23,
        body_fs=9.3,
    )
    draw_straight_arrow(ax, 0.744, ym3, 0.769, ym3, color="#4F46E5")

    # Card 3.4: x=0.771, w=0.205 -> right=0.976
    draw_card(
        ax,
        0.771,
        y3,
        0.205,
        h3,
        "12. Downstream Clinical Tasks",
        [
            "1. PTB-XL 12-Lead (Fold 10):",
            "   Super (79.8%), Sub (78.3%),",
            "   Form (72.7%), Rhythm (85.5%)",
            "2. MIMIC-IV-ECG Zero-Shot (78.1%)",
            "3. BIDMC ICU MSM (0.890 AUC)",
        ],
        header_bg="#14532D",
        body_bg="#FFFFFF",
        edge_color="#15803D",
        header_ratio=0.23,
        body_fs=9.3,
    )

    plt.tight_layout(pad=0.3)
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifact_dir = "/Users/rahulkrishnan/.gemini/antigravity/brain/c3e572a1-176f-4c92-b99a-8d46284c619a"
    repo_fig_dir = os.path.join(project_root, "results", "figures")
    ieee_fig_dir = os.path.join(project_root, "paper_ieee_jbhi", "figures")
    docs_fig_dir = os.path.join(project_root, "docs", "assets")

    generate_tokenizer_comparison_diagram(
        [
            os.path.join(artifact_dir, "raspro_fm_tokenizer_comparison.png"),
            os.path.join(repo_fig_dir, "raspro_fm_tokenizer_comparison.png"),
            os.path.join(ieee_fig_dir, "raspro_fm_tokenizer_comparison.png"),
            os.path.join(ieee_fig_dir, "raspro_fm_tokenizer_comparison.pdf"),
            os.path.join(docs_fig_dir, "raspro_fm_tokenizer_comparison.png"),
        ]
    )
    generate_end_to_end_diagram(
        [
            os.path.join(artifact_dir, "raspro_fm_architecture_diagram.png"),
            os.path.join(repo_fig_dir, "raspro_fm_architecture_diagram.png"),
            os.path.join(ieee_fig_dir, "raspro_fm_architecture_diagram.png"),
            os.path.join(ieee_fig_dir, "raspro_fm_architecture_diagram.pdf"),
            os.path.join(docs_fig_dir, "raspro_fm_architecture_diagram.png"),
        ]
    )
    print("Generated high-readability PNG and vector PDF architecture diagrams successfully.")
