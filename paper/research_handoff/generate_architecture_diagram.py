"""Generate revised publication-quality vector system architecture diagram for TrustStroke.

Addresses all visual clarity, layout, and scientific separation requirements:
1. Eliminates crowded text, overlapping labels, and arrow crossings.
2. Crystal-clear linear runtime pipeline in Tier 2.
3. Explicit separation of Offline Trust Diagnostics (DSI, CAS, RAI) from runtime routing.
4. Clean distinction between Dual Evaluation Regimes (Temporal Survey Transport vs. Locked Test Adaptive Benchmark).
5. Removal of cluttered numerical results from the diagram, retaining only component names,
   governing equations, and primary decision thresholds.
6. Clean IEEE-style typography and restrained, high-contrast palette.

Outputs:
- paper/research_handoff/truststroke_architecture.pdf
- paper/research_handoff/truststroke_architecture.svg
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Publication typography & vector settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def create_revised_architecture():
    # Canvas size: 14.2 x 9.2 inches for excellent aspect ratio and legibility
    fig = plt.figure(figsize=(14.2, 9.2), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 142)
    ax.set_ylim(0, 92)
    ax.axis('off')

    # Palette: Professional, restrained IEEE scientific palette
    c_title_main = "#0F172A"       # Deep Slate
    c_subtitle = "#475569"         # Slate Mid
    c_text_dark = "#1E293B"        # Primary Text
    c_text_muted = "#475569"       # Muted Secondary Text
    
    # Tier Theming
    # Tier 1 (Upstream Data & Modeling)
    t1_border = "#2563EB"          # Royal Blue
    t1_box_fill = "#F8FAFC"        # Pure Slate Light
    t1_box_border = "#94A3B8"
    
    # Tier 2 (Online Runtime Policy)
    t2_border = "#059669"          # Emerald Green
    t2_box_fill = "#F0FDF4"        # Soft Mint
    t2_box_border = "#10B981"
    
    # Tier 3 (Offline Trust Diagnostics)
    t3_border = "#D97706"          # Warm Amber
    t3_box_fill = "#FFFBEB"        # Soft Amber
    t3_box_border = "#F59E0B"
    
    # Tier 4 (Dual Evaluation Regimes)
    t4_border = "#4F46E5"          # Indigo
    t4_box_fill = "#EEF2FF"        # Soft Indigo
    t4_box_border = "#6366F1"

    # Helper function: draw rounded box
    def draw_box(x, y, w, h, fill, border, title="", subtitle="",
                 radius=1.2, lw=1.2, ls='-', title_color=c_text_dark,
                 title_size=8.8, subtitle_size=7.2, align='center', zorder=2):
        rect = patches.FancyBboxPatch((x, y), w, h,
                                      boxstyle=f"round,pad=0,rounding_size={radius}",
                                      facecolor=fill, edgecolor=border,
                                      linewidth=lw, linestyle=ls, zorder=zorder)
        ax.add_patch(rect)
        if title:
            tx = x + (w / 2 if align == 'center' else 1.6)
            ty = y + h - (1.6 if subtitle else h / 2)
            va = 'top' if subtitle else 'center'
            ax.text(tx, ty, title, color=title_color, fontsize=title_size,
                    fontweight='bold', ha=align, va=va, zorder=zorder + 1)
        if subtitle:
            sx = x + (w / 2 if align == 'center' else 1.6)
            sy = y + (h - 2.6) / 2
            ax.text(sx, sy, subtitle, color=c_text_muted, fontsize=subtitle_size,
                    ha=align, va='center', zorder=zorder + 1, linespacing=1.25)
        return rect

    # Helper function: draw arrow with clean text pill
    def draw_arrow(x1, y1, x2, y2, color="#334155", lw=1.3, ls='-',
                   label="", label_pos=0.5, label_offset=(0, 0), label_size=7.2,
                   connectionstyle="arc3,rad=0", zorder=4):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45",
                                    color=color, lw=lw, linestyle=ls,
                                    connectionstyle=connectionstyle),
                    zorder=zorder)
        if label:
            lx = x1 + (x2 - x1) * label_pos + label_offset[0]
            ly = y1 + (y2 - y1) * label_pos + label_offset[1]
            ax.text(lx, ly, label, color=color, fontsize=label_size,
                    fontweight='semibold', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.92),
                    zorder=zorder + 1)

    # =========================================================================
    # HEADER / TITLE BLOCK
    # =========================================================================
    ax.text(4, 88.5, "TrustStroke: Trust-Aware Classification & Adaptive Decision Framework",
            color=c_title_main, fontsize=12.5, fontweight='bold', ha='left', va='center')
    ax.text(138, 88.5, "Target: Lifetime Stroke History (CVDSTRK3) | CDC BRFSS Multi-Year Survey",
            color=c_subtitle, fontsize=8.2, fontstyle='italic', ha='right', va='center')

    # =========================================================================
    # SECTION 1: UPSTREAM DATA, SIMULATED AVAILABILITY & MODELING (TOP, y: 64 to 85)
    # =========================================================================
    draw_box(4, 64, 134, 21.5, "#FFFFFF", t1_border,
             "SECTION 1: POPULATION DATA INGESTION, AVAILABILITY MODELING & SCENARIO RECALIBRATION",
             "", radius=1.6, lw=1.6, title_color=t1_border, title_size=9.2, align='left')

    # Box 1A: Population Data & Survey Design
    draw_box(6.5, 66, 38, 16.5, t1_box_fill, t1_box_border,
             "1. CDC BRFSS Ingestion & Design",
             "• Sources: 2024 Source & 2025 Temporal Releases\n• Target: CVDSTRK3 (Self-reported stroke history)\n• 15 Canonical Predictors (4 Categorical, 11 Numeric)\n• Complex Survey Design: Weights, Strata & Clusters",
             radius=1.0, title_size=8.2, subtitle_size=7.2, align='left')

    # Box 1B: Predefined Availability Scenarios
    draw_box(48, 66, 42, 16.5, t1_box_fill, t1_box_border,
             "2. Simulated Availability Scenarios",
             "• Full (15 variables) & Demographics Only (2 variables)\n• Demographics + Disease History (7 variables)\n• Without Health State (12 variables; drops general/physical)\n• Controlled missingness encoding (zero-masked & indicators)",
             radius=1.0, title_size=8.2, subtitle_size=7.2, align='left')

    # Box 1C: Core Estimators & Platt Recalibration
    draw_box(93.5, 66, 42.5, 16.5, t1_box_fill, t1_box_border,
             "3. Estimators & Scenario Recalibration",
             "• Model Families: Boosted Trees, MLP, Logistic Regression\n• 4 Training Controls: Global, Scenario, Mixed-Mask, Subset\n• Platt Sigmoid Recalibration: P_cal = σ(A_s·logit(p) + B_s)\n• Resolves input representation mismatch without refitting",
             radius=1.0, title_size=8.2, subtitle_size=7.2, align='left')

    # Connections in Section 1
    draw_arrow(44.5, 74.25, 48, 74.25, color=t1_border, lw=1.3)
    draw_arrow(90, 74.25, 93.5, 74.25, color=t1_border, lw=1.3)

    # =========================================================================
    # SECTION 2: RUNTIME ADAPTIVE DECISION POLICY (MIDDLE, y: 30 to 60)
    # =========================================================================
    draw_box(4, 30.5, 134, 30, "#FFFFFF", t2_border,
             "SECTION 2: RUNTIME ADAPTIVE DECISION POLICY (Sequential Online Inference Loop)",
             "", radius=1.6, lw=1.6, title_color=t2_border, title_size=9.2, align='left')

    # Downward connectors into Runtime Policy
    draw_arrow(25.5, 64, 15, 52.5, color=c_subtitle, lw=1.2, label="Patient Record x_obs", label_offset=(0, 1.2))
    draw_arrow(114.5, 64, 40, 52.5, color=t1_border, lw=1.2, label="Scenario Calibrator Routing", label_offset=(0, 1.2))

    # Runtime Step 1: Observed Input
    draw_box(6.5, 34.5, 17, 18, t2_box_fill, t2_box_border,
             "Observed Input",
             "Initial subset x_obs\nUnobserved variables\nconcealed as NaN\n(No feature leakage)",
             radius=1.0, title_size=8.0, subtitle_size=7.0, align='center')

    # Runtime Step 2: CAS Uncertainty Gate
    draw_box(29, 34.5, 23, 18, t2_box_fill, t2_box_border,
             "CAS Uncertainty Gate",
             "Initial calibrated p & entropy\nCheck Acquisition Trigger:\n  H_total(p) ≥ τ_acq (0.30 bits)\n  or Disagreement ≥ τ_disagree",
             radius=1.0, title_size=8.0, subtitle_size=7.0, align='center')

    # Runtime Step 3: Feature Recommender Engine
    draw_box(57.5, 34.5, 24, 18, t2_box_fill, t2_box_border,
             "Feature Recommender",
             "Rank missing candidates:\n• Static Priority: CVDINFR4\n• ConditionalDonors: max ΔH\n  (k=64 nearest complete donors)\nStop if ΔH < γ_min",
             radius=1.0, title_size=8.0, subtitle_size=7.0, align='center')

    # Runtime Step 4: On-Demand Query Callback
    draw_box(87, 34.5, 21.5, 18, t2_box_fill, t2_box_border,
             "On-Demand Query",
             "Strict Callback Provider:\n  v* = Query(c*)\nUpdate x_obs ← x_obs ∪ {c*}\nSteps k ← k + 1 (k ≤ K_max)",
             radius=1.0, title_size=8.0, subtitle_size=7.0, align='center')

    # Runtime Step 5: Safety Abstention Gate
    draw_box(113.5, 34.5, 22.5, 18, t2_box_fill, t2_box_border,
             "Safety Abstention Gate",
             "Post-Acquisition Re-scoring\nCheck Abstention Criteria:\n  H_total(p) ≥ τ_abs (0.85 bits)\n  or Margin |p - θ| ≤ δ_margin",
             radius=1.0, title_size=8.0, subtitle_size=7.0, align='center')

    # Step 1 -> Step 2
    draw_arrow(23.5, 43.5, 29, 43.5, color=t2_border, lw=1.4)

    # Step 2 -> Step 3 (High Uncertainty)
    draw_arrow(52, 43.5, 57.5, 43.5, color=t2_border, lw=1.4,
               label="High Uncertainty", label_offset=(0, 1.3))

    # Step 3 -> Step 4 (Query feature c*)
    draw_arrow(81.5, 43.5, 87, 43.5, color=t2_border, lw=1.4,
               label="Query c*", label_offset=(0, 1.3))

    # Loopback from Step 4 back to Step 2 (Update & Recalibrate)
    draw_arrow(97.75, 34.5, 40.5, 34.5, color="#047857", lw=1.3, ls="--",
               connectionstyle="arc3,rad=-0.22", label="Update x_obs & Recalibrate (k < K_max)",
               label_pos=0.5, label_offset=(0, -2.2))

    # Step 4 -> Step 5 (Acquisition Complete or k = K_max)
    draw_arrow(108.5, 43.5, 113.5, 43.5, color=t2_border, lw=1.4)

    # Low Uncertainty Direct Bypass (Step 2 directly to Step 5)
    draw_arrow(40.5, 52.5, 124.75, 52.5, color=c_subtitle, lw=1.2, ls=":",
               connectionstyle="arc3,rad=0.18", label="Uncertainty Low (Skip Acquisition directly to Safety Gate)",
               label_pos=0.5, label_offset=(0, 2.2))

    # Output Actions (Emerging cleanly from right edge of Tier 2)
    draw_box(115, 22.5, 21, 6.5, "#ECFDF5", "#059669",
             "ACTION: CLASSIFY", "y_hat = I(p ≥ 0.14) | Emits Record",
             radius=0.8, title_size=7.5, subtitle_size=6.2, align='center')

    draw_box(91.5, 22.5, 21, 6.5, "#FEF2F2", "#DC2626",
             "ACTION: ABSTAIN", "y_hat = None | Emits Record",
             radius=0.8, title_size=7.5, subtitle_size=6.2, align='center')

    draw_arrow(125.5, 34.5, 125.5, 29, color="#059669", lw=1.4,
               label="Safe (H < τ_abs)", label_offset=(4.2, 0))
    draw_arrow(116, 34.5, 102, 29, color="#DC2626", lw=1.4,
               label="Ambiguous / Boundary", label_offset=(-4.5, 0))

    # =========================================================================
    # SECTION 3: OFFLINE TRUST DIAGNOSTICS (BOTTOM LEFT, y: 2 to 21)
    # =========================================================================
    draw_box(4, 2, 65, 19, "#FFFFFF", t3_border,
             "SECTION 3: OFFLINE TRUST DIAGNOSTICS (Model Auditing; Excluded from Runtime Routing)",
             "", radius=1.6, lw=1.5, ls="--", title_color=t3_border, title_size=8.5, align='left')

    # DSI Box
    draw_box(6.5, 3.8, 19.5, 14.5, t3_box_fill, t3_box_border,
             "Decision Stability (DSI)",
             "• Permutation Shapley\n  attribution similarity\n• Diagnoses Monte Carlo\n  path noise (16 to 256)\n• Evaluates retraining\n  stability (seeds/bootstraps)",
             radius=0.8, title_size=7.5, subtitle_size=6.5, align='left')

    # CAS Box
    draw_box(27.5, 3.8, 19.5, 14.5, t3_box_fill, t3_box_border,
             "Confidence Attribution (CAS)",
             "• H_total = H_within + H_between\n• Decomposes ensemble entropy\n  into member entropy &\n  Jensen-gap disagreement\n• Diagnostic audit tool for\n  misclassification detection",
             radius=0.8, title_size=7.5, subtitle_size=6.5, align='left')

    # RAI Box
    draw_box(48.5, 3.8, 18.5, 14.5, t3_box_fill, t3_box_border,
             "Resource Adaptability (RAI)",
             "• Brier headroom retention r_s\n  above null-prevalence baseline\n• Evaluates multiple missingness\n  patterns (MCAR & informative)\n• Complements scenario-level\n  proper scoring curves",
             radius=0.8, title_size=7.5, subtitle_size=6.5, align='left')

    # Visual distinction marker
    ax.text(36.5, 2.5, "* DSI, CAS decomposition, and RAI are offline diagnostics; they do not control online runtime routing.",
            color=t3_border, fontsize=6.8, fontstyle='italic', ha='center', va='center')

    # =========================================================================
    # SECTION 4: DUAL INDEPENDENT EVALUATION REGIMES (BOTTOM RIGHT, y: 2 to 21)
    # =========================================================================
    draw_box(73, 2, 65, 19, "#FFFFFF", t4_border,
             "SECTION 4: DUAL INDEPENDENT EVALUATION REGIMES (Separately Evaluated Protocols)",
             "", radius=1.6, lw=1.6, title_color=t4_border, title_size=8.5, align='left')

    # Regime A Box
    draw_box(75.5, 3.8, 30, 14.5, t4_box_fill, t4_box_border,
             "Regime A: Forward Temporal Transport",
             "• Core Conference Paper Evaluation\n• 2024 Development → 2025 External Transport\n• Complex Survey Weighting & Multi-Stage Clusters\n• Stratified Taylor Linearization (Discloses Singletons)\n• Compares FG vs. SC vs. MT vs. SR Controls",
             radius=0.8, title_size=7.5, subtitle_size=6.5, align='left')

    # Regime B Box
    draw_box(107, 3.8, 29, 14.5, t4_box_fill, t4_box_border,
             "Regime B: Sequential Adaptive Benchmark",
             "• Exploratory Dynamic Acquisition & Safety Study\n• 2024 Locked Test Partition (I.I.D. Resampling)\n• Paired Respondent-Level Bootstrap Differences\n• Evaluates ConditionalDonors vs. Static Priority\n• Documents Trade-offs of Safety Abstention",
             radius=0.8, title_size=7.5, subtitle_size=6.5, align='left')

    ax.text(105.5, 2.5, "* Regime A (Survey-Weighted Temporal) and Regime B (I.I.D. Adaptive) are distinct protocols and not combined.",
            color=t4_border, fontsize=6.8, fontstyle='italic', ha='center', va='center')

    # Save crisp vector outputs
    fig.savefig("paper/research_handoff/truststroke_architecture.pdf", format='pdf', bbox_inches='tight')
    fig.savefig("paper/research_handoff/truststroke_architecture.svg", format='svg', bbox_inches='tight')
    plt.close(fig)
    print("Revised vector PDF and SVG generated successfully.")

if __name__ == '__main__':
    create_revised_architecture()
