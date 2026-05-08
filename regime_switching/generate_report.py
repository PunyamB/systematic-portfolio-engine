"""
B2 TVTP Markov-Switching Regime Detection — Final Report Generator
====================================================================
Single comprehensive PDF with all 4 deliverables:
  1. Static Model Report (TVTP fit, BIC selection, regime classification)
  2. Rule-Based Comparison (M5: agreement, confusion matrix, early warning)
  3. Walk-Forward Backtest Results (per-window soft vs hard, aggregate)
  4. Integration Assessment (C1/C2/C3 evaluation, decision recommendation)

Run from SPE root:
    python -m regime_switching.generate_report
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
    Table, TableStyle, KeepTogether
)

from regime_switching.config import DATA_DIR, OUTPUT_DIR

# ── Colors ──
NAVY = HexColor("#1B2A4A")
ACCENT = HexColor("#2E75B6")
RED = HexColor("#D32F2F")
GREEN = HexColor("#388E3C")
GOLD = HexColor("#F0A800")
GRAY = HexColor("#757575")
LIGHT_BG = HexColor("#E8F0FE")
WHITE = HexColor("#FFFFFF")
BLACK = HexColor("#000000")

REPORT_PATH = OUTPUT_DIR / "B2_TVTP_Regime_Detection_Report.pdf"

# ── Styles ──
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "title", parent=styles["Title"],
    fontSize=24, textColor=NAVY, alignment=TA_CENTER,
    spaceAfter=10, leading=28,
)

subtitle_style = ParagraphStyle(
    "subtitle", parent=styles["Normal"],
    fontSize=14, textColor=GRAY, alignment=TA_CENTER,
    spaceAfter=20, italic=True,
)

h1_style = ParagraphStyle(
    "h1", parent=styles["Heading1"],
    fontSize=18, textColor=NAVY,
    spaceBefore=20, spaceAfter=10, leading=22,
)

h2_style = ParagraphStyle(
    "h2", parent=styles["Heading2"],
    fontSize=14, textColor=ACCENT,
    spaceBefore=14, spaceAfter=6, leading=18,
)

body_style = ParagraphStyle(
    "body", parent=styles["Normal"],
    fontSize=10, textColor=BLACK, alignment=TA_LEFT,
    spaceAfter=8, leading=14,
)

caption_style = ParagraphStyle(
    "caption", parent=styles["Normal"],
    fontSize=8, textColor=GRAY, alignment=TA_CENTER,
    spaceAfter=12, italic=True,
)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def make_table(headers, rows, col_widths=None):
    """Build a styled table."""
    data = [headers] + rows
    if col_widths is None:
        col_widths = [None] * len(headers)
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ALIGN", (0, 1), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def add_image(story, name, width=6.5, caption=None):
    """Add an image from outputs/ to the story."""
    img_path = OUTPUT_DIR / f"{name}.png"
    if img_path.exists():
        img = Image(str(img_path), width=width * inch, height=None)
        # Preserve aspect ratio
        img._restrictSize(width * inch, 9 * inch)
        story.append(img)
        if caption:
            story.append(Paragraph(f"<i>Figure: {caption}</i>", caption_style))
        story.append(Spacer(1, 0.15 * inch))


def build_report():
    """Build complete B2 report PDF."""
    story = []

    # ─────────── COVER PAGE ───────────
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("B2: TVTP Markov-Switching", title_style))
    story.append(Paragraph("Regime Detection", title_style))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "Time-Varying Transition Probability Markov-Switching model for "
        "probabilistic regime detection in systematic equity strategies",
        subtitle_style,
    ))
    story.append(Spacer(1, 1.5 * inch))

    # Author / project info
    info_data = [
        ["Project:", "B2 — TVTP Markov-Switching Regime Detection"],
        ["Author:", "Punyam"],
        ["Date:", datetime.now().strftime("%B %Y")],
        ["Pipeline:", "7 modules (M1-M7) + 4 deliverables"],
        ["Sample:", "1997-01-02 to 2026-03-19 (7,347 trading days)"],
        ["Strategy:", "Meridian Systematic Equity (S&P 500, 15 signals)"],
    ]
    info_table = Table(info_data, colWidths=[1.5 * inch, 4.5 * inch])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(PageBreak())

    # ─────────── EXECUTIVE SUMMARY ───────────
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(Paragraph(
        "This project implements a Time-Varying Transition Probability Markov-Switching "
        "(TVTP-MS) model as a probabilistic alternative to the Meridian system\'s rule-based "
        "regime detector. The model uses VIX, the 10Y-2Y yield curve slope, and the BofA "
        "high-yield credit spread as macro covariates that drive logistic transition "
        "probabilities between three latent regimes (bull, bear, crisis).",
        body_style,
    ))

    story.append(Paragraph(
        "The custom Hamilton filter and EM algorithm with numerical M-step were implemented "
        "from scratch (no off-the-shelf library exists for TVTP-MS). The model was validated "
        "on synthetic data with parameter recovery tests, cross-validated against statsmodels "
        "for the fixed-transition case, then applied to the full 1997-2026 sample. BIC strongly "
        "selects the 3-state TVTP variant over fixed-transition alternatives (BIC = -47,545).",
        body_style,
    ))

    story.append(Paragraph(
        "An 18-window walk-forward backtest compared the soft TVTP regime weighting against "
        "the existing rule-based hard regime switching within Meridian\'s production-grade "
        "backtest engine (cvxpy optimizer, trailing stops, sector neutralization, full constraint "
        "set preserved). Aggregate performance was indistinguishable: soft 23.44% CAGR vs hard "
        "23.56% CAGR, soft 1.308 Sharpe vs hard 1.309 Sharpe. Soft regime won 7 of 18 windows "
        "on both CAGR and Sharpe, with consistent outperformance only in the most recent holdout "
        "(2022-2026).",
        body_style,
    ))

    story.append(Paragraph(
        "Conclusion: The TVTP model is statistically valid, the methodology is rigorous, but "
        "the empirical result is that the rule-based regime detector is well-calibrated for "
        "this strategy. The TVTP model does not produce sufficient walk-forward improvement "
        "to justify replacing the existing detector. Integration into live Meridian is declined "
        "per the project\'s C3 quantitative criterion. The model retains analytical value as "
        "a standalone tool for tail risk monitoring and regime probability visualization.",
        body_style,
    ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # DELIVERABLE 1: STATIC MODEL REPORT
    # ═══════════════════════════════════════════════════════════════════
    story.append(Paragraph("1. Static Model Report", h1_style))

    # 1.1 Methodology
    story.append(Paragraph("1.1 Mathematical Framework", h2_style))

    story.append(Paragraph(
        "The TVTP-MS model assumes daily SPY log returns are drawn from a regime-dependent "
        "Gaussian distribution: r_t | S_t = k ~ N(mu_k, sigma_k^2). The latent regime state S_t "
        "follows a Markov chain with transition probabilities that depend on observable macro "
        "covariates z_t = [VIX, yield_curve, credit_spread]:",
        body_style,
    ))

    story.append(Paragraph(
        "<b>P(S_t = j | S_{t-1} = i, z_t) = exp(a_ij + b_ij\' z_t) / Σ_k exp(a_ik + b_ik\' z_t)</b>",
        body_style,
    ))

    story.append(Paragraph(
        "where a_ij is an intercept and b_ij is a coefficient vector for the i-to-j transition. "
        "Identification: a_i0 = 0 and b_i0 = 0 (state 0 as reference). Parameters estimated by "
        "maximum likelihood via the EM algorithm with a numerical M-step for the logistic "
        "coefficients. The custom Hamilton filter operates with time-varying transition "
        "matrices computed at each timestep from the current covariate values. Multiple "
        "random restarts (8) avoid local optima.",
        body_style,
    ))

    # 1.2 Model selection
    story.append(Paragraph("1.2 Model Selection (BIC)", h2_style))

    try:
        comp = load_json(DATA_DIR / "model_comparison.json")
        models_info = comp.get("models", {})
        if models_info:
            ms_rows = []
            for name, m in models_info.items():
                ms_rows.append([
                    name,
                    f"{m.get('log_likelihood', m.get('LL', 0)):,.2f}",
                    f"{m.get('bic', m.get('BIC', 0)):,.2f}",
                    f"{m.get('n_params', '-')}"
                ])
            story.append(make_table(
                ["Model", "Log-Likelihood", "BIC", "# Params"],
                ms_rows,
                col_widths=[2 * inch, 1.5 * inch, 1.5 * inch, 1 * inch],
            ))
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(
                f"<b>Selected model: {comp.get('selected_model', 'MS-3-TVTP')}</b> "
                "(lowest BIC). The 3-state TVTP variant outperforms both fixed-transition "
                "alternatives, demonstrating that macro covariates contain meaningful information "
                "about regime transitions beyond what state-history alone provides.",
                body_style,
            ))
    except Exception as e:
        story.append(Paragraph(f"Model comparison data unavailable: {e}", body_style))

    add_image(story, "05_model_comparison_bic", caption="BIC across 4 model variants. Lower is better.")

    # 1.3 Best model parameters
    story.append(Paragraph("1.3 Selected Model Parameters", h2_style))

    try:
        result = load_json(DATA_DIR / "ms3_tvtp_results.json")
        labels = result.get("state_labels", ["bull", "bear", "crisis"])
        mu_ann = result.get("mu_annualized", [])
        sigma_ann = result.get("sigma_annualized", [])

        param_rows = []
        for i, lbl in enumerate(labels):
            param_rows.append([
                lbl,
                f"{mu_ann[i]:+.2%}" if i < len(mu_ann) else "-",
                f"{sigma_ann[i]:.2%}" if i < len(sigma_ann) else "-",
            ])
        story.append(make_table(
            ["Regime", "Annualized Mean", "Annualized Vol"],
            param_rows,
            col_widths=[2 * inch, 2 * inch, 2 * inch],
        ))
        story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph(
            f"<b>Log-likelihood:</b> {result.get('log_likelihood', 0):,.2f}<br/>"
            f"<b>EM iterations:</b> {result.get('n_iter', '-')}<br/>"
            f"<b>Converged:</b> {result.get('converged', False)}",
            body_style,
        ))
    except Exception as e:
        story.append(Paragraph(f"Parameters unavailable: {e}", body_style))

    add_image(story, "02_transition_matrix",
              caption="Average transition matrix at sample-mean covariates. Diagonal dominance indicates regime persistence.")

    # 1.4 Logistic response curves
    story.append(Paragraph("1.4 TVTP Logistic Response Curves", h2_style))
    story.append(Paragraph(
        "How does each macro covariate affect transition probabilities? These plots show "
        "the fitted logistic response: holding other covariates at zero (mean), the "
        "x-axis sweeps each covariate over its standardized range, and the y-axis shows "
        "P(transition to state j) from each origin state.",
        body_style,
    ))
    add_image(story, "03_tvtp_logistic_curves",
              caption="P(transition) as a function of standardized covariate values, holding others constant.")

    # 1.5 Regime probability time series
    story.append(Paragraph("1.5 Regime Probability Time Series", h2_style))
    story.append(Paragraph(
        "The fitted TVTP model produces daily filtered probabilities for each regime over "
        "the full 1997-2026 sample. Crisis episodes (GFC 2008, COVID 2020, 2022 rate hikes) "
        "are correctly identified by elevated P(crisis) and reduced P(bull).",
        body_style,
    ))
    add_image(story, "01_regime_probabilities",
              caption="SPY cumulative returns (top) and stacked TVTP regime probabilities (bottom).")

    add_image(story, "12_rolling_transition_probs",
              caption="Filtered regime probabilities over time. Shaded bands mark major crisis periods.")

    add_image(story, "04_regime_conditional_returns",
              caption="Distribution of daily returns by max-probability regime label.")

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # DELIVERABLE 2: RULE-BASED COMPARISON
    # ═══════════════════════════════════════════════════════════════════
    story.append(Paragraph("2. Rule-Based Comparison", h1_style))

    story.append(Paragraph("2.1 Overlap Analysis (2009-2026)", h2_style))

    try:
        cmp = load_json(DATA_DIR / "comparison_results.json")
        agree = cmp.get("agreement", {})
        early = cmp.get("early_warning", {})

        story.append(Paragraph(
            f"Comparison period: {cmp.get('overlap_period', {}).get('start', '-')} to "
            f"{cmp.get('overlap_period', {}).get('end', '-')} "
            f"({cmp.get('overlap_period', {}).get('n_days', 0)} days). "
            f"Cohen\'s kappa = {agree.get('cohen_kappa', 0):.3f} "
            f"({agree.get('kappa_interpretation', '-')}).",
            body_style,
        ))

        story.append(Paragraph(
            "The slight kappa value reflects that the two systems organize the same data "
            "differently. The rule-based system uses 4 regimes (bull/recovery/bear/crisis) "
            "while the TVTP model uses 3 (bull/bear/crisis with no \'recovery\' state — "
            "transitional periods are captured by mixed probabilities). Notably, the "
            "regime-conditional return distributions reveal that the rule-based \'crisis\' "
            "label includes days with positive expected returns, suggesting the rule-based "
            "thresholds may not reliably isolate true crisis periods.",
            body_style,
        ))
    except Exception as e:
        story.append(Paragraph(f"Comparison data unavailable: {e}", body_style))

    add_image(story, "06_regime_timeline_comparison",
              caption="TVTP soft probabilities (top) vs rule-based hard regimes (bottom).")

    add_image(story, "08_confusion_matrix",
              caption="Regime classification overlap. Rows: rule-based, columns: TVTP.")

    # 2.2 Early warning
    story.append(Paragraph("2.2 Early Warning Analysis", h2_style))

    try:
        events = early.get("events", {})
        ew_rows = []
        for event, info in events.items():
            if info.get("status") == "no data in window":
                ew_rows.append([event, "no data", "-", "-"])
            else:
                ew_rows.append([
                    event,
                    info.get("ms_first_signal", "-"),
                    info.get("rb_first_signal", "-"),
                    f"+{info.get('ms_lead_days', 0)} days" if info.get("ms_earlier") else "-",
                ])
        story.append(make_table(
            ["Event", "TVTP First Signal", "Rule-Based First Signal", "TVTP Lead"],
            ew_rows,
            col_widths=[1.5 * inch, 1.6 * inch, 1.7 * inch, 1.2 * inch],
        ))
        story.append(Spacer(1, 0.1 * inch))

        story.append(Paragraph(
            f"<b>Average lead time: {early.get('avg_lead_days', 0):.0f} days</b> "
            f"across {early.get('n_events_measured', 0)} measurable events. "
            "TVTP detects regime transitions earlier than the rule-based system, providing "
            "actionable lead time for risk-off decisions. This is the strongest argument "
            "for the TVTP approach as an analytical complement to the existing system.",
            body_style,
        ))
    except Exception as e:
        story.append(Paragraph(f"Early warning data unavailable: {e}", body_style))

    add_image(story, "07_early_warning",
              caption="Crisis transition zooms. Dashed lines mark rule-based regime switches.")

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # DELIVERABLE 3: WALK-FORWARD BACKTEST
    # ═══════════════════════════════════════════════════════════════════
    story.append(Paragraph("3. Walk-Forward Backtest Results", h1_style))

    story.append(Paragraph("3.1 Methodology", h2_style))

    story.append(Paragraph(
        "An 18-window expanding walk-forward backtest evaluated whether soft regime weighting "
        "improves Meridian strategy performance over hard regime switching. Each window\'s "
        "TVTP model was fit on data through train_end (frozen parameters), then applied "
        "causally to the test period using filtered probabilities. The full Meridian backtest "
        "engine (cvxpy MV optimizer with Ledoit-Wolf covariance, sector neutralization, "
        "tracking error cap, turnover penalty, and trailing stops) was used for both soft "
        "and hard variants. The only difference: how regime multipliers are applied to signal "
        "weights — hard switching to a single regime\'s multipliers vs probability-weighted "
        "blend across all regimes.",
        body_style,
    ))

    story.append(Paragraph("3.2 Aggregate Results (2005-2026 stitched NAVs)", h2_style))

    try:
        m = load_json(DATA_DIR / "integration_metrics.json")
        agg = m.get("aggregate", {})
        soft = agg.get("soft", {})
        hard = agg.get("hard", {})

        agg_rows = [
            ["CAGR", f"{soft.get('cagr', 0) * 100:.2f}%", f"{hard.get('cagr', 0) * 100:.2f}%",
             f"{(soft.get('cagr', 0) - hard.get('cagr', 0)) * 100:+.2f}%"],
            ["Sharpe", f"{soft.get('sharpe', 0):.3f}", f"{hard.get('sharpe', 0):.3f}",
             f"{soft.get('sharpe', 0) - hard.get('sharpe', 0):+.3f}"],
            ["Max Drawdown", f"{soft.get('max_dd', 0) * 100:.2f}%", f"{hard.get('max_dd', 0) * 100:.2f}%",
             f"{(soft.get('max_dd', 0) - hard.get('max_dd', 0)) * 100:+.2f}%"],
        ]
        story.append(make_table(
            ["Metric", "Soft (TVTP)", "Hard (Rule-Based)", "Difference"],
            agg_rows,
            col_widths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch],
        ))
        story.append(Spacer(1, 0.1 * inch))

        n_total = m.get("n_windows", 18)
        story.append(Paragraph(
            f"Soft regime won <b>{m.get('windows_won_cagr', 0)}/{n_total}</b> windows on CAGR "
            f"and <b>{m.get('windows_won_sharpe', 0)}/{n_total}</b> windows on Sharpe ratio. "
            "Aggregate differences are statistically negligible (within natural noise across "
            "walk-forward windows).",
            body_style,
        ))
    except Exception as e:
        story.append(Paragraph(f"Backtest results unavailable: {e}", body_style))

    add_image(story, "09_integration_nav",
              caption="Stitched NAV: Soft (TVTP) vs Hard (rule-based), 2005-2026.")

    add_image(story, "10_per_window_comparison",
              caption="Per-window CAGR (top) and Sharpe (bottom). Soft (blue) vs Hard (navy).")

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # DELIVERABLE 4: INTEGRATION ASSESSMENT
    # ═══════════════════════════════════════════════════════════════════
    story.append(Paragraph("4. Integration Assessment", h1_style))

    story.append(Paragraph("4.1 Decision Criteria", h2_style))

    story.append(Paragraph(
        "Three quantitative criteria were defined ex-ante to gate integration into live "
        "Meridian:",
        body_style,
    ))

    story.append(Paragraph(
        "<b>C1: Regime classification accuracy</b> — Cohen\'s kappa &gt; 0.40 (moderate "
        "agreement) vs rule-based detector on the 2009-2026 overlap.<br/>"
        "<b>C2: Early warning lead time</b> — TVTP detects regime transitions ≥ 3 trading "
        "days earlier on average across measurable crisis events.<br/>"
        "<b>C3: Walk-forward backtest improvement</b> — CAGR improvement > 0 AND Sharpe "
        "improvement > 0 across the 18-window walk-forward + holdout.",
        body_style,
    ))

    add_image(story, "11_model_scorecard",
              caption="Integration criteria scorecard.")

    story.append(Paragraph("4.2 Decision", h2_style))

    story.append(Paragraph(
        "<b>Result: 1 of 3 criteria pass. Integration into live Meridian is declined.</b>",
        body_style,
    ))

    story.append(Paragraph(
        "<b>C1 (FAIL):</b> Cohen\'s kappa = 0.094 reflects substantial disagreement between "
        "the two systems. However, the rule-based baseline is itself unreliable (regime-conditional "
        "returns reveal the rule-based \'crisis\' label has positive mean returns), so this "
        "criterion is less informative than originally intended.",
        body_style,
    ))

    story.append(Paragraph(
        "<b>C2 (PASS):</b> TVTP detects regime transitions an average of 88 days earlier than "
        "the rule-based system across COVID 2020 and 2022 rate hikes. This is the strongest "
        "evidence that the TVTP model captures meaningful regime information.",
        body_style,
    ))

    story.append(Paragraph(
        "<b>C3 (FAIL):</b> Aggregate CAGR difference of -0.12% and Sharpe difference of "
        "-0.001 are statistical noise. The rule-based detector is well-calibrated for this "
        "strategy at the portfolio level. Soft regime weighting provides no walk-forward "
        "improvement to justify deployment.",
        body_style,
    ))

    story.append(Paragraph("4.3 Project Value", h2_style))

    story.append(Paragraph(
        "Despite the negative integration decision, this project has substantive value:",
        body_style,
    ))

    story.append(Paragraph(
        "<b>1. Methodological rigor.</b> Custom Hamilton filter and EM algorithm were "
        "implemented from scratch (no existing library supports TVTP-MS). The implementation "
        "passed synthetic recovery tests and cross-validated against statsmodels for the "
        "fixed-transition case. The masters-level mathematical work is itself a portfolio artifact.",
        body_style,
    ))

    story.append(Paragraph(
        "<b>2. Honest negative finding.</b> The rigorous walk-forward result is a reproducible, "
        "defensible empirical finding: the rule-based regime detector is more robust than its "
        "simplicity suggests for this strategy. This represents good research practice — "
        "preventing deployment of unproven changes per the system\'s research-before-deployment "
        "principle.",
        body_style,
    ))

    story.append(Paragraph(
        "<b>3. Standalone analytical value.</b> The TVTP model retains value as an independent "
        "analytical tool: probability vectors for risk reporting, early warning signals "
        "(88-day average lead) for monitoring dashboards, and regime-conditional analysis "
        "for performance attribution. These uses do not require integration into the trading "
        "engine.",
        body_style,
    ))

    story.append(Paragraph(
        "<b>4. Foundation for future research.</b> The fitted TVTP probabilities are saved "
        "per walk-forward window and can be reused without refitting. This enables future "
        "experiments (different signal weighting schemes, alternative covariate sets, ensemble "
        "regime detectors) to leverage this work as a free input.",
        body_style,
    ))

    story.append(Paragraph("4.4 Recommendation", h2_style))

    story.append(Paragraph(
        "<b>Do not integrate</b> the TVTP model into Meridian\'s live signal weighting. "
        "Retain the existing rule-based detector. Use the TVTP probabilities as a standalone "
        "monitoring tool with the early warning system as a separate alert channel. Revisit "
        "this decision after EXP009 (rolling parameter recalibration with 7-12 year lookback) "
        "completes — the soft regime mechanism may add value when combined with adaptive "
        "parameters that the current static-calibration system cannot leverage.",
        body_style,
    ))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # APPENDIX
    # ═══════════════════════════════════════════════════════════════════
    story.append(Paragraph("Appendix: Per-Window Results Detail", h1_style))

    try:
        m = load_json(DATA_DIR / "integration_metrics.json")
        per_window = m.get("per_window", [])

        win_rows = []
        for p in per_window:
            wid = p["window_id"]
            soft = p["soft"]
            hard = p["hard"]
            win_rows.append([
                str(wid),
                f"{soft['cagr'] * 100:.2f}%",
                f"{soft['sharpe']:.3f}",
                f"{hard['cagr'] * 100:.2f}%",
                f"{hard['sharpe']:.3f}",
                "S" if soft["cagr"] > hard["cagr"] else "H",
            ])

        story.append(make_table(
            ["Window", "Soft CAGR", "Soft Sharpe", "Hard CAGR", "Hard Sharpe", "Winner"],
            win_rows,
            col_widths=[0.9 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 0.9 * inch],
        ))
    except Exception as e:
        story.append(Paragraph(f"Per-window data unavailable: {e}", body_style))

    # Build the PDF
    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title="B2: TVTP Markov-Switching Regime Detection",
        author="Punyam",
    )
    doc.build(story)
    print(f"Report saved: {REPORT_PATH}")
    return REPORT_PATH


if __name__ == "__main__":
    build_report()
