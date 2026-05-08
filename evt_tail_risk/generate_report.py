"""
EVT Tail Risk — Final Report Generator
========================================
Produces a single comprehensive PDF containing all 4 deliverables:
  1. Static Tail Report
  2. Model Comparison
  3. Dynamic Tail Monitor
  4. Integration Assessment

Run from SPE root:
    python -m evt_tail_risk.generate_report
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
    Table, TableStyle, KeepTogether
)

from evt_tail_risk import config

# ── Colors ──────────────────────────────────────────────────────
NAVY = HexColor("#1B2A4A")
ACCENT = HexColor("#2E75B6")
RED = HexColor("#D32F2F")
GREEN = HexColor("#388E3C")
GRAY = HexColor("#757575")
LIGHT_BG = HexColor("#E8F0FE")
WHITE = HexColor("#FFFFFF")
BLACK = HexColor("#000000")

# ── Paths ───────────────────────────────────────────────────────
DATA_DIR = config.DATA_DIR
OUTPUT_DIR = config.OUTPUT_DIR
REPORT_PATH = os.path.join(OUTPUT_DIR, "EVT_Tail_Risk_Report.pdf")


def load_json(filename):
    with open(os.path.join(DATA_DIR, filename)) as f:
        return json.load(f)


def get_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "CustomTitle", parent=base["Title"],
            fontSize=28, textColor=NAVY, spaceAfter=6,
            fontName="Helvetica-Bold"
        ),
        "subtitle": ParagraphStyle(
            "CustomSubtitle", parent=base["Normal"],
            fontSize=14, textColor=ACCENT, spaceAfter=20,
            fontName="Helvetica-Oblique", alignment=TA_CENTER
        ),
        "h1": ParagraphStyle(
            "CustomH1", parent=base["Heading1"],
            fontSize=20, textColor=NAVY, spaceBefore=24, spaceAfter=12,
            fontName="Helvetica-Bold"
        ),
        "h2": ParagraphStyle(
            "CustomH2", parent=base["Heading2"],
            fontSize=15, textColor=ACCENT, spaceBefore=16, spaceAfter=8,
            fontName="Helvetica-Bold"
        ),
        "h3": ParagraphStyle(
            "CustomH3", parent=base["Heading3"],
            fontSize=12, textColor=NAVY, spaceBefore=12, spaceAfter=6,
            fontName="Helvetica-Bold"
        ),
        "body": ParagraphStyle(
            "CustomBody", parent=base["Normal"],
            fontSize=10, leading=14, spaceAfter=8,
            fontName="Helvetica"
        ),
        "body_bold": ParagraphStyle(
            "CustomBodyBold", parent=base["Normal"],
            fontSize=10, leading=14, spaceAfter=8,
            fontName="Helvetica-Bold"
        ),
        "small": ParagraphStyle(
            "CustomSmall", parent=base["Normal"],
            fontSize=8, leading=10, textColor=GRAY,
            fontName="Helvetica"
        ),
        "center": ParagraphStyle(
            "CustomCenter", parent=base["Normal"],
            fontSize=10, alignment=TA_CENTER,
            fontName="Helvetica"
        ),
        "finding": ParagraphStyle(
            "CustomFinding", parent=base["Normal"],
            fontSize=10, leading=14, spaceAfter=8,
            leftIndent=20, fontName="Helvetica",
            backColor=LIGHT_BG,
        ),
    }
    return styles


def make_table(headers, rows, col_widths=None):
    """Create a styled table."""
    data = [headers] + rows
    t = Table(data, colWidths=col_widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    # Alternating row shading
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), LIGHT_BG))

    t.setStyle(TableStyle(style))
    return t


def add_plot(story, filename, width=6.5 * inch, height=3.5 * inch):
    """Add a plot image if it exists."""
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        img = Image(path, width=width, height=height)
        story.append(img)
        story.append(Spacer(1, 8))
    else:
        story.append(Paragraph(f"[Plot not found: {filename}]", get_styles()["small"]))


def build_title_page(story, styles):
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("EVT Tail Risk Decomposition", styles["title"]))
    story.append(Paragraph(
        "Institutional-Grade Tail Risk Analysis Using Extreme Value Theory",
        styles["subtitle"]
    ))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        "SPY (1993-2026) and Meridian Systematic Equity (2005-2026)",
        styles["center"]
    ))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y')}",
        styles["center"]
    ))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Project B1  |  SystematicPortfolioEngine/evt_tail_risk/", styles["small"]))
    story.append(PageBreak())


def build_section1_static_tail(story, styles):
    """Section 1: Static Tail Report"""
    story.append(Paragraph("1. Static Tail Report", styles["h1"]))
    story.append(Paragraph(
        "Complete EVT analysis on the full sample: GPD fit parameters, "
        "VaR/ES at multiple confidence levels, tail index interpretation, "
        "and diagnostic validation.",
        styles["body"]
    ))

    # Load data
    meta = load_json("loss_metadata.json")
    gpd = load_json("gpd_fit.json")
    risk = load_json("risk_measures.json")

    # 1.1 Data Summary
    story.append(Paragraph("1.1 Data Summary", styles["h2"]))

    rows = []
    for key, name in [("spy", "SPY"), ("meridian", "Meridian")]:
        m = meta[key]
        s = m["stats"]
        rows.append([
            name,
            f"{m['date_range'][0]} to {m['date_range'][1]}",
            f"{s['count']:,}",
            f"{s['std']:.4f}",
            f"{s['skewness']:.2f}",
            f"{s['kurtosis']:.2f}",
        ])

    story.append(make_table(
        ["Series", "Date Range", "Obs", "Daily Std", "Skewness", "Excess Kurt"],
        rows,
        col_widths=[1*inch, 2.2*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.9*inch]
    ))
    story.append(Spacer(1, 12))

    # Distribution plots
    story.append(Paragraph("1.2 Distributional Analysis", styles["h2"]))
    story.append(Paragraph(
        "Both series exhibit significant departure from normality. The histograms show "
        "higher peaks and fatter tails than the fitted Gaussian, and the QQ plots show "
        "systematic curvature in both tails.",
        styles["body"]
    ))

    add_plot(story, "loss_histogram_spy.png", height=2.8*inch)
    add_plot(story, "qq_normal_spy.png", width=4*inch, height=4*inch)
    story.append(PageBreak())

    add_plot(story, "loss_histogram_meridian.png", height=2.8*inch)
    add_plot(story, "qq_normal_meridian.png", width=4*inch, height=4*inch)
    story.append(PageBreak())

    # Tail probability plots
    story.append(Paragraph("1.3 Tail Behavior", styles["h2"]))
    story.append(Paragraph(
        "The log-scale tail probability plots reveal power-law decay in both series, "
        "departing sharply from the exponentially-decaying normal tail. This is the "
        "fundamental observation that justifies EVT modeling.",
        styles["body"]
    ))
    add_plot(story, "tail_probability_spy.png", width=5*inch, height=3.5*inch)
    add_plot(story, "tail_probability_meridian.png", width=5*inch, height=3.5*inch)
    story.append(PageBreak())

    # 1.3 GPD Fit Results
    story.append(Paragraph("1.4 GPD Fit Results", styles["h2"]))

    thresh = load_json("threshold_results.json")
    rows = []
    for key, name in [("spy", "SPY"), ("meridian", "Meridian")]:
        g = gpd[key]
        t = thresh[key]
        rows.append([
            name,
            f"{t['threshold']:.6f}",
            f"{t['quantile']:.1%}",
            str(g["n_exceedances"]),
            f"{g['xi']:.4f}",
            f"[{g['xi_ci_95'][0]:.4f}, {g['xi_ci_95'][1]:.4f}]",
            f"{g['sigma']:.6f}",
            f"{g['ks_pvalue']:.4f}",
        ])

    story.append(make_table(
        ["Series", "Threshold", "Quantile", "N Exc", "Xi", "Xi 95% CI", "Sigma", "KS p"],
        rows,
        col_widths=[0.7*inch, 0.8*inch, 0.7*inch, 0.6*inch, 0.6*inch, 1.4*inch, 0.8*inch, 0.6*inch]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "<b>Interpretation:</b> Both series are in the heavy-tail Frechet domain (xi > 0). "
        "SPY has a heavier tail (xi=0.30) than Meridian (xi=0.22), suggesting the strategy's "
        "trailing stops and circuit breakers compress extreme loss behavior. "
        "Both KS tests pass comfortably (p >> 0.05), confirming the GPD fit is appropriate.",
        styles["body"]
    ))

    # GPD diagnostics
    add_plot(story, "gpd_4panel_spy.png", width=6.5*inch, height=5*inch)
    story.append(PageBreak())
    add_plot(story, "gpd_4panel_meridian.png", width=6.5*inch, height=5*inch)
    story.append(PageBreak())

    # 1.4 Risk Measures
    story.append(Paragraph("1.5 EVT Risk Measures", styles["h2"]))

    for key, name in [("spy", "SPY"), ("meridian", "Meridian")]:
        story.append(Paragraph(f"{name}", styles["h3"]))
        rows = []
        for level_key, m in risk[key].items():
            p = m["confidence_level"]
            rows.append([
                f"{p:.1%}",
                f"{m['evt_var']:.6f}",
                f"[{m['evt_var_ci'][0]:.6f}, {m['evt_var_ci'][1]:.6f}]",
                f"{m['empirical_var']:.6f}",
                f"{m['var_ratio_evt_vs_empirical']:.2f}",
                f"{m['evt_es']:.6f}",
                f"{m['empirical_es']:.6f}",
            ])

        story.append(make_table(
            ["Level", "EVT VaR", "VaR 95% CI", "Emp VaR", "Ratio", "EVT ES", "Emp ES"],
            rows,
            col_widths=[0.6*inch, 0.8*inch, 1.6*inch, 0.8*inch, 0.6*inch, 0.8*inch, 0.8*inch]
        ))
        story.append(Spacer(1, 12))

    story.append(Paragraph(
        "EVT and empirical estimates are closely aligned across all levels (ratios near 1.0), "
        "indicating the tails are well-sampled given 20-30 years of data spanning multiple "
        "crisis episodes. The value of EVT over empirical quantiles emerges in shorter samples "
        "and during the rolling analysis in Section 3.",
        styles["body"]
    ))
    story.append(PageBreak())


def build_section2_model_comparison(story, styles):
    """Section 2: Model Comparison"""
    story.append(Paragraph("2. VaR Model Comparison", styles["h1"]))
    story.append(Paragraph(
        "Five VaR models backtested across 30 years (SPY) and 20 years (Meridian) "
        "with Kupiec unconditional coverage and Christoffersen conditional coverage tests. "
        "The backtest uses strictly out-of-sample one-day-ahead forecasts.",
        styles["body"]
    ))

    bt = load_json("backtest_results.json")

    for key, name in [("spy", "SPY"), ("meridian", "Meridian")]:
        story.append(Paragraph(f"2.{1 if key=='spy' else 2} {name} Backtest Results", styles["h2"]))

        results = bt[key]["backtest"]
        rows = []
        for level in results:
            for model, r in results[level].items():
                kp = r["kupiec"]["p_value"]
                cp = r["christoffersen"]["p_value"]
                kp_pass = "PASS" if r["kupiec"]["pass"] else "FAIL"
                cp_pass = "PASS" if r["christoffersen"]["pass"] else "FAIL"
                rows.append([
                    model, level,
                    f"{r['violation_rate']:.4f}",
                    f"{r['expected_rate']:.4f}",
                    f"{kp:.4f}" if not np.isnan(kp) else "N/A",
                    kp_pass,
                    f"{cp:.4f}" if not np.isnan(cp) else "N/A",
                    cp_pass,
                ])

        story.append(make_table(
            ["Model", "Level", "Viol Rate", "Expected", "Kupiec p", "KP", "Christ p", "CP"],
            rows,
            col_widths=[1.1*inch, 0.6*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.5*inch, 0.7*inch, 0.5*inch]
        ))
        story.append(Spacer(1, 12))

    # Scorecard plots
    story.append(Paragraph("2.3 Model Scorecards", styles["h2"]))
    add_plot(story, "model_scorecard_spy.png", height=3*inch)
    add_plot(story, "model_scorecard_meridian.png", height=3*inch)
    story.append(PageBreak())

    # Violation plots
    story.append(Paragraph("2.4 Violation Timelines (99% VaR)", styles["h2"]))
    add_plot(story, "backtest_violations_spy_0990.png", height=5.5*inch)
    story.append(PageBreak())
    add_plot(story, "backtest_violations_meridian_0990.png", height=5.5*inch)
    story.append(PageBreak())

    # Key findings
    story.append(Paragraph("2.5 Key Findings", styles["h2"]))
    findings = [
        "Gaussian VaR systematically underestimates tail risk: violation rates are 2-2.4x the "
        "expected rate at 99% for both series. This is the baseline that justifies all other models.",
        "All models fail the Christoffersen test universally, meaning VaR violations cluster. "
        "This reflects the volatility clustering confirmed by Ljung-Box tests in the EDA stage, "
        "and motivates time-varying approaches.",
        "Cornish-Fisher is surprisingly competitive: it is the only model to pass Kupiec at 99% "
        "for both series, because it directly adjusts for the heavy kurtosis and skewness measured "
        "in the loss distribution.",
        "Meridian is better calibrated than SPY across all models, consistent with the finding "
        "that the strategy's risk management compresses tail behavior (lower xi).",
        "EVT-GPD performs well but does not dominate. Its primary advantage is in providing "
        "the tail index (xi) as a diagnostic that no other model captures, and in extrapolating "
        "beyond the observed sample to return levels not yet experienced.",
    ]
    for f in findings:
        story.append(Paragraph(f"- {f}", styles["body"]))

    story.append(PageBreak())


def build_section3_dynamic_monitor(story, styles):
    """Section 3: Dynamic Tail Monitor"""
    story.append(Paragraph("3. Dynamic Tail Monitor", styles["h1"]))
    story.append(Paragraph(
        "Rolling 500-day GPD estimation tracking tail index evolution over time. "
        "The shape parameter xi is refitted every 5 trading days, producing a daily "
        "time series of tail heaviness that responds to market conditions.",
        styles["body"]
    ))

    bt = load_json("backtest_results.json")

    for key, name in [("spy", "SPY"), ("meridian", "Meridian")]:
        xi_range = bt[key]["rolling_xi_range"]
        story.append(Paragraph(f"3.{1 if key=='spy' else 2} {name}", styles["h2"]))
        story.append(Paragraph(
            f"Rolling xi range: [{xi_range[0]:.4f}, {xi_range[1]:.4f}]",
            styles["body_bold"]
        ))

    # Rolling tail index plots
    story.append(Paragraph("3.1 Rolling Tail Index", styles["h2"]))
    add_plot(story, "rolling_tail_index_spy.png", height=2.8*inch)
    add_plot(story, "rolling_tail_index_meridian.png", height=2.8*inch)
    story.append(PageBreak())

    # Rolling volatility
    story.append(Paragraph("3.2 Rolling Volatility", styles["h2"]))
    add_plot(story, "rolling_vol_spy.png", height=2.8*inch)
    add_plot(story, "rolling_vol_meridian.png", height=2.8*inch)
    story.append(PageBreak())

    # Drawdowns
    story.append(Paragraph("3.3 Drawdown Analysis", styles["h2"]))
    add_plot(story, "drawdown_spy.png", height=2.8*inch)
    add_plot(story, "drawdown_meridian.png", height=2.8*inch)
    story.append(PageBreak())

    # ACF
    story.append(Paragraph("3.4 Volatility Clustering", styles["h2"]))
    story.append(Paragraph(
        "Autocorrelation of absolute returns and squared returns confirms strong "
        "volatility persistence in both series, with slow ACF decay characteristic "
        "of long-memory processes. This is the fundamental driver of VaR violation "
        "clustering observed in Section 2.",
        styles["body"]
    ))
    add_plot(story, "acf_volatility_spy.png", height=2.8*inch)
    add_plot(story, "acf_volatility_meridian.png", height=2.8*inch)
    story.append(PageBreak())


def build_section4_integration(story, styles):
    """Section 4: Integration Assessment"""
    story.append(Paragraph("4. Meridian Integration Assessment", styles["h1"]))
    story.append(Paragraph(
        "Evaluation of three quantitative criteria determining whether EVT tail risk "
        "monitoring should be integrated into Meridian's circuit breaker system.",
        styles["body"]
    ))

    # Load backtest results
    bt = load_json("backtest_results.json")

    # C1: Kupiec at 99%
    story.append(Paragraph("4.1 Criterion C1: EVT VaR Calibration", styles["h2"]))
    story.append(Paragraph(
        "<b>Threshold:</b> Kupiec p-value > 0.05 at 99% level on rolling out-of-sample.",
        styles["body"]
    ))

    spy_bt = bt["spy"]["backtest"]
    mer_bt = bt["meridian"]["backtest"]

    # Find EVT results at 99%
    spy_evt_99 = spy_bt.get("99.0%", {}).get("EVT-GPD", {})
    mer_evt_99 = mer_bt.get("99.0%", {}).get("EVT-GPD", {})

    spy_kp = spy_evt_99.get("kupiec", {}).get("p_value", float("nan"))
    mer_kp = mer_evt_99.get("kupiec", {}).get("p_value", float("nan"))
    spy_pass = spy_evt_99.get("kupiec", {}).get("pass", False)
    mer_pass = mer_evt_99.get("kupiec", {}).get("pass", False)

    rows = [
        ["SPY", f"{spy_kp:.4f}" if not np.isnan(spy_kp) else "N/A",
         "PASS" if spy_pass else "FAIL"],
        ["Meridian", f"{mer_kp:.4f}" if not np.isnan(mer_kp) else "N/A",
         "PASS" if mer_pass else "FAIL"],
    ]
    story.append(make_table(
        ["Series", "Kupiec p (99%)", "Result"],
        rows,
        col_widths=[1.5*inch, 1.5*inch, 1*inch]
    ))
    story.append(Spacer(1, 8))

    c1_pass = spy_pass and mer_pass
    story.append(Paragraph(
        f"<b>C1 Result: {'PASS' if c1_pass else 'FAIL'}</b> - "
        f"{'Both series pass Kupiec at 99%.' if c1_pass else 'EVT VaR does not achieve well-calibrated coverage at 99% for both series. Violation rates exceed the expected 1%, indicating the rolling GPD with a fixed 95th percentile threshold does not fully adapt to changing tail dynamics.'}",
        styles["body"]
    ))
    story.append(Spacer(1, 8))

    # C2: Early warning
    story.append(Paragraph("4.2 Criterion C2: Early Warning Lead Time", styles["h2"]))
    story.append(Paragraph(
        "<b>Threshold:</b> Rolling tail index increase detectable >= 5 trading days before "
        "CB T2 trigger (10% drawdown).",
        styles["body"]
    ))
    story.append(Paragraph(
        "Assessment: The rolling tail index (xi) shows clear spikes during crisis onset periods "
        "(visible in the rolling tail index plots). However, a formal lead-time analysis requires "
        "identifying specific CB T2 trigger dates in the backtest and measuring whether xi "
        "elevated beforehand. With only backtest data (no live CB triggers), this criterion "
        "is evaluated qualitatively: the tail index does respond to market stress, but the "
        "response is concurrent with, not ahead of, volatility spikes.",
        styles["body"]
    ))
    story.append(Paragraph("<b>C2 Result: INCONCLUSIVE</b> - Requires live CB trigger data for formal evaluation.", styles["body"]))
    story.append(Spacer(1, 8))

    # C3: False positive rate
    story.append(Paragraph("4.3 Criterion C3: False Positive Rate", styles["h2"]))
    story.append(Paragraph(
        "<b>Threshold:</b> < 10% of tail index alerts followed by no CB trigger within 20 days.",
        styles["body"]
    ))
    story.append(Paragraph(
        "Assessment: Without a defined alert threshold for xi (what level of xi constitutes "
        "an 'alert'?) and without CB trigger history in the backtest, this criterion cannot "
        "be formally evaluated. The rolling xi exhibits substantial variation (SPY range: "
        f"[{bt['spy']['rolling_xi_range'][0]:.2f}, {bt['spy']['rolling_xi_range'][1]:.2f}]), "
        "which suggests that a fixed alert threshold would generate many false positives "
        "during normal market fluctuations.",
        styles["body"]
    ))
    story.append(Paragraph("<b>C3 Result: INCONCLUSIVE</b> - Requires alert threshold calibration and CB trigger history.", styles["body"]))
    story.append(Spacer(1, 12))

    # Overall recommendation
    story.append(Paragraph("4.4 Integration Recommendation", styles["h2"]))

    story.append(Paragraph(
        "<b>Recommendation: EVT remains a standalone analytical tool. "
        "Integration into Meridian's circuit breaker is not supported at this time.</b>",
        styles["body_bold"]
    ))
    story.append(Spacer(1, 8))

    rationale = [
        "C1 (calibration) fails for SPY and is borderline for Meridian. The rolling EVT model "
        "with a fixed quantile threshold does not produce well-calibrated VaR at 99%.",
        "C2 and C3 cannot be evaluated without live circuit breaker trigger history. "
        "These criteria are designed for live monitoring, not backtest evaluation.",
        "The project's standalone value remains fully intact: the model comparison, tail "
        "decomposition, and rolling tail index provide insights that no other component "
        "of the Meridian system captures.",
        "Future path: if the system accumulates sufficient live CB trigger events, "
        "revisit C2 and C3 with concrete data. A time-varying threshold (rather than "
        "fixed 95th percentile) may improve C1 calibration.",
    ]
    for r in rationale:
        story.append(Paragraph(f"- {r}", styles["body"]))

    story.append(PageBreak())

    # Threshold diagnostic plots
    story.append(Paragraph("4.5 Threshold Selection Diagnostics", styles["h2"]))
    add_plot(story, "mrl_spy.png", height=3*inch)
    add_plot(story, "param_stability_spy.png", height=4*inch)
    story.append(PageBreak())
    add_plot(story, "mrl_meridian.png", height=3*inch)
    add_plot(story, "param_stability_meridian.png", height=4*inch)


def build_report():
    """Build the complete PDF report."""
    print("Building EVT Tail Risk Report...")

    styles = get_styles()

    doc = SimpleDocTemplate(
        REPORT_PATH,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    story = []

    # Title page
    build_title_page(story, styles)

    # Section 1: Static Tail Report
    print("  Section 1: Static Tail Report...")
    build_section1_static_tail(story, styles)

    # Section 2: Model Comparison
    print("  Section 2: Model Comparison...")
    build_section2_model_comparison(story, styles)

    # Section 3: Dynamic Tail Monitor
    print("  Section 3: Dynamic Tail Monitor...")
    build_section3_dynamic_monitor(story, styles)

    # Section 4: Integration Assessment
    print("  Section 4: Integration Assessment...")
    build_section4_integration(story, styles)

    # Build
    doc.build(story)
    print(f"\nReport saved to: {REPORT_PATH}")
    print(f"File size: {os.path.getsize(REPORT_PATH) / 1024:.0f} KB")


if __name__ == "__main__":
    build_report()
