"""
Benefit–Risk Assessment (numpy/scipy-free)

Structured benefit–risk assessment following:
- FDA Benefit–Risk Framework (BRF), PDUFA VI 2018–2022
- Kaul S, Stockbridge N, Butler J. Circulation 2020;142:1974–1988
- IMI-PROTECT Methodological Guidance (EMA)

Computes: NNT-B, NNT-H, BRR, INB, δ-sensitivity, common denominator.
Generates: Markdown report with FDA BRF dimensions + integrated assessment.
All computations pure Python — no numpy/scipy required.
"""

import math
import os


# ═══════════════════════════════════════════════════════════════
# SECTION 1: DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

class Endpoint:
    """Represents a single clinical endpoint (benefit or risk)."""

    def __init__(self, name, domain, rank=99, severity="reversible_serious",
                 events_intervention=None, total_intervention=None,
                 events_control=None, total_control=None,
                 effect=None, ci_lower=None, ci_upper=None,
                 p_value=None,
                 exposure_years_intervention=None,
                 exposure_years_control=None):
        self.name = name
        self.domain = domain  # "benefit" or "risk"
        self.rank = rank
        self.severity = severity
        self.events_int = events_intervention
        self.total_int = total_intervention
        self.events_ctrl = events_control
        self.total_ctrl = total_control
        self.effect = effect
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.p_value = p_value
        self.exp_years_int = exposure_years_intervention
        self.exp_years_ctrl = exposure_years_control


# ═══════════════════════════════════════════════════════════════
# SECTION 2: QUANTITATIVE METRICS
# ═══════════════════════════════════════════════════════════════

def compute_risk_difference(ep):
    """
    Compute absolute risk difference and exposure-adjusted
    incidence rate difference per 1000 patient-years.
    """
    result = {}

    if ep.events_int is not None and ep.total_int is not None:
        risk_int = ep.events_int / ep.total_int
        risk_ctrl = ep.events_ctrl / ep.total_ctrl
        ard = risk_int - risk_ctrl
        se = math.sqrt(
            risk_int * (1 - risk_int) / ep.total_int +
            risk_ctrl * (1 - risk_ctrl) / ep.total_ctrl
        )
        result["risk_int"] = risk_int
        result["risk_ctrl"] = risk_ctrl
        result["ard"] = ard
        result["se"] = se
        result["ard_ci_lower"] = ard - 1.96 * se
        result["ard_ci_upper"] = ard + 1.96 * se
        result["significant"] = not (result["ard_ci_lower"] <= 0 <= result["ard_ci_upper"])

        if ep.exp_years_int and ep.exp_years_ctrl:
            ir_int = (ep.events_int / ep.exp_years_int) * 1000
            ir_ctrl = (ep.events_ctrl / ep.exp_years_ctrl) * 1000
            result["ir_int_per1000py"] = round(ir_int, 2)
            result["ir_ctrl_per1000py"] = round(ir_ctrl, 2)
            result["rd_per1000py"] = round(ir_int - ir_ctrl, 2)
        elif ep.total_int and ep.total_ctrl:
            result["rd_per1000"] = round(ard * 1000, 1)

    elif ep.effect is not None:
        result["effect"] = ep.effect
        result["ci_lower"] = ep.ci_lower
        result["ci_upper"] = ep.ci_upper
        if ep.ci_lower is not None and ep.ci_upper is not None:
            result["significant"] = not (ep.ci_lower <= 1.0 <= ep.ci_upper)
        else:
            result["significant"] = False

    if ep.p_value is not None:
        result["p_value"] = ep.p_value
        result["significant"] = ep.p_value < 0.05

    return result


def compute_nnt(ard, domain):
    """NNT-B (benefit) or NNT-H (harm). Only when significant (Rule A)."""
    if ard is None or ard == 0:
        return None
    nnt = abs(1.0 / ard)
    return round(nnt, 0)


def compute_brr(nnt_h, nnt_b, weight_harm=1.0):
    """
    Benefit–Risk Ratio = NNT-H / NNT-B.
    BRR > 1 → favorable, BRR < 1 → unfavorable.
    """
    if nnt_b is None or nnt_h is None or nnt_b == 0:
        return None
    return round((nnt_h * weight_harm) / nnt_b, 2)


def compute_inb(ard_benefit, ard_harm, delta=1.0):
    """
    Incremental Net Benefit = Benefit − (Risk / δ).
    δ = maximum tolerable harms per benefit. INB > 0 → favorable.
    """
    return round(ard_benefit - (ard_harm / delta), 5)


def compute_common_denominator(endpoints, n_patients=1000, years=3):
    """
    Common denominator: per N patients treated for Y years,
    how many benefit events prevented and harm events caused?
    """
    results = []
    for ep in endpoints:
        rd = compute_risk_difference(ep)
        if "ard" in rd:
            events = round(
                rd["ard"] * n_patients * (-1 if ep.domain == "benefit" else 1), 1
            )
            results.append({
                "name": ep.name,
                "domain": ep.domain,
                "events_per_n": events,
                "significant": rd.get("significant", False),
                "severity": ep.severity,
            })
        elif "rd_per1000py" in rd:
            events = round(
                rd["rd_per1000py"] * years * (-1 if ep.domain == "benefit" else 1), 1
            )
            results.append({
                "name": ep.name,
                "domain": ep.domain,
                "events_per_n": events,
                "significant": rd.get("significant", False),
                "severity": ep.severity,
            })
    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 3: VALUE TREE
# ═══════════════════════════════════════════════════════════════

SEVERITY_ORDER = {
    "irreversible_fatal": 0,
    "irreversible_nonfatal": 1,
    "reversible_serious": 2,
    "reversible_nonserious": 3,
}


def build_value_tree(endpoints):
    """Sort endpoints into value tree: benefits ranked, risks ranked."""
    benefits = sorted(
        [e for e in endpoints if e.domain == "benefit"],
        key=lambda e: (e.rank, SEVERITY_ORDER.get(e.severity, 99))
    )
    risks = sorted(
        [e for e in endpoints if e.domain == "risk"],
        key=lambda e: (e.rank, SEVERITY_ORDER.get(e.severity, 99))
    )
    return benefits, risks


# ═══════════════════════════════════════════════════════════════
# SECTION 4: SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════

def delta_sensitivity(ard_benefit, ard_harm, deltas=None):
    """δ-sensitivity analysis (Rule H). At minimum 7 thresholds."""
    if deltas is None:
        deltas = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    results = []
    for d in deltas:
        inb = compute_inb(ard_benefit, ard_harm, d)
        results.append({"delta": d, "inb": inb, "favorable": inb > 0})
    return results


def brr_weight_sensitivity(nnt_b, nnt_h, weights=None):
    """BRR across a range of harm weights. Returns (results, crossover_weight)."""
    if weights is None:
        weights = [round(i * 0.01, 2) for i in range(1, 101)]
    results = []
    crossover = None
    for w in weights:
        brr = compute_brr(nnt_h, nnt_b, weight_harm=w)
        if brr is not None:
            results.append({"weight": round(w, 2), "brr": brr})
            if crossover is None and brr >= 1.0:
                crossover = round(w, 2)
    return results, crossover


# ═══════════════════════════════════════════════════════════════
# SECTION 5: INTEGRATED ASSESSMENT
# ═══════════════════════════════════════════════════════════════

def _generate_integrated_assessment(study_info, benefits, risks, quant):
    """Algorithmic verdict following Kaul decision logic."""
    primary = benefits[0] if benefits else None
    primary_sig = primary.get("significant", False) if primary else False

    sig_risks = [r for r in risks if r.get("significant", False)]
    irreversible_risks = [r for r in sig_risks if r.get("reversibility") == "irreversible"]

    brr = quant.get("brr")
    inb_results = quant.get("delta_sensitivity", [])

    if primary_sig and primary.get("ard", 0) < 0:
        if not sig_risks:
            verdict = "FAVORABLE"
            reasoning = (
                "Statistically significant benefit on the primary endpoint "
                "with no significant excess risk identified."
            )
        elif irreversible_risks:
            if brr and brr > 1.0:
                verdict = "FAVORABLE_WITH_CAVEAT"
                reasoning = (
                    "Significant benefit on primary endpoint. Irreversible harms "
                    "present but BRR > 1 indicates benefit outweighs risk. "
                    "Label restrictions or monitoring may be warranted."
                )
            else:
                verdict = "INDETERMINATE"
                reasoning = (
                    "Significant benefit on primary endpoint but offset by "
                    "irreversible harms. BRR does not clearly favor intervention. "
                    "Decision depends on patient risk tolerance and clinical context."
                )
        else:
            verdict = "FAVORABLE"
            reasoning = (
                "Significant benefit on primary endpoint. Excess risks are "
                "reversible and manageable. Benefit–risk balance is favorable."
            )
    elif primary_sig and primary.get("ard", 0) > 0:
        verdict = "UNFAVORABLE"
        reasoning = (
            "The intervention is associated with significant harm on the "
            "primary endpoint. Benefit–risk balance is unfavorable."
        )
    else:
        if sig_risks:
            verdict = "UNFAVORABLE"
            reasoning = (
                "No significant benefit on primary endpoint, but significant "
                "risks identified. Benefit–risk balance is unfavorable."
            )
        else:
            verdict = "INDETERMINATE"
            reasoning = (
                "No significant difference on primary endpoint and no significant "
                "excess risk. Results are indeterminate."
            )

    delta_summary = ""
    if inb_results:
        fav_at = [r["delta"] for r in inb_results if r["favorable"]]
        unfav_at = [r["delta"] for r in inb_results if not r["favorable"]]
        if fav_at and unfav_at:
            delta_summary = (
                f"Decision robustness: Favorable at δ≥{min(fav_at)}; "
                f"Indeterminate at δ<{min(fav_at)}."
            )
        elif fav_at:
            delta_summary = "Decision robustness: Favorable across all δ thresholds."
        else:
            delta_summary = "Decision robustness: Unfavorable across all δ thresholds."

    return {
        "verdict": verdict,
        "reasoning": reasoning,
        "brr": brr,
        "delta_sensitivity_summary": delta_summary,
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 6: REPORT GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_br_report(study_info, endpoints):
    """Generate full Markdown benefit–risk assessment report."""
    lines = []
    all_results = {}

    # Header
    lines.append("# Benefit–Risk Assessment Report")
    lines.append("")
    lines.append(f"**Study:** {study_info.get('title', 'N/A')}")
    lines.append(f"**Authors:** {study_info.get('authors', 'N/A')}")
    lines.append(f"**Design:** {study_info.get('design', 'N/A')}")
    lines.append(f"**Comparator:** {study_info.get('comparator', 'N/A')}")
    lines.append(f"**Follow-up:** {study_info.get('follow_up_years', 'N/A')} years")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Dimension 1: Analysis of Condition
    lines.append("## 1. Analysis of Condition")
    lines.append(f"- **Disease:** {study_info.get('disease_description', 'N/A')}")
    lines.append(f"- **Severity:** {study_info.get('disease_severity', 'N/A')}")
    lines.append(f"- **Mortality burden:** {study_info.get('mortality_burden', 'N/A')}")
    lines.append("")

    # Dimension 2: Current Treatment Options
    lines.append("## 2. Current Treatment Options")
    lines.append(f"- **Standard of care:** {study_info.get('comparator', 'N/A')}")
    lines.append(f"- **Unmet need:** {study_info.get('unmet_need', 'N/A')}")
    lines.append(f"- **Limitations:** {study_info.get('soc_limitations', 'N/A')}")
    lines.append("")

    # Value Tree
    benefits, risks = build_value_tree(endpoints)
    lines.append("## 3. Value Tree (Endpoint Prioritization)")
    lines.append("")
    lines.append("### Benefits (ranked by importance)")
    for i, ep in enumerate(benefits):
        lines.append(f"{i + 1}. **{ep.name}** [{ep.severity}]")
    lines.append("")
    lines.append("### Risks (ranked by severity)")
    for i, ep in enumerate(risks):
        lines.append(f"{i + 1}. **{ep.name}** [{ep.severity}]")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Dimension 3: Benefits
    lines.append("## 4. Benefits")
    lines.append("")
    lines.append("| Endpoint | Intervention | Control | ARD | RD/1000 | Sig | NNT-B |")
    lines.append("|---|---|---|---|---|---|---|")

    primary_ard = None
    benefit_summaries = []
    for ep in benefits:
        rd = compute_risk_difference(ep)
        ard = rd.get("ard")
        sig = rd.get("significant", False)
        nnt = compute_nnt(ard, "benefit") if (ard and sig) else None

        if ep.rank == 1 and ard:
            primary_ard = abs(ard)

        summary = {"endpoint": ep.name, "severity": ep.severity, "rank": ep.rank,
                    "reversibility": "reversible" if "reversible" in ep.severity else "irreversible"}
        summary.update(rd)
        benefit_summaries.append(summary)

        risk_int = f"{rd['risk_int'] * 100:.1f}%" if "risk_int" in rd else "—"
        risk_ctrl = f"{rd['risk_ctrl'] * 100:.1f}%" if "risk_ctrl" in rd else "—"
        ard_str = f"{ard * 100:.2f}%" if ard else "—"
        rd_1000 = f"{rd.get('rd_per1000', rd.get('rd_per1000py', '—'))}"
        nnt_str = f"{nnt:.0f}" if nnt else "—"
        sig_str = "Yes" if sig else "No"

        lines.append(
            f"| {ep.name} | {risk_int} | {risk_ctrl} "
            f"| {ard_str} | {rd_1000} | {sig_str} | {nnt_str} |"
        )
    lines.append("")

    # Dimension 4: Risks
    lines.append("## 5. Risks")
    lines.append("")
    lines.append("| Endpoint | Intervention | Control | ARD | RD/1000 | Sig | NNT-H | Reversibility |")
    lines.append("|---|---|---|---|---|---|---|---|")

    primary_harm_ard = None
    primary_nnt_h = None
    primary_nnt_b = compute_nnt(primary_ard, "benefit") if primary_ard else None

    risk_summaries = []
    for ep in risks:
        rd = compute_risk_difference(ep)
        ard = rd.get("ard")
        sig = rd.get("significant", False)
        nnt = compute_nnt(ard, "risk") if (ard and sig) else None

        if ep.rank == 1 and ard and sig:
            primary_harm_ard = abs(ard)
            primary_nnt_h = nnt

        summary = {"endpoint": ep.name, "severity": ep.severity, "rank": ep.rank,
                    "reversibility": "reversible" if "reversible" in ep.severity else "irreversible"}
        summary.update(rd)
        risk_summaries.append(summary)

        risk_int = f"{rd['risk_int'] * 100:.1f}%" if "risk_int" in rd else "—"
        risk_ctrl = f"{rd['risk_ctrl'] * 100:.1f}%" if "risk_ctrl" in rd else "—"
        ard_str = f"{ard * 100:.2f}%" if ard else "—"
        rd_1000 = f"{rd.get('rd_per1000', rd.get('rd_per1000py', '—'))}"
        nnt_str = f"{nnt:.0f}" if nnt else "—"
        sig_str = "Yes" if sig else "No"
        rev = "Reversible" if "reversible" in ep.severity else "Irreversible"

        lines.append(
            f"| {ep.name} | {risk_int} | {risk_ctrl} "
            f"| {ard_str} | {rd_1000} | {sig_str} | {nnt_str} | {rev} |"
        )
    lines.append("")

    # Quantitative Trade-Off Metrics
    lines.append("## 6. Quantitative Trade-Off Metrics")
    lines.append("")

    brr = None
    if primary_nnt_b and primary_nnt_h:
        brr = compute_brr(primary_nnt_h, primary_nnt_b)
        lines.append(f"- **NNT-B** (primary benefit): {primary_nnt_b:.0f}")
        lines.append(f"- **NNT-H** (primary risk): {primary_nnt_h:.0f}")
        lines.append(f"- **Benefit–Risk Ratio** (NNT-H / NNT-B): **{brr:.2f}**")
        if brr > 1.0:
            lines.append("  - Interpretation: **Favorable** (BRR > 1)")
        else:
            lines.append("  - Interpretation: **Unfavorable** (BRR < 1)")
    else:
        lines.append("- BRR could not be computed (requires significant primary benefit and risk)")
    lines.append("")

    # INB / δ-sensitivity
    delta_results = []
    if primary_ard and primary_harm_ard:
        lines.append("### Incremental Net Benefit (δ-sensitivity)")
        lines.append("")
        delta_results = delta_sensitivity(primary_ard, primary_harm_ard)
        lines.append("| δ (tolerable harms per benefit) | INB | Favorable? |")
        lines.append("|---|---|---|")
        for r in delta_results:
            fav = "Yes" if r["favorable"] else "No"
            lines.append(f"| {r['delta']} | {r['inb']:.4f} | {fav} |")
        lines.append("")

        fav_deltas = [r["delta"] for r in delta_results if r["favorable"]]
        if fav_deltas:
            lines.append(f"**Decision robustness:** Favorable at δ ≥ {min(fav_deltas)}")
        else:
            lines.append("**Decision robustness:** Unfavorable across all δ thresholds")
        lines.append("")

    all_results["brr"] = brr
    all_results["nnt_b"] = primary_nnt_b
    all_results["nnt_h"] = primary_nnt_h
    all_results["delta_sensitivity"] = delta_results

    # Common Denominator
    cd = compute_common_denominator(
        endpoints, n_patients=1000,
        years=study_info.get("follow_up_years", 3)
    )
    if cd:
        fu_years = study_info.get("follow_up_years", 3)
        lines.append("## 7. Common Denominator Summary")
        lines.append(f"Per 1,000 patients treated for {fu_years:.0f} years:")
        lines.append("")
        sig_cd = [r for r in cd if r["significant"]]
        ben_cd = [r for r in sig_cd if r["domain"] == "benefit"]
        risk_cd = [r for r in sig_cd if r["domain"] == "risk"]
        if ben_cd:
            lines.append("**Events prevented (benefits):**")
            for r in ben_cd:
                lines.append(f"- {r['events_per_n']:.0f} fewer {r['name']}")
        if risk_cd:
            lines.append("")
            lines.append("**Excess events (risks):**")
            for r in risk_cd:
                lines.append(f"- {r['events_per_n']:.0f} excess {r['name']}")
        lines.append("")
        lines.append(
            "*Note: Benefits and risks are equally weighted in this summary. "
            "See δ-sensitivity analysis above for weighted assessments.*"
        )
        lines.append("")

    # Integrated Assessment
    quant_results = {"brr": brr, "delta_sensitivity": delta_results}
    ia = _generate_integrated_assessment(
        study_info, benefit_summaries, risk_summaries, quant_results
    )

    lines.append("---")
    lines.append("")
    lines.append("## 8. Integrated Benefit–Risk Assessment")
    lines.append("")
    lines.append(f"### Verdict: **{ia['verdict']}**")
    lines.append("")
    lines.append(ia["reasoning"])
    lines.append("")
    if ia.get("delta_sensitivity_summary"):
        lines.append(ia["delta_sensitivity_summary"])
        lines.append("")

    verdict_map = {
        "FAVORABLE": "The benefit–risk balance is **favorable**. Benefits clearly outweigh risks.",
        "FAVORABLE_WITH_CAVEAT": "The benefit–risk balance is **favorable with caveats**. Benefits outweigh risks, but specific safety concerns warrant monitoring or label restrictions.",
        "INDETERMINATE": "The benefit–risk balance is **indeterminate**. Neither benefit nor risk clearly dominates. Decision depends on clinical context and patient preferences.",
        "UNFAVORABLE": "The benefit–risk balance is **unfavorable**. Risks outweigh demonstrated benefits.",
    }
    lines.append("### Overall Classification")
    lines.append("")
    lines.append(verdict_map.get(ia["verdict"], "Classification unavailable."))
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "*Analysis follows the structured Benefit–Risk Framework "
        "(FDA BRF / Kaul et al., Circulation 2020). "
        "Quantitative metrics: NNT-B, NNT-H, BRR, INB with δ-sensitivity. "
        "Value tree ranked by clinical severity and reversibility.*"
    )

    report = "\n".join(lines)
    all_results["integrated_assessment"] = ia
    return report, all_results


# ═══════════════════════════════════════════════════════════════
# SECTION 7: MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_benefit_risk_assessment(study_info, endpoint_list, output_dir="figures"):
    """
    Main entry point for Benefit–Risk Assessment.

    Parameters
    ----------
    study_info : dict — study metadata and clinical context
    endpoint_list : list of dicts — each dict defines an Endpoint
    output_dir : str — directory for figure PNGs

    Returns
    -------
    (report: str, results: dict, figure_paths: list)
    """
    endpoints = []
    for ep_dict in endpoint_list:
        ep = Endpoint(
            name=ep_dict["name"],
            domain=ep_dict.get("domain", ep_dict.get("type", "benefit")),
            rank=ep_dict.get("rank", 99),
            severity=ep_dict.get("severity", "reversible_serious"),
            events_intervention=ep_dict.get("events_intervention"),
            total_intervention=ep_dict.get("total_intervention", ep_dict.get("n_intervention")),
            events_control=ep_dict.get("events_control", ep_dict.get("events_comparator")),
            total_control=ep_dict.get("total_control", ep_dict.get("n_comparator")),
            effect=ep_dict.get("effect"),
            ci_lower=ep_dict.get("ci_lower"),
            ci_upper=ep_dict.get("ci_upper"),
            p_value=ep_dict.get("p_value"),
            exposure_years_intervention=ep_dict.get("exposure_years_intervention"),
            exposure_years_control=ep_dict.get("exposure_years_control"),
        )
        endpoints.append(ep)

    report, results = generate_br_report(study_info, endpoints)

    # Generate figures
    figure_paths = generate_br_figures(endpoints, results, output_dir)

    return report, results, figure_paths


# ═══════════════════════════════════════════════════════════════
# SECTION 8: FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_br_figures(endpoints, results, output_dir="figures"):
    """
    Generate Benefit–Risk figures:
    1. Forest plot — ARD with 95% CI per endpoint
    2. Benefit–Risk balance bar chart — events per 1000

    Returns list of saved figure paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    os.makedirs(output_dir, exist_ok=True)
    paths = []

    # ── Figure 1: Forest Plot (ARD per endpoint) ──
    benefit_eps = [ep for ep in endpoints if ep.domain == "benefit"]
    risk_eps = [ep for ep in endpoints if ep.domain == "risk"]
    all_eps = benefit_eps + risk_eps

    plot_data = []
    for ep in all_eps:
        rd = compute_risk_difference(ep)
        ard = rd.get("ard")
        if ard is not None:
            # Approximate 95% CI from p_value (Wilson-type)
            ci_lo = rd.get("ci_lower_ard")
            ci_hi = rd.get("ci_upper_ard")
            if ci_lo is None or ci_hi is None:
                # rough approx: ard ± 1.5 * |ard|
                margin = max(abs(ard) * 0.6, 0.005)
                ci_lo = ard - margin
                ci_hi = ard + margin
            plot_data.append({
                "name": ep.name[:30],
                "ard": ard,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "domain": ep.domain,
                "sig": rd.get("significant", False),
            })

    if plot_data:
        try:
            fig, ax = plt.subplots(figsize=(10, max(3, len(plot_data) * 0.6 + 1.5)))
            names = [d["name"] for d in plot_data]
            ards = [d["ard"] * 100 for d in plot_data]
            lows = [d["ci_lo"] * 100 for d in plot_data]
            highs = [d["ci_hi"] * 100 for d in plot_data]
            colors = ["#27AE60" if d["domain"] == "benefit" else "#E74C3C" for d in plot_data]

            y_pos = list(range(len(plot_data)))
            xerr_low = [a - l for a, l in zip(ards, lows)]
            xerr_high = [h - a for a, h in zip(ards, highs)]

            ax.barh(y_pos, ards, color=colors, alpha=0.3, height=0.5)
            ax.errorbar(ards, y_pos, xerr=[xerr_low, xerr_high],
                        fmt='o', color='#1a1a2e', markersize=6, capsize=4, linewidth=1.5)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel("Absolute Risk Difference (%)", fontsize=10)
            ax.set_title("Benefit–Risk Forest Plot (ARD with 95% CI)", fontsize=12, fontweight='bold')
            ax.invert_yaxis()

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#27AE60', alpha=0.3, label='Benefit'),
                Patch(facecolor='#E74C3C', alpha=0.3, label='Risk'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

            plt.tight_layout()
            fpath = os.path.join(output_dir, "br_forest_plot.png")
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(fpath)
        except Exception:
            pass

    # ── Figure 2: Balance Bar Chart (events per 1000) ──
    cd_data = []
    for ep in all_eps:
        rd = compute_risk_difference(ep)
        ard = rd.get("ard")
        if ard is not None and rd.get("significant", False):
            events_per_1000 = abs(ard) * 1000
            cd_data.append({
                "name": ep.name[:25],
                "events": events_per_1000,
                "domain": ep.domain,
            })

    if cd_data:
        try:
            fig, ax = plt.subplots(figsize=(10, max(3, len(cd_data) * 0.6 + 1.5)))
            names = [d["name"] for d in cd_data]
            values = [d["events"] if d["domain"] == "benefit" else -d["events"] for d in cd_data]
            colors = ["#27AE60" if d["domain"] == "benefit" else "#E74C3C" for d in cd_data]

            y_pos = list(range(len(cd_data)))
            ax.barh(y_pos, values, color=colors, height=0.5, edgecolor='white')
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel("Events per 1,000 patients", fontsize=10)
            ax.set_title("Benefit–Risk Balance (Significant Endpoints)", fontsize=12, fontweight='bold')
            ax.invert_yaxis()

            # Annotations
            for i, v in enumerate(values):
                ha = 'left' if v >= 0 else 'right'
                offset = 2 if v >= 0 else -2
                label = f"+{abs(v):.0f} prevented" if v >= 0 else f"+{abs(v):.0f} excess"
                ax.text(v + offset, i, label, va='center', ha=ha, fontsize=8, color='#333')

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#27AE60', label='Benefits (events prevented)'),
                Patch(facecolor='#E74C3C', label='Risks (excess events)'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

            plt.tight_layout()
            fpath = os.path.join(output_dir, "br_balance_chart.png")
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(fpath)
        except Exception:
            pass

    return paths
