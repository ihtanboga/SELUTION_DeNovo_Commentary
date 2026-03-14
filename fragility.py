"""
Fragility Index (scipy-free)

Computes Fragility Index (FI), Fragility Quotient (FQ), RR-based sensitivity FI,
and Reverse Fragility Index for binary / time-to-event RCT endpoints.

Based on: Walsh M, Srinathan SK, McAuley DF, et al. J Clin Epidemiol 2014;67:622-628.

All statistical computations are pure Python — no scipy or numpy required.
Fisher's exact test implemented via hypergeometric distribution using log-gamma.
"""

import math
import statistics


# ═══════════════════════════════════════════════════════════════
# PURE PYTHON FISHER'S EXACT TEST
# ═══════════════════════════════════════════════════════════════

def _log_factorial(n: int) -> float:
    """Log of n! using math.lgamma(n+1)."""
    if n < 0:
        return 0.0
    return math.lgamma(n + 1)


def _hypergeometric_pmf(k: int, K: int, n: int, N: int) -> float:
    """
    Hypergeometric PMF: P(X=k) where X ~ Hypergeometric(N, K, n).

    In 2×2 table terms:
      k  = cell count (a)
      K  = row 1 total (a+b)
      n  = column 1 total (a+c)
      N  = grand total (a+b+c+d)

    P(X=k) = C(K,k) * C(N-K, n-k) / C(N, n)
    """
    # Validate
    if k < 0 or k > K or k > n or (n - k) > (N - K):
        return 0.0

    log_p = (
        _log_factorial(K) - _log_factorial(k) - _log_factorial(K - k)
        + _log_factorial(N - K) - _log_factorial(n - k) - _log_factorial(N - K - n + k)
        - _log_factorial(N) + _log_factorial(n) + _log_factorial(N - n)
    )
    return math.exp(log_p)


def fisher_exact(table, alternative="two-sided"):
    """
    Fisher's exact test for a 2×2 contingency table.

    Parameters
    ----------
    table : list of lists  — [[a, b], [c, d]]
    alternative : str — "two-sided" (only two-sided implemented)

    Returns
    -------
    (odds_ratio, p_value)
    """
    a, b = table[0]
    c, d = table[1]

    # Marginals
    R1 = a + b   # row 1 total
    R2 = c + d   # row 2 total
    C1 = a + c   # col 1 total
    N = R1 + R2  # grand total

    # Odds ratio
    if b * c == 0:
        odds_ratio = float('inf') if a * d > 0 else 0.0
    else:
        odds_ratio = (a * d) / (b * c)

    # Probability of observed table
    p_observed = _hypergeometric_pmf(a, R1, C1, N)

    # Two-sided: sum P of all tables with P <= P_observed
    k_min = max(0, C1 - R2)
    k_max = min(R1, C1)

    p_value = 0.0
    for k in range(k_min, k_max + 1):
        p_k = _hypergeometric_pmf(k, R1, C1, N)
        if p_k <= p_observed + 1e-12:  # tolerance for floating point
            p_value += p_k

    p_value = min(p_value, 1.0)
    return odds_ratio, p_value


# ═══════════════════════════════════════════════════════════════
# SECTION 1: CORE FRAGILITY INDEX CALCULATION
# ═══════════════════════════════════════════════════════════════

def compute_fragility_index(events_int, total_int, events_ctrl, total_ctrl,
                            alpha=0.05):
    """
    Compute the Fragility Index for a 2×2 table.

    Algorithm (Walsh et al. 2014):
    1. Identify the group with fewer events.
    2. Add one event to that group (subtract one non-event to keep N constant).
    3. Recalculate two-sided P-value using Fisher's exact test.
    4. Repeat until P >= alpha.
    5. FI = number of events added.

    Returns
    -------
    dict with fi, fq, modified_group, original_p, final_p, iterations, tables
    """
    assert events_int <= total_int, "Events cannot exceed total in intervention"
    assert events_ctrl <= total_ctrl, "Events cannot exceed total in control"
    assert events_int >= 0 and events_ctrl >= 0, "Events must be non-negative"

    a = events_int
    b = total_int - a
    c = events_ctrl
    d = total_ctrl - c

    # Check original significance
    _, p_orig = fisher_exact([[a, b], [c, d]])

    if p_orig >= alpha:
        return {
            "fi": 0,
            "fq": 0.0,
            "modified_group": "none",
            "original_p": round(p_orig, 6),
            "final_p": round(p_orig, 6),
            "note": "Original result is not statistically significant (P >= alpha). FI = 0.",
            "iterations": [],
            "original_table": {"a": a, "b": b, "c": c, "d": d},
            "final_table": {"a": a, "b": b, "c": c, "d": d},
        }

    # Identify group with fewer events → add events to this group
    modify = "intervention" if a <= c else "control"

    fi = 0
    iterations = []
    current_a, current_b = a, b
    current_c, current_d = c, d
    p_new = p_orig
    max_iterations = max(total_int, total_ctrl)

    while fi < max_iterations:
        if modify == "intervention":
            if current_b <= 0:
                break
            current_a += 1
            current_b -= 1
        else:
            if current_d <= 0:
                break
            current_c += 1
            current_d -= 1

        fi += 1
        _, p_new = fisher_exact([[current_a, current_b], [current_c, current_d]])

        iterations.append({
            "step": fi,
            "a": current_a, "b": current_b,
            "c": current_c, "d": current_d,
            "p_value": round(p_new, 6),
        })

        if p_new >= alpha:
            break

    total_n = total_int + total_ctrl
    fq = round(fi / total_n, 6) if total_n > 0 else 0.0

    return {
        "fi": fi,
        "fq": fq,
        "modified_group": modify,
        "original_p": round(p_orig, 6),
        "final_p": round(p_new, 6) if iterations else round(p_orig, 6),
        "iterations": iterations,
        "original_table": {"a": a, "b": b, "c": c, "d": d},
        "final_table": {
            "a": current_a, "b": current_b,
            "c": current_c, "d": current_d,
        },
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 2: SENSITIVITY — RR-BASED FRAGILITY INDEX
# ═══════════════════════════════════════════════════════════════

def compute_fragility_index_rr(events_int, total_int, events_ctrl, total_ctrl):
    """
    Sensitivity analysis: FI defined as events needed until
    the 95% CI for Relative Risk includes 1.0.
    """
    a = events_int
    b = total_int - a
    c = events_ctrl
    d = total_ctrl - c

    modify = "intervention" if a <= c else "control"

    def rr_ci(a_, n1, c_, n2):
        """Compute RR and 95% CI using log-normal approximation."""
        if a_ == 0 or c_ == 0:
            a_h, c_h = a_ + 0.5, c_ + 0.5
            n1_h, n2_h = n1 + 0.5, n2 + 0.5
        else:
            a_h, c_h = a_, c_
            n1_h, n2_h = n1, n2

        rr = (a_h / n1_h) / (c_h / n2_h)
        log_rr = math.log(rr)
        se_log_rr = math.sqrt(1 / a_h - 1 / n1_h + 1 / c_h - 1 / n2_h)
        ci_lower = math.exp(log_rr - 1.96 * se_log_rr)
        ci_upper = math.exp(log_rr + 1.96 * se_log_rr)
        return rr, ci_lower, ci_upper

    rr_orig, ci_lo_orig, ci_hi_orig = rr_ci(a, total_int, c, total_ctrl)
    includes_one = ci_lo_orig <= 1.0 <= ci_hi_orig

    if includes_one:
        return {
            "fi_rr": 0,
            "note": "Original 95% CI for RR already includes 1.0. FI_RR = 0.",
            "rr": round(rr_orig, 4),
            "ci_lower": round(ci_lo_orig, 4),
            "ci_upper": round(ci_hi_orig, 4),
            "iterations": [],
        }

    fi = 0
    iterations = []
    current_a, current_b = a, b
    current_c, current_d = c, d
    max_iter = max(total_int, total_ctrl)
    rr_new, ci_lo, ci_hi = rr_orig, ci_lo_orig, ci_hi_orig

    while fi < max_iter:
        if modify == "intervention":
            if current_b <= 0:
                break
            current_a += 1
            current_b -= 1
        else:
            if current_d <= 0:
                break
            current_c += 1
            current_d -= 1

        fi += 1
        rr_new, ci_lo, ci_hi = rr_ci(
            current_a, total_int, current_c, total_ctrl
        )

        iterations.append({
            "step": fi,
            "rr": round(rr_new, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
        })

        if ci_lo <= 1.0 <= ci_hi:
            break

    return {
        "fi_rr": fi,
        "modified_group": modify,
        "original_rr": round(rr_orig, 4),
        "original_ci": (round(ci_lo_orig, 4), round(ci_hi_orig, 4)),
        "final_rr": round(rr_new, 4),
        "final_ci": (round(ci_lo, 4), round(ci_hi, 4)),
        "iterations": iterations,
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 3: REVERSE FRAGILITY INDEX
# ═══════════════════════════════════════════════════════════════

def compute_reverse_fragility_index(events_int, total_int,
                                    events_ctrl, total_ctrl, alpha=0.05):
    """
    Reverse Fragility Index: for a NON-significant result,
    how many events must be removed from the group with MORE events
    to make the result significant?
    """
    a = events_int
    b = total_int - a
    c = events_ctrl
    d = total_ctrl - c

    _, p_orig = fisher_exact([[a, b], [c, d]])

    if p_orig < alpha:
        return {
            "reverse_fi": None,
            "note": "Result is already significant. Reverse FI not applicable.",
            "original_p": round(p_orig, 6),
        }

    # Remove events from group with MORE events
    modify = "intervention" if a >= c else "control"

    rfi = 0
    current_a, current_b = a, b
    current_c, current_d = c, d
    p_new = p_orig

    while rfi < max(total_int, total_ctrl):
        if modify == "intervention":
            if current_a <= 0:
                break
            current_a -= 1
            current_b += 1
        else:
            if current_c <= 0:
                break
            current_c -= 1
            current_d += 1

        rfi += 1
        _, p_new = fisher_exact(
            [[current_a, current_b], [current_c, current_d]]
        )

        if p_new < alpha:
            break
    else:
        return {
            "reverse_fi": None,
            "note": "Could not achieve significance by removing events.",
            "original_p": round(p_orig, 6),
        }

    return {
        "reverse_fi": rfi,
        "modified_group": modify,
        "original_p": round(p_orig, 6),
        "final_p": round(p_new, 6),
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 4: BATCH ANALYSIS FOR ALL ENDPOINTS
# ═══════════════════════════════════════════════════════════════

def analyze_all_endpoints(endpoint_list):
    """
    Compute Fragility Index for all endpoints in a trial.

    Parameters
    ----------
    endpoint_list : list of dicts
        Each: {name, events_intervention, total_intervention,
               events_control, total_control, significant, is_primary, outcome_type}

    Returns
    -------
    list of dicts with FI results
    """
    results = []

    for ep in endpoint_list:
        name = ep["name"]
        a = ep["events_intervention"]
        n1 = ep["total_intervention"]
        c = ep["events_control"]
        n2 = ep["total_control"]
        sig = ep.get("significant", True)

        if sig:
            fi_result = compute_fragility_index(a, n1, c, n2)
            fi_rr_result = compute_fragility_index_rr(a, n1, c, n2)

            results.append({
                "name": name,
                "is_primary": ep.get("is_primary", False),
                "outcome_type": ep.get("outcome_type", "binary"),
                "significant": True,
                "fi_fisher": fi_result["fi"],
                "fi_rr": fi_rr_result.get("fi_rr", None),
                "fq": fi_result["fq"],
                "original_p": fi_result["original_p"],
                "final_p_fisher": fi_result["final_p"],
                "modified_group": fi_result["modified_group"],
                "total_n": n1 + n2,
                "total_events": a + c,
                "details_fisher": fi_result,
                "details_rr": fi_rr_result,
            })
        else:
            rfi_result = compute_reverse_fragility_index(a, n1, c, n2)
            results.append({
                "name": name,
                "is_primary": ep.get("is_primary", False),
                "outcome_type": ep.get("outcome_type", "binary"),
                "significant": False,
                "reverse_fi": rfi_result.get("reverse_fi"),
                "original_p": rfi_result.get("original_p"),
                "total_n": n1 + n2,
                "total_events": a + c,
                "details_reverse": rfi_result,
            })

    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 5: INTERPRETATION
# ═══════════════════════════════════════════════════════════════

def interpret_fi(fi, total_n, total_events):
    """Qualitative interpretation of the Fragility Index."""
    fq = fi / total_n if total_n > 0 else 0
    event_ratio = fi / total_events if total_events > 0 else 0

    if fi == 0:
        return (
            "The result is not statistically significant or became "
            "non-significant with zero additional events."
        )
    elif fi <= 3:
        return (
            f"FI = {fi}: The statistical significance of this result is "
            f"extremely fragile. Changing the outcome of just {fi} "
            f"patient(s) would render the result non-significant. "
            f"This represents {event_ratio * 100:.1f}% of total events "
            f"and {fq * 100:.3f}% of the total sample. "
            f"Results should be interpreted with great caution."
        )
    elif fi <= 10:
        return (
            f"FI = {fi}: The statistical significance shows moderate fragility. "
            f"Changing {fi} patient outcomes ({event_ratio * 100:.1f}% of events, "
            f"FQ = {fq * 100:.3f}%) would overturn the result. "
            f"Robustness is context-dependent."
        )
    elif fi <= 25:
        return (
            f"FI = {fi}: The result shows reasonable robustness. "
            f"{fi} additional events needed to overturn significance "
            f"({event_ratio * 100:.1f}% of events, FQ = {fq * 100:.3f}%)."
        )
    else:
        return (
            f"FI = {fi}: The result is robust. A large number of event "
            f"changes would be needed to overturn statistical significance "
            f"({event_ratio * 100:.1f}% of events, FQ = {fq * 100:.3f}%)."
        )


# ═══════════════════════════════════════════════════════════════
# SECTION 6: REPORT GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_fi_report(study_info, fi_results):
    """Generate Markdown report for Fragility Index analysis."""
    lines = []
    lines.append("# Fragility Index Report")
    lines.append("")
    lines.append(f"**Study:** {study_info.get('title', 'N/A')}")
    lines.append(f"**Total N:** {study_info.get('n_total', 'N/A')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    sig_results = [r for r in fi_results if r["significant"]]
    nonsig_results = [r for r in fi_results if not r["significant"]]

    if sig_results:
        lines.append("## Fragility Index — Significant Endpoints")
        lines.append("")
        lines.append(
            "| Endpoint | Primary | Events (Int/Ctrl) | N | "
            "Original P | FI (Fisher) | FI (RR) | FQ | Interpretation |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for r in sig_results:
            t = r["details_fisher"]["original_table"]
            events_str = f"{t['a']}/{t['c']}"
            primary = "Yes" if r["is_primary"] else "No"
            fi_rr = r.get("fi_rr", "—")
            fi_rr_str = f"{fi_rr}" if fi_rr is not None else "—"

            if r["fi_fisher"] <= 3:
                interp_short = "Extremely fragile"
            elif r["fi_fisher"] <= 10:
                interp_short = "Moderately fragile"
            elif r["fi_fisher"] <= 25:
                interp_short = "Reasonably robust"
            else:
                interp_short = "Robust"

            lines.append(
                f"| {r['name']} | {primary} | {events_str} | {r['total_n']} "
                f"| {r['original_p']:.4f} | **{r['fi_fisher']}** | {fi_rr_str} "
                f"| {r['fq']:.5f} | {interp_short} |"
            )
        lines.append("")

        # Detailed interpretations
        lines.append("<details><summary>Detailed interpretations</summary>")
        lines.append("")
        for r in sig_results:
            interp = interpret_fi(r["fi_fisher"], r["total_n"], r["total_events"])
            lines.append(f"**{r['name']}:** {interp}")
            lines.append("")
            if r.get("fi_rr") is not None and r["fi_rr"] != r["fi_fisher"]:
                lines.append(
                    f"- Sensitivity (RR-based): FI = {r['fi_rr']} "
                    f"(vs Fisher-based FI = {r['fi_fisher']})"
                )
                lines.append("")
        lines.append("</details>")
        lines.append("")

    if nonsig_results:
        lines.append("## Reverse Fragility Index — Non-Significant Endpoints")
        lines.append("")
        lines.append(
            "| Endpoint | Events (Int/Ctrl) | N | Original P | Reverse FI |"
        )
        lines.append("|---|---|---|---|---|")

        for r in nonsig_results:
            rfi = r.get("reverse_fi", "—")
            rfi_str = f"{rfi}" if rfi is not None else "N/A"
            lines.append(
                f"| {r['name']} | {r['total_events']} | {r['total_n']} "
                f"| {r.get('original_p', '—')} | {rfi_str} |"
            )
        lines.append("")

    # Summary
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    if sig_results:
        fis = [r["fi_fisher"] for r in sig_results]
        primary_fi = [r for r in sig_results if r["is_primary"]]

        lines.append(
            f"- **Endpoints analyzed:** {len(sig_results)} significant, "
            f"{len(nonsig_results)} non-significant"
        )
        lines.append(
            f"- **FI range:** {min(fis)} – {max(fis)} "
            f"(median {int(statistics.median(fis))})"
        )

        if primary_fi:
            pfi = primary_fi[0]["fi_fisher"]
            lines.append(f"- **Primary endpoint FI:** {pfi}")
            if pfi <= 3:
                lines.append(
                    "  - The primary endpoint result is **extremely fragile** "
                    "and should be interpreted with caution."
                )
            elif pfi <= 10:
                lines.append(
                    "  - The primary endpoint shows **moderate fragility**."
                )
    lines.append("")

    # Caveats
    lines.append("### Caveats")
    lines.append("")
    lines.append(
        "The Fragility Index has well-known limitations: it depends on "
        "sample size, event rate, and the alpha threshold; it does not "
        "account for effect size magnitude, clinical importance, or "
        "multiplicity adjustments; and it applies only to binary or "
        "event-count outcomes. FI should be interpreted alongside "
        "confidence intervals, Bayesian posterior probabilities, and "
        "clinical context — never as a standalone measure of evidence quality."
    )
    lines.append("")
    lines.append(
        "*For time-to-event outcomes, the 2×2 table uses total events "
        "during the entire follow-up period, not KM estimates.*"
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# SECTION 7: MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_fragility_analysis(study_info, endpoint_list):
    """
    Main entry point for Fragility Index computation.

    Parameters
    ----------
    study_info : dict  — {"title": str, "n_total": int, ...}
    endpoint_list : list of dicts — each with event counts + significance

    Returns
    -------
    (report: str, results: list)
    """
    results = analyze_all_endpoints(endpoint_list)
    report = generate_fi_report(study_info, results)
    return report, results
