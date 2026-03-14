"""
Bayesian Re-Analysis

Framework: Zampieri FG, Casey JD, Shankar-Hari M, Harrell FE Jr, Harhay MO.
           Am J Respir Crit Care Med 2021;203(5):543-552.

Zero external dependencies (no numpy/scipy).
Every RCT gets a Bayesian re-analysis with:
  - 3 priors (skeptical, optimistic, pessimistic)
  - Posterior computation (normal-normal conjugate)
  - Probability evidence ladder (P1-P7)
  - Posterior threshold curves (PNG figures)
  - Prior sensitivity assessment
"""

import json
import math
import os
# === Normal distribution functions (pure Python, no numpy/scipy) ===
# Using math.erf for CDF, rational approximation for PPF


def _geomspace(start, stop, num):
    """Pure Python replacement for numpy.geomspace."""
    if num <= 1:
        return [start]
    log_start = math.log(start)
    log_stop = math.log(stop)
    step = (log_stop - log_start) / (num - 1)
    return [math.exp(log_start + i * step) for i in range(num)]


class _NormDist:
    """Minimal normal distribution — drop-in replacement for scipy.stats.norm."""

    @staticmethod
    def cdf(x, loc=0, scale=1):
        """Normal CDF using error function."""
        z = (x - loc) / scale if scale > 0 else 0
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

    @staticmethod
    def ppf(p):
        """Normal PPF (inverse CDF) — rational approximation (Abramowitz & Stegun 26.2.23)."""
        if p <= 0:
            return -float('inf')
        if p >= 1:
            return float('inf')
        if p == 0.5:
            return 0.0

        if p < 0.5:
            t = math.sqrt(-2 * math.log(p))
        else:
            t = math.sqrt(-2 * math.log(1 - p))

        # Rational approximation coefficients
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308

        result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
        return -result if p < 0.5 else result

norm = _NormDist()


# ═══════════════════════════════════════════════════════════════
# SECTION 1: LIKELIHOOD COMPUTATION
# ═══════════════════════════════════════════════════════════════

def likelihood_from_events(a, b, c, d):
    """
    Compute log(OR) and SE from a 2x2 table.
    a = events intervention, b = non-events intervention
    c = events control, d = non-events control
    Haldane correction (+0.5) if any cell is zero.
    """
    if any(x == 0 for x in [a, b, c, d]):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    log_or = math.log((a / b) / (c / d))
    se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    return {"log_or": log_or, "se": se}


def likelihood_from_effect(effect, ci_lower, ci_upper):
    """Derive log(effect) and SE from reported OR/RR/HR with 95% CI."""
    log_effect = math.log(effect)
    se = (math.log(ci_upper) - math.log(ci_lower)) / (2 * 1.96)
    return {"log_or": log_effect, "se": se}


def likelihood_from_mean_diff(md, se):
    """For continuous outcomes: use mean difference directly."""
    return {"log_or": md, "se": se}


# ═══════════════════════════════════════════════════════════════
# SECTION 2: PRIOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

def find_sd_for_target_prob(mean_log_or, target_prob, direction="harm"):
    """
    Find SD of a normal prior such that the probability of
    opposite-direction effect equals target_prob.
    """
    for sd_x100 in range(1, 500):
        sd = sd_x100 * 0.01
        if direction == "harm":
            prob = 1 - norm.cdf(0, loc=mean_log_or, scale=sd)
        else:
            prob = norm.cdf(0, loc=mean_log_or, scale=sd)
        if abs(prob - target_prob) < 0.003:
            return round(sd, 3)
    # Analytic fallback
    return round(abs(mean_log_or) / norm.ppf(1 - target_prob), 3)


def build_priors(expected_or=None, evidence_level="moderate", outcome_type="binary"):
    """
    Build the minimum family of 3 priors following Zampieri framework.
    - Skeptical: centered at OR=1 (no effect)
    - Optimistic: centered at expected beneficial OR
    - Pessimistic: mirror of optimistic
    """
    priors = []

    # Skeptical prior
    if evidence_level in ["weak_rationale", "many_neutral"]:
        priors.append({
            "name": "Skeptical (Strong)",
            "mean": 0,
            "sd": 0.205,
            "rationale": (
                "Strong skeptical prior: N(0, 0.205). "
                "95% probability that OR is between 0.67 and 1.50. "
                "Used because prior evidence suggests weak rationale for "
                "a direct effect or multiple neutral trials exist."
            )
        })
    else:
        priors.append({
            "name": "Skeptical (Moderate)",
            "mean": 0,
            "sd": 0.355,
            "rationale": (
                "Moderate skeptical prior: N(0, 0.355). "
                "95% probability that OR is between 0.50 and 2.00. "
                "Represents clinical equipoise."
            )
        })

    # Determine delta (expected effect size)
    if expected_or is not None:
        delta = math.log(expected_or)
    elif outcome_type == "continuous":
        delta = -0.3
    else:
        delta = math.log(0.75)

    # Optimistic prior
    if evidence_level == "none":
        strength_label = "Weak"
        target_opposite = 0.30
    elif evidence_level in ["conflicting", "supportive"]:
        strength_label = "Moderate"
        target_opposite = 0.15
    elif evidence_level == "strong_support":
        strength_label = "Strong"
        target_opposite = 0.05
    else:
        strength_label = "Weak"
        target_opposite = 0.30

    sd_opt = find_sd_for_target_prob(delta, target_opposite, direction="harm")
    priors.append({
        "name": f"Optimistic ({strength_label})",
        "mean": round(delta, 4),
        "sd": sd_opt,
        "rationale": (
            f"Optimistic prior: N({delta:.3f}, {sd_opt:.3f}). "
            f"Centered at OR={math.exp(delta):.2f}. "
            f"SD chosen so that P(OR>1) = {target_opposite:.2f}. "
            f"Strength: {strength_label.lower()} ({evidence_level} prior evidence)."
        )
    })

    # Pessimistic prior (mirror)
    if evidence_level in ["none", "supportive", "weak_rationale", "many_neutral"]:
        pes_strength_label = "Weak"
        pes_target = 0.30
    elif evidence_level == "conflicting":
        pes_strength_label = "Moderate"
        pes_target = 0.15
    else:
        pes_strength_label = "Weak"
        pes_target = 0.30

    sd_pes = find_sd_for_target_prob(-delta, pes_target, direction="benefit")
    priors.append({
        "name": f"Pessimistic ({pes_strength_label})",
        "mean": round(-delta, 4),
        "sd": sd_pes,
        "rationale": (
            f"Pessimistic prior: N({-delta:.3f}, {sd_pes:.3f}). "
            f"Centered at OR={math.exp(-delta):.2f} (mirror of optimistic). "
            f"SD chosen so that P(OR<1) = {pes_target:.2f}. "
            f"Strength: {pes_strength_label.lower()}."
        )
    })

    return priors


# ═══════════════════════════════════════════════════════════════
# SECTION 3: POSTERIOR COMPUTATION (Normal-Normal Conjugate)
# ═══════════════════════════════════════════════════════════════

def compute_posterior(prior_mean, prior_sd, likelihood_mean, likelihood_se):
    """Analytic normal-normal conjugate update."""
    prior_var = prior_sd ** 2
    obs_var = likelihood_se ** 2

    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    post_mean = post_var * (prior_mean / prior_var + likelihood_mean / obs_var)
    post_sd = math.sqrt(post_var)

    post_or = math.exp(post_mean)
    cri_lower = math.exp(post_mean - 1.96 * post_sd)
    cri_upper = math.exp(post_mean + 1.96 * post_sd)

    p_benefit = norm.cdf(0, loc=post_mean, scale=post_sd)
    p_harm = 1.0 - p_benefit
    p_severe_harm = 1 - norm.cdf(math.log(1.25), post_mean, post_sd)
    p_notable_benefit = norm.cdf(math.log(0.80), post_mean, post_sd)

    rope_lo = math.log(1.0 / 1.1)
    rope_hi = math.log(1.1)
    p_rope = norm.cdf(rope_hi, post_mean, post_sd) - norm.cdf(rope_lo, post_mean, post_sd)

    return {
        "post_mean": round(post_mean, 4),
        "post_sd": round(post_sd, 4),
        "post_or": round(post_or, 3),
        "cri_lower": round(cri_lower, 3),
        "cri_upper": round(cri_upper, 3),
        "p_benefit": round(p_benefit, 4),
        "p_harm": round(p_harm, 4),
        "p_severe_harm": round(p_severe_harm, 4),
        "p_notable_benefit": round(p_notable_benefit, 4),
        "p_rope": round(p_rope, 4),
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 3B: PROBABILITY EVIDENCE LADDER & THRESHOLD CURVE
# ═══════════════════════════════════════════════════════════════

def convert_absolute_to_relative_ni_margin(absolute_margin, active_control_event_rate):
    """Convert absolute NI margin to relative margin (OR scale)."""
    relative_margin = (absolute_margin + active_control_event_rate) / active_control_event_rate
    return round(relative_margin, 4)


def compute_probability_ladder(post_mean, post_sd, ni_margin=None,
                                direction="benefit_below"):
    """
    Compute Harrell-style Probability Evidence Ladder.
    P1-P7: graduated scale of clinical evidence.
    """
    ni = ni_margin if ni_margin is not None else 1.10

    if direction == "benefit_below":
        ladder = [
            {"label": "P1", "threshold_or": 1.0,
             "description": "Any benefit",
             "probability": round(norm.cdf(math.log(1.0), post_mean, post_sd), 4)},
            {"label": "P2", "threshold_or": 1/1.05,
             "description": "More than trivial benefit",
             "probability": round(norm.cdf(math.log(1/1.05), post_mean, post_sd), 4)},
            {"label": "P3", "threshold_or": 1/1.25,
             "description": "Moderate benefit or greater",
             "probability": round(norm.cdf(math.log(1/1.25), post_mean, post_sd), 4)},
            {"label": "P4", "threshold_or": 1.0,
             "description": "Inefficacy or harm",
             "probability": round(1 - norm.cdf(math.log(1.0), post_mean, post_sd), 4)},
            {"label": "P5", "threshold_or": 1.05,
             "description": "More than trivial harm",
             "probability": round(1 - norm.cdf(math.log(1.05), post_mean, post_sd), 4)},
            {"label": "P6", "threshold_or": ni,
             "description": f"Non-inferiority (margin={ni:.2f})",
             "probability": round(norm.cdf(math.log(ni), post_mean, post_sd), 4)},
            {"label": "P7", "threshold_or": "4/5 to 5/4",
             "description": "Similarity (0.80 < OR < 1.25)",
             "probability": round(
                 norm.cdf(math.log(5/4), post_mean, post_sd) -
                 norm.cdf(math.log(4/5), post_mean, post_sd), 4)},
        ]
    else:
        ladder = [
            {"label": "P1", "threshold_or": 1.0,
             "description": "Any benefit",
             "probability": round(1 - norm.cdf(math.log(1.0), post_mean, post_sd), 4)},
            {"label": "P2", "threshold_or": 1.05,
             "description": "More than trivial benefit",
             "probability": round(1 - norm.cdf(math.log(1.05), post_mean, post_sd), 4)},
            {"label": "P3", "threshold_or": 1.25,
             "description": "Moderate benefit or greater",
             "probability": round(1 - norm.cdf(math.log(1.25), post_mean, post_sd), 4)},
            {"label": "P4", "threshold_or": 1.0,
             "description": "Inefficacy or harm",
             "probability": round(norm.cdf(math.log(1.0), post_mean, post_sd), 4)},
            {"label": "P5", "threshold_or": 1/1.05,
             "description": "More than trivial harm",
             "probability": round(norm.cdf(math.log(1/1.05), post_mean, post_sd), 4)},
            {"label": "P6", "threshold_or": 1/ni,
             "description": f"Non-inferiority (margin={1/ni:.2f})",
             "probability": round(
                 1 - norm.cdf(math.log(1/ni), post_mean, post_sd), 4)},
            {"label": "P7", "threshold_or": "4/5 to 5/4",
             "description": "Similarity (0.80 < OR < 1.25)",
             "probability": round(
                 norm.cdf(math.log(5/4), post_mean, post_sd) -
                 norm.cdf(math.log(4/5), post_mean, post_sd), 4)},
        ]

    return ladder


def generate_threshold_curve_data(post_mean, post_sd,
                                   or_range=(0.40, 2.50), n_points=200,
                                   direction="benefit_below"):
    """Generate data for Posterior Threshold Curve."""
    or_values = _geomspace(or_range[0], or_range[1], n_points)
    log_or_values = [math.log(v) for v in or_values]

    if direction == "benefit_below":
        prob_values = [norm.cdf(lv, loc=post_mean, scale=post_sd) for lv in log_or_values]
        ylabel = "P(true OR < threshold | data)"
    else:
        prob_values = [1 - norm.cdf(lv, loc=post_mean, scale=post_sd) for lv in log_or_values]
        ylabel = "P(true OR > threshold | data)"

    return {
        "or_values": [round(v, 4) for v in or_values],
        "prob_values": [round(v, 4) for v in prob_values],
        "xlabel": "OR Threshold",
        "ylabel": ylabel,
    }


def plot_multi_prior_threshold_curve(post_params_list, outcome_name,
                                      or_range=(0.40, 2.50), n_points=200,
                                      direction="benefit_below", save_path=None):
    """Overlay threshold curves for all priors on a single figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2563EB", "#DC2626", "#059669", "#7C3AED", "#D97706"]
    or_vals = _geomspace(or_range[0], or_range[1], n_points)
    log_or_vals = [math.log(v) for v in or_vals]

    for i, params in enumerate(post_params_list):
        if direction == "benefit_below":
            probs = [norm.cdf(lv, loc=params["post_mean"],
                              scale=params["post_sd"]) for lv in log_or_vals]
        else:
            probs = [1 - norm.cdf(lv, loc=params["post_mean"],
                                   scale=params["post_sd"]) for lv in log_or_vals]
        color = colors[i % len(colors)]
        ax.plot(or_vals, probs, color=color, linewidth=1.8,
                label=params["name"])

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axhline(y=0.95, color="#059669", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.80, color="#D97706", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axvline(x=1.0, color="black", linestyle="-", linewidth=1.0, alpha=0.5)

    benefit_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    harm_thresholds = [1.10, 1.25, 1.50]
    for kor in benefit_thresholds + harm_thresholds:
        if or_range[0] <= kor <= or_range[1]:
            ax.axvline(x=kor, color="#E5E7EB", linestyle=":", linewidth=0.5)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xticks([0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0, 1.10, 1.25, 1.50, 2.0, 2.5])
    ax.get_xaxis().set_tick_params(which='minor', size=0)

    ylabel = ("P(true OR < threshold | data)" if direction == "benefit_below"
              else "P(true OR > threshold | data)")
    ax.set_xlabel("OR Threshold (log scale)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"Posterior Threshold Curve \u2014 {outcome_name}",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlim(or_range[0], or_range[1])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9, edgecolor="#D1D5DB")
    ax.grid(True, alpha=0.15, which="both")

    if direction == "benefit_below":
        ax.axvspan(or_range[0], 1.0, alpha=0.04, color="#2563EB")
        ax.text(0.62, 0.96, "\u2190 Benefit", transform=ax.transAxes,
                fontsize=9, color="#2563EB", alpha=0.7, ha="center")
        ax.text(0.85, 0.96, "Harm \u2192", transform=ax.transAxes,
                fontsize=9, color="#DC2626", alpha=0.7, ha="center")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    else:
        plt.close(fig)
        return fig


# ═══════════════════════════════════════════════════════════════
# SECTION 4: SINGLE OUTCOME ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_outcome(outcome_name, outcome_type, likelihood, priors,
                    rope_range=(1/1.1, 1.1),
                    severe_harm_threshold=1.25,
                    notable_benefit_threshold=0.80,
                    ni_margin=None,
                    ni_margin_absolute=None,
                    active_control_event_rate=None,
                    direction="benefit_below"):
    """Run full Bayesian re-analysis for a single outcome."""
    log_or_obs = likelihood["log_or"]
    se_obs = likelihood["se"]

    # Resolve NI margin
    effective_ni_margin = ni_margin
    ni_conversion_note = None
    if ni_margin_absolute is not None and active_control_event_rate is not None:
        effective_ni_margin = convert_absolute_to_relative_ni_margin(
            ni_margin_absolute, active_control_event_rate
        )
        ni_conversion_note = (
            f"Absolute NI margin {ni_margin_absolute*100:.1f}% converted to "
            f"relative margin OR={effective_ni_margin:.4f} using anticipated "
            f"active control event rate {active_control_event_rate*100:.1f}%."
        )

    log_severe = math.log(severe_harm_threshold)
    log_benefit = math.log(notable_benefit_threshold)
    rope_lo = math.log(rope_range[0])
    rope_hi = math.log(rope_range[1])

    results = {
        "outcome_name": outcome_name,
        "outcome_type": outcome_type,
        "frequentist": {
            "log_or": round(log_or_obs, 4),
            "se": round(se_obs, 4),
            "or": round(math.exp(log_or_obs), 3),
            "ci_lower": round(math.exp(log_or_obs - 1.96 * se_obs), 3),
            "ci_upper": round(math.exp(log_or_obs + 1.96 * se_obs), 3),
        },
        "priors_used": [],
        "posteriors": [],
    }

    for prior in priors:
        post = compute_posterior(prior["mean"], prior["sd"], log_or_obs, se_obs)

        post_mean = post["post_mean"]
        post_sd = post["post_sd"]
        post["p_severe_harm"] = round(
            1 - norm.cdf(log_severe, post_mean, post_sd), 4)
        post["p_notable_benefit"] = round(
            norm.cdf(log_benefit, post_mean, post_sd), 4)
        post["p_rope"] = round(
            norm.cdf(rope_hi, post_mean, post_sd) - norm.cdf(rope_lo, post_mean, post_sd), 4)

        results["priors_used"].append({
            "name": prior["name"],
            "mean": prior["mean"],
            "sd": prior["sd"],
            "rationale": prior["rationale"],
        })
        results["posteriors"].append({
            "prior_name": prior["name"],
            **post,
        })

    # Probability evidence ladder
    for i, post in enumerate(results["posteriors"]):
        ladder = compute_probability_ladder(
            post["post_mean"], post["post_sd"],
            ni_margin=effective_ni_margin, direction=direction)
        results["posteriors"][i]["probability_ladder"] = ladder

    # Threshold curve data
    for i, post in enumerate(results["posteriors"]):
        curve = generate_threshold_curve_data(
            post["post_mean"], post["post_sd"], direction=direction)
        results["posteriors"][i]["threshold_curve"] = curve

    # Multi-prior curve params
    results["multi_prior_curve_params"] = [
        {"name": p["prior_name"], "post_mean": p["post_mean"], "post_sd": p["post_sd"]}
        for p in results["posteriors"]
    ]
    results["direction"] = direction
    results["ni_margin"] = effective_ni_margin
    results["ni_conversion_note"] = ni_conversion_note

    # Prior sensitivity assessment
    ors = [p["post_or"] for p in results["posteriors"]]
    harms = [p["p_harm"] for p in results["posteriors"]]
    max_or_diff = max(ors) - min(ors)
    all_same_direction = all(h > 0.5 for h in harms) or all(h < 0.5 for h in harms)

    if all_same_direction and max_or_diff < 0.10:
        sensitivity = "robust"
        sensitivity_text = (
            "Results are robust to prior selection. "
            "All priors yield the same direction with similar magnitude.")
    elif all_same_direction:
        sensitivity = "moderate"
        sensitivity_text = (
            "All priors yield the same direction, but magnitude differs somewhat.")
    else:
        sensitivity = "sensitive"
        sensitivity_text = (
            "WARNING: Results are sensitive to prior selection. "
            "Different priors yield different directional conclusions. "
            "Interpret with caution.")

    results["sensitivity"] = {
        "level": sensitivity,
        "max_or_difference": round(max_or_diff, 3),
        "interpretation": sensitivity_text,
    }

    # Clinical interpretation (based on skeptical prior)
    skeptic = [p for p in results["posteriors"] if "Skeptical" in p["prior_name"]]
    if skeptic:
        s = skeptic[0]
        if s["p_benefit"] > 0.95:
            verdict = "High probability of benefit (>0.95)."
        elif s["p_benefit"] > 0.80:
            verdict = "Probable benefit (>0.80), but some uncertainty remains."
        elif s["p_benefit"] > 0.60:
            verdict = "Slight trend toward benefit, substantial uncertainty."
        elif s["p_harm"] > 0.95:
            verdict = "High probability of harm (>0.95)."
        elif s["p_harm"] > 0.80:
            verdict = "Probable harm (>0.80), but some uncertainty remains."
        elif s["p_harm"] > 0.60:
            verdict = "Slight trend toward harm, substantial uncertainty."
        else:
            verdict = "No clear difference between intervention and control."

        if s["p_rope"] > 0.50:
            verdict += " ROPE analysis suggests the effect is likely clinically negligible."
        elif s["p_rope"] < 0.10:
            verdict += " ROPE analysis suggests the effect is likely clinically meaningful."

        results["interpretation"] = verdict

    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 5: HELPER — EXTRACT LIKELIHOOD
# ═══════════════════════════════════════════════════════════════

def _extract_likelihood(ep):
    """Extract likelihood from endpoint dict based on available data."""
    if "events_intervention" in ep and ep.get("events_intervention") is not None:
        a = ep["events_intervention"]
        b = ep["total_intervention"] - a
        c = ep["events_control"]
        d = ep["total_control"] - c
        return likelihood_from_events(a, b, c, d)
    elif "effect" in ep and ep.get("effect") is not None:
        return likelihood_from_effect(ep["effect"], ep["ci_lower"], ep["ci_upper"])
    elif "mean_diff" in ep and ep.get("mean_diff") is not None:
        return likelihood_from_mean_diff(ep["mean_diff"], ep["se"])
    else:
        raise ValueError(
            f"No likelihood data for endpoint: {ep.get('name', 'unknown')}")


# ═══════════════════════════════════════════════════════════════
# SECTION 6: MARKDOWN REPORT GENERATION
# ═══════════════════════════════════════════════════════════════

def format_outcome_report(result):
    """Format a single outcome analysis into Markdown."""
    lines = []
    freq = result["frequentist"]
    effect_label = "HR" if result["outcome_type"] == "time_to_event" else "OR"
    if result["outcome_type"] == "continuous":
        effect_label = "MD"

    lines.append(f"### {result['outcome_name']}")
    lines.append(f"- **Type:** {result['outcome_type']}")
    lines.append(
        f"- **Frequentist result:** {effect_label} = {freq['or']:.2f} "
        f"(95% CI: {freq['ci_lower']:.2f} \u2013 {freq['ci_upper']:.2f})")
    lines.append(
        f"- **Likelihood:** log({effect_label}) = {freq['log_or']:.3f}, "
        f"SE = {freq['se']:.3f}")
    lines.append("")

    # Posterior table
    lines.append(
        f"| Prior | Mean (SD) | Posterior {effect_label} (95% CrI) | "
        f"P(Benefit) | P(Harm) | P(Severe Harm) | P(Notable Benefit) | ROPE |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for i, post in enumerate(result["posteriors"]):
        pr = result["priors_used"][i]
        lines.append(
            f"| {post['prior_name']} "
            f"| {pr['mean']:.3f} ({pr['sd']:.3f}) "
            f"| {post['post_or']:.2f} ({post['cri_lower']:.2f} \u2013 {post['cri_upper']:.2f}) "
            f"| {post['p_benefit']:.3f} "
            f"| {post['p_harm']:.3f} "
            f"| {post['p_severe_harm']:.3f} "
            f"| {post['p_notable_benefit']:.3f} "
            f"| {post['p_rope']:.3f} |")

    lines.append("")
    lines.append(f"**Prior sensitivity:** {result['sensitivity']['interpretation']}")
    lines.append(
        f"(Max posterior OR difference across priors: "
        f"{result['sensitivity']['max_or_difference']:.3f})")
    lines.append("")
    lines.append(
        f"**Interpretation (skeptical prior):** {result.get('interpretation', 'N/A')}")
    lines.append("")

    # Prior rationales (collapsible)
    lines.append("<details><summary>Prior rationales</summary>")
    lines.append("")
    for pr in result["priors_used"]:
        lines.append(f"- **{pr['name']}:** {pr['rationale']}")
    lines.append("")
    lines.append("</details>")
    lines.append("")

    if result.get("ni_conversion_note"):
        lines.append(f"> **NI margin note:** {result['ni_conversion_note']}")
        lines.append("")

    # Probability evidence ladder (skeptical prior)
    skeptic_post = [p for p in result["posteriors"] if "Skeptical" in p["prior_name"]]
    if skeptic_post and "probability_ladder" in skeptic_post[0]:
        lines.append("#### Probability Evidence Ladder (Skeptical Prior)")
        lines.append("")
        lines.append("| | Probability | Evidence For | Threshold |")
        lines.append("|---|---|---|---|")
        for item in skeptic_post[0]["probability_ladder"]:
            thr = item["threshold_or"]
            if isinstance(thr, float):
                thr_str = f"OR {'<' if 'benefit' in item['description'].lower() or 'Non-inf' in item['description'] else '>'} {thr:.2f}"
            else:
                thr_str = f"OR in ({thr})"
            lines.append(
                f"| **{item['label']}** "
                f"| {item['probability']:.3f} "
                f"| {item['description']} "
                f"| {thr_str} |")
        lines.append("")

        # All priors ladder (collapsible)
        lines.append("<details><summary>Probability ladder for all priors</summary>")
        lines.append("")
        for post in result["posteriors"]:
            if "probability_ladder" in post:
                lines.append(f"**{post['prior_name']}:**")
                lines.append("")
                lines.append("| | Probability | Evidence For |")
                lines.append("|---|---|---|")
                for item in post["probability_ladder"]:
                    lines.append(
                        f"| {item['label']} "
                        f"| {item['probability']:.3f} "
                        f"| {item['description']} |")
                lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines)


def generate_full_report(study_info, outcomes):
    """Generate the complete Bayesian re-analysis Markdown report."""
    lines = []
    lines.append("# Bayesian Re-Analysis Report")
    lines.append("")
    lines.append(f"**Study:** {study_info.get('title', 'N/A')}")
    lines.append(f"**Authors:** {study_info.get('authors', 'N/A')}")
    lines.append(f"**Year:** {study_info.get('year', 'N/A')}")
    lines.append(f"**Design:** {study_info.get('design', 'RCT')}")
    lines.append(
        f"**Sample size:** N = {study_info.get('n_total', 'N/A')} "
        f"(Intervention: {study_info.get('n_intervention', 'N/A')}, "
        f"Control: {study_info.get('n_control', 'N/A')})")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Prior Strategy")
    lines.append(f"- **Evidence level:** {study_info.get('evidence_level', 'N/A')}")
    exp_or = study_info.get("expected_or")
    if exp_or:
        lines.append(
            f"- **Expected effect size:** OR = {exp_or:.2f} (from power calculation)")
    else:
        lines.append(
            "- **Expected effect size:** default assumption used "
            "(no power calculation reported)")
    lines.append(
        "- **Framework:** Zampieri et al., "
        "*Am J Respir Crit Care Med* 2021;203(5):543-552")
    lines.append("")
    lines.append("---")
    lines.append("")

    primary = [o for o in outcomes if o.get("_is_primary", False)]
    secondary = [o for o in outcomes if not o.get("_is_primary", False)]

    if primary:
        lines.append("## Primary Endpoint")
        lines.append("")
        for o in primary:
            lines.append(format_outcome_report(o))

    if secondary:
        lines.append("## Secondary Endpoints")
        lines.append("")
        for o in secondary:
            lines.append(format_outcome_report(o))

    if not primary and not secondary:
        lines.append("## Outcomes")
        lines.append("")
        for o in outcomes:
            lines.append(format_outcome_report(o))

    lines.append("---")
    lines.append("")
    lines.append("## Overall Assessment")
    lines.append("")
    for o in outcomes:
        interp = o.get("interpretation", "N/A")
        sens = o["sensitivity"]["level"]
        lines.append(f"- **{o['outcome_name']}:** {interp} [sensitivity: {sens}]")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*Analysis performed using normal-normal conjugate Bayesian updating. "
        "Prior family follows the minimum set recommended by Zampieri et al. (2021). "
        "ROPE default: OR 0.91-1.10. Severe harm threshold: OR > 1.25. "
        "Notable benefit threshold: OR < 0.80.*")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# SECTION 7: MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_bayesian_reanalysis(study_info, endpoint_list, output_dir="figures"):
    """
    Main entry point for Bayesian re-analysis.

    Parameters
    ----------
    study_info : dict — title, authors, year, design, n_*, expected_or, evidence_level
    endpoint_list : list of dicts — each with name, type, is_primary, likelihood data
    output_dir : str — directory for threshold curve PNGs

    Returns
    -------
    tuple: (markdown_report: str, figure_paths: list, all_outcomes: list)
    """
    priors = build_priors(
        expected_or=study_info.get("expected_or"),
        evidence_level=study_info.get("evidence_level", "moderate"),
        outcome_type="binary"
    )

    all_outcomes = []

    for ep in endpoint_list:
        lik = _extract_likelihood(ep)
        rope = ep.get("rope_range", (1/1.1, 1.1))
        severe = ep.get("severe_harm_threshold", 1.25)
        notable = ep.get("notable_benefit_threshold", 0.80)
        direction = ep.get("direction", "benefit_below")

        result = analyze_outcome(
            outcome_name=ep["name"],
            outcome_type=ep["type"],
            likelihood=lik,
            priors=priors,
            rope_range=rope,
            severe_harm_threshold=severe,
            notable_benefit_threshold=notable,
            ni_margin=ep.get("ni_margin"),
            ni_margin_absolute=ep.get("ni_margin_absolute"),
            active_control_event_rate=ep.get("active_control_event_rate"),
            direction=direction,
        )
        result["_is_primary"] = ep.get("is_primary", False)
        all_outcomes.append(result)

        # If composite, analyze each component
        if ep.get("is_composite") and ep.get("components"):
            for comp in ep["components"]:
                comp_lik = _extract_likelihood(comp)
                comp_result = analyze_outcome(
                    outcome_name=f"{ep['name']} \u2192 {comp['name']}",
                    outcome_type=comp.get("type", "binary"),
                    likelihood=comp_lik,
                    priors=priors,
                    rope_range=rope,
                    severe_harm_threshold=severe,
                    notable_benefit_threshold=notable,
                    ni_margin=ep.get("ni_margin"),
                    ni_margin_absolute=ep.get("ni_margin_absolute"),
                    active_control_event_rate=ep.get("active_control_event_rate"),
                    direction=direction,
                )
                comp_result["_is_primary"] = True
                all_outcomes.append(comp_result)

    # Generate figures
    figure_paths = _generate_all_figures(all_outcomes, output_dir=output_dir)

    report = generate_full_report(study_info, all_outcomes)
    return report, figure_paths, all_outcomes


def _generate_all_figures(all_outcomes, output_dir="figures"):
    """Generate threshold curve figures for all outcomes."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for result in all_outcomes:
        safe_name = result["outcome_name"].replace(" ", "_").replace("/", "-")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
        save_path = os.path.join(output_dir, f"threshold_curve_{safe_name}.png")
        direction = result.get("direction", "benefit_below")
        params = result.get("multi_prior_curve_params", [])

        if params:
            try:
                plot_multi_prior_threshold_curve(
                    post_params_list=params,
                    outcome_name=result["outcome_name"],
                    direction=direction,
                    save_path=save_path,
                )
                paths.append(save_path)
            except Exception as e:
                print(f"[BAYESIAN] Figure generation failed for {result['outcome_name']}: {e}")

    return paths
