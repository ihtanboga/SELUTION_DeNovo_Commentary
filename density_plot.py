"""
SELUTION DeNovo — Posterior Density Plot for TVF (Primary Endpoint)
Beta(1,1) prior on each proportion → Monte Carlo for RD posterior
P(inferiority) = P(RD > 2.44%), one-sided α = 0.025
"""
import math
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

random.seed(42)

def beta_sample(a, b):
    return random.betavariate(a, b)

NI_MARGIN = 0.0244
N_SIM = 500_000

populations = {
    "ITT": {
        "n_sel": 1661, "n_des": 1662,
        "e_sel": 88, "e_des": 73,
        "label": "ITT Population (FAS)",
    },
    "PP": {
        "n_sel": 1592, "n_des": 1602,
        "e_sel": 83, "e_des": 65,
        "label": "Per-Protocol Population",
    },
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for idx, (pop_key, pop) in enumerate(populations.items()):
    ax = axes[idx]

    e1, n1 = pop["e_sel"], pop["n_sel"]
    e2, n2 = pop["e_des"], pop["n_des"]

    a_sel, b_sel = 1 + e1, 1 + n1 - e1
    a_des, b_des = 1 + e2, 1 + n2 - e2

    rd_samples = []
    for _ in range(N_SIM):
        p_s = beta_sample(a_sel, b_sel)
        p_d = beta_sample(a_des, b_des)
        rd_samples.append(p_s - p_d)

    rd_samples.sort()

    rd_mean = sum(rd_samples) / N_SIM
    n_inf = sum(1 for r in rd_samples if r > NI_MARGIN)
    p_inf = n_inf / N_SIM
    p_ni = 1.0 - p_inf
    n_harm = sum(1 for r in rd_samples if r > 0)
    p_harm = n_harm / N_SIM
    p_sup = 1.0 - p_harm
    cri_lo = rd_samples[int(N_SIM * 0.025)]
    cri_hi = rd_samples[int(N_SIM * 0.975)]

    # ── Histogram bins ──
    n_bins = 250
    bin_min = min(rd_samples[0], -0.025)
    bin_max = max(rd_samples[-1], 0.050)
    bin_width = (bin_max - bin_min) / n_bins
    bins = [bin_min + i * bin_width for i in range(n_bins + 1)]

    # ── Compute density values per bin ──
    density_x = []
    density_y = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        c = sum(1 for r in rd_samples if lo <= r < hi)
        density_x.append((lo + hi) / 2)
        density_y.append(c / (N_SIM * bin_width))

    y_max = max(density_y)

    # ── Fill: non-inferior region (blue, under curve) ──
    ni_x = [x for x, y in zip(density_x, density_y) if x <= NI_MARGIN]
    ni_y = [y for x, y in zip(density_x, density_y) if x <= NI_MARGIN]
    ax.fill_between(ni_x, ni_y, alpha=0.25, color="#2563EB")

    # ── Fill: inferior region (red, under curve only) ──
    inf_x = [x for x, y in zip(density_x, density_y) if x >= NI_MARGIN]
    inf_y = [y for x, y in zip(density_x, density_y) if x >= NI_MARGIN]
    ax.fill_between(inf_x, inf_y, alpha=0.50, color="#DC2626")

    # ── Density curve ──
    ax.plot(density_x, density_y, color="#1a1a2e", linewidth=2.0)

    # ── NI margin line ──
    ax.axvline(x=NI_MARGIN, color="#DC2626", linestyle="--", linewidth=2.0, alpha=0.85)
    ax.text(NI_MARGIN + 0.0005, y_max * 0.60, "NI margin\n2.44%",
            color="#DC2626", fontsize=10, fontweight="bold", ha="left")

    # ── RD = 0 line ──
    ax.axvline(x=0, color="#6B7280", linestyle="-", linewidth=1.0, alpha=0.5)

    # ── Posterior mean line ──
    ax.axvline(x=rd_mean, color="#2563EB", linestyle=":", linewidth=1.8, alpha=0.7)
    ax.text(rd_mean, y_max * 1.08, f"Posterior RD\n{rd_mean*100:.2f}%",
            color="#2563EB", fontsize=10, fontweight="bold", ha="center")

    # ── Color legend squares ──
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2563EB", alpha=0.25, label=f"P(non-inferior) = {p_ni*100:.1f}%"),
        Patch(facecolor="#DC2626", alpha=0.50, label=f"P(inferiority) = {p_inf*100:.2f}%"),
    ]
    ax.legend(handles=legend_elements, fontsize=9.5, loc="lower right",
              bbox_to_anchor=(0.98, 0.02), framealpha=0.9, edgecolor="#D1D5DB")

    # ── Verdict box (mid-height) ──
    ni_met = p_inf < 0.025
    if ni_met:
        verdict_text = f"P(inferiority) = {p_inf*100:.2f}% < 2.5%\nNon-inferiority MET  \u2714"
        verdict_color = "#059669"
    else:
        verdict_text = f"P(inferiority) = {p_inf*100:.2f}% > 2.5%\nNon-inferiority NOT MET  \u2718"
        verdict_color = "#DC2626"

    ax.text(0.50, 0.72, verdict_text, transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=verdict_color,
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=verdict_color, linewidth=1.5, alpha=0.95))

    # ── Stats box (top left) ──
    stats_text = (
        f"SELUTION: {e1}/{n1} ({e1/n1*100:.2f}%)\n"
        f"DES: {e2}/{n2} ({e2/n2*100:.2f}%)\n"
        f"Prior: Beta(1,1) on each arm\n"
        f"Posterior RD: {rd_mean*100:.2f}% "
        f"[{cri_lo*100:.2f}, {cri_hi*100:.2f}]"
    )
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            fontsize=8.5, color="#374151", va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3F4F6",
                      edgecolor="#D1D5DB", alpha=0.85))

    # ── Formatting ──
    ax.set_title(f"{pop['label']}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Risk Difference (SELUTION \u2212 DES)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.set_ylim(0, y_max * 1.20)
    ax.grid(True, alpha=0.12)

    if idx == 0:
        ax.set_ylabel("Posterior Density", fontsize=11)

fig.suptitle("SELUTION DeNovo \u2014 Primary Endpoint (TVF) Posterior Distribution\n"
             "Prior: Beta(1,1)  |  NI margin: 2.44% absolute RD  |  "
             "One-sided \u03b1 = 0.025  |  Monte Carlo (500K draws)",
             fontsize=13, fontweight="bold", y=1.01)

plt.tight_layout(rect=[0, 0, 1, 0.93])

save_path = "/Users/apple/Desktop/selution/figures/tvf_posterior_density.png"
fig.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {save_path}")
