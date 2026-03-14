"""
SELUTION DeNovo — Threshold Curve for TVF (Primary Endpoint)
Beta(1,1) prior → Monte Carlo posterior for RD
P(RD > δ | data) as a function of threshold δ
"""
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

random.seed(42)

NI_MARGIN = 0.0244
N_SIM = 500_000

populations = {
    "ITT": {
        "n_sel": 1661, "n_des": 1662,
        "e_sel": 88, "e_des": 73,
        "label": "ITT Population (FAS)",
        "color": "#2563EB",
    },
    "PP": {
        "n_sel": 1592, "n_des": 1602,
        "e_sel": 83, "e_des": 65,
        "label": "Per-Protocol Population",
        "color": "#DC2626",
    },
}

# ── Generate posterior RD samples for each population ──
rd_data = {}
for pop_key, pop in populations.items():
    e1, n1 = pop["e_sel"], pop["n_sel"]
    e2, n2 = pop["e_des"], pop["n_des"]
    a_sel, b_sel = 1 + e1, 1 + n1 - e1
    a_des, b_des = 1 + e2, 1 + n2 - e2

    samples = []
    for _ in range(N_SIM):
        p_s = random.betavariate(a_sel, b_sel)
        p_d = random.betavariate(a_des, b_des)
        samples.append(p_s - p_d)
    samples.sort()
    rd_data[pop_key] = samples

# ── Threshold range ──
thresholds = np.linspace(-0.03, 0.06, 500)

# ── Compute P(RD > δ) for each threshold ──
fig, ax = plt.subplots(figsize=(10, 7))

for pop_key, pop in populations.items():
    samples = rd_data[pop_key]
    n = len(samples)
    probs = []
    for d in thresholds:
        # P(RD > δ) = fraction of samples above δ
        count = n - np.searchsorted(samples, d, side="right")
        probs.append(count / n)

    ax.plot(thresholds, probs, color=pop["color"], linewidth=2.5,
            label=pop["label"])

    # Mark the point at NI margin
    p_at_ni = probs[np.argmin(np.abs(thresholds - NI_MARGIN))]
    ax.plot(NI_MARGIN, p_at_ni, "o", color=pop["color"], markersize=8, zorder=5)
    ax.annotate(f"{p_at_ni*100:.2f}%",
                xy=(NI_MARGIN, p_at_ni),
                xytext=(NI_MARGIN + 0.006, p_at_ni + 0.03),
                fontsize=10, fontweight="bold", color=pop["color"],
                arrowprops=dict(arrowstyle="-", color=pop["color"], lw=1.2))

# ── Vertical lines ──
ax.axvline(x=0, color="#6B7280", linestyle="-", linewidth=1.2, alpha=0.5,
           label="RD = 0")
ax.axvline(x=NI_MARGIN, color="#059669", linestyle="--", linewidth=2.0, alpha=0.8,
           label=f"NI margin = {NI_MARGIN*100:.2f}%")

# ── Horizontal line at α = 0.025 ──
ax.axhline(y=0.025, color="#9333EA", linestyle=":", linewidth=1.5, alpha=0.7,
           label="α = 0.025")

# ── Formatting ──
ax.set_xlabel("Threshold δ (Risk Difference)", fontsize=12)
ax.set_ylabel("P(RD > δ | data)", fontsize=12)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax.set_xlim(-0.03, 0.06)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.15)
ax.legend(fontsize=10, loc="upper right", framealpha=0.9, edgecolor="#D1D5DB")

ax.set_title("SELUTION DeNovo — Threshold Curve: P(RD > δ | data)\n"
             "Prior: Beta(1,1)  |  Monte Carlo (500K draws)",
             fontsize=13, fontweight="bold")

plt.tight_layout()
save_path = "/Users/apple/Desktop/selution/figures/tvf_threshold_curve.png"
fig.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {save_path}")
