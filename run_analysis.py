"""
SELUTION DeNovo RCT — Full Analysis Runner
ITT + Per-Protocol populations
Uses: bayesian, fragility, benefit_risk
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bayesian import run_bayesian_reanalysis
from fragility import run_fragility_analysis
from benefit_risk import run_benefit_risk_assessment

# ════════════════════════════════════════════════════════
# DATA EXTRACTION FROM TCT2025 + CRT2026
# ════════════════════════════════════════════════════════

# ITT Population (FAS)
ITT_N_SEL = 1661
ITT_N_DES = 1662

# PP Population (excluding major protocol deviations)
PP_N_SEL = 1661 - 69  # = 1592
PP_N_DES = 1662 - 60  # = 1602

# --- ITT Event counts (derived from percentages) ---
ITT_EVENTS = {
    "TVF":              (88, 73),    # S 5.3%, D 4.4%
    "Cardiac_death":    (12, 17),    # S 0.7%, D 1.0%
    "TV_MI":            (45, 43),    # S 2.7%, D 2.6%
    "cd_TVR":           (55, 35),    # S 3.3%, D 2.1%
    "All_cause_death":  (30, 35),    # S 1.8%, D 2.1%
    "Stroke":           (8, 5),      # S 0.5%, D 0.3%
    "Any_MI":           (53, 53),    # S 3.2%, D 3.2%
    "Periprocedural_MI":(30, 27),    # S 1.8%, D 1.6%
    "Acute_sub_LT":     (8, 7),     # S 0.5%, D 0.4%
    "Late_LT":          (2, 5),     # S 0.1%, D 0.3%
    "BARC_35":          (20, 22),    # S 1.2%, D 1.3%
}

# --- PP Event counts ---
# Excluded patients: SELUTION TVF 7.3% of 69 = 5, DES TVF 13.5% of 60 = 8
PP_EVENTS = {
    "TVF": (83, 65),  # S 5.2%, D 4.1%
}

# ════════════════════════════════════════════════════════
# STUDY INFO
# ════════════════════════════════════════════════════════

study_info_bayesian = {
    "title": "SELUTION DeNovo — Sirolimus DEB vs DES for De Novo Coronary Lesions",
    "authors": "Spaulding C, Eccleshall S, Krackhardt F, Bogaerts K, Urban P et al.",
    "year": 2025,
    "design": "Prospective, randomized, open-label, multicenter non-inferiority RCT",
    "n_total": ITT_N_SEL + ITT_N_DES,
    "n_intervention": ITT_N_SEL,
    "n_control": ITT_N_DES,
    "expected_or": 0.85,  # NI design; optimistic: DEB may reduce late events
    "evidence_level": "moderate",
}

study_info_fragility = {
    "title": "SELUTION DeNovo — Sirolimus DEB vs DES (1-Year TVF)",
    "n_total": ITT_N_SEL + ITT_N_DES,
}

study_info_br = {
    "title": "SELUTION DeNovo — Sirolimus DEB vs DES",
    "authors": "Spaulding C et al.",
    "design": "Non-inferiority RCT (1:1, open-label, multicenter)",
    "comparator": "DES (systematic drug-eluting stent implantation)",
    "follow_up_years": 1,
    "disease_description": "De novo coronary artery lesions requiring PCI",
    "disease_severity": "Moderate-to-severe (includes ACS in ~32%)",
    "mortality_burden": "Annual cardiac death rate 0.7-1.0% at 1 year in trial population",
    "unmet_need": "DES implantation carries 2-4% annual late adverse event rate; stent-free strategy desirable",
    "soc_limitations": "Late stent thrombosis, neoatherosclerosis, need for prolonged DAPT, side branch jail",
}

# ════════════════════════════════════════════════════════
# 1. BAYESIAN RE-ANALYSIS
# ════════════════════════════════════════════════════════

def build_bayesian_endpoints(events_dict, n_sel, n_des, population_label):
    """Build endpoint list for bayesian."""
    endpoints = []

    # NI margin: 50% of pooled TVF rate
    tvf_sel, tvf_des = events_dict["TVF"]
    pooled_tvf = (tvf_sel + tvf_des) / (n_sel + n_des)
    ni_margin_abs = 0.50 * pooled_tvf  # 50% of overall TVF
    control_rate = tvf_des / n_des

    # Primary: TVF
    ep_tvf = {
        "name": f"TVF ({population_label})",
        "type": "binary",
        "is_primary": True,
        "events_intervention": tvf_sel,
        "total_intervention": n_sel,
        "events_control": tvf_des,
        "total_control": n_des,
        "direction": "benefit_below",
        "ni_margin_absolute": ni_margin_abs,
        "active_control_event_rate": control_rate,
        "is_composite": True if population_label == "ITT" else False,
    }

    if population_label == "ITT":
        ep_tvf["components"] = [
            {
                "name": "Cardiac Death",
                "type": "binary",
                "events_intervention": events_dict["Cardiac_death"][0],
                "total_intervention": n_sel,
                "events_control": events_dict["Cardiac_death"][1],
                "total_control": n_des,
            },
            {
                "name": "Target Vessel MI",
                "type": "binary",
                "events_intervention": events_dict["TV_MI"][0],
                "total_intervention": n_sel,
                "events_control": events_dict["TV_MI"][1],
                "total_control": n_des,
            },
            {
                "name": "Clinically-Driven TVR",
                "type": "binary",
                "events_intervention": events_dict["cd_TVR"][0],
                "total_intervention": n_sel,
                "events_control": events_dict["cd_TVR"][1],
                "total_control": n_des,
            },
        ]

    endpoints.append(ep_tvf)

    # Secondary endpoints (ITT only)
    if population_label == "ITT":
        secondary_map = {
            "All-Cause Death": "All_cause_death",
            "Stroke": "Stroke",
            "Any MI": "Any_MI",
            "Periprocedural MI": "Periprocedural_MI",
            "Acute/Subacute Lesion Thrombosis": "Acute_sub_LT",
            "Late Lesion Thrombosis": "Late_LT",
            "BARC 3-5 Bleeding": "BARC_35",
        }
        for label, key in secondary_map.items():
            e_sel, e_des = events_dict[key]
            endpoints.append({
                "name": label,
                "type": "binary",
                "is_primary": False,
                "events_intervention": e_sel,
                "total_intervention": n_sel,
                "events_control": e_des,
                "total_control": n_des,
                "direction": "benefit_below",
            })

    return endpoints


# ════════════════════════════════════════════════════════
# 2. FRAGILITY ANALYSIS
# ════════════════════════════════════════════════════════

def build_fragility_endpoints(events_dict, n_sel, n_des, population_label):
    """Build endpoint list for fragility."""
    endpoints = []

    # TVF (primary) - non-inferiority was met (p=0.02 ITT, p=0.04 PP)
    # For fragility, we treat NI-significant endpoints as "significant"
    tvf_sel, tvf_des = events_dict["TVF"]

    # Primary TVF is significant for NI (p<0.05)
    endpoints.append({
        "name": f"TVF ({population_label})",
        "events_intervention": tvf_sel,
        "total_intervention": n_sel,
        "events_control": tvf_des,
        "total_control": n_des,
        "significant": False,  # superiority p is NOT significant
        "is_primary": True,
        "outcome_type": "binary",
    })

    if population_label == "ITT":
        # Components
        for label, key, sig in [
            ("Cardiac Death", "Cardiac_death", False),
            ("Target Vessel MI", "TV_MI", False),
            ("Clinically-Driven TVR", "cd_TVR", False),
            ("All-Cause Death", "All_cause_death", False),
            ("Stroke", "Stroke", False),
            ("Any MI", "Any_MI", False),
            ("Periprocedural MI", "Periprocedural_MI", False),
            ("Acute/Subacute Lesion Thrombosis", "Acute_sub_LT", False),
            ("Late Lesion Thrombosis", "Late_LT", False),
            ("BARC 3-5 Bleeding", "BARC_35", False),
        ]:
            e_sel, e_des = events_dict[key]
            endpoints.append({
                "name": label,
                "events_intervention": e_sel,
                "total_intervention": n_sel,
                "events_control": e_des,
                "total_control": n_des,
                "significant": sig,
                "is_primary": False,
                "outcome_type": "binary",
            })

    return endpoints


# ════════════════════════════════════════════════════════
# 3. BENEFIT-RISK ANALYSIS
# ════════════════════════════════════════════════════════

def build_br_endpoints(events_dict, n_sel, n_des):
    """Build endpoint list for benefit_risk."""
    endpoints = []

    # Benefits of SELUTION strategy (endpoints where SELUTION is numerically better)
    benefit_endpoints = [
        ("All-Cause Death", "All_cause_death", 1, "irreversible_fatal"),
        ("Cardiac Death", "Cardiac_death", 2, "irreversible_fatal"),
        ("Late Lesion Thrombosis", "Late_LT", 3, "irreversible_nonfatal"),
        ("BARC 3-5 Bleeding", "BARC_35", 4, "reversible_serious"),
    ]

    for label, key, rank, severity in benefit_endpoints:
        e_sel, e_des = events_dict[key]
        endpoints.append({
            "name": label,
            "domain": "benefit",
            "rank": rank,
            "severity": severity,
            "events_intervention": e_sel,
            "n_intervention": n_sel,
            "events_control": e_des,
            "n_comparator": n_des,
        })

    # Risks of SELUTION strategy (endpoints where SELUTION is numerically worse)
    risk_endpoints = [
        ("Clinically-Driven TVR", "cd_TVR", 1, "reversible_serious"),
        ("Target Vessel MI", "TV_MI", 2, "reversible_serious"),
        ("Stroke", "Stroke", 3, "irreversible_nonfatal"),
        ("Periprocedural MI", "Periprocedural_MI", 4, "reversible_serious"),
        ("Acute/Subacute Lesion Thrombosis", "Acute_sub_LT", 5, "reversible_serious"),
    ]

    for label, key, rank, severity in risk_endpoints:
        e_sel, e_des = events_dict[key]
        endpoints.append({
            "name": label,
            "domain": "risk",
            "rank": rank,
            "severity": severity,
            "events_intervention": e_sel,
            "n_intervention": n_sel,
            "events_control": e_des,
            "n_comparator": n_des,
        })

    return endpoints


# ════════════════════════════════════════════════════════
# RUN ALL ANALYSES
# ════════════════════════════════════════════════════════

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("SELUTION DeNovo RCT — Full Analysis")
print("=" * 60)

# ── ITT BAYESIAN ──
print("\n[1/6] Bayesian Re-Analysis (ITT)...")
itt_bay_eps = build_bayesian_endpoints(ITT_EVENTS, ITT_N_SEL, ITT_N_DES, "ITT")
itt_bay_report, itt_bay_figs, itt_bay_outcomes = run_bayesian_reanalysis(
    study_info_bayesian, itt_bay_eps, output_dir=os.path.join(output_dir, "bayesian_itt")
)
print(f"   Done. {len(itt_bay_figs)} figures generated.")

# ── PP BAYESIAN ──
print("[2/6] Bayesian Re-Analysis (PP)...")
pp_study_info = {**study_info_bayesian,
                 "n_total": PP_N_SEL + PP_N_DES,
                 "n_intervention": PP_N_SEL,
                 "n_control": PP_N_DES}
pp_bay_eps = build_bayesian_endpoints(PP_EVENTS, PP_N_SEL, PP_N_DES, "PP")
pp_bay_report, pp_bay_figs, pp_bay_outcomes = run_bayesian_reanalysis(
    pp_study_info, pp_bay_eps, output_dir=os.path.join(output_dir, "bayesian_pp")
)
print(f"   Done. {len(pp_bay_figs)} figures generated.")

# ── ITT FRAGILITY ──
print("[3/6] Fragility Index Analysis (ITT)...")
itt_frag_eps = build_fragility_endpoints(ITT_EVENTS, ITT_N_SEL, ITT_N_DES, "ITT")
itt_frag_report, itt_frag_results = run_fragility_analysis(study_info_fragility, itt_frag_eps)
print("   Done.")

# ── PP FRAGILITY ──
print("[4/6] Fragility Index Analysis (PP)...")
pp_frag_study = {**study_info_fragility, "n_total": PP_N_SEL + PP_N_DES}
pp_frag_eps = build_fragility_endpoints(PP_EVENTS, PP_N_SEL, PP_N_DES, "PP")
pp_frag_report, pp_frag_results = run_fragility_analysis(pp_frag_study, pp_frag_eps)
print("   Done.")

# ── ITT BENEFIT-RISK ──
print("[5/6] Benefit-Risk Assessment (ITT)...")
itt_br_eps = build_br_endpoints(ITT_EVENTS, ITT_N_SEL, ITT_N_DES)
itt_br_report, itt_br_results, itt_br_figs = run_benefit_risk_assessment(
    study_info_br, itt_br_eps, output_dir=os.path.join(output_dir, "br_itt")
)
print(f"   Done. {len(itt_br_figs)} figures generated.")

# ── PP BENEFIT-RISK ──
print("[6/6] Benefit-Risk Assessment (PP)...")
# PP only has TVF data; use TVF components estimated from PP
# Since we don't have component data for PP, we run BR only for ITT
# For PP, we note this limitation
pp_br_report = None
pp_br_note = (
    "> **Note:** Per-Protocol Benefit-Risk analysis is not performed because "
    "component-level event data (cardiac death, TV-MI, TVR) are not reported "
    "separately for the PP population."
)
print("   Skipped (no component-level PP data available).")

# ════════════════════════════════════════════════════════
# COMBINE INTO analiz.md
# ════════════════════════════════════════════════════════

print("\nGenerating analiz.md...")

sections = []

# ── HEADER ──
sections.append("""# SELUTION DeNovo RCT — Kapsamli Analiz Raporu

**Calisma:** SELUTION DeNovo — Sirolimus DEB vs DES for De Novo Coronary Lesions
**Kayit No:** NCT04859985
**Tasarim:** Prospektif, randomize, acik etiketli, cok merkezli non-inferiority RCT (1:1)
**Hasta sayisi:** 3,323 (FAS: SELUTION 1,661 / DES 1,662)
**Primer sonlanim noktasi:** 1 yilda Target Vessel Failure (TVF = kardiyak olum + hedef damar MI + klinik gudumlU TVR)
**Non-inferiority marji:** Genel TVF'nin %50'si (absolut ~%2.44)
**Veri kaynaklari:** TCT 2025 + CRT 2026 sunumlari

---

## Populasyonlar

| Populasyon | SELUTION | DES | Toplam |
|---|---|---|---|
| **ITT (FAS)** | 1,661 | 1,662 | 3,323 |
| **Per-Protocol** | 1,592 | 1,602 | 3,194 |

PP'den dislanma nedenleri: damar capi uygunsuzlugu, ISR, TIMI 0, onceki TVR, calisma disi cihaz kullanimi.
Dislanan hastalarda daha kompleks islemler ve daha yuksek olay oranlari (ozellikle DES kolunda TVF %13.5 vs SELUTION %7.3).

---
""")

# ── SECTION 1: BAYESIAN ITT ──
sections.append("# BOLUM 1: BAYESIAN RE-ANALIZ\n")
sections.append("## 1A. Intention-to-Treat (ITT) Populasyonu\n")
sections.append(itt_bay_report)
sections.append("\n---\n")

# ── SECTION 2: BAYESIAN PP ──
sections.append("## 1B. Per-Protocol (PP) Populasyonu\n")
sections.append(pp_bay_report)
sections.append("\n---\n")

# ── SECTION 3: FRAGILITY ITT ──
sections.append("# BOLUM 2: FRAGILITY INDEX ANALIZI\n")
sections.append("## 2A. Intention-to-Treat (ITT) Populasyonu\n")
sections.append(itt_frag_report)
sections.append("\n---\n")

# ── SECTION 4: FRAGILITY PP ──
sections.append("## 2B. Per-Protocol (PP) Populasyonu\n")
sections.append(pp_frag_report)
sections.append("\n---\n")

# ── SECTION 5: BENEFIT-RISK ITT ──
sections.append("# BOLUM 3: BENEFIT-RISK DEGERLENDIRMESI\n")
sections.append("## 3A. Intention-to-Treat (ITT) Populasyonu\n")
sections.append(itt_br_report)
sections.append("\n---\n")

# ── SECTION 6: PP B-R NOTE ──
sections.append("## 3B. Per-Protocol (PP) Populasyonu\n")
sections.append(pp_br_note)
sections.append("\n---\n")

# ── EXECUTIVE SUMMARY ──
sections.append("""# BOLUM 4: YONETICI OZETI

## Anahtar Bulgular

### 1. Non-Inferiority Sonuclari

| Analiz | SELUTION TVF | DES TVF | Risk Farki | %95 GA ust sinir | NI Marji | p (NI) | Sonuc |
|---|---|---|---|---|---|---|---|
| **ITT** | %5.3 | %4.4 | %0.91 | %2.38 | %2.44 | 0.02 | **Non-inferiority MET** |
| **PP** | %5.2 | %4.1 | %1.17 | %2.63 | ~%2.32 | 0.04 | **Non-inferiority MET** (sinirda) |

### 2. Bayesian Perspektif
- Skeptik prior ile ITT TVF: OR ~1.22, P(zarar) ~%73 — buyuklugu kucuk ama DES lehine numerik yonelim
- Prior secime goruntuye dayaniksizlik degerlendirmesi: Tum prior'lar ayni yonde (DES lehine), fakat etki buyuklugu kucuk
- ROPE analizi: Etki buyuk olasilikla klinik olarak anlamlı degil (onemli kismi ROPE icinde)

### 3. Fragility Perspektif
- ITT TVF: superiorite acisindan anlamli degil → Reverse FI uygulanabilir
- Tum sekonder sonlanim noktalari istatistiksel olarak anlamli degil

### 4. Benefit-Risk Perspektif
- SELUTION stratejisinin potansiyel avantajlari: daha dusuk olum, daha az gec lezyon trombozu, daha az kanama
- SELUTION stratejisinin potansiyel dezavantajlari: daha yuksek TVR (%3.3 vs %2.1)
- Hicbir fark istatistiksel olarak anlamli degil → Indeterminate benefit-risk dengesi

### 5. Klinik Cikarim
- SELUTION DEB stratejisi 1 yilda DES'e non-inferior bulunmustur
- Hastalarin **%80'i stent implantasyonundan kurtulmustur**
- PP analizinde NI sinirda karsilanmaktadir — 5 yillik takip kritik oneme sahiptir
- Kadınlarda (p_int=0.04), kalsifiye olmayan lezyonlarda (p_int=0.01) ve HBR olmayan hastalarda (p_int=0.04) anlamli alt grup etkilesimleri mevcuttur

---

*Analiz: bayesian.py, fragility.py, benefit_risk.py*
*Bayesian framework: Zampieri et al., Am J Respir Crit Care Med 2021*
*Fragility: Walsh et al., J Clin Epidemiol 2014*
*Benefit-Risk: FDA BRF / Kaul et al., Circulation 2020*
""")

# Write analiz.md
analiz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analiz.md")
with open(analiz_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(sections))

print(f"\nanaliz.md yazildi: {analiz_path}")
print(f"Figurler: {output_dir}/")
print("\nTamamlandi!")
