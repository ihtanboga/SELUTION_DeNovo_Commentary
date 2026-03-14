"""
SELUTION DeNovo — Fragility Index for Non-Inferiority
RD scale, one-sided α = 0.025, NI margin = 2.44%

1) ITT (NI met): Kaç ek SELUTION olayı NI'yi bozar?
2) PP (NI not met): Kaç SELUTION olayı azalsa / DES olayı artsa NI karşılanırdı?
"""
import math

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def p_inferiority(e_sel, n_sel, e_des, n_des, ni_margin):
    p1 = e_sel / n_sel
    p2 = e_des / n_des
    rd = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n_sel + p2 * (1 - p2) / n_des)
    z = (ni_margin - rd) / se
    return 1.0 - norm_cdf(z), rd, se

NI_MARGIN = 0.0244
ALPHA = 0.025

print("=" * 75)
print("FRAGILITY INDEX FOR NON-INFERIORITY")
print(f"NI margin: {NI_MARGIN*100:.2f}%  |  α: {ALPHA}  |  RD scale")
print("=" * 75)

# ═══════════════════════════════════════════════════════
# 1) ITT: NI MET → kaç ek SELUTION olayı NI'yi bozar?
# ═══════════════════════════════════════════════════════

print("\n" + "─" * 75)
print("  ITT: Non-inferiority MET → Kaç ek SELUTION olayı ile NI bozulur?")
print("─" * 75)

e_sel, n_sel = 88, 1661
e_des, n_des = 73, 1662

p_orig, rd_orig, se_orig = p_inferiority(e_sel, n_sel, e_des, n_des, NI_MARGIN)
print(f"\n  Baslangic: SELUTION {e_sel}/{n_sel}, DES {e_des}/{n_des}")
print(f"  RD = {rd_orig*100:.3f}%, P(inf) = {p_orig:.4f} ({p_orig*100:.2f}%)")
print(f"  NI durumu: {'MET' if p_orig < ALPHA else 'NOT MET'}")
print()

fi = 0
current_e_sel = e_sel
print(f"  {'Adim':<6} {'SELUTION olaylari':<20} {'RD':<12} {'P(inf)':<15} {'NI?'}")
print(f"  {'─'*6} {'─'*20} {'─'*12} {'─'*15} {'─'*10}")

while fi < 200:
    current_e_sel += 1
    fi += 1
    p_new, rd_new, se_new = p_inferiority(current_e_sel, n_sel, e_des, n_des, NI_MARGIN)

    ni_status = "MET" if p_new < ALPHA else "BOZULDU!"
    print(f"  +{fi:<5} {current_e_sel}/{n_sel:<17} {rd_new*100:.3f}%      {p_new:.4f} ({p_new*100:.2f}%)  {ni_status}")

    if p_new >= ALPHA:
        break

print(f"\n  *** ITT FRAGILITY INDEX (NI) = {fi} ***")
print(f"  SELUTION kolunda {fi} ek olay ({e_sel} → {current_e_sel}) NI'yi bozar.")
print(f"  Bu {fi}/{n_sel+n_des} = {fi/(n_sel+n_des)*100:.3f}% toplam orneklemin (FQ)")
print(f"  Orijinal: RD={rd_orig*100:.3f}%, P(inf)={p_orig:.4f}")
print(f"  Final:    RD={rd_new*100:.3f}%, P(inf)={p_new:.4f}")

# Tersten de dene: DES'te olay azalırsa
print(f"\n  --- Alternatif: DES kolundan olay çıkarılırsa ---")
fi_des = 0
current_e_des = e_des
while fi_des < 200:
    current_e_des -= 1
    if current_e_des < 0:
        break
    fi_des += 1
    p_new2, rd_new2, _ = p_inferiority(e_sel, n_sel, current_e_des, n_des, NI_MARGIN)
    if p_new2 >= ALPHA:
        break

print(f"  DES kolunda {fi_des} olay azalsa ({e_des} → {current_e_des}) NI bozulur.")
print(f"  Final: RD={rd_new2*100:.3f}%, P(inf)={p_new2:.4f}")

# ═══════════════════════════════════════════════════════
# 2) PP: NI NOT MET → kaç olay değişikliği ile NI karşılanırdı?
# ═══════════════════════════════════════════════════════

print("\n" + "─" * 75)
print("  PP: Non-inferiority KARSILANMADI → Kaç olay değişikliği ile NI sağlanır?")
print("─" * 75)

e_sel_pp, n_sel_pp = 83, 1592
e_des_pp, n_des_pp = 65, 1602

p_orig_pp, rd_orig_pp, se_orig_pp = p_inferiority(e_sel_pp, n_sel_pp, e_des_pp, n_des_pp, NI_MARGIN)
print(f"\n  Baslangic: SELUTION {e_sel_pp}/{n_sel_pp}, DES {e_des_pp}/{n_des_pp}")
print(f"  RD = {rd_orig_pp*100:.3f}%, P(inf) = {p_orig_pp:.4f} ({p_orig_pp*100:.2f}%)")
print(f"  NI durumu: {'MET' if p_orig_pp < ALPHA else 'NOT MET'}")

# 2a) SELUTION'dan olay çıkar
print(f"\n  Senaryo A: SELUTION kolundan olay çıkarılırsa")
print(f"  {'Adim':<6} {'SELUTION olaylari':<20} {'RD':<12} {'P(inf)':<15} {'NI?'}")
print(f"  {'─'*6} {'─'*20} {'─'*12} {'─'*15} {'─'*10}")

rfi_a = 0
current_e_sel_pp = e_sel_pp
while rfi_a < 200:
    current_e_sel_pp -= 1
    if current_e_sel_pp < 0:
        break
    rfi_a += 1
    p_new_a, rd_new_a, _ = p_inferiority(current_e_sel_pp, n_sel_pp, e_des_pp, n_des_pp, NI_MARGIN)

    ni_status = "MET!" if p_new_a < ALPHA else "NOT MET"
    print(f"  -{rfi_a:<5} {current_e_sel_pp}/{n_sel_pp:<17} {rd_new_a*100:.3f}%      {p_new_a:.4f} ({p_new_a*100:.2f}%)  {ni_status}")

    if p_new_a < ALPHA:
        break

print(f"\n  *** PP REVERSE FRAGILITY (SELUTION olay azaltma) = {rfi_a} ***")
print(f"  SELUTION'da {rfi_a} olay az olsaydı ({e_sel_pp} → {current_e_sel_pp}) NI karşılanırdı.")

# 2b) DES'e olay ekle
print(f"\n  Senaryo B: DES koluna olay eklenirse")
print(f"  {'Adim':<6} {'DES olaylari':<20} {'RD':<12} {'P(inf)':<15} {'NI?'}")
print(f"  {'─'*6} {'─'*20} {'─'*12} {'─'*15} {'─'*10}")

rfi_b = 0
current_e_des_pp = e_des_pp
while rfi_b < 200:
    current_e_des_pp += 1
    rfi_b += 1
    p_new_b, rd_new_b, _ = p_inferiority(e_sel_pp, n_sel_pp, current_e_des_pp, n_des_pp, NI_MARGIN)

    ni_status = "MET!" if p_new_b < ALPHA else "NOT MET"
    print(f"  +{rfi_b:<5} {current_e_des_pp}/{n_des_pp:<17} {rd_new_b*100:.3f}%      {p_new_b:.4f} ({p_new_b*100:.2f}%)  {ni_status}")

    if p_new_b < ALPHA:
        break

print(f"\n  *** PP REVERSE FRAGILITY (DES olay artırma) = {rfi_b} ***")
print(f"  DES'te {rfi_b} olay fazla olsaydı ({e_des_pp} → {current_e_des_pp}) NI karşılanırdı.")

# ═══════════════════════════════════════════════════════
# ÖZET
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("ÖZET")
print("=" * 75)
print(f"""
  ITT (NI met):
    → SELUTION'a +{fi} olay eklenirse NI bozulur (FI = {fi})
    → veya DES'ten {fi_des} olay çıkarılırsa NI bozulur

  PP (NI not met):
    → SELUTION'dan {rfi_a} olay çıkarılırsa NI sağlanırdı
    → veya DES'e +{rfi_b} olay eklenirse NI sağlanırdı

  Yorum:
    ITT'de NI kararı {fi} olay değişikliğine dayanıyor.
    PP'de NI'ya sadece {rfi_a} olay uzak — oldukça sınırda.
""")
