#!/usr/bin/env python3
import os
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import fisher_exact

ROOT = Path(os.environ.get("DBIC_DATA_ROOT", "data/AD715"))

master_file = ROOT / "6_single_cell/07_multimodal_master/tables/AD715_master_multimodal.tsv"

outdir = ROOT / "6_single_cell/14_Morph_RNA_state_coupling"
tabdir = outdir / "tables"
figdir = outdir / "figures"
tabdir.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)

master = pd.read_csv(master_file, sep="\t")

if "Morph_state" not in master.columns:
    raise ValueError("Morph_state column not found in master table")
if "RNA_state" not in master.columns:
    raise ValueError("RNA_state column not found in master table")

master["Morph_state"] = master["Morph_state"].astype(str)
master["RNA_state_label"] = "RNA" + master["RNA_state"].astype(str)

morph_order = sorted(
    master["Morph_state"].dropna().unique(),
    key=lambda x: int(x.replace("Morph", "")) if x.startswith("Morph") else x
)

rna_order = sorted(
    master["RNA_state_label"].dropna().unique(),
    key=lambda x: int(x.replace("RNA", "")) if x.startswith("RNA") else x
)

morph_name = {
    "Morph0": "Compact",
    "Morph1": "Stress-fiber enriched",
    "Morph2": "Cortical-actin",
    "Morph3": "Nuclear remodeling",
    "Morph4": "Polarized nuclear-remodeling",
    "Morph5": "Highly elongated",
}

rna_name = {
    "RNA0": "RNA0",
    "RNA1": "RNA1",
    "RNA2": "RNA2",
    "RNA3": "RNA3",
}

rna_colors = {
    "RNA0": "#C9A51D",
    "RNA1": "#61C474",
    "RNA2": "#E61E86",
    "RNA3": "#3498DB",
}

# ------------------------------------------------------------
# 1. Morph x RNA composition
# ------------------------------------------------------------
comp = (
    master
    .groupby(["Morph_state", "RNA_state_label"])
    .size()
    .reset_index(name="n")
)

comp["fraction_within_Morph"] = (
    comp["n"] / comp.groupby("Morph_state")["n"].transform("sum")
)

comp["fraction_within_RNA"] = (
    comp["n"] / comp.groupby("RNA_state_label")["n"].transform("sum")
)

comp.to_csv(
    tabdir / "AD715_48a_Morph_RNA_composition.tsv",
    sep="\t",
    index=False
)

count_mat = (
    comp
    .pivot(index="Morph_state", columns="RNA_state_label", values="n")
    .fillna(0)
    .reindex(index=morph_order, columns=rna_order)
    .fillna(0)
)

frac_mat = (
    comp
    .pivot(index="Morph_state", columns="RNA_state_label", values="fraction_within_Morph")
    .fillna(0)
    .reindex(index=morph_order, columns=rna_order)
    .fillna(0)
)

count_mat.to_csv(tabdir / "AD715_48a_Morph_RNA_count_matrix.tsv", sep="\t")
frac_mat.to_csv(tabdir / "AD715_48a_Morph_RNA_fraction_matrix.tsv", sep="\t")

# ------------------------------------------------------------
# 2. Fisher enrichment Morph x RNA
# ------------------------------------------------------------
records = []

for morph in morph_order:
    for rna in rna_order:
        a = int(((master["Morph_state"] == morph) & (master["RNA_state_label"] == rna)).sum())
        b = int(((master["Morph_state"] == morph) & (master["RNA_state_label"] != rna)).sum())
        c = int(((master["Morph_state"] != morph) & (master["RNA_state_label"] == rna)).sum())
        d = int(((master["Morph_state"] != morph) & (master["RNA_state_label"] != rna)).sum())

        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")

        frac_in = a / max(1, a + b)
        bg_frac = c / max(1, c + d)

        records.append({
            "Morph_state": morph,
            "Morph_name": morph_name.get(morph, morph),
            "RNA_state": rna,
            "RNA_name": rna_name.get(rna, rna),
            "n_Morph_RNA": a,
            "n_Morph_not_RNA": b,
            "n_other_Morph_RNA": c,
            "n_other_Morph_not_RNA": d,
            "fraction_within_Morph": frac_in,
            "background_fraction": bg_frac,
            "agreement_delta": frac_in - bg_frac,
            "odds_ratio": odds,
            "pvalue": p,
        })

enrich = pd.DataFrame(records)
enrich = enrich.sort_values("pvalue").reset_index(drop=True)

m = enrich.shape[0]
enrich["rank"] = np.arange(1, m + 1)
enrich["padj"] = (enrich["pvalue"] * m / enrich["rank"]).clip(upper=1)
enrich["neglog10_padj"] = -np.log10(enrich["padj"] + 1e-300)

enrich["log2_odds_ratio"] = (
    np.log2(enrich["odds_ratio"].replace(0, np.nan))
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

enrich = enrich.sort_values(["Morph_state", "RNA_state"])

enrich.to_csv(
    tabdir / "AD715_48a_Morph_RNA_Fisher_enrichment.tsv",
    sep="\t",
    index=False
)

logor_mat = (
    enrich
    .pivot(index="Morph_state", columns="RNA_state", values="log2_odds_ratio")
    .reindex(index=morph_order, columns=rna_order)
    .fillna(0)
)

padj_mat = (
    enrich
    .pivot(index="Morph_state", columns="RNA_state", values="padj")
    .reindex(index=morph_order, columns=rna_order)
    .fillna(1)
)

delta_mat = (
    enrich
    .pivot(index="Morph_state", columns="RNA_state", values="agreement_delta")
    .reindex(index=morph_order, columns=rna_order)
    .fillna(0)
)

logor_mat.to_csv(tabdir / "AD715_48a_Morph_RNA_log2OR_matrix.tsv", sep="\t")
padj_mat.to_csv(tabdir / "AD715_48a_Morph_RNA_padj_matrix.tsv", sep="\t")
delta_mat.to_csv(tabdir / "AD715_48a_Morph_RNA_agreement_delta_matrix.tsv", sep="\t")

# ------------------------------------------------------------
# 3. Dominant RNA per Morph
# ------------------------------------------------------------
bridge_rows = []

for morph in morph_order:
    sub = frac_mat.loc[morph].sort_values(ascending=False)
    top_rna = sub.index[0]
    top_frac = float(sub.iloc[0])

    e = enrich[
        (enrich["Morph_state"] == morph) &
        (enrich["RNA_state"] == top_rna)
    ].iloc[0]

    bridge_rows.append({
        "Morph_state": morph,
        "Morph_name": morph_name.get(morph, morph),
        "dominant_RNA_state": top_rna,
        "dominant_RNA_name": rna_name.get(top_rna, top_rna),
        "dominant_RNA_fraction": top_frac,
        "log2_odds_ratio": e["log2_odds_ratio"],
        "agreement_delta": e["agreement_delta"],
        "padj": e["padj"],
        "n_cells_in_Morph": int(count_mat.loc[morph].sum()),
        "n_cells_Morph_dominant_RNA": int(count_mat.loc[morph, top_rna]),
    })

bridge = pd.DataFrame(bridge_rows)

bridge.to_csv(
    tabdir / "AD715_48a_Morph_to_dominant_RNA_bridge.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# 4. Figure: fraction heatmap
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 4.8))

im = ax.imshow(
    frac_mat.values,
    aspect="auto",
    cmap="viridis",
    vmin=0,
    vmax=max(0.01, frac_mat.values.max())
)

ax.set_xticks(np.arange(len(rna_order)))
ax.set_xticklabels(rna_order, fontsize=10)

ax.set_yticks(np.arange(len(morph_order)))
ax.set_yticklabels(
    [f"{m}\n{morph_name.get(m, m)}" for m in morph_order],
    fontsize=8
)

ax.set_title("RNA-state composition within morphology states", fontsize=12)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Fraction within Morph state", fontsize=9)

plt.tight_layout()
fig.savefig(figdir / "AD715_48a_Morph_RNA_fraction_heatmap.pdf")
fig.savefig(figdir / "AD715_48a_Morph_RNA_fraction_heatmap.png", dpi=500)
fig.savefig(figdir / "AD715_48a_Morph_RNA_fraction_heatmap.svg")
plt.close(fig)

# ------------------------------------------------------------
# 5. Figure: log2OR heatmap with significance stars
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 4.8))

im = ax.imshow(
    logor_mat.values,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-1.2,
    vmax=1.2
)

ax.set_xticks(np.arange(len(rna_order)))
ax.set_xticklabels(rna_order, fontsize=10)

ax.set_yticks(np.arange(len(morph_order)))
ax.set_yticklabels(
    [f"{m}\n{morph_name.get(m, m)}" for m in morph_order],
    fontsize=8
)

for i, morph in enumerate(morph_order):
    for j, rna in enumerate(rna_order):
        padj = padj_mat.loc[morph, rna]
        val = logor_mat.loc[morph, rna]

        if padj < 0.001:
            star = "***"
        elif padj < 0.01:
            star = "**"
        elif padj < 0.05:
            star = "*"
        else:
            star = ""

        if star:
            ax.text(
                j, i,
                star,
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

ax.set_title("Morphology-RNA state enrichment", fontsize=12)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("log2 odds ratio", fontsize=9)

plt.tight_layout()
fig.savefig(figdir / "AD715_48a_Morph_RNA_log2OR_heatmap.pdf")
fig.savefig(figdir / "AD715_48a_Morph_RNA_log2OR_heatmap.png", dpi=500)
fig.savefig(figdir / "AD715_48a_Morph_RNA_log2OR_heatmap.svg")
plt.close(fig)

# ------------------------------------------------------------
# 6. Figure: stacked bar
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 4.2))

x = np.arange(len(morph_order))
bottom = np.zeros(len(morph_order))

for rna in rna_order:
    vals = frac_mat[rna].values
    ax.bar(
        x,
        vals,
        bottom=bottom,
        width=0.72,
        color=rna_colors.get(rna, None),
        label=rna,
        edgecolor="white",
        linewidth=0.4
    )
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(
    [f"{m}\n{morph_name.get(m, m)}" for m in morph_order],
    rotation=35,
    ha="right",
    fontsize=8
)

ax.set_ylabel("Fraction within Morph state")
ax.set_ylim(0, 1.02)
ax.set_title("RNA-state composition of morphology states", fontsize=12)

ax.legend(
    frameon=False,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8,
    title="RNA state",
    title_fontsize=9
)

plt.tight_layout()
fig.savefig(figdir / "AD715_48a_Morph_RNA_stacked_bar.pdf")
fig.savefig(figdir / "AD715_48a_Morph_RNA_stacked_bar.png", dpi=500)
fig.savefig(figdir / "AD715_48a_Morph_RNA_stacked_bar.svg")
plt.close(fig)

# ------------------------------------------------------------
# 7. Figure: dominant bridge dotplot
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 3.8))

for i, row in bridge.iterrows():
    morph = row["Morph_state"]
    rna = row["dominant_RNA_state"]

    x0 = rna_order.index(rna)
    y0 = morph_order.index(morph)

    size = 800 * row["dominant_RNA_fraction"]

    ax.scatter(
        x0,
        y0,
        s=size,
        c=row["log2_odds_ratio"],
        cmap="RdBu_r",
        vmin=-1.2,
        vmax=1.2,
        edgecolor="black",
        linewidth=0.5
    )

    ax.text(
        x0,
        y0,
        f"{row['dominant_RNA_fraction']*100:.0f}%",
        ha="center",
        va="center",
        fontsize=7,
        color="black"
    )

ax.set_xticks(np.arange(len(rna_order)))
ax.set_xticklabels(rna_order, fontsize=10)

ax.set_yticks(np.arange(len(morph_order)))
ax.set_yticklabels(
    [f"{m}\n{morph_name.get(m, m)}" for m in morph_order],
    fontsize=8
)

ax.set_xlim(-0.5, len(rna_order) - 0.5)
ax.set_ylim(len(morph_order) - 0.5, -0.5)

ax.set_title("Dominant RNA state per morphology state", fontsize=12)

sm = plt.cm.ScalarMappable(
    cmap="RdBu_r",
    norm=plt.Normalize(vmin=-1.2, vmax=1.2)
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("log2 odds ratio", fontsize=9)

plt.tight_layout()
fig.savefig(figdir / "AD715_48a_Morph_to_dominant_RNA_dotplot.pdf")
fig.savefig(figdir / "AD715_48a_Morph_to_dominant_RNA_dotplot.png", dpi=500)
fig.savefig(figdir / "AD715_48a_Morph_to_dominant_RNA_dotplot.svg")
plt.close(fig)

print("[DONE] 48a Morphology state x RNA state coupling")
print(tabdir / "AD715_48a_Morph_RNA_composition.tsv")
print(tabdir / "AD715_48a_Morph_RNA_fraction_matrix.tsv")
print(tabdir / "AD715_48a_Morph_RNA_log2OR_matrix.tsv")
print(tabdir / "AD715_48a_Morph_RNA_Fisher_enrichment.tsv")
print(tabdir / "AD715_48a_Morph_to_dominant_RNA_bridge.tsv")
print(figdir / "AD715_48a_Morph_RNA_fraction_heatmap.pdf")
print(figdir / "AD715_48a_Morph_RNA_log2OR_heatmap.pdf")
print(figdir / "AD715_48a_Morph_RNA_stacked_bar.pdf")
print(figdir / "AD715_48a_Morph_to_dominant_RNA_dotplot.pdf")
print()
print("[Bridge]")
print(bridge)
print()
print("[Fraction matrix]")
print(frac_mat.round(3))
print()
print("[log2OR matrix]")
print(logor_mat.round(3))
