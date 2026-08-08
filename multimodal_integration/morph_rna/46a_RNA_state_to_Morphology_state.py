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

outdir = ROOT / "6_single_cell/12_Drug_RNA_Morph_integration"
tabdir = outdir / "tables"
figdir = outdir / "figures"
tabdir.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)

master = pd.read_csv(master_file, sep="\t")

master["RNA_state_label"] = "RNA" + master["RNA_state"].astype(str)
master["Morph_state"] = master["Morph_state"].astype(str)

rna_states = sorted(master["RNA_state_label"].unique(), key=lambda x: int(x.replace("RNA", "")))
morph_states = sorted(
    master["Morph_state"].unique(),
    key=lambda x: int(x.replace("Morph", "")) if x.startswith("Morph") else x
)

# ------------------------------------------------------------
# 1. RNA state x Morph state composition
# ------------------------------------------------------------
comp = (
    master
    .groupby(["RNA_state_label", "Morph_state"])
    .size()
    .reset_index(name="n")
)

comp["fraction_within_RNA_state"] = (
    comp["n"] / comp.groupby("RNA_state_label")["n"].transform("sum")
)

comp["fraction_within_Morph_state"] = (
    comp["n"] / comp.groupby("Morph_state")["n"].transform("sum")
)

comp.to_csv(
    tabdir / "AD715_46a_RNA_state_by_Morph_state_composition.tsv",
    sep="\t",
    index=False
)

mat_count = (
    comp
    .pivot(index="RNA_state_label", columns="Morph_state", values="n")
    .fillna(0)
    .loc[rna_states, morph_states]
)

mat_frac = (
    comp
    .pivot(index="RNA_state_label", columns="Morph_state", values="fraction_within_RNA_state")
    .fillna(0)
    .loc[rna_states, morph_states]
)

mat_count.to_csv(tabdir / "AD715_46a_RNA_state_by_Morph_state_count_matrix.tsv", sep="\t")
mat_frac.to_csv(tabdir / "AD715_46a_RNA_state_by_Morph_state_fraction_matrix.tsv", sep="\t")

# ------------------------------------------------------------
# 2. Fisher enrichment RNA state x Morph state
# ------------------------------------------------------------
records = []

for rna in rna_states:
    for morph in morph_states:
        a = int(((master["RNA_state_label"] == rna) & (master["Morph_state"] == morph)).sum())
        b = int(((master["RNA_state_label"] == rna) & (master["Morph_state"] != morph)).sum())
        c = int(((master["RNA_state_label"] != rna) & (master["Morph_state"] == morph)).sum())
        d = int(((master["RNA_state_label"] != rna) & (master["Morph_state"] != morph)).sum())

        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")

        records.append({
            "RNA_state": rna,
            "Morph_state": morph,
            "n_RNA_Morph": a,
            "odds_ratio": odds,
            "pvalue": p,
            "fraction_within_RNA_state": a / max(1, a + b),
            "background_fraction": c / max(1, c + d),
        })

enrich = pd.DataFrame(records)
enrich = enrich.sort_values("pvalue").reset_index(drop=True)

m = enrich.shape[0]
enrich["rank"] = np.arange(1, m + 1)
enrich["padj"] = (enrich["pvalue"] * m / enrich["rank"]).clip(upper=1)
enrich["neglog10_padj"] = -np.log10(enrich["padj"] + 1e-300)

enrich = enrich.sort_values(["RNA_state", "Morph_state"])

enrich.to_csv(
    tabdir / "AD715_46a_RNA_state_by_Morph_state_Fisher_enrichment.tsv",
    sep="\t",
    index=False
)

odds_mat = (
    enrich
    .pivot(index="RNA_state", columns="Morph_state", values="odds_ratio")
    .replace([np.inf], np.nan)
    .fillna(0)
    .loc[rna_states, morph_states]
)

logor_mat = np.log2(odds_mat.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
logor_mat.to_csv(tabdir / "AD715_46a_RNA_state_by_Morph_state_log2OR_matrix.tsv", sep="\t")

# ------------------------------------------------------------
# 3. Dominant Morph state per RNA state
# ------------------------------------------------------------
bridge = []

for rna, sub in master.groupby("RNA_state_label"):
    vc = sub["Morph_state"].value_counts(normalize=True)
    top = vc.index[0]

    bridge.append({
        "RNA_state": rna,
        "n_cells": sub.shape[0],
        "dominant_Morph_state": top,
        "dominant_Morph_fraction": float(vc.iloc[0]),
    })

bridge = pd.DataFrame(bridge)
bridge["RNA_state"] = pd.Categorical(bridge["RNA_state"], categories=rna_states, ordered=True)
bridge = bridge.sort_values("RNA_state")

bridge.to_csv(
    tabdir / "AD715_46a_RNA_state_to_dominant_Morph_state_bridge.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# 4. Network-ready links
# ------------------------------------------------------------
links = comp.copy()
links["source"] = links["RNA_state_label"]
links["target"] = links["Morph_state"]
links["edge_type"] = "RNA_state_to_Morph_state"
links["weight"] = links["fraction_within_RNA_state"]

links[["source", "target", "edge_type", "weight", "n"]].to_csv(
    tabdir / "AD715_46a_network_links_RNA_state_to_Morph_state.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# 5. Figure: stacked bar Morph composition within RNA state
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.4))

bottom = np.zeros(len(rna_states))
x = np.arange(len(rna_states))

for morph in morph_states:
    vals = mat_frac[morph].values
    ax.bar(x, vals, bottom=bottom, label=morph, width=0.72)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(rna_states, fontsize=10)
ax.set_ylabel("Fraction within RNA state")
ax.set_title("Morphology-state composition of RNA states")
ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

plt.tight_layout()
fig.savefig(figdir / "AD715_46a_RNA_state_by_Morph_state_stacked_bar.pdf")
fig.savefig(figdir / "AD715_46a_RNA_state_by_Morph_state_stacked_bar.png", dpi=400)
fig.savefig(figdir / "AD715_46a_RNA_state_by_Morph_state_stacked_bar.svg")
plt.close(fig)

# ------------------------------------------------------------
# 6. Figure: logOR heatmap
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4.2))

im = ax.imshow(
    logor_mat.values,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-2,
    vmax=2
)

ax.set_xticks(np.arange(len(morph_states)))
ax.set_xticklabels(morph_states, rotation=45, ha="right", fontsize=9)

ax.set_yticks(np.arange(len(rna_states)))
ax.set_yticklabels(rna_states, fontsize=10)

ax.set_title("RNA-state enrichment for morphology states")

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("log2 odds ratio")

plt.tight_layout()
fig.savefig(figdir / "AD715_46a_RNA_state_by_Morph_state_log2OR_heatmap.pdf")
fig.savefig(figdir / "AD715_46a_RNA_state_by_Morph_state_log2OR_heatmap.png", dpi=400)
fig.savefig(figdir / "AD715_46a_RNA_state_by_Morph_state_log2OR_heatmap.svg")
plt.close(fig)

print("[DONE] 46a RNA state -> Morphology state")
print(tabdir / "AD715_46a_RNA_state_by_Morph_state_composition.tsv")
print(tabdir / "AD715_46a_RNA_state_by_Morph_state_fraction_matrix.tsv")
print(tabdir / "AD715_46a_RNA_state_by_Morph_state_Fisher_enrichment.tsv")
print(tabdir / "AD715_46a_RNA_state_by_Morph_state_log2OR_matrix.tsv")
print(tabdir / "AD715_46a_RNA_state_to_dominant_Morph_state_bridge.tsv")
print(tabdir / "AD715_46a_network_links_RNA_state_to_Morph_state.tsv")
print(figdir / "AD715_46a_RNA_state_by_Morph_state_stacked_bar.pdf")
print(figdir / "AD715_46a_RNA_state_by_Morph_state_log2OR_heatmap.pdf")
print()
print("[Bridge]")
print(bridge)
print()
print("[Fraction matrix]")
print(mat_frac)
print()
print("[log2OR matrix]")
print(logor_mat)
