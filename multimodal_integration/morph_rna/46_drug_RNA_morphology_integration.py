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

drug_name_map = {
    "DMSO": "DMSO",
    "Drug01": "Cisplatin",
    "Drug02": "Etoposide",
    "Drug03": "Doxorubicin",
    "Drug04": "Actinomycin D",
    "Drug05": "Paclitaxel",
    "Drug06": "Epothilone B",
    "Drug07": "Vinblastine",
    "Drug08": "Pelitinib",
    "Drug09": "Raltitrexed",
    "Drug10": "Cyclophosphamide",
    "Drug11": "AT-9283",
}

moa_map = {
    "DMSO": "Control",
    "Drug01": "DNA damage",
    "Drug02": "DNA damage",
    "Drug03": "DNA damage",
    "Drug04": "Transcription inhibition",
    "Drug05": "Microtubule stabilization",
    "Drug06": "Microtubule stabilization",
    "Drug07": "Microtubule destabilization",
    "Drug08": "EGFR/HER inhibition",
    "Drug09": "DNA synthesis / replication stress",
    "Drug10": "Alkylating agent / weak in vitro",
    "Drug11": "Aurora kinase / mitotic checkpoint",
}

master = pd.read_csv(master_file, sep="\t")

master["condition"] = master["condition"].astype(str)
master["drug_name"] = master["condition"].map(drug_name_map).fillna(master["condition"])
master["MoA"] = master["condition"].map(moa_map).fillna("Unknown")
master["RNA_state_label"] = "RNA" + master["RNA_state"].astype(str)
master["Morph_state"] = master["Morph_state"].astype(str)

conditions = sorted(master["condition"].unique())
rna_states = sorted(master["RNA_state_label"].unique(), key=lambda x: int(x.replace("RNA", "")))
morph_states = sorted(master["Morph_state"].unique(), key=lambda x: int(x.replace("Morph", "")) if x.startswith("Morph") else x)

# ------------------------------------------------------------
# 1. Drug x RNA x Morph count table
# ------------------------------------------------------------
triple = (
    master
    .groupby(["condition", "drug_name", "MoA", "RNA_state_label", "Morph_state"])
    .size()
    .reset_index(name="n")
)

triple["fraction_within_condition"] = (
    triple["n"] / triple.groupby("condition")["n"].transform("sum")
)

triple["fraction_within_RNA_state_in_condition"] = (
    triple["n"] / triple.groupby(["condition", "RNA_state_label"])["n"].transform("sum")
)

triple["fraction_within_Morph_state_in_condition"] = (
    triple["n"] / triple.groupby(["condition", "Morph_state"])["n"].transform("sum")
)

triple.to_csv(
    tabdir / "AD715_Drug_RNA_Morph_triple_composition.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# 2. For each drug, dominant RNA and dominant Morph state
# ------------------------------------------------------------
drug_summary = []

for cond, sub in master.groupby("condition"):
    rna_count = sub["RNA_state_label"].value_counts(normalize=True)
    morph_count = sub["Morph_state"].value_counts(normalize=True)

    top_rna = rna_count.index[0]
    top_rna_frac = rna_count.iloc[0]

    top_morph = morph_count.index[0]
    top_morph_frac = morph_count.iloc[0]

    drug_summary.append({
        "condition": cond,
        "drug_name": drug_name_map.get(cond, cond),
        "MoA": moa_map.get(cond, "Unknown"),
        "n_cells": sub.shape[0],
        "dominant_RNA_state": top_rna,
        "dominant_RNA_fraction": top_rna_frac,
        "dominant_Morph_state": top_morph,
        "dominant_Morph_fraction": top_morph_frac,
    })

drug_summary = pd.DataFrame(drug_summary)
drug_summary.to_csv(
    tabdir / "AD715_Drug_dominant_RNA_and_Morph_state.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# 3. Drug x RNA state x dominant Morph within RNA
# ------------------------------------------------------------
rna_morph_within_drug = (
    triple
    .sort_values(["condition", "RNA_state_label", "n"], ascending=[True, True, False])
    .groupby(["condition", "RNA_state_label"])
    .head(1)
    .copy()
)

rna_morph_within_drug.to_csv(
    tabdir / "AD715_Drug_RNA_state_dominant_Morph_state.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# 4. Fisher enrichment for Drug-RNA-Morph combination
# ------------------------------------------------------------
records = []

for cond in conditions:
    for rna in rna_states:
        for morph in morph_states:
            inside = (
                (master["condition"] == cond) &
                (master["RNA_state_label"] == rna) &
                (master["Morph_state"] == morph)
            )

            cond_only = master["condition"] == cond
            combo_only = (
                (master["RNA_state_label"] == rna) &
                (master["Morph_state"] == morph)
            )

            a = int(inside.sum())
            b = int((cond_only & ~combo_only).sum())
            c = int((~cond_only & combo_only).sum())
            d = int((~cond_only & ~combo_only).sum())

            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")

            records.append({
                "condition": cond,
                "drug_name": drug_name_map.get(cond, cond),
                "MoA": moa_map.get(cond, "Unknown"),
                "RNA_state": rna,
                "Morph_state": morph,
                "n_condition_RNA_Morph": a,
                "odds_ratio": odds,
                "pvalue": p,
            })

enrich = pd.DataFrame(records)
enrich = enrich.sort_values("pvalue").reset_index(drop=True)

m = enrich.shape[0]
enrich["rank"] = np.arange(1, m + 1)
enrich["padj"] = (enrich["pvalue"] * m / enrich["rank"]).clip(upper=1)
enrich["neglog10_padj"] = -np.log10(enrich["padj"] + 1e-300)

enrich = enrich.sort_values(["condition", "RNA_state", "Morph_state"])
enrich.to_csv(
    tabdir / "AD715_Drug_RNA_Morph_Fisher_enrichment.tsv",
    sep="\t",
    index=False
)

# ------------------------------------------------------------
# 5. Heatmap: condition x RNA-Morph combination
# ------------------------------------------------------------
master["RNA_Morph"] = master["RNA_state_label"] + "|" + master["Morph_state"]

combo = (
    master
    .groupby(["condition", "RNA_Morph"])
    .size()
    .reset_index(name="n")
)

combo["fraction_within_condition"] = (
    combo["n"] / combo.groupby("condition")["n"].transform("sum")
)

combo_mat = combo.pivot(
    index="condition",
    columns="RNA_Morph",
    values="fraction_within_condition"
).fillna(0)

# order columns by RNA then Morph
ordered_cols = []
for rna in rna_states:
    for morph in morph_states:
        col = f"{rna}|{morph}"
        if col in combo_mat.columns:
            ordered_cols.append(col)

combo_mat = combo_mat.loc[conditions, ordered_cols]

combo_mat.to_csv(
    tabdir / "AD715_condition_by_RNA_Morph_fraction_matrix.tsv",
    sep="\t"
)

fig, ax = plt.subplots(figsize=(max(10, len(ordered_cols) * 0.32), 5.5))

im = ax.imshow(
    combo_mat.values,
    aspect="auto",
    cmap="viridis",
    vmin=0,
    vmax=max(0.01, combo_mat.values.max())
)

ax.set_xticks(np.arange(combo_mat.shape[1]))
ax.set_xticklabels(combo_mat.columns, rotation=90, fontsize=7)

ax.set_yticks(np.arange(combo_mat.shape[0]))
ax.set_yticklabels(
    [f"{drug_name_map.get(x, x)} ({x})" for x in combo_mat.index],
    fontsize=8
)

ax.set_title("Drug x RNA state x Morphology state composition", fontsize=12)

cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Fraction within drug condition", fontsize=9)

plt.tight_layout()
fig.savefig(figdir / "AD715_Drug_RNA_Morph_fraction_heatmap.pdf")
fig.savefig(figdir / "AD715_Drug_RNA_Morph_fraction_heatmap.png", dpi=300)
fig.savefig(figdir / "AD715_Drug_RNA_Morph_fraction_heatmap.svg")
plt.close(fig)

# ------------------------------------------------------------
# 6. Export network-ready links
# ------------------------------------------------------------
# Drug -> RNA
drug_rna = (
    master
    .groupby(["condition", "RNA_state_label"])
    .size()
    .reset_index(name="n")
)
drug_rna["weight"] = drug_rna["n"] / drug_rna.groupby("condition")["n"].transform("sum")
drug_rna["source"] = drug_rna["condition"]
drug_rna["target"] = drug_rna["RNA_state_label"]
drug_rna["edge_type"] = "Drug_to_RNA_state"

drug_rna[["source", "target", "edge_type", "weight", "n"]].to_csv(
    tabdir / "AD715_network_links_Drug_to_RNA_state.tsv",
    sep="\t",
    index=False
)

# RNA -> Morph
rna_morph = (
    master
    .groupby(["RNA_state_label", "Morph_state"])
    .size()
    .reset_index(name="n")
)
rna_morph["weight"] = rna_morph["n"] / rna_morph.groupby("RNA_state_label")["n"].transform("sum")
rna_morph["source"] = rna_morph["RNA_state_label"]
rna_morph["target"] = rna_morph["Morph_state"]
rna_morph["edge_type"] = "RNA_state_to_Morph_state"

rna_morph[["source", "target", "edge_type", "weight", "n"]].to_csv(
    tabdir / "AD715_network_links_RNA_state_to_Morph_state.tsv",
    sep="\t",
    index=False
)

print("[DONE] 46 Drug-RNA-Morph integration")
print(tabdir / "AD715_Drug_RNA_Morph_triple_composition.tsv")
print(tabdir / "AD715_Drug_dominant_RNA_and_Morph_state.tsv")
print(tabdir / "AD715_Drug_RNA_state_dominant_Morph_state.tsv")
print(tabdir / "AD715_Drug_RNA_Morph_Fisher_enrichment.tsv")
print(tabdir / "AD715_condition_by_RNA_Morph_fraction_matrix.tsv")
print(tabdir / "AD715_network_links_Drug_to_RNA_state.tsv")
print(tabdir / "AD715_network_links_RNA_state_to_Morph_state.tsv")
print(figdir / "AD715_Drug_RNA_Morph_fraction_heatmap.pdf")
print()
print("[Preview dominant drug states]")
print(drug_summary)
