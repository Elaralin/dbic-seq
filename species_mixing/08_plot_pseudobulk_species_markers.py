#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def split_feature_parts(feature: str):
    return [x.strip() for x in str(feature).split('---') if x.strip()]


def parse_feature_class(part: str):
    s = str(part).lower()
    if '__exon' in s:
        return 'exon'
    elif '__intron' in s:
        return 'intron'
    return 'other'


def parse_biotype(part: str):
    toks = str(part).split('__')
    if len(toks) >= 2:
        return toks[1]
    return 'other'


def keep_feature(feature: str):
    parts = split_feature_parts(feature)
    classes = [parse_feature_class(p) for p in parts]
    biotypes = [parse_biotype(p) for p in parts]
    return any((c == 'exon' and b == 'protein_coding') for c, b in zip(classes, biotypes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--expmat", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--call-col", default="final_call_round6")
    ap.add_argument("--chunksize", type=int, default=2000)
    ap.add_argument("--top-n", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    calls = pd.read_csv(args.calls, sep="\t")
    if args.call_col not in calls.columns:
        raise ValueError(f"{args.call_col} not found in calls table.")

    singlets = calls[calls[args.call_col].isin(["human", "mouse"])].copy()
    human_barcodes = list(singlets.loc[singlets[args.call_col] == "human", "barcode"].astype(str))
    mouse_barcodes = list(singlets.loc[singlets[args.call_col] == "mouse", "barcode"].astype(str))

    header = pd.read_csv(args.expmat, sep="\t", nrows=0)
    matrix_barcodes = list(header.columns[1:])
    barcode_to_idx = {b: i for i, b in enumerate(matrix_barcodes)}

    human_idx = [barcode_to_idx[b] for b in human_barcodes if b in barcode_to_idx]
    mouse_idx = [barcode_to_idx[b] for b in mouse_barcodes if b in barcode_to_idx]

    rows = []

    for chunk in pd.read_csv(args.expmat, sep="\t", chunksize=args.chunksize):
        features = chunk.iloc[:, 0].astype(str)
        keep_mask = features.apply(keep_feature).values
        if keep_mask.sum() == 0:
            continue

        sub = chunk.loc[keep_mask].copy()
        feats = sub.iloc[:, 0].astype(str).tolist()
        mat = sub.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float64)

        for i, feat in enumerate(feats):
            h_mean = mat[i, human_idx].mean() if len(human_idx) > 0 else 0.0
            m_mean = mat[i, mouse_idx].mean() if len(mouse_idx) > 0 else 0.0
            log2fc = np.log2((h_mean + 1.0) / (m_mean + 1.0))
            rows.append({
                "feature": feat,
                "human_mean": h_mean,
                "mouse_mean": m_mean,
                "log2FC_human_vs_mouse": log2fc
            })

    marker_df = pd.DataFrame(rows)
    marker_tsv = os.path.join(args.outdir, f"{args.sample}_pseudobulk_feature_stats.tsv")
    marker_df.to_csv(marker_tsv, sep="\t", index=False)

    top_h = marker_df.sort_values("log2FC_human_vs_mouse", ascending=False).head(args.top_n)
    top_m = marker_df.sort_values("log2FC_human_vs_mouse", ascending=True).head(args.top_n)

    top_df = pd.concat([top_h, top_m], ignore_index=True)
    top_df = top_df.drop_duplicates("feature")

    top_tsv = os.path.join(args.outdir, f"{args.sample}_top_species_markers.tsv")
    top_df.to_csv(top_tsv, sep="\t", index=False)

    heat = top_df[["human_mean", "mouse_mean"]].to_numpy(dtype=float)
    row_labels = list(top_df["feature"])
    col_labels = ["human", "mouse"]

    fig, ax = plt.subplots(figsize=(5.8, max(5, 0.22 * len(row_labels))))
    im = ax.imshow(heat, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title("Top species-enriched features")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean expression")

    plt.tight_layout()
    heatmap_pdf = os.path.join(args.outdir, f"{args.sample}_top_species_marker_heatmap.pdf")
    fig.savefig(heatmap_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, max(5, 0.22 * len(row_labels))))
    y = np.arange(len(row_labels))

    ax.scatter(top_df["human_mean"], y, s=50, label="human")
    ax.scatter(top_df["mouse_mean"], y, s=50, label="mouse")

    ax.set_yticks(y)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Mean expression")
    ax.set_title("Top species-enriched feature dotplot")
    ax.legend(frameon=False)

    plt.tight_layout()
    dotplot_pdf = os.path.join(args.outdir, f"{args.sample}_top_species_marker_dotplot.pdf")
    fig.savefig(dotplot_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(marker_tsv)
    print(top_tsv)
    print(heatmap_pdf)
    print(dotplot_pdf)


if __name__ == "__main__":
    main()
