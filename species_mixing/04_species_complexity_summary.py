#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def add_box_inside_violin(ax, pos, vals, box_w=0.16):
    if len(vals) == 0:
        return

    q1, med, q3 = np.percentile(vals, [25, 50, 75])

    ax.add_patch(
        plt.Rectangle(
            (pos - box_w / 2, q1),
            box_w,
            q3 - q1,
            facecolor="white",
            edgecolor="black",
            lw=0.8,
            zorder=3
        )
    )
    ax.plot([pos - box_w / 2, pos + box_w / 2], [med, med],
            color="black", lw=1.0, zorder=4)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="both", which="major",
                   labelsize=10, width=1.5, length=5, colors="black")


def make_violin(ax, df, value_col, ylabel):
    order = ["HeLa", "3T3", "Mixed"]
    colors = {
        "HeLa": "#7EA1D3",
        "3T3": "#E06A63",
        "Mixed": "#8FB9A8",
    }
    positions = [1, 2, 3]

    data = [df.loc[df["plot_class"] == cls, value_col].dropna().values for cls in order]

    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showextrema=False,
        showmedians=False
    )

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[order[i]])
        body.set_edgecolor("black")
        body.set_alpha(0.75)
        body.set_linewidth(0.8)

    for pos, vals in zip(positions, data):
        add_box_inside_violin(ax, pos, vals)

    ymax = max(np.max(v) if len(v) > 0 else 0 for v in data)
    ax.set_ylim(0, ymax * 1.05)

    ax.set_xticks(positions)
    ax.set_xticklabels(order, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    style_axis(ax)


def compute_barcode_complexity_from_expmat(expmat_tsv, chunksize=2000):
    """
    Compute nCount and nFeature for each barcode directly from expmat.tsv.
    Assumes first column is feature/gene, remaining columns are barcodes.
    """
    header = pd.read_csv(expmat_tsv, sep="\t", nrows=0)
    barcode_cols = [str(x) for x in header.columns[1:]]

    n_barcodes = len(barcode_cols)
    total_counts = np.zeros(n_barcodes, dtype=np.int64)
    total_features = np.zeros(n_barcodes, dtype=np.int64)

    for chunk in pd.read_csv(expmat_tsv, sep="\t", chunksize=chunksize):
        mat = chunk.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        total_counts += mat.sum(axis=0)
        total_features += (mat > 0).sum(axis=0)

    out_df = pd.DataFrame({
        "barcode": barcode_cols,
        "nCount": total_counts,
        "nFeature": total_features
    })
    return out_df


def main():
    ap = argparse.ArgumentParser(description="Make CM524 HeLa/3T3/Mixed complexity summary plot from expmat + species calls.")
    ap.add_argument("--calls", required=True,
                    help="CM524 species calls table, e.g. CM524_round6_strict_species_calls.tsv")
    ap.add_argument("--expmat", required=True,
                    help="Raw ASTRO expmat.tsv for CM524")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample", default="CM524")
    ap.add_argument("--call-col", default="final_call_round6")
    ap.add_argument("--chunksize", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    calls = pd.read_csv(args.calls, sep="\t")
    if "barcode" not in calls.columns:
        raise ValueError("barcode column not found in calls table")
    if args.call_col not in calls.columns:
        raise ValueError(f"{args.call_col} not found in calls table")

    # compute nCount / nFeature directly from expmat
    complexity = compute_barcode_complexity_from_expmat(args.expmat, chunksize=args.chunksize)
    complexity_out = os.path.join(args.outdir, f"{args.sample}_barcode_complexity.tsv")
    complexity.to_csv(complexity_out, sep="\t", index=False)

    df = calls.merge(complexity, on="barcode", how="left")

    class_map = {
        "human": "HeLa",
        "mouse": "3T3",
        "mixed": "Mixed"
    }
    df = df[df[args.call_col].isin(class_map.keys())].copy()
    df["plot_class"] = df[args.call_col].map(class_map)

    # save merged table
    merged_out = os.path.join(args.outdir, f"{args.sample}_complexity_merged.tsv")
    df.to_csv(merged_out, sep="\t", index=False)

    # summary stats
    summary = (
        df.groupby("plot_class")[["nCount", "nFeature"]]
        .agg(["count", "mean", "median"])
        .round(2)
    )
    summary_out = os.path.join(args.outdir, f"{args.sample}_complexity_summary_stats.tsv")
    summary.to_csv(summary_out, sep="\t")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    make_violin(axes[0], df, "nFeature", "Genes per barcode")
    make_violin(axes[1], df, "nCount", "UMIs per barcode")
    plt.tight_layout()

    out_png = os.path.join(args.outdir, f"{args.sample}_complexity_summary.png")
    out_pdf = os.path.join(args.outdir, f"{args.sample}_complexity_summary.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(out_png)
    print(out_pdf)
    print(summary_out)
    print(complexity_out)
    print(merged_out)


if __name__ == "__main__":
    main()
