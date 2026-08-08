#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--min-total", type=int, default=200)
    ap.add_argument("--call-col", default="final_call_round6")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.calls, sep="\t")
    if "total" not in df.columns:
        df["total"] = df["human"] + df["mouse"]

    sub = df[df["total"] >= args.min_total].copy()
    sub["human_frac"] = np.where(sub["total"] > 0, sub["human"] / sub["total"], 0.0)
    sub["mouse_frac"] = np.where(sub["total"] > 0, sub["mouse"] / sub["total"], 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].hist(sub["human_frac"], bins=50)
    axes[0].set_xlabel("human fraction")
    axes[0].set_ylabel("Barcode count")
    axes[0].set_title("Human fraction distribution")

    axes[1].hist(sub["mouse_frac"], bins=50)
    axes[1].set_xlabel("mouse fraction")
    axes[1].set_ylabel("Barcode count")
    axes[1].set_title("Mouse fraction distribution")

    plt.tight_layout()
    out_pdf = os.path.join(args.outdir, f"{args.sample}_species_fraction_hist.pdf")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(out_pdf)


if __name__ == "__main__":
    main()
