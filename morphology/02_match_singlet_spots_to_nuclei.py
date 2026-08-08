#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd


def find_column(df, candidates, required=True):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Cannot find required column among: {candidates}")
    return None


def parse_spot_id_from_row(row):
    candidates = ["spot_id", "spot", "barcode", "spot_name", "id", "name"]
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return str(row[c])
    if "row" in row.index and "col" in row.index:
        return f"A{int(row['row'])}B{int(row['col'])}"
    if "A" in row.index and "B" in row.index:
        return f"A{int(row['A'])}B{int(row['B'])}"
    return None


def find_class_column(df):
    candidates = ["class", "spot_class", "class_name", "final_class", "label", "category"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Cannot find class column")


def is_single_value(v):
    s = str(v).strip().lower()
    return s in ["single", "singlet", "1"]


def choose_match_two_stage(
    x,
    y,
    nuclei_xy,
    nuclei_df,
    r1,
    r2,
    min_gap,
    min_ratio,
    min_area,
    max_area
):
    d2 = (nuclei_xy[:, 0] - x) ** 2 + (nuclei_xy[:, 1] - y) ** 2
    order = np.argsort(d2)

    if len(order) == 0:
        return {
            "status": "no_nucleus_candidates",
            "distance_spot_to_nucleus": np.nan,
            "stage": "none"
        }

    idx1 = int(order[0])
    dist1 = float(np.sqrt(d2[idx1]))

    idx2 = int(order[1]) if len(order) > 1 else None
    dist2_ = float(np.sqrt(d2[idx2])) if idx2 is not None else np.inf

    nuc1 = nuclei_df.iloc[idx1]
    area1 = float(nuc1["area"])

    area_ok = (area1 >= min_area) and (area1 <= max_area)
    gap_ok = (dist2_ - dist1) >= min_gap
    ratio_ok = (dist2_ / max(dist1, 1e-6)) >= min_ratio

    # -------- Stage 2: strict matching radius --------
    if dist1 <= r1 and area_ok:
        return {
            "status": "matched_r50",
            "stage": "r50",
            "distance_spot_to_nucleus": dist1,
            "distance_to_second_nucleus": dist2_,
            "distance_gap": dist2_ - dist1,
            "distance_ratio": dist2_ / max(dist1, 1e-6),
            "nucleus_id": int(nuc1["nucleus_id"]),
            "nucleus_x": float(nuc1["centroid_x"]),
            "nucleus_y": float(nuc1["centroid_y"]),
            "nucleus_area": area1,
            "nucleus_equivalent_diameter": float(nuc1["equivalent_diameter"])
        }

    # -------- Stage 2: process only spots unmatched in Stage 1 --------
    # Use a more permissive radius while requiring clearer separation between the nearest and second-nearest nuclei
    if dist1 <= r2 and area_ok and (gap_ok or ratio_ok):
        return {
            "status": "matched_r80_rescue",
            "stage": "r80_rescue",
            "distance_spot_to_nucleus": dist1,
            "distance_to_second_nucleus": dist2_,
            "distance_gap": dist2_ - dist1,
            "distance_ratio": dist2_ / max(dist1, 1e-6),
            "nucleus_id": int(nuc1["nucleus_id"]),
            "nucleus_x": float(nuc1["centroid_x"]),
            "nucleus_y": float(nuc1["centroid_y"]),
            "nucleus_area": area1,
            "nucleus_equivalent_diameter": float(nuc1["equivalent_diameter"])
        }

    # ---------- Remaining unmatched spots ----------
    return {
        "status": "no_nucleus_within_radius",
        "stage": "none",
        "distance_spot_to_nucleus": dist1,
        "distance_to_second_nucleus": dist2_,
        "distance_gap": dist2_ - dist1,
        "distance_ratio": dist2_ / max(dist1, 1e-6)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot_summary", required=True)
    ap.add_argument("--nuclei_props", required=True)
    ap.add_argument("--out_tsv", required=True)

    ap.add_argument("--r1", type=float, default=50.0, help="strict first-pass radius")
    ap.add_argument("--r2", type=float, default=80.0, help="rescue second-pass radius")
    ap.add_argument("--min_gap", type=float, default=8.0, help="minimum d2-d1 for rescue match")
    ap.add_argument("--min_ratio", type=float, default=1.15, help="minimum d2/d1 for rescue match")
    ap.add_argument("--min_area", type=float, default=20.0)
    ap.add_argument("--max_area", type=float, default=3000.0)

    args = ap.parse_args()

    spot_df = pd.read_csv(args.spot_summary, sep="\t")
    nuclei_df = pd.read_csv(args.nuclei_props, sep="\t")

    class_col = find_class_column(spot_df)
    x_col = find_column(spot_df, ["x_center", "center_x", "cx", "x", "x_centroid"])
    y_col = find_column(spot_df, ["y_center", "center_y", "cy", "y", "y_centroid"])

    single_df = spot_df[spot_df[class_col].astype(str).map(is_single_value)].copy()
    single_df = single_df.reset_index(drop=True)

    nuclei_xy = nuclei_df[["centroid_x", "centroid_y"]].to_numpy(dtype=float)

    rows = []
    for _, row in single_df.iterrows():
        spot_id = parse_spot_id_from_row(row)
        x = float(row[x_col])
        y = float(row[y_col])

        rec = {
            "spot_id": spot_id,
            "spot_x": x,
            "spot_y": y
        }

        match = choose_match_two_stage(
            x=x,
            y=y,
            nuclei_xy=nuclei_xy,
            nuclei_df=nuclei_df,
            r1=args.r1,
            r2=args.r2,
            min_gap=args.min_gap,
            min_ratio=args.min_ratio,
            min_area=args.min_area,
            max_area=args.max_area
        )
        rec.update(match)
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_tsv, sep="\t", index=False)

    print("[OK] wrote:", args.out_tsv)
    print(out_df["status"].value_counts(dropna=False))
    print()
    print(out_df.groupby("status")["distance_spot_to_nucleus"].describe())


if __name__ == "__main__":
    main()
