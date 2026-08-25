#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def bilinear(p00, p10, p01, p11, u, v):
    """
    Bilinear interpolation inside a quadrilateral.
    p00 = top-left
    p10 = top-right
    p01 = bottom-left
    p11 = bottom-right
    u: left -> right, in [0, 1]
    v: top -> bottom, in [0, 1]
    """
    return (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + (1 - u) * v * p01 + u * v * p11


def normalize_channel(ch):
    ch = ch.astype(np.float32)
    lo, hi = np.percentile(ch, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((ch - lo) / (hi - lo), 0, 1)


def load_corner_points(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")

    required_cols = {"corner", "x", "y"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"Corner TSV must contain columns {required_cols}, got {df.columns.tolist()}"
        )

    df["corner"] = df["corner"].astype(str).str.strip().str.lower()

    expected = {"tl", "tr", "bl", "br"}
    have = set(df["corner"])
    if have != expected:
        raise ValueError(
            f"Corner labels must be exactly {expected}, got {have}"
        )

    lookup = {}
    for _, r in df.iterrows():
        lookup[r["corner"]] = np.array([float(r["x"]), float(r["y"])], dtype=float)

    return lookup["tl"], lookup["tr"], lookup["bl"], lookup["br"]


def convert_center_corners_to_boundary_corners(tl, tr, bl, br, rows, cols):
    """
    Interpret the 4 given points as the centers of the corner spots:
      tl = center of A1-B1
      tr = center of A1-B50
      bl = center of A50-B1
      br = center of A50-B50

    Convert them into the OUTER boundary corners of the full 50x50 grid.
    """
    step_top = (tr - tl) / (cols - 1)
    step_bottom = (br - bl) / (cols - 1)

    step_left = (bl - tl) / (rows - 1)
    step_right = (br - tr) / (rows - 1)

    # Expand from corner-spot centers to outer box corners
    tl_b = tl - 0.5 * step_top - 0.5 * step_left
    tr_b = tr + 0.5 * step_top - 0.5 * step_right
    bl_b = bl - 0.5 * step_bottom + 0.5 * step_left
    br_b = br + 0.5 * step_bottom + 0.5 * step_right

    return tl_b, tr_b, bl_b, br_b


def expand_grid_corners(tl, tr, bl, br, expand):
    """
    Globally expand or shrink the entire grid around its geometric center.
    expand = 1.00 means no change
    expand = 1.03 means enlarge by 3%
    """
    center = (tl + tr + bl + br) / 4.0
    tl2 = center + (tl - center) * expand
    tr2 = center + (tr - center) * expand
    bl2 = center + (bl - center) * expand
    br2 = center + (br - center) * expand
    return tl2, tr2, bl2, br2


def build_grid_df(tl, tr, bl, br, rows, cols):
    out_rows = []

    for r in range(rows):
        for c in range(cols):
            u0 = c / cols
            u1 = (c + 1) / cols
            v0 = r / rows
            v1 = (r + 1) / rows

            p_tl = bilinear(tl, tr, bl, br, u0, v0)
            p_tr = bilinear(tl, tr, bl, br, u1, v0)
            p_bl = bilinear(tl, tr, bl, br, u0, v1)
            p_br = bilinear(tl, tr, bl, br, u1, v1)
            p_ct = bilinear(tl, tr, bl, br, (u0 + u1) / 2.0, (v0 + v1) / 2.0)

            row_idx = r + 1
            col_idx = c + 1

            out_rows.append({
                "spot_id": f"R{row_idx:02d}C{col_idx:02d}",
                "row": row_idx,
                "col": col_idx,
                "A_barcode": f"A{row_idx}",
                "B_barcode": f"B{col_idx}",
                "center_x": p_ct[0],
                "center_y": p_ct[1],
                "x_tl": p_tl[0], "y_tl": p_tl[1],
                "x_tr": p_tr[0], "y_tr": p_tr[1],
                "x_br": p_br[0], "y_br": p_br[1],
                "x_bl": p_bl[0], "y_bl": p_bl[1],
            })

    return pd.DataFrame(out_rows)


def draw_overlay(df, red_img_path, blue_img_path, out_png, label_every=10, title=""):
    red = np.squeeze(tifffile.imread(red_img_path))
    blue = np.squeeze(tifffile.imread(blue_img_path))

    red_n = normalize_channel(red)
    blue_n = normalize_channel(blue)
    rgb = np.dstack([red_n, np.zeros_like(red_n), blue_n])

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.imshow(rgb)

    for _, r in df.iterrows():
        poly = np.array([
            [r["x_tl"], r["y_tl"]],
            [r["x_tr"], r["y_tr"]],
            [r["x_br"], r["y_br"]],
            [r["x_bl"], r["y_bl"]],
        ])
        ax.add_patch(Polygon(poly, fill=False, edgecolor="lime", linewidth=0.3))

        if (r["row"] - 1) % label_every == 0 and (r["col"] - 1) % label_every == 0:
            spot_w = 0.5 * (
                abs(r["x_tr"] - r["x_tl"]) +
                abs(r["x_br"] - r["x_bl"])
            )
            spot_h = 0.5 * (
                abs(r["y_bl"] - r["y_tl"]) +
                abs(r["y_br"] - r["y_tr"])
            )

            x_text = r["x_tl"] + 0.03 * spot_w
            y_text = r["y_tl"] + 0.08 * spot_h

            ax.text(
                x_text,
                y_text,
                f'{r["A_barcode"]}{r["B_barcode"]}',
                color="white",
                fontsize=5,
                ha="left",
                va="top",
                bbox=dict(
                    facecolor="black",
                    alpha=0.60,
                    edgecolor="none",
                    pad=0.5
                ),
                zorder=10
            )

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners", required=True, help="TSV with columns: corner, y, x")
    ap.add_argument("--rows", type=int, default=50)
    ap.add_argument("--cols", type=int, default=50)
    ap.add_argument("--red-image", required=True)
    ap.add_argument("--blue-image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlay-out", required=True)
    ap.add_argument("--label-every", type=int, default=10)
    ap.add_argument(
        "--corner-mode",
        choices=["boundary", "center"],
        default="center",
        help="Interpret input points as outer boundary corners or corner-spot centers"
    )
    ap.add_argument(
        "--expand",
        type=float,
        default=1.0,
        help="Global expansion factor. Example: 1.03 enlarges the grid by 3 percent"
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.overlay_out), exist_ok=True)

    tl, tr, bl, br = load_corner_points(args.corners)

    # Step 1: if input points are corner-spot centers, convert them to boundary corners
    if args.corner_mode == "center":
        tl, tr, bl, br = convert_center_corners_to_boundary_corners(
            tl, tr, bl, br, args.rows, args.cols
        )

    # Step 2: optional global expansion correction
    tl, tr, bl, br = expand_grid_corners(tl, tr, bl, br, args.expand)

    # Step 3: build spot polygons
    df = build_grid_df(tl, tr, bl, br, args.rows, args.cols)

    # Save grid table
    df.to_csv(args.out, sep="\t", index=False)

    # Save overlay image
    title = f"{args.rows}x{args.cols} DBiC spot grid overlay"
    draw_overlay(
        df,
        red_img_path=args.red_image,
        blue_img_path=args.blue_image,
        out_png=args.overlay_out,
        label_every=args.label_every,
        title=title
    )

    print(f"[OK] wrote table:   {args.out}")
    print(f"[OK] wrote overlay: {args.overlay_out}")


if __name__ == "__main__":
    main()
