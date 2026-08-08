#!/usr/bin/env python3
import argparse
import os
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, feature, measure, color


def normalize_channel(ch):
    ch = ch.astype(np.float32)
    lo, hi = np.percentile(ch, [1, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((ch - lo) / (hi - lo), 0, 1)


def relabel_sequential(lbl):
    ids = np.unique(lbl)
    ids = ids[ids > 0]
    out = np.zeros_like(lbl, dtype=np.int32)
    for i, old in enumerate(ids, start=1):
        out[lbl == old] = i
    return out


def remove_small_labels(lbl, min_size):
    out = np.zeros_like(lbl, dtype=np.int32)
    props = measure.regionprops(lbl)
    keep = [p.label for p in props if p.area >= min_size]
    for i, old in enumerate(keep, start=1):
        out[lbl == old] = i
    return out


# =========================================================
# nucleus hybrid segmentation:
# Keep small nuclei intact and split only large nuclear clusters
# =========================================================
def nuclei_segmentation_hybrid(
    blue_norm,
    min_area=20,
    percentile_hi=98.8,
    erosion_radius=2,
    split_area_threshold=220,
    split_min_distance=6
):
    """
    Hybrid nuclear segmentation strategy:
    1) First generate a small and conservative nucleus binary mask
    2) For small objects: retain directly without splitting
    3) For large objects: perform one controlled watershed split within the object
    """

    sm = filters.gaussian(blue_norm, sigma=1.2)
    vals = sm[sm > 0]
    if vals.size == 0:
        return np.zeros_like(blue_norm, dtype=np.int32)

    # Conservative small-nucleus handling
    thr = np.percentile(vals, percentile_hi)
    bw = sm > thr
    bw = morphology.remove_small_objects(bw, min_size=max(8, min_area // 2))
    bw = ndi.binary_fill_holes(bw)

    if erosion_radius > 0:
        bw = morphology.binary_erosion(bw, morphology.disk(erosion_radius))

    bw = morphology.remove_small_objects(bw, min_size=min_area)
    bw = ndi.binary_fill_holes(bw)

    if bw.sum() == 0:
        return np.zeros_like(blue_norm, dtype=np.int32)

    cc = measure.label(bw)
    out = np.zeros_like(cc, dtype=np.int32)
    next_id = 1

    for region in measure.regionprops(cc):
        lab = region.label
        area = region.area
        m = (cc == lab)

        # Small objects are retained as individual nuclei
        if area < split_area_threshold:
            out[m] = next_id
            next_id += 1
            continue

        # Large objects are split only within the object boundary
        dist = ndi.distance_transform_edt(m)

        coords = feature.peak_local_max(
            dist,
            labels=m,
            min_distance=split_min_distance,
            footprint=np.ones((9, 9)),
            exclude_border=False
        )

        # Too few peaks indicate a single nucleus; do not split
        if coords.shape[0] <= 1:
            out[m] = next_id
            next_id += 1
            continue

        markers = np.zeros_like(m, dtype=np.int32)
        for i, (rr, cc_) in enumerate(coords, start=1):
            markers[rr, cc_] = i

        ws = segmentation.watershed(-dist, markers, mask=m)

        # Remove very small fragments
        ws = remove_small_labels(ws, min_area)

        labs = np.unique(ws)
        labs = labs[labs > 0]

        # If <=1 object remains after cleanup, do not split
        if len(labs) <= 1:
            out[m] = next_id
            next_id += 1
            continue

        for wlab in labs:
            out[ws == wlab] = next_id
            next_id += 1

    return relabel_sequential(out)


def subtract_red_background(red_norm, sigma_bg=10):
    bg = filters.gaussian(red_norm, sigma=sigma_bg)
    fg = red_norm - bg
    fg[fg < 0] = 0
    if fg.max() > 0:
        fg = fg / fg.max()
    return fg, bg


# =========================================================
# Cell segmentation: retain the validated segmentation procedure unchanged
# =========================================================
def build_cell_candidate_mask(red_norm, nuclei_lbl, min_cell_area=50, sigma_bg=10):
    red_fg, red_bg = subtract_red_background(red_norm, sigma_bg=sigma_bg)

    vals = red_fg[red_fg > 0]
    if vals.size == 0:
        return np.zeros_like(red_norm, dtype=bool), red_fg, red_bg

    thr = np.percentile(vals, 60)
    bw = red_fg > thr

    bw = morphology.binary_opening(bw, morphology.disk(1))
    bw = morphology.binary_closing(bw, morphology.disk(2))
    bw = ndi.binary_fill_holes(bw)
    bw = morphology.remove_small_objects(bw, min_size=min_cell_area)

    labeled = measure.label(bw)
    out = np.zeros_like(bw, dtype=bool)

    for region in measure.regionprops(labeled):
        coords = region.coords
        rr, cc = coords[:, 0], coords[:, 1]
        if np.any(nuclei_lbl[rr, cc] > 0):
            out[rr, cc] = True

    out = out | morphology.binary_dilation(nuclei_lbl > 0, morphology.disk(2))
    out = ndi.binary_fill_holes(out)
    out = morphology.remove_small_objects(out, min_size=min_cell_area)

    return out, red_fg, red_bg


def segment_cells_from_candidate_mask(cell_mask, nuclei_lbl, min_cell_area=50):
    if nuclei_lbl.max() == 0:
        return np.zeros_like(nuclei_lbl, dtype=np.int32)

    dist = ndi.distance_transform_edt(cell_mask)
    cell_lbl = segmentation.watershed(-dist, markers=nuclei_lbl, mask=cell_mask)
    cell_lbl = remove_small_labels(cell_lbl, min_size=min_cell_area)
    return relabel_sequential(cell_lbl)


def save_preview(red_norm, blue_norm, nuclei_lbl, cell_lbl, out_png):
    rgb = np.dstack([red_norm, np.zeros_like(red_norm), blue_norm])
    nuc_bd = segmentation.find_boundaries(nuclei_lbl, mode="outer")
    cell_bd = segmentation.find_boundaries(cell_lbl, mode="outer")

    show = rgb.copy()
    show[cell_bd] = [1, 1, 0]   # yellow
    show[nuc_bd] = [0, 1, 1]    # cyan

    plt.figure(figsize=(8, 8))
    plt.imshow(show)
    plt.axis("off")
    plt.title("Crop segmentation preview (cyan=nuclei, yellow=cells)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_qc_panel(red_norm, blue_norm, red_fg, red_bg, nuclei_lbl, cell_mask, out_png):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].imshow(blue_norm, cmap="gray")
    axes[0].set_title("blue_norm")

    axes[1].imshow(red_norm, cmap="gray")
    axes[1].set_title("red_norm")

    axes[2].imshow(red_bg, cmap="gray")
    axes[2].set_title("red_background")

    axes[3].imshow(red_fg, cmap="gray")
    axes[3].set_title("red_foreground")

    axes[4].imshow(color.label2rgb(nuclei_lbl, bg_label=0, bg_color=(0, 0, 0)))
    axes[4].set_title("nuclei_labels_preview")

    axes[5].imshow(cell_mask, cmap="gray")
    axes[5].set_title("cell_candidate_mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_label_preview(lbl, out_png):
    rgb = color.label2rgb(lbl, bg_label=0, bg_color=(0, 0, 0))
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blue-image", required=True)
    ap.add_argument("--red-image", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--x0", type=int, required=True)
    ap.add_argument("--y0", type=int, required=True)
    ap.add_argument("--w", type=int, default=2048)
    ap.add_argument("--h", type=int, default=2048)

    # nucleus only
    ap.add_argument("--min-nucleus-area", type=int, default=20)
    ap.add_argument("--nucleus-percentile", type=float, default=98.8)
    ap.add_argument("--nucleus-erosion", type=int, default=2)
    ap.add_argument("--split-area-threshold", type=int, default=220)
    ap.add_argument("--split-min-distance", type=int, default=6)

    # cell fixed
    ap.add_argument("--min-cell-area", type=int, default=50)
    ap.add_argument("--red-bg-sigma", type=float, default=10)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    blue = np.squeeze(tifffile.imread(args.blue_image))
    red = np.squeeze(tifffile.imread(args.red_image))

    y1 = min(blue.shape[0], args.y0 + args.h)
    x1 = min(blue.shape[1], args.x0 + args.w)

    blue = blue[args.y0:y1, args.x0:x1]
    red = red[args.y0:y1, args.x0:x1]

    blue_norm = normalize_channel(blue)
    red_norm = normalize_channel(red)

    nuclei_lbl = nuclei_segmentation_hybrid(
        blue_norm,
        min_area=args.min_nucleus_area,
        percentile_hi=args.nucleus_percentile,
        erosion_radius=args.nucleus_erosion,
        split_area_threshold=args.split_area_threshold,
        split_min_distance=args.split_min_distance
    )

    cell_mask, red_fg, red_bg = build_cell_candidate_mask(
        red_norm,
        nuclei_lbl,
        min_cell_area=args.min_cell_area,
        sigma_bg=args.red_bg_sigma
    )

    cell_lbl = segment_cells_from_candidate_mask(
        cell_mask,
        nuclei_lbl,
        min_cell_area=args.min_cell_area
    )

    tifffile.imwrite(os.path.join(args.outdir, "crop_blue_norm.tif"), (blue_norm * 65535).astype(np.uint16))
    tifffile.imwrite(os.path.join(args.outdir, "crop_red_norm.tif"), (red_norm * 65535).astype(np.uint16))
    tifffile.imwrite(os.path.join(args.outdir, "crop_red_foreground.tif"), (red_fg * 65535).astype(np.uint16))
    tifffile.imwrite(os.path.join(args.outdir, "crop_red_background.tif"), (red_bg * 65535).astype(np.uint16))
    tifffile.imwrite(os.path.join(args.outdir, "crop_cell_candidate_mask.tif"), (cell_mask.astype(np.uint8) * 255))
    tifffile.imwrite(os.path.join(args.outdir, "crop_nuclei_labels.tif"), nuclei_lbl.astype(np.int32))
    tifffile.imwrite(os.path.join(args.outdir, "crop_cell_labels.tif"), cell_lbl.astype(np.int32))

    save_preview(
        red_norm, blue_norm, nuclei_lbl, cell_lbl,
        os.path.join(args.outdir, "crop_segmentation_preview.png")
    )
    save_qc_panel(
        red_norm, blue_norm, red_fg, red_bg, nuclei_lbl, cell_mask,
        os.path.join(args.outdir, "crop_qc_panel.png")
    )
    save_label_preview(
        nuclei_lbl, os.path.join(args.outdir, "crop_nuclei_labels_preview.png")
    )
    save_label_preview(
        cell_lbl, os.path.join(args.outdir, "crop_cell_labels_preview.png")
    )

    with open(os.path.join(args.outdir, "crop_info.txt"), "w") as f:
        f.write(f"x0\t{args.x0}\n")
        f.write(f"y0\t{args.y0}\n")
        f.write(f"w\t{args.w}\n")
        f.write(f"h\t{args.h}\n")
        f.write(f"x1\t{x1}\n")
        f.write(f"y1\t{y1}\n")
        f.write(f"min_nucleus_area\t{args.min_nucleus_area}\n")
        f.write(f"nucleus_percentile\t{args.nucleus_percentile}\n")
        f.write(f"nucleus_erosion\t{args.nucleus_erosion}\n")
        f.write(f"split_area_threshold\t{args.split_area_threshold}\n")
        f.write(f"split_min_distance\t{args.split_min_distance}\n")
        f.write(f"min_cell_area\t{args.min_cell_area}\n")
        f.write(f"red_bg_sigma\t{args.red_bg_sigma}\n")

    print("[OK] hybrid nuclei segmentation finished")
    print(f"[OK] output dir: {args.outdir}")


if __name__ == "__main__":
    main()
