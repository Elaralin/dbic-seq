#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage import exposure, filters, morphology, measure
from skimage.morphology import reconstruction
from cellpose import models


print("### USING OFFICIAL C416 V3 ULTRA-TIGHT SCRIPT ###", flush=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_unit_float(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32, copy=False)


def normalize(img, p1=1, p99=99.5):
    img = np.asarray(img, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(img, [p1, p99])
    if hi <= lo:
        hi = lo + 1.0
    out = (img - lo) / (hi - lo)
    return safe_unit_float(out)


def crop_with_bounds(img, cx, cy, half_size):
    h, w = img.shape[:2]
    x1 = max(0, int(round(cx)) - half_size)
    x2 = min(w, int(round(cx)) + half_size)
    y1 = max(0, int(round(cy)) - half_size)
    y2 = min(h, int(round(cy)) + half_size)
    crop = img[y1:y2, x1:x2]
    return crop, x1, y1, x2, y2


def border_touch_fraction(mask_bool):
    if mask_bool.sum() == 0:
        return 1.0
    border = np.zeros_like(mask_bool, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return float((mask_bool & border).sum()) / float(mask_bool.sum())


def build_actin_probability(red_crop):
    r = normalize(red_crop)

    r_blur = filters.gaussian(r, sigma=0.8, preserve_range=True)
    r_blur = safe_unit_float(r_blur)

    r_clahe = exposure.equalize_adapthist(r_blur, clip_limit=0.01)
    r_clahe = safe_unit_float(r_clahe)

    bg = filters.gaussian(r_blur, sigma=6.0, preserve_range=True)
    bg = safe_unit_float(bg)

    hp = r_blur - bg
    hp = hp - np.min(hp)
    mx = np.max(hp)
    if mx > 0:
        hp = hp / mx
    hp = safe_unit_float(hp)

    try:
        ridge = filters.meijering(r_clahe, sigmas=range(1, 5), black_ridges=False)
        ridge = safe_unit_float(ridge)
    except Exception:
        ridge = hp.copy()

    prob = 0.50 * r_clahe + 0.25 * hp + 0.25 * ridge
    prob = safe_unit_float(prob)
    return prob, r_clahe, ridge


def build_edge_map(red_crop):
    r = normalize(red_crop)
    r = filters.gaussian(r, sigma=1.0, preserve_range=True)
    r = safe_unit_float(r)

    e1 = filters.sobel(r)
    e2 = filters.scharr(r)
    edge = 0.5 * e1 + 0.5 * e2
    edge = safe_unit_float(edge)

    edge = exposure.equalize_adapthist(edge, clip_limit=0.01)
    edge = safe_unit_float(edge)
    return edge


def build_cellpose_rgb(red_crop, blue_crop, mode="raw"):
    r = normalize(red_crop)
    b = normalize(blue_crop)

    if mode == "enhanced":
        r = exposure.equalize_adapthist(r, clip_limit=0.01)
        r = safe_unit_float(r)
        b = exposure.equalize_adapthist(b, clip_limit=0.01)
        b = safe_unit_float(b)

        r = filters.gaussian(r, sigma=0.6, preserve_range=True)
        r = safe_unit_float(r)
        b = filters.gaussian(b, sigma=0.6, preserve_range=True)
        b = safe_unit_float(b)

    rgb = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = r
    rgb[..., 2] = b
    rgb = safe_unit_float(rgb)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    return rgb, r, b


def score_cellpose_candidate(mask_lbl, red_norm, nuc_local_x, nuc_local_y):
    labels = np.unique(mask_lbl)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return None

    h, w = mask_lbl.shape[:2]
    nx = int(round(nuc_local_x))
    ny = int(round(nuc_local_y))

    best = None
    best_score = -1e9

    for lab in labels:
        m = (mask_lbl == lab)
        area = int(m.sum())
        if area == 0:
            continue

        contains_nucleus = (0 <= nx < w and 0 <= ny < h and bool(m[ny, nx]))
        if not contains_nucleus:
            continue

        border_frac = border_touch_fraction(m)
        red_in = float(red_norm[m].mean()) if m.any() else 0.0
        red_out = float(red_norm[~m].mean()) if (~m).any() else 0.0
        contrast = red_in - red_out

        prop = measure.regionprops(m.astype(np.uint8))[0]
        eccentricity = float(prop.eccentricity) if prop.eccentricity is not None else 0.0
        solidity = float(prop.solidity) if prop.solidity is not None else 0.0

        score = 0.0
        score += 5.0
        score += 3.0 * contrast
        score += 0.4 * solidity
        score -= 2.5 * border_frac

        if area < 150:
            score -= 2.0
        if area > 30000:
            score -= 2.0

        rec = {
            "label": int(lab),
            "area": area,
            "contains_nucleus": contains_nucleus,
            "border_frac": border_frac,
            "contrast": contrast,
            "eccentricity": eccentricity,
            "solidity": solidity,
            "score": score
        }

        if score > best_score:
            best_score = score
            best = rec

    return best


def refine_with_seeded_growth(
    cp_mask,
    nucleus_mask,
    actin_prob,
    ridge,
    grow_low=0.30,
    ridge_low=0.22,
    max_growth_ratio=4.0
):
    seed = (cp_mask | morphology.binary_dilation(nucleus_mask, morphology.disk(2)))

    allowed = ((actin_prob > grow_low) & (ridge > ridge_low * 0.85)) | seed

    grown = reconstruction(
        seed.astype(np.uint8),
        allowed.astype(np.uint8),
        method="dilation"
    ) > 0

    grown = morphology.remove_small_holes(grown, area_threshold=64)

    lbl = measure.label(grown)
    if lbl.max() == 0:
        return cp_mask.copy()

    nuc_ys, nuc_xs = np.where(nucleus_mask)
    if len(nuc_xs) == 0:
        return cp_mask.copy()

    ny = int(np.median(nuc_ys))
    nx = int(np.median(nuc_xs))

    if lbl[ny, nx] == 0:
        return cp_mask.copy()

    keep = (lbl == lbl[ny, nx])

    cp_area = max(int(cp_mask.sum()), 1)
    grown_area = int(keep.sum())
    if grown_area > cp_area * max_growth_ratio:
        return cp_mask.copy()

    return keep


def cleanup_mask_shape(mask_bool, min_obj=40, min_hole=40):
    m = mask_bool.astype(bool)
    m = morphology.remove_small_objects(m, min_size=min_obj)
    m = morphology.remove_small_holes(m, area_threshold=min_hole)
    m = morphology.binary_opening(m, morphology.disk(1))
    m = morphology.binary_closing(m, morphology.disk(1))

    lbl = measure.label(m)
    if lbl.max() > 0:
        props = sorted(measure.regionprops(lbl), key=lambda x: x.area, reverse=True)
        m = (lbl == props[0].label)

    m = morphology.remove_small_holes(m, area_threshold=min_hole)
    return m


def snap_mask_to_edges(mask_bool, nucleus_mask, edge_map, max_dist=1, edge_low=0.26):
    m = mask_bool.astype(bool)
    if m.sum() == 0:
        return m

    dil = morphology.binary_dilation(m, morphology.disk(max_dist))
    ero = morphology.binary_erosion(m, morphology.disk(max_dist))
    band = dil ^ ero

    edge_support = edge_map > edge_low
    allowed = m | (band & edge_support)

    seed = morphology.binary_dilation(nucleus_mask, morphology.disk(1)) | morphology.binary_erosion(m, morphology.disk(1))
    seed = seed & allowed

    snapped = reconstruction(
        seed.astype(np.uint8),
        allowed.astype(np.uint8),
        method="dilation"
    ) > 0

    snapped = cleanup_mask_shape(snapped, min_obj=40, min_hole=40)
    return snapped


def tighten_boundary_preserve_shape(mask_bool, nucleus_mask, edge_map, edge_low=0.26):
    m = mask_bool.astype(bool)
    if m.sum() == 0:
        return m

    eroded = morphology.binary_erosion(m, morphology.disk(1))

    dil = morphology.binary_dilation(eroded, morphology.disk(1))
    recover_allowed = m & dil & (edge_map > edge_low)

    seed = eroded | morphology.binary_dilation(nucleus_mask, morphology.disk(1))
    seed = seed & (eroded | recover_allowed)

    out = reconstruction(
        seed.astype(np.uint8),
        (eroded | recover_allowed).astype(np.uint8),
        method="dilation"
    ) > 0

    out = cleanup_mask_shape(out, min_obj=40, min_hole=40)
    return out


def build_nucleus_mask_local(nuc_label_crop, nucleus_id):
    return (nuc_label_crop == nucleus_id)


def save_overlay(out_png, red_crop, blue_crop, mask_bool, spot_local, nuc_local, title_text):
    r = normalize(red_crop)
    b = normalize(blue_crop)

    rgb = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = r
    rgb[..., 2] = b
    rgb = safe_unit_float(rgb)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(rgb)

    if mask_bool is not None and mask_bool.sum() > 0:
        ax.contour(mask_bool.astype(float), levels=[0.5], colors='yellow', linewidths=1.0)

    sx, sy = spot_local
    nx, ny = nuc_local
    ax.scatter([sx], [sy], s=18, c='cyan', marker='x')
    ax.scatter([nx], [ny], s=16, c='lime', marker='o')

    ax.set_title(title_text, fontsize=7)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_cellpose_model():
    if hasattr(models, "CellposeModel"):
        try:
            return models.CellposeModel(gpu=False, model_type="cyto")
        except TypeError:
            return models.CellposeModel(gpu=False)

    if hasattr(models, "Cellpose"):
        try:
            return models.Cellpose(gpu=False, model_type="cyto")
        except TypeError:
            return models.Cellpose(gpu=False)

    raise AttributeError("Neither CellposeModel nor Cellpose is available in cellpose.models")


def run_cellpose_eval(model, rgb_input, diameter, cellprob_threshold, flow_threshold):
    rgb_input = safe_unit_float(rgb_input)
    rgb_input = np.ascontiguousarray(rgb_input, dtype=np.float32)

    out = model.eval(
        rgb_input,
        channels=[1, 3],
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold
    )

    if isinstance(out, (tuple, list)):
        masks = out[0]
    else:
        masks = out

    return masks


def process_one(row, blue_full, red_full, nuc_labels_full, model, args, out_dirs):
    spot_id = row["spot_id"]
    spot_x = float(row["spot_x"])
    spot_y = float(row["spot_y"])
    nucleus_id = int(row["nucleus_id"])
    nuc_x = float(row["nucleus_x"])
    nuc_y = float(row["nucleus_y"])

    match_status = str(row["status"])
    match_confidence = "high" if match_status == "matched_r50" else "rescue"

    best = None
    best_mask = None
    best_red = None
    best_blue = None
    best_crop = None

    for half in args.context_halves:
        red_crop, x1, y1, x2, y2 = crop_with_bounds(red_full, nuc_x, nuc_y, half)
        blue_crop, _, _, _, _ = crop_with_bounds(blue_full, nuc_x, nuc_y, half)
        nuc_crop, _, _, _, _ = crop_with_bounds(nuc_labels_full, nuc_x, nuc_y, half)

        if red_crop.size == 0 or blue_crop.size == 0:
            continue

        nucleus_mask = build_nucleus_mask_local(nuc_crop, nucleus_id)
        if nucleus_mask.sum() == 0:
            continue

        nuc_local_x = nuc_x - x1
        nuc_local_y = nuc_y - y1
        spot_local_x = spot_x - x1
        spot_local_y = spot_y - y1

        actin_prob, red_clahe, ridge = build_actin_probability(red_crop)
        edge_map = build_edge_map(red_crop)

        local_best = None
        local_final_mask = None

        for mode in args.modes:
            rgb_input, red_norm, blue_norm = build_cellpose_rgb(red_crop, blue_crop, mode=mode)

            for diam in args.diameters:
                try:
                    masks = run_cellpose_eval(
                        model=model,
                        rgb_input=rgb_input,
                        diameter=diam,
                        cellprob_threshold=args.cellprob_threshold,
                        flow_threshold=args.flow_threshold
                    )
                except Exception:
                    continue

                if masks is None or np.max(masks) == 0:
                    continue

                cp_best = score_cellpose_candidate(masks, red_norm, nuc_local_x, nuc_local_y)
                if cp_best is None:
                    continue

                cp_mask = (masks == cp_best["label"])

                grown_mask = refine_with_seeded_growth(
                    cp_mask=cp_mask,
                    nucleus_mask=nucleus_mask,
                    actin_prob=actin_prob,
                    ridge=ridge,
                    grow_low=args.grow_low,
                    ridge_low=args.ridge_low,
                    max_growth_ratio=args.max_growth_ratio
                )

                cleaned_mask = cleanup_mask_shape(
                    grown_mask,
                    min_obj=args.cleanup_min_obj,
                    min_hole=args.cleanup_min_hole
                )

                snapped_mask = snap_mask_to_edges(
                    cleaned_mask,
                    nucleus_mask=nucleus_mask,
                    edge_map=edge_map,
                    max_dist=args.snap_max_dist,
                    edge_low=args.edge_low
                )

                final_mask = tighten_boundary_preserve_shape(
                    snapped_mask,
                    nucleus_mask=nucleus_mask,
                    edge_map=edge_map,
                    edge_low=args.edge_low
                )

                final_border = border_touch_fraction(final_mask)
                final_area = int(final_mask.sum())

                final_score = cp_best["score"]
                final_score += 0.4 * np.log1p(final_area)
                final_score -= 2.0 * final_border

                rec = {
                    "spot_id": spot_id,
                    "status": "ok",
                    "match_status": match_status,
                    "match_confidence": match_confidence,
                    "spot_x": spot_x,
                    "spot_y": spot_y,
                    "nucleus_id": nucleus_id,
                    "nucleus_x": nuc_x,
                    "nucleus_y": nuc_y,
                    "roi_x1": x1,
                    "roi_y1": y1,
                    "roi_x2": x2,
                    "roi_y2": y2,
                    "roi_half": half,
                    "mode": mode,
                    "diameter": diam,
                    "cp_area": int(cp_mask.sum()),
                    "final_area": final_area,
                    "cp_border_frac": float(cp_best["border_frac"]),
                    "final_border_frac": float(final_border),
                    "cp_contrast": float(cp_best["contrast"]),
                    "cp_solidity": float(cp_best["solidity"]),
                    "cp_eccentricity": float(cp_best["eccentricity"]),
                    "score": float(final_score)
                }

                if local_best is None or rec["score"] > local_best["score"]:
                    local_best = rec
                    local_final_mask = final_mask.copy()

        if local_best is None:
            continue

        if best is None or local_best["score"] > best["score"]:
            best = local_best
            best_mask = local_final_mask
            best_red = red_crop.copy()
            best_blue = blue_crop.copy()
            best_crop = {
                "x1": x1,
                "y1": y1,
                "spot_local_x": spot_local_x,
                "spot_local_y": spot_local_y,
                "nuc_local_x": nuc_local_x,
                "nuc_local_y": nuc_local_y
            }

        if local_best["final_border_frac"] <= args.accept_border_frac:
            break

    if best is None:
        return {
            "spot_id": spot_id,
            "status": "fail_no_valid_mask",
            "match_status": match_status,
            "match_confidence": match_confidence,
            "spot_x": spot_x,
            "spot_y": spot_y,
            "nucleus_id": nucleus_id,
            "nucleus_x": nuc_x,
            "nucleus_y": nuc_y
        }

    mask_file = os.path.join(out_dirs["masks"], f"{spot_id}_mask.tif")
    overlay_file = os.path.join(out_dirs["overlays"], f"{spot_id}_overlay.png")
    roi_file = os.path.join(out_dirs["roi_rgb"], f"{spot_id}_roi_rgb.tif")

    tifffile.imwrite(mask_file, (best_mask.astype(np.uint8) * 255), compression="zlib")

    rr = normalize(best_red)
    bb = normalize(best_blue)
    rgb = np.zeros((rr.shape[0], rr.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = rr
    rgb[..., 2] = bb
    rgb = safe_unit_float(rgb)
    tifffile.imwrite(roi_file, (rgb * 255).astype(np.uint8), compression="zlib")

    title = (
        f"{spot_id} | {match_confidence} | mode={best['mode']} | "
        f"diam={best['diameter']} | score={best['score']:.2f} | "
        f"border={best['final_border_frac']:.3f}"
    )

    save_overlay(
        overlay_file,
        best_red,
        best_blue,
        best_mask,
        (best_crop["spot_local_x"], best_crop["spot_local_y"]),
        (best_crop["nuc_local_x"], best_crop["nuc_local_y"]),
        title
    )

    best["mask_file"] = mask_file
    best["overlay_file"] = overlay_file
    best["roi_file"] = roi_file
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blue", required=True)
    ap.add_argument("--red", required=True)
    ap.add_argument("--nuclei_labels", required=True)
    ap.add_argument("--matched_spots", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--context_halves", type=int, nargs="+", default=[140, 180, 240, 300])
    ap.add_argument("--diameters", type=int, nargs="+", default=[30, 40, 50])
    ap.add_argument("--modes", nargs="+", default=["raw", "enhanced"])

    ap.add_argument("--cellprob_threshold", type=float, default=-1.0)
    ap.add_argument("--flow_threshold", type=float, default=0.4)

    ap.add_argument("--grow_low", type=float, default=0.30)
    ap.add_argument("--ridge_low", type=float, default=0.22)
    ap.add_argument("--max_growth_ratio", type=float, default=4.0)

    ap.add_argument("--cleanup_min_obj", type=int, default=40)
    ap.add_argument("--cleanup_min_hole", type=int, default=40)
    ap.add_argument("--snap_max_dist", type=int, default=1)
    ap.add_argument("--edge_low", type=float, default=0.26)

    ap.add_argument("--accept_border_frac", type=float, default=0.005)

    ap.add_argument("--chunk_id", type=int, default=0)
    ap.add_argument("--n_chunks", type=int, default=1)

    args = ap.parse_args()

    out_dirs = {
        "masks": os.path.join(args.outdir, "masks"),
        "overlays": os.path.join(args.outdir, "overlays"),
        "roi_rgb": os.path.join(args.outdir, "roi_rgb"),
        "metadata": os.path.join(args.outdir, "metadata"),
    }
    for d in out_dirs.values():
        ensure_dir(d)

    blue_full = tifffile.imread(args.blue)
    red_full = tifffile.imread(args.red)
    nuc_labels_full = tifffile.imread(args.nuclei_labels)

    match_df = pd.read_csv(args.matched_spots, sep="\t")
    match_df = match_df[
        match_df["status"].isin(["matched_r50", "matched_r80_rescue"])
    ].copy().reset_index(drop=True)

    idx = np.arange(len(match_df))
    sub = match_df.loc[(idx % args.n_chunks) == args.chunk_id].copy().reset_index(drop=True)

    print(f"[INFO] matched usable spots total = {len(match_df)}", flush=True)
    print(f"[INFO] chunk {args.chunk_id}/{args.n_chunks}, n = {len(sub)}", flush=True)
    print(match_df["status"].value_counts(dropna=False), flush=True)

    model = make_cellpose_model()

    records = []
    for i, (_, row) in enumerate(sub.iterrows(), start=1):
        print(f"[{i}/{len(sub)}] {row['spot_id']} | {row['status']}", flush=True)
        try:
            rec = process_one(row, blue_full, red_full, nuc_labels_full, model, args, out_dirs)
        except Exception as e:
            rec = {
                "spot_id": row["spot_id"],
                "status": f"error: {repr(e)}",
                "match_status": row["status"],
                "match_confidence": "high" if row["status"] == "matched_r50" else "rescue"
            }
        records.append(rec)

    meta_file = os.path.join(out_dirs["metadata"], f"segmentation_metadata.chunk{args.chunk_id:03d}.tsv")
    pd.DataFrame(records).to_csv(meta_file, sep="\t", index=False)
    print("[OK] wrote:", meta_file, flush=True)


if __name__ == "__main__":
    main()
