#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from scipy import ndimage as ndi
from skimage import filters, morphology, segmentation, feature, measure


VALID_OVERRIDE_KEYS = {
    "min_nucleus_area",
    "nucleus_percentile",
    "nucleus_erosion",
    "split_area_threshold",
    "split_min_distance",
    "nucleus_gaussian_sigma",
    "nucleus_post_dilate",
    "split_min_child_area",
    "split_min_area_ratio",
    "nuc_abs_min",
    "min_cell_area",
    "red_bg_sigma",
    "cell_fg_percentile",
    "cell_post_erode",
    "max_nucleus_to_cell_distance",
    "spot_min_overlap_nucleus",
    "spot_min_overlap_cell",
    "count_min_nucleus_area",
    "count_min_cell_area",
    "count_min_cell_nucleus_area_ratio",
    "count_max_nucleus_fraction",
    # new: valid nucleus filtering
    "valid_nucleus_min_area_abs",
    "valid_nucleus_min_area_ratio_to_largest",
    "valid_nucleus_red_support_dilate_radius",
    "valid_nucleus_min_red_overlap_pixels",
    "valid_nucleus_min_red_overlap_ratio",
}


# =========================
# basic utils
# =========================
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


def parse_bool_like(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return x


def maybe_cast_value(val):
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    sl = s.lower()
    if sl in {"1", "0", "true", "false", "t", "f", "yes", "no", "y", "n"}:
        return parse_bool_like(sl)
    try:
        if any(c in s for c in [".", "e", "E"]):
            return float(s)
        return int(s)
    except Exception:
        return s


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================
# summary row helpers
# =========================
def append_class_summary_rows(df, class_col="spot_class"):
    if df.shape[0] == 0:
        return df

    single_n = int((df[class_col] == "single").sum()) if class_col in df.columns else 0
    empty_n = int((df[class_col] == "empty").sum()) if class_col in df.columns else 0
    multi_n = int((df[class_col] == "doublet_or_multiplet").sum()) if class_col in df.columns else 0
    total_n = int(df[~df.iloc[:, 0].astype(str).str.startswith("SUMMARY_")].shape[0])

    summary_rows = []
    for label, value in [
        ("SUMMARY_total_spots", total_n),
        ("SUMMARY_single_spots", single_n),
        ("SUMMARY_empty_spots", empty_n),
        ("SUMMARY_doublet_or_multiplet_spots", multi_n),
    ]:
        row = {c: "" for c in df.columns}
        if "spot_id" in df.columns:
            row["spot_id"] = label
        else:
            row[df.columns[0]] = label

        if "notes" in df.columns:
            row["notes"] = str(value)
        elif "review_reason" in df.columns:
            row["review_reason"] = str(value)
        elif len(df.columns) >= 2:
            row[df.columns[1]] = str(value)

        summary_rows.append(row)

    return pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)


# =========================
# override table logic
# =========================
def load_spot_overrides(path):
    if path is None:
        return {}

    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "spot_id" not in df.columns:
        raise ValueError(f"override TSV must contain column 'spot_id': {path}")

    df = df[~df["spot_id"].astype(str).str.startswith("SUMMARY_")].copy()

    overrides = {}
    for _, row in df.iterrows():
        spot_id = str(row["spot_id"]).strip()
        if not spot_id:
            continue

        use_override = True
        if "use_override" in df.columns:
            u = maybe_cast_value(row["use_override"])
            if u is not None and bool(u) is False:
                use_override = False

        if not use_override:
            continue

        one = {}
        for col in df.columns:
            if col in {
                "spot_id", "use_override", "notes",
                "current_class", "current_raw_n_nuclei", "current_n_nuclei", "current_n_cells",
                "current_total_nucleus_area", "current_total_cell_area",
                "current_max_nucleus_fraction", "current_min_cell_nucleus_ratio",
                "review_score", "review_reason"
            }:
                continue

            if col not in VALID_OVERRIDE_KEYS:
                continue

            val = maybe_cast_value(row[col])
            if val is None:
                continue
            one[col] = val

        overrides[spot_id] = one

    return overrides


def merge_params(base_params, override_params):
    out = dict(base_params)
    out.update(override_params)
    return out


# =========================
# nuclei: hybrid split
# =========================
def nuclei_segmentation_hybrid(
    blue_norm,
    min_area=20,
    percentile_hi=99.1,
    erosion_radius=2,
    split_area_threshold=420,
    split_min_distance=10,
    nuc_abs_min=0.29,
    gaussian_sigma=0.8,
    post_dilate_radius=0,
    split_min_child_area=30,
    split_min_area_ratio=0.22
):
    sm = filters.gaussian(blue_norm, sigma=gaussian_sigma)
    vals = sm[sm > 0]
    if vals.size == 0:
        return np.zeros_like(blue_norm, dtype=np.int32)

    auto_thr = np.percentile(vals, percentile_hi)
    final_thr = max(auto_thr, nuc_abs_min)

    bw = sm >= final_thr
    bw = morphology.remove_small_objects(bw, min_size=max(8, int(min_area) // 2))
    bw = ndi.binary_fill_holes(bw)

    if erosion_radius > 0:
        bw = morphology.binary_erosion(bw, morphology.disk(int(erosion_radius)))

    if post_dilate_radius > 0:
        bw = morphology.binary_dilation(bw, morphology.disk(int(post_dilate_radius)))

    bw = morphology.remove_small_objects(bw, min_size=int(min_area))
    bw = ndi.binary_fill_holes(bw)

    if bw.sum() == 0:
        return np.zeros_like(blue_norm, dtype=np.int32)

    cc = measure.label(bw)
    out = np.zeros_like(cc, dtype=np.int32)
    next_id = 1

    for region in measure.regionprops(cc):
        area = region.area
        m = (cc == region.label)

        if area < split_area_threshold:
            out[m] = next_id
            next_id += 1
            continue

        dist = ndi.distance_transform_edt(m)

        coords = feature.peak_local_max(
            dist,
            labels=m,
            min_distance=int(split_min_distance),
            footprint=np.ones((11, 11)),
            exclude_border=False
        )

        if coords.shape[0] <= 1:
            out[m] = next_id
            next_id += 1
            continue

        markers = np.zeros_like(m, dtype=np.int32)
        for i, (rr, cc_) in enumerate(coords, start=1):
            markers[rr, cc_] = i

        ws = segmentation.watershed(-dist, markers, mask=m)
        ws = remove_small_labels(ws, int(min_area))

        labs = np.unique(ws)
        labs = labs[labs > 0]

        if len(labs) <= 1:
            out[m] = next_id
            next_id += 1
            continue

        split_props = [p for p in measure.regionprops(ws) if p.label > 0]
        split_areas = sorted([p.area for p in split_props], reverse=True)

        if len(split_areas) >= 2 and min(split_areas) < split_min_child_area:
            out[m] = next_id
            next_id += 1
            continue

        if len(split_areas) >= 2:
            if split_areas[1] / split_areas[0] < split_min_area_ratio:
                out[m] = next_id
                next_id += 1
                continue

        for wlab in labs:
            out[ws == wlab] = next_id
            next_id += 1

    return relabel_sequential(out)


# =========================
# red foreground helpers
# =========================
def subtract_red_background(red_norm, sigma_bg=10):
    bg = filters.gaussian(red_norm, sigma=sigma_bg)
    fg = red_norm - bg
    fg[fg < 0] = 0
    if fg.max() > 0:
        fg = fg / fg.max()
    return fg, bg


def make_red_support_mask(red_fg):
    """
    Used to evaluate red-channel support for each nucleus
    Use a more permissive red foreground mask than that used for cell segmentation,
    # Remove very small fragments to reduce spurious nucleus support from random red-channel noise
    """
    vals = red_fg[red_fg > 0]
    if vals.size == 0:
        return np.zeros_like(red_fg, dtype=bool)

    # Use a permissive threshold while excluding low-level background noise
    thr = max(np.percentile(vals, 40), 0.05)
    bw = red_fg >= thr
    bw = morphology.binary_opening(bw, morphology.disk(1))
    bw = morphology.binary_closing(bw, morphology.disk(1))
    bw = morphology.remove_small_objects(bw, min_size=8)
    return bw


# =========================
# valid nucleus filtering
# =========================
def filter_valid_nuclei_by_size_and_red_support(
    nuclei_lbl,
    red_fg,
    valid_nucleus_min_area_abs=90,
    valid_nucleus_min_area_ratio_to_largest=0.22,
    valid_nucleus_red_support_dilate_radius=6,
    valid_nucleus_min_red_overlap_pixels=25,
    valid_nucleus_min_red_overlap_ratio=0.08
):
    """
    Rules:
    1) Remove nuclei with very small absolute area
    2) Remove nuclei whose area is too small relative to the largest nucleus
    3) Remove nuclei with insufficient overlap with red-channel support after dilation
    """
    if nuclei_lbl.max() == 0:
        empty = np.zeros_like(nuclei_lbl, dtype=np.int32)
        stats = {
            "raw_n_nuclei": 0,
            "filtered_small_abs_nuclei": 0,
            "filtered_small_ratio_nuclei": 0,
            "filtered_low_red_support_nuclei": 0,
            "kept_valid_nuclei": 0,
        }
        return empty, stats, np.zeros_like(nuclei_lbl, dtype=bool)

    props = measure.regionprops(nuclei_lbl)
    if len(props) == 0:
        empty = np.zeros_like(nuclei_lbl, dtype=np.int32)
        stats = {
            "raw_n_nuclei": 0,
            "filtered_small_abs_nuclei": 0,
            "filtered_small_ratio_nuclei": 0,
            "filtered_low_red_support_nuclei": 0,
            "kept_valid_nuclei": 0,
        }
        return empty, stats, np.zeros_like(nuclei_lbl, dtype=bool)

    red_support_mask = make_red_support_mask(red_fg)
    largest_area = max([p.area for p in props])

    out = np.zeros_like(nuclei_lbl, dtype=np.int32)
    next_id = 1

    filtered_small_abs = 0
    filtered_small_ratio = 0
    filtered_low_red_support = 0

    for p in props:
        nid = p.label
        area = int(p.area)
        nmask = (nuclei_lbl == nid)

        # 1) absolute min area
        if area < int(valid_nucleus_min_area_abs):
            filtered_small_abs += 1
            continue

        # 2) relative size to largest
        if largest_area > 0:
            area_ratio = area / float(largest_area)
            if area_ratio < float(valid_nucleus_min_area_ratio_to_largest):
                filtered_small_ratio += 1
                continue

        # 3) red support after dilation
        if int(valid_nucleus_red_support_dilate_radius) > 0:
            dm = morphology.binary_dilation(
                nmask,
                morphology.disk(int(valid_nucleus_red_support_dilate_radius))
            )
        else:
            dm = nmask

        overlap_pixels = int(np.sum(dm & red_support_mask))
        overlap_ratio = overlap_pixels / max(int(np.sum(dm)), 1)

        if overlap_pixels < int(valid_nucleus_min_red_overlap_pixels):
            filtered_low_red_support += 1
            continue

        if overlap_ratio < float(valid_nucleus_min_red_overlap_ratio):
            filtered_low_red_support += 1
            continue

        out[nmask] = next_id
        next_id += 1

    out = relabel_sequential(out)

    stats = {
        "raw_n_nuclei": len(props),
        "filtered_small_abs_nuclei": filtered_small_abs,
        "filtered_small_ratio_nuclei": filtered_small_ratio,
        "filtered_low_red_support_nuclei": filtered_low_red_support,
        "kept_valid_nuclei": int(out.max()),
    }
    return out, stats, red_support_mask


# =========================
# cells
# =========================
def build_cell_candidate_mask(
    red_norm,
    nuclei_lbl,
    min_cell_area=50,
    sigma_bg=10,
    cell_fg_percentile=60,
    cell_post_erode=0,
    max_nucleus_to_cell_distance=None
):
    red_fg, red_bg = subtract_red_background(red_norm, sigma_bg=sigma_bg)

    vals = red_fg[red_fg > 0]
    if vals.size == 0:
        return np.zeros_like(red_norm, dtype=bool), red_fg, red_bg

    thr = np.percentile(vals, float(cell_fg_percentile))
    bw = red_fg > thr

    bw = morphology.binary_opening(bw, morphology.disk(1))
    bw = morphology.binary_closing(bw, morphology.disk(2))
    bw = ndi.binary_fill_holes(bw)

    if cell_post_erode > 0:
        bw = morphology.binary_erosion(bw, morphology.disk(int(cell_post_erode)))

    bw = morphology.remove_small_objects(bw, min_size=int(min_cell_area))

    labeled = measure.label(bw)
    out = np.zeros_like(bw, dtype=bool)

    for region in measure.regionprops(labeled):
        coords = region.coords
        rr, cc = coords[:, 0], coords[:, 1]
        if np.any(nuclei_lbl[rr, cc] > 0):
            out[rr, cc] = True

    out = out | morphology.binary_dilation(nuclei_lbl > 0, morphology.disk(2))
    out = ndi.binary_fill_holes(out)

    if max_nucleus_to_cell_distance is not None:
        dist_to_nuc = ndi.distance_transform_edt(nuclei_lbl == 0)
        near_nuc = dist_to_nuc <= float(max_nucleus_to_cell_distance)
        out = out & near_nuc

    out = morphology.remove_small_objects(out, min_size=int(min_cell_area))

    return out, red_fg, red_bg


def segment_cells_from_candidate_mask(cell_mask, nuclei_lbl, min_cell_area=50):
    if nuclei_lbl.max() == 0:
        return np.zeros_like(nuclei_lbl, dtype=np.int32)

    dist = ndi.distance_transform_edt(cell_mask)
    cell_lbl = segmentation.watershed(-dist, markers=nuclei_lbl, mask=cell_mask)
    cell_lbl = remove_small_labels(cell_lbl, min_size=int(min_cell_area))
    return relabel_sequential(cell_lbl)


# =========================
# grid parsing
# =========================
def pick_first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def standardize_grid(df):
    row_colmap = {}
    c = pick_first_existing(df, ["row", "spot_row", "grid_row", "A_index", "a_idx"])
    if c is not None:
        row_colmap[c] = "row"

    c = pick_first_existing(df, ["col", "spot_col", "grid_col", "B_index", "b_idx"])
    if c is not None:
        row_colmap[c] = "col"

    c = pick_first_existing(df, ["spot_id", "spot", "barcode", "spot_name"])
    if c is not None:
        row_colmap[c] = "spot_id"

    df = df.rename(columns=row_colmap)

    if "spot_id" not in df.columns:
        if "row" in df.columns and "col" in df.columns:
            df["spot_id"] = [f"A{int(r)}-B{int(c)}" for r, c in zip(df["row"], df["col"])]
        else:
            df["spot_id"] = [f"spot_{i+1}" for i in range(len(df))]

    rename_map = {}

    c = pick_first_existing(df, ["x0", "xmin", "x_min", "left"])
    if c is not None:
        rename_map[c] = "x0"

    c = pick_first_existing(df, ["x1", "xmax", "x_max", "right"])
    if c is not None:
        rename_map[c] = "x1"

    c = pick_first_existing(df, ["y0", "ymin", "y_min", "top"])
    if c is not None:
        rename_map[c] = "y0"

    c = pick_first_existing(df, ["y1", "ymax", "y_max", "bottom"])
    if c is not None:
        rename_map[c] = "y1"

    c = pick_first_existing(df, ["x_center", "center_x", "xc", "xcent"])
    if c is not None:
        rename_map[c] = "x_center"

    c = pick_first_existing(df, ["y_center", "center_y", "yc", "ycent"])
    if c is not None:
        rename_map[c] = "y_center"

    df = df.rename(columns=rename_map)

    required_box = {"x0", "x1", "y0", "y1", "x_center", "y_center"}
    if required_box.issubset(df.columns):
        return df

    xcorners = [
        pick_first_existing(df, ["tl_x", "x_tl", "TL_x", "TLX"]),
        pick_first_existing(df, ["tr_x", "x_tr", "TR_x", "TRX"]),
        pick_first_existing(df, ["bl_x", "x_bl", "BL_x", "BLX"]),
        pick_first_existing(df, ["br_x", "x_br", "BR_x", "BRX"]),
    ]
    ycorners = [
        pick_first_existing(df, ["tl_y", "y_tl", "TL_y", "TLY"]),
        pick_first_existing(df, ["tr_y", "y_tr", "TR_y", "TRY"]),
        pick_first_existing(df, ["bl_y", "y_bl", "BL_y", "BLY"]),
        pick_first_existing(df, ["br_y", "y_br", "BR_y", "BRY"]),
    ]

    if all(c is not None for c in xcorners + ycorners):
        xs = df[xcorners].values.astype(float)
        ys = df[ycorners].values.astype(float)
        df["x0"] = xs.min(axis=1)
        df["x1"] = xs.max(axis=1)
        df["y0"] = ys.min(axis=1)
        df["y1"] = ys.max(axis=1)
        df["x_center"] = xs.mean(axis=1)
        df["y_center"] = ys.mean(axis=1)
        return df

    missing = required_box - set(df.columns)
    raise ValueError(
        f"grid file missing columns after standardization: {missing}\n"
        f"available columns: {df.columns.tolist()}"
    )


def load_grid(grid_tsv):
    df = pd.read_csv(grid_tsv, sep="\t")
    df = standardize_grid(df)

    required = {"spot_id", "x_center", "y_center", "x0", "x1", "y0", "y1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"grid file still missing columns: {missing}")

    if "row" not in df.columns:
        df["row"] = np.arange(1, len(df) + 1)
    if "col" not in df.columns:
        df["col"] = 1

    return df


# =========================
# spot utils
# =========================
def crop_with_margin(img, x0, x1, y0, y1, margin, H, W):
    xx0 = max(0, int(np.floor(x0 - margin)))
    xx1 = min(W, int(np.ceil(x1 + margin)))
    yy0 = max(0, int(np.floor(y0 - margin)))
    yy1 = min(H, int(np.ceil(y1 + margin)))
    return img[yy0:yy1, xx0:xx1], xx0, yy0


def labels_in_box_by_centroid_and_area(
    lbl,
    x0, x1, y0, y1,
    min_overlap=1,
    min_area_for_count=1,
    min_cell_nucleus_area_ratio=None,
    nuclei_lbl=None,
    max_nucleus_fraction=None
):
    keep = []

    for region in measure.regionprops(lbl):
        cy, cx = region.centroid
        area = region.area

        if area < min_area_for_count:
            continue

        if not (y0 <= cy < y1 and x0 <= cx < x1):
            continue

        rr = region.coords[:, 0]
        cc = region.coords[:, 1]
        overlap = np.sum(
            (rr >= y0) & (rr < y1) &
            (cc >= x0) & (cc < x1)
        )

        if overlap < min_overlap:
            continue

        if nuclei_lbl is not None and (min_cell_nucleus_area_ratio is not None or max_nucleus_fraction is not None):
            nuc_ids = np.unique(nuclei_lbl[rr, cc])
            nuc_ids = nuc_ids[nuc_ids > 0]

            nuc_area = 0
            for nid in nuc_ids:
                nuc_area += np.sum(nuclei_lbl == nid)

            if min_cell_nucleus_area_ratio is not None:
                ratio = area / max(nuc_area, 1)
                if ratio < float(min_cell_nucleus_area_ratio):
                    continue

            if max_nucleus_fraction is not None:
                frac = nuc_area / max(area, 1)
                if frac > float(max_nucleus_fraction):
                    continue

        keep.append(region.label)

    return np.array(keep, dtype=int)


def summarize_kept_objects(nucleus_lbl, cell_lbl, nuc_ids, cell_ids):
    nuc_areas = [int(np.sum(nucleus_lbl == nid)) for nid in nuc_ids]

    cell_areas = []
    cell_nucleus_ratios = []
    nucleus_fractions = []

    for cid in cell_ids:
        cmask = (cell_lbl == cid)
        c_area = int(np.sum(cmask))
        cell_areas.append(c_area)

        nuc_overlap_ids = np.unique(nucleus_lbl[cmask])
        nuc_overlap_ids = nuc_overlap_ids[nuc_overlap_ids > 0]
        nuc_area = 0
        for nid in nuc_overlap_ids:
            nuc_area += int(np.sum(nucleus_lbl == nid))

        ratio = c_area / max(nuc_area, 1)
        frac = nuc_area / max(c_area, 1)

        cell_nucleus_ratios.append(ratio)
        nucleus_fractions.append(frac)

    return {
        "total_nucleus_area": int(np.sum(nuc_areas)) if nuc_areas else 0,
        "max_nucleus_area": int(np.max(nuc_areas)) if nuc_areas else 0,
        "total_cell_area": int(np.sum(cell_areas)) if cell_areas else 0,
        "max_cell_area": int(np.max(cell_areas)) if cell_areas else 0,
        "min_cell_nucleus_ratio": float(np.min(cell_nucleus_ratios)) if cell_nucleus_ratios else np.nan,
        "max_nucleus_fraction": float(np.max(nucleus_fractions)) if nucleus_fractions else np.nan,
    }


def compute_review_flags(spot_class, raw_n_nuclei, n_nuclei, n_cells, metrics, filter_stats):
    """
    Generate quality-control flags for review.

    These flags are diagnostic only and do not alter the final
    empty/single/doublet_or_multiplet classification.
    """
    score = 0
    reasons = []

    if spot_class != "single":
        score += 3
        reasons.append(f"class={spot_class}")

    if raw_n_nuclei > n_nuclei:
        score += 1
        reasons.append("raw_nuclei_filtered")

    if filter_stats.get("filtered_small_abs_nuclei", 0) > 0:
        score += 1
        reasons.append("small_nucleus_filtered")

    if filter_stats.get("filtered_small_ratio_nuclei", 0) > 0:
        score += 1
        reasons.append("small_relative_nucleus_filtered")

    if filter_stats.get("filtered_low_red_support_nuclei", 0) > 0:
        score += 1
        reasons.append("insufficient_red_support")

    if n_cells > n_nuclei:
        score += 2
        reasons.append("cells_gt_nuclei")

    if n_nuclei == 0 and n_cells > 0:
        score += 2
        reasons.append("cell_without_nucleus")

    max_nucleus_fraction = metrics.get("max_nucleus_fraction", np.nan)
    min_cell_nucleus_ratio = metrics.get("min_cell_nucleus_ratio", np.nan)
    max_cell_area = metrics.get("max_cell_area", 0)

    if not np.isnan(max_nucleus_fraction) and max_nucleus_fraction > 0.70:
        score += 1
        reasons.append("high_nucleus_fraction")

    if not np.isnan(min_cell_nucleus_ratio) and min_cell_nucleus_ratio < 1.30:
        score += 1
        reasons.append("low_cell_nucleus_ratio")

    if max_cell_area > 0 and max_cell_area < 150:
        score += 1
        reasons.append("small_cell_area")

    if not reasons:
        reasons.append("no_qc_flags")

    return score, ";".join(reasons)


def classify_spot(n_nuclei, n_cells):
    if n_nuclei == 0:
        return "empty"
    if n_nuclei == 1:
        return "single"
    return "doublet_or_multiplet"


def parse_force_spot_ids(s):
    if s is None or str(s).strip() == "":
        return set()
    return set([x.strip() for x in str(s).split(",") if x.strip()])


def make_short_title(spot_id, raw_n_nuclei, n_nuclei, n_cells, spot_class):
    return f"{spot_id} | raw_nuclei={raw_n_nuclei}, nuclei={n_nuclei}, cells={n_cells}, class={spot_class}"


def make_param_text(P):
    items = [
        f"min_nucleus_area={P['min_nucleus_area']}",
        f"nucleus_percentile={P['nucleus_percentile']}",
        f"nucleus_erosion={P['nucleus_erosion']}",
        f"split_area_threshold={P['split_area_threshold']}",
        f"split_min_distance={P['split_min_distance']}",
        f"nucleus_gaussian_sigma={P['nucleus_gaussian_sigma']}",
        f"nucleus_post_dilate={P['nucleus_post_dilate']}",
        f"split_min_child_area={P['split_min_child_area']}",
        f"split_min_area_ratio={P['split_min_area_ratio']}",
        f"nuc_abs_min={P['nuc_abs_min']}",
        f"min_cell_area={P['min_cell_area']}",
        f"red_bg_sigma={P['red_bg_sigma']}",
        f"cell_fg_percentile={P['cell_fg_percentile']}",
        f"cell_post_erode={P['cell_post_erode']}",
        f"max_nucleus_to_cell_distance={P['max_nucleus_to_cell_distance']}",
        f"spot_min_overlap_nucleus={P['spot_min_overlap_nucleus']}",
        f"spot_min_overlap_cell={P['spot_min_overlap_cell']}",
        f"count_min_nucleus_area={P['count_min_nucleus_area']}",
        f"count_min_cell_area={P['count_min_cell_area']}",
        f"count_min_cell_nucleus_area_ratio={P['count_min_cell_nucleus_area_ratio']}",
        f"count_max_nucleus_fraction={P['count_max_nucleus_fraction']}",
        f"valid_nucleus_min_area_abs={P['valid_nucleus_min_area_abs']}",
        f"valid_nucleus_min_area_ratio_to_largest={P['valid_nucleus_min_area_ratio_to_largest']}",
        f"valid_nucleus_red_support_dilate_radius={P['valid_nucleus_red_support_dilate_radius']}",
        f"valid_nucleus_min_red_overlap_pixels={P['valid_nucleus_min_red_overlap_pixels']}",
        f"valid_nucleus_min_red_overlap_ratio={P['valid_nucleus_min_red_overlap_ratio']}",
    ]
    return " | ".join(items)


def save_spot_preview(red_crop, blue_crop, nuclei_lbl, cell_lbl, spot_box_local, out_png, title=""):
    rgb = np.dstack([red_crop, np.zeros_like(red_crop), blue_crop])

    nuc_bd = segmentation.find_boundaries(nuclei_lbl, mode="outer")
    cell_bd = segmentation.find_boundaries(cell_lbl, mode="outer")

    show = rgb.copy()
    show[cell_bd] = [1, 1, 0]
    show[nuc_bd] = [0, 1, 1]

    y0, y1, x0, x1 = spot_box_local
    rr = np.array([y0, y0, y1, y1, y0])
    cc = np.array([x0, x1, x1, x0, x0])

    plt.figure(figsize=(5, 5))
    plt.imshow(show)
    plt.plot(cc, rr, linewidth=1)
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def append_spot_pdf_page(
    pdf,
    spot_id,
    red_crop,
    blue_crop,
    red_fg,
    cell_mask,
    nuclei_lbl_show,
    cell_lbl_show,
    red_support_mask,
    spot_box_local,
    title="",
    param_text=""
):
    rgb = np.dstack([red_crop, np.zeros_like(red_crop), blue_crop])

    nuc_bd = segmentation.find_boundaries(nuclei_lbl_show, mode="outer")
    cell_bd = segmentation.find_boundaries(cell_lbl_show, mode="outer")

    overlay = rgb.copy()
    overlay[cell_bd] = [1, 1, 0]
    overlay[nuc_bd] = [0, 1, 1]

    y0, y1, x0, x1 = spot_box_local
    rr = np.array([y0, y0, y1, y1, y0])
    cc = np.array([x0, x1, x1, x0, x0])

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].imshow(blue_crop, cmap="gray")
    axes[0].plot(cc, rr, linewidth=1)
    axes[0].set_title("Blue crop")
    axes[0].axis("off")

    axes[1].imshow(red_crop, cmap="gray")
    axes[1].plot(cc, rr, linewidth=1)
    axes[1].set_title("Red crop")
    axes[1].axis("off")

    axes[2].imshow(red_fg, cmap="gray")
    axes[2].plot(cc, rr, linewidth=1)
    axes[2].set_title("Red foreground")
    axes[2].axis("off")

    axes[3].imshow(nuclei_lbl_show > 0, cmap="gray")
    axes[3].plot(cc, rr, linewidth=1)
    axes[3].set_title("Nuclei kept")
    axes[3].axis("off")

    axes[4].imshow(cell_mask, cmap="gray")
    axes[4].plot(cc, rr, linewidth=1)
    axes[4].set_title("Cell candidate mask")
    axes[4].axis("off")

    axes[5].imshow(overlay)
    axes[5].plot(cc, rr, linewidth=1)
    axes[5].set_title("Overlay")
    axes[5].axis("off")

    fig.suptitle(title if title else spot_id, fontsize=15)
    if param_text:
        fig.text(0.01, 0.01, param_text, fontsize=7, ha="left", va="bottom")

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    pdf.savefig(fig, dpi=160)
    plt.close(fig)


# =========================
# main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blue-image", required=True)
    ap.add_argument("--red-image", required=True)
    ap.add_argument("--grid-tsv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--margin", type=int, default=30)
    ap.add_argument("--save-preview-every", type=int, default=100)
    ap.add_argument("--preview-force-spot-ids", type=str, default="")
    ap.add_argument(
        "--spot-override-tsv",
        type=str,
        default=None,
        help="Optional external override TSV. If not provided, script auto-uses <outdir>/spot_override_edit.tsv"
    )

    ap.add_argument("--min-nucleus-area", type=int, default=20)
    ap.add_argument("--nucleus-percentile", type=float, default=99.1)
    ap.add_argument("--nucleus-erosion", type=int, default=2)
    ap.add_argument("--split-area-threshold", type=int, default=420)
    ap.add_argument("--split-min-distance", type=int, default=10)
    ap.add_argument("--nucleus-gaussian-sigma", type=float, default=0.8)
    ap.add_argument("--nucleus-post-dilate", type=int, default=0)
    ap.add_argument("--split-min-child-area", type=int, default=30)
    ap.add_argument("--split-min-area-ratio", type=float, default=0.22)
    ap.add_argument("--nuc-abs-min", type=float, default=0.29)

    ap.add_argument("--min-cell-area", type=int, default=50)
    ap.add_argument("--red-bg-sigma", type=float, default=10)
    ap.add_argument("--cell-fg-percentile", type=float, default=60)
    ap.add_argument("--cell-post-erode", type=int, default=0)
    ap.add_argument("--max-nucleus-to-cell-distance", type=float, default=-1)

    ap.add_argument("--spot-min-overlap-nucleus", type=int, default=8)
    ap.add_argument("--spot-min-overlap-cell", type=int, default=20)
    ap.add_argument("--count-min-nucleus-area", type=int, default=35)
    ap.add_argument("--count-min-cell-area", type=int, default=80)
    ap.add_argument("--count-min-cell-nucleus-area-ratio", type=float, default=1.0)
    ap.add_argument("--count-max-nucleus-fraction", type=float, default=1.0)

    # new: valid nucleus filtering
    ap.add_argument("--valid-nucleus-min-area-abs", type=int, default=90)
    ap.add_argument("--valid-nucleus-min-area-ratio-to-largest", type=float, default=0.22)
    ap.add_argument("--valid-nucleus-red-support-dilate-radius", type=int, default=6)
    ap.add_argument("--valid-nucleus-min-red-overlap-pixels", type=int, default=25)
    ap.add_argument("--valid-nucleus-min-red-overlap-ratio", type=float, default=0.08)

    args = ap.parse_args()

    ensure_dir(args.outdir)
    preview_dir = os.path.join(args.outdir, "spot_previews")
    ensure_dir(preview_dir)

    auto_override_path = os.path.join(args.outdir, "spot_override_edit.tsv")
    override_path = args.spot_override_tsv if args.spot_override_tsv else auto_override_path

    force_preview_spot_ids = parse_force_spot_ids(args.preview_force_spot_ids)

    blue = np.squeeze(tifffile.imread(args.blue_image))
    red = np.squeeze(tifffile.imread(args.red_image))
    H, W = blue.shape

    blue_norm_full = normalize_channel(blue)
    red_norm_full = normalize_channel(red)

    grid = load_grid(args.grid_tsv)
    spot_overrides = load_spot_overrides(override_path)

    print(f"[INFO] loaded grid rows = {len(grid)}")
    print(f"[INFO] grid columns = {grid.columns.tolist()}")
    print(f"[INFO] override file in use = {override_path}")
    print(f"[INFO] loaded spot overrides = {len(spot_overrides)}")
    if len(force_preview_spot_ids) > 0:
        print(f"[INFO] force preview spot IDs = {sorted(force_preview_spot_ids)}")

    global_params = {
        "min_nucleus_area": args.min_nucleus_area,
        "nucleus_percentile": args.nucleus_percentile,
        "nucleus_erosion": args.nucleus_erosion,
        "split_area_threshold": args.split_area_threshold,
        "split_min_distance": args.split_min_distance,
        "nucleus_gaussian_sigma": args.nucleus_gaussian_sigma,
        "nucleus_post_dilate": args.nucleus_post_dilate,
        "split_min_child_area": args.split_min_child_area,
        "split_min_area_ratio": args.split_min_area_ratio,
        "nuc_abs_min": args.nuc_abs_min,
        "min_cell_area": args.min_cell_area,
        "red_bg_sigma": args.red_bg_sigma,
        "cell_fg_percentile": args.cell_fg_percentile,
        "cell_post_erode": args.cell_post_erode,
        "max_nucleus_to_cell_distance": None if float(args.max_nucleus_to_cell_distance) < 0 else float(args.max_nucleus_to_cell_distance),
        "spot_min_overlap_nucleus": args.spot_min_overlap_nucleus,
        "spot_min_overlap_cell": args.spot_min_overlap_cell,
        "count_min_nucleus_area": args.count_min_nucleus_area,
        "count_min_cell_area": args.count_min_cell_area,
        "count_min_cell_nucleus_area_ratio": args.count_min_cell_nucleus_area_ratio,
        "count_max_nucleus_fraction": args.count_max_nucleus_fraction,
        # new valid nucleus params
        "valid_nucleus_min_area_abs": args.valid_nucleus_min_area_abs,
        "valid_nucleus_min_area_ratio_to_largest": args.valid_nucleus_min_area_ratio_to_largest,
        "valid_nucleus_red_support_dilate_radius": args.valid_nucleus_red_support_dilate_radius,
        "valid_nucleus_min_red_overlap_pixels": args.valid_nucleus_min_red_overlap_pixels,
        "valid_nucleus_min_red_overlap_ratio": args.valid_nucleus_min_red_overlap_ratio,
    }

    summary_rows = []
    params_rows = []
    review_rows = []
    template_rows = []

    forced_pdf_path = os.path.join(args.outdir, "spot_preview_forced.pdf")
    all_pdf_path = os.path.join(args.outdir, "spot_preview_all.pdf")
    pdf_forced = PdfPages(forced_pdf_path)
    pdf_all = PdfPages(all_pdf_path)

    for i, rec in grid.iterrows():
        spot_id = rec["spot_id"]
        row = int(rec["row"])
        col = int(rec["col"])

        P = merge_params(global_params, spot_overrides.get(spot_id, {}))

        x0 = rec["x0"]
        x1 = rec["x1"]
        y0 = rec["y0"]
        y1 = rec["y1"]

        blue_crop, ox, oy = crop_with_margin(blue_norm_full, x0, x1, y0, y1, args.margin, H, W)
        red_crop, _, _ = crop_with_margin(red_norm_full, x0, x1, y0, y1, args.margin, H, W)

        # 1) raw nuclei segmentation
        raw_nuclei_lbl = nuclei_segmentation_hybrid(
            blue_crop,
            min_area=P["min_nucleus_area"],
            percentile_hi=P["nucleus_percentile"],
            erosion_radius=P["nucleus_erosion"],
            split_area_threshold=P["split_area_threshold"],
            split_min_distance=P["split_min_distance"],
            nuc_abs_min=P["nuc_abs_min"],
            gaussian_sigma=P["nucleus_gaussian_sigma"],
            post_dilate_radius=P["nucleus_post_dilate"],
            split_min_child_area=P["split_min_child_area"],
            split_min_area_ratio=P["split_min_area_ratio"]
        )

        # 2) red fg first
        red_fg, red_bg = subtract_red_background(red_crop, sigma_bg=P["red_bg_sigma"])

        # 3) valid nucleus filtering:
        #    too small blue dots removed
        #    too small relative-to-largest removed
        #    no-red-support nuclei removed
        nuclei_lbl, filter_stats, red_support_mask = filter_valid_nuclei_by_size_and_red_support(
            raw_nuclei_lbl,
            red_fg,
            valid_nucleus_min_area_abs=P["valid_nucleus_min_area_abs"],
            valid_nucleus_min_area_ratio_to_largest=P["valid_nucleus_min_area_ratio_to_largest"],
            valid_nucleus_red_support_dilate_radius=P["valid_nucleus_red_support_dilate_radius"],
            valid_nucleus_min_red_overlap_pixels=P["valid_nucleus_min_red_overlap_pixels"],
            valid_nucleus_min_red_overlap_ratio=P["valid_nucleus_min_red_overlap_ratio"]
        )

        # 4) build cell mask only from valid nuclei
        cell_mask, red_fg2, red_bg2 = build_cell_candidate_mask(
            red_crop,
            nuclei_lbl,
            min_cell_area=P["min_cell_area"],
            sigma_bg=P["red_bg_sigma"],
            cell_fg_percentile=P["cell_fg_percentile"],
            cell_post_erode=P["cell_post_erode"],
            max_nucleus_to_cell_distance=P["max_nucleus_to_cell_distance"]
        )

        cell_lbl = segment_cells_from_candidate_mask(
            cell_mask,
            nuclei_lbl,
            min_cell_area=P["min_cell_area"]
        )

        local_x0 = int(round(x0 - ox))
        local_x1 = int(round(x1 - ox))
        local_y0 = int(round(y0 - oy))
        local_y1 = int(round(y1 - oy))

        local_x0 = max(0, local_x0)
        local_x1 = min(blue_crop.shape[1], local_x1)
        local_y0 = max(0, local_y0)
        local_y1 = min(blue_crop.shape[0], local_y1)

        # Valid nucleus count used for final spot classification
        nuc_ids = labels_in_box_by_centroid_and_area(
            nuclei_lbl,
            local_x0, local_x1, local_y0, local_y1,
            min_overlap=P["spot_min_overlap_nucleus"],
            min_area_for_count=P["count_min_nucleus_area"]
        )

        # Raw nucleus count retained for quality-control reporting
        raw_nuc_ids = labels_in_box_by_centroid_and_area(
            raw_nuclei_lbl,
            local_x0, local_x1, local_y0, local_y1,
            min_overlap=P["spot_min_overlap_nucleus"],
            min_area_for_count=max(1, min(P["count_min_nucleus_area"], P["min_nucleus_area"]))
        )

        cell_ids = labels_in_box_by_centroid_and_area(
            cell_lbl,
            local_x0, local_x1, local_y0, local_y1,
            min_overlap=P["spot_min_overlap_cell"],
            min_area_for_count=P["count_min_cell_area"],
            min_cell_nucleus_area_ratio=P["count_min_cell_nucleus_area_ratio"],
            nuclei_lbl=nuclei_lbl,
            max_nucleus_fraction=P["count_max_nucleus_fraction"]
        )

        raw_n_nuclei = len(raw_nuc_ids)
        n_nuclei = len(nuc_ids)
        n_cells = len(cell_ids)

        # Final class is determined only by the number of valid nuclei
        spot_class = classify_spot(n_nuclei, n_cells)

        metrics = summarize_kept_objects(nuclei_lbl, cell_lbl, nuc_ids, cell_ids)
        review_score, review_reason = compute_review_flags(
            spot_class, raw_n_nuclei, n_nuclei, n_cells, metrics, filter_stats
        )

        summary_rows.append({
            "spot_id": spot_id,
            "row": row,
            "col": col,
            "x_center": rec["x_center"],
            "y_center": rec["y_center"],
            "raw_n_nuclei": raw_n_nuclei,
            "n_nuclei": n_nuclei,
            "n_cells": n_cells,
            "spot_class": spot_class,
            "override_applied": int(spot_id in spot_overrides),

            "filtered_small_abs_nuclei": filter_stats["filtered_small_abs_nuclei"],
            "filtered_small_ratio_nuclei": filter_stats["filtered_small_ratio_nuclei"],
            "filtered_low_red_support_nuclei": filter_stats["filtered_low_red_support_nuclei"],

            "total_nucleus_area": metrics["total_nucleus_area"],
            "max_nucleus_area": metrics["max_nucleus_area"],
            "total_cell_area": metrics["total_cell_area"],
            "max_cell_area": metrics["max_cell_area"],
            "min_cell_nucleus_ratio": metrics["min_cell_nucleus_ratio"],
            "max_nucleus_fraction": metrics["max_nucleus_fraction"],
            "review_score": review_score,
            "review_reason": review_reason,
        })

        one_param_row = {"spot_id": spot_id}
        one_param_row.update(P)
        params_rows.append(one_param_row)

        review_rows.append({
            "spot_id": spot_id,
            "row": row,
            "col": col,
            "spot_class": spot_class,
            "raw_n_nuclei": raw_n_nuclei,
            "n_nuclei": n_nuclei,
            "n_cells": n_cells,
            "filtered_small_abs_nuclei": filter_stats["filtered_small_abs_nuclei"],
            "filtered_small_ratio_nuclei": filter_stats["filtered_small_ratio_nuclei"],
            "filtered_low_red_support_nuclei": filter_stats["filtered_low_red_support_nuclei"],
            "total_nucleus_area": metrics["total_nucleus_area"],
            "total_cell_area": metrics["total_cell_area"],
            "max_nucleus_fraction": metrics["max_nucleus_fraction"],
            "min_cell_nucleus_ratio": metrics["min_cell_nucleus_ratio"],
            "review_score": review_score,
            "review_reason": review_reason,
            "override_applied": int(spot_id in spot_overrides),
        })

        template_row = {
            "spot_id": spot_id,
            "use_override": 0,
            "notes": "",
            "current_class": spot_class,
            "current_raw_n_nuclei": raw_n_nuclei,
            "current_n_nuclei": n_nuclei,
            "current_n_cells": n_cells,
            "current_total_nucleus_area": metrics["total_nucleus_area"],
            "current_total_cell_area": metrics["total_cell_area"],
            "current_max_nucleus_fraction": metrics["max_nucleus_fraction"],
            "current_min_cell_nucleus_ratio": metrics["min_cell_nucleus_ratio"],
            "review_score": review_score,
            "review_reason": review_reason,
        }
        for k in sorted(VALID_OVERRIDE_KEYS):
            template_row[k] = P.get(k, "")
        template_rows.append(template_row)

        need_preview = (
            (i % args.save_preview_every == 0)
            or (spot_class != "single")
            or (spot_id in force_preview_spot_ids)
        )

        nuclei_lbl_show = np.where(np.isin(nuclei_lbl, nuc_ids), nuclei_lbl, 0)
        cell_lbl_show = np.where(np.isin(cell_lbl, cell_ids), cell_lbl, 0)

        short_title = make_short_title(spot_id, raw_n_nuclei, n_nuclei, n_cells, spot_class)
        param_text = make_param_text(P)

        if need_preview:
            out_png = os.path.join(preview_dir, f"{spot_id}.png")
            save_spot_preview(
                red_crop, blue_crop, nuclei_lbl_show, cell_lbl_show,
                spot_box_local=(local_y0, local_y1, local_x0, local_x1),
                out_png=out_png,
                title=short_title
            )

            append_spot_pdf_page(
                pdf=pdf_all,
                spot_id=spot_id,
                red_crop=red_crop,
                blue_crop=blue_crop,
                red_fg=red_fg,
                cell_mask=cell_mask,
                nuclei_lbl_show=nuclei_lbl_show,
                cell_lbl_show=cell_lbl_show,
                red_support_mask=red_support_mask,
                spot_box_local=(local_y0, local_y1, local_x0, local_x1),
                title=short_title,
                param_text=param_text
            )

        if spot_id in force_preview_spot_ids:
            append_spot_pdf_page(
                pdf=pdf_forced,
                spot_id=spot_id,
                red_crop=red_crop,
                blue_crop=blue_crop,
                red_fg=red_fg,
                cell_mask=cell_mask,
                nuclei_lbl_show=nuclei_lbl_show,
                cell_lbl_show=cell_lbl_show,
                red_support_mask=red_support_mask,
                spot_box_local=(local_y0, local_y1, local_x0, local_x1),
                title=short_title,
                param_text=param_text
            )

        if (i + 1) % 100 == 0:
            print(f"[INFO] processed {i+1}/{len(grid)} spots")

    pdf_forced.close()
    pdf_all.close()

    summary_df = pd.DataFrame(summary_rows)
    params_df = pd.DataFrame(params_rows)
    review_df = pd.DataFrame(review_rows).sort_values(
        by=["review_score", "spot_class", "n_cells", "n_nuclei"],
        ascending=[False, True, False, False]
    )
    template_df = pd.DataFrame(template_rows).sort_values(
        by=["review_score", "spot_id"],
        ascending=[False, True]
    )

    summary_df_out = append_class_summary_rows(summary_df, class_col="spot_class")
    review_df_out = append_class_summary_rows(review_df, class_col="spot_class")
    template_df_out = append_class_summary_rows(template_df, class_col="current_class")

    out_tsv = os.path.join(args.outdir, "spot_summary.tsv")
    params_tsv = os.path.join(args.outdir, "spot_params_used.tsv")
    review_tsv = os.path.join(args.outdir, "review_candidates.tsv")
    edit_override_tsv = auto_override_path

    summary_df_out.to_csv(out_tsv, sep="\t", index=False)
    params_df.to_csv(params_tsv, sep="\t", index=False)
    review_df_out.to_csv(review_tsv, sep="\t", index=False)

    if not os.path.exists(edit_override_tsv):
        template_df_out.to_csv(edit_override_tsv, sep="\t", index=False)
        print(f"[OK] created editable override table: {edit_override_tsv}")
    else:
        print(f"[OK] kept existing editable override table: {edit_override_tsv}")

    print("[OK] spot delineation + classification finished")
    print(f"[OK] output summary: {out_tsv}")
    print(f"[OK] output params: {params_tsv}")
    print(f"[OK] review candidates: {review_tsv}")
    print(f"[OK] editable override table: {edit_override_tsv}")
    print(f"[OK] preview dir: {preview_dir}")
    print(f"[OK] forced preview pdf: {forced_pdf_path}")
    print(f"[OK] all preview pdf: {all_pdf_path}")


if __name__ == "__main__":
    main()
