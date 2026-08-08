#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, segmentation, measure, feature


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize(img, p1=1, p99=99.8):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [p1, p99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img - lo) / (hi - lo), 0, 1)


def segment_nuclei_full(blue, min_area=25, max_area=3000):
    x = normalize(blue)
    x = filters.gaussian(x, sigma=1.0, preserve_range=True)

    # Remove slowly varying background
    bg = filters.gaussian(x, sigma=8.0, preserve_range=True)
    hp = x - bg
    hp = hp - hp.min()
    if hp.max() > 0:
        hp = hp / hp.max()

    hp = exposure.equalize_adapthist(hp, clip_limit=0.01)

    try:
        thr = filters.threshold_otsu(hp)
    except Exception:
        thr = np.percentile(hp, 85)

    bw = hp > max(thr, np.percentile(hp, 80))
    bw = morphology.remove_small_objects(bw, min_size=min_area)
    bw = morphology.remove_small_holes(bw, area_threshold=16)

    # watershed split
    dist = ndi.distance_transform_edt(bw)
    coords = feature.peak_local_max(
        dist,
        min_distance=4,
        threshold_abs=1.5,
        labels=bw
    )
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (yy, xx) in enumerate(coords, start=1):
        markers[yy, xx] = i

    if markers.max() > 0:
        labels = segmentation.watershed(-dist, markers, mask=bw)
    else:
        labels = measure.label(bw)

    # Filter objects by area and relabel
    out = np.zeros_like(labels, dtype=np.int32)
    rows = []
    new_id = 0
    props = measure.regionprops(labels, intensity_image=hp)
    for p in props:
        area = int(p.area)
        if area < min_area or area > max_area:
            continue
        new_id += 1
        out[labels == p.label] = new_id
        cy, cx = p.centroid
        minr, minc, maxr, maxc = p.bbox
        rows.append({
            "nucleus_id": new_id,
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "area": area,
            "mean_intensity": float(p.mean_intensity),
            "bbox_minr": int(minr),
            "bbox_minc": int(minc),
            "bbox_maxr": int(maxr),
            "bbox_maxc": int(maxc),
            "equivalent_diameter": float(p.equivalent_diameter),
            "eccentricity": float(p.eccentricity),
            "solidity": float(p.solidity) if p.solidity is not None else np.nan
        })

    props_df = pd.DataFrame(rows)
    return out, props_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blue", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min_area", type=int, default=25)
    ap.add_argument("--max_area", type=int, default=3000)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    blue = tifffile.imread(args.blue)
    if blue.ndim != 2:
        raise ValueError("Expected 2D single-channel blue image")

    labels, props_df = segment_nuclei_full(
        blue,
        min_area=args.min_area,
        max_area=args.max_area
    )

    label_file = os.path.join(args.outdir, "full_nuclei_labels.tif")
    props_file = os.path.join(args.outdir, "full_nuclei_props.tsv")

    tifffile.imwrite(label_file, labels.astype(np.int32), compression="zlib")
    props_df.to_csv(props_file, sep="\t", index=False)

    print("[OK] nuclei labels:", label_file)
    print("[OK] nuclei props :", props_file)
    print("[INFO] n nuclei =", len(props_df))


if __name__ == "__main__":
    main()
