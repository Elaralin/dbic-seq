#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import tifffile

IN_DIR = os.environ.get("DBIC_CAJAL_INPUT_DIR", "data/cajal_segmentation_input_raw")
OUT_DIR = os.environ.get("DBIC_CAJAL_OUTPUT_DIR", "results/cajal_segmentation_input_padded")
PAD = 10

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(IN_DIR, "*_mask.tif")))
n = 0

for f in files:
    arr = tifffile.imread(f)
    arr = (arr > 0).astype(np.uint8)   # background = 0, cell = 1
    arr = np.pad(arr, ((PAD, PAD), (PAD, PAD)), mode="constant", constant_values=0)

    out = os.path.join(OUT_DIR, os.path.basename(f))
    tifffile.imwrite(out, arr.astype(np.uint8), photometric="minisblack")
    n += 1

print(f"[OK] padded files = {n}")
print(f"[OK] outdir = {OUT_DIR}")
