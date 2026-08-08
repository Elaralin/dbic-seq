#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import inspect
import argparse
from cajal import sample_seg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infolder", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--n_sample", type=int, default=300)
    ap.add_argument("--n_proc", type=int, default=8)
    ap.add_argument("--background", type=int, default=0)
    ap.add_argument("--discard_cells_with_holes", action="store_true")
    ap.add_argument("--only_longest", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    sig = inspect.signature(sample_seg.compute_icdm_all)
    kwargs = dict(
        infolder=args.infolder,
        out_csv=args.out_csv,
        n_sample=args.n_sample,
        background=args.background,
        discard_cells_with_holes=args.discard_cells_with_holes,
        only_longest=args.only_longest,
    )

    if "num_processes" in sig.parameters:
        kwargs["num_processes"] = args.n_proc
    elif "num_cores" in sig.parameters:
        kwargs["num_cores"] = args.n_proc
    else:
        raise RuntimeError("Could not find num_processes / num_cores in compute_icdm_all signature")

    print("[INFO] compute_icdm_all kwargs:")
    for k, v in kwargs.items():
        print(f"  {k} = {v}")

    sample_seg.compute_icdm_all(**kwargs)
    print("[OK] finished")

if __name__ == "__main__":
    main()
