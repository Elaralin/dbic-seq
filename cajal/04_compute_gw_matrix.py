#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from cajal import run_gw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icdm_csv", required=True)
    ap.add_argument("--gw_csv", required=True)
    ap.add_argument("--n_proc", type=int, default=8)
    ap.add_argument("--coupling_npz", default=None)
    ap.add_argument("--return_coupling_mats", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.gw_csv), exist_ok=True)

    run_gw.compute_gw_distance_matrix(
        intracell_csv_loc=args.icdm_csv,
        gw_dist_csv_loc=args.gw_csv,
        num_processes=args.n_proc,
        gw_coupling_mat_npz_loc=args.coupling_npz,
        return_coupling_mats=args.return_coupling_mats
    )
    print("[OK] GW finished")

if __name__ == "__main__":
    main()
