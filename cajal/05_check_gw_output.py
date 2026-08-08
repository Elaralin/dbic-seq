#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from cajal import utilities

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gw_csv", required=True)
    args = ap.parse_args()

    cells, gw_dict = utilities.read_gw_dists(args.gw_csv, header=True)
    print(f"[OK] n_cells = {len(cells)}")
    print(f"[OK] n_pairs = {len(gw_dict)}")
    print("[HEAD]")
    print(cells[:10])

if __name__ == "__main__":
    main()
