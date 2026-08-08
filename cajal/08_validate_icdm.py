#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from cajal.run_gw import icdm_csv_validate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icdm_csv", required=True)
    args = ap.parse_args()

    icdm_csv_validate(args.icdm_csv)
    print("[OK] ICDM CSV validated")

if __name__ == "__main__":
    main()
