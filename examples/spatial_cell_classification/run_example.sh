#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXAMPLE_DIR="$ROOT/examples/spatial_cell_classification"
INPUT_DIR="$EXAMPLE_DIR/input"
RUN_DIR="$EXAMPLE_DIR/output"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "========================================"
echo "DBiC-seq spatial cell classification demo"
echo "========================================"

echo
echo "[1/3] Preparing output directory..."
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR/grid" "$RUN_DIR/classification"

echo
echo "[2/3] Constructing 6x6 spatial spot grid..."

"$PYTHON_BIN" \
"$ROOT/spatial_coordinate_classification/01_make_spot_grid.py" \
--corners "$INPUT_DIR/C416_example_corner_coordinates.tsv" \
--rows 6 \
--cols 6 \
--red-image "$INPUT_DIR/C416_example_red_actin.tif" \
--blue-image "$INPUT_DIR/C416_example_blue_nuclei.tif" \
--out "$RUN_DIR/grid/C416_example_grid.tsv" \
--overlay-out "$RUN_DIR/grid/C416_example_grid_overlay.png" \
--label-every 1 \
--corner-mode center \
--expand 1.0

echo
echo "[3/3] Delineating and classifying indexed spots..."

"$PYTHON_BIN" \
"$ROOT/spatial_coordinate_classification/03_delineate_and_classify_spots.py" \
--blue-image "$INPUT_DIR/C416_example_blue_nuclei.tif" \
--red-image "$INPUT_DIR/C416_example_red_actin.tif" \
--grid-tsv "$RUN_DIR/grid/C416_example_grid.tsv" \
--outdir "$RUN_DIR/classification" \
--save-preview-every 1

echo
echo "========================================"
echo "Example completed successfully."
echo
echo "Main outputs:"
echo "  $RUN_DIR/grid/C416_example_grid_overlay.png"
echo "  $RUN_DIR/classification/spot_summary.tsv"
echo "  $RUN_DIR/classification/spot_preview_all.pdf"
echo "========================================"
