# Spatial cell classification example

This example demonstrates the image-guided spatial indexing and cell-classification workflow used in DBiC-seq.

The example contains a cropped 6 × 6 region from a DBiC cell array together with nuclear and F-actin fluorescence images and the four corner-spot coordinates.

## Input files

```text
input/
├── C416_example_blue_nuclei.tif
├── C416_example_red_actin.tif
└── C416_example_corner_coordinates.tsv
```

The blue channel contains nuclear fluorescence and the red channel contains F-actin fluorescence.

## Run the example

From the repository root:

```bash
python --version
bash examples/spatial_cell_classification/run_example.sh
```

If a specific Python executable should be used:

```bash
PYTHON_BIN=/path/to/python \
bash examples/spatial_cell_classification/run_example.sh
```

The workflow performs two main steps:

1. Construction of a 6 × 6 spatially indexed DBiC spot grid.
2. Image-based delineation and classification of cells within each indexed spot.

## Output

Runtime outputs are written to:

```text
examples/spatial_cell_classification/output/
```

Main outputs include:

```text
output/grid/C416_example_grid_overlay.png
output/classification/spot_summary.tsv
output/classification/spot_preview_all.pdf
```

For this example, the expected classification result is:

```text
Total indexed spots: 36
Single-cell spots:   35
Empty spots:          1
```

No manual spot-classification overrides are required for this example.

## Expected output

Reference outputs are provided in:

```text
expected_output/
├── C416_example_classification.tsv
├── C416_example_grid_overlay.png
└── C416_example_spot_classification.pdf
```

These files allow users to compare their locally generated results with the reference output.

## Notes

This small example is included to reproduce the spatial coordinate assignment and image-based cell classification workflow. Full experimental datasets are not included in this repository.
