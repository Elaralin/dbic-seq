# Computational environments

DBiC-seq analyses use dedicated computational environments for image processing, morphology analysis, transcriptomic analysis, and CAJAL-based shape analysis.

## Software

The core workflows use Python, R, MATLAB, and CAJAL. Major Python and R dependencies include packages for numerical computing, image processing, single-cell transcriptomic analysis, statistical analysis, and visualization.

Environment-specific dependencies may vary across analysis modules.

## Data paths

Raw sequencing and imaging data are not stored in this repository.

For drug-perturbation analyses, the data root can be specified using:

```bash
export DBIC_DATA_ROOT=/path/to/AD715
```

If not specified, scripts use:

```text
data/AD715
```

For CAJAL mask preparation, input and output directories can be specified using:

```bash
export DBIC_CAJAL_INPUT_DIR=/path/to/input_masks
export DBIC_CAJAL_OUTPUT_DIR=/path/to/output_masks
```

Users may modify these paths according to their local data organization.
