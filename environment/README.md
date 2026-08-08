# Computational environments

DBiC analyses were performed using dedicated Python and Conda environments for image processing, morphology analysis, transcriptomic analysis, and CAJAL-based shape analysis.

## Data paths

For drug-perturbation analyses, specify the data root using:

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

Software dependencies and package versions will be provided with the final release.
