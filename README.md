# DBiC

DBiC is a deterministic barcoding framework for linking image-derived cellular phenotypes with sequencing-based molecular profiles at single-cell resolution.

This repository contains the core computational workflows used for image-indexed cell assignment, morphology analysis, CAJAL-based shape profiling, species-mixing validation, drug-response analysis, and morphology-transcriptome integration.

## Repository structure

### `spatial_coordinate_classification/`

Scripts for image-guided spatial indexing and cell classification.

- `00_select_corner_spots.m`  
  Interactive selection of the four corner spots from a nuclear image.

- `01_make_spot_grid.py`  
  Construction of the spatial spot grid from the selected corner coordinates.

- `02_segment_nuclei_by_spot.py`  
  Nuclear segmentation within spatially indexed regions.

- `03_segment_and_classify_spots.py`  
  Spot-level cell classification and singlet/multiplet identification.

### `morphology/`

Scripts for whole-cell segmentation and morphology extraction.

- `01_segment_nuclei_full.py`  
  Full-image nuclear segmentation.

- `02b_match_singlet_spots_to_nuclei_two_stage.py`  
  Matching of spatially indexed singlet spots to segmented nuclei.

- `03_extract_cells_fullimage_seeded_v3.py`  
  Seeded whole-cell segmentation and cell-shape extraction.

### `cajal/`

Core scripts for CAJAL-based cell-shape analysis.

- `06_pad_masks_for_cajal.py`  
  Preparation and padding of binary cell masks.

- `07_cajal_compute_icdm_from_segmentation.py`  
  Computation of intracellular distance matrices from segmented cell shapes.

- `08_validate_icdm.py`  
  Validation of generated intracellular distance matrices.

- `09_compute_gw_matrix.py`  
  Computation of pairwise Gromov-Wasserstein shape distances.

- `10_check_gw_output.py`  
  Quality-control checks for the resulting distance matrix.

### `species_mixing/`

Scripts for human-mouse species-mixing validation.

- `01_species_assignment.py`  
  Assignment of individual cells to human, mouse, or mixed populations.

- `7_plot_species_fraction_hist.py`  
  Visualization of species-assignment score distributions.

- `8_plot_pseudobulk_species_markers.py`  
  Pseudobulk validation using species-specific marker genes.

- `make_cm524_complexity_summary.py`  
  Summary of single-cell transcriptomic complexity.

### `drug_perturbation/`

Core transcriptomic analysis of drug-perturbation experiments.

#### `rna/`

- extraction of matched single-cell transcriptomes
- RNA-state analysis
- marker analysis
- comparison of RNA states across perturbations

#### `drug_response/`

- drug metadata construction
- pseudobulk aggregation
- differential-expression analysis with edgeR
- drug-response module analysis

### `multimodal_integration/`

Scripts for integrating morphology and transcriptomic states.

The `morph_rna/` workflow contains the core analyses used to relate drug perturbations, RNA states, and morphology states at the single-cell level.

### `environment/`

Information about computational environments and configurable data paths.

## Data organization

Raw sequencing and imaging data are not distributed through this repository.

Drug-perturbation analyses use the environment variable:

```bash
export DBIC_DATA_ROOT=/path/to/AD715
```

If not specified, scripts use:

```text
data/AD715
```

CAJAL mask preparation supports:

```bash
export DBIC_CAJAL_INPUT_DIR=/path/to/input_masks
export DBIC_CAJAL_OUTPUT_DIR=/path/to/output_masks
```

Users may modify these paths according to their local data organization.

## General workflow

A typical image-to-molecular analysis proceeds through:

```text
Image acquisition
      |
      v
Corner selection and spatial grid construction
      |
      v
Spot-level cell classification
      |
      v
Whole-cell segmentation
      |
      v
Morphology extraction and shape analysis
      |
      v
Single-cell transcriptomic analysis
      |
      v
Morphology-transcriptome integration
```

CAJAL-based shape analysis can be applied to the segmented cell masks to obtain pairwise morphology distances and downstream morphology manifolds.

## Requirements

The workflows use Python, R, MATLAB, and CAJAL depending on the analysis module.

Exact software dependencies and package versions will be documented in the final release.

## Data availability

Large raw sequencing and imaging datasets are hosted separately from this code repository. Public accession information will be provided with the associated publication.

## Repository scope

This repository provides the core computational workflows supporting the DBiC study. Development history, exploratory analyses, intermediate files, and nonessential figure-formatting scripts are not included.

## License

See `LICENSE`.
