# DBiC-seq

DBiC-seq (deterministic barcoding in cells for sequencing) is a spatially indexed platform that integrates live-cell array printing, multiplexed fluorescence imaging, in situ barcoding, and transcriptome profiling in the same cells.

This repository contains the core computational workflows used for spatially indexed cell assignment, morphology analysis, CAJAL-based shape profiling, species-mixing validation, drug-response analysis, and morphology-transcriptome integration.

## Repository structure

### `spatial_coordinate_classification/`

Scripts for image-guided spatial indexing and cell classification.

- `00_select_corner_spots.m`  
  Interactive selection of the four corner spots from a nuclear image.

- `01_make_spot_grid.py`  
  Construction of the spatial spot grid from the selected corner coordinates.

- `02_delineate_nuclei_by_spot.py`  
  Nuclear delineation within spatially indexed regions.

- `03_delineate_and_classify_spots.py`  
  Spot-level cell classification and singlet/multiplet identification.

### `morphology/`

Scripts for whole-cell delineation and morphology extraction.

- `01_delineate_nuclei_full_image.py`  
  Full-image nuclear delineation.

- `02_match_singlet_spots_to_nuclei.py`  
  Matching of spatially indexed singlet spots to delineated nuclei.

- `03_extract_cells_fullimage_seeded.py`  
  Seeded whole-cell delineation and cell-shape extraction.

### `cajal/`

Core scripts for CAJAL-based cell-shape analysis.

- `01_prepare_masks_for_cajal.py`  
  Preparation and padding of binary cell masks.

- `02_compute_icdm_from_cell_masks.py`  
  Computation of intracellular distance matrices from delineated cell masks.

- `03_validate_icdm.py`  
  Validation of generated intracellular distance matrices.

- `04_compute_gw_matrix.py`  
  Computation of pairwise Gromov-Wasserstein shape distances.

- `05_check_gw_output.py`  
  Quality-control checks for the resulting distance matrix.

### `species_mixing/`

Scripts for human-mouse species-mixing validation.

- `01_species_assignment.py`  
  Assignment of individual cells to human, mouse, or mixed populations.

- `02_plot_species_fraction_hist.py`  
  Visualization of species-assignment score distributions.

- `03_plot_pseudobulk_species_markers.py`  
  Pseudobulk validation using species-specific marker genes.

- `04_species_complexity_summary.py`  
  Summary of single-cell transcriptomic complexity.

### `drug_perturbation/`

Core transcriptomic analysis of drug-perturbation experiments.

#### `rna/`

- `01_extract_matched_scRNA.py`  
  Extraction of matched single-cell transcriptomic profiles.
- `02_build_rna_atlas.py`  
  Construction of the single-cell RNA-state atlas.
- `03_identify_rna_state_markers.py`  
  Identification of marker genes associated with RNA states.
- `04_compare_rna_states_across_drugs.py`  
  Comparison of RNA-state composition across perturbation conditions.

#### `drug_response/`

- `01_make_drug_metadata.py`  
  Construction of perturbation metadata for indexed cells.
- `02_pseudobulk.R`  
  Pseudobulk aggregation of single-cell transcriptomic profiles.
- `03_differential_expression_edgeR.R`  
  Differential-expression analysis using edgeR.
- `04_drug_response_modules.R`  
  Analysis of drug-associated transcriptional response modules.

### `multimodal_integration/`

Scripts for integrating morphology and transcriptomic states.

#### `morph_rna/`

- `01_morph_state_by_rna_state_coupling.py`
  Analysis of RNA-state composition and coupling across morphology states.
- `02_rna_state_by_morph_state_composition.py`
  Analysis of morphology-state composition across RNA states.
- `03_drug_rna_morphology_integration.py`
  Integrated analysis of drug perturbations, RNA states, and morphology states.

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
Whole-cell delineation
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

CAJAL-based shape analysis can be applied to the delineated cell masks to obtain pairwise morphology distances and downstream morphology manifolds.

## Requirements

The workflows use Python, R, MATLAB, and CAJAL depending on the analysis module.

Software requirements and environment-specific notes are provided in `environment/README.md`.

## Data availability

Large raw sequencing and imaging datasets are hosted separately from this code repository. Public accession information will be provided with the associated publication.

## Repository scope

This repository provides the core computational workflows supporting the DBiC-seq study. Development history, exploratory analyses, intermediate files, and nonessential figure-formatting scripts are not included.

## License

See `LICENSE`.

## Citation

This repository accompanies the DBiC-seq study. Citation information will be updated upon publication.
