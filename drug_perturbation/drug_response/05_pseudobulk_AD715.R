#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(Matrix)
})

ROOT <- Sys.getenv("DBIC_DATA_ROOT", unset = "data/AD715")
WORK <- file.path(ROOT, "5_drug_response")

GENE_MAT  <- file.path(ROOT, "0_matrix", "expmat_715.tsv")
META_FILE <- file.path(WORK, "step1_drug_metadata", "AD715_linked_with_drug_condition.tsv")
OUT_DIR   <- file.path(WORK, "05_pseudobulk")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "qc"), recursive = TRUE, showWarnings = FALSE)

log_msg <- function(...) {
  cat("[", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "] ",
      paste0(..., collapse = ""), "\n", sep = "")
}

log_msg("Reading RNA matrix: ", GENE_MAT)
gdt <- fread(GENE_MAT)
setnames(gdt, colnames(gdt)[1], "gene")

genes <- gdt$gene
matrix_cols <- setdiff(colnames(gdt), "gene")

log_msg("RNA matrix: ", nrow(gdt), " genes x ", length(matrix_cols), " columns")

log_msg("Reading metadata: ", META_FILE)
md <- fread(META_FILE)

need <- c("spot_id", "rna_row", "rna_col", "condition", "morph_row", "morph_col")
miss <- setdiff(need, colnames(md))
if (length(miss) > 0) stop("Missing metadata columns: ", paste(miss, collapse = ", "))

# Try common RNA matrix column formats
md[, id_1 := paste0(rna_row, "x", rna_col)]
md[, id_2 := paste0("X", rna_row, "x", rna_col)]
md[, id_3 := paste(rna_row, rna_col, sep = "_")]
md[, id_4 := paste0("R", sprintf("%02d", rna_row), "C", sprintf("%02d", rna_col))]
md[, id_5 := spot_id]

candidates <- c("id_1", "id_2", "id_3", "id_4", "id_5")
match_n <- sapply(candidates, function(x) sum(md[[x]] %in% matrix_cols))

log_msg("Matrix column matching:")
print(match_n)

best <- names(which.max(match_n))
if (max(match_n) == 0) {
  stop("No matching cells between metadata and expmat_715.tsv. Check column names.")
}

md[, matrix_col := get(best)]
md_use <- md[matrix_col %in% matrix_cols]
md_use <- md_use[!duplicated(matrix_col)]

log_msg("Best ID format: ", best)
log_msg("Matched cells: ", nrow(md_use))

# Pseudobulk replicate = physical morphology row
md_use[, replicate := paste0("R", sprintf("%02d", morph_row))]
md_use[, pseudobulk_id := paste(condition, replicate, sep = "__")]

matched_cols <- md_use$matrix_col

expr <- gdt[, c("gene", matched_cols), with = FALSE]
mat_dt <- expr[, ..matched_cols]

for (j in seq_along(mat_dt)) {
  suppressWarnings(set(mat_dt, j = j, value = as.numeric(mat_dt[[j]])))
}

mat <- as.matrix(mat_dt)
rownames(mat) <- expr$gene
mat[is.na(mat)] <- 0

pb_ids <- md_use$pseudobulk_id
unique_pb <- unique(pb_ids)

pb_mat <- matrix(
  0,
  nrow = nrow(mat),
  ncol = length(unique_pb),
  dimnames = list(rownames(mat), unique_pb)
)

for (i in seq_along(unique_pb)) {
  pb <- unique_pb[i]
  idx <- which(pb_ids == pb)
  pb_mat[, i] <- if (length(idx) == 1) mat[, idx] else rowSums(mat[, idx, drop = FALSE])
}

pb_mat <- Matrix(pb_mat, sparse = TRUE)

pb_meta <- unique(md_use[, .(
  pseudobulk_id,
  condition,
  group = condition,
  replicate,
  morph_row
)])

pb_meta[, n_cells := vapply(
  pseudobulk_id,
  function(x) sum(md_use$pseudobulk_id == x),
  numeric(1)
)]

pb_meta[, total_counts := Matrix::colSums(pb_mat)[match(pseudobulk_id, colnames(pb_mat))]]
setorder(pb_meta, condition, morph_row)

pb_mat <- pb_mat[, pb_meta$pseudobulk_id, drop = FALSE]

# Save outputs
fwrite(
  data.table(gene = rownames(pb_mat), as.matrix(pb_mat)),
  file.path(OUT_DIR, "AD715_pseudobulk_counts.tsv.gz"),
  sep = "\t"
)

saveRDS(pb_mat, file.path(OUT_DIR, "AD715_pseudobulk_counts.rds"))

fwrite(
  pb_meta,
  file.path(OUT_DIR, "AD715_pseudobulk_metadata.tsv"),
  sep = "\t"
)

fwrite(
  md_use[, .(
    spot_id, matrix_col, condition, morph_row, morph_col,
    rna_row, rna_col, replicate, pseudobulk_id
  )],
  file.path(OUT_DIR, "qc", "AD715_linked_cells_used.tsv"),
  sep = "\t"
)

summary_dt <- data.table(
  metric = c(
    "n_genes_input",
    "n_RNA_columns_input_matrix",
    "n_linked_cells_metadata",
    "n_cells_matched_matrix_and_metadata",
    "n_pseudobulk_samples"
  ),
  value = c(
    nrow(gdt),
    length(matrix_cols),
    nrow(md),
    nrow(md_use),
    nrow(pb_meta)
  )
)

fwrite(summary_dt, file.path(OUT_DIR, "AD715_pseudobulk_summary.tsv"), sep = "\t")

cell_count <- md_use[, .N, by = condition][order(condition)]
setnames(cell_count, "N", "n_cells")
fwrite(cell_count, file.path(OUT_DIR, "qc", "AD715_cell_count_by_condition.tsv"), sep = "\t")

pb_count <- pb_meta[, .N, by = condition][order(condition)]
setnames(pb_count, "N", "n_pseudobulk")
fwrite(pb_count, file.path(OUT_DIR, "qc", "AD715_pseudobulk_count_by_condition.tsv"), sep = "\t")

log_msg("Done.")
log_msg("Output: ", OUT_DIR)
