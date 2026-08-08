#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(edgeR)
})

ROOT <- Sys.getenv("DBIC_DATA_ROOT", unset = "data/AD715")
WORK <- file.path(ROOT, "5_drug_response")

IN_DIR  <- file.path(WORK, "05_pseudobulk")
OUT_DIR <- file.path(WORK, "06_edgeR")

COUNT_FILE <- file.path(IN_DIR, "AD715_pseudobulk_counts.tsv.gz")
META_FILE  <- file.path(IN_DIR, "AD715_pseudobulk_metadata.tsv")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "qc"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "normalized"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "per_drug"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "summary"), recursive = TRUE, showWarnings = FALSE)

sanitize_label <- function(x) {
  x <- gsub("[ /\\-]+", "_", x)
  x <- gsub("[^A-Za-z0-9_]", "", x)
  x
}

cat("[1/8] Reading pseudobulk counts and metadata...\n")

counts_dt <- fread(COUNT_FILE)
meta <- fread(META_FILE)

setnames(counts_dt, colnames(counts_dt)[1], "gene")

genes <- counts_dt$gene
count_mat <- as.matrix(counts_dt[, -1, with = FALSE])
rownames(count_mat) <- genes
mode(count_mat) <- "numeric"
count_mat[is.na(count_mat)] <- 0

sample_ids <- colnames(count_mat)

required <- c("pseudobulk_id", "condition", "replicate")
missing <- setdiff(required, colnames(meta))
if (length(missing) > 0) {
  stop("Metadata missing columns: ", paste(missing, collapse = ", "))
}

if (!all(sample_ids %in% meta$pseudobulk_id)) {
  stop("Some pseudobulk count columns are missing in metadata.")
}

meta <- meta[match(sample_ids, pseudobulk_id)]
stopifnot(identical(sample_ids, meta$pseudobulk_id))

meta[, condition := as.character(condition)]
meta[toupper(condition) %in% c("DMSO", "CONTROL", "CTRL", "VEHICLE"), condition := "DMSO"]

if (!("DMSO" %in% meta$condition)) {
  stop("No DMSO group found.")
}

cat("[2/8] Group summary...\n")

group_sizes <- meta[, .N, by = condition][order(condition)]
setnames(group_sizes, "N", "n_replicates")
fwrite(group_sizes, file.path(OUT_DIR, "qc", "AD715_group_sizes.tsv"), sep = "\t")
print(group_sizes)

valid_groups <- group_sizes[n_replicates >= 2, condition]

if (!("DMSO" %in% valid_groups)) {
  stop("DMSO has fewer than 2 pseudobulk replicates.")
}

meta <- meta[condition %in% valid_groups]
count_mat <- count_mat[, meta$pseudobulk_id, drop = FALSE]

condition_levels <- c("DMSO", sort(setdiff(unique(meta$condition), "DMSO")))
meta[, condition_group := factor(condition, levels = condition_levels)]

cat("[3/8] Building DGEList and filtering genes...\n")

dge <- DGEList(counts = count_mat)
dge$samples$sample_id <- meta$pseudobulk_id
dge$samples$condition <- meta$condition_group

design <- model.matrix(~ 0 + condition_group, data = meta)
colnames(design) <- gsub("^condition_group", "", colnames(design))

cat("Design columns:\n")
print(colnames(design))

genes0 <- rownames(dge)

tot0 <- rowSums(dge$counts)
det0 <- rowSums(dge$counts > 0)

keep <- grepl("__protein_coding__", genes0) &
        grepl("__exon", genes0) &
        tot0 >= 10 &
        det0 >= 5

cat("Custom feature filtering:\n")
cat("  protein_coding exon total>=10 detected>=5\n")
cat("  kept features =", sum(keep), "\n")

dge <- dge[keep, , keep.lib.sizes = FALSE]

filter_summary <- data.table(
  metric = c("input_genes", "kept_genes", "filtered_genes", "input_samples"),
  value = c(nrow(count_mat), nrow(dge), nrow(count_mat) - nrow(dge), ncol(count_mat))
)

fwrite(filter_summary, file.path(OUT_DIR, "qc", "AD715_filter_summary.tsv"), sep = "\t")
print(filter_summary)

dge <- calcNormFactors(dge, method = "TMM")

sample_qc <- data.table(
  sample_id = dge$samples$sample_id,
  condition = as.character(dge$samples$condition),
  lib_size = dge$samples$lib.size,
  norm_factors = dge$samples$norm.factors,
  effective_lib_size = dge$samples$lib.size * dge$samples$norm.factors
)

fwrite(sample_qc, file.path(OUT_DIR, "qc", "AD715_sample_qc.tsv"), sep = "\t")

logcpm <- cpm(dge, log = TRUE, prior.count = 2)

fwrite(
  data.table(gene = rownames(logcpm), as.data.table(logcpm)),
  file.path(OUT_DIR, "normalized", "AD715_gene_logCPM.tsv.gz"),
  sep = "\t"
)

cat("[4/8] Estimating dispersion and fitting GLM...\n")

dge <- estimateDisp(dge, design)
fit <- glmQLFit(dge, design, robust = TRUE)

disp_summary <- data.table(
  common_dispersion = dge$common.dispersion,
  trended_dispersion_mean = mean(dge$trended.dispersion, na.rm = TRUE),
  tagwise_dispersion_median = median(dge$tagwise.dispersion, na.rm = TRUE)
)

fwrite(disp_summary, file.path(OUT_DIR, "qc", "AD715_dispersion_summary.tsv"), sep = "\t")
print(disp_summary)

cat("[5/8] Running each drug vs DMSO...\n")

drug_names <- setdiff(colnames(design), "DMSO")

all_results_list <- list()
summary_list <- list()
top_list <- list()

for (drug_i in drug_names) {
  cat("  - ", drug_i, " vs DMSO\n", sep = "")
  
  contrast_vec <- rep(0, ncol(design))
  names(contrast_vec) <- colnames(design)
  contrast_vec[drug_i] <- 1
  contrast_vec["DMSO"] <- -1
  
  qlf <- glmQLFTest(fit, contrast = contrast_vec)
  tt <- topTags(qlf, n = Inf, sort.by = "PValue")$table
  
  res <- data.table(
    gene = rownames(tt),
    logFC = tt$logFC,
    logCPM = tt$logCPM,
    F = tt$F,
    PValue = tt$PValue,
    FDR = tt$FDR
  )
  
  res[, condition := drug_i]
  res[, comparison := paste0(drug_i, "_vs_DMSO")]
  res[, direction := fifelse(logFC > 0, "up", "down")]
  
  res[, sig_FDR_0.05 := FDR < 0.05]
  res[, sig_FDR_0.01 := FDR < 0.01]
  res[, sig_FDR_0.05_logFC1 := FDR < 0.05 & abs(logFC) >= 1]
  res[, sig_FDR_0.01_logFC1 := FDR < 0.01 & abs(logFC) >= 1]
  
  prefix <- file.path(OUT_DIR, "per_drug", paste0(sanitize_label(drug_i), "_vs_DMSO"))
  
  fwrite(res, paste0(prefix, ".full_results.tsv.gz"), sep = "\t")
  fwrite(res[FDR < 0.05], paste0(prefix, ".FDR0.05.tsv.gz"), sep = "\t")
  fwrite(res[FDR < 0.05 & abs(logFC) >= 1], paste0(prefix, ".FDR0.05_logFC1.tsv.gz"), sep = "\t")
  fwrite(res[FDR < 0.01 & abs(logFC) >= 1], paste0(prefix, ".FDR0.01_logFC1.tsv.gz"), sep = "\t")
  
  top_up <- res[logFC > 0][order(FDR, -logFC)][1:min(.N, 100)]
  top_down <- res[logFC < 0][order(FDR, logFC)][1:min(.N, 100)]
  
  if (nrow(top_up) > 0) {
    top_up[, marker_class := "top_up"]
    top_up[, rank_in_class := seq_len(.N)]
  }
  if (nrow(top_down) > 0) {
    top_down[, marker_class := "top_down"]
    top_down[, rank_in_class := seq_len(.N)]
  }
  
  top_markers <- rbindlist(list(top_up, top_down), fill = TRUE)
  fwrite(top_markers, paste0(prefix, ".top100_up_down.tsv.gz"), sep = "\t")
  
  summary_i <- data.table(
    condition = drug_i,
    comparison = paste0(drug_i, "_vs_DMSO"),
    n_total_genes = nrow(res),
    n_FDR_0.05 = sum(res$FDR < 0.05, na.rm = TRUE),
    n_FDR_0.01 = sum(res$FDR < 0.01, na.rm = TRUE),
    n_FDR_0.05_logFC1 = sum(res$FDR < 0.05 & abs(res$logFC) >= 1, na.rm = TRUE),
    n_FDR_0.01_logFC1 = sum(res$FDR < 0.01 & abs(res$logFC) >= 1, na.rm = TRUE),
    n_up_FDR_0.05_logFC1 = sum(res$FDR < 0.05 & res$logFC >= 1, na.rm = TRUE),
    n_down_FDR_0.05_logFC1 = sum(res$FDR < 0.05 & res$logFC <= -1, na.rm = TRUE)
  )
  
  all_results_list[[drug_i]] <- res
  summary_list[[drug_i]] <- summary_i
  top_list[[drug_i]] <- top_markers
}

cat("[6/8] Writing combined tables...\n")

all_results <- rbindlist(all_results_list, fill = TRUE)
de_summary <- rbindlist(summary_list, fill = TRUE)
top_markers_all <- rbindlist(top_list, fill = TRUE)

fwrite(
  all_results,
  file.path(OUT_DIR, "summary", "AD715_all_drugs_vs_DMSO.full_results.tsv.gz"),
  sep = "\t"
)

fwrite(
  de_summary,
  file.path(OUT_DIR, "summary", "AD715_DE_summary.tsv"),
  sep = "\t"
)

fwrite(
  top_markers_all,
  file.path(OUT_DIR, "summary", "AD715_top100_up_down_all_drugs.tsv.gz"),
  sep = "\t"
)

cat("[7/8] Helper tables...\n")

sig_mat <- dcast(
  all_results[, .(
    gene,
    condition,
    sig = as.integer(FDR < 0.05 & abs(logFC) >= 1)
  )],
  gene ~ condition,
  value.var = "sig",
  fill = 0
)

condition_cols <- setdiff(colnames(sig_mat), "gene")
sig_mat[, n_conditions_sig := rowSums(.SD), .SDcols = condition_cols]

fwrite(
  sig_mat,
  file.path(OUT_DIR, "summary", "AD715_gene_significance_matrix_FDR0.05_logFC1.tsv.gz"),
  sep = "\t"
)

shared <- sig_mat[n_conditions_sig >= 2][order(-n_conditions_sig)]
fwrite(
  shared,
  file.path(OUT_DIR, "summary", "AD715_shared_response_candidates_sig_in_ge2_conditions.tsv.gz"),
  sep = "\t"
)

best_hit <- all_results[, .SD[which.min(FDR)], by = gene]
best_hit[, abs_logFC := abs(logFC)]
setorder(best_hit, FDR, -abs_logFC)
best_hit[, abs_logFC := NULL]

fwrite(
  best_hit,
  file.path(OUT_DIR, "summary", "AD715_best_hit_per_gene_across_all_drugs.tsv.gz"),
  sep = "\t"
)

cat("[8/8] Run summary...\n")

run_summary <- data.table(
  input_count_file = COUNT_FILE,
  input_meta_file = META_FILE,
  n_input_samples = length(sample_ids),
  n_input_genes = length(genes),
  n_samples_used = ncol(dge),
  n_genes_after_filter = nrow(dge),
  drugs_tested = paste(drug_names, collapse = ", ")
)

fwrite(run_summary, file.path(OUT_DIR, "AD715_edgeR_run_summary.tsv"), sep = "\t")

cat("Done.\n")
cat("Output directory: ", OUT_DIR, "\n", sep = "")
