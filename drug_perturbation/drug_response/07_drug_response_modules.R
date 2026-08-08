#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(pheatmap)
})

ROOT <- Sys.getenv("DBIC_DATA_ROOT", unset = "data/AD715")
WORK <- file.path(ROOT, "5_drug_response")

IN_GENE_TABLE <- file.path(
  WORK,
  "07c_drug_response_module_atlas_compact",
  "tables",
  "AD715_compact_representative_response_genes.tsv"
)

IN_HEATMAP <- file.path(
  WORK,
  "07c_drug_response_module_atlas_compact",
  "tables",
  "AD715_compact_response_gene_heatmap_rowZ.tsv"
)

OUT_DIR <- file.path(WORK, "07f_drug_response_module_final")
FIG_DIR <- file.path(OUT_DIR, "figures")
TAB_DIR <- file.path(OUT_DIR, "tables")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TAB_DIR, recursive = TRUE, showWarnings = FALSE)

drug_info <- data.table(
  condition = c("DMSO","Drug01","Drug02","Drug03","Drug04","Drug05","Drug06","Drug07","Drug08","Drug09","Drug10","Drug11"),
  drug_name = c("DMSO","Cisplatin","Etoposide","Doxorubicin","Actinomycin D","Paclitaxel","Epothilone B","Vinblastine","Pelitinib","Raltitrexed","Cyclophosphamide","AT-9283"),
  target_moa = c("Vehicle","DNA crosslinking","Topoisomerase II inhibitor","Topoisomerase II inhibitor","Transcription inhibitor","Microtubule stabilizer","Microtubule stabilizer","Microtubule destabilizer","EGFR inhibitor","Thymidylate synthase inhibitor","Alkylating agent","Aurora kinase inhibitor"),
  drug_class = c("Control","DNA damage","DNA damage","DNA damage","Transcription","Microtubule","Microtubule","Microtubule","Kinase inhibitor","Antimetabolite","DNA damage","Kinase inhibitor")
)

drug_info[, label := paste0(drug_name, "\n(", target_moa, ")")]

module_sets <- list(
  DNA_damage_chromatin = c("XRCC5","HMGB3","HMGN1","PMS2","TERF2","H1-3","VEZF1","ZNF267"),
  Stress_response = c("AHNAK2","DNAJC5","S100A4","TXNIP","HSPA1A","HSPA1B","DNAJB1","ATF3","ATF4"),
  Cell_cycle_mitosis = c("EDC4","BOD1","CCNB1","CDC20","AURKA","AURKB","RCC1","TOP2A"),
  Cytoskeleton_motility = c("DCTN2","PFN2","SYNE2","RHOQ","WDR1","RHOA","CLIP4","CORO1C","STRN3"),
  Adhesion_integrin = c("SLC4A2","ITGA6","ADGRG6","MALL","TGFBI","ITGB5","ITGAV","PTPN12"),
  EGFR_MAPK_signaling = c("PRKCA","KRAS","MAPK9","CREB1","RHOA","RHOQ"),
  Metabolism_mitochondria = c("SLC7A5","COX7A2L","SLC39A10","ACACA","CTPS1","ALDH3B1"),
  Translation_ribosome = c("RPS10","RPS29","RPS24","RPL24"),
  RNA_processing_transcription = c("HNF4A","IRF2BP2","FIP1L1","PHF5A","PAPOLA","CREB1"),
  Ubiquitin_proteasome = c("VCPIP1","NPLOC4","TRIML2","CYP1B1","UBE3B","UBQLN2"),
  Vesicle_membrane_trafficking = c("ARHGEF11","HIP1","LRP8","CCDC66","KIAA0232","SLC26A2")
)

module_order <- c(
  "DNA_damage_chromatin",
  "Stress_response",
  "Cell_cycle_mitosis",
  "Cytoskeleton_motility",
  "Adhesion_integrin",
  "EGFR_MAPK_signaling",
  "Metabolism_mitochondria",
  "Translation_ribosome",
  "RNA_processing_transcription",
  "Ubiquitin_proteasome",
  "Vesicle_membrane_trafficking",
  "Other"
)

module_keep_n <- c(
  DNA_damage_chromatin = 5,
  Stress_response = 4,
  Cell_cycle_mitosis = 4,
  Cytoskeleton_motility = 7,
  Adhesion_integrin = 6,
  EGFR_MAPK_signaling = 4,
  Metabolism_mitochondria = 5,
  Translation_ribosome = 3,
  RNA_processing_transcription = 5,
  Ubiquitin_proteasome = 4,
  Vesicle_membrane_trafficking = 5,
  Other = 4
)

module_colors <- c(
  DNA_damage_chromatin = "#6A3D9A",
  Stress_response = "#E28E2C",
  Cell_cycle_mitosis = "#D73027",
  Cytoskeleton_motility = "#1F78B4",
  Adhesion_integrin = "#1F9E89",
  EGFR_MAPK_signaling = "#33A02C",
  Metabolism_mitochondria = "#A6CEE3",
  Translation_ribosome = "#FDBF6F",
  RNA_processing_transcription = "#CAB2D6",
  Ubiquitin_proteasome = "#FB9A99",
  Vesicle_membrane_trafficking = "#B2DF8A",
  Other = "#D9D9D9"
)

drug_colors <- list(
  DrugClass = c(
    "Control" = "#D0D0D0",
    "DNA damage" = "#6A3D9A",
    "Transcription" = "#E28E2C",
    "Microtubule" = "#86C443",
    "Kinase inhibitor" = "#1F9E89",
    "Antimetabolite" = "#377EB8"
  ),
  Direction = c("Up" = "#E85C47", "Down" = "#3E8FC4"),
  ResponseLayer = c("Shared module" = "#E28E2C", "Drug-specific" = "#2E91C2"),
  GeneModule = module_colors
)

assign_module <- function(gene) {
  g <- toupper(gene)
  for (m in names(module_sets)) {
    if (g %in% toupper(module_sets[[m]])) return(m)
  }
  if (grepl("^ZNF", g)) return("DNA_damage_chromatin")
  return("Other")
}

cat("[1/5] Reading 07c compact gene table and heatmap...\n")

gene_dt <- fread(IN_GENE_TABLE)
hm <- fread(IN_HEATMAP)

setnames(hm, 1, "gene_label")

mat <- as.matrix(hm[, -1, with = FALSE])
rownames(mat) <- hm$gene_label
mode(mat) <- "numeric"

gene_dt[, gene_label_clean := gene_label]
gene_dt[, GeneModule := vapply(gene_label_clean, assign_module, character(1))]
gene_dt[, GeneModule := factor(GeneModule, levels = module_order)]

anno_gene <- gene_dt[match(rownames(mat), gene_label_clean)]
anno_gene[is.na(GeneModule), GeneModule := factor("Other", levels = module_order)]
anno_gene[is.na(ResponseLayer), ResponseLayer := "Drug-specific"]
anno_gene[is.na(best_direction), best_direction := "Up"]

cat("[2/5] Keep top genes within each functional module...\n")

anno_keep <- rbindlist(lapply(module_order, function(m) {
  x <- anno_gene[as.character(GeneModule) == m]
  if (nrow(x) == 0) return(NULL)
  n_keep <- module_keep_n[[m]]
  x[order(-best_score, min_PValue)][1:min(.N, n_keep)]
}), fill = TRUE)

setorder(
  anno_keep,
  GeneModule,
  best_condition,
  -best_score,
  min_PValue
)

row_order <- anno_keep$gene_label_clean
row_order <- row_order[row_order %in% rownames(mat)]

mat2 <- mat[row_order, , drop = FALSE]
anno_gene2 <- anno_keep[match(row_order, gene_label_clean)]

cat("Final genes kept:", nrow(mat2), "\n")
cat("Other genes kept:", sum(as.character(anno_gene2$GeneModule) == "Other"), "\n")

cat("[3/5] Build annotations and labels...\n")

use_conditions <- c(
  "DMSO",
  "Drug01","Drug02","Drug03","Drug10",
  "Drug04",
  "Drug05","Drug06","Drug07",
  "Drug08","Drug11",
  "Drug09"
)
use_conditions <- use_conditions[seq_len(ncol(mat2))]

labels_col <- drug_info[match(use_conditions, condition), label]

anno_col <- data.frame(
  DrugClass = drug_info[match(use_conditions, condition), drug_class]
)
rownames(anno_col) <- colnames(mat2)

anno_row <- data.frame(
  GeneModule = as.character(anno_gene2$GeneModule),
  ResponseLayer = anno_gene2$ResponseLayer,
  Direction = anno_gene2$best_direction
)
rownames(anno_row) <- rownames(mat2)

fwrite(
  anno_gene2,
  file.path(TAB_DIR, "AD715_final_response_genes_with_modules.tsv"),
  sep = "\t"
)

fwrite(
  data.table(gene = rownames(mat2), mat2, check.names = FALSE),
  file.path(TAB_DIR, "AD715_final_response_module_heatmap_rowZ.tsv"),
  sep = "\t"
)

cat("[4/5] Plot final main-panel heatmap...\n")

heat_cols <- colorRampPalette(c("#3E8FC4", "#F2E1CF", "#E85C47"))(100)
gaps <- which(diff(as.integer(factor(anno_gene2$GeneModule, levels = module_order))) != 0)

pdf(
  file.path(FIG_DIR, "AD715_drug_response_module_atlas_final_mainpanel.pdf"),
  width = 10.8,
  height = 6.8
)
pheatmap(
  mat2,
  color = heat_cols,
  breaks = seq(-1.5, 1.5, length.out = 101),
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  border_color = NA,
  show_rownames = FALSE,
  show_colnames = TRUE,
  labels_col = labels_col,
  fontsize = 8,
  fontsize_col = 8,
  angle_col = 45,
  annotation_col = anno_col,
  annotation_row = anno_row,
  annotation_colors = drug_colors,
  annotation_legend = TRUE,
  gaps_row = gaps,
  main = "AD715 drug-response module atlas"
)
dev.off()

png(
  file.path(FIG_DIR, "AD715_drug_response_module_atlas_final_mainpanel.png"),
  width = 2350,
  height = 1550,
  res = 220
)
pheatmap(
  mat2,
  color = heat_cols,
  breaks = seq(-1.5, 1.5, length.out = 101),
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  border_color = NA,
  show_rownames = FALSE,
  show_colnames = TRUE,
  labels_col = labels_col,
  fontsize = 8,
  fontsize_col = 8,
  angle_col = 45,
  annotation_col = anno_col,
  annotation_row = anno_row,
  annotation_colors = drug_colors,
  annotation_legend = TRUE,
  gaps_row = gaps,
  main = "AD715 drug-response module atlas"
)
dev.off()

cat("[5/5] Labeled review version and summary...\n")

pdf(
  file.path(FIG_DIR, "AD715_drug_response_module_atlas_final_labeled_genes.pdf"),
  width = 11.8,
  height = 9.5
)
pheatmap(
  mat2,
  color = heat_cols,
  breaks = seq(-1.5, 1.5, length.out = 101),
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  border_color = NA,
  show_rownames = TRUE,
  show_colnames = TRUE,
  labels_col = labels_col,
  fontsize = 6,
  fontsize_row = 5.5,
  fontsize_col = 8,
  angle_col = 45,
  annotation_col = anno_col,
  annotation_row = anno_row,
  annotation_colors = drug_colors,
  annotation_legend = TRUE,
  gaps_row = gaps,
  main = "AD715 final representative drug-response genes"
)
dev.off()

summary <- anno_gene2[, .N, by = GeneModule][order(match(GeneModule, module_order))]
fwrite(summary, file.path(OUT_DIR, "AD715_final_module_summary.tsv"), sep = "\t")

run_summary <- data.table(
  metric = c(
    "input_gene_source",
    "selection_basis",
    "final_n_genes",
    "max_other_genes"
  ),
  value = c(
    "07c compact genes from edgeR drug-vs-DMSO ranked DE genes",
    "PValue<0.01, abs(logFC)>=1, protein_coding exon, response in <=2 drugs, top4 up/down per drug; then top genes per functional module",
    nrow(mat2),
    module_keep_n[["Other"]]
  )
)

fwrite(run_summary, file.path(OUT_DIR, "AD715_final_atlas_run_summary.tsv"), sep = "\t")

cat("Done.\n")
cat("Output directory:", OUT_DIR, "\n")
