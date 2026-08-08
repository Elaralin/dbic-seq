import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.environ.get("DBIC_DATA_ROOT", "data/AD715"))

expr_file = ROOT / "0_matrix/expmat_715.tsv"
cell_file = ROOT / "0_core_data/AD715_integrated_cells.tsv"
drug_file = ROOT / "5_drug_response/05_pseudobulk/qc/AD715_linked_cells_used.tsv"

outdir = ROOT / "6_single_cell/01_matrix"
outdir.mkdir(parents=True, exist_ok=True)

cells = pd.read_csv(cell_file, sep="\t")
drug = pd.read_csv(drug_file, sep="\t")

# merge drug condition
cells = cells.merge(
    drug[["spot_id", "matrix_col", "condition", "replicate", "pseudobulk_id"]],
    on="spot_id",
    how="left"
)

# official RNA column
if "rna_colname" in cells.columns:
    cells["matrix_col_final"] = cells["rna_colname"].astype(str)
else:
    cells["matrix_col_final"] = cells["matrix_col"].astype(str)

# read expression matrix
expr = pd.read_csv(expr_file, sep="\t", index_col=0)

keep_cols = [c for c in cells["matrix_col_final"] if c in expr.columns]
expr_sub = expr[keep_cols]

meta = cells[cells["matrix_col_final"].isin(keep_cols)].copy()
meta = meta.drop_duplicates("matrix_col_final")

expr_sub.to_csv(outdir / "AD715_scRNA_counts_all1275.tsv", sep="\t")
meta.to_csv(outdir / "AD715_scRNA_metadata_all1275.tsv", sep="\t", index=False)

# Export subsets defined by precomputed RNA quality-control flags in the input metadata
for flag in ["pass_RNA_300_100", "pass_RNA_500_100"]:
    if flag in meta.columns:
        m = meta[meta[flag] == True].copy()
        cols = [c for c in m["matrix_col_final"] if c in expr_sub.columns]
        expr_sub[cols].to_csv(outdir / f"AD715_scRNA_counts_{flag}.tsv", sep="\t")
        m.to_csv(outdir / f"AD715_scRNA_metadata_{flag}.tsv", sep="\t", index=False)
        print(flag, len(cols))

print("all cells:", expr_sub.shape[1])
print("genes:", expr_sub.shape[0])
print("saved:", outdir)
