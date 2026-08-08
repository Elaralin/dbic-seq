import os
import pandas as pd
from pathlib import Path

infile = Path(
    os.environ.get(
        "DBIC_DRUG_METADATA_INPUT",
        "data/AD715/morphology_rna_linked.tsv"
    )
)
outdir = Path("step1_drug_metadata")
outdir.mkdir(exist_ok=True)

df = pd.read_csv(infile, sep="\t")

# parse morphology row/col from spot_id
rc = df["spot_id"].astype(str).str.extract(r"R(\d+)C(\d+)")
df["morph_row"] = rc[0].astype(int)
df["morph_col"] = rc[1].astype(int)

# AD715 drug layout:
# R01-R06 = DMSO
# R07-R10 = Drug01
# R11-R14 = Drug02
# ...
# R47-R50 = Drug11
def assign_condition(r):
    if 1 <= r <= 6:
        return "DMSO"
    idx = (r - 7) // 4 + 1
    if 1 <= idx <= 11:
        return f"Drug{idx:02d}"
    return "Unassigned"

df["condition"] = df["morph_row"].apply(assign_condition)

meta_cols = [
    "spot_id",
    "morph_row",
    "morph_col",
    "rna_row",
    "rna_col",
    "condition",
    "nCount_RNA",
    "nFeature_RNA",
    "log10_UMI",
    "log10_Genes"
]

meta_cols = [c for c in meta_cols if c in df.columns]

meta = df[meta_cols].copy()

meta.to_csv(outdir / "AD715_drug_cell_metadata.tsv", sep="\t", index=False)

summary = (
    meta.groupby("condition")
    .agg(
        n_cells=("spot_id", "count"),
        median_UMI=("nCount_RNA", "median"),
        median_genes=("nFeature_RNA", "median")
    )
    .reset_index()
    .sort_values("condition")
)

summary.to_csv(outdir / "AD715_drug_cell_count_summary.tsv", sep="\t", index=False)

# also save linked table with condition added
df.to_csv(outdir / "AD715_linked_with_drug_condition.tsv", sep="\t", index=False)

print("Input cells:", df.shape[0])
print("Input columns:", df.shape[1])
print("\nCondition summary:")
print(summary.to_string(index=False))

print("\nSaved:")
print(outdir / "AD715_drug_cell_metadata.tsv")
print(outdir / "AD715_drug_cell_count_summary.tsv")
print(outdir / "AD715_linked_with_drug_condition.tsv")
