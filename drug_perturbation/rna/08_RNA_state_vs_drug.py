import os
import scanpy as sc
import pandas as pd
from pathlib import Path

ROOT = Path(os.environ.get("DBIC_DATA_ROOT", "data/AD715"))

rna_state_h5ad = Path(
    os.environ.get(
        "DBIC_RNA_STATE_H5AD",
        "data/AD715/AD715_RNA_state_atlas.h5ad"
    )
)

adata = sc.read_h5ad(rna_state_h5ad)

tab = pd.crosstab(
    adata.obs["condition"],
    adata.obs["RNA_state"],
    normalize="index"
)

outdir = ROOT/"6_single_cell/08_RNA_state_vs_drug"
outdir.mkdir(exist_ok=True)

tab.to_csv(
    outdir/"AD715_condition_vs_RNA_state_fraction.tsv",
    sep="\t"
)

print(tab.round(3))
