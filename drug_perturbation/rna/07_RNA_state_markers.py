import os
import scanpy as sc
import pandas as pd
from pathlib import Path

ROOT = Path(os.environ.get("DBIC_DATA_ROOT", "data/AD715"))

h5ad = Path(
    os.environ.get(
        "DBIC_RNA_STATE_H5AD",
        "data/AD715/AD715_RNA_state_atlas.h5ad"
    )
)

counts_file = Path(
    os.environ.get(
        "DBIC_RNA_COUNTS_FILE",
        "data/AD715/AD715_scRNA_counts_gene_symbol.tsv"
    )
)

outdir = ROOT / "6_single_cell/07_RNA_state_markers"
figdir = outdir / "figures"
tabdir = outdir / "tables"
figdir.mkdir(parents=True, exist_ok=True)
tabdir.mkdir(parents=True, exist_ok=True)

atlas = sc.read_h5ad(h5ad)

# Use full gene-symbol matrix for markers
counts = pd.read_csv(counts_file, sep="\t", index_col=0)
adata = sc.AnnData(counts.T)

common = adata.obs_names.intersection(atlas.obs_names)
adata = adata[common].copy()
atlas = atlas[common].copy()

adata.obs = atlas.obs.copy()
adata.obsm["X_umap"] = atlas.obsm["X_umap"]

sc.pp.filter_genes(adata, min_cells=3)
adata.layers["counts"] = adata.X.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.tl.rank_genes_groups(
    adata,
    groupby="RNA_state",
    method="wilcoxon",
    pts=True
)

markers = sc.get.rank_genes_groups_df(adata, group=None)
markers.to_csv(tabdir / "AD715_RNA_state_markers_all.tsv", sep="\t", index=False)

top = (
    markers
    .query("pvals_adj < 0.05")
    .sort_values(["group", "scores"], ascending=[True, False])
    .groupby("group")
    .head(50)
)

top.to_csv(tabdir / "AD715_RNA_state_markers_top50.tsv", sep="\t", index=False)

top10_genes = (
    top.groupby("group")
    .head(10)["names"]
    .unique()
    .tolist()
)

sc.settings.figdir = str(figdir)

sc.pl.dotplot(
    adata,
    var_names=top10_genes,
    groupby="RNA_state",
    standard_scale="var",
    save="_AD715_RNA_state_top10_marker_dotplot.pdf",
    show=False
)

adata.write(outdir / "AD715_RNA_state_marker_analysis.h5ad")

print("Saved:", outdir)
print("Top marker table:")
print(tabdir / "AD715_RNA_state_markers_top50.tsv")
print("Top genes:", top10_genes[:20])
