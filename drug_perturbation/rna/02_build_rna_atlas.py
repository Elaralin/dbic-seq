import os
import pandas as pd
from pathlib import Path
import scanpy as sc

ROOT = Path(os.environ.get("DBIC_DATA_ROOT", "data/AD715"))

count_file = Path(
    os.environ.get(
        "DBIC_RNA_COUNTS_FILE",
        "data/AD715/AD715_scRNA_counts_gene_symbol.tsv"
    )
)

meta_file = Path(
    os.environ.get(
        "DBIC_RNA_METADATA_FILE",
        "data/AD715/AD715_scRNA_metadata.tsv"
    )
)
outdir = ROOT / "6_single_cell/03_rna_atlas"
figdir = ROOT / "6_single_cell/03_rna_atlas/figures"
tabdir = ROOT / "6_single_cell/03_rna_atlas/tables"
outdir.mkdir(parents=True, exist_ok=True)
figdir.mkdir(parents=True, exist_ok=True)
tabdir.mkdir(parents=True, exist_ok=True)

counts = pd.read_csv(count_file, sep="\t", index_col=0)
meta = pd.read_csv(meta_file, sep="\t")

# cells x genes
adata = sc.AnnData(counts.T)
adata.obs = meta.set_index("matrix_col_final").loc[adata.obs_names]

# basic filtering
sc.pp.filter_genes(adata, min_cells=3)

adata.layers["counts"] = adata.X.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
adata = adata[:, adata.var["highly_variable"]].copy()

sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
sc.tl.umap(adata, min_dist=0.4)

for res in [0.2, 0.4, 0.6, 0.8]:
    sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{res}")

adata.write(outdir / "AD715_scRNA_atlas.h5ad")

umap = pd.DataFrame(
    adata.obsm["X_umap"],
    index=adata.obs_names,
    columns=["UMAP1", "UMAP2"]
)
umap = pd.concat([umap, adata.obs], axis=1)
umap.to_csv(tabdir / "AD715_scRNA_umap_metadata.tsv", sep="\t")

adata.var.to_csv(tabdir / "AD715_scRNA_HVG_genes.tsv", sep="\t")

# figures
sc.settings.figdir = str(figdir)

if "condition" in adata.obs.columns:
    sc.pl.umap(adata, color="condition", save="_condition.pdf", show=False)

for key in ["leiden_0.2", "leiden_0.4", "leiden_0.6", "leiden_0.8"]:
    sc.pl.umap(adata, color=key, legend_loc="on data", save=f"_{key}.pdf", show=False)

print("saved h5ad:", outdir / "AD715_scRNA_atlas.h5ad")
print("cells:", adata.n_obs)
print("HVG genes:", adata.n_vars)
