# scToolsRNA

A workbench of general-purpose single-cell RNA-seq analysis tools from the
**Wagner Lab (UCSF)**.

`scToolsRNA` collects utilities that have accumulated across lab projects into a
single pip-installable package built on top of [scanpy](https://scanpy.readthedocs.io)
and [AnnData](https://anndata.readthedocs.io). It covers preprocessing and QC,
feature selection and dimensionality reduction, differential expression,
kNN label transfer, trajectory analysis, network export, plotting, and I/O.

It is a companion to the lab's dataset-specific
[`zmap-tools`](https://github.com/WagnerLabUCSF/zmap-tools) package: where
`zmap-tools` is specialized for the Zebrafish Multi-Atlas Project, `scToolsRNA`
holds the organism-agnostic building blocks meant for everyday use.

## Installation

```bash
pip install sctoolsrna
```

or, from a checkout of this repository:

```bash
pip install .
```

All runtime dependencies (scanpy, anndata, scikit-learn, faiss-cpu, pydeseq2,
scrublet, umap-learn, harmonypy, plotly, igraph, leidenalg, …) are installed
automatically. Individual modules import heavier or optional dependencies lazily
(inside the functions that use them), so `import scToolsRNA` stays fast and does
not fail if a single optional system library is missing.

## Usage

The distribution installs as `sctoolsrna`, but the import name is `scToolsRNA`:

```python
import scanpy as sc
import scToolsRNA as sct

adata = sc.read_h5ad("my_data.h5ad")

# Feature selection + significant-PC estimation
sct.get_variable_genes(adata, top_n_genes=3000)
sct.get_sig_pcs(adata)

# kNN label transfer from a labeled reference
sct.transfer_labels_knn(
    adata_query,
    adata_ref,
    ref_label_col="cell_type",
    ref_basis="X_pca_harmony",
    query_basis="X_pca_harmony",
)
```

Every public function is also re-exported at the top level, so
`from scToolsRNA import get_variable_genes` continues to work for existing code.

## Modules

| Module | Contents |
| --- | --- |
| `preprocess` | Barcode/mito/ribo/doublet filtering and sampling QC |
| `dimensionality` | Variable-gene (V-score) selection, covarying genes, significant-PC estimation |
| `workflows` | Convenience pipelines (raw → normalized → UMAP/Leiden) |
| `diffexp` | Pseudobulk pyDESeq2 contrasts, DEG tables, volcano/clustermap plots |
| `classification` | Train/predict per-cell classifiers (sklearn) |
| `knn` | Portable FAISS/scikit-learn k-nearest-neighbor search |
| `labeltransfer` | kNN label and continuous-value transfer between datasets |
| `trajectories` | Diffusion-pseudotime dynamic-gene detection and plotting |
| `stitch` | STITCH temporal graph construction and diagnostics |
| `network` | GraphML / Pajek export for Gephi |
| `plotting` | 3D embeddings, UMAP animations, axis helpers |
| `colormaps` | Custom matplotlib colormaps |
| `utils` | Label smoothing, confusion matrices, stacked barplots |
| `readwrite` | STARsolo / alevin / alevin-fry / inDrops loaders, cell/gene metadata |
| `sparse` | Sparse-matrix helpers |

## License

See [LICENSE](LICENSE).
