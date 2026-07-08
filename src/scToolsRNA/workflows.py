
import scanpy as sc
import numpy as np
from .dimensionality import *
from .sparse import normalize_log1p


def adata2tpt(adata):

    # Perform TPT Normalization on X matrix of an adata object
    adata_tpt = adata.copy()
    # CP10K (target_sum=1e4) library-size normalization + log1p via shared helper
    adata_tpt.X = normalize_log1p(adata_tpt.layers['raw_nolog'], target_sum=1e4)
    adata_tpt.uns['log1p'] = {'base': None}

    return adata_tpt


def pp_raw2norm(adata, include_raw_layers=True, include_tpm_layers=True):

	# Store raw counts as separate layer
	if include_raw_layers: adata.layers['raw_nolog'] = adata.X.copy()
	if include_raw_layers: adata.layers['raw'] = sc.pp.log1p(adata.X.copy())

	# TPM (target_sum=1e6) library-size normalization via shared helper; keep the
	# un-logged result as the tpm_nolog layer, then log1p for X and the tpm layer.
	adata.X = normalize_log1p(adata.X, target_sum=1e6, log=False)
	if include_tpm_layers: adata.layers['tpm_nolog'] = adata.X.copy()
	sc.pp.log1p(adata)
	if include_tpm_layers: adata.layers['tpm'] = adata.X.copy()

	# Scale the genes by z-score (no zero center = large sparse matrix-friendly)
	sc.pp.scale(adata, zero_center=False)


def pp_norm2umap(adata, batch_key=None, n_neighbors=15, verbose=False, include_umap=True, include_leiden=True):

	# Perform PCA with a specified number of dimensions
	sc.pp.pca(adata, n_comps=adata.uns['n_sig_PCs'], zero_center=True)

	# Generate neighbor graph, incorporating Harmony integration if necessary
	if batch_key is not None:
		sc.external.pp.harmony_integrate(adata, batch_key, basis='X_pca', adjusted_basis='X_pca_harmony', max_iter_harmony=20, random_state=0, verbose=verbose)
		sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=adata.uns['n_sig_PCs'], metric='euclidean', use_rep='X_pca_harmony')
	else:
		sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=adata.uns['n_sig_PCs'], metric='euclidean', use_rep='X_pca')
	
	# Generate UMAP embedding
	if include_umap: sc.tl.umap(adata, n_components=2, spread=1)
	
	# Perform graph clustering
	if include_leiden: sc.tl.leiden(adata, resolution=1, key_added='leiden')


# LEGACY ALIASES
pp_get_embedding = pp_norm2umap