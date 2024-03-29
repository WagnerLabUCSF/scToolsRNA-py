
import scanpy as sc
import numpy as np
from .dimensionality import *


def pp_raw2norm(adata, include_raw_layers=True, include_tpm_layers=True):
	
	# Store raw counts as separate layer
	if include_raw_layers: adata.layers['raw_nolog'] = adata.X.copy()
	if include_raw_layers: adata.layers['raw'] = np.log1p(adata.X.copy())
	
	# Perform total counts normalization
	sc.pp.normalize_total(adata, target_sum=1e6, inplace=True) # TPM Normalization

	# Store tpm counts as a separate layer
	if include_tpm_layers: adata.layers['tpm_nolog'] = adata.X.copy()
	sc.pp.log1p(adata)
	if include_tpm_layers: adata.layers['tpm'] = adata.X.copy()
	
	# Scale the genes by z-score (no zero center = large sparse matrix-friendly)
	sc.pp.scale(adata, zero_center=False)


def pp_get_embedding(adata, batch_key=None, n_neighbors=10, verbose=False, include_umap=True, include_leiden=True):

	# Perform PCA with a specified number of dimensions
	sc.pp.pca(adata, n_comps=adata.uns['n_sig_PCs'], zero_center=True)

	# Generate neighbor graph, incorporating Harmony integration if necessary
	if batch_key == None:
		sc.pp.neighbors(adata, n_neighbors=10, n_pcs=adata.uns['n_sig_PCs'], metric='euclidean', use_rep='X_pca')
	else:
		sc.external.pp.harmony_integrate(adata, batch_key, basis='X_pca', adjusted_basis='X_pca_harmony', max_iter_harmony=20, random_state=0, verbose=verbose)
		sc.pp.neighbors(adata, n_neighbors=10, n_pcs=adata.uns['n_sig_PCs'], metric='euclidean', use_rep='X_pca_harmony')

	# Generate UMAP embedding
	if include_umap:
		sc.tl.umap(adata, n_components=2, spread=1)
	
	# Perform graph clustering
	if include_leiden:
		sc.tl.leiden(adata, resolution=1, key_added='leiden')


