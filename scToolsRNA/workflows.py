
import scanpy as sc
import numpy as np
from .dimensionality import *


def pp_raw2norm(adata, batch_key=None):
	
	# Store raw counts as separate layer
	adata.layers['raw_nolog'] = adata.X.copy()
	adata.layers['raw'] = np.log1p(adata.X.copy())
	
	# Perform total counts normalization
	sc.pp.normalize_total(adata, target_sum=1e6, inplace=True) # TPM Normalization

	# Store tpm counts as a separate layer
	adata.layers['tpm_nolog'] = adata.X.copy()
	sc.pp.log1p(adata)
	adata.layers['tpm'] = adata.X.copy()
	
	# Determine highly variable genes using default ScanPy method: 'seurat', store results in uns
	if batch_key == None:
		adata.uns['highly_variable_scanpy'] = sc.pp.highly_variable_genes(adata, max_mean=10, min_mean=0.1, inplace=False)
	else:
		adata.uns['highly_variable_scanpy'] = sc.pp.highly_variable_genes(adata, batch_key=batch_key, max_mean=10, min_mean=0.1, inplace=False)

	# Scale the genes by z-score (large sparse matrix-friendly)
	sc.pp.scale(adata, zero_center=False)

	return adata


def pp_get_dims(adata, batch_key=None, verbose=True):

	# Estimate the dimensionality of the dataset
	adata = run_dim_tests(adata, dim_test_n_comps_test=300, dim_test_n_trials=3, dim_test_vpctl=None, verbose=verbose)

	# Get the opimal # of variable genes and PC dimensions
	get_variable_genes(adata, norm_counts_per_cell=1e6, min_vscore_pctl=adata.uns['optim_vscore_pctl'], min_counts=3, min_cells=3, show_FF_plot=False, show_vscore_plot=False)
	get_significant_pcs(adata, n_iter = 20, n_comps_test = 300, show_plots=True, zero_center=True)

	return adata


def pp_get_embedding(adata, batch_key=None):

	# Perform PCA with a specified number of dimensions
	sc.pp.pca(adata, n_comps=adata.uns['n_sig_PCs'], zero_center=True)

	# Generate neighbor graph, incorporating Harmony integration if necessary
	if batch_key == None:
		sc.pp.neighbors(adata, n_neighbors=10, n_pcs=adata.uns['n_sig_PCs'], metric='euclidean', use_rep='X_pca')
	else:
		sc.external.pp.harmony_integrate(adata, batch_key, basis='X_pca', adjusted_basis='X_pca_harmony')
		sc.pp.neighbors(adata, n_neighbors=10, n_pcs=adata.uns['n_sig_PCs'], metric='euclidean', use_rep='X_pca_harmony')

	# Generate UMAP embedding
	sc.tl.umap(adata, n_components=2, spread=1)
	
	# Perform graph clustering
	sc.tl.leiden(adata, resolution=1, key_added='leiden')
	sc.tl.leiden(adata, resolution=2, key_added='leiden_2')
	sc.tl.leiden(adata, resolution=5, key_added='leiden_5')
	sc.tl.leiden(adata, resolution=10, key_added='leiden_10')

	return adata