import warnings
import pandas as pd
import scanpy as sc
import numpy as np
import scipy
from .dimensionality import *
from .workflows import *


# (from ingest)
#def project_to_pca(adata, n_pcs=None):
#    X = adata.X
#    X = X.toarray() if issparse(X) else X.copy()
#    X = X[:, adata_ref.var["highly_variable"]]
#    X_pca = np.dot(X, adata.varm["PCs"][:, :n_pcs])
#    return X_pca


def stitch(adata, timepoint_obs, batch_obs=None, n_neighbors=15, distance_metric='correlation', vscore_min_pctl=95, vscore_filter_method=None):

  # Determine the # of timepoints in adata
  timepoint_list = np.unique(adata.obs[timepoint_obs])
  n_timepoints = len(timepoint_list)
  n_stitch_rounds = n_timepoints - 1

  # Sort the cells in adata by timepoint
  time_sort_index = adata.obs[timepoint_obs].sort_values(inplace=False).index
  adata = adata[time_sort_index,:].copy()

  # Generate a list of individual timepoint adatas
  adata_list = []
  for tp in timepoint_list:
    adata_list.append(adata[adata.obs[timepoint_obs]==tp])

  # Get edge lists for each timepoint pair
  base_counter = 0
  edge_lists = []
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for n in range(n_stitch_rounds):
      print('Stitching Timepoints:', timepoint_list[n], '-', timepoint_list[n+1])

      # Specify individual adatas for the two timepoints in this round
      adata_t1 = adata_list[n].copy()
      adata_t2 = adata_list[n+1].copy()

      # Normalize the two timepoints separately
      pp_raw2norm(adata_t1)
      pp_raw2norm(adata_t2)

      # Define variable genes and nPCs for t2
      get_variable_genes(adata_t2, batch_key=batch_obs, filter_method=vscore_filter_method, min_vscore_pctl=vscore_min_pctl)
      nPCs_test_use = np.min([300, np.sum(adata_t2.var.highly_variable)-1])
      get_significant_pcs(adata_t2, n_iter=1, nPCs_test = nPCs_test_use, show_plots=False, verbose=False)
      print(np.sum(np.sum(adata_t2.var['highly_variable'])), adata_t2.uns['n_sig_PCs'])

      # Get a pca embedding for t2
      sc.pp.pca(adata_t2, n_comps=adata_t2.uns['n_sig_PCs'], zero_center=True)
      sc.pp.neighbors(adata_t2, n_neighbors=n_neighbors, n_pcs=adata_t2.uns['n_sig_PCs'], metric=distance_metric, use_rep='X_pca')

      # Project t1 into the pca subspace defined for t2
      sc.tl.ingest(adata_t1, adata_t2, embedding_method='pca')

      # Concatenate the pca projections for t1 & t2
      adata_t1t2 = adata_t1.concatenate(adata_t2, batch_categories=['t1', 't2'])

      # Generate a t1-t2 neighbor graph in the joint pca space
      if True: # include Harmony batch correction
        sc.external.pp.harmony_integrate(adata_t1t2, batch_obs, basis='X_pca', adjusted_basis='X_pca_harmony', max_iter_harmony=20, verbose=False)
        sc.pp.neighbors(adata_t1t2, n_neighbors=n_neighbors, metric=distance_metric, use_rep='X_pca_harmony')
        del adata_t1t2.uns['neighbors']['params']['use_rep']
      else: # version without Harmony
        sc.pp.neighbors(adata_t1t2, n_neighbors=n_neighbors, metric=distance_metric)
        stitch_neighbors_settings = adata_t1t2.uns['neighbors']

      # Convert csr graph connectivities to an edge list
      X = adata_t1t2.obsp['connectivities']
      edge_df = pd.DataFrame([[n1, n2, X[n1,n2]] for n1, n2 in zip(*X.nonzero())], columns=['n1','n2','connectivity'])

      # Adjust the node ids in the edge list based on their overall order
      edge_df['n1'] = edge_df['n1'] + base_counter
      edge_df['n2'] = edge_df['n2'] + base_counter
      edge_lists.append(edge_df)

      # Increase base_counter by the # of cells in adata_t1
      base_counter = base_counter + len(adata_t1)

  # Merge all edge lists
  combined_edge_df = pd.concat(edge_lists)

  # Store STITCH graph and neighbors settings to adata
  adata.obsp['connectivities'] = scipy.sparse.coo_matrix((combined_edge_df['connectivity'], (combined_edge_df['n1'], combined_edge_df['n2']))).tocsr().copy()
  adata.uns['neighbors'] = adata_t1t2.uns['neighbors']

  return adata

