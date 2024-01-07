import warnings
import pandas as pd
import scanpy as sc
import numpy as np
import scipy
from .dimensionality import *
from .workflows import *


def stitch(adata, timepoint_obs, n_neighbors=200, distance='correlation', vscore_min_pctl=95, vscore_batch_key=None, vscore_filter_method=None):

  # Determine the # of timepoints in adata
  timepoint_list = np.unique(adata.obs[timepoint_obs])
  print(timepoint_list)
  n_timepoints = len(timepoint_list)
  n_stitch_rounds = n_timepoints - 1

  # Sort adata by timepoint
  time_sort_index = adata.obs['stage.integer'].sort_values(inplace=False).index
  adata = adata[time_sort_index,:]

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
      print('This is round:', n+1)

      # Specify individual adatas for the two timepoints in this round
      adata_t1 = adata_list[n].copy()
      adata_t2 = adata_list[n+1].copy()

      print(np.unique(adata_t2.obs[timepoint_obs]))

      # Normalize the two timepoints separately
      dew.pp_raw2norm(adata_t1)
      dew.pp_raw2norm(adata_t2)

      # Define variable genes and nPCs for t2
      dew.get_variable_genes(adata_t2, batch_key=vscore_batch_key, filter_method=vscore_filter_method, min_vscore_pctl=vscore_min_pctl)
      dew.get_significant_pcs(adata_t2, n_iter=1, show_plots=False, verbose=False)
      print(np.sum(np.sum(adata_t2.var['highly_variable'])), adata_t2.uns['n_sig_PCs'])

      # Get a pca embedding for t2
      sc.pp.pca(adata_t2, n_comps=adata_t2.uns['n_sig_PCs'], zero_center=True)
      sc.pp.neighbors(adata_t2, n_neighbors=n_neighbors, n_pcs=adata_t2.uns['n_sig_PCs'], metric=distance, use_rep='X_pca')

      # Project t1 into the pca subspace defined for t2
      sc.tl.ingest(adata_t1, adata_t2, embedding_method='pca')

      # Concatenate the pca projections for t1 & t2
      adata_t1t2 = adata_t1.concatenate(adata_t2, batch_categories=['t1', 't2'])

      # Generate a t1-t2 neighbor graph in the joint pca space
      sc.pp.neighbors(adata_t1t2, n_neighbors=n_neighbors, metric=distance)
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

  # Convert the edge list back to a csr graph
  adata.obsp['connectivities'] = scipy.sparse.coo_matrix((combined_edge_df['connectivity'], (combined_edge_df['n1'], combined_edge_df['n2']))).tocsr()

  # Generate a combined UMAP for all timepoints
  adata.uns['neighbors'] = stitch_neighbors_settings
  sc.tl.umap(adata)

  return adata