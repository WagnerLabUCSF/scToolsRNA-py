import warnings
import pandas as pd
import scanpy as sc
import numpy as np
import scipy
import logging
from contextlib import contextmanager
from umap.umap_ import fuzzy_simplicial_set
from .dimensionality import *
from .workflows import *

@contextmanager
def logging_disabled(highest_level=logging.CRITICAL):
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)


def get_knn_ind_dist_from_csr(D: scipy.sparse.csr_matrix, n_neighbors: int):
    indices = np.zeros((D.shape[0], n_neighbors), dtype=int)
    distances = np.zeros((D.shape[0], n_neighbors), dtype=D.dtype)
    n_neighbors_m1 = n_neighbors - 1
    for i in range(indices.shape[0]):
        neighbors = D[i].nonzero()  # 'true' and 'spurious' zeros
        indices[i, 0] = i
        distances[i, 0] = 0
        # account for cases when there are more than n_neighbors due to an approximate search
        # [the point itself was not detected as its own neighbor during the search]
        if len(neighbors[1]) > n_neighbors_m1:
            sorted_indices = np.argsort(D[i][neighbors].A1)[:n_neighbors_m1]
            indices[i, 1:] = neighbors[1][sorted_indices]
            distances[i, 1:] = D[i][neighbors[0][sorted_indices], neighbors[1][sorted_indices]]
        # cases when n_neighbors match
        elif len(neighbors[1]) == n_neighbors_m1:
            indices[i, 1:] = neighbors[1]
            distances[i, 1:] = D[i][neighbors]
        # cases when there are fewer than n_neighbors
        else:
            print('Error - fewer than n_neighbors found for some nodes')
            return
    return indices, distances


def get_connectivities_from_dist_csr(D_csr, n_neighbors):
  # Extract knn indices and distances from sparse matrix
  n_nodes = D_csr.shape[0]
  knn_indices, knn_distances = get_knn_ind_dist_from_csr(D_csr, n_neighbors)
  D_empty = scipy.sparse.coo_matrix(([], ([], [])), shape=(n_nodes, 1))
  connectivities = fuzzy_simplicial_set(D_empty,
                                        n_neighbors,
                                        None,
                                        'correlation',
                                        knn_indices=knn_indices,
                                        knn_dists=knn_distances)[0]
  return connectivities


def stitch(adata, timepoint_obs, batch_obs=None, n_neighbors=15, distance_metric='correlation', vscore_min_pctl=95, vscore_filter_method=None, method='forward', use_harmony=True, max_iter_harmony=20, verbose=True):

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

  # Set directionality of time projections
  if method=='forward':
    arrow_str = '->'
    anchor_round = n_stitch_rounds - 1 # anchor = last round
  elif method=='reverse':
    arrow_str = '<-'
    anchor_round = 0 # anchor = first round

  # Initialize STITCH data lists
  base_counter = 0
  X_d_stitch_rows = []
  X_d_stitch_cols = []
  X_d_stitch_data = []
  stitch_nHVgenes = []
  stitch_nSigPCs = []

  # Get neighbor graph for each stitch_round (each timepoint pair)
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for n in range(n_stitch_rounds):
      
      if verbose: print('Stitching Timepoints:', timepoint_list[n], arrow_str, timepoint_list[n+1])

      # Specify the reference and projection adatas this round
      adata_t1 = adata_list[n].copy()
      adata_t2 = adata_list[n+1].copy()
      if method=='forward':
        adata_ref = adata_t2
        adata_prj = adata_t1
      elif method=='reverse':
        adata_ref = adata_t1
        adata_prj = adata_t2

      # Normalize the two adata objects separately
      pp_raw2norm(adata_t1)
      pp_raw2norm(adata_t2)
      
      # Get highly variable genes and significant PCs for adata_ref
      get_variable_genes(adata_ref, batch_key=batch_obs, filter_method=vscore_filter_method, min_vscore_pctl=vscore_min_pctl)
      nPCs_test_use = np.min([300, np.sum(adata_ref.var.highly_variable)-1]) # in case nHVgenes is < nPCs
      get_significant_pcs(adata_ref, n_iter=1, nPCs_test = nPCs_test_use, show_plots=False, verbose=False)
      this_round_nHVgenes = np.sum(np.sum(adata_ref.var['highly_variable']))      
      this_round_nSigPCs = adata_ref.uns['n_sig_PCs']
      if verbose: 
        print('nHVgenes:', this_round_nHVgenes)
        print('nSigPCs', this_round_nSigPCs)
      stitch_nHVgenes.append(this_round_nHVgenes)
      stitch_nSigPCs.append(this_round_nSigPCs)

      # Get a pca embedding for adata_ref
      sc.pp.pca(adata_ref, n_comps=adata_ref.uns['n_sig_PCs'], zero_center=True)
      sc.pp.neighbors(adata_ref, n_neighbors=n_neighbors, n_pcs=adata_ref.uns['n_sig_PCs'], metric=distance_metric, use_rep='X_pca')

      # Embed adata_prj into the pca subspace defined by adata_ref
      sc.tl.ingest(adata_prj, adata_ref, embedding_method='pca')

      # Concatenate the pca projections for both timepoints in chronological order
      adata_t1t2 = adata_t1.concatenate(adata_t2, batch_categories=['t1', 't2'])

      # Generate a t1-t2 neighbor graph (a sparse COO matrix) in the joint pca space
      if use_harmony: # include Harmony batch correction
        with logging_disabled():
          sc.external.pp.harmony_integrate(adata_t1t2, batch_obs, basis='X_pca', adjusted_basis='X_pca_harmony', max_iter_harmony=max_iter_harmony, verbose=False)
          sc.pp.neighbors(adata_t1t2, n_neighbors=n_neighbors, metric=distance_metric, use_rep='X_pca_harmony')
          del adata_t1t2.uns['neighbors']['params']['use_rep']
      else: # version without Harmony
        sc.pp.neighbors(adata_t1t2, n_neighbors=n_neighbors, metric=distance_metric)
      X_d_coo = adata_t1t2.obsp['distances'].tocoo()

      # Filter adata_ref self-edges in the non-anchor timepoints
      if n != anchor_round: 

        # Flag self-edges within adata_ref
        row_indices, col_indices = X_d_coo.nonzero()
        if method=='reverse':
          row_flag = row_indices<len(adata_ref)
          col_flag = col_indices<len(adata_ref)
        elif method=='forward':
          row_flag = row_indices>=len(adata_prj)
          col_flag = col_indices>=len(adata_prj)
        adata_ref_self_edge = row_flag & col_flag

        # Apply self-edge filter
        X_d_coo.data = X_d_coo.data[~adata_ref_self_edge]
        X_d_coo.col = X_d_coo.col[~adata_ref_self_edge]
        X_d_coo.row = X_d_coo.row[~adata_ref_self_edge]

      # Concatenate row, column, data for this stitch round
      X_d_stitch_rows = np.concatenate((X_d_stitch_rows, X_d_coo.row + base_counter))
      X_d_stitch_cols = np.concatenate((X_d_stitch_cols, X_d_coo.col + base_counter))
      X_d_stitch_data = np.concatenate((X_d_stitch_data, X_d_coo.data))

      # Increment base_counter by the # of cells in adata_t1
      base_counter = base_counter + len(adata_t1)

  # Assemble the full STITCH graph as a COO matrix
  X_d_stitch_combined = scipy.sparse.coo_matrix((X_d_stitch_data, (X_d_stitch_rows, X_d_stitch_cols)), shape=(len(adata), len(adata)))
  adata.obsp['distances'] = X_d_stitch_combined.tocsr()

  # Compute connectivities from neighbor distances (umap-style)
  adata.obsp['connectivities'] = get_connectivities_from_dist_csr(adata.obsp['distances'], n_neighbors)

  # Store run settings 
  adata.uns['stitch_settings'] = {'timepoint_obs': timepoint_obs,
                                  'batch_obs': batch_obs,
                                  'n_neighbors': n_neighbors,
                                  'distance_metric': distance_metric,
                                  'vscore_min_pctl': vscore_min_pctl,
                                  'vscore_filter_method': vscore_filter_method,
                                  'method': method,
                                  'use_harmony': use_harmony,
                                  'max_iter_harmony': max_iter_harmony}
  
  # Store stitch/neighbors params
  adata.uns['neighbors'] = adata_t1t2.uns['neighbors']
  adata.uns['stitch_params'] = {'stitch_timepoints': timepoint_list,
                                'stitch_n_timepoints': n_timepoints,
                                'stitch_n_rounds': n_stitch_rounds,
                                'stitch_nHVgenes': stitch_nHVgenes,
                                'stitch_nSigPCs': stitch_nSigPCs}
 
  return adata