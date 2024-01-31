import warnings
import pandas as pd
import scanpy as sc
import numpy as np
import scipy
import logging
from contextlib import contextmanager
from .dimensionality import *
from .workflows import *

@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)


def stitch(adata, timepoint_obs, batch_obs=None, n_neighbors=15, distance_metric='correlation', vscore_min_pctl=95, vscore_filter_method=None, method='forward'):

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
  dist_lists = []
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for n in range(n_stitch_rounds):

      # Specify individual adatas for the two timepoints in this round
      adata_t1 = adata_list[n].copy()
      adata_t2 = adata_list[n+1].copy()

      # Normalize the two adata objects separately
      pp_raw2norm(adata_t1)
      pp_raw2norm(adata_t2)

      # Set directionality of time projections
      if method=='forward':
        adata_ref = adata_t2
        adata_project = adata_t1
        arrow_str = '->'
      elif method=='reverse':
        adata_ref = adata_t1
        adata_project = adata_t2
        arrow_str = '<-'
      
      # Perform projections
      print('Stitching Timepoints:', timepoint_list[n], arrow_str, timepoint_list[n+1])

      # Define variable genes and nPCs for adata_ref
      get_variable_genes(adata_ref, batch_key=batch_obs, filter_method=vscore_filter_method, min_vscore_pctl=vscore_min_pctl)
      nPCs_test_use = np.min([300, np.sum(adata_ref.var.highly_variable)-1])
      get_significant_pcs(adata_ref, n_iter=1, nPCs_test = nPCs_test_use, show_plots=False, verbose=False)
      print('nHVgenes:', np.sum(np.sum(adata_ref.var['highly_variable'])))
      print('nSigPCs', adata_ref.uns['n_sig_PCs'])

      # Get a pca embedding for adata_ref
      sc.pp.pca(adata_ref, n_comps=adata_ref.uns['n_sig_PCs'], zero_center=True)
      sc.pp.neighbors(adata_ref, n_neighbors=n_neighbors, n_pcs=adata_ref.uns['n_sig_PCs'], metric=distance_metric, use_rep='X_pca')

      # Embed adata_project into the pca subspace defined by adata_ref
      sc.tl.ingest(adata_project, adata_ref, embedding_method='pca')
        
      # Concatenate the pca projections for both timepoints in chronological order (not projection order)
      adata_t1t2 = adata_t1.concatenate(adata_t2, batch_categories=['t1', 't2'])

      # Generate a t1-t2 neighbor graph in the joint pca space
      if True: # include Harmony batch correction
        with all_logging_disabled():
          sc.external.pp.harmony_integrate(adata_t1t2, batch_obs, basis='X_pca', adjusted_basis='X_pca_harmony', max_iter_harmony=20, verbose=False)
        sc.pp.neighbors(adata_t1t2, n_neighbors=n_neighbors, metric=distance_metric, use_rep='X_pca_harmony')
        del adata_t1t2.uns['neighbors']['params']['use_rep']
      else: # version without Harmony
        sc.pp.neighbors(adata_t1t2, n_neighbors=n_neighbors, metric=distance_metric)
        stitch_neighbors_settings = adata_t1t2.uns['neighbors']

      # Convert graph connectivities and distances (csr matrices) to edge list format
      X_c = adata_t1t2.obsp['connectivities']
      edge_df = pd.DataFrame([[n1, n2, X_c[n1,n2]] for n1, n2 in zip(*X_c.nonzero())], columns=['n1','n2','connectivity'])
      X_d = adata_t1t2.obsp['distances']
      dist_df = pd.DataFrame([[n1, n2, X_d[n1,n2]] for n1, n2 in zip(*X_d.nonzero())], columns=['n1','n2','distances'])

      # Adjust the node id counters
      edge_df['n1'] = edge_df['n1'] + base_counter
      edge_df['n2'] = edge_df['n2'] + base_counter
      dist_df['n1'] = dist_df['n1'] + base_counter
      dist_df['n2'] = dist_df['n2'] + base_counter
      
      # Append the most recent edge lists to the previous lists
      edge_lists.append(edge_df)
      dist_lists.append(dist_df)

      # Increment base_counter by the # of cells in adata_t1
      base_counter = base_counter + len(adata_t1)

  # Merge all lists - what to do with the one straggler timepoint???
  combined_edge_df = pd.concat(edge_lists)
  combined_dist_df = pd.concat(dist_lists)

  # Save STITCH graph/settings/params to adata
  adata.obsp['stitch_connectivities'] = scipy.sparse.coo_matrix((combined_edge_df['connectivity'], (combined_edge_df['n1'], combined_edge_df['n2']))).tocsr().copy()
  adata.obsp['connectivities'] = adata.obsp['stitch_connectivities'].copy()
  adata.obsp['distances'] = scipy.sparse.coo_matrix((combined_dist_df['distances'], (combined_dist_df['n1'], combined_dist_df['n2']))).tocsr().copy()
  adata.uns['neighbors'] = adata_t1t2.uns['neighbors']
  adata.uns['stitch_params'] = {'timepoint_obs': timepoint_obs,
                                'batch_obs': batch_obs,
                                'n_neighbors': n_neighbors, 
                                'distance_metric': distance_metric,
                                'vscore_min_pctl': vscore_min_pctl,
                                'vscore_filter_method': vscore_filter_method,
                                'method': method}

  return adata




def stitch_leiden(adata, timepoint_obs, batch_obs=None, n_neighbors=15, distance_metric='correlation', vscore_min_pctl=95, vscore_filter_method=None):

  # Determine the # of timepoints in adata
  timepoint_list = np.unique(adata.obs[timepoint_obs])
  n_timepoints = len(timepoint_list)

  # Sort the cells in adata by timepoint
  time_sort_index = adata.obs[timepoint_obs].sort_values(inplace=False).index
  adata = adata[time_sort_index,:].copy()

  # Generate a list of individual timepoint adatas
  adata_list = []
  for tp in timepoint_list:
    adata_list.append(adata[adata.obs[timepoint_obs]==tp])

  # Get embedding spaces and clustering assignments for each timepoint
  leiden_lists = []
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for n in range(n_timepoints):

      print(n, timepoint_list[n])

      # Specify individual adatas for the two timepoints in this round
      adata_this_t = adata_list[n].copy()

      # Normalize the two adata objects separately
      pp_raw2norm(adata_this_t)

      # Define variable genes and nPCs
      get_variable_genes(adata_this_t, batch_key=batch_obs, filter_method=vscore_filter_method, min_vscore_pctl=vscore_min_pctl)
      nPCs_test_use = np.min([300, np.sum(adata_this_t.var.highly_variable)-1])
      get_significant_pcs(adata_this_t, n_iter=1, nPCs_test = nPCs_test_use, show_plots=False, verbose=False)

      # Get a pca & neighbors embedding
      sc.pp.pca(adata_this_t, n_comps=adata_this_t.uns['n_sig_PCs'], zero_center=True)
      #sc.pp.neighbors(adata_this_t, n_neighbors=n_neighbors, n_pcs=adata_this_t.uns['n_sig_PCs'], metric=distance_metric, use_rep='X_pca')

      # Harmony version
      with all_logging_disabled():
        sc.external.pp.harmony_integrate(adata_this_t, batch_obs, basis='X_pca', adjusted_basis='X_pca_harmony', max_iter_harmony=20, verbose=False)
      sc.pp.neighbors(adata_this_t, n_neighbors=n_neighbors, n_pcs=adata_this_t.uns['n_sig_PCs'], metric=distance_metric, use_rep='X_pca_harmony')

      # Perform leiden clustering
      sc.tl.leiden(adata_this_t, resolution=1)
      leiden_lists.append(str(n)+'_'+adata_this_t.obs['leiden'].astype('str'))

  # Merge lists
  leiden_df = pd.concat(leiden_lists)

  # Save STITCH graph/settings/params to adata
  adata.obs['stitch_leiden'] = list(leiden_df)

  return adata