import warnings
import pandas as pd
import scanpy as sc
import numpy as np
import scipy
import logging
import gc

from contextlib import contextmanager
from umap.umap_ import fuzzy_simplicial_set
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.cm as cm

from .dimensionality import *
from .workflows import *


### UTILS ###

@contextmanager
def disable_logging(highest_level=logging.CRITICAL):
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
        # cases when there were fewer than n_neighbors found
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



### DIAGNOSTICS ###

def plot_stitch_hvgene_overlaps(adata, jaccard=True, cmap='jet', n_clust=3, zmax=None):

    labels = adata.uns['stitch']['timepoints'].astype('int').astype('str')
    hvgene_list = adata.uns['stitch']['HVgenes']

    # Calculate overlap scores between hv gene sets
    n = len(hvgene_list)
    overlap = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            intersection = list(set(hvgene_list[str(i)]) & set(hvgene_list[str(j)])) #map(eq, hvgene_list[str(i)], hvgene_list[str(j)])
            intersection_size = len(intersection)
            union_size = len(hvgene_list[str(i)]) + len(hvgene_list[str(j)])
            min_group_size = np.min([len(hvgene_list[str(i)]), len(hvgene_list[str(j)])])
            if jaccard == True:
                denom = union_size
            else:
                denom = min_group_size                        
            overlap[i, j] = intersection_size / denom if denom != 0 else 0
            overlap[j, i] = overlap[i, j]

    # Perform clustering
    distances = 1 - overlap  # convert similarities to distances
    np.fill_diagonal(distances, 0) # Set diagonal elements to 0 to avoid numerical issues
    condensed_dist = squareform(distances)
    Z = sch.linkage(condensed_dist, method='ward', metric='euclidean')
    clusters = sch.fcluster(Z, n_clust, criterion='maxclust')
    clusters = clusters.reshape(len(clusters),1)

    # Generate a list of categorical colors for the cluster key
    colors = [cm.tab10(i) for i in range(len(np.unique(clusters)))]
    colors = [f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})' for r, g, b, _ in colors]
    
    # Plot overlap matrix using Plotly heatmap
    heatmap1 = go.Heatmap(z=overlap[::-1, :], colorscale=cmap,
                          x=labels, y=labels[::-1],  # Reverse the order of labels in y
                          colorbar=dict(title=''),  # Add colorbar
                          hoverinfo='skip',  # Hide hoverinfo for cleaner display
                          showscale=True,  # Hide color scale for cleaner display
                          xgap=1, ygap=1,  # Add gaps between cells for gridlines
                          hovertemplate='Jaccard Index: %{z}<extra></extra>',
                          zauto=False, zmax=zmax)  # Custom hovertemplate

    # Create the second heatmap trace without color bar
    heatmap2 = go.Heatmap(z=clusters.transpose(), colorscale=colors, showscale=False)  # Only one column

    # Create subplots with two columns
    fig = make_subplots(rows=2, cols=1, row_heights=[0.02, 0.98], vertical_spacing=0.02)
    fig.add_trace(heatmap1, row=2, col=1)
    fig.add_trace(heatmap2, row=1, col=1)
    
    fig.update_layout(title='HV Gene Overlap Matrix with Clusters',
                      font=dict(family='Helvetica, sans-serif', color='black'),
                      width=600, height=600,  # set width and height to make the figure square
                      plot_bgcolor='black',  # set plot background color
                      xaxis2_showgrid=False, yaxis2_showgrid=False,  # hide gridlines
                      xaxis_gridcolor='black', yaxis_gridcolor='black',
                      margin=dict(l=100, r=100, t=100, b=100),
                      coloraxis_colorbar_thickness=5,  # Set the thickness of the color bar
                      coloraxis_colorbar_len=0.5,  # Set the length of the color bar
                      xaxis1=dict(showticklabels=False, showgrid=False),  # Hide x-axis labels and grid for the second heatmap
                      yaxis1=dict(showticklabels=False, showgrid=False))  # Hide x-axis labels and grid for the second heatmap
     
    fig.update_traces(colorbar=dict(thickness=10, len=0.5, x=1.05), selector=dict(type='heatmap'))


    fig.show()


def plot_stitch_pcgene_overlaps(adata, jaccard=True, cmap='jet', n_clust=3, zmax=None):

    labels = adata.uns['stitch']['timepoints'].astype('int').astype('str')
    pcgenes_list = adata.uns['stitch']['PCgenes']

    # Calculate overlap scores between pvgene sets
    n = len(pcgenes_list)
    overlap = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            intersection = list(set(pcgenes_list[str(i)]) & set(pcgenes_list[str(j)])) #map(eq, pcgenes_list[str(i)], pcgenes_list[str(j)])
            intersection_size = len(intersection)
            union_size = len(pcgenes_list[str(i)]) + len(pcgenes_list[str(j)])
            min_group_size = np.min([len(pcgenes_list[str(i)]), len(pcgenes_list[str(j)])])
            if jaccard == True:
                denom = union_size
            else:
                denom = min_group_size                        
            overlap[i, j] = intersection_size / denom if denom != 0 else 0
            overlap[j, i] = overlap[i, j]

    # Perform clustering
    distances = 1 - overlap  # Convert similarities to distances
    np.fill_diagonal(distances, 0) # Set diagonal elements to 0 to avoid numerical issues
    condensed_dist = squareform(distances)
    Z = sch.linkage(condensed_dist, method='ward', metric='euclidean')
    clusters = sch.fcluster(Z, n_clust, criterion='maxclust')
    clusters = clusters.reshape(len(clusters),1)

    # Generate a list of categorical colors for the cluster key
    colors = [cm.tab10(i) for i in range(len(np.unique(clusters)))]
    colors = [f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})' for r, g, b, _ in colors]
    
    # Plot overlap matrix using Plotly heatmap
    heatmap1 = go.Heatmap(z=overlap[::-1, :], colorscale=cmap,
                          x=labels, y=labels[::-1],  # Reverse the order of labels in y
                          colorbar=dict(title=''),  # Add colorbar
                          hoverinfo='skip',  # Hide hoverinfo for cleaner display
                          showscale=True,  # Hide color scale for cleaner display
                          xgap=1, ygap=1,  # Add gaps between cells for gridlines
                          hovertemplate='Jaccard Index: %{z}<extra></extra>',
                          zauto=False, zmax=zmax)  # Custom hovertemplate

    # Create the second heatmap trace without color bar
    heatmap2 = go.Heatmap(z=clusters.transpose(), colorscale=colors, showscale=False)  # Only one column

    # Create subplots with two columns
    fig = make_subplots(rows=2, cols=1, row_heights=[0.02, 0.98], vertical_spacing=0.02)
    fig.add_trace(heatmap1, row=2, col=1)
    fig.add_trace(heatmap2, row=1, col=1)
    
    fig.update_layout(title='PC Gene Overlap Matrix with Clusters',
                      font=dict(family='Helvetica, sans-serif', color='black'),
                      width=600, height=600,  # set width and height to make the figure square
                      plot_bgcolor='black',  # set plot background color
                      xaxis2_showgrid=False, yaxis2_showgrid=False,  # hide gridlines
                      xaxis_gridcolor='black', yaxis_gridcolor='black',
                      margin=dict(l=100, r=100, t=100, b=100),
                      coloraxis_colorbar_thickness=5,  # Set the thickness of the color bar
                      coloraxis_colorbar_len=0.5,  # Set the length of the color bar
                      xaxis1=dict(showticklabels=False, showgrid=False),  # Hide x-axis labels and grid for the second heatmap
                      yaxis1=dict(showticklabels=False, showgrid=False))  # Hide x-axis labels and grid for the second heatmap
     
    fig.update_traces(colorbar=dict(thickness=10, len=0.5, x=1.05), selector=dict(type='heatmap'))


    fig.show()


def plot_stitch_dims(adata):

    # Plot #s of HV genes and PC dimensions

    df = pd.DataFrame({'Timepoint': adata.uns['stitch']['timepoints'],
                       'nHVgenes':  adata.uns['stitch']['nHVgenes'],
                       'nSigPCs':   adata.uns['stitch']['nSigPCs']})

    fig, ax1 = plt.subplots()

    ax1.plot(df['Timepoint'], df['nHVgenes'], color='r', linewidth=2)
    ax1.set_ylabel('# Highly Variable Genes', color='r')
    ax1.set_xlabel('Timepoint Group (hpf)', color='k')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(df['Timepoint'], df['nSigPCs'], color='b', linewidth=2)
    ax2.set_ylabel('# Significant PC Dimensions', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_yscale('linear')

    plt.show()


def stitch_get_dims_df(adata):
    
    stitch_dims_df = pd.DataFrame({'nHVgenes': adata.uns['stitch']['nHVgenes'], 'nSigPCs': adata.uns['stitch']['nSigPCs']},
                        index=adata.uns['stitch']['timepoints'])
    
    return stitch_dims_df



### CORE STITCH METHODS ###

def stitch_get_dims(adata, timepoint_obs, batch_obs=None, vscore_filter_method='majority', vscore_min_pctl=90, vscore_top_n_genes=3000, sig_pcs_n_trials=1, sig_pcs_top_n_genes=20, downsample_cells=False, verbose=True):
  
  # Identify highly variable genes and the # of significant PCA dimensions across basis timepoints in adata
  # This function must be run prior to constructing the stitch neighbor graph
  # Returns an updated adata with embedding results stored in adata.uns['stitch']
   
  # Determine the # of timepoints in adata
  timepoint_list = np.unique(adata.obs[timepoint_obs])
  n_timepoints = len(timepoint_list)

  # Sort the cells in adata by timepoint
  time_sort_index = adata.obs[timepoint_obs].sort_values(inplace=False).index
  adata = adata[time_sort_index,:] #.copy()

  # Generate a list of individual timepoint adatas
  adata_list = []
  for tp in timepoint_list:
    adata_tmp = adata[adata.obs[timepoint_obs]==tp].copy()
    if downsample_cells:
        min_cells_per_timepoint = np.min(adata.obs[timepoint_obs].value_counts())
        adata_tmp = sc.pp.subsample(adata_tmp, n_obs=min_cells_per_timepoint, copy=True)
    adata_list.append(adata_tmp)

  # Initialize results containers
  nHVgenes = []
  nSigPCs = []
  HVgenes = {}
  PCgenes = {}
  
  # Get dimensionality info for each timepoint
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for n in range(n_timepoints):
      
      if verbose: print('Computing gene vscores and PC embeddings for:', timepoint_list[n])

      # Specify the adata for this timepoint
      adata_tmp = adata_list[n] # .copy()
      
      # Normalize and scale data
      # We will need 2 data layers: (1) tpm_no_log for finding variable genes, and (2) zscored/scaled for pca
      pp_raw2norm(adata_tmp, include_raw_layers=False)

      # Get the top highly variable genes and up to the first 300 PCs
      get_variable_genes(adata_tmp, batch_key=batch_obs, filter_method=vscore_filter_method, top_n_genes=vscore_top_n_genes, min_vscore_pctl=vscore_min_pctl)
      nPCs_test_use = np.min([300, np.sum(adata_tmp.var.highly_variable)-1]) # in case nHVgenes is < nPCs
      get_significant_pcs(adata_tmp, n_iter=sig_pcs_n_trials, nPCs_test=nPCs_test_use, show_plots=False, verbose=False)
      sc.pp.pca(adata_tmp, n_comps=adata_tmp.uns['n_sig_PCs'], zero_center=True) # perform pca using the # of determined significant PCs
      
      # Get a list of the top genes from PCA loading matrices
      this_round_PC_loadings = adata_tmp.varm['PCs']
      this_round_PC_genes = []
      for pc in range(adata_tmp.uns['n_sig_PCs']): # Only use significant PCs
          this_round_PC_gene_ind = list(np.argsort(np.absolute((this_round_PC_loadings[:,pc])))[::-1][:sig_pcs_top_n_genes]) 
          this_round_PC_genes.extend(adata.var_names[this_round_PC_gene_ind])

      # Organize results
      nHVgenes.append(np.sum(np.sum(adata_tmp.var['highly_variable'])))
      nSigPCs.append(adata_tmp.uns['n_sig_PCs'])
      HVgenes[str(n)]=list(adata_tmp.var[adata_tmp.var['highly_variable']==True].index)
      PCgenes[str(n)]=list(set(this_round_PC_genes))

      # Clean up objects from this round
      adata_list[n] = []
      gc.collect()

  # Save results to dictionary
  adata.uns['stitch'] = {'timepoint_obs': timepoint_obs, 'batch_obs': batch_obs, 
                         'vscore_filter_method': vscore_filter_method, 'vscore_min_pctl': vscore_min_pctl, 
                         'timepoints': timepoint_list, 'nTimepoints': n_timepoints, 'nHVgenes': nHVgenes, 
                         'HVgenes': HVgenes, 'nSigPCs': nSigPCs, 'PCgenes': PCgenes, 'downsample_cells': downsample_cells} 
                         
  return adata


def stitch_get_graph(adata, timepoint_obs, batch_obs=None, n_neighbors=15, distance_metric='correlation', method='reverse', self_edge_filter=True, use_harmony=True, max_iter_harmony=20, downsample_cells=False, verbose=True):

  # Builds the stitch neigbhor graph
  # This function returns an updated adata object with new info saved to adata.uns['stitch'] and
  # also generates graph objects/info stored in adata.obsp, adata.obsm, and adata.uns['neighbors']

  # Determine the # of timepoints in adata
  timepoint_list = adata.uns['stitch']['timepoints']
  n_timepoints = adata.uns['stitch']['nTimepoints']
  n_stitch_rounds = n_timepoints - 1

  # Get the previously built embedding info from each timepoint
  HVgenes = adata.uns['stitch']['HVgenes']
  nSigPCs = adata.uns['stitch']['nSigPCs']

  # Generate a list of individual timepoint adatas
  adata_list = []
  for tp in timepoint_list:
    adata_tmp = adata[adata.obs[timepoint_obs]==tp].copy()
    if downsample_cells:
        min_cells_per_timepoint = np.min(adata.obs[timepoint_obs].value_counts())
        adata_tmp = sc.pp.subsample(adata_tmp, n_obs=min_cells_per_timepoint, copy=True)
    adata_list.append(adata_tmp)
    del adata_tmp

  # Set directionality of time projections
  if method=='forward':
    arrow_str = '->'
    anchor_round = n_stitch_rounds - 1 # anchor = last round
  elif method=='reverse':
    arrow_str = '<-'
    anchor_round = 0 # anchor = first round

  # Initialize graph data containers
  base_counter = 0
  X_d_stitch_rows = []
  X_d_stitch_cols = []
  X_d_stitch_data = []

  # Get neighbor graph for each stitch_round (each timepoint pair)
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for n in range(n_stitch_rounds):
      
      if verbose: print('Stitching Timepoints:', timepoint_list[n], arrow_str, timepoint_list[n+1])
      
      # Load previously processed adata_t1 and adata_t2 for this round
      adata_t1 = adata_list[n].copy()
      adata_t2 = adata_list[n+1].copy()
      
      # Specify adatas for reference and projection
      if method=='forward':
        adata_ref = adata_t2
        adata_ref.var['highly_variable'] = adata_ref.var_names.isin(HVgenes[n+1])
        #adata_ref.var['highly_variable'] = HVgenes[n+1]
        adata_ref.uns['n_sig_PCs'] = nSigPCs[n+1]
        adata_prj = adata_t1
      elif method=='reverse':
        adata_ref = adata_t1
        adata_ref.var['highly_variable'] = adata_ref.var_names.isin(HVgenes[n])
        #adata_ref.var['highly_variable'] = HVgenes[n]
        adata_ref.uns['n_sig_PCs'] = nSigPCs[n]
        adata_prj = adata_t2

      # Normalize  
      pp_raw2norm(adata_ref, include_raw_layers=False, include_tpm_layers=False)
      pp_raw2norm(adata_prj, include_raw_layers=False, include_tpm_layers=False)

      # Perform PCA on reference timepoint
      sc.pp.pca(adata_ref, n_comps=adata_ref.uns['n_sig_PCs'], zero_center=True) # perform pca using the # of predetermined significant PCs
      if batch_obs is not None and use_harmony:
          with disable_logging():
              sc.external.pp.harmony_integrate(adata_ref, batch_obs, basis='X_pca', adjusted_basis='X_pca_harmony', max_iter_harmony=20, verbose=False)
      
      # Embed adata_prj into the pca subspace defined by adata_ref
      sc.pp.neighbors(adata_ref, n_neighbors=n_neighbors, n_pcs=adata_ref.uns['n_sig_PCs'], metric=distance_metric, use_rep='X_pca')
      sc.tl.ingest(adata_prj, adata_ref, embedding_method='pca')

      # Concatenate the pca projections for both timepoints in chronological order
      adata_t1t2 = adata_t1.concatenate(adata_t2, batch_categories=['t1', 't2'])

      # Generate a t1-t2 neighbor graph (a sparse COO matrix) in the joint pca space
      if batch_obs is not None and use_harmony:  # include Harmony batch correction
        with disable_logging():
          sc.external.pp.harmony_integrate(adata_t1t2, batch_obs, basis='X_pca', adjusted_basis='X_pca_harmony', max_iter_harmony=max_iter_harmony, verbose=False)
          sc.pp.neighbors(adata_t1t2, n_neighbors=n_neighbors, n_pcs=adata_ref.uns['n_sig_PCs'], metric=distance_metric, use_rep='X_pca_harmony')
      else: # version without Harmony
        sc.pp.neighbors(adata_t1t2, n_neighbors=n_neighbors, n_pcs=adata_ref.uns['n_sig_PCs'], metric=distance_metric, use_rep='X_pca')
      X_d_coo = adata_t1t2.obsp['distances'].tocoo()

      # Filter adata_ref self-edges in the non-anchor timepoints
      if self_edge_filter and n != anchor_round: 

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

      # Concatenate row, column, data from this round
      X_d_stitch_rows = np.concatenate((X_d_stitch_rows, X_d_coo.row + base_counter))
      X_d_stitch_cols = np.concatenate((X_d_stitch_cols, X_d_coo.col + base_counter))
      X_d_stitch_data = np.concatenate((X_d_stitch_data, X_d_coo.data))

      # Increment base_counter by the # of cells in adata_t1
      base_counter += len(adata_t1)

      # Cleanup objects from this round
      del adata_t1, adata_t2, adata_t1t2
      gc.collect()

  # Specify neighbor graph settings for downstream steps
  adata.obsm['X_stitch_dummy'] = np.zeros((adata.shape[0], 50)) # empty X for initializing umap later on
  adata.uns['neighbors'] = {'connectivities_key': 'connectivities', 'distances_key': 'distances',
                            'params': {'method': 'umap', 'metric': distance_metric, 'n_neighbors': n_neighbors,
                                       'n_pcs': 50, 'random_state': 0, 'use_rep': 'X_stitch_dummy'}}

  # Assemble the full STITCH graph as a COO matrix and compute 'umap-style' connectivities
  adata.obsp['distances'] = scipy.sparse.coo_matrix((X_d_stitch_data, (X_d_stitch_rows, X_d_stitch_cols)), shape=(len(adata), len(adata))).tocsr()
  adata.obsp['connectivities'] = get_connectivities_from_dist_csr(adata.obsp['distances'], n_neighbors)

  # Update STITCH results
  adata.uns['stitch'].update({'n_neighbors': n_neighbors,'distance_metric': distance_metric, 'stitch_method': method,
                              'use_harmony': use_harmony, 'max_iter_harmony': max_iter_harmony, 'self_edge_filter': self_edge_filter})
  
  return adata



### STITCH PIPELINE ###

def stitch(adata, 
           timepoint_obs,                     # column in adata.obs containing numeric timepoint assignments (required)
           batch_obs=None,                    # column in adata.obs containing batch assignments (optional)
           vscore_filter_method='majority',   # vscore batch filtering method: any | multiple | majority | all | top_n_genes
           vscore_min_pctl=90,                # vscore minimum percentile to call a gene as highly variable
           vscore_top_n_genes=3000,           # vscore number of highly variable genes to keep (only invoked if method = top_n_genes)
           sig_pcs_n_trials=1,                # number of random trials used to determine significant PCs
           sig_pcs_top_n_genes=20,            # number of top loaded genes to keep from each significant PC (used for plotting only)
           downsample_cells=False,            # whether to downsample all timepoints to the same number of cells
           n_neighbors=15,                    # number of outgoing nearest neighbors to retain for the stitch graph
           distance_metric='correlation',     # metric used to calculate nearest neighbor distances
           method='reverse',                  # directionality of the stitch method: forward | reverse
           self_edge_filter=True,             # whether to filter self edges from adata_ref
           use_harmony=True,                  # whether to use harmony batch correction within each timepoint pair (only invoked if batch_obs exists)
           max_iter_harmony=20,               # maximum number of harmony iterations (if used)
           verbose=True):                     # print additional details to screen 

    # Estimate dimensionality of each timepoint
    stitch_get_dims(adata=adata, timepoint_obs=timepoint_obs, batch_obs=batch_obs, vscore_filter_method=vscore_filter_method, vscore_min_pctl=vscore_min_pctl, vscore_top_n_genes=vscore_top_n_genes, sig_pcs_n_trials=sig_pcs_n_trials, sig_pcs_top_n_genes=sig_pcs_top_n_genes, downsample_cells=downsample_cells, verbose=verbose)
    
    # Dimensionality plots
    plot_stitch_dims(adata)
    plot_stitch_hvgene_overlaps(adata, cmap='jet')
    plot_stitch_pcgene_overlaps(adata, cmap='jet')
    
    # Build and embed the stitch graph
    stitch_get_graph(adata=adata, timepoint_obs=timepoint_obs, batch_obs=batch_obs, n_neighbors=n_neighbors, distance_metric=distance_metric, method=method, self_edge_filter=self_edge_filter, use_harmony=use_harmony, max_iter_harmony=max_iter_harmony, downsample_cells=downsample_cells, verbose=verbose)
    sc.tl.umap(adata)

    return adata




###################################
### LEGACY FXNS - DON'T TOUCH!! ###
###################################
    
def stitch_orig(adata, timepoint_obs, batch_obs=None, n_neighbors=15, distance_metric='correlation', vscore_min_pctl=85, vscore_filter_method=None, method='forward', use_harmony=True, max_iter_harmony=20, verbose=True):

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

  # Initialize results lists
  base_counter = 0
  X_d_stitch_rows = []
  X_d_stitch_cols = []
  X_d_stitch_data = []
  nHVgenes = []
  stitch_HVgene_flags = []
  nSigPCs = []
  stitch_nBatches = []
  coo_shape = (len(adata), len(adata))
  X_d_stitch_combined = scipy.sparse.coo_matrix(coo_shape)

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
      pp_raw2norm(adata_t1, include_raw_layers=False)
      pp_raw2norm(adata_t2, include_raw_layers=False)

      # Get highly variable genes and significant PCs for adata_ref
      get_variable_genes(adata_ref, batch_key=batch_obs, filter_method=vscore_filter_method, min_vscore_pctl=vscore_min_pctl)
      nPCs_test_use = np.min([300, np.sum(adata_ref.var.highly_variable)-1]) # in case nHVgenes is < nPCs
      get_significant_pcs(adata_ref, n_iter=1, nPCs_test = nPCs_test_use, show_plots=False, verbose=False)
      this_round_nHVgenes = np.sum(np.sum(adata_ref.var['highly_variable']))      
      this_round_nSigPCs = adata_ref.uns['n_sig_PCs']
      if verbose: 
        print('nHVgenes:', this_round_nHVgenes)
        print('nSigPCs', this_round_nSigPCs)
      nHVgenes.append(this_round_nHVgenes)
      nSigPCs.append(this_round_nSigPCs)
      stitch_HVgene_flags.append(adata_ref.var['highly_variable'])

      # Get a pca embedding for adata_ref
      sc.pp.pca(adata_ref, n_comps=adata_ref.uns['n_sig_PCs'], zero_center=True)
      sc.pp.neighbors(adata_ref, n_neighbors=n_neighbors, n_pcs=adata_ref.uns['n_sig_PCs'], metric=distance_metric, use_rep='X_pca')

      # Embed adata_prj into the pca subspace defined by adata_ref
      sc.tl.ingest(adata_prj, adata_ref, embedding_method='pca')

      # Concatenate the pca projections for both timepoints in chronological order
      adata_t1t2 = adata_t1.concatenate(adata_t2, batch_categories=['t1', 't2'])
      stitch_nBatches.append(len(np.unique(adata_ref.obs[batch_obs])))

      # Generate a t1-t2 neighbor graph (a sparse COO matrix) in the joint pca space
      if use_harmony: # include Harmony batch correction
        with disable_logging():
          sc.external.pp.harmony_integrate(adata_t1t2, batch_obs, basis='X_pca', adjusted_basis='X_pca_harmony', max_iter_harmony=max_iter_harmony, verbose=False)
          sc.pp.neighbors(adata_t1t2, n_neighbors=n_neighbors, metric=distance_metric, use_rep='X_pca_harmony')
          del adata_t1t2.uns['neighbors']['params']['use_rep']
      else: # version without Harmony
        sc.pp.neighbors(adata_t1t2, n_neighbors=n_neighbors, metric=distance_metric)
      X_d_coo = adata_t1t2.obsp['distances'].tocoo()
      neighbors_settings = adata_t1t2.uns['neighbors']

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
      #X_d_coo = scipy.sparse.coo_matrix((X_d_coo.data, (X_d_coo.row + base_counter, X_d_coo.col + base_counter)), shape=coo_shape)
      #X_d_stitch_combined = scipy.sparse.vstack([X_d_stitch_combined, X_d_coo])

      # Increment base_counter by the # of cells in adata_t1
      base_counter += len(adata_t1)

  # Assemble the full STITCH graph as a COO matrix
  adata.obsp['distances'] = scipy.sparse.coo_matrix((X_d_stitch_data, (X_d_stitch_rows, X_d_stitch_cols)), shape=(len(adata), len(adata))).tocsr()
  #adata.obsp['distances'] = X_d_stitch_combined.tocsr()

  # Compute connectivities from neighbor distances (umap-style)
  adata.obsp['connectivities'] = get_connectivities_from_dist_csr(adata.obsp['distances'], n_neighbors)

  # Save neighbor graph settings for downstream steps
  adata.obsm['X_stitch_dummy'] = np.zeros((adata.shape[0], 50)) # empty X for initializing umap later on
  adata.uns['neighbors'] = {'connectivities_key': 
                            'connectivities', 'distances_key': 'distances',
                            'params': {'method': 'umap',
                            'metric': distance_metric,
                            'n_neighbors': n_neighbors,
                            'n_pcs': 50,
                            'random_state': 0,
                            'use_rep': 'X_stitch_dummy'}}
  
  adata.uns['stitch_settings'] = {'timepoint_obs': timepoint_obs, 'batch_obs': batch_obs, 'n_neighbors': n_neighbors,'distance_metric': distance_metric,
                                  'vscore_min_pctl': vscore_min_pctl, 'vscore_filter_method': vscore_filter_method, 'method': method,
                                  'use_harmony': use_harmony, 'max_iter_harmony': max_iter_harmony}
  adata.uns['stitch_results'] = {'stitch_timepoints': timepoint_list, 'stitch_n_timepoints': n_timepoints, 'stitch_n_rounds': n_stitch_rounds,
                                'nHVgenes': nHVgenes, 'stitch_HVgene_flags': stitch_HVgene_flags, 'nSigPCs': nSigPCs, 'stitch_nBatches': stitch_nBatches}
                              
 
  return adata

