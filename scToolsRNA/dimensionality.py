
import sys
import warnings
import numpy as np
import scipy
import sklearn
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



# IDENTIFY HIGHLY VARIABLE GENES


def get_min_max_norm(data):

    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))


def runningquantile(x, y, p, nBins):
    """ calculate the quantile of y in bins of x """

    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]

    dx = (x[-1] - x[0]) / nBins
    xOut = np.linspace(x[0]+dx/2, x[-1]-dx/2, nBins)

    yOut = np.zeros(xOut.shape)

    for i in range(len(xOut)):
        ind = np.nonzero((x >= xOut[i]-dx/2) & (x < xOut[i]+dx/2))[0]
        if len(ind) > 0:
            yOut[i] = np.percentile(y[ind], p)
        else:
            if i > 0:
                yOut[i] = yOut[i-1]
            else:
                yOut[i] = np.nan

    return xOut, yOut


def get_vscores(E, min_mean=0, nBins=50, fit_percentile=0.1, error_wt=1):
    '''
    Calculate v-score (above-Poisson noise statistic) for genes in the input counts matrix
    Return v-scores and other stats
    '''

    ncell = E.shape[0]

    mu_gene = E.mean(axis=0).A.squeeze()
    gene_ix = np.nonzero(mu_gene > min_mean)[0]
    mu_gene = mu_gene[gene_ix]

    tmp = E[:, gene_ix]
    tmp.data **= 2
    var_gene = tmp.mean(axis=0).A.squeeze() - mu_gene ** 2
    del tmp
    FF_gene = var_gene / mu_gene

    data_x = np.log(mu_gene)
    data_y = np.log(FF_gene / mu_gene)

    x, y = runningquantile(data_x, data_y, fit_percentile, nBins)
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    def gLog(input): return np.log(input[1] * np.exp(-input[0]) + input[2])
    h, b = np.histogram(np.log(FF_gene[mu_gene > 0]), bins=200)
    b = b[:-1] + np.diff(b) / 2
    max_ix = np.argmax(h)
    c = np.max((np.exp(b[max_ix]), 1))

    def errFun(b2): return np.sum(abs(gLog([x, c, b2]) - y) ** error_wt)
    b0 = 0.1
    b = scipy.optimize.fmin(func=errFun, x0=[b0], disp=False)
    a = c / (1 + b) - 1

    v_scores = FF_gene / ((1 + a) * (1 + b) + b * mu_gene)
    CV_eff = np.sqrt((1 + a) * (1 + b) - 1)
    CV_input = np.sqrt(b)

    return v_scores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b


def get_vscores_adata(adata, norm_counts_per_cell=1e6, min_vscore_pctl=85, min_counts=3, min_cells=3, in_place=True):

    ''' 
    Identifies highly variable genes
    Requires a layer of adata that has been processed by total count normalization (e.g. tpm_nolog)
    '''

    E = adata.layers['tpm_nolog']
    
    # get variability statistics
    stats = {}
    stats['vscores'], stats['CV_eff'], stats['CV_input'], stats['gene_ix'], stats['mu_gene'], stats['FF_gene'], stats['a'], stats['b'] = get_vscores(E) # gene_ix = genes for which vscores could be returned
    stats['min_vscore_pctl'] = min_vscore_pctl

    # index genes based on vscores percentile
    ix2 = stats['vscores'] > 0 # ix2 = genes for which a positive vscore was obtained
    stats['min_vscore'] = np.percentile(stats['vscores'][ix2], min_vscore_pctl)    
    ix3 = (((E[:, stats['gene_ix'][ix2]] >= min_counts).sum(0).A.squeeze()>= min_cells) & (stats['vscores'][ix2] >= stats['min_vscore'])) # ix3 = genes passing final min cells & counts thresholds

    # highly variable genes = genes passing all 3 filtering steps: gene_ix, ix2, and ix3
    stats['hv_genes'] = adata.var_names[stats['gene_ix'][ix2][ix3]]
    
    if in_place:
      
        # save gene-level stats to adata.var
        adata.var['highly_variable'] = False
        adata.var.loc[stats['hv_genes'], 'highly_variable'] = True
        adata.var['vscore'] = np.nan
        adata.var.loc[adata.var_names[stats['gene_ix']], 'vscore'] = stats['vscores']
        adata.var['mu_gene'] = np.nan
        adata.var.loc[adata.var_names[stats['gene_ix']], 'mu_gene'] = stats['mu_gene']
        adata.var['ff_gene'] = np.nan
        adata.var.loc[adata.var_names[stats['gene_ix']], 'ff_gene'] = stats['FF_gene']
    
        # save vscore results to adata.uns
        adata.uns['vscore_stats'] = stats
        adata.uns['vscore_stats']['hv_genes'] = list(adata.uns['vscore_stats']['hv_genes'])

        return None
    
    else:
        
        # just return vscore stats
        return stats


def get_variable_genes(adata, batch_key=None, filter_method='all', top_n_genes=3000, norm_counts_per_cell=1e6, min_vscore_pctl=85, min_counts=3, min_cells=3, in_place=True):
    
    '''
    Filter variable genes based on their representation within individual sample batches
    '''

    # compute initial gene variability stats using the entire dataset
    get_vscores_adata(adata, norm_counts_per_cell=norm_counts_per_cell, min_vscore_pctl=min_vscore_pctl, min_counts=min_counts, min_cells=min_cells)
    
    # batch handling: if no batches are provided we are already done
    if batch_key == None: # 
        if in_place == True: 
            adata.var['vscore'] = get_min_max_norm(adata.var['vscore'])
            return adata
        else:
            return adata.var['highly_variable']
    
    # if multiple batches are present, then we will determine variable genes independently for each batch
    else:
        batch_ids = np.unique(adata.obs[batch_key])
        n_batches = len(batch_ids)

    # set hvgene filter method
    if filter_method == 'any':
        count_thresh = 0 # >0 = keep hvgenes identified in 1 or more batches
    elif filter_method == 'multiple':
        count_thresh = 1 # >1 = only keep hvgenes identified in 2 or more batches
    elif filter_method == 'majority':
        count_thresh = n_batches/2 # only keep hvgenes identified in >50% of batches
    elif filter_method == 'all':
        count_thresh = n_batches - 1 # only keep hvgenes identified in 100% of batches
    elif filter_method == 'top_n_genes': 
        min_vscore_pctl = 0 # return the top hv genes (# specified by 'top_n_genes') ranked by mean scaled vscore
    else:
        print('Invalid filtering method provided!')    

    # identify variable genes for each batch separately
    within_batch_hv_genes = []
    within_batch_vscores = np.full(shape=[adata.shape[1],n_batches], fill_value=np.nan)
    for n,b in enumerate(batch_ids):
        adata_batch = adata[adata.obs[batch_key] == b].copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            vscore_stats = get_vscores_adata(adata_batch, norm_counts_per_cell=norm_counts_per_cell, min_vscore_pctl=min_vscore_pctl, min_counts=min_counts, min_cells=min_cells, in_place=False)
        hv_genes_this_batch = list(vscore_stats['hv_genes'])
        within_batch_hv_genes.append(hv_genes_this_batch)
        within_batch_vscores[:,n] = get_min_max_norm(adata.var['vscore']) # scale vscores from 0 to 1

    # aggregate batch stats
    adata.var['vscore'] = np.nanmean(within_batch_vscores, axis=1) # return the mean of scaled vscores across all batches
    within_batch_hv_genes = [g for gene in within_batch_hv_genes for g in gene]
    within_batch_hv_genes, hv_batch_count = np.unique(within_batch_hv_genes, return_counts=True)
    
    # perform hv_gene filtering
    if filter_method is 'top_n_genes':
        hv_genes = adata.var['vscore'].sort_values(ascending=False)[0:top_n_genes].index
    else:
        hv_genes = within_batch_hv_genes[hv_batch_count > count_thresh]
        
    # update adata
    adata.var['highly_variable'] = False
    adata.var.loc[hv_genes, 'highly_variable'] = True    

    if in_place:
        return None
    else:
        return adata.var['highly_variable']


def plot_ff(adata, gene_ix=None, color=None):

  if gene_ix == None:
    gene_ix = adata.var['highly_variable']
  else:
    gene_ix = adata.var[gene_ix]

  if color == None:
    color = np.array(['blue'])

  mu_gene = adata.var['mu_gene']
  ff_gene = adata.var['ff_gene']
  a = adata.uns['vscore_stats']['a']
  b = adata.uns['vscore_stats']['b']

  x_min = 0.5 * np.min(mu_gene)
  x_max = 2 * np.max(mu_gene)
  xTh = x_min * np.exp(np.log(x_max / x_min) * np.linspace(0, 1, 100))
  yTh = (1 + a) * (1 + b) + b * xTh
  plt.figure(figsize=(6, 6))
  plt.scatter(np.log10(mu_gene), np.log10(ff_gene), c=np.array(['grey']), alpha=0.3, edgecolors=None, s=4)
  plt.scatter(np.log10(mu_gene)[gene_ix], np.log10(ff_gene)[gene_ix], c=color, alpha=0.3, edgecolors=None, s=4)

  plt.plot(np.log10(xTh), np.log10(yTh))
  plt.xlabel('Mean Transcripts Per Cell (log10)')
  plt.ylabel('Gene Fano Factor (log10)')
  plt.show()


def plot_vscores(adata, gene_ix=None, color=None):

  if gene_ix == None:
    gene_ix = adata.var['highly_variable']
  else:
    gene_ix = adata.var[gene_ix]
  
  if color == None:
    color = np.array(['blue'])
  
  mu_gene = adata.var['mu_gene']
  vscores_gene = adata.var['vscore']
  a = adata.uns['vscore_stats']['a']
  b = adata.uns['vscore_stats']['b']

  plt.figure(figsize=(6, 6))
  plt.scatter(np.log10(mu_gene), np.log10(vscores_gene), c=np.array(['grey']), alpha=0.3, edgecolors=None, s=4)
  plt.scatter(np.log10(mu_gene)[gene_ix], np.log10(vscores_gene)[gene_ix], c=color, alpha=0.3, edgecolors=None, s=4)

  plt.xlabel('Mean Transcripts Per Cell (log10)')
  plt.ylabel('Gene Vscores (log10)')
  plt.show()

 
def get_covar_genes(adata, minimum_correlation=0.2, show_hist=True):

    # Subset adata to highly variable genes x cells (counts matrix only)
    adata_tmp = sc.AnnData(adata[:,adata.var.highly_variable].X)

    # Determine if the input matrix is sparse
    sparse=False
    if scipy.sparse.issparse(adata_tmp.X):
      sparse=True

    # Get nn correlation distance for each highly variable gene
    gene_correlation_matrix = 1-sparse_corr(adata_tmp.X)
    np.fill_diagonal(gene_correlation_matrix, np.inf)
    max_neighbor_corr = 1-gene_correlation_matrix.min(axis=1)

    # filter genes whose nearest neighbor correlation is above threshold 
    ix_keep = np.array(max_neighbor_corr > minimum_correlation, dtype=bool).squeeze()
    
    # Prepare a randomized data matrix                    
    adata_tmp_rand = adata_tmp.copy()
    
    if sparse:
      mat = adata_tmp_rand.X.todense()
    else:
      mat = adata_tmp_rand.X
    
    # randomly permute each row of the counts matrix
    for c in range(mat.shape[1]):
        np.random.seed(seed=c)
        mat[:,c] = mat[np.random.permutation(mat.shape[0]),c]
    
    if sparse:        
      adata_tmp_rand.X = scipy.sparse.csr_matrix(mat)
    else:
      adata_tmp_rand.X = mat
    
    # Get nn correlation distances for randomized data
    gene_correlation_matrix = 1-sparse_corr(adata_tmp_rand.X)
    np.fill_diagonal(gene_correlation_matrix, np.inf)
    max_neighbor_corr_rand = 1-gene_correlation_matrix.min(axis=1)
    
    # Plot histogram of correlation distances
    plt.figure(figsize=(6, 6))
    plt.hist(max_neighbor_corr_rand, bins=np.linspace(0, 1, 100), density=False, alpha=0.5, label='random')
    plt.hist(max_neighbor_corr, bins=np.linspace(0, 1, 100), density=False, alpha=0.5, label='data')
    plt.axvline(x = minimum_correlation, color = 'k', linestyle = '--', alpha=0.5, linewidth=1)
    plt.xlabel('Nearest Neighbor Correlation')
    plt.ylabel('Counts')
    plt.legend(loc='upper right')
    plt.show()
    
    return adata_tmp.var[ix_keep]




# IDENTIFY SIGNIFICANT PCA DIMENSIONS


def get_sig_pcs(adata, n_iter = 3, nPCs_test = 300, threshold_method='95', show_plots=True, zero_center=True, verbose=True, in_place=True):

    # Subset adata to highly variable genes x cells (counts matrix only)
    adata_tmp = sc.AnnData(adata[:,adata.var.highly_variable].X)

    # Determine if the input matrix is sparse
    sparse=False
    if scipy.sparse.issparse(adata_tmp.X):
      sparse=True

    # Get eigenvalues from pca on data matrix
    if verbose: 
        print('Performing PCA on data')
    sc.pp.pca(adata_tmp, n_comps=nPCs_test, zero_center=zero_center)
    eig = adata_tmp.uns['pca']['variance']

    # Get eigenvalues from pca on randomly permuted data matrices
    if verbose: 
        print('Performing PCA on randomized data')
    eig_rand = np.zeros(shape=(n_iter, nPCs_test))
    eig_rand_max = []
    n_sig_PCs_trials = []
    for j in range(n_iter):

        if verbose and n_iter>1: sys.stdout.write('\rIteration %i / %i' % (j+1, n_iter)); sys.stdout.flush()
        
        adata_tmp_rand = adata_tmp.copy()
        
        if sparse:
          mat = adata_tmp_rand.X.todense()
        else:
          mat = adata_tmp_rand.X
        
        # randomly permute each row of the counts matrix
        for c in range(mat.shape[1]):
            np.random.seed(seed=j+c)
            mat[:,c] = mat[np.random.permutation(mat.shape[0]),c]
        
        if sparse:        
          adata_tmp_rand.X = scipy.sparse.csr_matrix(mat)
        else:
          adata_tmp_rand.X = mat
        
        sc.pp.pca(adata_tmp_rand, n_comps=nPCs_test, zero_center=zero_center)
        eig_rand_next = adata_tmp_rand.uns['pca']['variance']
        eig_rand[j,:] = eig_rand_next
        eig_rand_max.append(np.max(eig_rand_next))
        n_sig_PCs_trials.append(np.count_nonzero(eig>np.max(eig_rand_next)))

    # Set eigenvalue thresholding method
    if threshold_method == '95':
        method_string = 'Counting the # of PCs with eigenvalues above random in >95% of trials'
        eig_thresh = np.percentile(eig_rand_max,95)
    elif threshold_method == 'median':
        method_string = 'Counting the # of PCs with eigenvalues above random in >50% of trials'
        eig_thresh = np.percentile(eig_rand_max,50)
    elif threshold_method == 'all':
        method_string = 'Counting the # of PCs with eigenvalues above random across all trials'
        eig_thresh = np.percentile(eig_rand_max,100)
    
    # Determine # of PC dimensions with eigenvalues above threshold
    n_sig_PCs = np.count_nonzero(eig>eig_thresh)    

    if show_plots: 

        # Plot eigenvalue histograms
        bins = np.logspace(0, np.log10(np.max(eig)+10), 50)
        sns.histplot(eig_rand.flatten(), bins=bins, kde=False, alpha=1, label='random', stat='probability', color='orange')#, weights=np.zeros_like(data_rand) + 1. / len(data_rand))
        sns.histplot(eig, bins=bins, kde=False, alpha=0.5, label='data', stat='probability')#, weights=np.zeros_like(data) + 1. / len(data))
        plt.legend(loc='upper right')
        plt.axvline(x = eig_thresh, color = 'k', linestyle = '--', alpha=0.5, linewidth=1)
        plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Frequency')
        plt.show()

        # Plot scree (eigenvalues for each PC dimension)
        plt.plot([], label='data', color='#1f77b4', alpha=1)
        plt.plot([], label='random', color='#ff7f0e', alpha=1)
        plt.plot(eig, alpha=1, color='#1f77b4')
        for j in range(n_iter):
          plt.plot(eig_rand[j], alpha=1/n_iter, color='#ff7f0e')   
        plt.legend(loc='upper right')
        plt.axhline(y = eig_thresh, color = 'k', linestyle = '--', alpha=0.5, linewidth=1)
        plt.yscale('log')
        plt.xlabel('PC #')
        plt.ylabel('Eigenvalue')
        plt.show()

        # Plot nPCs above rand histograms
        sns.set_context(rc = {'patch.linewidth': 0.0})
        sns.histplot(n_sig_PCs_trials, kde=True, stat='probability', color='#1f77b4') 
        plt.xlabel('# PCs Above Random')
        plt.ylabel('Frequency')
        plt.xlim([0, nPCs_test])
        plt.show()

    # Print summary stats to screen
    if verbose: 
        print()
        print(method_string)
        print('Eigenvalue Threshold =', np.round(eig_thresh, 2))
        print('# Significant PCs =', n_sig_PCs)

    if in_place:
        adata.uns['n_sig_PCs'] = n_sig_PCs
        adata.uns['n_sig_PCs_trials'] = n_sig_PCs_trials
        return None
    
    else:
        return n_sig_PCs, n_sig_PCs_trials



# ESTIMATE DIMENSIONALITY 


def run_dim_tests_vscore(adata, batch_key=None, gene_filter_method='multiple', nPCs_test=300, n_trials=3, vpctl_tests=None, verbose=True):

  if vpctl_tests is None:
    vpctl_tests = [99, 97.5, 95, 92.5, 90, 87.5, 85, 82.5, 80, 77.5, 75, 72.5, 70, 67.5, 65, 62.5, 60]
    #vpctl_tests = [99, 95, 90, 85, 80, 75, 70, 65, 60]
    
  results_vpctl = []
  results_trial = []
  results_nHVgenes = []
  results_nPCs_each = []

  # Determine # of significant PC dimensions vs randomized data for different numbers of highly variable genes
  for n, vpctl in enumerate(vpctl_tests):
    
    if verbose: sys.stdout.write('\rRunning Dimensionality Test %i / %i' % (n+1, len(vpctl_tests))); sys.stdout.flush()
    
    # Get and filter variable genes for this vcptl 
    get_variable_genes(adata, batch_key=batch_key, filter_method=gene_filter_method, min_vscore_pctl=vpctl)
    
    # nPC dimensions tested cannot exceed the # of variable genes; adjust nPCs_test if needed
    nPCs_test_use = np.min([nPCs_test, np.sum(adata.var.highly_variable)-1])
    
    # Get # significant PCs for these variable genes & this vcptl
    _, n_sig_PCs_trials = get_sig_pcs(adata, n_iter = n_trials, nPCs_test = nPCs_test_use, show_plots=False, zero_center=True, verbose=False, in_place=False)
    
    # Report results from each independent trial
    for trial in range(0, n_trials):
      results_vpctl.append(vpctl)
      results_trial.append(trial)
      results_nHVgenes.append(np.sum(adata.var.highly_variable))
      results_nPCs_each.append(n_sig_PCs_trials[trial])

  # Export results to adata.uns
  results = pd.DataFrame({'vscore_pct': results_vpctl, 'trial': results_trial,'n_hv_genes': results_nHVgenes, 'n_sig_PCs': results_nPCs_each})
  adata.uns['dim_test_results'] = results 
  adata.uns['optim_vscore_pctl'] = results.vscore_pct[np.argmax(results.n_sig_PCs)]

  # Generate line plot
  plt.figure()
  ax = plt.subplot(111)
  sns.lineplot(x=results.vscore_pct, y=results.n_sig_PCs)
  ymin, ymax = ax.get_ylim()
  label_gap = (ymax - ymin) / 50 
  ix = results.trial==0
  [ax.text(x, ymax+label_gap, name, rotation=90, horizontalalignment='center') for x, y, name in zip(results[ix].vscore_pct, results[ix].n_sig_PCs, results[ix].n_hv_genes)]
  plt.ylabel('# PCs Above Random')
  plt.xlabel('vscore Percentile')
  ax.invert_xaxis()
  plt.show()


def run_dim_tests_scanpy(adata, batch_key=None, nPCs_test=300, n_trials=3, min_disp_tests=None, verbose=True):

  if min_disp_tests is None:
    min_disp_tests = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2]
    
    
  results_min_disp = []
  results_trial = []
  results_nHVgenes = []
  results_nPCs_each = []

  # Determine # of significant PC dimensions vs randomized data for different numbers of highly variable genes
  for n, min_disp in enumerate(min_disp_tests):
    
    if verbose: sys.stdout.write('\rRunning Dimensionality Test %i / %i' % (n+1, len(min_disp_tests))); sys.stdout.flush()
    
    # Get and filter variable genes for this vcptl 
    sc.pp.highly_variable_genes(adata, batch_key=batch_key, layer='tpm', min_mean=0.05, max_mean=10, n_bins=50, min_disp=min_disp)
    
    # nPC dimensions tested cannot exceed the # of variable genes; adjust nPCs_test if needed
    nPCs_test_use = np.min([nPCs_test, np.sum(adata.var.highly_variable)-1])
    
    # Get # significant PCs for these variable genes & this vcptl
    _, n_sig_PCs_trials = get_significant_pcs(adata, n_iter = n_trials, nPCs_test = nPCs_test_use, show_plots=False, zero_center=True, verbose=False, in_place=False)
    
    # Report results from each independent trial
    for trial in range(0, n_trials):
      results_min_disp.append(min_disp)
      results_trial.append(trial)
      results_nHVgenes.append(np.sum(adata.var.highly_variable))
      results_nPCs_each.append(n_sig_PCs_trials[trial])

  # Export results to adata.uns
  results = pd.DataFrame({'min_disp': results_min_disp, 'trial': results_trial,'n_hv_genes': results_nHVgenes, 'n_sig_PCs': results_nPCs_each})
  adata.uns['dim_test_results'] = results 

  # Generate line plot
  plt.figure()
  ax = plt.subplot(111)
  sns.lineplot(x=results.min_disp, y=results.n_sig_PCs)
  ymin, ymax = ax.get_ylim()
  label_gap = (ymax - ymin) / 50 
  ix = results.trial==0
  [ax.text(x, ymax+label_gap, name, rotation=90, horizontalalignment='center') for x, y, name in zip(results[ix].min_disp, results[ix].n_sig_PCs, results[ix].n_hv_genes)]
  plt.ylabel('# PCs Above Random')
  plt.xlabel('Minimum Normalized Dispersion')
  ax.invert_xaxis()
  plt.show()





# LEGACY ALIASES

get_significant_pcs = get_sig_pcs

