
import sys
import numpy as np
import scipy
import sklearn
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



# IDENTIFY HIGHLY VARIABLE GENES


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


def get_variable_genes_batch(adata, norm_counts_per_cell=1e6, batch_key=None, min_vscore_pctl=85, min_counts=3, min_cells=3):
    
    # Find variable genes for the entire adata object
    adata = get_variable_genes(adata, norm_counts_per_cell=1e6, min_vscore_pctl=min_vscore_pctl, min_counts=3, min_cells=3)

    # Now filter genes based on 
    # get a list of variable genes that were discovered within each batches
    batch_ids = np.unique(adata.obs[batch_key])
    n_batches = len(batch_ids)
    within_batch_hv_genes = []
    for b in batch_ids:
        adata_batch = adata[adata.obs[batch_key] == b]
        adata_batch = get_variable_genes(adata_batch)
        hv_genes_this_batch = list(adata_batch.uns['vscore_stats']['hv_genes']) 
        within_batch_hv_genes.append(hv_genes_this_batch)
    
    # filter variable genes based on the # of occurences
    within_batch_hv_genes = [g for gene in within_batch_hv_genes for g in gene]
    within_batch_hv_genes, c = np.unique(within_batch_hv_genes, return_counts=True)
    within_batch_hv_genes = within_batch_hv_genes[c >= 4]
    
    # update the highly variable gene flags in adata
    adata.var['highly_variable_all_batches'] = adata.var['highly_variable']
    adata.var['highly_variable'] = False
    adata.var.loc[within_batch_hv_genes, 'highly_variable'] = True

    return adata

def get_variable_genes(adata, norm_counts_per_cell=1e6, min_vscore_pctl=85, min_counts=3, min_cells=3):

    ''' 
    Identifies highly variable genes
    Requires a layer of adata that has been processed by total count normalization (e.g. tpm_nolog)
    '''

    E = adata.layers['tpm_nolog']
    
    # get variability statistics
    Vscores, CV_eff, CV_input, ix1, mu_gene, FF_gene, a, b = get_vscores(E) # ix1 = genes for which vscores could be returned

    # index genes based on vscores percentile
    ix2 = Vscores > 0 # ix2 = genes for which a positive vscore was obtained
    min_vscore = np.percentile(Vscores[ix2], min_vscore_pctl)    
    ix3 = (((E[:, ix1[ix2]] >= min_counts).sum(0).A.squeeze()>= min_cells) & (Vscores[ix2] >= min_vscore)) # ix3 = highly variable genes

    # annotate highly variable gene in adata
    if 'highly_variable' in adata.var.keys():
        adata.var['highly_variable_older'] = adata.var['highly_variable'].copy()
    hv_genes = adata.var_names[ix1[ix2][ix3]]
    adata.var['highly_variable'] = False
    adata.var.loc[hv_genes, 'highly_variable'] = True
    
    # save vscore stats 
    adata.var['vscore'] = np.nan
    adata.var.loc[adata.var_names[ix1], 'vscore'] = Vscores
    adata.var['mu_gene'] = np.nan
    adata.var.loc[adata.var_names[ix1], 'mu_gene'] = mu_gene
    adata.var['ff_gene'] = np.nan
    adata.var.loc[adata.var_names[ix1], 'ff_gene'] = FF_gene
    adata.uns['vscore_stats'] = {'hv_genes': hv_genes,
                                 'CV_eff': CV_eff,
                                 'CV_input': CV_input,
                                 'a': a,
                                 'b': b,
                                 'min_vscore': min_vscore}

    return adata


def plot_gene_ff(adata, gene_ix=None, color=None):

  if gene_ix == None:
    gene_ix = adata.var['highly_variable']
  
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


def plot_gene_vscores(adata, gene_ix=None, color=None):

  if gene_ix == None:
    gene_ix = adata.var['highly_variable']
  
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

 
def get_covarying_genes(adata, minimum_correlation=0.2, show_hist=True):

    # Subset adata to highly variable genes x cells (counts matrix only)
    adata_tmp = sc.AnnData(adata[:,adata.var.highly_variable].X)

    # Determine if the input matrix is sparse
    sparse=False
    if scipy.sparse.issparse(adata_tmp.X):
      sparse=True

    # Get nn correlation distance for each highly variable gene
    gene_correlation_matrix = 1-dew.sparse_corr(adata_tmp.X)
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
    gene_correlation_matrix = 1-dew.sparse_corr(adata_tmp_rand.X)
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


def get_significant_pcs(adata, n_iter = 3, n_comps_test = 200, threshold_method='95', show_plots=True, zero_center=True, verbose=True):

    # Subset adata to highly variable genes x cells (counts matrix only)
    adata_tmp = sc.AnnData(adata[:,adata.var.highly_variable].X)

    # Determine if the input matrix is sparse
    sparse=False
    if scipy.sparse.issparse(adata_tmp.X):
      sparse=True

    # Get eigenvalues from pca on data matrix
    if verbose: 
        print('Performing PCA on data matrix')
    sc.pp.pca(adata_tmp, n_comps=n_comps_test, zero_center=zero_center)
    eig = adata_tmp.uns['pca']['variance']

    # Get eigenvalues from pca on randomly permuted data matrices
    if verbose: 
        print('Performing PCA on randomized data matrices')
    eig_rand = np.zeros(shape=(n_iter, n_comps_test))
    eig_rand_max = []
    nPCs_above_rand = []
    for j in range(n_iter):

        if verbose: sys.stdout.write('\rIteration %i / %i' % (j+1, n_iter)); sys.stdout.flush()
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
        
        sc.pp.pca(adata_tmp_rand, n_comps=n_comps_test, zero_center=zero_center)
        eig_rand_next = adata_tmp_rand.uns['pca']['variance']
        eig_rand[j,:] = eig_rand_next
        eig_rand_max.append(np.max(eig_rand_next))
        nPCs_above_rand.append(np.count_nonzero(eig>np.max(eig_rand_next)))

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
        sns.histplot(nPCs_above_rand, kde=True, stat='probability', color='#1f77b4') 
        plt.xlabel('# PCs Above Random')
        plt.ylabel('Frequency')
        plt.xlim([0, n_comps_test])
        plt.show()

    # Print summary stats to screen
    if verbose: 
        print(method_string)
        print('Eigenvalue Threshold =', np.round(eig_thresh, 2))
        print('# Significant PCs =', n_sig_PCs)

    adata.uns['n_sig_PCs'] = n_sig_PCs
    adata.uns['n_sig_PCs_trials'] = nPCs_above_rand

    return adata




# ESTIMATE DIMENSIONALITY 


def run_dim_tests(adata, dim_test_n_comps_test=300, dim_test_n_trials=3, dim_test_vpctl=None, verbose=True):

  if dim_test_vpctl is None:
    dim_test_vpctl = [99, 97.5, 95, 92.5, 90, 87.5, 85, 82.5, 80, 75, 70, 65, 60, 55, 50]
    
  results_vpctl = []
  results_trial = []
  results_nHVgenes = []
  results_nPCs_each = []

  # Determine # of significant PC dimensions vs randomized data for different numbers of highly variable genes
  for n, vpctl in enumerate(dim_test_vpctl):
    if verbose:
      sys.stdout.write('\rRunning Dimensionality Test %i / %i' % (n+1, len(dim_test_vpctl))); sys.stdout.flush()
    get_variable_genes_batch(adata, batch_key=batch_key, min_vscore_pctl = vpctl)
    if dim_test_n_comps_test > np.sum(adata.var.highly_variable):
      # nPC dimensions tested cannot exceed the # of variable genes; adjust n_comps_test if needed
      get_significant_pcs(adata, n_iter = dim_test_n_trials, n_comps_test = np.sum(adata.var.highly_variable)-1, show_plots=False, zero_center=True, verbose=False)  
    else:
      get_significant_pcs(adata, n_iter = dim_test_n_trials, n_comps_test = dim_test_n_comps_test, show_plots=False, zero_center=True, verbose=False)
    
    # Report results from each independent trial
    for trial in range(0, dim_test_n_trials):
      results_vpctl.append(vpctl)
      results_trial.append(trial)
      results_nHVgenes.append(np.sum(adata.var.highly_variable))
      results_nPCs_each.append(adata.uns['n_sig_PCs_trials'][trial])

  # Export results to adata.uns
  results = pd.DataFrame({'vscore_pct': results_vpctl, 'trial': results_trial,'n_hv_genes': results_nHVgenes, 'n_sig_PCs': results_nPCs_each})
  adata.uns['dim_test_results'] = results 
  adata.uns['optim_vscore_pctl'] = results.vscore_pct[np.argmax(results.n_sig_PCs)]

  def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))

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

  return adata

