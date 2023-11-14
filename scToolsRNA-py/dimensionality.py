
import numpy as np
import scipy
import sklearn
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns



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


def get_variable_genes(E, base_ix=[], min_vscore_pctl=85, min_counts=3, min_cells=3, show_FF_plot=False, show_vscore_plot=False, return_stats=False, plot_title=''):

    ''' 
    Filter genes by expression level and variability
    Return list of filtered gene indices
    '''

    if len(base_ix) == 0:
        base_ix = np.arange(E.shape[0])

    # get variability statistics    
    Vscores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b = get_vscores(E[base_ix, :])

    # index genes with positive vscores     
    ix2 = Vscores > 0

    # index genes based on vscore percentile
    min_vscore = np.percentile(Vscores[ix2], min_vscore_pctl)    
    ix = (((E[:, gene_ix[ix2]] >= min_counts).sum(0).A.squeeze()>= min_cells) & (Vscores[ix2] >= min_vscore))

    if show_FF_plot:
        x_min = 0.5 * np.min(mu_gene[ix2])
        x_max = 2 * np.max(mu_gene[ix2])
        xTh = x_min * np.exp(np.log(x_max / x_min) * np.linspace(0, 1, 100))
        yTh = (1 + a) * (1 + b) + b * xTh
        plt.figure(figsize=(6, 6))
        plt.scatter(np.log10(mu_gene[ix2]), np.log10(FF_gene[ix2]), c=np.array(['grey']), alpha=0.3, edgecolors=None, s=4)
        plt.scatter(np.log10(mu_gene[ix2])[ix], np.log10(FF_gene[ix2])[ix], c=np.log10(Vscores[ix2])[ix], cmap='jet', alpha=0.3, edgecolors=None, s=4)
        plt.plot(np.log10(xTh), np.log10(yTh))
        plt.title(plot_title)
        plt.xlabel('Mean Transcripts Per Cell (log10)')
        plt.ylabel('Gene Fano Factor (log10)')
        plt.show()

    if show_vscore_plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(np.log10(mu_gene[ix2]), np.log10(Vscores[ix2]), c=np.array(['grey']), alpha=0.3, edgecolors=None, s=4)
        plt.scatter(np.log10(mu_gene[ix2])[ix], np.log10(Vscores[ix2])[ix], c=np.log10(FF_gene[ix2])[ix], cmap='jet', alpha=0.3, edgecolors=None, s=4)
        plt.title(plot_title)
        plt.xlabel('Mean Transcripts Per Cell (log10)')
        plt.ylabel('Vscores (log10)')
        plt.show()

    if return_stats:
        return {'gene_ix': gene_ix[ix2][ix],
                'vscores': Vscores[ix2][ix], 
                'mu_gene': mu_gene[ix2][ix],
                'FF_gene': FF_gene[ix2][ix],
                'CV_eff': CV_eff,
                'CV_input': CV_input,
                'a': a,
                'b': b,
                'min_vscore': min_vscore}
    else:
        return gene_ix[ix2][ix]


def get_covarying_genes(E, gene_ix, minimum_correlation=0.2, show_hist=False, sample_name=''):

    # subset input matrix to gene_ix
    E = E[:,gene_ix]
    
    # compute gene-gene correlation distance matrix (1-correlation)
    #gene_correlation_matrix1 = sklearn.metrics.pairwise_distances(E.todense().T, metric='correlation',n_jobs=-1)
    gene_correlation_matrix = 1-sparse_corr(E) # approx. 2X faster than sklearn
  
    # for each gene, get correlation to the nearest gene neighbor (ignoring self)
    np.fill_diagonal(gene_correlation_matrix, np.inf)
    max_neighbor_corr = 1-gene_correlation_matrix.min(axis=1)
  
    # filter genes whose nearest neighbor correlation is above threshold 
    ix_keep = np.array(max_neighbor_corr > minimum_correlation, dtype=bool).squeeze()
  
    # plot distribution of top gene-gene correlations
    if show_hist:
        plt.figure(figsize=(6, 6))
        plt.hist(max_neighbor_corr,bins=100)
        plt.title(sample_name)
        plt.xlabel('Nearest Gene Correlation')
        plt.ylabel('Counts')
        plt.show()
  
    return gene_ix[ix_keep]

    


# IDENTIFY SIGNIFICANT PCA DIMENSIONS


def get_significant_pcs(adata, n_iter = 3, n_comps_test = 200, threshold_method='95', show_plots=True, zero_center=True):

    # Subset adata to highly variable genes x cells (counts matrix only)
    adata_tmp = sc.AnnData(adata[:,adata.var.highly_variable].X)

    # Determine if the input matrix is sparse
    sparse=False
    if scipy.sparse.issparse(adata_tmp.X):
      sparse=True

    # Get eigenvalues from pca on data matrix
    print('Performing PCA on data matrix')
    sc.pp.pca(adata_tmp, n_comps=n_comps_test, zero_center=zero_center)
    eig = adata_tmp.uns['pca']['variance']

    # Get eigenvalues from pca on randomly permuted data matrices
    print('Performing PCA on randomized data matrices')
    eig_rand = np.zeros(shape=(n_iter, n_comps_test))
    eig_rand_max = []
    nPCs_above_rand = []
    for j in range(n_iter):
        #print('Iteration', j+1, '/', n_iter, end='\r')
        sys.stdout.write('\rIteration %i / %i' % (j+1, n_iter)); sys.stdout.flush()
        np.random.seed(seed=j)
        adata_tmp_rand = adata_tmp.copy()
        
        if sparse:
          mat = adata_tmp_rand.X.todense()
        else:
          mat = adata_tmp_rand.X
        
        # randomly permute each row of the counts matrix
        for c in range(mat.shape[1]):
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
    print(method_string)
    print('Eigenvalue Threshold =', np.round(eig_thresh, 2))
    print('# Significant PCs =', n_sig_PCs)

    adata.uns['n_sig_PCs'] = n_sig_PCs

    return adata




