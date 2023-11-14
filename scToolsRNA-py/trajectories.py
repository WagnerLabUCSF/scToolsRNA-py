
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import scipy



# TRAJECTORY ANALYSIS


def get_dynamic_genes(adata, sliding_window=100, fdr_alpha = 0.05, min_cells=20, nVarGenes=2000):

    '''
    Expects an AnnData object that has already been subsetted to cells and/or genes of interest.
    Cells are ranked by dpt pseudotime. Genes are tested for significant differential expression 
    between two sliding windows corresponding the highest and lowest average expression. FDR values
    are then calculated by thresholding p-values calculated from randomized data.
    Returns a copy of adata with the following fields added: 
        adata.var['dyn_peak_cell']: pseudotime-ordered cell with the highest mean expression
        adata.var['dyn_fdr']: fdr-corrected p-value for differential expression
        adata.var['dyn_fdr_flag']: boolean flag, true if fdr <= fdr_alpha
    '''
    

    # Function for calculating p-values for each gene from min & max sliding window expression values
    def get_slidingwind_pv(X, sliding_window):
        # construct a series of sliding windows over the cells in X
        wind=[]
        nCells = X.shape[0]
        for k in range(nCells-sliding_window+1):    
            wind.append(list(range(k, k+sliding_window)))
        # calculate p-values on the sliding windows
        pv = []
        max_cell_this_gene = []
        nGenes = X.shape[1]
        for j in range(nGenes):
            tmp_X_avg = []
            # get mean expression of gene j in each sliding window k
            for k in range(len(wind)-1):    
                tmp_X_avg.append(np.mean(X[wind[k],j]))
            # determine min and max sliding windows for this gene
            max_wind = np.argmax(tmp_X_avg)
            min_wind = np.argmin(tmp_X_avg)
            # determine if this gene displays significant differential expression
            _,p=scipy.stats.ttest_ind(X[wind[max_wind],j],X[wind[min_wind],j])
            pv.append(p[0])
            max_cell_this_gene.append(max_wind)
        return np.array(pv), np.array(max_cell_this_gene)

    # create a new adata object for the dynamic genes analysis
    adata_dyn = adata.copy()

    # reinitiate adata_dyn from raw counts
    if 'raw_nolog' in adata_dyn.layers:
        adata_dyn.X = adata_dyn.layers['raw_nolog']
    elif adata_dyn.raw:
        adata_dyn.X = adata_dyn.raw.X
    elif 'raw' in adata_dyn.layers:
        adata_dyn.X = adata_dyn.layers['raw']
    else:
        print('Error: raw counts layer required but not provided')
        return

    # pre-filter genes based on minimum expression 
    expressed_genes = np.squeeze(np.asarray(np.sum(adata_dyn.X  >= 1, axis=0) >= min_cells))
    adata_dyn = adata_dyn[:,expressed_genes]
    nGenes_expressed = adata_dyn.shape[1]

    # pre-filter genes based on variability
    nVarGenes = min([nGenes_expressed, nVarGenes])
    sc.pp.normalize_per_cell(adata_dyn, counts_per_cell_after=10**6) # TPM normalization
    sc.pp.log1p(adata_dyn)
    sc.pp.highly_variable_genes(adata_dyn, n_top_genes=nVarGenes)
    adata_dyn = adata_dyn[:,adata_dyn.var['highly_variable'] == True]
    
    # import counts and pseudotime from the AnnData object
    cell_order = np.argsort(adata_dyn.obs['dpt_pseudotime'])
    
    # reorder cells
    if scipy.sparse.issparse(adata_dyn.X):
        X = adata_dyn.X[cell_order,:].todense()
    else:
        X = adata_dyn.X[cell_order,:]

    # calculate p values on the pseudotime-ordered data
    print('calculating p-values')
    pv, peak_cell = get_slidingwind_pv(X, sliding_window)
    adata_dyn.var['dyn_peak_cell'] = peak_cell#np.argsort(gene_ord)
    print('done calculating p-values')
    
    # calculate p values on the randomized data
    print('calculating randomized p-values')
    np.random.seed(802)
    X_rand = X[np.random.permutation(cell_order),:]
    pv_rand, _ = get_slidingwind_pv(X_rand, sliding_window)
    print('done calculating randomized p-values')

    # calculate fdr as the fraction of randomized p-values that exceed this p-value
    print('calculating fdr')
    fdr = []
    fdr_flag = []
    nGenes = adata_dyn.shape[1]
    for j in range(nGenes):
        fdr.append(sum(pv_rand <= pv[j])/nGenes)
        fdr_flag.append(fdr[j] <= fdr_alpha)
    adata_dyn.var['dyn_fdr'] = fdr
    adata_dyn.var['dyn_fdr_flag'] = fdr_flag
    print('done calculating fdr')

    return adata_dyn


def plot_dpt_trajectory(adata, key, layer='raw', sliding_window=100, return_axes=False, save=None):
  
    '''
    Expects an AnnData object that has already been subsetted to cells and/or genes of interest.
    Generates a lineplot for a single gene or AnnData observation (obs matrix column) error bands 
    at +/- 1 sd.  
    '''

    # get xy plotting data from adata
    df=pd.DataFrame()
    df['x']=adata.obs['dpt_pseudotime']

    # key can be either a gene or a column in 'obs'
    if key in adata.var_names:
        df['y']=convert_to_dense(adata[:, adata.var_names==key].layers[layer])
    elif key in adata.obs.columns:
        df['y']=adata.obs[key]

    # calculate sliding window mean and std dev on sorted x
    df=df.sort_values('x')
    df['y_mn'] = df['y'].rolling(sliding_window).mean().tolist()
    df['y_std'] = df['y'].rolling(sliding_window).std().tolist()

    # Define variables to plot, upper and lower bounds = 1 * sd 
    y_mean = df['y_mn']
    x = df['x']
    y_std = df['y_std']
    lower = y_mean - y_std
    upper = y_mean + y_std

    # draw plot with error band and extra formatting to match seaborn style
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(x, y_mean, label='signal mean')
    ax.plot(x, lower, color='tab:blue', alpha=0.1)
    ax.plot(x, upper, color='tab:blue', alpha=0.1)
    ax.fill_between(x, lower, upper, alpha=0.2)
    ax.set_xlabel('dpt pseudotime')
    ax.set_ylabel(key)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save:
      plt.savefig('figures/dpt_lineplot'+save)

    if return_axes:
        return ax




