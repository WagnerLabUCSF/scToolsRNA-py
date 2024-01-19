
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import sklearn
import warnings



# EVALUATION OF CLUSTERING RESULTS
    

def get_confusion_matrix(labels_A, labels_B,
                         normalize=True,
                         title=None,
                         reorder_columns=True,
                         reorder_rows=True,
                         cmap=plt.cm.Blues,
                         overlay_values=False,
                         vmin=None,
                         vmax=None,
                         show_plot=True,
                         return_df=False,
                         figsize=4):
    '''
    Plots a confusion matrix comparing two sets labels. 
    '''
    
    # Filter labels if value is missing from either set
    nan_flag = labels_A.isnull() | labels_B.isnull()
    labels_A = labels_A[~nan_flag]
    labels_B = labels_B[~nan_flag]

    # Get all the unique values for each set of labels
    labels_A_unique = np.unique(labels_A.astype('string'))
    labels_B_unique = np.unique(labels_B.astype('string'))

    # Compute confusion matrix 
    cm = sklearn.metrics.confusion_matrix(labels_A, labels_B)
    non_empty_rows = cm.sum(axis=0)!=0
    non_empty_cols = cm.sum(axis=1)!=0
    cm = cm[:,non_empty_rows]
    cm = cm[non_empty_cols,:]
    cm = cm.T
    
    # Normalize by rows (label B)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Set title, colorbar, and axis names
    if normalize:
        colorbar_label = 'Fraction Overlap'
        if not title:
            title = 'Normalized confusion matrix'
    else:
        colorbar_label = '# Overlaps'
        if not title:
            title = 'Confusion matrix, without normalization'  
  
    # If available, get the label category names for plotting
    if hasattr(labels_A, 'name'):
        labels_A_name = labels_A.name 
    else:
        labels_A_name = 'Label A'
    if hasattr(labels_B, 'name'):
        labels_B_name = labels_B.name 
    else:
        labels_B_name = 'Label B'

    # If requested, reorder the rows and columns by best match
    if reorder_columns:
        top_match_c = np.argmax(cm, axis=0)
        reordered_columns = np.argsort(top_match_c)
        cm=cm[:,reordered_columns]
        labels_A_unique_sorted = labels_A_unique[reordered_columns]
    else:
        labels_A_unique_sorted = labels_A_unique

    if reorder_rows:
        top_match_r = np.argmax(cm, axis=1)
        reordered_rows = np.argsort(top_match_r)
        cm=cm[reordered_rows,:]
        labels_B_unique_sorted = labels_B_unique[reordered_rows]
    else:
        labels_B_unique_sorted = labels_B_unique

    # If requested, generate heatmap and format figure axes
    if show_plot:
        
        plt.rcParams['axes.grid'] = False
        fig, ax = plt.subplots(figsize=(figsize,figsize))
        im = plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect('equal') 
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_title(title)
        ax.set_ylabel(labels_B_name)
        ax.set_xlabel(labels_A_name)
        ax.set_xticklabels(labels_A_unique_sorted, rotation=90, ha='center', minor=False)
        ax.set_yticklabels(labels_B_unique_sorted)

        # Format colorbar
        cb=ax.figure.colorbar(im, ax=ax, shrink=0.5)
        #cb.ax.tick_params(labelsize=10) 
        cb.ax.set_ylabel(colorbar_label, rotation=90)

        # If requested, loop over data dimensions and create text annotations
        if overlay_values:
            fmt = '.1f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            size=8)
    
    # If requested, return dataframe mapping top A match for each B
    if return_df:
    
        if reorder_rows:
            labels_A_mapped = labels_A_unique_sorted[top_match_r]
        else:
            labels_A_mapped = labels_A_unique_sorted

        mapping_df = pd.DataFrame(data=labels_A_mapped, index=labels_B_unique, columns=['top_match'])
        
        # Sort the index labels, if possible
        orig_list = labels_B_unique
        orig_list_digits = [s for s in orig_list if s.isdigit()]
        if len(orig_list)==len(orig_list_digits):
            mapping_df.index = mapping_df.index.astype(int)
            mapping_df = mapping_df.sort_index()
        
        return mapping_df
        
            
plot_confusion_matrix = get_confusion_matrix # alias to legacy function name 


def transfer_top_obs_label(adata, column_from, column_to):
    
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      
      # Group the dataframe by the 'a' column and find the mode of the 'b' column within each group
      most_common_per_group = adata.obs.groupby(column_to)[column_from].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index()

      # Rename the columns
      new_column_name = column_from + '->' + column_to
      most_common_per_group.columns = [column_to, new_column_name]

      # Merge w/the original DataFrame
      result_df = pd.merge(adata.obs, most_common_per_group, on=column_to, how='left')
      result_df[new_column_name] = result_df[new_column_name].astype('category')

      # Save to adata
      adata.obs = result_df

    return adata


def plot_stacked_barplot(labels_A, 
                         labels_B, 
                         normalize='index', 
                         fig_width=4, 
                         fig_height=4):

    # Cross-tabulate the two sets of labels
    crstb = pd.crosstab(labels_A, labels_B, normalize=normalize)
    
    # Plot stacked bars
    crstb.plot.bar(stacked=True, width=0.8, figsize=(fig_width, fig_height))
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.ylim([0,1])
    plt.ylabel('Proportion')
    plt.grid(False)
    plt.show()
