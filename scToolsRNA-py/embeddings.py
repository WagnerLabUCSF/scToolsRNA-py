

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import plotly.express as px



# PLOTTING


def plot_umap3d(adata, color):
  
    # Generate 3D UMAP and store coordinates in obsm; preserve 2D coordinates if they exist
    if 'X_umap' in adata.obsm:
        tmp = adata.obsm['X_umap'].copy()
    sc.tl.umap(adata, n_components=3)
    adata.obsm['X_umap_3d'] = adata.obsm['X_umap']
    adata.obsm['X_umap']=tmp
  
    # Generate the plot using Plotly express
    fig = px.scatter_3d(pd.DataFrame(adata.obsm['X_umap_3d']), 
                      x=0, y=1, z=2, 
                      size_max=8, size=np.repeat(1,len(adata)), 
                      opacity=1, color=sc.get.obs_df(adata, color, layer='raw').tolist(), 
                      color_discrete_sequence=sc.pl.palettes.default_20, color_continuous_scale=px.colors.sequential.Viridis)#,
                      #height=plot_window_height, width=plot_window_width)
  
    fig.update_layout(scene = dict(xaxis = dict(visible=False), yaxis = dict(visible=False), zaxis = dict(visible=False)), 
                    scene_dragmode='orbit', scene_camera = dict(eye=dict(x=0, y=0, z=1.5)), 
                    coloraxis_colorbar_title_text = 'log<br>counts', showlegend=True, coloraxis_colorbar_thickness=10, legend_title_text=' ')
  
    fig.update_traces(marker=dict(line=dict(width=0)))
  
    fig.show()


def format_axes(eq_aspect='all', rm_colorbar=False):
    '''
    Gets axes from the current figure and applies custom formatting options
    In general, each parameter is a list of axis indices (e.g. [0,1,2]) that will be modified
    Colorbar is assumed to be the last set of axes
    '''
    
    # get axes from current figure
    ax = plt.gcf().axes

    # format axes aspect ratio
    if eq_aspect != 'all':
        for j in eq_aspect:
            ax[j].set_aspect('equal') 
    else:
        for j in range(len(ax)):
            ax[j].set_aspect('equal') 

    # remove colorbar
    if rm_colorbar:
        j=len(ax)-1
        if j>0:
            ax[j].remove()


def darken_cmap(cmap, scale_factor):
    cdat = np.zeros((cmap.N, 4))
    for ii in range(cdat.shape[0]):
        curcol = cmap(ii)
        cdat[ii,0] = curcol[0] * scale_factor
        cdat[ii,1] = curcol[1] * scale_factor
        cdat[ii,2] = curcol[2] * scale_factor
        cdat[ii,3] = 1
    cmap = cmap.from_list(cmap.N, cdat)
    return cmap

