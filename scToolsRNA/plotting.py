

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import plotly.express as px

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D



# PLOTTING


###


def plot_3d_embedding(plot_data_df, color_data, dims_to_plot=[0,1,2]):

  fig = px.scatter_3d(plot_data_df,
                      x=dims_to_plot[0], y=dims_to_plot[1], z=dims_to_plot[2],
                      color=color_data,
                      size_max=12, size=np.repeat(1,len(plot_data_df)),
                      opacity=1,
                      height=1000,
                      color_discrete_sequence=sc.pl.palettes.default_20,
                      color_continuous_scale=px.colors.sequential.Viridis)

  fig.update_layout(scene_dragmode='orbit',
                    scene = dict(xaxis = dict(visible=False),
                                yaxis = dict(visible=False),
                                zaxis = dict(visible=False)),
                    legend_title_text='',
                    coloraxis_colorbar_title_text = 'log<br>counts',
                    showlegend=True,
                    coloraxis_colorbar_thickness=10)

  fig.update_traces(marker=dict(line=dict(width=0)))

  fig.show()


def plot_umap3d(adata, color, window_height=1000):
  
    # Generate or use existing 3D UMAP coordinates in obsm; if present, make a backup copy of 2D UMAP coordinates
    if 'X_umap_3d' not in adata.obsm:        
        calc_umap3d = True
        if 'X_umap' in adata.obsm:
            adata.obsm['X_umap_2d'] = adata.obsm['X_umap'].copy()
        sc.tl.umap(adata, n_components=3)       
        adata.obsm['X_umap_3d'] = adata.obsm['X_umap'].copy()
    else:
        calc_umap3d = False

    # Reset the default X_umap embedding to the 2d version
    if 'X_umap_2d' in adata.obsm and calc_umap3d:  
        adata.obsm['X_umap'] = adata.obsm['X_umap_2d']

    # Generate the plot using Plotly express
    fig = px.scatter_3d(pd.DataFrame(adata.obsm['X_umap_3d']), 
                      x=0, y=1, z=2, 
                      size_max=8, size=np.repeat(1,len(adata)), 
                      opacity=1, color=sc.get.obs_df(adata, color, layer='raw').values.flatten(),
                      color_discrete_sequence=sc.pl.palettes.default_20, color_continuous_scale=px.colors.sequential.Viridis,
                      height=window_height)
  
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




# EXPORT GIF



def create_3d_rotation_animation(data, rotation_duration=3, fps=10, point_size=4, color_dimension=None, cmap='viridis', alpha=1, show_legend=True, show_axes=True, save_path='rotating_plot.gif'):
    """
    Create a rotating 3D scatter plot animation from a Pandas DataFrame or a NumPy array.

    Parameters:
    - data: Pandas DataFrame or NumPy array containing 3D coordinates.
    - rotation_duration: Duration of the rotation in seconds.
    - fps: Frames per second for the animation.
    - point_size: Size of the points in the scatter plot.
    - color_dimension: Array of values for the color dimension (use None for a single color).
    - show_legend: Whether to display the color legend (default is True).
    - save_path: File path to save the resulting GIF.

    Returns:
    - None (saves the animation as a GIF).
    """

    # Function to update the plot in each frame
    def update(frame, sc, ax):
        azimuthal_angle = (360 * frame) / num_frames  # Calculate azimuthal angle for rotation
        ax.view_init(elev=20, azim=azimuthal_angle)
        return sc,

    # Check if the input is a Pandas DataFrame or a NumPy array
    if isinstance(data, pd.DataFrame):
        # Extract coordinates from the DataFrame
        num_columns = data.shape[1]
        coordinates = [data.iloc[:, i].to_numpy() for i in range(num_columns)]
    elif isinstance(data, np.ndarray):
        # If the input is a NumPy array, assume it has shape (n, 3)
        if data.shape[1] != 3:
            raise ValueError("NumPy array must have shape (n, 3) for 3D coordinates.")
        coordinates = [data[:, i] for i in range(3)]
    else:
        raise ValueError("Unsupported input type. Please provide a Pandas DataFrame or a NumPy array.")

    # Create a 3D scatter plot with a larger figure size
    fig = plt.figure(figsize=(20, 20))  # Adjust width and height as needed
    ax = fig.add_subplot(111, projection='3d')

    # Determine if the color dimension is numeric or categorical
    if color_dimension is not None:
        if pd.api.types.is_categorical_dtype(color_dimension):
            # If categorical, assign unique colors to each category
            unique_categories = color_dimension.cat.categories
            category_colors = plt.cm.get_cmap(cmap, len(unique_categories))
            color_mapping = {category: category_colors(i) for i, category in enumerate(unique_categories)}
            color_array = [color_mapping[category] if pd.notna(category) else (0, 0, 0, 0) for category in color_dimension]
            sc = ax.scatter(*coordinates, s=point_size, c=color_array, marker='o', alpha=alpha)
            if show_legend:
                # Increase the legend size and remove the frame
                legend = ax.legend([plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[category], markersize=10) for category in unique_categories],
                                   unique_categories, loc='upper left', bbox_to_anchor=(0.1, 0.9), borderaxespad=0., frameon=False, fontsize=12)
    else:
        # Create scatter plot without color dimension
        sc = ax.scatter(*coordinates, c='blue', marker='o', s=point_size, alpha=alpha)

    # Set plot parameters
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for the axes
    ax.set_facecolor('white')  # Set background color to white
    ax.grid(False)  # Hide grid
    ax.axis('off')  # Hide axes

    # Plot X, Y, and Z axes in black with linewidth 5
    if show_axes:
        x_center = (coordinates[0].max() + coordinates[0].min())/2
        y_center = (coordinates[1].max() + coordinates[1].min())/2
        z_center = (coordinates[2].max() + coordinates[2].min())/2
        ax.plot([x_center,x_center], [y_center, y_center], [coordinates[2].min() - 2, coordinates[2].max() + 2], c = (0, 0, 0, 0.5), lw = 5)
        ax.plot([x_center,x_center], [coordinates[1].min() - 2, coordinates[1].max() + 2], [z_center, z_center], c = (0, 0, 0, 0.5), lw = 5)
        ax.plot([coordinates[0].min() - 2, coordinates[0].max() + 2], [y_center,y_center], [z_center, z_center], c = (0, 0, 0, 0.5), lw = 5)

    # Calculate the number of frames needed for a 360-degree rotation
    num_frames = int(rotation_duration * fps)

    # Set the interval between frames
    interval = 1000 / fps

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_frames, fargs=(sc, ax), interval=interval, blit=True)

    # Save the animation as a GIF with increased DPI
    ani.save(save_path, writer='imagemagick', fps=fps, dpi=100)  # You can adjust the dpi value as needed

    # Display the plot (optional)
    plt.show()

# Example usage with a Pandas DataFrame 'df' and a color dimension 'color_dim'
# Replace 'df' and 'color_dim' with your actual DataFrame containing 3D coordinates and the color dimension
# create_3d_rotation_animation(df, point_size=30, color_dimension=color_dim, show_legend=True)
