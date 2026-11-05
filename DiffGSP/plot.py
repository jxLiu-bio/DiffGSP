import matplotlib
# Use 'Agg' backend for non-interactive environments
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as ss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata # KEY: Interpolation tool


def plot_visium_3d_surface_mpl(
        adata: ad.AnnData,
        gene_name: str,
        coord_key: str = 'spatial',
        output_path: str = '3d_surface_plot.png',
        cmap_name: str = 'viridis',
        grid_resolution: int = 50  # Resolution of the interpolation grid
) -> str:
    """
    Generates a static 3D smooth surface plot for Visium spatial transcriptomics data
    using SciPy's interpolation and Matplotlib's plot_surface.

    The function maps the expression value (Z) of the target gene onto a regular
    (X, Y) grid to create a continuous, smooth surface.

    Args:
        adata: The AnnData object containing Visium data.
        gene_name: The name of the gene to be visualized.
        coord_key: The key in adata.obsm where the spatial coordinates are stored.
        output_path: The file path where the static image will be saved.
        cmap_name: The Matplotlib colormap to use.
        grid_resolution: The number of points for the interpolation grid (e.g., 50x50).

    Returns:
        The path of the saved file upon successful execution.
    """

    # 1. Input Validation and Data Extraction (Same as before)
    if gene_name not in adata.var_names:
        raise ValueError(f"Gene '{gene_name}' not found in adata.var_names.")

    if coord_key not in adata.obsm_keys():
        raise ValueError(f"Spatial coordinates key '{coord_key}' not found in adata.obsm.")

    # Extract X, Y, and Z (Expression) data
    X_coords = adata.obsm[coord_key][:, 0]
    Y_coords = adata.obsm[coord_key][:, 1]

    if ss.issparse(adata.X):
        Z_expression = adata[:, gene_name].X.toarray().flatten()
    else:
        Z_expression = adata[:, gene_name].X.flatten()

    # Combine coordinates into a single array for griddata
    points = np.stack((X_coords, Y_coords), axis=1)

    # 2. Interpolation Step (Creating the Smooth Surface)

    # Define the bounds of the new, regular grid
    x_min, x_max = X_coords.min(), X_coords.max()
    y_min, y_max = Y_coords.min(), Y_coords.max()

    # Create the regular grid coordinates (xi, yi)
    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    XI, YI = np.meshgrid(xi, yi)

    # Use griddata to interpolate the expression values (Z_expression) onto the new grid (ZI)
    # 'cubic' method provides a smooth, continuous surface
    ZI = griddata(
        points,
        Z_expression,
        (XI, YI),
        method='cubic'  # Other methods: 'linear', 'nearest'
    )

    # Note: Interpolation may produce NaN values outside the original convex hull.

    # 3. Create Static Matplotlib 3D Surface Plot

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Use plot_surface instead of scatter
    surface = ax.plot_surface(
        XI,
        YI,
        ZI,
        cmap=cmap_name,
        linewidth=0,
        antialiased=True,  # Helps with smoothness
        vmax=Z_expression.max()  # Ensure color map covers the full range
    )

    # 4. Set Titles and Labels
    ax.set_title(f'3D Surface Expression of {gene_name}', fontsize=16)
    ax.set_xlabel('Spatial X', fontsize=12)
    ax.set_ylabel('Spatial Y', fontsize=12)
    ax.set_zlabel(f'{gene_name} Expression (Height)', fontsize=12)

    # Add a color bar
    fig.colorbar(surface, shrink=0.5, aspect=5, label='Gene Expression Value')

    # Set the same scale for X and Y to accurately reflect spatial area
    ax.set_box_aspect([1, 1, 0.5])  # Adjust the last value (Z-axis scale) to your preference

    # 5. Save the Figure and Close
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return output_path


def spatial_plot(adata, color, marker='o', vmin=None, vmax=None, spot_size=15, cmap='magma', title=None, figsize=(4, 4)):
    """
    A custom replacement for sc.pl.spatial using matplotlib scatter.

    Parameters:
    - adata: AnnData object.
    - color: Gene name or observation key.
    - vmax: Maximum value for color scale.
    - spot_size: Size of the spots (s parameter).
    - cmap: Colormap (default 'magma').
    - title: Plot title.
    """
    # 1. Prepare data
    coords = adata.obsm['spatial']

    # Extract expression data (handle sparse matrices)
    if color in adata.var_names:
        counts = adata[:, color].X
        if hasattr(counts, "toarray"):
            counts = counts.toarray()
        counts = counts.flatten()
    elif color in adata.obs.columns:
        counts = adata.obs[color].values
    else:
        raise ValueError(f"'{color}' not found in var_names or obs.")

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=figsize)

    # 3. Your specific scatter configuration
    vmax_val = vmax if vmax is not None else np.percentile(counts, 99)

    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=counts,
        marker=marker,
        s=spot_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax_val,
        edgecolor='none',
        alpha=1.0
    )

    # 4. Refine aesthetics
    ax.set_aspect('equal')
    ax.set_title(title if title else color)
    ax.invert_yaxis()  # Standard for matching tissue slice orientation
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar
    plt.colorbar(sc, ax=ax, label='Expression')

    plt.show()

