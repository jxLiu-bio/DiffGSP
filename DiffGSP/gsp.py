import scanpy as sc
import pandas as pd
import numpy as np
import os
import random
import anndata
import torch
from scanpy.plotting import spatial
from torch.optim import LBFGS
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import scipy.sparse as ss
from tqdm import tqdm
import itertools
import warnings
from .svg_select_spagft import rank_gene_smooth
import numba as nb


# warnings.filterwarnings('ignore')


def run_diffgsp(adata,
                gene_network=False,
                k=2,
                alpha=0,
                device='cuda:0',
                max_iterations=100,
                lr=0.05,
                spatial_key='spatial',
                rad_cutoff=None,
                max_iter=5,
                method='BFGS',
                variable=None,
                normlization=False,
                data_type=None,
                bin_size=None,
                top_genes_train=None,
                scale=None,
                knn_method='Radius',
                k_cutoff=18):
    """
    The function is designed to address the transcript diffusion and enhance data quality for sequencing-based
    spatial transcriptomics.

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial transcriptomics dataset object, including expression matrix and spatial metadata.
    gene_network : bool, optional (default=False)
        Whether to include the gene graph structure for modeling gene-wise diffusion.
    k : int, optional (default=2)
        Order of graph filtering or the number of graph Fourier modes used.
    alpha : float, optional (default=0)
        Weighting factor balancing the loss between background and in-tissue regions.
    device : str, optional (default='cuda:0')
        Computation device, e.g., 'cpu' or 'cuda:0'.
    max_iterations : int, optional (default=100)
        Maximum number of optimization iterations (used in BFGS mode).
    lr : float, optional (default=0.05)
        Learning rate for the BFGS optimizer.
    spatial_key : str or list (default='spatial')
        Keys in `adata.obs` or `'spatial'` in `adata.obsm` used as spatial coordinates.
    rad_cutoff : float, optional (default=None)
        Spatial radius cutoff for graph construction; set automatically based on `data_type`.
    max_iter : int, optional (default=5)
        Internal maximum iteration for each LBFGS step.
    method : str, optional (default='BFGS')
        Optimization method: 'BFGS' for automatic optimization, 'manual' for fixed parameter input.
    variable : list[float] or torch.Tensor, optional (default=None)
        User-defined diffusion coefficients (used only when `method='manual'`).
    normlization : bool, optional (default=False)
        Whether to normalize the denoised matrix to preserve the original total expression per gene.
    data_type : str, optional (default=None)
        Type of spatial transcriptomics platform (e.g., 'Visium', 'Stereoseq', 'VisiumHD', 'Slideseq').
    bin_size : float, optional (default=None)
        Bin size or spatial resolution (required for 'Stereoseq' or 'VisiumHD').
    top_genes_train : int, optional (default=None)
        Number of top-expressed genes used during parameter training (full matrix restored after optimization).
    scale : float, optional (default=None)
        Spatial scaling factor for adjusting distances between spots during graph construction.
    knn_method : str, optional (default='Radius')
        Method to construct the spatial graph: 'Radius' (radius-based) or 'KNN' (k-nearest neighbors).
    k_cutoff : int, optional (default=18)
        Number of neighbors to use if knn_method='KNN'.

    Returns
    -------
    adata : anndata.AnnData
        Modified AnnData object with the denoised expression matrix (adata.X).
    optimal_solution : list[float] or torch.Tensor
        Learned diffusion parameters for spatial (and optionally gene) graphs.
    loss_list : list[float]
        List of loss values at each iteration (only in BFGS mode).
    constant_value : list
        Cached constants used in the denoising process, including graph Laplacian eigenvectors/values.

    Notes
    -----
    - BFGS mode performs automatic parameter optimization based on the input data.
    - Manual mode uses predefined or user-specified diffusion coefficients.
    - If the calculated spatial diffusion coefficient is exactly 0, a warning is triggered and it is forcibly set to
     0.1 to prevent degeneration.
    - If top_genes_train is used, only top genes are used during optimization, and the full matrix is reconstructed
     afterward.
    - The parameter 'in_tissue' must be accurate to ensure proper background/in-tissue masking.
    """

    #  Check input parameters
    assert method in ['BFGS', 'manual'], 'method should be BFGS or manual'
    if data_type in ['Stereoseq', 'VisiumHD']:
        if bin_size is None:
            raise ValueError(f'bin_size should be given for data type {data_type}')
        rad_cutoff = 1 if rad_cutoff is None else rad_cutoff
    elif data_type in ['Visium', 'ST']:
        rad_cutoff = 2 if rad_cutoff is None else rad_cutoff
    elif data_type == 'Slideseq':
        rad_cutoff = 5 if rad_cutoff is None else rad_cutoff

    if isinstance(spatial_key, str):
        if spatial_key not in adata.obsm.keys():
            raise ValueError(f'{spatial_key} not found in adata.obsm')
        if spatial_key !=  'spatial':
            adata.obsm['spatial'] = adata.obsm[spatial_key]
    elif isinstance(spatial_key, list):
        sp_array = adata.obs.loc[:, spatial_key].values
        adata.obsm['spatial'] = sp_array

    # Obtain spot network using spatial information
    spotnet = obtain_spotnet(adata,
                             knn_method=knn_method,
                             rad_cutoff=rad_cutoff,
                             data_type=data_type,
                             bin_size=bin_size,
                             scale=scale,
                             k_cutoff=k_cutoff)
    spot_eigvecs, spot_eigvals = GraphFourierTransform(spotnet, n_GFT=adata.shape[0])
    spot_eigvals = spot_eigvals * (spot_eigvals > 0)

    # Train n all genes or top genes
    if top_genes_train is not None:
        adata_raw = adata.copy()
        adata.var['total_counts'] = adata.X.sum(axis=0).reshape(-1, 1)
        adata.var = adata.var.sort_values(by='total_counts', ascending=False)
        adata = adata[:, adata.var_names[:top_genes_train]]

    # Obtain out-of-tissue mask
    if 'in_tissue' not in adata.obs.keys():
        print('Warning: in_tissue is not included in adata.obs, assuming all spots are in tissue')
        adata.obs['in_tissue'] = np.ones(adata.shape[0])
    A_out = 1 - adata.obs['in_tissue'].values
    A_out = np.repeat(A_out[:, np.newaxis], adata.shape[1], axis=1)
    out_sum = A_out.sum()
    if type(adata.X) == np.ndarray or isinstance(adata.X, anndata._core.views.ArrayView):
        x = adata.X.copy()
    else:
        x = adata.X.copy().todense()

    if gene_network:
        genenet = obtain_genenet(adata, cut=0.2)
        gene_eigvecs, gene_eigvals = GraphFourierTransform(genenet, n_GFT=adata.shape[1])
        gene_eigvals = gene_eigvals * (gene_eigvals > 0)
        constant_value = [A_out, x, spot_eigvecs, spot_eigvals, gene_eigvecs, gene_eigvals]
    else:
        constant_value = [A_out, x, spot_eigvecs, spot_eigvals]

    loss_list = []
    if method == 'BFGS':
        constant_value = [torch.from_numpy(item).float().to(device) for item in constant_value]
        variable = torch.tensor([0.0] * (3 if gene_network else 2), requires_grad=True, device=device)
        lower_bound = torch.tensor([0.0] * (3 if gene_network else 2), device=device)
        upper_bound = torch.tensor([0.999 / spot_eigvals.max()] + [np.inf] * (2 if gene_network else 1), device=device)
        optimizer = LBFGS([variable], lr=lr, max_iter=max_iter)

        if alpha != 0:
            in_sum = (1 - A_out).sum()
            exp_in = np.mean(adata[adata.obs['in_tissue'] == 1].X)
            exp_out = np.mean(adata[adata.obs['in_tissue'] == 0].X)
            fac = exp_out / exp_in

            def closure():
                optimizer.zero_grad()
                y_pred = denoise_function(variable, constant_value, k)
                loss = ((1 - alpha) * torch.sum((constant_value[0] * y_pred) ** 2) / out_sum
                        + alpha * torch.sum(
                            ((1 - constant_value[0]) * (y_pred - constant_value[1])) ** 2) / in_sum * fac)
                loss.backward()
                return loss
        else:
            def closure():
                optimizer.zero_grad()
                y_pred = denoise_function(variable, constant_value, k)
                loss = torch.sum((constant_value[0] * y_pred) ** 2) / out_sum
                loss.backward()
                return loss

        loss = closure()
        loss_max = loss.data.clone()
        with tqdm(total=max_iterations, desc='Training', ncols=100) as pbar:
            for epoch in range(max_iterations):
                pbar.set_postfix({'Loss': loss.data})
                pbar.update(1)
                optimal_solution = variable.data.clone()
                loss_list.append(loss.data)
                optimizer.step(closure)
                variable.data = torch.clamp(variable.data, lower_bound, upper_bound)

                loss = closure()
                if torch.isnan(loss) or loss <= loss_max * 1e-4 or (epoch > 5 and loss >= loss_list[-1] * (1 - 1e-4)):
                    break

        if optimal_solution[0] == torch.tensor(0, device=device):
            print("""The calculated value of diffusion coefficient is 0, which has been forcibly changed to 0.1. Please 
            check the data or redefine the in_tissue of the tissue boundary spots. The latter can try increasing the 
            in_rate parameter in downsampling function.""")
            optimal_solution = torch.tensor([0.1, 0.0], device=device)

        if top_genes_train is not None:
            adata = adata_raw.copy()
            if type(adata.X) == np.ndarray or isinstance(adata.X, anndata._core.views.ArrayView):
                x = adata.X.copy()
            else:
                x = adata.X.copy().todense()
            A_out = 1 - adata.obs['in_tissue'].values
            A_out = np.repeat(A_out[:, np.newaxis], adata.shape[1], axis=1)

            if gene_network:
                genenet = obtain_genenet(adata, cut=0)
                gene_eigvecs, gene_eigvals = GraphFourierTransform(genenet, n_GFT=adata.shape[1])
                gene_eigvals = gene_eigvals * (gene_eigvals > 0)
                constant_value = [A_out, x, spot_eigvecs, spot_eigvals, gene_eigvecs, gene_eigvals]
            else:
                constant_value = [A_out, x, spot_eigvecs, spot_eigvals]
            constant_value = [torch.from_numpy(item).float().to(device) if not isinstance(item, torch.Tensor)
                              else item for item in constant_value]

        y = denoise_function(optimal_solution, constant_value, k)
        if normlization:
            y = y / y.sum(axis=0) * constant_value[1].sum(axis=0)
        adata.X = y.cpu().detach().numpy()

    elif method == 'manual':
        if variable is None:
            if data_type == 'Visium':
                if len(constant_value) == 4:
                    optimal_solution = [0.3, 0.1]
                else:
                    optimal_solution = [0.3, 0.1, 0.1]
            elif data_type == 'Stereoseq':
                if len(constant_value) == 4:
                    optimal_solution = [0.05, 0.05]
                else:
                    optimal_solution = [0.05, 0.1, 0.05]
            elif data_type == 'VisiumHD':
                if len(constant_value) == 4:
                    optimal_solution = [0.02, 0.005]
                else:
                    optimal_solution = [0.02, 0.005, 0.005]
            elif data_type == 'Slideseq':
                if len(constant_value) == 4:
                    optimal_solution = [0.1, 0.05]
                else:
                    optimal_solution = [0.1, 0.05, 0.05]
        else:
            optimal_solution = variable
        if optimal_solution[0] >= 1 / spot_eigvals.max():
            print(f"Optimal solution (diffusion coefficient) fails to meet the the subgraph constraint!"
                  f"Current graph value: {optimal_solution[0]}, modify it to: {0.999 / spot_eigvals.max():.6f}")
            optimal_solution[0] = 0.999 / spot_eigvals.max()
        adata.X = denoise_function_numpy(optimal_solution, constant_value, k)
        if normlization:
            adata.X = adata.X / np.array(adata.X.sum(axis=0)) * np.array(x.sum(axis=0))
        adata.X = csr_matrix(adata.X)

    return adata, optimal_solution, loss_list, constant_value


def run_diffgsp_subgraph(adata,
                         k=2,
                         variable=None,
                         normlization=False,
                         gene_network=False,
                         partition=[1, 1],
                         spatial_key='spatial',
                         array_key=['array_row', 'array_col'],
                         rad_cutoff=None,
                         data_type=None,
                         bin_size=None,
                         scale=None,
                         dic_storage=True,
                         knn_method='Radius',
                         k_cutoff=18,
                         bgzero=False,
                         bgnone=False):
    """
    This function performs graph-based denoising on spatial transcriptomics data by applying reverse diffusion
    locally on subgraphs partitioned across spatial dimensions. It enhances expression data quality while
    preserving local spatial structure and reduces memory usage by processing subsets.

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial transcriptomics dataset containing expression and spatial coordinates.
    k : int, optional (default=2)
        Number of graph filter iterations or Fourier modes.
    variable : list[float], optional (default=None)
        Predefined diffusion coefficients [beta_spatial, lambda1, (lambda2)] depending on whether gene_network is used.
    normlization : bool, optional (default=False)
        Whether to normalize the denoised matrix to match original total expression.
    gene_network : bool, optional (default=False)
        Whether to include the gene graph structure for gene-wise smoothing.
    partition : list[int], optional (default=[1, 1])
        Number of spatial partitions along [x, y] axes for subgraph processing.
    spatial_key : list[str], optional (default='spatial')
        Keys in `adata.obs` or `'spatial'` in `adata.obsm` used as spatial coordinates.
    rad_cutoff : float, optional (default=None)
        Radius cutoff for spatial graph construction. Auto-assigned if not specified.
    data_type : str
        Spatial platform type: one of ['Visium', 'Stereoseq', 'Slideseq', 'VisiumHD'].
    bin_size : float, optional (default=None)
        Required for Stereoseq or VisiumHD to determine spatial scaling.
    scale : float, optional (default=None)
        Scaling factor for adjusting spot distances when constructing graphs.
    dic_storage : bool, optional (default=True)
        Whether to cache and reuse Laplacian eigendecompositions for identical subgraph sizes.
    knn_method : str, optional (default='Radius')
        Graph construction method: 'Radius' or 'KNN'.
    k_cutoff : int, optional (default=18)
        Max number of neighbors if knn_method='KNN'.
    bgzero : bool, optional (default=False)
        If True, sets the expression of background spots (`in_tissue == 0`) to zero in the final output.
    bgnone : bool, optional (default=False)
        If True, excludes background spots (`in_tissue == 0`) from the final output.

    Returns
    -------
    adata_result : anndata.AnnData
        A new AnnData object containing the denoised expression matrix across all subgraphs, aligned to original spot order.

    Notes
    -----
    - Subgraphs are created by spatially partitioning the tissue into rectangular blocks.
    - Useful for large ST datasets by lowering memory footprint and improving scalability.
    - If `dic_storage` is False, eigendecomposition results will not be cached and reused.
    - If `variable` is not provided, platform-specific default values are used based on `data_type`.
    - Either `bgzero` or `bgnone` can be used to handle background noise in final output.
    - Final matrix is converted to `csr_matrix` format for efficient storage and downstream compatibility.
    """
    if data_type in ['Stereoseq', 'VisiumHD']:
        assert bin_size is not None, 'bin_size should be given'

    if 'x' not in adata.obs or 'y' not in adata.obs:
        adata.obs['x'] = np.array(adata.obsm['spatial'][:, 0])
        adata.obs['y'] = np.array(adata.obsm['spatial'][:, 1])

    if isinstance(spatial_key, str):
        if spatial_key not in adata.obsm.keys():
            raise ValueError(f'{spatial_key} not found in adata.obsm')
        if spatial_key !=  'spatial':
            adata.obsm['spatial'] = adata.obsm[spatial_key]
    elif isinstance(spatial_key, list):
        sp_array = adata.obs.loc[:, spatial_key].values
        adata.obsm['spatial'] = sp_array

    sp_df = adata.obs.loc[:, array_key].values
    batch_x_coor = np.percentile(sp_df[:, 0], np.linspace(0, 100, partition[0] + 1))
    batch_y_coor = np.percentile(sp_df[:, 1], np.linspace(0, 100, partition[1] + 1))

    gene_eigvecs, gene_eigvals = None, None
    if gene_network:
        genenet = obtain_genenet(adata, cut=0.2)
        gene_eigvecs, gene_eigvals = GraphFourierTransform(genenet, n_GFT=adata.shape[1])
        gene_eigvals = gene_eigvals * (gene_eigvals > 0)

    if variable is None:
        if data_type == 'Visium':
            if not gene_network:
                optimal_solution = [0.3, 0.1]
            else:
                optimal_solution = [0.3, 0.1, 0.1]
        elif data_type == 'Stereoseq':
            if not gene_network:
                optimal_solution = [0.05, 0.05]
            else:
                optimal_solution = [0.05, 0.1, 0.05]
        elif data_type == 'VisiumHD':
            if not gene_network:
                optimal_solution = [0.02, 0.005]
            else:
                optimal_solution = [0.02, 0.005, 0.005]
        elif data_type == 'Slideseq':
            if not gene_network:
                optimal_solution = [0.1, 0.05]
            else:
                optimal_solution = [0.1, 0.05, 0.05]
    else:
        optimal_solution = variable
    spot_eigvecs_dic = {}
    adata_result = None

    total_iterations = partition[0] * partition[1]
    with tqdm(total=total_iterations, desc='Test', ncols=100) as pbar:
        count = 0
        for it_x in range(partition[0]):
            for it_y in range(partition[1]):
                min_x = batch_x_coor[it_x]
                max_x = batch_x_coor[it_x + 1]
                min_y = batch_y_coor[it_y]
                max_y = batch_y_coor[it_y + 1]
                temp_adata = adata.copy()
                temp_adata = temp_adata[temp_adata.obs[array_key[0]].map(lambda x: min_x <= x < max_x
                if it_x != partition[0] - 1 else min_x <= x <= max_x)]
                temp_adata = temp_adata[temp_adata.obs[array_key[1]].map(lambda y: min_y <= y < max_y
                if it_y != partition[1] - 1 else min_y <= y <= max_y)]
                if temp_adata.shape[0] == 0:
                    if count == 0:
                        adata_result = temp_adata
                        count += 1
                    continue
                names_order = temp_adata.obs.sort_values(['x', 'y']).index
                temp_adata = temp_adata[names_order, :]

                x_unique = len(np.unique(temp_adata.obs['x']))
                y_unique = len(np.unique(temp_adata.obs['y']))
                if (x_unique, y_unique, temp_adata.shape[0]) not in spot_eigvecs_dic.keys():
                    spotnet = obtain_spotnet(temp_adata, knn_method=knn_method, data_type=data_type,
                                             bin_size=bin_size, scale=scale, k_cutoff=k_cutoff)
                    spot_eigvecs, spot_eigvals = GraphFourierTransform(spotnet, n_GFT=temp_adata.shape[0])
                    spot_eigvals = spot_eigvals * (spot_eigvals > 0)

                    spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])] = [spot_eigvecs, spot_eigvals]
                    if optimal_solution[0] >= 1 / spot_eigvals.max():
                        print(f"Optimal solution (diffusion coefficient) fails to meet the the subgraph constraint!"
                              f"Current subgraph value: {optimal_solution[0]}, modify it to: {0.999 / spot_eigvals.max():.6f}")
                        optimal_solution[0] = 0.999 / spot_eigvals.max()
                if type(temp_adata.X) == np.ndarray or isinstance(temp_adata.X, anndata._core.views.ArrayView):
                    x = temp_adata.X.copy()
                else:
                    x = temp_adata.X.copy().todense()

                if len(variable) == 3:
                    constant_value = [None, x, spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])][0],
                                      spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])][1], gene_eigvecs, gene_eigvals]
                else:
                    constant_value = [None, x, spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])][0],
                                      spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])][1]]
                if dic_storage is False:
                    spot_eigvecs_dic.pop((x_unique, y_unique, temp_adata.shape[0]), None)
                temp_adata.X = csr_matrix(denoise_function_numpy(optimal_solution, constant_value, k))
                if count == 0:
                    adata_result = temp_adata
                else:
                    adata_result = anndata.concat([adata_result, temp_adata], axis=0, join='outer')
                count += 1
                pbar.update(1)
                pbar.set_postfix({'Finish': f'{count}/{total_iterations}'})
    adata_result = adata_result[adata.obs_names, :]
    if bgnone:
        adata_result = adata_result[adata_result.obs['in_tissue'] == 1]
    if bgzero:
        adata_result.X[adata_result.obs['in_tissue'] == 0, :] = 0
    if normlization:
        adata_result.X = adata_result.X / np.array(adata_result.X.sum(axis=0)) * np.array(adata.X.sum(axis=0))
    if not isinstance(adata_result.X, csr_matrix):
        adata_result.X = csr_matrix(adata_result.X, dtype=np.float64)

    return adata_result


def fill_adata(adata, bin_size=None):
    x_min, x_max = adata.obsm['spatial'][:, 0].min(), adata.obsm['spatial'][:, 0].max()
    y_min, y_max = adata.obsm['spatial'][:, 1].min(), adata.obsm['spatial'][:, 1].max()
    if bin_size is None:
        bin_size = adata.uns['bin_size']
    x_grid = np.arange(x_min, x_max + bin_size, bin_size).astype(int)
    y_grid = np.arange(y_min, y_max + bin_size, bin_size).astype(int)
    all_coords = np.array(list(itertools.product(x_grid, y_grid)))
    all_coords_dype = all_coords.dtype
    existing_coords = adata.obsm['spatial']

    all_coords = all_coords.view([('', all_coords.dtype)] * all_coords.shape[1])
    existing_coords = existing_coords.view([('', existing_coords.dtype)] * existing_coords.shape[1])
    missing_coords = np.setdiff1d(all_coords, existing_coords).view(all_coords_dype).reshape(-1, 2)

    uns = adata.uns.copy()
    adata_new = anndata.AnnData(X=csr_matrix((len(missing_coords), adata.X.shape[1])),
                                obs={'orig.ident': 'fill', 'x': missing_coords[:, 0],
                                     'y': missing_coords[:, 1], 'in_tissue': 0},
                                var=adata.var.copy(),
                                obsm={'spatial': missing_coords})
    adata_new.obs_names = [f'fill_{i}' for i in range(len(missing_coords))]
    combined_obs_names = list(adata.obs_names.tolist() + adata_new.obs_names.tolist())
    adata = adata.concatenate(adata_new)
    adata.obs_names = combined_obs_names
    adata.uns = uns
    return adata


def downsampling(adata, multiple=[5, 5], in_rate=0.5):
    spatial_coords = adata.obsm['spatial']
    length = spatial_coords.max(axis=0) - spatial_coords.min(axis=0)
    n_x = int(np.sqrt(adata.shape[0] * length[0] / length[1]) / multiple[0])
    n_y = int(np.sqrt(adata.shape[0] * length[1] / length[0]) / multiple[1])
    print('Number of downsampled spots:', n_x * n_y)
    x_range = np.percentile(spatial_coords[:, 0], np.linspace(0, 100, n_x + 1))
    y_range = np.percentile(spatial_coords[:, 1], np.linspace(0, 100, n_y + 1))

    new_expression_matrix = np.zeros((n_x * n_y, adata.shape[1]))
    new_coordinate_matrix = np.zeros((n_x * n_y, 2))
    new_obs_matrix = np.zeros(n_x * n_y)

    x_grid = np.digitize(spatial_coords[:, 0], x_range) - 1
    y_grid = np.digitize(spatial_coords[:, 1], y_range) - 1
    for i in range(n_x):
        for j in range(n_y):
            mask = (x_grid == i) & (y_grid == j)
            indices = np.where(mask)[0]
            if indices.size > 0:
                new_expression_matrix[i * n_y + j, :] = np.sum(adata.X[indices, :], axis=0)
                new_coordinate_matrix[i * n_y + j, :] = np.mean(spatial_coords[indices, :], axis=0)
                if np.sum(adata.obs['in_tissue'][indices]) > indices.size * in_rate:
                    new_obs_matrix[i * n_y + j] = 1
    adata_down = sc.AnnData(X=new_expression_matrix)
    adata_down.obsm['spatial'] = new_coordinate_matrix
    adata_down.obs['in_tissue'] = new_obs_matrix
    adata_down.var = adata.var.copy()
    return adata_down


def denoise_function_numpy(variables, constant, k):
    if len(constant) == 4:
        [_, x, spot_eigvecs, spot_eigvals] = constant
        filter_diffusion = np.diag(1 / (1 - variables[0] * spot_eigvals))  # a
        filter_spot = np.diag(1 / (1 + variables[1] * spot_eigvals))  # c
        y = spot_eigvecs @ filter_diffusion ** k @ filter_spot ** k @ spot_eigvecs.T @ x
    else:
        [_, x, spot_eigvecs, spot_eigvals, gene_eigvecs, gene_eigvals] = constant
        filter_diffusion = np.diag(1 / (1 - variables[0] * spot_eigvals))  # a
        filter_gene = np.diag(1 / (1 + variables[1] * gene_eigvals))  # b
        filter_spot = np.diag(1 / (1 + variables[2] * spot_eigvals))  # c
        y = (spot_eigvecs @ filter_diffusion ** k @ filter_spot ** k @ spot_eigvecs.T @ x @ gene_eigvecs @ filter_gene @
             gene_eigvecs.T)

    y = np.multiply(y, y >= 0)

    return y


def denoise_function(variables, constant, k):
    if len(constant) == 4:
        [_, x, spot_eigvecs, spot_eigvals] = constant
        filter_diffusion = torch.diag(1 / (1 - variables[0] * spot_eigvals))  # a
        filter_spot = torch.diag(1 / (1 + variables[1] * spot_eigvals))  # c
        y = spot_eigvecs @ filter_diffusion ** k @ filter_spot ** k @ spot_eigvecs.T @ x
    else:
        [_, x, spot_eigvecs, spot_eigvals, gene_eigvecs, gene_eigvals] = constant
        # filter_diffusion = torch.diag(1 + variables[0] * spot_eigvals)  # a
        filter_diffusion = torch.diag(1 / (1 - variables[0] * spot_eigvals))  # a
        filter_gene = torch.diag(1 / (1 + variables[1] * gene_eigvals))  # b
        filter_spot = torch.diag(1 / (1 + variables[2] * spot_eigvals))  # c
        # filter_spot = torch.diag(torch.exp(- variables[2] * spot_eigvals))  # c
        y = (spot_eigvecs @ filter_diffusion ** k @ filter_spot ** k @ spot_eigvecs.T @ x @ gene_eigvecs @ filter_gene @
             gene_eigvecs.T)

    y = y * (y >= 0)

    return y


def GraphFourierTransform(net, n_GFT):
    lap_mtx = _get_lap_mtx(net).toarray()
    eigvals, eigvecs = ss.linalg.eigsh(lap_mtx.astype('double'),
                                       k=round(n_GFT),
                                       which='SM',
                                       v0=[1 / np.sqrt(net.shape[0])] * net.shape[0])
    return eigvecs, eigvals


def _create_degree_mtx(diag):
    diag = np.array(diag)
    diag = diag.flatten()
    row_index = list(range(diag.size))
    col_index = row_index
    sparse_mtx = ss.coo_matrix((diag, (row_index, col_index)), shape=(diag.size, diag.size))
    return sparse_mtx


def _get_lap_mtx(net, normalization=False):
    diag = net.sum(axis=1)
    adj_mtx = ss.coo_matrix(net)
    if not normalization:
        deg_mtx = _create_degree_mtx(diag)
        lap_mtx = deg_mtx - adj_mtx
    else:
        diag = np.array(diag) ** (-0.5)
        deg_mtx = _create_degree_mtx(diag)
        lap_mtx = ss.identity(deg_mtx.shape[0]) - deg_mtx @ adj_mtx @ deg_mtx

    return lap_mtx


def obtain_spotnet(adata, rad_cutoff=None, k_cutoff=18, knn_method='Radius', data_type=None, bin_size=None, scale=None):
    if scale is None:
        if data_type == 'Visium':
            scale = 150
        elif data_type in ['Stereoseq', 'VisiumHD']:
            scale = 0.75 * bin_size
        elif data_type == 'Slideseq':
            scale = None
        elif data_type == 'ST':
            scale = 300

    coor = np.array(adata.obsm['spatial'])
    delta_x = coor[:, 0].max() - coor[:, 0].min()
    delta_y = coor[:, 1].max() - coor[:, 1].min()
    if rad_cutoff is None:
        rad_cutoff = (delta_x + delta_y) / 2 / 2
    else:
        rad_cutoff = (delta_x + delta_y) / 2 / rad_cutoff
    # nbrs = NearestNeighbors(radius=rad_cutoff) if knn_method == 'Radius' else NearestNeighbors(n_neighbors=k_cutoff + 1)
    nbrs = NearestNeighbors(radius=rad_cutoff, n_jobs=-1) if knn_method == 'Radius' else NearestNeighbors(
        n_neighbors=k_cutoff + 1, n_jobs=-1)
    nbrs.fit(coor)
    if knn_method == 'Radius':
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    elif knn_method == 'KNN':
        distances, indices = nbrs.kneighbors(coor)

    cell_indices = np.repeat(np.arange(len(coor)), [len(idx) for idx in indices])
    neighbor_indices = np.concatenate(indices)
    neighbor_distances = np.concatenate(distances)
    KNN_df = pd.DataFrame({'Cell1': cell_indices, 'Cell2': neighbor_indices, 'Distance': neighbor_distances})
    KNN_df = KNN_df[KNN_df['Distance'] > 0]

    if scale is not None:
        KNN_df['Distance'] = KNN_df['Distance'] / KNN_df['Distance'].min() * scale
    KNN_df['InverseDistance'] = 1 / KNN_df['Distance']
    spot_net = csr_matrix((KNN_df['InverseDistance'], (KNN_df['Cell1'], KNN_df['Cell2'])))
    spot_net = (spot_net + spot_net.transpose()) / 2

    return spot_net


def obtain_genenet(adata, cut=0.2, n_neighbor=15):
    adata = adata[adata.obs['in_tissue'] == 1, :]
    if type(adata.X) == np.ndarray or isinstance(adata.X, anndata._core.views.ArrayView):
        X = adata.X.copy()
    else:
        X = adata.X.copy().todense()
    df = pd.DataFrame(X)
    if adata.shape[0] > 4000:
        spot_choose = np.random.choice(list(range(adata.shape[0])), 4000, replace=False)
        df = df.loc[spot_choose, :]

    # nbrs = NearestNeighbors(n_neighbors=n_neighbor, metric='cosine').fit(df.transpose())
    # distances, indices = nbrs.kneighbors(df.transpose())
    # KNN_list = []
    # for it in range(indices.shape[0]):
    #     KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    # KNN_df = pd.concat(KNN_list)
    # KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    # KNN_df = KNN_df.loc[KNN_df['Distance'] > 0, :]
    # spot_net = csr_matrix((KNN_df['Distance'], (KNN_df['Cell1'], KNN_df['Cell2'])),
    #                       shape=(df.shape[1], df.shape[1]))
    # result = (spot_net + spot_net.transpose()) / 2
    # result = result.toarray()
    # result[result > 0] = 1

    genenet = df.corr(method='pearson').values
    genenet -= np.diag([1] * genenet.shape[0])
    # top_n_indices = np.argpartition(-genenet, n_neighbor, axis=1)[:, n_neighbor]
    result = genenet * (genenet >= cut)  # * (genenet >= genenet[:, top_n_indices])
    result = (result + result.T) / 2
    ave_connectivity = np.count_nonzero(result) / result.shape[0]
    print('Average connectivity of gene network:', ave_connectivity)
    print('Maximum connectivity of gene network:', (result != 0).sum(axis=0).max())
    return result


def prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=3, max_cells=None):
    adata = adata[adata.obs['in_tissue'] == 1, :]
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[1], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)
    return adata.var_names


def prefilter_spots(adata, min_counts=10, max_counts=None, min_genes=None, max_genes=None):
    if min_genes is None and min_counts is None and max_genes is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_cells, max_counts or max_cells.')
    id_tmp = np.asarray([True] * adata.shape[0], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_cells(adata.X, min_genes=min_genes)[0]) if min_genes is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_cells(adata.X, max_genes=max_genes)[0]) if max_genes is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_cells(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_cells(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_obs(id_tmp)


def select_svgs(adata, svg_method='gft', n_top=2000):
    """
    Select spatially variable genes using six methods, including 'gft', 'gft_top',
    'seurat', 'seurat_v3', 'cell_ranger' and 'mix'.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        svg_method: str, optional
            Methods for selecting spatially variable genes. Teh default is 'gft_top'.
        n_top: int, optional
            Number of spatially variable genes selected. The default is 3000.
        csvg: float, optional
            Smoothing coefficient of GFT for noise reduction. The default is 0.0001.
        smoothing: bool, optional
            Determine whether it is smooth for noise reduction. The default is True.

    Returns:
        adata: anndata
            AnnData object of scanpy package after choosing svgs and smoothing.
        adata_raw: anndata
            AnnData object of scanpy package before choosing svgs and smoothing.
    """
    svg_list = []
    adata = adata[adata.obs['in_tissue'] == 1, :]
    assert svg_method in ['gft', 'gft_top', 'seurat', 'seurat_v3', 'cell_ranger', 'mix']
    if svg_method == 'seurat_v3':
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top)
        adata = adata[:, adata.var['highly_variable']]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        svg_list = adata.var['highly_variable'].index
    elif svg_method == 'mix':
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=int(n_top / 2))
        seuv3_list = adata.var_names[adata.var['highly_variable']]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        gene_df = rank_gene_smooth(adata,
                                   spatial_info=['array_row', 'array_col'],
                                   ratio_low_freq=1,
                                   ratio_high_freq=1,
                                   ratio_neighbors=1,
                                   filter_peaks=True,
                                   S=6)
        svg_list = gene_df.index[:(n_top - len(seuv3_list))]
        merged_gene_list = np.union1d(seuv3_list, svg_list)
        svg_list = merged_gene_list
    else:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        if svg_method == 'seurat':
            sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top)
            adata = adata[:, adata.var['highly_variable']]
            svg_list = adata.var['highly_variable'].index
        elif svg_method == 'cell_ranger':
            sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes=n_top)
            adata = adata[:, adata.var['highly_variable']]
            svg_list = adata.var['highly_variable'].index
        elif svg_method == 'gft' or svg_method == 'gft_top':
            gene_df = rank_gene_smooth(adata,
                                       spatial_info='spatial',
                                       ratio_low_freq=1,
                                       ratio_high_freq=1,
                                       ratio_neighbors=1,
                                       filter_peaks=True,
                                       S=6)
            if svg_method == 'gft':
                svg_list = gene_df[gene_df.cutoff_gft_score][gene_df.qvalue < 0.05].index.tolist()
            elif svg_method == 'gft_top':
                svg_list = gene_df.index[:n_top].tolist()
    return svg_list


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
