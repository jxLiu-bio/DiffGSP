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
from DiffGSP.svg_select_spagft import rank_gene_smooth
from DiffGSP.utils import _get_lap_mtx, scale_exp_calculation, _normalize_library_size
from scipy.sparse import csr_matrix, diags
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')


def run_diffgsp(adata,
                gene_network=False,
                k=3,
                alpha=0,
                device='cuda:0',
                max_iterations=100,
                lr=0.05,
                spatial_key='spatial',
                rad_cutoff=None,
                max_iter=10,
                method='manual',
                variable=None,
                normalization=False,
                data_type=None,
                top_genes_train=None,
                knn_method='Radius',
                k_cutoff=18,
                c=0.0005,
                sigma=2,
                inv_cutoff='auto',
                inv_knee_upbound=2):
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
    normalization : bool, optional (default=False)
        Whether to normalize the denoised matrix to preserve the original total expression per gene.
    data_type : str, optional (default=None)
        Type of spatial transcriptomics platform (e.g., 'Visium', 'Stereoseq', 'VisiumHD', 'Slideseq').
    top_genes_train : int, optional (default=None)
        Number of top-expressed genes used during parameter training (full matrix restored after optimization).
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
    - If the calculated spatial diffusion coefficient is exactly 0, a warning is triggered, and it is forcibly set to
     0.1 to prevent degeneration.
    - If top_genes_train is used, only top genes are used during optimization, and the full matrix is reconstructed
     afterward.
    - The parameter 'in_tissue' must be accurate to ensure proper background/in-tissue masking.
    """

    #  Check input parameters
    assert method in ['BFGS', 'manual'], 'method should be BFGS or manual'
    if not torch.cuda.is_available():
        if device != 'cuda:0':
            raise NotImplementedError('CUDA is not available')
    if alpha != 0:
        if alpha > 1:
            raise ValueError('alpha should be less than 1')
    if data_type in ['Stereoseq', 'VisiumHD']:
        rad_cutoff = 0.1 if rad_cutoff is None else rad_cutoff
    elif data_type in ['Visium', 'ST']:
        rad_cutoff = 0.1 if rad_cutoff is None else rad_cutoff
    elif data_type == 'Slideseq':
        rad_cutoff = 0.1 if rad_cutoff is None else rad_cutoff
    else:
        rad_cutoff = 0.1 if rad_cutoff is None else rad_cutoff

    if isinstance(spatial_key, str):
        if spatial_key not in adata.obsm.keys():
            raise ValueError(f'{spatial_key} not found in adata.obsm')
        if spatial_key != 'spatial':
            adata.obsm['spatial'] = adata.obsm[spatial_key]
    elif isinstance(spatial_key, list):
        sp_array = adata.obs.loc[:, spatial_key].values
        adata.obsm['spatial'] = sp_array

    # Obtain spot network using spatial information
    spotnet = obtain_spotnet(adata,
                             knn_method=knn_method,
                             rad_cutoff=rad_cutoff,
                             k_cutoff=k_cutoff,
                             sigma=sigma)
    spot_eigvecs, spot_eigvals = spectral_decomposition(spotnet,
                                                        n_GFT=int(adata.shape[0]),
                                                        device=device)
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
    if not ss.issparse(adata.X):
        x = adata.X.copy()
    else:
        x = adata.X.toarray()
    # extract the max library_size
    max_library_size = np.percentile(x.sum(axis=1), 99)

    constant_value = [A_out, x, spot_eigvecs, spot_eigvals]

    loss_list = []
    if method == 'BFGS':
        constant_value = [torch.from_numpy(item).float().to(device) for item in constant_value]
        if data_type == 'Visium':
            low_value = 0.01
            high_value = 0.4
        elif data_type == 'VisiumHD':
            low_value = 0.01
            high_value = 0.4
        elif data_type == 'Stereoseq':
            low_value = 0.01
            high_value = 0.4
        elif data_type == 'Slideseq':
            low_value = 0.01
            high_value = 0.4
        else:
            low_value = 0.02
            high_value = 0.4
        low_value = low_value
        high_value = high_value
        variable = torch.tensor([low_value, 0.001 * high_value],
                                requires_grad=True,
                                device=device,
                                dtype=torch.float)
        lower_bound = torch.tensor([low_value, 0.001 * high_value],
                                   device=device,
                                   dtype=torch.float)
        upper_bound = torch.tensor([high_value, 0.5 * high_value],
                                   device=device,
                                   dtype=torch.float)
        optimizer = LBFGS([variable],
                          lr=lr,
                          max_iter=max_iter)
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
                if torch.isnan(loss) or loss <= loss_max * 1e-3 or (epoch > 5 and loss >= loss_list[-1] * (1 - 1e-2)):
                    break
        optimal_solution[0] = min(optimal_solution[0], 1 / spot_eigvals.max())
        if optimal_solution[0] == torch.tensor(0, device=device):
            print("""The calculated value of diffusion coefficient is 0, which has been forcibly changed to 0.1. Please 
            check the data or redefine the in_tissue of the tissue boundary spots. The latter can try increasing the 
            in_rate parameter in downsampling function.""")
            optimal_solution = torch.tensor([low_value, 0.0], device=device)

        if optimal_solution[1] >= optimal_solution[0]:
            optimal_solution[1] = 0.5 * optimal_solution[0]

        if top_genes_train is not None:
            adata = adata_raw.copy()
            if not ss.issparse(adata.X):
                x = adata.X.copy()
            else:
                x = adata.X.toarray()
            A_out = 1 - adata.obs['in_tissue'].values
            A_out = np.repeat(A_out[:, np.newaxis], adata.shape[1], axis=1)
            constant_value = [A_out, x, spot_eigvecs, spot_eigvals]
            constant_value = [torch.from_numpy(item).float().to(device) if not isinstance(item, torch.Tensor)
                              else item for item in constant_value]

        y = denoise_function(optimal_solution, constant_value, k)
        y = y.cpu().detach().numpy()
        constant_value = [item.cpu().numpy() if isinstance(item, torch.Tensor)
                              else item for item in constant_value]
        optimal_solution = [item.cpu().numpy() if isinstance(item, torch.Tensor)
                              else item for item in optimal_solution]
    # constant_value = [torch.from_numpy(item).float().to(device) if not isinstance(item, torch.Tensor)
    #                   else item for item in constant_value]
    elif method == 'manual':
        if variable is None:
            if data_type == 'Visium':
                if len(constant_value) == 4:
                    optimal_solution = [0.15, 0.05]
                else:
                    optimal_solution = [0.15, 0.05, 0.1]
            elif data_type == 'Stereoseq':
                if len(constant_value) == 4:
                    optimal_solution = [0.1, 0.01]
                else:
                    optimal_solution = [0.1, 0.01, 0.05]
            elif data_type == 'VisiumHD':
                if len(constant_value) == 4:
                    optimal_solution = [0.1, 0.05]
                else:
                    optimal_solution = [0.1, 0.01, 0.005]
            elif data_type == 'Slideseq':
                if len(constant_value) == 4:
                    optimal_solution = [0.1, 0.01]
                else:
                    optimal_solution = [0.1, 0.01, 0.005]
            else:
                if len(constant_value) == 4:
                    optimal_solution = [0.1, 0.05]
                else:
                    optimal_solution = [0.1, 0.01, 0.01]
        else:
            if len(constant_value) == 4:
                optimal_solution = variable[:2]
            else:
                optimal_solution = variable[:3]

        if optimal_solution[0] > 1 or optimal_solution[0] < 0:
            raise ValueError('The value of diffusion coefficient must be between 0 and 1.')
        if optimal_solution[1] > 1 or optimal_solution[1] < 0:
            raise ValueError('The value of low-pass filtering coefficient must be between 0 and 1.')

        # adjust in 20260609
        # optimal_solution[0] = optimal_solution[0] * 1 / spot_eigvals.max()
        # optimal_solution[1] = optimal_solution[1] * 1 / spot_eigvals.max()
        y = denoise_function_numpy(optimal_solution, constant_value, k)

    if inv_cutoff is not None and inv_cutoff != 'auto':
        if not _check_over_diffusion(x, y, cutoff=inv_cutoff):
            for para in np.linspace(optimal_solution[0],
                                    0.9 * 1 / spot_eigvals.max(),
                                    max(20, int(optimal_solution[0] / 0.01))):
                optimal_solution[0] = para
                y = denoise_function_numpy(optimal_solution, constant_value, k)
                if _check_over_diffusion(x, y, cutoff=inv_cutoff):
                    break
        if _check_over_diffusion(x, y, cutoff=inv_cutoff):
            for para in np.linspace(optimal_solution[0], 0,
                                    max(20, int(optimal_solution[0] / 0.01))):
                optimal_solution[0] = para
                y = denoise_function_numpy(optimal_solution, constant_value, k)
                if not _check_over_diffusion(x, y, cutoff=inv_cutoff):
                    break
    elif inv_cutoff == 'auto':
        para_list = np.linspace(0.01 * 1 / spot_eigvals.max(),
                                min(0.9 * 1 / spot_eigvals.max(),
                                    max(0.75 * 1 / spot_eigvals.max(),
                                        4 * optimal_solution[0])),
                                50)
        inv_ratio_list = []
        for para in para_list:
            y = denoise_function_numpy([para, optimal_solution[1]], constant_value, k)
            score = _check_over_diffusion_score(x, y)
            inv_ratio_list.append(score)
        inv_ratio_arr = np.array(inv_ratio_list)
        para_arr = para_list
        valid_mask = (~np.isnan(inv_ratio_arr)) & (inv_ratio_arr < inv_knee_upbound)

        # print(para_list)
        # print(inv_ratio_list)

        from kneed import KneeLocator
        knee_point = KneeLocator(para_arr[valid_mask],
                           inv_ratio_arr[valid_mask],
                           direction='increasing',
                           curve='convex',
                           S=1)
        if knee_point.knee:
            # print("Found knee", knee_point.knee, knee_point.knee_y)
            if knee_point.knee_y <= 1.3:
                optimal_solution[0] = knee_point.knee
            else:
                exp_para = np.e ** (- 0.1 * (knee_point.knee_y - 1)) * knee_point.knee
                optimal_solution[0] = exp_para
            print("New para", optimal_solution[0])
        y = denoise_function_numpy(optimal_solution, constant_value, k)

    if normalization:
        y = y / y.sum(axis=0) * constant_value[1].sum(axis=0)
    adata.X = y

    # print parameters
    print("Current parameters: \t", optimal_solution)
    if gene_network:
        adata = denoise_by_genenet(adata, cutoff=1, c=c, device=device)
    # adjust in 20260609
    # adata = _remove_other_factors(adata, max_library_size=max_library_size)
    if not ss.issparse(adata.X):
        adata.X = csr_matrix(adata.X, dtype=np.float32)

    return adata, optimal_solution, loss_list, constant_value


def _remove_other_factors(adata, max_library_size, in_tissue_key='in_tissue'):
    if adata.is_view:
        adata = adata.copy()

    original_sparse = ss.issparse(adata.X)
    if not in_tissue_key in adata.obs.keys():
        return adata
    adata_out = adata[adata.obs[in_tissue_key] == 0, :]
    if adata_out.shape[0] == 0:
        return adata
    out_exp = adata_out.X.toarray() if ss.issparse(adata_out.X) else adata_out.X
    median_exp = []
    for i in range(out_exp.shape[1]):
        col_data = out_exp[:, i]
        positive_values = col_data[col_data > 0]
        if len(positive_values) > 0.05 * out_exp.shape[0]:
            median_exp.append(np.median(positive_values))
        else:
            median_exp.append(0)
    median_exp = np.array(median_exp)
    if original_sparse:
        y_dense = adata.X.toarray()
        y_dense = y_dense - median_exp
        y_dense[y_dense < 0] = 0
        y_dense = _normalize_library_size(y_dense, target_max=max_library_size)
        adata.X = ss.csr_matrix(y_dense, dtype=np.float32)
    else:
        y = adata.X - median_exp
        y[y < 0] = 0
        y = _normalize_library_size(y, target_max=max_library_size)
        adata.X = y

    return adata


def _check_over_diffusion(original_exp, diffgsp_exp, cutoff=1.3):
    # obtain the orig
    #original_max = np.percentile(np.log1p(original_exp), 99.5, axis=0)
    #diffgsp_max = np.percentile(np.log1p(diffgsp_exp), 99.5, axis=0)
    original_max = np.percentile(original_exp, 99, axis=0)
    diffgsp_max = np.percentile(diffgsp_exp, 99, axis=0)
    mean_ratio = np.mean(np.divide(diffgsp_max + 1, original_max + 1))
    if mean_ratio > cutoff:
        return True
    else:
        return False


def _check_over_diffusion_score(original_exp, diffgsp_exp):
    # obtain the orig
    original_max = np.percentile(original_exp, 99, axis=0)
    # original_max = np.max(original_exp, axis=0)
    diffgsp_max = np.percentile(diffgsp_exp, 99, axis=0)
    # diffgsp_max = np.max(diffgsp_exp, axis=0)
    mean_ratio = np.mean(np.divide(diffgsp_max + 1, original_max + 1))
    return mean_ratio


def run_diffgsp_subgraph(adata,
                         k=2,
                         variable=None,
                         normalization=False,
                         gene_network=False,
                         c=0.0001,
                         partition=[2, 2],
                         spatial_key='spatial',
                         array_key=['array_row', 'array_col'],
                         rad_cutoff=None,
                         data_type=None,
                         dic_storage=True,
                         knn_method='Radius',
                         device='cpu',
                         k_cutoff=18,
                         inv_cutoff=1.1,
                         inv_knee_upbound=2,
                         sigma=2,
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
    normalization : bool, optional (default=False)
        Whether to normalize the denoised matrix to match the original total expression.
    gene_network : bool, optional (default=False)
        Whether to include the gene graph structure for gene-wise smoothing.
    c : float, optional (default=0.0001)
        The low-pass filtering coefficient in gene network smoothing.
    partition : list[int], optional (default=[2, 2])
        Number of spatial partitions along [x, y] axes for subgraph processing.
    spatial_key : list[str], optional (default='spatial')
        Keys in `adata.obs` or `'spatial'` in `adata.obsm` used as spatial coordinates.
    rad_cutoff : float, optional (default=None)
        Radius cutoff for spatial graph construction. Auto-assigned if not specified.
    data_type : str
        Spatial platform type: one of ['Visium', 'Stereoseq', 'Slideseq', 'VisiumHD'].
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
    if 'x' not in adata.obs or 'y' not in adata.obs:
        adata.obs['x'] = np.array(adata.obsm['spatial'][:, 0])
        adata.obs['y'] = np.array(adata.obsm['spatial'][:, 1])

    if isinstance(spatial_key, str):
        if spatial_key not in adata.obsm.keys():
            raise ValueError(f'{spatial_key} not found in adata.obsm')
        if spatial_key != 'spatial':
            adata.obsm['spatial'] = adata.obsm[spatial_key]

    elif isinstance(spatial_key, list):
        sp_array = adata.obs.loc[:, spatial_key].values
        adata.obsm['spatial'] = sp_array

    sp_df = adata.obs.loc[:, array_key].values
    batch_x_coor = np.percentile(sp_df[:, 0], np.linspace(0, 100, partition[0] + 1))
    batch_y_coor = np.percentile(sp_df[:, 1], np.linspace(0, 100, partition[1] + 1))
    gene_eigvecs, gene_eigvals = None, None

    if data_type in ['Stereoseq', 'VisiumHD']:
        rad_cutoff = 0.1 if rad_cutoff is None else rad_cutoff
    elif data_type in ['Visium', 'ST']:
        rad_cutoff = 0.1 if rad_cutoff is None else rad_cutoff
    elif data_type == 'Slideseq':
        rad_cutoff = 0.1 if rad_cutoff is None else rad_cutoff
    else:
        rad_cutoff = 0.1 if rad_cutoff is None else rad_cutoff
    if variable is None:
        if data_type == 'Visium':
            optimal_solution = [0.1, 0.1]
        elif data_type == 'Stereoseq':
            optimal_solution = [0.1, 0.05]
        elif data_type == 'VisiumHD':
            optimal_solution = [0.1, 0.05]
        elif data_type == 'Slideseq':
            optimal_solution = [0.1, 0.05]
        else:
            optimal_solution = [0.1, 0.05]
    else:
        optimal_solution = variable[:2]
    spot_eigvecs_dic = {}
    adata_result = None
    max_library_size = np.percentile(adata.X.sum(axis=1).flatten().tolist(), 99)
    total_iterations = partition[0] * partition[1]

    # adjust optimal solution
    it_x = int(partition[0] / 2)
    it_y = int(partition[1] / 2)
    min_x = batch_x_coor[it_x]
    max_x = batch_x_coor[it_x + 1]
    min_y = batch_y_coor[it_y]
    max_y = batch_y_coor[it_y + 1]
    step_x = max_x - min_x
    step_y = max_y - min_y
    temp_adata = adata.copy()
    temp_adata = temp_adata[
        temp_adata.obs[array_key[0]].map(lambda x: (min_x - 0.1 * step_x) <= x < (max_x + 0.1 * step_x)
        if it_x != partition[0] - 1 else (min_x - 0.1 * step_x) <= x <= max_x)]
    temp_adata = temp_adata[
        temp_adata.obs[array_key[1]].map(lambda y: (min_y - 0.1 * step_y) <= y < (max_y + 0.1 * step_y)
        if it_y != partition[1] - 1 else (min_y - 0.1 * step_y) <= y <= max_y)]

    print(temp_adata.shape)
    if temp_adata.shape[0] == 0:
        for it_x in range(partition[0]):
            for it_y in range(partition[1]):
                min_x = batch_x_coor[it_x]
                max_x = batch_x_coor[it_x + 1]
                min_y = batch_y_coor[it_y]
                max_y = batch_y_coor[it_y + 1]
                temp_adata = adata.copy()
                temp_adata = temp_adata[
                    temp_adata.obs[array_key[0]].map(lambda x: (min_x - 0.1 * step_x) <= x < (max_x + 0.1 * step_x)
                    if it_x != partition[0] - 1 else (min_x - 0.1 * step_x) <= x <= max_x)]
                temp_adata = temp_adata[
                    temp_adata.obs[array_key[1]].map(lambda y: (min_y - 0.1 * step_y) <= y < (max_y + 0.1 * step_y)
                    if it_y != partition[1] - 1 else (min_y - 0.1 * step_y) <= y <= max_y)]
                if temp_adata.shape[0] == 0:
                    continue

    names_order = temp_adata.obs.sort_values(['x', 'y']).index
    temp_adata = temp_adata[names_order, :]

    x_unique = len(np.unique(temp_adata.obs['x']))
    y_unique = len(np.unique(temp_adata.obs['y']))
    if (x_unique, y_unique, temp_adata.shape[0]) not in spot_eigvecs_dic.keys():
        spotnet = obtain_spotnet(temp_adata,
                                 knn_method=knn_method,
                                 rad_cutoff=rad_cutoff,
                                 k_cutoff=k_cutoff,
                                 sigma=sigma)
        spot_eigvecs, spot_eigvals = spectral_decomposition(spotnet,
                                                            n_GFT=int(adata.shape[0]),
                                                            device=device)
        spot_eigvals = spot_eigvals * (spot_eigvals > 0)
    spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])] = [spot_eigvecs, spot_eigvals]
    if not ss.issparse(temp_adata.X):
        x = temp_adata.X.copy()
    else:
        x = temp_adata.X.toarray()

    if len(variable) == 3:
        constant_value = [None, x, spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])][0],
                          spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])][1], gene_eigvecs,
                          gene_eigvals]
    else:
        constant_value = [None, x, spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])][0],
                          spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])][1]]

    y = denoise_function_numpy(optimal_solution, constant_value, k)
    if inv_cutoff is not None and inv_cutoff != 'auto':
        if not _check_over_diffusion(x, y, cutoff=inv_cutoff):
            for para in np.linspace(optimal_solution[0],
                                    0.9 * 1 / spot_eigvals.max(),
                                    max(20, int(optimal_solution[0] / 0.01))):
                optimal_solution[0] = para
                y = denoise_function_numpy(optimal_solution, constant_value, k)
                if _check_over_diffusion(x, y, cutoff=inv_cutoff):
                    break

        if _check_over_diffusion(x, y, cutoff=inv_cutoff):
            print(f"over inverse-diffusion under inv_cutoff parameter: {inv_cutoff}")
            for para in np.linspace(optimal_solution[0], 0, max(20, int(optimal_solution[0] / 0.01))):
                optimal_solution[0] = para
                y = denoise_function_numpy(optimal_solution, constant_value, k)
                if not _check_over_diffusion(x, y, cutoff=inv_cutoff):
                    break

    elif inv_cutoff == 'auto':
        para_list = np.linspace(0.01 * 1 / spot_eigvals.max(),
                                min(0.9 * 1 / spot_eigvals.max(),
                                    max(0.75 * 1 / spot_eigvals.max(),
                                        4 * optimal_solution[0])),
                                50)
        inv_ratio_list = []
        for para in para_list:
            y = denoise_function_numpy([para, optimal_solution[1]], constant_value, k)
            score = _check_over_diffusion_score(x, y)
            inv_ratio_list.append(score)
        inv_ratio_arr = np.array(inv_ratio_list)
        para_arr = para_list
        valid_mask = (~np.isnan(inv_ratio_arr)) & (inv_ratio_arr < inv_knee_upbound)
        # print(para_list)
        # print(inv_ratio_list)
        from kneed import KneeLocator
        knee_point = KneeLocator(para_arr[valid_mask],
                           inv_ratio_arr[valid_mask],
                           direction='increasing',
                           curve='convex',
                           S=1)
        if knee_point.knee:
            print("Found knee", knee_point.knee, knee_point.knee_y)
            if knee_point.knee_y <= 1.3:
                optimal_solution[0] = knee_point.knee
            else:
                exp_para = np.e ** (- 0.1 * (knee_point.knee_y - 1)) * knee_point.knee
                optimal_solution[0] = exp_para
            print("New para", optimal_solution[0])

    if optimal_solution[1] > optimal_solution[0]:
        optimal_solution[1] = 0.5 * optimal_solution[0]

    with tqdm(total=total_iterations, desc='Test', ncols=100) as pbar:
        count = 0
        for it_x in range(partition[0]):
            for it_y in range(partition[1]):
                min_x = batch_x_coor[it_x]
                max_x = batch_x_coor[it_x + 1]
                min_y = batch_y_coor[it_y]
                max_y = batch_y_coor[it_y + 1]
                step_x = max_x - min_x
                step_y = max_y - min_y
                temp_adata = adata.copy()
                temp_adata = temp_adata[
                    temp_adata.obs[array_key[0]].map(lambda x: (min_x - 0.1 * step_x) <= x < (max_x + 0.1 * step_x)
                    if it_x != partition[0] - 1 else (min_x - 0.1 * step_x) <= x <= max_x)]
                temp_adata = temp_adata[
                    temp_adata.obs[array_key[1]].map(lambda y: (min_y - 0.1 * step_y) <= y < (max_y + 0.1 * step_y)
                    if it_y != partition[1] - 1 else (min_y - 0.1 * step_y) <= y <= max_y)]
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
                    spotnet = obtain_spotnet(temp_adata,
                                             knn_method=knn_method,
                                             k_cutoff=k_cutoff,
                                             rad_cutoff=rad_cutoff,
                                             sigma=sigma)
                    spot_eigvecs, spot_eigvals = spectral_decomposition(spotnet, n_GFT=temp_adata.shape[0],
                                                                        device=device)
                    spot_eigvals = spot_eigvals * (spot_eigvals > 0)

                    spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])] = [spot_eigvecs, spot_eigvals]
                    if optimal_solution[0] >= 1 / spot_eigvals.max():
                        print(f"Optimal solution (diffusion coefficient) fails to meet the the subgraph constraint!"
                              f"Current subgraph value: {optimal_solution[0]},"
                              " modify it to: {0.999 / spot_eigvals.max():.6f}")
                        optimal_solution[0] = 0.999 / spot_eigvals.max()
                if not ss.issparse(temp_adata.X):
                    x = temp_adata.X.copy()
                else:
                    x = temp_adata.X.toarray()

                if len(variable) == 3:
                    constant_value = [None, x, spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])][0],
                                      spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])][1], gene_eigvecs,
                                      gene_eigvals]
                else:
                    constant_value = [None, x, spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])][0],
                                      spot_eigvecs_dic[(x_unique, y_unique, temp_adata.shape[0])][1]]
                if dic_storage is False:
                    spot_eigvecs_dic.pop((x_unique, y_unique, temp_adata.shape[0]), None)
                temp_adata.X = denoise_function_numpy(optimal_solution, constant_value, k)
                if temp_adata.is_view:
                    temp_adata = temp_adata.copy()
                temp_adata.X = csr_matrix(temp_adata.X, dtype=np.float32)
                # extract the results
                temp_adata = temp_adata[temp_adata.obs[array_key[0]].map(lambda x: min_x <= x < max_x
                    if it_x != partition[0] - 1 else min_x <= x <= max_x)]
                temp_adata = temp_adata[temp_adata.obs[array_key[1]].map(lambda y: min_y <= y < max_y
                    if it_y != partition[1] - 1 else min_y <= y <= max_y)]
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
    if normalization:
        adata_result.X = adata_result.X / np.array(adata_result.X.sum(axis=0)) * np.array(adata.X.sum(axis=0))
    if not isinstance(adata_result.X, csr_matrix):
        if adata_result.is_view:
            adata_result = adata_result.copy()
        adata_result.X = csr_matrix(adata_result.X, dtype=np.float32)

    if gene_network and adata.shape[1] <= 5000 and adata.shape[0] <= 20000:
        adata_result = denoise_by_genenet(adata_result, cutoff=1, c=c, device=device)
    adata_result = _remove_other_factors(adata_result, max_library_size=max_library_size)

    return adata_result, optimal_solution


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


def downsampling(adata_high, multiple=[5, 5], in_rate=0.5):
    spatial_coords = adata_high.obsm['spatial']
    length = spatial_coords.max(axis=0) - spatial_coords.min(axis=0)
    n_x = int(np.sqrt(adata_high.shape[0] * length[0] / length[1]) / multiple[0])
    n_y = int(np.sqrt(adata_high.shape[0] * length[1] / length[0]) / multiple[1])
    print('Number of down-sampled spots:', n_x * n_y)
    x_range = np.percentile(spatial_coords[:, 0], np.linspace(0, 100, n_x + 1))
    y_range = np.percentile(spatial_coords[:, 1], np.linspace(0, 100, n_y + 1))

    new_expression_matrix = np.zeros((n_x * n_y, adata_high.shape[1]))
    new_coordinate_matrix = np.zeros((n_x * n_y, 2))
    new_obs_matrix = np.zeros(n_x * n_y)

    x_grid = np.digitize(spatial_coords[:, 0], x_range) - 1
    y_grid = np.digitize(spatial_coords[:, 1], y_range) - 1
    for i in range(n_x):
        for j in range(n_y):
            mask = (x_grid == i) & (y_grid == j)
            indices = np.where(mask)[0]
            if indices.size > 0:
                new_expression_matrix[i * n_y + j, :] = np.sum(adata_high.X[indices, :], axis=0)
                new_coordinate_matrix[i * n_y + j, :] = np.mean(spatial_coords[indices, :], axis=0)
                if np.sum(adata_high.obs['in_tissue'][indices]) > indices.size * in_rate:
                    new_obs_matrix[i * n_y + j] = 1
    adata_down = sc.AnnData(X=new_expression_matrix)
    adata_down.obsm['spatial'] = new_coordinate_matrix
    adata_down.obs['in_tissue'] = new_obs_matrix
    adata_down.var = adata_high.var.copy()

    return adata_down


def denoise_function_numpy(variables, constant, k):
    if len(constant) == 4:
        [_, x, spot_eigvecs, spot_eigvals] = constant
        diff_coef = variables[0] / spot_eigvals.max()
        spot_coef = variables[1] / spot_eigvals.max()
        filter_diffusion = np.diag(1 / (1 - diff_coef * spot_eigvals))  # a
        filter_spot = np.diag(1 / (1 + spot_coef * spot_eigvals))  # c
        y = spot_eigvecs @ filter_diffusion ** k @ filter_spot ** k @ spot_eigvecs.T @ x
    else:
        [_, x, spot_eigvecs, spot_eigvals, gene_eigvecs, gene_eigvals] = constant
        diff_coef = variables[0] / spot_eigvals.max()
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
        diff_coef = variables[0] / spot_eigvals.max()
        spot_coef = variables[1] / spot_eigvals.max()
        filter_diffusion = torch.diag(1 / (1 - diff_coef * spot_eigvals))  # a
        filter_spot = torch.diag(1 / (1 + spot_coef * spot_eigvals))  # c
        y = spot_eigvecs @ filter_diffusion ** k @ filter_spot ** k @ spot_eigvecs.T @ x
    else:
        [_, x, spot_eigvecs, spot_eigvals, gene_eigvecs, gene_eigvals, max_exp_list] = constant
        # filter_diffusion = torch.diag(1 + variables[0] * spot_eigvals)  # a
        scaler_mtx_right = torch.diag(max_exp_list)
        scaler_mtx_left = torch.diag(1.0 / (max_exp_list + 1e-3))
        filter_diffusion = torch.diag(1 / (1 - variables[0] * spot_eigvals))  # a
        filter_gene = torch.diag(1 / (1 + variables[1] * gene_eigvals))  # b
        filter_spot = torch.diag(1 / (1 + variables[2] * spot_eigvals))  # c
        # filter_spot = torch.diag(torch.exp(- variables[2] * spot_eigvals))  # c
        y = (spot_eigvecs @ filter_diffusion ** k @ filter_spot ** k @ spot_eigvecs.T @ x @ scaler_mtx_left
             @ gene_eigvecs @ filter_gene @ gene_eigvecs.T @ scaler_mtx_right)

    y = y * (y >= 0)

    return y


def spectral_decomposition(net, n_GFT, device='cpu'):
    import numpy as np
    n = net.shape[0]
    full_spectrum = (n_GFT is None) or (round(n_GFT) >= n)
    if device == 'cpu' or net.shape[0] > 50000:
        lap_mtx = _get_lap_mtx(net).toarray()
        if full_spectrum:
            eigvals, eigvecs = np.linalg.eigh(lap_mtx.astype('double'))
        else:
            eigvals, eigvecs = ss.linalg.eigsh(
                lap_mtx.astype('double'),
                k=round(n_GFT),
                which='SM',
                v0=[1 / np.sqrt(n)] * n
            )
    else:
        if 'cuda' in str(device):
            device_id = int(str(device).split(':')[-1])
        else:
            device_id = 0
        import torch
        torch.cuda.set_device(device_id)
        import cupy as cp
        import cupyx.scipy.sparse as cpxs
        import cupyx.scipy.sparse.linalg as cpxsl

        with cp.cuda.Device(device_id):
            cp.get_default_memory_pool().free_all_blocks()
            lap_mtx_sp_cpu = _get_lap_mtx(net)
            if full_spectrum:
                L_gpu = cp.asarray(lap_mtx_sp_cpu.toarray(), dtype=cp.float32)
                eigvals_gpu, eigvecs_gpu = cp.linalg.eigh(L_gpu)
                eigvals = cp.asnumpy(eigvals_gpu)
                eigvecs = cp.asnumpy(eigvecs_gpu)
            else:
                lap_mtx_gpu = cpxs.csr_matrix(lap_mtx_sp_cpu, dtype=cp.float32)
                v0_gpu = cp.array([1 / np.sqrt(n)] * n, dtype=cp.float32)

                eigvals_k_gpu, eigvecs_k_gpu = cpxsl.eigsh(
                    lap_mtx_gpu,
                    k=min(round(n_GFT), n - 2),
                    which='SA',
                    v0=v0_gpu
                )
                eigvecs = cp.asnumpy(eigvecs_k_gpu)
                eigvals = cp.asnumpy(eigvals_k_gpu)


    return eigvecs, eigvals


def obtain_spotnet(adata, rad_cutoff=None, k_cutoff=18, knn_method='Radius', sigma=2):
    coor = np.array(adata.obsm['spatial'])
    delta_x = coor[:, 0].max() - coor[:, 0].min()
    delta_y = coor[:, 1].max() - coor[:, 1].min()
    if rad_cutoff is None:
        rad_cutoff = (delta_x + delta_y) / 2 * rad_cutoff
    else:
        rad_cutoff = (delta_x + delta_y) / 2 * 0.5

    # nbrs = NearestNeighbors(radius=rad_cutoff) if knn_method == 'Radius' else NearestNeighbors(n_neighbors=k_cutoff + 1)
    import platform
    if platform.system() == 'Windows':
        n_jobs = 1
    else:
        n_jobs = -1
    nbrs = NearestNeighbors(radius=rad_cutoff, n_jobs=n_jobs) if knn_method == 'Radius' else NearestNeighbors(
        n_neighbors=k_cutoff + 1, n_jobs=n_jobs)
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

    min_distance = np.percentile(KNN_df['Distance'], 1)
    min_distance = max(min_distance, KNN_df['Distance'].min())
    KNN_df['Distance'] = KNN_df['Distance'] / min_distance
    # KNN_df['InverseDistance'] = 1 / KNN_df['Distance']
    sigma = sigma * np.median(KNN_df["Distance"])
    KNN_df['InverseDistance'] = np.exp(- (KNN_df['Distance'] ** 2) / (2 * sigma ** 2))

    spot_net = csr_matrix((KNN_df['InverseDistance'], (KNN_df['Cell1'], KNN_df['Cell2'])))
    spot_net = (spot_net + spot_net.transpose()) / 2

    return spot_net


def obtain_genenet(adata, cut=0.2):
    adata = adata[adata.obs['in_tissue'] == 1, :]
    if adata.shape[0] > 5000:
        spot_choose = np.random.choice(adata.obs_names, 4000, replace=False)
    else:
        spot_choose = adata.obs_names
    if not ss.issparse(adata.X):
        X = adata[spot_choose, :].X.copy()
    else:
        X = adata[spot_choose, :].X.toarray()
    df = pd.DataFrame(X)
    genenet = df.corr(method='spearman').values
    genenet -= np.diag([1] * genenet.shape[0])
    max_deg = np.percentile(genenet, 99.5)
    cut = cut * max_deg
    result = genenet * (genenet >= cut)
    result = (result + result.T) / 2
    ave_connectivity = np.count_nonzero(result) / result.shape[0]
    print('Average connectivity of gene network:', ave_connectivity / 2)
    print('Maximum connectivity of gene network:', (result != 0).sum(axis=0).max() / 2)

    return result, ave_connectivity


def select_svgs(adata_copy, svg_method='gft', n_top=2000, boost=True):
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
        boost: bool, optional
            Boost the speed by down-sampling.

    Returns:
        adata: anndata
            AnnData object of scanpy package after choosing svgs and smoothing.
        adata_raw: anndata
            AnnData object of scanpy package before choosing svgs and smoothing.
    """
    adata = adata_copy.copy()
    if 'in_tissue' not in adata.obs.keys():
        adata.obs['in_tissue'] = 1
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
                                   spatial_info='spatial',
                                   ratio_low_freq=1,
                                   ratio_high_freq=1,
                                   ratio_neighbors=1,
                                   filter_peaks=True,
                                   S=6,
                                   boost=boost)
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
                                       boost=boost,
                                       S=6)
            if svg_method == 'gft':
                svg_list = gene_df[gene_df.cutoff_gft_score][gene_df.qvalue < 0.05].index.tolist()
            elif svg_method == 'gft_top':
                svg_list = gene_df.index[:n_top].tolist()

    return svg_list

def denoise_by_genenet(adata, cutoff=0.75, c=0.1, device='cpu'):
    import warnings
    warnings.filterwarnings('ignore')
    # log-transformation
    if ss.issparse(adata.X):
        x = adata.X.todense().A
    else:
        x = adata.X.copy()
    if isinstance(x, np.matrix):
        x = x.A
    x = np.log1p(x)

    # obtain gene_net
    genenet, mean_degree = obtain_genenet(adata, cut=cutoff)

    # spectral decomposition
    gene_eigvecs, gene_eigvals = spectral_decomposition(genenet, n_GFT=adata.shape[1], device=device)
    gene_eigvals = gene_eigvals * (gene_eigvals > 0)

    # check the mean expression
    mean_exp_list = scale_exp_calculation(x)
    mean_exp_list = np.array(mean_exp_list)

    # low-pass filtering
    c = c / gene_eigvals.max() if gene_eigvals.max() > 0 else 0.0001
    c = min(c, 0.1)
    # low-pass filter
    scaler_mtx_right = np.diag(mean_exp_list + 1e-4)
    scaler_mtx_left = np.diag(1.0 / (mean_exp_list + 1e-4))
    filter_gene = np.diag(1 / (1 + c * gene_eigvals))  # b
    x = x @ scaler_mtx_left @ gene_eigvecs @ filter_gene @ gene_eigvecs.T @ scaler_mtx_right


    adata.X = np.expm1(x)

    return adata