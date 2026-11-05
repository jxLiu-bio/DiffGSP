import torch
import os
import random
import scipy.sparse as ss
import warnings
import numpy as np
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.sparse import issparse, csr_matrix, hstack

warnings.filterwarnings('ignore')


def scale_exp_calculation(x, percentile=97):
    num_cols = x.shape[1]
    mean_positive_list = []
    for i in range(num_cols):
        column_data = x[:, i].flatten()
        positive_values = column_data[column_data > 0]
        if len(positive_values) > 0:
            mean_val = np.percentile(positive_values, q=percentile)
        else:
            mean_val = 0.1
        mean_positive_list.append(mean_val)

    return mean_positive_list


def _create_degree_mtx(diag):
    diag = np.array(diag)
    diag = diag.flatten()
    row_index = list(range(diag.size))
    col_index = row_index
    sparse_mtx = ss.coo_matrix((diag, (row_index, col_index)), shape=(diag.size, diag.size))
    return sparse_mtx


def _get_lap_mtx(net, normalization=True):
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


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = False
            torch.use_deterministic_algorithms(True)
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    except Exception:
        pass


def prefilter_genes(adata_copy, min_counts=None, max_counts=None, min_cells=3, max_cells=None):
    adata = adata_copy.copy()
    if 'in_tissue' in adata.obs.keys():
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


def correct_high_from_low(
        adata_low,
        adata_high,
        low_layer="X",
        high_layer="X",
        spatial_key="spatial",
        batch_size=2000,
        auto_scale=True
):
    """
    Performs mass-conserving redistribution handling different coordinate scales.

    Parameters
    ----------
    auto_scale : bool
        If True, normalizes both low and high coordinates to the [0, 1] range
        before finding neighbors. Use this if 'low' is 100x100 but 'high' is 5000x5000.
        Assumption: Both datasets cover the exact same biological Field of View (FOV).
    """

    # --- 1. Prepare Coordinates ---
    print("Extracting coordinates...")
    # Copy to avoid modifying original objects
    low_coords = np.array(adata_low.obsm[spatial_key], dtype=np.float32)
    high_coords = np.array(adata_high.obsm[spatial_key], dtype=np.float32)

    # --- 2. Coordinate Alignment (Scaling) ---
    if auto_scale:
        print(">> Auto-scaling coordinates to [0, 1] range for alignment...")
        def normalize_coords(coords):
            # Min-Max Normalization: (x - min) / (max - min)
            c_min = coords.min(axis=0)
            c_max = coords.max(axis=0)
            range_val = c_max - c_min
            # Avoid division by zero
            range_val[range_val == 0] = 1
            return (coords - c_min) / range_val

        # We normalize ONLY for the KDTree step.
        # We do not change the original .obsm['spatial']
        search_low_coords = normalize_coords(low_coords)
        search_high_coords = normalize_coords(high_coords)
    else:
        search_low_coords = low_coords
        search_high_coords = high_coords

    n_low = low_coords.shape[0]
    n_high = high_coords.shape[0]
    n_genes = adata_high.shape[1]

    # --- 3. Build Global Spatial Map (Using Scaled Coords) ---
    print(f"Mapping {n_high} high-res spots to {n_low} low-res spots...")

    # Use the normalized coordinates for distance calculation
    tree = cKDTree(search_low_coords)
    _, nearest_low_indices = tree.query(search_high_coords, k=1)

    # Calculate group counts for fallback (Uniform distribution)
    group_counts = np.bincount(nearest_low_indices, minlength=n_low)
    mapped_counts = group_counts[nearest_low_indices]
    mapped_counts[mapped_counts == 0] = 1
    uniform_weights_col = (1.0 / mapped_counts).astype(np.float32)[:, None]

    # --- 4. Process Genes in Batches ---
    print(f"Processing {n_genes} genes in batches of {batch_size}...")
    sparse_blocks = []

    def get_slice(adata, layer, s, e):
        X_slice = adata.X[:, s:e] if layer == "X" else adata.layers[layer][:, s:e]
        if issparse(X_slice): X_slice = X_slice.toarray()
        return X_slice.astype(np.float32)

    for start_idx in range(0, n_genes, batch_size):
        end_idx = min(start_idx + batch_size, n_genes)

        # Load Data
        low_chunk = get_slice(adata_low, low_layer, start_idx, end_idx)
        high_chunk = get_slice(adata_high, high_layer, start_idx, end_idx)
        current_batch_width = high_chunk.shape[1]

        # Calculate Sum per Group
        sum_high_chunk = np.zeros((n_low, current_batch_width), dtype=np.float32)
        np.add.at(sum_high_chunk, nearest_low_indices, high_chunk)
        mapped_sum_chunk = sum_high_chunk[nearest_low_indices]

        # Calculate Weights
        weights_chunk = np.zeros_like(high_chunk)
        valid_mask = mapped_sum_chunk > 0

        # Proportional vs Uniform
        weights_chunk[valid_mask] = high_chunk[valid_mask] / mapped_sum_chunk[valid_mask]
        if np.any(~valid_mask):
            weights_chunk[~valid_mask] = np.broadcast_to(uniform_weights_col, weights_chunk.shape)[~valid_mask]

        # Apply Correction
        mapped_low_chunk = low_chunk[nearest_low_indices]
        corrected_chunk = weights_chunk * mapped_low_chunk

        # Save Sparse
        sparse_blocks.append(csr_matrix(corrected_chunk, dtype=np.float32))
        del low_chunk, high_chunk, sum_high_chunk, weights_chunk, corrected_chunk

    # --- 5. Finalize ---
    print("Concatenating sparse blocks...")
    full_sparse_matrix = hstack(sparse_blocks, format='csr')
    print("Done.")

    if adata_high.is_view:
        adata_high = adata_high.copy()
    adata_high.X = full_sparse_matrix

    return adata_high


def _normalize_library_size(exp_mtx, target_max=10000, quantile=0.99):
    library_size = exp_mtx.sum(axis=1)
    q_val = np.quantile(library_size, quantile)

    scaling_factors = np.ones_like(library_size, dtype=float)

    high_mask = library_size > q_val
    normal_mask = ~high_mask & (library_size > 0)

    if np.any(high_mask):
        scaling_factors[high_mask] = target_max / library_size[high_mask]

    if np.any(normal_mask):
        normal_scale = target_max / q_val
        scaling_factors[normal_mask] = normal_scale

    return exp_mtx * scaling_factors[:, np.newaxis]