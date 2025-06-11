import pandas as pd
import numpy as np
import warnings
import scipy.sparse as ss
import scanpy as sc
from scipy.spatial.distance import pdist, squareform
import itertools
import scipy.sparse as ss
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
import networkx as nx

warnings.filterwarnings("ignore")


def _correct_pvalues_for_multiple_testing(pvalues,
                                          correction_type="Benjamini-Hochberg"):
    """
    Correct pvalues to obtain the adjusted pvalues 

    Parameters
    ----------
    pvalues : list | 1-D array
        The original p values. It should be a list.
    correction_type : str, optional
        Method used to correct p values. The default is "Benjamini-Hochberg".

    Returns
    -------
    new_pvalues : array
        Corrected p values.

    """
    from numpy import array, empty
    pvalues = array(pvalues)
    n = int(pvalues.shape[0])
    new_pvalues = empty(n)
    if correction_type == "Bonferroni":
        new_pvalues = n * pvalues
    elif correction_type == "Bonferroni-Holm":
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        for rank, vals in enumerate(values):
            pvalue, i = vals
            new_pvalues[i] = (n - rank) * pvalue
    elif correction_type == "Benjamini-Hochberg":
        values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
        values.sort()
        values.reverse()
        new_values = []
        for i, vals in enumerate(values):
            rank = n - i
            pvalue, index = vals
            new_values.append((n / rank) * pvalue)
        for i in range(0, int(n) - 1):
            if new_values[i] < new_values[i + 1]:
                new_values[i + 1] = new_values[i]
        for i, vals in enumerate(values):
            pvalue, index = vals
            new_pvalues[index] = new_values[i]
    return new_pvalues


def _permutation_signal(signal_array, num_permutaion=1000):
    """
    Permutate gene signals in spatial domain randomly.

    Parameters
    ----------
    signal_array : list | array
        A one-dimensional array indicate gene expression on all spots.
    num_permutaion : int, optional
        The number of permutation. The default is 1000.

    Returns
    -------
    total_signals : array
        The permutaed gene expression signals.

    """
    signal_array = np.array(signal_array)
    total_signals = signal_array * np.ones((num_permutaion,
                                            len(signal_array)))

    for i in range(num_permutaion):
        total_signals[i, :] = np.random.permutation(total_signals[i, :])

    return total_signals


def _significant_test_permutation(exp_mtx,
                                  gene_score,
                                  eigvals,
                                  eigvecs_T,
                                  num_permutaion=1000,
                                  num_pool=200,
                                  spec_norm='l1'):
    """
    To calculate p values for genes, permutate gene expression data and 
    calculate p values.

    Parameters
    ----------
    exp_mtx : 2D-array
        The count matrix of gene expresssion. (spots * genes)
    gene_score : 1D-array
        The calculated gene scores. 
    eigvals : array
        The eigenvalues of Laplacian matrix.
    eigvecs_T : array
        The eigenvectors of Laplacian matrix.
    num_permutaion : int, optional
        The number of permutations. The default is 1000.
    num_pool : int, optional
        The cores used for umltiprocess calculation to accelerate speed. The 
        default is 200.
    spec_norm : str, optional
        The method to normalize graph signals in spectral domain. The default 
        is 'l1'.

    Returns
    -------
    array
        The calculated p values.

    """
    from multiprocessing.dummy import Pool as ThreadPool
    from scipy.stats import mannwhitneyu

    def _test_by_permutaion(gene_index):
        (gene_index)
        graph_signal = exp_mtx[gene_index, :]
        total_signals = _permutation_signal(signal_array=graph_signal,
                                            num_permutaion=num_permutaion)
        frequency_array = np.matmul(eigvecs_T, total_signals.transpose())
        frequency_array = np.abs(frequency_array)
        if spec_norm != None:
            frequency_array = preprocessing.normalize(frequency_array,
                                                      norm=spec_norm,
                                                      axis=0)
        score_list = np.matmul(2 ** (-1 * eigvals), frequency_array)
        score_list = score_list / score_max
        pval = mannwhitneyu(score_list, gene_score[gene_index], alternative='less').pvalue
        return pval

    score_max = np.matmul(2 ** (-2 * eigvals), (1 / len(eigvals)) * \
                          np.ones(len(eigvals)))
    gene_index_list = list(range(exp_mtx.shape[0]))
    pool = ThreadPool(num_pool)
    res = pool.map(_test_by_permutaion, gene_index_list)

    return res


def _test_significant_freq(freq_array,
                           cutoff,
                           num_pool=200):
    """
    Significance test by camparing the intensities in low frequency FMs and 
    in high frequency FMs. 

    Parameters
    ----------
    freq_array : array
        The graph signals of genes in frequency domain. 
    cutoff : int
        Watershed between low frequency signals and high frequency signals.
    num_pool : int, optional
        The cores used for umltiprocess calculation to accelerate speed. The 
        default is 200.

    Returns
    -------
    array
        The calculated p values.

    """
    from scipy.stats import wilcoxon, mannwhitneyu, ranksums, combine_pvalues
    from multiprocessing.dummy import Pool as ThreadPool

    def _test_by_feq(gene_index):
        freq_signal = freq_array[gene_index, :]
        freq_1 = freq_signal[:cutoff]
        freq_1 = freq_1[freq_1 > 0]
        freq_2 = freq_signal[cutoff:]
        freq_2 = freq_2[freq_2 > 0]
        if freq_1.size <= 50 or freq_2.size <= 50:
            freq_1 = np.concatenate((freq_1, freq_1, freq_1, freq_1))
            freq_2 = np.concatenate((freq_2, freq_2, freq_2, freq_2))
        pval = ranksums(freq_1, freq_2, alternative='greater').pvalue
        return pval

    gene_index_list = list(range(freq_array.shape[0]))
    pool = ThreadPool(num_pool)
    res = pool.map(_test_by_feq, gene_index_list)

    return res


def _my_eigsh(args_tupple):
    """
    The function is used to multi-process calculate using Pool.

    Parameters
    ----------
    args_tupple : tupple 
        The args_tupple contains three elements, that are, Lplacian matrix, k 
        and which.

    Returns
    -------
    (eigvals, eigvecs)

    """
    lap_mtx = args_tupple[0]
    k = args_tupple[1]
    which = args_tupple[2]
    eigvals, eigvecs = ss.linalg.eigsh(lap_mtx.astype(float),
                                       k=k,
                                       which=which)
    return ((eigvals, eigvecs))


def low_pass_enhancement(adata,
                         ratio_low_freq='infer',
                         ratio_high_freq='infer',
                         ratio_neighbors='infer',
                         c=0.0001,
                         spatial_info=['array_row', 'array_col'],
                         normalize_lap=False,
                         inplace=False):
    """
    Implement gene expression with low-pass filter. After this step, the 
    spatially variables genes will be more smooth than previous. The function
    can also be treated as denoising. Note that the denosing results is related
    to spatial graph topology so that only the resulsts of spatially variable
    genes could be convincing.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinat-
        es of all spots should be found in adata.obs or adata.obsm.
    ratio_low_freq : float | "infer", optional
        The ratio_low_freq will be used to determine the number of the FMs with
        low frequencies. Indeed, the ratio_low_freq * sqrt(number of spots) low
        frequecy FMs will be calculated. The default is 'infer'.
    ratio_high_freq: float | 'infer', optional
        The ratio_high_freq will be used to determine the number of the FMs with
        high frequencies. Indeed, the ratio_high_freq * sqrt(number of spots) 
        high frequecy FMs will be calculated. If 'infer', the ratio_high_freq 
        will be set to 0. The default is 'infer'.
        A high can achieve better smothness. c should be setted to [0, 0.1].
    ratio_neighbors: float | 'infer', optional
        The ratio_neighbors will be used to determine the number of neighbors
        when contruct the KNN graph by spatial coordinates. Indeed, ratio_neig-
        hobrs * sqrt(number of spots) / 2 indicates the K. If 'infer', the para
        will be set to 1.0. The default is 'infer'.
    c: float, optional
        c balances the smoothness and difference with previous expresssion.
    spatial_info : list or tupple, optional
        The column names of spaital coordinates in adata.obs_names or key
        in adata.obsm_keys() to obtain spatial information. The default
        is ['array_row', 'array_col'].
    normalize_lap : bool. optional
        Whether need to normalize the Laplcian matrix. The default is False.
    inplace: bool, optional
        

    Returns
    -------
    count_matrix: DataFrame

    """
    import scipy.sparse as ss
    if ratio_low_freq == 'infer':
        if adata.shape[0] <= 800:
            num_low_frequency = min(15 * int(np.ceil(np.sqrt(adata.shape[0]))),
                                    adata.shape[0])
        elif adata.shape[0] <= 2000:
            num_low_frequency = 12 * int(np.ceil(np.sqrt(adata.shape[0])))
        elif adata.shape[0] <= 10000:
            num_low_frequency = 10 * int(np.ceil(np.sqrt(adata.shape[0])))
        else:
            num_low_frequency = 4 * int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * \
                                        ratio_low_freq))
    if ratio_high_freq == 'infer':
        num_high_frequency = 0 * int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * \
                                         ratio_high_freq))

    if ratio_neighbors == 'infer':
        if adata.shape[0] <= 500:
            num_neighbors = 4
        else:
            num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2))
    else:
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2 \
                                    * ratio_neighbors))

    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=1)
    # Get Laplacian matrix according to coordinates 
    lap_mtx = get_laplacian_mtx(adata,
                                num_neighbors=num_neighbors,
                                spatial_key=spatial_info,
                                normalization=normalize_lap)

    # Fourier modes of low frequency
    eigvals_s, eigvecs_s = ss.linalg.eigsh(lap_mtx.astype(float),
                                           k=num_low_frequency,
                                           which='SM')
    if num_high_frequency > 0:
        # Fourier modes of high frequency    
        eigvals_l, eigvecs_l = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_high_frequency,
                                               which='LM')
        eigvals = np.concatenate((eigvals_s, eigvals_l))  # eigenvalues
        eigvecs = np.concatenate((eigvecs_s, eigvecs_l), axis=1)  # eigenvectors
    else:
        eigvals = eigvals_s
        eigvecs = eigvecs_s

    # *********************** Graph Fourier Tranform **************************
    # Calculate GFT
    eigvecs_T = eigvecs.transpose()
    if not ss.issparse(adata.X):
        exp_mtx = adata.X
    else:
        exp_mtx = adata.X.toarray()
    frequency_array = np.matmul(eigvecs_T, exp_mtx)
    # low-pass filter
    filter_list = [1 / (1 + c * eigv) for eigv in eigvals]
    filter_array = np.matmul(np.diag(filter_list), frequency_array)
    filter_array = np.matmul(eigvecs, filter_array)
    filter_array[filter_array < 0] = 0

    # whether need to replace original count matrix
    if inplace and not ss.issparse(adata.X):
        adata.X = filter_array
    elif inplace:
        import scipy.sparse as ss
        adata.X = ss.csr.csr_matrix(filter_array)

    filter_array = pd.DataFrame(filter_array,
                                index=adata.obs_names,
                                columns=adata.var_names)
    pass


def select_num_fms(adata,
                   ratio_low_freq='infer',
                   ratio_high_freq='infer',
                   ratio_neighbors='infer',
                   spatial_info=['array_row', 'array_col'],
                   select_auto=True,
                   cutoff_fms=0.05,
                   normalized_lap=True):
    """
    Select FMs automatically acoording to corresponding frequencies.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinat-
        es of all spots could be found in adata.obs or adata.obsm.
    ratio_low_freq : float | "infer", optional
        The ratio_low_freq will be used to determine the number of the FMs with
        low frequencies. Indeed, the ratio_low_freq * sqrt(number of spots) low
        frequecy FMs will be calculated. If 'infer', the ratio_low_freq will be
        set to 1.0. The default is 'infer'.
    ratio_high_freq: float | 'infer', optional
        The ratio_high_freq will be used to determine the number of the FMs with
        high frequencies. Indeed, the ratio_high_freq * sqrt(number of spots) 
        high frequecy FMs will be calculated. If 'infer', the ratio_high_freq 
        will be set to 0. The default is 'infer'.
        A high can achieve better smothness. c should be setted to [0, 0.05].
    ratio_neighbors: float | 'infer', optional
        The ratio_neighbors will be used to determine the number of neighbors
        when contruct the KNN graph by spatial coordinates. Indeed, ratio_neig-
        hobrs * sqrt(number of spots) / 2 indicates the K. If 'infer', the para
        will be set to 1.0. The default is 'infer'.
    spatial_info : list or tupple, optional
        The column names of spaital coordinates in adata.obs_names or key
        in adata.varm_keys() to obtain spatial information. The default
        is ['array_row', 'array_col'].
    select_auto : bool, optional
        Determine the number of FMs automatically.. The default is True.
    cutoff_fms : float, optional
        Amount of information retained. The default is 0.95.
    normalized_lap : TYPE, optional
        DESCRIPTION. The default is True.

    Raises
    ------
    ValueError
        cutoff_fms should be in (0, 1]

    Returns
    -------
    float | DataFrame
        If select == True, return the number of FMs used, value * sqrt(N).
        Otherwise, return a dataframe contains infromation mount.

    """
    # Ensure parameters
    if ratio_low_freq == 'infer':
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * \
                                        ratio_low_freq))
    if ratio_high_freq == 'infer':
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * \
                                         ratio_high_freq))

    if ratio_neighbors == 'infer':
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2))
    else:
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2 \
                                    * ratio_neighbors))
    if adata.shape[0] <= 500:
        num_neighbors = 4

    # *************** Construct graph and corresponding matrixs ***************
    lap_mtx = get_laplacian_mtx(adata, num_neighbors=num_neighbors,
                                spatial_key=spatial_info,
                                normalization=normalized_lap)

    # Next, calculate the eigenvalues and eigenvectors of the Laplace matrix
    if num_high_frequency > 0:
        # Fourier bases of low frequency
        eigvals_s, eigvecs_s = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_low_frequency,
                                               which='SM')
        # Fourier bases of high frequency
        eigvals_l, eigvecs_l = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_high_frequency,
                                               which='LM')
        adata.uns['low_freq'] = eigvals_s
        adata.uns['high_freq'] = eigvals_l
        adata.uns['low_fms'] = eigvecs_s
        adata.uns['high_fms'] = eigvecs_l
    else:
        # Fourier bases of low frequency
        eigvals_s, eigvecs_s = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_low_frequency,
                                               which='SM')
        adata.uns['low_freq'] = eigvals_s
        adata.uns['low_fms'] = eigvecs_s

    # ********************** select fms ********************************
    eigvals_s = np.abs(eigvals_s)
    eigvals_s_power = 2 ** (-2 * eigvals_s)
    if select_auto:
        if cutoff_fms <= 0 or cutoff_fms > 1:
            raise ValueError("cutoff_fms should be in (0, 1]")
        frac_d = np.sum(eigvals_s_power)
        condition = 1
        for num in range(eigvals_s.size):
            if condition <= 1 - cutoff_fms:
                break
            condition -= eigvals_s_power[num] / frac_d

        return num / np.sqrt(adata.shape[1])
    else:
        plot_df = pd.DataFrame({'FMs': range(1, eigvals_s.size + 1),
                                'amount of information': eigvals_s_power})

        return plot_df


def rank_gene_smooth(adata,
                     ratio_low_freq='infer',
                     ratio_high_freq='infer',
                     ratio_neighbors='infer',
                     spatial_info=['array_row', 'array_col'],
                     normalize_lap=False,
                     filter_peaks=True,
                     S=5,
                     cal_pval=True):
    """
    Rank genes to find spatially variable genes by graph Fourier transform.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinat-
        es could be found in adata.obs or adata.obsm.
    ratio_low_freq : float | "infer", optional
        The ratio_low_freq will be used to determine the number of the FMs with
        low frequencies. Indeed, the ratio_low_freq * sqrt(number of spots) low
        frequecy FMs will be calculated. If 'infer', the ratio_low_freq will be
        set to 1.0. The default is 'infer'.
    ratio_high_freq: float | 'infer', optional
        The ratio_high_freq will be used to determine the number of the FMs of
        high frequencies. Indeed, the ratio_high_freq * sqrt(number of spots) 
        high frequecy FMs will be calculated. If 'infer', the ratio_high_freq 
        will be set to 1.0. The default is 'infer'.
    ratio_neighbors: float | 'infer', optional
        The ratio_neighbors will be used to determine the number of neighbors
        when contruct the KNN graph by spatial coordinates. Indeed, ratio_neig-
        hobrs * sqrt(number of spots) / 2 indicates the K. If 'infer', the para
        will be set to 1.0. The default is 'infer'.
    spatial_info : list | tupple | string, optional
        The column names of spaital coordinates in adata.obs_names or key
        in adata.varm_keys() to obtain spatial information. The default
        is ['array_row', 'array_col'].
    normalize_lap : bool, optional
        Whether need to normalize laplacian matrix. The default is false.
    filter_peaks: bool, optional
        For calculated vectors/signals in frequency/spectral domian, whether
        filter low peaks to stress the important peaks. The default is True.
    S: int, optional
        The sensitivity parameter in Kneedle algorithm. A large S will enable
        more genes indentified as SVGs according to gft_score. The default is
        5.
    cal_pval : bool, optional
        Whether need to calculate p val by mannwhitneyu. The default is False.
    Returns
    -------
    score_df : dataframe
        Return gene information.

    """
    # Ensure parameters
    if ratio_low_freq == 'infer':
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * \
                                        ratio_low_freq))
    if ratio_high_freq == 'infer':
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * \
                                         ratio_high_freq))

    if ratio_neighbors == 'infer':
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2))
    else:
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2 \
                                    * ratio_neighbors))
    if adata.shape[0] <= 500:
        num_neighbors = 4

    # Ensure gene index uniquely and all gene had expression  
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=1)

    # *************** Construct graph and corresponding matrixs ***************
    lap_mtx = get_laplacian_mtx(adata, num_neighbors=num_neighbors,
                                spatial_key=spatial_info,
                                normalization=normalize_lap)

    # Next, calculate the eigenvalues and eigenvectors of the Laplace matrix
    # Fourier bases of low frequency
    eigvals_s, eigvecs_s = ss.linalg.eigsh(lap_mtx.astype(float),
                                           k=num_low_frequency,
                                           which='SM')
    if num_high_frequency > 0:
        # Fourier bases of high frequency
        eigvals_l, eigvecs_l = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_high_frequency,
                                               which='LM')
        eigvals = np.concatenate((eigvals_s, eigvals_l))  # eigenvalues
        eigvecs = np.concatenate((eigvecs_s, eigvecs_l), axis=1)  # eigenvectors
    else:
        eigvals = eigvals_s
        eigvecs = eigvecs_s

    # ************************ Graph Fourier Tranform *************************
    # Calculate GFT
    eigvecs_T = eigvecs.transpose()
    if type(adata.X) == np.ndarray:
        exp_mtx = preprocessing.scale(adata.X)
    else:
        exp_mtx = preprocessing.scale(adata.X.toarray())

    frequency_array = np.matmul(eigvecs_T, exp_mtx)
    frequency_array = np.abs(frequency_array)

    # Filter noise peaks
    if filter_peaks == True:
        frequency_array_thres_low = \
            np.quantile(frequency_array[:num_low_frequency, :],
                        q=0.5, axis=0)
        frequency_array_thres_high = \
            np.quantile(frequency_array[num_low_frequency:, :],
                        q=0.5, axis=0)
        for j in range(frequency_array.shape[1]):
            frequency_array[:num_low_frequency, :] \
                [frequency_array[:num_low_frequency, j] <= \
                 frequency_array_thres_low[j], j] = 0
            frequency_array[num_low_frequency:, :] \
                [frequency_array[num_low_frequency:, j] <= \
                 frequency_array_thres_high[j], j] = 0

    frequency_array = preprocessing.normalize(frequency_array,
                                              norm='l1',
                                              axis=0)

    eigvals = np.abs(eigvals)
    eigvals_power = np.exp(-1 * eigvals)
    score_list = np.matmul(eigvals_power, frequency_array)
    score_max = np.matmul(eigvals_power, (1 / len(eigvals)) * \
                          np.ones(len(eigvals)))
    score_list = score_list / score_max
    # print("Graph Fourier Transform finished!")

    # Rank genes according to smooth score
    adata.var["gft_score"] = score_list
    score_df = adata.var["gft_score"]
    score_df = pd.DataFrame(score_df)
    score_df = score_df.sort_values(by="gft_score", ascending=False)
    score_df.loc[:, "svg_rank"] = range(1, score_df.shape[0] + 1)
    adata.var["svg_rank"] = score_df.reindex(adata.var_names).loc[:, "svg_rank"]
    # print("SVG ranking could be found in adata.var['svg_rank']")

    # Determine cutoff of gft_score
    from kneed import KneeLocator
    magic = KneeLocator(score_df.svg_rank.values,
                        score_df.gft_score.values,
                        direction='decreasing',
                        curve='convex',
                        S=S)
    score_df['cutoff_gft_score'] = False
    score_df['cutoff_gft_score'][:(magic.elbow + 1)] = True
    adata.var['cutoff_gft_score'] = score_df['cutoff_gft_score']
    # print("""The spatially variable genes judged by gft_score could be found
    #       in adata.var['cutoff_gft_score']""")
    adata.varm['freq_domain_svg'] = frequency_array.transpose()
    # print("""Gene signals in frequency domain when detect SVGs could be found
    #       in adata.varm['freq_domain_svg']""")
    adata.uns['frequencies_svg'] = eigvals
    adata.uns['fms_low'] = eigvecs_s
    adata.uns['fms_high'] = eigvecs_l

    # ****************** calculate pval ***************************
    if cal_pval == True:
        pval_list = _test_significant_freq(
            freq_array=adata.varm['freq_domain_svg'],
            cutoff=num_low_frequency)
        from statsmodels.stats.multitest import multipletests
        qval_list = multipletests(np.array(pval_list), method='fdr_by')[1]
        adata.var['pvalue'] = pval_list
        adata.var['qvalue'] = qval_list
        score_df = adata.var.loc[score_df.index, :].copy()

    return score_df


def calculate_frequcncy_domain(adata,
                               ratio_low_freq='infer',
                               ratio_high_freq='infer',
                               ratio_neighbors='infer',
                               spatial_info=['array_row', 'array_col'],
                               return_freq_domain=True,
                               normalize_lap=False,
                               filter_peaks=False):
    """
    Obtain gene signals in frequency/spectral domain for all genes in 
    adata.var_names.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinat-
        es could be found in adata.obs or adata.obsm.
    ratio_low_freq : float | "infer", optional
        The ratio_low_freq will be used to determine the number of the FMs with
        low frequencies. Indeed, the ratio_low_freq * sqrt(number of spots) low
        frequecy FMs will be calculated. If 'infer', the ratio_low_freq will be
        set to 1.0. The default is 'infer'.
    ratio_high_freq: float | 'infer', optional
        The ratio_high_freq will be used to determine the number of the FMs with
        high frequencies. Indeed, the ratio_high_freq * sqrt(number of spots) 
        high frequecy FMs will be calculated. If 'infer', the ratio_high_freq 
        will be set to 0. The default is 'infer'.
    ratio_neighbors: float | 'infer', optional
        The ratio_neighbors will be used to determine the number of neighbors
        when contruct the KNN graph by spatial coordinates. Indeed, ratio_neig-
        hobrs * sqrt(number of spots) / 2 indicates the K. If 'infer', the para
        will be set to 1.0. The default is 'infer'.
    spatial_info : list | tupple | str, optional
        The column names of spaital coordinates in adata.obs_keys() or 
        key in adata.obsm_keys. The default is ['array_row','array_col'].
    return_freq_domain : bool, optional
        Whether need to return gene signals in frequency domain. The default is 
        True.
    normalize_lap : bool, optional
        Whether need to normalize laplacian matrix. The default is false.
    filter_peaks: bool, optional
        For calculated vectors/signals in frequency/spectral domian, whether
        filter low peaks to stress the important peaks. The default is False.

    Returns
    -------
    If return_freq_domain, return DataFrame, the index indicates the gene and 
    the columns indicates corresponding frequecies/smoothness. 

    """
    # Critical parameters
    # Ensure parameters
    if ratio_low_freq == 'infer':
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * \
                                        ratio_low_freq))
    if ratio_high_freq == 'infer':
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * \
                                         ratio_high_freq))
    if adata.shape[0] >= 10000:
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0])))

    if ratio_neighbors == 'infer':
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2))
    else:
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2 \
                                    * ratio_neighbors))
    if adata.shape[0] <= 500:
        num_neighbors = 4

    # Ensure gene index uniquely and all gene had expression
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=1)

    # ************** Construct graph and corresponding matrixs ****************
    lap_mtx = get_laplacian_mtx(adata, num_neighbors=num_neighbors,
                                spatial_key=spatial_info,
                                normalization=normalize_lap)

    # Calculate the eigenvalues and eigenvectors of the Laplace matrix
    np.random.seed(123)
    if num_high_frequency > 0:
        # Fourier modes of low frequency
        eigvals_s, eigvecs_s = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_low_frequency,
                                               which='SM')
        # Fourier modes of high frequency    
        eigvals_l, eigvecs_l = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_high_frequency,
                                               which='LM')
        eigvals = np.concatenate((eigvals_s, eigvals_l))  # eigenvalues
        eigvecs = np.concatenate((eigvecs_s, eigvecs_l), axis=1)  # eigenvectors
    else:
        eigvals_s, eigvecs_s = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_low_frequency,
                                               which='SM')
        eigvecs = eigvecs_s
        eigvals = eigvals_s

    # ************************Graph Fourier Tranform***************************
    # Calculate GFT
    eigvecs = eigvecs.transpose()
    if not ss.issparse(adata.X):
        exp_mtx = preprocessing.scale(adata.X)
    else:
        exp_mtx = preprocessing.scale(adata.X.toarray())
    frequency_array = np.matmul(eigvecs, exp_mtx)
    frequency_array = np.abs(frequency_array)
    # Filter noise peaks
    if filter_peaks == True:
        frequency_array_thres = np.mean(frequency_array, axis=0)
        for j in range(adata.shape[0]):
            frequency_array[frequency_array[:, j] <= \
                            frequency_array_thres[j], j] = 0
    # Spectral domian normalization
    frequency_array = preprocessing.normalize(frequency_array,
                                              norm='l1',
                                              axis=0)

    # ********************** Results of GFT ***********************************
    frequency_df = pd.DataFrame(frequency_array, columns=adata.var_names,
                                index=['low_spec_' + str(low) \
                                       for low in range(1, num_low_frequency + 1)] \
                                      + ['high_spec_' + str(high) \
                                         for high in range(1, num_high_frequency + 1)])
    adata.varm['freq_domain'] = frequency_df.transpose()
    adata.uns['frequencies'] = eigvals

    tmp_adata = sc.AnnData(adata.varm['freq_domain'])
    sc.pp.neighbors(tmp_adata, use_rep='X')
    sc.tl.umap(tmp_adata)
    adata.varm['gft_umap'] = tmp_adata.obsm['X_umap']

    if return_freq_domain:
        return frequency_df


def freq2umap(adata,
              ratio_low_freq='infer',
              ratio_high_freq='infer',
              ratio_neighbors='infer',
              spatial_info=['array_row', 'array_col'],
              normalize_lap=False,
              filter_peaks=False):
    """
    Obtain gene signals in frequency/spectral domain for all genes in 
    adata.var_names and reduce dimension to 2 by UMAP.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinat-
        es could be found in adata.obs or adata.obsm.
    ratio_low_freq : float | "infer", optional
        The ratio_low_freq will be used to determine the number of the FMs with
        low frequencies. Indeed, the ratio_low_freq * sqrt(number of spots) low
        frequecy FMs will be calculated. If 'infer', the ratio_low_freq will be
        set to 1.0. The default is 'infer'.
    ratio_high_freq: float | 'infer', optional
        The ratio_high_freq will be used to determine the number of the FMs with
        high frequencies. Indeed, the ratio_high_freq * sqrt(number of spots) 
        high frequecy FMs will be calculated. If 'infer', the ratio_high_freq 
        will be set to 0. The default is 'infer'.
    ratio_neighbors: float | 'infer', optional
        The ratio_neighbors will be used to determine the number of neighbors
        when contruct the KNN graph by spatial coordinates. Indeed, ratio_neig-
        hobrs * sqrt(number of spots) / 2 indicates the K. If 'infer', the para
        will be set to 1.0. The default is 'infer'.
    spatial_info : list | tupple | str, optional
        The column names of spaital coordinates in adata.obs_keys() or 
        key in adata.obsm_keys. The default is ['array_row','array_col'].
    normalize_lap : bool, optional
        Whether need to normalize laplacian matrix. The default is false.
    filter_peaks: bool, optional
        For calculated vectors/signals in frequency/spectral domian, whether
        filter low peaks to stress the important peaks. The default is False.

    """
    if 'svg_rank' not in adata.var.columns:
        assert KeyError("adata.var['svg_rank'] is not available. Please run\
                        SpaGFT.rank_gene_smooth(adata) firstly.")
    tmp_adata = adata.copy()
    if 'log1p' in adata.uns_keys():
        tmp_adata.uns.pop('log1p')
    tmp_adata.X = adata.raw[:, adata.var_names].X
    sc.pp.log1p(tmp_adata)
    calculate_frequcncy_domain(tmp_adata,
                               ratio_low_freq=ratio_low_freq,
                               ratio_high_freq=ratio_high_freq,
                               ratio_neighbors=ratio_neighbors,
                               spatial_info=spatial_info,
                               filter_peaks=filter_peaks,
                               normalize_lap=normalize_lap,
                               return_freq_domain=False)
    adata.varm['gft_umap_svg'] = tmp_adata.varm['gft_umap']


def find_tissue_module(adata,
                       svg_list='infer',
                       ratio_fms='infer',
                       ratio_neighbors='infer',
                       spatial_info=['array_row', 'array_col'],
                       n_neighbors=15,
                       resolution=1,
                       sub_n_neighbors=15,
                       sub_resolution=0.5,
                       random_state=0,
                       quantile=0.85,
                       **kwargs):
    '''
    After identifying spatially variable genes, this function will group these
    spatially variable genes sharing common spatial patterns.

    Parameters
    ----------
   adata : AnnData
       adata.X is the normalized count matrix. Besides, the spatial coordinat-
       es could be found in adata.obs or adata.obsm; the gft_score should be 
       provided in adata.obs.
    svg_list : 'infer' | list, optional
        Determine SVGs used in clustering. If 'infer', SpaGFT 
        will determine the SVGs automatically according to kneedle
        algorithm.    
    ratio_fms : 'infer' | float optional
        The ratio_low_freq will be used to determine the number of the FMs with
        low frequencies. Indeed, the ratio_low_freq * sqrt(number of spots) low
        frequecy FMs will be calculated. The default is 'infer'.
    ratio_neighbors: float | 'infer', optional
        The ratio_neighbors will be used to determine the number of neighbors
        when contruct the KNN graph by spatial coordinates. Indeed, ratio_neig-
        hobrs * sqrt(number of spots) / 2 indicates the K. If 'infer', the para
        will be set to 1.0. The default is 'infer'.
    spatial_info : list | tupple | str, optional
        The column names of spaital coordinates in adata.obs_keys() or 
        key in adata.obsm_keys. The default is ['array_row','array_col'].
    n_neighbors : int, optional
        The neighbor number of k before clustering when detect tissue modules. 
        The default is 15.
    resolution : float, optional
        The resolution parameter used in Louvain clustering algorithm when 
        detect tissue modules.The default is 1.
    sub_n_neighbors : int, optional
        The neighbor number of k before clustering when detect sub-TMs. 
        The default is 15.
    sub_resolution : float, optional
        The resolution parameter used in Louvain clustering algorithm when 
        detect subTMs The default is 0.5.
    random_state : int, optional
        Random state when run Louvain algorithm. The default is 0.
    quantile : float, optional
        The quantile when binary tissue module pseudo expression. The default 
        is 0.85.
    **kwargs : TYPE
        Parameters in sc.tl.louvain.

    Raises
    ------
    ValueError
        'svg_rank' should in adata.obs. rank_gene_smooth should be implemented
        before this step.

    Returns
    -------
    None.

    '''
    # Find tissue module by grouping Spatially variable genes with similar 
    # spatial patterns acoording to louvain algorithm.
    # Check conditions and determine parameters
    if 'svg_rank' not in adata.var.columns:
        assert KeyError("adata.var['svg_rank'] is not available. Please run\
                        SpaGFT.rank_gene_smooth(adata) firstly.")
    if ratio_fms == 'infer':
        if adata.shape[0] <= 500:
            ratio_fms = 4
        elif adata.shape[0] <= 10000:
            ratio_fms = 2
        else:
            ratio_fms = 1
    gene_score = adata.var.sort_values(by='svg_rank')
    if svg_list == 'infer':
        svg_list = adata.var[adata.var.cutoff_gft_score] \
            [adata.var.qvalue < 0.05].index.tolist()

    tmp_adata = adata.copy()
    if 'log1p' in adata.uns_keys():
        tmp_adata.uns.pop('log1p')
    tmp_adata.X = adata.raw[:, adata.var_names].X
    sc.pp.log1p(tmp_adata)
    calculate_frequcncy_domain(tmp_adata,
                               ratio_low_freq=ratio_fms,
                               ratio_high_freq=0,
                               ratio_neighbors=ratio_neighbors,
                               spatial_info=spatial_info,
                               filter_peaks=False,
                               return_freq_domain=False)
    # Create new anndata to store freq domain information
    gft_adata = sc.AnnData(tmp_adata.varm['freq_domain'])
    # sc.pp.neighbors(gft_adata, n_neighbors=n_neighbors, use_rep='X')
    # sc.tl.umap(gft_adata)
    # adata.varm['gft_umap_tm'] = gft_adata.obsm['X_umap']
    # clustering
    gft_adata = gft_adata[svg_list, :]
    sc.pp.neighbors(gft_adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.umap(gft_adata)
    adata.uns['gft_umap_tm'] = pd.DataFrame(gft_adata.obsm['X_umap'],
                                            index=gft_adata.obs.index,
                                            columns=['UMAP_1', 'UMAP_2'])
    sc.tl.louvain(gft_adata, resolution=resolution, random_state=random_state,
                  **kwargs)
    gft_adata.uns['gft_genes_tm'] = [str(eval(i_tm) + 1) for i_tm in \
                                     gft_adata.obs.louvain.tolist()]
    gft_adata.obs.louvain = [str(eval(i_tm) + 1) for i_tm in \
                             gft_adata.obs.louvain.tolist()]
    gft_adata.obs.louvain = pd.Categorical(gft_adata.obs.louvain)
    adata.var['tissue_module'] = 'None'
    adata.var.loc[gft_adata.obs_names, 'tissue_module'] = gft_adata.obs.louvain
    adata.var['tissue_module'] = pd.Categorical(adata.var['tissue_module'])
    # tm pseudo expression
    all_tms = gft_adata.obs.louvain.cat.categories
    tm_df = pd.DataFrame(0, index=adata.obs_names, columns='tm_' + all_tms)
    for tm in all_tms:
        pseudo_exp = tmp_adata[:,
                     gft_adata.obs.louvain[gft_adata.obs.louvain == tm].index].X.sum(axis=1)
        pseudo_exp = np.ravel(pseudo_exp)
        tm_df['tm_' + str(tm)] = pseudo_exp
    adata.obsm['tm_pseudo_expression'] = tm_df.copy()
    tm_df[tm_df < np.quantile(tm_df, q=0.85, axis=0)] = 0
    tm_df[tm_df >= np.quantile(tm_df, q=0.85, axis=0)] = 1
    tm_df = tm_df.astype(int)
    tm_df = tm_df.astype(str)
    adata.obsm['tm_binary'] = tm_df.copy()
    # obtain freq signal
    freq_signal_tm_df = pd.DataFrame(0, index=tm_df.columns,
                                     columns=tmp_adata.varm['freq_domain'].columns)
    for tm in all_tms:
        tm_gene_list = gft_adata.obs.louvain[gft_adata.obs.louvain == tm].index
        freq_signal = tmp_adata.varm['freq_domain'].loc[tm_gene_list,
                      :].sum(axis=0)
        freq_signal = freq_signal / sum(freq_signal)
        freq_signal_tm_df.loc['tm_' + tm, :] = freq_signal
    adata.uns['freq_signal_tm'] = freq_signal_tm_df

    # sub tm expression clustering
    tm_df = pd.DataFrame(index=adata.obs_names)
    adata.var['sub_TM'] = 'None'
    #  sub tms frequency signals
    freq_signal_subtm_df = pd.DataFrame(
        columns=tmp_adata.varm['freq_domain'].columns)
    for tm in all_tms:
        tm_gene_list = gft_adata.obs.louvain[gft_adata.obs.louvain == tm].index
        sub_gft_adata = gft_adata[tm_gene_list, :].copy()
        sc.pp.neighbors(sub_gft_adata, n_neighbors=sub_n_neighbors, use_rep='X')
        sc.tl.louvain(sub_gft_adata, resolution=sub_resolution, random_state=random_state,
                      **kwargs)
        sub_gft_adata.obs.louvain = [str(eval(i_tm) + 1) for i_tm in \
                                     sub_gft_adata.obs.louvain.tolist()]
        sub_gft_adata.obs.louvain = pd.Categorical(sub_gft_adata.obs.louvain)
        all_sub_tms = sub_gft_adata.obs.louvain.cat.categories
        for sub_tm in all_sub_tms:
            # Obtain pseudo expression
            subTm_gene_list = sub_gft_adata.obs.louvain[sub_gft_adata.obs.louvain == sub_tm].index
            adata.var.loc[subTm_gene_list, 'sub_TM'] = sub_tm
            pseudo_exp = tmp_adata[:, subTm_gene_list].X.sum(axis=1)
            pseudo_exp = np.ravel(pseudo_exp)
            tm_df['tm-' + str(tm) + "_subTm-" + str(sub_tm)] = pseudo_exp
            # Obtain frequency signals
            freq_signal = tmp_adata.varm['freq_domain'].loc[subTm_gene_list,
                          :].sum(axis=0)
            freq_signal = freq_signal / sum(freq_signal)
            freq_signal_subtm_df.loc['tm-' + str(tm) + "_subTm-" + str(sub_tm)
            , :] = freq_signal

    adata.obsm['subTm_pseudo_expression'] = tm_df.copy()
    tm_df[tm_df < np.quantile(tm_df, q=quantile, axis=0)] = 0
    tm_df[tm_df >= np.quantile(tm_df, q=quantile, axis=0)] = 1
    tm_df = tm_df.astype(int)
    tm_df = tm_df.astype(str)
    adata.obsm['subTm_binary'] = tm_df.copy()
    adata.uns['freq_signal_subTM'] = freq_signal_subtm_df.copy()


def get_laplacian_mtx(adata, 
                      num_neighbors=6,
                      spatial_key=['array_row', 'array_col'],
                      normalization=False):
    """
    Obtain the Laplacian matrix or normalized laplacian matrix.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinat-
        es could be found in adata.obs or adata.obsm.
    num_neighbors: int, optional
        The number of neighbors for each node/spot/pixel when contrcut graph. 
        The defalut if 6.
    spatial_key=None : list | string
        Get the coordinate information by adata.obsm[spaital_key] or 
        adata.var[spatial_key]. The default is ['array_row', 'array_col'].
    normalization : bool, optional
        Whether need to normalize laplacian matrix. The default is False.

    Raises
    ------
    KeyError
        The coordinates should be found at adata.obs[spatial_names] or 
        adata.obsm[spatial_key]

    Returns
    -------
    lap_mtx : csr_matrix
        The laplcaian matrix or mormalized laplcian matrix.

    """
    if spatial_key in adata.obsm_keys():
        adj_mtx = kneighbors_graph(adata.obsm[spatial_key], 
                                    n_neighbors=num_neighbors)
    elif set(spatial_key) <= set(adata.obs_keys()):
        adj_mtx = kneighbors_graph(adata.obs[spatial_key], 
                                    n_neighbors=num_neighbors)
    else:
        raise KeyError("%s is not avaliable in adata.obsm_keys" % \
                       spatial_key +  " or adata.obs_keys")

    adj_mtx = nx.adjacency_matrix(nx.Graph(adj_mtx))
    # Degree matrix
    deg_mtx = adj_mtx.sum(axis=1)
    deg_mtx = create_degree_mtx(deg_mtx)
    # Laplacian matrix
    # Whether need to normalize laplcian matrix
    if not normalization:
        lap_mtx = deg_mtx - adj_mtx
    else:
        deg_mtx = np.array(adj_mtx.sum(axis=1)) ** (-0.5)
        deg_mtx = create_degree_mtx(deg_mtx)
        lap_mtx = ss.identity(deg_mtx.shape[0]) - deg_mtx @ adj_mtx @ deg_mtx
    
    return lap_mtx


def find_HVGs(adata, norm_method=None, num_genes=2000):
    """
    Find spatialy variable genes.

    Parameters
    ----------
    adata : AnnData
        The object to save sequencing data.
    norm_method : str | None, optional
        The method used to normalized adata. The default is None.
    num_genes : None | int, optional
        The number of highly variable genes. The default is 2000.

    Returns
    -------
    HVG_list : list
        Highly variable genes.

    """
    # Normalization
    if norm_method == "CPM":
        sc.pp.normalize_total(adata, target_sum=1e5)
        # log-transform
        sc.pp.log1p(adata)
    else:
        pass
    # Find high variable genes using sc.pp.highly_variable_genes() with default 
    # parameters
    sc.pp.highly_variable_genes(adata, n_top_genes=num_genes)
    HVG_list = adata.var.index[adata.var.highly_variable]
    HVG_list = list(HVG_list)
    
    return HVG_list


def create_adjacent_mtx(coor_df, spatial_names = ['array_row', 'array_col'],
                        num_neighbors=4):
    # Transform coordinate dataframe to coordinate array
    coor_array = coor_df.loc[:, spatial_names].values
    coor_array.astype(np.float32) 
    edge_list = []
    num_neighbors += 1
    for i in range(coor_array.shape[0]):
        point = coor_array[i, :]
        distances = np.sum(np.asarray((point - coor_array)**2), axis=1)
        distances = pd.DataFrame(distances, 
                                 index=range(coor_array.shape[0]),
                                 columns=["distance"])
        distances = distances.sort_values(by='distance', ascending=True)
        neighbors = distances[1:num_neighbors].index.tolist()
        edge_list.extend((i, j) for j in neighbors)
        edge_list.extend((j, i) for j in neighbors)
    # Remove duplicates
    edge_list = set(edge_list)
    edge_list = list(edge_list)
    row_index = []
    col_index = []
    row_index.extend(j[0] for j in edge_list)
    col_index.extend(j[1] for j in edge_list)
    
    sparse_mtx = ss.coo_matrix((np.ones_like(row_index), (row_index, col_index)),
                               shape=(coor_array.shape[0], coor_array.shape[0]))
    
    return sparse_mtx

def create_degree_mtx(diag):
    diag = np.array(diag)
    diag = diag.flatten()
    row_index = list(range(diag.size))
    col_index = row_index
    sparse_mtx = ss.coo_matrix((diag, (row_index, col_index)),
                               shape=(diag.size, diag.size))
    
    return sparse_mtx

def gene_clustering_kMeans(frequency_array, n_clusters, reduction=50):
    # Normlization
    frequency_array = preprocessing.StandardScaler().fit_transform(frequency_array)
    
    # Dimension reduction
    if reduction and frequency_array.shape[1] > reduction:
        pca = PCA(n_components=reduction)
        frequency_array = pca.fit_transform(frequency_array)
            
    # Clustering
    kmeans_model = KMeans(n_clusters=n_clusters).fit(frequency_array)
    
    return kmeans_model.labels_

def window_side_bin(adata, shape=(20, 20),
                    spatial_names = ['array_row', 'array_col'],
                    sparse=True):
    # Extract border
    max_x = adata.obs[spatial_names[1]].max()   
    min_x = adata.obs[spatial_names[1]].min()
    max_y = adata.obs[spatial_names[0]].max()
    min_y = adata.obs[spatial_names[0]].min()
    
    # Calculate bin-size
    bin_x = (max_x - min_x) / (shape[0] - 1)
    bin_y = (max_y - min_y) / (shape[1] - 1)
    

    # Create a new dataframe to store new coordinates
    new_coor_df = pd.DataFrame(0, index=adata.obs.index, columns=spatial_names)
    for i in range(adata.shape[0]):
        coor_x = adata.obs.iloc[i][spatial_names][1]
        coor_y = adata.obs.iloc[i][spatial_names][0]
        coor_x = int(np.floor((coor_x - min_x) / bin_x))
        coor_y = int(np.floor((coor_y - min_y) / bin_y))
        new_coor_df.iloc[i, :] = [coor_x, coor_y]
    
    # Merge the spots within the bins
    count_mtx = pd.DataFrame(columns=adata.var_names)
    final_coor_df = pd.DataFrame(columns=spatial_names)
    for i in range(shape[0]):
        for j in range(shape[1]):
            tmp_index = new_coor_df[new_coor_df.iloc[:, 0] == i]
            tmp_index = tmp_index[tmp_index.iloc[:, 1] == j]
            if tmp_index.shape[0] > 0:
                tmp_index = tmp_index.index
                count_mtx.loc["pseudo_" + str(i) + "_" + str(j), 
                              :] = adata[tmp_index, :].X.sum(axis=0)
                final_coor_df.loc["pseudo_" + str(i) + "_" + str(j),
                                  :] = [i, j]
            else:
                count_mtx.loc["pseudo_" + str(i) + "_" + str(j), 
                              :] = 0
                final_coor_df.loc["pseudo_" + str(i) + "_" + str(j),
                                  :] = [i, j]
    
    # Transform it to anndata
    from anndata import AnnData
    if sparse == True:
        from scipy import sparse
        new_adata = AnnData(sparse.coo_matrix(count_mtx))
        new_adata.obs_names = count_mtx.index.tolist()
        new_adata.var_names = count_mtx.columns.tolist()
        new_adata.obs = final_coor_df
    else:
        new_adata = AnnData(count_mtx)  
        new_adata.obs = final_coor_df
        
    return new_adata


def select_svg_normal(gene_score, num_sigma=1):
    mu = np.mean(gene_score['gft_score'])
    sigma = np.std(gene_score['gft_score'])
    gene_score['spatially_variable'] = 0
    gene_score.loc[gene_score['gft_score'] > mu + num_sigma * sigma, 
                   'spatially_variable'] = 1
    
    return gene_score


def select_svg_kmean(gene_score):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2)
    X = gene_score["smooth_score"].tolist()
    X = np.array(X).reshape(-1, 1)
    y_pred = kmeans.fit_predict(X)       
    gene_score['spatially_variable'] = y_pred
    
    return gene_score


def umap_spectral_domain(frequency_array, gene_score, n_dim=2):
    adata_gene = sc.AnnData(frequency_array).T
    sc.pp.pca(adata_gene)
    sc.pp.neighbors(adata_gene)
    sc.tl.umap(adata_gene)
    gene_score = select_svg_normal(gene_score, num_sigma=1)
    gene_score = gene_score.reindex(adata_gene.obs.index)
    adata_gene.obs = gene_score
    sc.pl.umap(adata_gene, color='spatially_variable')
    

def tsne_spectral_domain(frequency_array, gene_score, n_dims=2):
    adata_gene = sc.AnnData(frequency_array).T
    sc.pp.pca(adata_gene)
    sc.pp.neighbors(adata_gene)
    sc.tl.tsne(adata_gene)
    gene_score = select_svg_normal(gene_score, num_sigma=1)
    gene_score = gene_score.reindex(adata_gene.obs.index)
    adata_gene.obs = gene_score
    sc.pl.tsne(adata_gene, color='spatially_variable')


def fms_spectral_domain(frequency_array, gene_score, n_dims=2):
    adata_gene = sc.AnnData(frequency_array).T
    adata_gene.obsm['X_pca'] = frequency_array.transpose()
    gene_score = select_svg_normal(gene_score, num_sigma=3)
    gene_score = gene_score.reindex(adata_gene.obs.index)
    adata_gene.obs = gene_score
    sc.pl.pca(adata_gene, color='spatially_variable')


def pca_spatial_domain(adata, gene_score, n_dims=2):
    adata_gene = adata.copy()
    adata_gene = adata_gene.T
    gene_score = select_svg_normal(gene_score, num_sigma=1)
    adata_gene.obs['spatially_variable'] = 0
    svg_index = (gene_score[gene_score['spatially_variable'] == 1]).index
    adata_gene.obs.loc[svg_index, 'spatially_variable'] = 1
    sc.pp.pca(adata_gene)
    sc.pl.pca(adata_gene, color='spatially_variable')


def umap_spatial_domain(adata, gene_score, n_dim=2):
    adata_gene = adata.copy()
    adata_gene = adata_gene.T
    gene_score = select_svg_normal(gene_score, num_sigma=1)
    adata_gene.obs['spatially_variable'] = 0
    svg_index = (gene_score[gene_score['spatially_variable'] == 1]).index
    adata_gene.obs.loc[svg_index, 'spatially_variable'] = 1
    sc.pp.pca(adata_gene)
    sc.pp.neighbors(adata_gene)
    sc.tl.umap(adata_gene)
    sc.pl.umap(adata_gene, color='spatially_variable')


def tsne_spatial_domain(adata, gene_score, n_dim=2):
    adata_gene = adata.copy()
    adata_gene = adata_gene.T
    gene_score = select_svg_normal(gene_score, num_sigma=1)
    adata_gene.obs['spatially_variable'] = 0
    svg_index = (gene_score[gene_score['spatially_variable'] == 1]).index
    adata_gene.obs.loc[svg_index, 'spatially_variable'] = 1
    sc.pp.pca(adata_gene)
    sc.pp.neighbors(adata_gene)
    sc.tl.tsne(adata_gene)
    sc.pl.tsne(adata_gene, color='spatially_variable')
    
    
def cal_mean_expression(adata, gene_list):
    tmp_adata = adata[:, gene_list].copy()
    if 'log1p' not in adata.uns_keys():
        tmp_adata = sc.pp.log1p(tmp_adata)
    mean_vector = tmp_adata.X.mean(axis=1)
    mean_vector = np.array(mean_vector).ravel()
    
    return mean_vector
