########################################################################################################################
#
#   Support functions
#
#   Description:
#
#       - general support functions that are not specific to one script
#
#   Author: Lars Bollmann
#
#   Created: 21/03/2019
#
#   Structure:
#
#       - VECTOR/MATRIX OPERATIONS
#       - DIMENSIONALITY REDUCTION
#       - NEURAL DATA SPECIFIC
#       - OSCILLATIONS
#       - OTHERS
#
########################################################################################################################

import numpy as np
import math
import scipy.stats as sp
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from sklearn.manifold import MDS
from scipy.spatial import distance
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from matplotlib.colors import ListedColormap
from scipy.stats import ttest_ind
import networkx as nx
import random
import scipy.stats as sps
from scipy.stats import pearsonr, entropy, spearmanr, sem, mannwhitneyu, wilcoxon, ks_2samp, multivariate_normal, \
    zscore
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from scipy.special import factorial, gammaln
from sklearn.metrics import pairwise_distances
from scipy.signal import stft, butter, filtfilt
from matplotlib.cm import hsv
from scipy import stats
import pickle
from itertools import combinations

"""#####################################################################################################################
#   DATA LOADING FUNCTIONS
#####################################################################################################################"""

def read_integers(filename): # faster way of reading clu / res
    with open(filename) as f:
        return np.array([int(x) for x in f])


def read_arrays(filename, skip_first_row=False, sep=" "): # faster way of reading clu / res
    with open(filename) as f:
        if skip_first_row:
            next(f)
        return np.vstack([np.fromstring(x, sep=sep) for x in f])


"""#####################################################################################################################
#   GENERAL FUNCTIONS
#####################################################################################################################"""


def log_func(x, x_0, L=1, k=1):
    return L/(1+np.exp(-k*(x-x_0)))


"""#####################################################################################################################
#   VECTOR/MATRIX OPERATIONS
#####################################################################################################################"""


def unit_vector(vector):
    # ------------------------------------------------------------------------------------------------------------------
    # args: - vector, np.array
    #
    # returns: - the unit vector of the vector
    # ------------------------------------------------------------------------------------------------------------------
    return vector / np.linalg.norm(vector)


def calc_diff(a, b, diff_meas):

    # calculates column-wise difference between two matrices a and b
    D = np.zeros((a.shape[1],b.shape[1]))

    if diff_meas == "jaccard":
        # calculate difference using Jaccard

        # Jaccard similarity
        for i, pop_vec_ref in enumerate(a.T):
            for j, pop_vec_comp in enumerate(b.T):
                D[i, j] = jaccard_score(pop_vec_ref, pop_vec_comp)

        # want difference --> diff_jaccard = 1 - sim_jaccard
        D = 1 - D
        # plt.imshow(D)
        # plt.colorbar()
        # plt.show()
    elif diff_meas == "cos":
        # calculates column-wise difference between two matrices a and b

        # cosine
        for i, pop_vec_ref in enumerate(a.T):
            for j, pop_vec_comp in enumerate(b.T):
                D[i, j] = distance.cosine(pop_vec_ref, pop_vec_comp)
                # if one of the vectors contains only zeros --> division by zero for cosine
                # if math.isnan(D[i,j]):
                #     D[i, j] = 1

    elif diff_meas == "euclidean":
        # calculate difference matrix: euclidean distance

        # euclidean distance
        for i, pop_vec_ref in enumerate(a.T):
            for j, pop_vec_comp in enumerate(b.T):
                    D[i, j] = distance.euclidean(pop_vec_ref, pop_vec_comp)

    elif diff_meas == "L1":
        # calculate difference matrix: euclidean distance

        # euclidean distance
        for i, pop_vec_ref in enumerate(a.T):
            for j, pop_vec_comp in enumerate(b.T):
                    D[i, j] = norm(pop_vec_ref-pop_vec_comp,1)

    return D


def angle_between(v1, v2):
    #  Returns the angle in radians between vectors 'v1' and 'v2'::
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def correlateOneWithMany(one, many):
    """Return Pearson's correlation coef of 'one' with each row of 'many'."""
    pr_arr = np.zeros((many.shape[0], 2), dtype=np.float64)
    pr_arr[:] = np.nan
    for row_num in np.arange(many.shape[0]):
        pr_arr[row_num, :] = sps.pearsonr(one, many[row_num, :])
    return pr_arr


def upper_tri_without_diag(A):
    # compute upper triangle without diangle
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]


def angle_between_col_vectors(data_set):
    # computes angle between two subsequent transitions --> transitions: from one pop-vec to the next
    # returns row vector with angles in radiant
    angle_mat = np.zeros(data_set.shape[1]-1)

    for i,_ in enumerate(data_set.T[:-1,:]):
        angle_mat[i] = angle_between(data_set.T[i+1,:],data_set.T[i,:])

    # calculate relative change between subsequent vectors
    rel_angle_mat = np.zeros(angle_mat.shape[0] - 1)

    for i, _ in enumerate(angle_mat[:-1]):
        rel_angle_mat[i] = abs(1- angle_mat[i+1]/angle_mat[i])

    return angle_mat, rel_angle_mat


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))


def cross_correlate(x, y, shift_array=np.arange(-10, 10)):
    # ------------------------------------------------------------------------------------------------------------------
    # cross-correlates arrays x and y
    #
    # args: - shift array, array: bins by which to shift data
    # ------------------------------------------------------------------------------------------------------------------
    corr = np.zeros(len(shift_array))

    for i, shift in enumerate(shift_array):
        if shift >= 0:
            corr[i], _ = pearsonr(x[shift:], y[:y.shape[0] - shift])
        else:
            shift = np.abs(shift)
            corr[i], _ = pearsonr(y[shift:], x[:x.shape[0] - shift])

    return corr, shift_array


def cross_correlate_matrices(x, y, shift_array=np.arange(-10, 10)):
    # ------------------------------------------------------------------------------------------------------------------
    # cross-correlates columns of matrices x and y --> uses average
    #
    # args: - shift array, array: bins by which to shift data
    # ------------------------------------------------------------------------------------------------------------------
    corr = np.zeros(len(shift_array))

    for i, shift in enumerate(shift_array):
        if shift >= 0:
            corr_ = np.zeros(y.shape[1] - shift)
            x_ = x[:, shift:]
            y_ = y[:, :(y.shape[1] - shift)]
            # go through each pair of x_ and y_ to compute correlations
            for col_id, (col_x_, col_y_) in enumerate(zip(x_.T,y_.T)):
                corr_[col_id], _ = pearsonr(col_x_, col_y_)
            corr[i] = np.mean(corr_)
        else:
            shift = np.abs(shift)
            corr_ = np.zeros(y.shape[1] - shift)
            y_ = y[:, shift:]
            x_ = x[:, :(x.shape[1] - shift)]
            # go through each pair of x_ and y_ to compute correlations
            for col_id, (col_x_, col_y_) in enumerate(zip(x_.T, y_.T)):
                corr_[col_id], _ = pearsonr(col_x_, col_y_)
            corr[i] = np.mean(corr_)

    return corr, shift_array


def covariance_to_correlation(cov_matrix):
    # ------------------------------------------------------------------------------------------------------------------
    # function that converts covariance matrix to correlation
    #
    # args: - cov_matrix, array: covariance matrix
    # ------------------------------------------------------------------------------------------------------------------
    v = np.sqrt(np.diag(cov_matrix))
    outer_v = np.outer(v, v)
    corr_matrix = cov_matrix / outer_v
    corr_matrix[cov_matrix==0] = 0
    return corr_matrix


def compute_sparsity(matrix):
    # ------------------------------------------------------------------------------------------------------------------
    # computes sparsity
    #
    # args: - matrix, array: input matrix
    # ------------------------------------------------------------------------------------------------------------------
    mat = np.nan_to_num(matrix)
    return 1.0 - (np.count_nonzero(mat)/float(mat.size))


def down_sample_array_sum(x, chunk_size, axis=-1):
    # delete last column(s) to be able to downsample array
    to_cut = np.mod(x.shape[1], chunk_size)
    x = x[:,:(x.shape[1]-to_cut)]
    shape = x.shape
    if axis < 0:
        axis += x.ndim
    shape = shape[:axis] + (-1, chunk_size) + shape[axis+1:]
    x = x.reshape(shape)
    return x.sum(axis=axis+1)


def down_sample_array_mean(x, chunk_size, axis=-1):
    # delete last column(s) to be able to downsample array
    to_cut = np.mod(x.shape[1], chunk_size)
    x = x[:,:(x.shape[1]-to_cut)]
    shape = x.shape
    if axis < 0:
        axis += x.ndim
    shape = shape[:axis] + (-1, chunk_size) + shape[axis+1:]
    x = x.reshape(shape)
    return x.mean(axis=axis+1)


def correlations_from_raster(raster, bins_per_corr_matrix=20, only_upper_triangle=False,
                             overlap=None, sliding_window=False):
    """
    computes correlation matrices from raster data

    :param raster: data (n_cells, n_bins)
    :type raster: np.array
    :param bins_per_corr_matrix: how many bins to use to compute one correlation matrix
    :type bins_per_corr_matrix: int
    :param only_upper_triangle: whether to only use off-diagonal elements
    :type only_upper_triangle: bool
    :param overlap: by how many bins to shift data to compute next correlation matrix (overlap)
    :type overlap: int
    :param sliding_window: whether to use sliding window or subsequent windows
    :type sliding_window: bool
    :return: correlation matrices
    :rtype: list
    """

    if only_upper_triangle:
        corr_mat = np.zeros((int(raster.shape[0]*(raster.shape[0]-1)/2), 0))

    else:
        corr_mat = np.zeros((int(raster.shape[0]**2), 0))


    if sliding_window:

        print(" - COMPUTING CORRELATION MATRICES USING SLIDING WINDOWS ....\n")

        for entry in range(int(raster.shape[1] - bins_per_corr_matrix + 1)):
            # TODO: implement overlap

            corr_mat = np.hstack((corr_mat, np.expand_dims(np.corrcoef(raster[:, entry:(entry + bins_per_corr_matrix)]).flatten(),1)))

    else:

        print(" - COMPUTING SUBSEQUENT CORRELATION MATRICES ....\n")

        for entry in range(int(raster.shape[1] / bins_per_corr_matrix)):
            if only_upper_triangle:
                corr_mat = np.hstack((corr_mat, np.expand_dims(upper_tri_without_diag(
                    np.corrcoef(raster[:, entry * bins_per_corr_matrix:
                (entry+1)*bins_per_corr_matrix])).flatten(), 1)))
            else:
                corr_mat = np.hstack((corr_mat, np.expand_dims(np.corrcoef(raster[:, entry * bins_per_corr_matrix:
                (entry+1)*bins_per_corr_matrix]).flatten(), 1)))

    # if one vector is constant (e.g. all zeros) --> pearsonr return np.nan
    # set all nans to zero
    corr_mat = np.nan_to_num(corr_mat, posinf=0, neginf=0)

    # global_correlation_matrix = np.corrcoef(raster)
    # correlation_matrices = global_correlation_matrix - correlation_matrices

    print("  ... DONE\n")

    return corr_mat


def transition_matrix(transitions):
    n = 1+ max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M


"""#####################################################################################################################
#   DIMENSIONALITY REDUCTION
#####################################################################################################################"""


def multi_dim_scaling(act_mat, param_dic):
    # returns fitted multi scale model using defined difference measure

    # if param_dic.dr_method_p1 == "jaccard":
    #     # calculate difference matrix: Jaccard
    #     D = np.zeros([act_mat.shape[1],act_mat.shape[1]])
    #
    #     # Jaccard similarity
    #     for i,pop_vec_ref in enumerate(act_mat.T):
    #         for j,pop_vec_comp in enumerate(act_mat.T):
    #             if param_dic["dr_method_p3"]:
    #                 # make population vectors binary
    #                 vec1 = np.where(pop_vec_ref > 0, 1, 0)
    #                 vec2 = np.where(pop_vec_comp > 0, 1, 0)
    #                 D[i, j] = jaccard_score(vec1, vec2, average="micro")
    #             else:
    #                 D[i,j] = jaccard_score(pop_vec_ref.astype(int), pop_vec_comp.astype(int), average="micro")
    #
    #     # want difference --> diff_jaccard = 1 - sim_jaccard
    #     D = 1 - D
    #     # plt.imshow(D)
    #     # plt.colorbar()
    #     # plt.show()
    # elif param_dic.dr_method_p1 == "cos":
    #     # calculate difference matrix: cosine
    #     D = np.zeros([act_mat.shape[1], act_mat.shape[1]])
    #
    #     # cosine
    #     for i, pop_vec_ref in enumerate(act_mat.T):
    #         for j, pop_vec_comp in enumerate(act_mat.T):
    #                 D[i, j] = distance.cosine(pop_vec_ref, pop_vec_comp)
    #                 # if one of the vectors contains only zeros --> division by zero for cosine
    #                 if math.isnan(D[i,j]):
    #                     D[i, j] = 1
    #
    # elif param_dic.dr_method_p1 == "corr":
    #     # calculate difference matrix: correlation
    #     D = np.zeros([act_mat.shape[1], act_mat.shape[1]])
    #
    #     # cosine
    #     for i, pop_vec_ref in enumerate(act_mat.T):
    #         for j, pop_vec_comp in enumerate(act_mat.T):
    #                 D[i, j] = pearsonr(pop_vec_ref, pop_vec_comp)[0]
    #     # want distance
    #     D = 1-D
    #
    # elif param_dic.dr_method_p1 == "euclidean":
    #     # calculate difference matrix: cosine
    #     D = np.zeros([act_mat.shape[1], act_mat.shape[1]])
    #
    #     # euclidean distance
    #     for i, pop_vec_ref in enumerate(act_mat.T):
    #         for j, pop_vec_comp in enumerate(act_mat.T):
    #                 D[i, j] = distance.euclidean(pop_vec_ref, pop_vec_comp)
    #
    # elif param_dic.dr_method_p1 == "L1":
    #     # calculate difference matrix: L1
    #     D = np.zeros([act_mat.shape[1], act_mat.shape[1]])
    #
    #     # euclidean distance
    #     for i, pop_vec_ref in enumerate(act_mat.T):
    #         for j, pop_vec_comp in enumerate(act_mat.T):
    #                 D[i, j] = norm(pop_vec_ref-pop_vec_comp,1)

    D = pairwise_distances(act_mat.T, metric=param_dic.dr_method_p1)

    model = MDS(n_components=param_dic.dr_method_p2, dissimilarity='precomputed', random_state=1)
    return model.fit_transform(D)


def perform_PCA(act_mat, param_dic):
    # performs PCA
    pca = PCA(n_components=param_dic.dr_method_p2)
    pca_result = pca.fit_transform(act_mat.T)

    return pca_result, str(pca.explained_variance_ratio_)


def perform_TSNE(act_mat, param_dic):
    # performs TSNE
    return TSNE(n_components=param_dic.dr_method_p2, perplexity=5).fit_transform(act_mat.T)


def perform_isomap(act_mat, param_dic):
    # performs isomap on data set
    return Isomap(n_components=param_dic.dr_method_p2).fit_transform(act_mat.T)


"""#####################################################################################################################
#   DECODING
#####################################################################################################################"""


def bayes_likelihood(pop_vec, frm, log_likeli=False):
    # make sure that population vector is not empty!
    if np.count_nonzero(pop_vec) == 0:
        raise Exception("Cannot do decoding: pop. vec. is all zeros!")
    if log_likeli:
        return np.nansum(np.log(np.exp(-frm.T) * frm.T ** pop_vec / factorial(pop_vec)), axis=1)
    else:
        return np.prod(np.exp(-frm.T) * frm.T ** pop_vec / factorial(pop_vec), 1)


def bayes_likelihood_alternative(pop_vec_matrix, rate_maps_flat, log_likeli=False):
    """
    Computes bayes likelihood for e.g. Bayesian decoding

    :param pop_vec_matrix: matrix with column population vectors (n_cells, n_pop_vectors)
    :type pop_vec_matrix: numpy.array
    :param rate_maps_flat: flattened rate maps (nr_spatial_bins, nr_cells)
    :type rate_maps_flat: numpy.array
    :param log_likeli: whether to return log-likelihoods or likelihoods
    :type log_likeli: bool
    :return: (log) likelihoods per population vector in pop_vec_matrix (nr_spatial_bins, nr_pop_vecs)
    :rtype: numpy.array
    """
    # find number of population vectors
    nr_pop_vec = pop_vec_matrix.shape[1]

    # compute exp(-lambda) and repeat
    first_exp = np.repeat(np.exp(-rate_maps_flat)[:, :, np.newaxis], nr_pop_vec, axis=2)

    # compute lambda^k and repeat
    second_exp = np.repeat(rate_maps_flat[:, :, np.newaxis], nr_pop_vec, axis=2)**pop_vec_matrix

    # compute results per cell
    per_cell = first_exp * second_exp / factorial(pop_vec_matrix)

    if log_likeli:
        return np.nansum(np.exp(per_cell), axis=1)
    else:
        return np.prod(per_cell, axis=1)


def decode_using_phmm_modes(mode_means, event_spike_rasters, compression_factor, cell_selection="all"):
    """
    decodes sleep activity of provided events using pHMM modes from awake activity, based on Federico Stella's
    analysis --> saves to dictionary result_list (list with entries for each event:  entry contains array with
    [pop_vec_event, spatial_bin_template_map] probabilities

    @param mode_means: mode means (lambdas) from pHMM awake model
    @type mode_means: numpy.array
    @param event_spike_rasters: list with one raster (several pop. vectors, constant #spikes binning) per event
    @type event_spike_rasters: list
    @param compression_factor: scaling factor between sleep activity and awake activity --> if awake
    model was build with constant time bin size (e.g. 100ms): compute how many spikes on average happen during this
    window size (e.g. mean = 30), check how many #spikes you used for constant #spike binning (e.g. 12) for the
    sleep data --> awake_activity_scaling_factor = 12/30 --> = 0.4
    @type compression_factor: float
    @param cell_selection: whether to only use active cells or all ("all", "split", "only_active", "custom") -->
    if "custom": provide cell ids ("cells_to_use")
    @type cell_selection: str
    @param cells_to_use: which cells to use for decoding
    @type cells_to_use: array
    @return: results_list, one list element per event --> list elements contains array with prob. of each mode for
    each population vector
    @rtype:list
    """
    # load and pre-process template map (computed on awake data)
    # --------------------------------------------------------------------------------------------------------------
    nr_modes = mode_means.shape[0]

    # list to store results per event
    results_list = []

    # main decoding part
    # --------------------------------------------------------------------------------------------------------------

    # check if correct template is used (nr. of cells need to match)
    if not event_spike_rasters[0].shape[0] == mode_means.shape[1]:
        raise Exception("THIS DOES NOT SEEM TO BE THE RIGHT TEMPLATE:\n"
                        "NR. CELLS ARE DIFFERENT IN SPIKE RASTER AND LAMBDA VECTORS!")

    print("  - STARTED DECODING PROCEDURE ...")
    print("    - USING SCALING FACTOR OF "+str(compression_factor))
    # go through all SWR events
    for event_id, spike_raster in enumerate(event_spike_rasters):
        results_per_event = np.zeros((spike_raster.shape[1], nr_modes))
        results_per_event[:] = np.nan

        # go through all population vectors in spike raster
        for pop_vec_id, pop_vec in enumerate(spike_raster.T):

            if not cell_selection == "all":
                # find active/non-active cells
                act_cell_ids = np.argwhere(pop_vec > 0)
                non_act_cell_ids = np.argwhere(pop_vec == 0)

            # go trough all modes
            for mode_id, mean_vec in enumerate(mode_means):
                if cell_selection == "all":
                    # pythonic way
                    rates = compression_factor * mean_vec
                    prob_f = np.sum(np.log(np.power(rates, pop_vec) * np.exp(-1 * rates) /
                                           factorial(pop_vec)))
                elif cell_selection == "only_active":
                    rates = compression_factor * mean_vec[act_cell_ids]
                    prob_f = np.sum(np.log(np.power(rates, pop_vec[act_cell_ids]) *
                                           np.exp(-1 * rates) / factorial(pop_vec[act_cell_ids])))
                elif cell_selection == "split":
                    rates = compression_factor * mean_vec[act_cell_ids]
                    prob_f = np.sum(np.log(np.power(rates, pop_vec[act_cell_ids]) *
                                           np.exp(-1 * rates) / factorial(pop_vec[act_cell_ids])))
                    # compute contribution by silent cells
                    silent_rate = compression_factor * mean_vec[non_act_cell_ids]
                    prob_silent = -1 * np.sum(silent_rate)
                    prob_f += prob_silent

                # save probability for this mode
                results_per_event[pop_vec_id, mode_id] = np.exp(prob_f)
        results_list.append(results_per_event)

    print("  - ... DONE!")
    return results_list


def decode_using_ising_map(template_map, event_spike_rasters, compression_factor, cell_selection="all"):
    """
    decodes sleep activity using ising maps from awake activity

    @param template_map: ising map from awake activity (spatial_bins [x*y] * nr. of cells)
    @type template_map: numpy.array
    @param event_spike_rasters: constant #spike rasters
    @type event_spike_rasters: list
    @param compression_factor: used to scale awake activity (e.g. if constant #spikes = 12, and awake activity
    encoding with 100ms (mean #spikes per 100ms during awake: 30) --> compression factor = 12/30
    @type compression_factor: float
    @param cell_selection: whether to use all cells or subset ("all", "only_active", "split")
    @type cell_selection: str
    @return: result_list, probabilities for each population vector given the template
    @rtype: list
    """
    # --------------------------------------------------------------------------------------------------------------
    # decodes sleep activity using maps from awake activity
    #
    # args:     - decode_with_non_spiking_cells, bool: whether to do decoding with non spiking cells
    #           - template_file_name, string:   name of file containing the dictionary with template from
    #                                           awake/behavioral data
    #           - plot_for_control, bool: whether to plot intermediate results for control
    #
    # returns:  - list with entries for each SWR:  entry contains array with [pop_vec_SWR, spatial_bin_template_map]
    #                                              probabilities
    # --------------------------------------------------------------------------------------------------------------

    template_map_dim = np.array([template_map.shape[1], template_map.shape[2]])
    prob_per_spatial_bin = np.sum(template_map, axis=0)

    # reshape template (do not need 2D information --> just want to go through all spatial bins)
    template_map = template_map.reshape(-1, (template_map.shape[1]*template_map.shape[2]))
    prob_per_spatial_bin = prob_per_spatial_bin.reshape(1, (prob_per_spatial_bin.shape[0]*
                                                            prob_per_spatial_bin.shape[1]))

    # compute firing rate factor (sleep <-> awake activity with different binning & compression)
    # --------------------------------------------------------------------------------------------------------------

    # time_window_factor:   accounts for the different window length of awake encoding and window length
    #                       during sleep (variable window length because of #spike binning)

    # list to store results per SWR
    results_list = []

    print("  - STARTED DECODING PROCEDURE ...")
    print("    - USING SCALING FACTOR OF " + str(compression_factor))

    # main decoding part
    # --------------------------------------------------------------------------------------------------------------
    # go through all events
    for event_id, spike_raster in enumerate(event_spike_rasters):

        results_per_event = np.zeros((spike_raster.shape[1], template_map.shape[1]))

        # go through all population vectors in spike raster
        for pop_vec_id, pop_vec in enumerate(spike_raster.T):

            # find active/non-active cells
            act_cell_ids = np.argwhere(pop_vec > 0)
            non_act_cell_ids = np.argwhere(pop_vec == 0)
            # go trough all spatial bins
            for spatial_bin_id, (prob_firing, template_spatial_bin) in enumerate(zip(prob_per_spatial_bin.T,
                                                                                     template_map.T)):
                prob_f = 0
                # check if there is any cell firing within the current spatial bin --> if yes, continue with
                # decoding, TODO: can filter template_map already after loading to only include bins with firing
                if prob_firing:
                    if cell_selection == "all":
                        # pythonic way
                        rates = compression_factor * template_spatial_bin
                        prob_f = np.sum(np.log(np.power(rates, pop_vec) * np.exp(-1 * rates) /
                                               factorial(pop_vec)))
                    elif cell_selection == "only_active":
                        rates = compression_factor * template_spatial_bin[act_cell_ids]
                        prob_f = np.sum(np.log(np.power(rates, pop_vec[act_cell_ids]) *
                                               np.exp(-1 * rates) / factorial(pop_vec[act_cell_ids])))
                    elif cell_selection == "split":
                        rates = compression_factor * template_spatial_bin[act_cell_ids]
                        prob_f = np.sum(np.log(np.power(rates, pop_vec[act_cell_ids]) *
                                               np.exp(-1 * rates) / factorial(pop_vec[act_cell_ids])))
                        # compute contribution by silent cells
                        silent_rate = compression_factor * template_spatial_bin[non_act_cell_ids]
                        prob_silent = -1 * np.sum(silent_rate)
                        prob_f += prob_silent
                    elif cell_selection == "old":
                        # check if decoding is supposed to be done with non spiking cells as well
                        silent_rate = compression_factor * template_spatial_bin[non_act_cell_ids]
                        prob_f = -1*np.sum(silent_rate)
                        # go trough all active cells
                        for act_cell_id in act_cell_ids:
                            rate = compression_factor * template_spatial_bin[act_cell_id]
                            # compute log-Possionian probability
                            prob_fact = np.log((rate)**(pop_vec[act_cell_id])*np.exp(-1*rate)/
                                               factorial(pop_vec[act_cell_id]))
                            prob_f += prob_fact

                    # save probability for this spatial bin
                    results_per_event[pop_vec_id, spatial_bin_id] = np.exp(prob_f)

        results_list.append(results_per_event)

    print("    - ... DONE!\n" )

    return results_list


def compute_values_from_probabilities(pre_prob_list, post_prob_list, pre_prob_z_list, post_prob_z_list):
    """
    computes values from decoding probabilities

    @param pre_prob_list: list with prob. per event
    @type pre_prob_list: list
    @param post_prob_list: list with prob. per event
    @type post_prob_list: list
    @param pre_prob_z_list: list with prob. per event (z-scored before selecting winner)
    @type pre_prob_z_list: list
    @param post_prob_z_list: list with prob. per event (z-scored before selecting winner)
    @type post_prob_z_list: list
    @return:
    @rtype:
    """
    # per event results
    event_pre_post_ratio = []
    event_pre_post_ratio_z = []
    event_pre_prob = []
    event_post_prob = []
    event_len_seq = []

    # per population vector results
    pop_vec_pre_post_ratio = []
    pre_seq_list = []
    pre_seq_list_z = []
    post_seq_list = []
    pre_seq_list_prob = []
    post_seq_list_prob = []
    pop_vec_post_prob = []
    pop_vec_pre_prob = []

    # go trough all events
    for pre_array, post_array, pre_array_z, post_array_z in zip(pre_prob_list, post_prob_list, pre_prob_z_list,
                                                                post_prob_z_list):
        # make sure that there is any data for the current SWR
        if pre_array.shape[0] > 0:
            pre_sequence = np.argmax(pre_array, axis=1)
            pre_sequence_z = np.argmax(pre_array_z, axis=1)
            pre_sequence_prob = np.max(pre_array, axis=1)
            post_sequence = np.argmax(post_array, axis=1)
            post_sequence_prob = np.max(post_array, axis=1)
            pre_seq_list_z.extend(pre_sequence_z)
            pre_seq_list.extend(pre_sequence)
            post_seq_list.extend(post_sequence)
            pre_seq_list_prob.extend(pre_sequence_prob)
            post_seq_list_prob.extend(post_sequence_prob)

            # check how likely observed sequence is considering transitions from model (awake behavior)
            mode_before = pre_sequence[:-1]
            mode_after = pre_sequence[1:]
            event_len_seq.append(pre_sequence.shape[0])

            # per SWR computations
            # ----------------------------------------------------------------------------------------------
            # arrays: [nr_pop_vecs_per_SWR, nr_time_spatial_time_bins]
            # get maximum value per population vector and take average across the SWR
            if pre_array.shape[0] > 0:
                # save pre and post probabilities
                event_pre_prob.append(np.mean(np.max(pre_array, axis=1)))
                event_post_prob.append(np.mean(np.max(post_array, axis=1)))
                # compute ratio by picking "winner" mode by first comparing z scored probabilities
                # then the probability of the most over expressed mode (highest z-score) is used
                pre_sequence_z = np.argmax(pre_array_z, axis=1)
                prob_pre_z = np.mean(pre_array[:, pre_sequence_z])
                post_sequence_z = np.argmax(post_array_z, axis=1)
                prob_post_z = np.mean(post_array[:, post_sequence_z])
                event_pre_post_ratio_z.append((prob_post_z - prob_pre_z) / (prob_post_z + prob_pre_z))

                # compute ratio using probabilites
                prob_pre = np.mean(np.max(pre_array, axis=1))
                prob_post = np.mean(np.max(post_array, axis=1))
                event_pre_post_ratio.append((prob_post - prob_pre) / (prob_post + prob_pre))
            else:
                event_pre_prob.append(np.nan)
                event_post_prob.append(np.nan)
                event_pre_post_ratio.append(np.nan)

            # per population vector computations
            # ----------------------------------------------------------------------------------------------
            # compute per population vector similarity score
            prob_post = np.max(post_array, axis=1)
            prob_pre = np.max(pre_array, axis=1)
            pop_vec_pre_post_ratio.extend((prob_post - prob_pre) / (prob_post + prob_pre))

            if pre_array.shape[0] > 0:
                pop_vec_pre_prob.extend(np.max(pre_array, axis=1))
                pop_vec_post_prob.extend(np.max(post_array, axis=1))
            else:
                pop_vec_pre_prob.extend([np.nan])
                pop_vec_post_prob.extend([np.nan])

    pop_vec_pre_prob = np.array(pop_vec_pre_prob)
    pop_vec_post_prob = np.array(pop_vec_post_prob)
    pop_vec_pre_post_ratio = np.array(pop_vec_pre_post_ratio)
    pre_seq_list = np.array(pre_seq_list)

    return event_pre_post_ratio, event_pre_post_ratio_z, event_pre_prob, event_post_prob, event_len_seq,\
           pop_vec_pre_post_ratio, pre_seq_list, pre_seq_list_z, post_seq_list, pre_seq_list_prob, \
           post_seq_list_prob, pop_vec_post_prob, pop_vec_pre_prob


def nr_goals_coded_per_cell(rate_maps, occ_map, goal_locations, env_x_min, env_y_min, radius=20, gc_threshold=1,
                            plotting=False):
    """
    determines whether a cell codes for a goal location. Checks z-scored firing around goals (radius defines area)
    and either takes as a threshold max(z_scored firing around goals) > 2 or mean(z_scored firing around goals) > 1
    as a criterium to decide if cell codes for goal

    :param rate_maps: rate maps for all cells (x,y,nr_cells)
    :type rate_maps: np.array
    :param occ_map: occupancy map (x,y)
    :type occ_map: np.array
    :param goal_locations: goal locations
    :type goal_locations: np.array
    :param env_x_min: min x value of environment
    :type env_x_min: float
    :param env_y_min: min y value of environment
    :type env_y_min: float
    :param plotting: whether to plot or not
    :type plotting: bool
    :return: per_cell_nr_goals_coded, number of goals cell coded for
    :rtype: numpy.array
    """

    if plotting:
        # plot sigmoid
        nr_pixels = 500
        test_data = np.arange(0, nr_pixels)
        test_data = test_data / nr_pixels
        plt.plot(test_data, log_func(x=test_data, x_0=0.25, k=10), label="Simgoid")
        plt.vlines(0.25, 0, 1, color="yellow", label="25% of pixels")
        plt.vlines(0.5, 0, 1, color="orange", label="50% of pixels")
        plt.vlines(1, 0, 1, color="red", label="100% of pixels")
        plt.xlabel("#PIXELS")
        plt.ylabel("F(X)")
        plt.legend()
        plt.show()

    per_cell_nr_goals_coded = []
    for cell_rate_map in rate_maps.T:
        goal_coding_estimate = 0
        # only use visited bins
        cell_rate_map[occ_map.T == 0] = np.nan

        cell_rate_map = cell_rate_map.T

        cell_rate_map_z = (cell_rate_map - np.nanmean(cell_rate_map, keepdims=True)) / np.nanstd(cell_rate_map,
                                                                                                 keepdims=True)
        # only for plotting
        cell_rate_map_all_goals = np.zeros((cell_rate_map.shape[0], cell_rate_map.shape[1]))
        cell_rate_map_all_goals[:] = np.nan

        goal_coding_estimate_min = []

        for goal in goal_locations:
            cell_rate_map_gc = np.zeros((cell_rate_map.shape[0], cell_rate_map.shape[1]))
            cell_rate_map_gc[:] = np.nan
            y = np.arange(0, cell_rate_map.shape[0])
            x = np.arange(0, cell_rate_map.shape[1])
            cy = goal[0] - env_x_min
            cx = goal[1] - env_y_min
            # define mask to mark area around goal
            mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < radius ** 2
            # copy only masked z-values
            cell_rate_map_gc[mask] = cell_rate_map_z[mask]
            # copy also to map for all goals
            cell_rate_map_all_goals[mask] = cell_rate_map_z[mask]
            # compute total number of pixels (visited) around the current goal
            nr_pixels = np.count_nonzero(~np.isnan(cell_rate_map_gc))
            # check how many are above goal coding threshold
            above_two_std = np.count_nonzero(np.nan_to_num(cell_rate_map_gc) > gc_threshold)

            if nr_pixels > 0:
                # normalize values to lie between 0 and 1
                above_two_std_norm = above_two_std / nr_pixels

                # estimate goal coding using logistic function
                goal_coding_estimate += log_func(x=above_two_std_norm, x_0=0.0, k=10)

                goal_coding_estimate_min.append(log_func(x=above_two_std_norm, x_0=0.25, k=10))
            else:
                goal_coding_estimate_min.append(0)

        # per_cell_nr_goals_coded.append(min(goal_coding_estimate_min))
        per_cell_nr_goals_coded.append(goal_coding_estimate)

        if plotting == True:

            plt.imshow(cell_rate_map_all_goals.T)
            plt.title("NR. GOALS CODED: " + str(np.round(goal_coding_estimate,2)))
            a = plt.colorbar()
            a.set_label("FIRING RATE Z-SCORED")
            for goal in goal_locations:
                plt.scatter(goal[0] - env_x_min, goal[1] - env_y_min)
            plt.show()

            cell_rate_map_all_goals[cell_rate_map_all_goals>2] = 100
            cell_rate_map_all_goals[cell_rate_map_all_goals <= 2] = 0
            plt.imshow(cell_rate_map_all_goals.T)
            plt.title("NR. GOALS CODED: " + str(np.round(goal_coding_estimate,2))+"\n ABOVE 2 STD")
            for goal in goal_locations:
                plt.scatter(goal[0] - env_x_min, goal[1] - env_y_min)
            plt.show()

    return np.array(per_cell_nr_goals_coded)


def distance_peak_firing_to_closest_goal(rate_maps, goal_locations, env_x_min, env_y_min):
    """
    computes distance of peak firing location to closest goal

    """

    distances_all_cells = []
    for cell_rate_map in rate_maps.T:
        # only use visited bins
        cell_rate_map = cell_rate_map.T

        # only for plotting
        cell_rate_map_all_goals = np.zeros((cell_rate_map.shape[0], cell_rate_map.shape[1]))
        cell_rate_map_all_goals[:] = np.nan

        peak_loc = np.unravel_index(cell_rate_map.argmax(), cell_rate_map.shape)
        peak_loc_y = peak_loc[0]
        peak_loc_x = peak_loc[1]

        distance = []
        for goal in goal_locations:
            # plt.imshow(cell_rate_map)
            # plt.scatter(peak_loc_x, peak_loc_y, facecolor="none", edgecolors="r")
            cell_rate_map_gc = np.zeros((cell_rate_map.shape[0], cell_rate_map.shape[1]))
            cell_rate_map_gc[:] = np.nan
            y = np.arange(0, cell_rate_map.shape[0])
            x = np.arange(0, cell_rate_map.shape[1])
            cy = goal[0] - env_x_min
            cx = goal[1] - env_y_min
            # plt.scatter(cx,cy, color="white")
            distance.append(np.sqrt((peak_loc_x - cx) ** 2 + (peak_loc_y - cy) ** 2))
            # plt.title(np.sqrt((peak_loc_x - cx) ** 2 + (peak_loc_y - cy) ** 2))
            # plt.show()

        distances_all_cells.append(min(distance))
        # define mask to mark area around goal

    return np.array(distances_all_cells)


def nr_goals_coded_subset_of_cells(rate_maps, occ_map, goal_locations, env_x_min, env_y_min, radius=20, gc_threshold=1,
                            plotting=False, combining_by="mean"):
    """
    determines whether a subset of cells code for a a goal location. Checks z-scored firing around goals
    (radius defines area)
    and either takes as a threshold max(z_scored firing around goals) > 2 or mean(z_scored firing around goals) > 1
    as a criterium to decide if cell codes for goal

    :param rate_maps: rate maps for all cells (x,y,nr_cells)
    :type rate_maps: np.array
    :param occ_map: occupancy map (x,y)
    :type occ_map: np.array
    :param goal_locations: goal locations
    :type goal_locations: np.array
    :param env_x_min: min x value of environment
    :type env_x_min: float
    :param env_y_min: min y value of environment
    :type env_y_min: float
    :param plotting: whether to plot or not
    :type plotting: bool
    :return: per_cell_nr_goals_coded, number of goals cell coded for
    :rtype: numpy.array
    """

    if plotting:
        # plot sigmoid
        nr_pixels = 500
        test_data = np.arange(0, nr_pixels)
        test_data = test_data / nr_pixels
        plt.plot(test_data, log_func(x=test_data, x_0=0.25, k=10), label="Simgoid")
        plt.vlines(0.25, 0, 1, color="yellow", label="25% of pixels")
        plt.vlines(0.5, 0, 1, color="orange", label="50% of pixels")
        plt.vlines(1, 0, 1, color="red", label="100% of pixels")
        plt.xlabel("#PIXELS")
        plt.ylabel("F(X)")
        plt.legend()
        plt.show()


    if combining_by == "mean":
        cell_rate_map = np.mean(rate_maps, axis=2)
    elif combining_by == "sum":
        cell_rate_map = np.sum(rate_maps, axis=2)

    goal_coding_estimate = 0
    # only use visited bins
    cell_rate_map[occ_map == 0] = np.nan

    cell_rate_map = cell_rate_map

    cell_rate_map_z = (cell_rate_map - np.nanmean(cell_rate_map, keepdims=True)) / np.nanstd(cell_rate_map,
                                                                                             keepdims=True)
    # only for plotting
    cell_rate_map_all_goals = np.zeros((cell_rate_map.shape[0], cell_rate_map.shape[1]))
    cell_rate_map_all_goals[:] = np.nan

    goal_coding_estimate_min = []

    for goal in goal_locations:
        cell_rate_map_gc = np.zeros((cell_rate_map.shape[0], cell_rate_map.shape[1]))
        cell_rate_map_gc[:] = np.nan
        y = np.arange(0, cell_rate_map.shape[0])
        x = np.arange(0, cell_rate_map.shape[1])
        cy = goal[0] - env_x_min
        cx = goal[1] - env_y_min
        # define mask to mark area around goal
        mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < radius ** 2
        # copy only masked z-values
        cell_rate_map_gc[mask] = cell_rate_map_z[mask]
        # copy also to map for all goals
        cell_rate_map_all_goals[mask] = cell_rate_map_z[mask]
        # compute total number of pixels (visited) around the current goal
        nr_pixels = np.count_nonzero(~np.isnan(cell_rate_map_gc))
        # check how many are above goal coding threshold
        above_two_std = np.count_nonzero(np.nan_to_num(cell_rate_map_gc) > gc_threshold)
        # normalize values to lie between 0 and 1
        above_two_std_norm = above_two_std / nr_pixels

        # estimate goal coding using logistic function
        goal_coding_estimate += log_func(x=above_two_std_norm, x_0=0.0, k=10)

        goal_coding_estimate_min.append(log_func(x=above_two_std_norm, x_0=0.25, k=10))

    return goal_coding_estimate


def goal_coding_per_cell(rate_maps, occ_map, goal_locations, env_x_min, env_y_min, radius=20,plotting=False):
    """
    determines whether a cell codes for a goal location. Compares mean firing around goals with mean firing outside.
    Checks where it gets the highest mean inside / mean outside value (e.g. around one goal, all goals, two goals)

    :param rate_maps: rate maps for all cells (x,y,nr_cells)
    :type rate_maps: np.array
    :param occ_map: occupancy map (x,y)
    :type occ_map: np.array
    :param goal_locations: goal locations
    :type goal_locations: np.array
    :param env_x_min: min x value of environment
    :type env_x_min: float
    :param env_y_min: min y value of environment
    :type env_y_min: float
    :param plotting: whether to plot or not
    :type plotting: bool
    :return: per_cell_goal_coding, if entry is > 1: goals are more coded than overall environment
    :rtype: numpy.array
    """
    # look at stable cells first
    per_cell_goal_coding = []
    per_cell_nr_of_goals = []
    for cell_rate_map in rate_maps.T:
        # set all bins that were not visited to NaN
        cell_rate_map[occ_map.T == 0] = np.nan
        mean_firing = np.nanmean(cell_rate_map)

        cell_rate_map = cell_rate_map.T
        firing_around_goals = []
        firing_outside_goals = []
        firing_outside_all_goals = np.copy(cell_rate_map)
        for goal in goal_locations:
            y = np.arange(0, cell_rate_map.shape[0])
            x = np.arange(0, cell_rate_map.shape[1])
            cy = goal[0] - env_x_min
            cx = goal[1] - env_y_min
            # define mask to mark area around goal
            mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < radius ** 2
            # cell_rate_map_gc[mask]=20
            firing_around_goals.append(np.nanmean(cell_rate_map[mask]))
            outside_goal = np.copy(cell_rate_map)
            outside_goal[mask] = np.nan
            firing_outside_goals.append(np.nanmean(outside_goal))
            firing_outside_all_goals[mask] = np.nan

        firing_outside_all_goals_mean = np.nanmean(firing_outside_all_goals)
        firing_around_goals = np.array(firing_around_goals)
        firing_outside_goals = np.array(firing_outside_goals)

        # if cell only fires around goals
        if firing_outside_all_goals_mean == 0:
            firing_outside_all_goals_mean += 0.00000000001
        if not np.all(firing_outside_goals):
            firing_outside_goals[firing_outside_goals == 0] = np.nan

        # check first single goal coding
        single_goal = firing_around_goals/firing_outside_goals

        # check goal coding for 2 goals
        two_goals = []
        for doublets in combinations([0,1,2,3], 2):
                mean_within = np.mean([firing_around_goals[doublets[0]], firing_around_goals[doublets[1]]])
                inside_other_goals = np.delete(firing_around_goals, [doublets[0], doublets[1]])
                mean_outside = np.mean(np.hstack((inside_other_goals, firing_outside_all_goals_mean)))
                two_goals.append(mean_within/mean_outside)
        two_goals = np.array(two_goals)

        # check goal coding for 3 goals
        three_goals = []
        for triplet in combinations([0,1,2,3], 3):
            mean_within = np.mean([firing_around_goals[triplet[0]], firing_around_goals[triplet[1]],firing_around_goals[triplet[2]]])
            inside_other_goals = np.delete(firing_around_goals, [triplet[0], triplet[1], triplet[2]])
            mean_outside = np.mean(np.hstack((inside_other_goals, firing_outside_all_goals_mean)))
            three_goals.append(mean_within / mean_outside)

        three_goals = np.array(three_goals)

        # check coding for all four goals
        four_goals = np.mean(firing_around_goals)/firing_outside_all_goals_mean

        # combine all and pick maximum
        all_gc = np.hstack((np.max(single_goal), np.max(two_goals), np.max(three_goals), np.max(four_goals)))
        best_gc = np.max(all_gc)
        nr_goals = np.argmax(all_gc) + 1

        if plotting == True:
            plt.imshow(cell_rate_map.T)
            plt.title("GOAL CODING: " + str(np.round(best_gc, 2))+", #GOALS: "+str(nr_goals))
            plt.show()

        per_cell_goal_coding.append(best_gc)
        per_cell_nr_of_goals.append(nr_goals)

    return np.array(np.nan_to_num(per_cell_goal_coding))


def collective_goal_coding(normalized_rate_map, goal_locations, env_x_min, env_y_min, spatial_resolution,
                           max_radius=20, ring_width=2):

    for g_l in goal_locations:
        plt.scatter((g_l[0] - env_x_min) / spatial_resolution, (g_l[1] - env_y_min) / spatial_resolution,
                    color="white", label="Goal locations")
    # plt.xlim(55,260)
    plt.show()

    all_goals = []
    for goal in goal_locations:
        y = np.arange(0, normalized_rate_map.shape[0])
        x = np.arange(0, normalized_rate_map.shape[1])
        cy = (goal[0] - env_x_min) / spatial_resolution
        cx = (goal[1] - env_y_min) / spatial_resolution
        # define mask to mark area around goal

        per_goal_coding = []

        for radius in np.arange(0, max_radius, ring_width):
            mask = (radius ** 2 < (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2) &\
                    ((x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < (radius+ring_width) ** 2)

            map_copy = np.copy(normalized_rate_map)
            map_copy[~mask] = np.nan

            # plt.imshow(map_copy.T)
            # a = plt.colorbar()
            # a.set_label("Sum firing rate / normalized to 1")
            # plt.show()
            per_goal_coding.append(np.nanmean(map_copy))

        all_goals.append(per_goal_coding)

    return all_goals


"""#####################################################################################################################
#   NEURAL DATA SPECIFIC
#####################################################################################################################"""


def simulate_poisson_spiking(nr_neurons, time_bins, nr_states):
    # ------------------------------------------------------------------------------------------------------------------
    # generates samples with poisson emissions
    #
    # params:   - nr_neurons, int
    #           - time_bins, int: number of time bins
    #           - nr_states, int: number of states
    #
    # ------------------------------------------------------------------------------------------------------------------

    lambda_states = np.random.randint(0,10, size=((nr_neurons, nr_states)))

    mean_firing = []
    state_seq = np.zeros(time_bins)

    for i in range(time_bins):
        state = np.random.randint(0, nr_states)
        mean_firing.append(lambda_states[:, state])
        state_seq[i] = state

    mean_firing = np.array(mean_firing).T
    poisson_firing = np.random.poisson(mean_firing)

    return poisson_firing, state_seq


def calc_pop_vector_entropy(act_mat):
    # calculates shannon entropy for each population vector in act_mat
    pop_vec_entropy = np.zeros(act_mat.shape[1])
    # calculate entropy
    for i,pop_vec in enumerate(act_mat.T):
        # add small value because of log
        pop_vec_entropy[i] = sp.entropy(pop_vec+0.000001)
    return pop_vec_entropy


def synchronous_activity(act_mat):
    # computes %of cells that are active per time bin
    syn = np.zeros(act_mat.shape[1])
    # calculate entropy
    for i, pop_vec in enumerate(act_mat.T):
        syn[i] = np.count_nonzero(pop_vec)/act_mat.shape[0]
    return syn


def pop_vec_diff(data_set):
    # computes difference vectors between subsequent population vectors and returns matrix of difference vectors
    # calculate transition vector between two subsequent population states --> rows: cells, col: time bins
    diffMat = np.zeros((data_set.shape[0],data_set.shape[1]-1))
    for i,pop_vec in enumerate(data_set.T[:-1,:]):
        diffMat.T[i,:] = data_set.T[i+1,:] - data_set.T[i,:]
    return diffMat


def pop_vec_dist(data_set, measure):
    # computes distance between column vectors of data set
    # returns row vector with euclidean distances
    dist_mat = np.zeros(data_set.shape[1]-1)

    for i, _ in enumerate(data_set.T[:-1,:]):
        if measure == "euclidean":
            dist_mat[i] = distance.euclidean(data_set.T[i+1, :], data_set.T[i, :])
        elif measure == "cos":
            dist_mat[i] = distance.cosine(data_set.T[i + 1, :], data_set.T[i, :])
        elif measure == "L1":
            dist_mat[i] = norm(data_set.T[i + 1, :] - data_set.T[i, :],1)

    # calculate relative change between subsequent vectors
    rel_dist_mat = np.zeros(dist_mat.shape[0] - 1)

    for i, _ in enumerate(dist_mat[:-1]):
        rel_dist_mat[i] = abs(1- dist_mat[i+1]/dist_mat[i])

    return dist_mat, rel_dist_mat


def find_hse(x, std_above_avg=3, duration=[50, 750], min_perc_cells_active=0.1, plotting=False):
    # --------------------------------------------------------------------------------------------------------------
    # finds high synchrony events based on firing rate, #cells active and duration of event
    #
    # args:         - x: population vectors (rows: cells, columns: time bins)
    #               - std_above_avg, int: how many std above the average is used as a cutoff (usually = 3)
    #               - duration, list: duration of event to be classified as a high synchrony event (50-750ms)
    #               - min_perc_cells_active, float: how many cells need to be active simultaneously to classify
    #                 event as hse (in percent --> 0.1: 10% active)
    #
    # returns:      - array with time bins that are classified as high synchrony events
    # --------------------------------------------------------------------------------------------------------------

    # TODO: should I include duration of HSE during sleep? Usually I use time bins of > 50ms

    # compute std above average
    mean_act = np.mean(x, axis=0)
    z_scored = (mean_act - np.mean(mean_act)) / (np.std(mean_act))

    # compute % cells active
    bool_act = x.astype(bool)
    perc_active = np.sum(bool_act, axis=0) / x.shape[0]

    # identify hse candidates using above criteria
    hse_bool_mask = np.logical_and(perc_active > min_perc_cells_active, z_scored > std_above_avg)
    hse_ind = np.where(hse_bool_mask)

    if plotting:
        plt.imshow(x, interpolation='nearest', aspect='auto')
        mask = np.zeros(x.shape)
        mask[:, hse_ind] = 1

        plt.imshow(mask, interpolation='nearest', aspect='auto', alpha=0.1)
        plt.show()

    return hse_ind


def down_sample_modify_raster(raster, time_bin_size, binning_method, time_bin_size_after):
    # --------------------------------------------------------------------------------------------------------------
    # down samples existing raster and changes format
    #
    # args:         - raster, np.array [cells, time points]
    #               - time_bin_size, float (in sec): resolution of input data
    #               - binning_method, str: which binning to use ("temporal_spike", "temporal_binary", "temporal")
    #               - time_bin_size_after, float: which time bin size (in sec) to use
    #
    # returns:      - new_raster, binned data at time_bin_size_after resolution
    # --------------------------------------------------------------------------------------------------------------

    # check if data is actually at higher resolution --> if not we would need to interpolate
    if time_bin_size_after > time_bin_size:
        raise Exception("TO DOWN SAMPLE RASTER --> REQUESTED RESOLUTION MUST BE SMALLER THAN TIME BIN SIZE!")

    print("TRANSFORMING RASTER --> NEW FORMAT: " + binning_method + " - " + str(time_bin_size) + "s")

    # base raster has 10ms windows --> time_bin_scaler tells how many of these 10ms windows to combine to a new
    # window (e.g. 20ms windows --> time_bin_scaler = 2), width is given in seconds --> base raster: 0.01s
    time_bin_scaler = int(time_bin_size_after / time_bin_size)
    new_raster = np.zeros((raster.shape[0], int(raster.shape[1] / time_bin_scaler)))

    if binning_method == "temporal_spike":
        # down sample spikes by combining multiple bins
        for i in range(new_raster.shape[1]):
            new_raster[:, i] = np.sum(raster[:, (i * time_bin_scaler): ((1 + i) * time_bin_scaler)], axis=1)

    elif binning_method == "temporal_binary":
        # doesn't count spikes --> only sets 0 (no firing) or 1 (firing) for each cell/column
        # down sample spikes by combining multiple bins
        for i in range(new_raster.shape[1]):
            new_raster[:, i] = np.sum(raster[:, (i * time_bin_scaler): ((1 + i) * time_bin_scaler)], axis=1)
        # make data binary
        new_raster = np.array(new_raster, dtype=bool)
    elif binning_method == "temporal":
        # divides number of spikes per time bin by the time bin width --> unit: spikes/second
        raise Exception("NOT YET IMPLEMENTED")

    return new_raster


def compute_spatial_information(rate_maps, occ_map, only_visited=True):
    orig_rate_maps = rate_maps
    occ_map_orig = occ_map
    # reshape --> 2D bins to vector
    rate_maps = np.reshape(rate_maps,
                               (rate_maps.shape[0] * rate_maps.shape[1], rate_maps.shape[2]))
    occ_map = np.reshape(occ_map, (occ_map.shape[0] * occ_map.shape[1]))

    # compute occupancy probabilities
    prob_occ = occ_map / occ_map.sum()

    if only_visited:
        # only use bins that were visited
        prob_occ = occ_map[occ_map > 0] / occ_map[occ_map > 0].sum()
        rate_maps = rate_maps[occ_map > 0, :]

    # initialize sparsity and skaggs info list
    sparsity_list = []
    skaggs_info_per_sec_list = []
    skaggs_info_per_spike_list = []

    for cell_id, (rate_map, orig_rate_map) in enumerate(zip(rate_maps.T, orig_rate_maps.T)):
        if np.count_nonzero(rate_map) == 0:
            sparse_cell = np.nan
            skaggs_info_per_sec = np.nan
            skaggs_info_per_spike = np.nan
        else:
            # compute sparsity
            # sparse_cell = np.round(np.mean(rate_map_pre) ** 2 / np.mean(np.square(rate_map_pre)), 4)

            sparse_cell = (np.sum(prob_occ * rate_map) ** 2) / np.sum(prob_occ * (rate_map ** 2))

            # find good bins so that there is no problem with the log
            good_bins = (rate_map / rate_map.mean() > 0.0000001)

            mean_rate = np.sum(rate_map[good_bins] * prob_occ[good_bins])
            skaggs_info_per_sec = np.sum(rate_map[good_bins] * prob_occ[good_bins] *
                                         np.log(rate_map[good_bins] / mean_rate))
            skaggs_info_per_spike = np.sum(rate_map[good_bins] / mean_rate * prob_occ[good_bins] *
                                           np.log(rate_map[good_bins] / mean_rate))

        sparsity_list.append(sparse_cell)
        skaggs_info_per_sec_list.append(skaggs_info_per_sec)
        skaggs_info_per_spike_list.append(skaggs_info_per_spike)


    # sparsity = np.array(sparsity_list)
    # skaggs_info_per_sec = np.array(skaggs_info_per_sec_list)
    # skaggs_info_per_spike = np.array(skaggs_info_per_spike_list)

    # min_spars = np.nanmin(sparsity)
    # max_spars = np.nanmax(sparsity-min_spars)
    # min_sec = np.nanmin(skaggs_info_per_sec)
    # max_sec = np.nanmax(skaggs_info_per_sec-min_sec)
    # min_spike = np.nanmin(skaggs_info_per_spike)
    # max_spike = np.nanmax(skaggs_info_per_spike-min_spike)
    #
    #
    # for cell_id, (rate_map, orig_rate_map) in enumerate(zip(rate_maps.T, orig_rate_maps.T)):
    #     if np.count_nonzero(rate_map) == 0:
    #         sparse_cell = np.nan
    #         skaggs_info_per_sec = np.nan
    #         skaggs_info_per_spike = np.nan
    #     else:
    #         # compute sparsity
    #         # sparse_cell = np.round(np.mean(rate_map_pre) ** 2 / np.mean(np.square(rate_map_pre)), 4)
    #
    #         sparse_cell = (np.sum(prob_occ * rate_map) ** 2) / np.sum(prob_occ * (rate_map ** 2))
    #
    #         # find good bins so that there is no problem with the log
    #         good_bins = (rate_map / rate_map.mean() > 0.0000001)
    #
    #         mean_rate = np.sum(rate_map[good_bins] * prob_occ[good_bins])
    #         skaggs_info_per_sec = np.sum(rate_map[good_bins] * prob_occ[good_bins] *
    #                                      np.log(rate_map[good_bins] / mean_rate))
    #         skaggs_info_per_spike = np.sum(rate_map[good_bins] / mean_rate * prob_occ[good_bins] *
    #                                        np.log(rate_map[good_bins] / mean_rate))
    #
    #         per_bin_sec = rate_map[good_bins] * prob_occ[good_bins] * np.log(rate_map[good_bins] / mean_rate)
    #         per_bin_spike = rate_map[good_bins] / mean_rate * prob_occ[good_bins] * np.log(
    #             rate_map[good_bins] / mean_rate)
    #
    #         p_pre_sec = 1. * np.arange(per_bin_sec.shape[0]) / (per_bin_sec.shape[0] - 1)
    #         p_pre_spike = 1. * np.arange(per_bin_spike.shape[0]) / (per_bin_spike.shape[0] - 1)
    #         plt.subplot(2, 1, 1)
    #         plt.plot(np.sort(per_bin_sec), p_pre_sec, label="per_sec, " + str(np.round(skaggs_info_per_sec,3)))
    #         plt.plot(np.sort(per_bin_spike), p_pre_spike, label="per_spike, " + str(np.round(skaggs_info_per_spike, 3)))
    #         plt.legend()
    #         plt.subplot(2, 2, 3)
    #         orig_rate_map[occ_map_orig == 0] = np.nan
    #         plt.imshow(orig_rate_map)
    #         plt.colorbar()
    #         plt.subplot(2, 2, 4)
    #         plt.scatter(0, 1-(sparse_cell-min_spars)/max_spars)
    #         plt.scatter(1, (skaggs_info_per_sec-min_sec)/max_sec)
    #         plt.scatter(2, (skaggs_info_per_spike-min_spike)/max_spike)
    #         plt.xticks([0,1,2],["1-sparse", "per_sec", "per_spike"])
    #         plt.ylim(0,1)
    #         plt.show()
    #         print("HERE")


    return np.array(sparsity_list), np.array(skaggs_info_per_sec_list), np.array(skaggs_info_per_spike_list)


"""#####################################################################################################################
#   CLUSTERING
#####################################################################################################################"""


def evaluate_clustering_fit(real_data, samples, binning, time_bin_size, plotting=False):
    # --------------------------------------------------------------------------------------------------------------
    # evaluates fit by comparing basic statistics
    #
    # args:     - real_data, array: [cells, time bins]
    #           - samples, array: [cells, time bins]
    #           - binning, string: which type of binning was used for real_data & samples (e.g. temporal_spike)
    #           - time_bin_size, float: which bin size in seconds was used for real_data and samples (e.g. 0.1)
    #           - plotting, bool: whether to plot results
    #
    # --------------------------------------------------------------------------------------------------------------

    nr_samples = samples.shape[1]
    nr_real_data = real_data.shape[1]

    # CORRELATION VALUES
    # --------------------------------------------------------------------------------------------------------------

    corr_dic = {
        "samples": np.nan_to_num(np.corrcoef(samples)),
        "real": np.nan_to_num(np.corrcoef(real_data)),
    }

    # corr_dic["mwu"] = mannwhitneyu(corr_dic["samples"].flatten(), corr_dic["real"].flatten())
    corr_dic["wc"] = wilcoxon(corr_dic["samples"].flatten(), corr_dic["real"].flatten())
    corr_dic["norm_diff"] = np.linalg.norm(corr_dic["samples"] - corr_dic["real"])
    corr_dic["corr"] = pearsonr(corr_dic["samples"].flatten(), corr_dic["real"].flatten())

    # only use upper wihtout diagonal
    corr_dic["samples_triangle"] = upper_tri_without_diag(corr_dic["samples"])
    corr_dic["real_triangle"] = upper_tri_without_diag(corr_dic["real"])
    corr_dic["corr_triangles"] = pearsonr(corr_dic["samples_triangle"].flatten(), corr_dic["real_triangle"].flatten())
    corr_dic["corr_triangles_spear"] = spearmanr(corr_dic["samples_triangle"].flatten(),
                                          corr_dic["real_triangle"].flatten())
    # corr_dic["ks"] = ks_2samp(corr_dic["samples"].flatten(), corr_dic["real"].flatten())

    # K-STATISTIC
    # --------------------------------------------------------------------------------------------------------------

    k_dic = {
        "samples": synchronous_activity(samples),
        "real": synchronous_activity(real_data),
    }

    k_dic["mwu"] = mannwhitneyu(k_dic["samples"].flatten(), k_dic["real"].flatten())
    k_dic["ks"] = ks_2samp(k_dic["samples"].flatten(), k_dic["real"].flatten())
    k_dic["ttest"] = ttest_ind(k_dic["samples"].flatten(), k_dic["real"].flatten())
    k_dic["diff_med"] = abs(np.median(k_dic["samples"].flatten())-np.median(k_dic["real"].flatten()))

    # MEAN FIRING RATES
    # --------------------------------------------------------------------------------------------------------------

    mean_dic = {
        "samples": np.sum(samples, axis=1)/(nr_samples*time_bin_size),
        "real": np.sum(real_data, axis=1)/(nr_real_data*time_bin_size),
    }

    mean_dic["diff"] = mean_dic["samples"] - mean_dic["real"]
    # mean_dic["mwu"] = mannwhitneyu(mean_dic["samples"], mean_dic["real"])
    mean_dic["wc"] = wilcoxon(mean_dic["samples"], mean_dic["real"])
    # mean_dic["ks"] = ks_2samp(mean_dic["samples"], mean_dic["real"])
    mean_dic["norm_diff"] = np.linalg.norm(mean_dic["samples"] - mean_dic["real"])
    mean_dic["corr"] = pearsonr(mean_dic["samples"].flatten(), mean_dic["real"].flatten())
    mean_dic["corr_spear"] = spearmanr(mean_dic["samples"].flatten(), mean_dic["real"].flatten())

    if not plotting:
        # return results
        return mean_dic, corr_dic, k_dic
    else:
        # plot results

        # mean firing rates
        # ----------------------------------------------------------------------------------------------------
        plt.subplot(2, 2, 1)
        plt.hist(mean_dic["samples"].flatten(), histtype='stepfilled', alpha=0.8, density=True, bins=40,
                 color="#990000",
                 label="SAMPLES")
        plt.hist(mean_dic["real"].flatten(), histtype='stepfilled', alpha=0.3, density=True, bins=40, color="yellow",
                 label="REAL DATA")
        plt.title("AVERAGE CELL FIRING RATES")
        plt.xlabel("AVG. FIRING RATE / Hz")
        plt.ylabel("COUNTS")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.hist(mean_dic["diff"])
        plt.title("DIFF. CELL FIRING RATES")
        plt.xlabel("DIFF. AVG. FIRING RATE / Hz")
        plt.ylabel("COUNTS")

        plt.subplot(2, 2, 3)
        plt.scatter(mean_dic["samples"].flatten(), mean_dic["real"].flatten())
        plt.xlabel("MEAN FIRING RATE (SAMPLES)")
        plt.ylabel("MEAN FIRING RATE (REAL)")
        plt.plot([0, max(np.maximum(mean_dic["samples"].flatten(), mean_dic["real"].flatten()))],
                 [0,max(np.maximum(mean_dic["samples"].flatten(),mean_dic["real"].flatten()))],
                 linestyle="dashed", c="w")
        plt.title("PEARSON CORR: " + str(round(mean_dic["corr"][0], 5)))

        plt.subplot(2, 2, 4)
        # plt.text(0, 0.8, "* MWU p-value: " + str(round(mean_dic["mwu"][1], 8)), c="w")
        plt.text(0, 0.6, "* WC p-value: " + str(round(mean_dic["wc"][1], 8)), c="w")
        # plt.text(0, 0.4, "* KS p-value: " + str(round(mean_dic["ks"][1], 8)), c="w")
        # plt.text(0, 0.3, "p-value < 0.05 --> not the same")
        plt.axis("off")

        plt.show()

        # plot k-statistic
        # ------------------------------------------------------------------------------------------------------------------

        # plot real and sample data
        plt.subplot(2, 2, 1)
        plt.imshow(samples, interpolation='nearest', aspect='auto')
        if binning == "temporal_spike":
            plt.colorbar()
        plt.title("SAMPLES")
        plt.subplot(2, 2, 2)
        plt.imshow(real_data, interpolation='nearest', aspect='auto')
        if binning == "temporal_spike":
            plt.colorbar()
        plt.title("REAL DATA")

        plt.subplot(2, 2, 3)
        plt.hist(k_dic["samples"].flatten(), histtype='stepfilled', alpha=0.8, density=True, bins=40, color="#990000",
                 label="SAMPLES")

        plt.hist(k_dic["real"].flatten(), histtype='stepfilled', alpha=0.3, density=True, bins=40, color="yellow",
                 label="REAL DATA")
        plt.title("<k> STATISTIC")
        plt.xlabel("CELLS ACTIVE")
        plt.ylabel("COUNTS")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, "* MWU p-value: " + "{:.2e}".format(k_dic["mwu"][1]), c="w")
        # plt.text(0, 0.8, "* WC p-value: " + str(round(wc[1], 2)), c="w")
        plt.text(0.1, 0.4, "p-value < 0.05 --> not the same")
        plt.text(0.1, 0.6, "* KS p-value: " + "{:.2e}".format(k_dic["ks"][1]), c="w")
        plt.text(0.1,0.2, "Parametric test:")
        plt.text(0.1, 0.0, "* T-test: " + "{:.2e}".format(k_dic["ttest"][1]), c="w")
        plt.axis("off")

        plt.show()

        # plot corr values
        # ------------------------------------------------------------------------------------------------------------------

        # plot real and sample data
        plt.subplot(2, 2, 1)
        plt.imshow(corr_dic["samples"], interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        plt.title("CORR. MAT: SAMPLES")
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.imshow(corr_dic["real"], interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        plt.title("CORR. MAT: REAL DATA")
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.hist(corr_dic["samples"].flatten(), histtype='stepfilled', alpha=0.8, density=True, bins=40,
                 color="#990000",
                 label="SAMPLES")
        plt.xlabel("CORR. VAL.")
        plt.ylabel("COUNTS")

        plt.hist(corr_dic["real"].flatten(), histtype='stepfilled', alpha=0.3, density=True, bins=40, color="yellow",
                 label="REAL DATA")
        plt.title("CORR. VALUES")
        plt.xlabel("CORR.VAL")
        plt.ylabel("COUNTS")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.scatter(corr_dic["samples_triangle"].flatten(), corr_dic["real_triangle"].flatten())
        plt.xlabel("CORR. VAL. (SAMPLES)")
        plt.ylabel("CORR. VAL (REAL)")
        plt.plot([0, max(np.maximum(corr_dic["samples_triangle"].flatten(), corr_dic["real_triangle"].flatten()))],
                 [0,max(np.maximum(corr_dic["samples_triangle"].flatten(), corr_dic["real_triangle"].flatten()))],
                 linestyle="dashed", c="w")
        plt.title("PEARSON CORR: " + str(round(corr_dic["corr_triangles"][0], 5)))

        # plt.subplot(2, 2, 4)
        # # plt.text(0.1, 0.8, "* MWU p-value: " + str(round(corr_dic["mwu"][1], 8)), c="w")
        # plt.text(0.1, 0.6, "* WC p-value: " + str(round(corr_dic["wc"][1], 8)), c="w")
        # # plt.text(0.1, 0.4, "* KS p-value: " + str(round(corr_dic["ks"][1], 8)), c="w")
        # # plt.text(0.1, 0.2, "p-value < 0.05 --> not the same")
        # plt.axis("off")

        plt.show()

def generate_splits():
    all_random = []
    nr_samples = 10000
    nr_cv = 10
    nr_splits = 10
    test_ratio = 0.4
    for i in range(nr_samples):
        all_random.append(random.sample(range(0, nr_splits), int(test_ratio * nr_splits)))

    all_random = np.array(all_random)
    res_old = 0

    while (True):
        # pick random arrays
        sel = all_random[random.sample(range(0, nr_samples), nr_cv), :]
        sel_flattened = sel.flatten()
        unique, counts = np.unique(sel_flattened, return_counts=True)
        _, res = stats.kstest(sel_flattened, stats.uniform(loc=0.0, scale=10.0).cdf)
        if len(counts) == 10 and len(np.unique(counts)) == 1 or len(counts) == 10 and \
                len(np.unique(counts)) == 2 \
                and (np.unique(counts)[0] - np.unique(counts)[1]) <= 1:
            print(np.unique(counts)[0], np.unique(counts)[1])
            print(unique, counts)
            break

    cv_test_chunks_lo = []
    cv_test_chunks_hi = []

    for i in range(nr_cv):
        randomList = sel[i]
        randomList.sort()

        unobserved_lo = []
        unobserved_lo.append(randomList[0])
        unobserved_hi = []

        for i in range(1, len(randomList)):
            if not randomList[i] == (randomList[i - 1] + 1):
                # close test chunk
                unobserved_hi.append(randomList[i - 1] + 1)
                # start new test chunk
                unobserved_lo.append(randomList[i])

        unobserved_hi.append(randomList[-1] + 1)

        cv_test_chunks_lo.append(unobserved_lo)
        cv_test_chunks_hi.append(unobserved_hi)

        # if last element is 10 --> set last element to be the last time point

    for lo, hi in zip(cv_test_chunks_lo, cv_test_chunks_hi):
        print(lo)
        print(hi)

    pickle.dump(cv_test_chunks_hi, open("unobserved_hi_cv" + str(nr_cv), "wb"))
    pickle.dump(cv_test_chunks_lo, open("unobserved_lo_cv" + str(nr_cv), "wb"))


"""#####################################################################################################################
#   OSCILLATIONS
#####################################################################################################################"""


def compute_power_stft(input_data, input_data_time_res, output_data_time_res, window_size_s=0.3, freq_lo_bound=140,
                  freq_hi_bound=200):
    # ------------------------------------------------------------------------------------------------------------------
    # computes power from local field potential using short time Fourier transform. Computes power for each
    # tetrode/channel and averages to get one value per time bin
    #
    # args:         - input_data, np.array [time_points, channels/tetrods]
    #               - input_data_time_res, float: resolution in seconds ( --> 1 / f)
    #               - output_data_time_res, float: resolution in seconds ( --> 1 / f) of output data (power)
    #               - window_size_s, float: window size (in seconds) used to compute power
    #               - freq_lo_bound, int: lower bound for frequency range to be used
    #               - freq_up_bound, int: upper bound for frequency range to be used
    #                   --> SWR: freq_lo_bound=140, freq_hi_bound=200
    #
    # returns:      - output_data, array: avg. power across all channels/tetrodes at output_data_time_res (e.g.
    #                 ( output_data_time_res = 0.01 --> one value per 10 ms)
    # ------------------------------------------------------------------------------------------------------------------

    # compute nr. of time bins per window
    nr_bins_per_window = np.round(window_size_s/input_data_time_res).astype(int)

    # length of data in seconds
    len_data_s = input_data.shape[0] * input_data_time_res

    list_pow = []
    # go through all windows

    for ch in range(input_data.shape[1]):
        # f: frequencies, t: time points (beginning of window), zxx: magnitude
        f, t, zxx = stft(x=input_data[:, 0], fs=1 / input_data_time_res, nperseg=nr_bins_per_window)
        ind_freq_range = np.argwhere((freq_lo_bound < f) & (f < freq_hi_bound))
        # compute max or mean
        list_pow.append(np.mean(abs(zxx[ind_freq_range]), axis=0))

    list_pow = np.squeeze(np.array(list_pow), axis=1)
    # average across all channels --> one entry per time point in t (in seconds!)
    avg_across_ch = np.mean(list_pow, axis=0)
    # upsample to get output_data_time_res
    output_data_time_bins = np.arange(0, len_data_s, output_data_time_res)
    # interpolate
    output_data = np.interp(output_data_time_bins, t, avg_across_ch)
    return output_data


def butterworth_bandpass(input_data, freq_nyquist, freq_lo_bound, freq_hi_bound, filter_order=4):
    # ------------------------------------------------------------------------------------------------------------------
    # applies Butterworth bandpass filtering to input data
    #
    # args:         - input_data, np.array [time_points]
    #               - freq_nyquist, int: nyquist theorem --> need half the frequency
    #               - freq_lo_bound, int: lower bound for frequency range to be used
    #               - freq_up_bound, int: upper bound for frequency range to be used
    #                   --> SWR: freq_lo_bound=140, freq_hi_bound=200
    #               - filter_order, int: order of Butterworth filter to use
    #
    # returns:      - bandpass filtered signal
    # ------------------------------------------------------------------------------------------------------------------

    Wn = np.array([freq_lo_bound, freq_hi_bound])/freq_nyquist
    s = input_data.shape[0]
    filt_coeff = butter(N=filter_order, Wn=Wn, btype="bandpass")
    # add padding by adding the flipped signal to the left and the right to the original signal
    signal_orig = np.expand_dims(input_data, 0)
    signal_pad = np.hstack((np.fliplr(signal_orig), signal_orig, np.fliplr(signal_orig)))
    # apply filter: digital filtering by processing data in forward and reverse direction
    signal_filtered = filtfilt(filt_coeff[0], filt_coeff[1], signal_pad, padtype='odd', method='pad')
    # extract signal
    signal = np.squeeze(signal_filtered[:, s:(2*s)])

    return signal


def butterworth_lowpass(input_data, freq_nyquist, cut_off_freq, filter_order=4):
    # ------------------------------------------------------------------------------------------------------------------
    # applies Butterworth bandpass filtering to input data
    #
    # args:         - input_data, np.array [time_points]
    #               - freq_nyquist, int: nyquist theorem --> need half the frequency
    #               - cut_off_freq, int: cut off frequency
    #               - filter_order, int: order of Butterworth filter to use
    #
    # returns:      - bandpass filtered signal
    # ------------------------------------------------------------------------------------------------------------------

    Wn = cut_off_freq / freq_nyquist
    s = input_data.shape[0]
    filt_coeff = butter(N=filter_order, Wn=Wn, btype="lowpass")
    # add padding by adding the flipped signal to the left and the right to the original signal
    signal_orig = np.expand_dims(input_data, 0)
    signal_pad = np.hstack((np.fliplr(signal_orig), signal_orig, np.fliplr(signal_orig)))
    # apply filter: digital filtering by processing data in forward and reverse direction
    signal_filtered = filtfilt(filt_coeff[0], filt_coeff[1], signal_pad, padtype='odd', method='pad')
    # extract signal
    signal = np.squeeze(signal_filtered[:, s:(2 * s)])

    return signal


"""#####################################################################################################################
#   OTHERS
#####################################################################################################################"""


def log_multivariate_poisson_density(X, means):
    # ------------------------------------------------------------------------------------------------------------------
    # modeled on log_multivariate_normal_density from sklearn.mixture
    #
    # params:   - X, array: data
    #           - means, array: means of all modes [modes, cells]
    #
    # ------------------------------------------------------------------------------------------------------------------
    n_samples, n_dim = X.shape
    # -lambda + k log(lambda) - log(k!)
    means = np.nan_to_num(means)
    # TODO: why do i have negative lambdas???
    means[means<0] = 1e-3
    means[means == 0] = 1e-3

    log_means = np.where(means > 0, np.log(means), np.log(1e-3))
    lpr = np.dot(X, log_means.T)
    lpr = lpr - np.sum(means,axis=1) # rates for all elements are summed and then broadcast across the observation dimenension
    log_factorial = np.sum(gammaln(X + 1), axis=1)
    lpr = lpr - log_factorial[:,None] # logfactobs vector broad cast across the state dimension

    return lpr


def moving_average(a, n=3):
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def independent_shuffle(arr):
    # shuffles each column independently
    arr = arr.T
    x, y = arr.shape
    rows = np.indices((x, y))[0]
    cols = [np.random.permutation(y) for _ in range(x)]
    return arr[rows, cols].T


def simple_gaussian(xd, yd, std):
    return np.exp(-(np.square(xd) + np.square(yd)) / (std**2))


def calc_loc_and_speed(whl):
    # computes speed from the whl and returns speed in cm/s
    # need to smooth position data --> accuracy of measurement: about +-1cm --> error for speed: +-40m/s
    # last element of velocity vector is zero --> velocity is calculated using 2 locations

    # savitzky golay
    w_l = 31 # window length
    p_o = 5 # order of polynomial
    whl = signal.savgol_filter(whl, w_l, p_o)

    # one time bin: whl is recorded at 20kHz/512
    t_b = 1/(20e3/512)

    # upsampling to synchronize with spike data
    location = np.zeros(whl.shape[0] * 512)
    for i, loc in enumerate(whl):
        location[512 * i:(i + 1) * 512] = 512 * [loc]

    # calculate speed: x1-x0/dt
    temp_speed = np.zeros(whl.shape[0]-1)
    for i in range(temp_speed.shape[0]):
        temp_speed[i] = (whl[i+1]-whl[i])/t_b

    # smoothen speed using savitzky golay
    temp_speed = signal.savgol_filter(temp_speed, 15, 5)

    # upsampling to synchronize with spike data
    speed = np.zeros(whl.shape[0]*512)
    for i,bin_speed in enumerate(temp_speed):
        speed[512*i:(i+1)*512] = 512*[bin_speed]

    # plotting
    # t = np.arange(speed.shape[0])
    #
    # plt.plot(temp_speed,label="speed / m/s")
    # plt.plot(t/512,location, label = "location / cm")
    # plt.plot([0,350],[5,5], label = "threshold: 5cm/s")
    # plt.xlabel("time bins / 25.6ms")
    # plt.scatter(t/512,speed,color="b")
    # plt.legend()
    # plt.show()

    return location, speed


def calc_cohens_d(dat1, dat2):

    # calculates cohens D --> assumption of normal distributions
    pooled_std = np.sqrt(((dat1.shape[1]-1)*np.std(dat1,axis=1)**2+(dat2.shape[1]-1)*np.std(dat2,axis=1)**2)/
                         (dat1.shape[1]+dat2.shape[1]-2))
    # add small value to avoid division by zero
    pooled_std += 0.000001

    diff_avg = np.average(dat1,axis=1) - np.average(dat2,axis=1)

    return diff_avg/pooled_std


def graph_distance(ad_mat_1, ad_mat_2):
    # transforms 2 correlation matrices into graphs and computes distance between these two graphs
    print(ad_mat_1.shape)

    graph_1 = nx.to_networkx_graph(ad_mat_1)
    graph_2 = nx.to_networkx_graph(ad_mat_2)

    return nx.graph_edit_distance(graph_1, graph_2)


def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)


def make_square_axes(ax):
    """Make an axes square in screen units.

    Should be called after plotting.
    """
    ax.set_aspect(1 / ax.get_data_ratio())
