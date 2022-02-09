########################################################################################################################
#
#   MACHINE LEARNING METHODS
#
#   Description: contains classes that are used to perform analysis methods from ML
#
#   Author: Lars Bollmann
#
#   Created: 11/03/2020
#
#   Structure:
#
#               (1) MlMethodsOnePopulation: methods for one population
#
#               (2) MlMethodsTwoPopulations: methods for two populations
#
########################################################################################################################

from collections import OrderedDict
from function_files.support_functions import upper_tri_without_diag, multi_dim_scaling, perform_PCA, perform_TSNE, \
    perform_isomap, log_multivariate_poisson_density, correlateOneWithMany, find_hse, make_square_axes

from function_files.plotting_functions import plot_pop_clusters
from function_files.plotting_functions import plot_2D_scatter
from function_files.plotting_functions import plot_3D_scatter
from function_files.plotting_functions import plot_ridge_weight_vectors
from function_files.plotting_functions import plot_true_vs_predicted
from function_files.plotting_functions import plot_pop_cluster_analysis
from function_files.plotting_functions import plot_cca_loadings
from function_files.plotting_functions import cca_video
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import sklearn as sk
from scipy.stats import pearsonr
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend
import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing as mp
import tensorflow as tf
import pickle
from hmmlearn.base import _BaseHMM
import numpy as np
import sklearn.cluster as cluster
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from functools import partial
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
# from hmmlearn import hmm
from sklearn.decomposition import PCA
import rcca

########################################################################################################################
#   class MlMethodsOnePopulation
########################################################################################################################


class MlMethodsOnePopulation:
    """Machine learning methods for two populations"""

    def __init__(self, act_map=None, params=None, cell_type=None):
        self.cell_type = cell_type
        self.raster = act_map
        self.params = params
        self.X = None
        self.Y = None

    def reduce_dimension(self, input_data):
        # --------------------------------------------------------------------------------------------------------------
        # reduces the dimension using one of the defined methods
        # --------------------------------------------------------------------------------------------------------------

        # clear in case there was a previous result
        self.result_dr = []

        # use whole data or only a subset
        if input_data is None:
            act_map = self.map
        else:
            act_map = input_data

        # dimensionality reduction: multi dimensional scaling
        if self.params.dr_method == "MDS":
            self.result_dr = multi_dim_scaling(act_map, self.params)
        # dimensionality reduction: principal component analysis
        elif self.params.dr_method == "PCA":
            self.result_dr, explained_var = perform_PCA(act_map, self.params)
            print("PCA, VAR EXPLAINED: ", explained_var)
        elif self.params.dr_method == "TSNE":
            self.result_dr = perform_TSNE(act_map, self.params)
        elif self.params.dr_method == "isomap":
            self.result_dr = perform_isomap(act_map, self.params)

        return self.result_dr

    def plot_reduced_dimension(self, input_data=None):
        # --------------------------------------------------------------------------------------------------------------
        # reduces the dimension of the data set and plots data defined in data_range
        #
        # parameters:   - data_range, np.arange: np.arange(500, 1000, 50)
        # --------------------------------------------------------------------------------------------------------------
        # apply dimensionality reduction
        self.reduce_dimension(input_data)
        # generate plots
        if self.params.dr_method_p2 == 3:
            # create figure instance
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plot_3D_scatter(ax, self.result_dr, self.params, None)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot_2D_scatter(ax, self.result_dr, self.params, None)
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

    def parallelize_cross_val_model(self, nr_cluster_array, nr_cores, model_type, folder_name,
                                    raster_data=None, splits=None, cells_used="all_cells"):
        # --------------------------------------------------------------------------------------------------------------
        # parallelization of gmm cross validation
        #
        # parameters:   - nr_clusters_array, np.arange: for which number of clusters GMM is fit
        #               - nr_cores, int: how many cores to run in parallel
        # --------------------------------------------------------------------------------------------------------------

        # custom_splits_array = custom_splits * np.ones(nr_clusters_array.shape[0])
        # custom_splits_array = custom_splits_array.astype("bool")

        # result location
        if model_type == "pHMM":
                if self.params.cross_val_splits == "custom_splits":
                    res_dir = self.params.pre_proc_dir+"phmm/cross_val/"+cells_used+"/custom_splits/"+folder_name
                elif self.params.cross_val_splits == "standard_k_fold":
                    res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/standard_k_fold/" + folder_name
                elif self.params.cross_val_splits == "trial_splitting":
                    res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/trial_splitting/" + folder_name
        elif model_type == "GMM":
            if self.params.cross_val_splits == "custom_splits":
                res_dir = self.params.pre_proc_dir+"gmm/cross_val/custom_splits/"+folder_name
            elif self.params.cross_val_splits == "standard_k_fold":
                res_dir = self.params.pre_proc_dir + "gmm/cross_val/standard_k_fold/" + folder_name

        # check if directory exists already, otherwise create it
        if not os.path.isdir(res_dir):
            os.mkdir(res_dir)

        with mp.Pool(nr_cores) as p:
            # use partial to pass several arguments to cross_val_model, only nr_clusters_array is a changing one, the
            # others are constant
            multi_arg = partial(self.cross_val_model, res_dir=res_dir, raster_data=raster_data,
                                model_type=model_type, splits=splits)
            p.map(multi_arg, nr_cluster_array)

    def cross_val_model(self, nr_clusters, res_dir, raster_data=None, model_type="pHMM", splits=None):
        # --------------------------------------------------------------------------------------------------------------
        # cross validation of selected model
        #
        # parameters:   - nr_clusters, int: number of clusters to fit to the data
        #               - raster_data, array: input data [nr_cells, nr_time_bins]
        #               - model_type, string ["GMM", POISSON_HMM"]: defines which model to fit to data
        #               - custom_splits, bool: use custom splits or standard k-fold CV
        #
        # returns:  - saves result in text file [mean. log-likeli, std. log-likeli, mean. per sample log-likeli,
        #                                       std. per sample log-likeli, time_bin_size in seconds]
        # --------------------------------------------------------------------------------------------------------------

        # how many times to fit for each split
        max_nr_fitting = 5

        # print current process nr
        print(" - FITTING "+ model_type +" FOR #CLUSTERS = " + str(nr_clusters)+"\n")
        # print(mp.current_process())

        # check if result file exists already
        if os.path.isfile(res_dir + "/" + str(nr_clusters)):
            print("   --> RESULT EXISTS ALREADY ... SKIPPING\n")
            return

        if raster_data is None:
            # load data
            x = self.raster.T
        else:
            x = raster_data.T

        if splits is not None:
            # cross validation using provided splits
            # ----------------------------------------------------------------------------------------------------------
            nr_folds = len(splits)
            result_array = np.zeros(len(splits))
            result_array_per_sample = np.zeros(len(splits))

            for fold, test_range in enumerate(splits):
                X_test = x[test_range,:]
                X_train = np.delete(x,test_range, axis=0)

                # fit model several times to average over the influence of initialization
                el_fitting = np.zeros(max_nr_fitting)
                el_fitting_per_sample = np.zeros(max_nr_fitting)
                for nr_fitting in range(max_nr_fitting):
                    if model_type == "GMM":
                        model = GaussianMixture(n_components=nr_clusters)
                    elif model_type == "pHMM":
                        model = PoissonHMM(n_components=nr_clusters)
                    model.fit(X_train)
                    el_fitting[nr_fitting] = model.score(X_test)
                    el_fitting_per_sample[nr_fitting] = model.score(X_test) / X_test.shape[0]
                result_array[fold] = np.mean(el_fitting)
                result_array_per_sample[fold] = np.mean(el_fitting_per_sample)

        else:

            # standard n fold cross validation
            # ----------------------------------------------------------------------------------------------------------
            # number of folds --> 10 by default
            nr_folds = 10
            result_array = np.zeros(nr_folds)
            result_array_per_sample = np.zeros(nr_folds)
            skf = KFold(n_splits=nr_folds)

            for fold, (train_index, test_index) in enumerate(skf.split(x)):
                X_train, X_test = x[train_index], x[test_index]
                # fit model several times to average over the influence of initialization
                el_fitting = np.zeros(max_nr_fitting)
                el_fitting_per_sample = np.zeros(max_nr_fitting)
                for nr_fitting in range(max_nr_fitting):
                    if model_type == "GMM":
                        model = GaussianMixture(n_components=nr_clusters)
                    elif model_type == "pHMM":
                        model = PoissonHMM(n_components=nr_clusters)
                    model.fit(X_train)
                    el_fitting[nr_fitting] = model.score(X_test)
                    el_fitting_per_sample[nr_fitting] = model.score(X_test) / X_test.shape[0]
                result_array[fold] = np.mean(el_fitting)
                result_array_per_sample[fold] = np.mean(el_fitting_per_sample)

        # print current process nr
        print(" ... DONE WITH #CLUSTERS = " + str(nr_clusters)+"\n")

        # save results
        # --------------------------------------------------------------------------------------------------------------
        with open(res_dir + "/" + str(nr_clusters), "a") as f:
            # f.write(str(np.mean(result_array[:])) + "," + str(np.std(result_array[:])) + "\n")
            f.write(str(np.round(np.mean(result_array[:]),2)) + "," + str(np.round(np.std(result_array[:]),2)) + "," +
                    str(np.round(np.mean(result_array_per_sample[:]), 5)) + "," +
                    str(np.round(np.std(result_array_per_sample[:]), 5)) + "," + str(self.params.time_bin_size)+ "," +
                    str(nr_folds)+"\n")

    def cross_val_view_results(self, folder_name, range_to_plot=None, save_fig=False, cells_used="all_cells",
                               model_type="pHMM"):
        # --------------------------------------------------------------------------------------------------------------
        # view results of GMM cross validation (optimal number of clusters)
        #
        # parameters:   - identifier, str: only used when GMM cross val results are saved with an additional identifier
        #                 (e.g. z_scored)
        #               - first_data_point_to_plot: from which #modes to plot
        # --------------------------------------------------------------------------------------------------------------

        # result location
        # result location
        # result location
        if model_type == "pHMM":
                if self.params.cross_val_splits == "custom_splits":
                    res_dir = self.params.pre_proc_dir+"phmm/cross_val/"+cells_used+"/custom_splits/"+folder_name
                elif self.params.cross_val_splits == "standard_k_fold":
                    res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/standard_k_fold/" + folder_name
                elif self.params.cross_val_splits == "trial_splitting":
                    res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/trial_splitting/" + folder_name
        elif model_type == "GMM":
            if self.params.cross_val_splits == "custom_splits":
                res_dir = self.params.pre_proc_dir+"gmm/cross_val/custom_splits/"+folder_name
            elif self.params.cross_val_splits == "standard_k_fold":
                res_dir = self.params.pre_proc_dir + "gmm/cross_val/standard_k_fold/" + folder_name

        nr_cluster_list = []
        mean_values = []
        std_values = []
        mean_values_per_sample = []
        std_values_per_sample = []
        read_time_bin_size = True

        for file in os.listdir(res_dir):
            nr_cluster_list.append(int(file))
            with open(res_dir + "/" + file) as f:
                res = f.readline()
                mean_values.append(float(res.replace("\n", "").split(",")[0]))
                std_values.append(float(res.replace("\n", "").split(",")[1]))
                mean_values_per_sample.append(float(res.replace("\n", "").split(",")[2]))
                std_values_per_sample.append(float(res.replace("\n", "").split(",")[3]))
                if read_time_bin_size:
                    time_bin_size = float(res.replace("\n", "").split(",")[4])
                    try:
                        nr_folds = int(res.replace("\n", "").split(",")[5])
                    except:
                        print("NR. FOLDS NOT FOUND")
                        nr_folds = 10
                    read_time_bin_size = False
        # sort in right order
        mean_values = [x for _, x in sorted(zip(nr_cluster_list, mean_values))]
        std_values = [x for _, x in sorted(zip(nr_cluster_list, std_values))]
        mean_values_per_sample = [x for _, x in sorted(zip(nr_cluster_list, mean_values_per_sample))]
        std_values_per_sample = [x for _, x in sorted(zip(nr_cluster_list, std_values_per_sample))]
        nr_cluster_list = sorted(nr_cluster_list)

        if save_fig:
            plt.style.use('default')
        if range_to_plot is None:
            plt.plot(nr_cluster_list, mean_values_per_sample, color="red", marker='o')
        else:
            plt.plot(nr_cluster_list[range_to_plot[0]:range_to_plot[1]],
                     mean_values_per_sample[range_to_plot[0]:range_to_plot[1]], color="red", marker='o')
        plt.xlabel("#states")
        plt.ylabel("mean per sample log-likelihood ")
        plt.grid()
        if save_fig:
            make_square_axes(plt.gca())
            plt.title("cross-validated log-likelihood (10 fold)")
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("phmm_max_likelihood.svg", transparent="True")
        else:
            plt.title("GOODNESS OF FIT: "+str(nr_folds)+"-FOLD (5 FITS PER FOLD)\n PER SAMPLE, TIME BIN SIZE = "+
                      str(time_bin_size)+"s\n SPLITTING METHOD: "+self.params.cross_val_splits)
            plt.show()

        print("MAX. LOG.LIKELIHOOD: "+ str(max(mean_values)) +" --> "+
              str(nr_cluster_list[np.argmax(np.array(mean_values_per_sample))])+" CLUSTERS")

        if range_to_plot is None:
            plt.errorbar(nr_cluster_list, mean_values_per_sample, yerr = std_values_per_sample,
                         fmt='o-', label="TEST", c="#990000")
        else:
            plt.errorbar(nr_cluster_list[range_to_plot[0]:range_to_plot[1]],
                         mean_values_per_sample[range_to_plot[0]:range_to_plot[1]],
                         yerr = std_values_per_sample[range_to_plot[0]:range_to_plot[1]],
                         fmt='o-', label="TEST", c="#990000")

        plt.xlabel("#MODES")
        plt.ylabel("PER SAMPLE LOG-LIKELIHOOD (MEAN+STD OF 5 FITS)")
        plt.title("GOODNESS OF FIT: "+str(nr_folds)+"-FOLD (5 FITS PER FOLD)\n PER SAMPLE, TIME BIN SIZE = "+str(time_bin_size)+"s")
        plt.grid()
        plt.show()
        print("MAX. LOG.LIKELIHOOD: "+ str(max(mean_values)) +" --> "+
              str(nr_cluster_list[np.argmax(np.array(mean_values_per_sample))])+" CLUSTERS")

    def get_optimal_mode_number(self, folder_name, cells_used="all_cells",
                               model_type="pHMM"):
        # --------------------------------------------------------------------------------------------------------------
        # view results of GMM cross validation (optimal number of clusters)
        #
        # parameters:   - identifier, str: only used when GMM cross val results are saved with an additional identifier
        #                 (e.g. z_scored)
        #               - first_data_point_to_plot: from which #modes to plot
        # --------------------------------------------------------------------------------------------------------------

        if model_type == "pHMM":
                if self.params.cross_val_splits == "custom_splits":
                    res_dir = self.params.pre_proc_dir+"phmm/cross_val/"+cells_used+"/custom_splits/"+folder_name
                elif self.params.cross_val_splits == "standard_k_fold":
                    res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/standard_k_fold/" + folder_name
                elif self.params.cross_val_splits == "trial_splitting":
                    res_dir = self.params.pre_proc_dir + "phmm/cross_val/"+cells_used+"/trial_splitting/" + folder_name
        elif model_type == "GMM":
            if self.params.cross_val_splits == "custom_splits":
                res_dir = self.params.pre_proc_dir+"gmm/cross_val/custom_splits/"+folder_name
            elif self.params.cross_val_splits == "standard_k_fold":
                res_dir = self.params.pre_proc_dir + "gmm/cross_val/standard_k_fold/" + folder_name

        nr_cluster_list = []
        mean_values = []
        std_values = []
        mean_values_per_sample = []
        std_values_per_sample = []
        read_time_bin_size = True

        for file in os.listdir(res_dir):
            nr_cluster_list.append(int(file))
            with open(res_dir + "/" + file) as f:
                res = f.readline()
                mean_values.append(float(res.replace("\n", "").split(",")[0]))
                std_values.append(float(res.replace("\n", "").split(",")[1]))
                mean_values_per_sample.append(float(res.replace("\n", "").split(",")[2]))
                std_values_per_sample.append(float(res.replace("\n", "").split(",")[3]))
                if read_time_bin_size:
                    time_bin_size = float(res.replace("\n", "").split(",")[4])
                    try:
                        nr_folds = int(res.replace("\n", "").split(",")[5])
                    except:
                        print("NR. FOLDS NOT FOUND")
                        nr_folds = 10
                    read_time_bin_size = False
        # sort in right order
        mean_values = [x for _, x in sorted(zip(nr_cluster_list, mean_values))]
        std_values = [x for _, x in sorted(zip(nr_cluster_list, std_values))]
        mean_values_per_sample = [x for _, x in sorted(zip(nr_cluster_list, mean_values_per_sample))]
        std_values_per_sample = [x for _, x in sorted(zip(nr_cluster_list, std_values_per_sample))]
        nr_cluster_list = sorted(nr_cluster_list)

        return nr_cluster_list[np.argmax(np.array(mean_values_per_sample))]

    def plot_custom_splits(self):
        # plot custom splits
        # number of folds
        nr_folds = 10
        # how many times to fit for each split
        max_nr_fitting = 5
        # how many chunks (for pre-computed splits)
        nr_chunks = 10
        unobserved_lo_array = pickle.load(
            open("temp_data/unobserved_lo_cv" + str(nr_folds) + "_" + str(nr_chunks) + "_chunks", "rb"))
        unobserved_hi_array = pickle.load(
            open("temp_data/unobserved_hi_cv" + str(nr_folds) + "_" + str(nr_chunks) + "_chunks", "rb"))

        # set number of time bins
        bin_num = 1000
        bins = np.arange(bin_num + 1)

        # length of one chunk
        n_chunks = int(bin_num / nr_chunks)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for fold in range(nr_folds):

            # unobserved_lo: start bins (in spike data resolution) for all test data chunks
            unobserved_lo = []
            unobserved_hi = []
            for lo, hi in zip(unobserved_lo_array[fold], unobserved_hi_array[fold]):
                unobserved_lo.append(bins[lo * n_chunks])
                unobserved_hi.append(bins[hi * n_chunks])

            unobserved_lo = np.array(unobserved_lo)
            unobserved_hi = np.array(unobserved_hi)

            test_range = []
            for lo, hi in zip(unobserved_lo, unobserved_hi):
                test_range += (list(range(lo, hi)))
            test_range = np.array(test_range)
            ax.scatter(range(bin_num), fold * np.ones(bin_num), c="b", label="TRAIN")
            ax.scatter(test_range, fold * np.ones(test_range.shape[0]), c="r", label="TEST")

        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("CUSTOM SPLITS FOR CROSS-VALIDATION")
        plt.xlabel("DATA POINTS")
        plt.ylabel("FOLD NR.")

        plt.show()

    def gmm_fitting(self, nr_clusters):
        # --------------------------------------------------------------------------------------------------------------
        # finds clusters for population states using a mixture of Gaussians and plots result
        #
        # parameters:   - nr_clusters, int
        # --------------------------------------------------------------------------------------------------------------

        model = GaussianMixture(n_components=nr_clusters)
        labels = model.fit_predict(self.map.T)
        # for cov in model.covariances_:
        #     print("NEXt EIG")
        #     eigv = np.linalg.eig(cov)[0]
        #     print(eigv)

        samples = model.sample(n_samples=100000)[0].T
        real_data = self.map

        return samples, real_data, labels

    def gmm_cluster_analysis(self, nr_clusters):
        # --------------------------------------------------------------------------------------------------------------
        # finds clusters for population states using a mixture of Gaussians and analyzes results
        #
        # parameters:   - nr_clusters, int
        # --------------------------------------------------------------------------------------------------------------

        model = GaussianMixture(n_components=nr_clusters)
        labels = model.fit_predict(self.map.T)

        transition_matrix = np.zeros((nr_clusters, nr_clusters))
        for ind in range(labels.shape[0] - 1):
            transition_matrix[labels[ind]][labels[ind + 1]] += 1

        row_sum = np.sum(transition_matrix, axis=1)
        # divide to get probabilites
        transition_matrix /= row_sum[:, None]
        transition_matrix = np.nan_to_num(transition_matrix)

        # compute transition entropy --> This quantity has the interpretation that from mode β, transitions can be made
        # to roughly 2^Htrans different modes.

        h_trans = np.zeros(transition_matrix.shape[0])

        # don't consider self transitions --> remove diagonal
        trans_mat_copy = np.copy(transition_matrix)
        trans_mat_copy = \
            trans_mat_copy[~np.eye(trans_mat_copy.shape[0], dtype=bool)].reshape(trans_mat_copy.shape[0], -1)
        # get probabilites for self transition for normalization
        self_prob = transition_matrix.diagonal()
        print(self_prob.shape)

        for i, mode in enumerate(trans_mat_copy):
            h_trans_mode = 0
            for p in mode:
                # if probability is zero --> do not add anything
                if p == 0:
                    h_trans_mode = h_trans_mode
                else:
                    # normalize
                    p_norm = p/(1-self_prob[i])
                    h_trans_mode += p_norm*np.log2(p_norm)
            # write into array --> -1 because of negative sign for entropy
            h_trans[i] = -1* h_trans_mode

        # compute <k> statistic for each mode

        k_values_mean = np.zeros(nr_clusters)
        k_values_std = np.zeros(nr_clusters)
        nr_neurons = self.map.shape[0]
        for mode in range(nr_clusters):
            act_mat_mode = self.map[:, np.where(labels == mode)]
            k = []
            for time_bin in act_mat_mode.T:
                k.append(np.nan_to_num(np.count_nonzero(time_bin)/nr_neurons))
            k = np.array(k)
            k_values_mean[mode] = np.mean(k)
            k_values_std[mode] = np.std(k)



        # plot clustering analysis results
        plot_pop_cluster_analysis(labels, self.param_dic, nr_clusters, transition_matrix, h_trans, k_values_mean,
                                  k_values_std)

        # plot time series
        plot_pop_clusters(self.map, labels, self.param_dic, nr_clusters)

        # TODO: assess goodness of model --> are basic statistics maintained when you draw a random sample?

    def tree_hmm_cluster_analysis(self, file_name):
        # --------------------------------------------------------------------------------------------------------------
        # loads fit and plots results such as transition probabilities
        #
        # parameters:   - file_name, string
        # --------------------------------------------------------------------------------------------------------------

        data_base = pickle.load(open(file_name, "rb"), encoding='latin1')

        print(data_base["hist"].shape)
        print(data_base["emiss_prob"].shape)
        print(data_base["trans"].shape)
        print(data_base["P"].shape)
        A = data_base["trans"].T

        plt.subplot(1,2,1)
        plt.imshow(A, interpolation='nearest', aspect='auto')
        A = A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)
        plt.title("TRANS. PROB")
        plt.xlabel("MODE ID")
        plt.ylabel("MODE ID")
        plt.subplot(1,2,2)
        plt.imshow(A, interpolation='nearest', aspect='auto')
        plt.title("TRANS. PROB W/0 DIAGONAL")
        plt.xlabel("MODE ID")
        plt.show()

        plt.imshow(data_base["P"].T, interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.ylabel("MODE ID")
        plt.xlabel("TIME BINS")
        plt.title("MODE PROBABILITY")
        plt.show()

    def independent_model(self, custom_splits=True):
        # --------------------------------------------------------------------------------------------------------------
        # fit independent poisson model to data
        #
        # parameters:   -custom_splits, bool: whether to use previously generated splits or standard n fold cross-val
        # --------------------------------------------------------------------------------------------------------------

        x = self.map

        if custom_splits:
            nr_cv = 10
            nr_chunks = 10

            unobserved_lo_array = pickle.load(
                open(self.params.pre_proc_dir+"TreeHMM/unobserved_lo_cv" + str(nr_cv) + "_" + str(nr_chunks) + "_chunks", "rb"))
            unobserved_hi_array = pickle.load(
                open(self.params.pre_proc_dir+"TreeHMM/unobserved_hi_cv" + str(nr_cv) + "_" + str(nr_chunks) + "_chunks", "rb"))

            # set number of time bins
            bin_num = x.shape[1]
            bins = np.arange(bin_num + 1)

            # length of one chunk
            n_chunks = int(bin_num / nr_chunks)

            res = []

            for k in range(nr_cv):

                # unobserved_lo: start bins (in spike data resolution) for all test data chunks
                unobserved_lo = []
                unobserved_hi = []
                for lo, hi in zip(unobserved_lo_array[k], unobserved_hi_array[k]):
                    unobserved_lo.append(bins[lo * n_chunks])
                    unobserved_hi.append(bins[hi * n_chunks])

                unobserved_lo = np.array(unobserved_lo)
                unobserved_hi = np.array(unobserved_hi)

                test_range = []
                for lo, hi in zip(unobserved_lo, unobserved_hi):
                    test_range += (list(range(lo,hi)))
                test_range = np.array(test_range)
                x_test = x[:, test_range]
                x_train = np.delete(x, test_range, axis=1)

                # train independent model --> only need mean firing (binarized)
                p_i = np.mean(x_train, axis=1)

                # get log likelihood
                log_li = 0

                for pop_test in x_test.T:
                    log_li += np.dot(pop_test,np.log10(p_i))+ np.dot((1-pop_test),np.log10(1-p_i))

                # per sample --> divide by number of samples
                log_li /= x_test.shape[1]

                print("FOLD "+str(k)+" --> LOG-LIKELIHOOD = " + str(log_li))

                res.append(log_li)

            res = np.array(res)

            plt.scatter(np.ones(nr_cv), res, marker="*", c="w")

            plt.errorbar(1, np.mean(res),np.std(res), c="#990000", label="MEAN+STD", marker="o")
            plt.title("INDEPENDENT MODEL - " +str(self.params.time_bin_size)+"s - "+self.params.binning_method)
            plt.ylim([min(res)-0.2, -1.8])
            plt.ylabel("LOG-LIKELIHOOD")
            plt.xlabel("#MODES")
            plt.grid(True, c="grey")
            plt.show()


        else:
            x = x.T
            nr_folds = 10

            skf = KFold(n_splits=nr_folds)

            for fold, (train_index, test_index) in enumerate(skf.split(x)):
                x_train, x_test = x[train_index], x[test_index]

                x_train = x_train.T
                x_test = x_test.T

                # train independent model --> only need mean firing (binarized)
                p_i = np.mean(x_train, axis=1)

                # get log likelihood
                log_li = 0

                for pop_test in x_test.T:
                    log_li += np.sum(pop_test*np.log10(p_i) + (1-pop_test)*np.log10(1-p_i))

                # per sample --> divide by number of samples
                log_li /= x_test.shape[1]

                print("FOLD "+str(fold)+" --> LOG-LIKELIHOOD = " + str(log_li))

    def ridge_time_bin_progress(self, x, y, new_time_bin_size, alpha_fitting=True, alpha=100,
                                plotting=False, random_seed=42, return_weights=False):
        # --------------------------------------------------------------------------------------------------------------
        # fits ridge regression to predict time bin progression
        #
        # parameters:   - x, 2d-array: rows --> variables, columns --> samples
        #               - y, 1d-array: time bin ids
        #               - new_time_bin_size: time bin that was used for raster data x
        # --------------------------------------------------------------------------------------------------------------

        n_regressors = x.shape[0]
        n_samples = x.shape[1]

        print(" - #regressors: " + str(n_regressors))
        print(" - #samples: " + str(n_samples))

        # shuffle data
        per_ind = np.random.RandomState(seed=random_seed).permutation(np.arange(x.shape[1]))
        x_shuffled = x[:, per_ind].T
        y_shuffled = y[per_ind]

        # # normalize data to lie between 0 and 1
        # x_shuffled = preprocessing.StandardScaler().fit_transform(x_shuffled)
        # y_shuffled = preprocessing.StandardScaler().fit_transform(y_shuffled)

        if alpha_fitting:
            print("\n - ALPHA OPTIMIZATION STARTED ...\n")
            # 60% for training, 20% for testing, 15% for parameter optimization training, 5% for parameter optimization
            # testing
            x_train = x_shuffled[:int(x_shuffled.shape[0] * 0.6)]
            x_test = x_shuffled[int(x_shuffled.shape[0] * 0.6):int(x_shuffled.shape[0] * 0.8):]

            x_opt_train = x_shuffled[int(x_shuffled.shape[0] * 0.8):int(x_shuffled.shape[0] * 0.95):]
            y_opt_train = y_shuffled[int(y_shuffled.shape[0] * 0.8):int(y_shuffled.shape[0] * 0.95):]

            x_opt_test = x_shuffled[int(x_shuffled.shape[0] * 0.95):]
            y_opt_test = y_shuffled[int(y_shuffled.shape[0] * 0.95):]

            y_train = y_shuffled[:int(y_shuffled.shape[0] * 0.6)]
            y_test = y_shuffled[int(y_shuffled.shape[0] * 0.6):int(y_shuffled.shape[0] * 0.8):]

            # use 10% of training data for parameter optimization (alpha)
            alpha_list = range(100, 2000, 50)

            r2 = []
            mse = []

            for alpha in alpha_list:
                clf = Ridge(alpha=alpha)
                clf.fit(x_opt_train, y_opt_train)

                # # plot weight vectors
                # plot_ridge_weight_vectors(clf.coef_, cell_type_input, cell_type_output)

                # calculate mean square error & r2
                y_pred = clf.predict(x_opt_test)
                mse.append(sk.metrics.mean_squared_error(y_opt_test, y_pred))
                r2.append(clf.score(x_opt_test, y_opt_test))

            # plt.plot(alpha_list, r2)
            # plt.show()
            alpha_opt = alpha_list[np.argmax(np.array(r2))]
            print(" - OPTIMAL ALPHA="+str(alpha_list[np.argmax(np.array(r2))]))

            clf = Ridge(alpha=alpha_opt)

        else:
            x_train = x_shuffled[:int(x_shuffled.shape[0] * 0.8)]
            x_test = x_shuffled[int(x_shuffled.shape[0] * 0.8):]

            y_train = y_shuffled[:int(y_shuffled.shape[0] * 0.8)]
            y_test = y_shuffled[int(y_shuffled.shape[0] * 0.8):]

            clf = Ridge(alpha=alpha)

        clf.fit(x_train, y_train)
        # weights = clf.coef_
        # plt.bar(range(x.shape[0]), weights)
        # plt.title("REGRESSION WEIGHTS")
        # plt.xlabel("CELL IDS")
        # plt.ylabel("WEIGHT")
        # plt.show()

        true_values = y_test.T/60
        pred_values = clf.predict(x_test).T/60

        max_value = max(np.amax(true_values), np.amax(pred_values))

        y_pred = clf.predict(x_test)
        mse = sk.metrics.mean_squared_error(y_test, y_pred)
        r2 = clf.score(x_test, y_test)

        # compute adjusted r2
        r2_adj = 1-(1-r2)*(n_samples-1)/(n_samples-n_regressors-1)

        # control
        # --------------------------------------------------------------------------------------------------------------

        # y_train_shuffled = np.copy(y_train)
        # np.random.shuffle(y_train_shuffled)
        #
        # clf.fit(x_train, y_train_shuffled)
        #
        # true_values = y_test.T/60
        # pred_values_control = clf.predict(x_test).T/60
        #
        max_value = max(np.amax(true_values), np.amax(pred_values))
        min_value = min(np.amin(true_values), np.amin(pred_values))
        #
        # r2_control = clf.score(x_test, y_test)

        # plot true and predicted activity
        # plot_true_vs_predicted(true_values, pred_values, mse, r2, self.params)
        if plotting:
            plt.scatter(true_values, pred_values, label="MODEL", color="deepskyblue")
            # plt.scatter(true_values, pred_values_control, color="grey", alpha=0.4, label="CONTROL")
            plt.xlim(np.amin(np.hstack((true_values, pred_values))), np.amax(np.hstack((true_values, pred_values))))
            plt.ylim(np.amin(np.hstack((true_values, pred_values))), np.amax(np.hstack((true_values, pred_values))))
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel("TIME / min - TRUE VALUES")
            plt.ylabel("TIME / min - PREDICTED VALUES")
            plt.title("RIDGE REGRESSION: r2="+str(round(r2, 2))+", r2_adj="+str(round(r2_adj, 2))+"\n"
                            +"TIME BIN SIZE = " +str(new_time_bin_size) +"s")
            plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle= "--")
            plt.legend()
            plt.show()

        if return_weights:
            return clf.coef_
        else:
            return r2_adj

        # plt.imshow(x, interpolation='nearest', aspect='auto')
        # plt.show()

    def linear_separability(self, input_data, input_labels, C=1):

        X_train, X_test, y_train, y_test = train_test_split(input_data.T, input_labels, test_size=0.3,
                                                            random_state=np.random.randint(0, 1000), stratify=input_labels)
        # plt.subplot(2,1,1)
        # plt.imshow(np.log(X_train.T), interpolation='nearest', aspect='auto')
        # plt.subplot(2,1,2)
        # plt.imshow(np.expand_dims(y_train, 0), interpolation='nearest', aspect='auto')
        # plt.show()
        #
        # plt.subplot(2,1,1)
        # plt.imshow(np.log(X_test.T), interpolation='nearest', aspect='auto')
        # plt.subplot(2,1,2)
        # plt.imshow(np.expand_dims(y_test, 0), interpolation='nearest', aspect='auto')
        # plt.show()

        clf = svm.SVC(C=C, gamma="auto", kernel="linear")
        clf.fit(X_train, y_train)

        # compute score for REM (=1)
        X_test_rem = X_test[y_test == 1]
        y_test_rem = y_test[y_test == 1]

        # compute score for NREM (=1)
        X_test_nrem = X_test[y_test == 0]
        y_test_nrem = y_test[y_test == 0]

        return clf.score(X_test_rem, y_test_rem), clf.score(X_test_nrem, y_test_nrem), clf.score(X_test, y_test)

    def apply_pca(self):
        pca = PCA()
        pca.fit(self.raster.T)

        return pca.explained_variance_ratio_

    def parallelize_svm(self, m_subset, X_train, X_test, y_train, y_test, nr_subsets=10):

        with mp.Pool(nr_subsets) as p:
            # use partial to pass several arguments to cross_val_model, only nr_clusters_array is a changing one, the
            # others are constant
            multi_arg = partial(self.svm_with_subsets, X_train=X_train, X_test=X_test,
                                y_test=y_test, y_train=y_train)

            mean_accuracy = p.map(multi_arg, (np.ones(nr_subsets)*m_subset).astype(int))

            return mean_accuracy

    def svm_with_subsets(self, m_subset, X_train, X_test, y_train, y_test):

        subset = np.random.choice(a=range(X_train.shape[1]), size=m_subset, replace=False)

        X_train_subset = X_train[:, subset]
        X_test_subset = X_test[:,subset]

        # clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto', kernel="linear"))
        clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto', kernel="linear"))
        clf.fit(X_train_subset, y_train)

        return clf.score(X_test_subset, y_test)

########################################################################################################################
#   class MlMethodsTwoPopulation
########################################################################################################################


class MlMethodsTwoPopulations:
    """Machine learning methods for two populations"""

    def __init__(self, map_list=None, params=None, cell_type_array=None):
        self.cell_type_array = cell_type_array
        self.map_list = map_list
        self.params = params
        self.X = None
        self.Y = None

    def perform_global_cca(self, sel_range=None, co_firing_window_size=1, sliding_window=False, plotting=False,
                           remove_hse=False):
        # --------------------------------------------------------------------------------------------------------------
        # performs canonical correlation analysis on the entire input data and plots data in reduced space and loadings
        #
        # parameters:   - sel_range, range object: how much of the data to use
        #               - n_comp, int: number of components
        #
        # TODO: implement regularized CCA (Bilenko et al.) --> pyrCCA
        # --------------------------------------------------------------------------------------------------------------

        # compute how many time bins fit in one window
        time_bins_per_window = int(co_firing_window_size / self.params.time_bin_size)

        self.X = self.map_list[0]
        self.Y = self.map_list[1]

        if sel_range is not None:
            x = self.X[:, sel_range]
            y = self.Y[:, sel_range]
        else:
            x = self.X
            y = self.Y

        if remove_hse:
            ind_hse_x = np.array(find_hse(x=x)).flatten()
            ind_hse_y = np.array(find_hse(x=y)).flatten()

            ind_hse = np.unique(np.hstack((ind_hse_x, ind_hse_y)))

            # remove high synchrony events
            x = np.delete(x, ind_hse, axis=1)
            y = np.delete(y, ind_hse, axis=1)

        if sliding_window:

            data_x = np.empty((x.shape[0], 0))
            data_y = np.empty((y.shape[0], 0))

            for entry in range(int(x.shape[1] - time_bins_per_window + 1)):
                # print("PROGRESS: " + str(entry + 1) + "/" + str(int(x.shape[1] - time_bins_per_sliding_window))
                #       + " FRAMES")
                # extract chunk of data according to window size
                chunk_pop1 = x[:, entry:(entry + time_bins_per_window)]
                chunk_pop2 = y[:, entry:(entry + time_bins_per_window)]

                data_x = np.hstack((data_x, chunk_pop1))
                data_y = np.hstack((data_y, chunk_pop2))

            x = data_x.T
            y = data_y.T


        else:

            # transpose --> for CCA: (n_samples, n_features)
            x = x.T
            y = y.T

        # how many components to use
        n_comp = self.params.dr_method_p2

        cca = CCA(n_components=n_comp)
        cca.fit(x, y)
        x_c, y_c = cca.transform(x, y)

        x_loadings = np.nan_to_num(correlateOneWithMany(x_c[:, 0], x.T)[:, 0])
        y_loadings = np.nan_to_num(correlateOneWithMany(y_c[:, 1], y.T)[:, 0])

        x_weights = cca.x_weights_
        y_weights = cca.y_weights_

        x_trans = x @ x_weights

        result = np.corrcoef(x_c.T, y_c.T).diagonal(offset=n_comp)
        result = np.round(result, 2)

        nr_to_plot = 200
        print(x_trans.shape)
        time_bins = np.arange(x_trans.shape[0])*self.params.time_bin_size/60

        plt.subplot(2,1,1)
        plt.plot(time_bins[:nr_to_plot], x_trans[:nr_to_plot,0], label="CV1")
        plt.title("CV1 (corr:"+str(result[0])+")")
        plt.ylabel("CV1")
        plt.subplot(2,1,2)
        plt.plot(time_bins[:nr_to_plot], x_trans[:nr_to_plot, 1], label="CV2")
        plt.ylabel("CV2")
        plt.xlabel("TIME / min")
        plt.title("CV2 (corr: "+str(result[1])+")")

        plt.show()

        plot_cca_loadings(cca.x_weights_, cca.y_weights_, self.cell_type_array, self.params,
                          sliding_window)


        exit()
        if plotting:
            plot_cca_loadings(cca.x_loadings_, cca.y_loadings_, self.cell_type_array, self.params,
                              sliding_window)



            exit()
            if self.params.dr_method_p2 == 3:
                # create figure instance
                fig = plt.figure(figsize=(10, 5))
            elif self.params.dr_method_p2 == 2:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            for i in range(2):
                if i == 0:
                    data = x_c
                if i == 1:
                    data = y_c

                if self.params.dr_method_p2 == 3:
                    # create figure instance
                    ax_ = fig.add_subplot(1, 2, i+1, projection='3d')
                    plot_3D_scatter(ax_, data, self.params, None)
                    ax_.set_zlabel("CV3")
                elif self.params.dr_method_p2 == 2:
                    ax_ = ax[i]
                    plot_2D_scatter(ax_, data, self.params, None)
                    handles, labels = ax_.get_legend_handles_labels()
                    by_label = OrderedDict(zip(labels, handles))
                    ax_.legend(by_label.values(), by_label.keys())
                ax_.set_xlabel("CV1")
                ax_.set_ylabel("CV2")
                ax_.set_title("SCORES OF CANONICAL VARIATES FOR: "+ self.cell_type_array[i]+"\n VARIATE-CORR (CV1x-CV1y, ...): "
                         + str(result), y=1)
            plt.show()

        else:
            return x_loadings, y_loadings
    def perform_global_rcca(self, sel_range=None, co_firing_window_size=1, sliding_window=False, plotting=False,
                           remove_hse=False):
        # --------------------------------------------------------------------------------------------------------------
        # performs canonical correlation analysis on the entire input data and plots data in reduced space and loadings
        #
        # parameters:   - sel_range, range object: how much of the data to use
        #               - n_comp, int: number of components
        #
        # TODO: implement regularized CCA (Bilenko et al.) --> pyrCCA
        # --------------------------------------------------------------------------------------------------------------

        # compute how many time bins fit in one window
        time_bins_per_window = int(co_firing_window_size / self.params.time_bin_size)

        self.X = self.map_list[0]
        self.Y = self.map_list[1]

        if sel_range is not None:
            x = self.X[:, sel_range]
            y = self.Y[:, sel_range]
        else:
            x = self.X
            y = self.Y

        if remove_hse:
            ind_hse_x = np.array(find_hse(x=x)).flatten()
            ind_hse_y = np.array(find_hse(x=y)).flatten()

            ind_hse = np.unique(np.hstack((ind_hse_x, ind_hse_y)))

            # remove high synchrony events
            x = np.delete(x, ind_hse, axis=1)
            y = np.delete(y, ind_hse, axis=1)

        if sliding_window:

            data_x = np.empty((x.shape[0], 0))
            data_y = np.empty((y.shape[0], 0))

            for entry in range(int(x.shape[1] - time_bins_per_window + 1)):
                # print("PROGRESS: " + str(entry + 1) + "/" + str(int(x.shape[1] - time_bins_per_sliding_window))
                #       + " FRAMES")
                # extract chunk of data according to window size
                chunk_pop1 = x[:, entry:(entry + time_bins_per_window)]
                chunk_pop2 = y[:, entry:(entry + time_bins_per_window)]

                data_x = np.hstack((data_x, chunk_pop1))
                data_y = np.hstack((data_y, chunk_pop2))

            x = data_x.T
            y = data_y.T


        else:

            # transpose --> for CCA: (n_samples, n_features)
            x = x.T
            y = y.T

        # Split each dataset into two halves: training set and test set
        train1 = x[:x.shape[0] // 2]
        train2 = y[:y.shape[0] // 2]
        test1 = x[x.shape[0] // 2:]
        test2 = y[y.shape[0] // 2:]

        # how many components to use
        n_comp = self.params.dr_method_p2
        # Create a cca object as an instantiation of the CCA object class.

        # Initialize a cca object as an instantiation of the CCACrossValidate class.
        ccaCV = rcca.CCACrossValidate(kernelcca=False, numCCs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                                      regs=[0.0000001, 1e2, 1e4, 1e6])
        # Use the train() and validate() methods to run the analysis and perform cross-dataset prediction.
        ccaCV.train([train1, train2])
        testcorrsCV = ccaCV.validate([test1, test2])
        print('Optimal number of components: %d\nOptimal regularization coefficient: %d' % (
        ccaCV.best_numCC, ccaCV.best_reg))

        n_comp =11

        cca = rcca.CCA(kernelcca=False, reg=100, numCC=n_comp)

        # Use the train() method to find a CCA mapping between the two training sets.
        cca.train([train1, train2])

        # Use the validate() method to test how well the CCA mapping generalizes to the test data.
        # For each dimension in the test data, correlations between predicted and actual data are computed.
        testcorrs = cca.validate([test1, test2])

        # Plot canonical correlations (cca.cancorrs)
        plt.plot(np.arange(n_comp) + 1, cca.cancorrs, 'ro')
        plt.xlim(0.5, 0.5 + n_comp)
        plt.xticks(np.arange(n_comp) + 1)
        plt.xlabel('Canonical component')
        plt.ylabel('Canonical correlation')
        plt.title('Canonical correlations')
        # print('''The canonical correlations are:\n
        # Component 1: %.02f\n
        # Component 2: %.02f\n
        # Component 3: %.02f\n
        # Component 4: %.02f\n
        # ''' % tuple(cca.cancorrs))
        plt.show()
        from palettable.colorbrewer import qualitative
        # Plot correlations between actual test data and predictions
        # obtained by projecting the other test dataset via the CCA mapping for each dimension.
        nTicks = max(testcorrs[0].shape[0], testcorrs[1].shape[0])
        # bmap1 = qualitative.Dark2[3]
        bmap1 = qualitative.Dark2_3
        plt.plot(np.arange(testcorrs[0].shape[0]) + 1, testcorrs[0], 'o', color=bmap1.mpl_colors[0])
        plt.plot(np.arange(testcorrs[1].shape[0]) + 1, testcorrs[1], 'o', color=bmap1.mpl_colors[1])
        plt.xlim(0.5, 0.5 + nTicks + 3)
        plt.ylim(0.0, 1.0)
        plt.xticks(np.arange(nTicks) + 1)
        plt.xlabel('Dataset dimension')
        plt.ylabel('Prediction correlation')
        plt.title('Prediction accuracy')
        plt.legend(['Dataset 1', 'Dataset 2'])

        plt.show()

        exit()

        cca = CCA(n_components=n_comp)
        cca.fit(x, y)
        x_c, y_c = cca.transform(x, y)

        x_loadings = np.nan_to_num(correlateOneWithMany(x_c[:, 0], x.T)[:, 0])
        y_loadings = np.nan_to_num(correlateOneWithMany(y_c[:, 1], y.T)[:, 0])

        x_weights = cca.x_weights_
        y_weights = cca.y_weights_

        x_trans = x @ x_weights

        result = np.corrcoef(x_c.T, y_c.T).diagonal(offset=n_comp)
        result = np.round(result, 2)

        nr_to_plot = 200
        print(x_trans.shape)
        time_bins = np.arange(x_trans.shape[0])*self.params.time_bin_size/60

        plt.subplot(2,1,1)
        plt.plot(time_bins[:nr_to_plot], x_trans[:nr_to_plot,0], label="CV1")
        plt.title("CV1 (corr:"+str(result[0])+")")
        plt.ylabel("CV1")
        plt.subplot(2,1,2)
        plt.plot(time_bins[:nr_to_plot], x_trans[:nr_to_plot, 1], label="CV2")
        plt.ylabel("CV2")
        plt.xlabel("TIME / min")
        plt.title("CV2 (corr: "+str(result[1])+")")

        plt.show()

        plot_cca_loadings(cca.x_weights_, cca.y_weights_, self.cell_type_array, self.params,
                          sliding_window)


        exit()
        if plotting:
            plot_cca_loadings(cca.x_loadings_, cca.y_loadings_, self.cell_type_array, self.params,
                              sliding_window)



            exit()
            if self.params.dr_method_p2 == 3:
                # create figure instance
                fig = plt.figure(figsize=(10, 5))
            elif self.params.dr_method_p2 == 2:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            for i in range(2):
                if i == 0:
                    data = x_c
                if i == 1:
                    data = y_c

                if self.params.dr_method_p2 == 3:
                    # create figure instance
                    ax_ = fig.add_subplot(1, 2, i+1, projection='3d')
                    plot_3D_scatter(ax_, data, self.params, None)
                    ax_.set_zlabel("CV3")
                elif self.params.dr_method_p2 == 2:
                    ax_ = ax[i]
                    plot_2D_scatter(ax_, data, self.params, None)
                    handles, labels = ax_.get_legend_handles_labels()
                    by_label = OrderedDict(zip(labels, handles))
                    ax_.legend(by_label.values(), by_label.keys())
                ax_.set_xlabel("CV1")
                ax_.set_ylabel("CV2")
                ax_.set_title("SCORES OF CANONICAL VARIATES FOR: "+ self.cell_type_array[i]+"\n VARIATE-CORR (CV1x-CV1y, ...): "
                         + str(result), y=1)
            plt.show()

        else:
            return x_loadings, y_loadings

    def perform_local_cca(self, sel_range=None, sliding_window=True, co_firing_window_size=1, n_comp=3, plotting=True,
                          video_file_name="cca_test.mp4"):
        # --------------------------------------------------------------------------------------------------------------
        # performs canonical correlation analysis on subsequent windows or one sliding window and return loadings
        #
        # parameters:   - sel_range, range object: how much of the data to use
        #
        # TODO: implement regularized CCA (Bilenko et al.) --> pyrCCA
        # --------------------------------------------------------------------------------------------------------------

        # compute how many time bins fit in one window
        time_bins_per_window = int(co_firing_window_size / self.params.time_bin_size)

        # print(x.shape)
        # # remove cells that don't fire at all
        # x = x[~np.all(x == 0, axis=1)]
        # y = y[~np.all(y == 0, axis=1)]
        # print(x.shape)
        #
        # x_ind = ~np.all(x == 0, axis=0)
        # y_ind = ~np.all(y == 0, axis=0)
        # comb_ind = [a and b for a, b in zip(x_ind, y_ind)]
        #
        # x = x[:, comb_ind]
        # y = y[:, comb_ind]
        #
        # print(x.shape, y.shape)

        # transpose --> for CCA: (n_samples, n_features)

        self.X = self.map_list[0]
        self.Y = self.map_list[1]

        if sel_range is not None:
            x = self.X[:, sel_range]
            y = self.Y[:, sel_range]
        else:
            x = self.X
            y = self.Y

        # # remove cells that don't fire at all
        # x = x[~np.all(x == 0, axis=1)]
        # y = y[~np.all(y == 0, axis=1)]
        # print(x.shape)
        #
        # x_ind = ~np.all(x == 0, axis=0)
        # y_ind = ~np.all(y == 0, axis=0)
        # comb_ind = [a and b for a, b in zip(x_ind, y_ind)]
        #
        # x = x[:, comb_ind]
        # y = y[:, comb_ind]
        #
        # print(x.shape, y.shape)

        x_loadings_CV1 = []
        x_loadings_CV2 = []
        x_loadings_CV1_scipy = []
        x_loadings_CV2_scipy = []
        y_loadings_CV1_scipy = []
        y_loadings_CV2_scipy = []
        y_loadings = []
        corr_coff_CV1 = []
        corr_coff_CV2 = []
        corr_coff_CV1_scipy = []
        corr_coff_CV2_scipy = []

        if sliding_window:
            for entry in range(int(x.shape[1] - time_bins_per_window + 1)):
                # print("PROGRESS: " + str(entry + 1) + "/" + str(int(x.shape[1] - time_bins_per_sliding_window))
                #       + " FRAMES")
                # extract chunk of data according to window size
                chunk_pop1 = x[:, entry:(entry + time_bins_per_window)]
                chunk_pop2 = y[:, entry:(entry + time_bins_per_window)]

                # transpose --> for CCA: (n_samples, n_features)
                chunk_pop1 = chunk_pop1.T
                chunk_pop2 = chunk_pop2.T

                # cca = rcca.CCA(kernelcca=False, reg=0.000001, numCC=n_comp, )
                # # train on data
                # cca.train([chunk_pop1, chunk_pop2])
                # # compute loadings for x
                # # CV1
                #
                # # x_loadings_cv1 = correlateOneWithMany(cca.comps[0][:,0], chunk_pop1.T)[:,0]
                # # x_loadings_cv1 = np.nan_to_num(x_loadings_cv1)
                #
                # x_loadings_CV1.append(np.nan_to_num(correlateOneWithMany(cca.comps[0][:,0], chunk_pop1.T)[:,0]))
                # x_loadings_CV2.append(np.nan_to_num(correlateOneWithMany(cca.comps[0][:, 1], chunk_pop1.T)[:, 0]))
                # corr_coff_CV1.append(pearsonr(cca.comps[0][:,0],cca.comps[1][:,0] )[0])
                # corr_coff_CV2.append(pearsonr(cca.comps[0][:, 1], cca.comps[1][:, 1])[0])

                cca = CCA(n_components=n_comp, max_iter=1000)
                x_c, y_c = cca.fit_transform(chunk_pop1, chunk_pop2)

                x_loadings_CV1_scipy.append(np.nan_to_num(correlateOneWithMany(x_c[:, 0], chunk_pop1.T)[:, 0]))
                x_loadings_CV2_scipy.append(np.nan_to_num(correlateOneWithMany(x_c[:, 1], chunk_pop1.T)[:, 0]))

                y_loadings_CV1_scipy.append(np.nan_to_num(correlateOneWithMany(y_c[:, 0], chunk_pop2.T)[:, 0]))
                y_loadings_CV2_scipy.append(np.nan_to_num(correlateOneWithMany(y_c[:, 1], chunk_pop2.T)[:, 0]))
                #
                # result = np.corrcoef(x_c.T, y_c.T).diagonal(offset=n_comp)
                # x_loadings.append(cca.x_loadings_)
                # y_loadings.append(cca.y_loadings_)
                # plt.scatter(x_c[:, 0], y_c[:, 0])
                # plt.show()

                corr_coff_CV1_scipy.append(pearsonr(x_c[:, 0], y_c[:, 0])[0])
                corr_coff_CV2_scipy.append(pearsonr(x_c[:, 1], y_c[:, 1])[0])

        else:

            for window in range(int(x.shape[1]/time_bins_per_window)):
                # print("PROGRESS: " + str(entry + 1) + "/" + str(int(x.shape[1] - time_bins_per_sliding_window))
                #       + " FRAMES")
                # extract chunk of data according to window size
                chunk_pop1 = x[:, window * time_bins_per_window:(window + 1)*time_bins_per_window]
                chunk_pop2 = y[:, window * time_bins_per_window:(window + 1)*time_bins_per_window]

                # transpose --> for CCA: (n_samples, n_features)
                chunk_pop1 = chunk_pop1.T
                chunk_pop2 = chunk_pop2.T

                cca = CCA(n_components=n_comp, max_iter=1000)

                x_c, y_c = cca.fit_transform(chunk_pop1, chunk_pop2)

                result = np.corrcoef(x_c.T, y_c.T).diagonal(offset=n_comp)
                x_loadings.append(cca.x_loadings_)
                y_loadings.append(cca.y_loadings_)
                corr_coff.append(result)

        # x_loadings_CV1 = np.array(x_loadings_CV1)
        x_loadings_CV1_scipy = np.array(x_loadings_CV1_scipy)
        # x_loadings_CV2 = np.array(x_loadings_CV2)
        x_loadings_CV2_scipy = np.array(x_loadings_CV2_scipy)
        y_loadings_CV1_scipy = np.array(y_loadings_CV1_scipy)
        # y_loadings = np.array(y_loadings)
        # corr_coff = np.array(corr_coff)


        if plotting:
            plt.plot(corr_coff_CV1, label="CV1")
            plt.plot(corr_coff_CV2, label="CV2")
            plt.plot(corr_coff_CV1_scipy, label="CV1 scipy")
            plt.plot(corr_coff_CV2_scipy, label="CV2 scipy")
            plt.legend()
            plt.show()


            window_in_sec = self.params.time_bin_size * time_bins_per_window

            plt.subplot(2, 1, 1)
            plt.imshow(x_loadings_CV1.T, interpolation='nearest', aspect='auto',
                       extent=[0, self.params.time_bin_size * x_loadings_CV1.shape[0], x_loadings_CV1.shape[1], 0])
            plt.colorbar()
            plt.title("CCA LOADINGS CV1 (PYRCCA): " + str(self.params.time_bin_size) + "s TIME BIN, " + str(
                co_firing_window_size) + "s WINDOW")
            plt.ylabel("CELL ID")
            plt.subplot(2, 1, 2)
            plt.imshow(x_loadings_CV2.T, interpolation='nearest', aspect='auto',
                       extent=[0, self.params.time_bin_size * x_loadings_CV1.shape[0], x_loadings_CV1.shape[1], 0])
            plt.colorbar()
            plt.xlabel("START OF WINDOW (s)")
            plt.title("CCA LOADINGS CV2: " + str(self.params.time_bin_size) + "s TIME BIN, " + str(
                co_firing_window_size) + "s WINDOW")
            plt.ylabel("CELL ID")
            plt.show()

            plt.subplot(2, 1, 1)
            plt.imshow(x_loadings_CV1_scipy.T, interpolation='nearest', aspect='auto',
                       extent=[0, self.params.time_bin_size * x_loadings_CV1_scipy.shape[0],
                               x_loadings_CV1_scipy.shape[1], 0])
            plt.colorbar()
            plt.xlabel("START OF WINDOW (s)")
            plt.title("CCA LOADINGS CV1: " + str(self.params.time_bin_size) + "s TIME BIN, " + str(
                co_firing_window_size) + "s WINDOW")
            plt.ylabel("CELL ID")
            plt.subplot(2, 1, 2)
            plt.imshow(x_loadings_CV2_scipy.T, interpolation='nearest', aspect='auto',
                       extent=[0, self.params.time_bin_size * x_loadings_CV1.shape[0], x_loadings_CV1.shape[1], 0])
            plt.colorbar()
            plt.xlabel("START OF WINDOW (s)")
            plt.title("CCA LOADINGS CV2: " + str(self.params.time_bin_size) + "s TIME BIN, " + str(
                co_firing_window_size) + "s WINDOW")
            plt.ylabel("CELL ID")
            plt.show()
            plt.show()

            exit()

            # plot time course of correlation between variates

            # if sliding_window:
            #     x_ax = np.arange(corr_coff.shape[0]) * self.params.time_bin_size
            # else:
            #     x_ax = np.arange(corr_coff.shape[0])
            #
            # plt.plot(x_ax, corr_coff[:, 0], "o-", c="r", markersize=5, label="CV1")
            # plt.plot(x_ax, corr_coff[:, 1], "o-", c="b", markersize=4, label="CV2")
            # plt.plot(x_ax, corr_coff[:, 2], "o-", c="w", markersize=3, label="CV3")
            # plt.ylim(0, 1.1)
            # plt.xlabel("START WINDOW (s)")
            # plt.ylabel("PEARSON R")
            # plt.title("CORR. VAL. BETWEEN VARIATES - TIME BIN SIZE: " + str(self.params.time_bin_size) + "s\n" +
            #           "WINDOW SIZE: " + str(window_in_sec) + "s WINDOW")
            # plt.legend()
            # plt.show()

            cca_video(x_loadings, y_loadings, corr_coff, window_in_sec, self.params.time_bin_size,
                      sliding_window, video_file_name)
            # for x_load, y_load in zip(x_loadings, y_loadings):
            #     plot_cca_loadings(x_load, y_load, self.cell_type_array, self.params, time_bin_size,
            #                   sliding_window)
            #     plt.show()
        else:

            return x_loadings_CV1_scipy, y_loadings_CV1_scipy

    def ridge_reg_optimize_alpha(self, cell_type_input, cell_type_output):
        # --------------------------------------------------------------------------------------------------------------
        # calculates goodness of fit for ridge regression using different values for alpha
        #
        # parameters:   - cell_type_input, str
        #               - cell_type_output, str
        #
        # TODO: pre-compute optimal initial range for alpha (see Semedo et al)
        # --------------------------------------------------------------------------------------------------------------

        # assign correct data as input/output
        self.X = self.map_list[self.cell_type_array.index(cell_type_input)]
        self.Y = self.map_list[self.cell_type_array.index(cell_type_output)]

        # shuffle data
        per_ind = np.random.permutation(np.arange(self.X.shape[1]))
        X = self.X[:, per_ind].T
        Y = self.Y[:, per_ind].T

        # k-fold cross validation to find best alpha
        nr_folds = 10
        alpha_values = np.arange(0.01, 1500, 5)
        mean_scores = np.zeros(alpha_values.shape[0])

        for i, val in enumerate(alpha_values):

            result_array = np.zeros(nr_folds)
            skf = KFold(n_splits=nr_folds)

            for fold, (train_index, test_index) in enumerate(skf.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                clf = Ridge(alpha=val)
                clf.fit(X_train, Y_train)

                result_array[fold] = clf.score(X_test, Y_test)

            mean_scores[i] = np.mean(result_array)

        plt.scatter(alpha_values, mean_scores)
        plt.show()

    def ridge_reg(self, cell_type_input, cell_type_output, alpha, normalize):
        # --------------------------------------------------------------------------------------------------------------
        # performs multi variate regression on dataset
        #
        # parameters:   - cell_type_input, str
        #               - cell_type_output, str
        #               - alpha, int: alpha value for ridge regression (determine via cross-validation)
        #               - normalize, bool: if True, data is normalized to lie between 0 and 1
        # --------------------------------------------------------------------------------------------------------------

        # assign correct data as input/output
        x = self.map_list[self.cell_type_array.index(cell_type_input)]
        y = self.map_list[self.cell_type_array.index(cell_type_output)]

        # remove cells that dont fire at all
        x = x[~np.all(x == 0, axis=1)]
        y = y[~np.all(y == 0, axis=1)]

        # plt.imshow(X, interpolation='nearest', aspect='auto',)
        # plt.colorbar()
        # plt.show()

        # shuffle data
        per_ind = np.random.permutation(np.arange(x.shape[1]))
        x_shuffled = x[:, per_ind].T
        y_shuffled = y[:, per_ind].T

        if normalize:
            # # normalize data to lie between 0 and 1
            x_shuffled = preprocessing.StandardScaler().fit_transform(x_shuffled)
            y_shuffled = preprocessing.StandardScaler().fit_transform(y_shuffled)

        # 80% for training, 20% for validation
        x_train = x_shuffled[:int(x_shuffled.shape[0] * 0.8)]
        x_test = x_shuffled[int(x_shuffled.shape[0] * 0.8):]

        y_train = y_shuffled[:int(y_shuffled.shape[0] * 0.8)]
        y_test = y_shuffled[int(y_shuffled.shape[0] * 0.8):]

        clf = Ridge(alpha=alpha)
        clf.fit(x_train, y_train)

        # plot weight vectors
        plot_ridge_weight_vectors(clf.coef_, cell_type_input, cell_type_output)

        # calculate mean square error & r2
        y_pred = clf.predict(x_test)
        mse = sk.metrics.mean_squared_error(y_test, y_pred)
        r2 = clf.score(x_test, y_test)

        true_values = y_test.T
        pred_values = clf.predict(x_test).T

        # plot true and predicted activity
        plot_true_vs_predicted(true_values, pred_values, mse, r2, self.params)

    def dec_enc(self, cell_type_input, cell_type_output, normalizing):
        # --------------------------------------------------------------------------------------------------------------
        # constructs and trains neural network that generates output activity from input activity
        #
        # parameters:   - cell_type_input, str
        #               - cell_type_output, str
        #               - normalize, bool: if True, data is normalized to lie between 0 and 1
        # --------------------------------------------------------------------------------------------------------------

        tf.keras.backend.set_floatx('float64')

        # params
        k = 2  # [2,5,10,50,100]
        learning_rate = 0.5
        batch_size = 50
        epochs = 100

        # assign correct data as input/output
        self.X = self.map_list[self.cell_type_array.index(cell_type_input)]
        self.Y = self.map_list[self.cell_type_array.index(cell_type_output)]

        # remove cells that dont fire at all
        X = self.X[~np.all(self.X == 0, axis=1)]
        Y = self.Y[~np.all(self.Y == 0, axis=1)]

        # shuffle data
        per_ind = np.random.permutation(np.arange(self.X.shape[1]))
        X = X[:, per_ind].T
        Y = Y[:, per_ind].T

        if normalizing:
            # # normalize data to lie between 0 and 1
            X = preprocessing.StandardScaler().fit_transform(X)
            Y = preprocessing.StandardScaler().fit_transform(Y)

        # 90% for training, 10% for validation
        x_train = X[:int(X.shape[0] * 0.9)]
        x_test = X[int(X.shape[0] * 0.9):]

        y_train = Y[:int(Y.shape[0] * 0.9)]
        y_test = Y[int(Y.shape[0] * 0.9):]

        # build model
        model = tf.keras.Sequential()
        model.add(Dense(100, input_dim=X.shape[1], activation="relu", kernel_initializer="he_uniform"))
        model.add(Dense(k, activation=None))
        # model.add(Dense(100, activation="relu"))
        model.add(Dense(Y.shape[1], activation="relu"))

        # define optimizer
        opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
        # define loss
        loss = tf.keras.losses.MeanSquaredLogarithmicError()
        model.compile(loss=loss, optimizer=opt)
        # fit model
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, verbose=0)

        # evaluate the model
        train_mse = model.evaluate(x_train, y_train, verbose=0)
        test_mse = model.evaluate(x_test, y_test, verbose=0)
        print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
        # plot loss during training
        plt.title('LOSS: MEAN SQUARED ERROR')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.xlabel("EPOCHS")
        plt.ylabel("MSE")
        plt.legend()
        plt.show()

        true_values = y_test.T
        pred_values = model.predict(x_test).T
        plot_true_vs_predicted(true_values, pred_values)

        # get latent variable values
        get_latents = backend.function([model.layers[0].input], [model.layers[1].output])

        latents = get_latents([x_test])[0]

        for i, latent_dim in enumerate(latents.T):
            print("latent "+str(i)+": "+str(np.std(latent_dim)))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_2D_scatter(ax, latents, self.params, None)
        plt.show()

    def ridge_time_from_correlations(self, correlation_matrices):
        # --------------------------------------------------------------------------------------------------------------
        # predict time passed using correlation matrices
        #
        # parameters:   - correlation matrices
        # --------------------------------------------------------------------------------------------------------------

        train_length = 0.8

        corr_mat_flat = []

        for mat in correlation_matrices:
            corr_mat_flat.append(upper_tri_without_diag(mat).flatten())

        # correlation matrices is our X
        X = np.array(corr_mat_flat)

        # avg_corr = np.mean(corr_mat_flat, axis=1)
        # print(avg_corr.shape)
        # plt.plot(avg_corr)
        # plt.show()
        # exit()
        # print(X.shape)

        # apply PCA
        # pca = PCA(n_components=0.9, svd_solver="full")
        # pca.fit(X)
        # cumsum = np.cumsum(pca.explained_variance_ratio_)
        # plt.plot(cumsum)
        # plt.show()
        #
        # a = pca.fit_transform(X)
        # print(a.shape)
        #
        #
        # print(X.shape)
        # exit()

        # time points is our Y
        Y = np.arange(X.shape[0])


        # shuffle data

        # per_ind = np.random.permutation(np.arange(X.shape[0]))
        # x_shuffled = X[per_ind, :]
        # y_shuffled = Y[per_ind]
        x_shuffled = X
        y_shuffled = Y

        # split into test and train
        x_train = x_shuffled[:int(x_shuffled.shape[0] * train_length)]
        x_test = x_shuffled[int(x_shuffled.shape[0] * train_length):]

        y_train = y_shuffled[:int(y_shuffled.shape[0] * train_length)]
        y_test = y_shuffled[int(y_shuffled.shape[0] * train_length):]

        # clf = Ridge(alpha=0.01)
        # clf = Lasso()
        # clf.fit(x_train, y_train)

        # # calculate mean square error & r2
        # y_pred = clf.predict(x_test)
        # plt.scatter(y_pred, y_test)
        # plt.xlabel("time predicted")
        # plt.ylabel("time true")
        # plt.show()
        #
        # mse = sk.metrics.mean_squared_error(y_test, y_pred)
        # r2 = clf.score(x_test, y_test)
        # print(r2)
        # exit()

        # k-fold cross validation to find best alpha
        nr_folds = 5
        alpha_values = np.arange(0.01, 5, 0.1)
        mean_scores = np.zeros(alpha_values.shape[0])

        for i, val in enumerate(alpha_values):

            result_array = np.zeros(nr_folds)
            skf = KFold(n_splits=nr_folds)

            for fold, (train_index, test_index) in enumerate(skf.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                # clf = Ridge(alpha=val)
                clf = Lasso(alpha=val)
                clf.fit(X_train, Y_train)

                result_array[fold] = clf.score(X_test, Y_test)

            mean_scores[i] = np.mean(result_array)

        plt.scatter(alpha_values, mean_scores)
        plt.show()

    def ridge_avg_firing_rate(self, x, y, new_time_bin_size, alpha_fitting=True, alpha=100):
        # --------------------------------------------------------------------------------------------------------------
        # fits ridge regression to predict average firing rate
        #
        # parameters:   - x, 2d-array: population vectors
        #               - y, 1d-array: time bin ids
        #               - new_time_bin_size: time bin that was used for raster data x
        # --------------------------------------------------------------------------------------------------------------

        train_length = int(0.5*x.shape[1])
        x_train = x[:, :train_length].T
        x_test = x[:, train_length:].T
        y_train = y[:train_length]
        y_test = y[train_length:]

        # # normalize data to lie between 0 and 1
        # x_shuffled = preprocessing.StandardScaler().fit_transform(x_shuffled)
        # y_shuffled = preprocessing.StandardScaler().fit_transform(y_shuffled)

        if alpha_fitting:

            # shuffle data
            per_ind = np.random.permutation(np.arange(x_train.shape[1]))
            x_shuffled = x_train[per_ind,:]
            y_shuffled = y_train[per_ind]

            # 15% for parameter optimization training, 5% for parameter optimization
            # testing

            x_opt_train = x_shuffled[int(x_shuffled.shape[0] * 0.8):int(x_shuffled.shape[0] * 0.95):]
            y_opt_train = y_shuffled[int(y_shuffled.shape[0] * 0.8):int(y_shuffled.shape[0] * 0.95):]

            x_opt_test = x_shuffled[int(x_shuffled.shape[0] * 0.95):]
            y_opt_test = y_shuffled[int(y_shuffled.shape[0] * 0.95):]

            # use 10% of training data for parameter optimization (alpha)
            alpha_list = range(100, 2000, 50)

            r2 = []
            mse = []

            for alpha in alpha_list:
                clf = Ridge(alpha=alpha)
                clf.fit(x_opt_train, y_opt_train)

                # # plot weight vectors
                # plot_ridge_weight_vectors(clf.coef_, cell_type_input, cell_type_output)

                # calculate mean square error & r2
                y_pred = clf.predict(x_opt_test)
                mse.append(sk.metrics.mean_squared_error(y_opt_test, y_pred))
                r2.append(clf.score(x_opt_test, y_opt_test))

            # plt.plot(alpha_list, r2)
            # plt.show()
            alpha_opt = alpha_list[np.argmax(r2)]
            print("OPTIMAL ALPHA="+str(alpha_list[np.argmax(r2)]))

            clf = Ridge(alpha=alpha_opt)

        else:

            clf = Ridge(alpha=alpha)

        clf.fit(x_train, y_train)

        plt.bar(range(x.shape[0]), clf.coef_)
        plt.title("REGRESSION WEIGHTS")
        plt.xlabel("CELL IDS")
        plt.ylabel("WEIGHT")
        plt.show()

        true_values = y_test
        pred_values = clf.predict(x_test)

        max_value = max(np.amax(true_values), np.amax(pred_values))

        y_pred = clf.predict(x_test)
        mse = sk.metrics.mean_squared_error(y_test, y_pred)
        r2 = clf.score(x_test, y_test)

        # control
        # --------------------------------------------------------------------------------------------------------------

        y_train_shuffled = np.copy(y_train)
        np.random.shuffle(y_train_shuffled)

        clf.fit(x_train, y_train_shuffled)

        true_values = y_test.T
        pred_values_control = clf.predict(x_test).T

        max_value = max(np.amax(true_values), np.amax(pred_values), np.amax(pred_values_control))

        r2_control = clf.score(x_test, y_test)

        # plot true and predicted activity
        # plot_true_vs_predicted(true_values, pred_values, mse, r2, self.params)

        plt.scatter(true_values, pred_values, label="MODEL")
        plt.scatter(true_values, pred_values_control, color="grey", alpha=0.4, label="CONTROL")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("AVG. FIRING / #SPIKES - TRUE VALUES")
        plt.ylabel("AVG. FIRING / #SPIKES - PREDICTED VALUES")
        plt.title("RIDGE REGRESSION, PREDICTING TIME BINS: r2="+str(round(r2, 2))+"\n"
                             "r2_control="+str(round(r2_control, 2))+", TIME BIN SIZE = " +str(new_time_bin_size) +"s")
        plt.plot([0, max_value], [0, max_value], color="red", linestyle= "--")
        plt.legend()
        plt.show()

        # plt.imshow(x, interpolation='nearest', aspect='auto')
        # plt.show()



class PoissonHMM(_BaseHMM):
    """ Hidden Markov Model with independent Poisson emissions.
    Parameters
    ----------
    n_components : int
        Number of states.
    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.
    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.
    algorithm : string, one of the :data:`base.DECODER_ALGORITHMS`
        Decoder algorithm.
    random_state: RandomState or an int seed (0 by default)
        A random number generator instance.
    n_iter : int, optional
        Maximum number of iterations to perform.
    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.
    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.
    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.
    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.
    Attributes
    ----------
    n_features : int
        Dimensionality of the (independent) Poisson emissions.
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.
    means_ : array, shape (n_components, n_features)
        Mean parameters for each state.
    Examples
    --------
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    PoissonHMM(algorithm='viterbi',...
    """
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.means_prior = means_prior
        self.means_weight = means_weight
        self.time_bin_size = None

    def _check(self):
        super(PoissonHMM, self)._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

    def _compute_log_likelihood(self, obs):
        return log_multivariate_poisson_density(obs, self.means_)

    def _generate_sample_from_state(self, state, random_state=None):
        rng = check_random_state(random_state)
        return rng.poisson(self.means_[state])

    def _init(self, X, lengths=None, params='stmc'):
        super(PoissonHMM, self)._init(X, lengths=lengths)

        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features
        if 'm' in params or not hasattr(self, "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_

    def _initialize_sufficient_statistics(self):
        stats = super(PoissonHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(PoissonHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'm' in self.params in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

    def _do_mstep(self, stats):
        super(PoissonHMM, self)._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight

        denom = stats['post'][:, np.newaxis]
        if 'm' in self.params:
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))
            self.means_ = np.where(self.means_ > 1e-5, self.means_, 1e-3)

    def set_time_bin_size(self, time_bin_size):
        # set time bin size in seconds for later analysis (need to know what time bin size was used to fit model)
        self.time_bin_size = time_bin_size