########################################################################################################################
#
#
#   MAIN FOR NEURAL DATA ANALYSIS
#
#
#   Description:
#
#                   1)  takes args as input or runs debug by default
#                   2)  classes with analysis methods can either be called directly or in analysis scripts in the
#                       analysis_scripts directory
#
#
#   Author: Lars Bollmann
#
#   Created: 28/01/2020
#
#
#  TODO:    - make sure that experiment phase is either always past as a list or string
#           - cell_type: where to store it
#           - complete documentation (overal structure and functions)
#
########################################################################################################################

import argparse
import socket
import importlib
from function_files.sessions import SingleSession, MultipleSessions
from parameter_files.standard_analysis_parameters import Parameters
import numpy as np


if __name__ == '__main__':

    """#################################################################################################################
    #   Get parser arguments
    #################################################################################################################"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_step', default="debug", type=str)
    parser.add_argument('--cell_type', nargs="+", default=["p1"], help="CELL TYPE WITHIN """)
    parser.add_argument('--binning_method', default="temporal_spike", type=str,
                        help="[temporal, temporal_spike, temporal_binary, spike_binning]")
    parser.add_argument('--time_bin_size', default=0.1, type=float, help="TIME BIN SIZE IN SECONDS (FLOAT)")
    parser.add_argument("--data_to_use", default="ext", type=str,
                        help="use standard data [std] or extended (incl. lfp) data [ext]")
    # parser.add_argument('--session_name', nargs="+", default=["mjc163R4R_0114"], type=str)
    # parser.add_argument('--session_name', nargs="+", default=["mjc163R2R_0114", "mjc163R4R_0114"], type=str)
    # parser.add_argument('--session_name', nargs="+", default=["mjc163R3L_0114", "mjc169R1R_0114", "mjc169R4R_0114",
    #                                                           "mjc148R4R_0113"], type=str)
    parser.add_argument('--session_name', nargs="+", default=["mjc163R4R_0114", "mjc163R2R_0114", "mjc169R4R_0114",
                                                              "mjc163R1L_0114", "mjc148R4R_0113",
                                                              "mjc163R3L_0114", "mjc169R1R_0114"], type=str)
    parser.add_argument("--experiment_phase", nargs="+", default=["sleep_long_1"],
                        help="EXPERIMENT PHASE """)

    args = parser.parse_args()

    """#################################################################################################################
    #   LOAD STANDARD ANALYSIS PARAMETERS
    #################################################################################################################"""

    params = Parameters()

    """#################################################################################################################
    #   PRE-PROCESS ARGS
    #################################################################################################################"""

    if len(args.session_name) == 1:
        # if one phase is provided
        session_name = args.session_name[0]

    else:
        # if two experiment phases are supposed to be analyzed
        session_name = args.session_name

    if len(args.cell_type) == 1:
        # if only one cell type is provided --> analyzing one region
        cell_type = args.cell_type[0]

    else:
        # if two cell types are provided --> analyze both regions
        cell_type = args.cell_type

    """#################################################################################################################
    #   SAVING DIRECTORY FOR (TEMPORARY) RESULTS AND DATA
    #################################################################################################################"""

    # check which computer is used to determine directory for pre-processed data:
    # ------------------------------------------------------------------------------------------------------------------
    if socket.gethostname() == "lngrad210.ist.local":
        # work laptop
        params.pre_proc_dir = "/home/lbollman/Projects/01_memory_drift/02_data/00_pre_proc_data/"

    elif socket.gethostname() == "ryzen2.ista.local":
        # work station
        params.pre_proc_dir = "/mnt/hdl1/01_across_regions/02_data/00_pre_proc_data/"
    else:
        raise Exception("COMPUTER NOT RECOGNIZED ... DEFINE PRE_PROC_DIR DIRECTORY IN MAIN FILE!")

    """#################################################################################################################
    #   ANALYSIS PARAMETERS
    #
    #   - standard parameters are in parameter_files/standard_analysis_parameters
    #################################################################################################################"""

    # check again that all points lie within cheeseboard
    # ------------------------------------------------------------------------------------------------------------------
    params.additional_spatial_filter = False

    # binning
    # ------------------------------------------------------------------------------------------------------------------
    # "temporal" --> counts spikes per time bin and divides by time bin size
    # "temporal_spike" --> counts spikes per time bin
    # "temporal_binary" --> checks if cell fires within time bin
    params.binning_method = args.binning_method

    # sleep type: "nrem", "rem", "sw", "all"
    # ------------------------------------------------------------------------------------------------------------------
    params.sleep_type = "all"

    # interval for temporal binning in s
    # ------------------------------------------------------------------------------------------------------------------
    params.time_bin_size = args.time_bin_size

    # #spikes per bin for spike binning (usually between 10 and 15)
    params.spikes_per_bin = 12

    # spatial resolution in cm --> for rate maps etc.
    params.spatial_resolution = 1

    # # speed filter in cm/s --> every data with lower speed is neglected
    params.speed_filter = 5

    # stable cell method: which method to use to find stable cells ("k_means", "mean_firing_awake",
    # "mwu_sleep", "ks_sleep", "ks_awake", "mwu_awake")
    params.stable_cell_method = "mwu_awake"

    # which kind of splits to use for cross validation ("standard_k_fold", "custom_splits", "trial_splitting")
    params.cross_val_splits = "trial_splitting"

    # define method for dimensionality reduction
    # ------------------------------------------------------------------------------------------------------------------
    # "MDS" multi dimensional scaling
    # "PCA" principal component analysis
    # "TSNE"
    # "isomap"

    params.dr_method = "MDS"

    # first parameter of method:
    # MDS --> p1: difference measure ["jaccard","cosine","euclidean", "correlation"]
    # PCA --> p1 does not exist --> ""
    params.dr_method_p1 = "correlation"

    # second parameter of method:
    # MDS --> p2: number of components
    # PCA --> p2: number of components
    # CCA --> p2: number of components
    params.dr_method_p2 = 2

    # third parameter of method:
    # MDS + jaccard --> make binary: if True --> population vectors are first made binary
    params.dr_method_p3 = True

    # ------------------------------------------------------------------------------------------------------------------
    # PLOTTING PARAMETERS
    # -----------------------------------------------   ----------------------------------------------------------------

    # lines in scatter plot
    params.lines = False

    if len(args.session_name) == 1:
        """#############################################################################################################
        # SINGLE SESSION
        #############################################################################################################"""
        if args.compute_step == "debug":
            single_ses = SingleSession(session_name=session_name, params=params,
                                       cell_type=cell_type).cheese_board(experiment_phase=["learning_cheeseboard_1"], data_to_use="ext")
            single_ses.phase_preference_analysis(tetrode=9)
            exit()
            single_ses.firing_rates_around_swr()
            exit()
            #
            single_ses = SingleSession(session_name=session_name, params=params,
                                       cell_type=cell_type).sleep_before_pre_sleep()
            single_ses.pre_play_learning_phmm_modes(trials_to_use_for_decoding="all", n_smoothing=1000, cells_to_use="stable")
            single_ses.pre_play_learning_phmm_modes(trials_to_use_for_decoding="all", n_smoothing=1000, cells_to_use="decreasing")
        elif args.compute_step == "fitting_ising":
            single_ses = SingleSession(session_name=session_name, params=params,
                                       cell_type=cell_type).long_sleep()
            single_ses.memory_drift_long_sleep_compute_results(template_type="ising")
        else:
            # run existing analysis scripts (from analysis_scripts directory)
            routine = importlib.import_module("analysis_scripts."+args.compute_step)
            routine.execute(params=params)

        """#############################################################################################################
        # FOR MULTIPLE SESSIONS
        #############################################################################################################"""
    else:
        if args.compute_step == "debug":
            ls = MultipleSessions(session_names=session_name,
                                  cell_type=cell_type, params=params)
            ls.phase_preference_analysis(sleep_or_awake="awake")
            exit()
            ls.cheeseboard_firing_rates_gain_during_swr(save_fig=False)
            exit()
            # ls.cheeseboard_find_and_fit_optimal_phmm(cells_to_use="stable_cells", pre_or_post="post",
            #                                          cl_ar_init=np.arange(1, 40, 5))
            # exit()
            ls.before_sleep_sleep_diff_likelihoods_subsets(split_sleep=False)
            exit()
            ls.cheeseboard_find_and_fit_optimal_phmm(cells_to_use="stable_cells", pre_or_post="post",
                                                     cl_ar_init=np.arange(1, 40, 5))
            # ls.before_sleep_sleep_compute_likelihoods(cells_to_use="decreasing")
            exit()
            ls.long_sleep_firing_rates_all_cells()
            exit()
            ls.before_sleep_sleep_compare_max_post_probabilities(cells_to_use="stable")
            # ls.before_sleep_sleep_compare_likelihoods(cells_to_use="decreasing")

        else:
            # run existing analysis scripts (from analysis_scripts directory)
            routine = importlib.import_module("analysis_scripts."+args.compute_step)
            routine.execute(params=params)

