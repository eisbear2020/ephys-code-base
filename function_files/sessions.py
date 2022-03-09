########################################################################################################################
#
#   SESSIONS
#
#   Description:    - SingleSession:    class that contains all info about one session
#                                       (parameters, which method can be used etc.)
#
#                                       one session contains several experiments phases (e.g. sleep, task, exploration)
#
#                   - MultipleSessions: class that bundles multiple sessions to compute things using results from
#                                       from multiple sessions
#
#   Author: Lars Bollmann
#
#   Created: 30/04/2021
#
#   Structure:
#
########################################################################################################################

import importlib
from function_files.single_phase import Sleep, Exploration, TwoPopSleep, Cheeseboard, CrossMaze
from function_files.multiple_phases import MultPhasesOnePopulation, MultPhasesTwoPopulations, LongSleep, \
     PrePostCheeseboard, ExplorationNovelFamiliar, PreSleepPost, ExplFamPrePostCheeseboardExplFam, AllData, \
     PreProbPrePostPostProb, SleepBeforeSleep, SleepBeforePreSleep
from function_files.load_data import LoadData
from function_files.support_functions import moving_average
import numpy as np
import scipy
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.stats import pearsonr, mannwhitneyu, ks_2samp, spearmanr, ttest_ind, binom_test, ttest_1samp
import seaborn as sns
from scipy import optimize
import matplotlib


class SingleSession:
    """class for single session"""

    def __init__(self, session_name, cell_type, params):
        # --------------------------------------------------------------------------------------------------------------
        # args: - session_name, str: name of session
        # --------------------------------------------------------------------------------------------------------------

        # import analysis parameters that are specific for the current session
        # --------------------------------------------------------------------------------------------------------------
        session_parameter_file = importlib.import_module("parameter_files." + session_name)

        self.session_params = session_parameter_file.SessionParameters()

        self.cell_type = cell_type
        self.session_name = session_name
        self.params = params
        self.params.cell_type = cell_type

    def load_data(self, experiment_phase, data_to_use):
        data_obj = LoadData(session_name=self.session_name,
                            experiment_phase=experiment_phase,
                            pre_proc_dir=self.params.pre_proc_dir)
        # write experiment phase id (_1, _2, ..) and experiment_phase data to params
        self.session_params.experiment_phase_id = data_obj.get_experiment_phase_id()
        self.session_params.experiment_phase = data_obj.get_experiment_phase()

        # check whether standard or extended data (incl. lfp) needs to be used
        if data_to_use == "std":
            data_dic = data_obj.get_standard_data()
            self.params.data_to_use = "std"
        elif data_to_use == "ext":
            data_dic = data_obj.get_extended_data()
            self.params.data_to_use = "ext"
        return data_dic

    def load_data_object(self, experiment_phase):

        data_obj = LoadData(session_name=self.session_name,
                            experiment_phase=experiment_phase,
                            pre_proc_dir=self.params.pre_proc_dir)
        # write experiment phase id (_1, _2, ..) and experiment_phase data to params
        self.session_params.experiment_phase_id = data_obj.get_experiment_phase_id()
        self.session_params.experiment_phase = data_obj.get_experiment_phase()

        return data_obj

    """#################################################################################################################
    #  analyzing single experiment phase for one population
    #################################################################################################################"""

    def sleep(self, experiment_phase, data_to_use="std"):

        data_dic = self.load_data(experiment_phase=experiment_phase, data_to_use=data_to_use)

        return Sleep(data_dic=data_dic, cell_type=self.cell_type, params=self.params,
                     session_params=self.session_params, experiment_phase=experiment_phase)

    def exploration(self, experiment_phase, data_to_use="std"):

        data_dic = self.load_data(experiment_phase=experiment_phase, data_to_use=data_to_use)

        return Exploration(data_dic=data_dic, cell_type=self.cell_type, params=self.params,
                           session_params=self.session_params, experiment_phase=experiment_phase)

    def cheese_board(self, experiment_phase, data_to_use="std"):

        data_dic = self.load_data(experiment_phase=experiment_phase, data_to_use=data_to_use)

        return Cheeseboard(data_dic=data_dic, cell_type=self.cell_type, params=self.params,
                           session_params=self.session_params, experiment_phase=experiment_phase)

    def cross_maze(self, experiment_phase, data_to_use="std"):

        data_dic = self.load_data(experiment_phase=experiment_phase, data_to_use=data_to_use)

        return CrossMaze(data_dic=data_dic, cell_type=self.cell_type, params=self.params,
                           session_params=self.session_params, experiment_phase=experiment_phase)

    """#################################################################################################################
    #  analyzing multiple experiment phases
    #################################################################################################################"""

    def long_sleep(self, data_to_use="std", subset_of_sleep=None):

        # get all experiment phases that define long sleep
        long_sleep_exp_phases = self.session_params.long_sleep_experiment_phases
        if subset_of_sleep is not None:
            long_sleep_exp_phases = [long_sleep_exp_phases[subset_of_sleep]]
        data_obj = self.load_data_object(experiment_phase=long_sleep_exp_phases)
        self.params.data_to_use = data_to_use
        return LongSleep(sleep_data_obj=data_obj, params=self.params, session_params=self.session_params,
                         cell_type=self.cell_type)

    def pre_post_cheeseboard(self, data_to_use="std"):

        pre = self.cheese_board(experiment_phase=["learning_cheeseboard_1"])
        post = self.cheese_board(experiment_phase=["learning_cheeseboard_2"])

        return PrePostCheeseboard(pre=pre, post=post, params=self.params,
                                  session_params=self.session_params)

    def sleep_before_sleep(self, data_to_use="std"):

        sleep_before = self.sleep(experiment_phase=["sleep_cheeseboard_1"], data_to_use=data_to_use)
        sleep = self.sleep(experiment_phase=["sleep_long_1"], data_to_use=data_to_use)

        return SleepBeforeSleep(sleep_before=sleep_before, sleep=sleep, params=self.params,
                                  session_params=self.session_params)

    def sleep_before_pre_sleep(self, data_to_use="std"):

        sleep_before = self.sleep(experiment_phase=["sleep_cheeseboard_1"], data_to_use=data_to_use)
        sleep = self.sleep(experiment_phase=["sleep_long_1"], data_to_use=data_to_use)
        pre = self.cheese_board(experiment_phase=["learning_cheeseboard_1"])

        return SleepBeforePreSleep(sleep_before=sleep_before, sleep=sleep, pre=pre, params=self.params,
                                  session_params=self.session_params)

    def exp_fam_pre_post_cheeseboard_exp_fam(self):

        exp_fam_1 = self.exploration(experiment_phase=["exploration_familiar"])
        pre = self.cheese_board(experiment_phase=["learning_cheeseboard_1"])
        post = self.cheese_board(experiment_phase=["learning_cheeseboard_2"])
        exp_fam_2 = self.exploration(experiment_phase=["exploration_familiar_post"])

        return ExplFamPrePostCheeseboardExplFam(exp_fam_1=exp_fam_1, pre=pre, post=post, exp_fam_2=exp_fam_2,
                                                params=self.params, session_params=self.session_params)

    def cheeseboard_pre_prob_pre_post_post_prob(self):

        pre_probe = self.exploration(experiment_phase=["exploration_cheeseboard"])
        pre = self.cheese_board(experiment_phase=["learning_cheeseboard_1"])
        post = self.cheese_board(experiment_phase=["learning_cheeseboard_2"])
        post_probe = self.exploration(experiment_phase=["post_probe"])

        return PreProbPrePostPostProb(pre_probe=pre_probe, pre=pre, post=post, post_probe=post_probe,
                                                params=self.params, session_params=self.session_params)

    def pre_sleep_post(self, data_to_use="std", pre=["cross_maze_task_1"], post=["cross_maze_task_2"], sleep=["sleep_1"]):
        # get pre and post phase
        pre = self.cross_maze(experiment_phase=pre)
        post = self.cross_maze(experiment_phase=post)
        sleep = self.sleep(experiment_phase=sleep)
        # get all experiment phases that define long sleep
        self.params.data_to_use = data_to_use
        self.params.session_name = self.session_name
        return PreSleepPost(sleep=sleep, pre=pre, post=post, params=self.params,
                                session_params=self.session_params)

    def novel_fam(self):

        raise Exception("TO BE IMPLEMENTED")

        # TODO: create exploration with according phase names
        nov = None
        fam = None
        return ExplorationNovelFamiliar(nov=nov, fam=fam, params=self.params)

    def all_data(self):

        return AllData(params=self.params, session_params=self.session_params)


class TwoPopSingleSession:
    """class for single session"""

    def __init__(self, session_name, cell_type_1, cell_type_2, params):
        # --------------------------------------------------------------------------------------------------------------
        # args: - session_name, str: name of session
        # --------------------------------------------------------------------------------------------------------------

        # import analysis parameters that are specific for the current session
        # --------------------------------------------------------------------------------------------------------------
        session_parameter_file = importlib.import_module("parameter_files." + session_name)

        self.session_params = session_parameter_file.SessionParameters()

        self.cell_type_1 = cell_type_1
        self.cell_type_2 = cell_type_2
        self.session_name = session_name
        self.params = params
        self.params.cell_type_1 = cell_type_1
        self.params.cell_type_2 = cell_type_2

    def load_data(self, experiment_phase, data_to_use):
        data_obj = LoadData(session_name=self.session_name,
                            experiment_phase=experiment_phase,
                            pre_proc_dir=self.params.pre_proc_dir)
        # write experiment phase id (_1, _2, ..) and experiment_phase data to params
        self.session_params.experiment_phase_id = data_obj.get_experiment_phase_id()
        self.session_params.experiment_phase = data_obj.get_experiment_phase()

        # check whether standard or extended data (incl. lfp) needs to be used
        if data_to_use == "std":
            data_dic = data_obj.get_standard_data()
            self.params.data_to_use = "std"
        elif data_to_use == "ext":
            data_dic = data_obj.get_extended_data()
            self.params.data_to_use = "ext"
        return data_dic

    def load_data_object(self, experiment_phase):

        data_obj = LoadData(session_name=self.session_name,
                            experiment_phase=experiment_phase,
                            pre_proc_dir=self.params.pre_proc_dir)
        # write experiment phase id (_1, _2, ..) and experiment_phase data to params
        self.session_params.experiment_phase_id = data_obj.get_experiment_phase_id()
        self.session_params.experiment_phase = data_obj.get_experiment_phase()

        return data_obj

    """#################################################################################################################
    #  analyzing single experiment phase for one population
    #################################################################################################################"""

    def sleep(self, experiment_phase, data_to_use="std"):

        data_dic = self.load_data(experiment_phase=experiment_phase, data_to_use=data_to_use)

        return TwoPopSleep(data_dic=data_dic, cell_type_1=self.cell_type_1, cell_type_2=self.cell_type_2,
                           params=self.params, session_params=self.session_params, experiment_phase=experiment_phase)

class MultipleSessions:
    """class for multiple sessions"""

    def __init__(self, session_names, cell_type, params):
        self.params = params

        # initialize all sessions
        self.session_list = []
        for session_name in session_names:
            self.session_list.append(SingleSession(session_name=session_name, cell_type=cell_type, params=params))

    """#################################################################################################################
    #  cheeseboard - learning
    #################################################################################################################"""

    def learning_map_dynamics(self, adjust_pv_size=False):
        """
        Checks remapping of cells during learning
        @param adjust_pv_size: whether to subsample pv for decreasing/increasing/stable cells to have same number
        of cells
        @type adjust_pv_size: bool

        """

        # go trough all sessions to collect results
        remapping_stable = []
        remapping_shuffle_stable = []
        remapping_dec = []
        remapping_shuffle_dec = []
        remapping_pv_stable = []
        remapping_pv_stable_shuffle = []
        remapping_pv_dec = []
        remapping_pv_dec_shuffle = []
        for session in self.session_list:
            rs, rss, rd, rds, rpvs, rpvss, rpvd, rpvds = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).map_dynamics_learning(plot_results=False,
                                                                                   adjust_pv_size=adjust_pv_size)
            remapping_stable.append(rs)
            remapping_shuffle_stable.append(rss.flatten())
            remapping_dec.append(rd)
            remapping_shuffle_dec.append(rds.flatten())
            remapping_pv_stable.append(rpvs)
            remapping_pv_stable_shuffle.append(rpvss)
            remapping_pv_dec.append(rpvd)
            remapping_pv_dec_shuffle.append(rpvds)

        # ----------------------------------------------------------------------------------------------------------
        # compute for single cells
        # ----------------------------------------------------------------------------------------------------------

        # stable cells
        remapping_stable = np.expand_dims(np.hstack((remapping_stable)),1).flatten()
        remapping_shuffle_stable = np.expand_dims(np.hstack((remapping_shuffle_stable)),1).flatten()

        remapping_dec = np.expand_dims(np.hstack((remapping_dec)),1).flatten()
        remapping_shuffle_dec = np.expand_dims(np.hstack((remapping_shuffle_dec)),1).flatten()


        c = "white"
        res = [remapping_stable, remapping_shuffle_stable, remapping_dec, remapping_shuffle_dec]
        bplot = plt.boxplot(res, positions=[1,2,3,4], patch_artist=True,
                            labels=["Stable", "Stable shuffle", "Dec", "Dec shuffle"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),showfliers=False
                            )
        colors = ["magenta", 'magenta', "blue", "blue"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.title("Remapping using single cells")
        plt.ylabel("Pearson R: PRE map - POST map")
        plt.grid(color="grey", axis="y")
        plt.show()

        remapping_stable_sorted = np.sort(remapping_stable)
        remapping_stable_shuffle_sorted = np.sort(remapping_shuffle_stable)
        remapping_dec_sorted = np.sort(remapping_dec)
        remapping_dec_shuffle_sorted = np.sort(remapping_shuffle_dec)

        # plot on population vector level
        p_stable = 1. * np.arange(remapping_stable_sorted.shape[0]) / (remapping_stable_sorted.shape[0] - 1)
        p_stable_shuffle = 1. * np.arange(remapping_stable_shuffle_sorted.shape[0]) / (
                    remapping_stable_shuffle_sorted.shape[0] - 1)

        p_dec = 1. * np.arange(remapping_dec_sorted.shape[0]) / (remapping_dec_sorted.shape[0] - 1)
        p_dec_shuffle = 1. * np.arange(remapping_dec_shuffle_sorted.shape[0]) / (
                    remapping_dec_shuffle_sorted.shape[0] - 1)

        plt.plot(remapping_stable_sorted, p_stable, label="Stable", color="magenta")
        plt.plot(remapping_stable_shuffle_sorted, p_stable_shuffle, label="Stable shuffle", color="darkmagenta")

        plt.plot(remapping_dec_sorted, p_dec, label="Dec", color="aquamarine")
        plt.plot(remapping_dec_shuffle_sorted, p_dec_shuffle, label="Dec shuffle", color="lightseagreen")
        plt.legend()
        plt.ylabel("CDF")
        plt.xlabel("PEARSON R")
        plt.title("Per cell.")
        plt.show()

        # ----------------------------------------------------------------------------------------------------------
        # compute for population vectors
        # ----------------------------------------------------------------------------------------------------------

        # stable cells
        remapping_pv_stable= np.hstack((remapping_pv_stable))
        remapping_pv_stable_shuffle = np.hstack((remapping_pv_stable_shuffle))

        remapping_pv_stable = remapping_pv_stable[~np.isnan(remapping_pv_stable)]
        remapping_pv_stable_shuffle = remapping_pv_stable_shuffle[~np.isnan(remapping_pv_stable_shuffle)]

        # decreasing cells
        remapping_pv_dec= np.hstack((remapping_pv_dec))
        remapping_pv_dec_shuffle = np.hstack((remapping_pv_dec_shuffle))

        remapping_pv_dec = remapping_pv_dec[~np.isnan(remapping_pv_dec)]
        remapping_pv_dec_shuffle = remapping_pv_dec_shuffle[~np.isnan(remapping_pv_dec_shuffle)]

        # print("MWU for PV remapping:"+str(mannwhitneyu(remapping_pv, remapping_pv_shuffle)[1]))
        c = "white"

        plt.figure(figsize=(4, 5))
        res = [remapping_pv_stable, remapping_pv_stable_shuffle, remapping_pv_dec, remapping_pv_dec_shuffle]
        bplot = plt.boxplot(res, positions=[1, 2, 3, 4], patch_artist=True,
                            labels=["stable", "stable shuffle", "dec", "dec shuffle"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["magenta", 'magenta', "blue", "blue"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Pearson R: First trial vs. last trial")
        # plt.yticks([-0.5, 0, 0.5, 1])
        # plt.ylim(-0.5, 1)
        plt.grid(color="grey", axis="y")

        plt.title("Remapping using Population Vectors")
        plt.show()


        remapping_pv_stable_sorted = np.sort(remapping_pv_stable)
        remapping_pv_stable_shuffle_sorted = np.sort(remapping_pv_stable_shuffle)
        remapping_pv_dec_sorted = np.sort(remapping_pv_dec)
        remapping_pv_dec_shuffle_sorted = np.sort(remapping_pv_dec_shuffle)

        # plot on population vector level
        p_pv_stable = 1. * np.arange(remapping_pv_stable_sorted.shape[0]) / (remapping_pv_stable_sorted.shape[0] - 1)
        p_pv_stable_shuffle = 1. * np.arange(remapping_pv_stable_shuffle_sorted.shape[0]) / (
                    remapping_pv_stable_shuffle_sorted.shape[0] - 1)

        p_pv_dec = 1. * np.arange(remapping_pv_dec_sorted.shape[0]) / (remapping_pv_dec_sorted.shape[0] - 1)
        p_pv_dec_shuffle = 1. * np.arange(remapping_pv_dec_shuffle_sorted.shape[0]) / (
                    remapping_pv_dec_shuffle_sorted.shape[0] - 1)

        plt.plot(remapping_pv_stable_sorted, p_pv_stable, label="Stable", color="magenta")
        plt.plot(remapping_pv_stable_shuffle_sorted, p_pv_stable_shuffle, label="Stable shuffle", color="darkmagenta")

        plt.plot(remapping_pv_dec_sorted, p_pv_dec, label="Dec", color="aquamarine")
        plt.plot(remapping_pv_dec_shuffle_sorted, p_pv_dec_shuffle, label="Dec shuffle", color="lightseagreen")
        plt.legend()
        plt.ylabel("CDF")
        plt.xlabel("PEARSON R")
        plt.title("Per pop. vec.")
        plt.show()

    def learning_mean_firing(self, absolute_value=False):
        """
        Checks firing rate remapping of cells during learning
        @param absolute_value: weather to use absolute or relative value
        @type absolute_value: bool

        """

        # go trough all sessions to collect results
        diff_stable = []
        diff_dec = []

        for session in self.session_list:
            d_s, d_d = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).learning_mean_firing_rate(plotting=False,
                                                                                       absolute_value=absolute_value)
            diff_stable.append(d_s)
            diff_dec.append(d_d)

        diff_stable = np.hstack(diff_stable)
        diff_dec = np.hstack(diff_dec)

        diff_stable_sorted = np.sort(diff_stable)
        diff_dec_sorted = np.sort(diff_dec)

        p_diff_stable = 1. * np.arange(diff_stable.shape[0]) / (diff_stable.shape[0] - 1)

        p_diff_dec = 1. * np.arange(diff_dec.shape[0]) / (diff_dec.shape[0] - 1)

        plt.plot(diff_stable_sorted, p_diff_stable, label="stable")
        plt.plot(diff_dec_sorted, p_diff_dec, label="dec")
        if absolute_value:
            plt.xlabel("Abs. relative Difference firing rates")
        else:
            plt.xlabel("Rel. difference firing rates")
        plt.ylabel("cdf")
        plt.title("Change in mean firing rates through learning")
        plt.legend()
        plt.show()

    def learning_start_end_map_stability(self):
        """
        Checks remapping of cells by comparing initial trials vs. last trials

        """

        # go trough all sessions to collect results
        initial_pop_vec_sim_stable = []
        initial_pop_vec_sim_dec = []
        late_pop_vec_sim_stable = []
        late_pop_vec_sim_dec = []

        for session in self.session_list:
            i_s, i_d, l_s, l_d = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).map_heterogenity(plotting=False)
            initial_pop_vec_sim_stable.append(i_s)
            initial_pop_vec_sim_dec.append(i_d)
            late_pop_vec_sim_stable.append(l_s)
            late_pop_vec_sim_dec.append(l_d)

        initial_pop_vec_sim_stable = np.hstack(initial_pop_vec_sim_stable)
        initial_pop_vec_sim_dec = np.hstack(initial_pop_vec_sim_dec)
        late_pop_vec_sim_stable = np.hstack(late_pop_vec_sim_stable)
        late_pop_vec_sim_dec = np.hstack(late_pop_vec_sim_dec)

        # ----------------------------------------------------------------------------------------------------------
        # compute for single cells
        # ----------------------------------------------------------------------------------------------------------

        initial_pop_vec_sim_stable_sorted = np.sort(initial_pop_vec_sim_stable)
        initial_pop_vec_sim_dec_sorted = np.sort(initial_pop_vec_sim_dec)
        late_pop_vec_sim_stable_sorted = np.sort(late_pop_vec_sim_stable)
        late_pop_vec_sim_dec_sorted = np.sort(late_pop_vec_sim_dec)

        # plot on population vector level
        p_stable_init = 1. * np.arange(initial_pop_vec_sim_stable.shape[0]) / (initial_pop_vec_sim_stable.shape[0] - 1)
        p_dec_init = 1. * np.arange(initial_pop_vec_sim_dec.shape[0]) / (
                    initial_pop_vec_sim_dec.shape[0] - 1)

        p_stable_late = 1. * np.arange(late_pop_vec_sim_stable.shape[0]) / (late_pop_vec_sim_stable.shape[0] - 1)
        p_dec_late = 1. * np.arange( late_pop_vec_sim_dec.shape[0]) / (
                late_pop_vec_sim_dec.shape[0] - 1)

        plt.plot(initial_pop_vec_sim_stable_sorted, p_stable_init, label="Early-Stable", color="magenta", linestyle="dashed")
        plt.plot(initial_pop_vec_sim_dec_sorted, p_dec_init, label="Early-Dec", color="darkmagenta")

        plt.plot(late_pop_vec_sim_stable_sorted, p_stable_late, label="Late-Stable", color="aquamarine", linestyle="dashed")
        plt.plot(late_pop_vec_sim_dec_sorted, p_dec_late, label="Late_dec", color="lightseagreen")
        plt.legend()
        plt.ylabel("CDF")
        plt.xlabel("Cosine distance")
        plt.title("Pairwise PVs")
        plt.show()

    def learning_place_field_peak_shift(self, spatial_resolution=1):
        """
        Checks remapping of cells during learning using place field peak shift
        @param spatial_resolution: spatial bin size in cm2
        @type spatial_resolution: int

        """

        # go trough all sessions to collect results
        shift_stable = []
        shift_dec = []

        for session in self.session_list:
            s_s, s_d = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).learning_place_field_peak_shift(plotting=False,
                                                                                             spatial_resolution=spatial_resolution)
            shift_stable.append(s_s)
            shift_dec.append(s_d)

        # ----------------------------------------------------------------------------------------------------------
        # compute for single cells
        # ----------------------------------------------------------------------------------------------------------

        shift_stable = np.hstack(shift_stable)
        shift_dec = np.hstack(shift_dec)
        shift_stable_sorted = np.sort(shift_stable)
        shift_dec_sorted = np.sort(shift_dec)

        # plot on population vector level
        p_stable = 1. * np.arange(shift_stable.shape[0]) / (shift_stable.shape[0] - 1)

        p_dec = 1. * np.arange(shift_dec.shape[0]) / (shift_dec.shape[0] - 1)

        plt.plot(shift_stable_sorted, p_stable, label="Stable", color="magenta")
        plt.plot(shift_dec_sorted, p_dec, label="Dec", color="aquamarine")
        plt.legend()
        plt.ylabel("CDF")
        plt.xlabel("Place field peak shift / cm")
        plt.title("Place field peak shift during learning")
        plt.show()

    def learning_error_stable_vs_decreasing(self, nr_of_trials=10):
        """
        compares decoding error by cross-validation (first n trials to train, last n to test)

        @param nr_of_trials: how many trials to use for training/testing
        @type nr_of_trials: int
        """
        # go trough all sessions to collect results
        error_stable = []
        error_dec = []

        for session in self.session_list:
            e_s, e_d = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).decoding_error_stable_vs_decreasing(plotting=False,
                                                                                                 nr_of_trials=nr_of_trials)
            error_stable.append(e_s)
            error_dec.append(e_d)

        error_stable = np.hstack(error_stable)
        error_stable_sorted = np.sort(error_stable)
        p_error_stable = 1. * np.arange(error_stable.shape[0]) / (error_stable.shape[0] - 1)

        error_dec = np.hstack(error_dec)
        error_dec_sorted = np.sort(error_dec)
        p_error_dec = 1. * np.arange(error_dec.shape[0]) / (error_dec.shape[0] - 1)

        plt.plot(error_stable_sorted, p_error_stable, label="stable")
        plt.plot(error_dec_sorted, p_error_dec, label="dec")
        plt.legend()
        plt.show()

    def cheeseboard_place_field_goal_distance_temporal(self, save_fig=False, mean_firing_threshold=1, nr_trials=4, pre_or_post="pre"):
        """
        computes distance between place field peak and closest goal -- compares end of learning vs. after learning

        @param save_fig: save as .svg
        @param mean_firing_threshold: threshold to exclude low firing cells (in Hz)
        """

        dist_stable_learning_1 = []
        dist_stable_learning_2 = []
        dist_stable_learning_3 = []
        dist_dec_learning_1 = []
        dist_dec_learning_2 = []
        dist_dec_learning_3 = []
        dist_inc_learning_1 = []
        dist_inc_learning_2 = []
        dist_inc_learning_3 = []

        if pre_or_post == "pre":
            experiment_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            experiment_phase = ["learning_cheeseboard_2"]

        for session in self.session_list:
            nr_total_trials = session.cheese_board(experiment_phase=experiment_phase).nr_trials
            # first n trials
            d_s_learning, d_d_learning, d_i_learning = session.cheese_board(experiment_phase=experiment_phase).place_field_goal_distance(plotting=False,
                                                                                                  mean_firing_threshold=
                                                                                                  mean_firing_threshold,
                                                                                                  trials_to_use=range(nr_trials))
            dist_stable_learning_1.append(d_s_learning)
            dist_dec_learning_1.append(d_d_learning)
            dist_inc_learning_1.append(d_i_learning)

            # middle n trials
            d_s_learning, d_d_learning, d_i_learning = session.cheese_board(experiment_phase=experiment_phase).place_field_goal_distance(plotting=False,
                                                                                                  mean_firing_threshold=
                                                                                                  mean_firing_threshold,
                                                                                                  trials_to_use=range(int(nr_total_trials/2)-int(nr_trials/2), int(nr_total_trials/2)+int(nr_trials/2)))
            dist_stable_learning_2.append(d_s_learning)
            dist_dec_learning_2.append(d_d_learning)
            dist_inc_learning_2.append(d_i_learning)

            # last n trials
            d_s_learning, d_d_learning, d_i_learning = session.cheese_board(experiment_phase=experiment_phase).place_field_goal_distance(plotting=False,
                                                                                                  mean_firing_threshold=
                                                                                                  mean_firing_threshold,
                                                                                                  trials_to_use=range(nr_total_trials-nr_trials, nr_total_trials))
            dist_stable_learning_3.append(d_s_learning)
            dist_dec_learning_3.append(d_d_learning)
            dist_inc_learning_3.append(d_i_learning)

        dist_stable_learning_1 = np.hstack(dist_stable_learning_1)
        dist_stable_learning_2 = np.hstack(dist_stable_learning_2)
        dist_stable_learning_3 = np.hstack(dist_stable_learning_3)
        dist_dec_learning_1 = np.hstack(dist_dec_learning_1)
        dist_dec_learning_2 = np.hstack(dist_dec_learning_2)
        dist_dec_learning_3 = np.hstack(dist_dec_learning_3)
        dist_inc_learning_1 = np.hstack(dist_inc_learning_1)
        dist_inc_learning_2 = np.hstack(dist_inc_learning_2)
        dist_inc_learning_3 = np.hstack(dist_inc_learning_3)

        if pre_or_post == "pre":

            if save_fig:
                plt.style.use('default')
                c = "black"
            else:
                c = "white"

            res = [dist_stable_learning_1,dist_dec_learning_1,dist_stable_learning_2,dist_dec_learning_2,
                   dist_stable_learning_3,dist_dec_learning_3]

            plt.figure(figsize=(4, 5))
            bplot = plt.boxplot(res, positions=[1, 2, 3, 4, 5, 6], patch_artist=True,
                                labels=["Stable", "Dec", "Stable", "Dec", "Stable", "Dec"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", 'turquoise', "magenta", 'turquoise',"magenta", 'turquoise']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal (cm)")
            plt.grid(color="grey", axis="y")
            print(mannwhitneyu(dist_stable_learning_1,dist_dec_learning_1))
            print(mannwhitneyu(dist_stable_learning_2,dist_dec_learning_2))
            print(mannwhitneyu(dist_stable_learning_3,dist_dec_learning_3))
            plt.ylim(0,60)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("place_field_to_goal_distance_pre.svg", transparent="True")
            else:
                plt.show()
        elif pre_or_post == "post":

            if save_fig:
                plt.style.use('default')
                c = "black"
            else:
                c = "white"

            res = [dist_stable_learning_1, dist_inc_learning_1, dist_stable_learning_2, dist_inc_learning_2,
                   dist_stable_learning_3, dist_inc_learning_3]

            plt.figure(figsize=(4, 5))
            bplot = plt.boxplot(res, positions=[1, 2, 3, 4, 5, 6], patch_artist=True,
                                labels=["Stable", "Inc", "Stable", "Inc", "Stable", "Inc"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", 'orange', "magenta", 'orange', "magenta", 'orange']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal (cm)")
            plt.grid(color="grey", axis="y")
            print(mannwhitneyu(dist_stable_learning_1, dist_inc_learning_1))
            print(mannwhitneyu(dist_stable_learning_2, dist_inc_learning_2))
            print(mannwhitneyu(dist_stable_learning_3, dist_inc_learning_3))
            plt.ylim(0, 60)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("place_field_to_goal_distance_post.svg", transparent="True")
            else:
                plt.show()

    """#################################################################################################################
    #  cheeseboard - PRE (after initial learning phase)
    #################################################################################################################"""

    def pre_decode_single_goals(self, save_fig=False, subset_range=[4,8,12,18], nr_splits=20, nr_subsets=10):
        """
        tries to decode single goals using population vectors and SVM using different number of cells

        @param save_fig: save .svg file
        @param subset_range: nr. of cells to use (as a list)
        @param nr_splits: how often to split for cross-validation
        @param nr_subsets: how many times to subsample (how many different subsets to use)
        """
        print("Identifying single goals using SVM ...")

        stable = []
        decreasing = []
        increasing = []

        for session in self.session_list:
            s,i,d = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).identify_single_goal_multiple_subsets(
                subset_range=subset_range, nr_splits=nr_splits, nr_subsets=nr_subsets, plotting=False)
            stable.append(s)
            decreasing.append(d)
            increasing.append(i)

        stable = np.hstack(stable)
        dec = np.hstack(decreasing)
        inc = np.hstack(increasing)

        stable_mean = np.mean(stable, axis=1)
        dec_mean = np.mean(dec, axis=1)
        inc_mean = np.mean(inc, axis=1)

        stable_std = np.std(stable, axis=1)
        dec_std = np.std(dec, axis=1)
        inc_std = np.std(inc, axis=1)

        if save_fig:
            plt.style.use('default')

        plt.errorbar(x=subset_range, y=stable_mean, yerr=stable_std, label="stable", ls="--", fmt="o", capsize=5)
        # plt.errorbar(x=np.array(subset_range)+0.1, y=inc_mean, yerr=inc_std, label="inc")
        plt.errorbar(x=np.array(subset_range), y=dec_mean, yerr=dec_std, label="dec", ls="--", fmt="o",
                     capsize=5)
        plt.hlines(0.25, 4, 18, linestyles="--", colors="gray", label="chance")
        plt.ylabel("Mean accuracy - multiclass SVM (mean,std)")
        plt.xlabel("#cells")
        plt.legend()

        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("identifying_goals.svg", transparent="True")
        else:
            plt.show()

    def pre_goal_related_activity(self, save_fig=False, subset_range=[4,8,12,18], nr_splits=20, radius=15):
        """
        Tries to decode activity around goals and seperate it from activity away from goals

        @param save_fig: save .svg file
        @param subset_range: nr. of cells to use (as a list)
        @param nr_splits: how often to split for cross-validation
        @param radius: what radius around goals (in cm) to consider as goal related
        """
        stable = []
        decreasing = []
        increasing = []

        for session in self.session_list:
            s,i,d = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).detect_goal_related_activity_using_multiple_subsets(
                subset_range=subset_range, nr_splits=nr_splits, plotting=False, radius=radius)
            stable.append(s)
            decreasing.append(d)
            increasing.append(i)

        stable = np.hstack(stable)
        dec = np.hstack(decreasing)
        inc = np.hstack(increasing)

        stable_mean = np.mean(stable, axis=1)
        dec_mean = np.mean(dec, axis=1)
        inc_mean = np.mean(inc, axis=1)

        stable_std = np.std(stable, axis=1)
        dec_std = np.std(dec, axis=1)
        inc_std = np.std(inc, axis=1)

        if save_fig:
            plt.style.use('default')

        plt.errorbar(x=subset_range, y=stable_mean, yerr=stable_std, label="stable")
        plt.errorbar(x=np.array(subset_range)+0.1, y=inc_mean, yerr=inc_std, label="inc")
        plt.errorbar(x=np.array(subset_range)+0.2, y=dec_mean, yerr=dec_std, label="dec")
        plt.ylabel("Mean accuracy - SVM (mean,std)")
        plt.xlabel("#cells")
        plt.legend()

        if save_fig:
            plt.savefig("cell_classification_numbers.svg", transparent="True")
        else:
            plt.show()

    """#################################################################################################################
    #  cheeseboard - PRE or POST
    #################################################################################################################"""

    def cheeseboard_cross_val_phmm(self, cells_to_use="all_cells", cl_ar=np.arange(1, 50, 5), pre_or_post="pre"):
        if pre_or_post == "pre":
            exp_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            exp_phase = ["learning_cheeseboard_2"]

        # go through sessions and cross-validate phmm
        for session in self.session_list:
            print("\nCross-val pHMM, session: "+session.session_name+"(cells:"+cells_to_use+") ...\n")
            session.cheese_board(experiment_phase=exp_phase).cross_val_poisson_hmm(cells_to_use=cells_to_use,
                                                                                                    cl_ar=cl_ar)
            print(" \n... done with cross-val pHMM, session: " + session.session_name+"\n")

    def cheeseboard_find_and_fit_optimal_phmm(self, cells_to_use="all_cells", cl_ar_init=np.arange(1, 50, 5),
                                              pre_or_post="pre"):
        if pre_or_post == "pre":
            exp_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            exp_phase = ["learning_cheeseboard_2"]
        # go through sessions and cross-validate phmm
        for session in self.session_list:
            print("\nCross-val pHMM, session: "+session.session_name+"(cells:"+cells_to_use+") ...\n")
            session.cheese_board(experiment_phase=exp_phase).find_and_fit_optimal_number_of_modes(cells_to_use=
                                                                                                  cells_to_use,
                                                                                                  cl_ar_init=cl_ar_init)
            print(" \n... done with cross-val pHMM, session: " + session.session_name+"\n")

    def cheeseboard_place_field_goal_distance(self, save_fig=False, mean_firing_threshold=None, pre_or_post="pre",
                                              nr_trials=4):
        """
        computes distance between place field peak and closest goal

        @param save_fig: save as .svg
        @param mean_firing_threshold: threshold to exclude low firing cells (in Hz)
        """

        if pre_or_post == "pre":
            experiment_phase = ["learning_cheeseboard_1"]
        elif pre_or_post == "post":
            experiment_phase = ["learning_cheeseboard_2"]


        dist_stable = []
        dist_dec = []
        dist_inc = []

        for session in self.session_list:
            d_s, d_d, d_i = session.cheese_board(experiment_phase=experiment_phase).place_field_goal_distance(plotting=False,
                                                                                                  mean_firing_threshold=
                                                                                                  mean_firing_threshold)
            dist_stable.append(d_s)
            dist_dec.append(d_d)
            dist_inc.append(d_i)

        dist_stable = np.hstack(dist_stable)
        dist_dec = np.hstack(dist_dec)
        dist_inc = np.hstack(dist_inc)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        if pre_or_post == "pre":
            plt.figure(figsize=(4, 5))
            res = [dist_stable, dist_dec]
            bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                                labels=["Stable", "Decreasing"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", 'turquoise']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal (cm)")
            plt.grid(color="grey", axis="y")
            print(mannwhitneyu(dist_stable, dist_dec, alternative="less"))
            plt.ylim(0,60)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("place_field_to_goal_distance_pre.svg", transparent="True")
            else:
                plt.show()

        elif pre_or_post == "post":
            plt.figure(figsize=(4, 5))
            res = [dist_stable, dist_inc]
            bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                                labels=["Stable", "Increasing"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", 'orange']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal (cm)")
            plt.grid(color="grey", axis="y")
            print(mannwhitneyu(dist_stable, dist_inc, alternative="less"))
            plt.ylim(0,60)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("place_field_to_goal_distance_post.svg", transparent="True")
            else:
                plt.show()

    def cheeseboard_firing_rates_during_swr(self, time_bin_size=0.01, experiment_phase=["learning_cheeseboard_1"]):
        """
        compares firing rates during SWR for stable/decreasing/increasing cells

        @param time_bin_size: which time bin size (in s) to use for the computation of firing rates
        @param experiment_phase: PRE (["learning_cheeseboard_1"]) or POST (["learning_cheeseboard_2"])
        """
        swr_mean_firing_rates_stable = []
        swr_mean_firing_rates_dec = []
        swr_mean_firing_rates_inc = []

        for session in self.session_list:
            stable, dec, inc = session.cheese_board(experiment_phase=
                                                    experiment_phase,
                                                    data_to_use="ext").firing_rates_during_swr(time_bin_size=time_bin_size,
                                                                                               plotting=False)
            swr_mean_firing_rates_stable.append(stable)
            swr_mean_firing_rates_dec.append(dec)
            swr_mean_firing_rates_inc.append(inc)

        swr_mean_firing_rates_stable = np.hstack(swr_mean_firing_rates_stable)
        swr_mean_firing_rates_dec = np.hstack(swr_mean_firing_rates_dec)
        swr_mean_firing_rates_inc = np.hstack(swr_mean_firing_rates_inc)

        p_stable = 1. * np.arange(swr_mean_firing_rates_stable.shape[0]) / (swr_mean_firing_rates_stable.shape[0] - 1)
        p_inc = 1. * np.arange(swr_mean_firing_rates_inc.shape[0]) / (swr_mean_firing_rates_inc.shape[0] - 1)
        p_dec = 1. * np.arange(swr_mean_firing_rates_dec.shape[0]) / (swr_mean_firing_rates_dec.shape[0] - 1)

        plt.plot(np.sort(swr_mean_firing_rates_stable), p_stable, color="violet", label="stable")
        plt.plot(np.sort(swr_mean_firing_rates_dec), p_dec, color="turquoise", label="dec")
        plt.plot(np.sort(swr_mean_firing_rates_inc), p_inc, color="orange", label="inc")
        plt.ylabel("cdf")
        plt.xlabel("Mean firing rate during SWRs")
        plt.legend()
        plt.show()

    def cheeseboard_firing_rates_gain_during_swr(self, time_bin_size=0.01, threshold_stillness=15,
                                     experiment_phase=["learning_cheeseboard_1"], save_fig=False, threshold_firing=1):

        swr_gain_stable = []
        swr_gain_dec = []
        swr_gain_inc = []

        for session in self.session_list:
            stable, dec, inc = session.cheese_board(experiment_phase=
                                                    experiment_phase, data_to_use="ext").firing_rates_gain_during_swr(
                time_bin_size=time_bin_size, plotting=False, threshold_stillness=threshold_stillness, threshold_firing=threshold_firing)
            swr_gain_stable.append(stable)
            swr_gain_dec.append(dec)
            swr_gain_inc.append(inc)

        swr_gain_stable = np.hstack(swr_gain_stable)
        swr_gain_dec = np.hstack(swr_gain_dec)
        swr_gain_inc = np.hstack(swr_gain_inc)

        # filter out nan
        swr_gain_stable = swr_gain_stable[~np.isnan(swr_gain_stable)]
        swr_gain_dec = swr_gain_dec[~np.isnan(swr_gain_dec)]
        swr_gain_inc = swr_gain_inc[~np.isnan(swr_gain_inc)]

        # swr_gain_stable_log = np.log(swr_gain_stable+1e-15)
        # swr_gain_dec_log = np.log(swr_gain_dec+1e-15)
        # swr_gain_inc_log = np.log(swr_gain_inc+1e-15)

        p_stable = 1. * np.arange(swr_gain_stable.shape[0]) / (swr_gain_stable.shape[0] - 1)
        p_inc = 1. * np.arange(swr_gain_inc.shape[0]) / (swr_gain_inc.shape[0] - 1)
        p_dec = 1. * np.arange(swr_gain_dec.shape[0]) / (swr_gain_dec.shape[0] - 1)

        print("Two-sided: stable vs. inc:")
        print(mannwhitneyu(swr_gain_stable, swr_gain_inc))
        print("One-sided: stable vs. inc:")
        print(mannwhitneyu(swr_gain_stable, swr_gain_inc, alternative="greater"))
        print("Two-sided: stable vs. dec:")
        print(mannwhitneyu(swr_gain_stable, swr_gain_dec))
        print("One-sided: stable vs. dec:")
        print(mannwhitneyu(swr_gain_stable, swr_gain_dec, alternative="greater"))

        if save_fig:
            plt.style.use('default')
        plt.plot(np.sort(swr_gain_stable), p_stable, color="violet", label="stable")
        plt.plot(np.sort(swr_gain_dec), p_dec, color="turquoise", label="dec")
        plt.plot(np.sort(swr_gain_inc), p_inc, color="orange", label="inc")
        plt.ylabel("cdf")
        plt.xlabel("Within ripple - firing rate gain")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("within_swr_gain.svg", transparent="True")
        else:
            plt.show()

    def phase_preference_analysis(self, oscillation="theta", tetrode=10, sleep_or_awake="awake"):

        all_positive_angles_stable = []
        all_positive_angles_dec = []
        all_positive_angles_inc = []

        for session in self.session_list:
            if sleep_or_awake == "awake":
                stable, dec, inc = session.cheese_board(experiment_phase=["learning_cheeseboard_1"],
                                                        data_to_use="ext").phase_preference_analysis(tetrode=tetrode,
                                                                                                     plotting=False,
                                                                                                     oscillation=oscillation)
            elif sleep_or_awake == "sleep":
                stable, dec, inc = session.sleep(experiment_phase=["sleep_long_1"],
                                                        data_to_use="ext").phase_preference_analysis(tetrode=tetrode,
                                                                                                     plotting=False,
                                                                                                     oscillation=oscillation)

            all_positive_angles_stable.append(stable)
            all_positive_angles_dec.append(dec)
            all_positive_angles_inc.append(inc)

        all_positive_angles_stable = np.hstack(all_positive_angles_stable)
        all_positive_angles_dec = np.hstack(all_positive_angles_dec)
        all_positive_angles_inc = np.hstack(all_positive_angles_inc)

        bins_number = 10  # the [0, 360) interval will be subdivided into this
        # number of equal bins
        bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
        angles = all_positive_angles_stable
        n, _, _ = plt.hist(angles, bins, density=True)
        plt.title("stable")
        plt.show()
        plt.clf()
        width = 2 * np.pi / bins_number
        ax = plt.subplot(1, 1, 1, projection='polar')
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            bar.set_alpha(0.5)
        ax.set_title("stable cells")
        plt.show()


        angles = all_positive_angles_dec
        n, _, _ = plt.hist(angles, bins, density=True)
        plt.title("dec")
        plt.show()
        plt.clf()
        width = 2 * np.pi / bins_number
        ax = plt.subplot(1, 1, 1, projection='polar')
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            bar.set_alpha(0.5)
        ax.set_title("dec. cells")
        plt.show()

        angles = all_positive_angles_inc
        n, _, _ = plt.hist(angles, bins, density=True)
        plt.title("inc")
        plt.show()

        plt.clf()
        width = 2 * np.pi / bins_number
        ax = plt.subplot(1, 1, 1, projection='polar')
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
        for bar in bars:
            bar.set_alpha(0.5)
        ax.set_title("inc. cells")
        plt.show()

    """#################################################################################################################
    #  pre cheeseboard & post cheeseboard
    #################################################################################################################"""

    def pre_post_cheeseboard_remapping_stable_cells(self, lump_data_together=True, save_fig=False):
        """
        Checks remapping of stable cells between PRE and POST using (spatial) population vector similarity or single
        cell rate maps

        :param lump_data_together: lump data from all sessions together
        :type lump_data_together: bool
        :param save_fig: whether to save figure
        :type save_fig: bool
        """
        if lump_data_together:
            # go trough all sessions to collect results
            remapping_stable_list = []
            remapping_shuffle_list = []
            remapping_pv_list = []
            remapping_pv_shuffle_list = []
            for session in self.session_list:
                stable, shuffle, pv, pv_shuffle = session.pre_post_cheeseboard().remapping(plot_results=False,
                                                                                           return_distribution=True)
                remapping_stable_list.append(stable)
                remapping_shuffle_list.append(shuffle.flatten())
                remapping_pv_list.append(pv)
                remapping_pv_shuffle_list.append(pv_shuffle)

            # ----------------------------------------------------------------------------------------------------------
            # compute for population vectors
            # ----------------------------------------------------------------------------------------------------------
            remapping_pv = np.hstack((remapping_pv_list))
            remapping_pv_shuffle = np.hstack((remapping_pv_shuffle_list))

            remapping_pv = remapping_pv[~np.isnan(remapping_pv)]
            remapping_pv_shuffle = remapping_pv_shuffle[~np.isnan(remapping_pv_shuffle)]

            print("MWU for PV remapping:"+str(mannwhitneyu(remapping_pv, remapping_pv_shuffle)[1]))
            c = "white"
            if save_fig:
                plt.style.use('default')
                c = "black"
            plt.figure(figsize=(4, 5))
            res = [remapping_pv, remapping_pv_shuffle]
            bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                                labels=["Population vectors stable cells", "Shuffle"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", 'gray']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Pearson R: PRE map - POST map")
            plt.yticks([-0.5, 0, 0.5, 1])
            plt.ylim(-0.5, 1)
            plt.grid(color="grey", axis="y")

            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("persistent_map.svg", transparent="True")
            else:
                plt.title("Remapping using Population Vectors")
                plt.show()

            # ----------------------------------------------------------------------------------------------------------
            # compute for single cells
            # ----------------------------------------------------------------------------------------------------------
            remapping_stable_arr = np.expand_dims(np.hstack((remapping_stable_list)),1)
            remapping_shuffle_arr = np.expand_dims(np.hstack((remapping_shuffle_list)),1)

            remapping_stable_arr = remapping_stable_arr[~np.isnan(remapping_stable_arr)]
            remapping_shuffle_arr = remapping_shuffle_arr[~np.isnan(remapping_shuffle_arr)]

            c = "black"
            res = [remapping_stable_arr, remapping_shuffle_arr]
            bplot = plt.boxplot(res, positions=[1,2], patch_artist=True,
                                labels=["Stable", "Shuffle"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c),showfliers=False
                                )
            colors = ["magenta", 'gray']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.title("Remapping using single cells")
            plt.ylabel("Pearson R: PRE map - POST map")
            plt.grid(color="grey", axis="y")
            plt.show()

        else:

            # go trough all sessions to collect results
            perc_stable = []
            p_value_ks = []
            for session in self.session_list:
                perc, p = session.pre_post_cheeseboard().remapping(plot_results=False)
                perc_stable.append(perc)
                p_value_ks.append(p)
                # res.append(session.pre_post_cheeseboard().pre_post_firing_rates())

            perc_stable = np.vstack(perc_stable)
            plt.scatter(range(len(self.session_list)), perc_stable)
            plt.xlabel("SESSION ID")
            plt.ylabel("%STABLE CELLS WITH CONSTANT RATE MAPS")
            plt.title("RATE MAP STABILITY PRE - POST: %STABLE")
            plt.grid(c="gray")
            plt.ylim(0,100)
            plt.show()

            p_value_ks = np.vstack(p_value_ks)
            plt.scatter(range(len(self.session_list)), p_value_ks)
            plt.xlabel("SESSION ID")
            plt.ylabel("P-VALUE KS")
            plt.title("RATE MAP STABILITY PRE - POST: KS")
            plt.grid(c="gray")
            plt.ylim(0,max(p_value_ks))
            plt.show()

    def pre_post_cheeseboard_goal_coding(self):
        # go trough all sessions to collect results
        pre_res = []
        post_res = []
        pre_dec = []
        post_inc = []
        session_name_strings =[]
        for session in self.session_list:
            pre, post, pre_d, post_i = session.pre_post_cheeseboard().goal_coding(plotting=False)
            pre_res.append(pre)
            post_res.append(post)
            pre_dec.append(pre_d)
            post_inc.append(post_i)
            # res.append(session.pre_post_cheeseboard().pre_post_firing_rates())
            session_name_strings.append(session.session_name)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(range(len(self.session_list)), pre_res, label ="STABLE", color="y")
        plt.scatter(range(len(self.session_list)), pre_dec, label ="DEC", color="b")
        plt.xticks(range(len(self.session_list)), session_name_strings, rotation='vertical')
        plt.ylabel("MEDIAN GOAL CODING INDEX")
        plt.title("GOAL CODING PRE: STABLE VS. DEC")
        plt.grid(c="gray")
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(range(len(self.session_list)), post_res, label ="STABLE", color="y")
        plt.scatter(range(len(self.session_list)), post_inc, label ="INC", color="r")
        plt.xticks(range(len(self.session_list)), session_name_strings, rotation='vertical')
        plt.ylabel("MEDIAN GOAL CODING INDEX")
        plt.title("GOAL CODING POST: STABLE VS. INC")
        plt.grid(c="gray")
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(range(len(self.session_list)), pre_res, label ="PRE", color="r")
        plt.scatter(range(len(self.session_list)), post_res, label ="POST", color="b")
        plt.xticks(range(len(self.session_list)), session_name_strings, rotation='vertical')
        plt.ylabel("MEDIAN GOAL CODING INDEX")
        plt.title("GOAL CODING STABLE CELLS: PRE VS. POST")
        plt.grid(c="gray")
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

    def pre_post_cheeseboard_goal_coding_stable_cells_phmm(self):
        # go trough all sessions to collect results
        gain_res = []
        session_name_strings = []
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for session in self.session_list:
            gain = session.pre_post_cheeseboard().pre_post_models_goal_coding(gc_threshold=0.9)
            gain_res.append(gain)
            # res.append(session.pre_post_cheeseboard().pre_post_firing_rates())
            session_name_strings.append(session.session_name)

        plt.scatter(range(len(self.session_list)), gain_res, color="r")
        plt.xticks(range(len(self.session_list)), session_name_strings, rotation='vertical')
        plt.ylabel("GAIN IN GOAL CODING (POST - PRE)")
        plt.title("GOAL CODING GAIN FROM PRE TO POST \n USING LAMBDA AND MEAN FIRING RATE")
        plt.grid(c="gray")
        plt.show()

    def pre_post_cheeseboard_nr_goals(self, single_cells=True, mean_firing_thresh=None):
        # go trough all sessions to collect results
        pre_stable = []
        pre_dec = []
        post_stable = []
        post_inc = []
        session_name_strings = []

        for session in self.session_list:
            pre_s, pre_d, post_s, post_i = session.pre_post_cheeseboard().nr_goals_coded(plotting=False,
                                                                                         single_cells=single_cells,
                                                                                         mean_firing_thresh=
                                                                                         mean_firing_thresh)
            pre_stable.append(pre_s)
            pre_dec.append(pre_d)
            post_stable.append(post_s)
            post_inc.append(post_i)

            # res.append(session.pre_post_cheeseboard().pre_post_firing_rates())
            session_name_strings.append(session.session_name)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(range(len(self.session_list)), pre_stable, label ="STABLE", color="y")
        plt.scatter(range(len(self.session_list)), pre_dec, label ="DEC", color="b")
        plt.xticks(range(len(self.session_list)), session_name_strings, rotation='vertical')
        if single_cells:
            plt.ylabel("MEDIAN NR. GOALS CODED")
            plt.title("NR. GOALS CODED PER CELL: PRE")
        else:
            plt.ylabel("NR. GOALS CODED")
            plt.title("NR. GOALS FOR SUBSET: PRE")
        plt.grid(c="gray")
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(range(len(self.session_list)), post_stable, label ="STABLE", color="y")
        plt.scatter(range(len(self.session_list)), post_inc, label ="INC", color="r")
        plt.xticks(range(len(self.session_list)), session_name_strings, rotation='vertical')
        if single_cells:
            plt.ylabel("MEDIAN NR. GOALS CODED")
            plt.title("NR. GOALS CODED PER CELL: POST")
        else:
            plt.ylabel("NR. GOALS CODED")
            plt.title("NR. GOALS FOR SUBSET: POST")
        plt.grid(c="gray")
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

    def pre_post_cheeseboard_spatial_information(self, spatial_resolution=2, info_measure="sparsity",
                                                 lump_data_together=True, save_fig=False, plotting=True,
                                                 remove_nan=False, mean_firing_thresh=None):
        """
        computes spatial information of different subsets of cells (stable, increasing, decreasing)


        :param spatial_resolution: in cm to use for spatial binning
        :type spatial_resolution: int
        :param info_measure: which spatial information measure to use ("sparsity", "skaggs_info")
        :type info_measure: str
        :param lump_data_together: lump cells from all sessions together or keep them separate
        :type lump_data_together: bool
        :param save_fig: whether to save figure
        :type bool
        """
        if lump_data_together:
            pre_stable_list = []
            post_stable_list = []
            pre_dec_list = []
            post_inc_list = []
            post_dec_list = []
            pre_inc_list = []
            mean_firing_pre_stable = []
            mean_firing_pre_dec = []
            mean_firing_pre_inc = []
            mean_firing_post_stable = []
            mean_firing_post_dec = []
            mean_firing_post_inc = []
            for session in self.session_list:
                pre_stable, post_stable, pre_dec, post_inc, post_dec, pre_inc = \
                    session.pre_post_cheeseboard().spatial_information_per_cell(plotting=False,
                                                                                spatial_resolution=spatial_resolution,
                                                                                info_measure=info_measure,
                                                                                return_p_values=False,
                                                                                remove_nan=remove_nan)
                f_mean_pre_s, f_mean_post_s = session.pre_post_cheeseboard().firing_rates(mean_or_max="mean", cells_to_use="stable")
                f_mean_pre_d, f_mean_post_d = session.pre_post_cheeseboard().firing_rates(mean_or_max="mean", cells_to_use="decreasing")
                f_mean_pre_i, f_mean_post_i = session.pre_post_cheeseboard().firing_rates(mean_or_max="mean", cells_to_use="increasing")
                mean_firing_pre_stable.append(f_mean_pre_s)
                mean_firing_pre_dec.append(f_mean_pre_d)
                mean_firing_pre_inc.append(f_mean_pre_i)
                mean_firing_post_stable.append(f_mean_post_s)
                mean_firing_post_dec.append(f_mean_post_d)
                mean_firing_post_inc.append(f_mean_post_i)
                pre_stable_list.append(pre_stable)
                post_stable_list.append(post_stable)
                pre_dec_list.append(pre_dec)
                post_inc_list.append(post_inc)
                post_dec_list.append(post_dec)
                pre_inc_list.append(pre_inc)

            mean_firing_pre_stable = np.hstack(mean_firing_pre_stable)
            mean_firing_pre_dec = np.hstack(mean_firing_pre_dec)
            mean_firing_pre_inc = np.hstack(mean_firing_pre_inc)
            mean_firing_post_stable = np.hstack(mean_firing_post_stable)
            mean_firing_post_dec = np.hstack(mean_firing_post_dec)
            mean_firing_post_inc = np.hstack(mean_firing_post_inc)
            pre_stable_arr = np.hstack(pre_stable_list)
            post_stable_arr = np.hstack(post_stable_list)
            pre_dec_arr = np.hstack(pre_dec_list)
            post_inc_arr = np.hstack(post_inc_list)
            post_dec_arr = np.hstack(post_dec_list)
            pre_inc_arr = np.hstack(pre_inc_list)

            if mean_firing_thresh is not None:
                pre_dec_arr = pre_dec_arr[mean_firing_pre_dec > mean_firing_thresh]
                pre_stable_arr = pre_stable_arr[mean_firing_pre_stable > mean_firing_thresh]
                pre_inc_arr = pre_inc_arr[mean_firing_pre_inc > mean_firing_thresh]
                post_dec_arr = post_dec_arr[mean_firing_post_dec > mean_firing_thresh]
                post_stable_arr = post_stable_arr[mean_firing_post_stable > mean_firing_thresh]
                post_inc_arr = post_inc_arr[mean_firing_post_inc > mean_firing_thresh]

                print("nr. stable PRE: "+str(pre_stable_arr.shape[0]))
                print("nr. dec PRE: "+str(pre_dec_arr.shape[0]))
                print("nr. inc PRE: "+str(pre_inc_arr.shape[0]))

                print("nr. stable POST: "+str(post_stable_arr.shape[0]))
                print("nr. dec POST: "+str(post_dec_arr.shape[0]))
                print("nr. inc POST: "+str(post_inc_arr.shape[0]))


            # do stats tests
            if info_measure == "sparsity":
                p_pre_mwu_one_sided = mannwhitneyu(np.nan_to_num(pre_dec_arr), np.nan_to_num(pre_stable_arr),
                                                    alternative="greater")[1]
                print("p-value PRE: "  +str(p_pre_mwu_one_sided))

                p_post_mwu_one_sided = mannwhitneyu(np.nan_to_num(post_inc_arr), np.nan_to_num(post_stable_arr),
                                                    alternative="greater")[1]
                print("p-value POST: " +str(p_post_mwu_one_sided))

            else:

                p_pre_mwu_one_sided = mannwhitneyu(np.nan_to_num(pre_dec_arr), np.nan_to_num(pre_stable_arr))[1]
                print("p-value PRE: " + str(p_pre_mwu_one_sided))

                p_post_mwu_one_sided = mannwhitneyu(np.nan_to_num(post_inc_arr), np.nan_to_num(post_stable_arr))[1]
                print("p-value POST: " + str(p_post_mwu_one_sided))

            # compute for stable cells only
            p_stable_mwu = mannwhitneyu(np.nan_to_num(post_stable_arr), np.nan_to_num(pre_stable_arr))[1]

            # compare decreasing and increasing
            p_dec_inc_mwu = mannwhitneyu(np.nan_to_num(post_inc_arr), np.nan_to_num(pre_dec_arr))[1]
            print("p-value INC-DEC: " + str(p_dec_inc_mwu))
            # sort for CDF
            pre_stable_sorted = np.sort(pre_stable_arr)
            post_stable_sorted = np.sort(post_stable_arr)
            pre_dec_sorted = np.sort(pre_dec_arr)
            post_inc_sorted = np.sort(post_inc_arr)
            pre_inc_sorted = np.sort(pre_inc_arr)
            post_dec_sorted = np.sort(post_dec_arr)

            if plotting or save_fig:
                if save_fig:
                    plt.style.use('default')

                # ----------------------------------------------------------------------------------------------------------
                # PRE
                # ----------------------------------------------------------------------------------------------------------
                p_pre_stable = 1. * np.arange(pre_stable_sorted.shape[0]) / (pre_stable_sorted.shape[0] - 1)
                p_pre_dec = 1. * np.arange(pre_dec_sorted.shape[0]) / (pre_dec_sorted.shape[0] - 1)
                p_pre_inc = 1. * np.arange(pre_inc_sorted.shape[0]) / (pre_inc_sorted.shape[0] - 1)
                # plt.hlines(0.5, -0.02, 0.85, color="gray", linewidth=0.5)
                plt.plot(pre_stable_sorted, p_pre_stable, color="magenta", label="Stable")
                plt.plot(pre_dec_sorted, p_pre_dec, color="turquoise", label="Decreasing")
                # plt.plot(pre_inc_sorted, p_pre_inc, color="orange", label="Increasing")
                # plt.yticks([0, 0.25, 0.5, 0.75, 1])
                # plt.xticks([0, 0.45, 0.9])
                # plt.xlim(-0.05, .95)
                # plt.ylim(-0.05, 1.05)
                plt.ylabel("cdf")
                plt.xlabel(info_measure)
                plt.legend()
                if save_fig:
                    if info_measure == "sparsity":
                        plt.xlim(-0.02, 1.02)
                    elif info_measure == "skaggs_second":
                        plt.xlim(-0.2, 5.2)
                    plt.rcParams['svg.fonttype'] = 'none'
                    plt.savefig(info_measure+"_pre.svg", transparent="True")
                    plt.close()
                else:
                    plt.title(
                        "PRE: STABLE CELLS vs. DEC. CELLS \n" + "MWU , p-value = " + str(np.round(p_pre_mwu_one_sided, 5)))
                    plt.show()

                # ----------------------------------------------------------------------------------------------------------
                # POST
                # ----------------------------------------------------------------------------------------------------------
                # calculate the proportional values of samples
                p_post_stable = 1. * np.arange(post_stable_sorted.shape[0]) / (post_stable_sorted.shape[0] - 1)
                p_post_inc = 1. * np.arange(post_inc_sorted.shape[0]) / (post_inc_sorted.shape[0] - 1)
                p_post_dec = 1. * np.arange(post_dec_sorted.shape[0]) / (post_dec_sorted.shape[0] - 1)
                plt.plot(post_stable_sorted, p_post_stable, color="magenta", label="Stable")
                plt.plot(post_inc_sorted, p_post_inc, color="orange", label="Increasing")
                # plt.plot(post_dec_sorted, p_post_dec, color="turquoise", label="Decreasing")
                plt.ylabel("cdf")
                # plt.yticks([0, 0.25, 0.5, 0.75, 1])
                # plt.xticks([0, 0.45, 0.9])
                # plt.xlim(-0.05, 0.95)
                # plt.ylim(-0.02, 1.02)
                plt.xlabel(info_measure)
                # plt.hlines(0.5, -0.02, 0.87, color="gray", linewidth=0.5)
                plt.legend()
                if save_fig:
                    if info_measure == "sparsity":
                        plt.xlim(-0.02, 1.02)
                    elif info_measure == "skaggs_second":
                        plt.xlim(-0.4, 5.4)
                    plt.rcParams['svg.fonttype'] = 'none'
                    plt.savefig(info_measure+"_post.svg", transparent="True")
                    plt.close()
                else:
                    plt.title(
                        "POST: STABLE CELLS vs. INC. CELLS \n" + "MWU , p-value = " + str(np.round(p_post_mwu_one_sided, 5)))
                    plt.show()

                # spatial information PRE - POST for stable cells
                plt.plot(pre_stable_sorted, p_pre_stable, color="red", label="PRE")
                plt.plot(post_stable_sorted, p_post_stable, color="blue", label="POST")
                plt.ylabel("cdf")
                plt.xlabel(info_measure)
                plt.legend()
                if save_fig:
                    if info_measure == "sparsity":
                        plt.xlim(-0.02, 1.02)
                    elif info_measure == "skaggs_second":
                        plt.xlim(-0.2, 5.2)
                    plt.rcParams['svg.fonttype'] = 'none'
                    plt.savefig(info_measure+"_stable_pre_post.svg", transparent="True")
                    plt.close()
                else:
                    plt.title("PRE-POST, STABLE CELLS \n" + "MWU , p-value = " + str(np.round(p_stable_mwu, 5)))
                    plt.show()

                # spatial information increasing / decreasing cells
                # spatial information PRE - POST for stable cells
                plt.plot(pre_dec_sorted, p_pre_dec, color="#a0c4e4", label="Decreasing (PRE)")
                plt.plot(post_inc_sorted, p_post_inc, color="#f7959c", label="Increasing (POST)")
                plt.ylabel("cdf")
                # plt.yticks([0, 0.25, 0.5, 0.75, 1])
                # plt.xticks([0, 0.45, 0.9])
                # plt.xlim(-0.05, 0.95)
                plt.xlabel(info_measure)
                plt.legend()
                if save_fig:
                    if info_measure == "sparsity":
                        plt.xlim(-0.02, 1.02)
                    elif info_measure == "skaggs_second":
                        plt.xlim(-0.4, 5.4)
                    plt.rcParams['svg.fonttype'] = 'none'
                    plt.savefig(info_measure+"_dec_inc.svg", transparent="True")
                    plt.close()
                else:
                    plt.title("INC-DEC \n" + "MWU , p-value = " + str(np.round(p_dec_inc_mwu, 5)))
                    plt.show()
            else:

                return pre_stable_arr, post_stable_arr, pre_dec_arr, post_inc_arr, post_dec_arr, pre_inc_arr
        else:
            pre_list = []
            post_list = []
            stable_list = []
            all_list = []
            for session in self.session_list:
                pre, post, stable, all = session.pre_post_cheeseboard().spatial_information_per_cell(plotting=False,
                                                                                spatial_resolution=spatial_resolution,
                                                                                info_measure=info_measure)
                all_list.append(all)
                pre_list.append(pre)
                post_list.append(post)
                stable_list.append(stable)

            for i, p in enumerate(pre_list):
                plt.scatter(i,p)
            plt.ylabel("p-value")
            plt.title("PRE")
            plt.hlines(0.05, 0, len(pre_list), color="r")
            plt.show()

            for i, p in enumerate(post_list):
                plt.scatter(i,p)
            plt.ylabel("p-value")
            plt.hlines(0.05, 0, len(post_list), color="r")
            plt.title("POST")
            plt.show()

            for i, p in enumerate(stable_list):
                plt.scatter(i,p)
            plt.ylabel("p-value")
            plt.title("STABLE")
            plt.show()

            for i, p in enumerate(all_list):
                plt.scatter(i,p)
            plt.ylabel("p-value")
            plt.title("STABLE vs. NON-STABLE")
            plt.hlines(0.05, 0, len(post_list), color="r")
            plt.show()

    def pre_post_cheeseboard_firing_rate_changes(self, mean_or_max="mean"):
        pre_stable_list = []
        pre_dec_list = []
        post_stable_list = []
        post_inc_list = []
        for session in self.session_list:
            pre_stable, pre_dec, post_stable, post_inc = \
                session.pre_post_cheeseboard().firing_rate_changes(plotting=False, mean_or_max=mean_or_max)
            pre_stable_list.append(pre_stable)
            pre_dec_list.append(pre_dec)
            post_stable_list.append(post_stable)
            post_inc_list.append(post_inc)

        pre_stable_arr = np.hstack(pre_stable_list)
        post_stable_arr = np.hstack(post_stable_list)
        pre_dec_arr = np.hstack(pre_dec_list)
        post_inc_arr = np.hstack(post_inc_list)

        # sort for CDF
        pre_stable_sorted = np.sort(pre_stable_arr)
        post_stable_sorted = np.sort(post_stable_arr)
        pre_dec_sorted = np.sort(pre_dec_arr)
        post_inc_sorted = np.sort(post_inc_arr)

        p_pre_stable = 1. * np.arange(pre_stable_sorted.shape[0]) / (pre_stable_sorted.shape[0] - 1)
        p_pre_dec = 1. * np.arange(pre_dec_sorted.shape[0]) / (pre_dec_sorted.shape[0] - 1)
        # plt.hlines(0.5, -0.02, 0.85, color="gray", linewidth=0.5)
        plt.plot(pre_stable_sorted, p_pre_stable, color="#ffdba1", label="STABLE")
        plt.plot(pre_dec_sorted, p_pre_dec, color="#a0c4e4", label="DECREASING")
        # plt.yticks([0, 0.5, 1])
        # plt.xticks([0, 0.415, 0.83])
        # plt.xlim(-0.02, 0.85)
        # plt.ylim(-0.02, 1.02)
        plt.ylabel("CDF")
        plt.xlabel(mean_or_max+" firing")

        plt.legend()
        plt.show()

        p_post_stable = 1. * np.arange(post_stable_sorted.shape[0]) / (post_stable_sorted.shape[0] - 1)
        p_post_inc = 1. * np.arange(post_inc_sorted.shape[0]) / (post_inc_sorted.shape[0] - 1)
        plt.plot(post_stable_sorted, p_post_stable, color="#ffdba1", label="STABLE")
        plt.plot(post_inc_sorted, p_post_inc, color="#f7959c", label="INCREASING")
        plt.ylabel("CDF")
        # plt.yticks([0, 0.5, 1])
        # plt.xticks([0, 0.425, 0.85])
        # plt.xlim(-0.02, 0.87)
        # plt.ylim(-0.02, 1.02)
        plt.xlabel(mean_or_max+" firing")
        # plt.hlines(0.5, -0.02, 0.87, color="gray", linewidth=0.5)
        plt.legend()
        plt.show()

    def pre_post_cheeseboard_firing_around_goals(self, mean_or_max="mean"):
        pre_stable_list = []
        pre_dec_list = []
        post_stable_list = []
        post_inc_list = []
        for session in self.session_list:
            pre_stable, pre_dec, post_stable, post_inc = \
                session.pre_post_cheeseboard().distance_peak_firing_closest_goal(plotting=False)
            pre_stable_list.append(pre_stable)
            pre_dec_list.append(pre_dec)
            post_stable_list.append(post_stable)
            post_inc_list.append(post_inc)

        pre_stable_arr = np.hstack(pre_stable_list)
        post_stable_arr = np.hstack(post_stable_list)
        pre_dec_arr = np.hstack(pre_dec_list)
        post_inc_arr = np.hstack(post_inc_list)

        # sort for CDF
        pre_stable_sorted = np.sort(pre_stable_arr)
        post_stable_sorted = np.sort(post_stable_arr)
        pre_dec_sorted = np.sort(pre_dec_arr)
        post_inc_sorted = np.sort(post_inc_arr)

        p_pre_stable = 1. * np.arange(pre_stable_sorted.shape[0]) / (pre_stable_sorted.shape[0] - 1)
        p_pre_dec = 1. * np.arange(pre_dec_sorted.shape[0]) / (pre_dec_sorted.shape[0] - 1)
        plt.plot(pre_stable_sorted, p_pre_stable, color="#ffdba1", label="STABLE")
        plt.plot(pre_dec_sorted, p_pre_dec, color="#a0c4e4", label="DECREASING")
        plt.ylabel("CDF")
        plt.xlabel("Min. distance: peak firing loc. to closest goal / cm")
        plt.title("PRE")
        plt.legend()
        plt.show()

        p_post_stable = 1. * np.arange(post_stable_sorted.shape[0]) / (post_stable_sorted.shape[0] - 1)
        p_post_inc = 1. * np.arange(post_inc_sorted.shape[0]) / (post_inc_sorted.shape[0] - 1)
        plt.plot(post_stable_sorted, p_post_stable, color="#ffdba1", label="STABLE")
        plt.plot(post_inc_sorted, p_post_inc, color="#f7959c", label="INCREASING")
        plt.ylabel("CDF")
        plt.xlabel("Min. distance: peak firing loc. to closest goal / cm")
        plt.title("POST")
        plt.legend()
        plt.show()

        plt.plot(pre_dec_sorted, p_pre_dec, color="#a0c4e4", label="DECREASING")
        plt.plot(post_inc_sorted, p_post_inc, color="#f7959c", label="INCREASING")
        plt.ylabel("CDF")
        plt.xlabel("Min. distance: peak firing loc. to closest goal / cm")
        plt.title("PRE-POST")
        plt.legend()
        plt.show()

        plt.plot(pre_stable_sorted, p_pre_stable, color="b", label="STABLE PRE")
        plt.plot(post_stable_sorted, p_post_stable, color="r", label="STABLE POST")
        plt.ylabel("CDF")
        plt.xlabel("Min. distance: peak firing loc. to closest goal / cm")
        plt.title("STABLE CELLS: PRE-POST")
        plt.legend()
        plt.show()

    def pre_post_cheeseboard_learning_vs_drift_stable_cells(self, spatial_resolution=2):

        # learning - stable
        remapping_per_cell_learn_dec = []
        remapping_per_cell_shuffle_learn_dec = []
        remapping_pv_learn_dec = []
        remapping_pv_shuffle_learn_dec = []
        # learning - dec
        remapping_per_cell_learn = []
        remapping_per_cell_shuffle_learn = []
        remapping_pv_learn = []
        remapping_pv_shuffle_learn = []
        # drift (PRE - POST)
        remapping_per_cell_drift = []
        remapping_per_cell_shuffle_drift = []
        remapping_pv_drift = []
        remapping_pv_shuffle_drift = []

        for session in self.session_list:
            # get drift results
            remap_cell_, remap_cell_shuffle_, remap_pv_, remap_pv_shuffle_ =\
                session.pre_post_cheeseboard().remapping_pre_post_stable(plot_results=False,
                                                                      spatial_resolution=spatial_resolution,
                                                                      nr_trials_to_use=5, return_distribution=True)

            remapping_per_cell_drift.append(remap_cell_)
            remapping_per_cell_shuffle_drift.append(remap_cell_shuffle_)
            remapping_pv_drift.append(remap_pv_)
            remapping_pv_shuffle_drift.append(remap_pv_shuffle_)

            # get learning results
            remap_cell, remap_cell_shuffle, remap_cell_dec, remap_cell_shuffle_dec, remap_pv, remap_pv_shuffle, \
            remap_pv_dec, remap_pv_shuffle_dec = \
                session.cheese_board(
                    experiment_phase=["learning_cheeseboard_1"]).map_dynamics_learning(plot_results=False,
                                                                                       adjust_pv_size=False,
                                                                                       spatial_resolution=spatial_resolution)
            # stable cells
            remapping_per_cell_learn.append(remap_cell)
            remapping_per_cell_shuffle_learn.append(remap_cell_shuffle)
            remapping_pv_learn.append(remap_pv)
            remapping_pv_shuffle_learn.append(remap_pv_shuffle)
            # decreasing cells
            remapping_per_cell_learn_dec.append(remap_cell_dec)
            remapping_per_cell_shuffle_learn_dec.append(remap_cell_shuffle_dec)
            remapping_pv_learn_dec.append(remap_pv_dec)
            remapping_pv_shuffle_learn_dec.append(remap_pv_shuffle_dec)

        # pre-process results from learning - stable
        remapping_per_cell_learn = np.hstack(remapping_per_cell_learn)
        remapping_per_cell_shuffle_learn = np.vstack(remapping_per_cell_shuffle_learn).flatten()
        remapping_pv_learn = np.hstack(remapping_pv_learn)
        remapping_pv_cell_shuffle_learn = np.vstack(remapping_per_cell_shuffle_learn).flatten()

        # pre-process results from learning - dec
        remapping_per_cell_learn_dec = np.hstack(remapping_per_cell_learn_dec)
        remapping_per_cell_shuffle_learn_dec = np.vstack(remapping_per_cell_shuffle_learn_dec).flatten()
        remapping_pv_learn_dec = np.hstack(remapping_pv_learn_dec)
        remapping_pv_cell_shuffle_learn_dec = np.vstack(remapping_per_cell_shuffle_learn_dec).flatten()

        # remove nans
        remapping_per_cell_learn = remapping_per_cell_learn[~np.isnan(remapping_per_cell_learn)]
        remapping_per_cell_shuffle_learn = remapping_per_cell_shuffle_learn[~np.isnan(remapping_per_cell_shuffle_learn)]
        remapping_pv_learn = remapping_pv_learn[~np.isnan(remapping_pv_learn)]
        remapping_pv_cell_shuffle_learn = remapping_pv_cell_shuffle_learn[~np.isnan(remapping_pv_cell_shuffle_learn)]

        remapping_per_cell_learn_sorted = np.sort(remapping_per_cell_learn)
        remapping_per_cell_shuffle_learn_sorted = np.sort(remapping_per_cell_shuffle_learn)
        remapping_pv_learn_sorted = np.sort(remapping_pv_learn)
        remapping_pv_cell_shuffle_learn_sorted = np.sort(remapping_pv_cell_shuffle_learn)

        p_remapping_per_cell_learn = 1. * np.arange(remapping_per_cell_learn.shape[0]) / (remapping_per_cell_learn.shape[0] - 1)
        p_remapping_per_cell_learn_shuffle = 1. * np.arange(remapping_per_cell_shuffle_learn.shape[0]) / (remapping_per_cell_shuffle_learn.shape[0] - 1)
        p_remapping_pv_learn = 1. * np.arange(remapping_pv_learn.shape[0]) / (remapping_pv_learn.shape[0] - 1)
        p_remapping_pv_cell_shuffle_learn = 1. * np.arange(remapping_pv_cell_shuffle_learn.shape[0]) / (remapping_pv_cell_shuffle_learn.shape[0] - 1)

        # pre-process results from drift
        remapping_per_cell_drift = np.hstack(remapping_per_cell_drift)
        remapping_per_cell_shuffle_drift = np.vstack(remapping_per_cell_shuffle_drift).flatten()
        remapping_pv_drift = np.hstack(remapping_pv_drift)
        remapping_pv_cell_shuffle_drift = np.vstack(remapping_per_cell_shuffle_drift).flatten()

        # remove nans
        remapping_pv_drift = remapping_pv_drift[~np.isnan(remapping_pv_drift)]
        remapping_pv_cell_shuffle_drift = remapping_pv_cell_shuffle_drift[~np.isnan(remapping_pv_cell_shuffle_drift)]
        remapping_per_cell_drift = remapping_per_cell_drift[~np.isnan(remapping_per_cell_drift)]
        remapping_per_cell_shuffle_drift = remapping_per_cell_shuffle_drift[~np.isnan(remapping_per_cell_shuffle_drift)]

        remapping_per_cell_drift_sorted = np.sort(remapping_per_cell_drift)
        remapping_per_cell_shuffle_drift_sorted = np.sort(remapping_per_cell_shuffle_drift)
        remapping_pv_drift_sorted = np.sort(remapping_pv_drift)
        remapping_pv_cell_shuffle_drift_sorted = np.sort(remapping_pv_cell_shuffle_drift)

        p_remapping_per_cell_drift = 1. * np.arange(remapping_per_cell_drift.shape[0]) / (remapping_per_cell_drift.shape[0] - 1)
        p_remapping_per_cell_drift_shuffle = 1. * np.arange(remapping_per_cell_shuffle_drift.shape[0]) / (remapping_per_cell_shuffle_drift.shape[0] - 1)
        p_remapping_pv_drift = 1. * np.arange(remapping_pv_drift.shape[0]) / (remapping_pv_drift.shape[0] - 1)
        p_remapping_pv_cell_shuffle_drift = 1. * np.arange(remapping_pv_cell_shuffle_drift.shape[0]) / (remapping_pv_cell_shuffle_drift.shape[0] - 1)


        res_cells = [remapping_per_cell_learn, remapping_per_cell_learn_dec, remapping_per_cell_drift]
        res_pv = [remapping_pv_learn, remapping_pv_learn_dec, remapping_pv_drift]

        c = "black"
        bplot = plt.boxplot(res_cells, positions=[1, 2, 3], patch_artist=True,
                            labels=["Learning stable", "Learning decreasing", "PRE-POST stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["yellow", 'blue', "green"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.title("Correlation of cell maps")
        plt.ylabel("Pearson R")
        plt.grid(color="grey", axis="y")
        plt.show()

        c = "black"
        bplot = plt.boxplot(res_pv, positions=[1, 2, 3], patch_artist=True,
                            labels=["Learning stable", "Learning decreasing", "PRE-POST stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["yellow", 'blue', "green"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.title("Correlation of PV maps")
        plt.ylabel("Pearson R")
        plt.grid(color="grey", axis="y")
        plt.show()



        plt.plot(remapping_per_cell_learn_sorted, p_remapping_per_cell_learn, label="Stable - Learning", color="magenta")
        plt.plot(remapping_per_cell_shuffle_learn_sorted, p_remapping_per_cell_learn_shuffle, label="Shuffle - Learning",
                 color="plum",linestyle="--")
        plt.plot(remapping_per_cell_drift_sorted, p_remapping_per_cell_drift, label="Stable - Drift", color="blue")
        plt.plot(remapping_per_cell_shuffle_drift_sorted, p_remapping_per_cell_drift_shuffle, label="Shuffle - Drift",
                 color="aliceblue",linestyle="--")
        plt.title("Per cell")
        plt.xlabel("Pearson R")
        plt.ylabel("CDF")
        plt.legend()
        plt.show()

        plt.plot(remapping_pv_learn_sorted, p_remapping_pv_learn, label="Stable - Learning", color="magenta")
        plt.plot(remapping_pv_cell_shuffle_learn_sorted, p_remapping_pv_cell_shuffle_learn, label="Shuffle - Learning",
                 color="plum",linestyle="--")
        plt.plot(remapping_pv_drift_sorted, p_remapping_pv_drift, label="Stable - Drift", color="blue")
        plt.plot(remapping_pv_cell_shuffle_drift_sorted, p_remapping_pv_cell_shuffle_drift, label="Shuffle - Drift",
                 color="aliceblue",linestyle="--")

        plt.title("PV")
        plt.xlabel("Pearson R")
        plt.ylabel("CDF")
        plt.legend()
        plt.show()

    def pre_post_cheeseboard_learning_vs_drift(self, spatial_resolution=5, save_fig=False):

        # learning - stable
        remapping_pv_learning = []
        remapping_pv_drift = []
        remapping_rm_learning = []
        remapping_rm_drift = []

        for session in self.session_list:
            # get drift results
            remapping_pv_learning_, remapping_pv_drift_, remapping_rm_learning_, remapping_rm_drift_ =\
                session.pre_post_cheeseboard().remapping_learning_vs_drift(plotting=False,
                                                                           spatial_resolution=spatial_resolution,
                                                                           nr_trials_to_use=5)
            remapping_pv_learning.append(remapping_pv_learning_)
            remapping_pv_drift.append(remapping_pv_drift_)
            remapping_rm_learning.append(remapping_rm_learning_)
            remapping_rm_drift.append(remapping_rm_drift_)

        remapping_pv_learning = np.hstack(remapping_pv_learning)
        remapping_pv_drift = np.hstack(remapping_pv_drift)
        remapping_rm_learning = np.hstack(remapping_rm_learning)
        remapping_rm_drift = np.hstack(remapping_rm_drift)

        print("pv")
        print(mannwhitneyu(remapping_pv_learning, remapping_pv_drift, alternative="two-sided"))
        print("rm")
        print(mannwhitneyu(remapping_rm_learning, remapping_rm_drift, alternative="two-sided"))

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        bplot = plt.boxplot([remapping_pv_learning, remapping_pv_drift], positions=[1, 2], patch_artist=True,
                            labels=["Learning", "PRE-POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["dimgrey", 'lightgrey']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Pop. vec. correlations (Pearson R)")
        plt.grid(color="grey", axis="y")
        plt.ylim(-0.2,1.19)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("learning_vs_drfit_pvs.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        bplot = plt.boxplot([remapping_rm_learning, remapping_rm_drift], positions=[1, 2], patch_artist=True,
                            labels=["Learning", "PRE-POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["dimgrey", 'lightgrey']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Rate map correlations (Pearson R)")
        plt.grid(color="grey", axis="y")
        plt.ylim(-0.2,1)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("learning_vs_drfit_ratemaps.svg", transparent="True")
        else:
            plt.show()

    def pre_post_cheeseboard_stable_cells_pre_post_decoding(self, save_fig=False):

        # pre decoding error
        pre = []
        # post decoding error
        post = []
        # post decoding error shuffle
        post_shuffle = []

        for session in self.session_list:
            # get drift results
            error_pre, error_post, error_post_shuffle=\
                session.pre_post_cheeseboard().location_decoding_stable_cells(plotting=False)

            pre.append(error_pre)
            post.append(error_post)
            post_shuffle.append(error_post_shuffle)

        # pre-process results from learning
        pre = np.hstack(pre)
        post = np.hstack(post)
        post_shuffle = np.hstack(post_shuffle)

        # stats
        p_pre_shuffle = mannwhitneyu(pre, post_shuffle)[1]
        print("PRE vs. Shuffle, p = " +str(p_pre_shuffle))

        p_post_shuffle = mannwhitneyu(post, post_shuffle, alternative="less")[1]
        print("POST vs. Shuffle, p = "+str(p_post_shuffle))

        pre_sorted = np.sort(pre)
        post_sorted = np.sort(post)
        post_shuffle_sorted = np.sort(post_shuffle)

        p_pre = 1. * np.arange(pre.shape[0]) / (pre.shape[0] - 1)
        p_post = 1. * np.arange(post.shape[0]) / (post.shape[0] - 1)
        p_post_shuffle = 1. * np.arange(post_shuffle.shape[0]) / (post_shuffle.shape[0] - 1)

        if save_fig:
            plt.style.use('default')

        # want error in cm and not arbitrary units
        pre_sorted = pre_sorted
        post_shuffle_sorted =post_shuffle_sorted
        post_sorted = post_sorted
        plt.plot(pre_sorted, p_pre, label="PRE", color="blue")
        plt.plot(post_sorted, p_post, label="POST",
                 color="magenta")
        plt.plot(post_shuffle_sorted, p_post_shuffle, label="POST Shuffle", color="plum", linestyle="--")
        plt.xlabel("Decoding error (cm)")
        plt.ylabel("cdf")
        plt.xticks([0, 25, 50, 75, 100, 125])
        plt.xlim(-10, 135)
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoding_error_pre_post.svg", transparent="True")
            plt.close()
        else:
            plt.show()

    def pre_post_place_field_goal_distance_stable_cells(self, save_fig=False, mean_firing_threshold=1, nr_trials=4):
        """
        computes distance between place field peak and closest goal

        @param save_fig: save as .svg
        @param mean_firing_threshold: threshold to exclude low firing cells (in Hz)
        """

        dist_stable_pre = []
        dist_stable_post = []

        for session in self.session_list:
            total_nr_trials = session.cheese_board(experiment_phase=["learning_cheeseboard_1"]).nr_trials
            d_s, _, _ = session.cheese_board(experiment_phase=["learning_cheeseboard_1"]).place_field_goal_distance(plotting=False,
                                                                                                  mean_firing_threshold=
                                                                                                  mean_firing_threshold,
                                                                                                  trials_to_use=range(total_nr_trials-nr_trials, total_nr_trials))
            dist_stable_pre.append(d_s)

        for session in self.session_list:
            d_s, _, _ = session.cheese_board(experiment_phase=["learning_cheeseboard_2"]).place_field_goal_distance(plotting=False,
                                                                                                  mean_firing_threshold=
                                                                                                  mean_firing_threshold,
                                                                                                  trials_to_use=range(nr_trials))
            dist_stable_post.append(d_s)


        dist_stable_pre = np.hstack(dist_stable_pre)
        dist_stable_post = np.hstack(dist_stable_post)


        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(4, 5))
        res = [dist_stable_pre, dist_stable_post]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Last "+str(nr_trials)+"\n trials PRE", "First "+str(nr_trials)+"\n trials POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["gray", 'black']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Distance to closest goal (cm)")
        plt.grid(color="grey", axis="y")
        print(mannwhitneyu(dist_stable_pre, dist_stable_post, alternative="less"))
        plt.ylim(0, 60)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("place_field_to_goal_distance_pre_post_stable.svg", transparent="True")
        else:
            plt.show()

    """#################################################################################################################
    #  pre, long sleep, post
    #################################################################################################################"""

    def pre_long_sleep_post_over_expressed_modes_goal_coding(self, cells_to_compare="stable"):
        """
        Checks goal coding of over-expressed modes during sleep when using only a subset of cells for decoding

        @param cells_to_compare: which cells to use ("stable", "increasing", "decreasing")
        """
        # go trough all sessions to collect results
        res = []
        session_name_strings = []
        for session in self.session_list:
            gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem = \
                session.pre_long_sleep_post().over_expressed_modes_goal_coding(template_type="phmm", plotting=False,
                                                                               post_or_pre="pre",
                                                                               cells_to_compare=cells_to_compare)
            res.append([gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem])
            session_name_strings.append(session.session_name)

        res = np.array(res)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess, res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[0]*100, label="INC. WITH " +cells_to_compare+ " CELLS", color="r")
            plt.scatter(id_sess, res_sess[1]*100, label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("GOAL CODING")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("REM (PRE): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.ylim(0,100)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess,res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[2]*100, label="INC. WITH "+cells_to_compare+" CELLS", color="r")
            plt.scatter(id_sess, res_sess[3]*100, label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("GOAL CODING")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("NREM (PRE): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.ylim(0,100)
        plt.show()

        res = []
        for session in self.session_list:
            gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem = \
                session.pre_long_sleep_post().over_expressed_modes_goal_coding(template_type="phmm", plotting=False,
                                                                               post_or_pre="post",
                                                                               cells_to_compare=cells_to_compare)
            res.append([gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem])

        res = np.array(res)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess, res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[0] * 100, label="INC. WITH "+cells_to_compare+" CELLS", color="r")
            plt.scatter(id_sess, res_sess[1] * 100, label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("GOAL CODING")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("REM (POST): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.ylim(0, 100)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess, res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[2] * 100, label="INC. WITH "+cells_to_compare+" CELLS", color="r")
            plt.scatter(id_sess, res_sess[3] * 100, label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("GOAL CODING")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("NREM (POST): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.ylim(0, 100)
        plt.show()

    def pre_long_sleep_post_over_expressed_modes_spatial_info(self, cells_to_compare="stable"):
        """
        Checks spatial information of over-expressed modes during sleep when using only a subset of cells for decoding

        @param cells_to_compare: which cells to use ("stable", "increasing", "decreasing")
        """
        # go trough all sessions to collect results
        res = []
        session_name_strings = []
        for session in self.session_list:
            gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem = \
                session.pre_long_sleep_post().over_expressed_modes_spatial_information(template_type="phmm", plotting=False,
                                                                               post_or_pre="pre",
                                                                               cells_to_compare=cells_to_compare)
            res.append([gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem])
            session_name_strings.append(session.session_name)

        res = np.array(res)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess, res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[0], label="INC. WITH " +cells_to_compare+ " CELLS", color="r")
            plt.scatter(id_sess, res_sess[1], label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("MEAN(MEDIAN DISTANCE)")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("REM (PRE): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess,res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[2], label="INC. WITH "+cells_to_compare+" CELLS", color="r")
            plt.scatter(id_sess, res_sess[3], label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("MEAN(MEDIAN DISTANCE)")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("NREM (PRE): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.show()

        res = []
        for session in self.session_list:
            gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem = \
                session.pre_long_sleep_post().over_expressed_modes_spatial_information(template_type="phmm", plotting=False,
                                                                               post_or_pre="post",
                                                                               cells_to_compare=cells_to_compare)
            res.append([gc_increase_stable_rem, gc_stable_rem, gc_increase_stable_nrem, gc_stable_nrem])

        res = np.array(res)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess, res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[0], label="INC. WITH "+cells_to_compare+" CELLS", color="r")
            plt.scatter(id_sess, res_sess[1], label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("MEAN(MEDIAN DISTANCE)")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("REM (POST): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for id_sess, res_sess in enumerate(res):
            plt.scatter(id_sess, res_sess[2], label="INC. WITH "+cells_to_compare+" CELLS", color="r")
            plt.scatter(id_sess, res_sess[3], label="CONST./DEC. WITH "+cells_to_compare+" CELLS", color="b")
            plt.ylabel("MEAN(MEDIAN DISTANCE)")
        plt.xticks(range(res.shape[0]), session_name_strings, rotation='vertical')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("NREM (POST): INCREASE REACTIVATION OF MODES \n WITH ONLY "+cells_to_compare+" CELLS")
        plt.show()

    def pre_long_sleep_post_cell_type_correlations(self):
        """
        Computes correlations of sleep activity with PRE and POST behavioral activity

        """
        stable_pre = []
        stable_post = []
        inc_post = []
        dec_pre = []
        stable_inc_post = []
        stable_dec_pre = []
        for session in self.session_list:
            spre, spost, ipost, dpost, sipost, sdpre = \
                session.pre_long_sleep_post().cell_type_correlations(plotting=False)
            stable_pre.append(spre)
            stable_post.append(spost)
            inc_post.append(ipost)
            dec_pre.append(dpost)
            stable_inc_post.append(sipost)
            stable_dec_pre.append(sdpre)

        # plot stable pre/post
        for st_pre, st_post in zip(stable_pre, stable_post):
            plt.plot(st_pre, color="green", label="PRE")
            plt.plot(st_post, color="orange", label="POST")
            plt.ylabel("PEARSON R")
        plt.title("STABLE")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()

        # plot inc post
        for inc_po in inc_post:
            # plt.ylim([0, y_max])
            plt.plot(inc_po)
        plt.title("INCREASING")
        plt.show()

        # plot dec pre
        for de_pre in dec_pre:
            # plt.ylim([0, y_max])
            plt.plot(de_pre)
        plt.title("DECREASING")
        plt.show()

        # plot stable inc post
        for si_post in stable_inc_post:
            plt.plot(si_post)
        plt.title("STABLE-INC")
        plt.show()

        # plot stable dec pre
        for sd_post in stable_dec_pre:
            plt.plot(sd_post)
        plt.title("STABLE-DEC")
        plt.show()

    def pre_long_sleep_post_firing_rates(self, cells_to_use="stable", save_fig=True, separate_sleep_phase=True):

        if separate_sleep_phase:
            firing_pre_norm = []
            firing_sleep_rem_norm = []
            firing_sleep_nrem_norm = []
            firing_post_norm = []

            for session in self.session_list:
                pre_, rem_, nrem_, post_ = \
                    session.pre_long_sleep_post().firing_rate_distributions(cells_to_use=cells_to_use, plotting=False,
                                                                            separate_sleep_phases=True)

                firing_pre_norm.append(pre_)
                firing_sleep_rem_norm.append(rem_)
                firing_sleep_nrem_norm.append(nrem_)
                firing_post_norm.append(post_)

            firing_pre_norm = np.hstack(firing_pre_norm)
            firing_sleep_rem_norm = np.hstack(firing_sleep_rem_norm)
            firing_sleep_nrem_norm = np.hstack(firing_sleep_nrem_norm)
            firing_post_norm = np.hstack(firing_post_norm)

            p_pre_norm = 1. * np.arange(firing_pre_norm.shape[0]) / (firing_pre_norm.shape[0] - 1)
            p_sleep_nrem_norm = 1. * np.arange(firing_sleep_nrem_norm.shape[0]) / (firing_sleep_nrem_norm.shape[0] - 1)
            p_sleep_rem_norm = 1. * np.arange(firing_sleep_rem_norm.shape[0]) / (firing_sleep_rem_norm.shape[0] - 1)
            p_post_norm = 1. * np.arange(firing_post_norm.shape[0]) / (firing_post_norm.shape[0] - 1)

            if save_fig:
                plt.close()
                plt.style.use('default')

            plt.plot(np.sort(firing_pre_norm), p_pre_norm, label="PRE")
            plt.plot(np.sort(firing_sleep_rem_norm), p_sleep_rem_norm, label="REM")
            plt.plot(np.sort(firing_sleep_nrem_norm), p_sleep_nrem_norm, label="NREM")
            plt.plot(np.sort(firing_post_norm), p_post_norm, label="POST")
            plt.title(cells_to_use)
            plt.xlabel("Mean firing rate / normalized")
            plt.ylabel("CDF")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("firing_rates_pre_sleep_post_" + cells_to_use + ".svg", transparent="True")
            else:
                plt.title(cells_to_use)
                plt.show()

        else:
            pre = []
            sleep = []
            post = []

            for session in self.session_list:
                pre_, sleep_, post_ = \
                    session.pre_long_sleep_post().firing_rate_distributions(cells_to_use=cells_to_use, plotting=False,
                                                                            separate_sleep_phases=False)

                pre.append(pre_)
                sleep.append(sleep_)
                post.append(post_)

            pre = np.hstack(pre)
            sleep = np.hstack(sleep)
            post = np.hstack(post)

            # stats
            p_pre_sleep = mannwhitneyu(pre, sleep, alternative="less")[1]
            p_post_sleep = mannwhitneyu(post, sleep, alternative="less")[1]
            print("PRE-sleep, p-value = "+str(p_pre_sleep))
            print("POST-sleep, p-value = " + str(p_post_sleep))


            p_pre_stable = 1. * np.arange(pre.shape[0]) / (pre.shape[0] - 1)
            p_sleep_stable = 1. * np.arange(sleep.shape[0]) / (sleep.shape[0] - 1)
            p_post_stable = 1. * np.arange(post.shape[0]) / (post.shape[0] - 1)

            if save_fig:
                plt.close()
                plt.style.use('default')
            plt.plot(np.sort(pre), p_pre_stable, label="PRE")
            plt.plot(np.sort(sleep), p_sleep_stable, label="Sleep")
            plt.plot(np.sort(post), p_post_stable, label="POST")
            plt.xlabel("Mean firing rate / normalized")
            plt.ylabel("CDF")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("firing_rates_pre_sleep_post_"+cells_to_use+".svg", transparent="True")
            else:
                plt.title(cells_to_use)
                plt.show()

    def pre_long_sleep_post_firing_rates_stable_vs_decreasing(self, save_fig=True):

        # data for stable cells
        firing_pre_norm_stable = []
        firing_sleep_rem_norm_stable = []
        firing_sleep_nrem_norm_stable = []
        firing_post_norm_stable = []
        # data for decreasing cells
        firing_pre_norm_dec = []
        firing_sleep_rem_norm_dec = []
        firing_sleep_nrem_norm_dec = []
        firing_post_norm_dec = []

        for session in self.session_list:
            pre_, rem_, nrem_, post_ = \
                session.pre_long_sleep_post().firing_rate_distributions(cells_to_use="stable", plotting=False,
                                                                        separate_sleep_phases=True)

            firing_pre_norm_stable.append(pre_)
            firing_sleep_rem_norm_stable.append(rem_)
            firing_sleep_nrem_norm_stable.append(nrem_)
            firing_post_norm_stable.append(post_)

            pre_, rem_, nrem_, post_ = \
                session.pre_long_sleep_post().firing_rate_distributions(cells_to_use="decreasing", plotting=False,
                                                                        separate_sleep_phases=True)

            firing_pre_norm_dec.append(pre_)
            firing_sleep_rem_norm_dec.append(rem_)
            firing_sleep_nrem_norm_dec.append(nrem_)
            firing_post_norm_dec.append(post_)

        firing_pre_norm_stable = np.hstack(firing_pre_norm_stable)
        firing_sleep_rem_norm_stable = np.hstack(firing_sleep_rem_norm_stable)
        firing_sleep_nrem_norm_stable = np.hstack(firing_sleep_nrem_norm_stable)
        firing_post_norm_stable = np.hstack(firing_post_norm_stable)

        p_pre_norm_stable = 1. * np.arange(firing_pre_norm_stable.shape[0]) / (firing_pre_norm_stable.shape[0] - 1)
        p_sleep_nrem_norm_stable = 1. * np.arange(firing_sleep_nrem_norm_stable.shape[0]) / (firing_sleep_nrem_norm_stable.shape[0] - 1)
        p_sleep_rem_norm_stable = 1. * np.arange(firing_sleep_rem_norm_stable.shape[0]) / (firing_sleep_rem_norm_stable.shape[0] - 1)
        p_post_norm_stable = 1. * np.arange(firing_post_norm_stable.shape[0]) / (firing_post_norm_stable.shape[0] - 1)

        firing_pre_norm_dec = np.hstack(firing_pre_norm_dec)
        firing_sleep_rem_norm_dec = np.hstack(firing_sleep_rem_norm_dec)
        firing_sleep_nrem_norm_dec = np.hstack(firing_sleep_nrem_norm_dec)
        firing_post_norm_dec = np.hstack(firing_post_norm_dec)

        p_pre_norm_dec = 1. * np.arange(firing_pre_norm_dec.shape[0]) / (firing_pre_norm_dec.shape[0] - 1)
        p_sleep_nrem_norm_dec = 1. * np.arange(firing_sleep_nrem_norm_dec.shape[0]) / (firing_sleep_nrem_norm_dec.shape[0] - 1)
        p_sleep_rem_norm_dec = 1. * np.arange(firing_sleep_rem_norm_dec.shape[0]) / (firing_sleep_rem_norm_dec.shape[0] - 1)
        p_post_norm_dec = 1. * np.arange(firing_post_norm_dec.shape[0]) / (firing_post_norm_dec.shape[0] - 1)

        if save_fig:
            plt.close()
            plt.style.use('default')

        plt.plot(np.sort(firing_sleep_rem_norm_stable), p_sleep_rem_norm_stable, label="stable: REM", color="violet")
        plt.plot(np.sort(firing_sleep_nrem_norm_stable), p_sleep_nrem_norm_stable, label="stable: NREM", color="violet",
                 linestyle="--")
        plt.plot(np.sort(firing_sleep_rem_norm_dec), p_sleep_rem_norm_dec, label="dec: REM", color="turquoise")
        plt.plot(np.sort(firing_sleep_nrem_norm_dec), p_sleep_nrem_norm_dec, label="dec: NREM", color="turquoise",
                 linestyle="--")
        plt.xlabel("Mean firing rate / normalized")
        plt.ylabel("CDF")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("firing_rates_pre_sleep_post_stable_vs_decreasing.svg", transparent="True")
        else:
            plt.show()

    def pre_long_sleep_plot_cell_classification_mean_firing_rates_awake(self, save_fig = False):

        raster_ds_smoothed_stable = []
        pre_raster_mean_stable = []
        post_raster_mean_stable = []
        raster_ds_smoothed_decreasing = []
        pre_raster_mean_decreasing = []
        post_raster_mean_decreasing = []
        raster_ds_smoothed_increasing = []
        pre_raster_mean_increasing = []
        post_raster_mean_increasing = []

        for session in self.session_list:
            s_s, pre_s, post_s, s_d, pre_d, post_d, s_i, pre_i, post_i = \
                session.pre_long_sleep_post().plot_cell_classification_mean_firing_rates_awake(plotting=False)

            raster_ds_smoothed_stable.append(s_s)
            pre_raster_mean_stable.append(pre_s)
            post_raster_mean_stable.append(post_s)
            raster_ds_smoothed_decreasing.append(s_d)
            pre_raster_mean_decreasing.append(pre_d)
            post_raster_mean_decreasing.append(post_d)
            raster_ds_smoothed_increasing.append(s_i)
            pre_raster_mean_increasing.append(pre_i)
            post_raster_mean_increasing.append(post_i)

        raster_ds_smoothed_stable = np.vstack(raster_ds_smoothed_stable)
        pre_raster_mean_stable = np.hstack(pre_raster_mean_stable)
        post_raster_mean_stable = np.hstack(post_raster_mean_stable)
        raster_ds_smoothed_decreasing = np.vstack(raster_ds_smoothed_decreasing)
        pre_raster_mean_decreasing = np.hstack(pre_raster_mean_decreasing)
        post_raster_mean_decreasing = np.hstack(post_raster_mean_decreasing)
        raster_ds_smoothed_increasing = np.vstack(raster_ds_smoothed_increasing)
        pre_raster_mean_increasing = np.hstack(pre_raster_mean_increasing)
        post_raster_mean_increasing = np.hstack(post_raster_mean_increasing)

        raster_ds_smoothed_stable_sorted = raster_ds_smoothed_stable[pre_raster_mean_stable.argsort(), :]
        pre_raster_mean_stable_sorted = pre_raster_mean_stable[pre_raster_mean_stable.argsort()]
        post_raster_mean_stable_sorted = post_raster_mean_stable[pre_raster_mean_stable.argsort()]

        raster_ds_smoothed_increasing_sorted = raster_ds_smoothed_increasing[pre_raster_mean_increasing.argsort(), :]
        pre_raster_mean_increasing_sorted = pre_raster_mean_increasing[pre_raster_mean_increasing.argsort()]
        post_raster_mean_increasing_sorted = post_raster_mean_increasing[pre_raster_mean_increasing.argsort()]

        raster_ds_smoothed_decreasing_sorted = raster_ds_smoothed_decreasing[pre_raster_mean_decreasing.argsort(), :]
        pre_raster_mean_decreasing_sorted = pre_raster_mean_decreasing[pre_raster_mean_decreasing.argsort()]
        post_raster_mean_decreasing_sorted = post_raster_mean_decreasing[pre_raster_mean_decreasing.argsort()]

        # stack them back together for plotting
        pre_raster_mean_sorted = np.hstack(
            (pre_raster_mean_stable_sorted, pre_raster_mean_increasing_sorted, pre_raster_mean_decreasing_sorted))
        post_raster_mean_sorted = np.hstack(
            (post_raster_mean_stable_sorted, post_raster_mean_increasing_sorted, post_raster_mean_decreasing_sorted))

        raster_ds_smoothed_sorted = np.vstack((raster_ds_smoothed_stable_sorted, raster_ds_smoothed_increasing_sorted,
                                               raster_ds_smoothed_decreasing_sorted))

        nr_stable_cells = raster_ds_smoothed_stable_sorted.shape[0]
        nr_dec_cells = raster_ds_smoothed_decreasing_sorted.shape[0]
        nr_inc_cells = raster_ds_smoothed_increasing_sorted.shape[0]

        if save_fig:
            plt.style.use('default')

        fig = plt.figure(figsize=(5, 12))
        gs = fig.add_gridspec(15, 20)
        ax1 = fig.add_subplot(gs[:-3, :2])
        ax2 = fig.add_subplot(gs[:-3, 2:-2])
        ax3 = fig.add_subplot(gs[:-3, -2:])
        ax4 = fig.add_subplot(gs[-1, :10])
        ax1.imshow(np.expand_dims(pre_raster_mean_sorted, 1), vmin=0, vmax=1, interpolation='nearest',
                   aspect='auto')
        ax1.hlines(nr_stable_cells, -0.5, 0.5, color="red")
        ax1.hlines(nr_stable_cells + nr_inc_cells, -0.5, 0.5, color="red")
        ax1.set_xticks([])
        ax1.set_ylabel("Cells")
        ax1.set_ylim(nr_inc_cells + nr_stable_cells + nr_dec_cells, 0)
        cax = ax2.imshow(raster_ds_smoothed_sorted, vmin=0, vmax=1, interpolation='nearest',
                         aspect='auto')
        ax2.hlines(nr_stable_cells, -0.5, raster_ds_smoothed_sorted.shape[1] - 0.5, color="red")
        ax2.hlines(nr_stable_cells + nr_inc_cells, -0.5,
                   raster_ds_smoothed_sorted.shape[1] - 0.5,
                   color="red")
        ax2.set_yticks([])
        # ax2.set_xticks([0, 0.33 * raster_ds_smoothed_sorted.shape[1], 0.66 * raster_ds_smoothed_sorted.shape[1],
        #                 raster_ds_smoothed_sorted.shape[1] - 0.5])
        # ax2.set_xticklabels(
        #     ["0", str(int(np.round(duration_sleep_h * 0.33, 0))), str(int(np.round(duration_sleep_h * 0.66, 0))),
        #      str(duration_sleep_h)])
        ax2.set_xlabel("Normalized sleep duration")
        ax3.imshow(np.expand_dims(post_raster_mean_sorted, 1), vmin=0, vmax=1, interpolation='nearest',
                   aspect='auto')
        ax3.hlines(nr_stable_cells, -0.5, 0.5, color="red")
        ax3.hlines(nr_stable_cells + nr_inc_cells, -0.5, 0.5, color="red")
        ax3.set_yticks([])
        ax3.set_xticks([])
        a = fig.colorbar(mappable=cax, cax=ax4, orientation="horizontal", ticks=[0, 1])
        a.ax.set_xticklabels(["0", "1"])
        a.ax.set_xlabel("Normalized firing rate")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("cell_classification_all_cells.svg", transparent="True")
        else:
            plt.show()

    def pre_long_sleep_post_firing_rates_all_cells(self, measure="mean", save_fig=False, plotting=True, chunks_in_min=5):
        firing_pre_stable = []
        firing_pre_dec = []
        firing_pre_inc = []
        firing_sleep_stable = []
        firing_sleep_dec = []
        firing_sleep_inc = []
        firing_post_stable = []
        firing_post_dec = []
        firing_post_inc = []

        for session in self.session_list:
            pre_s, pre_d, pre_i, s_s, s_d, s_i, post_s, post_d, post_i = \
                session.pre_long_sleep_post().firing_rate_distributions_all_cells(plotting=False, measure=measure,
                                                                                  chunks_in_min=chunks_in_min)

            firing_pre_stable.append(pre_s)
            firing_pre_dec.append(pre_d)
            firing_pre_inc.append(pre_i)
            firing_sleep_stable.append(s_s)
            firing_sleep_dec.append(s_d)
            firing_sleep_inc.append(s_i)
            firing_post_stable.append(post_s)
            firing_post_dec.append(post_d)
            firing_post_inc.append(post_i)

        firing_pre_stable = np.hstack(firing_pre_stable)
        firing_pre_dec = np.hstack(firing_pre_dec)
        firing_pre_inc = np.hstack(firing_pre_inc)
        firing_sleep_stable = np.hstack(firing_sleep_stable)
        firing_sleep_dec = np.hstack(firing_sleep_dec)
        firing_sleep_inc = np.hstack(firing_sleep_inc)
        firing_post_stable = np.hstack(firing_post_stable)
        firing_post_dec = np.hstack(firing_post_dec)
        firing_post_inc = np.hstack(firing_post_inc)

        p_pre_stable = 1. * np.arange(firing_pre_stable.shape[0]) / (firing_pre_stable.shape[0] - 1)
        p_sleep_stable = 1. * np.arange(firing_sleep_stable.shape[0]) / (firing_sleep_stable.shape[0] - 1)
        p_post_stable = 1. * np.arange(firing_post_stable.shape[0]) / (firing_post_stable.shape[0] - 1)

        p_pre_dec = 1. * np.arange(firing_pre_dec.shape[0]) / (firing_pre_dec.shape[0] - 1)
        p_sleep_dec = 1. * np.arange(firing_sleep_dec.shape[0]) / (firing_sleep_dec.shape[0] - 1)
        p_post_dec = 1. * np.arange(firing_post_dec.shape[0]) / (firing_post_dec.shape[0] - 1)

        p_pre_inc = 1. * np.arange(firing_pre_inc.shape[0]) / (firing_pre_inc.shape[0] - 1)
        p_sleep_inc = 1. * np.arange(firing_sleep_inc.shape[0]) / (firing_sleep_inc.shape[0] - 1)
        p_post_inc = 1. * np.arange(firing_post_inc.shape[0]) / (firing_post_inc.shape[0] - 1)

        print("PRE:")
        print(" - dec vs. stable")
        print(mannwhitneyu(firing_pre_dec, firing_pre_stable))
        print("POST:")
        print(" - inc vs. stable")
        print(mannwhitneyu(firing_post_inc, firing_post_stable))

        fig = plt.figure(figsize=(10,6))
        fig.add_subplot(1, 3, 1)
        plt.title("PRE (Stable)")
        plt.hist(firing_pre_stable, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 5)
        plt.ylabel("z-scored max firing")
        fig.add_subplot(1, 3, 2)
        plt.title("Sleep")
        plt.hist(firing_sleep_stable, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 5)
        fig.add_subplot(1, 3, 3)
        plt.title("POST")
        plt.hist(firing_post_stable, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 5)
        plt.show()

        fig = plt.figure(figsize=(10,6))
        fig.add_subplot(1, 3, 1)
        plt.title("PRE (Dec)")
        plt.hist(firing_pre_dec, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 10)
        plt.ylabel("z-scored max firing")
        fig.add_subplot(1, 3, 2)
        plt.title("Sleep")
        plt.hist(firing_sleep_dec, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 10)
        fig.add_subplot(1, 3, 3)
        plt.title("POST")
        plt.hist(firing_post_dec, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 10)
        plt.show()

        fig = plt.figure(figsize=(10,6))
        fig.add_subplot(1, 3, 1)
        plt.title("PRE (inc)")
        plt.hist(firing_pre_inc, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 15)
        plt.ylabel("z-scored max firing")
        fig.add_subplot(1, 3, 2)
        plt.title("Sleep")
        plt.hist(firing_sleep_inc, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 15)
        fig.add_subplot(1, 3, 3)
        plt.title("POST")
        plt.hist(firing_post_inc, orientation="horizontal", density=True, bins=150)
        plt.ylim(-2, 15)
        plt.show()

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        print("Stable: PRE-sleep")
        print(mannwhitneyu(firing_pre_stable, firing_sleep_stable, alternative="less"))

        print("Stable: sleep-POST")
        print(mannwhitneyu(firing_sleep_stable, firing_post_stable, alternative="greater"))

        print("Stable: PRE-POST")
        print(mannwhitneyu(firing_pre_stable, firing_post_stable))

        y_dat = [firing_pre_stable, firing_sleep_stable, firing_post_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2,3], patch_artist=True,
                            labels=["PRE", "Sleep", "POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        # whis = [0.01, 99.99]
        plt.title("Stable")
        # plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-2, 2.5)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-1.3, 1.3)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("firing_rate_" + measure + "_stable.svg", transparent="True")
            plt.close()
        else:
            plt.show()


        print("Inc: PRE-sleep")
        print(mannwhitneyu(firing_pre_inc, firing_sleep_inc))

        print("Inc: PRE-POST")
        print(mannwhitneyu(firing_pre_inc, firing_post_inc))

        print("Inc: sleep-POST")
        print(mannwhitneyu(firing_sleep_inc, firing_post_inc, alternative="less"))
        y_dat = [firing_pre_inc, firing_sleep_inc, firing_post_inc]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2,3], patch_artist=True,
                            labels=["PRE", "Sleep", "POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        plt.title("Inc")
        # plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-2.8, 5.2)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-2.2, 6.2)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("firing_rate_" + measure + "_increasing.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        print("Dec: PRE-sleep")
        print(mannwhitneyu(firing_pre_dec, firing_sleep_dec, alternative="greater"))

        print("Dec: PRE-POST")
        print(mannwhitneyu(firing_pre_dec, firing_post_dec))

        print("Dec: sleep-POST")
        print(mannwhitneyu(firing_sleep_dec, firing_post_dec, alternative="greater"))
        y_dat = [firing_pre_dec, firing_sleep_dec, firing_post_dec]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2,3], patch_artist=True,
                            labels=["PRE", "Sleep", "POST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        plt.title("Dec")
        # plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-2, 4.8)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-1.5, 4.8)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("firing_rate_" + measure + "_decreasing.svg", transparent="True")
            plt.close()
        else:
            plt.show()


        if save_fig or plotting:

            if save_fig:
                plt.style.use('default')

            plt.plot(np.sort(firing_pre_stable), p_pre_stable, color="magenta", label="stable")
            plt.plot(np.sort(firing_pre_inc), p_pre_inc, color="orange", label="inc")
            plt.plot(np.sort(firing_pre_dec), p_pre_dec, color="turquoise", label="dec")
            plt.xlabel(measure + " firing rate (Hz)")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("PRE")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("firing_rate_"+measure+"_pre.svg", transparent="True")
                plt.close()
            else:
                plt.show()

            plt.plot(np.sort(firing_sleep_stable), p_sleep_stable, color="magenta", label="stable")
            plt.plot(np.sort(firing_sleep_inc), p_sleep_inc, color="orange", label="inc")
            plt.plot(np.sort(firing_sleep_dec), p_sleep_dec, color="turquoise", label="dec")
            plt.xlabel(measure + " firing rate (Hz)")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("Sleep")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("firing_rate_"+measure+"_sleep.svg", transparent="True")
                plt.close()
            else:
                plt.show()

            plt.plot(np.sort(firing_post_stable), p_post_stable, color="magenta", label="stable")
            plt.plot(np.sort(firing_post_inc), p_post_inc, color="orange", label="inc")
            plt.plot(np.sort(firing_post_dec), p_post_dec, color="turquoise", label="dec")
            plt.xlabel(measure + " firing rate (Hz)")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("Post")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("firing_rate_"+measure+"_post.svg", transparent="True")
                plt.close()
            else:
                plt.show()
        else:
            return firing_pre_stable, firing_pre_dec, firing_pre_inc, firing_sleep_stable, firing_sleep_dec, \
                   firing_sleep_inc, firing_post_stable, firing_post_dec, firing_post_inc

    def pre_long_sleep_post_firing_rate_ratios_all_cells(self, measure="mean", save_fig=False, plotting=True, chunks_in_min=5):
        ratio_pre_stable = []
        ratio_pre_dec = []
        ratio_pre_inc = []
        ratio_sleep_stable = []
        ratio_sleep_dec = []
        ratio_sleep_inc = []
        ratio_post_stable = []
        ratio_post_dec = []
        ratio_post_inc = []

        for session in self.session_list:
            pre_s, post_s, pre_d, post_d, pre_i, post_i = \
                session.pre_long_sleep_post().firing_rate_ratios_all_cells(plotting=False, measure=measure,
                                                                                  chunks_in_min=chunks_in_min)

            ratio_pre_stable.append(pre_s)
            ratio_post_stable.append(post_s)
            ratio_pre_dec.append(pre_d)
            ratio_post_dec.append(post_d)
            ratio_pre_inc.append(pre_i)
            ratio_post_inc.append(post_i)

        ratio_pre_stable = np.hstack(ratio_pre_stable)
        ratio_pre_dec = np.hstack(ratio_pre_dec)
        ratio_pre_inc = np.hstack(ratio_pre_inc)
        ratio_post_stable = np.hstack(ratio_post_stable)
        ratio_post_dec = np.hstack(ratio_post_dec)
        ratio_post_inc = np.hstack(ratio_post_inc)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        print("Stable")
        print(mannwhitneyu(ratio_pre_stable, ratio_post_stable))

        y_dat = [ratio_pre_stable, ratio_post_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Sleep/\nPRE", "Sleep/\nPOST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        # whis = [0.01, 99.99]
        plt.title("Stable")
        # plt.yscale("symlog")
        # if measure == "max":
        #     plt.ylim(-2, 2.5)
        #     plt.ylabel("Max firing rate (z-scored)")
        if measure == "mean":
            plt.ylim(-0.3, 12.3)
            plt.ylabel("Mean firing rate ratio")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("firing_rate_ratio_" + measure + "_stable.svg", transparent="True")
            plt.close()
        else:
            plt.show()


        print("Inc")
        print(mannwhitneyu(ratio_pre_inc, ratio_post_inc))

        y_dat = [ratio_pre_inc, ratio_post_inc]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Sleep/\nPRE", "Sleep/\nPOST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        plt.title("Inc")
        # plt.yscale("symlog")
        # if measure == "max":
        #     plt.ylim(-2.8, 5.2)
        #     plt.ylabel("Max firing rate (z-scored)")
        if measure == "mean":
            plt.ylim(-0.2, 7.2)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("firing_rate_ratio_" + measure + "_increasing.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        print("Dec")
        print(mannwhitneyu(ratio_pre_dec, ratio_post_dec))
        y_dat = [ratio_pre_dec, ratio_post_dec]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Sleep/\nPRE", "Sleep/\nPOST"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False)
        plt.title("Dec")
        # plt.yscale("symlog")
        # if measure == "max":
        #     plt.ylim(-2, 4.8)
        #     plt.ylabel("Max firing rate (z-scored)")
        if measure == "mean":
            plt.ylim(-0.2, 15.2)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("firing_rate_ratio_" + measure + "_decreasing.svg", transparent="True")
            plt.close()
        else:
            plt.show()


        if save_fig or plotting:

            if save_fig:
                plt.style.use('default')

            plt.plot(np.sort(firing_pre_stable), p_pre_stable, color="magenta", label="stable")
            plt.plot(np.sort(firing_pre_inc), p_pre_inc, color="orange", label="inc")
            plt.plot(np.sort(firing_pre_dec), p_pre_dec, color="turquoise", label="dec")
            plt.xlabel(measure + " firing rate (Hz)")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("PRE")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("firing_rate_"+measure+"_pre.svg", transparent="True")
                plt.close()
            else:
                plt.show()

            plt.plot(np.sort(firing_sleep_stable), p_sleep_stable, color="magenta", label="stable")
            plt.plot(np.sort(firing_sleep_inc), p_sleep_inc, color="orange", label="inc")
            plt.plot(np.sort(firing_sleep_dec), p_sleep_dec, color="turquoise", label="dec")
            plt.xlabel(measure + " firing rate (Hz)")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("Sleep")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("firing_rate_"+measure+"_sleep.svg", transparent="True")
                plt.close()
            else:
                plt.show()

            plt.plot(np.sort(firing_post_stable), p_post_stable, color="magenta", label="stable")
            plt.plot(np.sort(firing_post_inc), p_post_inc, color="orange", label="inc")
            plt.plot(np.sort(firing_post_dec), p_post_dec, color="turquoise", label="dec")
            plt.xlabel(measure + " firing rate (Hz)")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("Post")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("firing_rate_"+measure+"_post.svg", transparent="True")
                plt.close()
            else:
                plt.show()
        else:
            return firing_pre_stable, firing_pre_dec, firing_pre_inc, firing_sleep_stable, firing_sleep_dec, \
                   firing_sleep_inc, firing_post_stable, firing_post_dec, firing_post_inc

    def pre_long_sleep_drift_correlation_structure_equalized_firing_rates(self, save_fig=False, n_smoothing=40,
                                                                          plot_mean=False, cells_to_use="stable"):
        sim_ratio = []
        for session in self.session_list:
            s_r = session.pre_long_sleep_post().drift_correlation_structure_equalized_firing_rates(bins_per_corr_matrix
                                                                                                   =600, plotting=False,
                                                                                                   cells_to_use=
                                                                                                   cells_to_use)
            sim_ratio.append(s_r)

        if save_fig:
            plt.style.use('default')

        max_y = 0
        plt.figure(figsize=(4,6))
        # plt.figure(figsize=(3,4))
        fig = plt.figure()
        ax = fig.add_subplot()
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        first_val = []
        last_val = []
        sim_ratios_smooth = []
        sim_ratios_xaxis = []
        for i, session_sim_ratio in enumerate(sim_ratio):
            session_sim_ratio = np.array(session_sim_ratio)
            # smoothing
            session_sim_ratio = moving_average(a=session_sim_ratio, n=n_smoothing)
            sim_ratios_smooth.append(session_sim_ratio)
            sim_ratios_xaxis.append(np.linspace(0, 1, session_sim_ratio.shape[0]))
            first_val.append(session_sim_ratio[0])
            last_val.append(session_sim_ratio[-1])
            max_y = np.max(np.append(np.abs(session_sim_ratio), max_y))
            ax.plot(np.linspace(0, 1, session_sim_ratio.shape[0]),session_sim_ratio, linewidth=2, c=col[i], label=str(i))

        if plot_mean:
            nr_x_ticks = [x.shape[0] for x in sim_ratios_xaxis]
            x_axis_new = sim_ratios_xaxis[np.argmax(nr_x_ticks)]
            sim_ratios_smooth_equal_length = []
            for session_sim_ratio_smooth, x_axis in zip(sim_ratios_smooth, sim_ratios_xaxis):
                sim_ratios_smooth_equal_length.append(np.interp(x=x_axis_new, xp=x_axis, fp=session_sim_ratio_smooth))
            sim_ratios_smooth_equal_length = np.vstack(sim_ratios_smooth_equal_length)
            mean_sim_ratios_smooth = np.mean(sim_ratios_smooth_equal_length, axis=0)

            ax.plot(np.linspace(0, 1, mean_sim_ratios_smooth.shape[0]), mean_sim_ratios_smooth, linewidth=2, c="gray",
                    label="mean")
        plt.legend()
        # plt.ylim(-max_y-0.1, max_y+0.1)
        plt.ylim(-1,1)
        plt.xlim(0,1)
        plt.ylabel("sim_ratio correlations")
        plt.xlabel("Normalized sleep duration")
        plt.grid(axis='y', color="gray")
        plt.yticks([-1,-0.5, 0, 0.5,1], ["-1", "-0.5", "0", "0.5", "-1"])
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("drift_correlations_stable_equalized_firing_"+cells_to_use+".svg", transparent="True")
        else:
            plt.show()

    def pre_long_sleep_nrem_rem_autocorrelation_spikes_likelihood_vectors(self, template_type="phmm", save_fig=False,
                                                                          nr_pop_vecs=5):
        """
        compares the likelihoods from decoding for REM and NREM

        @param template_type: which template to use "phmm" or "ising"
        """

        # sleep data first
        # --------------------------------------------------------------------------------------------------------------

        rem = []
        nrem = []
        rem_exp = []
        nrem_exp = []

        for i, session in enumerate(self.session_list):
            re, nre, re_exp, nre_exp = \
                session.long_sleep().memory_drift_rem_nrem_autocorrelation_spikes_likelihood_vectors(plotting=False,
                                                                                                     template_type=template_type,
                                                                                                     nr_pop_vecs=nr_pop_vecs)
            rem_exp.append(re_exp)
            nrem_exp.append(nre_exp)
            rem.append(re)
            nrem.append(nre)

        rem = np.array(rem)
        nrem = np.array(nrem)
        rem_exp = np.array(rem_exp)
        nrem_exp = np.array(nrem_exp)

        rem_mean = np.mean(rem, axis=0)
        nrem_mean = np.mean(nrem, axis=0)
        rem_std = np.std(rem, axis=0)
        nrem_std = np.std(nrem, axis=0)

        # awake data
        # --------------------------------------------------------------------------------------------------------------

        awake = []
        awake_exp = []

        for i, session in enumerate(self.session_list):
            _, dat, a_exp = \
                session.cheese_board(experiment_phase=
                                     ["learning_cheeseboard_1"]).decode_awake_activity_autocorrelation_spikes_likelihood_vectors(plotting=False, nr_pop_vecs=nr_pop_vecs)

            awake.append(dat)
            awake_exp.append(a_exp)

        awake = np.array(awake)
        awake_exp = np.array(awake_exp)
        awake_mean = np.mean(awake, axis=0)
        awake_std = np.std(awake, axis=0)

        # plotting
        # --------------------------------------------------------------------------------------------------------------

        shift_array = np.arange(-1*int(nr_pop_vecs), int(nr_pop_vecs)+1)

        if exclude_zero_shift:
            rem_mean[int(rem_mean.shape[0] / 2)] = np.nan
            nrem_mean[int(nrem_mean.shape[0] / 2)] = np.nan
            awake_mean[int(awake_mean.shape[0] / 2)] = np.nan

        if save_fig:
            plt.style.use('default')
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.fill_between(shift_array*self.params.spikes_per_bin, rem_mean+rem_std ,rem_mean-rem_std, facecolor="salmon")
        ax.plot(shift_array*self.params.spikes_per_bin, rem_mean, c="red", label="REM")
        ax.fill_between(shift_array*self.params.spikes_per_bin, nrem_mean+nrem_std ,nrem_mean-nrem_std, facecolor="skyblue", alpha=0.6)
        ax.plot(shift_array*self.params.spikes_per_bin, nrem_mean, c="blue", label="NREM")

        ax.fill_between(shift_array*self.params.spikes_per_bin, awake_mean+awake_std ,awake_mean-awake_std, facecolor="lemonchiffon", alpha=0.7)
        ax.plot(shift_array*self.params.spikes_per_bin, awake_mean, c="yellow", label="Awake")
        plt.xlabel("Shift (#spikes)")
        plt.ylabel("Avg. Pearson correlation of likelihood vectors")
        plt.legend()

        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            if exclude_zero_shift:
                plt.savefig("auto_corr_likelihood_vectors_wo_zero.svg", transparent="True")
            else:
                plt.savefig("auto_corr_likelihood_vectors.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        y_dat = np.vstack((nrem_exp, rem_exp, awake_exp))

        plt.figure(figsize=(2,3))
        bplot = plt.boxplot(y_dat.T, positions=[1, 2, 3], patch_artist=True,
                            labels=["NREM", "REM", "AWAKE"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["blue", 'red', "yellow"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Exponential coefficient k")
        plt.grid(color="grey", axis="y")
        degrees = 45
        plt.xticks(rotation=degrees)
        # plt.yscale("symlog")
        plt.yticks([np.median(rem_exp), 0.1, -1, np.median(nrem_exp), np.median(awake_exp)])
        # plt.text(-0.001, np.median(rem_exp), np.str(np.round(np.median(rem_exp), 2)))
        # plt.text(-0.001, np.median(nrem_exp), np.str(np.round(np.median(nrem_exp), 2)))
        # plt.text(-0.001, np.median(awake_exp), np.str(np.round(np.median(awake_exp), 2)))
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("exponential_coeff_likelihood_vec.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        print("REM vs. NREM")
        print(mannwhitneyu(rem_exp, nrem_exp))
        print("REM vs. AWAKE")
        print(mannwhitneyu(rem_exp, awake_exp))
        print("NREM vs. AWAKE")
        print(mannwhitneyu(nrem_exp, awake_exp))

        # fitting of exponential: OPTIONAL
        # --------------------------------------------------------------------------------------------------------------
        exit()
        nrem_orig = nrem_mean[int(nrem_mean.shape[0] / 2):]
        rem_origa = rem_mean[int(rem_mean.shape[0] / 2):]
        awake_orig = awake_mean[int(awake_mean.shape[0] / 2):]

        def exponential(x, a, k, b):
            return a * np.exp(x * k) + b

        nrem_test_data = nrem_orig[1:]
        popt_exponential_nrem, pcov_exponential_nrem = optimize.curve_fit(exponential, np.arange(nrem_test_data.shape[0]),
                                                                        nrem_test_data, p0=[1, -0.5, 1])

        plt.plot(np.linspace(0, nrem_test_data.shape[0],150),
                 exponential(np.linspace(0, nrem_test_data.shape[0],150), popt_exponential_nrem[0],
                             popt_exponential_nrem[1], popt_exponential_nrem[2]), color="blue", label="fit_nrem" )
        plt.scatter(np.arange(nrem_test_data.shape[0]), nrem_test_data, linestyle="--", label="data_nrem", color="lightblue")

        rem_test_data = rem_orig[1:]
        popt_exponential_rem, pcov_exponential_rem = optimize.curve_fit(exponential, np.arange(rem_test_data.shape[0]),
                                                                        rem_test_data, p0=[1, -0.5, 1])

        plt.plot(np.linspace(0, rem_test_data.shape[0],150),
                 exponential(np.linspace(0, rem_test_data.shape[0],150), popt_exponential_rem[0],
                             popt_exponential_rem[1], popt_exponential_rem[2]), color="red", label="fit_rem" )
        plt.scatter(np.arange(rem_test_data.shape[0]), rem_test_data, linestyle="--", label="data_rem", color="salmon")
        plt.legend()

        awake_test_data = awake_orig[1:]
        popt_exponential_awake, pcov_exponential_awake = optimize.curve_fit(exponential, np.arange(awake_test_data.shape[0]),
                                                                        awake_test_data, p0=[1, -0.5, 1])

        plt.plot(np.linspace(0, awake_test_data.shape[0],150),
                 exponential(np.linspace(0, awake_test_data.shape[0],150), popt_exponential_awake[0],
                             popt_exponential_awake[1], popt_exponential_awake[2]), color="yellow", label="fit_awake" )
        plt.scatter(np.arange(awake_test_data.shape[0]), awake_test_data, linestyle="--", label="data_rem", color="lemonchiffon")
        plt.legend()
        plt.show()

    """#################################################################################################################
    #  long sleep
    #################################################################################################################"""

    def long_sleep_memory_drift_temporal(self, save_fig=False, smoothing=20000, template_type="phmm"):
        """
        plots memory drift for all sessions

        :param smoothing: how much smoothing to use for ratio (default: 20000)
        :param smoothing: int
        :param save_fig: whether to save figure (True)
        :type save_fig: bool
        """
        data_t0 = []
        data_t_end = []


        if save_fig:
            plt.style.use('default')

        fig = plt.figure()
        ax = fig.add_subplot()
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for i, session in enumerate(self.session_list):
            print("GETTING DATA FROM " +session.session_name)
            res = session.long_sleep().memory_drift_plot_temporal_trend(template_type=template_type,
                                                                        n_moving_average_pop_vec=smoothing,
                                                                        plotting=False)
            data_t0.append(res[0])
            data_t_end.append(res[-1])
            print("... DONE")
            ax.plot(np.linspace(0, 1, res.shape[0]),res, linewidth=1, c=col[i])

        # stats
        p_value_t0 = ttest_1samp(data_t0, 0, alternative="less")[1]
        print("T-test for t=0, data < 0 --> p = "+str(p_value_t0))

        p_value_t_end = ttest_1samp(data_t_end, 0, alternative="greater")[1]
        print("T-test for t_end, data > 0 --> p = "+str(p_value_t_end))

        plt.grid(axis='y')
        plt.xlabel("Normalized duration")
        plt.xlim(0, 1)
        # plt.ylim(-0.75, 0.25)
        plt.ylim(-1, 1)
        plt.yticks([-1,-0.5, 0, 0.5, 1], ["-1", "-0.5", "0", "0.5", "1"])
        plt.ylabel("sim_ratio")
        plt.title("All sessions")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("drift_all_sessions.svg", transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_data_vs_shuffle(self):

        z_scored_init = []
        z_scored_end = []

        for i, session in enumerate(self.session_list):
            zsi, zse = session.long_sleep().memory_drift_entire_sleep_spike_shuffle_vs_data_compute_p_values()
            z_scored_init.append(zsi)
            z_scored_end.append(zse)

    def long_sleep_memory_time_course(self, save_fig=False, smoothing=20000, template_type="phmm"):
        """
        plots memory drift for all sessions

        :param smoothing: how much smoothing to use for ratio (default: 20000)
        :param smoothing: int
        :param save_fig: whether to save figure (True)
        :type save_fig: bool
        """
        if save_fig:
            plt.style.use('default')

        r_delta_first_half = []
        r_delta_second_half = []

        for i, session in enumerate(self.session_list):

            r_d_f_h, r_d_s_h = session.long_sleep().memory_drift_time_course(template_type=template_type,
                                                                        n_moving_average_pop_vec=smoothing,
                                                                        )
            r_delta_first_half.append(r_d_f_h)
            r_delta_second_half.append(r_d_s_h)


        delta_ratio = np.array(r_delta_first_half)/np.array(r_delta_second_half)
        print(ttest_1samp(delta_ratio, popmean=1))
        print(ttest_1samp(delta_ratio, popmean=1, alternative="greater"))

        plt.figure(figsize=(4,6))
        # plt.figure(figsize=(3,4))
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for session_id, (first, second) in enumerate(zip(r_delta_first_half, r_delta_second_half)):
            plt.scatter([0.1, 0.2], [first,second], label=str(session_id), color=col[session_id], zorder=session_id)
            plt.plot([0.1, 0.2], [first,second], color=col[session_id], zorder=session_id)
            plt.xticks([0.1, 0.2], ["First half\nof sleep", "Second half\nof sleep"])
        plt.ylabel("Delta sim_ratio")
        plt.grid(axis="y", color="gray")
        plt.ylim(0,0.55)
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("drift_first_vs_second_half.svg", transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_opposing_nrem_rem(self, save_fig=False, template_type="phmm"):
        """
        compute cumulative effect on similarity ratio from NREM and REM

        :param save_fig: whether to save the figure (True)
        :type save_fig: bool
        """

        # --------------------------------------------------------------------------------------------------------------
        # get results from all sessions
        # --------------------------------------------------------------------------------------------------------------
        ds_rem_cum = []
        ds_nrem_cum = []
        for i, session in enumerate(self.session_list):
            if template_type =="ising" and session.session_name == "mjc148R4R_0113":
                # this session shows some weird behavior
                ds_rem_cum_, ds_nrem_cum_ = session.long_sleep().memory_drift_plot_rem_nrem(template_type=template_type,
                                                                                                     plotting=False,
                                                                                            n_moving_average_pop_vec=60,
                                                                                            rem_pop_vec_threshold=100)
            else:
                ds_rem_cum_, ds_nrem_cum_ = session.long_sleep().memory_drift_plot_rem_nrem(template_type=template_type,
                                                                                                     plotting=False)
            ds_rem_cum.append(ds_rem_cum_)
            ds_nrem_cum.append(ds_nrem_cum_)

        ds_rem_cum = np.array(ds_rem_cum)
        ds_nrem_cum = np.array(ds_nrem_cum)

        print("REM vs. NREM delta score (MWU):")
        print(mannwhitneyu(ds_rem_cum, ds_nrem_cum))

        # --------------------------------------------------------------------------------------------------------------
        # plotting
        # --------------------------------------------------------------------------------------------------------------
        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(3,5))
        plt.bar(1, np.mean(ds_rem_cum), 0.6, color="red", label="Mean REM")
        plt.scatter(np.ones(ds_rem_cum.shape[0]),ds_rem_cum, zorder=1000, edgecolors="black", facecolor="none",
                    label="Session values")
        plt.hlines(0,0.5,2.5)
        plt.xlim(0.5,2.5)
        plt.bar(2, np.mean(ds_nrem_cum), 0.6, color="blue", label="Mean NREM")
        plt.scatter(np.ones(ds_nrem_cum.shape[0])*2,ds_nrem_cum, zorder=1000, edgecolors="black",facecolor="none")
        plt.xticks([1,2],["REM", "NREM"])
        plt.ylabel("CUMULATIVE DELTA SCORE")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.title("Opposing net effect for \nREM and NREM")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("opp_rem_nrem.svg", transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_neighbouring_epochs(self, save_fig=False, template_type="phmm", first_type="rem"):
        """
        computes delta scores of similarity ratio for neighboring and non-neighboring epochs from all sessions
        and scatter plots them

        :param save_fig: whether to save figure (True) or not
        :type save_fig: bool
        """
        if save_fig:
            plt.style.use('default')

        def make_square_axes(ax):
            """Make an axes square in screen units.

            Should be called after plotting.
            """
            ax.set_aspect(1 / ax.get_data_ratio())

        # --------------------------------------------------------------------------------------------------------------
        # non-neighboring epochs
        # --------------------------------------------------------------------------------------------------------------
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        ds_rem_nn = []
        ds_nrem_nn = []
        for i, session in enumerate(self.session_list):
            ds_rem_arr, ds_nrem_arr = session.long_sleep().memory_drift_neighbouring_epochs(template_type= template_type,
                                                                                            plotting=False,
                                                                                            first_type=first_type)
            ds_rem_arr_nn = ds_rem_arr[2:]
            ds_nrem_arr_nn = ds_nrem_arr[:-2]
            plt.scatter(ds_rem_arr_nn, ds_nrem_arr_nn, color=col[i], alpha=0.8, label=str(i))
            ds_rem_nn.append(ds_rem_arr_nn)
            ds_nrem_nn.append(ds_nrem_arr_nn)
        ds_rem_nn = np.hstack(ds_rem_nn)
        ds_nrem_nn = np.hstack(ds_nrem_nn)
        print("Non-neighbouring epochs, "+str(pearsonr(ds_rem_nn, ds_nrem_nn)))
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_rem_nn, ds_nrem_nn)

        if save_fig:
            line_col = "black"
        else:
            line_col = "white"

        plt.plot(ds_rem_nn, intercept + slope * ds_rem_nn, color=line_col, label="R="+
                                                            str(np.round(pearsonr(ds_rem_nn, ds_nrem_nn)[0], 2)))
        plt.xlabel("DELTA REM")
        plt.ylabel("DELTA NREM")
        plt.title("Non-Neighbouring Epochs")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)
        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("nn_epochs_"+template_type+"_"+first_type+"_"+".svg", transparent="True")
            plt.close()
        else:
            plt.show()
        # --------------------------------------------------------------------------------------------------------------
        # neighboring epochs
        # --------------------------------------------------------------------------------------------------------------
        ds_rem = []
        ds_nrem = []
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for i, session in enumerate(self.session_list):
            ds_rem_arr, ds_nrem_arr = session.long_sleep().memory_drift_neighbouring_epochs(template_type=template_type,
                                                                                            plotting=False, first_type=first_type)
            plt.scatter(ds_rem_arr, ds_nrem_arr, color=col[i], alpha=0.8, label=str(i))
            ds_rem.append(ds_rem_arr)
            ds_nrem.append(ds_nrem_arr)
        ds_rem = np.hstack(ds_rem)
        ds_nrem = np.hstack(ds_nrem)
        print("Neighbouring epochs, "+str(pearsonr(ds_rem, ds_nrem)))
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_rem, ds_nrem)
        plt.plot(ds_rem, intercept + slope * ds_rem, color=line_col, label="R="+
                                                                          str(np.round(pearsonr(ds_rem, ds_nrem)[0], 2)))
        plt.xlabel("DELTA REM")
        plt.ylabel("DELTA NREM")
        plt.title("Neighbouring Epochs")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)
        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("n_epochs_"+template_type+"_"+first_type+"_"+".svg", transparent="True")
        else:
            plt.show()

    def long_sleep_memory_drift_neighbouring_epochs_same_sleep_phase(self, save_fig=False, template_type="phmm"):
        """
        computes delta scores of similarity ratio for neighboring and non-neighboring epochs from all sessions
        and scatter plots them

        :param save_fig: whether to save figure (True) or not
        :type save_fig: bool
        """
        if save_fig:
            plt.style.use('default')
            line_col = "black"
        else:
            line_col = "white"

        def make_square_axes(ax):
            """Make an axes square in screen units.

            Should be called after plotting.
            """
            ax.set_aspect(1 / ax.get_data_ratio())

        # --------------------------------------------------------------------------------------------------------------
        # non-neighboring epochs
        # --------------------------------------------------------------------------------------------------------------
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        ds_rem = []
        ds_nrem = []
        for i, session in enumerate(self.session_list):
            ds_rem_arr, ds_nrem_arr = session.long_sleep().memory_drift_neighbouring_epochs(template_type=
                                                                                                 template_type,
                                                                                                 plotting=False)
            ds_rem.append(ds_rem_arr)
            ds_nrem.append(ds_nrem_arr)
        ds_rem = np.hstack(ds_rem)
        ds_nrem = np.hstack(ds_nrem)

        # compute auto-correlation
        # --------------------------------------------------------------------------------------------------------------

        # for neighbouring REM epochs
        ds_rem_n = ds_rem[:-1]
        ds_rem_n_plus_1 = ds_rem[1:]
        print("Neighbouring REM epochs, " + str(pearsonr(ds_rem_n, ds_rem_n_plus_1)))
        plt.scatter(ds_rem_n,ds_rem_n_plus_1, color="lightgray", edgecolors="dimgray")
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_rem_n, ds_rem_n_plus_1)
        plt.plot(ds_rem_n, intercept + slope * ds_rem_n, color=line_col, label="R="+
                                                            str(np.round(pearsonr(ds_rem_n, ds_rem_n_plus_1)[0], 2)))
        plt.xlabel("DELTA REM n")
        plt.ylabel("DELTA REM n+1")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)
        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("neighboring_rem_epochs.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        # for neighbouring NREM epochs
        ds_nrem_n = ds_nrem[:-1]
        ds_nrem_n_plus_1 = ds_nrem[1:]
        print("Neighbouring NREM epochs, " + str(pearsonr(ds_nrem_n, ds_nrem_n_plus_1)))
        plt.scatter(ds_nrem_n,ds_nrem_n_plus_1, color="lightgray", edgecolors="dimgray")
        slope, intercept, r, p, stderr = scipy.stats.linregress(ds_nrem_n, ds_nrem_n_plus_1)
        plt.plot(ds_nrem_n, intercept + slope * ds_nrem_n, color=line_col, label="R="+
                                                            str(np.round(pearsonr(ds_nrem_n, ds_nrem_n_plus_1)[0], 2)))
        plt.xlabel("DELTA NREM n")
        plt.ylabel("DELTA NREM n+1")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)
        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("neighboring_nrem_epochs.svg", transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_firing_rate_changes(self, stats_test="t_test_two_sided", lump_data_together=False,
                                       use_firing_prob=True, save_fig=False):
        """
        compares firing rate changes during REM and NREM epochs for decreasing and increasing cells. Applies stat.
        test to see if difference is significant

        :param stats_test: which statistical test to use ("t_test", "mwu", "ks", "mwu_two_sided", "anova",
                           "t_test_one_sided")
        :type stats_test: str
        """

        if lump_data_together:
            rem_dec_list = []
            cum_rem_dec = []
            cum_nrem_dec = []
            cum_rem_inc = []
            cum_nrem_inc = []
            nrem_dec_list = []
            rem_inc_list = []
            nrem_inc_list = []
            for i, session in enumerate(self.session_list):
                rem_d, nrem_d, rem_i, nrem_i = \
                    session.long_sleep().firing_rate_changes(plotting=False, return_p_value=False,
                                                             use_firing_prob=use_firing_prob)
                rem_dec_list.append(rem_d)
                cum_rem_dec.append(np.sum(rem_d))
                cum_nrem_dec.append(np.sum(nrem_d))
                cum_rem_inc.append(np.mean(rem_i))
                cum_nrem_inc.append(np.mean(nrem_i))
                nrem_dec_list.append(nrem_d)
                rem_inc_list.append(rem_i)
                nrem_inc_list.append(nrem_i)

            rem_dec = np.hstack(rem_dec_list)
            nrem_dec = np.hstack(nrem_dec_list)
            rem_inc = np.hstack(rem_inc_list)
            nrem_inc = np.hstack(nrem_inc_list)

            rem_dec_sorted = np.sort(rem_dec)
            nrem_dec_sorted = np.sort(nrem_dec)
            rem_inc_sorted = np.sort(rem_inc)
            nrem_inc_sorted = np.sort(nrem_inc)


            p_rem_dec = 1. * np.arange(rem_dec_sorted.shape[0]) / (rem_dec_sorted.shape[0] - 1)
            p_nrem_dec = 1. * np.arange(nrem_dec_sorted.shape[0]) / (nrem_dec_sorted.shape[0] - 1)
            # plt.hlines(0.5, -0.02, 0.85, color="gray", linewidth=0.5)
            if save_fig:
                plt.style.use('default')
            plt.plot(rem_dec_sorted, p_rem_dec, color="red", label="REM")
            plt.plot(nrem_dec_sorted, p_nrem_dec, color="blue", label="NREM")
            plt.legend()
            plt.ylabel("cdf")
            plt.grid(axis="x", color="gray")
            if use_firing_prob:
                plt.xlabel("Delta: Firing prob.")
            else:
                plt.xlabel("Delta: Mean #spikes")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("firing_rate_changes_dec_all_sessions.svg", transparent="True")
                plt.close()
            else:
                plt.title("Decreasing cells \n"+"p-val:"+str(mannwhitneyu(nrem_dec, rem_dec, alternative="less")[1]))
                plt.show()

            p_rem_inc = 1. * np.arange(rem_inc_sorted.shape[0]) / (rem_inc_sorted.shape[0] - 1)
            p_nrem_inc = 1. * np.arange(nrem_inc_sorted.shape[0]) / (nrem_inc_sorted.shape[0] - 1)
            # plt.hlines(0.5, -0.02, 0.85, color="gray", linewidth=0.5)
            plt.plot(rem_inc_sorted, p_rem_inc, color="red", label="REM")
            plt.plot(nrem_inc_sorted, p_nrem_inc, color="blue", label="NREM")
            plt.legend()
            plt.ylabel("cdf")
            plt.grid(axis="x", color="gray")
            if use_firing_prob:
                plt.xlabel("Delta: Firing prob.")
            else:
                plt.xlabel("Delta: Mean #spikes")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("firing_rate_changes_inc_all_sessions.svg", transparent="True")
                plt.close()
            else:
                plt.title(
                    "Increasing cells\n" + "p-val:" + str(mannwhitneyu(nrem_inc, rem_inc, alternative="greater")[1]))
                plt.show()


        else:
            p_dec_list = []
            p_inc_list = []
            for i, session in enumerate(self.session_list):
                p_dec, p_inc = session.long_sleep().firing_rate_changes(plotting=False, stats_test=stats_test)
                p_dec_list.append(p_dec)
                p_inc_list.append(p_inc)

            # plotting
            # --------------------------------------------------------------------------------------------------------------

            for i, p in enumerate(p_dec_list):
                plt.scatter(i,p)
            plt.title("DECREASING")
            plt.yticks([0.05,0.01, 0.1,1])
            plt.ylabel("p-value "+stats_test)
            plt.grid(axis="y")
            plt.xlabel("SESSION")
            plt.show()

            for i, p in enumerate(p_inc_list):
                plt.scatter(i,p)
            plt.title("INCREASING")
            plt.yticks([0.05,0.01,0.1,1])
            plt.ylabel("p-value "+stats_test)
            plt.grid(axis="y")
            plt.xlabel("SESSION")
            plt.show()

    def long_sleep_firing_rate_changes_neighbouring_epochs(self, save_fig=False):

        nrem_dec_smooth = []
        rem_dec_smooth = []
        nrem_inc_smooth  = []
        rem_inc_smooth = []

        for i, session in enumerate(self.session_list):
            nrem_d, rem_d, nrem_i, rem_i = \
                session.long_sleep().firing_rate_changes_neighbouring_epochs()

            nrem_dec_smooth.append(nrem_d)
            rem_dec_smooth.append(rem_d)
            nrem_inc_smooth.append(nrem_i)
            rem_inc_smooth.append(rem_i)

        nrem_dec_smooth = np.hstack(nrem_dec_smooth)
        rem_dec_smooth = np.hstack(rem_dec_smooth)
        nrem_inc_smooth = np.hstack(nrem_inc_smooth)
        rem_inc_smooth = np.hstack(rem_inc_smooth)
        if save_fig:
            plt.style.use('default')
        print(pearsonr(nrem_dec_smooth, rem_dec_smooth))

        print(pearsonr(nrem_inc_smooth, rem_inc_smooth))

        print("Decreasing cells, neighbouring epochs, "+str(pearsonr(nrem_dec_smooth, rem_dec_smooth)))
        slope, intercept, r, p, stderr = scipy.stats.linregress(nrem_dec_smooth, rem_dec_smooth)
        plt.plot(rem_dec_smooth, intercept + slope * rem_dec_smooth, color="turquoise", label="R="+
                                                                          str(np.round(pearsonr(nrem_dec_smooth, rem_dec_smooth)[0], 2)))
        plt.scatter(rem_dec_smooth, nrem_dec_smooth, color="lightcyan", edgecolors="paleturquoise")
        plt.xlabel("DELTA firing prob. REM")
        plt.ylabel("DELTA firing prob. NREM")
        plt.title("Neighbouring Epochs: dec")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)
        def make_square_axes(ax):
            """Make an axes square in screen units.

            Should be called after plotting.
            """
            ax.set_aspect(1 / ax.get_data_ratio())
        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("n_epochs_firing_prob_dec.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        print("Increasing cells, neighbouring epochs, "+str(pearsonr(nrem_inc_smooth, rem_inc_smooth)))
        slope, intercept, r, p, stderr = scipy.stats.linregress(nrem_inc_smooth, rem_inc_smooth)
        plt.plot(rem_inc_smooth, intercept + slope * rem_inc_smooth, color="orange", label="R="+
                                                                          str(np.round(pearsonr(nrem_inc_smooth, rem_inc_smooth)[0], 2)))
        plt.scatter(rem_inc_smooth, nrem_inc_smooth, color="papayawhip", edgecolors="moccasin")
        plt.xlabel("DELTA firing prob. REM")
        plt.ylabel("DELTA firing prob. NREM")
        plt.title("Neighbouring Epochs: inc")
        plt.legend()
        y_min, y_max = plt.gca().get_ylim()
        y_lim = np.max(np.abs(np.array([y_min, y_max])))
        plt.vlines(0,-y_lim, y_lim, zorder=-1000, color="gray", linewidth=1)
        plt.ylim(-y_lim, y_lim)
        x_min, x_max = plt.gca().get_xlim()
        x_lim = np.max(np.abs(np.array([x_min, x_max])))
        plt.hlines(0,-x_lim, x_lim, zorder=-1000, color="gray", linewidth=1)
        plt.xlim(-x_lim, x_lim)
        def make_square_axes(ax):
            """Make an axes square in screen units.

            Should be called after plotting.
            """
            ax.set_aspect(1 / ax.get_data_ratio())
        make_square_axes(plt.gca())
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("n_epochs_firing_prob_inc.svg", transparent="True")
        else:
            plt.show()

        print("HERE")

    def long_sleep_firing_rates_all_cells(self, measure="mean", save_fig=False, chunks_in_min=2):
        firing_rem_stable = []
        firing_rem_dec = []
        firing_rem_inc = []
        firing_sleep_stable = []
        firing_sleep_dec = []
        firing_sleep_inc = []
        firing_nrem_stable = []
        firing_nrem_dec = []
        firing_nrem_inc = []

        for session in self.session_list:

            # get stable cell data
            # ----------------------------------------------------------------------------------------------------------

            sleep_z_stable = \
                session.long_sleep().firing_rate_distributions(plotting=False, measure=measure, cells_to_use="stable",
                                                               chunks_in_min=chunks_in_min, separate_sleep_phases=False)

            rem_z_stable, nrem_z_stable = \
                session.long_sleep().firing_rate_distributions(plotting=False, measure=measure, cells_to_use="stable",
                                                               chunks_in_min=chunks_in_min, separate_sleep_phases=True)

            firing_rem_stable.append(rem_z_stable)
            firing_sleep_stable.append(sleep_z_stable)
            firing_nrem_stable.append(nrem_z_stable)

            # get dec cell data
            # ----------------------------------------------------------------------------------------------------------

            sleep_z_dec = \
                session.long_sleep().firing_rate_distributions(plotting=False, measure=measure, cells_to_use="decreasing",
                                                               chunks_in_min=chunks_in_min, separate_sleep_phases=False)

            rem_z_dec, nrem_z_dec = \
                session.long_sleep().firing_rate_distributions(plotting=False, measure=measure, cells_to_use="decreasing",
                                                               chunks_in_min=chunks_in_min, separate_sleep_phases=True)

            firing_rem_dec.append(rem_z_dec)
            firing_sleep_dec.append(sleep_z_dec)
            firing_nrem_dec.append(nrem_z_dec)

            # get inc cell data
            # ----------------------------------------------------------------------------------------------------------

            sleep_z_inc = \
                session.long_sleep().firing_rate_distributions(plotting=False, measure=measure, cells_to_use="increasing",
                                                               chunks_in_min=chunks_in_min, separate_sleep_phases=False)

            rem_z_inc, nrem_z_inc = \
                session.long_sleep().firing_rate_distributions(plotting=False, measure=measure, cells_to_use="increasing",
                                                               chunks_in_min=chunks_in_min, separate_sleep_phases=True)

            firing_rem_inc.append(rem_z_inc)
            firing_sleep_inc.append(sleep_z_inc)
            firing_nrem_inc.append(nrem_z_inc)


        print("HERE")

        # combine session data
        firing_sleep_stable = np.hstack(firing_sleep_stable)
        firing_rem_stable = np.hstack(firing_rem_stable)
        firing_nrem_stable = np.hstack(firing_nrem_stable)

        # combine session data
        firing_sleep_dec = np.hstack(firing_sleep_dec)
        firing_rem_dec = np.hstack(firing_rem_dec)
        firing_nrem_dec = np.hstack(firing_nrem_dec)

        firing_sleep_inc = np.hstack(firing_sleep_inc)
        firing_rem_inc = np.hstack(firing_rem_inc)
        firing_nrem_inc = np.hstack(firing_nrem_inc)

        # combine dec and inc
        firing_sleep_unstable = np.hstack((firing_sleep_dec, firing_sleep_inc))
        firing_rem_unstable = np.hstack((firing_rem_dec, firing_rem_inc))
        firing_nrem_unstable = np.hstack((firing_nrem_dec, firing_nrem_inc))

        firing_rem_unstable = firing_rem_unstable[~np.isnan(firing_rem_unstable)]
        firing_nrem_unstable = firing_nrem_unstable[~np.isnan(firing_nrem_unstable)]

        # unstable vs. stable
        # --------------------------------------------------------------------------------------------------------------
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        print("Sleep")
        print(mannwhitneyu(firing_sleep_unstable, firing_sleep_stable))


        y_dat = [firing_sleep_unstable , firing_sleep_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Unstable", "Stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), whis=[0.001, 99.999], showfliers=False)
        plt.title("Sleep")
        plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-5, 200)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-4, 250)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("firing_rate_" + measure + "_sleep_stable_vs_unstable.svg", transparent="True")
            plt.close()
        else:
            plt.show()


        print("REM")
        print(mannwhitneyu(firing_rem_unstable, firing_rem_stable, alternative="less"))


        y_dat = [firing_rem_unstable , firing_rem_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Unstable", "Stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), whis=[0.001, 99.999], showfliers=False)
        plt.title("REM")
        plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-5, 200)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-4, 250)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("firing_rate_" + measure + "_rem_stable_vs_unstable.svg", transparent="True")
            plt.close()
        else:
            plt.show()


        print("NREM")
        print(mannwhitneyu(firing_nrem_unstable, firing_nrem_stable, alternative="greater"))


        y_dat = [firing_nrem_unstable , firing_nrem_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Unstable", "Stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), whis=[0.001, 99.999], showfliers=False)
        plt.title("NREM")
        plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-5, 200)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-4, 250)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("firing_rate_" + measure + "_nrem_stable_vs_unstable.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        # stable vs. dec vs. inc
        # --------------------------------------------------------------------------------------------------------------

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        print("Sleep: inc. vs dec.")
        print(mannwhitneyu(firing_sleep_dec, firing_sleep_inc))
        print("Sleep: inc. vs stable")
        print(mannwhitneyu(firing_sleep_inc, firing_sleep_stable))
        print("Sleep: dec. vs stable")
        print(mannwhitneyu(firing_sleep_dec, firing_sleep_stable))


        y_dat = [firing_sleep_dec, firing_sleep_inc, firing_sleep_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2, 3], patch_artist=True,
                            labels=["Dec", "Inc", "Stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), whis=[0.001, 99.999], showfliers=False)
        plt.title("Sleep")
        plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-5, 200)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-4, 250)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("firing_rate_" + measure + "_sleep_stable_vs_unstable.svg", transparent="True")
            plt.close()
        else:
            plt.show()


        print("REM")
        print("REM: inc. vs dec.")
        print(mannwhitneyu(firing_rem_dec, firing_rem_inc))
        print("REM: inc. vs stable")
        print(mannwhitneyu(firing_rem_inc, firing_rem_stable))
        print("REM: dec. vs stable")
        print(mannwhitneyu(firing_rem_dec, firing_rem_stable))

        y_dat = [firing_rem_dec, firing_rem_inc, firing_rem_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Dec", "Inc", "Stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), whis=[0.001, 99.999], showfliers=False)
        plt.title("REM")
        plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-5, 200)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-4, 250)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("firing_rate_" + measure + "_rem_stable_vs_unstable.svg", transparent="True")
            plt.close()
        else:
            plt.show()


        print("NREM")
        print("NREM: inc. vs dec.")
        print(mannwhitneyu(firing_nrem_dec, firing_nrem_inc))
        print("NREM: inc. vs stable")
        print(mannwhitneyu(firing_nrem_inc, firing_nrem_stable))
        print("NREM: dec. vs stable")
        print(mannwhitneyu(firing_nrem_dec, firing_nrem_stable))


        y_dat = [firing_nrem_dec, firing_nrem_inc , firing_nrem_stable]
        plt.figure(figsize=(2,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["Dec", "Inc", "Stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), whis=[0.001, 99.999], showfliers=False)
        plt.title("NREM")
        plt.yscale("symlog")
        if measure == "max":
            plt.ylim(-5, 200)
            plt.ylabel("Max firing rate (z-scored)")
        elif measure == "mean":
            plt.ylim(-4, 250)
            plt.ylabel("Mean firing rate (z-scored)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("firing_rate_" + measure + "_nrem_stable_vs_unstable.svg", transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_temporal_trend_stable_cells(self, save_fig=False):
        """
        Computes the slope of the memory drift (sim. ratio) using all or only stable cells and compares the two slopes.

        :param save_fig: whether to save (True) or show figure (False)
        :type save_fig: bool
        """
        slope_stable_list = []
        slope_all_list = []
        for i, session in enumerate(self.session_list):
            slope_all, slope_stable = session.long_sleep().memory_drift_plot_temporal_trend_stable_cells(plotting=False)
            slope_stable_list.append(slope_stable)
            slope_all_list.append(slope_all)

        slope_stable_arr = np.array(slope_stable_list)
        slope_all_arr = np.array(slope_all_list)

        # stats test
        print(mannwhitneyu(slope_stable_arr, slope_all_arr, alternative="less"))

        # plotting
        # --------------------------------------------------------------------------------------------------------------
        if save_fig:
            plt.style.use('default')
        plt.figure(figsize=(3,4))
        col = ["#8AAEA2", "#5E9080", "#26614E", "#3D7865", "#134D3A", "#6D9E50", "#94BE7B"]
        for session_id, (st,all) in enumerate(zip(slope_stable_arr, slope_all_arr)):
            plt.scatter([1,2], [all,st], label=str(session_id), color=col[session_id], zorder=session_id)
            plt.plot([1,2], [all,st], color=col[session_id], zorder=session_id)
            plt.xticks([1,2], ["All cells", "Only stable cells"])
        plt.yticks([0.0, 0.5, 1])
        plt.xlim([0.9,2.1])
        plt.ylim([-0.3,1.4])
        plt.ylabel("Normalized slope")
        plt.grid(axis="y")
        plt.legend()
        # save or show figure
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("slope_stable.svg", transparent="True")
        else:
            plt.show()

    def long_sleep_nrem_rem_likelihoods(self, template_type="phmm", save_fig=False):
        """
        compares the likelihoods from decoding for REM and NREM

        @param template_type: which template to use "phmm" or "ising"
        """
        pre_prob_rem_max = []
        pre_prob_nrem_max = []
        pre_prob_rem_flat = []
        pre_prob_nrem_flat = []
        pre_max_posterior_rem = []
        pre_max_posterior_nrem = []
        for i, session in enumerate(self.session_list):
            rem_max, nrem_max, rem_flat, nrem_flat, nrem_max_posterior, rem_max_posterior = \
                session.long_sleep().memory_drift_rem_nrem_likelihoods(plotting=False, template_type=template_type)

            pre_max_posterior_rem.append(rem_max_posterior)
            pre_max_posterior_nrem.append(nrem_max_posterior)
            pre_prob_rem_max.append(rem_max)
            pre_prob_nrem_max.append(nrem_max)
            pre_prob_rem_flat.append(rem_flat)
            pre_prob_nrem_flat.append(nrem_flat)

        pre_prob_rem_max = np.hstack(pre_prob_rem_max)
        pre_prob_nrem_max = np.hstack(pre_prob_nrem_max)
        pre_prob_rem_flat = np.hstack(pre_prob_rem_flat)
        pre_prob_nrem_flat = np.hstack(pre_prob_nrem_flat)
        pre_max_posterior_rem = np.hstack(pre_max_posterior_rem)
        pre_max_posterior_nrem = np.hstack(pre_max_posterior_nrem)

        p_mwu = mannwhitneyu(pre_prob_rem_max, pre_prob_nrem_max, alternative="greater")
        print("Max. likelihoods, MWU-test: p-value = " + str(p_mwu))

        p_mwu = mannwhitneyu(pre_max_posterior_rem, pre_max_posterior_nrem)
        print("Max. posterior prob., MWU-test: p-value = " + str(p_mwu))

        pre_prob_rem_max_sorted = np.sort(pre_prob_rem_max)
        pre_prob_nrem_max_sorted = np.sort(pre_prob_nrem_max)

        p_rem = 1. * np.arange(pre_prob_rem_max.shape[0]) / (pre_prob_rem_max.shape[0] - 1)
        p_nrem = 1. * np.arange(pre_prob_nrem_max.shape[0]) / (pre_prob_nrem_max.shape[0] - 1)
        if save_fig:
            plt.style.use('default')
            plt.close()
        plt.plot(pre_prob_rem_max_sorted, p_rem, color="red", label="REM")
        plt.plot(pre_prob_nrem_max_sorted, p_nrem, color="blue", label="NREM")
        plt.gca().set_xscale("log")
        plt.xlabel("max. likelihood per PV")
        plt.ylabel("CDF")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoding_all_sessions_max_likelihoods.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        pre_prob_rem_flat_sorted = np.sort(pre_prob_rem_flat)
        pre_prob_nrem_flat_sorted = np.sort(pre_prob_nrem_flat)

        p_rem_flat = 1. * np.arange(pre_prob_rem_flat.shape[0]) / (pre_prob_rem_flat.shape[0] - 1)
        p_nrem_flat = 1. * np.arange(pre_prob_nrem_flat.shape[0]) / (pre_prob_nrem_flat.shape[0] - 1)
        plt.plot(pre_prob_rem_flat_sorted, p_rem_flat, color="red", label="REM")
        plt.plot(pre_prob_nrem_flat_sorted, p_nrem_flat, color="blue", label="NREM")
        plt.gca().set_xscale("log")
        plt.xlabel("Likelihoods per PV")
        plt.ylabel("CDF")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoding_all_sessions_likelihoods.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        pre_max_posterior_rem_sorted = np.sort(pre_max_posterior_rem)
        pre_max_posterior_nrem_sorted = np.sort(pre_max_posterior_nrem)

        p_rem_flat = 1. * np.arange(pre_max_posterior_rem.shape[0]) / (pre_max_posterior_rem.shape[0] - 1)
        p_nrem_flat = 1. * np.arange(pre_max_posterior_nrem.shape[0]) / (pre_max_posterior_nrem.shape[0] - 1)
        plt.plot(pre_max_posterior_rem_sorted, p_rem_flat, color="red", label="REM")
        plt.plot(pre_max_posterior_nrem_sorted, p_nrem_flat, color="blue", label="NREM")
        # plt.gca().set_xscale("log")
        plt.xlabel("Max. posterior probability per PV")
        plt.ylabel("CDF")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoding_all_sessions_max_posterior_prob.svg", transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_nrem_rem_decoding_similarity(self, template_type="phmm", save_fig=False, pre_or_post="pre"):
        """
        compares the likelihoods from decoding for REM and NREM

        @param template_type: which template to use "phmm" or "ising"
        """
        pre_rem_mode_freq_norm = []
        pre_nrem_mode_freq_norm = []
        pre_nrem_mode_freq_norm_odd = []
        pre_nrem_mode_freq_norm_even = []
        pre_rem_mode_freq_norm_odd = []
        pre_rem_mode_freq_norm_even = []

        for i, session in enumerate(self.session_list):
            pre_rem, pre_nrem, pre_rem_odd, pre_rem_even, pre_nrem_odd, pre_nrem_even = \
                session.long_sleep().memory_drift_rem_nrem_decoding_similarity(plotting=False,
                                                                               template_type=template_type,
                                                                               pre_or_post=pre_or_post)

            pre_rem_mode_freq_norm.append(pre_rem)
            pre_nrem_mode_freq_norm.append(pre_nrem)
            pre_nrem_mode_freq_norm_odd.append(pre_rem_odd)
            pre_nrem_mode_freq_norm_even.append(pre_rem_even)
            pre_rem_mode_freq_norm_odd.append(pre_nrem_odd)
            pre_rem_mode_freq_norm_even.append(pre_nrem_even)

        pre_rem_mode_freq_norm = np.hstack(pre_rem_mode_freq_norm)
        pre_nrem_mode_freq_norm = np.hstack(pre_nrem_mode_freq_norm)
        # pre_nrem_mode_freq_norm_odd = np.hstack(pre_nrem_mode_freq_norm_odd)
        # pre_nrem_mode_freq_norm_even = np.hstack(pre_nrem_mode_freq_norm_even)
        # pre_rem_mode_freq_norm_odd = np.hstack(pre_rem_mode_freq_norm_odd)
        # pre_rem_mode_freq_norm_even = np.hstack(pre_rem_mode_freq_norm_even)

        diff = (pre_nrem_mode_freq_norm - pre_rem_mode_freq_norm)
        diff_sh = []
        nr_shuffle = 500
        for i in range(nr_shuffle):
            diff_sh.append((pre_nrem_mode_freq_norm - np.random.permutation(pre_rem_mode_freq_norm)))
        diff_sh = np.hstack(diff_sh)
        diff_sh = np.abs(diff_sh)
        diff = np.abs(diff)
        p_diff = 1. * np.arange(diff.shape[0]) / (diff.shape[0] - 1)
        p_diff_shuffle = 1. * np.arange(diff_sh.shape[0]) / (diff_sh.shape[0] - 1)

        def make_square_axes(ax):
            """Make an axes square in screen units.

            Should be called after plotting.
            """
            ax.set_aspect(1 / ax.get_data_ratio())
        if save_fig:
            plt.style.use('default')

        plt.scatter(pre_rem_mode_freq_norm, pre_nrem_mode_freq_norm, edgecolors="darkgray", facecolor="dimgrey",
                    linewidths=0.5)
        if template_type == "phmm":
            plt.text(0,0.6, "Spearman corr. = " + str(np.round(spearmanr(pre_rem_mode_freq_norm, pre_nrem_mode_freq_norm)[0],4)))
        elif template_type == "ising":
            plt.text(0,0.07, "Spearman corr. = " + str(np.round(spearmanr(pre_rem_mode_freq_norm, pre_nrem_mode_freq_norm)[0],4)))
        make_square_axes(plt.gca())
        if template_type == "phmm":
            plt.xlabel("mode decoding probability - REM")
            plt.ylabel("mode decoding probability - NREM")
        elif template_type == "ising":
            plt.xlabel("spatial bin decoding probability - REM")
            plt.ylabel("spatial bin decoding probability - NREM")
            plt.xlim(-0.005, 0.07)
            plt.ylim(-0.005, 0.08)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            if template_type == "phmm":
                plt.savefig("mode_decoding_prob.svg", transparent="True")
            elif template_type == "ising":
                plt.savefig("spatial_bin_decoding_prob.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        exit()
        ratio = pre_rem_mode_freq_norm / pre_nrem_mode_freq_norm
        ratio[ratio > 1e308] = np.nan
        ratio = ratio[~np.isnan(ratio)]
        plt.hist(ratio, bins=200)
        plt.xscale("log")
        plt.show()

        plt.plot(np.sort(diff), p_diff, label="data")
        plt.plot(np.sort(diff_sh), p_diff_shuffle, label="shuffle")
        plt.show()

        plt.scatter(pre_rem_mode_freq_norm_odd, pre_rem_mode_freq_norm_even)
        plt.xlabel("Norm. frequency - odd")
        plt.ylabel("Norm. frequency - even")
        plt.title(template_type+", REM: even vs. odd\n Pears.:"+str(pearsonr(pre_rem_mode_freq_norm_odd, pre_rem_mode_freq_norm_even)[0])+
                  "\n Spear.:"+str(spearmanr(pre_rem_mode_freq_norm_odd, pre_rem_mode_freq_norm_even)[0]))
        plt.show()


        plt.scatter(pre_nrem_mode_freq_norm_odd, pre_nrem_mode_freq_norm_even)
        plt.xlabel("Norm. frequency - odd")
        plt.ylabel("Norm. frequency - even")
        plt.title(template_type+", NREM: even vs. odd\n Pears.:"+str(pearsonr(pre_nrem_mode_freq_norm_odd, pre_nrem_mode_freq_norm_even)[0])+
                  "\n Spear.:"+str(spearmanr(pre_nrem_mode_freq_norm_odd, pre_nrem_mode_freq_norm_even)[0]))
        plt.show()

        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoding_all_sessions_max_likelihoods.svg", transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_nrem_rem_decoding_similarity_ising_and_phmm(self, save_fig=False):
        """
        compares the likelihoods from decoding for REM and NREM

        @param template_type: which template to use "phmm" or "ising"
        """
        r_phmm = []
        r_ising = []

        for i, session in enumerate(self.session_list):
            pre_rem_phmm, pre_nrem_phmm = \
                session.long_sleep().memory_drift_rem_nrem_decoding_similarity(plotting=False, template_type="phmm")

            r_phmm.append(pearsonr(pre_rem_phmm, pre_nrem_phmm)[0])

            pre_rem_ising, pre_nrem_ising = \
                session.long_sleep().memory_drift_rem_nrem_decoding_similarity(plotting=False, template_type="ising")

            r_ising.append(pearsonr(pre_rem_ising, pre_nrem_ising)[0])

        r_phmm = np.array(r_phmm)
        r_ising = np.array(r_ising)

        y_dat = [r_phmm, r_ising]

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(3,4))
        bplot = plt.boxplot(y_dat, positions=[1, 2], patch_artist=True,
                            labels=["pHMM", "Bayesian"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["darksalmon", 'bisque']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Pearson R")
        plt.ylim(0,1)
        plt.grid(color="grey", axis="y")
        plt.show()

        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("r_nrem_rem_similarity.svg", transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_nrem_rem_decoding_cleanliness_per_mode(self, template_type="phmm", save_fig=False, control_data="rem"):
        """
        compares the likelihoods from decoding for REM and NREM

        @param template_type: which template to use "phmm" or "ising"
        """
        per_mode_ratio = []
        per_mode_ratio_sign = []
        mean_post_prob_rem_significant = []
        mean_post_prob_nrem_significant = []
        post_prob_rem_significant = []
        post_prob_nrem_significant = []
        above_one = []
        above_one_sign = []

        for i, session in enumerate(self.session_list):
            pmr_sign, pmr, m_rem, m_nrem, pp_rem, pp_nrem = session.long_sleep().memory_drift_rem_nrem_decoding_cleanliness_per_mode(plotting=False,
                                                                                          template_type=template_type,
                                                                                          control_data=control_data)
            post_prob_rem_significant.append(pp_rem)
            post_prob_nrem_significant.append(pp_nrem)
            per_mode_ratio_sign.append(pmr_sign)
            per_mode_ratio.append(pmr)
            above_one.append(np.count_nonzero(pmr>1)/pmr.shape[0])
            above_one_sign.append(np.count_nonzero(pmr_sign > 1) / pmr_sign.shape[0])
            mean_post_prob_rem_significant.append(m_rem)
            mean_post_prob_nrem_significant.append(m_nrem)

        per_mode_ratio = np.hstack(per_mode_ratio)
        per_mode_ratio_sign = np.hstack(per_mode_ratio_sign)

        mean_post_prob_rem_significant = np.hstack(mean_post_prob_rem_significant)
        mean_post_prob_nrem_significant = np.hstack(mean_post_prob_nrem_significant)

        post_prob_rem_significant = np.hstack(post_prob_rem_significant)
        post_prob_nrem_significant = np.hstack(post_prob_nrem_significant)

        print("All sessions lumped, only sign. different ones:")
        print(ttest_1samp(per_mode_ratio_sign, popmean=1, alternative="greater"))
        print("All sessions lumped:")
        print(ttest_1samp(per_mode_ratio[~np.isnan(per_mode_ratio)], popmean=1, alternative="greater"))

        print("Ratios, only sign.")
        print(ttest_1samp(above_one_sign, popmean=0.5, alternative="greater"))
        print("Ratios")
        print(ttest_1samp(above_one, popmean=0.5, alternative="greater"))

        print("Raw mean post. probabilites")
        print(mannwhitneyu(mean_post_prob_rem_significant, mean_post_prob_nrem_significant))

        if save_fig:
            plt.style.use('default')
        sns.kdeplot(mean_post_prob_rem_significant, fill=True, color="red", label="REM")
        sns.kdeplot(mean_post_prob_nrem_significant, fill=True, color="blue", label="NREM")
        plt.xlabel("Mean post. probability per mode")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoding_cleanliness_per_mode.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        if save_fig:
            plt.style.use('default')
        g = sns.kdeplot(per_mode_ratio_sign, fill=True)
        max_y = g.viewLim.bounds[3]
        plt.vlines(1,0,max_y)
        plt.xlim(0.45, 1.55)
        plt.ylim(0, max_y)
        plt.xlabel("Post. prob. REM / Post. prob. NREM")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoding_cleanliness_per_mode_ratio.svg", transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_nrem_rem_decoding_cleanliness(self, template_type="phmm", save_fig=False):
        """
        compares the likelihoods from decoding for REM and NREM

        @param template_type: which template to use "phmm" or "ising"
        """
        rem_first_second_max_ratio = []
        nrem_first_second_max_ratio = []
        max_prob_rem = []
        max_prob_nrem = []

        for i, session in enumerate(self.session_list):
            rfs, nfs, mr, mn = \
                session.long_sleep().memory_drift_rem_nrem_decoding_cleanliness(plotting=False, template_type=template_type)

            rem_first_second_max_ratio.append(rfs)
            nrem_first_second_max_ratio.append(nfs)
            max_prob_rem.append(mr)
            max_prob_nrem.append(mn)

        rem_first_second_max_ratio = np.hstack(rem_first_second_max_ratio)
        nrem_first_second_max_ratio = np.hstack(nrem_first_second_max_ratio)
        max_prob_rem = np.hstack(max_prob_rem)
        max_prob_nrem = np.hstack(max_prob_nrem)

        p_ratio_rem = 1. * np.arange(rem_first_second_max_ratio.shape[0]) / (rem_first_second_max_ratio.shape[0] - 1)
        p_ratio_nrem = 1. * np.arange(nrem_first_second_max_ratio.shape[0]) / (nrem_first_second_max_ratio.shape[0] - 1)

        p_prob_rem = 1. * np.arange(max_prob_rem.shape[0]) / (max_prob_rem.shape[0] - 1)
        p_prob_nrem = 1. * np.arange(max_prob_nrem.shape[0]) / (max_prob_nrem.shape[0] - 1)

        plt.plot(np.sort(rem_first_second_max_ratio), p_ratio_rem, label="REM")
        plt.plot(np.sort(nrem_first_second_max_ratio), p_ratio_nrem, label="NREM")
        plt.xscale("log")
        plt.legend()
        plt.xlabel("max. likeli / second largest likeli")
        plt.ylabel("cdf")
        plt.show()
        print(mannwhitneyu(rem_first_second_max_ratio, nrem_first_second_max_ratio, alternative="greater"))

        plt.plot(np.sort(max_prob_rem), p_prob_rem, label="REM")
        plt.plot(np.sort(max_prob_nrem), p_prob_nrem, label="NREM")
        plt.ylabel("cdf")
        plt.xlabel("max. prob.")
        plt.legend()
        plt.show()

        print(mannwhitneyu(max_prob_rem, max_prob_nrem, alternative="greater"))
        exit()
        if save_fig:
            plt.style.use('default')
            plt.close()
        plt.scatter(pre_rem_mode_freq_norm, pre_nrem_mode_freq_norm)
        plt.xlabel("Norm. frequency - REM")
        plt.ylabel("Norm. frequency - NREM")
        plt.title(template_type+", REM vs. NREM\n Pears.:"+str(pearsonr(pre_rem_mode_freq_norm, pre_nrem_mode_freq_norm)[0])+
                  "\n Spear.:"+str(spearmanr(pre_rem_mode_freq_norm, pre_nrem_mode_freq_norm)[0]))
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoding_all_sessions_max_likelihoods.svg", transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_nrem_rem_autocorrelation_temporal(self, template_type="phmm", save_fig=False):
        """
        compares the likelihoods from decoding for REM and NREM

        @param template_type: which template to use "phmm" or "ising"
        """
        rem_exp = []
        nrem_exp = []

        for i, session in enumerate(self.session_list):
            re, nre = \
                session.long_sleep().memory_drift_rem_nrem_autocorrelation_temporal(plotting=False, template_type=template_type,
                                                                           duration_for_autocorrelation_nrem=1,
                                                                           duration_for_autocorrelation_rem=10)

            rem_exp.append(re)
            nrem_exp.append(nre)

        y_dat = np.vstack((np.array(rem_exp), np.array(nrem_exp)))

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"

        plt.figure(figsize=(3,4))
        bplot = plt.boxplot(y_dat.T, positions=[1, 2], patch_artist=True,
                            labels=["REM", "NREM"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["red", 'blue']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Exponential coefficient k")
        plt.grid(color="grey", axis="y")
        plt.yscale("symlog")
        plt.yticks([np.median(rem_exp), -1, -10, np.median(nrem_exp)])

        plt.text(-0.001, np.median(rem_exp), np.str(np.round(np.median(rem_exp), 2)))
        plt.text(-0.001, np.median(nrem_exp), np.str(np.round(np.median(nrem_exp), 2)))
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("exponential_coeff.svg", transparent="True")
            plt.close()
        else:
            plt.show()

    def long_sleep_constant_spike_bin_length(self, save_fig=False):
        """
        compares the likelihoods from decoding for REM and NREM

        @param template_type: which template to use "phmm" or "ising"
        """
        nrem_bin_length = []
        rem_bin_length = []

        for i, session in enumerate(self.session_list):
            n_l, r_l = \
                session.long_sleep().get_constant_spike_bin_length(plotting=False)

            nrem_bin_length.append(n_l)
            rem_bin_length.append(r_l)

        nrem_bin_length = np.hstack(nrem_bin_length)
        rem_bin_length = np.hstack(rem_bin_length)

        if save_fig:
            plt.style.use('default')

        p_rem = 1. * np.arange(rem_bin_length.shape[0]) / (rem_bin_length.shape[0] - 1)
        p_nrem = 1. * np.arange(nrem_bin_length.shape[0]) / (nrem_bin_length.shape[0] - 1)
        plt.plot(np.sort(nrem_bin_length), p_nrem, color="blue", label="NREM")
        plt.plot(np.sort(rem_bin_length), p_rem, color="red", label="REM")
        plt.legend()
        plt.xscale("log")
        plt.ylabel("cdf")
        plt.xlabel("12-spike-bin duration (s)")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("constant_spike_bin_length.svg", transparent="True")
            plt.close()
        else:
            plt.show()

    """#################################################################################################################
    #  others
    #################################################################################################################"""

    def compare_spatial_measures_and_firing_rates(self, spatial_resolution=2):

        firing_pre_stable, firing_pre_dec, firing_pre_inc, firing_sleep_stable, firing_sleep_dec, \
        firing_sleep_inc, firing_post_stable, firing_post_dec, firing_post_inc = \
            self.pre_long_sleep_post_firing_rates_all_cells(plotting=False, measure="mean")

        pre_stable_sk_s, post_stable_sk_s, pre_dec_sk_s, post_inc_sk_s, post_dec_sk_s, pre_inc_sk_s = \
            self.pre_post_cheeseboard_spatial_information(plotting=False, info_measure="skaggs_second",
                                                          remove_nan=False, spatial_resolution=spatial_resolution)
        pre_stable_spar, post_stable_spar, pre_dec_spar, post_inc_spar, post_dec_spar, pre_inc_spar = \
            self.pre_post_cheeseboard_spatial_information(plotting=False, info_measure="sparsity",
                                                          remove_nan=False, spatial_resolution=spatial_resolution)

        plt.scatter(pre_stable_sk_s, pre_stable_spar)
        plt.xlabel("Skaggs per second")
        plt.ylabel("Sparsity")
        plt.title("Stable (PRE)")
        plt.show()

        plt.scatter(firing_pre_stable, pre_stable_spar)
        plt.xlabel("Mean firing rate")
        plt.ylabel("Sparsity")
        plt.title("Stable (PRE)")
        plt.show()

        plt.scatter(firing_pre_stable, pre_stable_sk_s)
        plt.xlabel("Mean firing rate")
        plt.ylabel("Skaggs per second")
        plt.title("Stable (PRE)")
        plt.show()

        plt.scatter(pre_dec_sk_s, pre_dec_spar)
        plt.xlabel("Skaggs per second")
        plt.ylabel("Sparsity")
        plt.title("Decreasing (PRE)")
        plt.show()

        plt.scatter(firing_pre_dec, pre_dec_spar)
        plt.xlabel("Mean firing rate")
        plt.ylabel("Sparsity")
        plt.title("Decreasing (PRE)")
        plt.show()

        plt.scatter(firing_pre_dec, pre_dec_sk_s)
        plt.xlabel("Mean firing rate")
        plt.ylabel("Skaggs per second")
        plt.title("Decreasing (PRE)")
        plt.show()

    def post_cheeseboard_occupancy_around_goals(self, save_fig=False):

        around_goals = []
        wo_goals = []

        for session in self.session_list:
            occ_around_goals_per_cm2, occ_wo_goals_per_cm2 = session.cheese_board(experiment_phase=
                                                                                  ["learning_cheeseboard_2"]).occupancy_around_goals()
            around_goals.append(occ_around_goals_per_cm2)
            wo_goals.append(occ_wo_goals_per_cm2)

        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(4, 5))
        res = [around_goals, wo_goals]
        bplot = plt.boxplot(res, positions=[1, 2], patch_artist=True,
                            labels=["Around goals", "Away from goals"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )
        colors = ["gray", 'gray']
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Occupancy per spatial bin (s/m2)")
        plt.grid(color="grey", axis="y")
        print(mannwhitneyu(around_goals, wo_goals))
        plt.ylim(0, 0.3)
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("occupancy_post_all_sessions.svg", transparent="True")
        else:
            plt.show()

    def rate_map_stability_pre_probe_pre_post_post_probe(self, cells_to_use="all", spatial_resolution=5,
                                                         nr_of_splits=3, save_fig=False):

        sim_matrices = []
        for session in self.session_list:
            s_m = session.cheeseboard_pre_prob_pre_post_post_prob().rate_map_stability(cells_to_use=cells_to_use,
                                                                                    spatial_resolution=spatial_resolution,
                                                                                    plotting=False, nr_of_splits=nr_of_splits)
            sim_matrices.append(s_m)

        map_similarity=np.array(sim_matrices).mean(axis=0)


        labels = []
        phases = np.array(["pre-probe", "learn-PRE", "POST", "post-probe"])
        for phase in phases:
            for subdiv in range(nr_of_splits):
                labels.append(phase + "_" + str(subdiv))

        if save_fig:
            plt.style.use('default')
        # plt.figure(figsize=(6,5))
        plt.imshow(map_similarity, vmin=0, vmax=1)
        plt.yticks(np.arange(map_similarity.shape[0]), labels)
        plt.xticks(np.arange(map_similarity.shape[0]), labels, rotation='vertical')
        plt.xlim(-0.5, map_similarity.shape[0] - 0.5)
        plt.ylim(-0.5, map_similarity.shape[0] - 0.5)
        # plt.ylim(0, map_similarity.shape[0])
        a = plt.colorbar()
        a.set_label("Mean population vector correlation")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("rate_map_stability_"+cells_to_use+".svg", transparent="True")
            plt.close()
        else:
            plt.show()

    def rate_map_stability_pre_probe_pre_post_post_probe_cell_comparison(self, spatial_resolution=5, nr_of_splits=3,
                                                                         save_fig=False):

        sim_matrices_stable = []
        sim_matrices_dec = []
        sim_matrices_inc = []
        for session in self.session_list:
            s_m = session.cheeseboard_pre_prob_pre_post_post_prob().rate_map_stability(cells_to_use="stable",
                                                                                    spatial_resolution=spatial_resolution,
                                                                                    plotting=False, nr_of_splits=nr_of_splits)
            sim_matrices_stable.append(s_m)
            s_i = session.cheeseboard_pre_prob_pre_post_post_prob().rate_map_stability(cells_to_use="increasing",
                                                                                    spatial_resolution=spatial_resolution,
                                                                                    plotting=False, nr_of_splits=nr_of_splits)
            sim_matrices_inc.append(s_i)
            s_d = session.cheeseboard_pre_prob_pre_post_post_prob().rate_map_stability(cells_to_use="decreasing",
                                                                                    spatial_resolution=spatial_resolution,
                                                                                    plotting=False, nr_of_splits=nr_of_splits)
            sim_matrices_dec.append(s_d)

        map_similarity_stable=np.array(sim_matrices_stable).mean(axis=0)
        map_similarity_inc=np.array(sim_matrices_inc).mean(axis=0)
        map_similarity_dec=np.array(sim_matrices_dec).mean(axis=0)

        # compare PRE and POST
        # --------------------------------------------------------------------------------------------------------------
        pre_post_stable = map_similarity_stable[int((map_similarity_stable.shape[0]/2)):int((map_similarity_stable.shape[0]/2))+(nr_of_splits),
                          int((map_similarity_stable.shape[0]/2))-(nr_of_splits):int((map_similarity_stable.shape[0]/2))]

        pre_post_inc = map_similarity_inc[int((map_similarity_stable.shape[0]/2)):int((map_similarity_stable.shape[0]/2))+(nr_of_splits),
                          int((map_similarity_stable.shape[0]/2))-(nr_of_splits):int((map_similarity_stable.shape[0]/2))]

        pre_post_dec = map_similarity_dec[int((map_similarity_stable.shape[0]/2)):int((map_similarity_stable.shape[0]/2))+(nr_of_splits),
                          int((map_similarity_stable.shape[0]/2))-(nr_of_splits):int((map_similarity_stable.shape[0]/2))]

        # get all values (not mean) for stats test & boxplot
        # --------------------------------------------------------------------------------------------------------------
        sim_matrices_stable_arr = np.array(sim_matrices_stable)
        sim_matrices_dec_arr = np.array(sim_matrices_dec)
        sim_matrices_inc_arr = np.array(sim_matrices_inc)
        pre_post_stable_all_val = sim_matrices_stable_arr[:,int((map_similarity_stable.shape[0]/2)):int((map_similarity_stable.shape[0]/2))+(nr_of_splits),
                          int((map_similarity_stable.shape[0]/2))-(nr_of_splits):int((map_similarity_stable.shape[0]/2))].flatten()

        pre_post_inc_all_val = sim_matrices_inc_arr[:,int((map_similarity_stable.shape[0]/2)):int((map_similarity_stable.shape[0]/2))+(nr_of_splits),
                          int((map_similarity_stable.shape[0]/2))-(nr_of_splits):int((map_similarity_stable.shape[0]/2))].flatten()

        pre_post_dec_all_val = sim_matrices_dec_arr[:,int((map_similarity_stable.shape[0]/2)):int((map_similarity_stable.shape[0]/2))+(nr_of_splits),
                          int((map_similarity_stable.shape[0]/2))-(nr_of_splits):int((map_similarity_stable.shape[0]/2))].flatten()

        print("stable vs. dec")
        print(mannwhitneyu(pre_post_dec_all_val, pre_post_stable_all_val))
        print("stable vs. inc")
        print(mannwhitneyu(pre_post_inc_all_val, pre_post_stable_all_val))
        print("dec vs. inc")
        print(mannwhitneyu(pre_post_inc_all_val, pre_post_inc_all_val))

        y_dat = np.vstack((pre_post_dec_all_val, pre_post_inc_all_val, pre_post_stable_all_val))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(3,4))
        bplot = plt.boxplot(y_dat.T, positions=[1, 2, 3], patch_artist=True,
                            labels=["dec", "inc", "stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["blue", 'red', "yellow"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Mean population vector correlation")
        plt.grid(color="grey", axis="y")
        plt.ylim(0,1)
        # plt.yscale("symlog")
        # plt.yticks([np.median(rem_exp), 0.1, -1, np.median(nrem_exp), np.median(awake_exp)])
        # plt.text(-0.001, np.median(rem_exp), np.str(np.round(np.median(rem_exp), 2)))
        # plt.text(-0.001, np.median(nrem_exp), np.str(np.round(np.median(nrem_exp), 2)))
        # plt.text(-0.001, np.median(awake_exp), np.str(np.round(np.median(awake_exp), 2)))
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("rate_map_stability_all_cells_comparison.svg", transparent="True")
            plt.close()
        else:
            plt.show()


        probe_stable = map_similarity_stable[int(map_similarity_stable.shape[0]-nr_of_splits):int(map_similarity_stable.shape[0]+1),
                          0:int(nr_of_splits)]

        probe_inc = map_similarity_inc[int(map_similarity_stable.shape[0]-nr_of_splits):int(map_similarity_stable.shape[0]+1),
                          0:int(nr_of_splits)]

        probe_dec = map_similarity_dec[int(map_similarity_stable.shape[0]-nr_of_splits):int(map_similarity_stable.shape[0]+1),
                          0:int(nr_of_splits)]

        y_dat = np.vstack((probe_dec.flatten(), probe_inc.flatten(), probe_stable.flatten()))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        plt.figure(figsize=(3,4))
        bplot = plt.boxplot(y_dat.T, positions=[1, 2, 3], patch_artist=True,
                            labels=["dec", "inc", "stable"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c),
                            )
        colors = ["blue", 'red', "yellow"]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        plt.ylabel("Mean population vector correlation")
        plt.grid(color="grey", axis="y")
        plt.title("PRE_PROBE - POST_PROBE")
        # plt.yscale("symlog")
        # plt.yticks([np.median(rem_exp), 0.1, -1, np.median(nrem_exp), np.median(awake_exp)])
        # plt.text(-0.001, np.median(rem_exp), np.str(np.round(np.median(rem_exp), 2)))
        # plt.text(-0.001, np.median(nrem_exp), np.str(np.round(np.median(nrem_exp), 2)))
        # plt.text(-0.001, np.median(awake_exp), np.str(np.round(np.median(awake_exp), 2)))
        # if save_fig:
        #     plt.rcParams['svg.fonttype'] = 'none'
        #     plt.savefig("exponential_coeff_likelihood_vec.svg", transparent="True")
        #     plt.close()
        # else:
        plt.show()

        # check significance of difference between single entries
        # --------------------------------------------------------------------------------------------------------------

        sim_matrices_stable = np.array(sim_matrices_stable)
        sim_matrices_dec = np.array(sim_matrices_dec)
        sim_matrices_inc = np.array(sim_matrices_inc)

        # stable vs. decreasing
        stable_vs_dec = np.zeros(sim_matrices_stable.shape[1:])

        for iy,ix in np.ndindex(sim_matrices_stable.shape[1:]):
            if not iy==ix:
                stable_vs_dec[iy , ix] = mannwhitneyu(sim_matrices_stable[:,iy,ix],sim_matrices_dec[:,iy,ix])[1]
            else:
                stable_vs_dec[iy , ix] = np.nan


        labels = []
        phases = np.array(["pre-probe", "learn-PRE", "POST", "post-probe"])
        for phase in phases:
            for subdiv in range(nr_of_splits):
                labels.append(phase + "_" + str(subdiv))
        # plt.figure(figsize=(6,5))
        plt.imshow(stable_vs_dec)
        plt.yticks(np.arange(stable_vs_dec.shape[0]), labels)
        plt.xticks(np.arange(stable_vs_dec.shape[0]), labels, rotation='vertical')
        plt.xlim(-0.5, stable_vs_dec.shape[0] - 0.5)
        plt.ylim(-0.5, stable_vs_dec.shape[0] - 0.5)
        # plt.ylim(0, map_similarity.shape[0])
        a = plt.colorbar()
        a.set_label("p-value")
        plt.title("Stable vs. decreasing")
        plt.show()

        # stable vs. increasing
        stable_vs_inc = np.zeros(sim_matrices_stable.shape[1:])

        for iy,ix in np.ndindex(sim_matrices_stable.shape[1:]):
            if not iy==ix:
                stable_vs_inc[iy , ix] = mannwhitneyu(sim_matrices_stable[:,iy,ix],sim_matrices_inc[:,iy,ix])[1]
            else:
                stable_vs_inc[iy , ix] = np.nan


        labels = []
        phases = np.array(["pre-probe", "learn-PRE", "POST", "post-probe"])
        for phase in phases:
            for subdiv in range(nr_of_splits):
                labels.append(phase + "_" + str(subdiv))
        # plt.figure(figsize=(6,5))
        plt.imshow(stable_vs_inc)
        plt.yticks(np.arange(stable_vs_inc.shape[0]), labels)
        plt.xticks(np.arange(stable_vs_inc.shape[0]), labels, rotation='vertical')
        plt.xlim(-0.5, stable_vs_dec.shape[0] - 0.5)
        plt.ylim(-0.5, stable_vs_dec.shape[0] - 0.5)
        # plt.ylim(0, map_similarity.shape[0])
        a = plt.colorbar()
        a.set_label("p-value")
        plt.title("Stable vs. increasing")
        plt.show()

    def assess_stability_subsets(self, save_fig=False):
        stable_cell_z = []
        dec_cell_z = []
        inc_cell_z = []

        for session in self.session_list:
            s_z, d_z, i_z = session.all_data().assess_stability_subsets(plotting=False)
            stable_cell_z.append(s_z)
            dec_cell_z.append(d_z)
            inc_cell_z.append(s_z)

        all_stable_z = np.hstack(stable_cell_z)
        all_dec_z = np.hstack(dec_cell_z)
        all_inc_z = np.hstack(inc_cell_z)
        all_unstable_z = np.hstack((all_dec_z, all_inc_z))

        print("Stable vs. dec:")
        print(mannwhitneyu(all_stable_z, all_dec_z))

        print("Stable vs. inc:")
        print(mannwhitneyu(all_stable_z, all_inc_z))

        print("Stable vs. unstable:")
        print(mannwhitneyu(all_stable_z, all_unstable_z))

        p_stable = 1. * np.arange(all_stable_z.shape[0]) / (all_stable_z.shape[0] - 1)
        p_inc = 1. * np.arange(all_inc_z.shape[0]) / (all_inc_z.shape[0] - 1)
        p_dec = 1. * np.arange(all_dec_z.shape[0]) / (all_dec_z.shape[0] - 1)
        if save_fig:
            plt.style.use('default')
        plt.plot(np.sort(all_stable_z), p_stable, color="violet", label="stable")
        plt.plot(np.sort(all_dec_z), p_dec, color="turquoise", label="dec")
        plt.plot(np.sort(all_inc_z), p_inc, color="orange", label="inc")
        plt.ylabel("cdf")
        plt.xlabel("(mean_last_10%-mean_first_10%)/std_first_10% \n all features")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("clustering_stability.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        plt.plot(np.sort(np.abs(all_stable_z)), p_stable, color="violet", label="stable")
        plt.plot(np.sort(np.abs(all_dec_z)), p_dec, color="turquoise", label="dec")
        plt.plot(np.sort(np.abs(all_inc_z)), p_inc, color="orange", label="inc")
        plt.ylabel("cdf")
        plt.xlabel("abs((mean_last_10%-mean_first_10%)/std_first_10%) \n all features")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("clustering_stability_abs.svg", transparent="True")
        else:
            plt.show()

    def assess_stability(self, save_fig=False):
        cell_res = []

        for session in self.session_list:
            res = session.all_data().assess_stability(plotting=False)
            cell_res.append(res)

        # go through all sessions and test significance
        for sess_res in cell_res:
            print(mannwhitneyu(np.ma.masked_invalid(sess_res[0]),
                               np.ma.masked_invalid(sess_res[3])))

        # combine results from all sessions
        all_sess_res = []
        for hour in range(len(res)):
            hour_res = []
            for sess_res in cell_res:
                hour_res.extend(sess_res[hour])
            a = np.array(hour_res)
            all_sess_res.append(np.ma.masked_invalid(a))

        cmap = matplotlib.cm.get_cmap('viridis')
        colors_to_plot = cmap(np.linspace(0, 1, 4))
        if save_fig:
            plt.style.use('default')
            c = "black"
        else:
            c = "white"
        bplot = plt.boxplot(all_sess_res, positions=[1, 2, 3, 4], patch_artist=True,
                            labels=["2h-3h", "8h-9h", "16h-17h", "23h-24h"],
                            boxprops=dict(color=c),
                            capprops=dict(color=c),
                            whiskerprops=dict(color=c),
                            flierprops=dict(color=c, markeredgecolor=c),
                            medianprops=dict(color=c), showfliers=False
                            )

        for patch, color in zip(bplot['boxes'], colors_to_plot):
            patch.set_facecolor(color)
        plt.ylabel("Variance of z-scored features")
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("clustering_stability_all_cells_boxplot.svg", transparent="True")
            plt.close()
        else:
            plt.show()

        print("1 vs. 23:")
        print(ttest_ind(all_sess_res[0], all_sess_res[3]))
        print("8 vs. 16:")
        print(mannwhitneyu(all_sess_res[0], all_sess_res[3]))
        print("1 vs. 23:")
        print(mannwhitneyu(all_sess_res[0], all_sess_res[3]))
        print("1 vs. 23:")
        print(mannwhitneyu(all_sess_res[0], all_sess_res[3]))


        p_1 = 1. * np.arange(all_sess_res[0].shape[0]) / (all_sess_res[0].shape[0] - 1)
        p_8 = 1. * np.arange(all_sess_res[1].shape[0]) / (all_sess_res[1].shape[0] - 1)
        p_16 = 1. * np.arange(all_sess_res[2].shape[0]) / (all_sess_res[2].shape[0] - 1)
        p_23 = 1. * np.arange(all_sess_res[3].shape[0]) / (all_sess_res[3].shape[0] - 1)
        if save_fig:
            plt.style.use('default')
        plt.plot(np.sort(all_sess_res[0]), p_1, color=colors_to_plot[0], label="2h-3h")
        plt.plot(np.sort(all_sess_res[1]), p_8, color=colors_to_plot[1], label="8h-9h")
        plt.plot(np.sort(all_sess_res[2]), p_16, color=colors_to_plot[2], label="16h-17h")
        plt.plot(np.sort(all_sess_res[3]), p_23, color=colors_to_plot[3], label="23h-24h")
        plt.ylabel("cdf")
        plt.xlabel("Variance of z-scored features")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("clustering_stability_all_cells_cdf.svg", transparent="True")
            plt.close()
        else:
            plt.show()

    def plot_classified_cell_distribution(self, save_fig=False):
        """
        plots boxplot of nr. stable/decreasing/increasing cells

        @param save_fig: whether to save figure as svg
        @type save_fig: bool
        """
        stable = []
        decreasing = []
        increasing = []

        for session in self.session_list:
            cell_ids_stable, cell_ids_decreasing, cell_ids_increasing = session.cheese_board(
                experiment_phase=["learning_cheeseboard_1"]).get_cell_classification_labels()
            stable.append(cell_ids_stable.shape[0])
            decreasing.append(cell_ids_decreasing.shape[0])
            increasing.append(cell_ids_increasing.shape[0])

        print("stable: " + str(stable) + "\n")
        print("decreasing: " + str(decreasing) + "\n")
        print("increasing: " + str(increasing) + "\n")

        if save_fig:
            plt.style.use('default')

        plt.bar([0, 1, 2], [np.sum(np.array(stable)), np.sum(np.array(decreasing)), np.sum(np.array(increasing))],
                width=0.5, color=["violet", "green", "yellow"])
        plt.xticks([0, 1, 2], ["stable", "decreasing", "increasing"])
        plt.yticks([100, 200, 300, 400])
        plt.ylabel("Nr. cells")
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig("cell_classification_numbers.svg", transparent="True")
        plt.show()

    def before_sleep_sleep_compute_likelihoods(self, cells_to_use="all"):
        for session in self.session_list:
            session.sleep_before_sleep(data_to_use="ext").compute_likelihoods(cells_to_use=cells_to_use)

    def before_sleep_sleep_compare_likelihoods(self, use_max=True, save_fig=False, cells_to_use="all", split_sleep=True,
                                               z_score=False):

        if split_sleep:
            likelihood_sleep_before = []
            likelihood_sleep_1 = []
            likelihood_sleep_2 = []
        else:
            likelihood_sleep_before = []
            likelihood_sleep = []

        for session in self.session_list:
            if split_sleep:
                likeli_s_b, likeli_s1, likeli_s2 = session.sleep_before_sleep(data_to_use="std").compare_likelihoods(plotting=False,
                                                                                                         use_max=use_max,
                                                                                                         cells_to_use=cells_to_use,
                                                                                                         split_sleep=True)
                likelihood_sleep_before.append(likeli_s_b)
                likelihood_sleep_1.append(likeli_s1)
                likelihood_sleep_2.append(likeli_s2)
            else:
                likeli_s_b, likeli_s = session.sleep_before_sleep(data_to_use="std").compare_likelihoods(plotting=False,
                                                                                                         use_max=use_max,
                                                                                                         cells_to_use=cells_to_use,
                                                                                                         split_sleep=False)
                likelihood_sleep_before.append(likeli_s_b)
                likelihood_sleep.append(likeli_s)


        if split_sleep:

            likelihood_sleep_1 = np.hstack(likelihood_sleep_1)
            likelihood_sleep_2 = np.hstack(likelihood_sleep_2)
            likelihood_sleep_before = np.hstack(likelihood_sleep_before)

            print("Sleep before vs. sleep 1")
            print(mannwhitneyu(likelihood_sleep_before, likelihood_sleep_1))
            print("Sleep before vs. sleep 2")
            print(mannwhitneyu(likelihood_sleep_before, likelihood_sleep_2))

            p_sleep_before = 1. * np.arange(likelihood_sleep_before.shape[0]) / ( likelihood_sleep_before.shape[0] - 1)
            p_sleep_1 = 1. * np.arange(likelihood_sleep_1.shape[0]) / (likelihood_sleep_1.shape[0] - 1)
            p_sleep_2 = 1. * np.arange(likelihood_sleep_2.shape[0]) / (likelihood_sleep_2.shape[0] - 1)
            if save_fig:
                plt.style.use('default')
                plt.close()
            plt.plot(np.sort(likelihood_sleep_before), p_sleep_before, color="yellow", label="Sleep before")
            plt.plot(np.sort(likelihood_sleep_1), p_sleep_1, color="lightgreen", label="Sleep_1")
            plt.plot(np.sort(likelihood_sleep_2), p_sleep_2, color="limegreen", label="Sleep_2")
            if not z_score:
                plt.gca().set_xscale("log")
            if use_max:
                if z_score:
                    plt.xlabel("Max. likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Max. likelihood per PV")
            else:
                if z_score:
                    plt.xlabel("Likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Likelihood per PV")
            plt.ylabel("CDF")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("decoding_sleep_before_sleep_example_" + cells_to_use + ".svg", transparent="True")
                plt.close()
            else:
                plt.show()
        else:


            likelihood_sleep = np.hstack(likelihood_sleep)
            likelihood_sleep_before = np.hstack(likelihood_sleep_before)

            print("All sessions:\n")
            print(mannwhitneyu(likelihood_sleep_before, likelihood_sleep))

            p_sleep_before = 1. * np.arange(likelihood_sleep_before.shape[0]) / (likelihood_sleep_before.shape[0] - 1)
            p_sleep = 1. * np.arange(likelihood_sleep.shape[0]) / (likelihood_sleep.shape[0] - 1)
            if save_fig:
                plt.style.use('default')
                plt.close()
            plt.plot(np.sort(likelihood_sleep_before), p_sleep_before, color="greenyellow", label="Sleep before")
            plt.plot(np.sort(likelihood_sleep), p_sleep, color="limegreen", label="Sleep")
            plt.gca().set_xscale("log")
            if use_max:
                if z_score:
                    plt.xlabel("Max. likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Max. likelihood per PV")
            else:
                if z_score:
                    plt.xlabel("Likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Likelihood per PV")
            plt.ylabel("CDF")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("decoding_sleep_before_sleep_example_"+cells_to_use+".svg", transparent="True")
                plt.close()
            else:
                plt.show()
    def before_sleep_sleep_compare_max_post_probabilities(self, use_max=True, save_fig=False, cells_to_use="all"):

        max_posterior_prob_sleep_before = []
        max_posterior_prob_sleep_1 = []
        max_posterior_prob_sleep_2 = []

        for session in self.session_list:
            maxp_s_b, maxp_s1, maxp_s2 = session.sleep_before_sleep(data_to_use="std").compare_max_post_probabilities(plotting=False,
                                                                                                                      cells_to_use=cells_to_use)
            max_posterior_prob_sleep_before.append(maxp_s_b)
            max_posterior_prob_sleep_1.append(maxp_s1)
            max_posterior_prob_sleep_2.append(maxp_s2)


        max_posterior_prob_sleep_before = np.hstack(max_posterior_prob_sleep_before)
        max_posterior_prob_sleep_1 = np.hstack(max_posterior_prob_sleep_1)
        max_posterior_prob_sleep_2 = np.hstack(max_posterior_prob_sleep_2)

        p_sleep_before = 1. * np.arange(max_posterior_prob_sleep_before.shape[0]) / (
                    max_posterior_prob_sleep_before.shape[0] - 1)
        p_sleep_1 = 1. * np.arange(max_posterior_prob_sleep_1.shape[0]) / (
                    max_posterior_prob_sleep_1.shape[0] - 1)
        p_sleep_2 = 1. * np.arange(max_posterior_prob_sleep_2.shape[0]) / (
                    max_posterior_prob_sleep_2.shape[0] - 1)
        if save_fig:
            plt.style.use('default')
            plt.close()
        plt.plot(np.sort(max_posterior_prob_sleep_before), p_sleep_before, color="greenyellow",
                 label="Sleep before")
        plt.plot(np.sort(max_posterior_prob_sleep_1), p_sleep_1, color="lightgreen", label="Sleep_1")
        plt.plot(np.sort(max_posterior_prob_sleep_2), p_sleep_2, color="limegreen", label="Sleep_2")
        plt.gca().set_xscale("log")

        plt.xlabel("Max. post. probability per PV")
        plt.ylabel("CDF")
        plt.legend()
        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoding_sleep_before_sleep_example_" + cells_to_use + ".svg", transparent="True")
            plt.close()
        else:
            plt.show()

    def before_sleep_sleep_compute_likelihoods_subsets(self, cells_to_use="stable"):
        for session in self.session_list:
            session.sleep_before_sleep(data_to_use="ext").compute_likelihoods_subsets(cells_to_use=cells_to_use)

    def before_sleep_sleep_compare_likelihoods_subsets(self, use_max=True, save_fig=False, cells_to_use="stable", split_sleep=True,
                                               z_score=False):

        if split_sleep:
            likelihood_sleep_before = []
            likelihood_sleep_1 = []
            likelihood_sleep_2 = []
        else:
            likelihood_sleep_before = []
            likelihood_sleep = []

        for session in self.session_list:
            if split_sleep:
                likeli_s_b, likeli_s1, likeli_s2 = session.sleep_before_sleep(data_to_use="std").compare_likelihoods_subsets(plotting=False,
                                                                                                         use_max=use_max,
                                                                                                         cells_to_use=cells_to_use,
                                                                                                         split_sleep=True, z_score=z_score)
                likelihood_sleep_before.append(likeli_s_b)
                likelihood_sleep_1.append(likeli_s1)
                likelihood_sleep_2.append(likeli_s2)
            else:
                likeli_s_b, likeli_s = session.sleep_before_sleep(data_to_use="std").compare_likelihoods_subsets(plotting=False,
                                                                                                         use_max=use_max,
                                                                                                         cells_to_use=cells_to_use,
                                                                                                         split_sleep=False, z_score=z_score)
                likelihood_sleep_before.append(likeli_s_b)
                likelihood_sleep.append(likeli_s)


        if split_sleep:

            likelihood_sleep_1 = np.hstack(likelihood_sleep_1)
            likelihood_sleep_2 = np.hstack(likelihood_sleep_2)
            likelihood_sleep_before = np.hstack(likelihood_sleep_before)

            print("\n\n\n For all sessions:\n")
            print("Sleep before vs. sleep 1")
            print(mannwhitneyu(likelihood_sleep_before, likelihood_sleep_1))
            print("Sleep before vs. sleep 2")
            print(mannwhitneyu(likelihood_sleep_before, likelihood_sleep_2))

            p_sleep_before = 1. * np.arange(likelihood_sleep_before.shape[0]) / ( likelihood_sleep_before.shape[0] - 1)
            p_sleep_1 = 1. * np.arange(likelihood_sleep_1.shape[0]) / (likelihood_sleep_1.shape[0] - 1)
            p_sleep_2 = 1. * np.arange(likelihood_sleep_2.shape[0]) / (likelihood_sleep_2.shape[0] - 1)
            if save_fig:
                plt.style.use('default')
                plt.close()
            plt.plot(np.sort(likelihood_sleep_before), p_sleep_before, color="yellow", label="Sleep before")
            plt.plot(np.sort(likelihood_sleep_1), p_sleep_1, color="lightgreen", label="Sleep_1")
            plt.plot(np.sort(likelihood_sleep_2), p_sleep_2, color="limegreen", label="Sleep_2")
            if not z_score:
                plt.gca().set_xscale("log")
            if use_max:
                if z_score:
                    plt.xlabel("Max. likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Max. likelihood per PV")
            else:
                if z_score:
                    plt.xlabel("Likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Likelihood per PV")
            plt.ylabel("CDF")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("decoding_sleep_before_sleep_example_" + cells_to_use + ".svg", transparent="True")
                plt.close()
            else:
                plt.show()
        else:


            likelihood_sleep = np.hstack(likelihood_sleep)
            likelihood_sleep_before = np.hstack(likelihood_sleep_before)

            print("All sessions:\n")
            print(mannwhitneyu(likelihood_sleep_before, likelihood_sleep))

            p_sleep_before = 1. * np.arange(likelihood_sleep_before.shape[0]) / (likelihood_sleep_before.shape[0] - 1)
            p_sleep = 1. * np.arange(likelihood_sleep.shape[0]) / (likelihood_sleep.shape[0] - 1)
            if save_fig:
                plt.style.use('default')
                plt.close()
            plt.plot(np.sort(likelihood_sleep_before), p_sleep_before, color="greenyellow", label="Sleep before")
            plt.plot(np.sort(likelihood_sleep), p_sleep, color="limegreen", label="Sleep")
            plt.gca().set_xscale("log")
            if use_max:
                if z_score:
                    plt.xlabel("Max. likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Max. likelihood per PV")
            else:
                if z_score:
                    plt.xlabel("Likelihood per PV (z-scored)")
                else:
                    plt.xlabel("Likelihood per PV")
            plt.ylabel("CDF")
            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("decoding_sleep_before_sleep_example_"+cells_to_use+".svg", transparent="True")
                plt.close()
            else:
                plt.show()

    def before_sleep_sleep_diff_likelihoods_subsets(self, save_fig=False, split_sleep=False):

        if split_sleep:
            diff_sleep_sleep_1_stable = []
            diff_sleep_sleep_2_stable = []
            diff_sleep_sleep_1_dec = []
            diff_sleep_sleep_2_dec = []
        else:
            diff_sleep_stable = []
            diff_sleep_dec = []


        for session in self.session_list:
            if split_sleep:
                diff_s_1, diff_s_2  = session.sleep_before_sleep(data_to_use="std").likelihood_difference_subsets(plotting=False,
                                                                                                         cells_to_use="stable",
                                                                                                         split_sleep=True)
                diff_sleep_sleep_1_stable.append(diff_s_1)
                diff_sleep_sleep_2_stable.append(diff_s_2)

                diff_d_1, diff_d_2  = session.sleep_before_sleep(data_to_use="std").likelihood_difference_subsets(plotting=False,
                                                                                                         cells_to_use="decreasing",
                                                                                                         split_sleep=True)
                diff_sleep_sleep_1_dec.append(diff_d_1)
                diff_sleep_sleep_2_dec.append(diff_d_2)

            else:
                diff_s = session.sleep_before_sleep(data_to_use="std").likelihood_difference_subsets(plotting=False,
                                                                                                         cells_to_use="stable",
                                                                                                         split_sleep=False)
                diff_d = session.sleep_before_sleep(data_to_use="std").likelihood_difference_subsets(plotting=False,
                                                                                                         cells_to_use="decreasing",
                                                                                                         split_sleep=False)
                diff_sleep_stable.append(diff_s)
                diff_sleep_dec.append(diff_d)

        if split_sleep:

            diff_sleep_sleep_1_stable = np.hstack(diff_sleep_sleep_1_stable)
            diff_sleep_sleep_2_stable = np.hstack(diff_sleep_sleep_2_stable)
            diff_sleep_sleep_1_dec = np.hstack(diff_sleep_sleep_1_dec)
            diff_sleep_sleep_2_dec = np.hstack(diff_sleep_sleep_2_dec)

            print("\n\n\n For all sessions:\n")
            print("Sleep before vs. sleep 1")
            print(mannwhitneyu(diff_sleep_sleep_1_stable, diff_sleep_sleep_1_dec))
            print("Sleep before vs. sleep 2")
            print(mannwhitneyu(diff_sleep_sleep_2_stable, diff_sleep_sleep_2_dec))

            p_s1_stable = 1. * np.arange(diff_sleep_sleep_1_stable.shape[0]) / (diff_sleep_sleep_1_stable.shape[0] - 1)
            p_s1_dec = 1. * np.arange(diff_sleep_sleep_1_dec.shape[0]) / (diff_sleep_sleep_1_dec.shape[0] - 1)
            p_s2_stable = 1. * np.arange(diff_sleep_sleep_2_stable.shape[0]) / (diff_sleep_sleep_2_stable.shape[0] - 1)
            p_s2_dec = 1. * np.arange(diff_sleep_sleep_2_dec.shape[0]) / (diff_sleep_sleep_2_dec.shape[0] - 1)

            plt.plot(np.sort(diff_sleep_sleep_1_stable), p_s1_stable, color="violet", label="Stable")
            plt.plot(np.sort(diff_sleep_sleep_1_dec), p_s1_dec, color="turquoise", label="Decreasing")
            plt.xlabel("Likelihood diff. per mode")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("Sleep before - sleep 1")
            plt.xscale("log")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(".svg", transparent="True")
                plt.close()
            else:
                plt.show()
            plt.plot(np.sort(diff_sleep_sleep_2_stable), p_s2_stable, color="violet", label="Stable")
            plt.plot(np.sort(diff_sleep_sleep_2_dec), p_s2_dec, color="turquoise", label="Decreasing")
            plt.xlabel("Likelihood diff. per mode")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("Sleep before - sleep 2")
            plt.xscale("log")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(".svg", transparent="True")
                plt.close()
            else:
                plt.show()

        else:

            diff_sleep_sleep_stable = np.hstack(diff_sleep_stable)
            diff_sleep_sleep_dec = np.hstack(diff_sleep_dec)

            print("% pre-play modes for stable cells:")
            print(diff_sleep_sleep_stable[diff_sleep_sleep_stable>1].shape[0]/(diff_sleep_sleep_stable.shape[0]/100))
            print("% pre-play modes for dec cells:")
            print(diff_sleep_sleep_dec[diff_sleep_sleep_dec>1].shape[0]/(diff_sleep_sleep_dec.shape[0]/100))

            diff_sleep_sleep_stable = np.log(diff_sleep_sleep_stable)
            diff_sleep_sleep_dec = np.log(diff_sleep_sleep_dec)

            print("\n\n\n For all sessions:\n")
            print("Sleep before vs. sleep")
            print(mannwhitneyu(diff_sleep_sleep_stable, diff_sleep_sleep_dec))

            p_s_stable = 1. * np.arange(diff_sleep_sleep_stable.shape[0]) / (diff_sleep_sleep_stable.shape[0] - 1)
            p_s_dec = 1. * np.arange(diff_sleep_sleep_dec.shape[0]) / (diff_sleep_sleep_dec.shape[0] - 1)

            plt.plot(np.sort(diff_sleep_sleep_stable), p_s_stable, color="violet", label="Stable")
            plt.plot(np.sort(diff_sleep_sleep_dec), p_s_dec, color="turquoise", label="Decreasing")
            plt.xlabel("Per mode: mean likelihood sleep before/mean likelihood sleep after")
            plt.ylabel("CDF")
            plt.legend()
            plt.title("Sleep before - sleep after")
            plt.xscale("symlog")
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig(".svg", transparent="True")
                plt.close()
            else:
                plt.show()

    def before_sleep_pre_sleep_diff_likelihoods_subsets(self, save_fig=False, split_sleep=False):

        pre_modulation_stable = []
        likeli_ratio_sleep_stable = []
        pre_modulation_dec = []
        likeli_ratio_sleep_dec = []


        for session in self.session_list:
            pre_mod, sleep_mod = session.sleep_before_pre_sleep(data_to_use="std").pre_play_learning_phmm_modes(plotting=False,
                                                                                                     cells_to_use="stable")
            pre_modulation_stable.append(pre_mod)
            likeli_ratio_sleep_stable.append(sleep_mod)

            pre_mod, sleep_mod = session.sleep_before_pre_sleep(data_to_use="std").pre_play_learning_phmm_modes(plotting=False,
                                                                                                     cells_to_use="decreasing")
            pre_modulation_dec.append(pre_mod)
            likeli_ratio_sleep_dec.append(sleep_mod)

        pre_modulation_stable = np.hstack(pre_modulation_stable)
        likeli_ratio_sleep_stable = np.hstack(likeli_ratio_sleep_stable)
        pre_modulation_dec = np.hstack(pre_modulation_dec)
        likeli_ratio_sleep_dec = np.hstack(likeli_ratio_sleep_dec)

        print(pearsonr(pre_modulation_stable, likeli_ratio_sleep_stable)[0])
        plt.scatter(likeli_ratio_sleep_stable, pre_modulation_stable, edgecolor="blue", facecolor="lightblue", alpha=0.7)
        plt.ylabel("post_prob_first_20/post_prob_last_20")
        plt.xlabel("likeli_sleep_before/likeli_sleep_after")
        plt.yscale("symlog")
        plt.xscale("symlog")
        plt.xlim(-0.1,1.1)
        plt.ylim(-0.1,1.1)
        plt.hlines(1,0,1.1, color="white", zorder=-100)
        plt.vlines(1,0,1.1, color="white", zorder=-100)
        plt.title("Stable cells")
        # plt.gca().set_aspect('equal', 'box')
        plt.show()

        print(pearsonr(pre_modulation_dec, likeli_ratio_sleep_dec)[0])
        plt.scatter(likeli_ratio_sleep_dec, pre_modulation_dec, edgecolor="blue", facecolor="lightblue", alpha=0.7)
        plt.ylabel("post_prob_first_20/post_prob_last_20")
        plt.xlabel("likeli_sleep_before/likeli_sleep_after")
        plt.yscale("symlog")
        plt.xscale("symlog")
        plt.xlim(-0.1,1.1)
        plt.ylim(-0.1,1.1)
        plt.hlines(1,0,1.1, color="white", zorder=-100)
        plt.vlines(1,0,1.1, color="white", zorder=-100)
        plt.title("Decreasing cells")
        # plt.gca().set_aspect('equal', 'box')
        plt.show()