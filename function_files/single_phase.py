########################################################################################################################
#
#   ANALYSIS METHODS
#
#   Description: contains classes that are used to perform different types of analysis
#
#   Author: Lars Bollmann
#
#   Last modified: 10/10/2020
#
#   Structure:
#
#               (1) class BaseMethods: methods to analyze sleep and awake data for ONE POPULATION
#
#                   (a) class Sleep: derived from BaseMethods --> used to analyze sleep data
#
#                   (b) class Exploration: derived from BaseMethods --> used to analyze exploration data
#
#               (2) class BaseMethodsTwoPop: methods to analyze sleep and awake data for TWO POPULATIONS
#
#                   (a) class TwoPopSleep: derived from BaseMethods --> used to analyze sleep data
#
#                   (b) class TwoPopExploration: derived from BaseMethods --> used to analyze exploration data
#
#               (3) class Cheeseboard: methods to analyze cheeseboard task data
#
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import collections as matcoll
from matplotlib.patches import Circle
import numpy as matlib
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import median_absolute_deviation
import itertools
import copy
import time
import random
from functools import partial
import multiprocessing as mp
import os, glob, re
import pickle
import importlib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from numpy.linalg import norm
from math import floor
from scipy import io, signal
from scipy.special import logit, expit
from scipy.spatial import distance
from scipy.stats import pearsonr, entropy, spearmanr, sem, mannwhitneyu, wilcoxon, ks_2samp, multivariate_normal, \
    zscore, ttest_ind, ttest_ind_from_stats
from scipy.signal import find_peaks_cwt
import scipy.ndimage as nd
from scipy.io import loadmat
from sklearn.manifold import MDS
from sklearn.linear_model import Lasso
from sklearn.metrics import pairwise_distances
from scipy import optimize
from collections import OrderedDict

from function_files.support_functions import calc_pop_vector_entropy, synchronous_activity, pop_vec_dist, \
    upper_tri_without_diag, graph_distance, independent_shuffle, cross_correlate, \
    multi_dim_scaling, simple_gaussian, compute_power_stft, find_hse, down_sample_modify_raster, \
    butterworth_bandpass, butterworth_lowpass, moving_average, evaluate_clustering_fit, decode_using_ising_map, \
    decode_using_phmm_modes, compute_values_from_probabilities, perform_TSNE, perform_isomap, perform_PCA, \
    multi_dim_scaling, generate_colormap, correlations_from_raster, bayes_likelihood, collective_goal_coding, \
    distance_peak_firing_to_closest_goal, make_square_axes, compute_spatial_information, cross_correlate_matrices

from function_files.pre_processing import PreProcessSleep, PreProcessAwake

from function_files.plotting_functions import plot_2D_scatter, scatter_animation, scatter_animation_parallel, \
    plot_3D_scatter, plot_optimal_correlation_time_shift, view_dyn_co_firing, \
    plot_optimal_correlation_time_shift_hist, plot_optimal_correlation_time_shift_edges, plot_act_mat, \
    plot_multiple_gaussians, plot_pop_clusters

from function_files.ml_methods import MlMethodsOnePopulation, MlMethodsTwoPopulations, PoissonHMM

"""#####################################################################################################################
#   BASE CLASS
#####################################################################################################################"""


class BaseMethods:
    """Base class for general electro-physiological data analysis"""

    def __init__(self, data_dic, cell_type, params, session_params, experiment_phase):
        # --------------------------------------------------------------------------------------------------------------
        # args: - data_dic, dictionary with standard data
        #       - cell_type, string: cell type to do analysis with
        #       - params, python class: contains all parameters
        # --------------------------------------------------------------------------------------------------------------

        # get parameters
        self.params = copy.deepcopy(params)
        self.session_params = session_params
        self.session_name = self.session_params.session_name
        # get spatial factor: cm per .whl arbitrary unit
        self.spatial_factor = self.session_params.data_params_dictionary["spatial_factor"]

        # --------------------------------------------------------------------------------------------------------------
        # get phase specific info (ID & description)
        # --------------------------------------------------------------------------------------------------------------

        # check if list is passed:
        if isinstance(experiment_phase, list):
            experiment_phase = experiment_phase[0]

        self.experiment_phase = experiment_phase
        self.experiment_phase_id = session_params.data_description_dictionary[self.experiment_phase]

        # --------------------------------------------------------------------------------------------------------------
        # LFP PARAMETERS
        # --------------------------------------------------------------------------------------------------------------
        # if set to an integer --> only this tetrode is used to detect SWR etc., if None --> all tetrodes are used
        # check what tetrodes are assigned to different populations/hemispheres
        if hasattr(self.session_params, "tetrodes_p1_l") and hasattr(self.session_params, "tetrodes_p1_r"):
            if cell_type == "p1_l":
                self.session_params.lfp_tetrodes = self.session_params.tetrodes_p1_l
            elif cell_type == "p1_r":
                self.session_params.lfp_tetrodes = self.session_params.tetrodes_p1_r
        else:
            self.session_params.lfp_tetrodes = None

        # get data dictionary
        # check if list or dictionary is passed:
        if isinstance(data_dic, list):
            data_dic = data_dic[0]
        else:
            data_dic = data_dic

        # get all spike times - check if spikes from several populations (e.g. p2 and p3)
        if isinstance(cell_type, list) and len(cell_type)>1:
            self.firing_times ={**data_dic["spike_times"][cell_type[0]], **data_dic["spike_times"][cell_type[1]]}
        else:
            self.firing_times = data_dic["spike_times"][cell_type]

        # get last recorded spike
        if "last_spike" in data_dic.keys():
            self.last_spike = data_dic["last_spike"]
        else:
            self.last_spike = None

        # get location data
        self.whl = data_dic["whl"]

        # check if extended data dictionary is provided (contains lfp)
        if "eeg" in data_dic.keys():
            self.eeg = data_dic["eeg"]
        if "eegh" in data_dic.keys():
            self.eegh = data_dic["eegh"]

        # which cell type to be analyzed
        if isinstance(cell_type, list):
            self.cell_type = cell_type[0]+"_"+cell_type[1]
        else:
            self.cell_type = cell_type

        # initialize raster, loc and vel as None
        self.raster = None
        self.loc = None
        self.speed = None

        # initialize dimensionality reduction results as None
        self.result_dr = None

    """#################################################################################################################
    #   Standard visualization methods & simple computations
    #################################################################################################################"""

    def view_raster(self):
        """
        plot raster data
        """
        raster = self.raster[:, 0:100]
        plot_act_mat(raster, self.params, self.cell_type)

    def get_raster(self):
        """
         return raster
        """
        return self.raster

    def get_nr_cells(self):
        return len(self.firing_times)

    def save_raster(self, file_format=None, file_name=None):
        """
        save raster as file

        @param file_format: ["mat"] or None: file format e.g. for MATLAB
        @type file_format: str
        @param file_name: name of file
        @type file_name: str
        """

        if file_name is None:
            # if file name is not provided derive one
            file_name = self.cell_type + "_" + self.params.binning_method + "_" + \
                        str(self.params.time_bin_size)+"s"

        if file_format is None:
            np.save(file_name, self.raster, fix_imports=True)
        elif file_format == "mat":
            # export data as .mat file to use in matlab
            io.savemat(file_name+".mat", {"raster": self.raster})

    def get_raster_loc_vel(self):
        # --------------------------------------------------------------------------------------------------------------
        # return raster, location and speed
        # --------------------------------------------------------------------------------------------------------------

        return self.raster, self.loc, self.speed

    def get_location(self):
        # --------------------------------------------------------------------------------------------------------------
        # return transformed location data
        # --------------------------------------------------------------------------------------------------------------
        return self.loc

    def plot_location_data(self):
        plt.scatter(self.loc[:, 0], self.loc[:, 1])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def plot_speed(self):
        speed = self.speed
        plt.plot(speed)
        plt.show()

    def get_speed(self):
        speed = self.speed()
        return speed

    """#################################################################################################################
    #   Oscillation analysis
    #################################################################################################################"""

    def detect_swr_michele(self, thr=4):
        """
        detects sharp wave ripples (SWR) and returns start, peak and end timings at params.time_bin_size resolution

        @param thr: nr. std. to use for detection
        @type thr: int
        @return: SWR
        @rtype:
        """

        raise Exception("NEEDS TO BE IMPLEMENTED!")
        # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
        # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s
        # swr_pow = compute_power_stft(input_data=self.eegh, input_data_time_res=0.0002,
        #                         output_data_time_res=self.params.time_bin_size)
        #
        # plt.plot(swr_pow[:1000])
        # plt.show()

        # upper and lower bound in Hz for SWR
        freq_lo_bound=140
        freq_hi_bound=240

        # load data
        # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
        # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s

        # only select one tetrode
        data = self.eegh[:, 11]
        freq = 5000
        low_pass_cut_off_freq=30

        # nyquist theorem --> need half the frequency
        sig_bandpass = butterworth_bandpass(input_data=data, freq_nyquist=freq/2, freq_lo_bound=freq_lo_bound,
                                                   freq_hi_bound=freq_hi_bound)
        # compute rectified signal
        sig_abs = np.abs(sig_bandpass)
        # low pass filter signal
        sig_lo_pass = butterworth_lowpass(input_data=sig_abs, freq_nyquist=freq/2, cut_off_freq=low_pass_cut_off_freq)
        # z-score
        sig_z_scored = zscore(sig_lo_pass)

        SWR = []  # return [beginning, end, peak] of each event, in whl time indexes

        # add additional zero at the end
        swr_p = np.array(sig_z_scored, copy=True)
        swr_p = np.hstack([swr_p, 0])
        # check when it is above / below threshold
        der = np.array(swr_p[1:] > thr, dtype=int) - np.array(swr_p[:-1] > thr, dtype=int)
        # beginnings are where  der > 0
        begs = np.where(der > 0)[0] + 1
        last = 0
        for beg in begs:  # check each swr
            if beg > last:  # not to overlap
                # include 50 ms before: usually a lot of spiking happens before high SWR power
                first = max(beg - np.round(50/(1/freq)).astype(int), 0)
                # just a sanity check - something is wrong if this is not satisfied - probably threshold too low!!
                if np.min(swr_p[beg:beg + np.round(1000/self.params.time_bin_size).astype(int)]) < 0.8 * thr:
                    # end SWR where power is less 80% threshold
                    last = beg + np.where(swr_p[beg:beg + np.round(1000/(1/freq)).astype(int)]
                                          < 0.8 * thr)[0][0]
                    # peak power
                    peak = first + np.argmax(swr_p[first:last])
                    # check length: must be between 75 and 750 ms, else something is off
                    if (np.round(75/(1/freq)).astype(int) < last - first <
                            np.round(750/(1/freq)).astype(int)):
                        SWR.append([first, last, peak])
        return SWR

    def analyze_lfp(self):

        # upper and lower bound in Hz for SWR
        freq_lo_bound = 140
        freq_hi_bound = 240

        # load data
        # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
        # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s

        combined = []

        for tet in range(15):

            # only select one tetrode --> TODO: maybe select multiple, compute SWR times and take overlap
            data = self.eegh[:, tet]
            # freq of the input signal (eegh --> 5kHz --> freq=5000)
            freq = 5000
            # for low pass filtering of the signal before z-scoring (20-30Hz is good)
            low_pass_cut_off_freq = 30
            # min time gap between swr in seconds
            min_gap_between_events = 0.3


            # nyquist theorem --> need half the frequency
            sig_bandpass = butterworth_bandpass(input_data=data, freq_nyquist=freq/2, freq_lo_bound=freq_lo_bound,
                                                       freq_hi_bound=freq_hi_bound)

            # compute rectified signal
            sig_abs = np.abs(sig_bandpass)

            # if only peak position is supposed to be returned

            # low pass filter signal
            sig_lo_pass = butterworth_lowpass(input_data=sig_abs, freq_nyquist=freq/2,
                                              cut_off_freq=low_pass_cut_off_freq)

            combined.append(sig_lo_pass)

        combined = np.array(combined)
        sig_lo_pass = np.mean(combined, axis=0)
        # z-score
        sig_z_scored = zscore(sig_lo_pass)

        plt.plot(sig_z_scored[:100000])
        plt.show()

    def detect_swr(self, thr=4, plot_for_control=False):
        """
        detects swr in lfp and returns start, peak and end timings at params.time_bin_size resolution
        ripple frequency: 140-240 Hz

        @param thr: nr. std. above average to detect ripple event (usually: 4-6)
        @type thr: int
        @param plot_for_control: True to plot intermediate results
        @type plot_for_control: bool
        @return: start, end, peak of each swr in seconds
        @rtype: int, int, int
        """

        file_name = self.session_name + "_" + self.experiment_phase_id + "_swr_" + \
                    self.cell_type +"_tet_"+str(self.session_params.lfp_tetrodes)+ ".npy"
        # check if results exist already
        if not os.path.isfile(self.params.pre_proc_dir+"swr_periods/" + file_name):

            # check if results exist already --> if not

            # upper and lower bound in Hz for SWR
            freq_lo_bound = 140
            freq_hi_bound = 240

            # load data
            # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
            # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s

            # check if one tetrode or all tetrodes to use
            if self.session_params.lfp_tetrodes is None:
                print(" - DETECTING SWR USING ALL TETRODES ...\n")
                data = self.eegh[:, :]
            else:
                print(" - DETECTING SWR USING TETRODE(S) "+str(self.session_params.lfp_tetrodes) +" ...\n")
                data = self.eegh[:, self.session_params.lfp_tetrodes]
            # freq of the input signal (eegh --> 5kHz --> freq=5000)
            freq = 5000
            # for low pass filtering of the signal before z-scoring (20-30Hz is good)
            low_pass_cut_off_freq = 30
            # minimum gap in seconds between events. If two events have
            # a gap < min_gap_between_events --> events are joint and become one event
            min_gap_between_events = 0.1

            # if data is too large --> need to chunk it up

            if data.shape[0] > 10000000:
                start_times = np.zeros(0)
                peak_times = np.zeros(0)
                end_times = np.zeros(0)
                size_chunk = 10000000
                for nr_chunk in range(np.ceil(data.shape[0]/size_chunk).astype(int)):
                    chunk_data = data[nr_chunk*size_chunk:min(data.shape[0], (nr_chunk+1)*size_chunk)]

                    # compute offset in seconds for current chunk
                    offset_sec = nr_chunk * size_chunk * 1/freq

                    start_times_chunk, end_times_chunk, peak_times_chunk = self.detect_lfp_events(data=chunk_data,
                                                                                freq=freq, thr=thr,
                                                                                freq_lo_bound=freq_lo_bound,
                                                                                freq_hi_bound=freq_hi_bound,
                                                                                low_pass_cut_off_freq=low_pass_cut_off_freq,
                                                                                min_gap_between_events=min_gap_between_events,
                                                                                plot_for_control=plot_for_control)

                    # check if event was detected
                    if not start_times_chunk is None:
                        start_times = np.hstack((start_times, (start_times_chunk + offset_sec)))
                        end_times = np.hstack((end_times, (end_times_chunk + offset_sec)))
                        peak_times = np.hstack((peak_times, (peak_times_chunk + offset_sec)))

            else:
                # times in seconds
                start_times, end_times, peak_times = self.detect_lfp_events(data=data, freq=freq, thr=thr,
                                                                            freq_lo_bound=freq_lo_bound,
                                                                            freq_hi_bound=freq_hi_bound,
                                                                            low_pass_cut_off_freq=low_pass_cut_off_freq,
                                                                            min_gap_between_events=min_gap_between_events,
                                                                            plot_for_control=plot_for_control)

            result_dic = {
                "start_times": start_times,
                "end_times": end_times,
                "peak_times": peak_times
            }

            outfile = open(self.params.pre_proc_dir+"swr_periods/"+file_name, 'wb')
            pickle.dump(result_dic, outfile)
            outfile.close()

        # load results from file
        infile = open(self.params.pre_proc_dir+"swr_periods/" + file_name, 'rb')
        result_dic = pickle.load(infile)
        infile.close()

        start_times = result_dic["start_times"]
        end_times = result_dic["end_times"]
        peak_times = result_dic["peak_times"]

        print(" - " + str(start_times.shape[0]) + " SWRs FOUND\n")

        return start_times, end_times, peak_times

    @staticmethod
    def detect_lfp_events(data, freq, thr, freq_lo_bound, freq_hi_bound, low_pass_cut_off_freq,
                          min_gap_between_events, plot_for_control=False):
        """
        detects events in lfp and returns start, peak and end timings at params.time_bin_size resolution

        @param data: input data (either from one or many tetrodes)
        @type data: array [nxm]
        @param freq: sampling frequency of input data in Hz
        @type freq: int
        @param thr: nr. std. above average to detect ripple event
        @type thr: int
        @param freq_lo_bound: lower bound for frequency band in Hz
        @type freq_lo_bound: int
        @param freq_hi_bound: upper bound for frequency band in Hz
        @type freq_hi_bound: int
        @param low_pass_cut_off_freq: cut off frequency for envelope in Hz
        @type low_pass_cut_off_freq: int
        @param min_gap_between_events: minimum gap in seconds between events. If two events have
         a gap < min_gap_between_events --> events are joint and become one event
        @type min_gap_between_events: float
        @param plot_for_control: plot some examples to double check detection
        @type plot_for_control: bool
        @return: start_times, end_times, peak_times of each event in seconds --> are all set to None if no event was
        detected
        @rtype: array, array, array
        """

        # check if data from one or multiple tetrodes was provided
        if len(data.shape) == 1:
            # only one tetrode
            # nyquist theorem --> need half the frequency
            sig_bandpass = butterworth_bandpass(input_data=data, freq_nyquist=freq / 2, freq_lo_bound=freq_lo_bound,
                                                freq_hi_bound=freq_hi_bound)

            # compute rectified signal
            sig_abs = np.abs(sig_bandpass)

            # if only peak position is supposed to be returned

            # low pass filter signal
            sig_lo_pass = butterworth_lowpass(input_data=sig_abs, freq_nyquist=freq / 2,
                                              cut_off_freq=low_pass_cut_off_freq)
            # z-score
            sig_z_scored = zscore(sig_lo_pass)

        else:
            # multiple tetrodes
            combined_lo_pass = []
            # go trough all tetrodes
            for tet_data in data.T:
                # nyquist theorem --> need half the frequency
                sig_bandpass = butterworth_bandpass(input_data=tet_data, freq_nyquist=freq/2, freq_lo_bound=freq_lo_bound,
                                                           freq_hi_bound=freq_hi_bound)

                # compute rectified signal
                sig_abs = np.abs(sig_bandpass)

                # if only peak position is supposed to be returned

                # low pass filter signal
                sig_lo_pass = butterworth_lowpass(input_data=sig_abs, freq_nyquist=freq/2,
                                                  cut_off_freq=low_pass_cut_off_freq)

                combined_lo_pass.append(sig_lo_pass)

            combined_lo_pass = np.array(combined_lo_pass)
            avg_lo_pass = np.mean(combined_lo_pass, axis=0)

            # z-score
            sig_z_scored = zscore(avg_lo_pass)

        # find entries above the threshold
        bool_above_thresh = sig_z_scored > thr
        sig = bool_above_thresh.astype(int) * sig_z_scored

        # find event start / end
        diff = np.diff(sig)
        start = np.argwhere(diff > 0.8 * thr)
        end = np.argwhere(diff < -0.8 * thr)

        # check that first element is actually the start (not that event started before this chunk and we only
        # observe the end of the event)
        if end[0] < start[0]:
            # if first end is before first start --> need to delete first end
            print("  --> CURRENT CHUNK: FIRST END BEFORE FIRST START --> DELETED FIRST END ELEMENT ")
            end = end[1:]

        if end[-1] < start[-1]:
            # there is another start after the last end --> need to delete last start
            print("  --> CURRENT CHUNK: LAST START AFTER LAST END --> DELETED LAST START ELEMENT ")
            start = start[:-1]

        # join events if there are less than min_gap_between_events seconds apart --> this is then one event!
        # compute differences between start time of n+1th event with end time of nth --> if < gap --> delete both
        # entries
        gap = np.squeeze((start[1:] - end[:-1]) * 1 / freq)
        to_delete = np.argwhere(gap < min_gap_between_events)
        end = np.delete(end, to_delete)
        start = np.delete(start, to_delete + 1)

        # add 25ms to the beginning of event (many spikes occur in that window)
        pad_infront = np.round(0.025/(1/freq)).astype(int)
        start -= pad_infront
        # don't want negative values (in case event happens within the 50ms of the recording)
        start[start < 0] = 0

        # # add 20ms to the end of event
        # pad_end = np.round(0.02/(1/freq)).astype(int)
        # end += pad_end
        # # don't want to extend beyond the recording
        # end[end > sig.shape[0]] = sig.shape[0]

        # check length of events --> shouldn't be shorter than 95 ms or larger than 750 ms
        len_events = (end - start) * 1 / freq
        #
        # plt.hist(len_events, bins=50)
        # plt.show()
        # exit()

        to_delete_len = np.argwhere((0.75 < len_events) | (len_events < 0.05))

        start = np.delete(start, to_delete_len)
        end = np.delete(end, to_delete_len)

        peaks = []
        for s, e in zip(start,end):
            peaks.append(s+np.argmax(sig[s:e]))

        peaks = np.array(peaks)

        # check if there were any events detected --> if not: None
        if not peaks.size == 0:
            # get peak times in s
            time_bins = np.arange(data.shape[0]) * 1 / freq
            peak_times = time_bins[peaks]
            start_times = time_bins[start]
            end_times = time_bins[end]
        else:
            peak_times = None
            start_times = None
            end_times = None

        # plot some events with start, peak and end for control
        if plot_for_control:
            a = np.random.randint(0, start.shape[0], 5)
            # a = range(start.shape[0])
            for i in a:
                plt.plot(sig_z_scored, label="z-scored signal")
                plt.vlines(start[i], 0, 15, colors="r", label="start")
                plt.vlines(peaks[i], 0, 15, colors="y", label="peak")
                plt.vlines(end[i], 0, 15, colors="g", label="end")
                plt.xlim((start[i] - 5000),(end[i] + 5000))
                plt.ylabel("LFP FILTERED (140-240Hz) - Z-SCORED")
                plt.xlabel("TIME BINS / "+str(1/freq) + " s")
                plt.legend()
                plt.title("EVENT DETECTION, EVENT ID "+str(i))
                plt.show()

        return start_times, end_times, peak_times

    def phase_preference_per_cell_subset(self, angle_20k, cell_ids):

        # spike times at 20kHz
        spike_times = self.firing_times

        # get keys from dictionary and get correct order
        cell_names = []
        for key in spike_times.keys():
            cell_names.append(key[4:])
        cell_names = np.array(cell_names).astype(int)
        cell_names.sort()

        pref_angle = []

        for cell_id in cell_names[cell_ids]:
            all_cell_spikes = spike_times["cell" + str(cell_id)]
            # remove spikes that like outside array
            all_cell_spikes = all_cell_spikes[all_cell_spikes<angle_20k.shape[0]]
            # make array
            spk_ang = angle_20k[all_cell_spikes]
            pref_angle.append(np.angle(np.sum(np.exp(-1j * spk_ang))))

        return np.array(pref_angle)

    def phase_preference_analysis(self, oscillation="theta", tetrode=1, plot_for_control=False, plotting=True):

        # get lfp data
        # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
        # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s
        lfp = self.eegh[:, tetrode]

        # downsample to dt = 0.001 --> 1kHz --> take every 5th value
        lfp = lfp[::5]

        # Say you have an LFP signal LFP_Data and some spikes from a cell spk_t
        # First we extract the angle from the signal in a specific frequency band
        # Frequency Range to Extract, you can also select it AFTER running the wavelet on the entire frequency spectrum,
        # by using the variable frequency to select the desired ones
        if oscillation == "theta":
            Frq_Limits = [8, 12]
        elif oscillation == "slow_gamma":
            Frq_Limits = [20, 50]
        elif oscillation == "medium_gamma":
            Frq_Limits = [60, 90]
        else:
            raise Exception("Oscillation not defined!")
        # [8,12] Theta
        # [20,50] Slow Gamma
        # [60,90] Medium Gamma
        # LFP time bin duration in seconds
        # dt = 1/5e3
        dt=0.001
        # ‘morl’ wavelet
        wavelet = "cmor1.5-1.0" # 'cmor1.5-1.0'
        scales = np.arange(1,128)
        s2f = pywt.scale2frequency(wavelet, scales) / dt
        # This block is just to setup the wavelet analysis
        scales = scales[(s2f >= Frq_Limits[0]) * (s2f < Frq_Limits[1])]
        # scales = scales[np.logical_and(s2f >= Frq_Limits[0], s2f < Frq_Limits[1])]
        print(" - started wavelet decomposition ...")
        # Wavelet decomposition
        [cfs, frequencies] = pywt.cwt(data=lfp, scales=scales, wavelet=wavelet, sampling_period=dt, axis=0)
        print(" - done!")
        # This is the angle
        angl = np.angle(np.sum(cfs, axis=0))

        # plot for control
        if plot_for_control:
            plt.plot(lfp[:200])
            plt.xlabel("Time")
            plt.ylabel("LFP")
            plt.show()

            for i in range(frequencies.shape[0]):
                plt.plot(cfs[i, :200])
            plt.xlabel("Time")
            plt.ylabel("Coeff")
            plt.show()

            plt.plot(np.sum(cfs[:, :200], axis=0), label="coeff_sum")
            plt.plot(angl[:200]/np.max(angl[:200]), label="angle")
            plt.xlabel("Time")
            plt.ylabel("Angle (norm) / Coeff_sum (norm)")
            plt.legend()
            plt.show()

        # interpolate results to match 20k
        # --------------------------------------------------------------------------------------------------------------
        x_1k = np.arange(lfp.shape[0])*dt
        x_20k = np.arange(lfp.shape[0]*20)*1/20e3
        angle_20k = np.interp(x_20k, x_1k, angl, left=np.nan, right=np.nan)

        if plot_for_control:
            plt.plot(angle_20k[:4000])
            plt.ylabel("Angle")
            plt.xlabel("Time bin (20kHz)")
            plt.show()

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        stable = class_dic["stable_cell_ids"]
        dec = class_dic["decrease_cell_ids"]
        inc = class_dic["increase_cell_ids"]

        pref_angle_stable = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=stable)
        pref_angle_dec = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=dec)
        pref_angle_inc = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=inc)

        pref_angle_stable_deg = pref_angle_stable *180/np.pi
        pref_angle_dec_deg = pref_angle_dec * 180 / np.pi
        pref_angle_inc_deg = pref_angle_inc * 180 / np.pi

        if plotting:
            plt.hist(pref_angle_stable_deg, density=True, label="stable")
            plt.hist(pref_angle_dec_deg, density=True, label="dec")
            plt.hist(pref_angle_inc_deg, density=True, label="inc")
            plt.show()

        all_positive_angles_stable = np.copy(pref_angle_stable)
        all_positive_angles_stable[all_positive_angles_stable < 0] = 2*np.pi+all_positive_angles_stable[all_positive_angles_stable < 0]

        all_positive_angles_dec = np.copy(pref_angle_dec)
        all_positive_angles_dec[all_positive_angles_dec < 0] = 2 * np.pi + all_positive_angles_dec[
            all_positive_angles_dec < 0]

        all_positive_angles_inc = np.copy(pref_angle_inc)
        all_positive_angles_inc[all_positive_angles_inc < 0] = 2 * np.pi + all_positive_angles_inc[
            all_positive_angles_inc < 0]

        if plotting:

            bins_number = 10  # the [0, 360) interval will be subdivided into this
            # number of equal bins
            bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
            angles = all_positive_angles_stable
            n, _, _ = plt.hist(angles, bins, density=True)

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

            plt.clf()
            width = 2 * np.pi / bins_number
            ax = plt.subplot(1, 1, 1, projection='polar')
            bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
            for bar in bars:
                bar.set_alpha(0.5)
            ax.set_title("inc. cells")
            plt.show()

        else:
            return all_positive_angles_stable, all_positive_angles_dec, all_positive_angles_inc

    """#################################################################################################################
    #   Synchrony values (k-stats) / correlation value analysis
    #################################################################################################################"""

    def view_synchrony_values(self):
        """
        plots time course of synchrony (how many % of cells are active within one time bin)
        """
        syn = synchronous_activity(self.raster)
        plt.plot(syn)
        plt.show()
        plt.xlabel("TIME BINS")
        plt.ylabel("SYNCHRONY")
        plt.title("SYNCHRONY PER TIME BIN")

    def auto_correlation_synchrony_values(self):
        """
        computes auto-correlation of synchrony values

        @return:
        @rtype:
        """

        colors = cm.rainbow(np.linspace(0, 1, 5))

        for time_bin_size_counter, time_bin_size in enumerate([0.1, 0.07, 0.05, 0.02, 0.01]):

            # transform basis raster (10ms time bins) into appropriate time bin size:
            transformed_raster = down_sample_modify_raster(raster=self.raster, binning_method="temporal_spike",
                                                           time_bin_size=self.params.time_bin_size,
                                                           time_bin_size_after=time_bin_size)

            syn = synchronous_activity(transformed_raster)

            shift_array = np.arange(-30, 31)
            corr = np.zeros(len(shift_array))
            for i, shift in enumerate(shift_array):
                if shift >= 0:
                    corr[i], _ = pearsonr(syn[shift:], syn[:syn.shape[0] - shift])
                else:
                    shift = np.abs(shift)
                    corr[i], _ = pearsonr(syn[:syn.shape[0] - shift], syn[shift:])

            plt.plot(shift_array, corr, "-o", color=
            colors[time_bin_size_counter], label="time_bin: " + str(time_bin_size) + "s")
            plt.title("AUTO-CORRELATION OF SYNCHRONY VALUES - " + self.cell_type)
            plt.ylabel("PEARSON CORR. COEFF.")
            plt.xlabel("TIME BIN SHIFT - CENTERED ON " + self.cell_type)

        # get correlation value for shuffled data of last time bin size (usually: 10 ms)
        nr_shuffles = 500
        res_shuffles = np.zeros(nr_shuffles)
        for i in range(nr_shuffles):
            to_be_shuffled = syn
            shuffled_data = np.copy(to_be_shuffled)
            # shuffle synchrony values
            np.random.shuffle(shuffled_data)
            res_shuffles[i], _ = pearsonr(syn, shuffled_data)

        plt.hlines(np.mean(res_shuffles), -30, 30, color=colors[-1], linestyles="dashed", label="shuff. mean/3std")
        plt.hlines(3 * np.std(res_shuffles), -30, 30, color=colors[-1], linestyles="dotted")
        plt.legend()
        plt.show()

    def dynamic_co_firing(self, co_firing_window_size=2, sliding_window=True, sel_range=None):
        """

        @param co_firing_window_size: length of sliding window in seconds
        @type co_firing_window_size: int
        @param sliding_window: sliding window or discrete time windows
        @type sliding_window: bool
        @param sel_range: to only compute on subset of data
        @type sel_range: range object
        @return: correlation matrices
        @rtype: array
        """

        # compute how many time bins fit in one window
        time_bins_per_co_firing_matrix = int(co_firing_window_size / self.params.time_bin_size)

        if sel_range is None:
            raster = self.raster
        else:
            raster = self.raster[:, sel_range]

        if sliding_window:

            correlation_matrices = []

            print(" - COMPUTING SUBSEQUENT CORRELATION MATRICES ....\n")

            for entry in range(int(raster.shape[1] - time_bins_per_co_firing_matrix + 1)):
                # print("PROGRESS: "+ str(entry +1) + "/" + str(int(x.shape[1] - time_bins_per_co_firing_matrix)+1)
                #       +" FRAMES")
                correlation_matrices.append(np.corrcoef(raster[:, entry:(entry + time_bins_per_co_firing_matrix)]))

            correlation_matrices = np.array(correlation_matrices)

        else:

            correlation_matrices = []
            for entry in range(int(raster.shape[1] / time_bins_per_co_firing_matrix)):
                co_fir_numpy = np.corrcoef(raster[:, entry * time_bins_per_co_firing_matrix:
                (entry+1)*time_bins_per_co_firing_matrix])
                correlation_matrices.append(co_fir_numpy)

            correlation_matrices = np.array(correlation_matrices)

        # if one vector is constant (e.g. all zeros) --> pearsonr return np.nan
        # set all nans to zero

        correlation_matrices = np.nan_to_num(correlation_matrices, posinf=0, neginf=0)

        print("  ... DONE\n")

        return correlation_matrices

    """#################################################################################################################
    #   Population vector methods and others
    #################################################################################################################"""

    def population_vector_distance(self, distance_measure):
        # --------------------------------------------------------------------------------------------------------------
        # calculates and plots distance between subsequent population vectors (one time bin --> one vector)
        #
        # args:     - distance_measure, string: "cos", "euclidean", "L1"
        # --------------------------------------------------------------------------------------------------------------

        dis, rel = pop_vec_dist(self.raster, distance_measure)
        plt.plot(rel)
        plt.xlabel("TIME BINS")
        plt.ylabel("DISTANCE ("+distance_measure+")")
        plt.title("DISTANCE BETWEEN SUBSEQUENT POPULATION VECTORS")
        plt.show()

    def entropy_measure(self):
        # --------------------------------------------------------------------------------------------------------------
        # calculates entropy of population vectors and plots its time course
        # --------------------------------------------------------------------------------------------------------------

        ent = calc_pop_vector_entropy(self.raster)
        plt.plot(ent)
        plt.ylabel("ENTROPY")
        plt.xlabel("TIME BINS")
        plt.title("ENTROPY OF SUBSEQUENT POPULATION VECTORS")
        plt.show()

    def receptive_field(self, input):
        # --------------------------------------------------------------------------------------------------------------
        # user provided input as stimuli to find "receptive" field for each neuron in the population
        # assign correct data as input/output
        #
        # args:     - input, array with "stimuli" label
        # --------------------------------------------------------------------------------------------------------------

        y = self.raster

        all_receptive_fields = np.zeros((input.shape[0], y.shape[0]))
        # go through all neurons
        for neuron_id, neuron in enumerate(range(y.shape[0])):
            receptive_field_matrix = np.zeros((input.shape[0], input.shape[1]))
            # go through all time steps
            for i, (pop_vec_out, pop_vec_in) in enumerate(zip(y.T, input.T)):
                receptive_field_matrix[:, i] = pop_vec_out[neuron] * pop_vec_in
                print(i)
            all_receptive_fields[:, neuron_id] = np.mean(receptive_field_matrix, axis=1)

        plt.imshow(all_receptive_fields, interpolation='nearest', aspect='auto', cmap="jet")
        plt.title("RECEPTIVE FIELDS")
        plt.xlabel("NEURONS")
        plt.ylabel("RECEPTIVE FIELD")
        a = plt.colorbar()
        a.set_label("WEIGHT")
        plt.show()

    """#################################################################################################################
    #   Clustering / discrete system states analysis
    #################################################################################################################"""

    # poisson hmm
    # ------------------------------------------------------------------------------------------------------------------

    def cross_val_poisson_hmm(self, cl_ar=np.arange(1, 40, 5), sel_range=None):
        # --------------------------------------------------------------------------------------------------------------
        # cross validation of poisson hmm fits to data
        #
        # args:     - cl_ar, range object: #clusters to fit to data
        # --------------------------------------------------------------------------------------------------------------

        print(" - CROSS-VALIDATING POISSON HMM --> #modes ...\n")

        if sel_range is None:
            X = self.raster
        else:
            X = self.raster[:,sel_range]

        nr_cores = 12

        test_range_per_fold = None

        if self.params.cross_val_splits == "custom_splits":

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
            bin_num = X.shape[1]
            bins = np.arange(bin_num + 1)

            # length of one chunk
            n_chunks = int(bin_num / nr_chunks)
            test_range_per_fold = []
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
                test_range_per_fold.append(np.array(test_range))


        folder_name = self.params.session_name +"_"+self.experiment_phase_id+"_"+self.cell_type

        new_ml = MlMethodsOnePopulation(params=self.params)
        new_ml.parallelize_cross_val_model(nr_cluster_array=cl_ar, nr_cores=nr_cores, model_type="POISSON_HMM",
                                           raster_data=X, folder_name=folder_name, splits=test_range_per_fold)
        new_ml.cross_val_view_results(folder_name=folder_name)

    def view_cross_val_results(self, range_to_plot=None):
        # --------------------------------------------------------------------------------------------------------------
        # views cross validation results
        #
        # args:     - model_type, string: which type of model ("POISSON_HMM")
        #           - custom_splits, bool: whether custom splits were used for cross validation
        # --------------------------------------------------------------------------------------------------------------
        folder_name = self.params.session_name + "_" + str(
            int(self.params.experiment_phase_id[0])) + "_" + self.cell_type
        new_ml = MlMethodsOnePopulation(params=self.params)
        new_ml.cross_val_view_results(folder_name=folder_name, range_to_plot=range_to_plot)

    def fit_poisson_hmm(self, nr_modes):
        # --------------------------------------------------------------------------------------------------------------
        # fits poisson hmm to data
        #
        # args:     - nr_modes, int: #clusters to fit to data
        #           - file_identifier, string: string that is added at the end of file for identification
        # --------------------------------------------------------------------------------------------------------------

        print(" - FITTING POISSON HMM WITH "+str(nr_modes)+" MODES ...")

        X = self.raster

        model = PoissonHMM(n_components=nr_modes)
        model.fit(X.T)

        file_name = self.params.session_name + "_" + str(
            int(self.params.experiment_phase_id[0])) + "_" + self.cell_type +"_"+str(nr_modes)+"_modes"

        with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "wb") as file: pickle.dump(model, file)

    def decode_poisson_hmm(self, nr_modes=None, file_name=None, plotting=False):
        # --------------------------------------------------------------------------------------------------------------
        # loads model from file and decodes data
        #
        # args:     - nr_modes, int: #clusters to fit to data --> used to identify file that fits the data
        #           - file_name, string:    is used if model from a different experiment phase is supposed to be used
        #                                   (e.g. model from awake is supposed to be fit to sleep data)
        # --------------------------------------------------------------------------------------------------------------

        X = self.raster

        if (nr_modes is None) & (file_name is None):
            raise Exception("PROVIDE NR. MODES OR FILE NAME")

        if file_name is None:
            file_name = self.params.session_name + "_" + str(
                int(self.params.experiment_phase_id[0])) + "_" + self.cell_type +"_"+str(nr_modes)+"_modes"
        else:
            file_name =file_name

        with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "rb") as file:
            model = pickle.load(file)

        nr_modes_ = model.means_.shape[0]

        # compute most likely sequence
        sequence = model.predict(X.T)
        if plotting:
            plot_pop_clusters(map=X, labels=sequence, params=self.params,
                              nr_clusters=nr_modes, sel_range=range(100,200))

        return sequence, nr_modes_

    def load_poisson_hmm(self, nr_modes=None, file_name=None):
        # --------------------------------------------------------------------------------------------------------------
        # loads model from file and returns model
        #
        # args:     - nr_modes, int: #clusters to fit to data --> used to identify file that fits the data
        #           - file_name, string:    is used if model from a different experiment phase is supposed to be used
        #                                   (e.g. model from awake is supposed to be fit to sleep data)
        # --------------------------------------------------------------------------------------------------------------

        if (nr_modes is None) & (file_name is None):
            raise Exception("PROVIDE NR. MODES OR FILE NAME")

        if file_name is None:
            file_name = self.params.session_name + "_" + str(
                int(self.params.experiment_phase_id[0])) + "_" + self.cell_type +"_"+str(nr_modes)+"_modes"
        else:
            file_name =file_name

        with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "rb") as file:
            model = pickle.load(file)

        return model

    def plot_poisson_hmm_info(self, nr_modes):
        model = self.load_poisson_hmm(nr_modes=nr_modes)
        plt.imshow(model.means_.T, interpolation='nearest', aspect='auto', cmap="jet")
        plt.ylabel("CELL ID")
        plt.xlabel("MODE ID")
        a = plt.colorbar()
        a.set_label(r'$\lambda$'+ " (#SPIKES/100ms WINDOW)")
        plt.title("STATE NEURAL PATTERNS")
        plt.show()

    def evaluate_poisson_hmm(self, nr_modes):
        # --------------------------------------------------------------------------------------------------------------
        # fits poisson hmm to data and evaluates the goodness of the model by comparing basic statistics (avg. firing
        # rate, correlation values, k-statistics) between real data and data sampled from the model
        #
        # args:     - nr_modes, int: #clusters to fit to data
        #           - load_from_file, bool: whether to load model from file or to fit model again
        # --------------------------------------------------------------------------------------------------------------

        print(" - EVALUATING POISSON HMM FIT (BASIC STATISTICS) ...")

        X = self.raster
        nr_time_bins = X.shape[1]
        # X = X[:, :1000]

        file_name = self.params.session_name + "_" + str(
            int(self.params.experiment_phase_id[0])) + "_" + self.cell_type +"_"+str(nr_modes)+"_modes"

        # check if model file exists already --> otherwise fit model again
        if os.path.isfile(self.params.pre_proc_dir+"phmm/" + file_name + ".pkl"):
            print("- LOADING PHMM MODEL FROM FILE\n")
            with open(self.params.pre_proc_dir+"phmm/" + file_name + ".pkl", "rb") as file:
                model = pickle.load(file)
        else:
            print("- PHMM MODEL FILE NOT FOUND --> FITTING PHMM TO DATA\n")
            model = PoissonHMM(n_components=nr_modes)
            model.fit(X.T)

        samples, sequence = model.sample(nr_time_bins)
        samples = samples.T

        evaluate_clustering_fit(real_data=X, samples=samples, binning="TEMPORAL_SPIKE",
                                   time_bin_size=0.1, plotting=True)

    def poisson_hmm_fit_discreteness(self, nr_modes=80, analysis_method="LIN_SEP"):
        # --------------------------------------------------------------------------------------------------------------
        # checks how discrete identified modes are
        #
        # args:     - nr_modes, int: defines which number of modes were fit to data (for file identification)
        #           - analysis_method, string:  - "CORR" --> correlations between lambda vectors
        #                                       - "LIN_SEP" --> check linear separability of population vectors that
        #                                                       were assigned to different modes
        #                                       - "SAMPLING" --> samples from lambda vector using Poisson emissions and
        #                                                        plots samples color coded using MDS
        # --------------------------------------------------------------------------------------------------------------

        print(" - ASSESSING DISCRETENESS OF MODES ...")

        # load data
        X = self.raster

        with open(self.params.pre_proc_dir+"ML/poisson_hmm_" +
                  str(nr_modes) + "_modes_" + self.cell_type + ".pkl", "rb") as file:
            model = pickle.load(file)

        seq = model.predict(X.T)

        lambda_per_mode = model.means_.T

        nr_modes = lambda_per_mode.shape[1]

        if analysis_method == "CORR":
            plt.imshow(lambda_per_mode, interpolation='nearest', aspect='auto', cmap="jet")
            a = plt.colorbar()
            a.set_label("LAMBDA (SPIKES PER TIME BIN)")
            plt.ylabel("CELL ID")
            plt.xlabel("MODE ID")
            plt.title("LAMBDA VECTOR PER MODE")
            plt.show()

            # control --> shuffle each lambda vector
            nr_shuffles = 10000
            correlation_shuffle = []
            for i in range(nr_shuffles):
                shuffled_data = independent_shuffle(lambda_per_mode)
                correlation_shuffle.append(upper_tri_without_diag(np.corrcoef(shuffled_data.T)))

            corr_shuffle = np.array(correlation_shuffle).flatten()
            corr_shuffle_95_perc = np.percentile(corr_shuffle, 95)

            plt.hist(corr_shuffle, density=True, label="SHUFFLE")

            corr_data = np.corrcoef(lambda_per_mode.T)
            # get off diagonal elements
            corr_data_vals = upper_tri_without_diag(corr_data)
            plt.hist(corr_data_vals, density=True, alpha=0.5, label="DATA")
            plt.xlabel("PEARSON CORRELATION")
            plt.ylabel("COUNTS")
            plt.title("CORR. BETWEEN LAMBDA VECTORS FOR EACH MODE")
            y_max = plt.ylim()
            plt.vlines(corr_shuffle_95_perc, 0, 5, colors="r", label="95 percentile shuffle")
            plt.legend()
            plt.show()

            plt.imshow(corr_data, interpolation="nearest", aspect="auto", cmap="jet")
            a = plt.colorbar()
            a.set_label("PEARSON CORRELATION")
            plt.title("CORRELATION BETWEEN LAMBDA VECTORS PER MODE")
            plt.xlabel("MODE ID")
            plt.ylabel("MODE ID")
            plt.show()

        elif analysis_method == "LIN_SEP":
            # find pop vectors with mode id and try to fit svm

            D = np.zeros((nr_modes, nr_modes))
            for template_mode_id in np.arange(nr_modes):
                others = np.delete(np.arange(nr_modes), template_mode_id)
                for compare_mode_id in others:
                    mode_id_1 = template_mode_id
                    mode_id_2 = compare_mode_id

                    mode_1 = X[:, seq == mode_id_1]
                    mode_2 = X[:, seq == mode_id_2]
                    mode_id_1_label = mode_id_1 * np.ones(mode_1.shape[1])
                    mode_id_2_label = mode_id_2 * np.ones(mode_2.shape[1])

                    # plt.imshow(mode_1, interpolation='nearest', aspect='auto', cmap="jet")
                    # plt.show()
                    # plt.imshow(mode_2, interpolation='nearest', aspect='auto', cmap="jet")
                    # plt.show()

                    nr_mode_0 = mode_1.shape[1]

                    data = np.hstack((mode_1, mode_2))
                    labels = np.hstack((mode_id_1_label, mode_id_2_label))

                    D[template_mode_id, compare_mode_id] = MlMethodsOnePopulation(params=self.params).linear_separability(input_data=data, input_labels=labels)

            D[D == 0] = np.nan
            plt.imshow(D, interpolation='nearest', aspect='auto', cmap="jet")
            a = plt.colorbar()
            a.set_label("ACCURACY (SVM)")
            plt.title("LINEAR SEPARABILTY OF POP. VEC PER MODE")
            plt.xlabel("MODE ID")
            plt.ylabel("MODE ID")
            plt.show()

            return D

        elif analysis_method == "SAMPLING":

            lambda_per_mode = model.means_.T
            mode_id = np.arange(0, lambda_per_mode.shape[1])
            poisson_firing = np.empty((lambda_per_mode.shape[0],0))
            nr_samples = 50
            # sample from modes
            for i in range(nr_samples):
                poisson_firing = np.hstack((poisson_firing, np.random.poisson(lambda_per_mode)))

            rep_mode_id = np.tile(mode_id, nr_samples)

            reduced_dim = MlMethodsOnePopulation(params=self.params).reduce_dimension(input_data=poisson_firing)

            plt.scatter(reduced_dim[:, 0], reduced_dim[:, 1], c="gray")
            plt.scatter(reduced_dim[rep_mode_id == 0, 0], reduced_dim[rep_mode_id == 0, 1], c="r", label="MODE 0")
            plt.scatter(reduced_dim[rep_mode_id == 18, 0], reduced_dim[rep_mode_id == 18, 1], c="g", label="MODE 18")
            plt.scatter(reduced_dim[rep_mode_id == 50, 0], reduced_dim[rep_mode_id == 50, 1], c="b", label="MODE 50")
            plt.legend()
            plt.title("MODE SAMPLE SIMILARITY")
            plt.show()

    def poisson_hmm_mode_progression(self, nr_modes, window_size=10):
        """
        checks how discrete identified modes are

        @param nr_modes: #modes of model (to find according file containing the model)
        @type nr_modes: int
        @param window_size: length of window size in seconds to compute frequency of modes
        @type window_size: float
        """
        print(" - ASSESSING PROGRESSION OF MODES ...")

        file_name = self.params.session_name + "_" + str(
            int(self.params.experiment_phase_id[0])) + "_" + self.cell_type +"_"+str(nr_modes)+"_modes"

        X = self.raster
        with open(self.params.pre_proc_dir+"phmm/" + file_name +".pkl", "rb") as file: model = pickle.load(file)
        seq = model.predict(X.T)
        nr_modes = model.means_.shape[0]

        print(X.shape)
        log_prob, post = model.score_samples(X.T)
        print(post.shape)
        plt.imshow(post.T, interpolation='nearest', aspect='auto')
        a = plt.colorbar()
        a.set_label("POST. PROBABILITY")
        plt.ylabel("MODE ID")
        plt.xlabel("TIME BINS")
        plt.title("MODE PROBABILITIES")
        plt.show()
        print(np.sum(post.T, axis=0))


        bins_per_window = int(window_size / self.params.time_bin_size)

        nr_windows = int(X.shape[1]/bins_per_window)

        mode_prob = np.zeros((nr_modes, nr_windows))
        for i in range(nr_windows):
            a,b = np.unique(seq[i*bins_per_window:(i+1)*bins_per_window], return_counts=True)
            mode_prob[a, i] = b/bins_per_window

        max_mode_prob = np.max(mode_prob, axis=1)
        mode_prob_norm = mode_prob / mode_prob.max(axis=1)[:, None]

        plt.imshow(mode_prob_norm, interpolation='nearest', aspect='auto')
        plt.title("MODE FREQUENCY - NORMALIZED - "+str(window_size)+"s WINDOW")
        plt.ylabel("MODE ID")
        plt.xlabel("WINDOW ID")
        a = plt.colorbar()
        a.set_label("MODE FREQUENCY - NORMALIZED")

        # compute weighted average
        windows = np.tile(np.arange(nr_windows).T, nr_modes).reshape((nr_modes, nr_windows))
        weighted_av = np.average(windows, axis=1, weights=mode_prob)

        a = (windows - weighted_av[:, None]) ** 2

        weighted_std = np.sqrt(np.average(a, axis=1, weights=mode_prob))

        plt.scatter(weighted_av, np.arange(nr_modes), c="r", s=1, label="WEIGHTED AVERAGE")
        plt.legend()
        plt.show()

        plt.scatter(weighted_av, weighted_std)
        plt.title("MODE OCCURENCE: WEIGHTED AVERAGE & WEIGHTED STD")
        plt.ylabel("WEIGHTED STD")
        plt.xlabel("WEIGHTED AVERAGE")
        plt.show()

    def poisson_hmm_transitions(self, nr_modes=80):
        # --------------------------------------------------------------------------------------------------------------
        # views and analyzes transition matrix
        #
        # args:   - nr_modes, int: defines which number of modes were fit to data (for file identification)
        #
        # --------------------------------------------------------------------------------------------------------------
        print(" - ASSESSING TRANSITIONS OF MODES ...")

        with open(self.params.pre_proc_dir+"ML/poisson_hmm_" +
                  str(nr_modes) + "_modes_" + self.cell_type + ".pkl", "rb") as file:
            model = pickle.load(file)

        trans_mat = model.transmat_

        plt.imshow(trans_mat, interpolation='nearest', aspect='auto')

        plt.ylabel("MODE ID")
        plt.xlabel("MODE ID")
        plt.title("TRANSITION PROBABILITY")
        a = plt.colorbar()
        a.set_label("TRANSITION PROBABILITY")
        plt.show()

        np.fill_diagonal(trans_mat, np.nan)

        plt.imshow(trans_mat, interpolation='nearest', aspect='auto')

        plt.ylabel("MODE ID")
        plt.xlabel("MODE ID")
        plt.title("TRANSITION PROBABILITY WO DIAGONAL")
        a = plt.colorbar()
        a.set_label("TRANSITION PROBABILITY")
        plt.show()

    def poisson_hmm_mode_drift(self, file_name):
        # --------------------------------------------------------------------------------------------------------------
        # checks assemblies assigned to different modes drift
        # --------------------------------------------------------------------------------------------------------------

        print(" - ASSESSING DRIFT OF ASSEMBLIES WITHIN MODES ...")

        X = self.raster
        with open(self.params.pre_proc_dir+"ML/" + file_name , "rb") as file: model = pickle.load(file)
        seq = model.predict(X.T)
        nr_modes = model.means_.shape[0]
        change = []

        for mode_id in range(nr_modes):
            X_mode = X[:, seq == mode_id]
            X_mode = (X_mode - np.min(X_mode, axis=1, keepdims=True)) / \
                (np.max(X_mode, axis=1, keepdims=True) - np.min(X_mode, axis=1, keepdims=True))
            X_mode = np.nan_to_num(X_mode)

            # plt.imshow(X_mode, interpolation='nearest', aspect='auto')
            # plt.colorbar()
            # plt.show()
            res = []
            for cell in X_mode:
                try:
                    m, b = np.polyfit(range(X_mode.shape[1]), cell, 1)
                    res.append(m)
                except:
                    res.append(0)

            res = np.array(res)
            mean_change_cell = np.mean(np.abs(res))
            change.append(mean_change_cell)

        change = np.array(change)
        a = np.flip(np.argsort(change))
        print(a)
        plt.plot(change)
        plt.show()

        X_mode = X[:, seq == 90]

        plt.imshow(X_mode, interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.show()

        exit()

        for mode_id in range(nr_modes):
            X_mode = X[:, seq==mode_id]



            exit()


            X_temp = X_mode[:, :int(X_mode.shape[0]*0.1)]
            X_compare = X_mode[:, int(X_mode.shape[0]*0.1):]

            res = np.zeros(X_compare.shape[1])
            for i, x in enumerate(X_compare.T):
                all_comp = []
                for temp in X_temp.T:
                    all_comp.append(1-distance.euclidean(x, temp))
                res[i] = np.max(np.array(all_comp))
            try:
                m, b = np.polyfit(range(X_compare.shape[1]), res, 1)
                change.append(m)
            except:
                continue


        plt.plot(change)
        plt.show()
        exit()




        plt.scatter(range(X_mode.shape[1]), X_mode[102,:])
        # plt.plot(X_mode[102, :])
        plt.show()

    # tree hmm
    # ------------------------------------------------------------------------------------------------------------------

    def evaluate_tree_hmm_fit(self, file_name, time_bin_size=0.02):
        # --------------------------------------------------------------------------------------------------------------
        # evaluates fit by comparing basic statistics
        # --------------------------------------------------------------------------------------------------------------

        # treeHMM uses binary data
        binning = "temporal_binary"

        data_base = pickle.load(open(file_name, "rb"), encoding='latin1')
        samples = data_base["samples"].T

        real_data = down_sample_modify_raster(raster=self.raster, time_bin_size=self.params.time_bin_size,
                                                   binning_method=binning, time_bin_size_after=time_bin_size)

        evaluate_clustering_fit(real_data=real_data, samples=samples, binning=binning, time_bin_size=time_bin_size,
                                     plotting=True)

    def evaluate_multiple_tree_hmm_fits(self, dir_name, time_bin_size=0.02):
        # --------------------------------------------------------------------------------------------------------------
        # evaluates fit by comparing basic statistics
        # --------------------------------------------------------------------------------------------------------------

        # treeHMM uses binary data
        binning = "temporal_binary"

        # compute empirical statistics
        real_data = down_sample_modify_raster(raster=self.raster, binning_method=binning,
                                              time_bin_size=self.params.time_bin_size,
                                              time_bin_size_after=time_bin_size)

        mean_wc = []
        mean_diff = []
        mean_corr = []
        mean_corr_spear = []
        k_ks = []
        k_mwu = []
        corr_wc = []
        corr_corr_spear = []
        corr_corr = []
        nr_mode = []
        for file in os.listdir(dir_name):
            data_base = pickle.load(open(dir_name+file, "rb"), encoding='latin1')
            samples = data_base["samples"].T
            mean_dic, corr_dic, k_dic = evaluate_clustering_fit(real_data=real_data, samples=samples,
                                                                     binning=binning,
                                                                     time_bin_size=time_bin_size)
            nr_mode.append(int(file.split("modes")[0]))
            k_ks.append(k_dic["ks"][1])
            k_mwu.append(k_dic["mwu"][1])
            corr_wc.append(corr_dic["wc"][1])
            corr_corr.append(corr_dic["corr_triangles"][0])
            corr_corr_spear.append(corr_dic["corr_triangles_spear"][0])
            mean_wc.append(mean_dic["wc"][1])
            mean_diff.append(mean_dic["norm_diff"])
            mean_corr.append(mean_dic["corr"][0])
            mean_corr_spear.append(mean_dic["corr_spear"][0])
        sort_ind = np.argsort(np.array(nr_mode))
        nr_mode = np.array(nr_mode)[sort_ind]
        mean_wc = np.array(mean_wc)[sort_ind]
        # mean_diff = np.array(mean_diff)[sort_ind]
        mean_corr = np.array(mean_corr)[sort_ind]
        mean_corr_spear = np.array(mean_corr_spear)[sort_ind]
        k_ks = np.array(k_ks)[sort_ind]
        k_mwu = np.array(k_mwu)[sort_ind]
        corr_wc = np.array(corr_wc)[sort_ind]
        # corr_diff = np.array(corr_diff)[sort_ind]
        corr_corr = np.array(corr_corr)[sort_ind]
        corr_corr_spear = np.array(corr_corr_spear)[sort_ind]

        plt.subplot(3, 1, 1)
        plt.plot(nr_mode, mean_wc,"-o", color = "blue", label="mean_wc")
        # plt.plot(nr_mode, corr_wc, "-o", color="red", label="corr_wc")
        plt.plot(nr_mode, k_ks,"-o", color="lightgreen", label="k_ks")
        plt.plot(nr_mode, k_mwu,"-o", color="green", label="k_mwu")
        # plt.plot(nr_mode, corr_wc,"-o", color="yellow", label="corr_wc" )
        plt.grid(True, c="grey")
        plt.legend()
        plt.ylabel("P-VALUE")
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.subplot(3, 1, 2)
        plt.plot(nr_mode, mean_corr,"-o", color="blue", label="MEAN FIRING - PEAR.")
        plt.plot(nr_mode, mean_corr_spear, "-o", color="skyblue", label="MEAN FIRING - SPEAR.")
        plt.ylabel("CORRELATION")
        plt.grid(True, c="grey")
        plt.legend()
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.subplot(3, 1, 3)
        # plt.plot(nr_mode, corr_diff,"-o", color="red", label="CORR.: NORM(DIFF")
        plt.plot(nr_mode, corr_corr,"-o", color="red", label="CORR. VALUES - PEAR.")
        plt.plot(nr_mode, corr_corr_spear, "-o", color="salmon", label="CORR. VALUES - SPEAR.")
        # plt.plot(nr_mode, mean_corr[0],"-o", color="skyblue", label="MEAN.: SPEARMAN R")
        plt.ylabel("CORRELATION")
        plt.grid(True, c="grey")
        plt.legend()
        plt.xlabel("#MODES")
        plt.show()

    @staticmethod
    def tree_hmm_high_firing_cells(file_name, perc_threshold=0.5, prob_threshold=0.1, plotting=False):
        """ ------------------------------------------------------------------------------------------------------------
        # derives high firing cells for each mode
        #
        # args: - file_name, string: input file name (output of TreeHMM)
        #       - perc_threshold, float: threshold for cells with high firing probability (threshold * max prob value)
        #       - prob_threshold, float: threshold to filter out very low cell firing probabilities
        # -----------------------------------------------------------------------------------------------------------"""

        data_base = pickle.load(open(file_name, "rb"), encoding='latin1')

        ind_high_firing_cells = []
        all_modes_m = []
        # get high firing cells from mean firing prob from each mode using "params"
        for mode_id in range(len(data_base["params"])):
            # get mean firing rates from modes
            mean_fir_rates = data_base["params"][mode_id]["m"]
            all_modes_m.append(mean_fir_rates)
            corr_struct = data_base["params"][mode_id]["J"]

            ind_high_firing_cells.append(np.argwhere((mean_fir_rates > perc_threshold * np.max(mean_fir_rates)) &
                                                     (mean_fir_rates > prob_threshold))[:,0])

        all_modes_m = np.squeeze(np.array(all_modes_m), axis=2).T
        if plotting:
            plt.imshow(all_modes_m, interpolation='nearest', aspect='auto')
            a=plt.colorbar()
            a.set_label("FIRING PROBABILITY")
            plt.xlabel("MODE ID")
            plt.ylabel("CELL ID")
            plt.title("CELL FIRING PROB. PER MODE")
            plt.show()
        return ind_high_firing_cells, all_modes_m

    """#################################################################################################################
    #   Dimensionality reduction
    #################################################################################################################"""

    def view_dimensionality_reduction(self):
        # --------------------------------------------------------------------------------------------------------------
        # reduces dimensionality of data and plots result
        # --------------------------------------------------------------------------------------------------------------

        print(" - REDUCING DIMENSIONALITY OF DATA ...")

        raster = self.raster[:500]

        new_ml = MlMethodsOnePopulation(params=self.params)
        new_ml.plot_reduced_dimension(input_data=raster)


"""#####################################################################################################################
#   SLEEP CLASS
#####################################################################################################################"""


class Sleep(BaseMethods):
    """Base class for sleep analysis"""

    def __init__(self, data_dic, cell_type, params, session_params, experiment_phase=None):
        # --------------------------------------------------------------------------------------------------------------
        # args: - data_dic, dictionary with standard data
        #       - cell_type, string: cell type to do analysis with
        #       - params, python class: contains all parameters
        # --------------------------------------------------------------------------------------------------------------

        # get attributes from parent class
        BaseMethods.__init__(self, data_dic, cell_type, params, session_params, experiment_phase)

        # import analysis parameters that are specific for the current session
        # --------------------------------------------------------------------------------------------------------------
        # rem/nrem phases with speeds above this threshold are discarded
        self.session_name = self.session_params.session_name

        # compression factor:
        #
        # compression factor used for sleep decoding --> e.g when we use constant #spike bins with 12 spikes
        # we need to check how many spikes we have in e.g. 100ms windows if this was used for awake encoding
        # if we have a mean of 30 spikes for awake --> compression factor = 12/30 --> 0.4
        # is used to scale awake activity to fit sleep activity
        # --------------------------------------------------------------------------------------------------------------

        # default models for behavioral data
        # --------------------------------------------------------------------------------------------------------------

        # get data dictionary
        # check if list or dictionary is passed:
        if isinstance(data_dic, list):
            data_dic = data_dic[0]
        else:
            data_dic = data_dic
        # get time stamps for sleep type
        # time stamps for sleep (.nrem, .rem, .sw) are at 20 kHz --> just like .res data
        self.time_stamps = data_dic["timestamps"]

        # TODO: do pre processing!
        # should pre-process sleep to exclude moving periods --> is especially important for long sleep
        # PreProcessSleep(firing_times=self.firing_times, params=self.params, time_stamps=self.time_stamps,
        #                 whl=self.whl).speed_filter_raw_data(eegh=self.eegh)

        # initialize all as None, so that we do not have to go through the pre-processing everytime we generate
        # an instance of the class --> if this data is needed, call self.compute_raster_speed_loc
        # --------------------------------------------------------------------------------------------------------------
        self.raster = np.empty(0)
        self.speed = np.empty(0)
        self.loc = np.empty(0)

    # basic computations
    # ------------------------------------------------------------------------------------------------------------------
    def compute_raster_speed_loc(self):
        # check first if data exists already
        if self.raster.shape[0] == 0:
            pre_prop_sleep = PreProcessSleep(firing_times=self.firing_times, params=self.params, time_stamps=None,
                                             last_spike=self.last_spike, whl=self.whl, spatial_factor=self.spatial_factor)
            raster = pre_prop_sleep.get_raster()
            speed = pre_prop_sleep.get_speed()
            loc = pre_prop_sleep.get_loc()

            # trim to same length
            l = np.min([raster.shape[1], speed.shape[0], loc.shape[0]])

            self.raster = raster[:,:l]
            self.speed = speed[:l]
            self.loc = loc[:,:l]

    def compute_speed(self):

        # check if speed exists already
        if self.speed.shape[0] == 0:

            pre_prop_sleep = PreProcessSleep(firing_times=self.firing_times, params=self.params, time_stamps=None,
                                             last_spike=self.last_spike, whl=self.whl)
            self.speed = pre_prop_sleep.get_speed()

    def up_sample_binarize_hse(self, time_bin_size=0.02, plotting=False, file_name=None):
        # --------------------------------------------------------------------------------------------------------------
        # find HSE time bins and upsamples (20ms time bins by default) and binarizes the data
        #
        # args:     - time_bin_size, float: new time bin size in seconds
        # --------------------------------------------------------------------------------------------------------------

        # compute pre processed data
        self.compute_raster_speed_loc()

        x = self.raster

        # find high synchrony events
        ind_hse = np.array(find_hse(x=x)).flatten()

        binary_data = PreProcessSleep(self.firing_times, self.params, self.time_stamps,
                                     self.last_spike).temporal_binning_binary(time_bin_size=time_bin_size)

        ratio = int(self.params.time_bin_size / time_bin_size)

        # convert indices of hse with self.params.time_bin_size (e.g. 0.1s) into new time bin size (e.g. 0.02s)

        ind_upsampled = []
        for ind in ind_hse:
            ind_upsampled.extend(np.arange((ratio*ind), (ratio*ind+ratio)))
        ind_upsampled = np.array(ind_upsampled)

        if plotting:
            x_plot_lim = 600

            plt.subplot(2,1,1)
            plt.imshow(x, interpolation='nearest', aspect='auto')
            plt.colorbar()
            mask = np.zeros(x.shape)
            mask[:, ind_hse] = 1

            plt.imshow(mask, interpolation='nearest', aspect='auto', alpha=0.1)
            plt.xlim((0,x_plot_lim))

            plt.subplot(2,1,2)
            plt.imshow(binary_data, interpolation='nearest', aspect='auto')
            plt.colorbar()
            mask = np.zeros(binary_data.shape)
            mask[:, ind_upsampled] = 1

            plt.imshow(mask, interpolation='nearest', aspect='auto', alpha=0.1)
            plt.xlim((0,x_plot_lim*ratio))
            plt.show()

        # only select high synchrony events
        hse_data = binary_data[:, ind_upsampled]
        print("#HSE TIME BINS: " + str(hse_data.shape[1]))

        if plotting:
            plt.imshow(hse_data, interpolation='nearest', aspect='auto')
            plt.title("HSE binarized")
            plt.xlabel("TIME BINS")
            plt.ylabel("CELL IDS")
            plt.show()

        if file_name is not None:
            np.save(file_name, hse_data, fix_imports=True)

    # getter methods
    # ------------------------------------------------------------------------------------------------------------------

    def get_duration_sec(self):
        return self.whl.shape[0] * 0.0256

    def get_raster(self, speed_threshold=False):
        self.compute_raster_speed_loc()
        if speed_threshold:
            speed_threshold = self.session_params.sleep_phase_speed_threshold
            raster = self.raster[:, self.speed < speed_threshold]
        else:
            raster = self.raster
        return raster

    def get_speed(self):
        self.compute_raster_speed_loc()
        return self.speed

    def get_correlation_matrices(self, bins_per_corr_matrix, cell_selection=None, only_upper_triangle=False):

        file_name = self.session_name +"_"+self.experiment_phase+"_"+str(bins_per_corr_matrix)+"_"+self.cell_type+\
                    "_bin_p_m_"+"excl_diag_"+\
                    str(only_upper_triangle)+ "_all"


        # check first if correlation matrices have been computed and saved
        if not os.path.isfile(self.params.pre_proc_dir + "correlation_matrices/" + file_name+".npy"):
            self.compute_raster_speed_loc()
            raster = self.get_raster(speed_threshold=True)
            if cell_selection is not None:
                raster = raster[cell_selection, :]
            else:
                raster = raster
            corr_matrices = correlations_from_raster(raster=raster, bins_per_corr_matrix=bins_per_corr_matrix,
                                        only_upper_triangle=only_upper_triangle)
            np.save(self.params.pre_proc_dir + "correlation_matrices/" + file_name, corr_matrices)

        else:
            print("  - LOADING EXISTING CORRELATION MATRICES")
            # load data
            corr_matrices = np.load(self.params.pre_proc_dir + "correlation_matrices/" + file_name+".npy")

        return corr_matrices

    def get_spike_binned_raster(self, nr_spikes_per_bin=None, return_estimated_times=False):
        # --------------------------------------------------------------------------------------------------------------
        # returns spike binned raster (fixed #spikes per bin)
        #
        # args:   - nr_spikes_per_bin, int
        # --------------------------------------------------------------------------------------------------------------
        if nr_spikes_per_bin is None:
            nr_spikes_per_bin = self.params.spikes_per_bin

        return PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl, last_spike=self.last_spike).spike_binning(
            spikes_per_bin=nr_spikes_per_bin, return_estimated_times=return_estimated_times)

    def get_sleep_phase(self, sleep_phase, speed_threshold=None, classification_method="std"):
        """
        returns start & end (+peak for sharp waves) times for defined sleep phase in seconds

        - timestamps are at 20kHz resolution for sleep data when loaded from .rem, .nrem, .sw

        @param sleep_phase: start/end times for which sleep phase ("rem", "nrem", "sw")
        @type sleep_phase: str
        @param speed_threshold: used to filter periods of movements (only used when type="std"), when None -->
        parameter is loaded from params
        @type speed_threshold: float
        @param classification_method: which sleep classification to use --> "std": Jozsef's standard algorithm, "k_means": Juan's sleep
        classification
        @type classification_method: str
        """
        if classification_method == "std":
            # load timestamps at 20kHz resolution
            time_stamps_orig = self.time_stamps[sleep_phase]

        elif classification_method == "k_means":
            time_stamps_orig = np.loadtxt(self.params.pre_proc_dir+"sleep_states_k_means/"+self.session_name+"/"+
                                          self.session_name+"_"+\
                                          self.experiment_phase_id+"."+sleep_phase).astype(int)

        if speed_threshold is not None:
            # convert time stamps to same resolution like speed
            temp_time_stamps = (time_stamps_orig * 0.00005) / self.params.time_bin_size

            # get speed at params.time_bin_size
            self.compute_speed()
            speed = self.speed

            # good periods
            good_per = np.ones(time_stamps_orig.shape[0])
            # go trough all time stamps
            for i, t_s in enumerate(temp_time_stamps):
                speed_during_per = speed[np.round(t_s[0]).astype(int):np.round(t_s[1]).astype(int)]
                len_per = speed_during_per.shape[0]
                above = np.count_nonzero(np.nan_to_num(speed_during_per) > speed_threshold)
                # if there are periods longer than 1 sec above threshold or more than 50% above threshold,
                # and period shouldnt be shorter than 1 sec
                # --> need to be deleted
                if above * self.params.time_bin_size > 1 or above > 0.05 * len_per or len_per * self.params.time_bin_size < 3:
                    good_per[i] = 0

            time_stamps_orig = time_stamps_orig[good_per.astype(bool), :]

        # get time stamps in seconds
        time_stamps = time_stamps_orig * 0.00005

        # make sure there is no interval that is smaller than one second
        dur = time_stamps[:,1] - time_stamps[:,0]
        to_delete = np.where(dur<1)[0]
        time_stamps = np.delete(time_stamps, to_delete, axis=0)
        return time_stamps

    def get_event_spike_rasters(self, part_to_analyze, speed_threshold=None, plot_for_control=False,
                                return_event_times=False, pop_vec_threshold=None):

        file_name = self.session_name + "_" + self.experiment_phase_id
        result_dir = self.params.pre_proc_dir + "sleep_spike_rasters/" + part_to_analyze

        # check if results exist
        if os.path.isfile(result_dir + "/" + file_name):

            res_dic = pickle.load(open(result_dir + "/" + file_name, "rb"))

            event_spike_rasters = res_dic["event_spike_rasters"]
            event_spike_window_lengths = res_dic["event_spike_window_lengths"]
            start_times = res_dic["start_times"]
            end_times = res_dic["end_times"]

        else:

            # load speed threshold for REM/NREM from parameter file if not provided
            if speed_threshold is None:
                speed_threshold = self.session_params.sleep_phase_speed_threshold

            # check if SWR or REM phases are supposed to be analyzed
            if part_to_analyze == "all_swr":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # compute #spike binning for each event --> TODO: implement sliding window!
                event_spike_rasters, event_spike_window_lengths = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)
                # do not need this info here (all SWR)
                swr_to_nrem = None

            elif part_to_analyze == "nrem":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # only select SWR during nrem phases
                # get nrem phases in seconds
                n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold)
                swr_in_n_rem = np.zeros(event_times.shape[0])
                swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
                for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                    n_rem_start = n_rem_phase[0]
                    n_rem_end = n_rem_phase[1]
                    for i, e_t in enumerate(event_times):
                        event_start = e_t[0]
                        event_end = e_t[1]
                        if (n_rem_start < event_start) and (event_end < n_rem_end):
                            swr_in_n_rem[i] += 1
                            swr_in_which_n_rem[n_rem_phase_id, i] = 1

                event_times = event_times[swr_in_n_rem == 1]
                # assignment: which SWR belongs to which nrem phase (for plotting)
                swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

                print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:,
                            1]  # compute #spike binning for each event --> TODO: implement sliding window!
                event_spike_rasters, event_spike_window_lengths = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

            elif part_to_analyze == "rem":
                # get rem intervals in seconds
                event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold)

                print(" - " + str(event_times.shape[0]) + " REM phases found (speed thr.: " + str(
                    speed_threshold) + ")\n")

                # compute #spike binning for each event --> TODO: implement sliding window!
                event_spike_rasters, event_spike_window_lengths = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:, 1]


            result_post = {
                "event_spike_rasters": event_spike_rasters,
                "event_spike_window_lengths": event_spike_window_lengths,
                "start_times": start_times,
                "end_times": end_times
            }
            outfile = open(result_dir + "/" + file_name, 'wb')
            pickle.dump(result_post, outfile)
            print("  - SAVED NEW RESULTS!\n")

        if pop_vec_threshold is not None:
            # filter epochs that are too short
            a = [x.shape[1] for x in event_spike_rasters]
            to_delete = np.argwhere(np.array(a) < pop_vec_threshold).flatten()
            # remove results
            for i in reversed(to_delete):
                del event_spike_rasters[i]
                del event_spike_window_lengths[i]

            start_times = np.delete(start_times, to_delete)
            end_times = np.delete(end_times, to_delete)

        if return_event_times:

            return event_spike_rasters, event_spike_window_lengths, start_times, end_times

        else:

            return event_spike_rasters, event_spike_window_lengths

    def get_event_time_bin_rasters(self, sleep_phase, time_bin_size=None, speed_threshold=None,
                                   plot_for_control=False):

        # load speed threshold for REM/NREM from parameter file if not provided
        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        # check if SWR or REM phases are supposed to be analyzed
        if sleep_phase == "all_swr":

            # get SWR timings (in sec) & compute spike rasters (constant #spikes)
            # ------------------------------------------------------------------------------------------------------
            start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

            # convert to one array for event_spike_binning
            event_times = np.vstack((start_times, end_times, peak_times)).T

            # compute #spike binning for each event --> TODO: implement sliding window!
            event_time_bin_rasters = \
                PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl).event_temporal_binning(event_times=event_times, event_time_freq=1,
                                                                     time_bin_size=time_bin_size)

        elif sleep_phase == "nrem":

            # get SWR timings (in sec) & compute spike rasters (constant #spikes)
            # ------------------------------------------------------------------------------------------------------
            start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

            # convert to one array for event_spike_binning
            event_times = np.vstack((start_times, end_times, peak_times)).T

            # only select SWR during nrem phases
            # get nrem phases in seconds
            n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold)
            swr_in_n_rem = np.zeros(event_times.shape[0])
            swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
            for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                n_rem_start = n_rem_phase[0]
                n_rem_end = n_rem_phase[1]
                for i, e_t in enumerate(event_times):
                    event_start = e_t[0]
                    event_end = e_t[1]
                    if (n_rem_start < event_start) and (event_end < n_rem_end):
                        swr_in_n_rem[i] += 1
                        swr_in_which_n_rem[n_rem_phase_id, i] = 1

            event_times = event_times[swr_in_n_rem == 1]
            # assignment: which SWR belongs to which nrem phase (for plotting)
            swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

            print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

            # do not need this info here
            swr_to_nrem = None

            start_times = event_times[:, 0]
            end_times = event_times[:,
                        1]  # compute #spike binning for each event --> TODO: implement sliding window!
            event_time_bin_rasters = \
                PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl).event_temporal_binning(event_times=event_times, event_time_freq=1,
                                                                     time_bin_size=time_bin_size)

        elif sleep_phase == "rem":
            # get rem intervals in seconds
            event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold)

            print(" - " + str(event_times.shape[0]) + " REM phases found (speed thr.: " + str(
                speed_threshold) + ")\n")

            # compute #spike binning for each event --> TODO: implement sliding window!
            event_time_bin_rasters = \
                PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl).event_temporal_binning(event_times=event_times, event_time_freq=1,
                                                                     time_bin_size=time_bin_size)

            # do not need this info here
            swr_to_nrem = None

            start_times = event_times[:, 0]
            end_times = event_times[:, 1]

        return event_time_bin_rasters, start_times, end_times

    def get_pre_post_templates(self):
        return self.session_params.default_pre_phmm_model, self.session_params.default_post_phmm_model, \
               self.session_params.default_pre_ising_model, self.session_params.default_post_ising_model

    # visualization methods
    # ------------------------------------------------------------------------------------------------------------------

    def view_raster(self):
        """
        plot raster data
        """
        self.compute_raster_speed_loc()
        plot_act_mat(self.raster, self.params, self.cell_type)

    def save_spike_binned_raster(self):

        raster = self.get_event_spike_rasters(part_to_analyze="rem")
        raster = raster[0][0]
        plt.style.use('default')
        max_val = np.max(raster[:,:20])+1
        cmap = plt.get_cmap('viridis', max_val)
        plt.imshow(raster[:,:20], interpolation='nearest', aspect='auto', cmap=cmap)
        a = plt.colorbar()
        tick_locs = (np.arange(max_val) + 0.5) * (max_val - 1) / max_val
        a.set_ticks(tick_locs)
        a.set_ticklabels(np.arange(max_val).astype(int))
        plt.rcParams['svg.fonttype'] = 'none'
        #plt.show()
        plt.savefig("spike_bin_raster_2.svg", transparent="True")

    # saving
    # ------------------------------------------------------------------------------------------------------------------

    def save_spike_times(self, save_dir):
        # --------------------------------------------------------------------------------------------------------------
        # determines spike times of each cell and saves them as a list of list (each list: firing times of one cell)
        # --> used for TreeHMM
        #
        # args:   - save_dir, str
        # --------------------------------------------------------------------------------------------------------------

        spike_times = PreProcessSleep(self.firing_times, self.params, self.time_stamps).spike_times()
        filename = save_dir + "/" + self.params.session_name+"_"+self.sleep_type+"_"+self.cell_type
        # pickle in using python2 protocol
        with open(filename, "wb") as f:
            pickle.dump(spike_times, f, protocol=2)

    # sleep classification
    # ------------------------------------------------------------------------------------------------------------------

    def analyze_sleep_phase(self, speed_threshold=None):
        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        self.compute_speed()
        time_stamps_rem = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold)
        time_stamps_nrem = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold)
        # compute pre processed data
        plt.plot(np.arange(self.speed.shape[0])*self.params.time_bin_size, self.speed)
        plt.ylabel("SPEED / cm/s")
        plt.xlabel("TIME / s")
        # plt.title("SLEEP: "+ sleep_phase+" PHASES")

        rem_per = np.zeros(int(self.speed.shape[0]*self.params.time_bin_size))
        for t_s in time_stamps_rem:
            rem_per[int(t_s[0]):int(t_s[1])] = 10

        nrem_per = np.zeros(int(self.speed.shape[0]*self.params.time_bin_size))
        for t_s in time_stamps_nrem:
            nrem_per[int(t_s[0]):int(t_s[1])] = 10

        plt.plot(rem_per, color="r", label="REM")
        plt.plot(nrem_per, color="b", alpha=0.5, label="NREM")
        if speed_threshold is not None:
            axes = plt.gca()
            plt.hlines(speed_threshold, 0, self.speed.shape[0]*self.params.time_bin_size, color="yellow", zorder=1000,
                       label="THRESH.")
        plt.legend()

        # for t_s in time_stamps:
        #     # time stamps are in seconds --> need in time bin size
        #     plt.hlines(1, t_s[0], t_s[1], colors="r", zorder=1000)

        plt.show()

    def compare_sleep_classification(self, speed_threshold=None):

        time_stamps_rem_new = self.get_sleep_phase(sleep_phase="rem", classification_method="k_means")

        time_stamps_nrem_new = self.get_sleep_phase(sleep_phase="nrem", classification_method="k_means")

        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        speed_threshold = None

        self.compute_speed()
        time_stamps_rem = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold)
        time_stamps_nrem = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold)

        # compute pre processed data
        plt.plot(np.arange(self.speed.shape[0])*self.params.time_bin_size, self.speed)
        plt.ylabel("SPEED / cm/s")
        plt.xlabel("TIME / s")
        # plt.title("SLEEP: "+ sleep_phase+" PHASES")

        rem_per = np.zeros(int(self.speed.shape[0]*self.params.time_bin_size))
        for t_s in time_stamps_rem:
            rem_per[int(t_s[0]):int(t_s[1])] = 10

        rem_per_new = np.zeros(int(self.speed.shape[0]*self.params.time_bin_size))
        for t_s in time_stamps_rem_new:
            rem_per_new[int(t_s[0]):int(t_s[1])] = 10

        nrem_per_new = np.zeros(int(self.speed.shape[0]*self.params.time_bin_size))
        for t_s in time_stamps_nrem_new:
            nrem_per_new[int(t_s[0]):int(t_s[1])] = 10

        # plt.plot(rem_per, color="r", label="REM")
        plt.plot(rem_per*-1, color="orangered", alpha=0.5, label="REM")
        plt.plot(rem_per_new, color="red", alpha=0.5, label="REM NEW")
        plt.plot(nrem_per_new, color="blue", alpha=0.5, label="NREM NEW")
        if speed_threshold is not None:
            axes = plt.gca()
            plt.hlines(speed_threshold, 0, self.speed.shape[0]*self.params.time_bin_size, color="yellow", zorder=1000,
                       label="THRESH.")
        plt.legend()

        # for t_s in time_stamps:
        #     # time stamps are in seconds --> need in time bin size
        #     plt.hlines(1, t_s[0], t_s[1], colors="r", zorder=1000)

        plt.show()

    """#################################################################################################################
    #   memory drift analysis
    #################################################################################################################"""

    @staticmethod
    def compute_values_from_probabilities(pre_prob_list, post_prob_list, pre_prob_z_list, post_prob_z_list):
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

    # content based memory drift (decoding using ising map, phmm modes from PRE and POST)
    # ------------------------------------------------------------------------------------------------------------------

    def decode_activity_using_pre_post(self, template_type, part_to_analyze, pre_file_name=None, post_file_name=None,
                                       compression_factor=None, speed_threshold=None, plot_for_control=False,
                                       return_results=True, sleep_classification_method="std", cells_to_use="all",
                                       shuffling=False):
        """
        decodes sleep activity using pHMM modes/spatial bins from ising model from before and after awake activity and
        computes similarity measure

        @param template_type: whether to use "phmm" or "ising"
        @type template_type: str
        @param pre_file_name: name of file containing the template from PRE behavioral data --> if = None: file
        defined in session parameter file
        @type pre_file_name: str
        @param post_file_name: name of file containing the template from AFTER/POST awake/behavioral data --> if = None:
        file defined in session parameter file
        @type post_file_name: str
        @param part_to_analyze: which data to analyse ("all_swr", "rem", "nrem")
        @type part_to_analyze: str
        @param compression_factor: scaling factor between sleep activity and awake activity --> if awake
        model was build with constant time bin size (e.g. 100ms): compute how many spikes on average happen during this
        window size (e.g. mean = 30), check how many #spikes you used for constant #spike binning (e.g. 12) for the
        sleep data --> awake_activity_scaling_factor = 12/30 --> = 0.4
        if None --> is loaded from session specific parameter file
        @type compression_factor: float
        @param speed_threshold: filter for rem/nrem phases --> all phases with speeds above this threshold are
        neglected, if you don't want to use a speed_threshold set to None.
        @type speed_threshold: int
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param sleep_classification_method: which sleep classification to use --> std: Jozsef's script, k_mean: Juan's script
        @type sleep_classification_method: str
        @param cells_to_use: which cells to use ("all", "stable", "inc", "dec")
        @type cells_to_use: str
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        # load speed threshold for REM/NREM from parameter file if not provided
        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        if template_type == "phmm":
            # get template file name from parameter file of session if not provided
            if pre_file_name is None:
                pre_file_name = self.session_params.default_pre_phmm_model
            if post_file_name is None:
                post_file_name = self.session_params.default_post_phmm_model
            if sleep_classification_method == "std":
                print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using PHMM modes & std. "
                                                                                  "sleep classification...\n")
                if cells_to_use == "stable":
                    if shuffling:
                        result_dir = "phmm_decoding/stable_cells_shuffled_"+self.params.stable_cell_method
                    else:
                        result_dir = "phmm_decoding/stable_cells_"+self.params.stable_cell_method
                if cells_to_use == "increasing":
                    result_dir = "phmm_decoding/inc_cells_"+self.params.stable_cell_method
                if cells_to_use == "decreasing":
                    result_dir = "phmm_decoding/dec_cells_"+self.params.stable_cell_method
                elif cells_to_use == "all":
                    if shuffling:
                        result_dir = "phmm_decoding/spike_shuffled"
                    else:
                        result_dir = "phmm_decoding"
            elif sleep_classification_method == "k_means":
                print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using PHMM modes & k-means "
                                                                                  "sleep classification...\n")
                result_dir = "phmm_decoding/k_means_sleep_classification"

        elif template_type == "ising":
            print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using Ising model ...\n")
            result_dir = "ising_glm_decoding"
            # get template file name from parameter file of session if not provided
            if pre_file_name is None:
                pre_file_name = self.session_params.default_pre_ising_model
            if post_file_name is None:
                post_file_name = self.session_params.default_post_ising_model

        if pre_file_name is None or post_file_name is None:
            raise Exception("AT LEAST ONE TEMPLATE FILE WAS NEITHER PROVIDED\n NOR IN SESSION PARAMETER FILE DEFINED")

        # check that template files are from correct session
        # --------------------------------------------------------------------------------------------------------------
        #sess_pre = pre_file_name.split(sep="_")[0]+"_"+pre_file_name.split(sep="_")[1]
        #sess_post = post_file_name.split(sep="_")[0]+"_"+post_file_name.split(sep="_")[1]

        #if not (sess_pre == self.params.session_name and sess_post == self.params.session_name):
        #    raise Exception("TEMPLATE FILE AND CURRENT SESSION DO NOT MATCH!")

        file_name_pre = self.session_name +"_"+self.experiment_phase_id + "_"+\
                        part_to_analyze+"_"+ self.cell_type+"_PRE.npy"
        file_name_post = self.session_name + "_" + self.experiment_phase_id + "_"+\
                         part_to_analyze+"_"+self.cell_type+"_POST.npy"

        # check if PRE and POST result exists already
        # --------------------------------------------------------------------------------------------------------------
        if os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + file_name_pre) and \
            os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + file_name_post):
            print(" - PRE and POST results exist already -- using existing results\n")
        else:
            # if PRE and/or POST do not exist yet --> compute results

            # check if SWR or REM phases are supposed to be analyzed

            if part_to_analyze == "all_swr":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # compute #spike binning for each event --> TODO: implement sliding window!
                event_spike_rasters, event_spike_window_lenghts = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)
                # do not need this info here (all SWR)
                swr_to_nrem = None

            elif part_to_analyze == "nrem":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # only select SWR during nrem phases
                # get nrem phases in seconds
                n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                         classification_method=sleep_classification_method)
                swr_in_n_rem = np.zeros(event_times.shape[0])
                swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
                for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                    n_rem_start = n_rem_phase[0]
                    n_rem_end = n_rem_phase[1]
                    for i, e_t in enumerate(event_times):
                        event_start = e_t[0]
                        event_end = e_t[1]
                        if (n_rem_start < event_start) and (event_end < n_rem_end):
                            swr_in_n_rem[i] += 1
                            swr_in_which_n_rem[n_rem_phase_id, i] = 1

                event_times = event_times[swr_in_n_rem == 1]
                # assignment: which SWR belongs to which nrem phase (for plotting)
                swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

                print(" - "+str(event_times.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:, 1]       # compute #spike binning for each event --> TODO: implement sliding window!
                event_spike_rasters, event_spike_window_lenghts = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

            elif part_to_analyze == "rem":
                # get rem intervals in seconds
                event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                                   classification_method=sleep_classification_method)

                print(" - "+str(event_times.shape[0])+" REM phases found (speed thr.: "+str(speed_threshold)+")\n")

                # compute #spike binning for each event --> TODO: implement sliding window!
                event_spike_rasters, event_spike_window_lenghts = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:, 1]

            else:
                raise Exception("Provided part_to_analyze is not defined!")

            # plot raster and detected SWR, example spike rasters from SWRs
            if plot_for_control:
                # ------------------------------------------------------------------------------------------------------
                # plot detected events
                # ------------------------------------------------------------------------------------------------------
                # compute pre processed data
                self.compute_raster_speed_loc()
                to_plot = np.random.randint(0, start_times.shape[0], 5)
                for i in to_plot:
                    plt.imshow(self.raster, interpolation='nearest', aspect='auto')
                    plt.vlines(start_times[i] / self.params.time_bin_size, 0, self.raster.shape[0] - 1,
                               colors="r",
                               linewidth=0.5,
                               label="START")
                    plt.vlines(end_times[i] / self.params.time_bin_size, 0, self.raster.shape[0] - 1,
                               colors="g",
                               linewidth=0.5, label="END")
                    plt.xlim(start_times[i] / self.params.time_bin_size - 10,
                             end_times[i] / self.params.time_bin_size + 10)
                    plt.ylabel("CELL IDS")
                    plt.xlabel("TIME BINS / "+str(self.params.time_bin_size)+"s")
                    plt.legend()
                    a = plt.colorbar()
                    a.set_label("SPIKES PER BIN (" + str(self.params.time_bin_size) + " s)")
                    plt.title("EVENT ID " + str(i))
                    plt.show()

                    plt.imshow(event_spike_rasters[i], interpolation='nearest', aspect='auto')
                    a = plt.colorbar()
                    a.set_label("SPIKES PER CONST. #SPIKES BIN")
                    plt.xlabel("CONST. #SPIKE POP. VEC. ID")
                    plt.ylabel("CELL ID")
                    plt.title("BINNED EVENT ID " + str(i) + "\n#SPIKES PER BIN: " + str(
                        self.params.spikes_per_bin))
                    plt.show()

                # ------------------------------------------------------------------------------------------------------
                # compute length of constant #spike windows
                # ------------------------------------------------------------------------------------------------------
                event_spike_window_lengths_avg = np.mean(np.concatenate(event_spike_window_lenghts, axis=0))
                event_spike_window_lengths_median = np.median(np.concatenate(event_spike_window_lenghts, axis=0))

                y, x, _ = plt.hist(np.concatenate(event_spike_window_lenghts, axis=0), bins=30)
                plt.xlabel("LENGTH CONST #SPIKES POP. VEC / s")
                plt.ylabel("COUNTS")
                plt.title("POPULATION VECTOR LENGTH")
                plt.vlines(event_spike_window_lengths_avg, 0, y.max(), colors="r", label="MEAN: "+
                                        str(np.round(event_spike_window_lengths_avg,2)))
                plt.vlines(event_spike_window_lengths_median, 0, y.max(), colors="b", label="MEDIAN: "+
                                                str(np.round(event_spike_window_lengths_median,2)))
                plt.legend()
                plt.show()

            for result_file_name, template_file_name in zip([file_name_pre, file_name_post],
                                                            [pre_file_name, post_file_name]):
                if os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + result_file_name):
                    print(" - RESULT EXISTS ALREADY (" + result_file_name + ")\n")
                    continue
                else:
                    print(" - DECODING SLEEP ACTIVITY USING " + template_file_name + " ...\n")

                    if template_type == "phmm":
                        # load pHMM model
                        with open(self.params.pre_proc_dir + "phmm/" + template_file_name + '.pkl', 'rb') as f:
                            model_dic = pickle.load(f)
                        # get means of model (lambdas) for decoding
                        mode_means = model_dic.means_

                        time_bin_size_encoding = model_dic.time_bin_size

                        # check if const. #spike bins are correct for the loaded compression factor
                        if not self.params.spikes_per_bin == 12:
                            raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                            "BUT CURRENT #SPIKES PER BIN != 12")

                        # load correct compression factor (as defined in parameter file of the session)
                        if time_bin_size_encoding == 0.01:
                            compression_factor = np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
                        elif time_bin_size_encoding == 0.1:
                            compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
                        else:
                            raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

                        # if you want to use different compression factors for PRE/POST
                        # if "PRE" in result_file_name:
                        #     compression_factor = 0.4
                        # elif "POST" in result_file_name:
                        #     compression_factor = 0.6
                        if cells_to_use == "stable":
                            cell_selection = "all"
                            # load cell ids of stable cells
                            # get stable, decreasing, increasing cells
                            with open(self.params.pre_proc_dir + "cell_classification/" +
                                      self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                                      "rb") as f:
                                class_dic = pickle.load(f)

                            cells_ids = class_dic["stable_cell_ids"]

                        elif cells_to_use == "increasing":
                            cell_selection = "all"
                            # load cell ids of stable cells

                            if self.params.stable_cell_method == "k_means":
                                # load only stable cells
                                with open(self.params.pre_proc_dir + "cell_classification/" +
                                        self.params.session_name + "_k_means.pickle", "rb") as f:
                                    class_dic = pickle.load(f)
                                cells_ids = class_dic["increase_cell_ids"]

                            elif self.params.stable_cell_method == "mean_firing_awake":
                                # load only stable cells
                                with open(self.params.pre_proc_dir + "cell_classification/" +
                                        self.params.session_name + "_mean_firing_awake.pickle", "rb") as f:
                                    class_dic = pickle.load(f)

                                cells_ids = class_dic["increase_cell_ids"]

                            else:
                                raise Exception("NOT IMPLEMENTED YET!!!")

                        elif cells_to_use == "decreasing":
                            cell_selection = "all"
                            # load cell ids of stable cells

                            if self.params.stable_cell_method == "k_means":
                                # load only stable cells
                                with open(self.params.pre_proc_dir + "cell_classification/" +
                                        self.params.session_name + "_k_means.pickle", "rb") as f:
                                    class_dic = pickle.load(f)
                                cells_ids = class_dic["decrease_cell_ids"]

                            elif self.params.stable_cell_method == "mean_firing_awake":
                                # load only stable cells
                                with open(self.params.pre_proc_dir + "cell_classification/" +
                                        self.params.session_name + "_mean_firing_awake.pickle", "rb") as f:
                                    class_dic = pickle.load(f)

                                cells_ids = class_dic["decrease_cell_ids"]

                            else:
                                raise Exception("NOT IMPLEMENTED YET!!!")

                        elif cells_to_use == "all":
                            cell_selection = "all"

                        if cells_to_use in ["stable", "increasing", "decreasing"]:
                            # need to select event_spike_rasters only for subsets, the same for lambdas
                            # to only select cells that are wanted
                            event_spike_rasters_modified = []
                            for ev_r in event_spike_rasters:
                                event_spike_rasters_modified.append(ev_r[cells_ids, :])

                            event_spike_rasters = event_spike_rasters_modified

                            mode_means = mode_means[:, cells_ids]

                        if shuffling:
                            print(" -- STARTED SWAPPING PROCEDURE ...")
                            # merge all events
                            conc_data = np.hstack(event_spike_rasters)
                            nr_swaps = conc_data.shape[1]*10
                            for shuffle_id in range(nr_swaps):
                                # select two random time bins
                                t1 = 1
                                t2 = 1
                                while(t1 == t2):
                                    t1 = np.random.randint(conc_data.shape[1])
                                    t2 = np.random.randint(conc_data.shape[1])
                                # check in both time bins which cells are active
                                act_cells_t1 = np.argwhere(conc_data[:, t1].flatten()>0).flatten()
                                act_cells_t2 = np.argwhere(conc_data[:, t2].flatten()>0).flatten()
                                # find intersect (same cells need to be firing in t1 and t2 in order to exchange
                                # spikes
                                # original code
                                # --------------------------------------------------------------------------------------
                                # cells_firing_in_both = np.intersect1d(act_cells_t1, act_cells_t2)
                                # if cells_firing_in_both.shape[0] > 1:
                                #     # select first cell to swap
                                #     cell_1 = 1
                                #     cell_2 = 1
                                #     while (cell_1 == cell_2):
                                #         cell_1 = np.random.choice(cells_firing_in_both)
                                #         cell_2 = np.random.choice(cells_firing_in_both)
                                #     # do the actual swapping
                                #     conc_data[cell_1, t1] += 1
                                #     conc_data[cell_1, t2] -= 1
                                #     conc_data[cell_2, t1] -= 1
                                #     conc_data[cell_2, t2] += 1

                                if act_cells_t1.shape[0] > 1 and act_cells_t2.shape[0] > 1:
                                    # select first cell to swap
                                    cell_1 = 1
                                    cell_2 = 1
                                    while (cell_1 == cell_2):
                                        cell_1 = np.random.choice(act_cells_t2)
                                        cell_2 = np.random.choice(act_cells_t1)
                                    # do the actual swapping
                                    conc_data[cell_1, t1] += 1
                                    conc_data[cell_1, t2] -= 1
                                    conc_data[cell_2, t1] -= 1
                                    conc_data[cell_2, t2] += 1

                            print(" -- ... DONE!")
                            # split data again into list
                            event_lengths = [x.shape[1] for x in event_spike_rasters]

                            event_spike_rasters_shuffled = []
                            start = 0
                            for el in event_lengths:
                                event_spike_rasters_shuffled.append(conc_data[:,start:start+el])
                                start = el

                            event_spike_rasters = event_spike_rasters_shuffled
                        # start with actual decoding
                        # ----------------------------------------------------------------------------------------------

                        print(" - DECODING USING "+ cells_to_use + " CELLS")

                        results_list = decode_using_phmm_modes(mode_means=mode_means,
                                                               event_spike_rasters=event_spike_rasters,
                                                               compression_factor=compression_factor,
                                                               cell_selection=cell_selection)

                    elif template_type == "ising":
                        # load ising template
                        with open(self.params.pre_proc_dir + 'awake_ising_maps/' + template_file_name + '.pkl',
                                  'rb') as f:
                            model_dic = pickle.load(f)

                        # if compression_factor is not provided --> load from parameter file
                        if compression_factor is None:
                            # get time_bin_size of encoding
                            time_bin_size_encoding = model_dic["time_bin_size"]

                            # check if const. #spike bins are correct for the loaded compression factor
                            if not self.params.spikes_per_bin == 12:
                                raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                                "BUT CURRENT #SPIKES PER BIN != 12")

                            # load correct compression factor (as defined in parameter file of the session)
                            if time_bin_size_encoding == 0.01:
                                compression_factor = \
                                    np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
                            elif time_bin_size_encoding == 0.1:
                                compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
                            else:
                                raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

                        # get template map
                        template_map = model_dic["res_map"]

                        results_list = decode_using_ising_map(template_map=template_map,
                                                                       event_spike_rasters=event_spike_rasters,
                                                                       compression_factor=compression_factor,
                                                                       cell_selection="all")

                    # plot maps of some SWR for control
                    if plot_for_control:
                        swr_to_plot = np.random.randint(0, len(results_list), 10)
                        for swr in swr_to_plot:
                            res = results_list[swr]
                            plt.imshow(res.T, interpolation='nearest', aspect='auto')
                            plt.xlabel("POP.VEC. ID")
                            plt.ylabel("MODE ID")
                            a = plt.colorbar()
                            a.set_label("PROBABILITY")
                            plt.title("PROBABILITY MAP, SWR ID: " + str(swr))
                            plt.show()

                    # saving results
                    # --------------------------------------------------------------------------------------------------
                    # create dictionary with results
                    result_post = {
                        "results_list": results_list,
                        "event_times": event_times,
                        "swr_to_nrem": swr_to_nrem
                    }
                    outfile = open(self.params.pre_proc_dir + result_dir +"/" + result_file_name, 'wb')
                    pickle.dump(result_post, outfile)
                    print("  - SAVED NEW RESULTS!\n")

        if return_results:

            while True:
                # load decoded maps
                try:
                    result_pre = pickle.load(open(self.params.pre_proc_dir+result_dir + "/" + file_name_pre,"rb"))
                    result_post = pickle.load(open(self.params.pre_proc_dir+result_dir + "/" + file_name_post, "rb"))
                    break
                except:
                    continue

            pre_prob = result_pre["results_list"]
            post_prob = result_post["results_list"]
            event_times = result_pre["event_times"]
            swr_to_nrem = result_pre["swr_to_nrem"]

            return pre_prob, post_prob, event_times, swr_to_nrem

    def decode_activity_using_pre_post_plot_results(self, template_type, part_to_analyze, pre_file_name=None,
                                                    post_file_name=None, plot_for_control=False,
                                                    n_moving_average_events=10, n_moving_average_pop_vec=40,
                                                    only_stable_cells=False):
        """
        plots results of memory drift analysis (if results don't exist --> they are computed)

        @param template_type: whether to use "phmm" or "ising"
        @type template_type: str
        @param part_to_analyze: which data to analyse ("all_swr", "rem", "nrem", "nrem_rem")
        @type part_to_analyze: str
        @param pre_file_name: name of file containing the template from PRE behavioral data --> if = None: file
        defined in session parameter file
        @type pre_file_name: str
        @param post_file_name: name of file containing the template from AFTER/POST awake/behavioral data --> if = None:
        file defined in session parameter file
        @type post_file_name: str
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param n_moving_average_events: n for moving average across events
        @type n_moving_average_events: int
        @param n_moving_average_pop_vec: n for moving average across population vectors
        @type n_moving_average_pop_vec: int
        @param only_stable_cells: whether to only use cells with stable firing
        @type only_stable_cells: bool
        """
        if part_to_analyze == "nrem_rem":
            # plot nrem data first
            # ----------------------------------------------------------------------------------------------------------

            pre_prob_list, post_prob_list, event_times, swr_to_nrem = self.decode_activity_using_pre_post(
                template_type=template_type, pre_file_name=pre_file_name, post_file_name=post_file_name,
                plot_for_control=plot_for_control,
                part_to_analyze="nrem", only_stable_cells=only_stable_cells)


            pre_prob_arr = np.vstack(pre_prob_list)
            post_prob_arr = np.vstack(post_prob_list)

            # z-scoring of probabilites
            pre_prob_arr_z = zscore(pre_prob_arr + 1e-50, axis=0)
            post_prob_arr_z = zscore(post_prob_arr + 1e-50, axis=0)

            # assign array chunks to events again (either SWR or rem phases)
            event_lengths = [x.shape[0] for x in pre_prob_list]
            pre_prob_z = []
            post_prob_z = []
            first = 0
            for swr_id in range(len(event_lengths)):
                pre_prob_z.append(pre_prob_arr_z[first:first + event_lengths[swr_id], :])
                post_prob_z.append(post_prob_arr_z[first:first + event_lengths[swr_id], :])
                first += event_lengths[swr_id]

            event_pre_post_ratio, event_pre_post_ratio_z, event_pre_prob, event_post_prob, event_len_seq, \
            pop_vec_pre_post_ratio, pre_seq_list, pre_seq_list_z, post_seq_list, pre_seq_list_prob, \
            post_seq_list_prob, pop_vec_post_prob, pop_vec_pre_prob = compute_values_from_probabilities(
                pre_prob_list=pre_prob_list, post_prob_list=post_prob_list, post_prob_z_list=post_prob_z,
            pre_prob_z_list=pre_prob_z)

            # smoothen
            # ------------------------------------------------------------------------------------------------------
            event_pre_post_ratio_smooth = moving_average(a=np.array(event_pre_post_ratio), n=n_moving_average_events)
            event_pre_post_ratio_smooth_z = moving_average(a=np.array(event_pre_post_ratio_z), n=n_moving_average_events)
            pop_vec_pre_post_ratio = np.array(pop_vec_pre_post_ratio)
            # compute moving average to smooth signal
            pop_vec_pre_post_ratio_smooth = moving_average(a=pop_vec_pre_post_ratio, n=50)
            event_len_seq_smooth = moving_average(a=np.array(event_len_seq), n=n_moving_average_events)

            swr_to_nrem = swr_to_nrem[:,:event_pre_post_ratio_smooth.shape[0]]
            event_times = event_times[:event_pre_post_ratio_smooth.shape[0],:]
            # # plot per nrem phase
            fig = plt.figure()
            ax = fig.add_subplot()
            # for nrem_id in range(swr_to_nrem.shape[0]):
            #     ax.plot(event_times[swr_to_nrem[nrem_id,:]==1,1],
            #             event_pre_post_ratio_smooth[swr_to_nrem[nrem_id,:]==1],c="blue", label="NREM")

            # plot per nrem phase
            start = 0
            for rem_length, rem_time in zip(event_lengths, event_times):
                if start + rem_length > pop_vec_pre_post_ratio_smooth.shape[0]:
                    continue
                ax.plot(np.linspace(rem_time[0], rem_time[1], rem_length),
                        pop_vec_pre_post_ratio_smooth[start:start+rem_length],c="blue", label="NREM")
                start += rem_length


            # plot rem data
            # ----------------------------------------------------------------------------------------------------------
            pre_prob_list, post_prob_list, event_times, swr_to_nrem = self.decode_activity_using_pre_post(
                template_type=template_type, pre_file_name=pre_file_name, post_file_name=post_file_name,
                plot_for_control=plot_for_control,
                part_to_analyze="rem")

            pre_prob_arr = np.vstack(pre_prob_list)
            post_prob_arr = np.vstack(post_prob_list)

            # z-scoring of probabilites
            pre_prob_arr_z = zscore(pre_prob_arr + 1e-50, axis=0)
            post_prob_arr_z = zscore(post_prob_arr + 1e-50, axis=0)

            # assign array chunks to events again (either SWR or rem phases)
            event_lengths = [x.shape[0] for x in pre_prob_list]
            pre_prob_z = []
            post_prob_z = []
            first = 0
            for swr_id in range(len(event_lengths)):
                pre_prob_z.append(pre_prob_arr_z[first:first + event_lengths[swr_id], :])
                post_prob_z.append(post_prob_arr_z[first:first + event_lengths[swr_id], :])
                first += event_lengths[swr_id]

            event_pre_post_ratio, event_pre_post_ratio_z, event_pre_prob, event_post_prob, event_len_seq, \
            pop_vec_pre_post_ratio, pre_seq_list, pre_seq_list_z, post_seq_list, pre_seq_list_prob, \
            post_seq_list_prob, pop_vec_post_prob, pop_vec_pre_prob = self.compute_values_from_probabilities(
                pre_prob_list=pre_prob_list, post_prob_list=post_prob_list, post_prob_z_list=post_prob_z,
            pre_prob_z_list=pre_prob_z)

            # smoothen
            # ------------------------------------------------------------------------------------------------------
            event_pre_post_ratio_smooth = moving_average(a=np.array(event_pre_post_ratio), n=n_moving_average_events)
            event_pre_post_ratio_smooth_z = moving_average(a=np.array(event_pre_post_ratio_z), n=n_moving_average_events)
            pop_vec_pre_post_ratio = np.array(pop_vec_pre_post_ratio)
            # compute moving average to smooth signal
            pop_vec_pre_post_ratio_smooth = moving_average(a=pop_vec_pre_post_ratio, n=n_moving_average_pop_vec)
            event_len_seq_smooth = moving_average(a=np.array(event_len_seq), n=n_moving_average_events)

            # plot per nrem phase
            start = 0
            for rem_length, rem_time in zip(event_lengths, event_times):
                if start + rem_length > pop_vec_pre_post_ratio_smooth.shape[0]:
                    continue
                ax.plot(np.linspace(rem_time[0], rem_time[1], rem_length),
                        pop_vec_pre_post_ratio_smooth[start:start+rem_length],c="red", label="REM", alpha=0.5)
                start += rem_length

            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.xlabel("TIME / s")
            plt.ylabel("PRE_POST SIMILARITY")
            plt.title("PRE_POST SIMILARITY USING "+template_type)
            plt.show()

        else:
            pre_prob_list, post_prob_list, event_times, swr_to_nrem = self.decode_activity_using_pre_post(
                template_type=template_type, pre_file_name=pre_file_name, post_file_name=post_file_name,
                plot_for_control=plot_for_control,
                part_to_analyze=part_to_analyze, only_stable_cells=only_stable_cells)

            pre_prob_arr = np.vstack(pre_prob_list)
            post_prob_arr = np.vstack(post_prob_list)

            # z-scoring of probabilites: add very small value in case entries are all zero --> would get an error
            # for z-scoring otherwise
            pre_prob_arr_z = zscore(pre_prob_arr+1e-50, axis=0)
            post_prob_arr_z = zscore(post_prob_arr+1e-50, axis=0)

            # assign array chunks to events again (either SWR or rem phases)
            event_lengths = [x.shape[0] for x in pre_prob_list]
            pre_prob_z = []
            post_prob_z = []
            first = 0
            for swr_id in range(len(event_lengths)):
                pre_prob_z.append(pre_prob_arr_z[first:first + event_lengths[swr_id], :])
                post_prob_z.append(post_prob_arr_z[first:first + event_lengths[swr_id], :])
                first += event_lengths[swr_id]

            nr_modes_pre = pre_prob_list[0].shape[1]
            nr_modes_post = post_prob_list[0].shape[1]

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
            for pre_array, post_array, pre_array_z, post_array_z in zip(pre_prob_list, post_prob_list, pre_prob_z,
                                                                        post_prob_z):
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

            # to plot a detailed description of one phase of sleep

            # r_to_plot = range(0, 200)
            # plt.figure(figsize=(10, 15))
            # plt.subplot(3, 1, 1)
            # plt.plot(pop_vec_pre_prob[r_to_plot], label="MAX. PROB. PRE")
            # plt.plot(pop_vec_post_prob[r_to_plot], label="MAX. PROB. POST")
            # # plt.plot(pre_SWR_prob_arr[r_to_plot, 10], c="r", label="PROB. MODE 60")
            # plt.legend()
            # plt.ylabel("PROB")
            # plt.grid()
            # plt.yscale("log")
            # plt.title("RESULTS FOR PHMM TEMPLATE" if (template_type=="phmm") else "RESULTS FOR ISING TEMPLATE")
            # plt.subplot(3, 1, 2)
            # plt.scatter(r_to_plot, pop_vec_pre_post_ratio[r_to_plot], c="magenta")
            # plt.ylabel("PRE_POST RATIO")
            # plt.grid()
            # plt.subplot(3, 1, 3)
            # plt.scatter(r_to_plot, pre_seq_list[r_to_plot], c="y")
            # plt.ylabel("PRE MODE ID")
            # plt.grid()
            # plt.xlabel("POP. VEC. ID")
            # plt.show()

            # smoothen
            # ------------------------------------------------------------------------------------------------------
            event_pre_post_ratio_smooth = moving_average(a=np.array(event_pre_post_ratio), n=n_moving_average_events)
            event_pre_post_ratio_smooth_z = moving_average(a=np.array(event_pre_post_ratio_z), n=n_moving_average_events)
            pop_vec_pre_post_ratio = np.array(pop_vec_pre_post_ratio)
            # compute moving average to smooth signal
            pop_vec_pre_post_ratio_smooth = moving_average(a=pop_vec_pre_post_ratio, n=n_moving_average_pop_vec)
            event_len_seq_smooth = moving_average(a=np.array(event_len_seq), n=n_moving_average_events)

            # compute per mode info
            # ------------------------------------------------------------------------------------------------------
            pre_seq = np.array(pre_seq_list)
            post_seq = np.array(post_seq_list)
            mode_score_mean_pre = np.zeros(pre_prob_list[0].shape[1])
            mode_score_std_pre = np.zeros(pre_prob_list[0].shape[1])
            mode_score_mean_post = np.zeros(post_prob_list[0].shape[1])
            mode_score_std_post = np.zeros(post_prob_list[0].shape[1])

            # go through all pre modes and check the average score
            for i in range(pre_prob_list[0].shape[1]):
                ind_sel = np.where(pre_seq == i)[0]
                if ind_sel.size == 0 or ind_sel.size == 1:
                    mode_score_mean_pre[i] = np.nan
                    mode_score_std_pre[i] = np.nan
                else:
                    # delete all indices that are too large (becaue of moving average)
                    ind_sel = ind_sel[ind_sel < pop_vec_pre_post_ratio.shape[0]]
                    mode_score_mean_pre[i] = np.mean(pop_vec_pre_post_ratio[ind_sel])
                    mode_score_std_pre[i] = np.std(pop_vec_pre_post_ratio[ind_sel])

            # go through all post modes and check the average score
            for i in range(post_prob_list[0].shape[1]):
                ind_sel = np.where(post_seq == i)[0]
                if ind_sel.size == 0 or ind_sel.size == 1:
                    mode_score_mean_post[i] = np.nan
                    mode_score_std_post[i] = np.nan
                else:
                    # delete all indices that are too large (becaue of moving average)
                    ind_sel = ind_sel[ind_sel < pop_vec_pre_post_ratio.shape[0]]
                    mode_score_mean_post[i] = np.mean(pop_vec_pre_post_ratio[ind_sel])
                    mode_score_std_post[i] = np.std(pop_vec_pre_post_ratio[ind_sel])

            low_score_modes = np.argsort(mode_score_mean_pre)
            # need to skip nans
            nr_nans = np.count_nonzero(np.isnan(mode_score_mean_pre))
            high_score_modes = np.flip(low_score_modes)[nr_nans:]

            # check if modes get more often/less often reactivated over time
            pre_seq_list = np.array(pre_seq_list)
            nr_pop_vec = 20
            nr_windows = int(pre_seq_list.shape[0] / nr_pop_vec)
            occurence_modes_pre = np.zeros((nr_modes_pre, nr_windows))
            for i in range(nr_windows):
                seq = pre_seq_list[i * nr_pop_vec:(i + 1) * nr_pop_vec]
                mode, counts = np.unique(seq, return_counts=True)
                occurence_modes_pre[mode, i] = counts

            # check if modes get more often/less often reactivated over time
            post_seq_list = np.array(post_seq_list)
            nr_pop_vec = 20
            nr_windows = int(post_seq_list.shape[0] / nr_pop_vec)
            occurence_modes_post = np.zeros((nr_modes_pre, nr_windows))
            for i in range(nr_windows):
                seq = post_seq_list[i * nr_pop_vec:(i + 1) * nr_pop_vec]
                mode, counts = np.unique(seq, return_counts=True)
                occurence_modes_post[mode, i] = counts

            # per event (SWR/rem phase) data
            # ----------------------------------------------------------------------------------------------------------

            # check if plotting smoothed data makes sense
            if len(event_lengths) > 3 * n_moving_average_events:
                # per event pre-post ratio smoothed
                plt.plot(event_pre_post_ratio_smooth, c="r", label="n_mov_avg = " + str(n_moving_average_events))
                plt.title("PRE-POST RATIO FOR EACH EVENT: PHMM" if (template_type=="phmm") else
                          "PRE-POST RATIO FOR EACH EVENT: ISING")
                plt.xlabel("SWR ID" if part_to_analyze=="nrem" else "REM PHASE ID")
                plt.ylabel("PRE-POST SIMILARITY")
                plt.ylim(-1, 1)
                plt.grid()
                plt.legend()
                plt.show()

                # per event pre-post ratio after z-scoring smoothed
                plt.plot(event_pre_post_ratio_smooth_z, c="r", label="n_mov_avg = " + str(n_moving_average_events))
                plt.title(
                    "PRE-POST RATIO FOR EACH EVENT: PHMM\n Z-SCORED TO SELECT WINNER" if (template_type == "phmm") else
                    "PRE-POST RATIO FOR EACH EVENT: ISING\n Z-SCORED TO SELECT WINNER")
                plt.xlabel("SWR ID" if part_to_analyze == "nrem" else "REM PHASE ID")
                plt.ylabel("PRE-POST SIMILARITY")
                plt.ylim(-1, 1)
                plt.grid()
                plt.legend()
                plt.show()

                # event length smoothed
                plt.plot(event_len_seq_smooth, label="n_mov_avg = " + str(n_moving_average_events))
                plt.title("EVENT LENGTH")
                plt.xlabel("SWR ID" if part_to_analyze == "nrem" else "REM PHASE ID")
                plt.ylabel("#POP.VEC. PER SWR")
                plt.grid()
                plt.legend()
                plt.show()
            else:
                # per event pre-post ratio not smoothed
                plt.plot(event_pre_post_ratio, c="r", label="not smoothed")
                plt.title("PRE-POST RATIO FOR EACH EVENT: PHMM" if (template_type=="phmm") else
                          "PRE-POST RATIO FOR EACH EVENT: ISING")
                plt.xlabel("SWR ID" if part_to_analyze=="nrem" else "REM PHASE ID")
                plt.ylabel("PRE-POST SIMILARITY")
                plt.ylim(-1, 1)
                plt.grid()
                plt.legend()
                plt.show()

                # per event pre-post ratio after z-scoring not smoothed
                plt.plot(event_pre_post_ratio_z, c="r", label="not smoothed")
                plt.title(
                    "PRE-POST RATIO FOR EACH EVENT: PHMM\n Z-SCORED TO SELECT WINNER" if (template_type == "phmm") else
                    "PRE-POST RATIO FOR EACH EVENT: ISING\n Z-SCORED TO SELECT WINNER")
                plt.xlabel("SWR ID" if part_to_analyze == "nrem" else "REM PHASE ID")
                plt.ylabel("PRE-POST SIMILARITY")
                plt.ylim(-1, 1)
                plt.grid()
                plt.legend()
                plt.show()

                # event length not smoothed
                plt.plot(event_len_seq, label="not smoothed")
                plt.title("EVENT LENGTH")
                plt.xlabel("SWR ID" if part_to_analyze == "nrem" else "REM PHASE ID")
                plt.ylabel("#POP.VEC. PER SWR")
                plt.grid()
                plt.legend()
                plt.show()

            # per population vector data
            # ----------------------------------------------------------------------------------------------------------

            # per population vector ratio smoothed
            plt.plot(pop_vec_pre_post_ratio_smooth, label="n_mov_avg = " + str(n_moving_average_pop_vec))
            plt.title("PRE-POST RATIO FOR EACH POP. VECTOR: PHMM" if (template_type=="phmm") else
                      "PRE-POST RATIO FOR EACH POP. VECTOR: ISING")
            plt.xlabel("POP.VEC. ID")
            plt.ylabel("PRE-POST SIMILARITY")
            plt.ylim(-1, 1)
            plt.grid()
            plt.legend()
            plt.show()

            # per mode / per spatial bin data
            # ----------------------------------------------------------------------------------------------------------

            # occurrence of modes/spatial bins of PRE
            plt.imshow(occurence_modes_pre, interpolation='nearest', aspect='auto')
            plt.ylabel("MODE ID" if (template_type=="phmm") else "SPATIAL BIN")
            plt.xlabel("WINDOW ID")
            a = plt.colorbar()
            a.set_label("#WINS/" + str(nr_pop_vec) + " POP. VEC. WINDOW")
            plt.title("PRE: OCCURENCE (#WINS) OF MODES IN \nWINDOWS OF FIXED LENGTH" if (template_type=="phmm") else
                      "PRE: OCCURENCE (#WINS) OF SPATIAL BINS IN \nWINDOWS OF FIXED LENGTH")
            plt.show()

            # occurrence of modes/spatial bins of PRE
            plt.imshow(occurence_modes_post, interpolation='nearest', aspect='auto')
            plt.ylabel("MODE ID" if (template_type=="phmm") else "SPATIAL BIN")
            plt.xlabel("WINDOW ID")
            a = plt.colorbar()
            a.set_label("#WINS/" + str(nr_pop_vec) + " POP. VEC. WINDOW")
            plt.title("POST: OCCURENCE (#WINS) OF MODES IN \nWINDOWS OF FIXED LENGTH" if (template_type=="phmm") else
                      "POST: OCCURENCE (#WINS) OF SPATIAL BINS IN \nWINDOWS OF FIXED LENGTH")
            plt.show()

            # mode/spatial bin - score assignment: PRE
            plt.errorbar(range(pre_prob_list[0].shape[1]), mode_score_mean_pre, yerr=mode_score_std_pre,
                         linestyle="")
            plt.scatter(range(pre_prob_list[0].shape[1]), mode_score_mean_pre)
            plt.title("PRE-POST SCORE PER MODE: PRE" if (template_type=="phmm") else "PRE-POST SCORE PER SPATIAL BIN: PRE")
            plt.xlabel("MODE ID" if (template_type=="phmm") else "SPATIAL BIN ID")
            plt.ylabel("PRE-POST SCORE: MEAN AND STD")
            plt.show()

            # mode/spatial bin - score assignment: POST
            plt.errorbar(range(post_prob_list[0].shape[1]), mode_score_mean_post, yerr=mode_score_std_post,
                         linestyle="")
            plt.scatter(range(post_prob_list[0].shape[1]), mode_score_mean_post)
            plt.title("PRE-POST SCORE PER MODE: POST" if (template_type=="phmm") else "PRE-POST SCORE PER SPATIAL BIN: POST")
            plt.xlabel("MODE ID" if (template_type=="phmm") else "SPATIAL BIN ID")
            plt.ylabel("PRE-POST SCORE: MEAN AND STD")
            plt.show()

    def plot_likelihoods(self, template_type="phmm", part_to_analyze="rem"):

        pre_prob, post_prob, event_times, swr_to_nrem = self.decode_activity_using_pre_post(template_type=template_type,
                                                                                            part_to_analyze=part_to_analyze)

        print("HERE")
        # use only second time bin to plot likelihoods
        pre_like = pre_prob[0][1, :]
        post_like = post_prob[0][1, :]


        n_col = 8
        scaler = 1
        plt.style.use('default')

        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(6, n_col)

        max_val = pre_like[:9].max()
        min_val = pre_like[:9].min()

        for i in range(n_col-1):
            ax1 = fig.add_subplot(gs[:, i])
            a = ax1.imshow(np.expand_dims(np.expand_dims(pre_like[i], 1),0), vmin=min_val, vmax=max_val, cmap="YlOrRd")
            ax1.set_xticks([])
            ax1.set_yticks([], [])
            ax1.set_xlabel(str(i))
            ax1.set_aspect(scaler)

        plt.tight_layout()
        ax1 = fig.add_subplot(gs[:, n_col - 1])
        fig.colorbar(a, cax=ax1)
        plt.rcParams['svg.fonttype'] = 'none'
        #plt.savefig("pre_likeli.svg", transparent="True")
        plt.show()

        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(6, n_col)

        max_val = post_like[:7].max()
        min_val = post_like[:7].min()

        for i in range(n_col-1):
            ax1 = fig.add_subplot(gs[:, i])
            a = ax1.imshow(np.expand_dims(np.expand_dims(post_like[i], 1),0), vmin=min_val, vmax=max_val, cmap="YlOrRd")
            ax1.set_xticks([])
            ax1.set_yticks([], [])
            ax1.set_xlabel(str(i))
            ax1.set_aspect(scaler)

        plt.tight_layout()
        ax1 = fig.add_subplot(gs[:, n_col - 1])
        fig.colorbar(a, cax=ax1)
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig("post_likeli.svg", transparent="True")
        #plt.show()

    # decoding sleep activity using ONE model
    # ------------------------------------------------------------------------------------------------------------------

    def decode_poisson_hmm_sleep(self, part_to_analyze, template_file_name, use_full_model=False, speed_threshold=None,
                                 plot_for_control=False, return_results=False, cells_to_use="stable",
                                 classification_method="std"):
        """
        decodes sleep activity using pHMM modes using provided model

        :param part_to_analyze: which part to analyze ("rem" or "nrem")
        :type part_to_analyze: str
        :param template_file_name: poisson model file used for decoding
        :type template_file_name: str
        :param use_full_model: whether to use viterbi/posterior prob (True) or only max. likelihoods (False)
                               for decoding
        :type use_full_model: bool
        :param speed_threshold: whether to use defined speed threshold (if None: default speed threshold is used)
        :type speed_threshold: float or None
        :param plot_for_control: plotting intermediate results
        :type plot_for_control: bool
        :param return_results: return results (True) or only save them (False)
        :type return_results: bool
        :param classification_method: sleep classification method ("std" --> default, "k_means")
        :type classification_method: str
        :return: likelihood, event_times, swr_to_nrem, most_likely_state_sequence
        :rtype: np.array, np.array, np.array, np.array

        """

        # # load speed threshold for REM/NREM from parameter file if not provided
        # if speed_threshold is None:
        #     speed_threshold = self.session_params.sleep_phase_speed_threshold
        #
        # if use_full_model:
        #     if only_stable_cells:
        #         result_dir = "phmm_decoding/full_model/stable_cells_"+self.params.stable_cell_method
        #     else:
        #         result_dir = "phmm_decoding/full_model"
        # else:
        #     if only_stable_cells:
        #         result_dir = "phmm_decoding/stable_cells_"+self.params.stable_cell_method
        #     else:
        #         result_dir = "phmm_decoding"

        result_dir = "phmm_decoding"

        result_file_name = self.session_name +"_"+self.experiment_phase_id + "_"+\
                        part_to_analyze+"_"+ self.cell_type+"_one_model.npy"

        # check if results
        # --------------------------------------------------------------------------------------------------------------
        if os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + result_file_name):
            print(" - RESULTS EXIST ALREADY -- USING EXISTING RESULTS\n")
        else:
            # if PRE and/or POST do not exist yet --> compute results

            # check if SWR or REM phases are supposed to be analyzed

            if part_to_analyze == "all_swr":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # compute #spike binning for each event --> TODO: implement sliding window!
                event_spike_rasters, event_spike_window_lenghts = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)
                # do not need this info here (all SWR)
                swr_to_nrem = None

            elif part_to_analyze == "nrem":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # only select SWR during nrem phases
                # get nrem phases in seconds
                n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                         classification_method=classification_method)
                swr_in_n_rem = np.zeros(event_times.shape[0])
                swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
                for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                    n_rem_start = n_rem_phase[0]
                    n_rem_end = n_rem_phase[1]
                    for i, e_t in enumerate(event_times):
                        event_start = e_t[0]
                        event_end = e_t[1]
                        if (n_rem_start < event_start) and (event_end < n_rem_end):
                            swr_in_n_rem[i] += 1
                            swr_in_which_n_rem[n_rem_phase_id, i] = 1

                event_times = event_times[swr_in_n_rem == 1]
                # assignment: which SWR belongs to which nrem phase (for plotting)
                swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

                print(" - "+str(event_times.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:, 1]       # compute #spike binning for each event --> TODO: implement sliding window!
                event_spike_rasters, event_spike_window_lenghts = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

            elif part_to_analyze == "rem":
                # get rem intervals in seconds
                event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                                   classification_method=classification_method)

                print(" - "+str(event_times.shape[0])+" REM phases found (speed thr.: "+str(speed_threshold)+")\n")

                # compute #spike binning for each event --> TODO: implement sliding window!
                event_spike_rasters, event_spike_window_lenghts = \
                    PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:, 1]

            print(" - DECODING SLEEP ACTIVITY USING " + template_file_name + " ...\n")
            # load pHMM model
            with open(self.params.pre_proc_dir + "phmm/" + template_file_name + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)
            # get means of model (lambdas) for decoding
            mode_means = model_dic.means_

            time_bin_size_encoding = model_dic.time_bin_size

            # check if const. #spike bins are correct for the loaded compression factor
            if not self.params.spikes_per_bin == 12:
                raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                "BUT CURRENT #SPIKES PER BIN != 12")

            # load correct compression factor (as defined in parameter file of the session)
            if time_bin_size_encoding == 0.01:
                compression_factor = np.round(self.params.sleep_compression_factor_12spikes_100ms * 10, 3)
            elif time_bin_size_encoding == 0.1:
                compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
            else:
                raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

            if cells_to_use == "stable":
                cell_selection = "custom"
                # load cell ids of stable cells
                # get stable, decreasing, increasing cells
                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                          "rb") as f:
                    class_dic = pickle.load(f)

                cells_ids = class_dic["stable_cell_ids"]

            elif cells_to_use == "increasing":
                cell_selection = "custom"
                # load cell ids of stable cells

                if self.params.stable_cell_method == "k_means":
                    # load only stable cells
                    with open(self.params.pre_proc_dir + "cell_classification/" +
                              self.params.session_name + "_k_means.pickle", "rb") as f:
                        class_dic = pickle.load(f)
                    cells_ids = class_dic["increase_cell_ids"]

                elif self.params.stable_cell_method == "mean_firing_awake":
                    # load only stable cells
                    with open(self.params.pre_proc_dir + "cell_classification/" +
                              self.params.session_name + "_mean_firing_awake.pickle", "rb") as f:
                        class_dic = pickle.load(f)

                    cells_ids = class_dic["increase_cell_ids"]

                else:
                    raise Exception("NOT IMPLEMENTED YET!!!")

            elif cells_to_use == "decreasing":
                cell_selection = "custom"
                # load cell ids of stable cells

                if self.params.stable_cell_method == "k_means":
                    # load only stable cells
                    with open(self.params.pre_proc_dir + "cell_classification/" +
                              self.params.session_name + "_k_means.pickle", "rb") as f:
                        class_dic = pickle.load(f)
                    cells_ids = class_dic["decrease_cell_ids"]

                elif self.params.stable_cell_method == "mean_firing_awake":
                    # load only stable cells
                    with open(self.params.pre_proc_dir + "cell_classification/" +
                              self.params.session_name + "_mean_firing_awake.pickle", "rb") as f:
                        class_dic = pickle.load(f)

                    cells_ids = class_dic["decrease_cell_ids"]

                else:
                    raise Exception("NOT IMPLEMENTED YET!!!")

            elif cells_to_use == "all":
                cell_selection = "all"
                cells_ids = np.empty(0)

            most_likely_state_sequence = []
            if use_full_model:
                results_list = []
                for spike_raster in event_spike_rasters:
                    # need to increase #spikes by inverse of compression factor
                    if spike_raster.shape[1] == 0:
                        continue
                    spike_raster = (spike_raster * 1/compression_factor).T
                    results_list.append(model_dic.predict_proba(spike_raster))
                    most_likely_state_sequence.append(model_dic.predict(spike_raster))

            else:
                results_list = decode_using_phmm_modes(mode_means=mode_means,
                                                       event_spike_rasters=event_spike_rasters,
                                                       compression_factor=compression_factor,
                                                       cell_selection=cell_selection, cells_to_use=cells_ids)

            # saving results
            # --------------------------------------------------------------------------------------------------
            # create dictionary with results
            result_post = {
                "results_list": results_list,
                "event_times": event_times,
                "swr_to_nrem": swr_to_nrem,
                "most_likely_state_sequence": most_likely_state_sequence
            }
            outfile = open(self.params.pre_proc_dir + result_dir +"/" + result_file_name, 'wb')
            pickle.dump(result_post, outfile)
            print("  - SAVED NEW RESULTS!\n")

        if return_results:

            while True:
                # load decoded maps
                try:
                    result = pickle.load(open(self.params.pre_proc_dir+result_dir + "/" + result_file_name,"rb"))
                    break
                except:
                    continue

            likelihood = result["results_list"]
            event_times = result["event_times"]
            swr_to_nrem = result["swr_to_nrem"]
            try:
                most_likely_state_sequence = result["most_likely_state_sequence"]
            except:
                most_likely_state_sequence = []

            return likelihood, event_times, swr_to_nrem, most_likely_state_sequence

    def decode_phmm_one_model_cell_subset(self, part_to_analyze, template_file_name, speed_threshold=None,
                              plot_for_control=False, return_results=True, file_extension="PRE",
                              cells_to_use="stable", shuffling=False, cell_selection="all",
                              compute_spike_bins_with_subsets=True, sleep_classification_method="std"):
        """
        decodes sleep activity using pHMM modes/spatial bins from ising model from before and after awake activity and
        computes similarity measure

        @param part_to_analyze: which data to analyse ("all_swr", "rem", "nrem")
        @type part_to_analyze: str
        @param speed_threshold: filter for rem/nrem phases --> all phases with speeds above this threshold are
        neglected, if you don't want to use a speed_threshold set to None.
        @type speed_threshold: int
        @param compression_factor: scaling factor between sleep activity and awake activity --> if awake
        model was build with constant time bin size (e.g. 100ms): compute how many spikes on average happen during this
        window size (e.g. mean = 30), check how many #spikes you used for constant #spike binning (e.g. 12) for the
        sleep data --> awake_activity_scaling_factor = 12/30 --> = 0.4
        if None --> is loaded from session specific parameter file
        @type compression_factor: float
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param sleep_classification_method: which sleep classification to use --> std: Jozsef's script, k_mean: Juan's script
        @type sleep_classification_method: str
        @param cells_to_use: which cells to use ("all", "stable", "inc", "dec")
        @type cells_to_use: str
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        # load speed threshold for REM/NREM from parameter file if not provided
        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using PHMM modes & std. "
                                                                          "sleep classification...\n")
        if cells_to_use == "stable":
            result_dir = "phmm_decoding/stable_subset/"
        if cells_to_use == "increasing":
            result_dir = "phmm_decoding/increasing_subset/"
        if cells_to_use == "decreasing":
            result_dir = "phmm_decoding/decreasing_subset/"

        result_file_name = self.session_name +"_"+self.experiment_phase_id + "_"+\
                        part_to_analyze+"_"+ self.cell_type+"_"+file_extension

        # Check if results exist already
        # ----------------------------------------------------------------------------------------------------------
        if os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + result_file_name):
            print(" - Result exists already (" + result_file_name + ")\n")

        else:
            print(" - Decoding sleep ("+ part_to_analyze +") "+ template_file_name + ", using "+cells_to_use
                  + " cells ...\n")

            if cells_to_use == "stable":
                # load cell ids of stable cells
                # get stable, decreasing, increasing cells
                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                          "rb") as f:
                    class_dic = pickle.load(f)

                cells_ids = class_dic["stable_cell_ids"]

            elif cells_to_use == "increasing":
                # load cell ids of stable cells

                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                          "rb") as f:
                    class_dic = pickle.load(f)

                cells_ids = class_dic["increase_cell_ids"]

            elif cells_to_use == "decreasing":

                # load cell ids of stable cells

                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                          "rb") as f:
                    class_dic = pickle.load(f)

                cells_ids = class_dic["decrease_cell_ids"]

            # check if SWR or REM phases are supposed to be analyzed

            if part_to_analyze == "all_swr":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # compute #spike binning for each event --> TODO: implement sliding window!
                if compute_spike_bins_with_subsets:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                          cell_ids=cells_ids)
                else:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)
                # do not need this info here (all SWR)
                swr_to_nrem = None

            elif part_to_analyze == "nrem":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # only select SWR during nrem phases
                # get nrem phases in seconds
                n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                         classification_method=sleep_classification_method)
                swr_in_n_rem = np.zeros(event_times.shape[0])
                swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
                for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                    n_rem_start = n_rem_phase[0]
                    n_rem_end = n_rem_phase[1]
                    for i, e_t in enumerate(event_times):
                        event_start = e_t[0]
                        event_end = e_t[1]
                        if (n_rem_start < event_start) and (event_end < n_rem_end):
                            swr_in_n_rem[i] += 1
                            swr_in_which_n_rem[n_rem_phase_id, i] = 1

                event_times = event_times[swr_in_n_rem == 1]
                # assignment: which SWR belongs to which nrem phase (for plotting)
                swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

                print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:,1]
                # compute #spike binning for each event --> TODO: implement sliding window!

                event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                          cell_ids=cells_ids)

            elif part_to_analyze == "rem":
                # get rem intervals in seconds
                event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                                   classification_method=sleep_classification_method)

                print(" - " + str(event_times.shape[0]) + " REM phases found (speed thr.: " + str(
                    speed_threshold) + ")\n")

                # compute #spike binning for each event --> TODO: implement sliding window!

                event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                          cell_ids=cells_ids)


                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:, 1]

            else:
                raise Exception("Provided part_to_analyze is not defined!")

            # plot raster and detected SWR, example spike rasters from SWRs
            if plot_for_control:
                # ------------------------------------------------------------------------------------------------------
                # plot detected events
                # ------------------------------------------------------------------------------------------------------
                # compute pre processed data
                self.compute_raster_speed_loc()
                to_plot = np.random.randint(0, start_times.shape[0], 5)
                for i in to_plot:
                    if cells_to_use == "all":
                        raster_for_plotting = self.raster
                    else:
                        raster_for_plotting =self.raster[cells_ids, :]
                    plt.imshow(raster_for_plotting, interpolation='nearest', aspect='auto')
                    plt.vlines(start_times[i] / self.params.time_bin_size, 0, raster_for_plotting.shape[0] - 1,
                               colors="r",
                               linewidth=0.5,
                               label="START")
                    plt.vlines(end_times[i] / self.params.time_bin_size, 0, raster_for_plotting.shape[0] - 1,
                               colors="g",
                               linewidth=0.5, label="END")
                    plt.xlim(start_times[i] / self.params.time_bin_size - 10,
                             end_times[i] / self.params.time_bin_size + 10)
                    plt.ylabel("CELL IDS")
                    plt.xlabel("TIME BINS / " + str(self.params.time_bin_size) + "s")
                    plt.legend()
                    a = plt.colorbar()
                    a.set_label("SPIKES PER BIN (" + str(self.params.time_bin_size) + " s)")
                    plt.title("EVENT ID " + str(i))
                    plt.show()

                    plt.imshow(event_spike_rasters[i], interpolation='nearest', aspect='auto')
                    a = plt.colorbar()
                    a.set_label("SPIKES PER CONST. #SPIKES BIN")
                    plt.xlabel("CONST. #SPIKE POP. VEC. ID")
                    plt.ylabel("CELL ID")
                    plt.title("BINNED EVENT ID " + str(i) + "\n#SPIKES PER BIN: " + str(
                        self.params.spikes_per_bin))
                    plt.show()

                # ------------------------------------------------------------------------------------------------------
                # compute length of constant #spike windows
                # ------------------------------------------------------------------------------------------------------
                event_spike_window_lengths_avg = np.mean(np.concatenate(event_spike_window_lenghts, axis=0))
                event_spike_window_lengths_median = np.median(np.concatenate(event_spike_window_lenghts, axis=0))

                y, x, _ = plt.hist(np.concatenate(event_spike_window_lenghts, axis=0), bins=30)
                plt.xlabel("LENGTH CONST #SPIKES POP. VEC / s")
                plt.ylabel("COUNTS")
                plt.title("POPULATION VECTOR LENGTH")
                plt.vlines(event_spike_window_lengths_avg, 0, y.max(), colors="r", label="MEAN: " +
                                                                                         str(np.round(
                                                                                             event_spike_window_lengths_avg,
                                                                                             2)))
                plt.vlines(event_spike_window_lengths_median, 0, y.max(), colors="b", label="MEDIAN: " +
                                                                                            str(np.round(
                                                                                                event_spike_window_lengths_median,
                                                                                                2)))
                plt.legend()
                plt.show()

            # load pHMM model
            if cells_to_use == "stable":
                with open(self.params.pre_proc_dir + "phmm/stable_cells/" + template_file_name + '.pkl', 'rb') as f:
                    model_dic = pickle.load(f)
            elif cells_to_use == "decreasing":
                with open(self.params.pre_proc_dir + "phmm/decreasing_cells/" + template_file_name + '.pkl', 'rb') as f:
                    model_dic = pickle.load(f)
            # get means of model (lambdas) for decoding
            mode_means = model_dic.means_

            # get time bin size at time of decoding
            time_bin_size_encoding = model_dic.time_bin_size

            # check if const. #spike bins are correct for the loaded compression factor
            if not self.params.spikes_per_bin == 12:
                raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                "BUT CURRENT #SPIKES PER BIN != 12")

            # load correct compression factor (as defined in parameter file of the session)
            if time_bin_size_encoding == 0.01:
                compression_factor = np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
            elif time_bin_size_encoding == 0.1:
                if cells_to_use == "stable":
                    compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms_stable_cells
                elif cells_to_use == "decreasing":
                    compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms_decreasing_cells
            else:
                raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

            # if you want to use different compression factors for PRE/POST
            # if "PRE" in result_file_name:
            #     compression_factor = 0.4
            # elif "POST" in result_file_name:
            #     compression_factor = 0.6

            if shuffling:
                print(" -- STARTED SWAPPING PROCEDURE ...")
                # merge all events
                conc_data = np.hstack(event_spike_rasters)
                nr_swaps = conc_data.shape[1]*10
                for shuffle_id in range(nr_swaps):
                    # select two random time bins
                    t1 = 1
                    t2 = 1
                    while(t1 == t2):
                        t1 = np.random.randint(conc_data.shape[1])
                        t2 = np.random.randint(conc_data.shape[1])
                    # check in both time bins which cells are active
                    act_cells_t1 = np.argwhere(conc_data[:, t1].flatten()>0).flatten()
                    act_cells_t2 = np.argwhere(conc_data[:, t2].flatten()>0).flatten()
                    # find intersect (same cells need to be firing in t1 and t2 in order to exchange
                    # spikes
                    # original code
                    # --------------------------------------------------------------------------------------
                    # cells_firing_in_both = np.intersect1d(act_cells_t1, act_cells_t2)
                    # if cells_firing_in_both.shape[0] > 1:
                    #     # select first cell to swap
                    #     cell_1 = 1
                    #     cell_2 = 1
                    #     while (cell_1 == cell_2):
                    #         cell_1 = np.random.choice(cells_firing_in_both)
                    #         cell_2 = np.random.choice(cells_firing_in_both)
                    #     # do the actual swapping
                    #     conc_data[cell_1, t1] += 1
                    #     conc_data[cell_1, t2] -= 1
                    #     conc_data[cell_2, t1] -= 1
                    #     conc_data[cell_2, t2] += 1

                    if act_cells_t1.shape[0] > 1 and act_cells_t2.shape[0] > 1:
                        # select first cell to swap
                        cell_1 = 1
                        cell_2 = 1
                        while (cell_1 == cell_2):
                            cell_1 = np.random.choice(act_cells_t2)
                            cell_2 = np.random.choice(act_cells_t1)
                        # do the actual swapping
                        conc_data[cell_1, t1] += 1
                        conc_data[cell_1, t2] -= 1
                        conc_data[cell_2, t1] -= 1
                        conc_data[cell_2, t2] += 1

                print(" -- ... DONE!")
                # split data again into list
                event_lengths = [x.shape[1] for x in event_spike_rasters]

                event_spike_rasters_shuffled = []
                start = 0
                for el in event_lengths:
                    event_spike_rasters_shuffled.append(conc_data[:,start:start+el])
                    start = el

                event_spike_rasters = event_spike_rasters_shuffled
            # start with actual decoding
            # ----------------------------------------------------------------------------------------------

            results_list = decode_using_phmm_modes(mode_means=mode_means,
                                                   event_spike_rasters=event_spike_rasters,
                                                   compression_factor=compression_factor,
                                                   cell_selection=cell_selection)

            # plot maps of some SWR for control
            if plot_for_control:
                swr_to_plot = []
                n_swr = 0
                while (len(swr_to_plot) < 10):
                    if results_list[n_swr].shape[0]>0:
                        swr_to_plot.append(n_swr)
                    n_swr += 1

                for swr in swr_to_plot:
                    res = results_list[swr]
                    plt.imshow(res.T, interpolation='nearest', aspect='auto')
                    plt.xlabel("POP.VEC. ID")
                    plt.ylabel("MODE ID")
                    a = plt.colorbar()
                    a.set_label("PROBABILITY")
                    plt.title("PROBABILITY MAP, SWR ID: " + str(swr))
                    plt.show()

            # saving results
            # --------------------------------------------------------------------------------------------------
            # create dictionary with results
            result_post = {
                "results_list": results_list,
                "event_times": event_times,
                "swr_to_nrem": swr_to_nrem
            }
            outfile = open(self.params.pre_proc_dir + result_dir +"/" + result_file_name, 'wb')
            print("  - saving new results ...")
            pickle.dump(result_post, outfile)
            outfile.close()
            print("  - ... done!\n")
        if return_results:

            while True:
                # load decoded maps
                try:
                    result = pickle.load(open(self.params.pre_proc_dir+result_dir + "/" + result_file_name,"rb"))
                    break
                except:
                    continue

            likelihood = result["results_list"]
            event_times = result["event_times"]
            swr_to_nrem = result["swr_to_nrem"]
            try:
                most_likely_state_sequence = result["most_likely_state_sequence"]
            except:
                most_likely_state_sequence = []

            return likelihood, event_times, swr_to_nrem, most_likely_state_sequence

    def decode_phmm_one_model(self, part_to_analyze, template_file_name, template_type ="phmm", speed_threshold=None,
                              plot_for_control=False, return_results=True, sleep_classification_method="std",
                              cells_to_use="all", shuffling=False, compression_factor=None, cell_selection="all",
                              compute_spike_bins_with_subsets=True):
        """
        decodes sleep activity using pHMM modes/spatial bins from ising model from before and after awake activity and
        computes similarity measure

        @param part_to_analyze: which data to analyse ("all_swr", "rem", "nrem")
        @type part_to_analyze: str
        @param speed_threshold: filter for rem/nrem phases --> all phases with speeds above this threshold are
        neglected, if you don't want to use a speed_threshold set to None.
        @type speed_threshold: int
        @param compression_factor: scaling factor between sleep activity and awake activity --> if awake
        model was build with constant time bin size (e.g. 100ms): compute how many spikes on average happen during this
        window size (e.g. mean = 30), check how many #spikes you used for constant #spike binning (e.g. 12) for the
        sleep data --> awake_activity_scaling_factor = 12/30 --> = 0.4
        if None --> is loaded from session specific parameter file
        @type compression_factor: float
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param sleep_classification_method: which sleep classification to use --> std: Jozsef's script, k_mean: Juan's script
        @type sleep_classification_method: str
        @param cells_to_use: which cells to use ("all", "stable", "inc", "dec")
        @type cells_to_use: str
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        # load speed threshold for REM/NREM from parameter file if not provided
        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        if template_type == "phmm":
            if sleep_classification_method == "std":
                print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using PHMM modes & std. "
                                                                                  "sleep classification...\n")
                if cells_to_use == "stable":
                    if shuffling:
                        result_dir = "phmm_decoding/stable_cells_shuffled_"+self.params.stable_cell_method
                    else:
                        result_dir = "phmm_decoding/stable_cells_"+self.params.stable_cell_method
                if cells_to_use == "increasing":
                    result_dir = "phmm_decoding/inc_cells_"+self.params.stable_cell_method
                if cells_to_use == "decreasing":
                    result_dir = "phmm_decoding/dec_cells_"+self.params.stable_cell_method
                elif cells_to_use == "all":
                    if shuffling:
                        result_dir = "phmm_decoding/spike_shuffled"
                    else:
                        result_dir = "phmm_decoding"
            elif sleep_classification_method == "k_means":
                print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using PHMM modes & k-means "
                                                                                  "sleep classification...\n")
                result_dir = "phmm_decoding/k_means_sleep_classification"

        elif template_type == "ising":
            print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using Ising model ...\n")
            result_dir = "ising_glm_decoding"

        result_file_name = self.session_name +"_"+self.experiment_phase_id + "_"+\
                        part_to_analyze+"_"+ self.cell_type+"_one_model.npy"

        # Check if results exist already
        # ----------------------------------------------------------------------------------------------------------
        if os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + result_file_name):
            print(" - Result exists already (" + result_file_name + ")\n")

        else:
            print(" - Decoding sleep ("+ part_to_analyze +") "+ template_file_name + ", using "+cells_to_use
                  + " cells ...\n")

            if cells_to_use == "stable":
                # load cell ids of stable cells
                # get stable, decreasing, increasing cells
                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                          "rb") as f:
                    class_dic = pickle.load(f)

                cells_ids = class_dic["stable_cell_ids"]

            elif cells_to_use == "increasing":
                # load cell ids of stable cells

                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                          "rb") as f:
                    class_dic = pickle.load(f)

                cells_ids = class_dic["increase_cell_ids"]

            elif cells_to_use == "decreasing":

                # load cell ids of stable cells

                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                          "rb") as f:
                    class_dic = pickle.load(f)

                cells_ids = class_dic["decrease_cell_ids"]

            elif cells_to_use == "all":

                cells_ids = None

            # check if SWR or REM phases are supposed to be analyzed

            if part_to_analyze == "all_swr":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # compute #spike binning for each event --> TODO: implement sliding window!
                if compute_spike_bins_with_subsets:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                          cell_ids=cells_ids)
                else:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)
                # do not need this info here (all SWR)
                swr_to_nrem = None

            elif part_to_analyze == "nrem":

                # get SWR timings (in sec) & compute spike rasters (constant #spikes)
                # ------------------------------------------------------------------------------------------------------
                start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

                # convert to one array for event_spike_binning
                event_times = np.vstack((start_times, end_times, peak_times)).T

                # only select SWR during nrem phases
                # get nrem phases in seconds
                n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                         classification_method=sleep_classification_method)
                swr_in_n_rem = np.zeros(event_times.shape[0])
                swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
                for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
                    n_rem_start = n_rem_phase[0]
                    n_rem_end = n_rem_phase[1]
                    for i, e_t in enumerate(event_times):
                        event_start = e_t[0]
                        event_end = e_t[1]
                        if (n_rem_start < event_start) and (event_end < n_rem_end):
                            swr_in_n_rem[i] += 1
                            swr_in_which_n_rem[n_rem_phase_id, i] = 1

                event_times = event_times[swr_in_n_rem == 1]
                # assignment: which SWR belongs to which nrem phase (for plotting)
                swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

                print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:,1]
                # compute #spike binning for each event --> TODO: implement sliding window!
                if compute_spike_bins_with_subsets:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                          cell_ids=cells_ids)
                else:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

            elif part_to_analyze == "rem":
                # get rem intervals in seconds
                event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold,
                                                   classification_method=sleep_classification_method)

                print(" - " + str(event_times.shape[0]) + " REM phases found (speed thr.: " + str(
                    speed_threshold) + ")\n")

                # compute #spike binning for each event --> TODO: implement sliding window!
                if compute_spike_bins_with_subsets:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                          cell_ids=cells_ids)
                else:
                    event_spike_rasters, event_spike_window_lenghts = \
                        PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                        whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

                # do not need this info here
                swr_to_nrem = None

                start_times = event_times[:, 0]
                end_times = event_times[:, 1]

            else:
                raise Exception("Provided part_to_analyze is not defined!")

            # plot raster and detected SWR, example spike rasters from SWRs
            if plot_for_control:
                # ------------------------------------------------------------------------------------------------------
                # plot detected events
                # ------------------------------------------------------------------------------------------------------
                # compute pre processed data
                self.compute_raster_speed_loc()
                to_plot = np.random.randint(0, start_times.shape[0], 5)
                for i in to_plot:
                    if cells_to_use == "all":
                        raster_for_plotting = self.raster
                    else:
                        raster_for_plotting =self.raster[cells_ids, :]
                    plt.imshow(raster_for_plotting, interpolation='nearest', aspect='auto')
                    plt.vlines(start_times[i] / self.params.time_bin_size, 0, raster_for_plotting.shape[0] - 1,
                               colors="r",
                               linewidth=0.5,
                               label="START")
                    plt.vlines(end_times[i] / self.params.time_bin_size, 0, raster_for_plotting.shape[0] - 1,
                               colors="g",
                               linewidth=0.5, label="END")
                    plt.xlim(start_times[i] / self.params.time_bin_size - 10,
                             end_times[i] / self.params.time_bin_size + 10)
                    plt.ylabel("CELL IDS")
                    plt.xlabel("TIME BINS / " + str(self.params.time_bin_size) + "s")
                    plt.legend()
                    a = plt.colorbar()
                    a.set_label("SPIKES PER BIN (" + str(self.params.time_bin_size) + " s)")
                    plt.title("EVENT ID " + str(i))
                    plt.show()

                    plt.imshow(event_spike_rasters[i], interpolation='nearest', aspect='auto')
                    a = plt.colorbar()
                    a.set_label("SPIKES PER CONST. #SPIKES BIN")
                    plt.xlabel("CONST. #SPIKE POP. VEC. ID")
                    plt.ylabel("CELL ID")
                    plt.title("BINNED EVENT ID " + str(i) + "\n#SPIKES PER BIN: " + str(
                        self.params.spikes_per_bin))
                    plt.show()

                # ------------------------------------------------------------------------------------------------------
                # compute length of constant #spike windows
                # ------------------------------------------------------------------------------------------------------
                event_spike_window_lengths_avg = np.mean(np.concatenate(event_spike_window_lenghts, axis=0))
                event_spike_window_lengths_median = np.median(np.concatenate(event_spike_window_lenghts, axis=0))

                y, x, _ = plt.hist(np.concatenate(event_spike_window_lenghts, axis=0), bins=30)
                plt.xlabel("LENGTH CONST #SPIKES POP. VEC / s")
                plt.ylabel("COUNTS")
                plt.title("POPULATION VECTOR LENGTH")
                plt.vlines(event_spike_window_lengths_avg, 0, y.max(), colors="r", label="MEAN: " +
                                                                                         str(np.round(
                                                                                             event_spike_window_lengths_avg,
                                                                                             2)))
                plt.vlines(event_spike_window_lengths_median, 0, y.max(), colors="b", label="MEDIAN: " +
                                                                                            str(np.round(
                                                                                                event_spike_window_lengths_median,
                                                                                                2)))
                plt.legend()
                plt.show()

            if template_type == "phmm":
                # load pHMM model
                with open(self.params.pre_proc_dir + "phmm/" + template_file_name + '.pkl', 'rb') as f:
                    model_dic = pickle.load(f)
                # get means of model (lambdas) for decoding
                mode_means = model_dic.means_

                # only select lambdas of cells to be used
                if cells_to_use in ["decreasing", "stable", "increasing"]:
                    mode_means = mode_means[:, cells_ids]

                # get time bin size at time of decoding
                time_bin_size_encoding = model_dic.time_bin_size

                # check if const. #spike bins are correct for the loaded compression factor
                if not self.params.spikes_per_bin == 12:
                    raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                    "BUT CURRENT #SPIKES PER BIN != 12")

                # load correct compression factor (as defined in parameter file of the session)
                if time_bin_size_encoding == 0.01:
                    compression_factor = np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
                elif time_bin_size_encoding == 0.1:
                    compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
                else:
                    raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

                # if you want to use different compression factors for PRE/POST
                # if "PRE" in result_file_name:
                #     compression_factor = 0.4
                # elif "POST" in result_file_name:
                #     compression_factor = 0.6

                if shuffling:
                    print(" -- STARTED SWAPPING PROCEDURE ...")
                    # merge all events
                    conc_data = np.hstack(event_spike_rasters)
                    nr_swaps = conc_data.shape[1]*10
                    for shuffle_id in range(nr_swaps):
                        # select two random time bins
                        t1 = 1
                        t2 = 1
                        while(t1 == t2):
                            t1 = np.random.randint(conc_data.shape[1])
                            t2 = np.random.randint(conc_data.shape[1])
                        # check in both time bins which cells are active
                        act_cells_t1 = np.argwhere(conc_data[:, t1].flatten()>0).flatten()
                        act_cells_t2 = np.argwhere(conc_data[:, t2].flatten()>0).flatten()
                        # find intersect (same cells need to be firing in t1 and t2 in order to exchange
                        # spikes
                        # original code
                        # --------------------------------------------------------------------------------------
                        # cells_firing_in_both = np.intersect1d(act_cells_t1, act_cells_t2)
                        # if cells_firing_in_both.shape[0] > 1:
                        #     # select first cell to swap
                        #     cell_1 = 1
                        #     cell_2 = 1
                        #     while (cell_1 == cell_2):
                        #         cell_1 = np.random.choice(cells_firing_in_both)
                        #         cell_2 = np.random.choice(cells_firing_in_both)
                        #     # do the actual swapping
                        #     conc_data[cell_1, t1] += 1
                        #     conc_data[cell_1, t2] -= 1
                        #     conc_data[cell_2, t1] -= 1
                        #     conc_data[cell_2, t2] += 1

                        if act_cells_t1.shape[0] > 1 and act_cells_t2.shape[0] > 1:
                            # select first cell to swap
                            cell_1 = 1
                            cell_2 = 1
                            while (cell_1 == cell_2):
                                cell_1 = np.random.choice(act_cells_t2)
                                cell_2 = np.random.choice(act_cells_t1)
                            # do the actual swapping
                            conc_data[cell_1, t1] += 1
                            conc_data[cell_1, t2] -= 1
                            conc_data[cell_2, t1] -= 1
                            conc_data[cell_2, t2] += 1

                    print(" -- ... DONE!")
                    # split data again into list
                    event_lengths = [x.shape[1] for x in event_spike_rasters]

                    event_spike_rasters_shuffled = []
                    start = 0
                    for el in event_lengths:
                        event_spike_rasters_shuffled.append(conc_data[:,start:start+el])
                        start = el

                    event_spike_rasters = event_spike_rasters_shuffled
                # start with actual decoding
                # ----------------------------------------------------------------------------------------------

                print(" - DECODING USING "+ cells_to_use + " CELLS")

                if not compute_spike_bins_with_subsets:
                    # event_spike_raster size and mode_means size don't match --> need to go through event_spike_rasters
                    # to only select cells that are wanted
                    event_spike_rasters_modified = []
                    for ev_r in event_spike_rasters:
                        event_spike_rasters_modified.append(ev_r[cells_ids, :])

                    event_spike_rasters = event_spike_rasters_modified

                results_list = decode_using_phmm_modes(mode_means=mode_means,
                                                       event_spike_rasters=event_spike_rasters,
                                                       compression_factor=compression_factor,
                                                       cell_selection=cell_selection)
            elif template_type == "ising":
                # load ising template
                with open(self.params.pre_proc_dir + 'awake_ising_maps/' + template_file_name + '.pkl',
                          'rb') as f:
                    model_dic = pickle.load(f)

                # if compression_factor is not provided --> load from parameter file
                if compression_factor is None:
                    # get time_bin_size of encoding
                    time_bin_size_encoding = model_dic["time_bin_size"]

                    # check if const. #spike bins are correct for the loaded compression factor
                    if not self.params.spikes_per_bin == 12:
                        raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                        "BUT CURRENT #SPIKES PER BIN != 12")

                    # load correct compression factor (as defined in parameter file of the session)
                    if time_bin_size_encoding == 0.01:
                        compression_factor = \
                            np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
                    elif time_bin_size_encoding == 0.1:
                        compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
                    else:
                        raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

                # get template map
                template_map = model_dic["res_map"]

                results_list = decode_using_ising_map(template_map=template_map, event_spike_rasters=event_spike_rasters,
                                                               compression_factor=compression_factor,
                                                               cell_selection="all")

            # plot maps of some SWR for control
            if plot_for_control:
                swr_to_plot = []
                n_swr = 0
                while (len(swr_to_plot) < 10):
                    if results_list[n_swr].shape[0]>0:
                        swr_to_plot.append(n_swr)
                    n_swr += 1

                for swr in swr_to_plot:
                    res = results_list[swr]
                    plt.imshow(res.T, interpolation='nearest', aspect='auto')
                    plt.xlabel("POP.VEC. ID")
                    plt.ylabel("MODE ID")
                    a = plt.colorbar()
                    a.set_label("PROBABILITY")
                    plt.title("PROBABILITY MAP, SWR ID: " + str(swr))
                    plt.show()

            # saving results
            # --------------------------------------------------------------------------------------------------
            # create dictionary with results
            result_post = {
                "results_list": results_list,
                "event_times": event_times,
                "swr_to_nrem": swr_to_nrem
            }
            outfile = open(self.params.pre_proc_dir + result_dir +"/" + result_file_name, 'wb')
            print("  - saving new results ...")
            pickle.dump(result_post, outfile)
            outfile.close()
            print("  - ... done!\n")
        if return_results:

            while True:
                # load decoded maps
                try:
                    result = pickle.load(open(self.params.pre_proc_dir+result_dir + "/" + result_file_name,"rb"))
                    break
                except:
                    continue

            likelihood = result["results_list"]
            event_times = result["event_times"]
            swr_to_nrem = result["swr_to_nrem"]
            try:
                most_likely_state_sequence = result["most_likely_state_sequence"]
            except:
                most_likely_state_sequence = []

            return likelihood, event_times, swr_to_nrem, most_likely_state_sequence

    def view_decoding_poisson_hmm_sleep_results(self, part_to_analyze="rem", only_stable_cells=False, use_full_model=False):
        """
        loads and plots results of decoding analysis

        :param part_to_analyze: which part to analyze ("rem" or "nrem")
        :type part_to_analyze: str
        :param only_stable_cells: only use stable cells for decoding (True)
        :type only_stable_cells: bool
        :param use_full_model: whether to use viterbi/posterior prob (True) or only max. likelihoods (False)
                               for decoding
        :type use_full_model: bool
        """
        if use_full_model:
            if only_stable_cells:
                result_dir = "phmm_decoding/full_model/stable_cells_" + self.params.stable_cell_method
            else:
                result_dir = "phmm_decoding/full_model"
        else:
            if only_stable_cells:
                result_dir = "phmm_decoding/stable_cells_" + self.params.stable_cell_method
            else:
                result_dir = "phmm_decoding"

        result_file_name = self.params.session_name + "_" + self.experiment_phase_id + "_" + \
                           part_to_analyze + "_" + self.cell_type + "_PRE_POST.npy"

        result = pickle.load(open(self.params.pre_proc_dir + result_dir + "/" + result_file_name, "rb"))

        likelihood = result["results_list"]
        event_times = result["event_times"]
        swr_to_nrem = result["swr_to_nrem"]

        plt.imshow(likelihood[0])
        plt.show()

    def compare_decoding_methods_poisson_hmm(self, part_to_analyze="rem", only_stable_cells=False):
        """
        compare decoding results selecting active mode either by maximum likelihood, the viterbi algorithm or
        maximum posterior probability

        :param part_to_analyze: which sleep phase to decode (either "rem" or "nrem")
        :type part_to_analyze: str
        :param only_stable_cells: whether to only use stable cells
        :type only_stable_cells: bool
        """

        result_dir_full = "phmm_decoding/full_model"
        result_dir = "phmm_decoding/"

        # find result file using current session name
        result_file_name = self.params.session_name + "_" + self.experiment_phase_id + "_" + \
                           part_to_analyze + "_" + self.cell_type + "_PRE_POST.npy"

        # get result data for full model (viterbi and posterior probability)
        result_full = pickle.load(open(self.params.pre_proc_dir + result_dir_full + "/" + result_file_name, "rb"))
        # get result data using maximum likelihood
        result = pickle.load(open(self.params.pre_proc_dir + result_dir + "/" + result_file_name, "rb"))

        # extract results from dictionaries
        likelihood = result["results_list"]
        post_prob_full = result_full["results_list"]
        seq_full = result_full["most_likely_state_sequence"]

        # normalize likelihood for visualization
        likeli_norm = likelihood[0]/np.sum(likelihood[0], axis=1, keepdims=True)
        plt.subplot(2,1,1)
        plt.imshow(likeli_norm.T)
        a = plt.colorbar()
        a.set_label("LIKELIHOOD (NORM)")
        plt.title("USING ONLY LAMBDA VECTORS")
        plt.ylabel("MODE ID")
        plt.subplot(2,1,2)
        plt.imshow(post_prob_full[0].T)
        a = plt.colorbar()
        a.set_label("POSTERIOR PROB.")
        plt.title("USING FULL MODEL")
        plt.ylabel("MODE ID")
        plt.xlabel("TIME BINS")
        plt.show()

        # plot most likely sequence
        plt.scatter(range(seq_full[0].shape[0]), seq_full[0], color="r", label="FULL")
        plt.scatter(range(seq_full[0].shape[0]),np.argmax(likelihood[0], axis=1), color="b", label="LAMBDAS")
        plt.legend()
        plt.xlabel("TIME BINS")
        plt.ylabel("MODE IDS")
        plt.title("MOST LIKELY SEQUENCE")
        plt.show()

    # compression factor analysis & optimization
    # ------------------------------------------------------------------------------------------------------------------

    def optimize_compression_factor(self, phmm_file_name, result_file):

        print(" - COMPRESSION FACTOR OPTIMIZATION ...\n")

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr()

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem")
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times = event_times[swr_in_n_rem == 1]

        print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

        # compute #spike binning for each event --> TODO: implement sliding window!
        event_spike_rasters, event_spike_window_lenghts = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

        # load pHMM model
        with open(self.params.pre_proc_dir + "phmm/" + phmm_file_name + '.pkl', 'rb') as f:
            model_dic = pickle.load(f)
        # get means of model (lambdas) for decoding
        mode_means = model_dic.means_

        scaling_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        max_med_list = []
        glob_med_list = []
        for awake_activity_scaling_factor in scaling_factors:

            results_list = self.decode_using_phmm_modes(mode_means=mode_means,
                                                        event_spike_rasters=event_spike_rasters,
                                                        awake_activity_scaling_factor=
                                                        awake_activity_scaling_factor)
            prob_arr = np.vstack(results_list)
            prob_arr_flat = prob_arr.flatten()
            prob_arr_z = zscore(prob_arr_flat)

            glob_med_list.append(np.mean(prob_arr_flat[prob_arr_z < 3]))

            max_prob_arr = np.max(prob_arr, axis=1)
            max_prob_arr_z = zscore(max_prob_arr)
            max_med_list.append(np.mean(max_prob_arr[max_prob_arr_z < 3]))

        results = {
            "scaling_factors": scaling_factors,
            "glob_med_list": glob_med_list,
            "max_med_list": max_med_list
        }
        outfile = open(self.params.pre_proc_dir +"compression_optimization/"+result_file+".pkl", 'wb')
        pickle.dump(results, outfile)

    def optimize_compression_factor_plot_results(self, result_file_pre, result_file_post):

        for i, (result_file, name) in enumerate(zip([result_file_pre, result_file_post], ["PRE", "POST"])):
            results = np.load(self.params.pre_proc_dir +"compression_optimization/"+result_file+".pkl",
                              allow_pickle=True)

            scaling_factors = results["scaling_factors"]
            glob_med_list = results["glob_med_list"]
            max_med_list = results["max_med_list"]
            plt.subplot(2,1,i+1)
            plt.plot(scaling_factors, max_med_list, ".-", c="blue", label="mean(max.per.vec)")
            plt.legend()
            plt.ylabel("PROB.")
            plt.grid()
        plt.xlabel("SCALING FACTOR")
        plt.show()

        for i, (result_file, name) in enumerate(zip([result_file_pre, result_file_post], ["PRE", "POST"])):
            results = np.load(self.params.pre_proc_dir +"compression_optimization/"+result_file+".pkl",
                              allow_pickle=True)

            scaling_factors = results["scaling_factors"]
            glob_med_list = results["glob_med_list"]
            max_med_list = results["max_med_list"]
            plt.subplot(2,1,i+1)
            plt.plot(scaling_factors, glob_med_list, ".-", c="r", label="global_mean - "+name)
            plt.legend()
            plt.ylabel("PROB.")
            plt.grid()
        plt.xlabel("SCALING FACTOR")
        plt.show()

    def content_based_memory_drift_optimal_nrem_binning(self, template_type, pre_file_name=None, post_file_name=None,
                                   compression_factor=None, speed_threshold=None, plot_for_control=False,
                                   classification_method="std", cells_to_use="all",
                                   shuffling=False):
        """
        decodes sleep activity using pHMM modes/spatial bins from ising model from before and after awake activity and
        computes similarity measure

        @param template_type: whether to use "phmm" or "ising"
        @type template_type: str
        @param pre_file_name: name of file containing the template from PRE behavioral data --> if = None: file
        defined in session parameter file
        @type pre_file_name: str
        @param post_file_name: name of file containing the template from AFTER/POST awake/behavioral data --> if = None:
        file defined in session parameter file
        @type post_file_name: str
        @param part_to_analyze: which data to analyse ("all_swr", "rem", "nrem")
        @type part_to_analyze: str
        @param compression_factor: scaling factor between sleep activity and awake activity --> if awake
        model was build with constant time bin size (e.g. 100ms): compute how many spikes on average happen during this
        window size (e.g. mean = 30), check how many #spikes you used for constant #spike binning (e.g. 12) for the
        sleep data --> awake_activity_scaling_factor = 12/30 --> = 0.4
        if None --> is loaded from session specific parameter file
        @type compression_factor: float
        @param speed_threshold: filter for rem/nrem phases --> all phases with speeds above this threshold are
        neglected, if you don't want to use a speed_threshold set to None.
        @type speed_threshold: int
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param classification_method: which sleep classification to use --> std: Jozsef's script, k_mean: Juan's script
        @type classification_method: str
        @param cells_to_use: which cells to use ("all", "stable", "inc", "dec")
        @type cells_to_use: str
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        part_to_analyze = "nrem"

        # load speed threshold for REM/NREM from parameter file if not provided
        if speed_threshold is None:
            speed_threshold = self.session_params.sleep_phase_speed_threshold

        if template_type == "phmm":
            print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using PHMM modes ...\n")
            if cells_to_use == "stable":
                if shuffling:
                    result_dir = "phmm_decoding/stable_cells_shuffled_"+self.params.stable_cell_method
                else:
                    result_dir = "phmm_decoding/stable_cells_"+self.params.stable_cell_method
            if cells_to_use == "increasing":
                result_dir = "phmm_decoding/inc_cells_"+self.params.stable_cell_method
            if cells_to_use == "decreasing":
                result_dir = "phmm_decoding/dec_cells_"+self.params.stable_cell_method
            elif cells_to_use == "all":
                result_dir = "phmm_decoding"
            # get template file name from parameter file of session if not provided
            if pre_file_name is None:
                pre_file_name = self.session_params.default_pre_phmm_model
            if post_file_name is None:
                post_file_name = self.session_params.default_post_phmm_model

        elif template_type == "ising":
            print(" - Decoding "+self.experiment_phase+" ("+part_to_analyze + ") using Ising model ...\n")
            result_dir = "ising_glm_decoding"
            # get template file name from parameter file of session if not provided
            if pre_file_name is None:
                pre_file_name = self.session_params.default_pre_ising_model
            if post_file_name is None:
                post_file_name = self.session_params.default_post_ising_model

        if pre_file_name is None or post_file_name is None:
            raise Exception("AT LEAST ONE TEMPLATE FILE WAS NEITHER PROVIDED\n NOR IN SESSION PARAMETER FILE DEFINED")

        # check that template files are from correct session
        # --------------------------------------------------------------------------------------------------------------
        #sess_pre = pre_file_name.split(sep="_")[0]+"_"+pre_file_name.split(sep="_")[1]
        #sess_post = post_file_name.split(sep="_")[0]+"_"+post_file_name.split(sep="_")[1]

        #if not (sess_pre == self.params.session_name and sess_post == self.params.session_name):
        #    raise Exception("TEMPLATE FILE AND CURRENT SESSION DO NOT MATCH!")

        file_name_pre = self.session_name +"_"+self.experiment_phase_id + "_"+\
                        part_to_analyze+"_"+ self.cell_type+"_PRE.npy"
        file_name_post = self.session_name + "_" + self.experiment_phase_id + "_"+\
                         part_to_analyze+"_"+self.cell_type+"_POST.npy"


        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold,
                                                 classification_method=classification_method)
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times = event_times[swr_in_n_rem == 1]
        # assignment: which SWR belongs to which nrem phase (for plotting)
        swr_to_nrem = swr_in_which_n_rem[:, swr_in_n_rem == 1]

        print(" - "+str(event_times.shape[0])+" WERE ASSIGNED TO NREM PHASES\n")

        # do not need this info here
        swr_to_nrem = None

        start_times = event_times[:, 0]
        end_times = event_times[:, 1]       # compute #spike binning for each event --> TODO: implement sliding window!

        for spikes_per_bin in [2, 5, 8, 10, 12, 14]:
            event_spike_rasters, event_spike_window_lenghts = \
                PreProcessSleep(firing_times=self.firing_times, params=self.params,
                                whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1,
                                                                  spikes_per_bin=spikes_per_bin)

            # plot raster and detected SWR, example spike rasters from SWRs
            if plot_for_control:
                # ------------------------------------------------------------------------------------------------------
                # plot detected events
                # ------------------------------------------------------------------------------------------------------
                # compute pre processed data
                self.compute_raster_speed_loc()
                to_plot = np.random.randint(0, start_times.shape[0], 5)
                for i in to_plot:
                    plt.imshow(self.raster, interpolation='nearest', aspect='auto')
                    plt.vlines(start_times[i] / self.params.time_bin_size, 0, self.raster.shape[0] - 1,
                               colors="r",
                               linewidth=0.5,
                               label="START")
                    plt.vlines(end_times[i] / self.params.time_bin_size, 0, self.raster.shape[0] - 1,
                               colors="g",
                               linewidth=0.5, label="END")
                    plt.xlim(start_times[i] / self.params.time_bin_size - 10,
                             end_times[i] / self.params.time_bin_size + 10)
                    plt.ylabel("CELL IDS")
                    plt.xlabel("TIME BINS / "+str(self.params.time_bin_size)+"s")
                    plt.legend()
                    a = plt.colorbar()
                    a.set_label("SPIKES PER BIN (" + str(self.params.time_bin_size) + " s)")
                    plt.title("EVENT ID " + str(i))
                    plt.show()

                    plt.imshow(event_spike_rasters[i], interpolation='nearest', aspect='auto')
                    a = plt.colorbar()
                    a.set_label("SPIKES PER CONST. #SPIKES BIN")
                    plt.xlabel("CONST. #SPIKE POP. VEC. ID")
                    plt.ylabel("CELL ID")
                    plt.title("BINNED EVENT ID " + str(i) + "\n#SPIKES PER BIN: " + str(
                        self.params.spikes_per_bin))
                    plt.show()

                # ------------------------------------------------------------------------------------------------------
                # compute length of constant #spike windows
                # ------------------------------------------------------------------------------------------------------
                event_spike_window_lengths_avg = np.mean(np.concatenate(event_spike_window_lenghts, axis=0))
                event_spike_window_lengths_median = np.median(np.concatenate(event_spike_window_lenghts, axis=0))

                y, x, _ = plt.hist(np.concatenate(event_spike_window_lenghts, axis=0), bins=30)
                plt.xlabel("LENGTH CONST #SPIKES POP. VEC / s")
                plt.ylabel("COUNTS")
                plt.title("POPULATION VECTOR LENGTH")
                plt.vlines(event_spike_window_lengths_avg, 0, y.max(), colors="r", label="MEAN: "+
                                        str(np.round(event_spike_window_lengths_avg,2)))
                plt.vlines(event_spike_window_lengths_median, 0, y.max(), colors="b", label="MEDIAN: "+
                                                str(np.round(event_spike_window_lengths_median,2)))
                plt.legend()
                plt.show()


            result_file_name = file_name_pre
            template_file_name = pre_file_name
            print(" - DECODING SLEEP ACTIVITY USING " + template_file_name + " ...\n")

            if template_type == "phmm":
                # load pHMM model
                with open(self.params.pre_proc_dir + "phmm/" + template_file_name + '.pkl', 'rb') as f:
                    model_dic = pickle.load(f)
                # get means of model (lambdas) for decoding
                mode_means = model_dic.means_

                time_bin_size_encoding = model_dic.time_bin_size

                # load correct compression factor (as defined in parameter file of the session)
                if time_bin_size_encoding == 0.01:
                    compression_factor = np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
                elif time_bin_size_encoding == 0.1:
                    compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
                else:
                    raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")



                cell_selection = "all"
                cells_ids = np.empty(0)

                print(" - DECODING USING "+ cells_to_use + " CELLS")

                results_list = decode_using_phmm_modes(mode_means=mode_means,
                                                       event_spike_rasters=event_spike_rasters,
                                                       compression_factor=compression_factor,
                                                       cell_selection=cell_selection, cells_to_use=cells_ids)

            elif template_type == "ising":
                # load ising template
                with open(self.params.pre_proc_dir + 'awake_ising_maps/' + template_file_name + '.pkl',
                          'rb') as f:
                    model_dic = pickle.load(f)

                # if compression_factor is not provided --> load from parameter file
                if compression_factor is None:
                    # get time_bin_size of encoding
                    time_bin_size_encoding = model_dic["time_bin_size"]

                    # check if const. #spike bins are correct for the loaded compression factor
                    if not self.params.spikes_per_bin == 12:
                        raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                        "BUT CURRENT #SPIKES PER BIN != 12")

                    # load correct compression factor (as defined in parameter file of the session)
                    if time_bin_size_encoding == 0.01:
                        compression_factor = \
                            np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
                    elif time_bin_size_encoding == 0.1:
                        compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
                    else:
                        raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

                # get template map
                template_map = model_dic["res_map"]

                results_list = decode_using_ising_map(template_map=template_map,
                                                               event_spike_rasters=event_spike_rasters,
                                                               compression_factor=compression_factor,
                                                               cell_selection="all")

            all_likelihoods = np.vstack(results_list)
            max_likelihoods = np.max(all_likelihoods, axis=1)

            all_likelihoods = all_likelihoods.flatten()
            all_likelihoods_sorted = np.sort(all_likelihoods)
            max_likelihoods_sorted = np.sort(max_likelihoods)

            p_all = 1. * np.arange(all_likelihoods_sorted.shape[0]) / (all_likelihoods_sorted.shape[0] - 1)
            p_max = 1. * np.arange(max_likelihoods.shape[0]) / (max_likelihoods.shape[0] - 1)

            plt.plot(all_likelihoods_sorted, p_all, label=str(spikes_per_bin))
            # plt.plot(pre_prob_nrem_max_sorted, p_nrem, color="blue", label="NREM")
        plt.gca().set_xscale("log")
        plt.xlabel("All likelihoods per PV")
        plt.ylabel("CDF")
        plt.legend()
        plt.show()

    def check_compression_factor(self,phmm_file_pre, phmm_file_post):

        print(" - COMPRESSION FACTOR OPTIMIZATION ...\n")

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr()

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem")
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times = event_times[swr_in_n_rem == 1]

        print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

        # compute #spike binning for each event --> TODO: implement sliding window!
        event_spike_rasters, event_spike_window_lenghts = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

        # load pHMM model
        with open(self.params.pre_proc_dir + "phmm/" + phmm_file_pre + '.pkl', 'rb') as f:
            model_dic = pickle.load(f)
        # get means of model (lambdas) for decoding
        mode_means = model_dic.means_

        for awake_activity_scaling_factor in [0.4]:

            results_list = self.decode_using_phmm_modes(mode_means=mode_means,
                                                        event_spike_rasters=event_spike_rasters,
                                                        awake_activity_scaling_factor=
                                                        awake_activity_scaling_factor)
            pre_prob_arr = np.vstack(results_list)

        # load pHMM model
        with open(self.params.pre_proc_dir + "phmm/" + phmm_file_post + '.pkl', 'rb') as f:
            model_dic = pickle.load(f)
        # get means of model (lambdas) for decoding
        mode_means = model_dic.means_

        post_prob_list = []
        for awake_activity_scaling_factor in [0.4, 0.5]:

            results_list = self.decode_using_phmm_modes(mode_means=mode_means,
                                                        event_spike_rasters=event_spike_rasters,
                                                        awake_activity_scaling_factor=
                                                        awake_activity_scaling_factor)
            post_prob_list.append(np.vstack(results_list))

        post_0_4 = post_prob_list[0]
        post_0_5 = post_prob_list[1]

        results = {
            "post_0_4": post_0_4,
            "post_0_5": post_0_5,
            "pre_prob_arr": pre_prob_arr
        }
        outfile = open(self.params.pre_proc_dir + "compression_optimization/" + "test" + ".pkl", 'wb')
        pickle.dump(results, outfile)

    def check_compression_factor_plot(self, result_file_name):
        results = np.load(self.params.pre_proc_dir +"compression_optimization/"+result_file_name+".pkl",
                          allow_pickle=True)

        pre = results["pre_prob_arr"]
        post_04 = results["post_0_4"]
        post_05 = results["post_0_5"]

        pre_max = np.max(pre, axis=1)
        post_04_max = np.max(post_04, axis=1)
        post_05_max = np.max(post_05, axis=1)


        pre_post_ratio_04 = (post_04_max - pre_max)/(post_04_max + pre_max)
        pre_post_ratio_05 = (post_05_max - pre_max)/(post_05_max + pre_max)
        p_p_r_s_04 = moving_average(a=pre_post_ratio_04, n=40)
        p_p_r_s_05 = moving_average(a=pre_post_ratio_05, n=40)

        plt.plot(p_p_r_s_04, label="POST_04")
        plt.plot(p_p_r_s_05, c="r", alpha=0.5, label="POST_05")
        plt.legend()
        plt.xlabel("POP. VEC ID")
        plt.ylabel("PRE_POST_RATIO")
        plt.show()

        plt.plot(post_04_max, label="POST_04, AVG: "+str(np.mean(post_04_max)))
        plt.plot(post_05_max, c="r", alpha=0.5, label="POST_05, AVG: "+str(np.mean(post_05_max)))
        plt.title("MAX. POST LIKELIHOOD")
        plt.xlabel("POP. VEC. ID")
        plt.ylabel("MAX. LIKELIHOOD")
        plt.legend()
        plt.show()
        nom_04 = post_04_max - pre_max
        nom_05 = post_05_max - pre_max

        plt.plot(pre_max, c="r", alpha=0.5, label="PRE, AVG: "+str(np.mean(pre_max)))
        plt.title("MAX. POST LIKELIHOOD")
        plt.xlabel("POP. VEC. ID")
        plt.ylabel("MAX. LIKELIHOOD")
        plt.legend()
        plt.show()
        exit()


        # exit()
        # plt.plot(post_04_max - pre_max, label="POST_04, AVG: "+str(np.mean(post_04_max - pre_max)))
        # plt.plot(post_05_max - pre_max, c="r", alpha=0.5, label="POST_05, AVG: "+str(np.mean(post_05_max - pre_max)))
        # plt.title("NOMINATOR")
        # plt.legend()
        # plt.show()

        denom_04 = post_04_max + pre_max
        denom_05 = post_05_max + pre_max

        a = nom_04/denom_04
        ne = moving_average(a=a, n=40)
        plt.plot(ne, label="POST_04")
        a = nom_05/denom_04
        ne = moving_average(a=a, n=40)
        plt.plot(ne, label="POST_05",c="r", alpha=0.5)
        plt.legend()
        plt.show()

        # exit()
        #
        # a = denom_04-denom_05
        # plt.hist(a, bins=100)
        # plt.show()
        # exit()

        plt.hist(nom_04, bins=10000, density=True, label="POST_04")
        plt.hist(nom_05, bins=10000, color="red", alpha=0.5, density=True, label="POST_05")
        plt.xlim(-0.05e-13, 0.1e-13)
        plt.legend()
        plt.xlabel("NOM")
        plt.ylabel("DENSITY")
        plt.title("NOMINATOR")
        plt.show()
        exit()

        plt.plot(post_04_max + pre_max, label="POST_04, AVG: "+str(np.mean(post_04_max + pre_max)))
        plt.plot(post_05_max + pre_max, c="r", alpha=0.5, label="POST_05, AVG: "+str(np.mean(post_05_max + pre_max)))
        plt.title("DENOMINATOR")
        plt.legend()
        plt.show()

        plt.plot((post_04_max - pre_max)/(post_05_max - pre_max))
        plt.yscale("log")
        plt.title("NOMINATOR")
        plt.show()

        plt.plot((post_04_max + pre_max)/((post_05_max + pre_max)*0.01))
        plt.yscale("log")
        plt.title("DENOMINATOR")
        plt.show()

    def nrem_vs_rem_compression(self):

        speed_threshold = self.session_params.sleep_phase_speed_threshold

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # ------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr()

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times, peak_times)).T

        # only select SWR during nrem phases
        # get nrem phases in seconds
        n_rem_time_stamps = self.get_sleep_phase(sleep_phase="nrem", speed_threshold=speed_threshold)
        swr_in_n_rem = np.zeros(event_times.shape[0])
        swr_in_which_n_rem = np.zeros((n_rem_time_stamps.shape[0], event_times.shape[0]))
        for n_rem_phase_id, n_rem_phase in enumerate(n_rem_time_stamps):
            n_rem_start = n_rem_phase[0]
            n_rem_end = n_rem_phase[1]
            for i, e_t in enumerate(event_times):
                event_start = e_t[0]
                event_end = e_t[1]
                if (n_rem_start < event_start) and (event_end < n_rem_end):
                    swr_in_n_rem[i] += 1
                    swr_in_which_n_rem[n_rem_phase_id, i] = 1

        event_times = event_times[swr_in_n_rem == 1]

        print(" - " + str(event_times.shape[0]) + " WERE ASSIGNED TO NREM PHASES\n")

        # compute #spike binning for each event --> TODO: implement sliding window!
        _, event_spike_window_lenghts_nrem = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

        event_spike_window_lenghts_nrem = np.hstack(event_spike_window_lenghts_nrem)
        # get rem intervals in seconds
        event_times = self.get_sleep_phase(sleep_phase="rem", speed_threshold=speed_threshold)

        print(" - " + str(event_times.shape[0]) + " REM phases found (speed thr.: " + str(speed_threshold) + ")\n")

        # compute #spike binning for each event --> TODO: implement sliding window!
        _, event_spike_window_lenghts_rem = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)

        event_spike_window_lenghts_rem = np.hstack(event_spike_window_lenghts_rem)

        plt.hist(event_spike_window_lenghts_nrem,
                 color="blue", label="NREM, MEDIAN = "+str(np.median(event_spike_window_lenghts_nrem)), density=True)
        plt.hist(event_spike_window_lenghts_rem,
                 color="red", label="REM, MEDIAN = "+str(np.median(event_spike_window_lenghts_rem)), density=True)
        plt.xlabel("DURATION OF 12 SPIKE BINS / s")
        plt.ylabel("DENSITY")
        plt.legend()
        plt.show()

    # predicting time bin progression
    # ------------------------------------------------------------------------------------------------------------------
    def predict_bin_progression(self, time_bin_size=None, norm_firing=False):
        """
        analysis of drift using population vectors

        @param time_bin_size: which time bin size to use for prediction --> if None:
        standard time bin size from parameter object is used
        @type time_bin_size: float
        """
        # compute pre processed data
        self.compute_raster_speed_loc()

        if time_bin_size is None:
            time_bin_size = self.params.time_bin_size

        x = self.raster

        if norm_firing:
            x = (x - np.min(x, axis=1, keepdims=True)) / \
                (np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True))

        # plot activation matrix (matrix of population vectors)
        plt.imshow(x, vmin=0, vmax=x.max(), cmap='jet', aspect='auto')
        plt.imshow(x, interpolation='nearest', aspect='auto', cmap="jet",
                   extent=[0, x.shape[1], x.shape[0] - 0.5, 0.5])
        plt.ylabel("CELL ID")
        plt.xlabel("TIME BINS")
        plt.title("SPIKE BINNING RASTER")
        a = plt.colorbar()
        a.set_label("# SPIKES")
        plt.show()

        y = np.arange(x.shape[1])

        new_ml = MlMethodsOnePopulation()
        new_ml.ridge_time_bin_progress(x=x, y=y, new_time_bin_size=time_bin_size, plotting=True)

    def predict_time_bin_pop_vec_non_hse(self, normalize_firing_rates=True):
        # --------------------------------------------------------------------------------------------------------------
        # analysis of drift using population vectors
        #
        # args:   - normalize_firing_rates, bool: yes if true
        # --------------------------------------------------------------------------------------------------------------
        # compute pre processed data
        self.compute_raster_speed_loc()

        x = self.raster

        # find high synchrony events
        ind_hse = find_hse(x=x)

        # remove high synchrony events
        x_wo_hse = np.delete(x, ind_hse, axis=1)

        for new_time_bin_size in [0.1, 1, 5]:

            # down/up sample data
            time_bin_scaler = int(new_time_bin_size / self.params.time_bin_size)

            new_raster = np.zeros((x_wo_hse.shape[0], int(x_wo_hse.shape[1] / time_bin_scaler)))

            # down sample spikes by combining multiple bins
            for i in range(new_raster.shape[1]):
                new_raster[:, i] = np.sum(x_wo_hse[:, (i * time_bin_scaler): ((1 + i) * time_bin_scaler)], axis=1)

            plt.imshow(new_raster,  interpolation='nearest', aspect='auto')
            plt.colorbar()
            plt.show()


            if normalize_firing_rates:
                x = (new_raster-np.min(new_raster, axis=1, keepdims=True))/\
                    (np.max(new_raster, axis=1, keepdims=True)-np.min(new_raster, axis=1, keepdims=True))

            else:
                x = new_raster

            plt.imshow(x,  interpolation='nearest', aspect='auto')
            plt.colorbar()
            plt.show()

            y = np.arange(x.shape[1])*new_time_bin_size

            new_ml = MlMethodsOnePopulation()
            new_ml.ridge_time_bin_progress(x=x, y=y, new_time_bin_size=new_time_bin_size)

    def hse_predict_time_bin(self):
        # --------------------------------------------------------------------------------------------------------------
        # analysis of hse drift using population vectors
        #
        # args:
        # --------------------------------------------------------------------------------------------------------------

        # compute pre processed data
        self.compute_raster_speed_loc()

        x = self.raster

        # find high synchrony events
        ind_hse = np.array(find_hse(x=x)).flatten()

        # only select high synchrony events
        x = x[:, ind_hse]

        print("#HSE: "+str(x.shape[1]))

        y = ind_hse * self.params.time_bin_size

        new_ml = MlMethodsOnePopulation()
        new_ml.ridge_time_bin_progress(x=x, y=y, new_time_bin_size=self.params.time_bin_size, alpha_fitting=False,
                                       alpha=500)

    # population vector / HSE similarity
    # ------------------------------------------------------------------------------------------------------------------
    def hse_similarity(self):
        # --------------------------------------------------------------------------------------------------------------
        # analysis of hse similarity using population vectors
        #
        # args:
        # --------------------------------------------------------------------------------------------------------------

        # compute pre processed data
        self.compute_raster_speed_loc()

        x = self.raster

        # find high synchrony events
        ind_hse = np.array(find_hse(x=x)).flatten()

        # only select high synchrony events
        x = x[:, ind_hse]

        print("#HSE: "+str(x.shape[1]))

        # new_ml = MlMethodsOnePopulation(act_map=x, params=self.params)
        # new_ml.plot_reduced_dimension()
        # exit()

        # D = multi_dim_scaling(x, self.params)
        # scatter_animation(D, self.params)

        # D = perform_TSNE(x, self.params)
        # scatter_animation(D, self.params)

        # new_ml = MlMethodsOnePopulation(act_map=x, params=self.params)
        # new_ml.plot_reduced_dimension()
        #
        # exit()

        D = np.zeros([x.shape[1], x.shape[1]])

        for i, pop_vec_ref in enumerate(x.T):
            for j, pop_vec_comp in enumerate(x.T):
                    D[i, j] = distance.cosine(pop_vec_ref, pop_vec_comp)
        D = 1 - D
        plt.imshow(D, interpolation="nearest", aspect="auto", cmap="jet")
        a = plt.colorbar()
        a.set_label("COSINE SIMILARITY")
        plt.title("COSINE SIMILARITY OF HSE POP. VECTORS")
        plt.xlabel("HSE ID")
        plt.ylabel("HSE ID")
        plt.show()

        from sklearn.cluster import spectral_clustering
        labels = spectral_clustering(D, n_clusters=10)
        sorted = np.argsort(labels)
        print(sorted.shape)
        D = D[:, sorted]
        plt.imshow(D, interpolation="nearest", aspect="auto", cmap="jet")
        a = plt.colorbar()
        a.set_label("COSINE SIMILARITY")
        plt.title("COSINE SIMILARITY OF HSE POP. VECTORS")
        plt.xlabel("HSE ID")
        plt.ylabel("HSE ID")
        plt.show()



        exit()

        D = np.zeros([x.shape[0], x.shape[0]])

        for i, pop_vec_ref in enumerate(x):
            for j, pop_vec_comp in enumerate(x):
                    D[i, j] = distance.cosine(pop_vec_ref, pop_vec_comp)
        D = 1 - D
        plt.imshow(D, interpolation="nearest", aspect="auto", cmap="jet")
        a = plt.colorbar()
        a.set_label("COSINE SIMILARITY")
        plt.title("COSINE SIMILARITY CELL FIRING")
        plt.xlabel("CELL ID")
        plt.ylabel("CELL ID")
        plt.show()

        exit()

        eigenval, eigenvec = np.linalg.eig(D)
        sorted_ind = np.flip(np.argsort(eigenval))
        sorted_eigval = eigenval[sorted_ind]
        sorted_eigvec = eigenvec[:, sorted_ind]

        print(eigenvec.shape)
        print(x.shape)
        x_proj = np.real(x @ sorted_eigvec[:,:3])

        if self.params.dr_method_p2 == 3:
            # create figure instance
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plot_3D_scatter(ax, x_proj, self.params, None)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot_2D_scatter(ax, x_proj, self.params, None)
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

    def population_vector_similarity_non_hse(self, new_time_bin_size):
        # --------------------------------------------------------------------------------------------------------------
        # analysis of similarity using population vectors
        #
        # args:
        # --------------------------------------------------------------------------------------------------------------

        # compute pre processed data
        self.compute_raster_speed_loc()

        x = self.raster

        # find high synchrony events
        ind_hse = np.array(find_hse(x=x)).flatten()

        # remove high synchrony events
        raster_sleep = np.delete(x, ind_hse, axis=1)

        time_bin_scaler = int(new_time_bin_size / self.params.time_bin_size)

        new_raster = np.zeros((raster_sleep.shape[0], int(raster_sleep.shape[1] / time_bin_scaler)))

        # down sample spikes by combining multiple bins
        for i in range(new_raster.shape[1]):
            new_raster[:, i] = np.sum(raster_sleep[:, (i * time_bin_scaler): ((1 + i) * time_bin_scaler)], axis=1)

        x = new_raster

        D = multi_dim_scaling(x, self.params)
        #scatter_animation(D, self.params)

        if self.params.dr_method_p2 == 3:
            # create figure instance
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plot_3D_scatter(ax=ax, mds=D, params=self.params)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot_2D_scatter(ax=ax, mds=D, params=self.params)
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

        exit()
        # D = perform_TSNE(x, self.params)
        # scatter_animation(D, self.params)

        from scipy.spatial import distance

        D = np.zeros([x.shape[1], x.shape[1]])

        for i, pop_vec_ref in enumerate(x.T):
            for j, pop_vec_comp in enumerate(x.T):
                    D[i, j] = distance.cosine(pop_vec_ref, pop_vec_comp)
        D = 1 - D
        plt.imshow(D, interpolation="nearest", aspect="auto", cmap="jet")
        a = plt.colorbar()
        a.set_label("COSINE SIMILARITY")
        plt.title("COSINE SIMILARITY OF HSE POP. VECTORS")
        plt.xlabel("HSE ID")
        plt.ylabel("HSE ID")
        plt.show()

    """#################################################################################################################
    #   SWR sequential content analysis
    #################################################################################################################"""

    def sequential_content_phmm(self, pHMM_file_pre, pHMM_file_post, SWR=True,
                                   plot_for_control=False):
        # --------------------------------------------------------------------------------------------------------------
        # decodes sleep activity using pHMM model and investigates sequential content of activity (how similar are the
        # sequence observed during sleep to sequences observed during behavior)
        #
        # args:     - pHMM_file_pre:    name of file containing the dictionary with pHMM model from PRE behavioral data
        #           - pHMM_file_post:    name of file containing the dictionary with pHMM model from
        #                                                   AFTER/POST awake/behavioral data
        #
        #           - SWR, bool: whether to use only SWR (=True) or all activity
        #           - plot_for_control, bool: plots intermediate results if True
        #           - n_moving_average, int: n to use for moving average
        #           - return_reslts: if True --> returns results instead of plotting
        #
        # returns:  - list with entries for each SWR:  entry contains array with [pop_vec_SWR, spatial_bin_template_map]
        #                                              probabilities
        # --------------------------------------------------------------------------------------------------------------

        print(" - CONTENT BASED MEMORY DRIFT USING PHMM MODES (HMMLEARN)...\n")

        if SWR:

            # generate results for pre
            pre_log_prob, pre_sequence_list, event_times = \
                self.decode_swr_sequence_using_phmm(phmm_file_name=pHMM_file_pre, plot_for_control=plot_for_control)
            # generate results for post
            post_log_prob, post_sequence_list, post_event_times = \
                self.decode_swr_sequence_using_phmm(phmm_file_name=pHMM_file_post, plot_for_control=plot_for_control)

            # get model from PRE and transition matrix
            model_pre = self.load_poisson_hmm(file_name=pHMM_file_pre)
            transmat_pre = model_pre.transmat_
            model_post = self.load_poisson_hmm(file_name=pHMM_file_pre)
            transmat_post = model_post.transmat_

            pre_swr_seq_similarity = []
            post_swr_seq_similarity = []

            for pre_sequence, post_sequence in zip(pre_sequence_list, post_sequence_list):

                # make sure that there is any data for the current SWR

                if pre_sequence.shape[0] > 0:

                    # check how likely observed sequence is considering transitions from model (awake behavior)
                    mode_before = pre_sequence[:-1]
                    mode_after = pre_sequence[1:]
                    transition_prob = 0
                    # go trough each transition of the sequence
                    for bef, aft in zip(mode_before, mode_after):
                        transition_prob += np.log(transmat_pre[bef, aft])

                    pre_swr_seq_similarity.append(np.exp(transition_prob))

                    # POST
                    # check how likely observed sequence is considering transitions from model (awake behavior)
                    mode_before = post_sequence[:-1]
                    mode_after = post_sequence[1:]
                    transition_prob = 0
                    # go trough each transition of the sequence
                    for bef, aft in zip(mode_before, mode_after):
                        transition_prob += np.log(transmat_post[bef, aft])

                    post_swr_seq_similarity.append(np.exp(transition_prob))

            # sequence probability
            # pre_swr_seq_similarity = moving_average(a=np.array(pre_swr_seq_similarity), n=10)
            plt.plot(pre_swr_seq_similarity)
            plt.title("PROBABILITY SWR PHMM MODE SEQUENCES \n USING VITERBI + AWAKE TRANSITION PROB. PRE")
            plt.ylabel("JOINT PROBABILITY")
            plt.xlabel("SWR ID")
            plt.show()

            # sequence probability
            # post_swr_seq_similarity = moving_average(a=np.array(post_swr_seq_similarity), n=10)
            plt.plot(post_swr_seq_similarity)
            plt.title("PROBABILITY SWR PHMM MODE SEQUENCES \n USING VITERBI + AWAKE TRANSITION PROB. POST")
            plt.ylabel("JOINT PROBABILITY")
            plt.xlabel("SWR ID")
            plt.show()

        else:
            raise Exception("TO BE IMPLEMENTED!")

    def decode_swr_sequence_using_phmm(self, phmm_file_name, plot_for_control=False,
                              time_bin_size_encoding=0.1):
        """
        decodes sleep activity sequence during SWR using pHMM model from awake activity (uses hmmlearn predict_proba)

        @param phmm_file_name: name of file containing the dictionary with template from awake/behavioral data
        @type phmm_file_name: str
        @param plot_for_control: whether to plot intermediate results for control
        @type plot_for_control: bool
        @param time_bin_size_encoding: which time bin size was used for encoding (usually 100ms --> should save this
        info with the model
        @type time_bin_size_encoding: float
        """

        print(" - DECODING SLEEP ACTIVITY SEQUENCE USING HMMLEARN CODE & "+ phmm_file_name +" ...\n")

        # load and pre-process template map (computed on awake data)
        # --------------------------------------------------------------------------------------------------------------

        # load pHMM model
        with open('temp_data_old/phmm/' + phmm_file_name + '.pkl', 'rb') as f:
             phmm_model = pickle.load(f)

        mode_means = phmm_model.means_
        nr_modes = mode_means.shape[0]

        # get SWR timings (in sec) & compute spike rasters (constant #spikes)
        # --------------------------------------------------------------------------------------------------------------
        start_times, end_times, peak_times = self.detect_swr(plot_for_control=plot_for_control)

        # convert to one array for event_spike_binning
        event_times = np.vstack((start_times, end_times)).T

        # compute #spike binning for each event --> TODO: implement sliding window!
        event_spike_rasters, event_spike_window_lenghts = \
            PreProcessSleep(firing_times=self.firing_times, params=self.params,
                            whl=self.whl).event_spike_binning(event_times=event_times, event_time_freq=1)
        # plot raster and detected SWR, example spike rasters from SWRs
        if plot_for_control:
                        # compute pre processed data
            self.compute_raster_speed_loc()
            to_plot = np.random.randint(0, start_times.shape[0], 5)
            for i in to_plot:
                plt.imshow(self.raster, interpolation='nearest', aspect='auto')
                plt.vlines(start_times[i] / self.params.time_bin_size, 0, self.raster.shape[0] - 1, colors="r",
                           linewidth=0.5,
                           label="SWR START")
                plt.vlines(end_times[i] / self.params.time_bin_size, 0, self.raster.shape[0] - 1, colors="g",
                           linewidth=0.5, label="SWR END")
                plt.xlim(start_times[i] / self.params.time_bin_size - 10,
                         end_times[i] / self.params.time_bin_size + 10)
                plt.ylabel("CELL IDS")
                plt.xlabel("TIME BINS")
                plt.legend()
                a = plt.colorbar()
                a.set_label("SPIKES PER BIN (" + str(self.params.time_bin_size) + " s)")
                plt.title("SWR EVENT ID " + str(i))
                plt.show()

                plt.imshow(event_spike_rasters[i], interpolation='nearest', aspect='auto')
                a = plt.colorbar()
                a.set_label("SPIKES PER CONST. #SPIKES BIN")
                plt.xlabel("CONST. #SPIKE POP. VEC. ID")
                plt.ylabel("CELL ID")
                plt.title("BINNED SWR EVENT ID " +str(i) +"\n#SPIKES PER BIN: "+str(self.params.spikes_per_bin))
                plt.show()

        # compute firing rate factor (sleep <-> awake activity with different binning & compression)
        # --------------------------------------------------------------------------------------------------------------

        # time_window_factor:   accounts for the different window length of awake encoding and window length
        #                       during sleep (variable window length because of #spike binning)

        # compute average window length for swr activity
        event_spike_window_lengths_avg = np.mean(np.concatenate(event_spike_window_lenghts, axis=0))
        time_window_factor = event_spike_window_lengths_avg / time_bin_size_encoding

        # compression_factor: additional compression factor
        compression_factor = 1

        # firing_rate_factor: need this because we are comparing sleep activity (compressed) with awake activity
        firing_rate_factor = time_window_factor*compression_factor

        # list to store results per SWR
        sequence_list = []
        log_prob_list = []

        # main decoding part
        # --------------------------------------------------------------------------------------------------------------
        # go through all SWR events
        for event_id, spike_raster in enumerate(event_spike_rasters):

            # instead of multiplying awake activity by firing_rate_factor --> divide population vectors by
            # firing rate factor
            spike_raster /= firing_rate_factor
            res = phmm_model.decode(spike_raster.T, algorithm="viterbi")
            sequence_list.append(res[1])
            log_prob_list.append(res[0])


        return log_prob_list, sequence_list, event_times


"""#####################################################################################################################
#   EXPLORATION CLASS
#####################################################################################################################"""


class Exploration(BaseMethods):
    """Base class for exploration analysis"""

    def __init__(self, data_dic, cell_type, params, session_params=None, experiment_phase=None):
        # --------------------------------------------------------------------------------------------------------------
        # args: - data_dic, dictionary with standard data
        #       - cell_type, string: cell type to do analysis with
        #       - params, python class: contains all parameters
        # --------------------------------------------------------------------------------------------------------------

        # get attributes from parent class
        BaseMethods.__init__(self, data_dic, cell_type, params, session_params, experiment_phase)

        # get spatial factor: cm per .whl arbitrary unit
        self.spatial_factor = self.session_params.data_params_dictionary["spatial_factor"]

        # other data
        self.time_bin_size = params.time_bin_size

        # compute raster, location and speed data
        self.raster, self.loc, self.speed = PreProcessAwake(self.firing_times, self.params, self.whl,
                                                            spatial_factor=self.spatial_factor).get_raster_loc_vel()

        # # get dimensions of environment
        self.x_min, self.x_max, self.y_min, self.y_max = \
            min(self.loc[:, 0]), max(self.loc[:, 0]), min(self.loc[:, 1]), max(self.loc[:, 1])

    def plot_linearized_location_and_speed(self):

        # plotting
        t = np.arange(len(self.vel))
        plt.plot(t / 20e3, self.vel, label="speed")
        plt.plot(t / 20e3, self.loc, label="location")
        plt.plot([0, t[-1] / 20e3], [5, 5], label="threshold")
        plt.xlabel("time / s")
        plt.ylabel("location / cm - speed / (cm/s)")
        plt.legend()
        plt.show()

    def linearized_whl_calc_loc_and_speed(self):
        # computes speed from the whl and returns speed in cm/s and upsamples location data to match spike timing
        # need to smooth position data --> accuracy of measurement: about +-1cm --> error for speed: +-40cm/s
        # last element of velocity vector is zero --> velocity is calculated using 2 locations

        # savitzky golay
        w_l = 31  # window length
        p_o = 5  # order of polynomial
        whl = signal.savgol_filter(self.whl, w_l, p_o)

        # one time bin: whl is recorded at 20kHz/512
        t_b = 1 / (20e3 / 512)

        # upsampling to synchronize with spike data
        location = np.zeros(whl.shape[0] * 512)
        for i, loc in enumerate(whl):
            location[512 * i:(i + 1) * 512] = 512 * [loc]

        # calculate speed: x1-x0/dt
        temp_speed = np.zeros(whl.shape[0] - 1)
        for i in range(temp_speed.shape[0]):
            temp_speed[i] = (whl[i + 1] - whl[i]) / t_b

        # smoothen speed using savitzky golay
        temp_speed = signal.savgol_filter(temp_speed, 15, 5)

        # upsampling to synchronize with spike data
        speed = np.zeros(whl.shape[0] * 512)
        for i, bin_speed in enumerate(temp_speed):
            speed[512 * i:(i + 1) * 512] = 512 * [bin_speed]

        return location, speed

    def view_occupancy(self, spatial_resolution=None):
        # --------------------------------------------------------------------------------------------------------------
        # returns occupancy map in seconds
        #
        # args:   - spatial_resolution, int: spatial resolution of occupancy map in cm
        # --------------------------------------------------------------------------------------------------------------

        # occ_map = PreProcessAwake(self.firing_times, self.params, self.whl).occupancy_map(time_window=time_window)
        # exp_time = np.round(np.sum(occ_map.flatten())/60, 3)
        # # remove all elements outside the environment (rows and col that only contain zeros)
        # # occ_map = occ_map[:, ~(occ_map==0).all(0)]
        # # occ_map = occ_map[~(occ_map == 0).all(1), :]
        # occ_map = occ_map.T

        rate_map, occ = self.rate_map_from_data(loc=self.loc, raster=self.raster, spatial_resolution=spatial_resolution)
        exp_time = np.round(np.sum(occ.flatten()) / 60, 3)

        plt.imshow(occ.T, interpolation='nearest', aspect='equal', vmin=0, vmax=2,origin='lower')
        a = plt.colorbar()
        a.set_label("OCCUPANCY / SEC")
        plt.ylabel("Y")
        plt.xlabel("X")
        plt.title("OCCUPANCY (EXPLORATION TIME: "+str(exp_time)+" min)")
        plt.show()

    def view_all_rate_maps(self):
        rate_maps = PreProcessAwake(self.firing_times, self.params, self.whl).spatial_rate_map()
        for map in rate_maps:
            plt.imshow(map, interpolation='nearest', aspect='auto', origin="lower")
            plt.colorbar()
            plt.show()

    def view_one_rate_map(self, cell_id, sel_range=None):
        if range is None:
            loc = self.loc
            raster = self.raster
        else:
            loc = self.loc[sel_range,:]
            raster = self.raster[:,sel_range]

        rate_map, occ = self.rate_map_from_data(loc=loc, raster=raster)

        plt.scatter(loc[:,0], loc[:,1])
        plt.show()

        plt.imshow(occ.T, origin="lower")
        plt.title("OCCUPANCY")
        plt.colorbar()
        plt.show()

        plt.imshow(rate_map[:,:,cell_id].T, interpolation='nearest', aspect='auto', origin="lower")
        a = plt.colorbar()
        a.set_label("FIRING RATE")
        plt.title("RATE MAP FOR CELL " + str(cell_id))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def get_rate_maps(self, spatial_resolution=None):
        rate_maps, occ = self.rate_map_from_data(loc=self.loc, raster=self.raster,
                                                 spatial_resolution=spatial_resolution)
        return rate_maps

    def get_occ_map(self, spatial_resolution=None):
        rate_maps, occ = self.rate_map_from_data(loc=self.loc, raster=self.raster,
                                                 spatial_resolution=spatial_resolution)
        return occ

    def get_rate_maps_occ_maps_temporal_splits(self, spatial_resolution=None, nr_of_splits=2, env_dim=None):
        """
        computes rate maps using splits of the data (in terms of time)

        :param spatial_resolution: spatial resolution of rate maps in cm2
        :type spatial_resolution: int
        :param nr_of_splits: in how many splits to divide the data
        :type nr_of_splits: int
        :return: list of rate maps (one list entry with rate maps for all cells for each split) and list of occ maps
        :rtype: list, list
        """

        len_split = int(self.loc.shape[0]/nr_of_splits)

        list_rate_maps = []
        list_occ_maps = []

        for split_id in range(nr_of_splits):
            rate_maps=None
            occ = None
            rate_maps, occ = self.rate_map_from_data(loc=self.loc[split_id*len_split:(split_id+1)*len_split, :],
                                                     raster=self.raster[:, split_id*len_split:(split_id+1)*len_split],
                                                     spatial_resolution=spatial_resolution, env_dim=env_dim)

            list_rate_maps.append(rate_maps)
            list_occ_maps.append(occ)

        return list_rate_maps, list_occ_maps

    def place_field_similarity(self, plotting=False):
        rate_maps = PreProcessAwake(self.firing_times, self.params, self.whl).spatial_rate_map()
        pfs_matrix = np.zeros((len(rate_maps), len(rate_maps)))
        for i, r_m_1 in enumerate(rate_maps):
            for j, r_m_2 in enumerate(rate_maps):
                corr = pearsonr(r_m_1.flatten(), r_m_2.flatten())
                if corr[1] < 0.05:
                    pfs_matrix[i, j] = corr[0]
        if plotting:
            plt.imshow(pfs_matrix, interpolation='nearest', aspect='auto')
            a = plt.colorbar()
            a.set_label("PEARSON CORRELATION R")
            plt.xlabel("CELL ID")
            plt.ylabel("CELL ID")
            plt.title("PLACE FIELD SIMILARITY")
            plt.show()
        return pfs_matrix

    def place_field_entropy(self, plotting=False):
        # --------------------------------------------------------------------------------------------------------------
        # computes entropy of the rate map of each cell
        #
        # args:   - plotting, bool: whether to plot rate maps with maximum and minimum entropy
        #
        # returns:      -
        # --------------------------------------------------------------------------------------------------------------
        rate_maps = PreProcessAwake(self.firing_times, self.params, self.whl).spatial_rate_map()
        ent = np.zeros(len(rate_maps))
        for i, r_m in enumerate(rate_maps):
            ent[i] = entropy(r_m.flatten())

        if plotting:
            plt.subplot(2,1,1)
            plt.imshow(rate_maps[np.argmax(ent)], interpolation='nearest', aspect='auto')
            print(np.argmax(ent))
            a = plt.colorbar()
            a.set_label("FIRING RATE / Hz")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.subplot(2, 1, 2)
            plt.imshow(rate_maps[np.argmin(ent)], interpolation='nearest', aspect='auto')
            print(np.argmin(ent))
            plt.xlabel("X")
            plt.ylabel("Y")
            a = plt.colorbar()
            a.set_label("FIRING RATE / Hz")
            plt.show()
        return ent

    def find_correlations(self):
        # --------------------------------------------------------------------------------------------------------------
        # plots scatter plot with entropy and place field similarity
        #
        # args:   -
        #
        # returns:      -
        # --------------------------------------------------------------------------------------------------------------

        ent = self.place_field_entropy()
        pcf_mat = self.place_field_similarity()

        pcf_mean = np.mean(pcf_mat, axis=1)

        plt.scatter(ent, pcf_mean)
        plt.ylabel("MEAN PLACE FIELD SIMILARITY")
        plt.xlabel("RATE MAP ENTROPY")
        plt.show()

    def rate_map_from_data(self, loc, raster, spatial_resolution=None, gaussian_std=1, env_dim=None):
        # --------------------------------------------------------------------------------------------------------------
        # computes #spikes per bin from input data
        #
        # args:   - loc, array: location data (2D) synchronized with raster data
        #               - raster, array [nr_cells, nr_time_bins]: #spikes per cell /time bin
        #               - spatial_resolution, int: in cm
        #               - gaussian_std, float: std of gaussians for smoothing of rate map --> if 0: no smoothing
        #
        # returns:      - rate_map, array [x_coord, y_coord, cell_id], per spatial bin: spike rate (1/s)
        # --------------------------------------------------------------------------------------------------------------

        if spatial_resolution is None:
            spatial_resolution = self.params.spatial_resolution

        nr_cells = raster.shape[0]
        loc_ds = np.floor(loc / spatial_resolution).astype(int)

        if env_dim is None:
            # get dimensions of environment
            x_min, x_max, y_min, y_max = self.x_min, self.x_max, self.y_min, self.y_max
        else:
            x_min, x_max, y_min, y_max = env_dim[0], env_dim[1], env_dim[2], env_dim[3]


        # get size of environment
        x_span = x_max - x_min
        y_span = y_max - y_min

        # width - length ratio of environment
        w_l_ratio = y_span/x_span

        nr_spatial_bins = int(np.round(x_span / spatial_resolution))

        centers_x = np.linspace(x_min, x_max + 0.1, nr_spatial_bins)
        centers_y = np.linspace(y_min, y_max + 0.1, int(round(nr_spatial_bins*w_l_ratio)))

        dx = centers_x[1] - centers_x[0]
        dy = centers_y[1] - centers_y[0]

        # split location data into x and y coordinates
        x_loc = loc[:, 0]
        y_loc = loc[:, 1]

        x_loc[x_loc > x_max] = x_min - 0.01
        y_loc[y_loc > y_max] = y_min - 0.01

        occ = np.zeros((centers_x.shape[0], centers_y.shape[0]))
        raster_2d = np.zeros((nr_spatial_bins, int(np.round(nr_spatial_bins * w_l_ratio)), nr_cells))

        for i,(_, pop_vec) in enumerate(zip(loc_ds, raster.T)):
            xi = int(np.floor((x_loc[i]-x_min)/dx))+1
            yi = int(np.floor((y_loc[i] - y_min) / dy)) + 1
            if xi*yi > 0:
                occ[xi, yi] += 1
                raster_2d[xi, yi, :] += pop_vec

        # one pop vec is at time bin resolution --> need occupancy in seconds
        occ = occ * self.params.time_bin_size
        # make zeros to nan for division
        occ[occ == 0] = np.nan
        rate_map = np.nan_to_num(raster_2d / occ[..., None])
        occ = np.nan_to_num(occ)

        # rate[occ > 0.05] = rate[occ > 0.05] / occ[occ > 0.05]
        # if sigma_gauss > 0:
        #     rate = nd.gaussian_filter(rate, sigma=sigma_gauss)
        # rate[occ == 0] = 0

        # apply gaussian filtering --> smooth place fields
        for i in range(nr_cells):
            rate_map[:, :, i] = nd.gaussian_filter(rate_map[:, :, i], sigma=gaussian_std)

        return rate_map, occ

    def test_binning(self):
        # --------------------------------------------------------------------------------------------------------------
        # test binning using externally generated rasters (e.g. through matlab)
        #
        # args:   -
        #
        # returns:      -
        # --------------------------------------------------------------------------------------------------------------
        sel_range = range(1000)
        cell_to_plot = 5

        print(self.raster.shape, self.loc.shape)

        raster = self.raster[:, sel_range]
        loc = self.loc[sel_range, :]
        print(raster.shape, loc.shape)

        # load reference data

        mat_files = loadmat("matlab.mat")
        raster_ref = mat_files["UnitSp"][:, sel_range]
        occ_ref = mat_files["UnitBinOcc"][:, sel_range]
        loc_ref = mat_files["PathRely"][sel_range, :]

        print(raster_ref.shape, occ_ref.shape, loc_ref.shape)

        rate_map_ref, occ_map_ref = self.rate_map_from_data(loc=loc_ref, raster=raster_ref, gaussian_std=0)
        plt.imshow(rate_map_ref[:,:,cell_to_plot].T)
        plt.colorbar()
        plt.title("REF")
        plt.show()

        rate_map_ref, occ_map_ref = self.rate_map_from_data(loc=loc, raster=raster, gaussian_std=0)
        plt.imshow(rate_map_ref[:,:,cell_to_plot].T)
        plt.colorbar()
        plt.title("ORIG")
        plt.show()

    def test_occupancy(self):
        # --------------------------------------------------------------------------------------------------------------
        # test occupancy map by creating artifical
        #
        # args:   -
        #
        # returns:      -
        # --------------------------------------------------------------------------------------------------------------
        raise Exception("To be implemented!")

    def get_env_dim(self):
        return self.x_min, self.x_max, self.y_min, self.y_max

    """#################################################################################################################
    #   clustering / discrete system states analysis
    #################################################################################################################"""

    def fit_spatial_gaussians_for_modes(self, file_name, min_nr_bins_active, plot_awake_fit=False):

        state_sequence, nr_modes = self.decode_poisson_hmm(file_name=file_name)

        mode_id, freq = np.unique(state_sequence, return_counts=True)
        modes_to_plot = mode_id[freq > min_nr_bins_active]

        means = np.zeros((2, nr_modes))
        cov = np.zeros((2, nr_modes))
        for mode in np.arange(nr_modes):
            mode_data = self.loc[state_sequence == mode, :]

            if len(mode_data) == 0:
                means[:, mode] = np.nan
                cov[:, mode] = np.nan
            else:
                means[:, mode] = np.mean(mode_data, axis=0)
                cov[:, mode] = np.var(mode_data, axis=0)

        loc_data = self.loc[:, :]

        center = np.min(loc_data, axis=0) + (np.max(loc_data, axis=0) - np.min(loc_data, axis=0)) * 0.5
        dist = loc_data - center

        rad = max(np.sqrt(np.square(dist[:, 0]) + np.square(dist[:, 1]))) + 1

        std_modes = np.sqrt(cov[0,:]+cov[1,:])
        std_modes[std_modes == 0] = np.nan

        if plot_awake_fit:

            for mode_to_plot in modes_to_plot:

                mean = means[:, mode_to_plot]
                cov_ = cov[:, mode_to_plot]
                std_ = std_modes[mode_to_plot]

                # Parameters to set
                mu_x = mean[0]
                variance_x = cov_[0]

                mu_y = mean[1]
                variance_y = cov_[1]

                # Create grid and multivariate normal
                x = np.linspace(center[0] - rad, center[0]+rad, int(2.2*rad))
                y = np.linspace(0, 250, 250)
                X, Y = np.meshgrid(x, y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
                rv_normalized = rv.pdf(pos) / np.sum(rv.pdf(pos).flatten())

                fig, ax = plt.subplots()
                gauss = ax.imshow(rv_normalized)
                env = Circle((center[0], center[1]), rad, color="white", fill=False)
                ax.add_artist(env)
                ax.set_ylim(center[1] - 1.1*rad, center[1]+1.1*rad)
                ax.scatter(loc_data[state_sequence == mode_to_plot, 0], loc_data[state_sequence == mode_to_plot, 1],
                           alpha=1, c="white", marker=".", s=0.3, label="MODE "+ str(mode_to_plot) +" ASSIGNED")
                cb = plt.colorbar(gauss)
                cb.set_label("PROBABILITY")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title("STD: "+str(np.round(std_, 2)))
                plt.legend()
                plt.show()

        # compute frequency of each mode
        mode_freq = np.zeros(nr_modes)
        mode_freq[mode_id] = freq
        mode_freq = mode_freq.astype(int)

        env = Circle((center[0], center[1]), rad, color="white", fill=False)

        return means, std_modes, mode_freq, env, state_sequence

    def phmm_mode_spatial_information_from_model(self, spatial_resolution=5, nr_modes=None, file_name=None,
                                            plot_for_control=False):
        """
        loads poisson hmm model and weighs rate maps by lambda vectors --> then computes spatial information (sparsity,
        skaggs information)

        @param spatial_resolution: spatial resolution in cm
        @type spatial_resolution: int
        @param nr_modes: nr of modes for model file identification
        @type nr_modes: int
        @param file_name: file containing the model --> is used when nr_modes is not provided to identify file
        @type file_name: string
        @param plot_for_control: whether to plot intermediate results
        @type plot_for_control: bool
        @return: sparsity, skaggs info for each mode
        """

        print(" - SPATIAL INFORMATION OF PHMM MODES USING MODEL\n")

        if file_name is None:
            file_name = self.params.session_name + "_" + self.experiment_phase_id + \
                        "_" + self.cell_type +"_"+str(nr_modes)+"_modes"

        with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "rb") as file:
            model = pickle.load(file)

        # get means for all modes (lambda vectors)
        means = model.means_

        ################################################################################################################
        # get spatial information of mode by weighing rate maps
        ################################################################################################################

        # compute rate maps and occupancy
        rate_maps, occ = self.rate_map_from_data(loc=self.loc, raster=self.raster,
                                                 spatial_resolution=spatial_resolution, gaussian_std=0)
        # compute binary occupancy map --> used as a mask to filter non-visited bins
        occ_mask = np.where(occ > 0, 1, np.nan)

        sparsity_list = []
        skaggs_list = []

        # go through all modes
        for mode_id, means_mode in enumerate(means):
            # weigh rate map of each cell using mean firing from lambda vector --> compute mean across all cells
            rate_map_mode_orig = np.mean(rate_maps * means_mode, axis=2)
            # generate filtered rate map by masking non visited places
            rate_map_mode_orig = np.multiply(rate_map_mode_orig, occ_mask)
            rate_map_mode = rate_map_mode_orig[~np.isnan(rate_map_mode_orig)]
            # need to filter bins with zero firing rate --> otherwise log causes an error
            rate_map_mode = rate_map_mode[rate_map_mode > 0]

            # compute sparsity
            sparse_mode = np.round(np.mean(rate_map_mode.flatten())**2/np.mean(np.square(rate_map_mode.flatten())), 2)

            # compute Skagg's information criterium
            raise Exception("DOUBLE CHECK SKAGGS INFORMATION FORMULATION")
            skaggs_info = np.round(np.sum((rate_map_mode.flatten()/np.mean(rate_map_mode.flatten())) *
                                np.log(rate_map_mode.flatten()/np.mean(rate_map_mode.flatten()))), 4)

            skaggs_list.append(skaggs_info)
            sparsity_list.append(sparse_mode)
            if plot_for_control:
                # plot random examples
                rand_float = np.random.randn(1)
                if rand_float > 0.5:
                    plt.imshow(rate_map_mode_orig)
                    plt.colorbar()
                    plt.title(str(sparse_mode)+", "+str(skaggs_info))
                    plt.show()

        if plot_for_control:
            plt.hist(skaggs_list)
            plt.title("SKAGGS INFO.")
            plt.xlabel("SKAGGS INFO.")
            plt.ylabel("COUNTS")
            plt.show()

            plt.hist(sparsity_list)
            plt.title("SPARSITY")
            plt.xlabel("SPARSITY")
            plt.ylabel("COUNTS")
            plt.show()

            plt.scatter(skaggs_list, sparsity_list)
            plt.title("SKAGGS vs. SPARSITY\n"+str(pearsonr(skaggs_list, sparsity_list)))
            plt.xlabel("SKAGGS")
            plt.ylabel("SPARSITY")
            plt.show()

        return np.array(sparsity_list), np.array(skaggs_list)

    def phmm_mode_spatial_information_from_fit(self, nr_modes, spatial_resolution):
        """
        get spatial information of mode by fitting model to data and analyzing "mode locations" --> e.g taking
        distances between points

        @param nr_modes: nr. of modes to identify file containing model
        @type nr_modes: int
        @param spatial_resolution: spatial resolution for spatial binning in cm
        @type spatial_resolution: int
        """

        print(" - SPATIAL INFORMATION OF PHMM MODES USING FIT")

        state_sequence, nr_modes_ = self.decode_poisson_hmm(nr_modes=nr_modes)

        loc_data = self.loc[:, :]

        center = np.min(loc_data, axis=0) + (np.max(loc_data, axis=0) - np.min(loc_data, axis=0)) * 0.5
        dist = loc_data - center

        rad = max(np.sqrt(np.square(dist[:, 0]) + np.square(dist[:, 1]))) + 1

        # get dimensions of environment
        x_min, x_max, y_min, y_max = min(self.loc[:, 0]), max(self.loc[:, 0]), min(self.loc[:, 1]), max(
            self.loc[:, 1])

        # get size of environment
        x_span = x_max - x_min
        y_span = y_max - y_min

        # width - length ratio of environment
        w_l_ratio = y_span / x_span

        nr_spatial_bins = int(np.round(x_span / spatial_resolution))

        centers_x = np.linspace(x_min, x_max + 0.1, nr_spatial_bins)
        centers_y = np.linspace(y_min, y_max + 0.1, int(round(nr_spatial_bins * w_l_ratio)))

        dx = centers_x[1] - centers_x[0]
        dy = centers_y[1] - centers_y[0]

        # results from fit
        sparsity_fit_list = []
        skaggs_fit_list = []

        for mode in np.arange(nr_modes_):
            mode_loc = self.loc[state_sequence == mode, :]

            # compute pairwise distances (euclidean)
            pd = upper_tri_without_diag(pairwise_distances(mode_loc))

            mean_dist = np.mean(pd)
            std_dist = np.std(pd)

            if plot_for_control:

                fig, ax = plt.subplots()
                env = Circle((center[0], center[1]), rad, color="white", fill=False)
                ax.add_artist(env)
                ax.set_ylim(center[1] - 1.1 * rad, center[1] + 1.1 * rad)
                ax.set_xlim(center[0] - 1.1 * rad, center[0] + 1.1 * rad)
                ax.scatter(mode_loc[:, 0], mode_loc[:, 1])
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title(str(mean_dist) + ", " + str(std_dist))
                plt.show()

                continue
                # discretize data
                # split location data into x and y coordinates
                x_loc = mode_loc[:, 0]
                y_loc = mode_loc[:, 1]

                x_loc[x_loc > x_max] = x_min - 0.01
                y_loc[y_loc > y_max] = y_min - 0.01

                mode_act = np.zeros((centers_x.shape[0], centers_y.shape[0]))

                for i in range(x_loc.shape[0]):
                    xi = int(np.floor((x_loc[i] - x_min) / dx)) + 1
                    yi = int(np.floor((y_loc[i] - y_min) / dy)) + 1
                    if xi * yi > 0:
                        mode_act[xi, yi] += 1

                # mask non-visited places
                mode_act_orig = np.multiply(mode_act, occ_mask)

                plt.imshow(mode_act_orig.T, origin="lower")
                plt.show()

    def phmm_mode_spatial_information_visual(self, file_name):
        """
        uses phmm fit location

        @param file_name:
        @type file_name:
        """
        print(" - SPATIAL INFORMATION OF PHMM MODES - VISUALIZATION")

        with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "rb") as file:
            model = pickle.load(file)

        # get means for all modes (lambda vectors)
        means = model.means_
        # get transition matrix
        transmat = model.transmat_

        means, std_modes, mode_freq, env, state_sequence = \
            self.fit_spatial_gaussians_for_modes(file_name=file_name, plot_awake_fit=False, min_nr_bins_active=20)

        # compute distance between means --> correlate with transition matrix

        dist_mat = np.zeros((means.shape[1], means.shape[1]))

        for i, mean_1 in enumerate(means.T):
            for j, mean_2 in enumerate(means.T):
                dist_mat[i, j] = np.linalg.norm(mean_1 - mean_2)

        dist_flat = upper_tri_without_diag(dist_mat)
        transition = upper_tri_without_diag(transmat)

        plt.scatter(dist_flat, transition)
        plt.xlabel("DISTANCE BETWEEN MEANS / cm")
        plt.ylabel("TRANSITION PROBABILITY")
        plt.title("DISTANCE & TRANSITION PROBABILITES BETWEEN MODES")
        plt.show()

        std = np.nan_to_num(std_modes)

        constrained_means = means[:, (std < 25) & (std > 0) & (mode_freq > 100)]
        fig, ax = plt.subplots()
        ax.scatter(means[0, :], means[1, :], c="grey", label="ALL MEANS")
        ax.scatter(constrained_means[0, :], constrained_means[1, :], c="red", label="SPATIALLY CONSTRAINED")
        ax.add_artist(env)
        ax.set_ylim(40, 220)
        ax.set_xlim(0, 175)
        ax.set_aspect("equal")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

    def phmm_single_mode_details(self, mode_id, poisson_model_file):
        # --------------------------------------------------------------------------------------------------------------
        # loads poisson hmm model from file and fits to awake behavior and analysis details of a single mode
        #
        # args:   - mode_id, integer: which mode to look at
        #               - poisson_model_file, string: file that contains the trained model
        #
        # returns: -
        # --------------------------------------------------------------------------------------------------------------

        with open(self.params.pre_proc_dir+"phmm/" + poisson_model_file+".pkl", "rb") as file:
            model = pickle.load(file)

        X = self.raster

        state_sequence = model.predict(X.T)

        trans_mat = model.transmat_

        mode_lambda = model.means_

        # plt.hist(state_sequence, bins=80)
        # plt.show()

        mode_ids, freq = np.unique(state_sequence, return_counts=True)
        nr_modes = model.means_.shape[0]
        # compute frequency of each mode
        mode_freq = np.zeros(nr_modes)
        mode_freq[mode_ids] = freq
        mode_freq = mode_freq.astype(int)

        means = np.zeros((2, nr_modes))
        cov = np.zeros((2, nr_modes))
        for mode in np.arange(nr_modes):
            mode_data = self.loc[state_sequence == mode, :]

            if len(mode_data) == 0:
                means[:, mode] = np.nan
                cov[:, mode] = np.nan
            else:
                means[:, mode] = np.mean(mode_data, axis=0)
                cov[:, mode] = np.var(mode_data, axis=0)

        loc_data = self.loc[:, :]

        center = np.min(loc_data, axis=0) + (np.max(loc_data, axis=0) - np.min(loc_data, axis=0)) * 0.5
        dist = loc_data - center

        rad = max(np.sqrt(np.square(dist[:, 0]) + np.square(dist[:, 1]))) + 1

        std_modes = np.sqrt(cov[0, :] + cov[1, :])
        std_modes[std_modes == 0] = np.nan


        mean = means[:, mode_id]
        cov_ = cov[:, mode_id]
        std_ = std_modes[mode_id]

        # Parameters to set
        mu_x = mean[0]
        variance_x = cov_[0]

        mu_y = mean[1]
        variance_y = cov_[1]

        # Create grid and multivariate normal
        x = np.linspace(center[0] - rad, center[0] + rad, int(2.2 * rad))
        y = np.linspace(0, 250, 250)
        X, Y = np.meshgrid(x, y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
        rv_normalized = rv.pdf(pos) / np.sum(rv.pdf(pos).flatten())

        fig, ax = plt.subplots()
        gauss = ax.imshow(rv_normalized)
        env = Circle((center[0], center[1]), rad, color="white", fill=False)
        ax.add_artist(env)
        ax.set_ylim(center[1] - 1.1 * rad, center[1] + 1.1 * rad)
        ax.scatter(loc_data[state_sequence == mode_id, 0], loc_data[state_sequence == mode_id,1],
                   alpha=1, c="white", marker=".", s=0.3, label="MODE " + str(mode_id) + " ASSIGNED")
        cb = plt.colorbar(gauss)
        cb.set_label("PROBABILITY")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("STD: " + str(np.round(std_, 2)))
        plt.legend()
        plt.show()

    """#################################################################################################################
    #  glm
    #################################################################################################################"""

    def infer_glm_awake(self, nr_gauss=20, std_gauss=10, spatial_bin_size_map=None, sel_range=None,
                        plot_for_control=False):
        # --------------------------------------------------------------------------------------------------------------
        # infers glm (paper: Correlations and Functional Connections in a Population of Grid Cells, 2015) from awake
        # data
        #
        # args:   - file_name, str: file name to save model
        #               - nr_gauss, int: how many Gaussians to distribute in environment
        #               - std_gauss, int: which standard deviation to use for all Gaussians
        #               - sel_range, range object: which data chunk to use for inference
        #               - spatial_bin_size_map, int: size of a spatial bin in cm
        #               - plot_for_control, bool: if True --> plot single steps of generating maps
        #
        # returns:      -
        # --------------------------------------------------------------------------------------------------------------

        # check if time bin size < 20 ms --> needed for binary assumption
        if self.params.time_bin_size > 0.02:
            raise Exception("TIME BIN SIZE MUST BE < 20 MS!")

        print(" - INFERENCE GLM USING AWAKE DATA ...\n")

        # params
        # --------------------------------------------------------------------------------------------------------------
        learning_rates = [100, 10, 1, 0.1, 0.01]
        likelihood = np.zeros(len(learning_rates))
        max_likelihood_per_iteration = []
        max_iter = 250
        cell_to_plot = 5

        file_name = self.params.session_name+"_"+self.experiment_phase_id+"_"+\
                    str(spatial_bin_size_map)+"cm_bins"+"_"+self.cell_type

        # place Gaussians uniformly across environment
        # --------------------------------------------------------------------------------------------------------------

        # get dimensions of environment
        x_min, x_max, y_min, y_max = min(self.loc[:, 0]), max(self.loc[:, 0]), min(self.loc[:, 1]), max(self.loc[:, 1])

        # get size of environment
        x_span = x_max - x_min
        y_span = y_max - y_min

        # width - length ratio of environment
        w_l_ratio = y_span/x_span

        # tile x and y with centers of Gaussians
        centers_gauss_x = np.linspace(x_min, x_max, nr_gauss)
        centers_gauss_y = np.linspace(y_min, y_max, int(np.round(nr_gauss*w_l_ratio)))

        # compute grid with x and y values of centers
        centers_gauss_x, centers_gauss_y = np.meshgrid(centers_gauss_x, centers_gauss_y)

        # get data used to infer model
        # --------------------------------------------------------------------------------------------------------------
        if sel_range is None:
            loc = self.loc
            raster = self.raster
        else:
            loc = self.loc[sel_range, :]
            raster = self.raster[:, sel_range]

        # data and parameter preparation for optimization
        # --------------------------------------------------------------------------------------------------------------

        # split location data into x and y coordinates
        x_loc = loc[:, 0]
        y_loc = loc[:, 1]

        # make binary raster --> if cell fires --> 1, if cell doesn't fire --> -1 --> ISING MODEL!
        bin_raster = -1 * np.ones((raster.shape[0], raster.shape[1]))
        bin_raster[raster > 0] = 1
        bin_raster = bin_raster.T

        # x_loc_m = loadmat("matlab.mat")["posx"]
        # y_loc_m = loadmat("matlab.mat")["posy"]

        # how many time bins / cells
        nr_time_bins = bin_raster.shape[0]-1
        nr_cells = bin_raster.shape[1]

        # compute distance from center of each Gaussian for every time bin
        dist_to_center_x = matlib.repmat(np.expand_dims(centers_gauss_x.flatten("F"), 1), 1, x_loc.shape[0] - 1) - \
                   matlib.repmat(x_loc[:-1].T, centers_gauss_x.flatten().shape[0], 1)
        dist_to_center_y = matlib.repmat(np.expand_dims(centers_gauss_y.flatten("F"), 1), 1, y_loc.shape[0] - 1) - \
                   matlib.repmat(y_loc[:-1].T, centers_gauss_y.flatten().shape[0], 1)

        # compute values of each Gaussian for each time bin
        gauss_values = simple_gaussian(xd=dist_to_center_x, yd=dist_to_center_y, std=std_gauss)

        # optimization of alpha values --> maximize data likelihood
        # --------------------------------------------------------------------------------------------------------------

        # alpha --> weights for Gaussians [#gaussians, #cells]
        alpha = np.zeros((gauss_values.shape[0], bin_raster.shape[1]))

        # bias --> firing rates of neurons (constant across time points!!!)
        bias = matlib.repmat(np.random.rand(nr_cells, 1), 1, nr_time_bins)

        for iter in range(max_iter):
            # compute gradient for alpha values --> dLikeli/dalpha
            dalpha=((gauss_values @ bin_raster[1:, :]) -
                    (np.tanh(alpha.T @ gauss_values + bias) @ gauss_values.T).T)/nr_time_bins

            # compute change in cost
            dcost = np.sum(bin_raster[1:, :].T - np.tanh(alpha.T @ gauss_values+bias), axis=1)/nr_time_bins

            # try different learning rates to maximize likelihood
            for i, l_r in enumerate(learning_rates):
                # compute new alpha values with gradient and learning rate
                alpha_n = alpha + l_r * dalpha
                # compute cost using old cost and update
                bias_n = bias + matlib.repmat(l_r*np.expand_dims(dcost, 1), 1, nr_time_bins)

                likelihood[i] = np.trace((alpha_n.T @ gauss_values + bias_n) @ bin_raster[1:, :])-np.sum(
                    np.sum(np.log(2*np.cosh(alpha_n.T @ gauss_values + bias_n)), axis=1))

            max_likelihood = np.max(likelihood)
            max_likelihood_per_iteration.append(max_likelihood)
            best_learning_rate_index = np.argmax(likelihood)

            # update bias --> optimize the bias term first before optimizing alpha values
            bias = bias + matlib.repmat(learning_rates[best_learning_rate_index]*
                                        np.expand_dims(dcost, 1), 1, nr_time_bins)

            # only start optimizing alpha values after n iterations
            if iter > 50:

                alpha = alpha + learning_rates[best_learning_rate_index] * dalpha

        # generation of maps for spatial bin size defined
        # --------------------------------------------------------------------------------------------------------------

        # if spatial_bin_size_map was not provided --> use spatial_resolution from parameter file
        if spatial_bin_size_map is None:
            spatial_bin_size_map = self.params.spatial_resolution

        nr_spatial_bins = int(np.round(x_span / spatial_bin_size_map))

        # generate grid by spatial binning
        x_map = np.linspace(x_min, x_max, nr_spatial_bins)
        y_map = np.linspace(y_min, y_max, int(np.round(nr_spatial_bins * w_l_ratio)))

        # compute grid from x and y values
        x_map, y_map = np.meshgrid(x_map, y_map)

        dist_to_center_x_map = matlib.repmat(np.expand_dims(centers_gauss_x.flatten("F"), 1), 1, x_map.flatten().shape[0])\
                               -matlib.repmat(x_map.flatten("F"), centers_gauss_x.flatten().shape[0], 1)
        dist_to_center_y_map = matlib.repmat(np.expand_dims(centers_gauss_y.flatten("F"), 1), 1, y_map.flatten().shape[0])\
                               -matlib.repmat(y_map.flatten("F"), centers_gauss_y.flatten().shape[0], 1)

        # compute values of each Gaussian for each time bin
        gauss_values_map = simple_gaussian(xd=dist_to_center_x_map, yd=dist_to_center_y_map, std=std_gauss)

        # compute resulting map
        res_map = np.exp(alpha.T @ gauss_values_map + matlib.repmat(bias[:, 0], gauss_values_map.shape[1], 1).T)/ \
                   (2*np.cosh(alpha.T @ gauss_values_map + matlib.repmat(bias[:, 0], gauss_values_map.shape[1], 1).T))

        # reshape to reconstruct 2D map
        res_map = matlib.reshape(res_map, (res_map.shape[0], nr_spatial_bins, int(np.round(nr_spatial_bins * w_l_ratio))))

        # compute occupancy --> mask results (remove non visited bins)
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # Fede's implementation:
        # ----------------------
        # centers_x = np.linspace(x_min, x_max + 0.1, nr_spatial_bins)
        # centers_y = np.linspace(y_min, y_max + 0.1, int(round(nr_spatial_bins*w_l_ratio)))
        #
        # dx = centers_x[1] - centers_x[0]
        # dy = centers_y[1] - centers_y[0]
        #
        # occ = np.zeros((centers_x.shape[0], centers_y.shape[0]))
        #
        # x_loc[x_loc > x_max] = x_min - 0.01
        # y_loc[y_loc > y_max] = y_min - 0.01
        #
        # for i in range(x_loc.shape[0]):
        #     xi = int(np.floor((x_loc[i]-x_min)/dx))+1
        #     yi = int(np.floor((y_loc[i] - y_min) / dy)) + 1
        #     if xi*yi > 0:
        #         occ[xi, yi] += 1
        #
        # occ_mask_fede = np.where(occ > 0, 1, 0)
        # occ_mask = np.repeat(occ_mask[np.newaxis, :, :], nr_cells, axis=0)
        # --------------------------------------------------------------------------------------------------------------

        # compute actual rate maps from used data --> to validate results
        rate_map, occ = self.rate_map_from_data(loc=loc, raster=raster, spatial_resolution=spatial_bin_size_map)

        # compute binary occupancy map --> used as a mask to filter non-visited bins
        occ_mask = np.where(occ > 0, 1, 0)
        occ_mask_plot = np.where(occ > 0, 1, np.nan)
        occ_mask = np.repeat(occ_mask[np.newaxis, :, :], nr_cells, axis=0)

        # save original map before applying the mask for plotting
        res_orig = res_map

        # apply mask to results
        res_map = np.multiply(res_map, occ_mask)

        if plot_for_control:

            plt.plot(max_likelihood_per_iteration)
            plt.title("LIKELIHOOD PER ITERATION")
            plt.xlabel("ITERATION")
            plt.ylabel("LIKELIHOOD")
            plt.show()

            # compute actual rate maps from used data --> to validate results
            rate_map_to_plot = np.multiply(rate_map[:, :, cell_to_plot], occ_mask_plot)
            plt.imshow(rate_map_to_plot.T, origin="lower")
            plt.scatter((centers_gauss_x-x_min)/spatial_bin_size_map, (centers_gauss_y-y_min)/spatial_bin_size_map
                        , s=0.1, label="GAUSS. CENTERS")
            plt.title("RATE MAP + GAUSSIAN CENTERS")
            a = plt.colorbar()
            a.set_label("FIRING RATE / 1/s")
            plt.legend()
            plt.show()

            a = alpha.T @ gauss_values_map
            a = matlib.reshape(a, (a.shape[0], nr_spatial_bins, int(np.round(nr_spatial_bins * w_l_ratio))))
            plt.imshow(a[cell_to_plot, :, :].T, interpolation='nearest', aspect='auto', origin="lower")
            plt.colorbar()
            plt.title("ALPHA.T @ GAUSSIANS")
            plt.show()

            plt.imshow(res_orig[cell_to_plot, :, :].T, origin="lower")
            plt.title("RES_MAP ORIGINAL (W/O OCC. MASK)")
            a = plt.colorbar()
            a.set_label("PROB. OF FIRING IN WINDOW (" +str(self.params.time_bin_size)+"s")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            plt.imshow(occ.T, origin="lower")
            plt.title("OCC MAP")
            a = plt.colorbar()
            a.set_label("SEC")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            plt.imshow(occ_mask[cell_to_plot, :, :].T, origin="lower")
            plt.title("OCC MAP BINARY")
            a = plt.colorbar()
            a.set_label("OCC: YES/NO")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            plt.imshow(res_map[cell_to_plot, :, :].T, origin="lower")
            plt.title("RES_MAP MASKED")
            a = plt.colorbar()
            a.set_label("PROB. OF FIRING IN WINDOW (" +str(self.params.time_bin_size)+"s)")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

        model_dic = {
            "rate_map": rate_map,
            "occ_map": occ,
            "occ_mask_plot": occ_mask_plot,
            "res_map": res_map,
            "alpha": alpha,
            "bias": bias,
            "centers_gauss_x": centers_gauss_x,
            "centers_gauss_y": centers_gauss_y,
            "std_gauss": std_gauss,
            "likelihood": max_likelihood_per_iteration,
            "time_bin_size": self.params.time_bin_size
        }

        with open('temp_data_old/awake_ising_maps/' + file_name + '.pkl', 'wb') as f:
            pickle.dump(model_dic, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_glm_awake(name, cell_id=None):
        with open('temp_data_old/awake_ising_maps/' + name + '.pkl', 'rb') as f:
             model_dic = pickle.load(f)

        centers_gauss_x = model_dic["centers_gauss_x"]
        centers_gauss_y = model_dic["centers_gauss_y"]
        std_gauss = model_dic["std_gauss"]
        alpha = model_dic["alpha"]
        rate_map = model_dic["rate_map"]
        res_map = model_dic["res_map"]
        occ_mask_plot = model_dic["occ_mask_plot"]
        time_bin_size = model_dic["time_bin_size"]

        # compute actual rate maps from used data --> to validate results
        rate_map_to_plot = np.multiply(rate_map[:, :, cell_id], occ_mask_plot)
        plt.imshow(rate_map_to_plot.T, origin="lower")

        plt.title("RATE MAP")
        a = plt.colorbar()
        a.set_label("FIRING RATE / Hz")
        plt.show()

        # print res map
        plt.imshow(res_map[cell_id, :, :].T, origin='lower', interpolation='nearest', aspect='auto')
        plt.title("RES MAP")
        plt.xlabel("X")
        plt.ylabel("Y")
        a = plt.colorbar()
        a.set_label("PROB. OF FIRING IN WINDOW (" + str(time_bin_size) + "s")
        plt.show()


"""#####################################################################################################################
#   TRIAL PARENT CLASS
#####################################################################################################################"""


class TrialParentClass:
    """Base class for tasked data with trial structure (e.g. cheeseboard, cross maze, t-maze)

    """

    def __init__(self, data_dic, cell_type, params, session_params, experiment_phase=None):
        """
        initializes TrialBaseClass

        :param data_dic: dictionary containing spike data
        :type data_dic: python dic
        :param cell_type: which cell type to use
        :type cell_type: str
        :param params: general analysis params
        :type params: class
        :param session_params: sessions specific params
        :type session_params: class
        """

        # get standard analysis parameters
        self.params = copy.deepcopy(params)
        # get session analysis parameters
        self.session_params = session_params

        self.cell_type = cell_type
        self.time_bin_size = params.time_bin_size

        # get spatial factor: cm per .whl arbitrary unit
        self.spatial_factor = self.session_params.data_params_dictionary["spatial_factor"]

        # check if list or dictionary is passed:
        # --------------------------------------------------------------------------------------------------------------
        if isinstance(data_dic, list):
            self.data_dic = data_dic[0]
        else:
            self.data_dic = data_dic

        # check if extended data dictionary is provided (contains lfp)
        if "eeg" in self.data_dic.keys():
            self.eeg = self.data_dic["eeg"]
        if "eegh" in self.data_dic.keys():
            self.eegh = self.data_dic["eegh"]

        # get all spike times --> check if two populations need to be combined
        if isinstance(cell_type, list) and len(cell_type) > 1:
            self.firing_times = {}
            for ct in cell_type:
                d = self.data_dic["spike_times"][ct]
                for k, v in d.items():  # d.items() in Python 3+
                    self.firing_times.setdefault(k, []).append(v)
        else:
            self.firing_times = self.data_dic["spike_times"][cell_type]
        # # get location data
        self.whl = self.data_dic["whl"]

        # get last recorded spike
        if "last_spike" in self.data_dic.keys():
            self.last_spike = self.data_dic["last_spike"]
        else:
            self.last_spike = None

        # --------------------------------------------------------------------------------------------------------------
        # get phase specific info (ID & description)
        # --------------------------------------------------------------------------------------------------------------

        # check if list is passed:
        if isinstance(experiment_phase, list):
            experiment_phase = experiment_phase[0]

        self.experiment_phase = experiment_phase
        self.experiment_phase_id = session_params.data_description_dictionary[self.experiment_phase]
        # --------------------------------------------------------------------------------------------------------------
        # extract session analysis parameters
        # --------------------------------------------------------------------------------------------------------------
        self.session_name = self.session_params.session_name
        self.nr_trials = len(self.data_dic["trial_data"])

        # initialize environment dimensions --> are updated later while loading all trial data
        self.x_min = np.inf
        self.x_max = -np.inf
        self.y_min = np.inf
        self.y_max = -np.inf

    """#################################################################################################################
    #  Data processing
    #################################################################################################################"""

    def get_spike_times(self, trials_to_use=None):
        # get rasters from trials

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))

        if trials_to_use is "all":
            for trial_id, _ in enumerate(self.trial_raster_list):
                raster = np.hstack((raster, self.trial_raster_list[trial_id]))

        elif trials_to_use is None:
            trials_to_use = self.default_trials

            for trial_id in trials_to_use:
                raster = np.hstack((raster, self.trial_raster_list[trial_id]))

        return raster

    def rate_map_from_data(self, loc, raster, spatial_resolution=None, gaussian_std=1, env_dim=None):
        # --------------------------------------------------------------------------------------------------------------
        # computes #spikes per bin from input data
        #
        # args:   - loc, array: location data (2D) synchronized with raster data
        #               - raster, array [nr_cells, nr_time_bins]: #spikes per cell /time bin
        #               - spatial_resolution, int: in cm
        #               - gaussian_std, float: std of gaussians for smoothing of rate map --> if 0: no smoothing
        #
        # returns:      - rate_map, array [x_coord, y_coord, cell_id], per spatial bin: spike rate (1/s)
        # --------------------------------------------------------------------------------------------------------------

        if spatial_resolution is None:
            spatial_resolution = self.params.spatial_resolution

        nr_cells = raster.shape[0]
        loc_ds = np.floor(loc / spatial_resolution).astype(int)

        if env_dim is None:
            # get dimensions of environment
            x_min, x_max, y_min, y_max = self.x_min, self.x_max, self.y_min, self.y_max
        else:
            x_min, x_max, y_min, y_max = env_dim[0], env_dim[1], env_dim[2], env_dim[3]

        # get size of environment
        x_span = x_max - x_min
        y_span = y_max - y_min

        # width - length ratio of environment
        w_l_ratio = y_span / x_span

        nr_spatial_bins = int(np.round(x_span / spatial_resolution))

        centers_x = np.linspace(x_min, x_max + 0.1, nr_spatial_bins)
        centers_y = np.linspace(y_min, y_max + 0.1, int(round(nr_spatial_bins * w_l_ratio)))

        dx = centers_x[1] - centers_x[0]
        dy = centers_y[1] - centers_y[0]

        # split location data into x and y coordinates
        x_loc = loc[:, 0]
        y_loc = loc[:, 1]

        x_loc[x_loc > x_max] = x_min - 0.01
        y_loc[y_loc > y_max] = y_min - 0.01

        occ = np.zeros((centers_x.shape[0], centers_y.shape[0]))
        raster_2d = np.zeros((nr_spatial_bins, int(np.round(nr_spatial_bins * w_l_ratio)), nr_cells))

        for i, (_, pop_vec) in enumerate(zip(loc_ds, raster.T)):
            xi = int(np.floor((x_loc[i] - x_min) / dx)) + 1
            yi = int(np.floor((y_loc[i] - y_min) / dy)) + 1
            if xi * yi > 0:
                occ[xi, yi] += 1
                raster_2d[xi, yi, :] += pop_vec

        # one pop vec is at time bin resolution --> need occupancy in seconds
        occ = occ * self.params.time_bin_size
        # make zeros to nan for division
        occ[occ == 0] = np.nan
        rate_map = np.nan_to_num(raster_2d / occ[..., None])
        occ = np.nan_to_num(occ)

        # rate[occ > 0.05] = rate[occ > 0.05] / occ[occ > 0.05]
        # if sigma_gauss > 0:
        #     rate = nd.gaussian_filter(rate, sigma=sigma_gauss)
        # rate[occ == 0] = 0

        # apply gaussian filtering --> smooth place fields
        for i in range(nr_cells):
            rate_map[:, :, i] = nd.gaussian_filter(rate_map[:, :, i], sigma=gaussian_std)

        return rate_map, occ

    """#################################################################################################################
    #  Getter methods
    #################################################################################################################"""

    def get_raster(self, trials_to_use=None):
        # get rasters from trials

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))

        if trials_to_use is "all":
            for trial_id, _ in enumerate(self.trial_raster_list):
                raster = np.hstack((raster, self.trial_raster_list[trial_id]))

        elif trials_to_use is None:
            trials_to_use = self.default_trials

            for trial_id in trials_to_use:
                raster = np.hstack((raster, self.trial_raster_list[trial_id]))

        return raster

    def get_raster_stillness(self, threshold_stillness=15, time_bin_size=0.01):

        # detect periods of stillness
        # --------------------------------------------------------------------------------------------------------------

        speed = self.get_speed()

        good_bins = np.zeros(speed.shape[0])
        good_bins[speed < threshold_stillness] = 1

        transitions = np.diff(good_bins)

        start = []
        end = []

        if good_bins[0] == 1:
            # first data point during stillness
            start.append(0)

        for bin_nr, tran in enumerate(transitions):
            if tran == -1:
                end.append(bin_nr)
            if tran == 1:
                start.append(bin_nr+1)

        if good_bins[-1] == 1:
            # last data point during stillness
            end.append(good_bins.shape[0])

        start = np.array(start)
        end = np.array(end)
        duration = (end-start)*0.0256

        # delete all intervals that are shorter than time bin size
        start = start[duration > time_bin_size]
        end = end[duration > time_bin_size]

        event_times = np.vstack((start, end)).T

        stillness_rasters = []
        # get stillness rasters
        for e_t in event_times:
            stillness_rasters.append(PreProcessAwake(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl, spatial_factor=self.spatial_factor).interval_temporal_binning(interval=e_t, interval_freq=1,
                                                                            time_bin_size=time_bin_size))

        stillness_rasters = np.hstack(stillness_rasters)

        return stillness_rasters

    def get_speed(self):
        """
        returns speed of all data (not split into trials) at 0.0256s
        :return:
        :rtype:
        """
        return PreProcessAwake(firing_times=self.firing_times, params=self.params,
                        whl=self.whl, spatial_factor=self.spatial_factor).get_speed()

    def get_goal_locations(self):
        return self.goal_locations

    def get_raster_and_trial_times(self, trials_to_use=None):

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))

        trial_lengths = []
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            trial_lengths.append(self.trial_raster_list[trial_id].shape[1])

        return raster, trial_lengths

    def get_env_dim(self):
        return [self.x_min, self.x_max, self.y_min, self.y_max]

    def get_raster_location_speed(self, trials_to_use=None):

        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))
        speed = np.empty(0)
        location = np.empty((0, 2))

        if trials_to_use is None:
            trials_to_use = self.default_trials

        if trials_to_use is "all":
            for trial_id, _ in enumerate(self.trial_raster_list):
                raster = np.hstack((raster, self.trial_raster_list[trial_id]))
                speed = np.hstack((speed, self.trial_speed_list[trial_id]))
                location = np.vstack((location, self.trial_loc_list[trial_id]))


        else:
            for trial_id in trials_to_use:
                raster = np.hstack((raster, self.trial_raster_list[trial_id]))
                speed = np.hstack((speed, self.trial_speed_list[trial_id]))
                location = np.vstack((location, self.trial_loc_list[trial_id]))

        return raster, location, speed

    def get_basic_info(self):

        print("NUMBER CELLS: " +str(self.trial_raster_list[0].shape[0])+"\n")

        print("TRIAL LENGTH:\n")
        len_trials = []
        for trial_id, trial_raster in enumerate(self.trial_raster_list):
            len_trials.append(np.round(trial_raster.shape[1]*self.time_bin_size,2))
            print(" - TRIAL " + str(trial_id) + ": " + str(np.round(trial_raster.shape[1]*self.time_bin_size,2))+ "s")

        plt.plot(len_trials, marker=".", c="r")
        plt.title("TRIAL DURATION")
        plt.xlabel("TRIAL ID")
        plt.ylabel("TRIAL DURATION / s")
        plt.grid()
        plt.show()

        print("\n#TRIALS FOR DURATION STARTING FROM FIRST:\n")
        cs = np.cumsum(np.array(len_trials))
        for i,c in enumerate(cs):
            print(" - TRIAL " + str(i) +": "+str(np.round(c/60,2))+"min")

        print("\n#TRIALS FOR DURATION STARTING FROM LAST:\n")
        cs = np.cumsum(np.flip(np.array(len_trials)))
        for i,c in enumerate(cs):
            print(" - TRIAL " + str(cs.shape[0]-i-1) +": "+str(np.round(c/60,2))+"min")

    def get_cell_classification_labels(self):
        """
        returns cell labels for stable, increasing, decreasing cells

        @return: cell indices for stable, decreasing, increasing cells
        """
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_decreasing = class_dic["decrease_cell_ids"].flatten()
        cell_ids_increasing = class_dic["increase_cell_ids"].flatten()

        return cell_ids_stable, cell_ids_decreasing, cell_ids_increasing

    def get_info_for_trial(self, trial_id, cell_id):
        """
        shows basic data for trial and cell

        @param trial_id: which trial to use
        @type trial_id: int
        @param cell_id: which cell to use
        @type cell_id: int
        """
        loc = self.trial_loc_list[trial_id]

        plt.scatter(loc[:, 0], loc[:, 1])
        plt.show()

        raster = self.trial_raster_list[trial_id]
        rate_map, occ = self.rate_map_from_data(loc=loc, raster=raster, gaussian_std=5)

        plt.imshow(occ.T,origin="lower" )
        plt.colorbar()
        plt.show()
        plt.imshow(rate_map[:,:,cell_id].T,origin="lower")
        plt.colorbar()
        plt.show()

    def get_rate_maps(self, spatial_resolution=5, env_dim=None, trials_to_use=None):

        raster, location, speed = self.get_raster_location_speed(trials_to_use=trials_to_use)

        rate_maps, _ = self.rate_map_from_data(loc=location, raster=raster, spatial_resolution=spatial_resolution,
                                              env_dim=env_dim)

        return rate_maps

    def get_nr_of_trials(self):
        return len(self.trial_loc_list)

    def get_occ_map(self, spatial_resolution=1, env_dim=None, trials_to_use=None):

        raster, location, speed = self.get_raster_location_speed(trials_to_use=trials_to_use)

        _, occ_map = self.rate_map_from_data(loc=location, raster=raster, spatial_resolution=spatial_resolution,
                                              env_dim=env_dim)

        return occ_map

    def get_rate_maps_occ_maps_temporal_splits(self, spatial_resolution=None, nr_of_splits=2, env_dim=None,
                                               exclude_first_trial=False):
        """
        computes rate maps using splits of the data (in terms of time)

        :param spatial_resolution: spatial resolution of rate maps in cm2
        :type spatial_resolution: int
        :param nr_of_splits: in how many splits to divide the data
        :type nr_of_splits: int
        :return: list of rate maps (one list entry with rate maps for all cells for each split) and list of occ maps
        :rtype: list, list
        """

        if exclude_first_trial:
            # rasters & location from all except first trial
            nr_trials = len(self.trial_raster_list)
            raster, loc, _ = self.get_raster_location_speed(trials_to_use=np.arange(1, nr_trials))

        else:
            # get all rasters & location
            raster, loc, _ = self.get_raster_location_speed(trials_to_use="all")

        len_split = int(loc.shape[0]/nr_of_splits)

        list_rate_maps = []
        list_occ_maps = []

        for split_id in range(nr_of_splits):
            rate_maps=None
            occ = None
            rate_maps, occ = self.rate_map_from_data(loc=loc[split_id*len_split:(split_id+1)*len_split, :],
                                                     raster=raster[:, split_id*len_split:(split_id+1)*len_split],
                                                     spatial_resolution=spatial_resolution, env_dim=env_dim)

            list_rate_maps.append(rate_maps)
            list_occ_maps.append(occ)

        return list_rate_maps, list_occ_maps

    """#################################################################################################################
    #  Plotting methods
    #################################################################################################################"""

    def plot_speed_per_trial(self):

        avg = []
        for s in self.trial_speed_list:
            avg.append(np.mean(s))

        plt.plot(avg, marker=".")
        plt.title("AVG. SPEED PER TRIAL")
        plt.xlabel("TRIAL NR.")
        plt.ylabel("AVG. SPEED / cm/s")
        plt.show()

    def plot_rate_maps(self, spatial_resolution=5):

        rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution)

        for cell_id, rate_map in enumerate(rate_maps.T):
            plt.imshow(rate_map)
            plt.title("CELL "+str(cell_id))
            a = plt.colorbar()
            a.set_label("FIRING RATE (Hz)")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

    def plot_rate_map(self, cell_id, spatial_resolution=5, save_fig=False, trials_to_use=None):

        plt.style.use('default')
        rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution, trials_to_use=trials_to_use)
        rate_map_to_plot = rate_maps[:,:,cell_id]
        # rate_map_to_plot[rate_map_to_plot == 0] = np.nan
        plt.imshow(rate_map_to_plot)
        a = plt.colorbar()
        a.set_label("FIRING RATE (Hz)")
        plt.xlabel("X")
        plt.ylabel("Y")

        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("rate_map_example2.svg", transparent="True")
        else:
            plt.show()

    def plot_summary_rate_map(self, cells_to_use="all", spatial_resolution=3, normalize=False):
        # load all rate maps
        rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        if cells_to_use == "stable":

            cell_ids = class_dic["stable_cell_ids"].flatten()

        elif cells_to_use == "increasing":

            cell_ids = class_dic["increase_cell_ids"].flatten()

        elif cells_to_use == "decreasing":

            cell_ids = class_dic["decrease_cell_ids"].flatten()

        elif cells_to_use == "all":

            cell_ids = np.arange(rate_maps.shape[2])

        # normalize rate maps
        max_per_cell = np.max(np.reshape(rate_maps, (rate_maps.shape[0]*rate_maps.shape[1], rate_maps.shape[2])), axis=0)
        max_per_cell[max_per_cell == 0] = 1e-22
        norm_rate_maps = rate_maps / max_per_cell

        # compute summed up rate map
        sum_rate_map = np.sum(norm_rate_maps[:, :, cell_ids], axis=2)

        # mask with occupancy
        occ = self.get_occ_map(spatial_resolution=spatial_resolution)
        sum_rate_map[occ==0] = np.nan

        if normalize:
            sum_rate_map = sum_rate_map / np.sum(np.nan_to_num(sum_rate_map.flatten()))
            plt.imshow(sum_rate_map.T)
            a = plt.colorbar()
            a.set_label("Sum firing rate / normalized to 1")
        else:
            plt.imshow(sum_rate_map.T)
            a = plt.colorbar()
            a.set_label("SUMMED FIRING RATE")
        for g_l in self.goal_locations:
            plt.scatter((g_l[0]-self.x_min)/spatial_resolution, (g_l[1]-self.y_min)/spatial_resolution,
                        color="white", label="Goal locations")
        # plt.xlim(55,260)
        plt.title("CELLS USED: " + cells_to_use)
        plt.show()
        # exit()
        # plt.rcParams['svg.fonttype'] = 'none'
        # plt.savefig("dec_firing_changes.svg", transparent="True")

    def plot_tracking(self, ax=None, trials_to_use=None):
        if trials_to_use is None:
            trials_to_use = self.default_trials
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))
        ax.scatter(loc[:, 0], loc[:, 1], color="grey", s=1, label="TRACKING")
        plt.show()

    """#################################################################################################################
    #  Saving methods
    #################################################################################################################"""

    def save_raster(self, filename, trials_to_use=None):
        raster, _, _ = self.get_raster_location_speed(trials_to_use=trials_to_use)
        with open(filename+'.pkl', 'wb') as f:
            pickle.dump(raster, f)

    """#################################################################################################################
    #  Standard analysis
    #################################################################################################################"""

    def bayesian_decoding(self, test_perc=0.5):

        if self.params.stable_cell_method == "k_means":
            # load only stable cells
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_k_means.pickle", "rb") as f:
                class_dic = pickle.load(f)

        elif self.params.stable_cell_method == "mean_firing_awake":
            # load only stable cells
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_mean_firing_awake.pickle", "rb") as f:
                class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()
        inc_cells = class_dic["increase_cell_ids"].flatten()

        trials_to_use = self.default_trials

        first_trial = trials_to_use[0]
        last_trial = trials_to_use[-1]
        nr_test_trials = int((last_trial-first_trial)*test_perc)

        shuff_trials_to_use = np.array(np.copy(trials_to_use))
        np.random.shuffle(shuff_trials_to_use)

        test_trials = shuff_trials_to_use[:nr_test_trials]
        train_trials = shuff_trials_to_use[nr_test_trials:]

        # get rate map --> need to add x_min, y_min of environment to have proper location info
        rate_maps = self.get_rate_maps(spatial_resolution=1, trials_to_use=train_trials)

        # flatten rate maps
        rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0]*rate_maps.shape[1], rate_maps.shape[2]))

        test_raster_orig, test_loc_orig, _ = self.get_raster_location_speed(trials_to_use=test_trials)

        # test_raster = test_raster_orig[:,:50]
        # test_loc = test_loc_orig[:50, :]
        test_raster = test_raster_orig
        test_loc = test_loc_orig

        test_raster_stable = test_raster[stable_cells,:]
        rate_maps_flat_stable = rate_maps_flat[:,stable_cells]
        pred_loc_stable = []
        error_stable=[]
        for pop_vec, loc in zip(test_raster_stable.T, test_loc):
            bl = bayes_likelihood(frm=rate_maps_flat_stable.T, pop_vec=pop_vec, log_likeli=False)
            bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
            pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
            pred_x = pred_bin[0] + self.x_min
            pred_y = pred_bin[1] + self.y_min
            pred_loc_stable.append([pred_x, pred_y])

            # plt.scatter(pred_x, pred_y, color="red")
            # plt.scatter(loc[0], loc[1], color="gray")
            error_stable.append(np.sqrt((pred_x-loc[0])**2+(pred_y-loc[1])**2))
        pred_loc = np.array(pred_loc_stable)

        test_raster_dec = test_raster[dec_cells, :]
        rate_maps_flat_dec = rate_maps_flat[:, dec_cells]
        pred_loc_dec = []
        error_dec = []
        for pop_vec, loc in zip(test_raster_dec.T, test_loc):
            bl = bayes_likelihood(frm=rate_maps_flat_dec.T, pop_vec=pop_vec, log_likeli=False)
            bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
            pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
            pred_x = pred_bin[0] + self.x_min
            pred_y = pred_bin[1] + self.y_min
            pred_loc_stable.append([pred_x, pred_y])

            # plt.scatter(pred_x, pred_y, color="red")
            # plt.scatter(loc[0], loc[1], color="gray")
            error_dec.append(np.sqrt((pred_x - loc[0]) ** 2 + (pred_y - loc[1]) ** 2))
        pred_loc = np.array(pred_loc_stable)

        plt.hist(error_stable, density=True, color="#ffdba1", label="STABLE")
        plt.hist(error_dec, density=True, color="#a0c4e4", label="DECREASING", alpha=0.6)

        plt.xlabel("ERROR (cm)")
        plt.ylabel("DENSITY")
        plt.legend()
        _, y_max = plt.gca().get_ylim()
        plt.vlines(np.median(np.array(error_stable)), 0, y_max, colors="y")
        plt.vlines(np.median(np.array(error_dec)), 0, y_max, colors="b")
        plt.show()


        exit()

        plt.plot(test_loc[:, 0], test_loc[:, 1], color="lightgray")
        # plt.plot(pred_loc[:,0], pred_loc[:,1], color="lightcoral")
        plt.scatter(pred_loc[:,0], pred_loc[:,1], color="red", label="PREDICTED LOC")
        plt.scatter(test_loc[:,0], test_loc[:,1], color="white", label="TRUE LOC")
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("BAYESIAN DECODING")
        plt.legend()
        plt.show()
        # for gl in self.goal_locations:
        #     plt.scatter(gl[0], gl[1], color="w")

        plt.hist(error, density=True)
        plt.xlabel("ERROR (cm)")
        plt.ylabel("DENSITY")
        plt.title("ERROR")
        plt.show()

        # get rate map --> need to add x_min, y_min of environment to have proper location info
        rate_maps = self.get_rate_maps(spatial_resolution=1, trials_to_use=None)
                # flatten rate maps
        rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0]*rate_maps.shape[1], rate_maps.shape[2]))

        pred_loc = []
        error = []
        for pop_vec, loc in zip(test_raster.T, test_loc):
            bl = bayes_likelihood(frm=rate_maps_flat.T, pop_vec=pop_vec, log_likeli=False)
            bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
            pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
            pred_x = pred_bin[0] + self.x_min
            pred_y = pred_bin[1] + self.y_min
            pred_loc.append([pred_x, pred_y])

            # plt.scatter(pred_x, pred_y, color="red")
            # plt.scatter(loc[0], loc[1], color="gray")
            error.append(np.sqrt((pred_x - loc[0]) ** 2 + (pred_y - loc[1]) ** 2))
        pred_loc = np.array(pred_loc)

        plt.plot(test_loc[:, 0], test_loc[:, 1], color="lightgray")
        # plt.plot(pred_loc[:,0], pred_loc[:,1], color="lightcoral")
        plt.scatter(pred_loc[:, 0], pred_loc[:, 1], color="red", label="PREDICTED LOC")
        plt.scatter(test_loc[:, 0], test_loc[:, 1], color="white", label="TRUE LOC")
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("BAYESIAN DECODING")
        plt.legend()
        plt.show()
        # for gl in self.goal_locations:
        #     plt.scatter(gl[0], gl[1], color="w")

        plt.hist(error, density=True)
        plt.xlabel("ERROR (cm)")
        plt.ylabel("DENSITY")
        plt.title("ERROR")
        plt.show()

    def nr_spikes_per_time_bin(self, trials_to_use=None, cells_to_use="all_cells"):

        if trials_to_use is None:
            trials_to_use = self.default_trials

        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.zeros((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))

        if cells_to_use == "stable_cells":
            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            stable_cells = class_dic["stable_cell_ids"].flatten()
            raster = raster[stable_cells, :]

        elif cells_to_use == "decreasing_cells":
                        # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            dec_cells = class_dic["decrease_cell_ids"].flatten()
            raster = raster[dec_cells, :]

        spikes_per_bin = np.sum(raster, axis=0)
        y,_,_ = plt.hist(spikes_per_bin, bins=50)
        plt.vlines(np.mean(spikes_per_bin), 0, y.max(), colors="r",
                   label="MEAN: "+str(np.round(np.mean(spikes_per_bin),2)))
        plt.vlines(np.median(spikes_per_bin), 0, y.max(), colors="blue",
                   label="MEDIAN: "+str(np.median(spikes_per_bin)))
        plt.legend()
        plt.title("#SPIKES PER "+str(self.params.time_bin_size)+"s TIME BIN \n "+cells_to_use)
        plt.xlabel("#SPIKES PER TIME BIN")
        plt.ylabel("COUNT")
        plt.show()

    def phase_preference_per_cell_subset(self, angle_20k, cell_ids, trials_to_use=None):
        if trials_to_use is None:
            trials_to_use = range(len(self.trial_raster_list))

        # spike times at 20kHz
        spike_times = self.data_dic["spike_times"][self.cell_type]

        # get keys from dictionary and get correct order
        cell_names = []
        for key in spike_times.keys():
            cell_names.append(key[4:])
        cell_names = np.array(cell_names).astype(int)
        cell_names.sort()

        # start_times, end_times of trials at .whl resolution (20kHz/512) --> up-sample to match spike frequency
        trial_timestamps_20k = self.data_dic["trial_timestamps"] * 512

        pref_angle = []

        for cell_id in cell_names[cell_ids]:
            cell_spike_times = spike_times["cell" + str(cell_id)]
            # concatenate trial data
            all_cell_spikes = []
            for trial_id in trials_to_use:
                all_cell_spikes.extend(cell_spike_times[np.logical_and(trial_timestamps_20k[0,trial_id] < cell_spike_times,
                                                       cell_spike_times < trial_timestamps_20k[1,trial_id])])

            # make array
            spk_ang = angle_20k[all_cell_spikes]
            pref_angle.append(np.angle(np.sum(np.exp(-1j * spk_ang))))

        return np.array(pref_angle)

    def phase_preference_analysis(self, oscillation="theta", tetrode=1, plot_for_control=False, plotting=True):

        # get lfp data
        # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
        # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s
        lfp = self.eegh[:, tetrode]

        # downsample to dt = 0.001 --> 1kHz --> take every 5th value
        lfp = lfp[::5]

        # Say you have an LFP signal LFP_Data and some spikes from a cell spk_t
        # First we extract the angle from the signal in a specific frequency band
        # Frequency Range to Extract, you can also select it AFTER running the wavelet on the entire frequency spectrum,
        # by using the variable frequency to select the desired ones
        if oscillation == "theta":
            Frq_Limits = [8, 12]
        elif oscillation == "slow_gamma":
            Frq_Limits = [20, 50]
        elif oscillation == "medium_gamma":
            Frq_Limits = [60, 90]
        else:
            raise Exception("Oscillation not defined!")
        # [8,12] Theta
        # [20,50] Slow Gamma
        # [60,90] Medium Gamma
        # LFP time bin duration in seconds
        # dt = 1/5e3
        dt=0.001
        # ‘morl’ wavelet
        wavelet = "cmor1.5-1.0" # 'cmor1.5-1.0'
        scales = np.arange(1,128)
        s2f = pywt.scale2frequency(wavelet, scales) / dt
        # This block is just to setup the wavelet analysis
        scales = scales[(s2f >= Frq_Limits[0]) * (s2f < Frq_Limits[1])]
        # scales = scales[np.logical_and(s2f >= Frq_Limits[0], s2f < Frq_Limits[1])]
        print(" - started wavelet decomposition ...")
        # Wavelet decomposition
        [cfs, frequencies] = pywt.cwt(data=lfp, scales=scales, wavelet=wavelet, sampling_period=dt, axis=0)
        print(" - done!")
        # This is the angle
        angl = np.angle(np.sum(cfs, axis=0))

        # plot for control
        if plot_for_control:
            plt.plot(lfp[:200])
            plt.xlabel("Time")
            plt.ylabel("LFP")
            plt.show()

            for i in range(frequencies.shape[0]):
                plt.plot(cfs[i, :200])
            plt.xlabel("Time")
            plt.ylabel("Coeff")
            plt.show()

            plt.plot(np.sum(cfs[:, :200], axis=0), label="coeff_sum")
            plt.plot(angl[:200]/np.max(angl[:200]), label="angle")
            plt.xlabel("Time")
            plt.ylabel("Angle (norm) / Coeff_sum (norm)")
            plt.legend()
            plt.show()

        # interpolate results to match 20k
        # --------------------------------------------------------------------------------------------------------------
        x_1k = np.arange(lfp.shape[0])*dt
        x_20k = np.arange(lfp.shape[0]*20)*1/20e3
        angle_20k = np.interp(x_20k, x_1k, angl, left=np.nan, right=np.nan)

        if plot_for_control:
            plt.plot(angle_20k[:4000])
            plt.ylabel("Angle")
            plt.xlabel("Time bin (20kHz)")
            plt.show()

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        stable = class_dic["stable_cell_ids"]
        dec = class_dic["decrease_cell_ids"]
        inc = class_dic["increase_cell_ids"]

        pref_angle_stable = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=stable)
        pref_angle_dec = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=dec)
        pref_angle_inc = self.phase_preference_per_cell_subset(angle_20k=angle_20k, cell_ids=inc)

        pref_angle_stable_deg = pref_angle_stable *180/np.pi
        pref_angle_dec_deg = pref_angle_dec * 180 / np.pi
        pref_angle_inc_deg = pref_angle_inc * 180 / np.pi

        if plotting:
            plt.hist(pref_angle_stable_deg, density=True, label="stable")
            plt.hist(pref_angle_dec_deg, density=True, label="dec")
            plt.hist(pref_angle_inc_deg, density=True, label="inc")
            plt.show()

        all_positive_angles_stable = np.copy(pref_angle_stable)
        all_positive_angles_stable[all_positive_angles_stable < 0] = 2*np.pi+all_positive_angles_stable[all_positive_angles_stable < 0]

        all_positive_angles_dec = np.copy(pref_angle_dec)
        all_positive_angles_dec[all_positive_angles_dec < 0] = 2 * np.pi + all_positive_angles_dec[
            all_positive_angles_dec < 0]

        all_positive_angles_inc = np.copy(pref_angle_inc)
        all_positive_angles_inc[all_positive_angles_inc < 0] = 2 * np.pi + all_positive_angles_inc[
            all_positive_angles_inc < 0]

        if plotting:

            bins_number = 10  # the [0, 360) interval will be subdivided into this
            # number of equal bins
            bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
            angles = all_positive_angles_stable
            n, _, _ = plt.hist(angles, bins, density=True)

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

            plt.clf()
            width = 2 * np.pi / bins_number
            ax = plt.subplot(1, 1, 1, projection='polar')
            bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)
            for bar in bars:
                bar.set_alpha(0.5)
            ax.set_title("inc. cells")
            plt.show()

        else:
            return all_positive_angles_stable, all_positive_angles_dec, all_positive_angles_inc

    def check_oscillation(self, oscillation="theta", plot_for_control=False, plotting=True):
        if oscillation == "theta":
            Frq_Limits = [8, 12]
        elif oscillation == "slow_gamma":
            Frq_Limits = [20, 50]
        elif oscillation == "medium_gamma":
            Frq_Limits = [60, 90]
        else:
            raise Exception("Oscillation not defined!")
        for tetrode in range(10):
            # get lfp data
            # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
            # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s
            lfp = self.eegh[:, tetrode]

            # downsample to dt = 0.001 --> 1kHz --> take every 5th value
            lfp = lfp[::5]

            # bandpass: freq_nyquist --> sampling freq / 2
            sig_filtered = butterworth_bandpass(input_data=lfp, freq_nyquist=1000 / 2, freq_lo_bound=Frq_Limits[0],
                                                freq_hi_bound=Frq_Limits[1])

            plt.plot(sig_filtered[1000:1300], label="tet" + str(tetrode))
        plt.legend()
        plt.show()

    """#################################################################################################################
    #  SWR analysis
    #################################################################################################################"""

    def detect_swr(self, thr=4, plot_for_control=False):
        """
        detects swr in lfp and returns start, peak and end timings at params.time_bin_size resolution
        ripple frequency: 140-240 Hz

        @param thr: nr. std. above average to detect ripple event (usually: 4-6)
        @type thr: int
        @param plot_for_control: True to plot intermediate results
        @type plot_for_control: bool
        @return: start, end, peak of each swr in seconds
        @rtype: int, int, int
        """
        if not hasattr(self.session_params, 'lfp_tetrodes'):
            self.session_params.lfp_tetrodes = None

        file_name = self.session_name + "_" + self.experiment_phase_id + "_swr_" + \
                    self.cell_type +"_tet_"+str(self.session_params.lfp_tetrodes)+ ".npy"
        # check if results exist already
        if not os.path.isfile(self.params.pre_proc_dir+"swr_periods/" + file_name):

            # check if results exist already --> if not

            # upper and lower bound in Hz for SWR
            freq_lo_bound = 140
            freq_hi_bound = 240

            # load data
            # ["eeg"] --> all channels downsampled 16 times (1.25kHz)
            # ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz) --> time bin size = 0.0002s

            # check if one tetrode or all tetrodes to use
            if self.session_params.lfp_tetrodes is None:
                print(" - DETECTING SWR USING ALL TETRODES ...\n")
                data = self.eegh[:, :]
            else:
                print(" - DETECTING SWR USING TETRODE(S) "+str(self.session_params.lfp_tetrodes) +" ...\n")
                data = self.eegh[:, self.session_params.lfp_tetrodes]
            # freq of the input signal (eegh --> 5kHz --> freq=5000)
            freq = 5000
            # for low pass filtering of the signal before z-scoring (20-30Hz is good)
            low_pass_cut_off_freq = 30
            # minimum gap in seconds between events. If two events have
            # a gap < min_gap_between_events --> events are joint and become one event
            min_gap_between_events = 0.1

            # if data is too large --> need to chunk it up

            if data.shape[0] > 10000000:
                start_times = np.zeros(0)
                peak_times = np.zeros(0)
                end_times = np.zeros(0)
                size_chunk = 10000000
                for nr_chunk in range(np.ceil(data.shape[0]/size_chunk).astype(int)):
                    chunk_data = data[nr_chunk*size_chunk:min(data.shape[0], (nr_chunk+1)*size_chunk)]

                    # compute offset in seconds for current chunk
                    offset_sec = nr_chunk * size_chunk * 1/freq

                    start_times_chunk, end_times_chunk, peak_times_chunk = self.detect_lfp_events(data=chunk_data,
                                                                                freq=freq, thr=thr,
                                                                                freq_lo_bound=freq_lo_bound,
                                                                                freq_hi_bound=freq_hi_bound,
                                                                                low_pass_cut_off_freq=low_pass_cut_off_freq,
                                                                                min_gap_between_events=min_gap_between_events,
                                                                                plot_for_control=plot_for_control)

                    # check if event was detected
                    if not start_times_chunk is None:
                        start_times = np.hstack((start_times, (start_times_chunk + offset_sec)))
                        end_times = np.hstack((end_times, (end_times_chunk + offset_sec)))
                        peak_times = np.hstack((peak_times, (peak_times_chunk + offset_sec)))

            else:
                # times in seconds
                start_times, end_times, peak_times = self.detect_lfp_events(data=data, freq=freq, thr=thr,
                                                                            freq_lo_bound=freq_lo_bound,
                                                                            freq_hi_bound=freq_hi_bound,
                                                                            low_pass_cut_off_freq=low_pass_cut_off_freq,
                                                                            min_gap_between_events=min_gap_between_events,
                                                                            plot_for_control=plot_for_control)

            result_dic = {
                "start_times": start_times,
                "end_times": end_times,
                "peak_times": peak_times
            }

            outfile = open(self.params.pre_proc_dir+"swr_periods/"+file_name, 'wb')
            pickle.dump(result_dic, outfile)
            outfile.close()

        # load results from file
        infile = open(self.params.pre_proc_dir+"swr_periods/" + file_name, 'rb')
        result_dic = pickle.load(infile)
        infile.close()

        start_times = result_dic["start_times"]
        end_times = result_dic["end_times"]
        peak_times = result_dic["peak_times"]

        print(" - " + str(start_times.shape[0]) + " SWRs FOUND\n")

        return start_times, end_times, peak_times

    @staticmethod
    def detect_lfp_events(data, freq, thr, freq_lo_bound, freq_hi_bound, low_pass_cut_off_freq,
                          min_gap_between_events, plot_for_control=False):
        """
        detects events in lfp and returns start, peak and end timings at params.time_bin_size resolution

        @param data: input data (either from one or many tetrodes)
        @type data: array [nxm]
        @param freq: sampling frequency of input data in Hz
        @type freq: int
        @param thr: nr. std. above average to detect ripple event
        @type thr: int
        @param freq_lo_bound: lower bound for frequency band in Hz
        @type freq_lo_bound: int
        @param freq_hi_bound: upper bound for frequency band in Hz
        @type freq_hi_bound: int
        @param low_pass_cut_off_freq: cut off frequency for envelope in Hz
        @type low_pass_cut_off_freq: int
        @param min_gap_between_events: minimum gap in seconds between events. If two events have
         a gap < min_gap_between_events --> events are joint and become one event
        @type min_gap_between_events: float
        @param plot_for_control: plot some examples to double check detection
        @type plot_for_control: bool
        @return: start_times, end_times, peak_times of each event in seconds --> are all set to None if no event was
        detected
        @rtype: array, array, array
        """

        # check if data from one or multiple tetrodes was provided
        if len(data.shape) == 1:
            # only one tetrode
            # nyquist theorem --> need half the frequency
            sig_bandpass = butterworth_bandpass(input_data=data, freq_nyquist=freq / 2, freq_lo_bound=freq_lo_bound,
                                                freq_hi_bound=freq_hi_bound)

            # compute rectified signal
            sig_abs = np.abs(sig_bandpass)

            # if only peak position is supposed to be returned

            # low pass filter signal
            sig_lo_pass = butterworth_lowpass(input_data=sig_abs, freq_nyquist=freq / 2,
                                              cut_off_freq=low_pass_cut_off_freq)
            # z-score
            sig_z_scored = zscore(sig_lo_pass)

        else:
            # multiple tetrodes
            combined_lo_pass = []
            # go trough all tetrodes
            for tet_data in data.T:
                # nyquist theorem --> need half the frequency
                sig_bandpass = butterworth_bandpass(input_data=tet_data, freq_nyquist=freq/2, freq_lo_bound=freq_lo_bound,
                                                           freq_hi_bound=freq_hi_bound)

                # compute rectified signal
                sig_abs = np.abs(sig_bandpass)

                # if only peak position is supposed to be returned

                # low pass filter signal
                sig_lo_pass = butterworth_lowpass(input_data=sig_abs, freq_nyquist=freq/2,
                                                  cut_off_freq=low_pass_cut_off_freq)

                combined_lo_pass.append(sig_lo_pass)

            combined_lo_pass = np.array(combined_lo_pass)
            avg_lo_pass = np.mean(combined_lo_pass, axis=0)

            # z-score
            sig_z_scored = zscore(avg_lo_pass)

        # find entries above the threshold
        bool_above_thresh = sig_z_scored > thr
        sig = bool_above_thresh.astype(int) * sig_z_scored

        # find event start / end
        diff = np.diff(sig)
        start = np.argwhere(diff > 0.8 * thr)
        end = np.argwhere(diff < -0.8 * thr)

        # check that first element is actually the start (not that event started before this chunk and we only
        # observe the end of the event)
        if end[0] < start[0]:
            # if first end is before first start --> need to delete first end
            print("  --> CURRENT CHUNK: FIRST END BEFORE FIRST START --> DELETED FIRST END ELEMENT ")
            end = end[1:]

        if end[-1] < start[-1]:
            # there is another start after the last end --> need to delete last start
            print("  --> CURRENT CHUNK: LAST START AFTER LAST END --> DELETED LAST START ELEMENT ")
            start = start[:-1]

        # join events if there are less than min_gap_between_events seconds apart --> this is then one event!
        # compute differences between start time of n+1th event with end time of nth --> if < gap --> delete both
        # entries
        gap = np.squeeze((start[1:] - end[:-1]) * 1 / freq)
        to_delete = np.argwhere(gap < min_gap_between_events)
        end = np.delete(end, to_delete)
        start = np.delete(start, to_delete + 1)

        # add 25ms to the beginning of event (many spikes occur in that window)
        pad_infront = np.round(0.025/(1/freq)).astype(int)
        start -= pad_infront
        # don't want negative values (in case event happens within the 50ms of the recording)
        start[start < 0] = 0

        # # add 20ms to the end of event
        # pad_end = np.round(0.02/(1/freq)).astype(int)
        # end += pad_end
        # # don't want to extend beyond the recording
        # end[end > sig.shape[0]] = sig.shape[0]

        # check length of events --> shouldn't be shorter than 95 ms or larger than 750 ms
        len_events = (end - start) * 1 / freq
        #
        # plt.hist(len_events, bins=50)
        # plt.show()
        # exit()

        to_delete_len = np.argwhere((0.75 < len_events) | (len_events < 0.05))

        start = np.delete(start, to_delete_len)
        end = np.delete(end, to_delete_len)

        peaks = []
        for s, e in zip(start,end):
            peaks.append(s+np.argmax(sig[s:e]))

        peaks = np.array(peaks)

        # check if there were any events detected --> if not: None
        if not peaks.size == 0:
            # get peak times in s
            time_bins = np.arange(data.shape[0]) * 1 / freq
            peak_times = time_bins[peaks]
            start_times = time_bins[start]
            end_times = time_bins[end]
        else:
            peak_times = None
            start_times = None
            end_times = None

        # plot some events with start, peak and end for control
        if plot_for_control:
            a = np.random.randint(0, start.shape[0], 5)
            # a = range(start.shape[0])
            for i in a:
                plt.plot(sig_z_scored, label="z-scored signal")
                plt.vlines(start[i], 0, 15, colors="r", label="start")
                plt.vlines(peaks[i], 0, 15, colors="y", label="peak")
                plt.vlines(end[i], 0, 15, colors="g", label="end")
                plt.xlim((start[i] - 5000),(end[i] + 5000))
                plt.ylabel("LFP FILTERED (140-240Hz) - Z-SCORED")
                plt.xlabel("TIME BINS / "+str(1/freq) + " s")
                plt.legend()
                plt.title("EVENT DETECTION, EVENT ID "+str(i))
                plt.show()

        return start_times, end_times, peak_times

    def firing_rates_during_swr(self, time_bin_size=0.01, plotting=True):

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        inc_cells = class_dic["increase_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        start_times, end_times, peak_times = self.detect_swr()
        event_times = np.vstack((start_times, end_times)).T

        swr_rasters = []
        # get within swr rasters
        for e_t in event_times:
            swr_rasters.append(PreProcessAwake(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).interval_temporal_binning(interval=e_t, interval_freq=1,
                                                                            time_bin_size=time_bin_size))

        swr_raster = np.hstack(swr_rasters)

        swr_mean_firing_rates = np.mean(swr_raster, axis=1)/time_bin_size

        swr_mean_firing_rates_stable = swr_mean_firing_rates[stable_cells]
        swr_mean_firing_rates_inc = swr_mean_firing_rates[inc_cells]
        swr_mean_firing_rates_dec = swr_mean_firing_rates[dec_cells]

        if plotting:
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
        else:
            return swr_mean_firing_rates_stable, swr_mean_firing_rates_dec, swr_mean_firing_rates_inc

    def firing_rates_gain_during_swr(self, time_bin_size=0.01, plotting=True, threshold_stillness=15, threshold_firing=1):

        # get entire raster to determine firing rates
        # --------------------------------------------------------------------------------------------------------------
                # get entire raster to determine mean firing
        entire_raster = self.get_raster(trials_to_use="all")
        mean_firing = np.mean(entire_raster, axis=1)/self.params.time_bin_size
        cells_above_threshold = mean_firing > threshold_firing

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        inc_cells = class_dic["increase_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        # only use cells above the firing threshold
        dec_cells = dec_cells[cells_above_threshold[dec_cells]]
        stable_cells = stable_cells[cells_above_threshold[stable_cells]]
        inc_cells = inc_cells[cells_above_threshold[inc_cells]]

        stillness_rasters = self.get_raster_stillness(threshold_stillness=threshold_stillness, time_bin_size=time_bin_size)

        stillness_mean_firing_rates = np.mean(stillness_rasters, axis=1)/time_bin_size

        stillness_mean_firing_rates_stable = stillness_mean_firing_rates[stable_cells]
        stillness_mean_firing_rates_inc = stillness_mean_firing_rates[inc_cells]
        stillness_mean_firing_rates_dec = stillness_mean_firing_rates[dec_cells]

        # now get SWR data
        # --------------------------------------------------------------------------------------------------------------

        start_times, end_times, peak_times = self.detect_swr()
        event_times = np.vstack((start_times, end_times)).T

        swr_rasters = []
        # get within swr rasters
        for e_t in event_times:
            swr_rasters.append(PreProcessAwake(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl, spatial_factor=self.spatial_factor).interval_temporal_binning(interval=e_t, interval_freq=1,
                                                                            time_bin_size=time_bin_size))

        swr_raster = np.hstack(swr_rasters)

        swr_mean_firing_rates = np.mean(swr_raster, axis=1)/time_bin_size

        swr_mean_firing_rates_stable = swr_mean_firing_rates[stable_cells]
        swr_mean_firing_rates_inc = swr_mean_firing_rates[inc_cells]
        swr_mean_firing_rates_dec = swr_mean_firing_rates[dec_cells]

        # compute swr gain

        swr_gain_stable = swr_mean_firing_rates_stable / stillness_mean_firing_rates_stable
        swr_gain_inc = swr_mean_firing_rates_inc / stillness_mean_firing_rates_inc
        swr_gain_dec = swr_mean_firing_rates_dec / stillness_mean_firing_rates_dec

        if plotting:

            p_stable = 1. * np.arange(swr_gain_stable.shape[0]) / (swr_gain_stable.shape[0] - 1)
            p_inc = 1. * np.arange(swr_gain_inc.shape[0]) / (swr_gain_inc.shape[0] - 1)
            p_dec = 1. * np.arange(swr_gain_dec.shape[0]) / (swr_gain_dec.shape[0] - 1)

            plt.plot(np.sort(swr_gain_stable), p_stable, color="violet", label="stable")
            plt.plot(np.sort(swr_gain_dec), p_dec, color="turquoise", label="dec")
            plt.plot(np.sort(swr_gain_inc), p_inc, color="orange", label="inc")
            plt.ylabel("cdf")
            plt.xlabel("SWR gain")
            plt.legend()
            plt.show()

        else:
            return swr_gain_stable, swr_gain_dec, swr_gain_inc

    def firing_rates_around_swr(self, time_bin_size=0.01, plotting=True, threshold_firing=1):

        # get gain data
        # swr_gain_stable, swr_gain_dec, swr_gain_inc = self.firing_rates_gain_during_swr(plotting=False)

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        inc_cells = class_dic["increase_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        start_times, end_times, peak_times = self.detect_swr()
        event_times = np.vstack((peak_times-5, peak_times+5)).T

        swr_rasters = []
        # get within swr rasters
        for e_t in event_times:
            swr_rasters.append(PreProcessAwake(firing_times=self.firing_times, params=self.params,
                                    whl=self.whl).interval_temporal_binning(interval=e_t, interval_freq=1,
                                                                            time_bin_size=time_bin_size))
        swr_raster = np.hstack(swr_rasters)

        # z-score using all values during stillness
        stillness_raster = self.get_raster_stillness(time_bin_size=time_bin_size)
        mean_stillness=np.mean(stillness_raster, axis=1)
        std_stillness=np.std(stillness_raster, axis=1)

        # get entire raster to determine mean firing
        entire_raster = self.get_raster(trials_to_use="all")
        mean_firing = np.mean(entire_raster, axis=1)/self.params.time_bin_size
        cells_above_threshold = mean_firing > threshold_firing
        dec_cells = dec_cells[cells_above_threshold[dec_cells]]
        stable_cells = stable_cells[cells_above_threshold[stable_cells]]
        inc_cells = inc_cells[cells_above_threshold[inc_cells]]
        # compute mean firing rate for decreasing cells, than use mean and std of mean firing cells during stillness
        # to z-score
        all_dec_data = []
        for raster in swr_rasters:
            all_dec_data.append(np.mean(raster[dec_cells, :], axis=0))
        all_dec_data = np.vstack(all_dec_data)

        mean_dec_cells_stillness = np.mean(np.mean(stillness_raster[dec_cells,:], axis=1))
        std_dec_cells_stillness = np.std(np.mean(stillness_raster[dec_cells,:], axis=1))

        all_dec_data_z = (all_dec_data-mean_dec_cells_stillness)/std_dec_cells_stillness

        plot_range = 500
        plt.subplot(2, 1, 1)
        plt.plot(np.mean(all_dec_data_z, axis=0)[int((all_dec_data_z.shape[1] / 2) - int(plot_range / 2)):(int(all_dec_data_z.shape[1] / 2) + int(plot_range / 2))])
        plt.xlim(0, plot_range)
        plt.gca().get_xaxis().set_ticks([])
        plt.ylabel("Mean firing rate (z-sco.)")
        plt.title("Decreasing cells")
        plt.subplot(2, 1, 2)
        plt.imshow(all_dec_data_z[:, int((all_dec_data_z.shape[1] / 2) - int(plot_range / 2)):(
                    int(all_dec_data_z.shape[1] / 2) + int(plot_range / 2))] / time_bin_size,
            interpolation='nearest', aspect='auto')
        plt.ylabel("SWR ID")
        plt.xticks(np.arange(0, plot_range + 1, plot_range / 5),
                   (np.arange(-int(plot_range / 2), int(plot_range / 2) + 1, plot_range / 5) * time_bin_size).astype(
                       str))
        plt.xlabel("Offset from SWR peak (s)")
        plt.show()

        # compute mean firing rate for stable cells, than use mean and std of mean firing cells during stillness
        # to z-score
        all_stable_data = []
        for raster in swr_rasters:
            all_stable_data.append(np.mean(raster[stable_cells, :], axis=0))
        all_stable_data = np.vstack(all_stable_data)

        mean_stable_cells_stillness = np.mean(np.mean(stillness_raster[stable_cells,:], axis=1))
        std_stable_cells_stillness = np.std(np.mean(stillness_raster[stable_cells,:], axis=1))

        all_stable_data_z = (all_stable_data-mean_stable_cells_stillness)/std_stable_cells_stillness

        plot_range = 500
        plt.subplot(2, 1, 1)
        plt.plot(np.mean(all_stable_data_z, axis=0)[int((all_stable_data_z.shape[1] / 2) - int(plot_range / 2)):(int(all_stable_data_z.shape[1] / 2) + int(plot_range / 2))])
        plt.xlim(0, plot_range)
        plt.gca().get_xaxis().set_ticks([])
        plt.ylabel("Mean firing rate (z-sco.)")
        plt.title("Stable cells")
        plt.subplot(2, 1, 2)
        plt.imshow(all_stable_data_z[:, int((all_stable_data_z.shape[1] / 2) - int(plot_range / 2)):(
                    int(all_stable_data_z.shape[1] / 2) + int(plot_range / 2))] / time_bin_size,
            interpolation='nearest', aspect='auto')
        plt.ylabel("SWR ID")
        plt.xticks(np.arange(0, plot_range + 1, plot_range / 5),
                   (np.arange(-int(plot_range / 2), int(plot_range / 2) + 1, plot_range / 5) * time_bin_size).astype(
                       str))
        plt.xlabel("Offset from SWR peak (s)")
        plt.show()

        # for each cell compute z-scored firing around SWR and take the mean across cells

        # stable cells
        all_data_z_scored = []
        for raster in swr_rasters:
            all_data_z_scored.append(np.divide((raster-np.tile(mean_stillness,(raster.shape[1],1)).T),
                                               np.tile(std_stillness,(raster.shape[1],1)).T))


        all_data_z_scored_stable = []
        for raster in all_data_z_scored:
            all_data_z_scored_stable.append(np.mean(raster[stable_cells, :], axis=0))
        all_data_z_scored_stable = np.vstack(all_data_z_scored_stable)

        plot_range = 200
        plt.subplot(2, 1, 1)
        plt.plot(np.mean(all_data_z_scored_stable, axis=0)[int((all_data_z_scored_stable.shape[1] / 2) - int(plot_range / 2)):(int(all_data_z_scored_stable.shape[1] / 2) + int(plot_range / 2))])
        plt.xlim(0, plot_range)
        plt.gca().get_xaxis().set_ticks([])
        plt.ylabel("Mean firing rate (z-sco.)")
        plt.title("Stable cells")
        plt.subplot(2, 1, 2)
        plt.imshow(all_data_z_scored_stable[:, int((all_data_z_scored_stable.shape[1] / 2) - int(plot_range / 2)):(
                    int(all_data_z_scored_stable.shape[1] / 2) + int(plot_range / 2))] / time_bin_size,
            interpolation='nearest', aspect='auto')
        plt.ylabel("SWR ID")
        plt.xticks(np.arange(0, plot_range + 1, plot_range / 5),
                   (np.arange(-int(plot_range / 2), int(plot_range / 2) + 1, plot_range / 5) * time_bin_size).astype(
                       str))
        plt.xlabel("Offset from SWR peak (s)")
        plt.show()

        # dec cells
        all_data_z_scored_dec = []
        for raster in all_data_z_scored:
            all_data_z_scored_dec.append(np.mean(raster[dec_cells, :], axis=0))
        all_data_z_scored_dec = np.vstack(all_data_z_scored_dec)

        plot_range = 200
        plt.subplot(2, 1, 1)
        plt.plot(np.mean(all_data_z_scored_dec, axis=0)[int((all_data_z_scored_dec.shape[1] / 2) - int(plot_range / 2)):(int(all_data_z_scored_dec.shape[1] / 2) + int(plot_range / 2))])
        plt.xlim(0, plot_range)
        plt.gca().get_xaxis().set_ticks([])
        plt.ylabel("Mean firing rate (z-sco.)")
        plt.title("Decreasing cells")
        plt.subplot(2, 1, 2)
        plt.imshow(all_data_z_scored_dec[:, int((all_data_z_scored_dec.shape[1] / 2) - int(plot_range / 2)):(
                    int(all_data_z_scored_dec.shape[1] / 2) + int(plot_range / 2))] / time_bin_size,
            interpolation='nearest', aspect='auto')
        plt.ylabel("SWR ID")
        plt.xticks(np.arange(0, plot_range + 1, plot_range / 5),
                   (np.arange(-int(plot_range / 2), int(plot_range / 2) + 1, plot_range / 5) * time_bin_size).astype(
                       str))
        plt.xlabel("Offset from SWR peak (s)")
        plt.show()


        if plotting:
            plot_range = 500
            all_cell_data = []
            for raster in swr_rasters:
                all_cell_data.append(np.mean(raster[dec_cells, :], axis=0))
            all_data = np.vstack(all_cell_data)
            plt.subplot(2, 1, 1)
            plt.plot(zscore(
                np.mean(all_data, axis=0)[
                int((all_data.shape[1] / 2) - int(plot_range / 2)):(int(all_data.shape[1] / 2) + int(plot_range / 2))]))
            plt.xlim(0, plot_range)
            plt.gca().get_xaxis().set_ticks([])
            plt.ylabel("Mean firing rate (z-sco.)")
            plt.title("Decreasing cells")
            plt.subplot(2, 1, 2)
            plt.imshow(
                all_data[:, int((all_data.shape[1] / 2) - int(plot_range / 2)):(
                            int(all_data.shape[1] / 2) + int(plot_range / 2))] / time_bin_size,
                interpolation='nearest', aspect='auto')
            plt.ylabel("SWR ID")
            plt.xticks(np.arange(0, plot_range + 1, plot_range / 5),
                       (np.arange(-int(plot_range / 2), int(plot_range / 2) + 1, plot_range / 5) * time_bin_size).astype(
                           str))
            plt.xlabel("Offset from SWR peak (s)")
            plt.show()

            all_cell_data = []
            for raster in swr_rasters:
                all_cell_data.append(np.mean(raster[stable_cells, :], axis=0))
            all_data = np.vstack(all_cell_data)
            plt.subplot(2, 1, 1)
            plt.plot(zscore(
                np.mean(all_data, axis=0)[
                int((all_data.shape[1] / 2) - int(plot_range / 2)):(int(all_data.shape[1] / 2) + int(plot_range / 2))]))
            plt.xlim(0, plot_range)
            plt.gca().get_xaxis().set_ticks([])
            plt.ylabel("Mean firing rate (z-sco.)")
            plt.title("Stable cells")
            plt.subplot(2, 1, 2)
            plt.imshow(
                all_data[:, int((all_data.shape[1] / 2) - int(plot_range / 2)):(
                            int(all_data.shape[1] / 2) + int(plot_range / 2))] / time_bin_size,
                interpolation='nearest', aspect='auto')
            plt.ylabel("SWR ID")
            plt.xticks(np.arange(0, plot_range + 1, plot_range / 5),
                       (np.arange(-int(plot_range / 2), int(plot_range / 2) + 1, plot_range / 5) * time_bin_size).astype(
                           str))
            plt.xlabel("Offset from SWR peak (s)")
            plt.show()

        else:
            return swr_raster[stable_cells, :], swr_raster[dec_cells, :], swr_raster[inc_cells, :]

    """#################################################################################################################
    #  poisson hmm
    #################################################################################################################"""

    def cross_val_poisson_hmm(self, trials_to_use=None, cl_ar=np.arange(1, 50, 5), cells_to_use="all_cells"):
        # --------------------------------------------------------------------------------------------------------------
        # cross validation of poisson hmm fits to data
        #
        # args:     - cl_ar, range object: #clusters to fit to data
        # --------------------------------------------------------------------------------------------------------------

        print(" - CROSS-VALIDATING POISSON HMM ON CHEESEBOARD --> OPTIMAL #MODES ...")
        print("  - nr modes to compute: "+str(cl_ar))

        if trials_to_use is None:
            trials_to_use = self.default_trials

        print("   --> USING TRIALS: "+str(trials_to_use[0])+"-"+str(trials_to_use[-1])+"\n")

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))

        trial_lengths = []
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            trial_lengths.append(self.trial_raster_list[trial_id].shape[1])

        # if subset of cells to use
        if cells_to_use == "stable_cells":
            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            stable_cells = class_dic["stable_cell_ids"].flatten()
            raster = raster[stable_cells, :]

        elif cells_to_use == "decreasing_cells":
                        # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            dec_cells = class_dic["decrease_cell_ids"].flatten()
            raster = raster[dec_cells, :]

        if self.params.cross_val_splits == "trial_splitting":
            trial_end = np.cumsum(np.array(trial_lengths))
            trial_start = np.concatenate([[0], trial_end[:-1]])
            # test_range = np.vstack((trial_start, trial_end))
            test_range_per_fold = []
            for lo, hi in zip(trial_start, trial_end):
                test_range_per_fold.append(np.array(list(range(lo, hi))))

        elif self.params.cross_val_splits == "custom_splits":

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
            bin_num = raster.shape[1]
            bins = np.arange(bin_num + 1)

            # length of one chunk
            n_chunks = int(bin_num / nr_chunks)
            test_range_per_fold = []
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
                test_range_per_fold.append(np.array(test_range))

        nr_cores = 12

        folder_name = self.session_name +"_"+self.experiment_phase_id+"_"+self.cell_type+\
                      "_trials_"+str(trials_to_use[0])+"_"+str(trials_to_use[-1])

        new_ml = MlMethodsOnePopulation(params=self.params)
        new_ml.parallelize_cross_val_model(nr_cluster_array=cl_ar, nr_cores=nr_cores, model_type="pHMM",
                                           raster_data=raster, folder_name=folder_name, splits=test_range_per_fold,
                                           cells_used=cells_to_use)
        # new_ml.cross_val_view_results(folder_name=folder_name)

    def plot_custom_splits(self):
        new_ml = MlMethodsOnePopulation(params=self.params)
        new_ml.plot_custom_splits()

    def find_and_fit_optimal_number_of_modes(self, cells_to_use="all_cells", cl_ar_init = np.arange(1, 50, 5)):
        # compute likelihoods with standard spacing first
        self.cross_val_poisson_hmm(cells_to_use=cells_to_use, cl_ar=cl_ar_init)
        # get optimal number of modes for coarse grained
        trials_to_use = self.default_trials
        folder_name = self.session_name + "_" + str(
            int(self.experiment_phase_id)) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1])
        new_ml = MlMethodsOnePopulation(params=self.params)
        opt_nr_coarse = new_ml.get_optimal_mode_number(folder_name=folder_name, cells_used=cells_to_use)
        print("Coarse opt. number of modes: "+str(opt_nr_coarse))
        self.cross_val_poisson_hmm(cells_to_use=cells_to_use, cl_ar=np.arange(opt_nr_coarse - 2, opt_nr_coarse + 3, 2))
        opt_nr_fine = new_ml.get_optimal_mode_number(folder_name=folder_name, cells_used=cells_to_use)
        print("Fine opt. number of modes: " + str(opt_nr_fine))
        self.fit_poisson_hmm(nr_modes=opt_nr_fine, cells_to_use=cells_to_use)

    def view_cross_val_results(self, trials_to_use=None, range_to_plot=None, save_fig=False, cells_used="all_cells"):
        # --------------------------------------------------------------------------------------------------------------
        # views cross validation results
        #
        # args:     - model_type, string: which type of model ("POISSON_HMM")
        #           - custom_splits, bool: whether custom splits were used for cross validation
        # --------------------------------------------------------------------------------------------------------------
        if trials_to_use is None:
            trials_to_use = self.default_trials

        folder_name = self.session_name + "_" + str(
            int(self.experiment_phase_id[0])) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1])
        new_ml = MlMethodsOnePopulation(params=self.params)
        new_ml.cross_val_view_results(folder_name=folder_name, range_to_plot=range_to_plot, save_fig=save_fig,
                                      cells_used=cells_used)

    def fit_poisson_hmm(self, nr_modes, trials_to_use=None, cells_to_use="all_cells"):
        # --------------------------------------------------------------------------------------------------------------
        # fits poisson hmm to data
        #
        # args:     - nr_modes, int: #clusters to fit to data
        #           - file_identifier, string: string that is added at the end of file for identification
        # --------------------------------------------------------------------------------------------------------------

        print(" - FITTING POISSON HMM WITH "+str(nr_modes)+" MODES ...\n")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))

        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))

        # if subset of cells to use
        if cells_to_use == "stable_cells":
            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            stable_cells = class_dic["stable_cell_ids"].flatten()
            raster = raster[stable_cells, :]

        elif cells_to_use == "decreasing_cells":
                        # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            dec_cells = class_dic["decrease_cell_ids"].flatten()
            raster = raster[dec_cells, :]

        log_li = -1*np.inf
        # fit 10 times to select model with best highest log-likelihood (NOT CROSS-VALIDATED!!!)
        for i in range(10):
            test_model = PoissonHMM(n_components=nr_modes)
            test_model.fit(raster.T)
            log_li_test = test_model.score(raster.T)
            if log_li_test > log_li:
                model = test_model
                log_li = log_li_test

        model.set_time_bin_size(time_bin_size=self.params.time_bin_size)

        if cells_to_use == "stable_cells":
            save_dir = self.params.pre_proc_dir+"phmm/stable_cells/"
        elif cells_to_use == "decreasing_cells":
            save_dir = self.params.pre_proc_dir+"phmm/decreasing_cells/"
        elif cells_to_use == "all_cells":
            save_dir = self.params.pre_proc_dir+"phmm/"
        file_name = self.session_name + "_" + str(
            int(self.experiment_phase_id)) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1])+"_"+str(nr_modes)+"_modes"

        with open(save_dir+file_name+".pkl", "wb") as file: pickle.dump(model, file)

        print("  - ... DONE!\n")

    def evaluate_poisson_hmm(self, nr_modes, trials_to_use=None, save_fig=False):
        # --------------------------------------------------------------------------------------------------------------
        # fits poisson hmm to data and evaluates the goodness of the model by comparing basic statistics (avg. firing
        # rate, correlation values, k-statistics) between real data and data sampled from the model
        #
        # args:     - nr_modes, int: #clusters to fit to data
        #           - load_from_file, bool: whether to load model from file or to fit model again
        # --------------------------------------------------------------------------------------------------------------

        print(" - EVALUATING POISSON HMM FIT (BASIC STATISTICS) ...")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
        nr_time_bins = raster.shape[1]
        # X = X[:, :1000]

        file_name = self.session_name + "_" + str(
            int(self.experiment_phase_id[0])) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1])+"_"+str(nr_modes)+"_modes"

        # check if model file exists already --> otherwise fit model again
        if os.path.isfile(self.params.pre_proc_dir+"phmm/" + file_name + ".pkl"):
            print("- LOADING PHMM MODEL FROM FILE\n")
            with open(self.params.pre_proc_dir+"phmm/" + file_name + ".pkl", "rb") as file:
                model = pickle.load(file)
        else:
            print("- PHMM MODEL FILE NOT FOUND --> FITTING PHMM TO DATA\n")
            model = PoissonHMM(n_components=nr_modes)
            model.fit(raster.T)

        samples, sequence = model.sample(nr_time_bins*50)
        samples = samples.T

        if save_fig:
            mean_dic, corr_dic, k_dic = evaluate_clustering_fit(real_data=raster, samples=samples, binning="TEMPORAL_SPIKE",
                                   time_bin_size=0.1, plotting=False)

            plt.style.use('default')
            k_samples_sorted = np.sort(k_dic["samples"])
            k_data_sorted = np.sort(k_dic["real"])

            p_samples = 1. * np.arange(k_samples_sorted.shape[0]) / (k_samples_sorted.shape[0] - 1)
            p_data = 1. * np.arange(k_data_sorted.shape[0]) / (k_data_sorted.shape[0] - 1)
            # plt.hlines(0.5, -0.02, 0.85, color="gray", linewidth=0.5)
            plt.plot(k_data_sorted, p_data, color="darkorange", label="Data")
            plt.plot(k_samples_sorted, p_samples, color="bisque", label="Model")
            plt.ylabel("CDF")
            plt.xlabel("% active cells per time bin")
            plt.legend()
            plt.title("Model quality: k-statistic")
            make_square_axes(plt.gca())

            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("model_qual_k.svg", transparent="True")
            plt.close()

            # plot single figures
            plt.plot([-0.2, max(np.maximum(corr_dic["samples"].flatten(), corr_dic["real"].flatten()))],
                     [-0.2,max(np.maximum(corr_dic["samples"].flatten(),corr_dic["real"].flatten()))],
                     linestyle="dashed", c="gray")
            plt.scatter(corr_dic["samples"].flatten(), corr_dic["real"].flatten(), color="pink")
            plt.xlabel("Correlations (samples)")
            plt.ylabel("Correlations (data)")
            plt.title("Model quality: correlations ")
            plt.text(0.0, 0.9, "R = " + str(round(corr_dic["corr"][0], 4)))
            make_square_axes(plt.gca())
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("model_qual_correlations.svg", transparent="True")
            plt.close()

            # plot single figures
            plt.plot([0, max(np.maximum(mean_dic["samples"].flatten(), mean_dic["real"].flatten()))],
                     [0,max(np.maximum(mean_dic["samples"].flatten(),mean_dic["real"].flatten()))],
                     linestyle="dashed", c="gray")
            plt.scatter(mean_dic["samples"].flatten(), mean_dic["real"].flatten(), color="turquoise")
            plt.xlabel("Mean firing rate (samples)")
            plt.ylabel("Mean firing rate (data)")
            plt.title("Model quality: mean firing ")
            plt.text(1.2, 12, "R = " + str(round(mean_dic["corr"][0], 4)))
            make_square_axes(plt.gca())
            # plt.show()
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("model_qual_mean_firing.svg", transparent="True")
            plt.close()
        else:

            evaluate_clustering_fit(real_data=raster, samples=samples, binning="TEMPORAL_SPIKE",
                                    time_bin_size=0.1, plotting=True)

    def evaluate_multiple_poisson_hmm_models(self, nr_modes_range, trials_to_use=None):
        # --------------------------------------------------------------------------------------------------------------
        # fits poisson hmm to data and evaluates the goodness of the model by comparing basic statistics (avg. firing
        # rate, correlation values, k-statistics) between real data and data sampled from the model
        #
        # args:     - nr_modes, int: #clusters to fit to data
        #           - load_from_file, bool: whether to load model from file or to fit model again
        # --------------------------------------------------------------------------------------------------------------

        print("- EVALUATING POISSON HMM FIT (BASIC STATISTICS) ...")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
        nr_time_bins = raster.shape[1]
        # X = X[:, :1000]

        mean_res = []
        corr_res = []
        k_res = []

        for nr_modes in nr_modes_range:
            fit_mean_res = []
            fit_corr_res = []
            fit_k_res = []
            for nr_fit in range(10):
                print(" - FITTING PHMM MODEL WITH "+str(nr_modes)+" MODES - FIT NR: "+str(nr_fit))
                model = PoissonHMM(n_components=nr_modes)
                model.fit(raster.T)

                samples, sequence = model.sample(nr_time_bins*50)
                samples = samples.T
                mean_dic, corr_dic, k_dic = evaluate_clustering_fit(real_data=raster, samples=samples,
                                                                    binning="TEMPORAL_SPIKE",
                                                                    time_bin_size=0.1, plotting=False)
                fit_mean_res.append(mean_dic["corr"][0])
                fit_corr_res.append(corr_dic["corr_triangles"][0])
                fit_k_res.append(k_dic["diff_med"])


            mean_res.append([np.mean(np.array(fit_mean_res)), np.std(np.array(fit_mean_res))])
            corr_res.append([np.mean(np.array(fit_corr_res)), np.std(np.array(fit_corr_res))])
            k_res.append([np.mean(np.array(fit_k_res)), np.std(np.array(fit_k_res))])

        mean_res = np.array(mean_res)
        corr_res = np.array(corr_res)
        k_res = np.array(k_res)

        # plot results
        plt.figure(figsize=(5,10))
        plt.subplot(3,1,1)
        plt.errorbar(nr_modes_range, mean_res[:,0],yerr=mean_res[:,1], color="r")
        plt.grid(color="gray")
        plt.ylabel("PEARSON R: MEAN FIRING")
        plt.ylim(min(mean_res[:,0])-max(mean_res[:,1]), max(mean_res[:,0])+max(mean_res[:,1]))
        plt.subplot(3,1,2)
        plt.errorbar(nr_modes_range, corr_res[:,0],yerr=corr_res[:,1], color="r")
        plt.ylabel("PEARSON R: CORR")
        plt.grid(color="gray")
        plt.ylim(min(corr_res[:,0])-max(corr_res[:,1]), max(corr_res[:,0])+max(corr_res[:,1]))
        plt.subplot(3,1,3)
        plt.errorbar(nr_modes_range, k_res[:,0],yerr=k_res[:,1], color="r")
        plt.ylabel("DIFF. MEDIANS")
        plt.xlabel("NR. MODES")
        plt.grid(color="gray")
        plt.ylim(min(k_res[:,0])-max(k_res[:,1]), max(k_res[:,0])+max(k_res[:,1]))
        plt.show()

    def decode_poisson_hmm(self, trials_to_use=None, file_name=None, cells_to_use="all"):
        # --------------------------------------------------------------------------------------------------------------
        # loads model from file and decodes data
        #
        # args:     - nr_modes, int: #clusters to fit to data --> used to identify file that fits the data
        #           - file_name, string:    is used if model from a different experiment phase is supposed to be used
        #                                   (e.g. model from awake is supposed to be fit to sleep data)
        # --------------------------------------------------------------------------------------------------------------

        if trials_to_use is None:
            trials_to_use = self.default_trials
        if trials_to_use is "all":
            trials_to_use = range(self.nr_trials)


        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
        nr_time_bins = raster.shape[1]

        if file_name is None:
            file_name = self.default_phmm
        if cells_to_use == "all":
            with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "rb") as file:
                model = pickle.load(file)
        elif cells_to_use == "stable":
            with open(self.params.pre_proc_dir+"phmm/stable_cells/"+file_name+".pkl", "rb") as file:
                model = pickle.load(file)
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                      "rb") as f:
                class_dic = pickle.load(f)
            raster = raster[class_dic["stable_cell_ids"], :]

        elif cells_to_use == "decreasing":
            with open(self.params.pre_proc_dir+"phmm/decreasing_cells/"+file_name+".pkl", "rb") as file:
                model = pickle.load(file)
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                      "rb") as f:
                class_dic = pickle.load(f)
            raster = raster[class_dic["decrease_cell_ids"], :]

        nr_modes_ = model.means_.shape[0]

        # compute most likely sequence
        sequence = model.predict(raster.T)
        post_prob = model.predict_proba(raster.T)
        return sequence, nr_modes_, post_prob

    def load_poisson_hmm(self, trials_to_use=None, nr_modes=None, file_name=None, cells_to_use="all_cells"):
        # --------------------------------------------------------------------------------------------------------------
        # loads model from file and returns model
        #
        # args:     - nr_modes, int: #clusters to fit to data --> used to identify file that fits the data
        #           - file_name, string:    is used if model from a different experiment phase is supposed to be used
        #                                   (e.g. model from awake is supposed to be fit to sleep data)
        # --------------------------------------------------------------------------------------------------------------

        if trials_to_use is None:
            trials_to_use = self.default_trials

        if (nr_modes is None) & (file_name is None):
            raise Exception("PROVIDE NR. MODES OR FILE NAME")

        if file_name is None:
            file_name = self.params.session_name + "_" + str(
                int(self.params.experiment_phase_id[0])) + "_" + self.cell_type + "_trials_" + str(trials_to_use[0]) + \
                        "_" + str(trials_to_use[-1]) + "_" + str(nr_modes) + "_modes"
        else:
            file_name =file_name

        if cells_to_use == "stable_cells":
            save_dir = self.params.pre_proc_dir+"phmm/stable_cells"
        elif cells_to_use == "decreasing_cells":
            save_dir = self.params.pre_proc_dir+"phmm/decreasing_cells"
        elif cells_to_use == "all_cells":
            save_dir = self.params.pre_proc_dir+"phmm/"

        with open(save_dir+file_name+".pkl", "rb") as file:
            model = pickle.load(file)

        return model

    def fit_spatial_gaussians_for_modes(self, nr_modes=None, file_name=None, trials_to_use=None,
                                        min_nr_bins_active=5, plot_awake_fit=False, plot_modes=False):

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))

        state_sequence, nr_modes, _ = self.decode_poisson_hmm(file_name=file_name,
                                                           trials_to_use=trials_to_use)

        mode_id, freq = np.unique(state_sequence, return_counts=True)
        modes_to_plot = mode_id[freq > min_nr_bins_active]

        cmap = generate_colormap(nr_modes)
        if plot_modes:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for mode in np.arange(nr_modes):
                mode_data = loc[state_sequence == mode, :]
                # rgb = (random.random(), random.random(), random.random())
                ax.scatter(mode_data[:, 0], mode_data[:, 1],
                           alpha=1, marker=".", s=1, label="DIFFERENT MODES", c=np.array([cmap(mode)]))

            ax.set_ylim(self.y_min, self.y_max)
            ax.set_xlim(self.x_min, self.x_max)
            for g_l in self.goal_locations:
                ax.scatter(g_l[0], g_l[1], color="w", label="GOALS")

            plt.gca().set_aspect('equal', adjustable='box')
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("MODE ASSIGNMENT AWAKE DATA")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            for mode in np.arange(nr_modes):
                mode_data = loc[state_sequence == mode, :]
                fig, ax = plt.subplots()
                ax.scatter(loc[:,0], loc[:,1], color="grey", s=1, label="TRACKING")
                ax.scatter(mode_data[:, 0], mode_data[:, 1],
                       alpha=1, marker=".", s=1, label="MODE " + str(mode) + " ASSIGNED",
                           color="red")
                for g_l in self.goal_locations:
                    ax.scatter(g_l[0], g_l[1], color="w", label="GOALS")
                ax.set_ylim(self.y_min, self.y_max)
                ax.set_xlim(self.x_min, self.x_max)
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.gca().set_aspect('equal', adjustable='box')
                handles, labels = ax.get_legend_handles_labels()
                by_label = OrderedDict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())
                plt.show()

        means = np.zeros((2, nr_modes))
        cov = np.zeros((2, nr_modes))
        for mode in np.arange(nr_modes):
            mode_data = loc[state_sequence == mode, :]

            if len(mode_data) == 0:
                means[:, mode] = np.nan
                cov[:, mode] = np.nan
            else:
                means[:, mode] = np.mean(mode_data, axis=0)
                cov[:, mode] = np.var(mode_data, axis=0)

        loc_data = loc[:, :]

        center = np.min(loc_data, axis=0) + (np.max(loc_data, axis=0) - np.min(loc_data, axis=0)) * 0.5
        dist = loc_data - center

        rad = max(np.sqrt(np.square(dist[:, 0]) + np.square(dist[:, 1]))) + 1

        std_modes = np.sqrt(cov[0,:]+cov[1,:])
        std_modes[std_modes == 0] = np.nan

        if plot_awake_fit:

            for mode_to_plot in modes_to_plot:

                mean = means[:, mode_to_plot]
                cov_ = cov[:, mode_to_plot]
                std_ = std_modes[mode_to_plot]

                # Parameters to set
                mu_x = mean[0]
                variance_x = cov_[0]

                mu_y = mean[1]
                variance_y = cov_[1]

                # Create grid and multivariate normal
                x = np.linspace(center[0] - rad, center[0]+rad, int(2.2*rad))
                y = np.linspace(0, 250, 250)
                X, Y = np.meshgrid(x, y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
                rv_normalized = rv.pdf(pos) / np.sum(rv.pdf(pos).flatten())

                fig, ax = plt.subplots()
                gauss = ax.imshow(rv_normalized)
                env = Circle((center[0], center[1]), rad, color="white", fill=False)
                ax.add_artist(env)
                ax.set_ylim(center[1] - 1.1*rad, center[1]+1.1*rad)
                ax.scatter(loc_data[state_sequence == mode_to_plot, 0], loc_data[state_sequence == mode_to_plot, 1],
                           alpha=1, c="white", marker=".", s=0.3, label="MODE "+ str(mode_to_plot) +" ASSIGNED")
                cb = plt.colorbar(gauss)
                cb.set_label("PROBABILITY")
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title("STD: "+str(np.round(std_, 2)))
                plt.legend()
                plt.show()

        # compute frequency of each mode
        mode_freq = np.zeros(nr_modes)
        mode_freq[mode_id] = freq
        mode_freq = mode_freq.astype(int)

        env = Circle((center[0], center[1]), rad, color="white", fill=False)

        result_dic = {
            "std_modes": std_modes
        }

        with open("temp_data/test1", "wb") as f:
            pickle.dump(result_dic, f)


        return means, std_modes, mode_freq, env, state_sequence

    def plot_all_phmm_modes_spatial(self, nr_modes):
        for i in range(nr_modes):
            self.plot_phmm_mode_spatial(mode_id = i)

    def plot_phmm_mode_spatial(self, mode_id, ax=None, save_fig=False, use_viterbi=True, cells_to_use="all"):
        plt.style.use('default')
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        trials_to_use = self.default_trials

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))

        # plot all tracking data
        ax.scatter(loc[:,0], loc[:,1], color="lightgray", s=1, label="Tracking")

        if use_viterbi:
            if cells_to_use == "all":
                file_name = self.default_phmm
            elif cells_to_use == "stable":
                file_name = self.default_phmm_stable
            state_sequence, nr_modes, _ = self.decode_poisson_hmm(file_name=file_name,
                                                               trials_to_use=trials_to_use, cells_to_use=cells_to_use)
        else:
            cells_to_use = "stable"
            # load phmm model
            with open(self.params.pre_proc_dir + "phmm/" + self.default_phmm + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)
            # get means of model (lambdas) for decoding
            mode_means = model_dic.means_
            cell_selection = "custom"
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            if cells_to_use == "stable":
                cell_ids = class_dic["stable_cell_ids"]
            results_list = decode_using_phmm_modes(mode_means=mode_means, event_spike_rasters=[raster],
                                                   compression_factor=1, cells_to_use=cell_ids,
                                                   cell_selection=cell_selection)
            likelihoods = results_list[0]
            state_sequence = np.argmax(likelihoods, axis=1)

        for g_l in self.goal_locations:
            ax.scatter(g_l[0], g_l[1], color="black", label="Goal locations")

        mode_data = loc[state_sequence == mode_id, :]
        ax.scatter(mode_data[:, 0], mode_data[:, 1],
                   alpha=1, marker=".", s=1, color="red")
        # ax.set_ylim(30, 230)
        # ax.set_xlim(70, 300)
        #ax.set_xlim(self.x_min, self.x_max)
        plt.xlabel("X (cm)")
        plt.ylabel("Y (cm)")
        plt.gca().set_aspect('equal', adjustable='box')
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("State " + str(mode_id))
        if save_fig:
            plt.savefig("state_"+str(mode_id)+".svg", transparent="True")
            plt.rcParams['svg.fonttype'] = 'none'
        else:
            plt.show()

    def plot_phmm_state_neural_patterns(self, nr_modes):

        trials_to_use = self.default_trials

        file_name = self.params.session_name + "_" + str(
            int(self.params.experiment_phase_id[0])) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1])+"_"+str(nr_modes)+"_modes"

        with open(self.params.pre_proc_dir+"phmm/" + file_name + ".pkl", "rb") as file:
            model = pickle.load(file)

        # get means
        means = model.means_

        x_max_ = np.max(means.flatten())
        x_min_ = np.min(means.flatten())

        n_col = 10
        scaler = 0.1
        plt.style.use('default')
        fig = plt.figure(figsize=(8,6))
        gs = fig.add_gridspec(6, n_col)

        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(np.expand_dims(means[0, :], 1))
        ax1.set_xticks([])
        ax1.set_xlabel(str(0))
        ax1.set_aspect(scaler)
        ax1.set_ylabel("CELL ID")

        for i in range(1, n_col-1):
            ax1 = fig.add_subplot(gs[:, i])
            a = ax1.imshow(np.expand_dims(means[i,:], 1))
            ax1.set_xticks([])
            ax1.set_yticks([], [])
            ax1.set_xlabel(str(i))
            ax1.set_aspect(scaler)

        plt.tight_layout()
        ax1 = fig.add_subplot(gs[:, n_col-1])
        fig.colorbar(a, cax=ax1)
        plt.rcParams['svg.fonttype'] = 'none'
        #plt.show()
        plt.savefig("state_neural_pattern.svg", transparent="True")

        print("HERE")

    def analyze_modes_spatial_information(self, file_name, mode_ids, plotting=True):

        trials_to_use = self.default_trials

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))

        state_sequence, nr_modes_, _ = self.decode_poisson_hmm(file_name=file_name,
                                                           trials_to_use=trials_to_use)

        _, loc_data, _ = self.get_raster_location_speed()

        # only use data from mode ids provided
        # all_modes_data = loc_data[np.isin(state_sequence, mode_ids), :]
        # pd_all_modes = upper_tri_without_diag(pairwise_distances(all_modes_data))
        # all_modes_med_dist = np.median(pd_all_modes)

        med_dist_list = []

        for mode in mode_ids:
            mode_loc = loc_data[state_sequence == mode, :]

            # compute pairwise distances (euclidean)
            pd = upper_tri_without_diag(pairwise_distances(mode_loc))

            med_dist = np.median(pd)
            # std_dist = np.std(np.array(dist_list))
            med_dist_list.append(med_dist)

            if plotting:

                fig, ax = plt.subplots()
                ax.scatter(loc_data[:,0], loc_data[:,1], color="gray", s=1, label="TRACKING")
                ax.scatter(mode_loc[:, 0], mode_loc[:, 1], color="red", label="MODE "+str(mode)+" ASSIGNED", s=1)
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title("MODE "+str(mode)+"\nMEDIAN DISTANCE: "+ str(np.round(med_dist,2)))
                plt.legend()
                plt.show()

        return np.mean(np.array(med_dist_list))

    def phmm_mode_spatial_information_from_model(self, spatial_resolution=1, file_name=None,
                                            plot_for_control=False):
        """
        loads poisson hmm model and weighs rate maps by lambda vectors --> then computes spatial information (sparsity,
        skaggs information)

        @param spatial_resolution: spatial resolution in cm
        @type spatial_resolution: int
        @param nr_modes: nr of modes for model file identification
        @type nr_modes: int
        @param file_name: file containing the model --> is used when nr_modes is not provided to identify file
        @type file_name: string
        @param plot_for_control: whether to plot intermediate results
        @type plot_for_control: bool
        @return: sparsity, skaggs info for each mode
        """

        print(" - SPATIAL INFORMATION OF PHMM MODES USING MODEL\n")

        if file_name is None:
            file_name = self.default_phmm

        with open(self.params.pre_proc_dir+"phmm/"+file_name+".pkl", "rb") as file:
            model = pickle.load(file)

        # get means for all modes (lambda vectors)
        means = model.means_

        ################################################################################################################
        # get spatial information of mode by weighing rate maps
        ################################################################################################################

        # compute rate maps and occupancy
        rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution, trials_to_use=None)
        occ = self.get_occ_map(spatial_resolution=spatial_resolution, trials_to_use=None)
        prob_occ = occ / occ.sum()
        # compute binary occupancy map --> used as a mask to filter non-visited bins
        occ_mask = np.where(occ > 0, 1, np.nan)
        prob_occ_orig = np.multiply(prob_occ, occ_mask)

        sparsity_list = []
        skaggs_per_second_list = []
        skaggs_per_spike_list = []

        # go through all modes
        for mode_id, means_mode in enumerate(means):
            # weigh rate map of each cell using mean firing from lambda vector --> compute mean across all cells
            rate_map_mode_orig = np.mean(rate_maps * means_mode, axis=2)
            # generate filtered rate map by masking non visited places
            rate_map_mode_orig = np.multiply(rate_map_mode_orig, occ_mask)

            rate_map_mode = rate_map_mode_orig[~np.isnan(rate_map_mode_orig)]
            prob_occ = prob_occ_orig[~np.isnan(prob_occ_orig)]

            # need to filter bins with zero firing rate --> otherwise log causes an error
            rate_map_mode = rate_map_mode[rate_map_mode > 0]
            prob_occ = prob_occ[rate_map_mode > 0]

            # compute sparsity
            sparse_mode = (np.sum(prob_occ * rate_map_mode) ** 2) / np.sum(prob_occ * (rate_map_mode ** 2))

            # find good bins so that there is no problem with the log
            good_bins = (rate_map_mode / rate_map_mode.mean() > 0.0000001)
            mean_rate = np.sum(rate_map_mode[good_bins] * prob_occ[good_bins])
            skaggs_info_per_sec = np.sum(rate_map_mode[good_bins] * prob_occ[good_bins] *
                                         np.log(rate_map_mode[good_bins] / mean_rate))
            skaggs_info_per_spike = np.sum(rate_map_mode[good_bins] / mean_rate * prob_occ[good_bins] *
                                           np.log(rate_map_mode[good_bins] / mean_rate))

            skaggs_per_second_list.append(skaggs_info_per_sec)
            skaggs_per_spike_list.append(skaggs_info_per_spike)
            sparsity_list.append(sparse_mode)
            if plot_for_control:
                # plot random examples
                rand_float = np.random.randn(1)
                if rand_float > 0.5:
                    plt.imshow(rate_map_mode_orig)
                    plt.colorbar()
                    plt.title("Sparsity: "+str(sparse_mode)+"\n Skaggs per second: "+str(skaggs_info_per_sec)+
                              "\n Skaggs per spike: "+ str(skaggs_info_per_spike))
                    plt.show()

        if plot_for_control:
            plt.hist(skaggs_per_second_list)
            plt.title("SKAGGS INFO (PER SECOND)")
            plt.xlabel("SKAGGS INFO.")
            plt.ylabel("COUNTS")
            plt.show()

            plt.hist(skaggs_per_spike_list)
            plt.title("SKAGGS INFO (PER SPIKE)")
            plt.xlabel("SKAGGS INFO.")
            plt.ylabel("COUNTS")
            plt.show()

            plt.hist(sparsity_list)
            plt.title("SPARSITY")
            plt.xlabel("SPARSITY")
            plt.ylabel("COUNTS")
            plt.show()

            plt.scatter(skaggs_per_second_list, sparsity_list)
            plt.title("SKAGGS (PER SEC) vs. SPARSITY\n"+str(pearsonr(skaggs_per_second_list, sparsity_list)))
            plt.xlabel("SKAGGS (PER SEC)")
            plt.ylabel("SPARSITY")
            plt.show()

            plt.scatter(skaggs_per_spike_list, sparsity_list)
            plt.title("SKAGGS (PER SPIKE) vs. SPARSITY\n"+str(pearsonr(skaggs_per_spike_list, sparsity_list)))
            plt.xlabel("SKAGGS (PER SPIKE)")
            plt.ylabel("SPARSITY")
            plt.show()

            plt.scatter(skaggs_per_second_list, skaggs_per_spike_list)
            plt.title("SKAGGS (PER SEC) vs. SKAGGS (PER SPIKE)\n"+str(pearsonr(skaggs_per_second_list, skaggs_per_spike_list)))
            plt.xlabel("SKAGGS (PER SEC)")
            plt.ylabel("SKAGGS (PER SPIKE)")
            plt.show()

        return np.array(sparsity_list), np.array(skaggs_per_second_list), np.array(skaggs_per_spike_list)

    def nr_spikes_per_mode(self, trials_to_use=None, nr_modes=None):

        if trials_to_use is None:
            trials_to_use = self.default_trials

        model = self.load_poisson_hmm(trials_to_use, nr_modes)
        means = model.means_

        spikes_per_mode = np.sum(means, axis=1)
        y,_,_ = plt.hist(spikes_per_mode, bins=50)
        plt.vlines(np.mean(spikes_per_mode), 0, y.max(), colors="r",
                   label="MEAN: "+str(np.round(np.mean(spikes_per_mode),2)))
        plt.vlines(np.median(spikes_per_mode), 0, y.max(), colors="blue",
                   label="MEDIAN: "+str(np.median(spikes_per_mode)))
        plt.legend()
        plt.title("#SPIKES PER MODE")
        plt.xlabel("AVG. #SPIKES PER MODE")
        plt.ylabel("COUNT")
        plt.show()

    def decode_awake_activity_spike_binning(self, model_name=None, trials_to_use=None, plot_for_control=False,
                                   return_results=True, cells_to_use="all"):
        """
        decodes sleep activity using pHMM modes/spatial bins from ising model from before and after awake activity and
        computes similarity measure

        @param model_name: name of file containing the pHMM file
        @type model_name: str
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        print(" - AWAKE DECODING USING PHMM MODES ...\n")
        result_dir = "phmm_decoding"
        # get template file name from parameter file of session if not provided
        if model_name is None:
            model_name = self.session_params.default_pre_phmm_model

        if model_name is None:
            raise Exception("MODEL FILE NOT FOUND\n NOR IN SESSION PARAMETER FILE DEFINED")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        print("   --> USING TRIALS: "+str(trials_to_use[0])+"-"+str(trials_to_use[-1])+"\n")

        # check that template files are from correct session
        # --------------------------------------------------------------------------------------------------------------

        file_name = self.session_name +"_"+self.experiment_phase_id +\
                        "_"+ self.cell_type+"_AWAKE_DEC_"+cells_to_use+".npy"

        # check if PRE and POST result exists already
        # --------------------------------------------------------------------------------------------------------------
        if os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + file_name):
            print(" - RESULTS EXIST ALREADY -- USING EXISTING RESULTS\n")
        else:
            # if results don't exist --> compute results
            # go trough all trials and compute constant #spike bins
            spike_rasters = []

            for trial_id in trials_to_use:
                key = "trial"+str(trial_id)
                # compute spike rasters --> time stamps here are at .whl resolution (20kHz/512 --> 0.0256s)
                spike_raster = PreProcessAwake(
                    firing_times=self.data_dic["trial_data"][key]["spike_times"][self.cell_type],
                    params=self.params, whl=self.data_dic["trial_data"][key]["whl"],
                    ).spike_binning()

                if plot_for_control:
                    if random.uniform(0, 1) > 0.5:
                        plt.imshow(spike_raster, interpolation='nearest', aspect='auto')
                        plt.title("TRIAL"+str(trial_id)+": CONST. #SPIKES BINNING, 12 SPIKES PER BIN")
                        plt.xlabel("BIN ID")
                        a = plt.colorbar()
                        a.set_label("#SPIKES")
                        plt.ylabel("CELL ID")
                        plt.show()

                        plt.imshow(self.trial_raster_list[trial_id], interpolation='nearest', aspect='auto')
                        plt.title("TIME BINNING, 100ms TIME BINS")
                        plt.xlabel("TIME BIN ID")
                        plt.ylabel("CELL ID")
                        a = plt.colorbar()
                        a.set_label("#SPIKES")
                        plt.show()

                spike_rasters.append(spike_raster)

            # load phmm model
            with open(self.params.pre_proc_dir + "phmm/" + model_name + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)
            # get means of model (lambdas) for decoding
            mode_means = model_dic.means_

            time_bin_size_encoding = model_dic.time_bin_size

            # check if const. #spike bins are correct for the loaded compression factor
            if not self.params.spikes_per_bin == 12:
                raise Exception("TRYING TO LOAD COMPRESSION FACTOR FOR 12 SPIKES PER BIN, "
                                "BUT CURRENT #SPIKES PER BIN != 12")

            # load correct compression factor (as defined in parameter file of the session)
            if time_bin_size_encoding == 0.01:
                compression_factor = \
                    np.round(self.session_params.sleep_compression_factor_12spikes_100ms * 10, 3)
            elif time_bin_size_encoding == 0.1:
                compression_factor = self.session_params.sleep_compression_factor_12spikes_100ms
            else:
                raise Exception("COMPRESSION FACTOR NEITHER PROVIDED NOR FOUND IN PARAMETER FILE")

            if cells_to_use == "all":
                cell_selection = "all"
                cell_ids = np.empty(0)

            else:

                cell_selection = "custom"
                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name +"_"+ self.params.stable_cell_method +".pickle","rb") as f:
                    class_dic = pickle.load(f)

                if cells_to_use == "stable":
                    cell_ids = class_dic["stable_cell_ids"]
                elif cells_to_use == "increasing":
                    cell_ids = class_dic["increase_cell_ids"]
                elif cells_to_use == "decreasing":
                    cell_ids = class_dic["decrease_cell_ids"]

            print(" - DECODING USING " + cells_to_use + " CELLS")

            # decode activity
            results_list = decode_using_phmm_modes(mode_means=mode_means, event_spike_rasters=spike_rasters,
                                                   compression_factor=compression_factor, cells_to_use=cell_ids,
                                                   cell_selection=cell_selection)

            # plot maps of some SWR for control
            if plot_for_control:
                for trial_id, res in zip(trials_to_use, results_list):
                    plt.imshow(np.log(res.T), interpolation='nearest', aspect='auto')
                    plt.xlabel("POP.VEC. ID")
                    plt.ylabel("MODE ID")
                    a = plt.colorbar()
                    a.set_label("LOG-PROBABILITY")
                    plt.title("LOG-PROBABILITY MAP: TRIAL "+str(trial_id))
                    plt.show()

            # saving results
            # --------------------------------------------------------------------------------------------------
            # create dictionary with results
            result_post = {
                "results_list": results_list,
            }
            outfile = open(self.params.pre_proc_dir + result_dir +"/" + file_name, 'wb')
            pickle.dump(result_post, outfile)
            print("  - SAVED NEW RESULTS!\n")

        if return_results:

            # load decoded maps
            result_pre = pickle.load(open(self.params.pre_proc_dir+result_dir + "/" + file_name, "rb"))

            pre_prob = result_pre["results_list"]

            return pre_prob

    def decode_awake_activity_time_binning(self, model_name=None, trials_to_use=None, plot_for_control=False,
                                   return_results=True, cells_to_use="all"):
        """
        decodes sleep activity using pHMM modes/spatial bins from ising model from before and after awake activity and
        computes similarity measure

        @param model_name: name of file containing the pHMM file
        @type model_name: str
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        print(" - AWAKE DECODING USING PHMM MODES ...\n")
        result_dir = "phmm_decoding"
        # get template file name from parameter file of session if not provided
        if model_name is None:
            model_name = self.session_params.default_phmm_model

        if model_name is None:
            raise Exception("MODEL FILE NOT FOUND\n NOR IN SESSION PARAMETER FILE DEFINED")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        print("   --> USING TRIALS: "+str(trials_to_use[0])+"-"+str(trials_to_use[-1])+"\n")

        # check that template files are from correct session
        # --------------------------------------------------------------------------------------------------------------

        file_name = self.session_name +"_"+self.experiment_phase_id +\
                        "_"+ self.cell_type+"_AWAKE_DEC_"+cells_to_use+"_time_binning"+".npy"

        # check if PRE and POST result exists already
        # --------------------------------------------------------------------------------------------------------------
        if os.path.isfile(self.params.pre_proc_dir + result_dir + "/" + file_name):
            print(" - RESULTS EXIST ALREADY -- USING EXISTING RESULTS\n")
        else:
            # if results don't exist --> compute results
            # go trough all trials and compute constant #spike bins
            spike_rasters = []

            for trial_id in trials_to_use:
                key = "trial"+str(trial_id)
                # compute spike rasters --> time stamps here are at .whl resolution (20kHz/512 --> 0.0256s)
                spike_raster= PreProcessAwake(
                    firing_times=self.data_dic["trial_data"][key]["spike_times"][self.cell_type],
                    params=self.params, whl=self.data_dic["trial_data"][key]["whl"],
                    ).interval_temporal_binning(interval_start=self.data_dic["trial_timestamps"][0, trial_id],
                                                interval_end=self.data_dic["trial_timestamps"][1, trial_id],
                                                interval_freq=0.0256)

                if plot_for_control:
                    if random.uniform(0, 1) > 0.5:
                        plt.imshow(spike_raster, interpolation='nearest', aspect='auto')
                        plt.title("TRIAL"+str(trial_id)+": CONST. #SPIKES BINNING, 12 SPIKES PER BIN")
                        plt.xlabel("BIN ID")
                        a = plt.colorbar()
                        a.set_label("#SPIKES")
                        plt.ylabel("CELL ID")
                        plt.show()

                        plt.imshow(self.trial_raster_list[trial_id], interpolation='nearest', aspect='auto')
                        plt.title("TIME BINNING, 100ms TIME BINS")
                        plt.xlabel("TIME BIN ID")
                        plt.ylabel("CELL ID")
                        a = plt.colorbar()
                        a.set_label("#SPIKES")
                        plt.show()

                spike_rasters.append(spike_raster)

            # load phmm model
            with open(self.params.pre_proc_dir + "phmm/" + model_name + '.pkl', 'rb') as f:
                model_dic = pickle.load(f)
            # get means of model (lambdas) for decoding
            mode_means = model_dic.means_

            time_bin_size_encoding = model_dic.time_bin_size

            if not time_bin_size_encoding == self.params.time_bin_size:
                raise Exception("Time bin size of model and data are not matching!")

            if cells_to_use == "all":
                cell_selection = "all"
                cell_ids = np.empty(0)

            else:

                cell_selection = "custom"
                with open(self.params.pre_proc_dir + "cell_classification/" +
                          self.session_name +"_"+ self.params.stable_cell_method +".pickle","rb") as f:
                    class_dic = pickle.load(f)

                if cells_to_use == "stable":
                    cell_ids = class_dic["stable_cell_ids"]
                elif cells_to_use == "increasing":
                    cell_ids = class_dic["increase_cell_ids"]
                elif cells_to_use == "decreasing":
                    cell_ids = class_dic["decrease_cell_ids"]

            print(" - DECODING USING " + cells_to_use + " CELLS")

            # decode activity
            results_list = decode_using_phmm_modes(mode_means=mode_means, event_spike_rasters=spike_rasters,
                                                   compression_factor=1, cells_to_use=cell_ids,
                                                   cell_selection=cell_selection)

            # plot maps of some SWR for control
            if plot_for_control:
                for trial_id, res in zip(trials_to_use, results_list):
                    plt.imshow(np.log(res.T), interpolation='nearest', aspect='auto')
                    plt.xlabel("POP.VEC. ID")
                    plt.ylabel("MODE ID")
                    a = plt.colorbar()
                    a.set_label("LOG-PROBABILITY")
                    plt.title("LOG-PROBABILITY MAP: TRIAL "+str(trial_id))
                    plt.show()

            # saving results
            # --------------------------------------------------------------------------------------------------
            # create dictionary with results
            result_post = {
                "results_list": results_list,
            }
            outfile = open(self.params.pre_proc_dir + result_dir +"/" + file_name, 'wb')
            pickle.dump(result_post, outfile)
            print("  - SAVED NEW RESULTS!\n")

        if return_results:

            # load decoded maps
            result_pre = pickle.load(open(self.params.pre_proc_dir+result_dir + "/" + file_name, "rb"))

            pre_prob = result_pre["results_list"]

            return pre_prob

    def decode_awake_activity_visualization(self, cells_to_use="all", binning="spike_binning"):

        if binning == "spike_binning":
            results = self.decode_awake_activity_spike_binning(cells_to_use=cells_to_use)
        elif binning == "time_binning":
            results = self.decode_awake_activity_time_binning(cells_to_use=cells_to_use)

        a = np.vstack(results)
        b = np.argmax(a, axis=1)

        mode_ids, nr_counts = np.unique(b, return_counts=True)


        fig, ax = plt.subplots()

        lines = []
        for i in range(mode_ids.shape[0]):
            pair = [(mode_ids[i], 0), (mode_ids[i], nr_counts[i])]
            lines.append(pair)

        linecoll = matcoll.LineCollection(lines, colors="lightskyblue")

        ax.add_collection(linecoll)
        plt.scatter(mode_ids, nr_counts)
        plt.xlabel("Mode ID")
        plt.ylabel("Times assigned")
        plt.title("Awake decoding - "+cells_to_use)
        plt.show()

        exit()

        sep_array = np.cumsum(np.array([x.shape[0] for x in results]))
        labels = [str(x) for x in self.params.default_trials]

        results = np.log(np.vstack(results).T)

        rd = multi_dim_scaling(act_mat=results, param_dic=self.params)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot_2D_scatter(ax=ax, mds=rd, data_sep=sep_array, labels=labels, params=self.params)
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

    def decode_awake_activity_autocorrelation_spikes_likelihood_vectors(self, plot_for_control=True, plotting=True,
                                                                        bootstrapping=False, nr_pop_vecs=10, save_fig=False):

        # get likelihood vectors
        likeli_vecs_list = self.decode_awake_activity_spike_binning()
        likeli_vecs = np.vstack(likeli_vecs_list)
        # compute correlations
        shift_array = np.arange(-1*int(nr_pop_vecs),
                                     int(nr_pop_vecs)+1)

        auto_corr, _ = cross_correlate_matrices(likeli_vecs.T, likeli_vecs.T, shift_array=shift_array)

        # fitting exponential
        # --------------------------------------------------------------------------------------------------------------
        # only take positive part (symmetric) & exclude first data point
        autocorr_test_data = auto_corr[int(auto_corr.shape[0] / 2):][1:]

        def exponential(x, a, k, b):
            return a * np.exp(x * k) + b

        popt_exponential_awake, pcov_exponential_awake = optimize.curve_fit(exponential, np.arange(autocorr_test_data.shape[0]),
                                                                        autocorr_test_data, p0=[1, -0.5, 1])

        if plotting or save_fig:

            if save_fig:
                plt.style.use('default')
            plt.plot(shift_array, (auto_corr-auto_corr[-1])/(1-auto_corr[-1]), c="y", label="Awake")
            plt.xlabel("Shift (#spikes)")
            plt.ylabel("Avg. Pearson correlation of likelihood vectors")
            plt.legend()
            # plt.xticks([-100, -75, -50, -25, 0, 25, 50, 75, 100], np.array([-100, -75, -50, -25, 0, 25, 50, 75, 100]) * 12)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("sim_ratio_autocorr_spikes.svg", transparent="True")
                plt.close()
            else:
                print("HERE")
                plt.show()

        else:
            auto_corr_norm = (auto_corr-auto_corr[-1])/(1-auto_corr[-1])
            return auto_corr, auto_corr_norm, popt_exponential_awake[1]

        # nrem_test_data = auto_corr_nrem[int(auto_corr_nrem.shape[0] / 2)+1:]
        # rem_test_data = auto_corr_rem[int(auto_corr_rem.shape[0] / 2) + 1:]
        #
        # def exponential(x, a, k, b):
        #     return a * np.exp(x * k) + b
        #
        # popt_exponential_rem, pcov_exponential_rem = optimize.curve_fit(exponential, np.arange(rem_test_data.shape[0]),
        #                                                                 rem_test_data, p0=[1, -0.5, 1])
        # popt_exponential_nrem, pcov_exponential_nrem = optimize.curve_fit(exponential, np.arange(nrem_test_data.shape[0]),
        #                                                                 nrem_test_data, p0=[1, -0.5, 1])
        # if plotting or save_fig:
        #
        #     if save_fig:
        #         plt.style.use('default')
        #     # plot fits
        #     plt.text(3, 10, "k = " +str(np.round(popt_exponential_rem[1], 2)), c="red" )
        #     plt.scatter(np.arange(rem_test_data.shape[0]), rem_test_data, c="salmon", label="REM data")
        #     plt.plot((np.arange(rem_test_data.shape[0]))[1:], exponential((np.arange(rem_test_data.shape[0]))[1:],
        #                                                             a=popt_exponential_rem[0], k=popt_exponential_rem[1],
        #                                                             b=popt_exponential_rem[2]), c="red", label="REM fit")
        #     plt.text(0.05, 5, "k = " +str(np.round(popt_exponential_nrem[1], 2)), c="blue" )
        #     plt.scatter(np.arange(nrem_test_data.shape[0]), nrem_test_data, c="lightblue", label="NREM data")
        #     plt.plot((np.arange(nrem_test_data.shape[0]))[1:], exponential((np.arange(nrem_test_data.shape[0]))[1:],
        #                                                             a=popt_exponential_nrem[0], k=popt_exponential_nrem[1],
        #                                                             b=popt_exponential_nrem[2]), c="blue", label="NREM fit")
        #
        #     plt.legend(loc=2)
        #     plt.ylabel("Pearson R (z-scored)")
        #     plt.xlabel("nr. spikes")
        #     plt.ylim(-3, 18)
        #     if save_fig:
        #         plt.rcParams['svg.fonttype'] = 'none'
        #         plt.savefig("exponential_fit_spikes.svg", transparent="True")
        #         plt.close()
        #     else:
        #         plt.show()
        #
        # if bootstrapping:
        #
        #     # bootstrapping
        #     n_boots = 500
        #     n_samples_perc = 0.8
        #
        #     nrem_exp = []
        #     rem_exp = []
        #
        #     for boots_id in range(n_boots):
        #         per_ind = np.random.permutation(np.arange(rem_test_data.shape[0]))
        #         sel_ind = per_ind[:int(n_samples_perc*per_ind.shape[0])]
        #         # select subset
        #         x_rem = np.arange(nrem_test_data.shape[0])[sel_ind]
        #         x_nrem = np.arange(nrem_test_data.shape[0])[sel_ind]
        #         y_rem = rem_test_data[sel_ind]
        #         y_nrem = nrem_test_data[sel_ind]
        #         try:
        #             popt_exponential_rem, _ = optimize.curve_fit(exponential,x_rem, y_rem, p0=[1, -0.5, 1])
        #             popt_exponential_nrem, _ = optimize.curve_fit(exponential, x_nrem, y_nrem, p0=[1, -0.5, 1])
        #         except:
        #             continue
        #
        #         rem_exp.append(popt_exponential_rem[1])
        #         nrem_exp.append(popt_exponential_nrem[1])
        #
        #     if plotting:
        #         plt.hist(rem_exp, label="rem", color="red", bins=10, density=True)
        #         plt.xlabel("k from exp. function")
        #         plt.ylabel("density")
        #         plt.legend()
        #         plt.show()
        #         plt.hist(nrem_exp, label="nrem", color="blue", alpha=0.8, bins=10, density=True)
        #         # plt.xlim(-2,0.1)
        #         # plt.title("k from exponential fit (bootstrapped)\n"+"Ttest one-sided: p="+\
        #         #           str(ttest_ind(rem_exp, nrem_exp, alternative="greater")[1]))
        #         # plt.xscale("log")
        #         plt.show()
        #     else:
        #         return np.median(np.array(rem_exp)), np.median(np.array(nrem_exp))
        # else:
        #     return popt_exponential_rem[1], popt_exponential_nrem[1]

    """#################################################################################################################
    #  location decoding analysis
    #################################################################################################################"""

    def decode_location_phmm(self, trial_to_decode, model_name=None, trials_to_use=None, save_fig=False):
        """
        decodes awake activity using pHMM modes from same experiment phase

        @param trial_to_decode: which trial to decode
        @type trial_to_decode: int
        @param model_name: name of file containing the pHMM file
        @type model_name: str
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        print(" - DECODE LOCATION USING PHMM MODES ...\n")
        result_dir = "phmm_decoding"
        # get template file name from parameter file of session if not provided
        if model_name is None:
            model_name = self.session_params.default_pre_phmm_model

        if model_name is None:
            raise Exception("MODEL FILE NOT FOUND\n NOR IN SESSION PARAMETER FILE DEFINED")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        print("   --> USING TRIALS: "+str(trials_to_use[0])+"-"+str(trials_to_use[-1])+"\n")

        # check that template files are from correct session
        # --------------------------------------------------------------------------------------------------------------
        sess_pre = model_name.split(sep="_")[0]+"_"+model_name.split(sep="_")[1]

        if not (sess_pre == self.session_name):
            raise Exception("TEMPLATE FILE AND CURRENT SESSION DO NOT MATCH!")

        # load goal locations
        goal_loc = np.loadtxt(self.params.pre_proc_dir+"goal_locations"+"/"+"mjc163R2R_0114_11.gcoords")

        # get spatial information from pHMM model
        means, _, _, _, _ = self.fit_spatial_gaussians_for_modes(file_name=model_name)

        # get location data and raster from trial
        raster = self.trial_raster_list[trial_to_decode]
        loc = self.trial_loc_list[trial_to_decode]

        model = self.load_poisson_hmm(file_name=model_name)

        prob = model.predict_proba(raster.T)

        # decode activity
        # prob_poiss = decode_using_phmm_modes(mode_means=model.means_,
        #                                        event_spike_rasters=[raster],
        #                                        compression_factor=1)[0]
        #
        # prob_poiss_norm = prob_poiss / np.sum(prob_poiss, axis=1, keepdims=True)
        #
        # plt.imshow(prob_poiss_norm, interpolation='nearest', aspect='auto')
        # plt.show()
        # plt.imshow(prob, interpolation='nearest', aspect='auto')
        # plt.show()

        if save_fig:
            plt.style.use('default')
            err = np.zeros(prob.shape[0])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            prev_dec_location = None
            prev_location = None
            for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
                # compute decoded location using weighted average
                dec_location = np.average(means, weights=current_prob, axis=1)
                err[i] = np.linalg.norm(dec_location - current_loc)
                if i > 1:
                    # ax.plot([dec_location[0], prev_dec_location[0]], [dec_location[1], prev_dec_location[1]], color="mistyrose",
                    #         zorder=-1000)
                    ax.plot([current_loc[0], prev_location[0]], [current_loc[1], prev_location[1]], color="lightgray",
                            zorder=-1000)
                ax.scatter(dec_location[0], dec_location[1], color="red", label="Decoded locations")
                ax.scatter(current_loc[0], current_loc[1], color="lightgray", label="True locations")
                prev_dec_location = dec_location
                prev_location = current_loc
            for g_l in self.goal_locations:
                ax.scatter(g_l[0], g_l[1], color="black", label="Goals", marker="x")
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("Decoding using weighted means")
            plt.xlabel("X (cm)")
            plt.ylabel("Y (cm)")
            plt.ylim(10, 110)
            plt.xlim(30, 140)
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoded_locations_phmm.svg", transparent="True")
            # plt.show()
            plt.close()

            plt.hist(err, density=True, color="indianred", bins=int(err.shape[0]/5))
            plt.title("Decoding error")
            plt.xlabel("Error (cm)")
            plt.xlim(-5,100)
            plt.ylabel("Density")
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig("decoding_error_phmm.svg", transparent="True")
            # plt.show()
        else:
            col_map_red = cm.Reds(np.linspace(0, 1, prob.shape[0]))
            close_to_goal=np.zeros(prob.shape[0])
            err = np.zeros(prob.shape[0])
            fig = plt.figure()
            ax = fig.add_subplot(111)

            for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
                # compute decoded location using weighted average
                dec_location = np.average(means, weights=current_prob, axis=1)
                err[i] = np.linalg.norm(dec_location - current_loc)
                ax.scatter(dec_location[0], dec_location[1], color="blue", label="DECODED")
                ax.scatter(current_loc[0], current_loc[1], color=col_map_red[i], label="TRUE")
                ax.plot([dec_location[0], current_loc[0]], [dec_location[1], current_loc[1]], color="gray",
                         zorder=-1000, label="ERRORS")
                for gl in goal_loc:
                    if np.linalg.norm(current_loc - gl) < 20:
                        close_to_goal[i] = 1
                        ax.scatter(current_loc[0], current_loc[1], facecolors='none', edgecolors="white", label="CLOSE TO GOAL LOC.")
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("DECODING USING WEIGHTED MODE MEANS")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
            # err = moving_average(a = np.array(err), n=20)
            plt.plot(err, color="gray", label="ERROR")
            err = moving_average(a=np.array(err), n=20)
            plt.plot(err, color="lightcoral", label="ERROR SMOOTHED")
            plt.plot(close_to_goal*10, color="w", label="CLOSE TO GOAL LOC.")
            plt.ylabel("ERROR / cm")
            plt.xlabel("TIME BIN")
            plt.legend()
            plt.show()

            sequence = model.decode(raster.T, algorithm="map")[1]

            err = []
            close_to_goal=np.zeros(sequence.shape[0])
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # compute error
            for i, (mode_act, current_loc) in enumerate(zip(sequence, loc)):
                ax.scatter(means[0,mode_act], means[1,mode_act], color="blue", label="DECODED")
                ax.scatter(current_loc[0], current_loc[1], color=col_map_red[i], label="TRUE")
                ax.plot([means[0,mode_act], current_loc[0]], [means[1,mode_act], current_loc[1]], color="gray",
                         zorder=-1000, label="ERROR")
                err.append(np.linalg.norm(means[:,mode_act]-current_loc))
                for gl in goal_loc:
                    if np.linalg.norm(current_loc - gl) < 20:
                        close_to_goal[i] = 1
                        ax.scatter(current_loc[0], current_loc[1], facecolors='none', edgecolors="white", label="CLOSE TO GOAL LOC.")

            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("DECODING USING MOST LIKELY MODE SEQUENCE")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
            # err = moving_average(a = np.array(err), n=20)
            plt.plot(err, color="gray", label="ERROR")
            err = moving_average(a=np.array(err), n=20)
            plt.plot(err, color="lightcoral", label="ERROR SMOOTHED")
            plt.plot(close_to_goal*10, color="w", label="CLOSE TO GOAL LOC.")
            plt.legend()
            plt.ylabel("ERROR / cm")
            plt.xlabel("TIME BIN")
            plt.show()

            mean_err = []
            median_err = []

            for trial_to_decode in trials_to_use:
                raster = self.trial_raster_list[trial_to_decode]
                loc = self.trial_loc_list[trial_to_decode]
                prob = model.predict_proba(raster.T)
                err = []
                for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
                    # compute decoded location using weighted average
                    dec_location = np.average(means, weights=current_prob, axis=1)
                    err.append(np.linalg.norm(dec_location - current_loc))

                mean_err.append(np.mean(np.array(err)))
                median_err.append(np.median(np.array(err)))

            plt.plot(trials_to_use, mean_err, label="MEAN")
            plt.plot(trials_to_use, median_err, label="MEDIAN")
            plt.ylabel("ERROR / cm")
            plt.xlabel("TRIAL ID")
            plt.ylim(10,40)
            plt.legend()
            plt.show()

    def decode_location_ising(self, trial_to_decode, model_name=None, trials_to_use=None):
        """
        decodes awake activity using spatial bins from ising model from same experiment phase

        @param trial_to_decode: which trial to decode
        @type trial_to_decode: int
        @param model_name: name of file containing the pHMM file
        @type model_name: str
        @param plot_for_control: plots intermediate results if True
        @type plot_for_control: bool
        @param return_results: whether to return results or not
        @type return_results: bool
        @return: pre_prob (probabilities of PRE), post_prob (probabilities of ), event_times (in sec),
        swr_in_which_n_rem (assignement SWR - nrem phase)
        @rtype: list, list, list, numpy.array
        """

        print(" - DECODE LOCATION USING ISING ...\n")

        # get template file name from parameter file of session if not provided
        if model_name is None:
            model_name = self.params.default_pre_ising_model

        if model_name is None:
            raise Exception("MODEL FILE NOT FOUND\n NOR IN SESSION PARAMETER FILE DEFINED")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        print("   --> USING TRIALS: "+str(trials_to_use[0])+"-"+str(trials_to_use[-1])+"\n")

        # check that template files are from correct session
        # --------------------------------------------------------------------------------------------------------------
        sess_pre = model_name.split(sep="_")[0]+"_"+model_name.split(sep="_")[1]

        if not (sess_pre == self.params.session_name):
            raise Exception("TEMPLATE FILE AND CURRENT SESSION DO NOT MATCH!")

            # load ising template
        with open(self.params.pre_proc_dir + 'awake_ising_maps/' + model_name + '.pkl',
                  'rb') as f:
            model_dic = pickle.load(f)

        # load goal locations
        goal_loc = np.loadtxt(self.params.pre_proc_dir+"goal_locations"+"/"+"mjc163R2R_0114_11.gcoords")

        # get location data and raster from trial
        raster = self.trial_raster_list[trial_to_decode]
        loc = self.trial_loc_list[trial_to_decode]

        # decode activity
        # get template map
        template_map = model_dic["res_map"]

        # need actual spatial bin position to do decoding
        bin_size_x = np.round((self.x_max - self.x_min)/template_map.shape[1], 0)
        bins_x = np.linspace(self.x_min+bin_size_x/2, self.x_max-bin_size_x/2, template_map.shape[1])
        # bins_x = np.repeat(bins_x[None, :], template_map.shape[2], axis=0)

        # bins_x = bins_x.reshape(-1, (template_map.shape[1] * template_map.shape[2]))

        bin_size_y = np.round((self.y_max - self.y_min)/template_map.shape[2], 0)
        bins_y = np.linspace(self.y_min+bin_size_y/2, self.y_max-bin_size_y/2, template_map.shape[2])
        # bins_y = np.repeat(bins_y[:, None], template_map.shape[1], axis=1)

        # bins_y = bins_y.reshape(-1, (template_map.shape[1] * template_map.shape[2]))

        prob = decode_using_ising_map(template_map=template_map,
                                              event_spike_rasters=[raster],
                                              compression_factor=10,
                                              cell_selection="all")[0]


        # prob_poiss_norm = prob_poiss / np.sum(prob_poiss, axis=1, keepdims=True)

        # plt.imshow(prob_poiss_norm, interpolation='nearest', aspect='auto')
        # plt.show()
        # plt.imshow(np.log(prob_poiss), interpolation='nearest', aspect='auto')
        # plt.show()

        col_map_red = cm.Reds(np.linspace(0, 1, prob.shape[0]))
        close_to_goal=np.zeros(prob.shape[0])
        err = np.zeros(prob.shape[0])
        fig = plt.figure()
        ax = fig.add_subplot(111)

        for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):

            prob_map = current_prob.reshape(template_map.shape[1],template_map.shape[2])
            max_ind = np.unravel_index(prob_map.argmax(), prob_map.shape)
            # plt.imshow(prob_map.T, origin="lower")
            # plt.colorbar()
            # plt.show()
            # compute decoded location using weighted average
            dec_location_x = bins_x[max_ind[0]]
            dec_location_y = bins_y[max_ind[1]]
            dec_location = np.array([dec_location_x, dec_location_y])
            err[i] = np.linalg.norm(dec_location - current_loc)
            ax.scatter(dec_location[0], dec_location[1], color="blue", label="DECODED")
            ax.scatter(current_loc[0], current_loc[1], color=col_map_red[i], label="TRUE")
            ax.plot([dec_location[0], current_loc[0]], [dec_location[1], current_loc[1]], color="gray",
                     zorder=-1000, label="ERRORS")
            for gl in goal_loc:
                if np.linalg.norm(current_loc - gl) < 20:
                    close_to_goal[i] = 1
                    ax.scatter(current_loc[0], current_loc[1], facecolors='none', edgecolors="white", label="CLOSE TO GOAL LOC.")
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.title("BAYESIAN DECODING")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        # err = moving_average(a = np.array(err), n=20)
        plt.plot(err, color="gray", label="ERROR")
        err = moving_average(a=np.array(err), n=20)
        plt.plot(err, color="lightcoral", label="ERROR SMOOTHED")
        plt.plot(close_to_goal*10, color="w", label="CLOSE TO GOAL LOC.")
        plt.ylabel("ERROR / cm")
        plt.xlabel("TIME BIN")
        plt.legend()
        plt.show()


        mean_err = []
        median_err = []

        for trial_to_decode in trials_to_use:
            raster = self.trial_raster_list[trial_to_decode]
            loc = self.trial_loc_list[trial_to_decode]
            prob = decode_using_ising_map(template_map=template_map,
                                          event_spike_rasters=[raster],
                                          compression_factor=10,
                                          cell_selection="all")[0]
            err = []
            for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
                # compute decoded location using weighted average
                prob_map = current_prob.reshape(template_map.shape[1], template_map.shape[2])
                max_ind = np.unravel_index(prob_map.argmax(), prob_map.shape)
                # plt.imshow(prob_map.T, origin="lower")
                # plt.colorbar()
                # plt.show()
                # compute decoded location using weighted average
                dec_location_x = bins_x[max_ind[0]]
                dec_location_y = bins_y[max_ind[1]]
                dec_location = np.array([dec_location_x, dec_location_y])
                err.append(np.linalg.norm(dec_location - current_loc))

            mean_err.append(np.mean(np.array(err)))
            median_err.append(np.median(np.array(err)))

        plt.plot(trials_to_use, mean_err, label="MEAN")
        plt.plot(trials_to_use, median_err, label="MEDIAN")
        plt.ylabel("ERROR / cm")
        plt.xlabel("TRIAL ID")
        plt.legend()
        plt.show()

    def decode_location_bayes(self, cell_subset=None, trials_train=None, trials_test=None, save_fig=False,
                              plotting=False):
        """
        Location decoding using Bayes

        :param trials_train: trials used to generate rate maps
        :type trials_train: iterable
        :param trials_test: trials used for testing (pop.vec. & location)
        :type trials_test: iterable
        :param save_fig: whether to save figure or not
        :type save_fig: bool
        :param plotting: whether to plot results or return results (error)
        :type plotting: bool
        :param cell_subset: subset of cells that is used for decoding
        :type cell_subset: array
        """
        print(" - DECODING LOCATION USING BAYESIAN ...\n")

        # if no trials are provided: train/test on default trials without cross-validation
        if trials_train is None:
            trials_train = self.default_trials
        if trials_test is None:
            trials_test = self.default_trials

        # get train data --> rate maps
        # --------------------------------------------------------------------------------------------------------------
        rate_maps = self.get_rate_maps(spatial_resolution=1, trials_to_use=trials_train)

        if cell_subset is not None:
            rate_maps = rate_maps[:, :, cell_subset]

        # flatten rate maps
        rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0] * rate_maps.shape[1], rate_maps.shape[2]))

        # get test data --> population vectors and location
        # --------------------------------------------------------------------------------------------------------------
        raster = []
        loc = []
        for trial_id in trials_test:
            raster.append(self.trial_raster_list[trial_id])
            loc.append(self.trial_loc_list[trial_id])

        loc = np.vstack(loc)
        raster = np.hstack(raster)

        if cell_subset is not None:
            raster = raster[cell_subset, :]

        if plotting or save_fig:
            plt.style.use('default')
            fig = plt.figure()
            ax = fig.add_subplot(111)

        decoding_err = []
        for i, (pop_vec, current_loc) in enumerate(zip(raster.T, loc)):
            if np.count_nonzero(pop_vec) == 0:
                continue
            bl = bayes_likelihood(frm=rate_maps_flat.T, pop_vec=pop_vec, log_likeli=False)
            bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
            pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
            pred_x = pred_bin[0] + self.x_min
            pred_y = pred_bin[1] + self.y_min
            dec_location = np.array([pred_x, pred_y])
            decoding_err.append(np.sqrt((pred_x - current_loc[0]) ** 2 + (pred_y - current_loc[1]) ** 2))

            if plotting or save_fig:
                ax.scatter(dec_location[0], dec_location[1], color="red", label="Decoded locations")
                ax.scatter(current_loc[0], current_loc[1], color="lightgray", label="True locations")

        if plotting or save_fig:
            for g_l in self.goal_locations:
                ax.scatter(g_l[0], g_l[1], color="black", label="Goals", marker="x")
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.title("Bayesian decoding")
            plt.xlabel("X (cm)")
            plt.ylabel("Y (cm)")
            plt.ylim(10, 110)
            plt.xlim(30, 140)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("decoded_locations_bayes.svg", transparent="True")
                plt.close()
            else:
                plt.show()

            decoding_err = np.array(decoding_err)
            plt.hist(decoding_err, density=True, color="indianred", bins=int(decoding_err.shape[0]/5))
            plt.title("Decoding error")
            plt.xlabel("Error (cm)")
            plt.ylabel("Density")
            plt.xlim(-5, 100)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("decoding_error_bayes.svg", transparent="True")
            else:
                plt.show()

        if (not plotting) & (not save_fig):
            return decoding_err


"""#####################################################################################################################
#   CHEESEBOARD CLASS
#####################################################################################################################"""


class Cheeseboard(TrialParentClass):
    """Base class for cheese board task data analysis

       ATTENTION: this is only used for the task data --> otherwise use Sleep class!

    """

    def __init__(self, data_dic, cell_type, params, session_params, experiment_phase=None):
        """
        initializes cheeseboard class

        :param data_dic: dictionary containing spike data
        :type data_dic: python dic
        :param cell_type: which cell type to use
        :type cell_type: str
        :param params: general analysis params
        :type params: class
        :param session_params: sessions specific params
        :type session_params: class
        :param exp_phase_id: which experiment phase id
        :type exp_phase_id: int
        """

        # get attributes from parent class
        TrialParentClass.__init__(self, data_dic, cell_type, params, session_params, experiment_phase)

        # --------------------------------------------------------------------------------------------------------------
        # get phase specific info (ID & description)
        # --------------------------------------------------------------------------------------------------------------


        # compression factor:
        #
        # compression factor used for sleep decoding --> e.g when we use constant #spike bins with 12 spikes
        # we need to check how many spikes we have in e.g. 100ms windows if this was used for awake encoding
        # if we have a mean of 30 spikes for awake --> compression factor = 12/30 --> 0.4
        # is used to scale awake activity to fit sleep activity
        # --------------------------------------------------------------------------------------------------------------

        if cell_type == "p1_l":
            self.session_params.sleep_compression_factor_12spikes_100ms = \
                self.session_params.sleep_compression_factor_12spikes_100ms_p1_l
        elif cell_type == "p1_r":
            self.session_params.sleep_compression_factor_12spikes_100ms = \
                self.session_params.sleep_compression_factor_12spikes_100ms_p1_r
        else:
            self.session_params.sleep_compression_factor_12spikes_100ms = \
                self.session_params.sleep_compression_factor_12spikes_100ms

        # default models for behavioral data
        # --------------------------------------------------------------------------------------------------------------

        if cell_type == "p1_l":
            self.session_params.default_pre_phmm_model = self.session_params.default_pre_phmm_model_p1_l
            self.session_params.default_post_phmm_model = self.session_params.default_post_phmm_model_p1_l
            self.session_params.default_pre_ising_model = self.session_params.default_pre_ising_model_p1_l
            self.session_params.default_post_ising_model = self.session_params.default_post_ising_model_p1_l
        elif cell_type == "p1_r":
            self.session_params.default_pre_phmm_model = self.session_params.default_pre_phmm_model_p1_r
            self.session_params.default_post_phmm_model = self.session_params.default_post_phmm_model_p1_r
            self.session_params.default_pre_ising_model = self.session_params.default_pre_ising_model_p1_r
            self.session_params.default_post_ising_model = self.session_params.default_post_ising_model_p1_r
        else:
            self.session_params.default_pre_phmm_model = self.session_params.default_pre_phmm_model
            self.session_params.default_post_phmm_model = self.session_params.default_post_phmm_model
            self.session_params.default_pre_ising_model = self.session_params.default_pre_ising_model
            self.session_params.default_post_ising_model = self.session_params.default_post_ising_model

        # goal locations
        # --------------------------------------------------------------------------------------------------------------
        try:
            self.goal_locations = self.session_params.goal_locations
        except:
            print("GOAL LOCATIONS NOT FOUND")

        # convert goal locations from a.u. to cm
        self.goal_locations = np.array(self.goal_locations) * self.spatial_factor

        # get pre-selected trials (e.g. last 10 min of learning 1 and first 10 min of learning 2)
        # --------------------------------------------------------------------------------------------------------------
        if session_params.experiment_phase == "learning_cheeseboard_1":
            self.default_trials = self.session_params.default_trials_lcb_1
            self.default_ising = self.session_params.default_pre_ising_model
            self.default_phmm = self.session_params.default_pre_phmm_model
            self.default_phmm_stable = self.session_params.default_pre_phmm_model_stable
        elif session_params.experiment_phase == "learning_cheeseboard_2":
            self.default_trials = self.session_params.default_trials_lcb_2
            self.default_ising = self.session_params.default_post_ising_model
            self.default_phmm = self.session_params.default_post_phmm_model


        # compute raster, location speed
        # --------------------------------------------------------------------------------------------------------------
        self.trial_loc_list = []
        self.trial_raster_list = []
        self.trial_speed_list = []

        # get x-max from start box --> discard all data that is smaller than x-max
        x_max_sb = self.session_params.data_params_dictionary["start_box_coordinates"][1] * self.spatial_factor

        # compute center of cheeseboard (x_max_sb + 110 cm, assumption: cheeseboard diameter: 220cm)
        x_c = x_max_sb + 110 * self.spatial_factor

        # use center of start box to find y coordinate of center of cheeseboard
        y_c = self.session_params.data_params_dictionary["start_box_coordinates"][2]+ \
              (self.session_params.data_params_dictionary["start_box_coordinates"][3] - \
              self.session_params.data_params_dictionary["start_box_coordinates"][2])/2

        cb_center = np.expand_dims(np.array([x_c, y_c]), 0)

        # compute raster, location & speed for each trial
        for trial_id, key in enumerate(self.data_dic["trial_data"]):
            # compute raster, location and speed data
            raster, loc, speed = PreProcessAwake(firing_times=self.data_dic["trial_data"][key]["spike_times"][cell_type],
                                                 params=self.params, whl=self.data_dic["trial_data"][key]["whl"],
                                                 spatial_factor=self.spatial_factor
                                                 ).interval_temporal_binning_raster_loc_vel(
                interval_start=self.data_dic["trial_timestamps"][0,trial_id],
                interval_end=self.data_dic["trial_timestamps"][1,trial_id])

            # TODO: improve filtering of spatial locations outside cheeseboard
            if self.params.additional_spatial_filter:
                # filter periods that are outside the cheeseboard
                dist_from_center = distance.cdist(loc, cb_center, metric="euclidean")

                raster = np.delete(raster, np.where(dist_from_center > 110* self.spatial_factor), axis=1)
                speed = np.delete(speed, np.where(dist_from_center > 110* self.spatial_factor))
                loc = np.delete(loc, np.where(dist_from_center > 110* self.spatial_factor), axis=0)

            # TODO delete this section below
            # if self.session_name == "mjc163R4R_0114":
            #     dist_from_center = distance.cdist(loc, cb_center, metric="euclidean")
            #
            #     raster = np.delete(raster, np.where(dist_from_center > 110* self.spatial_factor), axis=1)
            #     speed = np.delete(speed, np.where(dist_from_center > 110* self.spatial_factor))
            #     loc = np.delete(loc, np.where(dist_from_center > 110* self.spatial_factor), axis=0)

            # if self.session_name == "mjc148R4R_0113":
            #     raster = np.delete(raster, np.where(loc[:,0] < 85* self.spatial_factor), axis=1)
            #     speed = np.delete(speed, np.where(loc[:, 0] < 85* self.spatial_factor))
            #     loc = np.delete(loc, np.where(loc[:, 0] < 85), axis=0)

            if self.session_name in ["mjc163R2R_0114", "mjc163R4R_0114", "mjc169R4R_0114", "mjc163R1L_0114",
                                     "mjc163R3L_0114", "mjc169R1R_0114"]:
                raster = np.delete(raster, np.where(loc[:,0] < x_max_sb), axis=1)
                speed = np.delete(speed, np.where(loc[:, 0] < x_max_sb))
                loc = np.delete(loc, np.where(loc[:, 0] < x_max_sb), axis=0)

            # update environment dimensions
            self.x_min = min(self.x_min, min(loc[:,0]))
            self.x_max = max(self.x_max, max(loc[:, 0]))
            self.y_min = min(self.y_min, min(loc[:,1]))
            self.y_max = max(self.y_max, max(loc[:, 1]))

            self.trial_raster_list.append(raster)
            self.trial_loc_list.append(loc)
            self.trial_speed_list.append(speed)

    """#################################################################################################################
    #  Plotting methods
    #################################################################################################################"""

    def plot_tracking_and_goals(self, ax=None, trials_to_use=None):
        if trials_to_use is None:
            trials_to_use = self.default_trials
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))
        ax.scatter(loc[:, 0], loc[:, 1], color="grey", s=1, label="TRACKING")

        # plot actual trajectory
        for part in range(loc.shape[0]-1):
            ax.plot(loc[part:part+2, 0], loc[part:part+2, 1], color="red")

        for g_l in self.goal_locations:
            ax.scatter(g_l[0], g_l[1], marker="x", color="w", label="GOALS")
        plt.show()

    def plot_tracking_and_goals_first_or_last_trial(self, ax=None, save_fig=False, trial="first"):

        if save_fig:
            plt.style.use('default')
        if trial=="first":
            trials_to_use = [0]
        elif trial =="last":
            trials_to_use = [len(self.trial_raster_list) - 1]
        else:
            raise Exception("Define [first] or [last] trial")

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))
        # ax.scatter(loc[:, 0], loc[:, 1], color="lightblue", s=1, label="TRACKING")

        # plot actual trajectory
        for part in range(loc.shape[0]-1):
            ax.plot(loc[part:part+2, 0], loc[part:part+2, 1], color="lightblue", label="Tracking")

        for g_l in self.goal_locations:
            ax.scatter(g_l[0], g_l[1], color="black", label="Goals")

        plt.gca().set_aspect('equal', adjustable='box')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.xlim(20, 160)
        plt.ylim(10, 130)
        plt.xlabel("X (cm)")
        plt.ylabel("Y (cm)")

        if save_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(self.experiment_phase+"_tracking_"+trial+"_trial.svg", transparent="True")
            plt.close()
        else:
            plt.show()

    """#################################################################################################################
    #  Saving methods
    #################################################################################################################"""

    def save_goal_coding_all_modes(self, nr_modes, out_file_name):

        trials_to_use = self.default_trials

        file_name = self.session_name + "_" + str(
            int(self.params.experiment_phase_id[0])) + "_" + self.cell_type + "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1])+"_"+str(nr_modes)+"_modes"

        frac_per_mode = []
        for mode_id in range(nr_modes):
            frac_close_to_goal = self.analyze_modes_goal_coding(file_name=file_name, mode_ids=mode_id)
            frac_per_mode.append(frac_close_to_goal)

        np.save(out_file_name, np.array(frac_per_mode))
        plt.hist(frac_per_mode, bins=20)
        plt.xlabel("FRACTION AROUND GOAL")
        plt.ylabel("COUNT")
        plt.show()

    """#################################################################################################################
    #  learning
    #################################################################################################################"""

    def map_dynamics_learning(self, nr_shuffles=500, plot_results=True, n_trials=5,
                              spatial_resolution=3, adjust_pv_size=False):

        # get maps for first trials
        initial = self.get_rate_maps(trials_to_use=range(n_trials), spatial_resolution=spatial_resolution)
        initial_occ = self.get_occ_map(trials_to_use=range(n_trials), spatial_resolution=spatial_resolution)
        # get maps for last trial
        last = self.get_rate_maps(trials_to_use=range(len(self.trial_loc_list)-n_trials, len(self.trial_loc_list)),
                                  spatial_resolution=spatial_resolution)
        last_occ = self.get_occ_map(trials_to_use=range(len(self.trial_loc_list)-n_trials, len(self.trial_loc_list)),
                                    spatial_resolution=spatial_resolution)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()
        nr_stable_cells = stable_cells.shape[0]
        nr_dec_cells = dec_cells.shape[0]

        # compute remapping based on population vectors per bin
        # --------------------------------------------------------------------------------------------------------------

        pop_vec_initial = np.reshape(initial,
                                 (initial.shape[0] * initial.shape[1], initial.shape[2]))
        pop_vec_last = np.reshape(last, (last.shape[0] * last.shape[1], last.shape[2]))

        # only select spatial bins that were visited in PRE and POST
        comb_occ_map = np.logical_and(initial_occ.flatten() > 0, last_occ.flatten() > 0)
        pop_vec_initial = pop_vec_initial[comb_occ_map, :]
        pop_vec_last = pop_vec_last[comb_occ_map, :]

        comb_occ_map_spatial = np.reshape(comb_occ_map,(initial.shape[0], initial.shape[1]))
        common_loc = np.where(comb_occ_map_spatial)
        # stable cells

        pop_vec_initial_stable = pop_vec_initial[:, stable_cells]
        pop_vec_last_stable = pop_vec_last[:, stable_cells]

        plt.imshow(pop_vec_initial_stable.T, interpolation='none', aspect='auto')
        plt.title("First 5 trials")
        plt.ylabel("Stable cells")
        plt.xlabel("Spatial bins")
        plt.show()

        plt.imshow(pop_vec_last_stable.T, interpolation='none', aspect='auto')
        plt.title("Last 5 trials")
        plt.ylabel("Stable cells")
        plt.xlabel("Spatial bins")
        plt.show()

        remapping_pv_stable = []

        for pre, post in zip(pop_vec_initial_stable, pop_vec_last_stable):

            remapping_pv_stable.append(pearsonr(pre.flatten(), post.flatten())[0])

        remapping_pv_stable = np.nan_to_num(np.array(remapping_pv_stable))
        # remapping_pv_stable = np.array(remapping_pv_stable)



        if plot_results:
            cmap = plt.cm.get_cmap('jet')
            # plotting
            for r, x, y in zip(remapping_pv_stable, common_loc[0], common_loc[1]):
                plt.scatter(x,y,color=cmap(r))
            a = plt.colorbar(cm.ScalarMappable(cmap=cmap))
            a.set_label("PV Pearson R")

            for g_l in self.goal_locations:
                plt.scatter((g_l[0]-self.x_min)/spatial_resolution, (g_l[1]-self.y_min)/spatial_resolution,
                            color="w",marker="x" )
            plt.title("Stable cells (mean="+str(np.round(np.mean(remapping_pv_stable),4))+")")
            plt.show()

        pop_vec_initial_dec = pop_vec_initial[:, dec_cells]
        pop_vec_last_dec = pop_vec_last[:, dec_cells]

        if adjust_pv_size:
            print("Adjusting PV size ...")

            # check if there are more dec cells than stable
            if nr_dec_cells < nr_stable_cells:
                raise Exception("Cannot adjust PV size: there are less decreasing cells than stable ones")

            remapping_pv_dec = []
            for i in range(nr_shuffles):
                # pick random cells from population vector
                rand_cells = np.random.randint(low=0, high=pop_vec_initial_dec.shape[1], size=nr_stable_cells)
                pop_vec_pre_dec_sub = pop_vec_initial_dec[:, rand_cells]
                pop_vec_post_dec_sub = pop_vec_last_dec[:, rand_cells]

                remapping_pv_dec_sub = np.zeros(pop_vec_pre_dec_sub.shape[0])
                for i, (pre, post) in enumerate(zip(pop_vec_pre_dec_sub, pop_vec_post_dec_sub)):
                    remapping_pv_dec_sub[i] = pearsonr(pre.flatten(), post.flatten())[0]
                remapping_pv_dec.append(np.nan_to_num(remapping_pv_dec_sub))

            remapping_pv_dec = np.mean(np.vstack(remapping_pv_dec), axis=0)

        else:

            remapping_pv_dec = []

            for pre, post in zip(pop_vec_initial_dec, pop_vec_last_dec):
                remapping_pv_dec.append(pearsonr(pre.flatten(), post.flatten())[0])

            remapping_pv_dec = np.nan_to_num(np.array(remapping_pv_dec))

        plt.imshow(pop_vec_initial_dec.T, interpolation='none', aspect='auto')
        plt.title("First 5 trials")
        plt.ylabel("Decreasing cells")
        plt.xlabel("Spatial bins")
        plt.show()

        plt.imshow(pop_vec_last_dec.T, interpolation='none', aspect='auto')
        plt.title("Last 5 trials")
        plt.ylabel("Decreasing cells")
        plt.xlabel("Spatial bins")
        plt.show()

        if plot_results:
            cmap = plt.cm.get_cmap('jet')
            # plotting
            for r, x, y in zip(remapping_pv_dec, common_loc[0], common_loc[1]):
                plt.scatter(x,y,color=cmap(r))
            a = plt.colorbar(cm.ScalarMappable(cmap=cmap))
            a.set_label("PV Pearson R")

            for g_l in self.goal_locations:
                plt.scatter((g_l[0]-self.x_min)/spatial_resolution, (g_l[1]-self.y_min)/spatial_resolution,
                            color="w",marker="x" )
            plt.title("Decreasing cells (mean="+str(np.round(np.mean(remapping_pv_dec),4))+")")
            plt.show()

        remapping_pv_stable_shuffle = []
        for i in range(nr_shuffles):
            shuffle_res = []
            per_ind = np.random.permutation(np.arange(pop_vec_last_stable.shape[0]))
            shuffled_pop_vec_post = pop_vec_last_stable[per_ind, :]
            for pre, post in zip(pop_vec_initial_stable, shuffled_pop_vec_post):
                shuffle_res.append(pearsonr(pre.flatten(), post.flatten())[0])
            remapping_pv_stable_shuffle.append(shuffle_res)

        remapping_pv_stable_shuffle = np.array(remapping_pv_stable_shuffle)
        remapping_pv_stable_shuffle_flat = remapping_pv_stable_shuffle.flatten()

        remapping_pv_stable_sorted = np.sort(remapping_pv_stable)
        remapping_pv_stable_shuffle_sorted = np.sort(remapping_pv_stable_shuffle_flat)

        # compute statistics
        # _, p = ks_2samp(remapping_pv_stable, remapping_pv_stable_shuffle_flat)

        # dec cells

        remapping_pv_dec_shuffle = []
        for i in range(nr_shuffles):
            shuffle_res = []
            per_ind = np.random.permutation(np.arange(pop_vec_last_dec.shape[0]))
            shuffled_pop_vec_post = pop_vec_last_dec[per_ind, :]
            for pre, post in zip(pop_vec_initial_dec, shuffled_pop_vec_post):
                shuffle_res.append(pearsonr(pre.flatten(), post.flatten())[0])
            remapping_pv_dec_shuffle.append(shuffle_res)

        remapping_pv_dec_shuffle = np.array(remapping_pv_dec_shuffle)
        remapping_pv_dec_shuffle_flat = remapping_pv_dec_shuffle.flatten()

        remapping_pv_dec_sorted = np.sort(remapping_pv_dec)
        remapping_pv_dec_shuffle_sorted = np.sort(remapping_pv_dec_shuffle_flat)

        # --------------------------------------------------------------------------------------------------------------
        # compute remapping (correlation of rate maps early learning vs. late learning)
        remapping = []

        for early, late in zip(initial.T, last.T):
            if np.count_nonzero(early) > 0 and np.count_nonzero(late) > 0:
                remapping.append(pearsonr(early.flatten(), late.flatten())[0])
            else:
                remapping.append(np.nan)

        remapping = np.array(remapping)
        # compute shuffled data
        remapping_shuffle = []
        for pre, post in zip(initial.T, last.T):
            shuffle_list = []
            post_flat = post.flatten()
            for i in range(nr_shuffles):
                if np.count_nonzero(pre) > 0 and np.count_nonzero(post) > 0:
                    np.random.shuffle(post_flat)
                    shuffle_list.append(pearsonr(pre.flatten(), post_flat)[0])
                else:
                    shuffle_list.append(0)
            remapping_shuffle.append(shuffle_list)
        remapping_shuffle = np.vstack(remapping_shuffle)

        remapping_stable = remapping[stable_cells]
        remapping_stable = remapping_stable[remapping_stable != np.nan]
        remapping_shuffle_stable = remapping_shuffle[stable_cells, :]

        remapping_dec = remapping[dec_cells]
        remapping_dec = remapping_dec[remapping_dec != np.nan]
        remapping_shuffle_dec = remapping_shuffle[dec_cells, :]

        # check how many cells did not remapped
        const = 0
        for data, control in zip(remapping_stable, remapping_shuffle_stable):
            # if data is 2 std above the mean of control --> no significant remapping
            if data > np.mean(control) + 2 * np.std(control):
                const += 1

        percent_stable_place = np.round(const / nr_stable_cells * 100, 2)

        if plot_results:

            stable_cell_remap_sorted = np.sort(remapping_stable)
            stable_cell_remap_shuffle_sorted = np.sort(remapping_shuffle_stable.flatten())

            dec_cell_remap_sorted = np.sort(remapping_dec)
            dec_cell_remap_shuffle_sorted = np.sort(remapping_shuffle_dec.flatten())

            # plot on cell level
            p_stable_cell_data = 1. * np.arange(stable_cell_remap_sorted.shape[0]) / (stable_cell_remap_sorted.shape[0] - 1)
            p_stable_cell_shuffle = 1. * np.arange(stable_cell_remap_shuffle_sorted.shape[0]) / (stable_cell_remap_shuffle_sorted.shape[0] - 1)

            p_dec_cell_data = 1. * np.arange(dec_cell_remap_sorted.shape[0]) / (dec_cell_remap_sorted.shape[0] - 1)
            p_dec_cell_shuffle = 1. * np.arange(dec_cell_remap_shuffle_sorted.shape[0]) / (dec_cell_remap_shuffle_sorted.shape[0] - 1)

            plt.plot(stable_cell_remap_sorted, p_stable_cell_data, label="Stable", color="magenta")
            plt.plot(stable_cell_remap_shuffle_sorted, p_stable_cell_shuffle, label="Stable shuffle", color="darkmagenta", linestyle="dashed")
            plt.plot(dec_cell_remap_sorted, p_dec_cell_data, label="Dec", color="aquamarine")
            plt.plot(dec_cell_remap_shuffle_sorted, p_dec_cell_shuffle, label="Dec shuffle", color="lightseagreen", linestyle="dashed")
            plt.legend()
            plt.ylabel("CDF")
            plt.xlabel("PEARSON R")
            plt.title("Per cell")
            plt.show()

            # plot on population vector level
            p_pv_stable = 1. * np.arange(remapping_pv_stable_sorted.shape[0]) / (remapping_pv_stable_sorted.shape[0] - 1)
            p_pv_stable_shuffle = 1. * np.arange(remapping_pv_stable_shuffle_sorted.shape[0]) / (remapping_pv_stable_shuffle_sorted.shape[0] - 1)

            p_pv_dec = 1. * np.arange(remapping_pv_dec_sorted.shape[0]) / (remapping_pv_dec_sorted.shape[0] - 1)
            p_pv_dec_shuffle = 1. * np.arange(remapping_pv_dec_shuffle_sorted.shape[0]) / (remapping_pv_dec_shuffle_sorted.shape[0] - 1)

            plt.plot(remapping_pv_stable_sorted, p_pv_stable, label="Stable", color="magenta")
            plt.plot(remapping_pv_stable_shuffle_sorted, p_pv_stable_shuffle, label="Stable shuffle",
                     color="darkmagenta", linestyle="dashed")

            plt.plot(remapping_pv_dec_sorted, p_pv_dec, label="Dec",  color="aquamarine")
            plt.plot(remapping_pv_dec_shuffle_sorted, p_pv_dec_shuffle, label="Dec shuffle",
                     color="lightseagreen", linestyle="dashed")
            plt.legend()
            plt.ylabel("CDF")
            plt.xlabel("PEARSON R")
            plt.title("Per pop. vec.")
            plt.show()

        else:

            return remapping_stable, remapping_shuffle_stable, remapping_dec, remapping_shuffle_dec, \
                   remapping_pv_stable, remapping_pv_stable_shuffle, remapping_pv_dec, remapping_pv_dec_shuffle

    def learning_pv_corr_temporal(self, spatial_resolution=2, average_trials=False):

        # check how many trials are there
        nr_trials = len(self.trial_loc_list)

        # how many trials to skip or average
        nr_trials_in_between = 5

        nr_data = floor(nr_trials/nr_trials_in_between)

        rate_maps = []

        occ_maps = []

        for i in range(nr_data):
            if average_trials:
                new_map = self.get_rate_maps(trials_to_use=range(i*nr_trials_in_between,(i+1)*nr_trials_in_between),
                                             spatial_resolution=spatial_resolution)
                occ_map = self.get_occ_map(trials_to_use=range(i*nr_trials_in_between,(i+1)*nr_trials_in_between),
                                             spatial_resolution=spatial_resolution)

            else:
                new_map = self.get_rate_maps(trials_to_use=range(i*nr_trials_in_between, i*nr_trials_in_between+1),
                                             spatial_resolution=spatial_resolution)
                occ_map = self.get_occ_map(trials_to_use=range(i*nr_trials_in_between, i*nr_trials_in_between+1),
                                             spatial_resolution=spatial_resolution)
            rate_maps.append(new_map)
            occ_maps.append(occ_map)

        # load cell labelaverage_trials=Falses
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        pv_corr_stable_mean = []
        pv_corr_stable_std = []
        pv_corr_dec_mean = []
        pv_corr_dec_std = []
        # go trough all maps
        for i_map in range(len(rate_maps)-1):
            # get population vectors
            pvs_first = np.reshape(rate_maps[i_map], (rate_maps[i_map].shape[0]*rate_maps[i_map].shape[1],
                                   rate_maps[i_map].shape[2]))
            pvs_second = np.reshape(rate_maps[i_map+1], (rate_maps[i_map].shape[0]*rate_maps[i_map].shape[1],
                                   rate_maps[i_map].shape[2]))

            occ_map_first = occ_maps[i_map]
            occ_map_second = occ_maps[i_map+1]
            comb_occ_map = np.logical_and(occ_map_first.flatten() > 0, occ_map_second.flatten() > 0)

            # only use spatial bins that were visited on both trials
            pvs_first = pvs_first[comb_occ_map,:]
            pvs_second = pvs_second[comb_occ_map, :]

            # first compute for stable cells
            pvs_first_stable = pvs_first[:, stable_cells]
            pvs_second_stable = pvs_second[:, stable_cells]
            pv_corr = []
            for pv_first, pv_second in zip(pvs_first_stable, pvs_second_stable):
                if np.count_nonzero(pv_first) > 0 and np.count_nonzero(pv_second) > 0:
                    pv_corr.append(pearsonr(pv_first, pv_second)[0])
                else:
                    continue
            pv_corr_stable_mean.append(np.mean(np.array(pv_corr)))
            pv_corr_stable_std.append(np.std(np.array(pv_corr)))

            # compute for dec cells
            pvs_first_dec = pvs_first[:, dec_cells]
            pvs_second_dec = pvs_second[:, dec_cells]
            pv_corr = []
            for pv_first, pv_second in zip(pvs_first_dec, pvs_second_dec):
                if np.count_nonzero(pv_first) > 0 and np.count_nonzero(pv_second) > 0:
                    pv_corr.append(pearsonr(pv_first, pv_second)[0])
                else:
                    continue
            pv_corr_dec_mean.append(np.mean(np.array(pv_corr)))
            pv_corr_dec_std.append(np.std(np.array(pv_corr)))


        pv_corr_dec_mean = np.array(pv_corr_dec_mean)
        pv_corr_stable_mean = np.array(pv_corr_stable_mean)
        pv_corr_dec_std = np.array(pv_corr_dec_std)
        pv_corr_stable_std = np.array(pv_corr_stable_std)

        plt.errorbar(x=np.arange(pv_corr_stable_mean.shape[0]), y=pv_corr_stable_mean, yerr=pv_corr_stable_std,
                     color="magenta", label="stable", capsize=2)
        plt.errorbar(x=np.arange(pv_corr_stable_mean.shape[0]), y=pv_corr_dec_mean, yerr=pv_corr_dec_std,
                     color="turquoise", label="decreasing", capsize=2, alpha=0.8)
        plt.ylabel("Mean PV correlations")
        plt.xlabel("Comparison ID (time)")
        plt.legend()
        plt.show()

    def learning_rate_map_corr_temporal(self, spatial_resolution=2, average_trials=False, instances_to_compare=None,
                                        plotting=True, save_fig=False, nr_trials_in_between=5):

        # check how many trials are there
        nr_trials = len(self.trial_loc_list)

        nr_data = floor(nr_trials/nr_trials_in_between)

        rate_maps = []
        block_labels = []

        for i in range(nr_data):
            if average_trials:
                new_map = self.get_rate_maps(trials_to_use=range(i*nr_trials_in_between,(i+1)*nr_trials_in_between),
                                             spatial_resolution=spatial_resolution)
                block_labels.append(str(i*nr_trials_in_between)+"-"+str((i+1)*nr_trials_in_between))
            else:
                new_map = self.get_rate_maps(trials_to_use=range(i*nr_trials_in_between, i*nr_trials_in_between+1),
                                             spatial_resolution=spatial_resolution)
            rate_maps.append(new_map)

        if instances_to_compare is None:
            # compare all instances (in time)
            time_steps = np.arange(len(rate_maps))

        else:
            if instances_to_compare > (len(rate_maps)):
                raise Exception("To many instances defined - choose smaller number")
            time_steps = np.linspace(0,len(rate_maps)-1, instances_to_compare, endpoint=True).astype(int)

        shift_between = []
        # go trough all instances
        for i_map in range(time_steps.shape[0]-1):
            map_first = rate_maps[time_steps[i_map]]
            map_second = rate_maps[time_steps[i_map+1]]
            remapping = []
            # go trough all rate maps per cell
            for map_init, map_last in zip(map_first.T, map_second.T):

                if np.count_nonzero(map_init) > 0 and np.count_nonzero(map_last) > 0:
                    # plt.subplot(1,2,1)
                    # plt.imshow(map_init)
                    # plt.subplot(1,2,2)
                    # plt.imshow(map_last)
                    # plt.title(pearsonr(map_init.flatten(), map_last.flatten())[0])
                    # plt.show()
                    remapping.append(pearsonr(map_init.flatten(), map_last.flatten())[0])
                else:
                    remapping.append(0)

            shift_between.append(remapping)

        shift_between = np.array(shift_between)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        stable_mean = []
        stable_std = []
        stable_nr_obs = []
        dec_mean = []
        dec_std = []
        dec_nr_obs = []

        # compute max for stable and dec (to normalized between 0 and 1)
        max_stable = np.max(shift_between[:,stable_cells])
        max_dec = np.max(shift_between[:,dec_cells])

        for i, current_shift in enumerate(shift_between):
            stable_mean.append(np.mean(current_shift[stable_cells]))
            dec_mean.append(np.mean(current_shift[dec_cells]))
            stable_std.append(np.std(current_shift[stable_cells]))
            dec_std.append(np.std(current_shift[dec_cells]))
            stable_nr_obs.append(current_shift[stable_cells].shape[0])
            dec_nr_obs.append(current_shift[dec_cells].shape[0])

        stable_mean = np.array(stable_mean)
        stable_std = np.array(stable_std)

        stable_mean_norm = (stable_mean - np.min(stable_mean))/np.max(stable_mean - np.min(stable_mean))
        stable_std_norm = stable_std / np.max(stable_mean - np.min(stable_mean))

        dec_mean_norm = (dec_mean - np.min(dec_mean))/np.max(dec_mean - np.min(dec_mean))
        dec_std_norm = dec_std / np.max(dec_mean - np.min(dec_mean))

        stats = []

        for s_mean, s_std, d_mean, d_std, s_obs, d_obs in zip(stable_mean_norm, stable_std_norm, dec_mean_norm, dec_std_norm,
                                                stable_nr_obs, dec_nr_obs):
            stats.append(np.round(ttest_ind_from_stats(mean1=s_mean, std1=s_std, mean2=d_mean, std2=d_std,
                                                       nobs1=s_obs, nobs2=d_obs)[1],2))

        m = stable_mean.shape[0]

        if plotting or save_fig:
            plt.errorbar(x=np.arange(len(stable_mean)), y=stable_mean,yerr=stable_std, color="magenta", label="stable",
                         capsize=2)
            plt.errorbar(x=np.arange(len(dec_mean)), y=dec_mean,yerr=dec_std, color="turquoise", label="decreasing",
                         capsize=2, alpha=0.8)
            plt.ylabel("Rate map correlations")
            plt.xlabel("Comparison ID (time)")
            plt.legend()
            plt.show()
            print(str(stats)+", alpha="+str(np.round(0.05/m,3)))
            # build x labels
            x_labels = []
            for i in range(len(block_labels)-1):
                x_labels.append(block_labels[i]+"\nvs\n"+block_labels[i+1])
            if save_fig:
                plt.style.use('default')
            plt.errorbar(x=np.arange(len(stable_mean_norm)), y=stable_mean_norm,yerr=stable_std_norm, color="magenta", label="stable",
                         capsize=2)
            plt.errorbar(x=np.arange(len(dec_mean_norm)), y=dec_mean_norm,yerr=dec_std_norm, color="turquoise", label="decreasing",
                         capsize=2, alpha=0.8)
            plt.ylabel("Rate map correlations (normalized)")
            plt.xticks(range(len(block_labels)),x_labels)
            plt.xlabel("Trials compared")
            plt.ylim(-1.5,3.5)
            plt.xlim(-0.3, len(stable_mean_norm)-0.7)
            # write n.s. above all
            for i, (m, s) in enumerate(zip(stable_mean_norm, stable_std_norm)):
                plt.text(i-0.1, m+s+0.2,"n.s.")

            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("learning_rate_map_correlations.svg", transparent="True")
            else:
                plt.show()

        else:
            return shift_between
    def learning_rate_map_corr_with_first_trial_temporal(self, spatial_resolution=2, average_trials=False,
                                        plotting=True, save_fig=False):

        # check how many trials are there
        nr_trials = len(self.trial_loc_list)

        rate_maps = []
        block_labels = []

        for i in range(nr_trials):
            new_map = self.get_rate_maps(trials_to_use=range(i,(i+1)),spatial_resolution=spatial_resolution)
            rate_maps.append(new_map)
        shift_between = []
        map_first = rate_maps[0]
        # compare all rate maps with first
        for map_to_compare in rate_maps[1:]:
            remapping = []
            # go trough all rate maps per cell
            for map_init, map_last in zip(map_first.T, map_to_compare.T):

                if np.count_nonzero(map_init) > 0 and np.count_nonzero(map_last) > 0:
                    # plt.subplot(1,2,1)
                    # plt.imshow(map_init)
                    # plt.subplot(1,2,2)
                    # plt.imshow(map_last)
                    # plt.title(pearsonr(map_init.flatten(), map_last.flatten())[0])
                    # plt.show()
                    remapping.append(pearsonr(map_init.flatten(), map_last.flatten())[0])
                else:
                    remapping.append(0)

            shift_between.append(remapping)

        shift_between = np.array(shift_between)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        if self.session_name == "mjc163R4R_0114":
            # delete first dec cell --> has correlation of always 1
            dec_cells = dec_cells[1:]

        stable_mean = []
        stable_std = []
        stable_nr_obs = []
        dec_mean = []
        dec_std = []
        dec_nr_obs = []

        # compute max for stable and dec (to normalized between 0 and 1)
        max_stable = np.max(shift_between[:,stable_cells])
        max_dec = np.max(shift_between[:,dec_cells])

        for i, current_shift in enumerate(shift_between):
            stable_mean.append(np.mean(current_shift[stable_cells]))
            dec_mean.append(np.mean(current_shift[dec_cells]))
            stable_std.append(np.std(current_shift[stable_cells]))
            dec_std.append(np.std(current_shift[dec_cells]))
            stable_nr_obs.append(current_shift[stable_cells].shape[0])
            dec_nr_obs.append(current_shift[dec_cells].shape[0])

        stable_mean = np.array(stable_mean)
        stable_std = np.array(stable_std)

        stable_mean_norm = (stable_mean - np.min(stable_mean))/np.max(stable_mean - np.min(stable_mean))
        stable_std_norm = stable_std / np.max(stable_mean - np.min(stable_mean))

        dec_mean_norm = (dec_mean - np.min(dec_mean))/np.max(dec_mean - np.min(dec_mean))
        dec_std_norm = dec_std / np.max(dec_mean - np.min(dec_mean))

        stats = []

        for s_mean, s_std, d_mean, d_std, s_obs, d_obs in zip(stable_mean_norm, stable_std_norm, dec_mean_norm, dec_std_norm,
                                                stable_nr_obs, dec_nr_obs):
            stats.append(np.round(ttest_ind_from_stats(mean1=s_mean, std1=s_std, mean2=d_mean, std2=d_std,
                                                       nobs1=s_obs, nobs2=d_obs)[1],2))

        m = stable_mean.shape[0]

        if plotting or save_fig:
            plt.errorbar(x=np.arange(len(stable_mean)), y=stable_mean,yerr=stable_std, color="magenta", label="stable",
                         capsize=2)
            plt.errorbar(x=np.arange(len(dec_mean)), y=dec_mean,yerr=dec_std, color="turquoise", label="decreasing",
                         capsize=2, alpha=0.8)
            plt.ylabel("Rate map correlations")
            plt.xlabel("Comparison ID (time)")
            plt.legend()
            plt.show()
            print(str(stats)+", alpha="+str(np.round(0.05/m,3)))
            # build x labels
            x_labels = []
            for i in range(len(block_labels)-1):
                x_labels.append(block_labels[i]+"\nvs\n"+block_labels[i+1])
            if save_fig:
                plt.style.use('default')
            plt.errorbar(x=np.arange(len(stable_mean_norm)), y=stable_mean_norm,yerr=stable_std_norm, color="magenta", label="stable",
                         capsize=2)
            plt.errorbar(x=np.arange(len(dec_mean_norm)), y=dec_mean_norm,yerr=dec_std_norm, color="turquoise", label="decreasing",
                         capsize=2, alpha=0.8)
            plt.ylabel("Rate map correlations (normalized)")
            plt.xticks(range(len(block_labels)),x_labels)
            plt.xlabel("Trials compared")
            plt.ylim(-1.5,3.5)
            plt.xlim(-0.3, len(stable_mean_norm)-0.7)
            # write n.s. above all
            for i, (m, s) in enumerate(zip(stable_mean_norm, stable_std_norm)):
                plt.text(i-0.1, m+s+0.2,"n.s.")

            plt.legend()
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("learning_rate_map_correlations.svg", transparent="True")
            else:
                plt.show()

        else:
            return shift_between

    def learning_place_field_peak_shift_temporal(self, plotting=True, spatial_resolution=3, average_trials=False):

        # check how many trials are there
        nr_trials = len(self.trial_loc_list)

        # how many trials to skip or average
        nr_trials_in_between = 5

        nr_data = floor(nr_trials/nr_trials_in_between)

        rate_maps = []

        for i in range(nr_data):
            if average_trials:
                new_map = self.get_rate_maps(trials_to_use=range(i*nr_trials_in_between,(i+1)*nr_trials_in_between),
                                             spatial_resolution=spatial_resolution)
            else:
                new_map = self.get_rate_maps(trials_to_use=range(i*nr_trials_in_between, i*nr_trials_in_between+1),
                                             spatial_resolution=spatial_resolution)
            rate_maps.append(new_map)

        shift_between = []
        # go trough all cells and compute shift in peak
        for i_map in range(len(rate_maps)-1):
            map_first = rate_maps[i_map]
            map_second = rate_maps[i_map+1]
            shift = []
            for map_init, map_last in zip(map_first.T, map_second.T):
                # get peak during first trials
                peak_loc_init = np.unravel_index(map_init.argmax(), map_init.shape)
                peak_loc_y_init = peak_loc_init[0]
                peak_loc_x_init = peak_loc_init[1]
                # get peak during later trials
                peak_loc_last = np.unravel_index(map_last.argmax(), map_last.shape)
                peak_loc_y_last = peak_loc_last[0]
                peak_loc_x_last = peak_loc_last[1]

                distance = np.sqrt((peak_loc_x_init - peak_loc_x_last) ** 2 + (peak_loc_y_init - peak_loc_y_last) ** 2)
                distance_cm = distance / spatial_resolution
                shift.append(distance_cm)

            shift_between.append(shift)

        shift_between = np.array(shift_between)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()

        stable_mean = []
        dec_mean = []
        stable_std = []
        dec_std = []
        for i, current_shift in enumerate(shift_between):
            stable_mean.append(np.mean(current_shift[stable_cells]))
            dec_mean.append(np.mean(current_shift[dec_cells]))
            stable_std.append(np.std(current_shift[stable_cells]))
            dec_std.append(np.std(current_shift[dec_cells]))

        plt.errorbar(x=np.arange(len(stable_mean)), y=stable_mean,yerr=stable_std, color="magenta", label="stable",
                     capsize=2)
        plt.errorbar(x=np.arange(len(dec_mean)), y=dec_mean,yerr=dec_std, color="turquoise", label="decreasing",
                     capsize=2, alpha=0.8)
        plt.ylabel("Place field shift")
        plt.xlabel("Comparison ID (time)")
        plt.legend()
        plt.show()

    def learning_place_field_peak_shift(self, plotting=True, spatial_resolution=3, all_cells=False):

        # get maps for first trial
        initial = self.get_rate_maps(trials_to_use=range(5), spatial_resolution=spatial_resolution)
        # get maps for last trial
        last = self.get_rate_maps(trials_to_use=range(len(self.trial_loc_list)-5, len(self.trial_loc_list)),
                                  spatial_resolution=spatial_resolution)

        shift = []
        # go trough all cells and compute shift in peak
        for map_init, map_last in zip(initial.T, last.T):
            # get peak during first trials
            peak_loc_init = np.unravel_index(map_init.argmax(), map_init.shape)
            peak_loc_y_init = peak_loc_init[0]
            peak_loc_x_init = peak_loc_init[1]
            # get peak during later trials
            peak_loc_last = np.unravel_index(map_last.argmax(), map_last.shape)
            peak_loc_y_last = peak_loc_last[0]
            peak_loc_x_last = peak_loc_last[1]

            distance = np.sqrt((peak_loc_x_init - peak_loc_x_last) ** 2 + (peak_loc_y_init - peak_loc_y_last) ** 2)
            distance_cm = distance / spatial_resolution
            shift.append(distance_cm)

        shift = np.array(shift)

        if all_cells:
            if plotting:
                # plot on population vector level
                shift_sorted = np.sort(shift)
                p = 1. * np.arange(shift.shape[0]) / (shift.shape[0] - 1)

                plt.plot(shift_sorted, p, label="All cells", color="aquamarine")
                plt.legend()
                plt.ylabel("CDF")
                plt.xlabel("Place field peak shift / cm")
                plt.title("Place field peak shift during learning")
                plt.show()
            else:
                return shift

        else:

            # load cell labels
            with open(self.params.pre_proc_dir + "cell_classification/" +
                      self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
                class_dic = pickle.load(f)

            stable_cells = class_dic["stable_cell_ids"].flatten()
            dec_cells = class_dic["decrease_cell_ids"].flatten()
            nr_stable_cells = stable_cells.shape[0]
            nr_dec_cells = dec_cells.shape[0]

            shift_stable = shift[stable_cells]
            shift_stable_sorted = np.sort(shift_stable)

            shift_dec = shift[dec_cells]
            shift_dec_sorted = np.sort(shift_dec)

            if plotting:

                # plot on population vector level
                p_stable = 1. * np.arange(shift_stable.shape[0]) / (shift_stable.shape[0] - 1)

                p_dec = 1. * np.arange(shift_dec.shape[0]) / (shift_dec.shape[0] - 1)

                plt.plot(shift_stable_sorted, p_stable, label="Stable", color="magenta")

                plt.plot(shift_dec_sorted, p_dec, label="Dec",  color="aquamarine")
                plt.legend()
                plt.ylabel("CDF")
                plt.xlabel("Place field peak shift / cm")
                plt.title("Place field peak shift during learning")
                plt.show()

            else:

                return shift_stable, shift_dec

    def learning_mean_firing_rate(self, plotting=False, absolute_value=False):
        """
        Computes the relative change in mean firing rate between first and last 5 trials of learning

        @return: diff, relative difference in firing from first 5 trials and last 5 trials
        """
        # get maps for first trial
        initial, _, _ = self.get_raster_location_speed(trials_to_use=range(5))
        # get maps for last trial
        last, _, _ = self.get_raster_location_speed(trials_to_use=range(len(self.trial_loc_list)-5,
                                                                        len(self.trial_loc_list)))

        initial_mean = np.mean(initial, axis=1)
        last_mean = np.mean(last, axis=1)

        diff = (last_mean - initial_mean)/ (last_mean + initial_mean)

        if absolute_value:
            diff = np.abs(diff)

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()

        cell_ids_dec = class_dic["decrease_cell_ids"].flatten()

        diff_stable = diff[cell_ids_stable]
        diff_dec = diff[cell_ids_dec]

        if plotting:

            diff_stable_sorted = np.sort(diff_stable)
            diff_dec_sorted = np.sort(diff_dec)

            p_diff_stable = 1. * np.arange(diff_stable.shape[0]) / (diff_stable.shape[0] - 1)

            p_diff_dec = 1. * np.arange(diff_dec.shape[0]) / (diff_dec.shape[0] - 1)

            plt.plot(diff_stable_sorted, p_diff_stable, label="stable")
            plt.plot(diff_dec_sorted, p_diff_dec, label="dec")
            plt.legend()
            plt.show()

        else:

            return diff_stable, diff_dec

    def learning_rate_map_corr(self, spatial_resolution=3):

        # get maps for first trial
        initial = self.get_rate_maps(trials_to_use=range(5), spatial_resolution=spatial_resolution)
        # get maps for last trial
        last = self.get_rate_maps(trials_to_use=range(len(self.trial_loc_list) - 5, len(self.trial_loc_list)),
                                  spatial_resolution=spatial_resolution)

        remapping = []

        for pre, post in zip(initial.T, last.T):
            if np.count_nonzero(pre) > 0 and np.count_nonzero(post) > 0:
                remapping.append(pearsonr(pre.flatten(), post.flatten())[0])
            else:
                remapping.append(0)

        remapping = np.array(remapping)
        return remapping

    def map_heterogenity(self, nr_shuffles=500, plot_results=True,
                              spatial_resolution=3, adjust_pv_size=False, plotting=True, metric="cosine"):

        # get maps for first trial
        initial = self.get_rate_maps(trials_to_use=range(5), spatial_resolution=spatial_resolution)
        initial_occ = self.get_occ_map(trials_to_use=range(5), spatial_resolution=spatial_resolution)
        # get maps for last trial
        last = self.get_rate_maps(trials_to_use=range(len(self.trial_loc_list)-5, len(self.trial_loc_list)),
                                  spatial_resolution=spatial_resolution)
        last_occ = self.get_occ_map(trials_to_use=range(len(self.trial_loc_list)-5, len(self.trial_loc_list)),
                                    spatial_resolution=spatial_resolution)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()
        nr_stable_cells = stable_cells.shape[0]
        nr_dec_cells = dec_cells.shape[0]

        # compute remapping based on population vectors per bin
        # --------------------------------------------------------------------------------------------------------------

        pop_vec_initial = np.reshape(initial,
                                 (initial.shape[0] * initial.shape[1], initial.shape[2]))
        pop_vec_last = np.reshape(last, (last.shape[0] * last.shape[1], last.shape[2]))

        # only select spatial bins that were visited in PRE and POST
        comb_occ_map = np.logical_and(initial_occ.flatten() > 0, last_occ.flatten() > 0)
        pop_vec_initial = pop_vec_initial[comb_occ_map, :]
        pop_vec_last = pop_vec_last[comb_occ_map, :]

        comb_occ_map_spatial = np.reshape(comb_occ_map,(initial.shape[0], initial.shape[1]))
        common_loc = np.where(comb_occ_map_spatial)
        # stable cells

        pop_vec_pre_stable = pop_vec_initial[:, stable_cells]
        pop_vec_post_stable = pop_vec_last[:, stable_cells]

        pop_vec_pre_dec = pop_vec_initial[:, dec_cells]
        pop_vec_post_dec = pop_vec_last[:, dec_cells]

        initial_pop_vec_sim_stable = distance.pdist(pop_vec_pre_stable, metric=metric)
        initial_pop_vec_sim_dec = distance.pdist(pop_vec_pre_dec, metric=metric)
        late_pop_vec_sim_stable = distance.pdist(pop_vec_post_stable, metric=metric)
        late_pop_vec_sim_dec = distance.pdist(pop_vec_post_dec, metric=metric)

        if plotting:

            initial_pop_vec_sim_stable_sorted = np.sort(initial_pop_vec_sim_stable)
            initial_pop_vec_sim_dec_sorted = np.sort(initial_pop_vec_sim_dec)

            p_init_stable = 1. * np.arange(initial_pop_vec_sim_stable.shape[0]) / (initial_pop_vec_sim_stable.shape[0] - 1)
            p_init_dec = 1. * np.arange(initial_pop_vec_sim_dec.shape[0]) / (initial_pop_vec_sim_dec.shape[0] - 1)

            plt.plot(initial_pop_vec_sim_stable_sorted, p_init_stable, label="stable")
            plt.plot(initial_pop_vec_sim_dec_sorted, p_init_dec, label="dec")
            plt.legend()
            plt.title("Before learning")
            plt.show()

            late_pop_vec_sim_stable_sorted = np.sort(late_pop_vec_sim_stable)
            late_pop_vec_sim_dec_sorted = np.sort(late_pop_vec_sim_dec)

            p_late_stable = 1. * np.arange(late_pop_vec_sim_stable.shape[0]) / (late_pop_vec_sim_stable.shape[0] - 1)
            p_late_dec = 1. * np.arange(late_pop_vec_sim_dec.shape[0]) / (late_pop_vec_sim_dec.shape[0] - 1)

            plt.plot(late_pop_vec_sim_stable_sorted, p_late_stable, label="stable")
            plt.plot(late_pop_vec_sim_dec_sorted, p_late_dec, label="dec")
            plt.legend()
            plt.title("After learning")
            plt.show()


            plt.plot(initial_pop_vec_sim_stable_sorted, p_init_stable, label="before")
            plt.plot(late_pop_vec_sim_stable_sorted, p_late_stable, label="after")
            plt.legend()
            plt.title("Stable cells: change through learning")
            plt.show()

            plt.plot(initial_pop_vec_sim_dec_sorted, p_init_dec, label="before")
            plt.plot(late_pop_vec_sim_dec_sorted, p_late_dec, label="after")
            plt.legend()
            plt.title("Dec cells: change through learning")
            plt.show()
        else:
            return initial_pop_vec_sim_stable, initial_pop_vec_sim_dec, late_pop_vec_sim_stable, late_pop_vec_sim_dec

    def learning_phmm_modes_activation(self, trials_to_use_for_decoding=None, cells_to_use="stable"):
        if cells_to_use == "stable":
            # get phmm model to decode awake
            pre_phmm_model = self.session_params.default_pre_phmm_model_stable
        elif cells_to_use == "decreasing":
            # get phmm model to decode awake
            pre_phmm_model = self.session_params.default_pre_phmm_model_dec
        # decode awake activity
        seq, _, post_prob = self.decode_poisson_hmm(file_name=pre_phmm_model, cells_to_use=cells_to_use,
                                                      trials_to_use=trials_to_use_for_decoding)

        return post_prob, seq

    """#################################################################################################################
    #  goal coding
    #################################################################################################################"""

    def collective_goal_coding(self, cells_to_use="all", spatial_resolution=3, max_radius=10, ring_width=1):
        # load all rate maps
        rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution)

        # load cell labels
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle",
                  "rb") as f:
            class_dic = pickle.load(f)

        if cells_to_use == "stable":

            cell_ids = class_dic["stable_cell_ids"].flatten()

        elif cells_to_use == "increasing":

            cell_ids = class_dic["increase_cell_ids"].flatten()

        elif cells_to_use == "decreasing":

            cell_ids = class_dic["decrease_cell_ids"].flatten()

        elif cells_to_use == "all":

            cell_ids = np.arange(rate_maps.shape[2])

        # normalize rate maps
        max_per_cell = np.max(np.reshape(rate_maps, (rate_maps.shape[0]*rate_maps.shape[1], rate_maps.shape[2])), axis=0)
        max_per_cell[max_per_cell == 0] = 1e-22
        norm_rate_maps = rate_maps / max_per_cell

        # compute summed up rate map
        sum_rate_map = np.sum(norm_rate_maps[:, :, cell_ids], axis=2)

        # mask with occupancy
        occ = self.get_occ_map(spatial_resolution=spatial_resolution)
        sum_rate_map[occ==0] = np.nan

        sum_rate_map = sum_rate_map / np.sum(np.nan_to_num(sum_rate_map.flatten()))

        plt.imshow(sum_rate_map.T)
        a = plt.colorbar()
        a.set_label("Sum firing rate / normalized to 1")

        all_goals = collective_goal_coding(normalized_rate_map=sum_rate_map, goal_locations=self.goal_locations,
                               env_x_min=self.x_min, env_y_min=self.y_min, spatial_resolution=spatial_resolution,
                                           max_radius=max_radius, ring_width=ring_width)

        all_goals_arr = np.vstack(all_goals)
        for i, one_goal in enumerate(all_goals):
            plt.plot(np.arange(0, max_radius*spatial_resolution, ring_width*spatial_resolution), one_goal, label="Goal " + str(i))
        plt.plot(np.arange(0, max_radius*spatial_resolution, ring_width*spatial_resolution), np.mean(all_goals_arr, axis=0), c="w", label="Mean", linewidth=3)
        plt.xlabel("Distance to goal / cm")
        plt.ylabel("Mean density")
        plt.title(cells_to_use)
        plt.legend()
        plt.show()
        # exit()
        # plt.rcParams['svg.fonttype'] = 'none'
        # plt.savefig("dec_firing_changes.svg", transparent="True")

    def phmm_mode_occurrence(self, n_smoothing=1000):

        phmm_file = self.session_params.default_pre_phmm_model

        all_trials = range(len(self.trial_loc_list))

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in all_trials:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))

        state_sequence, nr_modes, post_prob = self.decode_poisson_hmm(file_name=phmm_file,
                                                           trials_to_use=all_trials)

        # get occurences
        modes, mode_occurrences = np.unique(state_sequence, return_counts=True)

        occurrences = np.zeros(post_prob.shape[1])
        occurrences[modes] = mode_occurrences

        smooth_post_prob = []
        m = []
        # compute probabilites in moving window
        for mode_post_prob in post_prob.T:
            mode_post_prob_smooth = moving_average(a=mode_post_prob, n=n_smoothing)
            mode_post_prob_smooth_norm = mode_post_prob_smooth/np.max(mode_post_prob_smooth)
            smooth_post_prob.append(mode_post_prob_smooth_norm)
            coef = np.polyfit(np.linspace(0,1,mode_post_prob_smooth_norm.shape[0]), mode_post_prob_smooth_norm, 1)
            m.append(coef[0])
            # plt.plot(mode_post_prob_smooth_norm)
            # poly1d_fn = np.poly1d(coef)
            # plt.plot(np.arange(mode_post_prob_smooth_norm.shape[0]),
            # poly1d_fn(np.linspace(0,1,mode_post_prob_smooth_norm.shape[0])), '--w')
            # plt.title(coef[0])
            # plt.show()

        m = np.array(m)

        return m, occurrences

    def place_field_goal_distance(self, trials_to_use=None, plotting=True, mean_firing_threshold=None):

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # get maps for first trial
        rate_maps = self.get_rate_maps(trials_to_use=trials_to_use, spatial_resolution=1)
        raster = self.get_raster()
        mean_rates = np.mean(raster, axis=1)/self.params.time_bin_size

        # compute distances
        distances = distance_peak_firing_to_closest_goal(rate_maps, goal_locations=self.goal_locations,
                                                         env_x_min=self.x_min, env_y_min=self.y_min)

        # load cell labelaverage_trials=Falses
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name + "_" + self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        stable_cells = class_dic["stable_cell_ids"].flatten()
        dec_cells = class_dic["decrease_cell_ids"].flatten()
        inc_cells = class_dic["increase_cell_ids"].flatten()

        dist_stable = distances[stable_cells]
        dist_dec = distances[dec_cells]
        dist_inc = distances[inc_cells]

        if mean_firing_threshold is not None:
            mean_stable = mean_rates[stable_cells]
            mean_dec = mean_rates[dec_cells]
            mean_inc = mean_rates[inc_cells]

            dist_stable = dist_stable[mean_stable > mean_firing_threshold]
            dist_dec = dist_dec[mean_dec > mean_firing_threshold]
            dist_inc = dist_inc[mean_inc > mean_firing_threshold]

        if plotting:
            c = "white"
            plt.figure(figsize=(4, 5))
            res = [dist_stable, dist_dec, dist_inc]
            bplot = plt.boxplot(res, positions=[1, 2, 3], patch_artist=True,
                                labels=["Stable", "Decreasing", "Increasing"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c), showfliers=False
                                )
            colors = ["magenta", 'turquoise']
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylabel("Distance to closest goal")
            plt.grid(color="grey", axis="y")
            plt.show()

        else:
            return dist_stable, dist_dec, dist_inc

    """#################################################################################################################
    #  glm
    #################################################################################################################"""

    def infer_glm_awake(self, trials_to_use=None, nr_gauss=20, std_gauss=10, spatial_bin_size_map=None,
                        plot_for_control=False):
        """
        infers glm (paper: Correlations and Functional Connections in a Population of Grid Cells, 2015) from awake
        data

        @param trials_to_use: which trials to use to infer glm
        @type trials_to_use: range
        @param nr_gauss: how many Gaussians to distribute in environment
        @type nr_gauss: int
        @param std_gauss: which standard deviation to use for all Gaussians
        @type std_gauss: int
        @param spatial_bin_size_map: size of a spatial bin in cm
        @type spatial_bin_size_map: int
        @param plot_for_control: if True --> plot single steps of generating maps
        @type plot_for_control: bool
        """

        # check if time bin size < 20 ms --> needed for binary assumption
        if self.params.time_bin_size != 0.01:
            raise Exception("TIME BIN SIZE MUST BE 10 MS!")

        print(" - INFERENCE GLM USING CHEESEBOARD DATA ...\n")

        if trials_to_use is None:
            trials_to_use = self.default_trials

        # params
        # --------------------------------------------------------------------------------------------------------------
        learning_rates = [100, 10, 1, 0.1, 0.01]
        likelihood = np.zeros(len(learning_rates))
        max_likelihood_per_iteration = []
        max_iter = 250
        cell_to_plot = 5

        if spatial_bin_size_map is None:
            spatial_bin_size_map = self.params.spatial_resolution

        file_name = self.session_name+"_"+self.experiment_phase_id+"_"+\
                    str(spatial_bin_size_map)+"cm_bins"+"_"+self.cell_type+ "_trials_"+str(trials_to_use[0])+\
                      "_"+str(trials_to_use[-1])

        # place Gaussians uniformly across environment
        # --------------------------------------------------------------------------------------------------------------

        # get dimensions of environment
        x_min, x_max, y_min, y_max = self.x_min, self.x_max, self.y_min, self.y_max

        # get size of environment
        x_span = x_max - x_min
        y_span = y_max - y_min

        # width - length ratio of environment
        w_l_ratio = y_span/x_span

        # tile x and y with centers of Gaussians
        centers_gauss_x = np.linspace(x_min, x_max, nr_gauss)
        centers_gauss_y = np.linspace(y_min, y_max, int(np.round(nr_gauss*w_l_ratio)))

        # compute grid with x and y values of centers
        centers_gauss_x, centers_gauss_y = np.meshgrid(centers_gauss_x, centers_gauss_y)

        # get data used to infer model --> concatenate single trial data
        # --------------------------------------------------------------------------------------------------------------

        # check how many cells
        nr_cells = self.trial_raster_list[0].shape[0]
        raster = np.empty((nr_cells, 0))
        loc = np.empty((0,2))

        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))

        # data and parameter preparation for optimization
        # --------------------------------------------------------------------------------------------------------------

        # split location data into x and y coordinates
        x_loc = loc[:, 0]
        y_loc = loc[:, 1]

        # make binary raster --> if cell fires --> 1, if cell doesn't fire --> -1 --> ISING MODEL!
        bin_raster = -1 * np.ones((raster.shape[0], raster.shape[1]))
        bin_raster[raster > 0] = 1
        bin_raster = bin_raster.T

        # x_loc_m = loadmat("matlab.mat")["posx"]
        # y_loc_m = loadmat("matlab.mat")["posy"]

        # how many time bins / cells
        nr_time_bins = bin_raster.shape[0]-1
        nr_cells = bin_raster.shape[1]

        # compute distance from center of each Gaussian for every time bin
        dist_to_center_x = matlib.repmat(np.expand_dims(centers_gauss_x.flatten("F"), 1), 1, x_loc.shape[0] - 1) - \
                   matlib.repmat(x_loc[:-1].T, centers_gauss_x.flatten().shape[0], 1)
        dist_to_center_y = matlib.repmat(np.expand_dims(centers_gauss_y.flatten("F"), 1), 1, y_loc.shape[0] - 1) - \
                   matlib.repmat(y_loc[:-1].T, centers_gauss_y.flatten().shape[0], 1)

        # compute values of each Gaussian for each time bin
        gauss_values = simple_gaussian(xd=dist_to_center_x, yd=dist_to_center_y, std=std_gauss)

        # optimization of alpha values --> maximize data likelihood
        # --------------------------------------------------------------------------------------------------------------

        # alpha --> weights for Gaussians [#gaussians, #cells]
        alpha = np.zeros((gauss_values.shape[0], bin_raster.shape[1]))

        # bias --> firing rates of neurons (constant across time points!!!)
        bias = matlib.repmat(np.random.rand(nr_cells, 1), 1, nr_time_bins)

        for iter in range(max_iter):
            # compute gradient for alpha values --> dLikeli/dalpha
            dalpha=((gauss_values @ bin_raster[1:, :]) -
                    (np.tanh(alpha.T @ gauss_values + bias) @ gauss_values.T).T)/nr_time_bins

            # compute change in cost
            dcost = np.sum(bin_raster[1:, :].T - np.tanh(alpha.T @ gauss_values+bias), axis=1)/nr_time_bins

            # try different learning rates to maximize likelihood
            for i, l_r in enumerate(learning_rates):
                # compute new alpha values with gradient and learning rate
                alpha_n = alpha + l_r * dalpha
                # compute cost using old cost and update
                bias_n = bias + matlib.repmat(l_r*np.expand_dims(dcost, 1), 1, nr_time_bins)

                likelihood[i] = np.trace((alpha_n.T @ gauss_values + bias_n) @ bin_raster[1:, :])-np.sum(
                    np.sum(np.log(2*np.cosh(alpha_n.T @ gauss_values + bias_n)), axis=1))

            max_likelihood = np.max(likelihood)
            max_likelihood_per_iteration.append(max_likelihood)
            best_learning_rate_index = np.argmax(likelihood)

            # update bias --> optimize the bias term first before optimizing alpha values
            bias = bias + matlib.repmat(learning_rates[best_learning_rate_index]*
                                        np.expand_dims(dcost, 1), 1, nr_time_bins)

            # only start optimizing alpha values after n iterations
            if iter > 50:

                alpha = alpha + learning_rates[best_learning_rate_index] * dalpha

        # generation of maps for spatial bin size defined
        # --------------------------------------------------------------------------------------------------------------

        # if spatial_bin_size_map was not provided --> use spatial_resolution from parameter file
        if spatial_bin_size_map is None:
            spatial_bin_size_map = self.params.spatial_resolution

        nr_spatial_bins = int(np.round(x_span / spatial_bin_size_map))

        # generate grid by spatial binning
        x_map = np.linspace(x_min, x_max, nr_spatial_bins)
        y_map = np.linspace(y_min, y_max, int(np.round(nr_spatial_bins * w_l_ratio)))

        # compute grid from x and y values
        x_map, y_map = np.meshgrid(x_map, y_map)

        dist_to_center_x_map = matlib.repmat(np.expand_dims(centers_gauss_x.flatten("F"), 1), 1, x_map.flatten().shape[0])\
                               -matlib.repmat(x_map.flatten("F"), centers_gauss_x.flatten().shape[0], 1)
        dist_to_center_y_map = matlib.repmat(np.expand_dims(centers_gauss_y.flatten("F"), 1), 1, y_map.flatten().shape[0])\
                               -matlib.repmat(y_map.flatten("F"), centers_gauss_y.flatten().shape[0], 1)

        # compute values of each Gaussian for each time bin
        gauss_values_map = simple_gaussian(xd=dist_to_center_x_map, yd=dist_to_center_y_map, std=std_gauss)

        # compute resulting map
        res_map = np.exp(alpha.T @ gauss_values_map + matlib.repmat(bias[:, 0], gauss_values_map.shape[1], 1).T)/ \
                   (2*np.cosh(alpha.T @ gauss_values_map + matlib.repmat(bias[:, 0], gauss_values_map.shape[1], 1).T))

        # reshape to reconstruct 2D map
        res_map = matlib.reshape(res_map, (res_map.shape[0], nr_spatial_bins, int(np.round(nr_spatial_bins * w_l_ratio))))

        # compute occupancy --> mask results (remove non visited bins)
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # Fede's implementation:
        # ----------------------
        # centers_x = np.linspace(x_min, x_max + 0.1, nr_spatial_bins)
        # centers_y = np.linspace(y_min, y_max + 0.1, int(round(nr_spatial_bins*w_l_ratio)))
        #
        # dx = centers_x[1] - centers_x[0]
        # dy = centers_y[1] - centers_y[0]
        #
        # occ = np.zeros((centers_x.shape[0], centers_y.shape[0]))
        #
        # x_loc[x_loc > x_max] = x_min - 0.01
        # y_loc[y_loc > y_max] = y_min - 0.01
        #
        # for i in range(x_loc.shape[0]):
        #     xi = int(np.floor((x_loc[i]-x_min)/dx))+1
        #     yi = int(np.floor((y_loc[i] - y_min) / dy)) + 1
        #     if xi*yi > 0:
        #         occ[xi, yi] += 1
        #
        # occ_mask_fede = np.where(occ > 0, 1, 0)
        # occ_mask = np.repeat(occ_mask[np.newaxis, :, :], nr_cells, axis=0)
        # --------------------------------------------------------------------------------------------------------------

        # compute actual rate maps from used data --> to validate results
        rate_map, occ = self.rate_map_from_data(loc=loc, raster=raster, spatial_resolution=spatial_bin_size_map)

        # compute binary occupancy map --> used as a mask to filter non-visited bins
        occ_mask = np.where(occ > 0, 1, 0)
        occ_mask_plot = np.where(occ > 0, 1, np.nan)
        occ_mask = np.repeat(occ_mask[np.newaxis, :, :], nr_cells, axis=0)

        # save original map before applying the mask for plotting
        res_orig = res_map

        # apply mask to results
        res_map = np.multiply(res_map, occ_mask)

        if plot_for_control:

            plt.plot(max_likelihood_per_iteration)
            plt.title("LIKELIHOOD PER ITERATION")
            plt.xlabel("ITERATION")
            plt.ylabel("LIKELIHOOD")
            plt.show()

            # compute actual rate maps from used data --> to validate results
            rate_map_to_plot = np.multiply(rate_map[:, :, cell_to_plot], occ_mask_plot)
            plt.imshow(rate_map_to_plot.T, origin="lower")
            plt.scatter((centers_gauss_x-x_min)/spatial_bin_size_map, (centers_gauss_y-y_min)/spatial_bin_size_map
                        , s=0.1, label="GAUSS. CENTERS")
            plt.title("RATE MAP + GAUSSIAN CENTERS")
            a = plt.colorbar()
            a.set_label("FIRING RATE / 1/s")
            plt.legend()
            plt.show()

            a = alpha.T @ gauss_values_map
            a = matlib.reshape(a, (a.shape[0], nr_spatial_bins, int(np.round(nr_spatial_bins * w_l_ratio))))
            plt.imshow(a[cell_to_plot, :, :].T, interpolation='nearest', aspect='auto', origin="lower")
            plt.colorbar()
            plt.title("ALPHA.T @ GAUSSIANS")
            plt.show()

            plt.imshow(res_orig[cell_to_plot, :, :].T, origin="lower")
            plt.title("RES_MAP ORIGINAL (W/O OCC. MASK)")
            a = plt.colorbar()
            a.set_label("PROB. OF FIRING IN WINDOW (" +str(self.params.time_bin_size)+"s")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            plt.imshow(occ.T, origin="lower")
            plt.title("OCC MAP")
            a = plt.colorbar()
            a.set_label("SEC")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            plt.imshow(occ_mask[cell_to_plot, :, :].T, origin="lower")
            plt.title("OCC MAP BINARY")
            a = plt.colorbar()
            a.set_label("OCC: YES/NO")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

            plt.imshow(res_map[cell_to_plot, :, :].T, origin="lower")
            plt.title("RES_MAP MASKED")
            a = plt.colorbar()
            a.set_label("PROB. OF FIRING IN WINDOW (" +str(self.params.time_bin_size)+"s)")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

        model_dic = {
            "rate_map": rate_map,
            "occ_map": occ,
            "occ_mask_plot": occ_mask_plot,
            "res_map": res_map,
            "alpha": alpha,
            "bias": bias,
            "centers_gauss_x": centers_gauss_x,
            "centers_gauss_y": centers_gauss_y,
            "std_gauss": std_gauss,
            "likelihood": max_likelihood_per_iteration,
            "time_bin_size": self.params.time_bin_size
        }

        with open(self.params.pre_proc_dir+'awake_ising_maps/' + file_name + '.pkl', 'wb') as f:
            pickle.dump(model_dic, f, pickle.HIGHEST_PROTOCOL)

    def load_glm_awake(self, model_name=None, cell_id=None, plotting=False):

        if model_name is None:
            model_name = self.default_ising
        with open(self.params.pre_proc_dir + 'awake_ising_maps/' + model_name + '.pkl', 'rb') as f:
            model_dic = pickle.load(f)

        centers_gauss_x = model_dic["centers_gauss_x"]
        centers_gauss_y = model_dic["centers_gauss_y"]
        std_gauss = model_dic["std_gauss"]
        alpha = model_dic["alpha"]
        rate_map = model_dic["rate_map"]
        res_map = model_dic["res_map"]
        occ_mask_plot = model_dic["occ_mask_plot"]
        time_bin_size = model_dic["time_bin_size"]

        if plotting:
            # compute actual rate maps from used data --> to validate results
            rate_map_to_plot = np.multiply(rate_map[:, :, cell_id], occ_mask_plot)

            plt.imshow(rate_map_to_plot.T, origin="lower")

            plt.title("RATE MAP")
            a = plt.colorbar()
            a.set_label("FIRING RATE / Hz")
            plt.show()


            # print res map
            plt.imshow(res_map[cell_id, :, :].T, origin='lower', interpolation='nearest', aspect='auto')
            plt.title("RES MAP")
            plt.xlabel("X")
            plt.ylabel("Y")
            a = plt.colorbar()
            a.set_label("PROB. OF FIRING IN WINDOW (" + str(time_bin_size) + "s)")
            plt.show()
        else:
            return res_map

    """#################################################################################################################
    #  poisson hmm
    #################################################################################################################"""

    def analyze_modes_goal_coding(self, file_name, mode_ids, thr_close_to_goal=25, plotting=False):

        trials_to_use = self.default_trials

        # get location data from trials
        nr_cells = self.trial_raster_list[0].shape[0]
        loc = np.empty((0,2))
        raster = np.empty((nr_cells, 0))
        for trial_id in trials_to_use:
            raster = np.hstack((raster, self.trial_raster_list[trial_id]))
            loc = np.vstack((loc, self.trial_loc_list[trial_id]))

        state_sequence, nr_modes, _ = self.decode_poisson_hmm(file_name=file_name,
                                                           trials_to_use=trials_to_use)
        # only use data from mode ids provided
        mode_data = loc[np.isin(state_sequence, mode_ids), :]

        # compute distance from each goal
        close_to_goal = []
        for g_l in self.goal_locations:
            dist_g_l = np.linalg.norm(mode_data - g_l, axis=1)
            close_to_goal.append(dist_g_l < thr_close_to_goal)

        close_to_goal = np.array(close_to_goal)
        close_to_goal = np.logical_or.reduce(close_to_goal, axis=0)

        if close_to_goal.shape[0] > 0:
            # compute fraction of points that lie within border of goal location
            frac_close_to_goal = np.count_nonzero(close_to_goal) / close_to_goal.shape[0]
        else:
            frac_close_to_goal = 0

        if plotting:
            for g_l in self.goal_locations:
                plt.scatter(g_l[0], g_l[1], color="w")
            plt.scatter(mode_data[:,0], mode_data[:,1], color="gray", s=1, label="ALL LOCATIONS")
            plt.scatter(mode_data[close_to_goal,0], mode_data[close_to_goal,1], color="red", s=1, label="CLOSE TO GOAL")
            plt.title("LOCATIONS OF SELECTED MODES (GOAL RADIUS: "+str(thr_close_to_goal)+"cm)\n "
                      "FRACTION CLOSE TO GOAL: "+str(np.round(frac_close_to_goal,2)))
            plt.legend()
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

        return frac_close_to_goal

    """#################################################################################################################
    #  location decoding analysis
    #################################################################################################################"""

    def compare_decoding_location_phmm_bayesian(self):

        # decoding using phmm
        # --------------------------------------------------------------------------------------------------------------
        trials_to_use = self.default_trials
        model = self.load_poisson_hmm(file_name=self.default_phmm)
                # get spatial information from pHMM model
        means, _, _, _, _ = self.fit_spatial_gaussians_for_modes(file_name=self.default_phmm, plot_awake_fit=False)

        mean_err = []
        median_err = []
        phmm_all_err = []

        for trial_to_decode in trials_to_use:
            raster = self.trial_raster_list[trial_to_decode]
            loc = self.trial_loc_list[trial_to_decode]
            prob = model.predict_proba(raster.T)
            err = []
            for i, (current_prob, current_loc) in enumerate(zip(prob, loc)):
                # compute decoded location using weighted average
                dec_location = np.average(means, weights=current_prob, axis=1)
                err.append(np.linalg.norm(dec_location - current_loc))
                phmm_all_err.append(np.linalg.norm(dec_location - current_loc))

            mean_err.append(np.mean(np.array(err)))
            median_err.append(np.median(np.array(err)))

        plt.plot(trials_to_use, mean_err, label="MEAN")
        plt.plot(trials_to_use, median_err, label="MEDIAN")
        plt.ylabel("ERROR / cm")
        plt.xlabel("TRIAL ID")
        # plt.ylim(10,40)
        plt.legend()
        plt.show()

        # plot cdf
        phmm_all_err = np.array(phmm_all_err)
        phmm_all_error_sorted = np.sort(phmm_all_err)
        p_phmm = 1. * np.arange(phmm_all_err.shape[0]) / (phmm_all_err.shape[0] - 1)
        plt.plot(phmm_all_error_sorted, p_phmm, color="#ffdba1", label="pHMM")


        # decoding using Bayesian
        # --------------------------------------------------------------------------------------------------------------
        bayes_all_err = []
        rate_maps = self.get_rate_maps(spatial_resolution=1)
        # flatten rate maps
        rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0] * rate_maps.shape[1], rate_maps.shape[2]))
        for trial_to_decode in trials_to_use:
            raster = self.trial_raster_list[trial_to_decode]
            loc = self.trial_loc_list[trial_to_decode]
            for pop_vec, loc in zip(raster.T, loc):
                if np.count_nonzero(pop_vec) == 0:
                    continue
                bl = bayes_likelihood(frm=rate_maps_flat.T, pop_vec=pop_vec, log_likeli=False)
                bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
                pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
                pred_x = pred_bin[0] + self.x_min
                pred_y = pred_bin[1] + self.y_min

                # plt.scatter(pred_x, pred_y, color="red")
                # plt.scatter(loc[0], loc[1], color="gray")
                bayes_all_err.append(np.sqrt((pred_x - loc[0]) ** 2 + (pred_y - loc[1]) ** 2))

        bayes_all_err = np.array(bayes_all_err)
        bayes_all_error_sorted = np.sort(bayes_all_err)
        p_bayes = 1. * np.arange(bayes_all_err.shape[0]) / (bayes_all_err.shape[0] - 1)
        plt.plot(bayes_all_error_sorted, p_bayes, color="blue", label="bayes")
        plt.legend()
        plt.xlabel("Error (cm)")
        plt.ylabel("CDF")
        plt.show()

    def decode_location_during_learning(self, cells_to_use="stable", trial_window=3, restrict_to_goals=False,
                                        plot_for_control=False):

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        if cells_to_use == "stable":
            cell_ids = class_dic["stable_cell_ids"].flatten()
        if cells_to_use == "decreasing":
            cell_ids = class_dic["decrease_cell_ids"].flatten()

        test_trials = range(len(self.trial_loc_list)-trial_window, len(self.trial_loc_list))
        test_raster, test_loc, _ = self.get_raster_location_speed(trials_to_use=test_trials)
        test_raster = test_raster[cell_ids, :]

        if restrict_to_goals:
            radius = 20
            close_to_all_goals = np.zeros(test_loc.shape[0]).astype(bool)
            for g_l in self.goal_locations:
                dist_to_goal = np.sqrt((test_loc[:, 0] - g_l[0]) ** 2 + (test_loc[:, 1] - g_l[1]) ** 2)
                close_to_goal = dist_to_goal < radius
                close_to_all_goals = np.logical_or(close_to_all_goals, close_to_goal)
            if plot_for_control:
                plt.scatter(test_loc[:, 0], test_loc[:, 1])
                plt.scatter(test_loc[close_to_all_goals,0], test_loc[close_to_all_goals,1], color="r")
                plt.scatter(g_l[0], g_l[1], marker="x")
                plt.show()
            test_raster = test_raster[:,close_to_all_goals]
            test_loc = test_loc[close_to_all_goals,:]

        nr_windows = np.round((len(self.trial_loc_list)-trial_window)/trial_window).astype(int)

        col = plt.cm.get_cmap("jet")

        for window_id in range(nr_windows):
            train_trials = range(window_id*trial_window, (window_id+1)*trial_window)

            # get rate map --> need to add x_min, y_min of environment to have proper location info
            rate_maps = self.get_rate_maps(spatial_resolution=1, trials_to_use=train_trials)

            # flatten rate maps
            rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0] * rate_maps.shape[1], rate_maps.shape[2]))

            rate_maps_flat = rate_maps_flat[:, cell_ids]

            error = []
            for pop_vec, loc in zip(test_raster.T, test_loc):
                if np.count_nonzero(pop_vec) == 0:
                    continue
                bl = bayes_likelihood(frm=rate_maps_flat.T, pop_vec=pop_vec, log_likeli=False)
                bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
                pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
                pred_x = pred_bin[0] + self.x_min
                pred_y = pred_bin[1] + self.y_min



                # plt.scatter(pred_x, pred_y, color="red")
                # plt.scatter(loc[0], loc[1], color="gray")
                error.append(np.sqrt((pred_x - loc[0]) ** 2 + (pred_y - loc[1]) ** 2))

            error_sorted = np.sort(error)
            p_error = 1. * np.arange(error_sorted.shape[0]) / (error_sorted.shape[0] - 1)
            plt.plot(error_sorted, p_error, label="WINDOW "+str(window_id), c=col(window_id/nr_windows))
            # plt.gca().set_xscale("log")
        plt.ylabel("CDF")
        plt.xlabel("Error (cm)")
        if restrict_to_goals:
            plt.title("Location decoding around goals during learning: " + cells_to_use + " cells")
        else:
            plt.title("Location decoding during learning: "+cells_to_use+" cells")
        plt.legend()
        plt.show()

    def decode_location_beginning_end_learning(self, cells_to_use="stable", trial_window=5, restrict_to_goals=False,
                                        plotting=False):

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        if cells_to_use == "stable":
            cell_ids = class_dic["stable_cell_ids"].flatten()
        if cells_to_use == "decreasing":
            cell_ids = class_dic["decrease_cell_ids"].flatten()

        test_trials = range(len(self.trial_loc_list)-trial_window, len(self.trial_loc_list))
        test_raster, test_loc, _ = self.get_raster_location_speed(trials_to_use=test_trials)
        test_raster = test_raster[cell_ids, :]

        train_trials = range(trial_window)

        # get rate map --> need to add x_min, y_min of environment to have proper location info
        rate_maps = self.get_rate_maps(spatial_resolution=1, trials_to_use=train_trials)

        # flatten rate maps
        rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0] * rate_maps.shape[1], rate_maps.shape[2]))

        rate_maps_flat = rate_maps_flat[:, cell_ids]

        error = []
        for pop_vec, loc in zip(test_raster.T, test_loc):
            if np.count_nonzero(pop_vec) == 0:
                continue
            bl = bayes_likelihood(frm=rate_maps_flat.T, pop_vec=pop_vec, log_likeli=False)
            bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
            pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
            pred_x = pred_bin[0] + self.x_min
            pred_y = pred_bin[1] + self.y_min

            error.append(np.sqrt((pred_x - loc[0]) ** 2 + (pred_y - loc[1]) ** 2))

        error_sorted = np.sort(error)

        if plotting:
            p_error = 1. * np.arange(error_sorted.shape[0]) / (error_sorted.shape[0] - 1)
            # plt.gca().set_xscale("log")
            plt.ylabel("CDF")
            plt.xlabel("Error (cm)")
            plt.plot(error_sorted, p_error)

            plt.title("Location decoding during learning: "+cells_to_use+" cells")
            plt.show()
        else:
            return error_sorted

    def decode_location_end_of_learning(self, cells_to_use="stable", nr_of_trials=10, plotting=False):

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        if cells_to_use == "stable":
            cell_ids = class_dic["stable_cell_ids"].flatten()
        if cells_to_use == "decreasing":
            cell_ids = class_dic["decrease_cell_ids"].flatten()

        test_train_trials = range(len(self.trial_loc_list)-nr_of_trials, len(self.trial_loc_list))
        raster, loc, _ = self.get_raster_location_speed(trials_to_use=test_train_trials)
        raster = raster[cell_ids, :]

        # get rate map --> need to add x_min, y_min of environment to have proper location info
        rate_maps = self.get_rate_maps(spatial_resolution=1, trials_to_use=test_train_trials)
        occ_map = self.get_occ_map(spatial_resolution=1, trials_to_use=test_train_trials)

        # flatten rate maps
        rate_maps_flat = np.reshape(rate_maps, (rate_maps.shape[0] * rate_maps.shape[1], rate_maps.shape[2]))

        rate_maps_flat = rate_maps_flat[:, cell_ids]

        error = []
        pred_loc = []
        for pop_vec, loc_curr in zip(raster.T, loc):
            if np.count_nonzero(pop_vec) == 0:
                continue
            bl = bayes_likelihood(frm=rate_maps_flat.T, pop_vec=pop_vec, log_likeli=False)
            bl_area = np.reshape(bl, (rate_maps.shape[0], rate_maps.shape[1]))
            pred_bin = np.unravel_index(bl_area.argmax(), bl_area.shape)
            pred_x = pred_bin[0]
            pred_y = pred_bin[1]
            true_x = loc_curr[0] - self.x_min
            true_y = loc_curr[1] - self.y_min
            if plotting:
                plt.imshow(bl_area.T)
                plt.scatter(loc[:,0] - self.x_min, loc[:,1] - self.y_min, s=1, alpha=0.5)
                plt.scatter(true_x, true_y, c="r")
                plt.scatter(pred_bin[0],pred_bin[1], edgecolors="r", facecolor="none", s=50)
                a = plt.colorbar()
                a.set_label("Likelihood")
                plt.title(np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2))
                plt.show()
            pred_loc.append([pred_x, pred_y])
            error.append(np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2))

        error = np.array(error)
        error_sorted = np.sort(error)
        p_error = 1. * np.arange(error.shape[0]) / (error.shape[0] - 1)

        if plotting:
            plt.plot(error_sorted, p_error)
            plt.xlabel("Error (cm)")
            plt.ylabel("CDF")
            plt.show()

        pred_loc_arr = np.vstack(pred_loc)
        pos, times_decoded = np.unique(pred_loc_arr, axis=0, return_counts=True)

        times_decoded_prob = times_decoded / times_decoded.sum()

        max_prob = times_decoded_prob.max()
        times_decoded_prob_norm = times_decoded_prob / max_prob

        if plotting:
            fig, ax = plt.subplots()
            col_map = plt.cm.get_cmap("jet")
            ax.scatter(pos[:,0], pos[:,1], color=col_map(times_decoded_prob_norm))
            for g_l in self.goal_locations:
                ax.scatter(g_l[0]-self.x_min, g_l[1]-self.y_min, marker="x", color="w")
            a = fig.colorbar(cm.ScalarMappable(cmap=col_map), ticks=[0,1])
            a.ax.set_yticklabels(["0", "{:.2e}".format(max_prob)])
            a.set_label("Decoding probability: "+cells_to_use)
            plt.show()

        spatial_resolution = 1

        # get size of environment
        x_span = self.x_max - self.x_min
        y_span = self.y_max - self.y_min

        # width - length ratio of environment
        w_l_ratio = y_span / x_span

        nr_spatial_bins = int(np.round(x_span / spatial_resolution))

        centers_x = np.linspace(self.x_min, self.x_max + 0.1, nr_spatial_bins)
        centers_y = np.linspace(self.y_min, self.y_max + 0.1, int(round(nr_spatial_bins * w_l_ratio)))

        dx = centers_x[1] - centers_x[0]
        dy = centers_y[1] - centers_y[0]

        # split location data into x and y coordinates
        x_loc = loc[:, 0]
        y_loc = loc[:, 1]

        x_loc[x_loc > self.x_max] = self.x_min - 0.01
        y_loc[y_loc > self.y_max] = self.y_min - 0.01

        location_decoded_matrix = np.zeros((centers_x.shape[0], centers_y.shape[0]))

        for loc in pred_loc:
            xi = int(np.floor((loc[0]) / dx)) + 1
            yi = int(np.floor((loc[1]) / dy)) + 1
            if xi * yi > 0:
                location_decoded_matrix[xi, yi] += 1

        location_decoded_matrix_prob = location_decoded_matrix / np.sum(location_decoded_matrix.flatten())
        if plotting:
            plt.imshow(location_decoded_matrix_prob.T)
            a = plt.colorbar()
            a.set_label("Decoding probability")
            for g_l in self.goal_locations:
                plt.scatter((g_l[0]-self.x_min)/spatial_resolution, (g_l[1]-self.y_min)/spatial_resolution, marker="x", color="w")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Decoding prob. "+cells_to_use+ " cells")
            plt.show()

        if not plotting:
            return error

    def decoding_error_stable_vs_decreasing(self, plotting=False, nr_of_trials=10, subset_range=[4,8,12,18],
                                            nr_subsets=10, cross_val=False):

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_dec = class_dic["decrease_cell_ids"].flatten()

        # use last n trials without cross-validation
        test_train_trials = np.arange(len(self.trial_loc_list)-nr_of_trials, len(self.trial_loc_list))
        if cross_val:
            test_train_trials_per = np.random.permutation(test_train_trials)
            trials_test = test_train_trials[:int(test_train_trials_per.shape[0]*0.5)]
            trials_train = test_train_trials[int(test_train_trials_per.shape[0] * 0.5):]
        else:
            trials_test = test_train_trials
            trials_train = test_train_trials

        decoding_error_stable = []
        decoding_error_dec = []

        for subset_size in subset_range:
            stable_subsets = []
            dec_subsets = []
            # generate subsets
            for n in range(nr_subsets):
                stable_subsets.append(np.random.choice(a=cell_ids_stable, size=subset_size, replace=False))
                dec_subsets.append(np.random.choice(a=cell_ids_dec, size=subset_size, replace=False))

            # compute in parallel for different subsets: first for stable cells
            with mp.Pool(nr_subsets) as p:
                multi_arg = partial(self.decode_location_bayes, trials_train=trials_train,
                                    trials_test=trials_test)
                decoding_error_stable_subset = p.map(multi_arg, stable_subsets)

            # compute in parallel for different subsets: decreasing cells
            with mp.Pool(nr_subsets) as p:
                multi_arg = partial(self.decode_location_bayes, trials_train=trials_train,
                                    trials_test=trials_test)
                decoding_error_dec_subset = p.map(multi_arg, dec_subsets)

            decoding_error_stable.append(decoding_error_stable_subset)
            decoding_error_dec.append(decoding_error_dec_subset)

        error_stable = [np.hstack(x) for x in decoding_error_stable]
        error_dec = [np.hstack(x) for x in decoding_error_dec]

        med_stable = [np.median(x) for x in error_stable]
        mad_stable = [median_absolute_deviation(x) for x in error_stable]

        med_dec = [np.median(x) for x in error_dec]
        mad_dec = [median_absolute_deviation(x) for x in error_dec]
        error_stable_all = np.hstack(error_stable)
        error_dec_all = np.hstack(error_dec)

        if plotting:
            p_stable_all = 1. * np.arange(error_stable_all.shape[0]) / (error_stable_all.shape[0] - 1)
            p_dec_all = 1. * np.arange(error_dec_all.shape[0]) / (error_dec_all.shape[0] - 1)
            plt.plot(np.sort(error_stable_all), p_stable_all, label="stable")
            plt.plot(np.sort(error_dec_all), p_dec_all, label="dec")
            plt.title("Error for all subsets\n p-val:" + str(mannwhitneyu(error_stable_all, error_dec_all)[1]))
            plt.xlabel("Error (cm)")
            plt.ylabel("cdf")
            plt.legend()
            plt.show()

            for i, (stable, dec) in enumerate(zip(error_stable, error_dec)):
                p_stable = 1. * np.arange(stable.shape[0]) / (stable.shape[0] - 1)
                p_dec = 1. * np.arange(dec.shape[0]) / (dec.shape[0] - 1)
                plt.plot(np.sort(stable), p_stable, label="stable")
                plt.plot(np.sort(dec), p_dec, label="dec")
                plt.title("#cells in subset: " + str(subset_range[i]) + "\n p-val:" + str(mannwhitneyu(stable, dec)[1]))
                plt.xlabel("Error (cm)")
                plt.ylabel("cdf")
                plt.legend()
                plt.show()

            plt.errorbar(x=np.array(subset_range)+0.1, y=med_stable, yerr=mad_stable, label="stable")
            plt.errorbar(x=subset_range, y=med_dec, yerr=mad_dec, label="dec")
            plt.legend()
            plt.show()
        else:
            return error_stable, error_dec

    """#################################################################################################################
    #  SVM analysis
    #################################################################################################################"""

    def distinguishing_goals(self, radius=10, plot_for_control=False, plotting=False):

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_decrease = class_dic["decrease_cell_ids"].flatten()
        cell_ids_increase = class_dic["increase_cell_ids"].flatten()

        # get rasters
        raster, loc, speed = self.get_raster_location_speed()

        # if cells_to_use == "random_subset":
        #     cell_ids = random.sample(range(raster.shape[0]), 20)

        all_goals = []
        # select population vectors around goals
        for i, g_l in enumerate(self.goal_locations):
            close_to_goal = []
            for pv, location_pv in zip(raster.T, loc):
                if norm(location_pv - g_l) < radius:
                    close_to_goal.append(pv)
                    if plot_for_control:
                        plt.scatter(location_pv[0], location_pv[1], color="gray")
            all_goals.append(close_to_goal)
            if plot_for_control:
                plt.scatter(g_l[0], g_l[1], label="Goal "+str(i))
        if plot_for_control:
            plt.legend()
            plt.show()

        all_goal_ids = [0, 1, 2, 3]
        multi_fits = []

        for fits in range(30):
            predictability = np.zeros((4, 4, 4))
            predictability[:] = np.nan

            cell_ids_stable_subset = np.random.choice(a=cell_ids_stable, size=10, replace=False)
            cell_ids_increase_subset = np.random.choice(a=cell_ids_increase, size=10, replace=False)
            cell_ids_decrease_subset = np.random.choice(a=cell_ids_decrease, size=10, replace=False)

            for pair in itertools.combinations(all_goal_ids, r=2):

                # compare two goals using SVM
                goal_1 = np.array(all_goals[pair[0]])
                goal_2 = np.array(all_goals[pair[1]])

                all_data = np.vstack((goal_1, goal_2))

                labels = np.zeros(all_data.shape[0])
                labels[:goal_1.shape[0]] = 1

                # permute data

                per_ind = np.random.permutation(np.arange(all_data.shape[0]))
                X_orig = all_data[per_ind,:]
                y = labels[per_ind]

                for i, cell_ids in enumerate([cell_ids_stable_subset, cell_ids_decrease_subset,
                                              cell_ids_increase_subset, None]):

                    if cell_ids is not None:
                        X = X_orig[:, cell_ids]
                    else:
                        X = X_orig

                    train_per = 0.7
                    X_train = X[:int(train_per*X.shape[0]),:]
                    X_test = X[int(train_per * X.shape[0]):, :]
                    y_train = y[:int(train_per*X.shape[0])]
                    y_test = y[int(train_per * X.shape[0]):]

                    clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto', kernel="linear"))
                    clf.fit(X_train, y_train)
                    predictability[i, pair[0], pair[1]] = clf.score(X_test, y_test)
            multi_fits.append(predictability)

        multi_fits = np.array(multi_fits)
        stable = np.nanmean(multi_fits[:, 0, :, :], axis=0)
        dec = np.nanmean(multi_fits[:, 1, :, :], axis=0)
        inc = np.nanmean(multi_fits[:, 2, :, :], axis=0)
        all = np.nanmean(multi_fits[:, 3, :, :], axis=0)

        if plotting:
            plt.imshow(stable, vmin=0, vmax=1)
            plt.xticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            plt.yticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            a = plt.colorbar()
            a.set_label("Mean accuracy SVM")
            plt.title("stable")
            plt.show()
            plt.imshow(dec, vmin=0, vmax=1)
            plt.xticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            plt.yticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            a = plt.colorbar()
            a.set_label("Mean accuracy SVM")
            plt.title("decreasing")
            plt.show()
            plt.imshow(inc, vmin=0, vmax=1)
            plt.xticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            plt.yticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            a = plt.colorbar()
            a.set_label("Mean accuracy SVM")
            plt.title("increasing")
            plt.show()

            plt.imshow(all, vmin=0, vmax=1)
            plt.xticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            plt.yticks([0, 1, 2, 3], ["Goal 0", "Goal 1", "Goal 2", "Goal 3"])
            a = plt.colorbar()
            a.set_label("Mean accuracy SVM")
            plt.title("all cells")
            plt.show()
        else:
            return stable, dec, inc, all

    def identify_single_goal(self, radius=10, plot_for_control=False, plotting=True):
        """
        identifies single goals using PV and multi-class SVM

        @param radius:
        @param plot_for_control:
        @param plotting:
        @return:
        """
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_decrease = class_dic["decrease_cell_ids"].flatten()
        cell_ids_increase = class_dic["increase_cell_ids"].flatten()

        # get rasters
        raster, loc, speed = self.get_raster_location_speed()

        # if cells_to_use == "random_subset":
        #     cell_ids = random.sample(range(raster.shape[0]), 20)

        all_goals = []
        # select population vectors around goals
        for i, g_l in enumerate(self.goal_locations):
            close_to_goal = []
            for pv, location_pv in zip(raster.T, loc):
                if norm(location_pv - g_l) < radius:
                    close_to_goal.append(pv)
                    if plot_for_control:
                        plt.scatter(location_pv[0], location_pv[1], color="gray")
            all_goals.append(close_to_goal)
            if plot_for_control:
                plt.scatter(g_l[0], g_l[1], label="Goal "+str(i))
        if plot_for_control:
            plt.legend()
            plt.show()

        data_len = [len(x) for x in all_goals]
        all_data = np.vstack(all_goals)
        labels = np.zeros(all_data.shape[0])
        start = 0
        for label, l in enumerate(data_len):
            labels[start:start+l] = label
            start = start+l

        nr_fits = 30
        mean_accuracy = np.zeros((nr_fits, 4))
        for fit_id, fits in enumerate(range(nr_fits)):


            per_ind = np.random.permutation(np.arange(all_data.shape[0]))
            X_orig = all_data[per_ind, :]
            y = labels[per_ind]

            for cell_sel_id, cell_ids in enumerate([cell_ids_stable, cell_ids_decrease, cell_ids_increase, None]):
                if cell_ids is None:
                    X = X_orig
                else:
                    X = X_orig[:, cell_ids]

                train_per = 0.7
                X_train = X[:int(train_per * X.shape[0]), :]
                X_test = X[int(train_per * X.shape[0]):, :]
                y_train = y[:int(train_per * X.shape[0])]
                y_test = y[int(train_per * X.shape[0]):]

                clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto', kernel="linear"))
                clf.fit(X_train, y_train)

                mean_accuracy[fit_id, cell_sel_id] = clf.score(X_test, y_test)

        mean_acc_stable = mean_accuracy[:,0]
        mean_acc_dec = mean_accuracy[:, 1]
        mean_acc_inc = mean_accuracy[:, 2]
        mean_acc_all = mean_accuracy[:, 3]

        if plotting:
            c="white"
            res = [mean_acc_stable, mean_acc_dec, mean_acc_inc, mean_acc_all]
            bplot = plt.boxplot(res, positions=[1,2,3,4], patch_artist=True,
                                labels=["Stable", "Dec", "Inc", "All"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c),showfliers=False)
            plt.ylabel("Mean accuracy SVM - Multi-class")
            plt.show()

        else:
            return mean_acc_stable, mean_acc_dec, mean_acc_inc, mean_acc_all

    def identify_single_goal_subsets(self, radius=10, plot_for_control=False, plotting=True, m_subset=5,
                                     nr_splits=5, nr_subsets=10):
        """
        identifies single goals using PV and multi-class SVM

        @param radius:
        @param plot_for_control:
        @param plotting:
        @return:
        """
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_decrease = class_dic["decrease_cell_ids"].flatten()
        cell_ids_increase = class_dic["increase_cell_ids"].flatten()

        # get rasters
        raster, loc, speed = self.get_raster_location_speed()

        all_goals = []
        # select population vectors around goals
        for i, g_l in enumerate(self.goal_locations):
            close_to_goal = []
            for pv, location_pv in zip(raster.T, loc):
                if norm(location_pv - g_l) < radius:
                    close_to_goal.append(pv)
                    if plot_for_control:
                        plt.scatter(location_pv[0], location_pv[1], color="gray")
            all_goals.append(close_to_goal)
            if plot_for_control:
                plt.scatter(g_l[0], g_l[1], label="Goal "+str(i))
        if plot_for_control:
            plt.legend()
            plt.show()

        data_len = [len(x) for x in all_goals]
        all_data = np.vstack(all_goals)
        labels = np.zeros(all_data.shape[0])
        start = 0
        for label, l in enumerate(data_len):
            labels[start:start+l] = label
            start = start+l

        mean_accuracy_stable = []
        mean_accuracy_dec = []
        mean_accuracy_inc = []

        for i in range(nr_splits):

            per_ind = np.random.permutation(np.arange(all_data.shape[0]))
            X_orig = all_data[per_ind, :]
            y = labels[per_ind]

            for cell_sel_id, cell_ids in enumerate([cell_ids_stable, cell_ids_decrease, cell_ids_increase, None]):
                if cell_ids is None:
                    X = X_orig
                else:
                    X = X_orig[:, cell_ids]

                train_per = 0.7
                X_train = X[:int(train_per * X.shape[0]), :]
                X_test = X[int(train_per * X.shape[0]):, :]
                y_train = y[:int(train_per * X.shape[0])]
                y_test = y[int(train_per * X.shape[0]):]

                res = MlMethodsOnePopulation().parallelize_svm(X_train=X_train, X_test=X_test, y_train=y_train,
                                                                y_test=y_test, m_subset=m_subset, nr_subsets=nr_subsets)

                if cell_sel_id == 0:
                    mean_accuracy_stable.append(res)
                elif cell_sel_id == 1:
                    mean_accuracy_dec.append(res)
                elif cell_sel_id == 2:
                    mean_accuracy_inc.append(res)

        mean_acc_stable = np.vstack(mean_accuracy_stable).flatten()
        mean_acc_dec = np.vstack(mean_accuracy_dec).flatten()
        mean_acc_inc = np.vstack(mean_accuracy_inc).flatten()

        if plotting:
            c="white"
            res = [mean_acc_stable, mean_acc_dec, mean_acc_inc]
            bplot = plt.boxplot(res, positions=[1,2,3], patch_artist=True,
                                labels=["Stable", "Dec", "Inc"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c),showfliers=False)
            plt.ylabel("Mean accuracy SVM - Multi-class")
            plt.show()
        else:
            return mean_acc_stable, mean_acc_dec, mean_acc_inc

    def identify_single_goal_multiple_subsets(self, radius=10, plotting=True,
                                              subset_range=[4, 8, 12, 18], nr_splits=10, nr_subsets=10):

        stable_mean = []
        stable_std = []
        dec_mean = []
        dec_std = []
        inc_mean = []
        inc_std = []
        stable = []
        dec = []
        inc = []

        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_decrease = class_dic["decrease_cell_ids"].flatten()
        cell_ids_increase = class_dic["increase_cell_ids"].flatten()

        raster = self.get_raster()
        # compute mean
        mean_fir = np.mean(raster, axis=1)/self.params.time_bin_size

        mean_fir_stable = mean_fir[cell_ids_stable]
        mean_fir_stable = mean_fir_stable[mean_fir_stable>1]
        print(mean_fir_stable.shape[0])

        for m_subset in subset_range:
            mean_acc_stable, mean_acc_dec, mean_acc_inc = self.identify_single_goal_subsets(plotting=False,
                                                                                            m_subset=m_subset,
                                                                                            radius=radius,
                                                                                            nr_splits=nr_splits,
                                                                                            nr_subsets=nr_subsets)
            stable_mean.append(np.mean(mean_acc_stable))
            stable_std.append(np.std(mean_acc_stable))
            dec_mean.append(np.mean(mean_acc_dec))
            dec_std.append(np.std(mean_acc_dec))
            inc_mean.append(np.mean(mean_acc_inc))
            inc_std.append(np.std(mean_acc_inc))
            stable.append(mean_acc_stable)
            dec.append(mean_acc_dec)
            inc.append(mean_acc_inc)

        if plotting:
            plt.errorbar(x=subset_range, y=stable_mean, yerr=stable_std, label="stable")
            plt.errorbar(x=subset_range, y=inc_mean, yerr=inc_std, label="inc")
            plt.errorbar(x=subset_range, y=dec_mean, yerr=dec_std, label="dec")
            plt.ylabel("Mean accuracy - multiclass SVM (mean,std)")
            plt.xlabel("#cells")
            plt.legend()
            plt.show()
        else:
            return stable, inc, dec

    def detect_goal_related_activity_using_subsets(self, radius=15, plot_for_control=False, plotting=True, m_subset=3,
                                     nr_splits=30):
        """
        identifies if animal is around any goal using SVM

        @param radius:
        @param plot_for_control:
        @param plotting:
        @return:
        """
        with open(self.params.pre_proc_dir + "cell_classification/" +
                  self.session_name +"_"+self.params.stable_cell_method + ".pickle", "rb") as f:
            class_dic = pickle.load(f)

        cell_ids_stable = class_dic["stable_cell_ids"].flatten()
        cell_ids_decrease = class_dic["decrease_cell_ids"].flatten()
        cell_ids_increase = class_dic["increase_cell_ids"].flatten()

        # get rasters
        raster, loc, speed = self.get_raster_location_speed()

        close_to_goal = []
        away_from_goal = []
        away_from_goal_loc = []
        close_to_goal_loc = []
        # select population vectors around goals
        for pv, location_pv in zip(raster.T, loc):
            close_to_goal_flag = False
            for g_l in self.goal_locations:
                if norm(location_pv - g_l) < radius:
                    close_to_goal_flag = True
            if close_to_goal_flag:
                close_to_goal.append(pv)
                close_to_goal_loc.append(location_pv)
            else:
                away_from_goal.append(pv)
                away_from_goal_loc.append(location_pv)

        if plot_for_control:
            close_to_goal_loc = np.array(close_to_goal_loc)
            plt.scatter(close_to_goal_loc[:,0],close_to_goal_loc[:,1], color="b", s=1)
            away_from_goal_loc = np.array(away_from_goal_loc)
            plt.scatter(away_from_goal_loc[:,0],away_from_goal_loc[:,1], color="r", s=1)
            plt.show()

        y = np.zeros(len(close_to_goal)+len(away_from_goal))
        y[:len(close_to_goal)] = 1
        X = np.vstack((close_to_goal, away_from_goal))

        per_ind = np.random.permutation(np.arange(X.shape[0]))
        X = X[per_ind, :]
        y = y[per_ind]

        mean_accuracy_stable = []
        mean_accuracy_dec = []
        mean_accuracy_inc = []

        sss = StratifiedShuffleSplit(n_splits=nr_splits, test_size=0.3, random_state=0)

        for train_index, test_index in sss.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for cell_sel_id, cell_ids in enumerate([cell_ids_stable, cell_ids_decrease, cell_ids_increase]):
                res = None

                # only use data from selected cells
                X_train_sel = X_train[:, cell_ids]
                X_test_sel = X_test[:, cell_ids]

                res = MlMethodsOnePopulation().parallelize_svm(X_train=X_train_sel, X_test=X_test_sel, y_train=y_train,
                                                                y_test=y_test, m_subset=m_subset)

                if cell_sel_id == 0:
                    mean_accuracy_stable.append(res)
                elif cell_sel_id == 1:
                    mean_accuracy_dec.append(res)
                elif cell_sel_id == 2:
                    mean_accuracy_inc.append(res)

        mean_acc_stable = np.vstack(mean_accuracy_stable).flatten()
        mean_acc_dec = np.vstack(mean_accuracy_dec).flatten()
        mean_acc_inc = np.vstack(mean_accuracy_inc).flatten()

        if plotting:
            c="white"
            res = [mean_acc_stable, mean_acc_dec, mean_acc_inc]
            bplot = plt.boxplot(res, positions=[1,2,3], patch_artist=True,
                                labels=["Stable", "Dec", "Inc"],
                                boxprops=dict(color=c),
                                capprops=dict(color=c),
                                whiskerprops=dict(color=c),
                                flierprops=dict(color=c, markeredgecolor=c),
                                medianprops=dict(color=c),showfliers=False)
            plt.ylabel("Mean accuracy SVM - Multi-class")
            plt.show()
        else:
            return mean_acc_stable, mean_acc_dec, mean_acc_inc

    def detect_goal_related_activity_using_multiple_subsets(self, radius=15, plot_for_control=False, plotting=True,
                                              subset_range=[4, 8, 12, 18], nr_splits=10):

        stable_mean = []
        stable_std = []
        dec_mean = []
        dec_std = []
        inc_mean = []
        inc_std = []
        stable = []
        dec = []
        inc = []

        for m_subset in subset_range:
            mean_acc_stable, mean_acc_dec, mean_acc_inc = self.detect_goal_related_activity_using_subsets(plotting=False,
                                                                                            m_subset=m_subset,
                                                                                            radius=radius,
                                                                                            nr_splits=nr_splits)
            stable_mean.append(np.mean(mean_acc_stable))
            stable_std.append(np.std(mean_acc_stable))
            dec_mean.append(np.mean(mean_acc_dec))
            dec_std.append(np.std(mean_acc_dec))
            inc_mean.append(np.mean(mean_acc_inc))
            inc_std.append(np.std(mean_acc_inc))
            stable.append(mean_acc_stable)
            dec.append(mean_acc_dec)
            inc.append(mean_acc_inc)

        if plotting:
            plt.errorbar(x=subset_range, y=stable_mean, yerr=stable_std, label="stable")
            plt.errorbar(x=subset_range, y=inc_mean, yerr=inc_std, label="inc")
            plt.errorbar(x=subset_range, y=dec_mean, yerr=dec_std, label="dec")
            plt.ylabel("Mean accuracy - multiclass SVM (mean,std)")
            plt.xlabel("#cells")
            plt.legend()
            plt.show()
        else:
            return stable, inc, dec

    """#################################################################################################################
    #  others
    #################################################################################################################"""

    def spatial_information_vs_firing_rate(self, spatial_resolution=5):
        nr_trials = self.get_nr_of_trials()
        # get rate maps
        rate_maps = self.get_rate_maps(spatial_resolution=spatial_resolution, trials_to_use=range(nr_trials))
        occ_map = self.get_occ_map(spatial_resolution=spatial_resolution, trials_to_use=range(nr_trials))
        raster = self.get_raster(trials_to_use="all")

        mean_firing_rate = np.mean(raster, axis=1)/self.params.time_bin_size
        max_firing_rate = np.max(raster, axis=1) / self.params.time_bin_size

        sparsity, info_per_sec, info_per_spike = compute_spatial_information(rate_maps=rate_maps, occ_map=occ_map)

        plt.scatter(mean_firing_rate, sparsity)
        plt.xlabel("mean firing rate")
        plt.ylabel("sparsity")
        plt.show()

        plt.scatter(mean_firing_rate, info_per_sec)
        plt.xlabel("mean firing rate")
        plt.ylabel("info_per_sec")
        plt.show()

        plt.scatter(mean_firing_rate, info_per_spike)
        plt.xlabel("mean firing rate")
        plt.ylabel("info_per_spike")
        plt.show()
        c_id = (mean_firing_rate- np.min(mean_firing_rate)) / np.max(mean_firing_rate- np.min(mean_firing_rate))
        plt.scatter(sparsity, info_per_sec, c = c_id)
        plt.xlabel("sparsity")
        plt.ylabel("info per sec")
        plt.set_cmap("Reds")
        a = plt.colorbar()
        a.set_label("mean firing (norm)")
        plt.show()

        plt.scatter(sparsity, info_per_spike, c=c_id)
        plt.xlabel("sparsity")
        plt.ylabel("info per spike")
        plt.set_cmap("Reds")
        a = plt.colorbar()
        a.set_label("mean firing (norm)")
        plt.show()

        c_id = (max_firing_rate- np.min(max_firing_rate)) / np.max(max_firing_rate- np.min(max_firing_rate))
        plt.scatter(sparsity, info_per_sec, c = c_id)
        plt.xlabel("sparsity")
        plt.ylabel("info per sec")
        plt.set_cmap("Reds")
        a = plt.colorbar()
        a.set_label("max firing (norm)")
        plt.show()

        plt.scatter(sparsity, info_per_spike, c=c_id)
        plt.xlabel("sparsity")
        plt.ylabel("info per spike")
        plt.set_cmap("Reds")
        a = plt.colorbar()
        a.set_label("max firing (norm)")
        plt.show()

    def occupancy_around_goals(self, radius=10, plot_for_control=False, plotting=False,
                               save_fig=False):
        occ_map = self.get_occ_map(spatial_resolution=1)


        # only for plotting
        occ_map_all_goals = np.zeros((occ_map.shape[0], occ_map.shape[1]))
        occ_map_all_goals[:] = np.nan
        occ_map_no_goals = np.copy(occ_map)

        for goal in self.goal_locations:
            occ_map_gc = np.zeros((occ_map.shape[0], occ_map.shape[1]))
            occ_map_gc[:] = np.nan
            y = np.arange(0, occ_map.shape[0])
            x = np.arange(0, occ_map.shape[1])
            cy = goal[0] - self.x_min
            cx = goal[1] - self.y_min
            # define mask to mark area around goal
            mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < radius ** 2
            # copy only masked z-values
            occ_map_gc[mask] = occ_map[mask]
            # copy also to map for all goals
            occ_map_all_goals[mask] = occ_map[mask]

            # copy only masked z-values
            occ_map_no_goals[mask] = np.nan

        if plot_for_control:
            plt.imshow(occ_map_all_goals)
            plt.colorbar()
            plt.show()

            plt.imshow(occ_map_no_goals)
            plt.colorbar()
            plt.show()

        occ_map_all_goals = np.nan_to_num(occ_map_all_goals)
        occ_map_no_goals = np.nan_to_num(occ_map_no_goals)
        # compute occupancy per cm2 around goals
        occ_around_goals_per_cm2 = np.sum(occ_map_all_goals.flatten())/(4*np.pi*radius**2)
        # compute occupancy per cm2 away from goals --> use tracking data to compute cheeseboard size
        # (or use radius of 60cm)
        min_diam = np.min([self.x_max-self.x_min, self.y_max-self.y_min])
        area_covered = np.pi*(min_diam/2)**2
        occ_wo_goals_per_cm2 = np.sum(occ_map_no_goals.flatten())/(area_covered - 4*np.pi*radius**2)

        if save_fig or plotting:
            plt.style.use('default')
            occ_map[occ_map==0] = np.nan
            b = plt.imshow(occ_map.T, cmap="Oranges")
            for goal in self.goal_locations:
                cy = goal[0] - self.x_min
                cx = goal[1] - self.y_min
                plt.scatter(cy,cx, c="black", label="Goal locations")
            a = plt.colorbar(mappable=b)
            a.set_label("Occupancy / s")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.axis('off')
            # circle1 = plt.Circle((45,40), 60)
            # plt.gca().add_patch(circle1)
            if save_fig:
                plt.rcParams['svg.fonttype'] = 'none'
                plt.savefig("occupancy_post.svg", transparent="True")
                plt.close()
            else:
                plt.show()

        else:
            return occ_around_goals_per_cm2, occ_wo_goals_per_cm2


"""#####################################################################################################################
#   CROSS-MAZE TASK
#####################################################################################################################"""


class CrossMaze(TrialParentClass):
    """Class for cross-maze task data analysis

       ATTENTION: this is only used for the task data --> otherwise use Sleep class!

    """

    def __init__(self, data_dic, cell_type, params, session_params, experiment_phase=None):
        """
        initializes cheeseboard class

        :param data_dic: dictionary containing spike data
        :type data_dic: python dic
        :param cell_type: which cell type to use
        :type cell_type: str
        :param params: general analysis params
        :type params: class
        :param session_params: sessions specific params
        :type session_params: class
        :param exp_phase_id: which experiment phase id
        :type exp_phase_id: int
        """

        # get attributes from parent class
        TrialParentClass.__init__(self, data_dic, cell_type, params, session_params, experiment_phase)

        # select all trials by default
        self.default_trials = range(self.data_dic["timestamps"].shape[0])

        # --------------------------------------------------------------------------------------------------------------
        # compute raster, location speed per trial
        # --------------------------------------------------------------------------------------------------------------
        self.trial_loc_list = []
        self.trial_raster_list = []
        self.trial_speed_list = []

        # go trough all trials
        for trial_id, key in enumerate(self.data_dic["trial_data"]):
            # check if two cell-types were provided --> will be combined
            if isinstance(cell_type, list) and len(cell_type) > 1:
                raster = []
                for ct in cell_type:
                    raster_cell_type, loc, speed = PreProcessAwake(
                        firing_times=self.data_dic["trial_data"][key]["spike_times"][ct],
                        params=self.params, whl=self.data_dic["trial_data"][key]["whl"],
                        spatial_factor=self.spatial_factor
                        ).interval_temporal_binning_raster_loc_vel(
                        interval_start=self.data_dic["timestamps"][trial_id, 0],
                        interval_end=self.data_dic["timestamps"][trial_id, 4])
                    raster.append(raster_cell_type)
                raster = np.vstack(raster)
            else:
                raster, loc, speed = PreProcessAwake(firing_times=self.data_dic["trial_data"][key]["spike_times"][cell_type],
                                                     params=self.params, whl=self.data_dic["trial_data"][key]["whl"],
                                                     spatial_factor=self.spatial_factor
                                                     ).interval_temporal_binning_raster_loc_vel(
                    interval_start=self.data_dic["timestamps"][trial_id, 0],
                    interval_end=self.data_dic["timestamps"][trial_id, 4])

            # update environment dimensions
            self.x_min = min(self.x_min, min(loc[:,0]))
            self.x_max = max(self.x_max, max(loc[:, 0]))
            self.y_min = min(self.y_min, min(loc[:,1]))
            self.y_max = max(self.y_max, max(loc[:, 1]))


            self.trial_raster_list.append(raster)
            self.trial_loc_list.append(loc)
            self.trial_speed_list.append(speed)


"""#####################################################################################################################
#   BASE CLASS TWO POPULATIONS
#####################################################################################################################"""


class BaseClassTwoPop:
    """Base class for data analysis of two populations"""

    def __init__(self, data_dic, cell_type_1, cell_type_2, params, session_params, experiment_phase):
        self.cell_type_1 = cell_type_1
        self.cell_type_2 = cell_type_2
        self.params = params

        self.nr_cell_type_1 = None
        self.nr_cell_type_2 = None

        self.session_params = session_params
        self.data_dic = data_dic

"""#####################################################################################################################
#   SLEEP CLASS TWO POPULATIONS
#####################################################################################################################"""


class TwoPopSleep(BaseClassTwoPop):
    """Class for cross-maze task data analysis

       ATTENTION: this is only used for the task data --> otherwise use Sleep class!

    """

    def __init__(self, data_dic, cell_type_1, cell_type_2, session_params, params, experiment_phase):
        """
        initializes cheeseboard class

        :param data_dic: dictionary containing spike data
        :type data_dic: python dic
        :param cell_type_1: which cell type to use
        :type cell_type_1: str
        :param cell_type_2: which cell type to use
        :type cell_type_2: str
        :param params: general analysis params
        :type params: class
        :param session_params: sessions specific params
        :type session_params: class
        :param exp_phase_id: which experiment phase id
        :type exp_phase_id: int
        """

        if isinstance(cell_type_1, list) and len(cell_type_1) == 1:
            cell_type_1 = cell_type_1[0]
        if isinstance(cell_type_2, list) and len(cell_type_2) == 1:
            cell_type_2 = cell_type_2[0]

        # get attributes from parent class
        BaseClassTwoPop.__init__(self, data_dic, cell_type_1, cell_type_2, params, session_params, experiment_phase)

        self.pop_1 = Sleep(data_dic=data_dic, cell_type=cell_type_1, params=params, session_params=session_params,
                           experiment_phase=experiment_phase)
        self.pop_2 = Sleep(data_dic=data_dic, cell_type=cell_type_2, params=params, session_params=session_params,
                           experiment_phase=experiment_phase)

    def plot_rasters(self):
        self.pop_1.view_raster()
        self.pop_2.view_raster()

    def get_rasters(self):
        return self.pop_1.get_raster(), self.pop_2.get_raster()





"""#####################################################################################################################
#   BASE METHODS TWO POPULATIONS --> delete when all methods are copied to the above
#####################################################################################################################"""

class BaseMethodsTwoPop:
    """Base class for general electro physiological data analysis of two popoulations"""

    def __init__(self, cell_type_array, params):

        self.params = params
        self.raster_list = []
        self.cell_type_array = cell_type_array

        self.nr_cells_pop_1 = None
        self.nr_cells_pop_2 = None

    """#################################################################################################################
    #   helper functions
    #################################################################################################################"""

    def firing_rate_hist(self):
        # --------------------------------------------------------------------------------------------------------------
        # plot firing rate histograms of both populations. To compute the firing rates the standard bin size is used (as
        # defined in main.py)
        # --------------------------------------------------------------------------------------------------------------

        for i, rate_map in enumerate(self.raster_list):
            if self.params.binning_method == "temporal_spike":
                avg_firing = np.mean(rate_map/self.params.time_bin_size, 1)
            elif self.params.binning_method == "temporal":
                avg_firing = np.mean(rate_map, 1)
            plt.subplot(2,1,i+1)
            plt.hist(avg_firing, bins=15)
            plt.xlim([0, 6])
            plt.ylabel("COUNTS")
            plt.xlabel("FIRING RATE / Hz")
            plt.title("FIRING RATE HISTOGRAM - "+str(self.params.time_bin_size)+ "s window - "+
                      str(self.cell_type_array[i]))
        plt.show()

    def spikes_per_bin(self):
        # --------------------------------------------------------------------------------------------------------------
        # computes histogram of #spikes per time bin
        # --------------------------------------------------------------------------------------------------------------
        if self.params.binning_method == "temporal_spike":
            for i, spike_map in enumerate(self.raster_list):
                plt.subplot(2,1,i+1)
                plt.hist(spike_map, bins=15)
                plt.xlim([0, 6])
                plt.ylabel("COUNTS")
                plt.xlabel("SPIKES / TIME BIN")
                plt.title("SPIKES PER TIME BIN - "+str(self.params.time_bin_size)+ "s window - "+
                          str(self.cell_type_array[i]))
            plt.show()
        else:
            raise Exception("NEED TEMPORAL SPIKE DATA AS INPUT")

    """#################################################################################################################
    #   synchrony analysis methods
    #################################################################################################################"""

    def sync_activity(self, raster_list_input=None, plotting=False):
        # --------------------------------------------------------------------------------------------------------------
        # computes nr. of cells that are active within one time bin for each population separately
        #
        # args:     - plotting: bool, plot results if True
        #           - raster_list: list with arrays, activity rasters from two populations
        # --------------------------------------------------------------------------------------------------------------

        if raster_list_input is not None:
            raster_list = raster_list_input
        else:
            raster_list = self.raster_list

        syn_array = [None] * len(raster_list)

        for i, map in enumerate(raster_list):
            syn = synchronous_activity(map)
            syn_array[i] = syn
            if plotting:
                plt.subplot(2, 1, i + 1)
                plt.plot(syn)
                plt.title(str(self.cell_type_array[i]) + " - " + str(self.params.time_bin_size) + "s window")

                plt.ylabel("% CELLS ACTIVE")

        if plotting:
            plt.xlabel("TIME")
            plt.show()

            # plot histogram with synchrony values
            for i, syn in enumerate(syn_array):
                plt.subplot(2, 1, i + 1)
                plt.hist(syn, bins=15)
                plt.xlim([0, 1])
                plt.ylabel("COUNTS")
                plt.xlabel("%ACTIVE CELLS")
                plt.title("SYNCHRONY - " + str(self.params.time_bin_size) + "s window - " + str(
                    self.cell_type_array[i]))
            plt.show()

        return syn_array

    def sync_activity_combined(self):
        # --------------------------------------------------------------------------------------------------------------
        # takes synchrony value (% of active cells) from both populations and multiplies them to find phases of high
        # synchrony. The resulting value is divided by the difference between the synchrony values of both populaions
        # to make sure that high synchrony in both populations is observed (and not only very high synchrony in one
        # population)
        # --------------------------------------------------------------------------------------------------------------

        syn_array = self.sync_activity()

        syn_1 = syn_array[0]
        syn_2 = syn_array[1]

        comb_syn = np.zeros(syn_1.shape[0])

        for i, (a, b) in enumerate(zip(syn_1, syn_2)):
            comb_syn[i] = a * b / (abs(a - b + np.finfo(float).eps))
        plt.xlabel("TIME BINS")
        plt.ylabel("COMB. SYNC - sync(p1)*sync(pe)/abs(sync(p1)-sync(pe))")
        plt.title("COMBINED SYNCHRONY - " + str(self.params.time_bin_size) + "s TIME WINDOW")
        plt.plot(comb_syn)
        plt.show()

    def sync_activity_corr(self):
        # --------------------------------------------------------------------------------------------------------------
        # computes correlation between synchrony values (% of cells active) of both populations
        #
        # args:   - plotting: bool, plot results if True
        # --------------------------------------------------------------------------------------------------------------

        syn_array = self.sync_activity(None, False)
        corr, _ = pearsonr(syn_array[0], syn_array[1])

        plt.scatter(syn_array[0], syn_array[1])
        plt.ylabel("% CELLS ACTIVE - " + str(self.cell_type_array[1]))
        plt.xlabel("% CELLS ACTIVE - " + str(self.cell_type_array[0]))
        plt.title("CORR SYNC - " + str(self.params.time_bin_size) + "s window" + " - PEARSON: " +
                  str(round(corr, 2)))
        plt.show()

    def sync_activity_cross_corr(self):
        # --------------------------------------------------------------------------------------------------------------
        # computes correlation between synchrony values of both populations with different times shifts (integer values
        # that define by how many time bins the synchrony values are shifted)
        # --------------------------------------------------------------------------------------------------------------

        import matplotlib.cm as cm
        colors = cm.rainbow(np.linspace(0, 1, 5))

        if not self.params.time_bin_size == 0.01:
            raise Exception("Time bin size parameter must be set to 0.01 for this analysis")

        for time_bin_size_counter, time_bin_size in enumerate([0.1, 0.07, 0.05, 0.02, 0.01]):
            transformed_rasters = []

            for pop in self.raster_list:

                transformed_rasters.append(down_sample_modify_raster(raster=pop, binning_method="temporal_spike",
                                                                     time_bin_size=self.params.time_bin_size,
                                                                     time_bin_size_after=time_bin_size))

            syn_array = self.sync_activity(transformed_rasters)

            shift_array = np.arange(-30, 31)

            corr = cross_correlate(x=syn_array[0], y=syn_array[1], shift_array=shift_array)

            plt.plot(shift_array, corr, "-o", color=
            colors[time_bin_size_counter], label="time_bin: " + str(time_bin_size) + "s")
            plt.title("CROSS-CORRELATION OF SYNCHRONY VALUES \n FROM BOTH POPULATION")
            plt.ylabel("PEARSON CORR. COEFF.")
            plt.ylim(0.0, 1.0)
            plt.xlabel("TIME BIN SHIFT - CENTERED ON " + self.cell_type_array[0])

        # get correlation value for shuffled data of last time bin size (usually: 10 ms)
        nr_shuffles = 500
        res_shuffles = np.zeros(nr_shuffles)
        for i in range(nr_shuffles):
            to_be_shuffled = syn_array[1]
            shuffled_data = np.copy(to_be_shuffled)
            # shuffle synchrony values
            np.random.shuffle(shuffled_data)
            res_shuffles[i], _ = pearsonr(syn_array[0], shuffled_data)

        plt.hlines(np.mean(res_shuffles), -30, 30, color=colors[-1], linestyles="dashed", label="shuff. mean/3std")
        plt.hlines(3 * np.std(res_shuffles), -30, 30, color=colors[-1], linestyles="dotted")
        plt.legend()
        plt.show()

    """#################################################################################################################
    #   synchrony - average firing rate analysis methods
    #################################################################################################################"""

    def sync_to_average_firing_rate(self, pop_for_avg_firing_rate, plotting=True):
        # --------------------------------------------------------------------------------------------------------------
        # computes the relationship between synchrony (% of cells firing within one time bin) in one are and average
        # firing rate in the other area
        #
        # args:     - pop_for_avg_firing_rate, str: for which population/area should average firing rate be
        #             calculated
        # --------------------------------------------------------------------------------------------------------------

        map_other = self.raster_list[self.cell_type_array.index(pop_for_avg_firing_rate)]

        map_own = self.raster_list[~self.cell_type_array.index(pop_for_avg_firing_rate)]

        # compute synchrony values
        syn_array = self.sync_activity(None, False)
        syn = syn_array[~self.cell_type_array.index(pop_for_avg_firing_rate)]

        # find unique values of synchrony
        syn_val = np.unique(syn)

        # computing global firing rate per time bin:

        # # calculate average firing rate for each time bin
        # avg_firing_rate_other_pop = np.zeros(map_other.shape[1])
        # avg_firing_rate_own_pop = np.zeros(map_own.shape[1])
        #
        # for i, (bin_other, bin_own) in enumerate(zip(map_other.T, map_own.T)):
        #     avg_firing_rate_other_pop[i] = np.mean(bin_other)/self.params.time_bin_size
        #     avg_firing_rate_own_pop[i] = np.mean(bin_own) / self.params.time_bin_size

        # computing avg. spike per bin for each cell first and then the global average

        avg_spike_per_cell_other = np.zeros((map_other.shape[0], syn_val.shape[0]))
        avg_spike_per_cell_own = np.zeros((map_own.shape[0], syn_val.shape[0]))
        for j, syn_un_val in enumerate(syn_val):
            for i, cell_row in enumerate(map_other):
                avg_spike_per_cell_other[i, j] = np.mean(cell_row[np.where(syn == syn_un_val)])
            for i, cell_row in enumerate(map_own):
                avg_spike_per_cell_own[i, j] = np.mean(cell_row[np.where(syn == syn_un_val)])

        plt.imshow(avg_spike_per_cell_other, interpolation='nearest', aspect='auto')
        plt.xticks(range(0, syn_val.shape[0], 5), np.round(syn_val, 3)[::5], rotation="vertical"
                   , fontsize=10)
        a = plt.colorbar()
        a.set_label("AVG. #SPIKES PER TIME BIN (" + str(self.params.time_bin_size) + "s)")
        plt.tight_layout()
        plt.title("SYNCHRONY - AVG. #SPIKES FOR EACH CELL")
        plt.xlabel("%CELLS ACTIVE - " + self.cell_type_array[~self.cell_type_array.index(pop_for_avg_firing_rate)])
        plt.ylabel("CELL IDS - " + self.cell_type_array[self.cell_type_array.index(pop_for_avg_firing_rate)])
        plt.show()

        if plotting:
            corr_with_own = []
            corr_with_other = []
            # plot correlation values
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax1 = fig.add_subplot(1, 2, 2)
            for cell_id, avg_spike_array in enumerate(avg_spike_per_cell_own):
                corr_val, p = pearsonr(avg_spike_array, syn_val)
                if p < 0.05:
                    corr_with_own.append(corr_val)
                    ax.scatter(cell_id, corr_val, color="blue", label=
                    self.cell_type_array[~self.cell_type_array.index(pop_for_avg_firing_rate)])

            # plot correlation values
            for cell_id, avg_spike_array in enumerate(avg_spike_per_cell_other):
                corr_val, p = pearsonr(avg_spike_array, syn_val)
                if p < 0.05:
                    corr_with_other.append(corr_val)
                    ax.scatter(cell_id, corr_val, color="red", label=
                    self.cell_type_array[self.cell_type_array.index(pop_for_avg_firing_rate)])
                    # plt.scatter(cell_id, p, edgecolors="red", label="r")

            ax.hlines(0, 0, max(avg_spike_per_cell_other.shape[0], avg_spike_per_cell_own.shape[0]), colors="w")
            ax.set_xlabel("CELL IDS")
            ax.set_ylabel("PEARSON R: AVG. FIRING - SYNCHRONY (" +
                          self.cell_type_array[~self.cell_type_array.index(pop_for_avg_firing_rate)] + ")")
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

            ax1.hist(corr_with_own, edgecolor="blue", density=True, fill=False, label=
            self.cell_type_array[~self.cell_type_array.index(pop_for_avg_firing_rate)])
            ax1.hist(corr_with_other, edgecolor="red", density=True, fill=False,
                     label=self.cell_type_array[self.cell_type_array.index(pop_for_avg_firing_rate)])
            ax1.set_xlabel("PEARSON R AVG. FIRING - \n SYNCHRONY" + " (" +
                           self.cell_type_array[~self.cell_type_array.index(pop_for_avg_firing_rate)] + ")")
            ax1.set_ylabel("COUNTS (NORMALIZED)")
            plt.legend()
            fig.suptitle("SIGN. CORRELATIONS: AVG. FIRING - SYNCHRONY" + " (" +
                         self.cell_type_array[~self.cell_type_array.index(pop_for_avg_firing_rate)] + ")", y=1.00)
            # fig.subplots_adjust(bottom=0.0, top=0.9)
            plt.show()

            avg_avg_firing_rate_other = np.zeros(syn_val.shape)
            std_avg_firing_rate_other = np.zeros(syn_val.shape)
            avg_avg_firing_rate_own = np.zeros(syn_val.shape)
            std_avg_firing_rate_own = np.zeros(syn_val.shape)

            for i, syn_un_val in enumerate(syn_val):
                avg_avg_firing_rate_other[i] = np.mean(avg_spike_per_cell_other[:, i] / self.params.time_bin_size)
                std_avg_firing_rate_other[i] = sem(avg_spike_per_cell_other[:, i] / self.params.time_bin_size)
                avg_avg_firing_rate_own[i] = np.mean(avg_spike_per_cell_own[:, i] / self.params.time_bin_size)
                std_avg_firing_rate_own[i] = sem(avg_spike_per_cell_own[:, i] / self.params.time_bin_size)

            plt.errorbar(syn_val, avg_avg_firing_rate_other, std_avg_firing_rate_other,
                         label=self.cell_type_array[self.cell_type_array.index(pop_for_avg_firing_rate)], c="r")
            plt.errorbar(syn_val, avg_avg_firing_rate_own, std_avg_firing_rate_own,
                         label=self.cell_type_array[~self.cell_type_array.index(pop_for_avg_firing_rate)])
            plt.title("SYNCHRONY - AVG. FIRING RATE - " + str(self.params.time_bin_size) + "s TIME BIN")
            plt.xlabel("%CELLS ACTIVE - " + self.cell_type_array[~self.cell_type_array.index(pop_for_avg_firing_rate)])
            plt.ylabel("POP. AVG. FIRING RATE + SEM / Hz")
            plt.legend()
            plt.show()

        else:
            return syn_val, avg_spike_per_cell_own, avg_spike_per_cell_other

    """#########_id########################################################################################################
    #   static co-firing analysis (using entire length of data)
    #################################################################################################################"""

    def correlation_matrix(self, x=None, y=None, shift=None, plotting=False):
        # --------------------------------------------------------------------------------------------------------------
        # calculates correlation matrix, y is shifted with respect to x by the value shift
        #
        # args:     - shift, integer:   - shift = 1 --> y is one time bin behind y
        #                               - shift = -1 --> y is one time bin before x
        #           - x, array: pop1, rows: cells, col: time points
        #           - y, array: pop2, rows: cells, col: time points
        # --------------------------------------------------------------------------------------------------------------

        if x is None or y is None:
            # if no input is provided use raster from class
            print("USING RASTERS X AND Y FROM CLASS")
            x = self.raster_list[0]
            y = self.raster_list[1]

        data_mat = np.vstack((x,y))

        # TODO: look at positive/negative shifting

        if shift is not None and shift !=0:
            separator = data_mat.shape[0]
            if shift > 0:
                data_mat_1 = data_mat[:, shift:]
                data_mat_2 = data_mat[:, :data_mat.shape[1] - shift]

            data_mat_shifted = np.vstack((data_mat_1, data_mat_2))
            correlation_matrix = np.nan_to_num(np.corrcoef(data_mat_shifted))
            # only select entries from shifted correlation
            correlation_matrix = correlation_matrix[separator:, :separator]

        else:

            correlation_matrix = np.nan_to_num(np.corrcoef(data_mat))

        if plotting:
            plt.imshow(correlation_matrix, interpolation='nearest', aspect='auto', cmap="jet")
            plt.title("CO-FIRING - " + "TIME BIN SIZE: "+str(self.params.time_bin_size)+"s")
            plt.xlabel("CELL IDS")
            plt.ylabel("CELL IDS")
            a = plt.colorbar()
            a.set_label("PEARSON CORRELATION R")
            plt.show()

        else:
            return correlation_matrix

    def split_correlation_matrix(self, plotting=False, shift=None):
        # --------------------------------------------------------------------------------------------------------------
        # separates correlation matrix into within population 1, within population 2, and across
        # --------------------------------------------------------------------------------------------------------------

        res = self.correlation_matrix(shift=shift)
        nr_cells_pop1 = self.nr_cells_pop_1

        # within pop1 correlation values are in upper triangle of first block
        corr_within_pop1 = res[:nr_cells_pop1 + 1, :nr_cells_pop1 + 1]
        # within pop1 correlation values are in upper triangle of second block
        corr_within_pop2 = res[(nr_cells_pop1 + 1):, (nr_cells_pop1 + 1):]
        # across pop correlation values all other entries
        corr_across = (res[:(nr_cells_pop1 + 1), (nr_cells_pop1 + 1):])

        if plotting:

            # get upper triangles to generate histograms
            tr_corr_within_pop1 = upper_tri_without_diag(corr_within_pop1)
            tr_corr_within_pop2 = upper_tri_without_diag(corr_within_pop2)

            print(np.max(tr_corr_within_pop1), np.max(tr_corr_within_pop2))

            max_value = 1
            min_value = -0.15

            plt.subplot(3,1,1)
            plt.title("WITHIN LEFT ("+ str(self.params.time_bin_size) + "s)")
            plt.hist(tr_corr_within_pop1.flatten())
            plt.xlim(min_value, max_value)
            plt.yscale('log', nonposy='clip')
            plt.subplot(3, 1, 2)
            plt.title("WITHIN RIGHT")
            plt.hist(tr_corr_within_pop2.flatten())
            plt.xlim(min_value, max_value)
            plt.yscale('log', nonposy='clip')
            plt.subplot(3, 1, 3)
            plt.title("ACROSS")
            plt.hist(corr_across.flatten())
            plt.xlim(min_value, max_value)
            plt.yscale('log', nonposy='clip')
            plt.xlabel("PEARSON CORRELATION R")
            plt.show()

            plt.imshow(corr_within_pop1, interpolation='nearest', aspect='auto', cmap="jet",
                       vmin=0, vmax=1)
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("PEARSON R")
            plt.title("POS. CORR. WITHIN POP1 (" + str(self.params.time_bin_size) + "s)")
            plt.show()

            plt.imshow(corr_within_pop2, interpolation='nearest', aspect='auto', cmap="jet",
                       vmax=1,vmin=0)
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("PEARSON R")
            plt.title("POS. CORR. WITHIN POP2 (" + str(self.params.time_bin_size) + "s)")
            plt.show()

            plt.imshow(corr_across.T, interpolation='nearest', aspect='auto', cmap="jet",
                       vmax=1, vmin=0)
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("PEARSON R")
            plt.title("CORR. ACROSS (" + str(self.params.time_bin_size) + "s)")
            plt.show()

        else:
            return corr_within_pop1, corr_within_pop2, corr_across

    def co_firing_shuffled(self, x=None, y=None, shift=None, nr_shuffles=10):
        # --------------------------------------------------------------------------------------------------------------
        # calculates co-firing of shuffled data, y is shifted behind x by the value shift
        #
        # args:     - shift, integer: shift = 1 --> y is one time bin behind
        #           - x, array: pop1, rows: cells, col: time points
        #           - y, array: pop2, rows: cells, col: time points
        # --------------------------------------------------------------------------------------------------------------

        if x is None or y is None:
            # if no input is provided use raster from class
            print("USING RASTERS X AND Y FROM CLASS")
            x = self.raster_list[0]
            y = self.raster_list[1]

        if shift is not None:
            if shift > 0:
                x = x[:, shift:]
                y = y[:, :y.shape[1]-shift]
            elif shift < 0:
                shift = abs(shift)
                x = x[:, :x.shape[1]-shift]
                y = y[:, shift:]

        # shuffles
        shuffle_list = []

        for shuffle_id in range(nr_shuffles):
            x_shuffled = np.copy(x).T
            np.random.shuffle(x_shuffled)
            x_shuffled = x_shuffled.T

            y_shuffled = np.copy(y).T
            np.random.shuffle(y_shuffled)
            y_shuffled = y_shuffled.T

            co_firing_matrix = np.zeros((x.shape[0], y.shape[0]))
            # go through all neurons of one population
            for i, neuron_in in enumerate(x_shuffled):
                # neuron_in is template we compute correlation with using all neurons from other area
                for j, neuron_out in enumerate(y_shuffled):
                    co_firing_matrix[i, j], _ = pearsonr(neuron_in, neuron_out)
            shuffle_list.append(co_firing_matrix)

        shuffle_3d_matrix = np.array(shuffle_list)

        mean_shuffled_co_firing = np.mean(shuffle_3d_matrix, axis=0)
        std_shuffled_co_firing = np.std(shuffle_3d_matrix, axis=0)

        return mean_shuffled_co_firing, std_shuffled_co_firing

    def co_firing_z_scored(self, cell_type_input, cell_type_output, full=True, sel_range=None, nr_shuffles=10,
                           split_co_firing_matrix=True):
        # --------------------------------------------------------------------------------------------------------------
        # compute correlation values --> z-score with mean/std of shuffled distribution
        # cell_type_output is shifted behind by the value shift
        #
        # args:         - shift, integer: shift = 1 --> cell_type_output is one time bin behind
        #               - cell_type_input, str
        #               - cell_type_output, str
        #               - full, bool: whether to use full co-firing matrix or only across
        #               - sel_range, range object: provide if only a temporal subset of the data should be used
        #               - nr_shuffles, int: how many shuffles to perform for shuffled distributions
        #               - split_co_firing_matrix, bool: split co-firing matrix for visualization if True
        # --------------------------------------------------------------------------------------------------------------

        # get number of cells from both populations
        nr_cells_pop1 = self.raster_list[self.cell_type_array.index(cell_type_input)].shape[0]
        nr_cells_pop2 = self.raster_list[self.cell_type_array.index(cell_type_output)].shape[0]

        x = self.raster_list[self.cell_type_array.index(cell_type_input)]
        y = self.raster_list[self.cell_type_array.index(cell_type_output)]

        if sel_range is not None:
            x = x[:, sel_range]
            y = y[:, sel_range]

        # remove cells that don't fire at all
        # x = x[~np.all(x == 0, axis=1)]
        # y = y[~np.all(y == 0, axis=1)]

        if full:
            x = np.vstack((x,y))
            y = np.copy(x)

        # shift_array = np.arange(-3, 4)
        shift_array = [None]

        time_shifted_corr_matrices = []
        time_shifted_shuffle_mean_matrices = []
        time_shifted_shuffle_std_matrices = []

        co_firing_root_name = self.params.pre_proc_dir+"co_firing_matrices/" + self.params.session_name + "_" +\
                              str(self.params.time_bin_size)+"s"

        if not os.path.isfile(co_firing_root_name + "_corr_val"):

            for shift in shift_array:

                # compute result with true data
                res = self.co_firing(x, y, shift, False)

                # compute mean/std using shuffles
                mean_shuff, std_shuff = self.co_firing_shuffled(cell_type_input, cell_type_output, shift, True,
                                                                sel_range, nr_shuffles)

                time_shifted_corr_matrices.append(res)
                time_shifted_shuffle_mean_matrices.append(mean_shuff)
                time_shifted_shuffle_std_matrices.append(std_shuff)

            # save results
            outfile = open(co_firing_root_name + "_corr_val", 'wb')
            pickle.dump(time_shifted_corr_matrices, outfile)
            outfile.close()

            outfile = open(co_firing_root_name + "_shuffle_mean", 'wb')
            pickle.dump(time_shifted_shuffle_mean_matrices, outfile)
            outfile.close()

            outfile = open(co_firing_root_name + "_shuffle_std", 'wb')
            pickle.dump(time_shifted_shuffle_std_matrices, outfile)
            outfile.close()

        else:

            res = np.array(pickle.load(open(co_firing_root_name + "_corr_val", "rb")))
            mean_shuff = np.array(pickle.load(open(co_firing_root_name + "_shuffle_mean", "rb")))
            std_shuff = np.array(pickle.load(open(co_firing_root_name + "_shuffle_std", "rb")))

        # check if results contain time bin shift
        if res.shape[0] > 1:
            # select some time bin shift and do stuff
            res = res[2, :, :]
            mean_shuff = mean_shuff[2, :, :]
            std_shuff = std_shuff[2, :, :]

        # if no time bin shift continue with standard procedure
        else:

            res = np.nan_to_num(np.squeeze(res, 0))
            mean_shuff = np.nan_to_num(np.squeeze(mean_shuff, 0))
            std_shuff = np.nan_to_num(np.squeeze(std_shuff, 0))

            # z_scored data
            res_z_scored = np.divide((res - mean_shuff), std_shuff)
            res_z_scored = np.nan_to_num(res_z_scored)

        if split_co_firing_matrix:

            # within pop1 correlation values are in upper triangle of first block
            corr_within_pop1 = res[:nr_cells_pop1 + 1, :nr_cells_pop1 + 1]
            # within pop1 correlation values are in upper triangle of second block
            corr_within_pop2 = res[(nr_cells_pop1 + 1):, (nr_cells_pop1 + 1):]
            # across pop correlation values all other entries
            corr_across = (res[:(nr_cells_pop1 + 1), (nr_cells_pop1 + 1):])

            # get upper triangles to generate histograms
            tr_corr_within_pop1 = upper_tri_without_diag(corr_within_pop1)
            tr_corr_within_pop2 = upper_tri_without_diag(corr_within_pop2)

            print(np.max(tr_corr_within_pop1), np.max(tr_corr_within_pop2))

            max_value = 1
            min_value = -0.15

            plt.subplot(3,1,1)
            plt.title("WITHIN LEFT ("+ str(self.params.time_bin_size) + "s)")
            plt.hist(tr_corr_within_pop1.flatten())
            plt.xlim(min_value, max_value)
            plt.yscale('log', nonposy='clip')
            plt.subplot(3, 1, 2)
            plt.title("WITHIN RIGHT")
            plt.hist(tr_corr_within_pop2.flatten())
            plt.xlim(min_value, max_value)
            plt.yscale('log', nonposy='clip')
            plt.subplot(3, 1, 3)
            plt.title("ACROSS")
            plt.hist(corr_across.flatten())
            plt.xlim(min_value, max_value)
            plt.yscale('log', nonposy='clip')
            plt.xlabel("PEARSON CORRELATION R")
            plt.show()

            plt.imshow(corr_within_pop1, interpolation='nearest', aspect='auto', cmap="jet",
                       vmin=0, vmax=1)
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("PEARSON R")
            plt.title("POS. CORR. WITHIN POP1 (" + str(self.params.time_bin_size) + "s)")
            plt.show()

            plt.imshow(corr_within_pop2, interpolation='nearest', aspect='auto', cmap="jet",
                       vmax=1,vmin=0)
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("PEARSON R")
            plt.title("POS. CORR. WITHIN POP2 (" + str(self.params.time_bin_size) + "s)")
            plt.show()

            plt.imshow(corr_across.T, interpolation='nearest', aspect='auto', cmap="jet",
                       vmax=1, vmin=0)
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("PEARSON R")
            plt.title("CORR. ACROSS (" + str(self.params.time_bin_size) + "s)")
            plt.show()

            # z-scored results
            # ----------------------------------------------------------------------------------------------------------

            # within pop1 correlation values are in upper triangle of first block
            corr_within_pop1_z_scored = res_z_scored[:nr_cells_pop1 + 1, :nr_cells_pop1 + 1]
            # within pop1 correlation values are in upper triangle of second block
            corr_within_pop2_z_scored = res_z_scored[(nr_cells_pop1 + 1):, (nr_cells_pop1 + 1):]
            # across pop correlation values all other entries
            corr_across_z_scored = (res_z_scored[:(nr_cells_pop1 + 1), (nr_cells_pop1 + 1):])

            # get upper triangles to generate histograms
            tr_corr_within_pop1_z_scored = upper_tri_without_diag(corr_within_pop1_z_scored)
            tr_corr_within_pop2_z_scored = upper_tri_without_diag(corr_within_pop2_z_scored)
            # do not need upper triangle for across values

            max_value = 110
            min_value = -15

            plt.subplot(3,1,1)
            plt.title("WITHIN LEFT ("+ str(self.params.time_bin_size) + "s)")
            plt.hist(tr_corr_within_pop1_z_scored.flatten())
            plt.xlim(min_value, max_value)
            plt.yscale('log', nonposy='clip')
            plt.subplot(3, 1, 2)
            plt.title("WITHIN RIGHT")
            plt.hist(tr_corr_within_pop2_z_scored.flatten())
            plt.xlim(min_value, max_value)
            plt.yscale('log', nonposy='clip')
            plt.subplot(3, 1, 3)
            plt.title("ACROSS")
            plt.hist(corr_across_z_scored.flatten())
            plt.xlim(min_value, max_value)
            plt.yscale('log', nonposy='clip')
            plt.xlabel("Z-SCORED PEARSON CORRELATION R")
            plt.show()

            # plot separated co-firing matrices

            plt.imshow(corr_within_pop1_z_scored, interpolation='nearest', aspect='auto', cmap="jet",
                       vmin=0, vmax=100)
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("Z-SCORED PEARSON R")
            plt.title("POS. Z-SCORED CORR. WITHIN POP1 (" + str(self.params.time_bin_size) + "s)")
            plt.show()

            plt.imshow(corr_within_pop2_z_scored, interpolation='nearest', aspect='auto', cmap="jet",
                       vmax=100,vmin=0)
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("Z-SCORED PEARSON R")
            plt.title("POS. Z-SCORED CORR. WITHIN POP2 (" + str(self.params.time_bin_size) + "s)")
            plt.show()

            plt.imshow(corr_across_z_scored.T, interpolation='nearest', aspect='auto', cmap="jet",
                       vmax=100, vmin=0)
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("Z-SCORED PEARSON R")
            plt.title("POS. Z-SCORED CORR. ACROSS (" + str(self.params.time_bin_size) + "s)")
            plt.show()

        else:

            plt.subplot(2,2,1)
            plt.imshow(mean_shuff, interpolation='nearest', aspect='auto', cmap="jet")
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("MEAN PEARSON R")
            plt.title("MEAN SHUFF. CORR. (" + str(self.params.time_bin_size) + "s)")

            plt.subplot(2, 2, 2)
            plt.imshow(std_shuff, interpolation='nearest', aspect='auto',
            cmap = "jet")
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("STD PEARSON R")
            plt.title("STD SHUFF. CORR. (" + str(self.params.time_bin_size) + "s)")

            plt.subplot(2, 2, 3)
            plt.imshow(res, interpolation='nearest', aspect='auto',
            cmap = "jet")
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("PEARSON R")
            plt.title("CORRELATION VALUES (" + str(self.params.time_bin_size) + "s)")

            plt.subplot(2, 2, 4)
            plt.imshow(res_z_scored, interpolation='nearest', aspect='auto',
            cmap = "seismic", vmax=50, vmin=-50)
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("Z-SCORED PEARSON R")
            plt.title("Z-SCORED CORR. (" + str(self.params.time_bin_size) + "s)")
            plt.show()

            # insert one extra col/row to show within
            ins = -1000 * np.ones(res_z_scored.shape[0])
            res_z_scored = np.insert(res_z_scored, nr_cells_pop1, ins, axis=1)
            ins = -1000 * np.ones(res_z_scored.shape[0]+1).T
            res_z_scored = np.insert(res_z_scored, nr_cells_pop1, ins.T, axis=0)

            plt.imshow(res_z_scored, interpolation='nearest', aspect='auto',
                       cmap="seismic", vmax=50, vmin=-50)
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("Z-SCORED PEARSON R")
            plt.title("Z-SCORED CORR. (" + str(self.params.time_bin_size) + "s)")
            plt.show()

            # remove diagonal for visualization
            res = res[~np.eye(res.shape[0], dtype=bool)].reshape(res.shape[0], -1)

            plt.imshow(res, interpolation='nearest', aspect='auto',
            cmap = "jet")
            plt.ylabel("CELL IDs")
            plt.xlabel("CELL IDs")
            a = plt.colorbar()
            a.set_label("PEARSON R")
            plt.title("CORRELATION VALUES (" + str(self.params.time_bin_size) + "s)")
            plt.show()

    """#################################################################################################################
    #   static correlation analysis (using entire length of data) WITH time bin shift
    #################################################################################################################"""

    def cdf_across_correlation_values_time_shifted(self):
        # --------------------------------------------------------------------------------------------------------------
        # computes cdf of across population correlation values, one cdf centered on one population and another cdf
        # centered on the other population
        # --------------------------------------------------------------------------------------------------------------

        # compute cdf centered on first population and then on the other population

        x_cell_type = self.cell_type_array[0]

        for shift in [0, 1, 2, 3, 5, 10]:

            _,_, across = self.split_correlation_matrix(shift=shift, plotting=False)
            # generate pdf
            n_bins = 500
            plt.hist(across.flatten(), n_bins, density=True, histtype='step',
                           cumulative=True, label='TIME BINS SHIFT: '+str(shift))
        plt.xlabel("PEARSON CORRELATION R")
        plt.ylabel("CDF")
        plt.title("CDF OF ACROSS CORRELATION VALUES FROM CO-FIRING MATRIX \n AT DIFFERENT TIME LAGS (TIME BIN: "+
                  str(self.params.time_bin_size)+"s)"+" CENTERED ON: "+x_cell_type)
        plt.legend(loc="lower right")
        plt.show()

    def correlation_matrix_time_shift(self, shift, full=True):
        # --------------------------------------------------------------------------------------------------------------
        # computes and plots correlation matrices at zero and shift time delay next to each other
        #
        # args:         - shift, integer:   shift = 1 --> y is one time bin behind x
        #                                   - for full correlation matrix: shift = 1: everything is shifted one behind
        #               - full, bool: whether to use full correlation matrix (True) or only across values (False)
        # --------------------------------------------------------------------------------------------------------------

        # compute correlation with and without time bin shifted on each population
        for i in range(2):

            if full:

                if i == 0:
                    res_zero_lag = self.correlation_matrix(shift=None, plotting=False)
                    # shift one behind
                    res_one_lag = self.correlation_matrix(shift=shift, plotting=False)
                    x_cell_type = self.cell_type_array[0]

                if i == 1:
                    # shift one ahead
                    shift = -1*shift
                    res_one_lag = self.correlation_matrix(shift=shift, plotting=False)
                    x_cell_type = self.cell_type_array[0]

            else:
                _, _, across_zero_lag = self.split_correlation_matrix(shift=None, plotting=False)
                _, _, across_one_lag = self.split_correlation_matrix(shift=shift, plotting=False)

                # transpose to always the population with more cells on the y-axis
                if across_zero_lag.shape[0] < across_zero_lag.shape[1]:
                    res_zero_lag = across_zero_lag.T
                    res_one_lag = across_one_lag.T
                else:
                    res_zero_lag = across_zero_lag
                    res_one_lag = across_one_lag

            # find maximum change for scaling of color plot
            max_change = np.max(np.abs((res_one_lag-res_zero_lag)))

            if full:
                # if full correlation matrix is supposed to be used --> plot both (zero lag and shifted one) in
                # separate plots

                if i == 0:
                    plt.imshow(res_zero_lag, interpolation='nearest', aspect='auto', cmap="jet")
                    plt.title("CO-FIRING \n " + "TIME BIN SIZE: " + str(self.params.time_bin_size) + "s")
                    plt.hlines(self.nr_cells_pop_1 + 0.5, -0.5, res_zero_lag.T.shape[1] - 0.5, colors="white")
                    plt.vlines(self.nr_cells_pop_1 + 0.5, -0.5, res_zero_lag.T.shape[1] - 0.5, colors="white")
                    plt.xlabel("CELL IDS")
                    plt.ylabel("CELL IDS")
                    a = plt.colorbar()
                    a.set_label("PEARSON CORRELATION R")
                    plt.show()
                plt.imshow((res_one_lag-res_zero_lag), interpolation='nearest', aspect='auto', cmap="seismic",
                           vmin=-max_change, vmax=max_change)
                plt.hlines(self.nr_cells_pop_1 + 0.5, -0.5, res_zero_lag.T.shape[1] - 0.5, colors="black")
                plt.vlines(self.nr_cells_pop_1 + 0.5, -0.5, res_zero_lag.T.shape[1] - 0.5, colors="black")
                plt.title("CHANGE IN CO-FIRING \n " + "TIME BIN SHIFT: " + str(shift))
                plt.xlabel("CELL IDS")
                plt.ylabel("CELL IDS")
                a = plt.colorbar()
                a.set_label("CHANGE IN PEARSON CORRELATION R")
                plt.show()

            else:
                # if across population correlation values are supposed to be used --> subplots
                plt.subplot(1,2,1)
                plt.imshow(res_zero_lag, interpolation='nearest', aspect='auto', cmap="jet")
                plt.title("CO-FIRING \n " + "TIME BIN SIZE: " + str(self.params.time_bin_size) + "s")
                plt.xlabel("CELL IDS POPULATION A")
                plt.ylabel("CELL IDS POPULATION B")
                a = plt.colorbar()
                a.set_label("PEARSON CORRELATION R")
                plt.subplot(1, 2, 2)
                plt.imshow((res_one_lag-res_zero_lag), interpolation='nearest', aspect='auto', cmap="seismic",
                           vmin=-max_change, vmax=max_change)
                plt.title("CHANGE IN CO-FIRING \n " + "TIME BIN SHIFT: " + str(shift))
                plt.xlabel("CELL IDS POPULATION A")
                a = plt.colorbar()
                a.set_label("CHANGE IN PEARSON CORRELATION R")
                plt.show()

    def optimal_time_shift_for_highest_correlation(self, x=None, y=None):
        # --------------------------------------------------------------------------------------------------------------
        # computes optimal time bin shift for each correlation entry (edge between cells)
        #
        # args:     -x,y: input arrays (rows:cells, columns:time bins), if not provided --> class attributes are used
        # --------------------------------------------------------------------------------------------------------------

        # shift populations according to shift_array
        shift_array = np.arange(0, 4)

        # index for zero time delay
        ind_zero_time_delay = np.argwhere(shift_array == 0)[0][0]

        if x is None and y is None:
            # use data from class

            # nr cells in population 1
            nr_cells_x = self.raster_list[0].shape[0]

            time_shifted_corr_matrices = []
            for shift in shift_array:
                time_shifted_corr_matrices.append(self.correlation_matrix(shift=shift))
            time_shifted_corr_matrices = np.array(time_shifted_corr_matrices)


        else:
            # nr cells in population 1
            nr_cells_x = x.shape[0]

            # compute data with x and y input
            x = np.vstack((x, y))
            y = np.copy(x)

            time_shifted_corr_matrices = []
            for shift in shift_array:
                time_shifted_corr_matrices.append(self.correlation_matrix(x=x, y=y, shift=shift))
            time_shifted_corr_matrices = np.array(time_shifted_corr_matrices)

        # get entries with highest correlation value from all correlation matrices with different time bin shifts
        best_shift_ind = np.argmax(time_shifted_corr_matrices, axis=0)

        # best_shift matrix
        best_shift_matrix = np.zeros(best_shift_ind.shape)

        for index, _ in np.ndenumerate(best_shift_matrix):
            best_shift_matrix[index] = shift_array[best_shift_ind[index]]

        # separate within and across results
        # --------------------------------------------------------------------------------------------------------------

        # within pop1 correlation values are in upper triangle of first block
        best_shift_within_pop1 = upper_tri_without_diag(best_shift_matrix[:nr_cells_x+1, :nr_cells_x+1]).flatten()
        # within pop1 correlation values are in upper triangle of second block
        best_shift_within_pop2 = upper_tri_without_diag(best_shift_matrix[(nr_cells_x+1):, (nr_cells_x+1):]).flatten()
        # across pop correlation values all other entries
        best_shift_across = best_shift_matrix[:(nr_cells_x+1), (nr_cells_x+1):].flatten()

        # WITHIN 1
        # --------------------------------------------------------
        corr_list_within_pop1 = []
        corr_list_within_pop2 = []
        corr_list_across = []

        for corr_matrix in time_shifted_corr_matrices:
            # within pop1 correlation values are in upper triangle of first block
            corr_list_within_pop1.append(upper_tri_without_diag(corr_matrix[:nr_cells_x+1, :nr_cells_x+1]).flatten())
            # within pop1 correlation values are in upper triangle of second block
            corr_list_within_pop2.append(upper_tri_without_diag(corr_matrix[(nr_cells_x+1):, (nr_cells_x+1):]).flatten())
            # across pop correlation values all other entries
            corr_list_across.append(corr_matrix[:(nr_cells_x+1), (nr_cells_x+1):].flatten())

        corr_across_time_shifted = np.array(corr_list_across)
        corr_within1_time_shifted = np.array(corr_list_within_pop1)
        corr_within2_time_shifted = np.array(corr_list_within_pop2)

        # sort according to maximum correlation values
        max_r = np.argmax(corr_across_time_shifted, axis=0)
        max_r = shift_array[max_r]
        sorted_ind_across = max_r.argsort()
        corr_across_time_shifted_sorted = corr_across_time_shifted.T[sorted_ind_across, :]

        # sort according to maximum correlation values
        max_r = np.argmax(corr_within1_time_shifted, axis=0)
        max_r = shift_array[max_r]
        sorted_ind_within1 = max_r.argsort()
        corr_within1_time_shifted_sorted = corr_within1_time_shifted.T[sorted_ind_within1, :]

        # sort according to maximum correlation values
        max_r = np.argmax(corr_within2_time_shifted, axis=0)
        max_r = shift_array[max_r]
        sorted_ind_within2 = max_r.argsort()
        corr_within2_time_shifted_sorted = corr_within2_time_shifted.T[sorted_ind_within2, :]

        if x is not None and y is not None:
            # return indices of edges to sort according to correlation value peak
            return corr_across_time_shifted, corr_within1_time_shifted, corr_within2_time_shifted, \
                   sorted_ind_across, sorted_ind_within1, sorted_ind_within2

        else:
            # plot results
            plot_optimal_correlation_time_shift(best_shift_matrix, self.params, True, nr_cells_x)
            plot_optimal_correlation_time_shift_hist(best_shift_within_pop1, best_shift_within_pop2, best_shift_across,
                                                     self.params, self.cell_type_array)
            plot_optimal_correlation_time_shift_edges(corr_across_time_shifted_sorted, corr_within1_time_shifted_sorted,
                                                      corr_within2_time_shifted_sorted, self.params, shift_array,
                                                      self.cell_type_array)

    """#################################################################################################################
    #   static correlation analysis using different section of the sleep session
    # ###############################################################################################################"""

    def optimal_time_shift_for_highest_correlation_section_wise(self, sel_range_1, sel_range_2):
        # --------------------------------------------------------------------------------------------------------------
        # computes optimal time bin shift for each correlation entry (edge between cells) and compares values for
        # different time intervals of the data: edges are ordered according to maximum correlation of the first provided
        # sel_range_1
        #
        #   args:   - sel_range_1: range object, defines time interval 1
        #           - sel_range_2: range object, defines time interval 2
        #
        # --------------------------------------------------------------------------------------------------------------
        x = self.raster_list[0]
        y = self.raster_list[1]

        # compute for first interval
        # --------------------------------------------------------------------------------------------------------------
        x = x[:, sel_range_1]
        y = y[:, sel_range_1]

        x = np.vstack((x, y))
        y = np.copy(x)

        corr_across_time_shifted, corr_within1_time_shifted, corr_within2_time_shifted, \
        sorted_ind_across, sorted_ind_within1, sorted_ind_within2 = \
            self.optimal_time_shift_for_highest_correlation(x,y)

        # sort according to maximum correlation values
        corr_across_time_shifted_sorted = corr_across_time_shifted.T[sorted_ind_across, :]

        # sort according to maximum correlation values
        corr_within1_time_shifted_sorted = corr_within1_time_shifted.T[sorted_ind_within1, :]

        # sort according to maximum correlation values
        corr_within2_time_shifted_sorted = corr_within2_time_shifted.T[sorted_ind_within2, :]

        shift_array = np.arange(-3, 4)

        plot_optimal_correlation_time_shift_edges(corr_across_time_shifted_sorted, corr_within1_time_shifted_sorted,
                                                  corr_within2_time_shifted_sorted, self.params, shift_array,
                                                  self.cell_type_array)
        # compute for second interval
        # --------------------------------------------------------------------------------------------------------------
        x = self.raster_list[0]
        y = self.raster_list[1]

        x = x[:, sel_range_2]
        y = y[:, sel_range_2]

        x = np.vstack((x, y))
        y = np.copy(x)

        corr_across_time_shifted, corr_within1_time_shifted, corr_within2_time_shifted, \
        _, _, _ = \
            self.optimal_time_shift_for_highest_correlation(x,y)

        # sort according to maximum correlation values of first section
        corr_across_time_shifted_sorted = corr_across_time_shifted.T[sorted_ind_across, :]

        # sort according to maximum correlation values of first section
        corr_within1_time_shifted_sorted = corr_within1_time_shifted.T[sorted_ind_within1, :]

        # sort according to maximum correlation values of first section
        corr_within2_time_shifted_sorted = corr_within2_time_shifted.T[sorted_ind_within2, :]

        plot_optimal_correlation_time_shift_edges(corr_across_time_shifted_sorted, corr_within1_time_shifted_sorted,
                                                  corr_within2_time_shifted_sorted, self.params, shift_array,
                                                  self.cell_type_array)

    def segment_co_firing_mds(self, co_firing_window_size=0.1, sel_range=None, distance_measure="frob", video=False):
        # --------------------------------------------------------------------------------------------------------------
        # visualization of co-firing matrices of different sections of the sleep session WITHOUT sliding window
        #
        # args:     - co_firing_window_size: length of window in seconds for which correlation values are computed
        #           - sel_range, range object: for which range to compute MDS
        #           - distance_measure: which distance measure to use for correlation matrices
        #                                   - frobenius:    subtract one correlation matrix from the other and use
        #                                                   Frobenius norm of difference matrix
        #
        #
        # --------------------------------------------------------------------------------------------------------------
        x = self.raster_list[0]
        y = self.raster_list[1]

        if sel_range is not None:

            x = x[:, sel_range]
            y = y[:, sel_range]

        # stack up for full correlation matrix
        x = np.vstack((x, y))
        y = np.copy(x)

        # compute how many time bins fit in one window
        time_bins_per_co_firing_matrix = int(co_firing_window_size / self.params.time_bin_size)

        co_firing_matrices = self.dynamic_co_firing(x=x, y=y,
                                                    time_bins_per_co_firing_matrix=time_bins_per_co_firing_matrix,
                                                    sliding_window=False)

        dist = np.zeros((co_firing_matrices.shape[0], co_firing_matrices.shape[0]))

        # to apply MDS for visualization --> calculate frobenius norm of differences between all matrices
        for i, template in enumerate(co_firing_matrices):
            # choose one co-firing matrix as template and compare to all the others
            # one_left_co_firing_matrix = np.delete(co_firing_matrix, i, axis=0)
            # print(str(i / co_firing_matrix.shape[0] * 100) + "%")
            for e, compare in enumerate(co_firing_matrices):
                # dist[i, e] = abs(np.linalg.norm(template)-np.linalg.norm(compare))
                if distance_measure == "frob":
                    dist[i, e] = np.linalg.norm(template - compare)
                elif distance_measure == "graph":
                    dist[i, e] = graph_distance(template, compare)

        # outfile = open("dist", "wb")
        # pickle.dump(dist, outfile)
        # outfile.close()

        # infile = open("dist", 'rb')
        # dist = pickle.load(infile)
        # infile.close()

        model = MDS(n_components=self.params.dr_method_p2, dissimilarity='precomputed', random_state=1)
        result = model.fit_transform(dist)

        if video:

            scatter_animation(result, self.params)

        else:
            if self.params.dr_method_p2 == 3:
                # create figure instance
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plot_3D_scatter(ax=ax, mds=result, params=self.params)
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plot_2D_scatter(ax=ax, mds=result, params=self.params)
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.show()

    def segment_co_firing_pca(self, co_firing_window_size, sel_range=None, video=False):
        # --------------------------------------------------------------------------------------------------------------
        # decomposition of correlation matrices --> takes correlaton matrices, flattens them and computes pc
        #
        # args:     - co_firing_window_size: length of window in seconds for which correlation values are computed
        #           - sel_range, range object: for which range to compute MDS
        #           - distance_measure: which distance measure to use for correlation matrices
        #                                   - frobenius:    subtract one correlation matrix from the other and use
        #                                                   Frobenius norm of difference matrix
        #
        #
        # --------------------------------------------------------------------------------------------------------------
        x = self.raster_list[0]
        y = self.raster_list[1]

        if sel_range is not None:

            x = x[:, sel_range]
            y = y[:, sel_range]

        # stack up for full correlation matrix
        x = np.vstack((x, y))
        y = np.copy(x)

        # compute how many time bins fit in one window
        time_bins_per_co_firing_matrix = int(co_firing_window_size / self.params.time_bin_size)

        co_firing_matrices = self.dynamic_co_firing(raster=x,
                                                    time_bins_per_co_firing_matrix=time_bins_per_co_firing_matrix,
                                                    sliding_window=False)
        # flatten correlation matrices
        co_fir_mat_flat = []
        for co_fir_mat in co_firing_matrices:
            co_fir_mat_flat.append(co_fir_mat.flatten())
        co_fir_mat_flat = np.array(co_fir_mat_flat).T

        # delete entries with faulty values
        co_fir_mat_flat = co_fir_mat_flat[np.std(co_fir_mat_flat, axis=1) != 0.0,:]


        #
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=10, random_state=0).fit(co_fir_mat_flat)
        a = np.argsort(kmeans.labels_)
        co_fir_mat_flat = co_fir_mat_flat[a,:]

        plt.imshow(co_fir_mat_flat, interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.show()


        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.params.dr_method_p2)
        result = pca.fit_transform(co_fir_mat_flat)
        print(pca.explained_variance_ratio_)
        print(result.shape)

        if video:

            scatter_animation(result, self.params)

        else:
            if self.params.dr_method_p2 == 3:
                # create figure instance
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plot_3D_scatter(ax=ax, mds=result, params=self.params)
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plot_2D_scatter(ax=ax, mds=result, params=self.params)
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.show()

    """#################################################################################################################
    #   dynamic correlation analysis (using sliding window) WITHOUT time bin shift
    #################################################################################################################"""

    def dynamic_co_firing(self, raster=None, time_bins_per_co_firing_matrix=20, shift=None, sliding_window=True,
                          sel_range=None):
        # --------------------------------------------------------------------------------------------------------------
        # calculates co-firing for each time bin --> each time bin is split into equal chunks of length time_bin_size
        # and the pearson correlation for all possible cell pairs is calculated
        #
        # args:         - shift, integer: shift = 1 --> cell_type_output is one time bin behind
        #               - sel_range, range object: time range to use
        #               - sliding window, bool: sliding window or discrete time windows
        #               - time_bins_per_co_firing_matrix, int:  how many time bins should be used to construct one
        #                                                       co-firing matrix (e.g. time_bin_size = 10 ms and
        #                                                       time_bins_per_co_firing_matrix = 20 --> one co-firing
        #                                                       matrix per 200ms
        # --------------------------------------------------------------------------------------------------------------

        if raster is None:
            # if no input is provided use raster from class
            print("USING RASTER FROM CLASS")
            x = self.raster_list[0]
            y = self.raster_list[1]
            # remove cells that don't fire at all
            # x = x[~np.all(x == 0, axis=1)]
            # y = y[~np.all(y == 0, axis=1)]
            # generate full correlation matrix
            raster = np.vstack((x, y))
        if sel_range is not None:
            raster = raster[:, sel_range]

        if shift:
            if shift > 0:
                x = x[:, shift:]
                y = y[:, :y.shape[1] - shift]
            elif shift < 0:
                shift = abs(shift)
                x = x[:, :x.shape[1] - shift]
                y = y[:, shift:]

        if sliding_window:

            correlation_matrices = []

            print(" - COMPUTING SUBSEQUENT CORRELATION MATRICES ....\n")

            for entry in range(int(raster.shape[1] - time_bins_per_co_firing_matrix + 1)):
                # print("PROGRESS: "+ str(entry +1) + "/" + str(int(x.shape[1] - time_bins_per_co_firing_matrix)+1)
                #       +" FRAMES")
                correlation_matrices.append(np.corrcoef(raster[:, entry:(entry + time_bins_per_co_firing_matrix)]))

            correlation_matrices = np.array(correlation_matrices)

        else:

            correlation_matrices = []
            for entry in range(int(raster.shape[1] / time_bins_per_co_firing_matrix)):
                co_fir_numpy = np.corrcoef(raster[:, entry * time_bins_per_co_firing_matrix:
                (entry+1)*time_bins_per_co_firing_matrix])
                correlation_matrices.append(co_fir_numpy)

            correlation_matrices = np.array(correlation_matrices)

        # if one vector is constant (e.g. all zeros) --> pearsonr return np.nan
        # set all nans to zero

        correlation_matrices = np.nan_to_num(correlation_matrices, posinf=0, neginf=0)

        # global_correlation_matrix = np.corrcoef(raster)
        # correlation_matrices = global_correlation_matrix - correlation_matrices

        print("  ... DONE\n")

        return correlation_matrices

    def dynamic_co_firing_generator(self, raster=None, time_bins_per_co_firing_matrix=20, sel_range=None):
        # --------------------------------------------------------------------------------------------------------------
        # calculates correlation matrix for a sliding window. Yields every matrix after it is computed, in order to
        # not exhaust memory
        #
        # args:         - sel_range, range object: time range to use
        #               - time_bins_per_co_firing_matrix, int:  how many time bins should be used to construct one
        #                                                       co-firing matrix (e.g. time_bin_size = 10 ms and
        #                                                       time_bins_per_co_firing_matrix = 20 --> one co-firing
        #                                                       matrix per 200ms
        # --------------------------------------------------------------------------------------------------------------

        if raster is None:
            # if no input is provided use raster from class
            x = self.raster_list[0]
            y = self.raster_list[1]
            # remove cells that don't fire at all
            # x = x[~np.all(x == 0, axis=1)]
            # y = y[~np.all(y == 0, axis=1)]
            # generate full correlation matrix
            raster = np.vstack((x, y))
        if sel_range is not None:
            raster = raster[:, sel_range]
        entry = 0
        while entry < int(raster.shape[1] - time_bins_per_co_firing_matrix + 1):
            # print("PROGRESS: "+ str(entry +1) + "/" + str(int(x.shape[1] - time_bins_per_co_firing_matrix)+1)
            #       +" FRAMES")
            yield np.nan_to_num(np.corrcoef(raster[:, entry:(entry + time_bins_per_co_firing_matrix)]))
            entry += 1

    def regression_for_causal_connectivity(self, x=None, y=None, time_bins_per_matrix=20, sliding_window=True):
        # --------------------------------------------------------------------------------------------------------------
        # uses linear regression plus lasso to estimate causal connectivity
        # and the pearson correlation for all possible cell pairs is calculated
        #
        # args:         - sel_range, range object: time range to use
        #               - sliding window, bool: sliding window or discrete time windows
        #               - time_bins_per_matrix, int:  how many time bins should be used to construct one
        #                                                       connectivity matrix (e.g. time_bin_size = 10 ms and
        #                                                       time_bins_per_co_firing_matrix = 20 --> one co-firing
        #                                                       matrix per 200ms
        # --------------------------------------------------------------------------------------------------------------

        if x is None or y is None:
            # if no input is provided use raster from class
            print("USING RASTERS X AND Y FROM CLASS")
            x = self.raster_list[0]
            y = self.raster_list[1]
            # remove cells that don't fire at all
            x = x[~np.all(x == 0, axis=1)]
            y = y[~np.all(y == 0, axis=1)]
            # generate full correlation matrix
            x = np.vstack((x, y))
            y = np.copy(x)


        # if shift:
        #     if shift > 0:
        #         x = x[:, shift:]
        #         y = y[:, :y.shape[1] - shift]
        #     elif shift < 0:
        #         shift = abs(shift)
        #         x = x[:, :x.shape[1] - shift]
        #         y = y[:, shift:]

        if sliding_window:
            print("TO BE IMPLEMENTED")
            exit()

            co_firing_matrices = np.zeros(
                (int(x.shape[1] - time_bins_per_matrix + 1), x.shape[0], y.shape[0]))

            # construct co_firing matrix: pearson correlation value between cells from both regions using multiple
            # time bins (time_bins_per_co_firing_matrix)

            for entry in range(int(x.shape[1] - time_bins_per_matrix + 1)):
                print("PROGRESS: " + str(entry + 1) + "/" + str(int(x.shape[1] - time_bins_per_matrix) + 1)
                      + " FRAMES")
                for i, pop_vec_in in enumerate(x[:, entry:(entry + time_bins_per_matrix)]):
                    for j, pop_vec_out in enumerate(y[:, entry:(entry + time_bins_per_matrix)]):
                        co_firing_matrices[entry, i, j], _ = pearsonr(pop_vec_in, pop_vec_out)

        else:
            causal_connect_matrices = []
            # TODO: find optimal alpha
            # normalize x
            import sklearn
            x = sklearn.preprocessing.minmax_scale(x.T).T
            # plt.imshow(x, interpolation='nearest', aspect='auto')
            # plt.colorbar()
            # plt.show()
            # exit()
            regression = Lasso(fit_intercept=False, alpha=0.01, max_iter=5000)
            for entry in range(int(x.shape[1] / time_bins_per_matrix)):
                window = x[:, entry * time_bins_per_matrix:
                     (entry + 1) * time_bins_per_matrix]
                for neuron_id in range(x.shape[0]):
                    W = np.delete(window, neuron_id, axis=0).T
                    Y = window[neuron_id, :].T + np.finfo(float).eps
                    # if not np.count_nonzero(Y):
                    #     V = np.zeros(x.shape[0]-1)
                    #     Y = logit(window[neuron_id, :])
                    # else:
                    #     print(Y)
                    #     Y = np.nan_to_num(logit(Y),neginf=0 )
                    Y = logit(Y)
                    print(Y)
                    regression.fit(W, Y)
                    V = regression.coef_
                    causal_connect_matrices.append(V)

                causal_connect_matrices = np.array(causal_connect_matrices)
                causal_connect_matrices /= np.max(causal_connect_matrices)
                plt.imshow(causal_connect_matrices, vmin=-1, vmax=1)
                plt.colorbar()
                plt.show()

                co_fir_numpy = np.corrcoef(x[:, entry * time_bins_per_matrix:
                                                (entry + 1) * time_bins_per_matrix])
                co_fir_numpy = np.nan_to_num(co_fir_numpy)
                # delete diagonal
                co_fir_numpy = co_fir_numpy[~np.eye(co_fir_numpy.shape[0], dtype=bool)].reshape(co_fir_numpy.shape[0], -1)
                plt.imshow(co_fir_numpy, vmin=-1, vmax=1)
                plt.colorbar()
                plt.show()
                exit()
                causal_connect_matrices.append(co_fir_numpy)

            co_firing_matrices = np.array(causal_connect_matrices)

            # OWN IMPLEMENTATION --> much slower than standard numpy method
            # co_firing_matrices = np.zeros((int(x.shape[1]/time_bins_per_co_firing_matrix), x.shape[0], y.shape[0]))
            #
            # # construct co_firing matrix: pearson correlation value between cells from both regions using multiple
            # # time bins (time_bins_per_co_firing_matrix)
            #
            # for entry in range(int(x.shape[1]/time_bins_per_co_firing_matrix)):
            #     for i, pop_vec_in in enumerate(x[:, entry * time_bins_per_co_firing_matrix:
            #     (entry+1)*time_bins_per_co_firing_matrix]):
            #         for j, pop_vec_out in enumerate(y[:, entry * time_bins_per_co_firing_matrix:
            #         (entry+1)*time_bins_per_co_firing_matrix]):
            #             co_firing_matrices[entry, i, j], _ = pearsonr(pop_vec_in, pop_vec_out)

        # if one vector is constant (e.g. all zeros) --> pearsonr return np.nan
        # set all nans to zero
        co_firing_matrices = np.nan_to_num(co_firing_matrices, posinf=0, neginf=0)

        return causal_connect_matrices

    def load_or_create_dynamic_co_firing(self, co_firing_window_size, sel_range, x=None, y=None):
        # --------------------------------------------------------------------------------------------------------------
        # checks if dynamic co-firing matrices exist, otherwise calls dynamic_co_firing to create them
        # --------------------------------------------------------------------------------------------------------------

        if x is None or y is None:
            # if no input is provided use raster from class
            print("USING RASTERS X AND Y FROM CLASS")
            x = self.raster_list[0]
            y = self.raster_list[1]
            # remove cells that don't fire at all
            x = x[~np.all(x == 0, axis=1)]
            y = y[~np.all(y == 0, axis=1)]
            # generate full correlation matrix
            x = np.vstack((x, y))
            y = np.copy(x)

        data_name = self.params.session_name+"_"+ self.params.experiment_phase_id[0] +"_"+\
                    str(self.params.time_bin_size)+\
                                       "s_time_bin_"+str(co_firing_window_size)+"s_window"

        saving_dir = self.params.pre_proc_dir+"co_firing_matrices/" + data_name +"/"

        # --------------------------------------------------------------------------------------------------------------
        # check if requested data range is already contained in existing files
        # --------------------------------------------------------------------------------------------------------------
        # function used to sort file entries numerically
        numbers = re.compile(r'(\d+)')

        def numericalSort(value):
            parts = numbers.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts

        co_firing_matrices = None

        for i, infile in enumerate(sorted(glob.glob(saving_dir + "*"), key=numericalSort)):
            m = re.findall('\((.*?)\)', infile)
            m = np.array(m[0].split(",")).astype("int")
            # sliding window --> need to shorten actual range of string
            if m[0] > (co_firing_window_size/self.params.time_bin_size-1):
                m[0] = m[0] - (co_firing_window_size / self.params.time_bin_size - 1)
            m[1] = m[1] - (co_firing_window_size/self.params.time_bin_size-1)

            if m[0] <= sel_range[0] < m[1] and m[0] <= sel_range[-1] < m[1]:
                # all data is contained in one file
                dat_from_file = pickle.load(open(infile, "rb"))
                co_firing_matrices = dat_from_file[(sel_range[0]-m[0]):(sel_range[-1]-m[0]+1), :, :]
                break
            elif m[0] <= sel_range[0] < m[1]:
                # where does the range start
                dat_from_file = pickle.load(open(infile, "rb"))
                co_firing_matrices = dat_from_file[(sel_range[0] - m[0]):, :, :]
                print("TO BE CORRECTED - DOES NOT GENERATE THE EXACT DATA")
            elif m[0] <= sel_range[-1] < m[1]:
                dat_from_file = pickle.load(open(infile, "rb"))
                co_firing_matrices = np.concatenate((co_firing_matrices, dat_from_file[:(sel_range[-1]-m[0]+1), :, :]),
                axis=0)
            elif sel_range[0] < m[0] and m[1] < sel_range[-1]:
                # in this case the entire file needs to be added
                dat_from_file = pickle.load(open(infile, "rb"))
                co_firing_matrices = np.concatenate(
                    (co_firing_matrices, dat_from_file),axis=0)

        if co_firing_matrices is None:
            # if data could not be extracted from existing files --> need to create correlation matrices

            co_firing_matrices_file_name = self.params.session_name+"_"+str(self.params.time_bin_size)+\
                                           "s_time_bin_"+str(co_firing_window_size)+"s_window_"+str(sel_range)

            # compute how many time bins fit in one window
            time_bins_per_co_firing_matrix = int(co_firing_window_size/self.params.time_bin_size)

            # if directory does not exist --> create it
            if not os.path.isdir(saving_dir):
                os.mkdir(saving_dir)

            if not os.path.isfile(saving_dir + co_firing_matrices_file_name):

                # generate matrices
                co_firing_matrices = self.dynamic_co_firing(x, y, time_bins_per_co_firing_matrix)

                # save first dictionary as pickle
                filename = saving_dir + co_firing_matrices_file_name
                outfile = open(filename, 'wb')
                pickle.dump(co_firing_matrices, outfile)
                outfile.close()

            # if dictionary exists --> return
            else:
                co_firing_matrices = pickle.load(open(saving_dir + co_firing_matrices_file_name, "rb"))

        return co_firing_matrices

    def concatenate_co_firing_matrices(self, co_firing_window_size):
        # --------------------------------------------------------------------------------------------------------------
        # concatenate co firing matrices
        #
        # args: - co_firing_window_size, float: window size in seconds
        #
        # --------------------------------------------------------------------------------------------------------------

        # function used to sort file entries numerically
        numbers = re.compile(r'(\d+)')

        def numericalSort(value):
            parts = numbers.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts

        data_name = self.params.session_name + "_" + str(self.params.time_bin_size) + \
                    "s_time_bin_" + str(co_firing_window_size) + "s_window"

        saving_dir = self.params.pre_proc_dir+"co_firing_matrices/" + data_name + "/"

        # if concatenated file exists already --> delete it and create it again
        for file in os.listdir(saving_dir):
            if file.endswith("_concat"):
                os.remove(saving_dir + file)
                print("DELETED PREVIOUS VERSION OF CONCATENATED CORRELATION MATRICES")

        first_file = True
        print("GENERATING CONCATENATED ARRAY ...")
        for infile in sorted(glob.glob(saving_dir + "*"), key=numericalSort):
            print(" ADDING THE FOLLOWING FILE: " + infile)
            try:
                new_data = pickle.load(open(infile, "rb")).astype("float16")
                # if first file: check size of matrices
                if first_file:
                    all_data_array = np.zeros((0, new_data.shape[1], new_data.shape[2])).astype("float16")
                    first_file = False
                all_data_array = np.concatenate((all_data_array, new_data), axis=0)
            except:
                break

        print(all_data_array.shape)
        filename = saving_dir + data_name + "_concat"
        outfile = open(filename, 'wb')
        pickle.dump(all_data_array, outfile)
        outfile.close()

    def visualize_dynamic_correlation_values(self, co_firing_window_size=1, clustering=True, sel_range=None,
                                             nr_kmeans_clusters=10, remove_hse=False):
        # --------------------------------------------------------------------------------------------------------------
        # visualize correlation matrices
        #
        # args: - co_firing_window_size, float: window size in seconds
        #
        # --------------------------------------------------------------------------------------------------------------

        x = self.raster_list[0]
        nr_cells_x = x.shape[0]
        y = self.raster_list[1]

        if sel_range is not None:

            x = x[:, sel_range]
            y = y[:, sel_range]

        if remove_hse:
            ind_hse_x = np.array(find_hse(x=x)).flatten()
            ind_hse_y = np.array(find_hse(x=y)).flatten()

            ind_hse = np.unique(np.hstack((ind_hse_x, ind_hse_y)))

            # remove high synchrony events
            x = np.delete(x, ind_hse, axis=1)
            y = np.delete(y, ind_hse, axis=1)

        # stack up for full correlation matrix
        x = np.vstack((x, y))

        # compute how many time bins fit in one window
        time_bins_per_co_firing_matrix = int(co_firing_window_size / self.params.time_bin_size)

        co_firing_matrices = self.dynamic_co_firing(time_bins_per_co_firing_matrix=time_bins_per_co_firing_matrix,
                                                    raster=x)


        co_fir_mat_flat = []
        co_fir_within1_flat = []
        co_fir_within2_flat = []
        co_fir_across_flat = []

        ind_matrix = np.indices((co_firing_matrices[0].shape[0],co_firing_matrices[0].shape[1]))

        first_ind = upper_tri_without_diag(ind_matrix[0,:,:]).flatten()
        second_ind = upper_tri_without_diag(ind_matrix[1,:,:]).flatten()
        ind_array = np.vstack((first_ind, second_ind)).T

        # across: 0
        disting_temp = np.zeros((co_firing_matrices[0].shape[0],co_firing_matrices[0].shape[1]))
        # within first: 1
        disting_temp[:nr_cells_x+1, :nr_cells_x+1] = 1
        # within second: 2
        disting_temp[(nr_cells_x+1):, (nr_cells_x+1):] = 2
        disting_temp_flat = upper_tri_without_diag(disting_temp).flatten()

        for mat in co_firing_matrices:
            co_fir_mat_flat.append(upper_tri_without_diag(mat).flatten())
            co_fir_within1_flat.append(upper_tri_without_diag(mat[:nr_cells_x+1, :nr_cells_x+1]).flatten())
            co_fir_within2_flat.append(upper_tri_without_diag(mat[(nr_cells_x+1):, (nr_cells_x+1):]).flatten())
            co_fir_across_flat.append(mat[:(nr_cells_x+1), (nr_cells_x+1):].flatten())


        co_fir_mat_flat = np.array(co_fir_mat_flat).T
        co_fir_within1_flat = np.array(co_fir_within1_flat).T
        co_fir_within2_flat = np.array(co_fir_within2_flat).T
        co_fir_across_flat = np.array(co_fir_across_flat).T

        data_to_vis = co_fir_mat_flat

        if clustering:
            from sklearn.cluster import KMeans
            nr_clusters = nr_kmeans_clusters
            kmeans = KMeans(n_clusters=nr_clusters, random_state=0).fit(data_to_vis)
            a = np.argsort(kmeans.labels_)
            data_to_vis = data_to_vis[a,:]
            disting_temp_flat = disting_temp_flat[a]
            labels_sorted = kmeans.labels_[a]
            ind_array_sorted = ind_array[a,:]

            cluster_ass_cell = np.zeros((nr_clusters, np.unique(ind_array_sorted).shape[0]))

            cluster_ass = np.zeros((nr_clusters, 3))
            for cl_id in range(nr_clusters):
                ind_cl = np.argwhere(labels_sorted == cl_id)
                cluster_ass[cl_id, 0] = np.sum(disting_temp_flat[ind_cl] == 0)
                cluster_ass[cl_id, 1] = np.sum(disting_temp_flat[ind_cl] == 1)
                cluster_ass[cl_id, 2] = np.sum(disting_temp_flat[ind_cl] == 2)
                for cell_id in ind_array_sorted[ind_cl]:
                    cluster_ass_cell[cl_id, cell_id[0][0]] += 1
                    cluster_ass_cell[cl_id, cell_id[0][1]] += 1

            cluster_ass_cell_clust_norm = cluster_ass_cell/np.sum(cluster_ass_cell, axis=1)[:,None]
            cluster_ass_cell_cell_norm = cluster_ass_cell/np.sum(cluster_ass_cell, axis=0)

            plt.imshow(cluster_ass_cell_clust_norm*100, interpolation='nearest', aspect='auto')
            a = plt.colorbar()
            a.set_label("CELL CONTRIBUTION PER CLUSTER / %")
            plt.xlabel("CELL ID")
            plt.ylabel("CLUSTER ID")
            plt.show()
            #
            # plt.scatter(range(np.unique(ind_array_sorted).shape[0]), cluster_ass_cell[0,:],  marker=".")
            # plt.scatter(range(np.unique(ind_array_sorted).shape[0]), cluster_ass_cell[1, :], marker=".")
            # plt.scatter(range(np.unique(ind_array_sorted).shape[0]), cluster_ass_cell[2, :], marker=".")
            # plt.scatter(range(np.unique(ind_array_sorted).shape[0]), cluster_ass_cell[3, :], marker=".")
            # plt.scatter(range(np.unique(ind_array_sorted).shape[0]), cluster_ass_cell[4, :], marker=".")
            # plt.show()
            # exit()

            # for i in range(380):
            #     plt.scatter(range(nr_clusters), cluster_ass_cell_cell_norm[:,i])
            #     # plt.hist(cluster_ass_cell_cell_norm[:,i], density=True)
            # plt.show()

            from matplotlib import cm
            c_map = cm.get_cmap("tab20", nr_clusters)
            c_map = c_map(np.linspace(0,1,nr_clusters))
            from matplotlib.colors import ListedColormap, LinearSegmentedColormap
            newcmp = ListedColormap(c_map)

            for i in range(nr_clusters):
                plt.hist(cluster_ass_cell_clust_norm[i,:], density=True, edgecolor=c_map[i], fill=False)
            plt.show()

            cluster_ass /= np.sum(cluster_ass, axis=0)
            cluster_ass *= 100
            labels = np.arange(nr_clusters)
            print(cluster_ass)
            x = np.arange(len(labels))  # the label locations
            width = 0.2  # the width of the bars
            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width-0.02, cluster_ass[:, 0], width, label='across', hatch="\\", color=c_map, edgecolor="w")
            rects2 = ax.bar(x, cluster_ass[:, 1], width, label='within 1', hatch="o", color=c_map, edgecolor="w")
            rects3 = ax.bar(x + width+0.02, cluster_ass[:, 2], width, label='within 2', hatch="+", color=c_map, edgecolor="w")
            plt.legend()
            plt.xlabel("CLUSTER ID (K-MEANS)")
            plt.ylabel("CLUSTER ASSIGNEMENT (%)")
            plt.show()

            labels_sorted = np.expand_dims(labels_sorted, axis=1)
            fig = plt.figure(figsize=[10,5])
            gs = fig.add_gridspec(6, 20)

            ax1 = fig.add_subplot(gs[:,:1])
            ax1.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off
            ax1.get_yaxis().set_ticks([])
            ax1.set_ylabel("CLUSTER ID")
            ax1.imshow(labels_sorted, aspect='auto', cmap=newcmp, vmin=0, vmax=nr_clusters)

            ax2 = fig.add_subplot(gs[:,1:])
            # ax2.yaxis.tick_right(rotation='vertical')
            ax2.get_yaxis().set_ticks([])
            ax2.yaxis.set_label_position("right")

            im = ax2.imshow(data_to_vis, interpolation='nearest', aspect='auto',
                       extent=[sel_range[0]*self.params.time_bin_size,
                               (sel_range[0]+co_fir_mat_flat.shape[1])*self.params.time_bin_size,
                               co_fir_mat_flat.shape[0],0])
            cbar = fig.colorbar(im, ax=ax2)
            cbar.set_label('PEARSON CORRELATION')
            ax2.set_ylabel("EDGES (#:"+str(data_to_vis.shape[0])+")")
            ax2.set_xlabel("START SLIDING WINDOW (s) - " +str(co_firing_window_size)+"s WINDOW, TIME BIN: "+
                           str(self.params.time_bin_size)+"s")

            # ax2.colorbar()
            plt.show()

        else:
            plt.imshow(data_to_vis, interpolation='nearest', aspect='auto')
            plt.colorbar()
            plt.show()

    def subsequent_co_firing(self, co_firing_window_size=1, sel_range=None):
        # --------------------------------------------------------------------------------------------------------------
        # visualization of difference (Frobenius norm) between subsequent co-firing matrices (sliding window)
        #
        # args: - co_firing_window_size: length of window in seconds for which correlation values are computed
        # --------------------------------------------------------------------------------------------------------------

        x = self.raster_list[0]
        nr_cells_x = x.shape[0]
        y = self.raster_list[1]

        if sel_range is not None:

            x = x[:, sel_range]
            y = y[:, sel_range]

        # stack up for full correlation matrix
        x = np.vstack((x, y))

        # compute how many time bins fit in one window
        time_bins_per_co_firing_matrix = int(co_firing_window_size / self.params.time_bin_size)

        correlation_matrices = self.dynamic_co_firing(time_bins_per_co_firing_matrix=time_bins_per_co_firing_matrix,
                                                        raster=x)

        # calculate similarity between subsequent co-firing matrices
        dist_mat = correlation_matrices[1:, :, :] - correlation_matrices[:-1, :, :]

        dist_val = np.zeros(dist_mat.shape[0])
        within_1_dist = np.zeros(dist_mat.shape[0])
        within_2_dist = np.zeros(dist_mat.shape[0])
        across_dist = np.zeros(dist_mat.shape[0])

        for i, m in enumerate(dist_mat):
            dist_val[i] = np.linalg.norm(m)
            # separate within and across values
            # ----------------------------------------------------------------------------------------------------------
            within_1_dist[i] = np.linalg.norm(upper_tri_without_diag(m[:nr_cells_x+1, :nr_cells_x+1]).flatten())
            within_2_dist[i] = np.linalg.norm(upper_tri_without_diag(m[(nr_cells_x+1):, (nr_cells_x+1):]).flatten())
            across_dist[i] = np.linalg.norm(m[:(nr_cells_x+1), (nr_cells_x+1):].flatten())

        within_1_dist_normalized = within_1_dist/np.mean(within_1_dist)
        within_2_dist_normalized = within_2_dist/np.mean(within_2_dist)
        across_dist_normalized = across_dist/np.mean(across_dist)

        # compute how many time bins fit in one window
        time_bins_per_co_firing_matrix = int(co_firing_window_size/self.params.time_bin_size)

        plt.figure(figsize=[9, 6])
        # separate within and across values
        # --------------------------------------------------------------------------------------------------------------
        plt.plot(self.params.time_bin_size*np.arange(len(dist_val)), within_1_dist_normalized,
                 label="WITHIN "+ self.cell_type_array[0], c="#990000")
        plt.plot(self.params.time_bin_size * np.arange(len(dist_val)), within_2_dist_normalized,
                 label="WITHIN "+self.cell_type_array[1], c="#0099cc")
        plt.plot(self.params.time_bin_size * np.arange(len(dist_val)), across_dist_normalized,
                 label="ACROSS", linestyle=":", color="white")
        plt.title("CHANGE IN COFIRING MATRICES - TIME BIN: "+
                  str(self.params.time_bin_size)+"s, WINDOW SIZE: "+
                      str(self.params.time_bin_size*time_bins_per_co_firing_matrix)+"s")
        plt.ylabel("||COFIRING_MAT(t2)-COFIRING_MAT(t1)|| - NORMALIZED")
        plt.xlabel("BEGINNING OF WINDOW / SEC")
        plt.legend()
        plt.show()

        # cross correlate values

        shift_array = np.arange(-5, 5)

        corr = cross_correlate(x=within_1_dist_normalized, y=within_2_dist_normalized, shift_array=shift_array)

        plt.plot(shift_array, corr, "-o", color="red")
        plt.title("CROSS-CORRELATION OF CORR. MAT. CHANGES\n FROM BOTH POPULATION")
        plt.ylabel("PEARSON CORR. COEFF.")
        # plt.ylim(0.0, 1.0)
        plt.xlabel("TIME BIN SHIFT - CENTERED ON " + self.cell_type_array[0])
        plt.grid()

        plt.show()

    def subsequent_co_firing_sparsity(self, co_firing_window_size=1, sel_range=None, corr_thres=0.3):
        # --------------------------------------------------------------------------------------------------------------
        # visualization of difference (Frobenius norm) between subsequent co-firing matrices (sliding window)
        #
        # args: - co_firing_window_size: length of window in seconds for which correlation values are computed
        #       - sel_range, range: which data to use
        #       - corr_thres, float: which correlation values to set to zero
        # --------------------------------------------------------------------------------------------------------------

        x = self.raster_list[0]
        nr_cells_x = x.shape[0]
        y = self.raster_list[1]
        nr_cells_y = y.shape[0]

        if sel_range is not None:

            x = x[:, sel_range]
            y = y[:, sel_range]

        # stack up for full correlation matrix
        x = np.vstack((x, y))

        # compute how many time bins fit in one window
        time_bins_per_co_firing_matrix = int(co_firing_window_size / self.params.time_bin_size)

        correlation_matrices = self.dynamic_co_firing(time_bins_per_co_firing_matrix=time_bins_per_co_firing_matrix,
                                                        raster=x)

        within_1_sparsity = np.zeros(correlation_matrices.shape[0])
        within_2_sparsity = np.zeros(correlation_matrices.shape[0])
        across_sparsity = np.zeros(correlation_matrices.shape[0])

        for i, m in enumerate(correlation_matrices):
            # set threshold for correlation values to compute sparsity
            # plt.imshow(m, interpolation='nearest', aspect=_'auto')
            # plt.colorbar()
            # plt.show()
            m[np.abs(m) < corr_thres] = 0
            # plt.imshow(m, interpolation='nearest', aspect='auto')
            # plt.colorbar()
            # plt.show()
            # exit()
            # separate within and across values
            # ----------------------------------------------------------------------------------------------------------
            within_1_sparsity[i] = 1-np.count_nonzero(upper_tri_without_diag(
                m[:nr_cells_x+1, :nr_cells_x+1]))/(nr_cells_x**2-nr_cells_x)
            within_2_sparsity[i] = 1-np.count_nonzero(upper_tri_without_diag(
                m[(nr_cells_x+1):, (nr_cells_x+1):]))/(nr_cells_y**2-nr_cells_y)
            # across_sparsity[i] = m[:(nr_cells_x+1), (nr_cells_x+1):]

        # compute how many time bins fit in one window
        time_bins_per_co_firing_matrix = int(co_firing_window_size/self.params.time_bin_size)

        plt.figure(figsize=[9, 6])
        # separate within and across values
        # --------------------------------------------------------------------------------------------------------------
        plt.plot(self.params.time_bin_size*np.arange(within_1_sparsity.shape[0]), within_1_sparsity,
                 label="WITHIN "+ self.cell_type_array[0], c="#990000")
        plt.plot(self.params.time_bin_size * np.arange(within_2_sparsity.shape[0]), within_2_sparsity,
                 label="WITHIN "+self.cell_type_array[1], c="#0099cc")
        # plt.plot(self.params.time_bin_size * np.arange(len(dist_val)), across_dist_normalized,
        #          label="ACROSS", linestyle=":", color="white")
        plt.title("SPARSITY OF CORRELATION MATRICES - TIME BIN: "+
                  str(self.params.time_bin_size)+"s, WINDOW SIZE: "+
                      str(self.params.time_bin_size*time_bins_per_co_firing_matrix)+"s")
        plt.ylabel("SPARSITY")
        plt.xlabel("BEGINNING OF WINDOW / SEC")
        plt.legend()
        plt.show()

        # cross correlate values

        shift_array = np.arange(-5, 5)

        corr = cross_correlate(x=within_1_sparsity, y=within_2_sparsity, shift_array=shift_array)

        plt.plot(shift_array, corr, "-o", color="red")
        plt.title("CROSS-CORRELATION OF SPARSITY VALUES\n FROM BOTH POPULATION")
        plt.ylabel("PEARSON CORR. COEFF.")
        # plt.ylim(0.0, 1.0)
        plt.xlabel("TIME BIN SHIFT - CENTERED ON " + self.cell_type_array[0])
        plt.grid()

        plt.show()

    def dynamic_co_firing_mds(self, co_firing_window_size=1, distance_measure="frob", sliding_window=True,
                              sel_range_1=None, sel_range_2=None, sel_range_3=None, video=False):
        # --------------------------------------------------------------------------------------------------------------
        # visualization of co-firing matrices
        #
        # args:     - co_firing_window_size: length of window in seconds for which correlation values are computed
        #           - sel_range: for which range to compute MDS
        #           - distance_measure: which distance measure to use for correlation matrices
        #                                   - frobenius:    subtract one correlation matrix from the other and use
        #                                                   Frobenius norm of difference matrix
        #
        #
        # --------------------------------------------------------------------------------------------------------------

        # compute how many time bins fit in one window
        time_bins_per_co_firing_matrix = int(co_firing_window_size / self.params.time_bin_size)

        data_sep = None
        data_sep_2 = None

        x = self.raster_list[0]
        y = self.raster_list[1]

        # # stack up for full correlation matrix
        raster = np.vstack((x, y))

        if sel_range_1 is not None:

            raster_chunk = raster[:, sel_range_1]

            correlation_matrices = self.dynamic_co_firing(raster=raster_chunk, sliding_window=sliding_window,
                                                        time_bins_per_co_firing_matrix=time_bins_per_co_firing_matrix)
        if sel_range_2 is not None:

            # length of existing data set:
            data_sep = correlation_matrices.shape[0]

            raster_chunk = raster[:, sel_range_2]

            correlation_matrices_2 = self.dynamic_co_firing(raster=raster_chunk, sliding_window=sliding_window,
                                                          time_bins_per_co_firing_matrix=time_bins_per_co_firing_matrix)

            correlation_matrices = np.concatenate((correlation_matrices, correlation_matrices_2), axis=0)

        if sel_range_3 is not None:
            # length of existing data set:
            data_sep_2 = correlation_matrices.shape[0]

            raster_chunk = raster[:, sel_range_3]

            correlation_matrices_3 = self.dynamic_co_firing(raster=raster_chunk, sliding_window=sliding_window,
                                                            time_bins_per_co_firing_matrix=time_bins_per_co_firing_matrix)

            correlation_matrices = np.concatenate((correlation_matrices, correlation_matrices_3), axis=0)

        dist = np.zeros((correlation_matrices.shape[0], correlation_matrices.shape[0]))

        # to apply MDS for visualization --> calculate frobenius norm of differences between all matrices
        for i, template in enumerate(correlation_matrices):
            # choose one co-firing matrix as template and compare to all the others
            # one_left_co_firing_matrix = np.delete(co_firing_matrix, i, axis=0)
            # print(str(i / co_firing_matrix.shape[0] * 100) + "%")
            for e, compare in enumerate(correlation_matrices):
                # dist[i, e] = abs(np.linalg.norm(template)-np.linalg.norm(compare))
                if distance_measure == "frob":
                    dist[i, e] = np.linalg.norm(template - compare)
                elif distance_measure == "graph":
                    dist[i, e] = graph_distance(template, compare)

        # outfile = open("dist", "wb")
        # pickle.dump(dist, outfile)
        # outfile.close()

        # infile = open("dist", 'rb')
        # dist = pickle.load(infile)
        # infile.close()

        model = MDS(n_components=self.params.dr_method_p2, dissimilarity='precomputed', random_state=1)
        result = model.fit_transform(dist)

        if video:

            scatter_animation(result, self.params, data_sep)

        else:
            if self.params.dr_method_p2 == 3:
                # create figure instance
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plot_3D_scatter(ax=ax, mds=result, params=self.params, data_sep=data_sep, data_sep2=data_sep_2)
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plot_2D_scatter(ax=ax, mds=result, params=self.params, data_sep=data_sep, data_sep2=data_sep_2)
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            plt.show()

    def dynamic_co_firing_mds_parallel(self, sel_range_1, co_firing_window_size=1, distance_measure="frob",
                                       sliding_window=True, video_file_name=None):
        # --------------------------------------------------------------------------------------------------------------
        # visualization of co-firing matrices of two populations next to each other
        #
        # args:     - co_firing_window_size: length of window in seconds for which correlation values are computed
        #           - sel_range: for which range to compute MDS
        #           - distance_measure: which distance measure to use for correlation matrices
        #                                   - frobenius:    subtract one correlation matrix from the other and use
        #                                                   Frobenius norm of difference matrix
        #
        #
        # --------------------------------------------------------------------------------------------------------------

        print(" - DYNAMIC CORRELATION ANALYSIS: PARALLEL MDS ...\n")

        # compute how many time bins fit in one window
        time_bins_per_co_firing_matrix = int(co_firing_window_size / self.params.time_bin_size)

        x = self.raster_list[0]
        y = self.raster_list[1]


        raster_chunk_pop_1 = x[:, sel_range_1]
        raster_chunk_pop_2 = y[:, sel_range_1]

        correlation_matrices_1 = self.dynamic_co_firing(raster=raster_chunk_pop_1, sliding_window=sliding_window,
                                                    time_bins_per_co_firing_matrix=time_bins_per_co_firing_matrix)

        correlation_matrices_2 = self.dynamic_co_firing(raster=raster_chunk_pop_2, sliding_window=sliding_window,
                                                    time_bins_per_co_firing_matrix=time_bins_per_co_firing_matrix)


        dist_1 = np.zeros((correlation_matrices_1.shape[0], correlation_matrices_1.shape[0]))
        dist_2 = np.zeros((correlation_matrices_2.shape[0], correlation_matrices_2.shape[0]))

        # to apply MDS for visualization --> calculate frobenius norm of differences between all matrices
        for i, (template_1, template_2) in enumerate(zip(correlation_matrices_1, correlation_matrices_2)):
            # choose one co-firing matrix as template and compare to all the others
            # one_left_co_firing_matrix = np.delete(co_firing_matrix, i, axis=0)
            # print(str(i / co_firing_matrix.shape[0] * 100) + "%")
            for e, (compare_1, compare_2) in enumerate(zip(correlation_matrices_1, correlation_matrices_2)):
                # dist[i, e] = abs(np.linalg.norm(template)-np.linalg.norm(compare))
                if distance_measure == "frob":
                    dist_1[i, e] = np.linalg.norm(template_1 - compare_1)
                    dist_2[i, e] = np.linalg.norm(template_2 - compare_2)


        model = MDS(n_components=self.params.dr_method_p2, dissimilarity='precomputed', random_state=1)
        result_1 = model.fit_transform(dist_1)
        result_2 = model.fit_transform(dist_2)

        scatter_animation_parallel(results_1=result_1, results_2=result_2, params=self.params,
                                   video_file_name=video_file_name)

    def dynamic_co_firing_mds_load_and_plot(self, input_file_name):
        data = pickle.load(open(input_file_name, "rb"))
        # max_steps = 20500
        #
        # data = a[:max_steps, :max_steps]

        mds_res = multi_dim_scaling(data, self.params)

        # generate plots
        if self.params.dr_method_p2 == 3:
            # create figure instance
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plot_3D_scatter(ax, mds_res, self.params, None)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot_2D_scatter(ax, mds_res, self.params, None)
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()

    def video_co_firing_matrices(self, co_firing_window_size, sorting=None, save_file=False, saving_dir="movies",
                                 video_file_name=None, sel_range=None, remove_hse=False):
        # --------------------------------------------------------------------------------------------------------------
        # create video of co-firing matrices
        #
        #
        # args:   - co_firing_window_size: length of window in seconds for which correlation values are computed
        #
        #
        #
        # --------------------------------------------------------------------------------------------------------------

        # compute how many time bins fit in one window
        time_bins_per_co_firing_matrix = int(co_firing_window_size / self.params.time_bin_size)

        x = self.raster_list[0]
        y = self.raster_list[1]

        if sel_range is not None:

            x = x[:, sel_range]
            y = y[:, sel_range]

        if remove_hse:
            ind_hse_x = np.array(find_hse(x=x)).flatten()
            ind_hse_y = np.array(find_hse(x=y)).flatten()

            ind_hse = np.unique(np.hstack((ind_hse_x, ind_hse_y)))

            # remove high synchrony events
            x = np.delete(x, ind_hse, axis=1)
            y = np.delete(y, ind_hse, axis=1)

        # stack up for full correlation matrix
        x = np.vstack((x, y))

        # get number of cells in population 1
        nr_cells_x = self.raster_list[0].shape[0]

        co_firing_matrices = self.dynamic_co_firing(raster=x,
                                                    time_bins_per_co_firing_matrix=time_bins_per_co_firing_matrix)

        # order co-firing matrix elements for better visualization
        if sorting == "sort_cells":
            # order according to mean value of each cell --> keep cell pairings
            # cells with maximum correlation values are shifted to bottom-right
            m = np.mean(co_firing_matrices, axis=0)
            col_mean = np.mean(m, axis=0)
            row_mean = np.mean(m, axis=1)
            col_sort_ind = np.argsort(col_mean)
            row_sort_ind = np.argsort(row_mean)

            for i, mat in enumerate(co_firing_matrices):
                temp = mat[row_sort_ind, :]
                temp = temp[:, col_sort_ind]
                co_firing_matrices[i, :, :] = temp

        if sorting == "sort_edges":
            # matrix is sorted using the values of single entries --> cell pairs are left
            # entries ("connections") are moved to bottom-right according to their mean/std value
            # --> bottom-right: high mean correlation value and low std value
            # --> top-left: low (negative) correlation values and low std value
            # --> center: small correlations/high std values
            m = np.mean(co_firing_matrices, axis=0)/(np.std(co_firing_matrices, axis=0)+10e-9)

            # m = np.mean(co_firing_matrices, axis=0)

            row_sort_list = []
            col_sort_list = []

            # find correct indices
            for i in range(m.shape[0]):
                row_sort_ind = np.argsort(m[i, :])
                row_sort_list.append(row_sort_ind)
                m[i, :] = m[i, row_sort_ind]

            row_ind = np.stack(row_sort_list, axis=0)

            for i in range(m.shape[1]):
                col_sort_ind = np.argsort(m[:, i])
                col_sort_list.append(col_sort_ind)

            col_ind = np.stack(col_sort_list, axis=0)

            for time_bin, mat in enumerate(co_firing_matrices):
                # go through rows and sort
                for i in range(m.shape[0]):
                    row_sort_ind = row_ind[i,:]
                    co_firing_matrices[time_bin, i, :] = co_firing_matrices[time_bin, i, row_sort_ind]

                # go through columns and sort
                for i in range(m.shape[1]):
                    col_sort_ind = col_ind[i,:]
                    co_firing_matrices[time_bin, :, i] = co_firing_matrices[time_bin, col_sort_ind, i]

        if video_file_name:
            view_dyn_co_firing(co_firing_matrices, sorting, co_firing_window_size, self.params.time_bin_size,
                               nr_cells_x, save_file, saving_dir, video_file_name)
        else:
            view_dyn_co_firing(co_firing_matrices, sorting, co_firing_window_size, self.params.time_bin_size,
                               nr_cells_x, save_file)

    def predict_time_from_correlations(self, co_firing_window_size=10, sel_range=None):
        # --------------------------------------------------------------------------------------------------------------
        # visualization of difference (Frobenius norm) between subsequent co-firing matrices (sliding window)
        #
        # args: - co_firing_window_size: length of window in seconds for which correlation values are computed
        # --------------------------------------------------------------------------------------------------------------

        corr_mat = self.dynamic_co_firing(sliding_window=False)

        new_ml = MlMethodsTwoPopulations(self.raster_list)
        new_ml.ridge_time_from_correlations(correlation_matrices=corr_mat)


        exit()
        x = self.raster_list[0]
        nr_cells_x = x.shape[0]
        y = self.raster_list[1]

        if sel_range is not None:

            x = x[:, sel_range]
            y = y[:, sel_range]

        # stack up for full correlation matrix
        x = np.vstack((x, y))
        y = np.copy(x)

        # get correlation matrices
        co_firing_matrices = self.load_or_create_dynamic_co_firing(x, y, co_firing_window_size, sel_range)

        new_ml = MlMethodsTwoPopulations(self.raster_list)
        new_ml.ridge_time_from_correlations(correlation_matrices=co_firing_matrices)

        exit()
        # calculate similarity between subsequent co-firing matrices
        dist_mat = co_firing_matrices[1:, :, :] - co_firing_matrices[:-1, :, :]

        dist_val = np.zeros(dist_mat.shape[0])
        within_1_dist = np.zeros(dist_mat.shape[0])
        within_2_dist = np.zeros(dist_mat.shape[0])
        across_dist = np.zeros(dist_mat.shape[0])
        for i, m in enumerate(dist_mat):
            dist_val[i] = np.linalg.norm(m)
            # separate within and across values
            # ----------------------------------------------------------------------------------------------------------
            within_1_dist[i] = np.linalg.norm(upper_tri_without_diag(m[:nr_cells_x+1, :nr_cells_x+1]).flatten())
            within_2_dist[i] = np.linalg.norm(upper_tri_without_diag(m[(nr_cells_x+1):, (nr_cells_x+1):]).flatten())
            across_dist[i] = np.linalg.norm(m[:(nr_cells_x+1), (nr_cells_x+1):].flatten())

    """#################################################################################################################
    #   HSE analysis
    #################################################################################################################"""

    def visualize_hse(self, co_firing_window_size=1, clustering=True, sel_range=None,
                                             nr_kmeans_clusters=10):
        # --------------------------------------------------------------------------------------------------------------
        # visualize HSE
        #
        # args: - co_firing_window_size, float: window size in seconds
        #
        # --------------------------------------------------------------------------------------------------------------

        x = self.raster_list[0]
        nr_cells_x = x.shape[0]
        y = self.raster_list[1]

        if sel_range is not None:

            x = x[:, sel_range]
            y = y[:, sel_range]

        # stack up for full correlation matrix
        x = np.vstack((x, y))

        # compute how many time bins fit in one window
        time_bins_per_co_firing_matrix = int(co_firing_window_size / self.params.time_bin_size)

        co_firing_matrices = self.dynamic_co_firing(time_bins_per_co_firing_matrix=time_bins_per_co_firing_matrix,
                                                    raster=x)

        co_fir_mat_flat = []
        co_fir_within1_flat = []
        co_fir_within2_flat = []
        co_fir_across_flat = []

        ind_matrix = np.indices((co_firing_matrices[0].shape[0],co_firing_matrices[0].shape[1]))

        first_ind = upper_tri_without_diag(ind_matrix[0,:,:]).flatten()
        second_ind = upper_tri_without_diag(ind_matrix[1,:,:]).flatten()
        ind_array = np.vstack((first_ind, second_ind)).T

        # across: 0
        disting_temp = np.zeros((co_firing_matrices[0].shape[0],co_firing_matrices[0].shape[1]))
        # within first: 1
        disting_temp[:nr_cells_x+1, :nr_cells_x+1] = 1
        # within second: 2
        disting_temp[(nr_cells_x+1):, (nr_cells_x+1):] = 2
        disting_temp_flat = upper_tri_without_diag(disting_temp).flatten()

        for mat in co_firing_matrices:
            co_fir_mat_flat.append(upper_tri_without_diag(mat).flatten())
            co_fir_within1_flat.append(upper_tri_without_diag(mat[:nr_cells_x+1, :nr_cells_x+1]).flatten())
            co_fir_within2_flat.append(upper_tri_without_diag(mat[(nr_cells_x+1):, (nr_cells_x+1):]).flatten())
            co_fir_across_flat.append(mat[:(nr_cells_x+1), (nr_cells_x+1):].flatten())


        co_fir_mat_flat = np.array(co_fir_mat_flat).T

        plt.subplot(2,1,1)
        plt.imshow(co_fir_mat_flat, interpolation='nearest', aspect='auto')

        per_window_avg = np.mean(co_fir_mat_flat, axis=0)
        all_windows_std = np.std(per_window_avg)
        all_windows_mean = np.mean(per_window_avg)

        z_scored = abs(per_window_avg - all_windows_mean)/all_windows_std

        plt.subplot(2,1,2)
        plt.plot(z_scored)
        plt.show()

        # only select high synchrony events (> 1.5 std above the average correlation)
        corr_mat_flat_hse = co_fir_mat_flat[:, z_scored > 1.5]
        plt.imshow(corr_mat_flat_hse, interpolation='nearest', aspect='auto')
        plt.show()

        co_fir_within1_flat = np.array(co_fir_within1_flat).T
        co_fir_within2_flat = np.array(co_fir_within2_flat).T
        co_fir_across_flat = np.array(co_fir_across_flat).T

        data_to_vis = corr_mat_flat_hse.T

        from sklearn.cluster import KMeans
        nr_clusters = nr_kmeans_clusters
        kmeans = KMeans(n_clusters=nr_clusters, random_state=0).fit(data_to_vis)
        # a = np.argsort(kmeans.labels_)
        # data_to_vis = data_to_vis[a, :]
        # disting_temp_flat = disting_temp_flat[a]
        labels = kmeans.labels_
        # ind_array_sorted = ind_array[a, :]

        from matplotlib import cm
        c_map = cm.get_cmap("tab20", nr_clusters)
        c_map = c_map(np.linspace(0, 1, nr_clusters))
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        newcmp = ListedColormap(c_map)

        labels = np.expand_dims(labels, axis=1).T
        fig = plt.figure(figsize=[10, 5])
        gs = fig.add_gridspec(6, 20)

        ax1 = fig.add_subplot(gs[:1, :])
        ax1.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax1.get_yaxis().set_ticks([])
        ax1.set_xlabel("CLUSTER ID")
        ax1.xaxis.set_label_position('top')
        ax1.imshow(labels, aspect='auto', cmap=newcmp, vmin=0, vmax=nr_clusters)

        ax2 = fig.add_subplot(gs[1:, :])
        # ax2.yaxis.tick_right(rotation='vertical')
        ax2.get_yaxis().set_ticks([])
        ax2.yaxis.set_label_position("right")

        im = ax2.imshow(data_to_vis.T, interpolation='nearest', aspect='auto',
                        extent=[sel_range[0] * self.params.time_bin_size,
                                (sel_range[0] + co_fir_mat_flat.shape[1]) * self.params.time_bin_size,
                                co_fir_mat_flat.shape[0], 0])
        # cbar = fig.colorbar(im, ax=ax2, location='bottom')
        # cbar.set_label('PEARSON CORRELATION')
        ax2.set_ylabel("EDGES (#:" + str(data_to_vis.shape[1]) + ")")
        ax2.set_xlabel("#TIME BIN")

        # ax2.colorbar()
        plt.show()

    """#################################################################################################################
    #   dynamic correlation analysis (using sliding window) WITH time bin shift
    #################################################################################################################"""

    def find_best_time_shifted_correlation_dynamic(self, cell_type_input, cell_type_output,
                                                   time_bins_per_co_firing_matrix=20, full=False, sel_range=None):
        # --------------------------------------------------------------------------------------------------------------
        # calculates co-firing for each time bin --> each time bin is split into equal chunks of length time_bin_size
        # and the pearson correlation for all possible cell pairs is calculated --> the data is then shifted by n
        # time bins to find optimal correlation
        #
        # args:   - shift, integer: shift = 1 --> cell_type_output is one time bin behind
        #               - cell_type_input, str
        #               - cell_type_output, str
        #               - time_bins_per_co_firing_matrix, int:  how many time bins should be used to construct one
        #                                                       co-firing matrix (e.g. time_bin_size = 10 ms and
        #                                                       time_bins_per_co_firing_matrix = 20 --> one co-firing
        #                                                       matrix per 200ms
        # --------------------------------------------------------------------------------------------------------------

        # co-firing maps name
        co_firing_mat_name = self.params.pre_proc_dir+"co_firing_matrices/"+ self.params.session_name + "_" + \
                             str(self.params.time_bin_size)+"s"+"_dynamic"

        shift_array = np.arange(-3, 4)

        if not os.path.isfile(co_firing_mat_name):

            time_shifted_corr_matrices = []
            for shift in shift_array:
                time_shifted_corr_matrices.append(self.dynamic_co_firing_shift_within_time_bin(cell_type_input,
                                        cell_type_output, time_bins_per_co_firing_matrix, shift, full, sel_range))
            time_shifted_corr_matrices = np.array(time_shifted_corr_matrices)
            outfile = open(co_firing_mat_name, 'wb')
            pickle.dump(time_shifted_corr_matrices, outfile)
            outfile.close()

        # if dictionary exists --> return
        else:
            time_shifted_corr_matrices = pickle.load(open(co_firing_mat_name, "rb"))

        corr_avg = []
        for i, shift in enumerate(shift_array):
            corr_avg.append(np.mean(time_shifted_corr_matrices[i, :, :, :], axis=0))
            a = np.mean(time_shifted_corr_matrices[i, :, :, :], axis=0)
            # remove diagonal
            a = a[~np.eye(a.shape[0], dtype=bool)].reshape(a.shape[0], -1)
            plt.imshow(a, interpolation='nearest', aspect='auto', cmap="hot", vmin=0, vmax=1)
            plt.title(shift)

            plt.colorbar()
            plt.show()
        exit()

        best_shift_ind = np.argmax(time_shifted_corr_matrices, axis=0)

        # best_shift matrix
        best_shift_matrix = np.zeros(best_shift_ind.shape)

        for index, _ in np.ndenumerate(best_shift_matrix):
            best_shift_matrix[index] = shift_array[best_shift_ind[index]]

        plot_optimal_correlation_time_shift(best_shift_matrix, self.params, full)

    def dynamic_co_firing_shift_within_time_bin(self, cell_type_input, cell_type_output,
                    time_bins_per_co_firing_matrix=20, shift=None, full=False, sel_range=None):
        # --------------------------------------------------------------------------------------------------------------
        # calculates co-firing for each time bin --> each time bin is split into equal chunks of length time_bin_size
        # and the pearson correlation for all possible cell pairs is calculated
        #
        # args:   - shift, integer: shift = 1 --> cell_type_output is one time bin behind
        #               - cell_type_input, str
        #               - cell_type_output, str
        #               - time_bins_per_co_firing_matrix, int:  how many time bins should be used to construct one
        #                                                       co-firing matrix (e.g. time_bin_size = 10 ms and
        #                                                       time_bins_per_co_firing_matrix = 20 --> one co-firing
        #                                                       matrix per 200ms
        # --------------------------------------------------------------------------------------------------------------

        x = self.raster_list[self.cell_type_array.index(cell_type_input)]
        y = self.raster_list[self.cell_type_array.index(cell_type_output)]

        # remove cells that don't fire at all
        x = x[~np.all(x == 0, axis=1)]
        y = y[~np.all(y == 0, axis=1)]

        if sel_range is not None:
            x = x[:, sel_range]
            y = y[:, sel_range]

        if full:
            x = np.vstack((x, y))
            y = np.copy(x)

        co_firing_matrices = np.zeros((int(x.shape[1]/time_bins_per_co_firing_matrix), x.shape[0], y.shape[0]))

        # construct co_firing matrix: pearson correlation value between cells from both regions using multiple
        # time bins (time_bins_per_co_firing_matrix)

        for entry in range(int(x.shape[1]/time_bins_per_co_firing_matrix)):
            if shift:
                # shift x and y according to shift value
                if shift > 0:
                    for i, pop_vec_in in enumerate(x[:, entry * time_bins_per_co_firing_matrix + shift:
                    (entry + 1) * time_bins_per_co_firing_matrix]):
                        for j, pop_vec_out in enumerate(y[:, entry * time_bins_per_co_firing_matrix:
                        (entry + 1) * time_bins_per_co_firing_matrix - shift]):
                            co_firing_matrices[entry, i, j], _ = pearsonr(pop_vec_in, pop_vec_out)
                if shift < 0:
                    shift = abs(shift)
                    for i, pop_vec_in in enumerate(x[:, entry * time_bins_per_co_firing_matrix:
                    (entry + 1) * time_bins_per_co_firing_matrix - shift]):
                        for j, pop_vec_out in enumerate(y[:, entry * time_bins_per_co_firing_matrix + shift:
                        (entry + 1) * time_bins_per_co_firing_matrix]):
                            co_firing_matrices[entry, i, j], _ = pearsonr(pop_vec_in, pop_vec_out)
            else:
                # without shift --> simply calculate correlation
                for i, pop_vec_in in enumerate(x[:, entry * time_bins_per_co_firing_matrix:
                (entry+1)*time_bins_per_co_firing_matrix]):
                    for j, pop_vec_out in enumerate(y[:, entry * time_bins_per_co_firing_matrix:
                    (entry+1)*time_bins_per_co_firing_matrix]):
                        co_firing_matrices[entry, i, j], _ = pearsonr(pop_vec_in, pop_vec_out)

        # if one vector is constant (e.g. all zeros) --> pearsonr return np.nan
        # set all nans to zero
        co_firing_matrices = np.nan_to_num(co_firing_matrices)

        return co_firing_matrices

    """#################################################################################################################
    #   population vector analysis
    #################################################################################################################"""

    def pop_vec_distance(self, distance_measure):
        # --------------------------------------------------------------------------------------------------------------
        # computes distance between subsequent population vectors for each region separately and creates plot
        #
        # args:   - distance_measure, str: "L1, "euclidean", "cos"
        # --------------------------------------------------------------------------------------------------------------

        pop_vec_array = [None] * len(self.raster_list)

        for i, map in enumerate(self.raster_list):
            plt.subplot(2, 1, i + 1)
            dis, rel = pop_vec_dist(map, distance_measure)
            pop_vec_array[i] = dis
            plt.plot(dis)
            plt.title(str(self.cell_type_array[i]) + " - " + str(self.params.time_bin_size) + "s window")
            plt.ylabel("DIST. SUBS. POP. VEC - "+ distance_measure)
        plt.xlabel("BIN")
        plt.show()

        plt.xlabel("TIME")
        plt.show()

        corr, _ = pearsonr(np.nan_to_num(pop_vec_array[0]), np.nan_to_num(pop_vec_array[1]))

        plt.scatter(pop_vec_array[0], pop_vec_array[1])
        plt.ylabel("POP. VEC. DISTANCE - "+str(self.cell_type_array[1]))
        plt.xlabel("POP. VEC. DISTANCE - " + str(self.cell_type_array[0]))
        plt.title("CORR POP. VEC. DISTANCE - " + str(self.params.time_bin_size) + "s window" + " - PEARSON: "+str(corr))
        plt.show()

    """#################################################################################################################
    #   memory drift analysis
    #################################################################################################################"""

    def predict_avg_firing_pop_vec(self, new_time_bin_size=0.1):
        # --------------------------------------------------------------------------------------------------------------
        # predict firing average firing rate of one population using population vectors of the user population
        #
        # args:   - new_time_bin_size, float: time bin size to use for prediction
        # --------------------------------------------------------------------------------------------------------------

        print(" - PREDICTING AVG. FIRING RATE USING POP. VEC. ...")

        x = self.raster_list[0]
        y = self.raster_list[1]

        ind_hse_x = np.array(find_hse(x=x)).flatten()
        ind_hse_y = np.array(find_hse(x=y)).flatten()

        ind_hse = np.unique(np.hstack((ind_hse_x, ind_hse_y)))

        # remove high synchrony events
        x = np.delete(x, ind_hse, axis=1)
        y = np.delete(y, ind_hse, axis=1)

        # x = (x - np.min(x, axis=1, keepdims=True)) / \
        #     (np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True))

        # down/up sample data
        time_bin_scaler = int(new_time_bin_size / self.params.time_bin_size)

        new_raster_x = np.zeros((x.shape[0], int(x.shape[1] / time_bin_scaler)))
        new_raster_y = np.zeros((y.shape[0], int(y.shape[1] / time_bin_scaler)))

        # down sample spikes by combining multiple bins
        for i in range(new_raster_x.shape[1]):
            new_raster_x[:, i] = np.sum(x[:, (i * time_bin_scaler): ((1 + i) * time_bin_scaler)], axis=1)
            new_raster_y[:, i] = np.sum(y[:, (i * time_bin_scaler): ((1 + i) * time_bin_scaler)], axis=1)

        x = new_raster_x

        # compute average firing rate for y

        y = np.mean(new_raster_x, axis=0)

        new_ml = MlMethodsTwoPopulations()
        new_ml.ridge_avg_firing_rate(x=x, y=y, new_time_bin_size=new_time_bin_size, alpha=100)

    def analyse_poisson_hmm_fits(self, file_name_1, file_name_2):
        # --------------------------------------------------------------------------------------------------------------
        # fits poisson hmm to data
        #
        # args: - nr_clusters, int: #clusters to fit to data
        # --------------------------------------------------------------------------------------------------------------

        X_1 = self.raster_list[0]
        X_2 = self.raster_list[1]

        with open(self.params.pre_proc_dir+"ML/" + file_name_1 + ".pkl", "rb") as file: model_1 = pickle.load(file)
        with open(self.params.pre_proc_dir+"ML/" + file_name_2 + ".pkl", "rb") as file: model_2 = pickle.load(file)

        seq_1 = model_1.predict(X_1.T)
        seq_2 = model_2.predict(X_2.T)
        nr_modes = model_1.means_.shape[0]

        D = np.zeros((nr_modes, nr_modes))

        # simultaneous occurence
        for i in np.arange(nr_modes):
            first = np.zeros(seq_1.shape[0])
            first[seq_1 == i] = 1
            for j in np.arange(nr_modes):
                second = np.zeros(seq_1.shape[0])
                second[seq_2 == j] = 1

                l_or = first + second
                nr_active = np.count_nonzero(l_or)

                l_and = first * second
                nr_active_together = np.count_nonzero(l_and)
                D[i,j] = nr_active_together / nr_active

        plt.title("CO-OCCURENCE OF MODES IN BOTH POPULATIONS")
        plt.imshow(D, interpolation='nearest', aspect='auto', cmap="jet")
        plt.ylabel("MODE ID LEFT")
        plt.xlabel("MODE ID RIGHT")
        a = plt.colorbar()
        a.set_label("CO-OCCURENCE (TOGETHER/TOTAL)")
        plt.show()

        plt.title("HISTOGRAM OF CO-OCCURENCE VALUES")
        plt.xlabel("CO-OCCURENCE (TOGETHER/TOTAL)")
        plt.ylabel("COUNTS")
        plt.hist(D.flatten(), bins=100, log=True)
        plt.show()


