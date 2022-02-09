########################################################################################################################
#
#   PRE-PROCESSING
#
#   Description: contains classes that are used to perform pre-processing on raw data such as binning
#
#   Author: Lars Bollmann
#
#   Created: 22/07/2019
#
#   Structure:
#
#               - class PreProcess: base class containing methods & attributes that are used for sleep and awake data
#               - class PreProcessSleep: base class containing methods & attributes that are used for sleep data
#               - class PreProcessAwake: base class containing methods & attributes that are used for awake data
#
#
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
from collections import OrderedDict
from function_files.plotting_functions import plot_act_mat
from function_files.support_functions import moving_average
import scipy.ndimage as nd
import numpy.matlib as matlib
import time

########################################################################################################################
#   PRE-PROCESS BASE CLASS
########################################################################################################################


class PreProcess:
    """ Base class for spike data analysis"""

    def __init__(self, firing_times, whl, params, spatial_factor=None, time_stamps=None, last_spike=None):

        # set this flag to plot intermediate steps of pre-processing
        self.plot_for_control = False

        self.params = params

        # binning method
        self.firing_times = firing_times

        # binning method
        self.binning_method = params.binning_method

        # time bin size in seconds
        self.time_bin_size = params.time_bin_size

        # file name for saving
        self.file_name = params.file_name

        # duration of segment
        # ATTENTION:    time stamps for sleep (.nrem, .rem, .sw) are at 20 kHz --> just like .res data
        #               time stamps for awake behavior are at 20 kHz/512 --> just like .whl data
        self.time_stamps = time_stamps

        # last spike from all cell types --> estimate of length of recording
        self.last_spike = last_spike

        # save raw tracking data (lost frames are not interpolated)
        # multiply spatial factor to get data in cm! (and not in arbitrary units)
        if spatial_factor is not None:
            self.whl = whl * spatial_factor
        else:
            print("Spatial factor wasn't provided - computing everything with arbitrary units")
            self.whl = whl

        # interpolate lost frames
        self.whl_interpol = self.interpolate_lost_frames(self.whl)

        # set location and speed data to None --> only compute when needed (otherwise, it slows down init)
        self.loc = None
        self.speed = None

    def compute_loc_speed(self):
        """
        # computes location and speed data by interpolating
        """

        # compute speed at .whl resolution (20kHz/512 --> 25.6ms)
        speed_at_whl_res = self.compute_speed_fast(loc=self.whl_interpol, dt=0.0256)

        # up/down sample location & speed data to match requested time bin size
        self.loc = self.align_location(loc=self.whl_interpol, time_bin_size=self.params.time_bin_size)
        self.speed = self.align_speed(speed=speed_at_whl_res, time_bin_size=self.params.time_bin_size)

        # plotting to check that down-sampling & interpolation worked!
        if self.plot_for_control:
            ratio = self.time_bin_size / 0.0256
            plt.plot(self.whl[:int(3000 * ratio), 0], label="RAW .WHL DATA")
            plt.plot(np.linspace(0, int(3000 * ratio), 3000), self.loc[:3000, 0], color="r", alpha=0.5, label=
            "INTERPOL. & UP-SAMPLED")
            plt.title("TRACKING DATA")
            plt.legend()
            plt.show()

            plt.plot(speed_at_whl_res[:int(300 * ratio)], label="RAW SPEED DATA AT 20kHz")
            plt.plot(np.linspace(0, int(300 * ratio), 300), self.speed[:300], color="r", alpha=0.5, label=
            "INTERPOL. & DOWNSAMPLED")
            plt.title("SPEED")
            plt.legend()
            plt.show()

        # compute dimensions of field during behavior [x_min, x_max, y_min, y_max]
        self.field_dim = [np.nanmin(self.loc[:, 0]), np.nanmax(self.loc[:, 0]),
                          np.nanmin(self.loc[:, 1]), np.nanmax(self.loc[:, 1])]

    def interval_temporal_binning(self, interval, interval_freq, time_bin_size=None):
        # --------------------------------------------------------------------------------------------------------------
        # computes activity raster: time_bin_size in seconds --> sums up the activity within one time interval and
        # divides by length of interval if write_firing_rates is set to True
        #
        # ATTENTION:    time stamps for sleep (.nrem, .rem, .sw) are at 20 kHz --> just like .res data
        #               time stamps for awake behavior are at 20 kHz/512 --> just like .whl data
        #
        # parameters:   - time_bin_size, float (in seconds): which time bin size for output data
        #               - interval, array [start, end]: which interval to use
        #               - interval_freq: sampling frequency of interval in Hz
        #                 (e.g. interval in seconds --> interval_freq = 1)
        #
        # output:       - rows: cells
        #               - columns: time bins
        #
        # --------------------------------------------------------------------------------------------------------------

        if time_bin_size is None:
            time_bin_size = self.params.time_bin_size

        # compute interval in 20kHz resolution (to match spike rate)
        interval_spike_rate = interval * (1/interval_freq)/(0.05 * 1e-3)
        interval_spike_rate = np.round(interval_spike_rate).astype(int)

        # duration of requested interval in seconds
        dur_interval_s= (interval[1] - interval[0])*(1/interval_freq)
        # nr. of time bins for requested interval
        nr_time_bins = int(dur_interval_s / time_bin_size)
        # if the interval is too short to fit one time bin of defined size --> return None
        if nr_time_bins == 0:
            return None
        # length of time bin in 20kHz resolution (to match spike time resolution!)
        len_time_bin = int((dur_interval_s/(0.05 * 1e-3))/nr_time_bins)

        # matrix with population vectors
        act_mat = np.zeros([len(self.firing_times.keys()), nr_time_bins+1])

        # generate binned data using the entire data
        for cell_iter, (cell_ID, cell_firing) in enumerate(self.firing_times.items()):
            # only select spikes that are within the provided interval
            cell_firing = cell_firing[(interval_spike_rate[0] < cell_firing) & (cell_firing < interval_spike_rate[1])]
            # subtract start of the interval
            cell_firing -= interval_spike_rate[0]
            # divide cell firing times by interval size --> spike in which time bin
            cell_firing_per_bin = (cell_firing / len_time_bin).astype(int)
            # count how many spikes happened per time bin
            bins_and_nr_spikes = np.unique(cell_firing_per_bin, return_counts=True)
            # write nr of spikes or firing rate to according time bin (entry in act_mat)
            # check if any spikes where detected
            if bins_and_nr_spikes[0].size > 0:
                act_mat[cell_iter, bins_and_nr_spikes[0]] = bins_and_nr_spikes[1]

        return act_mat

    def temporal_binning(self, write_firing_rates=False, time_bin_size=None):
        # --------------------------------------------------------------------------------------------------------------
        # computes activity matrix: time_bin_size in seconds --> sums up the activity within one time interval and
        # divides by length of interval if write_firing_rates is set to True
        #
        # ATTENTION:    time stamps for sleep (.nrem, .rem, .sw) are at 20 kHz --> just like .res data
        #               time stamps for awake behavior are at 20 kHz/512 --> just like .whl data
        #
        # parameters:   - write_firing_rates:   if True --> writes firing rates instead of number of spikes for each bin
        #
        # output:       - rows: cells
        #               - columns: time bins
        #
        # --------------------------------------------------------------------------------------------------------------

        if time_bin_size is None:
            time_bin_size = self.params.time_bin_size

        if self.last_spike is None:
            # use last firing to determine duration of recording
            last_firing = 0

            for key, value in self.firing_times.items():
                last_firing = int(np.amax([last_firing, np.amax(value)]))

        else:
            last_firing = self.last_spike

        # duration of sleep in seconds (one time bin: 0.05ms --> 20kHz)
        dur_sleep = last_firing * 0.05 * 1e-3

        nr_intervals = int(dur_sleep / time_bin_size)
        size_intervals = int(last_firing / nr_intervals)
        size_interval_sec = size_intervals * 0.05 * 1e-3

        # matrix with population vectors
        act_mat = np.zeros([len(self.firing_times.keys()), nr_intervals+1])

        # generate binned data using the entire data
        for cell_iter, (cell_ID, cell_firing) in enumerate(self.firing_times.items()):
            # divide cell firing times by interval size --> spike in which time bin
            cell_firing_per_bin = (cell_firing / size_intervals).astype(int)
            # count how many spikes happened per time bin
            bins_and_nr_spikes = np.unique(cell_firing_per_bin, return_counts=True)
            # write nr of spikes or firing rate to according time bin (entry in act_mat)
            if write_firing_rates:
                act_mat[cell_iter, bins_and_nr_spikes[0]] = bins_and_nr_spikes[1] / size_interval_sec
            else:
                act_mat[cell_iter, bins_and_nr_spikes[0]] = bins_and_nr_spikes[1]

        # check if time stamps are provided --> if yes: only select time bins within time stamp window
        if self.time_stamps is not None:
            # indices to be selected
            indices = []
            # go trough all time stamps
            for time_stamp in self.time_stamps:
                indices.extend(np.arange((time_stamp[0]/ size_intervals).astype(int),
                                         (time_stamp[1]/ size_intervals).astype(int)))
            indices = np.array(indices)
            act_mat = act_mat[:, indices]

        return act_mat

    ####################################################################################################################
    #   GETTER METHODS
    ####################################################################################################################

    def get_loc(self):
        if self.loc is None:
            self.compute_loc_speed()
        return self.loc

    def get_speed(self):
        if self.speed is None:
            self.compute_loc_speed()
        return self.speed

    ####################################################################################################################
    #   SUPPORT METHODS
    ####################################################################################################################

    @staticmethod
    def compute_speed(whl, dt, n_smoothing=8):
        # --------------------------------------------------------------------------------------------------------------
        # compute speed from the whl - moving average - not the best but works
        #
        # parameters:   - whl, np.array: x, y location of the animal
        #               - dt, float: time resolution of provided location data in seconds (e.g. 20kHz --> dt = 0.00005,
        #                 if data is upsampled to match spike resolution or dt = 0.0256 if at 20kHz/512 resolution[whl])
        #               - n_smoothing, int: size of smoothing (moving average) window
        #
        # returns:      - speed, np.array: speed in cm/s
        # --------------------------------------------------------------------------------------------------------------

        # need half the size for the computation below
        half_smoothing = np.round(n_smoothing/2).astype(int)

        temp = np.zeros(whl.shape[0])
        speed = np.zeros(whl.shape[0])
        for i in range(half_smoothing, whl.shape[0] - half_smoothing):
            if whl[i, 0] > 0 and whl[i + 1, 0] > 0 and whl[i, 1] > 0 and whl[i + 1, 1] > 0:
                temp[i] = np.sqrt((whl[i, 0] - whl[i + 1, 0]) ** 2 + (whl[i, 1] - whl[i + 1, 1]) ** 2)
            else:
                temp[i] = -1
        for i in range(half_smoothing, whl.shape[0] - half_smoothing):
            t = temp[i - half_smoothing:i + half_smoothing]
            t = t[t >= 0]
            if (len(t) > 0):
                speed[i] = np.mean(t)
            else:
                speed[i] = np.nan
        return np.nan_to_num(speed / dt)

    @staticmethod
    def compute_speed_fast(loc, dt, n_smoothing=8):
        # --------------------------------------------------------------------------------------------------------------
        # compute speed from location data
        #
        # parameters:   - loc, np.array: x, y location of the animal
        #               - dt, float: time resolution of provided location data in seconds (e.g. 20kHz --> dt = 0.00005)
        #                 if data is upsampled to match spike resolution or dt = 0.0256 if at 20kHz/512 resolution[whl])
        #               - n_smoothing, int: size of smoothing (moving average) window
        #
        #
        # returns:      - speed, np.array: speed in cm/s
        # --------------------------------------------------------------------------------------------------------------

        # compute distance between subsequent bins
        dist = np.diff(loc, n=1, axis=0)
        dxy = np.sqrt(dist[:, 0]**2 + dist[:, 1]**2)

        # apply moving average filtering
        dxy = moving_average(dxy, n=n_smoothing)

        # compute speed by dividing distance by time
        speed = dxy / dt
        return speed

    @staticmethod
    def compute_speed_fede(loc, dt, n_smoothing=8):
        # --------------------------------------------------------------------------------------------------------------
        # compute speed from location data --> based on Federico's Matlab implementation
        #
        # parameters:   - loc, np.array: x, y location of the animal
        #               - dt, float: time resolution of provided location data in seconds (e.g. 20kHz --> dt = 0.00005)
        #                 if data is upsampled to match spike resolution or dt = 0.0256 if at 20kHz/512 resolution[whl])
        #               - n_smoothing, int: size of smoothing (moving average) window
        #
        # returns:      - speed, np.array: speed in cm/s
        # --------------------------------------------------------------------------------------------------------------

        # need half the size of the averaging window during the computation below
        half_smoothing_window = np.round(n_smoothing/2).astype(int)

        # compute distance between subsequent bins
        dist = np.diff(loc, n=1, axis=0)
        dxy = np.sqrt(dist[:, 0]**2 + dist[:, 1]**2)

        dxy_smooth = np.zeros(dxy.shape[0])
        for ss in range(dxy_smooth.shape[0]):
            swin = max(0, ss-(half_smoothing_window))
            ewin = min(dxy_smooth.shape[0], ss+(half_smoothing_window+1))
            # print(swin, ewin)
            dxy_smooth[ss] = np.median(dxy[swin:ewin])

        # compute speed by dividing distance by time
        speed = dxy_smooth / dt

        return speed

    @staticmethod
    def interpolate_lost_frames(whl):
        """
        interpolates lost frames from .whl file

        @param whl: .whl file data (x,y) coordinates of tracking
        @type whl: numpy.array
        """
        # get all frames where tracking is not lost (lost when [-1,-1])
        non_lost_frames = np.squeeze(np.argwhere(whl[:, 0] > 0))

        # interpolate lost data for x and y separately
        x_interpolated = np.interp(range(whl.shape[0]), non_lost_frames, whl[non_lost_frames, 0], left=np.nan,
                                   right=np.nan)
        y_interpolated = np.interp(range(whl.shape[0]), non_lost_frames, whl[non_lost_frames, 1], left=np.nan,
                                   right=np.nan)
        # combine x and y again to have same format like .whl
        whl_interpol = np.vstack((x_interpolated, y_interpolated)).T
        return whl_interpol

    @staticmethod
    def align_location(loc, time_bin_size, up_sampling_res=0.001):
        """
        Compute location data by first up-sampling and then combining bins and taking their mean.
        Interpolates lost frames.

        @param loc: x, y location of the animal at 20kHz resolution
        @type loc: numpy.array
        @param time_bin_size: time bin size in seconds of output data (20kHz --> time_bin_size = 0.00005)
        @type time_bin_size: float
        @param up_sampling_res: up sampling resolution in seconds (e.g. 1ms --> 0.001)
        @type up_sampling_res: float
        @return: down-sampled location data
        @rtype: numpy.array
        """

        # set nans from interpolation to -1
        loc[np.argwhere(np.isnan(loc[:, 0]))] = -1
        # loc = np.nan_to_num(loc)

        # get all frames where tracking is not lost (lost when [-1,-1])
        non_lost_frames = np.squeeze(np.argwhere(loc[:, 0] > 0))

        # compute frames at a defined resolution --> then down-sample (otherwise we can only use multiple of 25.6ms!)
        non_lost_frames_high_res = non_lost_frames * np.round(0.0256 / up_sampling_res).astype(int)
        new_frame_times = np.arange(np.round((0.0256 / up_sampling_res) * loc.shape[0]).astype(int))
        x_interpolated = np.interp(new_frame_times, non_lost_frames_high_res, loc[non_lost_frames, 0], left=np.nan,
                                   right=np.nan)

        y_interpolated = np.interp(new_frame_times, non_lost_frames_high_res, loc[non_lost_frames, 1], left=np.nan,
                                   right=np.nan)
        loc_up = np.vstack((x_interpolated, y_interpolated)).T

        # determine size of new array --> multiple of defined resolution
        size_intervals = np.round(time_bin_size / up_sampling_res).astype(int)
        nr_intervals = np.round(loc_up.shape[0] / size_intervals).astype(int)

        loc_downsampled = np.zeros((nr_intervals, 2))

        # down sample location data by combining multiple bins and taking their mean
        for i in range(nr_intervals):
            # check if there is a location that is not valid (e.g. [-1,-1]
            # if -1 in loc_up[(i * size_intervals): ((1 + i) * size_intervals), :]:
            #     loc_downsampled[i, :] = -1
            # else:
            loc_downsampled[i, :] = np.mean(
                loc_up[(i * size_intervals): ((1 + i) * size_intervals), :], axis=0)

        return loc_downsampled

    @staticmethod
    def align_speed(speed, time_bin_size):
        """
        Compute speed data at desired resolution

        @param speed: speed data at 20kHz resolution
        @type speed: numpy.array
        @param time_bin_size: time bin size in seconds of output data (20kHz --> time_bin_size = 0.00005)
        @type time_bin_size: float
        @return: down-sampled location data
        @rtype: numpy.array
        """

        old_frame_times_high_res = np.arange(speed.shape[0])* np.round(0.0256 / 0.0001).astype(int)

        new_frame_times = np.arange(np.round((0.0256 / 0.0001) * speed.shape[0]).astype(int))
        speed_interpolated = np.interp(new_frame_times, old_frame_times_high_res, speed, left=np.nan,
                                   right=np.nan)

        # determine size of new array --> multiple of 0.0001s
        size_intervals = np.round(time_bin_size / 0.0001).astype(int)
        nr_intervals = np.round(speed_interpolated.shape[0] / size_intervals).astype(int)

        speed_downsampled = np.zeros(nr_intervals)

        # down sample speed data by combining multiple bins and taking their mean
        for i in range(nr_intervals):
            # check if there is a location that is not valid (e.g. [-1,-1]
            # if -1 in loc_up[(i * size_intervals): ((1 + i) * size_intervals), :]:
            #     loc_downsampled[i, :] = -1
            # else:
            speed_downsampled[i] = np.mean(
                speed_interpolated[(i * size_intervals): ((1 + i) * size_intervals)], axis=0)

        return speed_downsampled


"""#####################################################################################################################
#   PRE-PROCESS CLASS FOR SLEEP DATA
#####################################################################################################################"""


class PreProcessSleep(PreProcess):
    """Extended class for sleep data"""

    def __init__(self, firing_times, params, whl=None, time_stamps=None, last_spike=None, spatial_factor=None):

        # get attributes from parent class
        PreProcess.__init__(self, firing_times=firing_times, whl=whl, params=params, time_stamps=time_stamps,
                            last_spike=last_spike, spatial_factor=spatial_factor)

    def speed_filter_raw_data(self, eegh=None, eeg=None):
        # speed/location filter raw data --> exclude periods when the animal is moving during sleep


        # get interpolated tracking data
        loc = self.interpolate_lost_frames(self.whl)
        plt.scatter(loc[:,0], loc[:,1])
        plt.show()
        speed = self.compute_speed_fast(loc=loc, dt=0.0256)
        plt.plot(speed)
        plt.show()

        exit()

    def spike_times(self, close_gaps=True):
        # returns spike times starting at one --> list of list ((firing times cell 1), (firing times cell 2), ...)
        #
        # parameters:   - close_gaps: closes gaps between different time intervals if True

        # list that contains lists for single cells
        all_cf_list = []

        # check if time stamps are provided
        if self.time_stamps is not None:

            # find first time stamp entry to subtract from all firing times
            global_offset = self.time_stamps[0][0]

            # go through all cells: cell_ID is not used --> only firing times
            for cell_iter, (cell_ID, cell_firing_times) in enumerate(self.firing_times.items()):
                # generate list with firing times of cell
                cf_list = []
                # don't subtract anything from first time interval except for global offset
                prev_time_stamp_last_firing = global_offset
                for time_stamp in self.time_stamps:
                    start_interval = time_stamp[0]
                    end_interval = time_stamp[1]
                    if close_gaps:
                        # calculate time between last and current time interval and offset that value
                        local_offset = start_interval - prev_time_stamp_last_firing - 1
                        cf_list.extend(
                            [x - global_offset - local_offset for x in cell_firing_times if
                             start_interval <= x < end_interval])
                        prev_time_stamp_last_firing = end_interval
                    else:
                        cf_list.extend(
                            [x-global_offset for x in cell_firing_times if start_interval <= x < end_interval])
                # append list for current cell to list of all cells
                all_cf_list.append(cf_list)

        else:
            # if time stamps are not provided --> use first firing to offset from all
            first_firing = np.inf

            for key, value in self.firing_times.items():
                first_firing = int(np.amin([first_firing, np.amin(value)]))

            global_offset = first_firing

            # go through all cells: cell_ID is not used --> only firing times
            for cell_iter, (cell_ID, cell_firing_times) in enumerate(self.firing_times.items()):
                # generate list with firing times of cell
                cf_list = []
                cf_list.extend([x-global_offset for x in cell_firing_times])
                # append list for current cell to list of all cells
                all_cf_list.append(cf_list)

        return all_cf_list

    def temporal_binning_binary(self, time_bin_size=None):
        # --------------------------------------------------------------------------------------------------------------
        # creates binned binary matrix from spike times
        #
        # ATTENTION: makes binary raster --> doesn't consider how many times fired within one time bin, only if it fired
        #
        # our data is recorded at 20kHz
        # 10 ms time bin --> binsize = 200
        # 100 ms time bin --> binsize = 2000
        # --------------------------------------------------------------------------------------------------------------

        if time_bin_size is None:
            time_bin_size = self.params.time_bin_size

        if self.last_spike is None:
            # use last firing to determine duration of recording
            last_firing = 0

            for key, value in self.firing_times.items():
                last_firing = int(np.amax([last_firing, np.amax(value)]))

        else:
            last_firing = self.last_spike

        # duration of sleep in seconds (one time bin: 0.05ms --> 20kHz)
        dur_sleep = last_firing * 0.05 * 1e-3

        nr_intervals = int(dur_sleep / time_bin_size)
        size_intervals = int(last_firing / nr_intervals)

        # matrix with population vectors
        act_mat = np.zeros([len(self.firing_times.keys()), nr_intervals+1])

        # generate binned data using the entire data
        for cell_iter, (cell_ID, cell_firing) in enumerate(self.firing_times.items()):
            # divide cell firing times by interval size --> spike in which time bin
            cell_firing_per_bin = (cell_firing / size_intervals).astype(int)
            # count how many spikes happened per time bin
            bins_and_nr_spikes = np.unique(cell_firing_per_bin, return_counts=True)
            # write nr of spikes or firing rate to according time bin (entry in act_mat)

            act_mat[cell_iter, bins_and_nr_spikes[0]] = 1

        # check if time stamps are provided --> if yes: only select time bins within time stamp window
        if self.time_stamps is not None:
            # indices to be selected
            indices = []
            # go trough all time stamps
            for time_stamp in self.time_stamps:
                indices.extend(np.arange((time_stamp[0]/ size_intervals).astype(int),
                                         (time_stamp[1]/ size_intervals).astype(int)))
            indices = np.array(indices)
            act_mat = act_mat[:, indices]

        return act_mat

    def spike_binning(self, spikes_per_bin=None, return_estimated_times=False):
        # --------------------------------------------------------------------------------------------------------------
        # computes raster matrix: fixed number of spikes per bin
        #
        # ATTENTION:    time stamps for sleep (.nrem, .rem, .sw) are at 20 kHz --> just like .res data
        #               time stamps for awake behavior are at 20 kHz/512 --> just like .whl data
        #
        #               --> variable bin length in seconds!
        #
        # output:       - rows: cells
        #               - columns: bins
        #
        # --------------------------------------------------------------------------------------------------------------

        if spikes_per_bin is None:
            spikes_per_bin = self.params.spikes_per_bin

        # combine all firing times of all cells, remember cell labels
        comb_firing_times = np.zeros((0))
        cell_labels = np.zeros((0))

        for cell_id, (_, cell_fir) in enumerate(self.firing_times.items()):
            comb_firing_times = np.hstack((comb_firing_times, cell_fir))
            cell_labels = np.hstack((cell_labels, np.ones(cell_fir.shape[0])*cell_id))

        sorted_ind = comb_firing_times.argsort()
        comb_firing_times_sorted = comb_firing_times[sorted_ind]
        cell_labels_sorted = cell_labels[sorted_ind]

        nr_bins = np.round(cell_labels_sorted.shape[0] / spikes_per_bin, 0).astype(int)

        nr_cells = len(self.firing_times)

        spike_bin_raster = np.zeros((nr_cells, nr_bins))
        # estimated time for each time bin
        estimated_time_s = np.zeros(nr_bins)

        # fill up spike bins --> TODO: there might be a faster way
        for raster_col in range(nr_bins-1):
            cell_ids, spikes = np.unique(cell_labels_sorted[
                                        (raster_col*spikes_per_bin):(raster_col+1)*spikes_per_bin], return_counts=True)

            spike_bin_raster[cell_ids.astype(int), raster_col] = spikes
            estimated_time_s[raster_col] = np.mean(comb_firing_times_sorted[
                                                 (raster_col*spikes_per_bin):(raster_col+1)*spikes_per_bin])*1/20e3

        if return_estimated_times:
            return spike_bin_raster, estimated_time_s
        else:
            return spike_bin_raster

    def get_raster(self, plotting=False):
        if self.binning_method == "temporal":
            # temporal binning: counting spikes per time bin and dividing by bin size --> unit: Hz
            raster = self.temporal_binning(write_firing_rates=True)
        elif self.binning_method == "temporal_spike":
            # temporal spike binning: counting spikes per time bin --> unit: #spikes
            raster = self.temporal_binning()
        elif self.binning_method == "temporal_binary":
            # temporal spike binning: counting spikes per time bin --> unit: #spikes
            raster = self.temporal_binning_binary()
        elif self.binning_method == "spike_binning":
            # spike binning: bins with constant number of spikes
            raster = self.spike_binning()
        if plotting:
            plot_act_mat(raster, self.params)
        return raster

    def event_spike_binning(self, event_times, event_time_freq, spikes_per_bin=None, cell_ids=None):
        # --------------------------------------------------------------------------------------------------------------
        # computes raster matrices for single events: fixed number of spikes per bin
        #
        # args:     - event_times, array [n_events, [start, end]]
        #           - event_time_freq, int: frequency of event time in Hz (e.g. event times in seconds -->
        #             event_time_freq = 1)
        #           - spikes_per_bin, how many spikes per bin to compute
        #
        # returns:  - spike rasters per event (nr. rows = nr. cells, nr. column = nr. of bins with constant spikes
        #             --> each event is a list entry
        #           - length of each constant spike bin per event (each event is one list entry)
        # --------------------------------------------------------------------------------------------------------------

        print("  - BUILDING CONSTANT #SPIKES POPULATION VECTORS ...")

        if spikes_per_bin is None:
            spikes_per_bin = self.params.spikes_per_bin

        # compute time binned raster at 10kHz
        time_bin_size_temp_raster = 0.0001
        # rasters for each event
        event_spike_rasters = []
        # window_length for each bin per event
        event_spike_window_lenghts = []

        for e_t in event_times:
            event_raster = self.interval_temporal_binning(interval=e_t, interval_freq=event_time_freq,
                                                          time_bin_size=time_bin_size_temp_raster)

            if cell_ids is not None:
                event_raster = event_raster[cell_ids, :]

            # compute sum per time bin
            sum_per_bin = np.sum(event_raster, axis=0)
            # cumulative sum
            cum_sum = np.cumsum(sum_per_bin)
            # check when multiples of spikes_per_bin occur in the cum sum
            indices_to_split = np.mod(cum_sum, spikes_per_bin)
            # detect negative change for modulo --> next spike packet starts
            dist = np.diff(indices_to_split)
            ind = (np.argwhere(dist < 0) + 2).flatten()
            ind = np.insert(ind, 0, 0, axis=0)
            # if ind contains last element --> don't split at the last element (delete last ind entry)
            if ind[-1] == event_raster.shape[1]:
                ind = ind[:-1]
            # split event raster according to computed indices & sum up spikes
            spike_raster = np.add.reduceat(event_raster, ind, axis=1)
            # check last bin --> if it contains less than spikes_per_bin - 1 --> delete
            if np.sum(spike_raster[:, -1], axis=0) < (spikes_per_bin -1):
                spike_raster = spike_raster[:,:-1]
            # compute spike window length in seconds
            window_length = np.diff(ind) * time_bin_size_temp_raster
            event_spike_rasters.append(spike_raster)
            event_spike_window_lenghts.append(window_length)

        # event_spike_rasters = np.array(event_spike_rasters)
        # event_spike_window_lenghts = np.array(event_spike_window_lenghts)

        print("  - ... DONE!\n")

        return event_spike_rasters, event_spike_window_lenghts

    def event_temporal_binning(self, event_times, event_time_freq, time_bin_size=None):
        # --------------------------------------------------------------------------------------------------------------
        # computes raster matrices for single events
        #
        # args:     - event_times, array [n_events, [start, end]]
        #           - event_time_freq, int: frequency of event time in Hz (e.g. event times in seconds -->
        #             event_time_freq = 1)
        #           - time_bin_size, in seconds
        #
        # returns:
        #
        # --------------------------------------------------------------------------------------------------------------

        print("  - BUILDING TIME BINNED POPULATION VECTORS ...")

        if time_bin_size is None:
            time_bin_size = self.params.time_bin_size

        # rasters for each event
        event_time_rasters = []

        for e_t in event_times:
            event_raster = self.interval_temporal_binning(interval=e_t, interval_freq=event_time_freq,
                                                          time_bin_size=time_bin_size)
            if event_raster is not None:
                event_time_rasters.append(event_raster)

        print("  - ... DONE!\n")

        return event_time_rasters

    # TODO: check if this method is obsolete

    def compute_base_raster(self):
        # TODO: obsolete!
        # --------------------------------------------------------------------------------------------------------------
        # computes base raster: bin width: 10 ms --> sums up the #spikes within one time bin of length 10ms
        #
        # ATTENTION:    time stamps for sleep (.nrem, .rem, .sw) are at 20 kHz --> just like .res data
        #               time stamps for awake behavior are at 20 kHz/512 --> just like .whl data
        #
        # parameters:
        #
        # output:       - rows: cells
        #               - columns: time bins
        #                   (resulting data has unit "spikes")
        #
        # TODO: maybe can modifiy temporal_binning_binary in a way to make this binning here faster
        # --------------------------------------------------------------------------------------------------------------

        # base map consists of 10ms bins --> in seconds: 0.01
        time_bin_size = 0.01

        print("COMPUTING BASE RASTER USING 10ms TIME BINS ... ")

        # check if time stamps are provided
        if self.time_stamps is not None:
            # compute duration of segment
            duration = 0
            for time_stamp in self.time_stamps:
                duration += time_stamp[1]-time_stamp[0]

            # duration of sleep (one time bin: 0.05ms --> 20kHz)
            dur_sleep_sec = duration * 0.05 * 1e-3

            nr_intervals = int(dur_sleep_sec / time_bin_size)
            size_intervals = int(duration / nr_intervals)
            size_interval_sec = size_intervals * 0.05 * 1e-3

            # matrix with population vectors
            act_mat = np.zeros([len(self.firing_times.keys()), nr_intervals])
            act_mat[:] = np.nan

            # go through all cells: cell_ID is not used --> only firing times
            for cell_iter, (cell_ID, cell) in enumerate(self.firing_times.items()):
                print(" - PROCESSING CELL "+str(cell_iter+1)+ "/" + str(len(self.firing_times.keys())+1))
                # counter to write values into act_mat
                entry_counter = 0
                # go through each time stamp entry
                for time_stamp in self.time_stamps:
                    # counter that iterates over intervals
                    interval_counter = 0
                    start_interval = time_stamp[0]
                    end_interval = time_stamp[0] + size_intervals
                    while end_interval <= time_stamp[1]:
                        # write population vectors
                        cell_spikes_intv = [x for x in cell if start_interval <= x < end_interval]
                        act_mat[cell_iter, entry_counter] = len(cell_spikes_intv)
                        entry_counter += 1
                        interval_counter += 1
                        start_interval = time_stamp[0] + interval_counter * size_intervals
                        end_interval = time_stamp[0] + (interval_counter + 1) * size_intervals

            # remove all columns with NAN (above computation only considers intervals that fully fit into the segment
            # intervals that are cut at the end of the segment are not considered

            act_mat = act_mat[:, ~np.all(np.isnan(act_mat), axis=0)]

        else:
            # if time stamps are not provided --> use first and last firing to determine segment duration
            first_firing = np.inf
            last_firing = 0

            for key, value in self.firing_times.items():
                first_firing = int(np.amin([first_firing, np.amin(value)]))
                last_firing = int(np.amax([last_firing, np.amax(value)]))

            # duration of sleep (one time bin: 0.05ms --> 20kHz)
            dur_sleep = (last_firing - first_firing) * 0.05 * 1e-3

            nr_intervals = int(dur_sleep / time_bin_size)
            size_intervals = int((last_firing - first_firing) / nr_intervals)
            size_interval_sec = size_intervals * 0.05 * 1e-3

            # matrix with population vectors
            act_mat = np.zeros([len(self.firing_times.keys()), nr_intervals])

            # go through all cells: cell_ID is not used --> only firing times
            for cell_iter, (cell_ID, cell) in enumerate(self.firing_times.items()):
                print(" - PROCESSING CELL " + str(cell_iter + 1) + "/" + str(len(self.firing_times.keys()) + 1))
                # go through all temporal intervals
                for i in range(nr_intervals):
                    start_interval = first_firing + i * size_intervals
                    end_interval = first_firing + (i + 1) * size_intervals

                    # write population vectors
                    cell_spikes_intv = [x for x in cell if start_interval <= x < end_interval]
                    act_mat[cell_iter, i] = len(cell_spikes_intv)

        return act_mat


"""#####################################################################################################################
#   PRE-PROCESS CLASS FOR AWAKE DATA
#####################################################################################################################"""


class PreProcessAwake(PreProcess):
    """Extended class for behavioral data"""

    def __init__(self, firing_times, params, whl, spatial_factor=None):

        # get attributes from parent class
        PreProcess.__init__(self, firing_times=firing_times, params=params, whl=whl, spatial_factor=spatial_factor)

        # spatial bin size in cm
        self.spatial_resolution = params.spatial_resolution

        # speed filter: if speed is below this threshold (in cm/s) --> activity is neglected
        self.spf = params.speed_filter

        self.compute_loc_speed()

    def spike_times(self, close_gaps=True):
        # returns spike times starting at one --> list of list ((firing times cell 1), (firing times cell 2), ...)
        #
        # parameters:   - close_gaps: closes gaps between different time intervals if True

        # list that contains lists for single cells
        all_cf_list = []

        # check if time stamps are provided
        if self.time_stamps is not None:

            # go through all cells: cell_ID is not used --> only firing times
            for cell_iter, (cell_ID, cell_firing_times) in enumerate(self.firing_times.items()):
                # generate list with firing times of cell
                cf_list = []
                for time_stamp in self.time_stamps:
                    start_interval = time_stamp[0]
                    end_interval = time_stamp[1]
                    if close_gaps:
                        # calculate time between last and current time interval and offset that value
                        local_offset = start_interval - 1
                        cf_list.extend(
                            [x - local_offset for x in cell_firing_times if
                             start_interval <= x < end_interval])
                        prev_time_stamp_last_firing = end_interval
                    else:
                        cf_list.extend(
                            [x for x in cell_firing_times if start_interval <= x < end_interval])
                # append list for current cell to list of all cells
                all_cf_list.append(cf_list)

        else:
            # if time stamps are not provided --> use first firing to offset from all
            first_firing = np.inf

            for key, value in self.firing_times.items():
                first_firing = int(np.amin([first_firing, np.amin(value)]))

            # go through all cells: cell_ID is not used --> only firing times
            for cell_iter, (cell_ID, cell_firing_times) in enumerate(self.firing_times.items()):
                # generate list with firing times of cell
                cf_list = []
                cf_list.extend([x for x in cell_firing_times])
                # append list for current cell to list of all cells
                all_cf_list.append(cf_list)

        return all_cf_list

    def temporal_binning(self, speed_filter=None, time_bin_size=None):
        # --------------------------------------------------------------------------------------------------------------
        # computes binned raster, location and velocity (aligned!)
        #
        # ATTENTION:    time stamps for sleep (.nrem, .rem, .sw) are at 20 kHz --> just like .res data
        #               time stamps for awake behavior are at 20 kHz/512 --> just like .whl data
        #
        # parameters:   - speed_filter, float: bins with speeds (in cm/s) below this treshold are neglected
        #               - time_bin_size, float (in sec): time bin size of results (raster, location, velocity)
        #               - comp_speed_method, string: one of the following
        #                                               - "fede":   Federico Stella's implementation with loop for
        #                                                           moving average
        #                                               - "standard":   Michele Nardin's implementation with loops
        #                                                               (relatively slow)
        #                                               - "fast":   matrix operatios with numpy's moving average
        #
        # output:       - raster, np.array: [cells, time bins]
        #               - location, np.array: [x_pos, y_pos]
        #               - velocity, np.array: [vel]
        #
        # --------------------------------------------------------------------------------------------------------------

        if speed_filter is None:
            speed_filter = self.params.speed_filter

        if time_bin_size is None:
            time_bin_size = self.params.time_bin_size

        if self.last_spike is None:
            # use last firing to determine duration of recording
            last_firing = 0

            for key, value in self.firing_times.items():
                if len(value) == 0:
                    continue
                else:
                    last_firing = int(np.amax([last_firing, np.amax(value)]))

        else:
            last_firing = self.last_spike

        # duration of exploration in seconds (one time bin: 0.05ms --> 20kHz)
        dur_exploration = last_firing * 0.05 * 1e-3
        nr_intervals = int(dur_exploration / time_bin_size)
        size_intervals = int(last_firing / nr_intervals)
        size_interval_sec = size_intervals * 0.05 * 1e-3

        # matrix with population vectors
        act_mat = np.zeros([len(self.firing_times.keys()), nr_intervals+1])

        # generate binned data using the entire data
        for cell_iter, (cell_ID, cell_firing) in enumerate(self.firing_times.items()):
            # check if spike times are provided as list entries --> convert to numpy array
            if isinstance(cell_firing, list):
                cell_firing = np.array(cell_firing)
            # divide cell firing times by interval size --> spike in which time bin
            cell_firing_per_bin = (cell_firing / size_intervals).astype(int)
            # count how many spikes happened per time bin
            bins_and_nr_spikes = np.unique(cell_firing_per_bin, return_counts=True)
            # write nr of spikes or firing rate to according time bin (entry in act_mat)
            act_mat[cell_iter, bins_and_nr_spikes[0]] = bins_and_nr_spikes[1]

        # load location data (is already at time_bin_size resolution -- trim all to same length)
        min_data_length = min(act_mat.shape[1], self.loc.shape[0], self.speed.shape[0])
        loc = self.loc[:min_data_length, :]
        speed = self.speed[:min_data_length]
        act_mat = act_mat[:,:min_data_length]

        # filter data using speed and location data --> if speed is below speed_filter or location contains np.nan
        ind_nan = np.argwhere(np.isnan(loc[:, 0]))
        ind_slow = np.argwhere(np.nan_to_num(speed) < speed_filter)

        bins_to_exclude = np.unique(np.vstack((ind_nan, ind_slow)))

        raster = np.delete(act_mat, bins_to_exclude, axis=1)
        loc_mat = np.delete(loc, bins_to_exclude, axis=0)
        speed_vec = np.delete(speed, bins_to_exclude)

        return raster, loc_mat, speed_vec

    def interval_temporal_binning_raster_loc_vel(self, interval_start, interval_end, speed_filter=None, time_bin_size=None):
        """
        computes binned raster, location and velocity (aligned!) for provided interval (in .whl resolution)
        --> used for trial data/when there are time stamps

        :param interval_start: int, start of interval in .whl (25.6ms) resolution
        :param interval_end: int, end of interval in .whl (25.6ms) resolution
        :param speed_filter: float: bins with speeds (in cm/s) below this treshold are neglected
        :param comp_speed_method: string: one of the following
        - "fede":   Federico Stella's implementation with loop for moving average
        - "standard":   Michele Nardin's implementation with loops (relatively slow)
        - "fast":   matrix operatios with numpy's moving average
        :param time_bin_size: float (in sec): time bin size of results (raster, location, velocity)
        :return:  -raster, np.array: [cells, time bins]
                  - location, np.array: [x_pos, y_pos]
                  - velocity, np.array: [vel]
        """

        if speed_filter is None:
            speed_filter = self.params.speed_filter

        if time_bin_size is None:
            time_bin_size = self.params.time_bin_size

        # duration of interval in seconds (one time bin: 0.0256ms --> 20/512kHz)
        dur_interval_s = (interval_end - interval_start) * 0.0256
        nr_time_bins = np.round(dur_interval_s / time_bin_size).astype(int)
        # compute size of time bin in .res resolution (20kHz ---> 0.00005s)
        size_time_bins = np.round(time_bin_size / 0.00005)
        # compute start of interval in 20kHz resolution to offset all firing times (want them to start at 0 for each
        # trial)
        interval_start_20kHz = interval_start * 512

        # matrix with population vectors
        act_mat = np.zeros([len(self.firing_times.keys()), nr_time_bins +1])

        # generate binned data using the entire data
        for cell_iter, (cell_ID, cell_firing) in enumerate(self.firing_times.items()):
            # need to offset by start of interval!!!
            cell_firing = cell_firing - interval_start_20kHz
            # divide cell firing times by interval size --> spike in which time bin
            cell_firing_per_bin = (cell_firing / size_time_bins).astype(int)
            # count how many spikes happened per time bin
            bins_and_nr_spikes = np.unique(cell_firing_per_bin, return_counts=True)
            # write nr of spikes or firing rate to according time bin (entry in act_mat)
            act_mat[cell_iter, bins_and_nr_spikes[0]] = bins_and_nr_spikes[1]

        # load location data (is already at time_bin_size resolution -- trim all to same length)
        min_data_length = min(act_mat.shape[1], self.loc.shape[0], self.speed.shape[0])
        loc = self.loc[:min_data_length, :]
        speed = self.speed[:min_data_length]
        act_mat = act_mat[:,:min_data_length]

        # filter data using speed and location data --> if speed is below speed_filter or location contains np.nan
        ind_nan = np.argwhere(np.isnan(loc[:, 0]))
        ind_slow = np.argwhere(np.nan_to_num(speed) < speed_filter)

        bins_to_exclude = np.unique(np.vstack((ind_nan, ind_slow)))

        raster = np.delete(act_mat, bins_to_exclude, axis=1)
        loc_mat = np.delete(loc, bins_to_exclude, axis=0)
        speed_vec = np.delete(speed, bins_to_exclude)

        return raster, loc_mat, speed_vec

    # TODO: check if we can delete this method (use method of parent class instead)

    # def interval_temporal_binning(self, interval_start, interval_end, time_bin_size=None):
    #     """
    #     computes binned raster, location and velocity (aligned!) for provided interval (in .whl resolution)
    #     --> used for trial data/when there are time stamps
    #
    #     :param interval_start: int, start of interval in .whl (25.6ms) resolution
    #     :param interval_end: int, end of interval in .whl (25.6ms) resolution
    #     :param speed_filter: float: bins with speeds (in cm/s) below this treshold are neglected
    #     :param comp_speed_method: string: one of the following
    #     - "fede":   Federico Stella's implementation with loop for moving average
    #     - "standard":   Michele Nardin's implementation with loops (relatively slow)
    #     - "fast":   matrix operatios with numpy's moving average
    #     :param time_bin_size: float (in sec): time bin size of results (raster, location, velocity)
    #     :return:  -raster, np.array: [cells, time bins]
    #               - location, np.array: [x_pos, y_pos]
    #               - velocity, np.array: [vel]
    #     """
    #
    #     if time_bin_size is None:
    #         time_bin_size = self.params.time_bin_size
    #
    #     # duration of interval in seconds (one time bin: 0.0256ms --> 20/512kHz)
    #     dur_interval_s = (interval_end - interval_start) * 0.0256
    #     nr_time_bins = np.round(dur_interval_s / time_bin_size).astype(int)
    #     # compute size of time bin in .res resolution (20kHz ---> 0.00005s)
    #     size_time_bins = np.round(time_bin_size / 0.00005)
    #     # compute start of interval in 20kHz resolution to offset all firing times (want them to start at 0 for each
    #     # trial)
    #     interval_start_20kHz = interval_start * 512
    #
    #     # matrix with population vectors
    #     act_mat = np.zeros([len(self.firing_times.keys()), nr_time_bins +1])
    #
    #     # generate binned data using the entire data
    #     for cell_iter, (cell_ID, cell_firing) in enumerate(self.firing_times.items()):
    #         # need to offset by start of interval!!!
    #         cell_firing = cell_firing - interval_start_20kHz
    #         # divide cell firing times by interval size --> spike in which time bin
    #         cell_firing_per_bin = (cell_firing / size_time_bins).astype(int)
    #         # count how many spikes happened per time bin
    #         bins_and_nr_spikes = np.unique(cell_firing_per_bin, return_counts=True)
    #         # write nr of spikes or firing rate to according time bin (entry in act_mat)
    #         act_mat[cell_iter, bins_and_nr_spikes[0]] = bins_and_nr_spikes[1]
    #
    #     return act_mat

    def interval_spike_binning(self, interval_start, interval_end, spikes_per_bin=None):
        # --------------------------------------------------------------------------------------------------------------
        # :param interval_start: int, start of interval in .whl (25.6ms) resolution
        # :param interval_end: int, end of interval in .whl (25.6ms) resolution
        #
        # args:     - event_times, array [n_events, [start, end]]
        #           - event_time_freq, int: frequency of event time in Hz (e.g. event times in seconds -->
        #             event_time_freq = 1)
        #           - spikes_per_bin, how many spikes per bin to compute
        #
        # returns:
        #
        # --------------------------------------------------------------------------------------------------------------

        if spikes_per_bin is None:
            spikes_per_bin = self.params.spikes_per_bin

        # compute time binned raster at 10kHz
        time_bin_size_temp_raster = 0.0001

        # only need raster, not location or velocity
        event_raster = self.interval_temporal_binning(interval=np.array([interval_start, interval_end]),
                                                      interval_freq=int(20e3/512),
                                                      time_bin_size=time_bin_size_temp_raster)
        # compute sum per time bin
        sum_per_bin = np.sum(event_raster, axis=0)
        # cumulative sum
        cum_sum = np.cumsum(sum_per_bin)
        # check when multiples of spikes_per_bin occur in the cum sum
        indices_to_split = np.mod(cum_sum, spikes_per_bin)
        # detect negative change for modulo --> next spike packet starts
        dist = np.diff(indices_to_split)
        ind = (np.argwhere(dist < 0) + 2).flatten()
        ind = np.insert(ind, 0, 0, axis=0)
        # if ind contains last element --> don't split at the last element (delete last ind entry)
        if ind[-1] == event_raster.shape[1]:
            ind = ind[:-1]
        # split event raster according to computed indices & sum up spikes
        spike_raster = np.add.reduceat(event_raster, ind, axis=1)
        # check last bin --> if it contains less than spikes_per_bin - 1 --> delete
        if np.sum(spike_raster[:, -1], axis=0) < (spikes_per_bin -1):
            spike_raster = spike_raster[:,:-1]
        # compute spike window length in seconds
        window_length = np.diff(ind) * time_bin_size_temp_raster


        # event_spike_rasters = np.array(event_spike_rasters)
        # event_spike_window_lenghts = np.array(event_spike_window_lenghts)

        return spike_raster, window_length

    def spike_binning(self, spikes_per_bin=None, return_estimated_times=False):
        # --------------------------------------------------------------------------------------------------------------
        # computes raster matrix: fixed number of spikes per bin
        #
        # ATTENTION:    time stamps for sleep (.nrem, .rem, .sw) are at 20 kHz --> just like .res data
        #               time stamps for awake behavior are at 20 kHz/512 --> just like .whl data
        #
        #               --> variable bin length in seconds!
        #
        # output:       - rows: cells
        #               - columns: bins
        #
        # --------------------------------------------------------------------------------------------------------------

        if spikes_per_bin is None:
            spikes_per_bin = self.params.spikes_per_bin

        # combine all firing times of all cells, remember cell labels
        comb_firing_times = np.zeros((0))
        cell_labels = np.zeros((0))

        for cell_id, (_, cell_fir) in enumerate(self.firing_times.items()):
            # check if spike times are provided as list entries --> convert to numpy array
            if isinstance(cell_fir, list):
                cell_fir = np.array(cell_fir)
            comb_firing_times = np.hstack((comb_firing_times, cell_fir))
            cell_labels = np.hstack((cell_labels, np.ones(cell_fir.shape[0])*cell_id))

        sorted_ind = comb_firing_times.argsort()
        comb_firing_times_sorted = comb_firing_times[sorted_ind]
        cell_labels_sorted = cell_labels[sorted_ind]

        nr_bins = np.round(cell_labels_sorted.shape[0] / spikes_per_bin, 0).astype(int)

        nr_cells = len(self.firing_times)

        spike_bin_raster = np.zeros((nr_cells, nr_bins))
        # estimated time for each time bin
        estimated_time_s = np.zeros(nr_bins)

        # fill up spike bins --> TODO: there might be a faster way
        for raster_col in range(nr_bins-1):
            cell_ids, spikes = np.unique(cell_labels_sorted[
                                        (raster_col*spikes_per_bin):(raster_col+1)*spikes_per_bin], return_counts=True)

            spike_bin_raster[cell_ids.astype(int), raster_col] = spikes
            estimated_time_s[raster_col] = np.mean(comb_firing_times_sorted[
                                                 (raster_col*spikes_per_bin):(raster_col+1)*spikes_per_bin])*1/20e3

        if return_estimated_times:
            return spike_bin_raster, estimated_time_s
        else:
            return spike_bin_raster

    def temporal_binning_binary(self):
        # --------------------------------------------------------------------------------------------------------------
        # creates binned binary matrix from spike times
        #
        # ATTENTION: makes binary raster --> doesn't consider how many times fired within one time bin, only if it fired
        #
        # our data is recorded at 20kHz
        # 10 ms time bin --> binsize = 200
        # 100 ms time bin --> binsize = 2000
        # --------------------------------------------------------------------------------------------------------------

        raise Exception("TO BE IMPLEMENTED!")

        spike_times = self.spike_times()
        loc = self.whl
        print(loc)
        exit()

        time_bin_size = self.params.time_bin_size
        binsize =time_bin_size * 20000

        maxBins = 1
        nNeurons = len(spike_times)
        # loop since not a numpy array, each neuron has different number of spike times --> find last firing of all
        # cells and use this value to determine that number of necessary time bins
        for nrnnum in range(nNeurons):
            try:
                # if cell doesn't fire at all --> list is empty and we cannot find max number of bins
                # here we keep the not firing cells --> maybe should remove them somewhere before
                maxBins = max((maxBins, int(np.amax(spike_times[nrnnum]) / binsize) + 1))
            except:
                continue
        spikeRaster = np.zeros((nNeurons, maxBins))
        for nrnnum in range(nNeurons):
            spikeRaster[nrnnum, (np.array(spike_times[nrnnum]) / binsize).astype(int)] = 1.
        raster = spikeRaster
        return raster

    def get_raster_loc_vel(self, plotting=False):
        if self.binning_method == "temporal":
            # temporal binning: counting spikes per time bin and dividing by bin size --> unit: Hz
            raster, loc_mat, speed_vec = self.temporal_binning()
        elif self.binning_method == "temporal_spike":
            # temporal spike binning: counting spikes per time bin --> unit: #spikes
            raster, loc_mat, speed_vec = self.temporal_binning()
        elif self.binning_method == "temporal_binary":
            # temporal spike binning: counting spikes per time bin --> unit: #spikes
            raster, loc_mat, speed_vec = self.temporal_binning()
        if plotting:
            plot_act_mat(raster, self.params)
        return raster, loc_mat, speed_vec

    # TODO: THESE METHODS DON'T BELONG TO PRE-PROCESSING --> MOVE TO ANALYSIS METHODS.py

    def spatial_rate_map(self):
        speed = self.compute_speed(whl=self.loc, dt=self.time_bin_size)
        occ = self.occupancy_map()
        sigma_gauss = 2

        rate_maps = []

        for cell_id, cell_f_t in self.firing_times.items():
            # down sample so that firing times match the tracking data
            cell_f_t = (cell_f_t/512).astype(int)
            cell_f_t = cell_f_t[speed[cell_f_t] > self.spf]
            rate = np.zeros((int(self.field_dim[1]/self.spatial_resolution)+1, int(self.field_dim[3]/self.spatial_resolution)+1))
            spkp = np.floor(self.whl[cell_f_t, :] / self.spatial_resolution).astype(int)

            for i in range(spkp.shape[0]):
                if (-1 < spkp[i, 0] < self.field_dim[1] and -1 < spkp[i, 1] < self.field_dim[3]):
                    rate[spkp[i, 0], spkp[i, 1]] += 1
            rate[occ > 0.05] = rate[occ > 0.05] / occ[occ > 0.05]
            if sigma_gauss > 0:
                rate = nd.gaussian_filter(rate, sigma=sigma_gauss)
            rate[occ == 0] = 0

            # trim rate map to proper dimensions
            rate = rate[(int(self.field_dim[0]/self.spatial_resolution)):,(int(self.field_dim[2]/self.spatial_resolution)):]

            rate_maps.append(rate)

        return rate_maps

    def plot_loc_and_speed(self):
        speed = self.compute_speed_fede(loc=self.loc, dt=self.params.time_bin_size)
        # plotting
        t = np.arange(len(speed))
        plt.plot(t/20e3, speed,label="speed")
        plt.plot(t/20e3,self.loc, label = "location")
        plt.plot([0, t[-1]/20e3],[5,5], label = "threshold")
        plt.xlabel("time / s")
        plt.ylabel("location / cm - speed / (cm/s)")
        plt.legend()
        plt.show()

    def occupancy_map(self, time_window=None):
        # speed represents the speed at each step of the whl
        # spf is the speed filter - anything between 3 and 9 is fine
        # plt.scatter(self.whl[:, 0], self.whl[: ,1])
        # plt.show()
        # whl = np.floor(self.whl[:, :] / 3).astype(int)
        # plt.scatter(whl[:, 0], whl[:, 1])
        # plt.show()
        # exit()20kHz / 512
        raise Exception("NEEDS TO BE CHECKED!")
        speed = self.compute_speed(whl=self.whl, dt=0.0256)

        whl = np.floor(self.whl[speed > self.spf, :] / self.spatial_resolution).astype(int)  # go from 0 to 150 cm

        if time_window is None:
            time_points = range(whl.shape[0])
        else:
            # convert time window (in min) to time points
            time_points = range(int(time_window[0]*60 * 39.0625), int(time_window[1]*60 * 39.0625))

        occ = np.zeros((int(self.field_dim[1]/self.spatial_resolution)+1, int(self.field_dim[3]/self.spatial_resolution)+1))
        for i in time_points:
            if (-1 < whl[i, 0] < self.field_dim[1] and -1 < whl[i, 1] < self.field_dim[3]):
                occ[whl[i, 0], whl[i, 1]] += 1
        # need occupancy in seconds
        occ = occ / 39.0625
        return occ

    # TODO: obsolete --> check if we can delete it

    def compute_base_raster_and_location(self):
        # --------------------------------------------------------------------------------------------------------------
        # computes base raster: bin width: 10 ms --> sums up the #spikes within one time bin of length 10ms
        #
        # ATTENTION:    time stamps for sleep (.nrem, .rem, .sw) are at 20 kHz --> just like .res data
        #               time stamps for awake behavior are at 20 kHz/512 --> just like .whl data
        #
        # parameters:
        #
        # output:       - rows: cells
        #               - columns: time bins
        #                   (resulting data has unit "spikes")
        #
        # TODO: modifiy temporal_binning_binary in a way to make this binning here faster
        # --------------------------------------------------------------------------------------------------------------

        # base map consists of 10ms bins --> in seconds: 0.01
        time_bin_size = 0.01

        print("COMPUTING BASE RASTER + LOCATION DATA USING 10ms TIME BINS ... ")

        # check if time stamps are provided
        if self.time_stamps is not None:
            # compute duration of segment
            duration = 0
            for time_stamp in self.time_stamps:
                duration += time_stamp[1]-time_stamp[0]

            # duration of sleep (one time bin: 0.05ms --> 20kHz)
            dur_sleep_sec = duration * 0.05 * 1e-3

            nr_intervals = int(dur_sleep_sec / time_bin_size)
            size_intervals = int(duration / nr_intervals)
            size_interval_sec = size_intervals * 0.05 * 1e-3

            # matrix with population vectors
            act_mat = np.zeros([len(self.firing_times.keys()), nr_intervals])
            act_mat[:] = np.nan

            # go through all cells: cell_ID is not used --> only firing times
            for cell_iter, (cell_ID, cell) in enumerate(self.firing_times.items()):
                print(" - PROCESSING CELL "+str(cell_iter+1)+ "/" + str(len(self.firing_times.keys())+1))
                # counter to write values into act_mat
                entry_counter = 0
                # go through each time stamp entry
                for time_stamp in self.time_stamps:
                    # counter that iterates over intervals
                    interval_counter = 0
                    start_interval = time_stamp[0]
                    end_interval = time_stamp[0] + size_intervals
                    while end_interval <= time_stamp[1]:
                        # write population vectors
                        cell_spikes_intv = [x for x in cell if start_interval <= x < end_interval]
                        act_mat[cell_iter, entry_counter] = len(cell_spikes_intv)
                        entry_counter += 1
                        interval_counter += 1
                        start_interval = time_stamp[0] + interval_counter * size_intervals
                        end_interval = time_stamp[0] + (interval_counter + 1) * size_intervals

            # remove all columns with NAN (above computation only considers intervals that fully fit into the segment
            # intervals that are cut at the end of the segment are not considered

            act_mat = act_mat[:, ~np.all(np.isnan(act_mat), axis=0)]

        else:
            # if time stamps are not provided --> use first and last firing to determine segment duration

            first_firing = np.inf
            last_firing = 0

            for key, value in self.firing_times.items():
                # check if cell fires at all
                if len(value):
                    first_firing = int(np.amin([first_firing, np.amin(value)]))
                    last_firing = int(np.amax([last_firing, np.amax(value)]))
                else:
                    continue

            # duration of sleep (one time bin: 0.05ms --> 20kHz)
            dur_sleep = (last_firing - first_firing) * 0.05 * 1e-3

            nr_intervals = int(dur_sleep / time_bin_size)
            size_intervals = int((last_firing - first_firing) / nr_intervals)
            size_interval_sec = size_intervals * 0.05 * 1e-3

            # upsample whl to match firing data
            # 20kHz / 512
            whl_up_sample = self.whl.copy().repeat(512, axis=0)
            # trim up-sampled whl to same length like firing time raster
            whl_up_sample = whl_up_sample[:(last_firing - first_firing), :]

            # matrix with population vectors
            act_mat = np.zeros([len(self.firing_times.keys()), nr_intervals])
            # array with location data
            loc_mat = np.zeros((nr_intervals, 2))

            # generate location array
            for i in range(nr_intervals):
                start_interval = first_firing + i * size_intervals
                end_interval = first_firing + (i + 1) * size_intervals
                loc_mat[i, :] = np.mean(whl_up_sample[start_interval:end_interval, :], axis=0)

            loc_mat = loc_mat.astype(int)

            # go through all cells: cell_ID is not used --> only firing times
            for cell_iter, (cell_ID, cell) in enumerate(self.firing_times.items()):
                print(" - PROCESSING CELL " + str(cell_iter + 1) + "/" + str(len(self.firing_times.keys()) + 1))
                # go through all temporal intervals
                for i in range(nr_intervals):
                    start_interval = first_firing + i * size_intervals
                    end_interval = first_firing + (i + 1) * size_intervals

                    # write population vectors
                    cell_spikes_intv = [x for x in cell if start_interval <= x < end_interval]
                    act_mat[cell_iter, i] = len(cell_spikes_intv)

        return act_mat, loc_mat