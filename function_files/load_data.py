########################################################################################################################
#
#   LOAD DATA
#
#   Description:
#
#           - returns data object containing data such as spike times, location, timestamps etc.
#
#           - imports data from .clu, .res, .timestamp and saves conditional data to "temp_data_old" directory
#
#           - criteria for selection: directory, experiment, cell type, environment, trial conditions (e.g
#             success/fail)
#
#   TODO: - clean up variable "session_name", "file_name" (is parameter of constructor and also one entry in parameter
#           file -- if they are always the same, use only one of them inside this class!)
#
#   Author: Lars Bollmann
#
#   Created: 11/03/2019
#
#
#
########################################################################################################################

import numpy as np
import pickle
import os
import importlib
import matplotlib.pyplot as plt

from function_files.support_functions import read_arrays, read_integers


class LoadData:
    """Base class to get and pre-process data
       - loads all needed files (such as .res, .whl, ...)
       - selects ids of cells to be used based on cell_type (e.g. "p1", "pe")
       - saves/loads data in form of a python dictionary --> data: location, cell spike timings etc.
    """

    def __init__(self, session_name, experiment_phase, pre_proc_dir):
        # --------------------------------------------------------------------------------------------------------------
        # creates SelectedData object
        #
        # args:     - session_name, str: session name, there needs to be a parameter file in parameter_files/ with the
        #             same name
        #           - experiment_phase, str: defines which phase of the experiment is supposed to be used
        #             (e.g. EXPLORATION_NOVEL, SLEEP_NOVEL)
        #           - cell_type, list of strings: which type of cell(s) to use (e.g. ["p1", "pe"]
        #           - pre_proc_dir, str: directory where pre-processed data is stored
        #
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # load parameter file
        # --------------------------------------------------------------------------------------------------------------
        data_parameter_file = importlib.import_module("parameter_files." + session_name)
        session_parameters = data_parameter_file.SessionParameters()

        # --------------------------------------------------------------------------------------------------------------
        # import parameters that describe the data
        # --------------------------------------------------------------------------------------------------------------
        self.data_description_dictionary = session_parameters.data_description_dictionary
        self.data_params_dictionary = session_parameters.data_params_dictionary
        self.cell_type_sub_div_dic = self.data_params_dictionary["cell_type_sub_div"]

        # --------------------------------------------------------------------------------------------------------------
        # check if only one or several experiment phases are supposed to be returned
        # --------------------------------------------------------------------------------------------------------------

        self.data_description = []
        self.experiment_phase_id = []
        for exp_ph in experiment_phase:
            self.data_description.append(exp_ph)
            self.experiment_phase_id.append(self.data_description_dictionary[exp_ph])

        # if two experiment phases are supposed to be analyzed
        self.experiment_phase = experiment_phase
        self.nr_phases = len(experiment_phase)

        # --------------------------------------------------------------------------------------------------------------
        # data file name --> file is stored in /temp_data_old subdirectory
        # --------------------------------------------------------------------------------------------------------------
        self.file_name = session_name

        # --------------------------------------------------------------------------------------------------------------
        # data direction for pre-processed data
        # --------------------------------------------------------------------------------------------------------------
        self.pre_proc_dir = pre_proc_dir

    """#################################################################################################################
    # Main functions
    #################################################################################################################"""

    def get_standard_data(self):
        # --------------------------------------------------------------------------------------------------------------
        # checks if pickled data exists already in subdirectory temp_data_old. If not, pickled data is generated and saved.
        # Then, pickled data is stored in dictionary and the dictionary returned.
        #
        # args:     -
        #
        # returns:  - data_dic_list, list: contains data dictionaries
        #
        #               - data dictionaries: dictionary containing requested data, keys:
        #
        #                   - in general:
        #
        #                   ["spike_times"][CELL_TYPE] --> contains spike times at 20kHz resolution
        #                                                  (e.g. CELL_TYPE = "p1")
        #                   ["last_firing"] --> contains timing of last spike (from all cells) at 20kHz resolution
        #
        #                   ["whl"] --> location data at 20kHz/512
        #
        #                   - additionally for SLEEP data:
        #
        #                   ["timestamps"][SLEEP_TYPE] --> time stamps for different sleep phases at 20kHz resolution
        #                                                  (SLEEP_TYPE can be "rem", "nrem", "sws")
        #
        # --------------------------------------------------------------------------------------------------------------

        # initialize empty dictionary
        data_dic = []
        for exp_phase_id, exp_phase_description in zip(self.experiment_phase_id, self.data_description):
            # check if data exists as pickle --> if not, create pickled data
            if not os.path.isfile(self.pre_proc_dir + self.file_name + "_" + exp_phase_id):
                self.save_standard_data(exp_phase_id, exp_phase_description)
            # loading the data
            infile = open(self.pre_proc_dir + self.file_name + "_" + exp_phase_id, 'rb')
            data_dic.append(pickle.load(infile))
            infile.close()
        return data_dic

    def get_extended_data(self):
        # --------------------------------------------------------------------------------------------------------------
        # loads extended data (including LFP data)
        #
        # args:     -
        #
        # returns:  - data_dic_list, list: contains data dictionaries
        #
        #               - data dictionaries: dictionary containing requested data, keys:
        #
        #                   - in general:
        #
        #                   ["spike_times"][CELL_TYPE] --> contains spike times at 20kHz resolution
        #                                                  (e.g. CELL_TYPE = "p1")
        #                   ["last_firing"] --> contains timing of last spike (from all cells) at 20kHz resolution
        #
        #                   ["whl"] --> location data at 20kHz/512
        #
        #                   ["eeg"] --> all channels downsampled 16 times (1.25kHz)
        #
        #                   ["eegh"] --> one channel per tetrode downsampled 4 times (5kHz)
        #
        #                   - additionally for SLEEP data:
        #
        #                   ["timestamps"][SLEEP_TYPE] --> time stamps for different sleep phases at 20kHz resolution
        #                                                  (SLEEP_TYPE can be "rem", "nrem", "sws")
        #
        # --------------------------------------------------------------------------------------------------------------

        print("LOADING LFP FILE ...")
        # initialize empty dictionary
        data_dic_list = []
        for exp_phase_id, exp_phase_description in zip(self.experiment_phase_id, self.data_description):
            # check if standard data exists as pickle --> if not, create pickled data
            if not os.path.isfile(self.pre_proc_dir + self.file_name + "_" + exp_phase_id):
                self.save_standard_data(exp_phase_id, exp_phase_description)
            # loading the data
            infile = open(self.pre_proc_dir + self.file_name + "_" + exp_phase_id, 'rb')
            data_dic = pickle.load(infile)
            infile.close()

            # check if lfp data exists as pickle --> if not, create pickled data
            if not os.path.isfile(self.pre_proc_dir + "lfp/" + self.file_name + "_" + exp_phase_id):
                self.save_lfp_data(exp_phase_id, exp_phase_description)
            # loading the data
            infile = open(self.pre_proc_dir + "lfp/" + self.file_name + "_" + exp_phase_id, 'rb')
            lfp_data_dic = pickle.load(infile)
            infile.close()
            data_dic.update(lfp_data_dic)
            data_dic_list.append(data_dic)

        print("... DONE!")

        return data_dic_list

    def save_standard_data(self, exp_phase_id, exp_phase_description):
        # --------------------------------------------------------------------------------------------------------------
        # Extracts data from directory containing [.res, .whl, .clu ..] files and saves data as pickle for faster
        # loading
        #
        # args:   - exp_phase_id, int: id describing phase of the experiment (extension of files --> e.g. _1
        #                 is exploration familiar --> exp_phase_id = 1
        #               - exp_phase_description, str: string that describes experiment phase (e.g. EXPLORATION_FAMILIAR)
        #
        # returns:      -
        # --------------------------------------------------------------------------------------------------------------
        data_dir = self.data_params_dictionary["data_dir"]

        # select session
        # --------------------------------------------------------------------------------------------------------------
        session_name = self.data_params_dictionary["session_name"]

        # cell types to extract
        # --------------------------------------------------------------------------------------------------------------
        cell_type_array = self.data_params_dictionary["cell_type_array"]

        # exp_phase_id --> _1, _2 etc.
        # --------------------------------------------------------------------------------------------------------------
        exp_phase = exp_phase_id

        # description of experiment phase --> e.g. EXPLORATION_FAMILIAR
        # --------------------------------------------------------------------------------------------------------------
        data_description = exp_phase_description

        # create dictionary
        data_dic = {}

        print("CREATING STANDARD DATA FILE ...")
        print(" - DATA DESCRIPTION: " + data_description)

        # load cluster IDs (from .clu) and times of spikes (from .res)
        clu = read_integers(data_dir + "/" + session_name + "/" + session_name + "_" + exp_phase + ".clu")
        res = read_integers(data_dir + "/" + session_name + "/" + session_name + "_" + exp_phase + ".res")

        # load trajectory data from .whl file
        # --------------------------------------------------------------------------------------------------------------
        whl = np.loadtxt(data_dir + "/" + session_name + "/" + session_name + "_" + exp_phase + ".whl").astype(float)

        """ ------------------------------------------------------------------------------------------------------------
        # check which type of data needs to be imported 
        # -----------------------------------------------------------------------------------------------------------"""
        if "learning_cheeseboard" in data_description:
            """ --------------------------------------------------------------------------------------------------------
            # this section is used for cheese board task: splits data into trials
            # -------------------------------------------------------------------------------------------------------"""

            # load start box coordinates
            x_min_sb, x_max_sb, y_min_sb, y_max_sb = self.data_params_dictionary["start_box_coordinates"]

            # get interpolated tracking data ("fill" lost frames)
            whl_interpol = self.interpolate_lost_frames(whl=whl)

            # get start and end times of trials
            start_times, end_times = self.trial_timestamps_from_whl(whl_interpol=whl_interpol, x_min_sb=x_min_sb,
                                                                    x_max_sb=x_max_sb, y_min_sb=y_min_sb, y_max_sb=
                                                                    y_max_sb)
            # get start and end times of each trial
            data_dic["trial_timestamps"] = np.vstack((start_times, end_times))
            data_dic["trial_data"] = {}

            # go through all trials and get location and spike data
            for trial_id, (s_t, e_t) in enumerate(zip(start_times, end_times)):
                data_dic["trial_data"]["trial" + str(trial_id)] = {}
                """ ----------------------------------------------------------------------------------------------------
                get location data for the trial
                ---------------------------------------------------------------------------------------------------- """
                data_dic["trial_data"]["trial" + str(trial_id)]["whl"] = whl[s_t:e_t]

                """ ----------------------------------------------------------------------------------------------------
                get spike data for the trial
                ---------------------------------------------------------------------------------------------------- """
                data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"] = {}
                # go through all cell types and get spike times
                for cell_type in cell_type_array:
                    # check if cells need to be split into two populations (defined in parameter file)
                    if self.cell_type_sub_div_dic and cell_type in self.cell_type_sub_div_dic:
                        cell_ids = self.get_cell_id(data_dir, session_name, cell_type)
                        # split into two subsets using the provided cutoff
                        cell_ids_ss_1 = [i for i in cell_ids if i <= self.cell_type_sub_div_dic[cell_type][0] + 2]
                        cell_ids_ss_2 = [i for i in cell_ids if i > self.cell_type_sub_div_dic[cell_type][0] + 2]
                        data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"][self.
                            cell_type_sub_div_dic[cell_type][1]] = {}
                        data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"][self.
                            cell_type_sub_div_dic[cell_type][2]] = {}
                        # get spike data
                        data_ss_1 = self.load_spikes_for_interval(cell_IDs=cell_ids_ss_1, clu=clu, res=res,
                                                                  start_interval=s_t, end_interval=e_t)
                        data_ss_2 = self.load_spikes_for_interval(cell_IDs=cell_ids_ss_2, clu=clu, res=res,
                                                                  start_interval=s_t, end_interval=e_t)
                        # write to dictionary (one key for each cell type)
                        data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"][self.
                            cell_type_sub_div_dic[cell_type][1]] = data_ss_1
                        data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"][self.
                            cell_type_sub_div_dic[cell_type][2]] = data_ss_2
                    # if cells are not supposed to be split --> save everything for one cell type entry
                    else:
                        data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"][cell_type] = {}
                        cell_ids = self.get_cell_id(data_dir, session_name, cell_type)
                        # get spike data
                        data = self.load_spikes_for_interval(cell_IDs=cell_ids, clu=clu, res=res, start_interval=s_t,
                                                             end_interval=e_t)
                        # write to dictionary
                        data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"][cell_type] = data

        elif "t_maze_task" in data_description:
            """ --------------------------------------------------------------------------------------------------------
            # this section is used for t-maze task: split data into trials, save additional data such as success/fail
            # -------------------------------------------------------------------------------------------------------"""

            print("T MAZE TASK")
            raise Exception("EXTRACTION OF T MAZE TASK DATA NEEDS TO BE IMPLEMENTED!")
            # TODO: T-maze task --> include trials in dictionary
            # for env in data_dic:
            #
            #     timestamps = np.loadtxt(data_dir+"/"+s_exp+"/"+s_exp+"_"+env+".timestamps").astype(int)
            #
            #     trial_sel = {"startarm": startarm, "goalarm": goalarm, "ruletype": ruletype, "errortrial": errortrial}
            #     trial_IDs, rule_identifier, new_rule_trial = sel_trials(timestamps, trial_sel)
            #
            #     # if no matching trials were found throw error
            #     if not trial_IDs:
            #         print("Environment "+env+": no matching trials found")
            #         continue
            #
            #     # get location data
            #     # --------------------------------------------------------------------------------------------------------
            #     whl_rot = np.loadtxt(data_dir + "/" + s_exp + "/" + s_exp + "_" + env + ".whl_rot").astype(int)
            #
            #     loc = get_location(trial_IDs, whl_rot, timestamps)
            #
            #     # get spike data
            #     # --------------------------------------------------------------------------------------------------------
            #
            #     # load cluster IDs and time of spikes
            #     clu = np.loadtxt(data_dir+"/"+s_exp+"/"+s_exp+"_"+env+".clu").astype(int)
            #     res = np.loadtxt(data_dir + "/" + s_exp + "/" + s_exp + "_" + env + ".res").astype(int)
            #
            #     # extract data
            #     data = get_data(trial_IDs, cell_IDs, clu, res, timestamps)
            #
            #     # save data dictionary as pickle
            #     data_dic[env] = {
            #         "whl_lin": loc,
            #         "res": data,
            #         "info": {"new_rule_trial": new_rule_trial, "rule_order": rule_identifier}
            #     }

        elif "cross_maze_task" in data_description:
            """ --------------------------------------------------------------------------------------------------------
            # this section is used for cross maze task: split data into trials, save additional data such as success/fail
            # -------------------------------------------------------------------------------------------------------"""

            print(" - processing cross maze task")

            # start,centrebegin, centreend, goalbegin, goalend, startarm, goalarm, control, lightarm, ruletype, errortrial
            timestamps = np.loadtxt(
                data_dir + "/" + session_name + "/" + session_name + "_" + exp_phase + ".timestamps").astype(int)

            # get start and end timestamp of each trial
            start_times = timestamps[:, 0]
            end_times = timestamps[:, 4]

            data_dic["trial_data"] = {}

            # go through all trials and get location and spike data
            for trial_id, (s_t, e_t) in enumerate(zip(start_times, end_times)):
                data_dic["trial_data"]["trial" + str(trial_id)] = {}
                """ ----------------------------------------------------------------------------------------------------
                get location data for the trial
                ---------------------------------------------------------------------------------------------------- """
                data_dic["trial_data"]["trial" + str(trial_id)]["whl"] = whl[s_t:e_t]

                """ ----------------------------------------------------------------------------------------------------
                get spike data for the trial
                ---------------------------------------------------------------------------------------------------- """
                data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"] = {}
                # go through all cell types and get spike times
                for cell_type in cell_type_array:
                    # check if cells need to be split into two populations (defined in parameter file)
                    if self.cell_type_sub_div_dic and cell_type in self.cell_type_sub_div_dic:
                        cell_ids = self.get_cell_id(data_dir, session_name, cell_type)
                        # split into two subsets using the provided cutoff
                        cell_ids_ss_1 = [i for i in cell_ids if i <= self.cell_type_sub_div_dic[cell_type][0] + 2]
                        cell_ids_ss_2 = [i for i in cell_ids if i > self.cell_type_sub_div_dic[cell_type][0] + 2]
                        data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"][self.
                            cell_type_sub_div_dic[cell_type][1]] = {}
                        data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"][self.
                            cell_type_sub_div_dic[cell_type][2]] = {}
                        # get spike data
                        data_ss_1 = self.load_spikes_for_interval(cell_IDs=cell_ids_ss_1, clu=clu, res=res,
                                                                  start_interval=s_t, end_interval=e_t)
                        data_ss_2 = self.load_spikes_for_interval(cell_IDs=cell_ids_ss_2, clu=clu, res=res,
                                                                  start_interval=s_t, end_interval=e_t)
                        # write to dictionary (one key for each cell type)
                        data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"][self.
                            cell_type_sub_div_dic[cell_type][1]] = data_ss_1
                        data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"][self.
                            cell_type_sub_div_dic[cell_type][2]] = data_ss_2
                    # if cells are not supposed to be split --> save everything for one cell type entry
                    else:
                        data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"][cell_type] = {}
                        cell_ids = self.get_cell_id(data_dir, session_name, cell_type)
                        # get spike data
                        data = self.load_spikes_for_interval(cell_IDs=cell_ids, clu=clu, res=res, start_interval=s_t,
                                                             end_interval=e_t)
                        # write to dictionary
                        data_dic["trial_data"]["trial" + str(trial_id)]["spike_times"][cell_type] = data

        """ --------------------------------------------------------------------------------------------------------
        # for all data: load spike times for all cell types for the entire data without splitting into trials
        # -------------------------------------------------------------------------------------------------------"""
        data_dic["whl"] = whl
        data_dic["spike_times"] = {}
        # go through all cell types and get spike times
        for cell_type in cell_type_array:
            # check if cells need to be split into two populations (defined in parameter file)
            if self.cell_type_sub_div_dic and cell_type in self.cell_type_sub_div_dic:
                cell_ids = self.get_cell_id(data_dir, session_name, cell_type)
                # split into two subsets using the provided cutoff
                cell_ids_ss_1 = [i for i in cell_ids if i <= self.cell_type_sub_div_dic[cell_type][0] + 2]
                cell_ids_ss_2 = [i for i in cell_ids if i > self.cell_type_sub_div_dic[cell_type][0] + 2]
                data_dic["spike_times"][self.cell_type_sub_div_dic[cell_type][1]] = {}
                data_dic["spike_times"][self.cell_type_sub_div_dic[cell_type][2]] = {}
                # get spike data
                data_ss_1 = self.load_spikes(cell_ids_ss_1, clu, res)
                data_ss_2 = self.load_spikes(cell_ids_ss_2, clu, res)
                # write to dictionary (one key for each cell type)
                data_dic["spike_times"][self.cell_type_sub_div_dic[cell_type][1]] = data_ss_1
                data_dic["spike_times"][self.cell_type_sub_div_dic[cell_type][2]] = data_ss_2
            # if cells are not supposed to be split --> save everything for one cell type entry
            else:
                data_dic["spike_times"][cell_type] = {}
                cell_ids = self.get_cell_id(data_dir, session_name, cell_type)
                # get spike data
                data = self.load_spikes(cell_ids, clu, res)
                # write to dictionary
                data_dic["spike_times"][cell_type] = data

        # for entire sleep data --> get last firing to determine sleep duration
        last_firing = 0

        # determine different cell_types in dic
        all_cell_type_array = data_dic["spike_times"].keys()

        for cell_type in all_cell_type_array:
            for key, value in data_dic["spike_times"][cell_type].items():
                if not len(value) == 0:
                    last_firing = int(np.amax([last_firing, np.amax(value)]))

        # save time of last spike of sleep
        data_dic["last_spike"] = last_firing

        """ ------------------------------------------------------------------------------------------------------------
        # ADDITIONAL DATA: check which additional data needs to be loaded depending on data type (SLEEP, EXPLORATION ..)
        # -----------------------------------------------------------------------------------------------------------"""
        if "sleep" in data_description:
            # ----------------------------------------------------------------------------------------------------------
            # get time stamps for different sleep types (rem, non-rem, sharp waves)
            # ----------------------------------------------------------------------------------------------------------
            data_dic["timestamps"] = {}
            sleep_sub_div = ["rem", "nrem", "sw"]

            for sleep_part in sleep_sub_div:
                data_dic[sleep_part] = {}
                if not os.path.isfile(data_dir + "/" + session_name + "/" + session_name + "_" + exp_phase + "." +
                                      sleep_part) or os.path.getsize(data_dir + "/" + session_name + "/" +
                                                                     session_name + "_" + exp_phase + "." + sleep_part) == 0:
                    print("   --> SLEEP PART " + sleep_part + " FILE NOT FOUND:")
                    print("       " + data_dir + "/" + session_name + "/" + session_name + "_" + exp_phase + "." +
                          sleep_part)
                    continue
                # get time intervals for sleep part
                try:
                    timestamps = np.loadtxt(data_dir + "/" + session_name + "/" + session_name
                                            + "_" + exp_phase + "." + sleep_part).astype(int)
                # value missing
                except:
                    timestamps = np.genfromtxt(data_dir + "/" + session_name + "/" + session_name
                                               + "_" + exp_phase + "." + sleep_part).astype(int)

                # save timestamps of sleep for time binning
                # if timestamp has only one entry --> expand dimension
                if timestamps.ndim == 1:
                    timestamps = np.expand_dims(timestamps, 0)
                data_dic["timestamps"][sleep_part] = timestamps

        elif "exploration" in data_description:
            # ----------------------------------------------------------------------------------------------------------
            # get additional exploration data --> nothing at this point in time
            # ----------------------------------------------------------------------------------------------------------
            pass

        elif "cheeseboard" in data_description:
            # ----------------------------------------------------------------------------------------------------------
            # get additional cheeseboard data --> coordinates of start box
            # ----------------------------------------------------------------------------------------------------------
            data_dic["start_box"] = self.data_params_dictionary["start_box_coordinates"]

        elif "cross_maze_task" in data_description:
            # ----------------------------------------------------------------------------------------------------------
            # get additional cross_maze_task data --> start_arm, rule etc.
            # ----------------------------------------------------------------------------------------------------------
            # start,centrebegin, centreend, goalbegin, goalend, startarm, goalarm,
            # control, lightarm, ruletype, errortrial
            trial_info = np.loadtxt(data_dir + "/" + session_name + "/" + session_name + "_" +
                                    exp_phase + ".timestamps").astype(int)
            data_dic["timestamps"] = trial_info[:, :5]
            data_dic["start_arm"] = trial_info[:, 5]
            data_dic["goal_arm"] = trial_info[:, 6]
            data_dic["control"] = trial_info[:, 7]
            data_dic["lightarm"] = trial_info[:, 8]
            data_dic["ruletype"] = trial_info[:, 9]
            data_dic["errortrial"] = trial_info[:, 10]

        filename = self.pre_proc_dir + session_name + "_" + exp_phase

        outfile = open(filename, 'wb')
        pickle.dump(data_dic, outfile)
        outfile.close()

    def save_lfp_data(self, exp_phase_id, exp_phase_description):
        # --------------------------------------------------------------------------------------------------------------
        # Extracts lfp data from directory containing [.eeg, .eegh] files and saves data as pickle for faster
        # loading
        #
        # args:   - exp_phase_id, int: id describing phase of the experiment (extension of files --> e.g. _1
        #                 is exploration familiar --> exp_phase_id = 1
        #               - exp_phase_description, str: string that describes experiment phase (e.g. EXPLORATION_FAMILIAR)
        #
        # returns:      -
        # --------------------------------------------------------------------------------------------------------------
        data_dir = self.data_params_dictionary["data_dir"]

        # select session
        # --------------------------------------------------------------------------------------------------------------
        session_name = self.data_params_dictionary["session_name"]

        # exp_phase_id --> _1, _2 etc.
        # --------------------------------------------------------------------------------------------------------------
        exp_phase = exp_phase_id

        # description of experiment phase --> e.g. EXPLORATION_FAMILIAR
        # --------------------------------------------------------------------------------------------------------------
        data_description = exp_phase_description

        # create dictionary
        data_dic = {}

        print("CREATING LFP DATA FILE ...")
        print(" - DATA DESCRIPTION: " + data_description)

        # --------------------------------------------------------------------------------------------------------------
        # get info about electrodes/channels from .par file
        # --------------------------------------------------------------------------------------------------------------
        par_file = data_dir + "/" + session_name + "/" + session_name + "_" + exp_phase + ".par"
        par_list = []
        with open(par_file, "r") as file:
            for line in file:
                for par in line.split():
                    par_list.append(par)
        # first entry: number of channels
        nr_channels = int(par_list[0])
        # 5th entry (=4 in python): number of tetrodes
        nr_tetrodes = int(par_list[4])

        # --------------------------------------------------------------------------------------------------------------
        # LFP: load .eeg and .eegh data
        # --------------------------------------------------------------------------------------------------------------

        eeg_file = data_dir + "/" + session_name + "/" + session_name + "_" + exp_phase + ".eeg"
        if os.path.isfile(eeg_file):
            eeg = np.fromfile(eeg_file, dtype=np.int16)
            # all channels downsampled 16 times (1.25kHz)
            eeg.shape = (-1, nr_channels)
            data_dic["eeg"] = eeg
        else:
            print("   --> .eeg FILE NOT FOUND --> CONTINUE")

        eegh_file = data_dir + "/" + session_name + "/" + session_name + "_" + exp_phase + ".eegh"
        if os.path.isfile(eegh_file):
            eegh = np.fromfile(eegh_file, dtype=np.int16)
            # one channel per tetrode downsampled 4 times (5kHz)
            eegh.shape = (-1, nr_tetrodes)
            data_dic["eegh"] = eegh
        else:
            print("   --> .eegh FILE NOT FOUND --> CONTINUE")

        # check if lfp folder exists, otherwise create it
        if not os.path.exists(self.pre_proc_dir + "lfp/"):
            os.makedirs(self.pre_proc_dir + "lfp/")

        filename = self.pre_proc_dir + "lfp/" + session_name + "_" + exp_phase

        outfile = open(filename, 'wb')
        pickle.dump(data_dic, outfile, protocol=4)
        outfile.close()

    """#################################################################################################################
    # Helper functions
    #################################################################################################################"""

    @staticmethod
    def get_cell_id(data_dir, session_name, cell_type):
        # --------------------------------------------------------------------------------------------------------------
        # returns cell IDs from .des file for the selected cell type
        #
        # args:         - data_dir, str: directory where data folders are stored (directory containig all mjc.. folders)
        #               - session_name, str: "e.g. mjc163_2_0104"
        #               - cell type, str: can be on of the following (updated: 16.11.2020)
        #
        #                   - u1: unidentified?
        #                   - p1: pyramidal cells of the HPC
        #                   - p2 - p3: pyramidal cells of the PFC
        #                   - b1: interneurons of HPC
        #                   - b2 - b3: interneurons of HPC
        #                   - pe: pyramidal cells MEC
        #
        # returns:      - cell_ids, list: list containing cell ids
        # --------------------------------------------------------------------------------------------------------------

        with open(data_dir + "/" + session_name + "/" + session_name + ".des") as f:
            des = f.read()
        des = des.splitlines()
        # offset by 2 entries
        cell_ids = [i + 2 for i in range(len(des)) if des[i] == cell_type]

        return cell_ids

    @staticmethod
    def load_spikes(cell_IDs, clu, res):
        # --------------------------------------------------------------------------------------------------------------
        # loads spike times for each cell and writes times (in 20kHz resolution) to dictionary
        #
        # args:     - cell_IDs, list: list containing cell ids of cells that are supposed to be used
        #           - clu, np.array: array containing entry <-> cell_id associations
        #           - res, np.array: spike times (at 20kHz resolution
        #
        # returns:  - data, dictionary:     data["cell" + cell_ID] --> for each cell one np.array with spike times in
        #                                   20kHz resolution
        #
        # --------------------------------------------------------------------------------------------------------------
        data = {}

        # go through all cell ids that were provided
        for cell_ID in cell_IDs:
            # find all entries of the cell_ID in the clu list
            entries_cell = np.where(clu == cell_ID)

            # append entries from res file (data is shifted by -1 with respect to clu list)
            ind_res_file = entries_cell[0] - 1

            # select spikes using found indices
            cell_spikes = res[ind_res_file]

            # write spikes to entry in dictionary
            data["cell" + str(cell_ID)] = cell_spikes

        return data

    @staticmethod
    def load_spikes_for_interval(cell_IDs, clu, res, start_interval, end_interval):
        """
        loads spike times for each cell and for the defined interval and writes times (in 20kHz resolution)
        to dictionary

        @param cell_IDs: list containing cell ids of cells that are supposed to be used
        @type cell_IDs: list
        @param clu: array containing entry <-> cell_id associations
        @type clu: numpy.array
        @param res: spike times (at 20kHz resolution)
        @type res: numpy.array
        @param start_interval: start of interval at .whl resolution (20kHz/512)
        @type start_interval: int
        @param end_interval: end of interval at .whl resolution (20kHz/512)
        @type end_interval: int
        @return: data, data["cell" + cell_ID] --> for each cell one np.array with spike times in 20kHz resolution
        @rtype: dict
        """
        data = {}

        # -----------------------------------------------
        # timestamps: 20kHz/512 --> 25.6 ms per time bin
        # res data: 20kHz
        # time interval for res data: timestamp data * 512
        t_start = start_interval * 512
        t_end = end_interval * 512

        # go through all cell ids that were provided
        for cell_ID in cell_IDs:
            # find all entries of the cell_ID in the clu list
            entries_cell = np.where(clu == cell_ID)

            # append entries from res file (data is shifted by -1 with respect to clu list)
            ind_res_file = entries_cell[0] - 1

            # only use spikes that correspond to time interval of the trial
            cell_spikes = [x for x in res[ind_res_file] if t_start < x < t_end]

            # write spikes to entry in dictionary
            data["cell" + str(cell_ID)] = cell_spikes

        return data

    # for cheeseboard task
    # ------------------------------------------------------------------------------------------------------------------

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
    def trial_timestamps_from_whl(whl_interpol, x_min_sb, x_max_sb, y_min_sb, y_max_sb,
                                  min_trial_length=8, min_sb_duration=2):
        """
        computes timestamps for single trials of cheese board task using times spend in the start box to separate
        trials

        @param whl_interpol: location data with interpolated values (no lost frames, e.g. -1/-1)
        @type whl_interpol: numpy.array
        @param x_min_sb: min. x boundary of start box in cm
        @type x_min_sb: float
        @param x_max_sb: max. x boundary of start box in cm
        @type x_max_sb: float
        @param y_min_sb: min. y boundary of start box in cm
        @type y_min_sb: float
        @param y_max_sb: max. y boundary of start box in cm
        @type y_max_sb: float
        @param min_trial_length: minimum length of trial in seconds --> used to close gaps (animal in start box, but
        some location values are outside the start box)
        @type min_trial_length: int
        @param min_sb_duration: minimum duration in start box --> used to filter out periods where animal maybe passes
        by the start box and some location values appear to be in the start box without the animal actually staying
        @type min_sb_duration: int
        @return: start_times, end_times of trials at .whl resolution (20kHz/512)
        @rtype: numpy.array, numpy.array
        """
        in_sb = np.zeros(whl_interpol.shape[0])
        # find times where animal is in start box (min. time = XX)
        for i, loc in enumerate(whl_interpol):
            if (x_min_sb < loc[0] < x_max_sb) and (y_min_sb < loc[1] < y_max_sb):
                in_sb[i] = 1

        # remove start box duration that is too short
        # min. start box duration in 20kHz/512 --> 0.0256ms resolution
        min_sb_duration_whl_res = np.round(min_sb_duration / 0.0256).astype(int)
        for i in range(min_sb_duration_whl_res, in_sb.shape[0] - min_sb_duration_whl_res):
            if in_sb[i] == 1:
                if (np.count_nonzero(in_sb[(i - min_sb_duration_whl_res):i] == 0) > 0) & \
                        (np.count_nonzero(in_sb[i:(i + min_sb_duration_whl_res)] == 0) > 0):
                    in_sb[i] = 0

        # close gaps
        # min. trial length in 20kHz/512 --> 0.0256ms resolution
        min_trial_length_whl_res = np.round(min_trial_length / 0.0256).astype(int)
        for i in range(min_trial_length_whl_res, in_sb.shape[0] - min_trial_length_whl_res):
            if in_sb[i] == 0:
                if (np.count_nonzero(in_sb[i:(i + min_trial_length_whl_res)]) > 0) & \
                        np.count_nonzero(in_sb[(i - min_trial_length_whl_res):i]) > 0:
                    in_sb[i] = 1

        # detect changes (from 0 to 1 or 1 to 0)
        diff = np.diff(in_sb)

        # get start times of trials: cut last entry of start_times/first entry of end times
        start_times = np.squeeze(np.argwhere(diff == -1)[:-1])
        end_times = np.squeeze(np.argwhere(diff == 1)[1:])

        # check one more time that no trials that are too short are selected
        trials_to_delete = []
        for trial_id in range(start_times.shape[0]):
            if (end_times[trial_id] - start_times[trial_id]) < min_trial_length_whl_res:
                trials_to_delete.append(trial_id)
        start_times = np.delete(start_times, trials_to_delete)
        end_times = np.delete(end_times, trials_to_delete)

        return start_times, end_times

    # for trial data
    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def sel_trials(timestamps, trial_sel):
        # --------------------------------------------------------------------------------------------------------------
        # returns trial IDs of trials that meet the conditions in trial_sel
        # TODO: finish description
        #
        # args:         - timestamps, list: time stamps of trials
        #               - trial_sel, dic: which trial to use
        #
        # returns:      - trial_intervals,
        #               - rule_identifier,
        #               - trial_new_rule,
        # --------------------------------------------------------------------------------------------------------------

        rule_identifier = []
        trial_intervals = []
        trial_new_rule = 0
        temp = 0
        # go through all trials:
        for trial_ID, trial in enumerate(timestamps):
            # check if trial agrees with conditions
            # start,centrebegin, centreend, goalbegin, goalend, startarm, goalarm, control, lightarm,
            # ruletype, errortrial
            if trial[9] in trial_sel["ruletype"] and trial[10] in trial_sel["errortrial"] and \
                    trial[5] in trial_sel["startarm"] and trial[6] in trial_sel["goalarm"]:
                trial_intervals.append(trial_ID)
                # remember number of trials saved to identify rule switch case
                temp += 1
                # check if new rule --> if yes, append to rule identifier
                if not rule_identifier:
                    rule_identifier.append(trial[9])
                elif trial[9] != rule_identifier[-1]:
                    rule_identifier.append(trial[9])
                    trial_new_rule = temp
        return trial_intervals, rule_identifier, trial_new_rule

    @staticmethod
    def load_spikes_trial(trial_IDs, cell_IDs, clu, res, timestamps):
        # returns dictionary with spike data for selected trials and selected cells
        #
        # ATTENTION: timestamps of awake behavior are at 20 kHz/512 just like the data in the .whl file

        data = {}
        # go through selected trials
        for trial_ID in trial_IDs:
            # create entry in dictionary
            data["trial" + str(trial_ID)] = {}
            # -----------------------------------------------
            # timestamps: 20kHz/512 --> 25.6 ms per time bin
            # res data: 20kHz
            # time interval for res data: timestamp data * 512
            t_start = timestamps[trial_ID, 0] * 512
            t_end = timestamps[trial_ID, 4] * 512
            # go through selected cells
            for cell_ID in cell_IDs:
                cell_spikes_trial = []
                # find all entries of the cell_ID in the clu list
                entries_cell = np.where(clu == cell_ID)

                # append entries from res file (data is shifted by -1 with respect to clu list)
                ind_res_file = entries_cell[0] - 1
                # only use spikes that correspond to time interval of the trial
                cell_spikes_trial = [x for x in res[ind_res_file] if t_start < x < t_end]
                # append data
                data["trial" + str(trial_ID)]["cell" + str(cell_ID)] = cell_spikes_trial

        return data

    def get_linearized_location_trials(self, trial_IDs, whl_rot, timestamps):
        # returns dictionary with location for selected trials
        # linearized location
        whl_lin = self.linearize_location(whl_rot, timestamps)

        # dictionary with locations
        loc = {}
        # go through selected trials
        for trial_ID in trial_IDs:
            # -----------------------------------------------
            #  timestamps: 20kHz/512 --> 25.6 ms per time bin
            #  whl: 20kHz/512 --> both have the same order of magnitude
            t_start = timestamps[trial_ID, 0]
            t_end = timestamps[trial_ID, 4]
            # select locations that correspond to time interval of the trial
            loc_trial = whl_lin[t_start:t_end]
            # append data
            loc["trial" + str(trial_ID)] = loc_trial

        return loc

    @staticmethod
    def linearize_location(whl_rot, timestamps):
        # returns linearized location, skipping locations that have -1/1 (recording errors)
        # - calculates distance from the center (101,116) using euclidean distance
        # - inverts results for start arm (location center - location)
        # - adds location of center to goal arm

        data = whl_rot.copy()

        for trial in range(len(timestamps)):
            ind = range(timestamps[trial, 0], timestamps[trial, 4] + 1)
            indx = np.where(data[ind, 0] > -1)[0] + ind[0]
            indy = np.where(data[ind, 1] > -1)[0] + ind[0]
            data[ind, 0] = np.interp(ind, np.where(data[ind, 0] > -1)[0] + ind[0], data[indx, 0])
            data[ind, 1] = np.interp(ind, np.where(data[ind, 1] > -1)[0] + ind[0], data[indy, 1])

        # location of center
        p = (101, 116)
        dis = np.zeros(len(data))
        for trial in range(len(timestamps)):
            ind = range(timestamps[trial, 0], timestamps[trial, 4] + 1)

            for i in ind:
                # euclidean distance to center
                dis[i] = np.sqrt((data[i, 0] - p[0]) ** 2 + (data[i, 1] - p[1]) ** 2)
        di = dis.copy()
        for trial in range(len(timestamps)):
            ind = range(timestamps[trial, 0], timestamps[trial, 4] + 1)
            switch = np.where(di[ind] == np.min(di[ind]))[0][0] + ind[0]  # index over entire whl
            for i in ind:
                di[i] = di[i] - np.min(di[ind])
            di[ind[0]:switch + 1] = max(di[ind[0]:switch + 1]) - di[ind[0]:switch + 1]
            di[switch + 1:ind[-1]] = di[switch] + di[switch + 1:ind[-1]]
            di[ind[0]:ind[-1]] = di[ind[0]:ind[-1]] - di[switch] + 100  # aligning to centre 100

        return di

    @staticmethod
    def get_location_trials(whl, timestamps):
        # returns dictionary with location for selected trials

        # dictionary with locations
        loc = {}
        # go through selected trials
        for trial_ID in range(timestamps.shape[0]):
            # -----------------------------------------------
            #  timestamps: 20kHz/512 --> 25.6 ms per time bin
            #  whl: 20kHz/512 --> both have the same order of magnitude
            t_start = timestamps[trial_ID, 0]
            t_end = timestamps[trial_ID, 1]
            # select locations that correspond to time interval of the trial
            loc_trial = whl[t_start:t_end]
            # append data
            loc["trial" + str(trial_ID)] = loc_trial

        return loc

    """#################################################################################################################
    # Getter functions
    #################################################################################################################"""

    def get_experiment_phase_id(self):
        return self.experiment_phase_id

    def get_experiment_phase(self):

        if len(self.experiment_phase) == 1:
            return self.experiment_phase[0]
        else:
            return self.experiment_phase



