import numpy as np


class Parameters:
    """class that stores parameters"""

    def __init__(self):
        # DATA DESCRIPTION
        # --------------------------------------------------------------------------------------------------------------

        # TODO: delete this section --> only want general analysis parameters
        # session and file name
        self.data_description = ""
        self.session_name = ""
        self.file_name = ""
        
        # ANALYSIS PARAMETERS
        # --------------------------------------------------------------------------------------------------------------

        # check again that all points lie within cheeseboard
        # ------------------------------------------------------------------------------------------------------------------
        self.additional_spatial_filter = False

        # binning
        # "temporal" --> counts spikes per time bin and divides by time bin size
        # "temporal_spike" --> counts spikes per time bin
        self.binning_method = "temporal_spike"

        # sleep type: "nrem", "rem", "sw", "all"
        self.sleep_type = ""

        # z-score bins
        self.z_score = False

        # interval for temporal binning in s
        self.time_bin_size = 0

        # number of temporal bins
        self.nr_time_bins = 12

        # grid size for place fields etc in cm
        self.spatial_resolution = 5

        # speed filter, in cm/s --> usually between 5 and 10 (Jozsef uses 5!)
        self.speed_filter = 0

        # interval for spatial binning in cm
        self.spatial_bin_size = 50

        # spatial bins to exclude: e.g. first 2 (e.g 0-10cm and 10-20cm) and last (190-200cm) --> [2,-1]
        self.spat_bins_excluded = []

        # exclude population vectors with all zero values
        self.zero_filter = False
        
        # define method for dimensionality reduction
        # "MDS" multi dimensional scaling
        # "PCA" principal component analysis
        # "TSNE"
        # "isomap"
        self.dr_method = "MDS"
        
        # first parameter of method:
        # MDS --> p1: difference measure ["jaccard","cos","euclidean
        # PCA --> p1 does not exist --> ""
        self.dr_method_p1 = "jaccard"
        
        # second parameter of method:
        # MDS --> p2: number of components
        # PCA --> p2: number of components
        self.dr_method_p2 = 2
        
        # third parameter of method:
        # MDS + jaccard --> make binary: if True --> populaiton vectors are first made binary
        self.dr_method_p3 = True
        
        # number of trials to compare
        self.nr_of_trials = 21
        # selected trial
        self.sel_trial = 3

        # which kind of splits to use for cross validation ("standard_k_fold", "custom_splits")
        self.cross_val_splits = None
        
        # QUANTITATIVE ANALYSIS
        # ------------------------------------------------------------------------------------------------------------------
        # statistical method: Kruskal-Wallis --> "KW", Mann-Whitney-U --> "MWU"
        self.stats_method = "MWU"
        
        # alpha value
        self.stats_alpha = 0.01
        
        # remapping characteristic: percent of total distance that is used to compute the number of needed cells
        # (default: 0.8 --> 80%)
        self.percent_of_total_distance = 0.8
        # how many times is the order permuted to compute cell contribution
        self.nr_order_permutations = 500
        
        # PLOTTING PARAMETERS
        # ------------------------------------------------------------------------------------------------------------------
        
        # lines in scatter plot
        self.lines = False
        
        # sort cells according to peak for visualization
        self.sort_cells = True

        # saving directory & name for results dictionary
        self.result_dictionary_name = "RESULT_DIC"
        self.saving_dir_result_dictionary = "temp_data_old/"
        
        # length of spatial segment for plotting (track [200cm] will be divided into equal length segments)
        # set to 20: TODO --> adapt for different lengths
        self.spat_seg_plotting = 20
        
        # saving figure file name
        self.plot_file_name = "trans_analysis" + "_" + self.dr_method + "_" + self.dr_method_p1 + "_" \
                                      + str(self.dr_method_p2) + "D" + self.binning_method
        self.saving_dir_maps = "temp_data/"
        # TODO: automatically use maximum value from all data for axis limits
        # axis limit for plotting
        # jaccard: [-0.2,0.2]
        # cos: [-1,1]
        # 3D: [-0.5,0.5]
        # tSNE 2D: -50,50
        # PCA: -10,10
        # axis_lim = np.zeros(6)
        # axis_lim[0] = axis_lim[2]= axis_lim[4]= -50
        # axis_lim[1] = axis_lim[3] = axis_lim[5] =50
        # self.axis_lim = axis_lim
        self.axis_lim = []


