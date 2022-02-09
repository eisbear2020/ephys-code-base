########################################################################################################################
#
#   Plotting functions
#
#   Description:
#
#       - functions that help plotting results from analysis
#
#   Author: Lars Bollmann
#
#   Created: 21/03/2019
#
#   Structure:
#
#       - plotActMat: plot activation matrix (matrix of population vectors)
#       - plot2DScatter: generates 2D scatter plots for one or multiple data sets
#       - plot3DScatter: generates 3D scatter plots for one or multiple data sets
#
#
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from collections import OrderedDict
import matplotlib.colors as clr
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from scipy import stats
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from matplotlib.colors import LogNorm
from statsmodels import robust
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
from scipy.stats import pearsonr, entropy, spearmanr, sem, mannwhitneyu, wilcoxon, ks_2samp, multivariate_normal
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import mpl_toolkits.mplot3d.axes3d as p3

# use dark background for presentation
plt.style.use('dark_background')


def plot_act_mat(act_mat, params, cell_type=""):
    # check if spatial or temporal binning
    if params.binning_method == "temporal":
        bin_interval = params.time_bin_size
        # plot activation matrix (matrix of population vectors)
        plt.imshow(act_mat, vmin=0, vmax=act_mat.max(), cmap='jet', aspect='auto')
        plt.imshow(act_mat, interpolation='nearest', aspect='auto', cmap="jet",
                   extent=[0, act_mat.shape[1], act_mat.shape[0] - 0.5, 0.5])
        plt.ylabel("CELL ID")
        plt.xlabel("TIME BINS")
        plt.title("SPIKE RATE RASTER ("+ str(bin_interval) +"s bin)"+" "+cell_type)
        a = plt.colorbar()
        a.set_label("FIRING RATE / Hz")
    elif params.binning_method == "temporal_spike":
        bin_interval = params.time_bin_size
        # plot activation matrix (matrix of population vectors)
        plt.imshow(act_mat, vmin=0, vmax=act_mat.max(), cmap='jet', aspect='auto')
        plt.imshow(act_mat, interpolation='nearest', aspect='auto', cmap="jet",
                   extent=[0, act_mat.shape[1], act_mat.shape[0] - 0.5, 0.5])
        plt.ylabel("CELL ID")
        plt.xlabel("TIME BINS")
        plt.title("SPIKE RASTER ("+ str(bin_interval) +"s bin)"+" "+cell_type)
        a = plt.colorbar()
        if params.z_score:
            a.set_label("Z SCORE")
        else:
            a.set_label("# SPIKES")
    elif params.binning_method == "temporal_binary":
        bin_interval = params.time_bin_size
        # invert for viewing purpose (white spikes on black)
        act_mat = ~np.array(act_mat, dtype=bool)
        act_mat = np.array(act_mat, dtype=int)
        # plot activation matrix (matrix of population vectors)
        plt.imshow(act_mat, vmin=0, vmax=act_mat.max(), aspect='auto', cmap='Greys')
        plt.imshow(act_mat, interpolation='nearest', aspect='auto',cmap='Greys',
                   extent=[0, act_mat.shape[1], act_mat.shape[0] - 0.5, 0.5])
        plt.ylabel("CELL ID")
        plt.xlabel("TIME BINS")
        plt.title("BINARY RASTER ("+ str(bin_interval) +"s bin)"+" "+cell_type)
    elif params.binning_method == "spatial":
        bin_interval = params.time_bin_size
        # plot activation matrix (matrix of population vectors)
        plt.imshow(act_mat, vmin=0, vmax=act_mat.max(), cmap='jet', aspect='auto')
        plt.imshow(act_mat, interpolation='nearest', aspect='auto', cmap="jet",
                   extent=[0, act_mat.shape[1], act_mat.shape[0] - 0.5, 0.5])
        plt.ylabel("CELL ID")
        plt.xlabel("SPATIAL BINS / CM")
        plt.title("RATE MAP ("+ str(bin_interval) +"CM bin)")
        a = plt.colorbar()
        a.set_label("FIRING RATE / Hz")
    elif params.binning_method == "spike_binning":
        spikes_per_bin = params.spikes_per_bin
        # plot activation matrix (matrix of population vectors)
        plt.imshow(act_mat, vmin=0, vmax=act_mat.max(), cmap='jet', aspect='auto')
        plt.imshow(act_mat, interpolation='nearest', aspect='auto', cmap="jet",
                   extent=[0, act_mat.shape[1], act_mat.shape[0] - 0.5, 0.5])
        plt.ylabel("CELL ID")
        plt.xlabel("TIME BINS")
        plt.title("SPIKE BINNING RASTER ("+ str(spikes_per_bin) +" spikes per bin)"+" "+cell_type)
        a = plt.colorbar()
        a.set_label("# SPIKES")

    plt.show()


def plot_2D_scatter(ax, mds, params, data_sep=None, data_sep2=None, loc_vec = [], labels=["DAT1", "DAT2", "DAT3"]):
    # generates 2D scatter plot with selected data --> for more than 1 data set, data_sep needs to be defined to
    # separate the data sets
    # for more than one data set
    if data_sep is not None:
        # if lines between points should be drawn
        if params.lines:
            for i,c in enumerate(mds):
                if i < data_sep-1:
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], c="#ff3300", marker="o", markerfacecolor='#990000',
                            label="DATA 1")
                elif data_sep <= i:
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], c="#0099cc", marker="o", markerfacecolor="#000099",
                            label="DATA 2")
                if data_sep2 is not None:
                    if data_sep2 <= i:
                        ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], c="yellow", marker="o", markerfacecolor="yellow",
                                label="DATA 3")
            # ax.scatter(mds[:data_sep, 0], mds[:data_sep, 1], color="b",label=params.data_descr[0])
            # ax.scatter(mds[data_sep:, 0], mds[data_sep:, 1], color="r",label=params.data_descr[1])
            ax.scatter(mds [0, 0], mds [0, 1],color="white", marker="x",label="start",zorder=200)
            ax.scatter(mds[data_sep-1, 0], mds[data_sep-1, 1], color="white", label="end",zorder=200)
            ax.scatter(mds [data_sep, 0], mds [data_sep, 1],color="white", marker="x",label="start",zorder=200)
            ax.scatter(mds[-1, 0], mds[-1, 1], color="white", label="end",zorder=200)
        # color points instead of lines
        else:
            # check if positions are provided
            if len(loc_vec):
                # length of track
                s_l = params.spat_seg_plotting
                l_track = 200
                nr_seg = int(l_track/s_l)
                col_map_blue = cm.Reds(np.linspace(0, 1, nr_seg + 5))
                col_map_red = cm.Blues(np.linspace(0, 1, nr_seg + 5))
                col_map_blue = col_map_blue[5:,:]
                col_map_red = col_map_red[5:, :]
                # linearized track is 200 cm long
                norm_loc_vec = loc_vec / s_l
                for i,c in enumerate(mds):
                    if i <= data_sep:
                        ax.scatter(mds[i, 0], mds[i, 1],  color=col_map_blue[int(np.ceil(norm_loc_vec[i][0]))-1, :],
                                   label=params.data_descr[0]+ str(int(np.ceil(norm_loc_vec[i][0]))*s_l)+" cm")
                    elif data_sep < i:
                        ax.scatter(mds[i, 0], mds[i, 1], color=col_map_red[int(np.ceil(norm_loc_vec[i][0]))-1, :],
                                   label=params.data_descr[1] + str(
                                       int(np.ceil(norm_loc_vec[i + 1][0])) * s_l) + " cm")

            else:
                # get colors for number of data sets
                if data_sep.shape[0] > 3:
                    colors = plt.cm.get_cmap("tab20b", data_sep.shape[0])
                else:
                    colors = ["r", "b", "y"]
                start=0
                # gp through different data_sep and plot
                for i, (end, label) in enumerate(zip(data_sep, labels)):
                    if data_sep.shape[0] > 3:
                        ax.scatter(mds[start:end, 0], mds[start:end, 1], marker="o", color=colors(i), label=label)
                    else:
                        ax.scatter(mds[start:end, 0], mds[start:end, 1], marker="o", color=colors[i], label=label)
                    start = end

    # for one data set
    else:
        # draw lines if option is set to True
        if params.lines:
            # use locations for line coloring if location vector is provided
            if len(loc_vec):
                # length of track
                s_l = params.spat_seg_plotting
                l_track = 200
                nr_seg = int(l_track/s_l)
                col_map = cm.rainbow(np.linspace(0, 1, nr_seg))
                # linearized track is 200 cm long
                norm_loc_vec = loc_vec / s_l
                for i in range(0, mds.shape[0] - 1):
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], color=col_map[int(np.ceil(norm_loc_vec[i+1]))-1, :],
                            label=str(int(np.ceil(norm_loc_vec[i+1]))*s_l)+" cm")

            else:
                colors = cm.cool(np.linspace(0, 1, mds.shape[0] - 1))
                for i, c in zip(range(0, mds.shape[0] - 1), colors):
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], color=c)
            # plt.title(title)
            ax.scatter(mds[:, 0], mds[:, 1], color="grey")
            ax.scatter(mds[0, 0], mds[0, 1], color="white", marker="x", label="start", zorder=200)
            ax.scatter(mds[-1, 0], mds[-1, 1], color="white", label="end", zorder=200)

        else:
            # use locations for point coloring if location vector is provided
            if len(loc_vec):
                # length of track
                s_l = params.spat_seg_plotting
                l_track = 200
                nr_seg = int(l_track/s_l)
                col_map = cm.rainbow(np.linspace(0, 1, nr_seg))
                # linearized track is 200 cm long
                norm_loc_vec = loc_vec / s_l
                for i in range(mds.shape[0]):
                    ax.scatter(mds[i, 0], mds[i, 1], color=col_map[int(np.ceil(norm_loc_vec[i]))-1, :],
                            label=str(int(np.ceil(norm_loc_vec[i]))*s_l)+" cm")
            else:
                # no locations are provided (e.g. sleep) color code with increasing time
                colors = cm.cool(np.linspace(0, 1, mds.shape[0]))
                ax.scatter(mds[1:-1, 0],mds[1:-1, 1], color=colors[1:-1])
                ax.scatter(mds[0, 0], mds[0, 1], color=colors[0], marker="x", label="start", zorder=200)
                ax.scatter(mds[-1, 0], mds[-1, 1], marker="x",color=colors[-1], label="end", zorder=200)
        # if axis limits are defined apply them
        if len(params.axis_lim):
            axis_lim = params.axis_lim
            ax.set_xlim(axis_lim[0], axis_lim[1])
            ax.set_ylim(axis_lim[2], axis_lim[3])

    ax.set_yticklabels([])
    ax.set_xticklabels([])

def plot_3D_scatter(ax, mds, params, data_sep=None, data_sep2=None, loc_vec=[], labels=["DAT1", "DAT2", "DAT3"]):
    # generates 3D scatter plot with selected data --> for more than 1 data set, data_sep needs to be defined to
    # separate the data sets

    # for more than one data set
    if data_sep:
        data_sep = int(data_sep)

        # if lines between points should be drawn
        if params.lines:
            for i,c in enumerate(mds):
                if i < data_sep-1:
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1],mds[i:i + 2, 2], c="#ff3300", marker="o", markerfacecolor='#990000',
                            label="DATA 1")
                if data_sep2 is None:
                    if data_sep <= i:
                        ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1],mds[i:i + 2, 2], c="#0099cc", marker="o", markerfacecolor="#000099",
                                label="DATA 2")
                else:
                    if data_sep <= i and i < data_sep2-1:
                        ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], mds[i:i + 2, 2], c="#0099cc", marker="o",
                                markerfacecolor="#000099",
                                label="DATA 2")
                    elif data_sep2 <= i:
                        ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], mds[i:i + 2, 2], c="yellow", marker="o",
                                markerfacecolor="yellow", label="DATA 3")

            # ax.scatter(mds[:data_sep, 0], mds[:data_sep, 1],mds[:data_sep, 2], color="b")
            # ax.scatter(mds[data_sep:, 0], mds[data_sep:, 1],mds[data_sep:, 2], color="r")
            # ax.scatter(mds[:data_sep, 0], mds[:data_sep, 1],mds[:data_sep, 2], color="b",label=params.data_descr[0])
            # ax.scatter(mds[data_sep:, 0], mds[data_sep:, 1],mds[data_sep:, 2], color="r",label=params.data_descr[1])
            ax.scatter(mds [0, 0], mds [0, 1],mds [0, 2],color="white", marker="x",label="start",zorder=200)
            ax.scatter(mds[data_sep-1, 0], mds[data_sep-1, 1],mds[data_sep-1, 2], color="white", label="end",zorder=200)
            ax.scatter(mds [data_sep, 0], mds [data_sep, 1],mds [data_sep, 2],color="white", marker="x",label="start",zorder=200)
            ax.scatter(mds[-1, 0], mds[-1, 1],mds[-1, 2], color="white", label="end",zorder=200)

        # color points instead of lines
        else:
            # check if positions are provided
            if len(loc_vec):
                # length of track
                s_l = params.spatial_bin_size
                l_track = 200
                nr_seg = int(l_track/s_l)
                col_map_blue = cm.Reds(np.linspace(0, 1, nr_seg + 2))
                col_map_red = cm.Blues(np.linspace(0, 1, nr_seg +2 ))
                col_map_blue = col_map_blue[2:,:]
                col_map_red = col_map_red[2:, :]
                # linearized track is 200 cm long
                norm_loc_vec = loc_vec / s_l
                for i,c in enumerate(mds):
                    if i <= data_sep:
                        ax.scatter(mds[i, 0], mds[i, 1],mds[i,2],  color=col_map_blue[int(np.ceil(norm_loc_vec[i][0]))-1, :],
                                   label=params.data_descr[0]+ str(int(np.ceil(norm_loc_vec[i][0]))*s_l)+" cm")
                    elif data_sep < i:
                        ax.scatter(mds[i, 0], mds[i, 1], mds[i,2], color=col_map_red[int(np.ceil(norm_loc_vec[i][0]))-1, :],
                                   label=params.data_descr[1] + str(
                                       int(np.ceil(norm_loc_vec[i + 1][0])) * s_l) + " cm")
            else:

                ax.scatter(mds[:data_sep, 0], mds[:data_sep, 1], mds[:data_sep, 2], c="#ff3300", label=labels[0])
                if data_sep2 is None:
                    ax.scatter(mds[data_sep:, 0], mds[data_sep:, 1], mds[data_sep:, 2], c="#0099cc", label=labels[1])
                else:
                    ax.scatter(mds[data_sep:data_sep2, 0], mds[data_sep:data_sep2, 1],
                               mds[data_sep:data_sep2, 2], c="#0099cc", label="DATA 2")

                    ax.scatter(mds[data_sep2:, 0], mds[data_sep2:, 1], mds[data_sep2:, 2], c="yellow", label=labels[2])

    # for one data set
    else:
        if params.lines:
            # use locations for line coloring if location vector is provided
            if len(loc_vec):
                # length of track
                s_l = params.spatial_bin_size
                l_track = 200
                nr_seg = int(l_track/s_l)
                col_map = cm.rainbow(np.linspace(0, 1, nr_seg))
                # linearized track is 200 cm long
                norm_loc_vec = loc_vec / s_l
                for i in range(0, mds.shape[0] - 1):
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], mds[i:i + 2, 2],color=col_map[int(np.ceil(norm_loc_vec[i+1][0]))-1, :],
                            label=str(int(np.ceil(norm_loc_vec[i+1][0]))*s_l)+" cm")
            else:
                colors = cm.cool(np.linspace(0, 1, mds.shape[0] - 1))
                for i, c in zip(range(0, mds.shape[0] - 1), colors):
                    ax.plot(mds[i:i + 2, 0], mds[i:i + 2, 1], mds[i:i + 2, 2], color=c)

            ax.scatter(mds[:, 0], mds[:, 1], mds[:, 2], color="white")
            # ax.scatter(mds[0, 0], mds[0, 1], mds[0, 2], color="white", marker="x", label="start", zorder=200)
            # ax.scatter(mds[-1, 0], mds[-1, 1], mds[-1, 2], color="white", label="end", zorder=200)
        else:
            # use locations for point coloring if location vector is provided
            if len(loc_vec):
                # length of track
                s_l = params.spat_seg_plotting
                l_track = 200
                nr_seg = int(l_track/s_l)
                col_map = cm.rainbow(np.linspace(0, 1, nr_seg))
                # linearized track is 200 cm long
                norm_loc_vec = loc_vec / s_l
                for i in range(mds.shape[0]):
                    ax.scatter(mds[i, 0], mds[i, 1], mds[i, 2], color=col_map[int(np.ceil(norm_loc_vec[i][0]))-1, :],
                            label=str(int(np.ceil(norm_loc_vec[i][0]))*s_l)+" cm")
            else:
                colors = cm.cool(np.linspace(0, 1, mds.shape[0] - 1))
                ax.scatter(mds[:-1, 0], mds[:-1, 1], mds[:-1, 2], color=colors)

        # if axis limits are defined apply them
        if len(params.axis_lim):
            axis_lim = params.axis_lim
            ax.set_xlim(axis_lim[0], axis_lim[1])
            ax.set_ylim(axis_lim[2], axis_lim[3])
            ax.set_zlim(axis_lim[4], axis_lim[5])
    # hide labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    # set pane alpha value to zero --> transparent
    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
    ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
    ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))


def plot_compare(data, params, data_sep, rule_sep = []):
# plots data in one plot and colors rules differently if "rule_sep" is provided. Otherwise each trial is colored with a
# different color

    if rule_sep:
        # to color 2 different subsets of trials (e.g. for different rules): new_rule_trial --> first trial with new
        # rule

        # create rgba color map
        col_map = np.zeros((len(data_sep)+1,4))
        col_map[:rule_sep] = colors.to_rgba_array("r")
        col_map[rule_sep:] = colors.to_rgba_array("b")

        # create label array
        label_arr = np.zeros((len(data_sep)+1), dtype=object)
        label_arr[:rule_sep] = params.data_descr[0]
        label_arr[rule_sep:] = params.data_descr[1]

    else:
        col_map = cm.rainbow(np.linspace(0, 1, len(data_sep)))

    # 2D plot
    if params.dr_method_p2 == 2:
        fig, ax = plt.subplots()
        for data_ID in range(len(data_sep)-1):
            data_subset = data[int(data_sep[data_ID]):int(data_sep[data_ID + 1]), :]

            for i in range(0, data_subset.shape[0] - 1):
                # check if lines should be plotted
                if params.lines:
                    # check if trial or rule is meant to be colored
                    # rule
                    if rule_sep:
                        # check if lines should be plotted
                        ax.plot(data_subset[i:i + 2, 0], data_subset[i:i + 2, 1], color=col_map[data_ID, :],
                                label=label_arr[data_ID])
                    # trials
                    else:
                        ax.plot(data_subset[i:i + 2, 0], data_subset[i:i + 2, 1], color=col_map[data_ID, :],
                                label="TRIAL " + str(data_ID))
                    ax.scatter(data_subset[:, 0], data_subset[:, 1], color="grey")
                    # ax.scatter(data_subset[0, 0], data_subset[0, 1], color="black", marker="x", label="start", zorder=200)
                    # ax.scatter(data_subset[-1, 0], data_subset[-1, 1], color="black", label="end", zorder=200)
                # plot without lines
                else:
                    # color rules
                    if rule_sep:
                        ax.scatter(data_subset[i, 0], data_subset[i, 1], color=col_map[data_ID, :],
                                label=label_arr[data_ID])
                    # color trial
                    else:
                        ax.scatter(data_subset[i, 0], data_subset[i, 1], color=col_map[data_ID, :],
                                   label="TRIAL " + str(data_ID))

    # 3D plot

    if params.dr_method_p2 == 3:
        # create figure instance
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for data_ID in range(len(data_sep)-1):
            data_subset = data[int(data_sep[data_ID]):int(data_sep[data_ID + 1]), :]

            for i in range(0, data_subset.shape[0] - 1):
                # check if lines should be plotted
                if params.lines:
                    if rule_sep:
                        ax.plot(data_subset[i:i + 2, 0], data_subset[i:i + 2, 1], data_subset[i:i + 2, 2],
                                color=col_map[data_ID, :],
                                label=label_arr[data_ID])
                    else:
                        ax.plot(data_subset[i:i + 2, 0], data_subset[i:i + 2, 1], data_subset[i:i + 2, 2],
                                color=col_map[data_ID, :],
                                label="TRIAL " + str(data_ID))
                    ax.scatter(data_subset[:, 0], data_subset[:, 1], data_subset[:, 2], color="grey")
                    # ax.scatter(data_subset[0, 0], data_subset[0, 1], data_subset[0, 2], color="white", marker="x", label="start",
                    #            zorder=200)
                    # ax.scatter(data_subset[-1, 0], data_subset[-1, 1], data_subset[-1, 2], color="white", label="end", zorder=200)
                else:
                    if rule_sep:
                        ax.scatter(data_subset[i, 0], data_subset[i, 1], data_subset[i, 2],
                                color=col_map[data_ID, :],
                                label=label_arr[data_ID])
                    else:
                        ax.scatter(data_subset[i, 0], data_subset[i, 1], data_subset[i, 2],
                                color=col_map[data_ID, :],
                                label="TRIAL " + str(data_ID))
            # hide z label
            ax.set_zticklabels([])
            # set pane alpha value to zero --> transparent
            ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
            ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
            ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))

    # hide labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])


    # return figure for saving
    return fig


def scatter_animation(results, params, data_sep = None):
    # --------------------------------------------------------------------------------------------------------------
    # creates animation with scatter data (rotating)
    #
    # parameters:   - results: 2D or 3D array with data to be plotted
    #               - params: parameter dictionary
    #               - data_sep: separates data in case two data sets are plotted sequentially
    # --------------------------------------------------------------------------------------------------------------

    if params.dr_method_p2 == 2:
        # 2D scatter
        fig = plt.figure()
        ims = []

        for i, mat in enumerate(results):
            #
            # draw lines
            c = cm.Greys_r(np.linspace(0, 1, i+1))
            to_be_plotted = []

            dots = plt.scatter(results[:i + 1, 0], results[:i + 1, 1], edgecolors="grey", animated=True, facecolors="none")
            im = plt.scatter(mat[0], mat[1], animated=True, color="white")
            if i >= 1:
                for e in range(i):
                    lines, = plt.plot(results[e:e+2, 0], results[e:e+2, 1], color=c[e], animated=True)
                    to_be_plotted.append(lines)

                to_be_plotted.append(dots)
                to_be_plotted.append(im)
                ims.append(to_be_plotted)
            else:
                ims.append([im, dots])
        ani = animation.ArtistAnimation(fig, ims, interval=200, blit=False,
                                        repeat_delay=1000)

    elif params.dr_method_p2 == 3:

        if params.lines:

            if data_sep is None:
                # how many times to rotate before drawing the next data point
                mul_rot = 10

                # smoothness of rotation --> the smaller the smoother
                deg_per_iter = 0.5

                # First set up the figure, the axis, and the plot element we want to animate
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                # sct, = ax.plot([], [], [], c="#990000", marker="o", alpha=.5)
                sct1, = ax.plot([], [], [], c="gray", marker="o", markerfacecolor='white')
                sct2, = ax.plot([], [], [], c="#990000", marker="o", markerfacecolor="#990000")
                ttl = ax.text2D(0, 0, "", horizontalalignment='center',
                                verticalalignment='bottom', color="w", size=10,
                                transform=ax.transAxes)

                # animation function.  This is called sequentially
                def animate(i):
                    e = int(i / mul_rot)
                    if e <= results.shape[0]:
                        if np.mod(i, mul_rot) == 0:

                            x = results[:e,0]
                            y = results[:e,1]
                            z = results[:e, 2]
                            sct1.set_data(x, y)
                            sct1.set_3d_properties(z)

                            x = results[e-1:e+1,0]
                            y = results[e-1:e+1,1]
                            z = results[e-1:e + 1, 2]
                            sct2.set_data(x, y)
                            sct2.set_3d_properties(z)

                            ttl.set_text("START WINDOW: "+ str(round(params.time_bin_size*e,2))+"s")

                    ax.view_init(elev=45., azim=30+deg_per_iter*i)

                    return sct1, sct2, ttl

                ax.set_xlim(min(results[:,0]), max(results[:,0]))
                ax.set_ylim(min(results[:,1]), max(results[:,1]))
                ax.set_zlim(min(results[:,2]), max(results[:,2]))

                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_zticklabels([])
                # set pane alpha value to zero --> transparent
                ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
                ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
                ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))

                # call the animator.  blit=True means only re-draw the parts that have changed.
                anim = animation.FuncAnimation(fig, animate,
                                               frames=results.shape[0]*mul_rot+int(360/deg_per_iter),
                                               interval=20, blit=False, repeat_delay=1000)
            else:
                # how many times to rotate before drawing the next data point
                mul_rot = 10

                # smoothness of rotation --> the smaller the smoother
                deg_per_iter = 0.5

                # First set up the figure, the axis, and the plot element we want to animate
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                # data from first data set
                sct1, = ax.plot([], [], [], c="#ff3300", marker="o", markerfacecolor='#990000')
                # data from second data set
                sct2, = ax.plot([], [], [], c="#0099cc", marker="o", markerfacecolor="#000099")
                # new point
                sct3, = ax.plot([], [], [], c="white", marker="o", markerfacecolor="white")
                ttl = ax.text2D(0, 0, "", horizontalalignment='center',
                                verticalalignment='bottom', color="w", size=10,
                                transform=ax.transAxes)

                # animation function.  This is called sequentially
                def animate(i):
                    e = int(i / mul_rot)
                    if e <= results.shape[0]:
                        if np.mod(i, mul_rot) == 0:
                            if e < data_sep:
                                x = results[:e, 0]
                                y = results[:e, 1]
                                z = results[:e, 2]
                                sct1.set_data(x, y)
                                sct1.set_3d_properties(z)
                            else:
                                x = results[:data_sep, 0]
                                y = results[:data_sep, 1]
                                z = results[:data_sep, 2]
                                sct1.set_data(x, y)
                                sct1.set_3d_properties(z)

                                x = results[data_sep:e, 0]
                                y = results[data_sep:e, 1]
                                z = results[data_sep:e, 2]
                                sct2.set_data(x, y)
                                sct2.set_3d_properties(z)



                            x = results[e - 1:e + 1, 0]
                            y = results[e - 1:e + 1, 1]
                            z = results[e - 1:e + 1, 2]
                            sct3.set_data(x, y)
                            sct3.set_3d_properties(z)

                            ttl.set_text("START WINDOW: " + str(round(params.time_bin_size * e, 2)) + "s")

                    ax.view_init(elev=45., azim=30 + deg_per_iter * i)

                    return sct1, sct2, sct3, ttl

                ax.set_xlim(min(results[:, 0]), max(results[:, 0]))
                ax.set_ylim(min(results[:, 1]), max(results[:, 1]))
                ax.set_zlim(min(results[:, 2]), max(results[:, 2]))

                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_zticklabels([])
                # set pane alpha value to zero --> transparent
                ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
                ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
                ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))

                # call the animator.  blit=True means only re-draw the parts that have changed.
                anim = animation.FuncAnimation(fig, animate,
                                               frames=results.shape[0] * mul_rot + int(360 / deg_per_iter),
                                               interval=20, blit=False, repeat_delay=1000)

        else:

            # how many times to rotate before drawing the next data point
            mul_rot = 10

            # smoothness of rotation --> the smaller the smoother
            deg_per_iter = 0.5

            # First set up the figure, the axis, and the plot element we want to animate
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sct = ax.scatter([], [], [], facecolor="#990000", marker="o")

            # initialization function: plot the background of each frame
            def init():
                sct._offsets3d = ([], [], [])
                return sct

            # animation function.  This is called sequentially
            def animate(i):
                e = int(i / mul_rot)
                if e <= results.shape[0]:
                    # c = cm.Greys_r(np.linspace(0, 1, e + 1))
                    if np.mod(i, mul_rot) == 0:
                        x = results[:e + 1, 0]
                        y = results[:e + 1, 1]
                        z = results[:e + 1, 2]
                        sct._offsets3d = (x, y, z)
                ax.view_init(elev=45., azim=30 + deg_per_iter * i)

                return sct

            ax.set_xlim(min(results[:, 0]), max(results[:, 0]))
            ax.set_ylim(min(results[:, 1]), max(results[:, 1]))
            ax.set_zlim(min(results[:, 2]), max(results[:, 2]))

            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_zticklabels([])
            # set pane alpha value to zero --> transparent
            ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
            ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))
            ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 0.0))

            # call the animator.  blit=True means only re-draw the parts that have changed.
            anim = animation.FuncAnimation(fig, animate,
                                           frames=results.shape[0] * mul_rot + int(360 / deg_per_iter),
                                           interval=20, blit=False, repeat_delay=1000)


    plt.show()


def scatter_animation_parallel(results_1, results_2, params, video_file_name=None):
    # --------------------------------------------------------------------------------------------------------------
    # creates animation with scatter data (rotating)
    #
    # parameters:   - results: 2D or 3D array with data to be plotted
    #               - params: parameter dictionary
    #               - data_sep: separates data in case two data sets are plotted sequentially
    # --------------------------------------------------------------------------------------------------------------

    if params.dr_method_p2 == 2:
        # 2D scatter
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ims = []

        for i, (mat_1, mat_2) in enumerate(zip(results_1, results_2)):
            #
            # draw lines
            c = cm.Greys_r(np.linspace(0, 1, i+1))
            to_be_plotted_1 = []

            dots_1 = ax1.scatter(results_1[:i + 1, 0], results_1[:i + 1, 1], edgecolors="grey", animated=True, facecolors="none")
            im_1 = ax1.scatter(mat_1[0], mat_1[1], animated=True, color="white")
            dots_2 = ax2.scatter(results_2[:i + 1, 0], results_2[:i + 1, 1], edgecolors="grey", animated=True, facecolors="none")
            im_2 = ax2.scatter(mat_2[0], mat_2[1], animated=True, color="white")
            if i >= 1:
                for e in range(i):
                    lines, = ax1.plot(results_1[e:e+2, 0], results_1[e:e+2, 1], color=c[e], animated=True)
                    to_be_plotted_1.append(lines)
                    lines, = ax2.plot(results_2[e:e+2, 0], results_2[e:e+2, 1], color=c[e], animated=True)
                    to_be_plotted_1.append(lines)

                to_be_plotted_1.append(dots_1)
                to_be_plotted_1.append(dots_2)
                to_be_plotted_1.append(im_1)
                to_be_plotted_1.append(im_2)
                ims.append(to_be_plotted_1)
            else:
                ims.append([im_1, dots_1, im_2, dots_2])

        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                        repeat_delay=1000)
        if video_file_name is not None:
            ani.save("movies" + "/" + video_file_name)

    elif params.dr_method_p2 == 3:
        raise Exception("3D parallel scatter not implemented yet!")


def plot_remapping_summary(cross_diff, within_diff_1, within_diff_2, stats_array, params):
    x_axis = np.arange(0, 200, params.spatial_bin_size)
    x_axis = x_axis[params.spat_bins_excluded[0]:params.spat_bins_excluded[-1]]

    med_1 = np.median(within_diff_1, axis=1)
    med_2 = np.median(within_diff_2, axis=1)

    med = np.median(cross_diff, axis=1)
    err = robust.mad(cross_diff, c=1, axis=1)


    plt.subplot(2, 2, 1)
    plt.errorbar(x_axis, med, yerr=err, fmt="o")
    plt.grid()
    plt.xlabel("MAZE LOCATION / cm")
    plt.ylabel("MEDIAN DISTANCE (COS) - MED & MAD")
    plt.title("ABSOLUTE")

    # add significance marker
    for i, p_v in enumerate(stats_array[:, 1]):
        if p_v < params.stats_alpha:
            plt.scatter(x_axis[i] + 2, med[i] + 0.02, marker="*", edgecolors="Red",
                        label=params.stats_method+", "+str(params.stats_alpha))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.subplot(2, 2, 2)
    plt.scatter(x_axis, med / med_1)
    plt.xlabel("MAZE LOCATION / cm")
    plt.ylabel("MEDIAN DISTANCE (COS)")
    plt.title("NORMALIZED BY DATA SET 1")
    plt.grid()

    # add significance marker
    for i, p_v in enumerate(stats_array[:, 1]):
        if p_v < params.stats_alpha:
            plt.scatter(x_axis[i] + 2, med[i] / med_1[i] + 0.05, marker="*", edgecolors="Red",
                        label=params.stats_method+", "+str(params.stats_alpha))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.subplot(2, 2, 3)
    plt.scatter(x_axis, med / med_2)
    plt.xlabel("MAZE LOCATION / cm")
    plt.ylabel("MEDIAN DISTANCE (COS)")
    plt.title("NORMALIZED BY DATA SET 2")
    plt.grid()

    # add significance marker
    for i, p_v in enumerate(stats_array[:, 1]):
        if p_v < params.stats_alpha:
            plt.scatter(x_axis[i] + 2, med[i] / med_2[i] + 0.05, marker="*", edgecolors="Red",
                        label=params.stats_method+", "+str(params.stats_alpha))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.subplot(2, 2, 4)
    plt.scatter(x_axis, stats_array[:, 1])
    plt.hlines(params.stats_alpha, min(x_axis), max(x_axis), colors="Red", label=str(params.stats_alpha))
    plt.xlabel("MAZE LOCATION / cm")
    plt.ylabel("P-VALUE")
    plt.title(params.stats_method+": WITHIN-RULE vs. ACROSS-RULES")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_cell_charact(cell_avg_rate_map, cohens_d, cell_to_diff_contribution, xlabel,
                      sort_cells):

    if sort_cells:
        # sort according to appearance of peak
        peak_array = np.zeros(cell_avg_rate_map.shape[0])
        # find peak in for every cell
        for i, cell in enumerate(cell_avg_rate_map):
            # if no activity
            if max(cell) == 0.0:
                peak_array[i] = -1
            else:
                peak_array[i] = np.argmax(cell)

        peak_array += 1
        peak_order = peak_array.argsort()
        cell_avg_rate_map = cell_avg_rate_map[np.flip(peak_order[::-1], axis=0), :]
        cohens_d = cohens_d[np.flip(peak_order[::-1], axis=0), :]
        cell_to_diff_contribution = cell_to_diff_contribution[np.flip(peak_order[::-1], axis=0), :]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.subplots_adjust(wspace=0.1)

    im1 = ax1.imshow(cell_avg_rate_map, interpolation='nearest', aspect='auto',cmap="jet",extent=[min(xlabel),max(xlabel),cell_avg_rate_map.shape[0]-0.5,0.5])
    ax1_divider = make_axes_locatable(ax1)
    # add an axes to the right of the main axes.
    cax1 = ax1_divider.append_axes("top", size="7%", pad="2%")
    cb1 = colorbar(im1, cax=cax1,orientation="horizontal")
    cax1.xaxis.set_ticks_position("top")
    #ax1.set_xlabel("LINEARIZED POSITION / cm")
    ax1.set_ylabel("CELLS SORTED")
    cax1.set_title("AVERAGE \n FIRING RATE")

    # hide y label
    ax2.set_yticklabels([])
    im2 = ax2.imshow(cohens_d, interpolation='nearest', aspect='auto',cmap="jet",extent=[min(xlabel),max(xlabel),cell_avg_rate_map.shape[0]-0.5,0.5])
    ax2_divider = make_axes_locatable(ax2)
    # add an axes above the main axes.
    cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
    cb2 = colorbar(im2, cax=cax2, orientation="horizontal")
    # change tick position to top. Tick position defaults to bottom and overlaps
    # the image.
    cax2.xaxis.set_ticks_position("top")
    ax2.set_xlabel("LINEARIZED POSITION / cm")
    cax2.set_title("EFFECT SIZE: \n RULE A vs. RULE B")

    # hide y label
    ax3.set_yticklabels([])
    im3 = ax3.imshow(np.log(cell_to_diff_contribution), interpolation='nearest',
                     aspect='auto',cmap="jet",extent=[min(xlabel),max(xlabel),cell_avg_rate_map.shape[0]-0.5,0.5])
    ax3_divider = make_axes_locatable(ax3)
    # add an axes above the main axes.
    cax3 = ax3_divider.append_axes("top", size="7%", pad="2%")
    cb3 = colorbar(im3, cax=cax3, orientation="horizontal")
    # change tick position to top. Tick position defaults to bottom and overlaps
    # the image.
    cax3.xaxis.set_ticks_position("top")
    #ax3.set_xlabel("LINEARIZED POSITION / cm")
    cax3.set_title("REL. CONTRIBUTION TO DIFF \n (RULE A vs. RULE B) / LOG")

    # # hide y label
    # ax4.set_yticklabels([])
    # im4 = ax4.imshow(cell_to_p_value_contribution, interpolation='nearest', aspect='auto',cmap="jet",extent=[min(xlabel),max(xlabel),cell_avg_rate_map.shape[0]-0.5,0.5])
    # ax4_divider = make_axes_locatable(ax4)
    # # add an axes above the main axes.
    # cax4 = ax4_divider.append_axes("top", size="7%", pad="2%")
    # cb4 = colorbar(im4, cax=cax4, orientation="horizontal")
    # # change tick position to top. Tick position defaults to bottom and overlaps
    # # the image.
    # cax4.xaxis.set_ticks_position("top")
    # ax4.set_xlabel("LINEARIZED POSITION / cm")
    # cax4.set_title("CHANGE OF P-VALUE KW(RULE A vs. RULE B)")

    plt.show()


def plot_transition_comparison(x_axis, dics, rel_dics, params, stats_array, measure):

    fig, ax = plt.subplots(2,2)

    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[1, 0]
    ax4 = ax[1, 1]

    for data_set_ID, dic in enumerate(dics):
        med = np.full(x_axis.shape[0], np.nan)
        all_values = np.empty((1, 0))
        for i, key in enumerate(dic):
            if dic[key].size == 0:
                continue
            med[i] = np.median(dic[key],axis=1)
            all_values = np.hstack((all_values, dic[key]))
            err = robust.mad(dic[key], c=1, axis=1)
            ax1.errorbar(x_axis[i]+data_set_ID*2, med[i], yerr=err,ecolor="gray")
        ax1.plot(x_axis+data_set_ID*2,med, marker="o", label=params.data_descr[data_set_ID])
        ax1.set_title(measure.upper() + " BETWEEN SUBS. POP. VECT.")
        ax1.set_ylabel(measure.upper() + " - MED & MAD")
        ax1.set_xlabel("MAZE POSITION / CM")
        ax1.legend()

        ax3.hist(all_values[~np.isnan(all_values)],bins=50, alpha=0.5, label=params.data_descr[data_set_ID])
        ax3.set_title("HIST OF " +measure.upper()+" BETWEEN SUBS. POP. VECT.")
        ax3.set_xlabel(measure.upper())
        ax3.set_ylabel("COUNTS")
        ax3.legend()

    # add significance marker
    for i, p_v in enumerate(stats_array):
        if p_v < params.stats_alpha:
            ax1.scatter(x_axis[i] + 2, ax1.get_ylim()[1]-0.1*ax1.get_ylim()[1], marker="*", edgecolors="Red",
                        label=params.stats_method + ", " + str(params.stats_alpha))
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())


    for data_set_ID, rel_dic in enumerate(rel_dics):

        # calculate relative step length
        rel_med = np.full(x_axis.shape[0], np.nan)
        all_rel_values = np.empty((1, 0))
        for i, key in enumerate(rel_dic):
            if rel_dic[key].size == 0:
                continue
            rel_med[i] = np.median(rel_dic[key],axis=1)
            all_rel_values = np.hstack((all_rel_values, rel_dic[key]))
            err = robust.mad(rel_dic[key], c=1, axis=1)
            ax2.errorbar(x_axis[i]+data_set_ID*2, rel_med[i], yerr=err,ecolor="gray")
        ax2.plot(x_axis+data_set_ID*2, rel_med, marker="o", label=params.data_descr[data_set_ID])
        ax2.set_title("RELATIVE CHANGE OF "+measure.upper()+" BETWEEN SUBS. POP. VECT.")
        ax2.set_ylabel("RELATIVE CHANGE")
        ax2.set_xlabel("MAZE POSITION / CM")

        ax4.hist(all_rel_values[~np.isnan(all_rel_values) & ~np.isinf(all_rel_values)],
                 bins=50, alpha=0.5, label=params.data_descr[data_set_ID])
        ax4.set_title("HIST OF RELATIVE CHANGE OF "+measure.upper()+" BETWEEN SUBS. POP. VECT.")
        ax4.set_xlabel("RELATIVE CHANGE")
        ax4.set_ylabel("COUNTS")
        ax4.legend()

    plt.show()


def plot_operations_comparison(x_axis, operation_dics, nr_of_cells_arr, params):

    fig, ax = plt.subplots(2, 2)
    ax1 = ax[0, 0]
    ax2 = ax[0, 1]
    ax3 = ax[1, 0]

    for data_set_ID, (operation_dic, nr_of_cells) in enumerate(zip(operation_dics, nr_of_cells_arr)):
        med = np.full(x_axis.shape[0], np.nan)
        all_values = np.empty((1, 0))
        for i, key in enumerate(operation_dic):
            if operation_dic[key].size == 0:
                continue
            med[i] = np.median(operation_dic[key][0]/nr_of_cells*100)
            #all_values = np.hstack((all_values, operation_dic[key][0]))
            err = robust.mad(operation_dic[key][0]/nr_of_cells*100, c=1)
            ax1.errorbar(x_axis[i]+data_set_ID*2, med[i], yerr=err, ecolor="gray")
        ax1.plot(x_axis+data_set_ID*2, med, marker="o", label=params.data_descr[data_set_ID])
        ax1.set_title("SILENCED CELLS")
        ax1.set_ylabel("% OF CELLS")
        ax1.set_xlabel("MAZE POSITION / CM")
        ax1.set_ylim(0, 30)
        ax1.legend(loc=1)

        med = np.full(x_axis.shape[0], np.nan)
        all_values = np.empty((1, 0))
        for i, key in enumerate(operation_dic):
            if operation_dic[key].size == 0:
                continue
            med[i] = np.median(operation_dic[key][1]/nr_of_cells*100)
            #all_values = np.hstack((all_values, operation_dic[key][0]))
            err = robust.mad(operation_dic[key][1]/nr_of_cells*100, c=1)
            ax2.errorbar(x_axis[i]+data_set_ID*2, med[i], yerr=err, ecolor="gray")
        ax2.plot(x_axis+data_set_ID*2, med, marker="o", label=params.data_descr[data_set_ID])
        ax2.set_title("UNCHANGED CELLS")
        ax2.set_ylabel("% OF CELLS")
        ax2.set_xlabel("MAZE POSITION / CM")
        ax2.set_ylim(60, 95)
        ax2.legend(loc=1)

        med = np.full(x_axis.shape[0], np.nan)
        all_values = np.empty((1, 0))
        for i, key in enumerate(operation_dic):
            if operation_dic[key].size == 0:
                continue
            med[i] = np.median(operation_dic[key][2]/nr_of_cells*100)
            #all_values = np.hstack((all_values, operation_dic[key][0]))
            err = robust.mad(operation_dic[key][2]/nr_of_cells*100, c=1)
            ax3.errorbar(x_axis[i]+data_set_ID*2, med[i], yerr=err, ecolor="gray")
        ax3.plot(x_axis+data_set_ID*2, med, marker="o", label=params.data_descr[data_set_ID])
        ax3.set_title("ACTIVATED CELLS")
        ax3.set_ylabel("% OF CELLS")
        ax3.set_xlabel("MAZE POSITION / CM")
        ax3.set_ylim(0,30)
        ax3.legend(loc=1)

    plt.show()


def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])


def plot_optimal_correlation_time_shift(data, params, full, nr_cells_pop1 = None):
    # make a color map of fixed colors
    #cmap = colors.ListedColormap(['white', 'red'])
    cmap = cm.Spectral
    bounds = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    img = plt.imshow(data.T, interpolation='none', aspect='auto', cmap=cmap,
               norm=norm)
    a = plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-3, -2, -1, -0, 1, 2, 3])
    a.set_label("OPTIMAL TIME BINS SHIFT")
    if full:
        plt.xlabel("CELL IDS")
        plt.ylabel("CELL IDS")
        plt.hlines(nr_cells_pop1+0.5, -0.5, data.T.shape[1]-0.5, colors="black")
        plt.vlines(nr_cells_pop1+0.5, -0.5, data.T.shape[1]-0.5, colors="black")
        plt.title("OPTIMAL TIME BIN SHIFT FOR CORRELATION VALUE (" + str(params.time_bin_size) + "s)")

    else:
        plt.xlabel("CELL IDS POPULATION A")
        plt.ylabel("CELL IDS POPULATION B")
        plt.title("OPT. TIME BIN SHIFT FOR ACROSS CORR. VALUE ("+str(params.time_bin_size)+"s)")
    plt.show()


def plot_optimal_correlation_time_shift_hist(best_shift_within_pop1, best_shift_within_pop2, best_shift_across,
                                             params, cell_type_array):

    fig, axs = plt.subplots(3, 1, tight_layout=True)
    bins = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    axs[0].set_title("POP1: "+cell_type_array[0], fontsize=10)
    axs[0].hist(best_shift_within_pop1, bins=bins)
    axs[0].set_yscale('log', nonposy='clip')
    axs[1].set_title("POP2: "+cell_type_array[1] , fontsize=10)
    axs[1].hist(best_shift_within_pop2, bins=bins)
    axs[1].set_yscale('log', nonposy='clip')
    axs[2].set_title("ACROSS", fontsize=10)
    axs[2].hist(best_shift_across, bins=bins)
    axs[2].set_yscale('log', nonposy='clip')

    fig.suptitle("OPTIMAL TIME BIN SHIFT ("+ str(params.time_bin_size) + "s)"+" WITHOUT DIAGONAL",
                 fontsize=10, y=1)

    plt.xlabel("OPTIMAL TIME BIN SHIFT")
    plt.show()


    fig, axs = plt.subplots(3, 1, tight_layout=True)
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5]
    axs[0].set_title("POP1: "+cell_type_array[0], fontsize=10)
    axs[0].hist(np.abs(best_shift_within_pop1), bins=bins)
    axs[0].set_yscale('log', nonposy='clip')
    axs[1].set_title("POP2: "+cell_type_array[1] , fontsize=10)
    axs[1].hist(np.abs(best_shift_within_pop2), bins=bins)
    axs[1].set_yscale('log', nonposy='clip')
    axs[2].set_title("ACROSS", fontsize=10)
    axs[2].hist(np.abs(best_shift_across), bins=bins)
    axs[2].set_yscale('log', nonposy='clip')

    fig.suptitle("OPTIMAL ABS. TIME BIN SHIFT ("+ str(params.time_bin_size) + "s)"+" WITHOUT DIAGONAL",
                 fontsize=10, y=1)

    plt.xlabel("OPTIMAL ABS. TIME BIN SHIFT")
    plt.show()


def plot_optimal_correlation_time_shift_edges(corr_across_time_shifted_sorted, corr_within1_time_shifted_sorted,
                                              corr_within2_time_shifted_sorted, params, shift_array, cell_type_array):
    plt.imshow(corr_across_time_shifted_sorted, interpolation='nearest', aspect='auto', cmap="jet")
    plt.ylabel("EDGES")
    plt.xlabel("TIME BIN SHIFT")
    plt.title("CORRELATION VALUES ACROSS (" + str(params.time_bin_size) + "s)")
    a = plt.colorbar()
    a.set_label("PEARSON R")
    plt.xticks(range(shift_array.shape[0]), shift_array)
    plt.show()

    plt.imshow(corr_within1_time_shifted_sorted, interpolation='nearest', aspect='auto', cmap="jet")
    plt.ylabel("EDGES")
    plt.xlabel("TIME BIN SHIFT")
    plt.title("CORRELATION VALUES WITHIN: "+ cell_type_array[0]+ " (" + str(params.time_bin_size) + "s)")
    a = plt.colorbar()
    a.set_label("PEARSON R")
    plt.xticks(range(shift_array.shape[0]), shift_array)
    plt.show()

    plt.imshow(corr_within2_time_shifted_sorted, interpolation='nearest', aspect='auto', cmap="jet")
    plt.ylabel("EDGES")
    plt.xlabel("TIME BIN SHIFT")
    plt.title("CORRELATION VALUES WITHIN: "+ cell_type_array[1]+ " (" + str(params.time_bin_size) + "s)")
    a = plt.colorbar()
    a.set_label("PEARSON R")
    plt.xticks(range(shift_array.shape[0]), shift_array)
    plt.show()


def cca_video(x_loadings, y_loadings, corr_coff, window_in_sec, time_bin_size, sliding, video_file_name):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    print(x_loadings.shape)

    ims = []

    for window in range(x_loadings.shape[0]):
        # print(corr_coff[window,:])

        im1 = ax1.scatter(range(x_loadings.shape[1]), x_loadings[window, :, 0], marker=".", c="r", label="CV1")
        im2 = ax1.scatter(range(x_loadings.shape[1]), x_loadings[window, :, 1], marker=".", c='blue', s=10,
                          label="CV2")
        im3 = ax1.scatter(range(x_loadings.shape[1]), x_loadings[window, :, 2], marker=".", c='w', s=5,
                          label="CV3")
        ax1.set_title(("SLIDING " if sliding else "")+ "WINDOW SIZE: "+ str(window_in_sec) + "s" +" - "
                      + str(time_bin_size)+"s TIME BIN", fontsize=10)
        ax1.set_ylabel("LOADINGS POP1")

        ttl = ax1.text(0.5, 2.4, str(corr_coff[window,:]), horizontalalignment='center', verticalalignment='bottom',
                       transform=ax2.transAxes)

        im4 = ax2.scatter(range(y_loadings.shape[1]), y_loadings[window, :, 0], marker=".", c="r", label="CV1")
        im5 = ax2.scatter(range(y_loadings.shape[1]), y_loadings[window, :, 1], marker=".", c='blue', s=10,
                          label="CV2")
        im6 = ax2.scatter(range(y_loadings.shape[1]), y_loadings[window, :, 2], marker=".", c='w', s=5,
                          label="CV3")

        ax2.set_xlabel("CELL ID")
        ax2.set_ylabel("LOADINGS POP2")

        handles, labels = ax1.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())

        handles, labels = ax2.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys())

        ims.append([im1, im2, im3, im4, im5, im6, ttl])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                    repeat_delay=1000)

    ani.save("movies/" + video_file_name)


def view_dyn_co_firing(co_firing_matrices, sorting, window_size, time_bin_size, nr_cells_pop_1,
                       saving=False, saving_dir="movies", video_file_name="movie.mp4"):

    ims = []
    fig = plt.figure()

    for i, mat in enumerate(co_firing_matrices):

        # # insert one extra col/row to show within
        # ins = np.ones(mat.shape[0])
        # mat = np.insert(mat, nr_cells_pop_1, ins, axis=1)
        # ins = np.ones(mat.shape[0] + 1).T
        # mat = np.insert(mat, nr_cells_pop_1, ins.T, axis=0)

        #  remove diagonal for better visualization
        mat = mat[~np.eye(mat.shape[0], dtype=bool)].reshape(mat.shape[0], -1)

        # plt.colorbar()
        # im = plt.imshow(mat, interpolation='nearest', aspect='auto', cmap="seismic", vmin=-1, vmax=1 , animated=True)
        im = plt.imshow(mat, interpolation='nearest', aspect='auto', cmap="hot", vmin=0, vmax=1,
                        animated=True)
        plt.hlines(nr_cells_pop_1+0.5, -0.5, mat.T.shape[0]-0.5, colors="white", linewidth=0.05)
        # -1 because diagonal was removed!
        plt.vlines(nr_cells_pop_1-1+0.5, -0.5, mat.T.shape[1]-0.5, colors="white", linewidth=0.05)

        im.axes.get_xaxis().set_ticks([])
        im.axes.get_yaxis().set_ticks([])
        if sorting == "sort_cells":
            im.axes.set_title("CO-FIRING - CELLS SORTED: MEAN\n TIME BIN SIZE: "+
                              str(time_bin_size*1000)+"ms,  WINDOW SIZE: "+str(window_size)+"s")
        if sorting == "sort_edges":
            im.axes.set_title("CO-FIRING - ENTRIES SORTED: MEAN/STD\n TIME BIN SIZE: "+
                              str(time_bin_size*1000)+"ms,  WINDOW SIZE: "+str(window_size)+"s")
        else:
            im.axes.set_title("CO-FIRING - TIME BIN SIZE: "+
                              str(time_bin_size*1000)+"ms,  WINDOW SIZE: "+str(window_size)+"s")

        ims.append([im])
    a = plt.colorbar()
    a.set_label("PEARSON R")

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=False,
                                    repeat_delay=1000)
    if saving:
        ani.save(saving_dir+"/"+video_file_name)
    else:
        plt.show()


def plot_pop_clusters(map, labels, params, nr_clusters, sel_range=None):
    if sel_range is not None:
        map = map[:,sel_range]
        labels=labels[sel_range]

    plt.subplot(2, 1, 1)
    plt.scatter(range(labels.shape[0]),labels, s=0.55, color="r")
    plt.xticks([])
    plt.xlim(0,labels.shape[0])
    plt.title("INFERRED MODES")
    plt.ylabel("MODE IDs")
    plt.subplot(2, 1, 2)
    if params.binning_method == "temporal" or params.binning_method == "temporal_spike":
        rate_map = plt.imshow(map, interpolation='nearest', aspect='auto', cmap="jet",
                   extent=[0, map.shape[1], map.shape[0] - 0.5, 0.5])
        plt.xlabel("TIME BINS ("+str(params.time_bin_size)+"s)")
        plt.ylabel("CELL IDs")
        # a = plt.colorbar(rate_map)
        # a.set_label("# SPIKES")
    elif params.binning_method == "temporal_binary":
        # invert for viewing purpose (white spikes on black)
        act_mat = ~np.array(map, dtype=bool)
        act_mat = np.array(act_mat, dtype=int)
        # plot activation matrix (matrix of population vectors)
        plt.imshow(act_mat, vmin=0, vmax=act_mat.max(), aspect='auto', cmap='Greys')
        plt.imshow(act_mat, interpolation='nearest', aspect='auto',cmap='Greys',
                   extent=[0, act_mat.shape[1], act_mat.shape[0] - 0.5, 0.5])
        plt.ylabel("CELL ID")
        plt.xlabel("TIME BINS")

    print(np.unique(labels).shape)
    plt.show()

    exit()
    fig = plt.figure()
    gs = fig.add_gridspec(6, 20)

    ax1 = fig.add_subplot(gs[0, :-1])
    ax1.set_title("CLUSTER IDs (#CLUSTERS: "+str(nr_clusters)+")")
    ax2 = fig.add_subplot(gs[1:, :-1])
    ax3 = fig.add_subplot(gs[1:, -1:])

    # plotting

    ax1.imshow(np.expand_dims(labels, 0), aspect="auto")
    ax1.axis("off")
    rate_map = ax2.imshow(map, interpolation='nearest', aspect='auto', cmap="jet",
               extent=[0, map.shape[1], map.shape[0] - 0.5, 0.5])
    ax2.set_xlabel("TIME BINS ("+str(params.time_bin_size)+"s)")
    ax2.set_ylabel("CELL IDs")
    a = plt.colorbar(rate_map, cax=ax3)
    a.set_label("# SPIKES")
    plt.show()


def plot_pop_cluster_analysis(labels, params, nr_clusters, transition_matrix, h_trans, k_val_mean, k_val_std):

    plt.subplot(2, 2, 3)
    plt.scatter(range(nr_clusters), h_trans)
    plt.xticks(range(nr_clusters))
    plt.title("TRANSITION ENTROPY")
    plt.xlabel("MODE")
    plt.ylabel("TRANSITION ENTROPY (bits)")
    plt.xticks(rotation='vertical', fontsize=7)

    plt.subplot(2, 2, 2)
    plt.imshow(transition_matrix)
    a = plt.colorbar()
    a.set_label("PROBABILITY")
    plt.title("TRANSITION PROB. \n BETWEEN MODES")
    plt.ylabel("BEFORE")
    plt.xlabel("AFTER")
    plt.xticks(range(nr_clusters),rotation='vertical', fontsize=5)
    plt.yticks(range(nr_clusters), fontsize=5)

    plt.subplot(2, 2, 1)
    bins = range(np.unique(labels).shape[0] + 1)
    plt.hist(labels, bins=bins)
    bins_labels(bins)
    plt.title("MODE ASSIGNMENT")
    plt.ylabel("TIME BINS PER MODE")
    plt.xlabel("MODE")
    plt.xticks(rotation='vertical', fontsize=7)

    plt.subplot(2, 2, 4)
    plt.errorbar(range(nr_clusters), k_val_mean, k_val_std, fmt="o")
    plt.xticks(range(nr_clusters))
    plt.title("<k> STATISTIC")
    plt.xlabel("MODE")
    plt.ylabel("%CELLS ACTIVE")
    plt.xticks(rotation='vertical', fontsize=7)

    plt.show()


def plot_ridge_weight_vectors(mat, cell_type_input, cell_type_output):
    max_val = np.max(np.abs((mat)))

    plt.tight_layout()
    plt.imshow(mat, interpolation='nearest', aspect='auto', cmap="seismic",
               vmin=-max_val, vmax=max_val)
    plt.title("WEIGHT VECTORS")
    plt.xlabel("INPUT - " + cell_type_input)
    plt.ylabel("OUTPUT - " + cell_type_output)
    cbar = plt.colorbar(orientation="horizontal")
    cbar.set_label("WEIGHT VALUE")
    plt.show()


def plot_true_vs_predicted(true, predicted, mse, r2, params):
    # plots results of any predictions vs. true values
    plt.tight_layout()
    plt.subplot(2, 1, 1)
    plt.title("TRUE, BIN SIZE: "+str(params.time_bin_size*1000)+"ms")
    plt.imshow(true, interpolation='nearest', aspect='auto', cmap="jet")
    a = plt.colorbar()
    a.set_label("#SPIKES")
    plt.ylabel("CELL IDS")
    plt.subplot(2, 1, 2)
    plt.title("PREDICTED, R2: "+str(round(r2, 3))+", MSE: "+str(round(mse, 3)))
    plt.imshow(predicted, interpolation='nearest', aspect='auto', cmap="jet")
    plt.xlabel("TIME BINS")
    plt.ylabel("CELL IDS")
    a = plt.colorbar()
    a.set_label("#SPIKES")
    plt.show()


def plot_cca_loadings(x_loadings, y_loadings, cell_type_array, params, sliding):

    for cv in range(x_loadings.shape[1]):
        plt.subplot(2, 1, 1)
        plt.scatter(range(x_loadings.shape[0]), x_loadings[:, cv], marker=".", c="r")
        plt.title("CV"+str(cv+1)+", "+str(params.time_bin_size) + "s TIME BINS" + (', SLIDING' if sliding else ''))
        plt.ylabel("WEIGHTS: " + cell_type_array[0])

        plt.subplot(2, 1, 2)
        plt.scatter(range(y_loadings.shape[0]), y_loadings[:, cv], marker=".", c="b")
        plt.ylabel("WEIGHTS: " + cell_type_array[1])
        plt.xlabel("CELL IDS")
        plt.show()


def plot_multiple_gaussians(centers_gauss_x, centers_gauss_y, std_gauss, alpha, cell_id=None, field_dim=None):

    centers_gauss_x = centers_gauss_x.T.flatten()
    centers_gauss_y = centers_gauss_y.T.flatten()

    variance_x = std_gauss
    variance_y = std_gauss

    if field_dim is None:
        dim_x = int(np.max(centers_gauss_x) - np.min(centers_gauss_x))
        dim_y = int(np.max(centers_gauss_y) - np.min(centers_gauss_y))
        # Create grid and multivariate normal
        x = np.linspace(np.min(centers_gauss_x)-5, np.max(centers_gauss_x)+5, dim_x)
        y = np.linspace(np.min(centers_gauss_y)-5, np.max(centers_gauss_y)+5, dim_y)
    else:
        dim_x = int(field_dim[1] - field_dim[0])
        dim_y = int(field_dim[3] - field_dim[2])
        x = np.linspace(field_dim[0], field_dim[1], dim_x)
        y = np.linspace(field_dim[2], field_dim[3], dim_y)

    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # plot all cells if cell_id is not provided
    if cell_id is None:
        for cell_id in range(alpha.shape[0]):
            all_gauss = np.zeros((dim_y, dim_x))
            fig, ax = plt.subplots()

            for mu_x, mu_y, weight in zip(centers_gauss_x, centers_gauss_y, alpha[:, cell_id]):
                rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
                rv_normalized = rv.pdf(pos) / np.sum(rv.pdf(pos).flatten())
                all_gauss += weight * rv_normalized

            gauss = ax.imshow(all_gauss, origin='lower', interpolation='nearest', aspect='auto')


            cb = plt.colorbar(gauss)
            cb.set_label("ALPHA VALUES * GAUSSIANS")
            plt.title("ALPHA VALUES FOR CELL " + str(cell_id))
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()
    else:
        all_gauss = np.zeros((dim_y, dim_x))
        fig, ax = plt.subplots()

        for mu_x, mu_y, weight in zip(centers_gauss_x, centers_gauss_y, alpha[:, cell_id]):
            rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
            rv_normalized = rv.pdf(pos) / np.sum(rv.pdf(pos).flatten())
            all_gauss += weight * rv_normalized

        gauss = ax.imshow(all_gauss, origin='lower', interpolation='nearest', aspect='auto')

        cb = plt.colorbar(gauss)
        cb.set_label("ALPHA VALUES * GAUSSIANS")
        plt.scatter(centers_gauss_x - field_dim[0], centers_gauss_y - field_dim[2], s=0.1, c="r", label="GAUSS. CENTERS")
        plt.title("ALPHA VALUES * GAUSSIANS FOR CELL " + str(cell_id))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()