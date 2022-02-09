########################################################################################################################
#
#   Analysis for memory drift paper with Federico
#
########################################################################################################################

from function_files.sessions import SingleSession, MultipleSessions
import numpy as np


def execute(params):

    # params
    # ------------------------------------------------------------------------------------------------------------------

    cell_type = "p1"

    all_sessions = ["mjc163R2R_0114", "mjc163R4R_0114", "mjc169R4R_0114", "mjc163R1L_0114", "mjc148R4R_0113",
                    "mjc163R3L_0114", "mjc169R1R_0114"]
    session_name = "mjc163R2R_0114"

    """#################################################################################################################
    #   Figure 1
    #################################################################################################################"""

    # stability of single cells
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc163R2R_0114", params=params,
    #                            cell_type=cell_type).all_data()
    # single_ses.plot_waveforms_single_cells(cell_id=3, electrode=[0,1,2,3], y_min=-900, y_max=580, save_fig=True)
    # single_ses.plot_waveforms_single_cells(cell_id=1, electrode=[0,1,2,3], y_min=-900, y_max=580, save_fig=True)
    # single_ses.plot_waveforms_single_cells(cell_id=18, electrode=[0,1,2,3], y_min=-900, y_max=580, save_fig=True)

    # recording stability stable, dec, inc
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.assess_stability()

    # trajectories learning, PRE, POST
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc163R3L_0114"
    # psp = SingleSession(session_name=session_name,
    #                    cell_type=cell_type, params=params).cheese_board(experiment_phase=["learning_cheeseboard_1"])
    # psp.plot_tracking_and_goals_first_or_last_trial(trial="first", save_fig="true")
    # psp.plot_tracking_and_goals_first_or_last_trial(trial="last", save_fig="true")
    # psp = SingleSession(session_name=session_name,
    #                    cell_type=cell_type, params=params).cheese_board(experiment_phase=["learning_cheeseboard_2"])
    # psp.plot_tracking_and_goals_first_or_last_trial(trial="first", save_fig="true")
    # occupancy in POST
    # ------------------------------------------------------------------------------------------------------------------
    # single_ses = SingleSession(session_name="mjc163R4R_0114", params=params,
    #                            cell_type=cell_type).cheese_board(experiment_phase=["learning_cheeseboard_2"])
    # single_ses.occupancy_around_goals(save_fig=True)

    # ls = MultipleSessions(session_names=all_sessions,
    #                          cell_type=cell_type, params=params).post_cheeseboard_occupancy_around_goals()

    """#################################################################################################################
    #   Figure 2
    #################################################################################################################"""

    # session_name = "mjc163R1L_0114"
    # single_ses = SingleSession(session_name=session_name, params=params,
    #                            cell_type=cell_type).long_sleep()
    # single_ses.memory_drift_entire_sleep_spike_shuffle_vs_data(save_fig=False, nr_shuffles=30,
    #                                                            swapping_factor=50)

    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_temporal(save_fig=False)
    # compute p-values for data vs. shuffle at t=0 and t=1
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses.long_sleep_memory_drift_data_vs_shuffle()

    """#################################################################################################################
    #   Figure 3
    #################################################################################################################"""

    # plot ratio and zoom
    # session_name = "mjc163R1L_0114"
    # cb_long_sleep = SingleSession(session_name=session_name,
    #                               cell_type=cell_type, params=params).long_sleep()
    # cb_long_sleep.memory_drift_plot_rem_nrem(template_type="phmm", n_moving_average_pop_vec=400, plotting=True)

    # example with delta score
    # session_name = "mjc163R4R_0114"
    # cb_long_sleep = SingleSession(session_name=session_name,
    #                               cell_type=cell_type, params=params).long_sleep()
    # cb_long_sleep.memory_drift_plot_rem_nrem_delta_score(template_type="phmm", n_moving_average_pop_vec=20,
    #                                                      plotting=True)

    # cumulative score for all sessions
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_memory_drift_opposing_nrem_rem(save_fig=False)

    """#################################################################################################################
    #   Figure 4
    #################################################################################################################"""

    # REM vs. NREM max likelihoods
    # ------------------------------------------------------------------------------------------------------------------
    # ls = MultipleSessions(session_names=all_sessions,
    #                          cell_type=cell_type, params=params)
    # ls.long_sleep_nrem_rem_likelihoods(template_type="phmm", save_fig=True)

    # REM vs. NREM per mode posterior prob.
    # ------------------------------------------------------------------------------------------------------------------
    # ls = MultipleSessions(session_names=all_sessions,
    #                          cell_type=cell_type, params=params)
    # ls.long_sleep_nrem_rem_decoding_cleanliness_per_mode(template_type="phmm", save_fig=False, control_data="nrem")

    # REM vs. NREM autocorrelation as a function of population vectors
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc163R1L_0114"
    # psp = SingleSession(session_name=session_name,
    #                    cell_type=cell_type, params=params).long_sleep()
    # psp.memory_drift_rem_nrem_autocorrelation_spikes_likelihood_vectors(template_type="phmm", save_fig=False)

    #
    # pp_cb = MultipleSessions(session_names=all_sessions,
    #                          cell_type=cell_type, params=params)
    # pp_cb.long_sleep_nrem_rem_decoding_similarity()

    # Example of decoded maps
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc169R4R_0114"
    # psp = SingleSession(session_name=session_name,
    #                                cell_type=cell_type, params=params).pre_long_sleep_post()
    # psp.memory_drift_rem_nrem_decoding_similarity_plot_decoded_map(save_fig=False, pre_or_post="pre")

    # Mode decoding probability REM-NREM: Ising
    # ------------------------------------------------------------------------------------------------------------------
    # pp_cb = MultipleSessions(session_names=all_sessions, cell_type=cell_type, params=params)
    # pp_cb.long_sleep_nrem_rem_decoding_similarity(template_type="ising")

    # Mode decoding probability REM-NREM: pHMM
    # ------------------------------------------------------------------------------------------------------------------
    # pp_cb = MultipleSessions(session_names=all_sessions, cell_type=cell_type, params=params)
    # pp_cb.long_sleep_nrem_rem_decoding_similarity()

    """#################################################################################################################
    #   Fig 5: Classification of cells, Firing rate changes NREM/REM
    #################################################################################################################"""

    # session_name = "mjc169R1R_0114"
    # ls = SingleSession(session_name=session_name, cell_type=cell_type, params=params).long_sleep()
    # ls.memory_drift_plot_temporal_trend_stable_cells(n_moving_average_pop_vec=5000, save_fig=True)

    # pre_s_post = SingleSession(session_name=session_name,
    #                             cell_type=cell_type, params=params).pre_long_sleep_post()
    # pre_s_post.plot_cell_classification_mean_firing_rates_awake(save_fig=True)

    # example session for firing rate changes
    # ------------------------------------------------------------------------------------------------------------------
    # session_name = "mjc163R3L_0114"
    # pre_s_post = SingleSession(session_name=session_name,
    #                             cell_type=cell_type, params=params).long_sleep()
    # pre_s_post.firing_rate_changes(save_fig=False, plotting=True)

    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.long_sleep_temporal_trend_stable_cells(save_fig=True)

    """#################################################################################################################
    #   Fig 6: Stable cells & learning
    #################################################################################################################"""

    # firing rate differences between subsets
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses = MultipleSessions(session_names=all_sessions, params=params, cell_type=cell_type)
    # mul_ses.pre_long_sleep_post_firing_rates_all_cells(save_fig=False, chunks_in_min=2, measure="mean")
    # mul_ses.pre_post_cheeseboard_stable_cells_pre_post_decoding(save_fig=False)
    # decoding POST using PRE decoder
    # mul_ses.pre_post_cheeseboard_stable_cells_pre_post_decoding(save_fig=True)
    # sparsity
    # pp_cb.pre_post_cheeseboard_spatial_information(spatial_resolution=2, save_fig=True)

    # mul_ses.pre_long_sleep_post_firing_rates()

    # rate map stability
    # ------------------------------------------------------------------------------------------------------------------
    # mul_ses.rate_map_stability_pre_probe_pre_post_post_probe_cell_comparison(save_fig=True)
    # mul_ses.rate_map_stability_pre_probe_pre_post_post_probe(cells_to_use="stable", save_fig=True)
    # mul_ses.rate_map_stability_pre_probe_pre_post_post_probe(cells_to_use="increasing", save_fig=True)
    # mul_ses.rate_map_stability_pre_probe_pre_post_post_probe(cells_to_use="decreasing", save_fig=True)