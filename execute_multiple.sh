#!/bin/sh
# to run this file: ./execute_multiple.sh

########################################################################################################################
# Execute multiple
########################################################################################################################


nohup python main.py --session_name "mjc163R2R_0114" --compute_step "fitting_ising" --time_bin_size 0.01 > mjc163R2R_0114_md.out &
nohup python main.py --session_name "mjc163R4R_0114" --compute_step "fitting_ising" --time_bin_size 0.01 > mjc163R4R_0114_md.out &
nohup python main.py --session_name "mjc163R1L_0114" --compute_step "fitting_ising" --time_bin_size 0.01 > mjc163R1L_0114_md.out &
nohup python main.py --session_name "mjc163R3L_0114" --compute_step "fitting_ising" --time_bin_size 0.01 > mjc163R3L_0114_md.out &
nohup python main.py --session_name "mjc169R4R_0114" --compute_step "fitting_ising" --time_bin_size 0.01 > mjc169R4R_0114_md.out &
nohup python main.py --session_name "mjc169R1R_0114" --compute_step "fitting_ising" --time_bin_size 0.01 > mjc169R1R_0114_md.out &
nohup python main.py --session_name "mjc148R4R_0113" --compute_step "fitting_ising" --time_bin_size 0.01 > mjc148R4R_0113_md.out &





########################################################################################################################
# cross validation: optimal number of modes
########################################################################################################################
#
#nohup python main.py --session_name "mjc163R2R_0114" --experiment_phase "learning_cheeseboard_1" --compute_step "cb_phmm" > mjc163R2R_0114_lc1.out &
#nohup python main.py --session_name "mjc163R2R_0114" --experiment_phase "learning_cheeseboard_2" --compute_step "cb_phmm" > mjc163R2R_0114_lc2.out &
#nohup python main.py --session_name "mjc163R4R_0114" --experiment_phase "learning_cheeseboard_1" --compute_step "cb_phmm" > mjc163R4R_0114_lc1.out &
#nohup python main.py --session_name "mjc163R4R_0114" --experiment_phase "learning_cheeseboard_2" --compute_step "cb_phmm" > mjc163R4R_0114_lc2.out &
#nohup python main.py --session_name "mjc169R4R_0114" --experiment_phase "learning_cheeseboard_1" --compute_step "cb_phmm" > mjc169R4R_0114_lc1.out &
#nohup python main.py --session_name "mjc169R4R_0114" --experiment_phase "learning_cheeseboard_2" --compute_step "cb_phmm" > mjc169R4R_0114_lc2.out &
#nohup python main.py --session_name "mjc163R1L_0114" --experiment_phase "learning_cheeseboard_1" --compute_step "cb_phmm" > mjc163R1L_0114_lc1.out &
#nohup python main.py --session_name "mjc163R1L_0114" --experiment_phase "learning_cheeseboard_2" --compute_step "cb_phmm" > mjc163R1L_0114_lc2.out &
#nohup python main.py --session_name "mjc169R1R_0114" --experiment_phase "learning_cheeseboard_1" --compute_step "cb_phmm" > mjc169R1R_0114_lc1.out &
#nohup python main.py --session_name "mjc169R1R_0114" --experiment_phase "learning_cheeseboard_2" --compute_step "cb_phmm" > mjc169R1R_0114_lc2.out &
#nohup python main.py --session_name "mjc148R4R_0113" --experiment_phase "learning_cheeseboard_1" --compute_step "cb_phmm" > mjc148R4R_0113_lc1.out &
#nohup python main.py --session_name "mjc148R4R_0113" --experiment_phase "learning_cheeseboard_2" --compute_step "cb_phmm" > mjc148R4R_0113_lc2.out &
#nohup python main.py --session_name "mjc163R1L_0114" --experiment_phase "learning_cheeseboard_1" --compute_step "cb_phmm" > mjc163R1L_0114_lc1.out &
#nohup python main.py --session_name "mjc163R1L_0114" --experiment_phase "learning_cheeseboard_2" --compute_step "cb_phmm" > mjc163R1L_0114_lc2.out &