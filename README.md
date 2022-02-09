# Electro-physiological data analysis

Scripts to study several phenomena using electro-physiological data

# File structure

## main.py

All analysis steps are started from here!

* analysis can either be started here directly (using debug string) or by running
a script from the analysis_scripts directory
* uses parameters from parameter_files/
* all other parameters are either set as parser arguments or inside the
"params" class
* calls classes from sessions.py for analysis

## function_files/ subdirectory
contains all .py files with relevant functions & classes.

### sessions.py
* class SingleSession: loads data and returns object of single phase (see single_phase.py)
* class MultipleSessions: analysis using multiple sessions

### single_phase.py
Contains classes for different single experiment phases. Single phases can be:

For one population:

* BaseMethods (parent class)
  * sleep
  * cheeseboard
* exploration

For two populations:
* BaseMethodTwoPop (parent class)
  * TwoPopSleep
  * TwoPopExploration

### multiple_phases.py
Classes that combine several phases from single_phase.py to e.g. analyze sleep and exploration


### load_data.py
This is the first script that needs to be called to get the data.

* class LoadData: selects and pre-processes raw data (.des, .whl, .clu) according to session_name, experiment_phase, cell_type
  * depending on type of data (SLEEP, EXPLORATION, TASK) different parts of the raw data are returned/processes
  * returns object with data for subsequent analysis

### pre_processing.py

Contains classes/functions for pre-processing data (e.g. binning) . Is used by analysis_methods.py

* class PreProcess: base class
  * class PreProcess: derived from PreProcess. Computes e.g. temporal spike binning or
  constant nr. of spikes binning
  * class PreProcess: derived from PreProcess. Computes rate maps, occupancy, temporal
  binning

### support_functions.py
Contains simple functions that are needed to compute things. Used by almost all functions/classes.

### ml_methods.py
Machine learning approaches for one or two populations:
* class MlMethodsOnePopulation
* class MlMethodsTwoPopulation

### plotting_functions.py
Contains most of the functions that are used to plot results.

## parameter_files subdirectory
* contains one parameter file for each data set
  * 3 dictionaries:
     * data description dictionary: from .des file --> which index
  corresponds to which phase of the experiment
     * data selection dictionary: where to find the data, which cell types
     * cell_type_sub_div_dic: defines where cell types are split (
     e.g. left and right hemisphere cells)
* class standard_analysis_parameters: stores all standard parameters/
place holders that are set at a later stage --> almost all later computations
use parameters stored in this class

## temp_data/ subdirectory
dictionaries with data for different experiment phases (generated
and loaded by select_data.py)

## results/ subdirectory
Contains .md files with results for different data sets
* plots are referenced from plots/ subdirectory

## plots/ subdirectory
Contains subfolders with result plots from different data sets

## external_files/ subdirectory
Contains additional files that are used (e.g. external libraries)

## analysis_scrips/ subdirectory
Contains selection of analysis (e.g. all analysis for a poster/paper)
