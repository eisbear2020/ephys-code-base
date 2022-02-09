# Electro-physiological data analysis

Scripts to study several phenomena using electro-physiological data:

* interactions between different regions (e.g. the Hippocampus and the medial entorhinal cortex, or 
left and right hemisphere of the Hippocampus)
* changes in correlation structure
* memory drift (representational drift)

# To dos:

* ising glm --> check that density of Gaussians is the same for pre and post

* consider storing .eeg and .eegh data separately (not in the created dictionaries under
/temp_data --> makes files very large and loading slower!)

* use .whl file to determine length of experiment (not last spike!)

* decide when to do .whl interpolation for lost data --> in load_data or pre_processing?

# File structure

## main.py

All analysis steps are started from here!

* uses parameters from parameter_files/
* all other parameters are either set as parser arguments or inside the 
"params" class
* automatically chooses correct analysis to be performed (e.g. one population, sleep data or 
two populations and awake data)

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

## function_files/ subdirectory 
contains all .py files with relevant functions & classes.

### load_data.py
This is the first script that needs to be called to get the data.

* class LoadData: selects and pre-processes raw data (.des, .whl, .clu) according to session_name,
experiment_phase, cell_type
  * depending on type of data (SLEEP, EXPLORATION, TASK) different parts of the raw data are 
  returned/processes
  * returns object with data for subsequent analysis

### analysis_methods.py

Contains classes/functions to analyze ONE phase (e.g. sleep or behavior) using one
or two populations.

* class BaseMethods: methods to analyze sleep and awake data for ONE POPULATION
  * class Sleep: derived from BaseMethods --> used to analyze sleep data
  * class Exploration: derived from BaseMethods --> used to analyze exploration data
* class BaseMethodsTwoPop: methods to analyze sleep and awake data for TWO POPULATIONS
  * class TwoPopSleep: derived from BaseMethods --> used to analyze sleep data
  * class TwoPopExploration: derived from BaseMethods --> used to analyze exploration data

### multiple_phases.py

Contains classes/functions to analyze TWO or more phase (e.g. sleep and behavior) using one
or two populations.

### multiple_sessions.py

Contains classes/functions to analyze multiple sessions (e.g. different animals/days)

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

## temp_data/ subdirectory
dictionaries with data for different experiment phases (generated 
and loaded by select_data.py)
#### ML/ subdirectory
results of ML analysis

#### TreeHMM/ subdirectory
firing times lists for treeHMM analysis

## results/ subdirectory
contains .md files with results for different data sets
* plots are referenced from plots/ subdirectory

## plots/ subdirectory
contains subfolders with result plots from different data sets

## external_files/ subdirectory 
* contains additional files that are used (e.g. external libraries)

# Data description
 
## Peter's exploration data: familiar vs. novel (Cell, 2018)
  
* cells from left hemisphere (first 16 electrodes):
  * mjc161_2: 72 (e.g. first 72 cells in .des file are from the left
  hemisphere, the rest is from the right hemisphere --> corresponds
  to index 2 to 74 in the global .clu file)
  * mjc163_2: 257
  * mjc169_2: 132
  * mjc186_3: 177 
  