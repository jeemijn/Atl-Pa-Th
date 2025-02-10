#!/usr/bin/python3

# This script creates parameter files suitable for running the Bern3D model,
# based on desired parameter sets as given in parameters_XXX_sampled_after_constraints.csv
# For each row of the csv file (1 parameter set), the 6 parameter files needed for 1 model run are generated.

# Run this script via $./0_methods_B_tuning_3_generate_param_files.py or $python 0_methods_B_tuning_3_generate_param_files.py

# Authors: Jeemijn Scheen, Marco Steinacher, Sebastian Lienert

# This script was published in the zenodo repository https://doi.org/10.5281/zenodo.10622403 along with the paper:
# Jeemijn Scheen, Jörg Lippold, Frerk Pöppelmeier, Finn Süfke and Thomas F. Stocker. Promising regions for detecting 
# the overturning circulation in Atlantic 231Pa/230Th: a model-data comparison. Paleoceanography and Paleoclimatology, 2025.

# load packages
from pathlib import Path      # path objects to avoid inter-platform errors
import numpy as np
import subprocess             # to execute terminal commmands
import pandas as pd

#############
## OPTIONS ##
#############

# ensemble
ensemble = 'KDE'  # choose '2TU' '3P5' or 'KDE'

# number of runs
nr_runs = 84

# folders
# KEEP THE Path() FUNCTION AND USE FORWARD SLASHES '/' ON EVERY OPERATING SYSTEM
in_dir = Path('./0_methods_B_tuning_files/')                            # location of csv_file
template_dir = Path('./0_methods_B_tuning_files/paramfile_templates/')  # location of template param files
out_dir = Path('./0_methods_B_tuning_files/paramfiles')                 # location to write resulting param files to

# file naming details
csv_file = 'parameters_' + ensemble + '_sampled_after_constraints.csv'  # input file with parameter sets as rows; header on first row
# file name of resulting param files = prefix + str(ID) + suffix e.g. KDE1001_PI
param_prefix = ensemble
param_suffix = '_PI'

# template to use determining all other, fixed, parameters
if ensemble in ['2TU', 'KDE']:
    template_param = 'EEENNNN_PI'
elif ensemble == '3P5':
    template_param = 'GGGNNNN_PI'

#################
## END OPTIONS ##
#################

## read in csv with pandas
all_param_sets = pd.read_csv(in_dir / csv_file, index_col=None, 
                            encoding='utf-8-sig').replace(u'\ufeff', '')  # encoding avoids \ufeff string in python3...
if all_param_sets.columns[1] == '\ufeffBGC_PaThWs':
    all_param_sets.rename(columns={"\ufeffBGC_PaThWs" : "BGC_PaThWs"}, inplace=True)     # ... but somehow didn't work

# assertion to make sure user is doing as intended
assert len(all_param_sets)==nr_runs, "ERROR: nr of runs, " + str(nr_runs) + ", not equal to csv file rows, " + str(len(all_param_sets))

param_sets_IDs = np.array(all_param_sets['# setID'])
assert len(param_sets_IDs) == len(all_param_sets), "ERROR: something is wrong with param_sets_IDs"
assert len(param_prefix) + len(str(param_sets_IDs[0])) + len(param_suffix) == 10, "ERROR: resulting name is not 10 chars long"

## determine which variables to adjust
vars_to_adjust = [var for var in all_param_sets.columns if var[0:4]=='BGC_']

# ASSUMPTION: in our case only bgc.parameter needs adjustment (all vars in heading start with BGC_). Checked here.
assert len(all_param_sets.columns)==len(vars_to_adjust)+1, "ERROR: assumption of only adjusting BGC vars seems violated"

## read in bgc.parameter template
file_handler = open(template_dir / (template_param + ".bgc.parameter"), mode="r")
bgc_parameter_template = file_handler.read()
file_handler.close()
del file_handler

## generate all parameter files except bgc.parameter
paramfile_kinds = ['.ebm.parameter', '.forcing.parameter', '.fw.parameter', '.main.parameter', '.ocn.parameter', ]
# N.B. '.lpj.filenames', '.lpj.parameter','.pisces.parameter', '.parameter.sed.parameter' are omitted (not relevant)

for this_ID in param_sets_IDs:
    ## 1.) BGC PARAMETER FILE
    this_param_set = all_param_sets.where(all_param_sets['# setID'] == this_ID).dropna()

    # adjust bgc.parameter file
    this_bgc_param = bgc_parameter_template  # is a string containing entire text file
    for param in vars_to_adjust:
        # replace @@@ placeholders with desired parameter based on csv line of this run
        this_bgc_param = this_bgc_param.replace('@@@'+param+'@@@',str(this_param_set[param].item()))

    # save resulting bgc parameter file
    textfile = open(out_dir / (param_prefix+str(this_ID)+param_suffix+'.bgc.parameter'), "w")
    nr_chars = textfile.write(this_bgc_param)
    assert nr_chars == len(this_bgc_param), "ERROR: file not entirely written for this_ID="+str(this_ID)
    textfile.close()
    
    ## 2.) OTHER PARAMETER FILES: 
    # just copy from templates (no adjustments needed); see ASSUMPTION above
    for kind in paramfile_kinds:
        res = subprocess.run(['cp', template_dir/(template_param+kind), 
                              out_dir/(param_prefix+str(this_ID)+param_suffix+kind)])#, capture_output=False)
        # print(res.stdout) # for debugging with capture_output=True; this syntax only works for python >= 3.7

#EOF
