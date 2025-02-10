TUNING README
This readme describes the tuning of the Pa/Th module of the Bern3D model via the usage of .py, .ipynb, .csv, and .parameter files,
located in the current folder and subfolder 0_methods_B_tuning_files.

This readme was published in the zenodo repository https://doi.org/10.5281/zenodo.10622403 along with the paper:
Jeemijn Scheen, Jörg Lippold, Frerk Pöppelmeier, Finn Süfke and Thomas F. Stocker. Promising regions for detecting 
the overturning circulation in Atlantic 231Pa/230Th: a model-data comparison. Paleoceanography and Paleoclimatology, 2025.

We gratefully acknowledge Marco Steinacher and Sebastian Lienert for their contributions to the latin hypercube sampling scripts. 


THREE ENSEMBLES WERE RUN, NAMED:
ensemble '2TU' (run ids 1000-1510) for ws
ensemble 'KDE' (run ids 1000-1083) for kdes
ensemble '3P5' (run ids 1000-3999) for sigmas


STEPS OF TUNING PROCEDURE (for each ensemble):

1. SAMPLE PARAMETER VALUES FROM DESIRED PARAMETER DISTRIBUTIONS:
- Using: 
	0_methods_B_tuning_1_sample.py
	0_methods_B_tuning_files/parameters_XXX.csv
- For each ensemble (e.g. '3P5'), parameters_3P5.csv was filled with the desired settings about the parameters, 
	their ranges and assumed statistical distributions to sample from. We chose a uniform distribution in all cases.
- 0_methods_B_tuning_1_sample.py generated the desired number of parameter sets, based on parameters_3P5.csv. Optionally, 
	the script can also generate some plots. The script writes out parameter sets (e.g. 3000) to parameters_3P5_sampled.csv.
- NOTE: for ensemble KDE, no sampling was performed. Instead, parameters_KDE_sampled_after_constraints.csv was created manually directly.

2. CONSTRAIN PARAMETER SETS:
- Using: 
	0_methods_B_tuning_2_constrain_param_sets.ipynb
	0_methods_B_tuning_files/parameters_XXX_sampled.csv
- The notebook 0_methods_B_tuning_2_constrain_param_sets.ipynb was used to edit the parameter sets. 
	It reads in parameters_3P5_sampled.csv and writes out parameters_3P5_sampled_after_constraints.csv.
	It reads in parameters_2TU_sampled.csv and writes out parameters_2TU_sampled_after_constraints.csv.
	What this notebook does: 
		I.	change setIDS from 0, 1, ..., 3000 to 1000, 1001, ..., 4000 for practical reasons 
		(runnames must have same length and trailing zeros like 0001 did not work as they are deleted in slurm array jobs)
		II. [only for ensemble 2TU:] constrain the parameter sets to only keep combinations that make sense, based on the literature. 
		Since applying these constraints in ensemble 2TU did not solve the issues with tuning all parameters simultaneously, 
		this approach was discarded in later ensembles such as 3P5. Instead we used a 3-step tuning approach (1. ws, 2. kdes, 3. sigmas). 
	NOTE: for ensemble 3P5, the resulting csv is also called _sampled_after_constraints.csv for consistency, even though no constraints 
	were applied (only the setIDs were changed).
- Recall: for ensemble KDE, the file parameters_KDE_sampled_after_constraints.csv was created manually directly.

3. GENERATE PARAMETER FILES:
- Using: 
	0_methods_B_tuning_3_generate_param_files.py
	0_methods_B_tuning_files/paramfile_templates/
	0_methods_B_tuning_files/parameters_XXX_sampled_after_constraints.csv
- 0_methods_B_tuning_3_generate_param_files.py creates parameter files suitable for running the Bern3D model,
based on desired parameter sets as given in parameters_XXX_sampled_after_constraints.csv
For each row of the csv file (1 parameter set), the 6 parameter files needed for 1 model run are generated.
- 0_methods_B_tuning_files/paramfile_templates/ is used as input and 0_methods_B_tuning_files/paramfiles/ as output.

RUN SIMULATIONS:
- Using:
	0_methods_B_tuning_files/paramfiles
	Bern3D model
- The filename convention for model runs is EEENNNNPPP, with:
	EEE the ensemble (2TU, KDE or 3P5); NNNN the Set ID starting from 1000 in steps of 1; PPP the phase (always _PI)
- The output from the 3595 tuning runs is not published along, but resulting MAEs, median concentrations, and mean residence times are.
- After running all tuning simulations with the Bern3D model, the results are analyzed:

4. FIND MEAN ABSOLUTE ERRORS (MAEs):
- Using: 
	0_methods_B_tuning_4_find_MAEs.ipynb
	0_methods_B_tuning_files/parameters_2TU_sampled_after_constraints.csv
	0_methods_B_tuning_files/parameters_KDE_sampled_after_constraints.csv
	0_methods_B_tuning_files/parameters_3P5_sampled_after_constraints.csv
	model output of the tuning runs (not provided)
- This notebook does the postprocessing of the tuning runs and computes Mean Absolute Errors (MAEs) of model versus observations.  
- We share this notebook for transparency, but it cannot be executed as model output from the 3595 tuning runs is not published online.
- The resulting MAEs were saved in 0_methods_B_tuning_files/total_table_ensemble_XXX.csv
- From this table: MAEs weighted by obs. uncertainty were used; also provided are: unweighted MAEs, medians and mean residence times (MRTs).

5. CONCLUSION OF TUNING - FIND BEST FIT PARAMETERS:
- Using:
	0_methods_B_tuning_5_conclusion.ipynb
	0_methods_B_tuning_files/total_table_ensemble_2TU.csv
	0_methods_B_tuning_files/total_table_ensemble_KDE.csv
	0_methods_B_tuning_files/total_table_ensemble_3P5.csv
- This notebook analyzes the results from the previous notebook to conclude which parameters give the best fit with observations 
- The procedure and choices are as described in Scheen et al. 2025, main text and supplement.
- This notebook also makes Fig. 4, S2, S3 and S4.
