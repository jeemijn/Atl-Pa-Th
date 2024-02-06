# Atl-Pa-Th
Study on Atlantic protactinium (Pa) and thorium (Th).  

- This repository contains code for analysis + figures of the manuscript:  
"Promising regions for detecting the overturning circulation in Atlantic Pa/Th: a model-data comparison"  
- Intended use is via the .ipynb notebooks. (These call functions from the .py scripts when needed.)
  - 1_results.xxx.ipynb are notebooks that create the figures
  - in addition to a folder 1_results_C_powerpoint_figures for postprocessing
  - 0_methods_xxx.ipynb are notebooks about the methods: A) nepheloid-layer model development, B) tuning, C) preparing runs with scaled particles.
- At the moment, the notebooks 1_results_A.ipynb, 1_results_B.ipynb, 0_methods_A.ipynb, 0_methods_C.ipynb are entirely cleaned up and clarified. The 0_methods_B_xxx.ipynb notebooks can still use more cleanup, which follows later.

### Steps
1. Download the model output from the model output repository on zenodo (https://doi.org/10.5281/zenodo.10621275) and place in subfolder 'modeloutput'.
2. Download the Geotraces data files Pad_Thd_IDP2021.txt & Pad_Thd_IDP2021.txt (2x ~25 Mb) and place in subfolder 'data'. (A better manual follows later)
3. Use the .ipynb notebooks to reproduce or edit. Documentation is inside the notebooks. It is not necessary to read the .py scripts.

### Where to find which figure:  
| Figure |is made in  |
|--------|-------------|
| 1 |	1_results_C_powerpoint_figures  |
| 2 |	1_results_B_other_figures.ipynb  |
| 3a |	figures/fig3a_cores_map.png  |
| 3b |	1_results_B_other_figures.ipynb  |
| 3ab |	1_results_C_powerpoint_figures  |
| 4 |	0_methods_B_tuning_4_analyse.ipynb  |
| 5 |	1_results_A_transect_figures.ipynb  |
| 6 |	1_results_B_other_figures.ipynb  |
| 7 |	1_results_B_other_figures.ipynb  |
| 7 |	1_results_C_powerpoint_figures  |
| 8 |	1_results_B_other_figures.ipynb  |
| 9 |	1_results_B_other_figures.ipynb  |
| 10 |	1_results_B_other_figures.ipynb  |
| 11 |	1_results_A_transect_figures.ipynb  |
| 11 |	1_results_C_powerpoint_figures  |
| 12 |	1_results_B_other_figures.ipynb  |
| C1 |	1_results_A_transect_figures.ipynb  |
| C2 |	1_results_B_other_figures.ipynb  |
| C2 |	1_results_C_powerpoint_figures  |
| C3 |	1_results_B_other_figures.ipynb  |
| C4 |	1_results_B_other_figures.ipynb  |
| C5 |	1_results_B_other_figures.ipynb  |
| S1 |	0_A_process_Gardner_data.ipynb  |
| S2 |	0_methods_B_tuning_4_analyse.ipynb  |
| S3 |	0_methods_B_tuning_4_analyse.ipynb  |
| S4 |	0_methods_B_tuning_4_analyse.ipynb  |
| S5 |	1_results_B_other_figures.ipynb  |
| S6 |	1_results_B_other_figures.ipynb  |
| S7 |	1_results_B_other_figures.ipynb  |
