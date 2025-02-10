# Atl-Pa-Th
Study on Atlantic protactinium (Pa) and thorium (Th).  

- This repository contains code for analysis + figures of the manuscript:  
Scheen, Jeemijn and Lippold, Jörg and Pöppelmeier, Frerk and Süfke, Finn and Stocker, Thomas F., "Promising regions for detecting the overturning circulation in Atlantic Pa/Th: a model-data comparison", Paleoceanography & Paleoclimatology, 2025
- Intended use is via the .ipynb notebooks. (These call functions from the .py scripts when needed.)
  - 1_results.xxx.ipynb are notebooks that create the figures
  - 1_results_C_powerpoint_figures is a folder with figures made/postprocessed in powerpoint
  - 0_methods_xxx.ipynb are notebooks about the methods: A) nepheloid-layer model development, B) tuning, C) preparing runs with scaled particle export fluxes. Step B) also contains .py scripts, a readme and additional files in subfolders.

### Steps
1. Download the model output from the model output repository on zenodo (https://doi.org/10.5281/zenodo.14791063) and place in subfolder 'modeloutput'.
2. Use the .ipynb notebooks to reproduce or edit figures. Documentation is inside the notebooks. It is not necessary to read the .py scripts.

### Redistributed data
- The subfolder 'data' already contains additional small data files needed for the code to run:
  - compilation of Pap/Thp sediment measurements (as in Table 2 of the manuscript; see references therein)
  - global compilation of benthic nepheloid layers (thickness & excess particulate matter mass), as presented in Gardner, W. D., Richardson, M. J., Mishonov, A. V., & Biscaye, P. E. 2018 Progress in Oceanography, https://doi.org/10.1016/j.pocean.2018.09.008 (also published later at https://odv.awi.de/data/ocean/global-transmissometer-database/)
  - Pad and Thd seawater measurements published in the papers (as cited in the manuscript):
    - GEOTRACES Intermediate Data Product Group, 2021, The GEOTRACES Intermediate Data Product 2021 (IDP2021), NERC EDS British Oceanographic Data Centre NOC, https://doi.org/10.5285/cf2d9ba9-d51d-3b7c-e053-8486abc0f5fd
    - Deng et al. 2018, Biogeosciences, https://doi.org/10.5194/bg-15-7299-2018  
      with data published at BODC: https://doi.org/10.5285/7a150d33-956b-0fec-e053-6c86abc0b35c
    - Ng et al. 2020, Marine Chemistry, https://doi.org/10.1016/j.marchem.2020.103894  
      with data published at BCO-DMO: https://www.bco-dmo.org/dataset/813379 & https://doi.org/10.26008/1912/bco-dmo.813379.2
    - Pavia et al. 2020, Global Biogeochemical Cycles, https://doi.org/10.1029/2020GB006760  
      with data published at PANGAEA: https://doi.org/10.1594/PANGAEA.926855
      
### Where to find which figure:  
| Figure |is made in  |
|--------|-------------|
| 1 | 1_results_C_powerpoint_figures  |
| 2 | 1_results_B_other_figures.ipynb  |
| 3b |  1_results_B_other_figures.ipynb  |
| 3cd | 1_results_A_transect_figures.ipynb  |
| 3abcd | 1_results_C_powerpoint_figures  |
| 4 | 0_methods_B_tuning_5_conclusion.ipynb  |
| 5 | 1_results_A_transect_figures.ipynb  |
| 6 | 1_results_B_other_figures.ipynb  |
| 7 | 1_results_B_other_figures.ipynb  |
| 8 | 1_results_B_other_figures.ipynb  |
| 8 | 1_results_C_powerpoint_figures  |
| 9 | 1_results_B_other_figures.ipynb  |
| 10 |  1_results_B_other_figures.ipynb  |
| 11 |  1_results_A_transect_figures.ipynb  |
| 11 |  1_results_C_powerpoint_figures  |
| C1 |  1_results_A_transect_figures.ipynb  |
| C2 |  1_results_A_transect_figures.ipynb  |
| C3 |  1_results_A_transect_figures.ipynb  |
| C4 |  1_results_B_other_figures.ipynb  |
| C4 |  1_results_C_powerpoint_figures  |
| C5 |  1_results_B_other_figures.ipynb  |
| C6 |  1_results_B_other_figures.ipynb  |
| S1 |  0_A_process_Gardner_data.ipynb  |
| S2 |  0_methods_B_tuning_5_conclusion.ipynb  |
| S3 |  0_methods_B_tuning_5_conclusion.ipynb  |
| S4 |  0_methods_B_tuning_5_conclusion.ipynb  |
| S5 |  1_results_B_other_figures.ipynb  |
| S6 |  1_results_B_other_figures.ipynb  |
| S7 |  1_results_B_other_figures.ipynb  |
| S8 |  1_results_B_other_figures.ipynb  |
