#!/usr/bin/env python3

## 0_methods_B_tuning_3_find_best_fit_script.py

# This script is a summary of 0_methods_B_tuning_3_find_best_fit.ipynb notebook and outputs a csv table of a.o. resulting MAEs for a given ensemble
# run in terminal with $python 0_methods_B_tuning_3_find_best_fit_script.py

################################
# SETTINGS
################################

# set ensemble to 1 of: 1TU 2TU 3TU PAR 1P5 2P5 3P5 KDE
ensemble = '3P5'

## INTERMEZZO: define basins/cruises of interest ################## 
import numpy as np

# using obs_d since obs_p would only give a subset of that
cruises_incl_arctic = np.unique(obs_d_incl_arctic.cruise)  # for completion; not used
cruises_all = np.unique(obs_d.cruise)                      # already excludes Arctic cruises

# only 1 basin incl its SO sector
cruises_Atl = ['GA02', 'GAc02', 'GA03', 'GAc03', 'GA10','GIPY04', 'GIPY05', 
               'deng', 'ng'] # deng=geovide
cruises_Pac = ['GP16', 'GPc01', 'GSc02', 'pavia']

# SO (Southern Ocean)
cruises_SO = ['GSc02','GIPY04', 'GIPY05','GIpr05', 'pavia'] 
# chosing here the definition that GA10 is not SO

# for testing purposes; NOTE: these are only in the obs_d_incl_arctic, obs_p_incl_arctic objects
cruises_Arctic = ['GN01', 'GN02', 'GN03', 'GN04']  
cruises_Labr = ['GN02']
################################

# define all MAEs and RMSEs to be computed as fourplets and name them (without the _Pad, _Thd)
# fourplets (cruises, wo_surface, weighted_vol, weighted_unc) 
# MAE_tasks = {'MAE_tot_weights':(cruises_all,False,True,True), 
#              'MAE_vol_weights':(cruises_all,False,True,False), 
#              'MAE_unc_weights':(cruises_all,False,False,True), 
#              'MAE_no_weights':(cruises_all,False,False,False),
#              'MAE_incl_arctic_tot_weights':(cruises_incl_arctic,False,True,True), 
#              'MAE_incl_arctic_vol_weights':(cruises_incl_arctic,False,True,False),
#              'MAE_incl_arctic_unc_weights':(cruises_incl_arctic,False,False,True), 
#              'MAE_incl_arctic_no_weights':(cruises_incl_arctic,False,False,False)}
# RMSE_tasks = {'RMSE_tot_weights':(cruises_all,False,True,True), 
#               'RMSE_vol_weights':(cruises_all,False,True,False), 
#               'RMSE_unc_weights':(cruises_all,False,False,True), 
#               'RMSE_no_weights':(cruises_all,False,False,False), 
#               'RMSE_incl_arctic_tot_weights':(cruises_incl_arctic,False,True,True), 
#               'RMSE_incl_arctic_vol_weights':(cruises_incl_arctic,False,True,False),
#               'RMSE_incl_arctic_unc_weights':(cruises_incl_arctic,False,False,True), 
#               'RMSE_incl_arctic_no_weights':(cruises_incl_arctic,False,False,False)}

MAE_tasks = {'MAE_no_weights':(cruises_all,False,False,False),
             'MAE_unc_weights':(cruises_all,False,False,True)}
RMSE_tasks = {'RMSE_unc_weights':(cruises_all,False,False,True), 
              'RMSE_no_weights':(cruises_all,False,False,False)}


from pathlib import Path                       # path objects to avoid inter-platform errors
# DEFINE PATHS FOR DATA & FOR GENERATED FIGURES
# KEEP THE Path() FUNCTION AND USE FORWARD SLASHES / ON EVERY OPERATING SYSTEM
modeldir = Path('/modeloutput')
obsdir = Path('./data')
savedir = Path('./figures')

################################
# END OF SETTINGS
################################



################################
# SET UP SCRIPT
################################

## CHECK FILEPATHS
# expand paths because np.loadtxt can't handle home directory ~
savedir = savedir.expanduser()
obsdir = obsdir.expanduser()
modeldir = modeldir.expanduser()
def check_dir(path):
    if not path.exists():
        raise Exception('File path ' + str(path) + ' does not exit. Correct or create first.')
check_dir(savedir)
check_dir(obsdir)
check_dir(modeldir)

## IMPORT PACKAGES
# first time install missing packages via $conda install numpy OR $pip3 install numpy (be consistent)
import xarray as xr
import pandas as pd

# check python version
if 1/2 == 0:
    raise Exception("You are using python 2. Please use python 3 for a correct display of the figures.") 

# set a random run (the 1st run actually) as control simulation; doesn't matter as only model grid is used
fnctrl = modeldir / '1TU1000_PI'  

# in the model output, simulation year 0 is called 1765 CE (pre-industrial)
spinup_yr = 1765

## INFORMATION ON ENSEMBLES
# only used for ensemble '3TU'
all_runs_ws = range(0,14)
all_runs_kdes = range(100,118)
all_runs_sigmas_poc = range(200,221)
all_runs_sigmas_ca = range(300,319)
all_runs_sigmas_op = range(400,422)
all_runs_sigmas_du = range(500,521)
all_runs_sigmas_ne = range(600,619)
all_runs_all_sigmas = range(701,711)   # varying all sigmas from 'min of obs range' to 'max of obs range +50%'
all_runs_try_combined = range(800,810) # manual choices that combine optimal parameters that were result of the other 3TU runs

# only used for ensemble 'KDE':
kdes_runs_batch1 = list(range(0,20))+list(range(40,48))
kdes_runs_batch2 = list(range(20,40))+list(range(48,56))
kdes_runs_batch3 = list(range(56,84))

all_run_nrs = {'1TU' : range(0,2951),
               '2TU' : range(0,511),
               '3TU' : list(all_runs_ws)+list(all_runs_kdes)+list(all_runs_sigmas_poc)+list(all_runs_sigmas_ca)+list(all_runs_sigmas_op)
                       +list(all_runs_sigmas_du)+list(all_runs_sigmas_ne)+list(all_runs_all_sigmas)+list(all_runs_try_combined),
               'PAR' : ['_WS0000', '_WS0100', '_WS0500', '_WS1000', '_WS1500', '_WS2000', '_WS3000', '_WS4000', '_WS5000', 
                        '_PTCA00', '_PTCA01', '_PTCA02', '_PTCA03', '_PTCA04', '_PTCA05', '_PTCA06', '_PTCA07', '_PTCA08'],
               '1P5' : range(0,3006),
               '2P5' : range(0,3000),
               '3P5' : range(0,3000),
               'KDE' : kdes_runs_batch1+kdes_runs_batch2+kdes_runs_batch3}
# NOTE FOR 'PAR' ENSEMBLE: for this ensemble I messed up the naming. It went as: PAR_WS0001 and PAR_PTCA01
# quick fixes are implemented inside the functions
# it can be replaced now by 3TU, which also varies Ws and sigmajCa, but with other sigmas fixed at different vals that seem a bit better

## LOAD USER-DEFINED FUNCTIONS:
import functions as f                          # my own functions; call via f.function_name()


################################
# LOAD MODEL RUNS
################################

# read in the csv on which the tuning runs were based
# this already gives us columns run_ID and the 13 parameters that were adjusted (all in the BGC routine, of which Pa, Th is a part)
# using var 'ensemble' as set on top of this notebook
total_table = pd.read_csv(obsdir / ('parameters_'+ensemble+'_sampled_after_constraints.csv'))

total_table.rename(columns={'# setID':'run_ID', '﻿BGC_PaThWs':'PaThWs', 'BGC_PaDesConst':'PaDesConst', 'BGC_PaThWs':'PaThWs', 'BGC_ThDesConst':'ThDesConst', 
                            'BGC_sigmaPaPOC':'sigmaPaPOC', 'BGC_sigmaPaCa':'sigmaPaCa', 
                            'BGC_sigmaPaOp':'sigmaPaOp', 'BGC_sigmaPaDu':'sigmaPaDu', 'BGC_sigmaPaNeph':'sigmaPaNeph', 
                            'BGC_sigmaThPOC':'sigmaThPOC', 'BGC_sigmaThCa':'sigmaThCa', 
                            'BGC_sigmaThOp':'sigmaThOp', 'BGC_sigmaThDu':'sigmaThDu', 'BGC_sigmaThNeph':'sigmaThNeph'}, 
                   inplace=True) # BGC_PaThWs column name contained still DOF unicode char

# add a column with the runname for completeness (corresponding parameter files, .out files and output files carry this name)
#     N.B. technically speaking: with runname we mean parameterfilesname here. We dont mean executablename, 
#     since the executable is the same for the entire ensemble.
total_table.insert(1, 'runname', [f.runname(n, ID=True, ensemble=ensemble) for n in total_table.run_ID]) # add new column runnames after column 1


################################
# MEAN RESIDENCE TIME MRT 
################################

## Load MEAN RESIDENCE TIME for all runs
runs = [f.runname(n, ensemble=ensemble) for n in all_run_nrs[ensemble]]
modelled_mrt = f.load_var_multiple_runs(['Mrt_Pa_bac','Mrt_Th_bac'], 
                                        'timeseries_inst', modeldir, runs, spinup_yr=spinup_yr)
sec_to_yr = 1.0 / (365.25*24*3600)

## check if runs are in equilibrium & insert MRT to table
# what is the 'internal variability' of the above curve? (trial and error by eye gives 0.1 yr)
delta_significant = 0.1  # in yr; a difference is significant if > delta_significant
# hardcoded: 50 yr average of MRT at end is saved

# fill lists with information on the way
mrts_Pa = []
mrts_Th = []
for n in all_run_nrs[ensemble]:
    ds = modelled_mrt[f.runname(n, ensemble=ensemble)]
    # t1 is 200 yr before end;     t2 is at end (50 yr average)
    mrt_Pa_t1 = ds.Mrt_Pa_bac.isel(time=-200) * sec_to_yr                    
    mrt_Pa_t2 = ds.Mrt_Pa_bac.isel(time=slice(-50,-1)).mean() * sec_to_yr 
    mrt_Th_t1 = ds.Mrt_Th_bac.isel(time=-200) * sec_to_yr 
    mrt_Th_t2 = ds.Mrt_Th_bac.isel(time=slice(-50,-1)).mean() * sec_to_yr   

    if abs(mrt_Pa_t2 - mrt_Pa_t1) < delta_significant:
        # simulation ran into equilibrium
        mrts_Pa.append(mrt_Pa_t2.item())
        mrts_Th.append(mrt_Th_t2.item())
    else:
        # mrts_Pa.append(mrt_Pa_t2)
        # mrts_Th.append(mrt_Th_t2)
        # save nan instead as a sign that this run is not in equilibrium
        mrts_Pa.append(np.nan)
        mrts_Th.append(np.nan)        
        print("MRT is not constant i.e. no equilibrium for run nr", n, "which has runname", f.runname(n,ensemble=ensemble))
        
print('All',len(all_run_nrs[ensemble]),'runs were succesfull IF this is the first print statement. Done.')
# surprisingly, all runs of 2TU are in equilibrium!

# add resulting MRTs to table
total_table.insert(2, 'MRT_Pa', mrts_Pa) # add new column MRT_Pa after column 2
total_table.insert(3, 'MRT_Th', mrts_Th) 

## delete objects again to save memory
del modelled_mrt


################################
# MEDIAN & KADS_SUM => LATTER COMMENTED OUT
################################

# COMPUTE median & K_ads_Pa_summed & K_ads_Th_summed AND ADD TO TABLE
run_nrs = all_run_nrs[ensemble]    # defined at top of notebook

## ASSUMING PARTICLE EXPORT FIELDS ARE THE SAME FOR ALL RUNS

## load all model data
runnames = [f.runname(nr=i, ensemble=ensemble, ID=False) for i in run_nrs]
load_vars = ['Pad','Thd','Pap','Thp','rho_SI','boxvol']
# load_vars = ['Pad','Thd','Pap','Thp','FLDPOM','FLDCA','FLDOP','rho_SI','boxvol']
data_fulls = f.load_var_multiple_runs(variables=load_vars, file_type='full_ave', 
                                      folder=modeldir, runs=runnames)  # still has model units of dpm/m3; converted below

# # COMMENT OUT KADSTOTAL

# ## prepare dust field for K_ads dust term
# # outside loop over runs because always the same
# # read in dust field (based on Mahowald et al., 2006, JGR)
# bgc_input_fields = xr.open_dataset(obsdir / 'world_41x40.BGC.nc', decode_times=False)
# dust_field = bgc_input_fields.dust_dep_mod.mean(dim='time') * 1000 # avg over 12 months; result in g-dust/m**2/s
# # process same way as in model: conversion of dust-units; kg-dust/m^2/s to g-dust/m**2/s                                                                                                                                                   
# flux_du = dust_field * 365.25 * 24 * 3600

# ## prepare sigma values
# sigmas_all_runs = total_table.loc[:,('run_ID','sigmaPaPOC', 'sigmaPaCa', 'sigmaPaOp', 'sigmaPaDu', 
#                                      'sigmaThPOC', 'sigmaThCa', 'sigmaThOp', 'sigmaThDu')]

# prepare fluxes F_i
# ASSUMING PARTICLE EXPORT FIELDS ARE THE SAME FOR ALL RUNS; otherwise move inside for loop
n = run_nrs[0]
run = f.runname(nr=n, ensemble=ensemble, ID=False)
# # Find export fluxes F_i(theta,phi) at surface
# # output is in C/(m^2*s) resp. C/(m^2*s) resp. Si/(m^2*s) resp. g dust/(m^2*s) 
# # but convert /s to /yr:
# flux_poc = data_fulls[run]['FLDPOM'].isel(time=-1) * 365.25 * 24 * 3600 
# flux_ca = data_fulls[run]['FLDCA'].isel(time=-1) * 365.25 * 24 * 3600
# flux_op = data_fulls[run]['FLDOP'].isel(time=-1) * 365.25 * 24 * 3600

# prepare grid cell volume and mask it with wet cells only
# used for k_ads_sum
vol = data_fulls[run].boxvol  
test_var = data_fulls[run].Pad.isel(time=-1)
vol = xr.where(np.isnan(test_var), np.nan, vol)
# correct: since vol is in denum and nom, this mask is applied to both sums over theta,phi,z

## loop over runs and fill information on the way
median_Pad = []
median_Thd = []
median_Pap = []
median_Thp = []
# k_ads_pa_sum = []
# k_ads_th_sum = []
for i,n in enumerate(run_nrs):
    run = f.runname(nr=n, ensemble=ensemble, ID=False)
    
    # convert model units dpm/m3 to uBq/kg
    obj = data_fulls[run].isel(time=-1) # output of this run, still in dpm/m3
    obj = f.model_to_sw_unit(obj, data_fulls[run].rho_SI.isel(time=-1)) # to uBq/kg

    # compute and save median
    median_Pad.append(obj.Pad.median().item())
    median_Thd.append(obj.Thd.median().item())
    median_Pap.append(obj.Pap.median().item())
    median_Thp.append(obj.Thp.median().item())
    
#     ## Compute and save K_ads^j_summed, where sum is over all grid cells
#     # TODO ignoring direct effect of neph on Kads for now (indirect: other diffusion is w/i run)

#     # Find sigmas of this run nr n
#     if ensemble == 'PAR':
#         # quick fix with using i instead of n (n is a string in case of PAR ensemble); breaks if specific run nrs are asked
#         [junk_ID, sig_pa_poc, sig_pa_ca, sig_pa_op, sig_pa_du, 
#          sig_th_poc, sig_th_ca, sig_th_op, sig_th_du] = sigmas_all_runs.loc[i].values
#     else:
#         # [sig_pa_poc, sig_pa_ca, sig_pa_op, sig_pa_du, 
#         #  sig_th_poc, sig_th_ca, sig_th_op, sig_th_du] = sigmas_all_runs.loc[n].values
        
#         [junk_ID, sig_pa_poc, sig_pa_ca, sig_pa_op, sig_pa_du, 
#          sig_th_poc, sig_th_ca, sig_th_op, sig_th_du] = sigmas_all_runs[sigmas_all_runs.run_ID == n+1000].values[0]
    
#     # Compute k_ads_sum
#     this_kads_pa_sum = 0  # unit is /yr, same as k_ads_pa
#     this_kads_th_sum = 0
#     for z in data_fulls[run].z_t:
#         # vol.sel(z_t=z) & flux_poc both have coords (lat,lon), which are preserved; rest scalars
#         res = vol.sel(z_t=z) * (flux_poc * sig_pa_poc * f.remin_curve_val(z, 'POC') +
#                                 flux_ca * sig_pa_ca * f.remin_curve_val(z, 'CaCO3') +
#                                 flux_op * sig_pa_op * f.remin_curve_val(z, 'opal') +
#                                 flux_du * sig_pa_du * f.remin_curve_val(z, 'dust') ) # numerator (theta,phi,z)
#         this_kads_pa_sum += res.sum(dim=('lat_t','lon_t')).item()

#         # repeat for Th
#         res = vol.sel(z_t=z) * (flux_poc * sig_th_poc * f.remin_curve_val(z, 'POC') +
#                                 flux_ca * sig_th_ca * f.remin_curve_val(z, 'CaCO3') +
#                                 flux_op * sig_th_op * f.remin_curve_val(z, 'opal') +
#                                 flux_du * sig_th_du * f.remin_curve_val(z, 'dust') ) 
#         this_kads_th_sum += res.sum(dim=('lat_t','lon_t')).item()
#     # denominator: divide by total weights
#     k_ads_pa_sum.append(this_kads_pa_sum / vol.sum().item())
#     k_ads_th_sum.append(this_kads_th_sum / vol.sum().item())
    
# add results to table
total_table.insert(4, 'median_Pad', median_Pad)
total_table.insert(5, 'median_Thd', median_Thd)
total_table.insert(6, 'median_Pap', median_Pap)
total_table.insert(7, 'median_Thp', median_Thp)
# total_table.insert(8, 'k_ads_pa_sum', k_ads_pa_sum)
# total_table.insert(9, 'k_ads_th_sum', k_ads_th_sum)

# delete (large) datasets again to save memory
del data_fulls


################################
# LOAD OBSERVATIONS
################################

# dissolved
# load geotraces observations
fnobs = obsdir / 'Pad_Thd_IDP2021.txt'
obs_d_geotraces = f.get_obs_geotraces(fnobs, dissolved_type='BOTTLE', 
                                      drop_meta_data=True, good_quality=True)

# load other dissolved observations from 3 additional studies
obs_d_deng = f.get_obs_other(obsdir / 'Deng2018Pad_Thd_formatted_uBq_per_kg.csv') # is geovide
obs_d_ng = f.get_obs_other(obsdir / 'Ng2020Pad_Thd_formatted_dpm_per_1000kg.csv')
for var in ['Pad','Pad_err','Thd','Thd_err']:
    # convert dpm/1000kg to uBq/kg; using 1 dpm = 1/60 * 1e6 uBq
    obs_d_ng[var] = obs_d_ng[var] / 60.0 * 1e3  
obs_d_pavia = f.get_obs_other(obsdir / 'Pavia2020Pad_Thd_formatted_uBq_per_kg.csv')

# combine all dissolved data
obs_d = pd.concat([obs_d_geotraces,obs_d_deng,obs_d_ng,obs_d_pavia], join='outer')
obs_d_incl_arctic = pd.concat([obs_d_geotraces_incl_arctic,obs_d_deng,obs_d_ng
                               ,obs_d_pavia], join='outer')  # for completeness; not used
[obs_d, obs_d_ave, obs_d_ave_num] = f.obs_to_model_grid(obs_d, fnctrl)

# particle-bound
fnobs = obsdir / 'Pap_Thp_IDP2021.txt'
obs_p_incl_arctic = f.get_obs_geotraces(fnobs, drop_meta_data=True, p_type='combined', good_quality=True)
obs_p = obs_p_incl_arctic[~obs_p_incl_arctic.cruise.isin(
    ['GN01','GN02','GN03','GN04'])].copy()
[obs_p, obs_p_ave, obs_p_ave_num] = f.obs_to_model_grid(obs_p, fnctrl)


################################
# COMPUTE MAEs & RMSEs
################################

######### COMPUTE ALL DESIRED MAEs: TAKES A LONG WHILE ###########
model_run_nrs = all_run_nrs[ensemble]    # all; takes ca. 1 hour for 1TU and 10 mins for 2TU;  MULTIPLIED WITH number of cruises_restricted below

if ensemble == 'PAR':
    modelrunIDs = model_run_nrs
else:
    modelrunIDs = np.asarray(model_run_nrs) + 1000  # must be ID   

print('--- NUMBER OF TASKS: GOING TO COMPUTE', len(MAE_tasks), 'different versions of MAEs (actually x 4 for Pad,Thd,Pap,Thp).')

MAE_results = {}
counter = 0
for key in MAE_tasks:
    (cruises, wo_surface, weighted_vol, weighted_unc) = MAE_tasks[key]
    
    [junk, this_obs_d_ave, junk2] = f.subset_of_obs(obs_d, fnctrl, cruises=cruises, wo_surface=wo_surface)
    [junk, this_obs_p_ave, junk2] = f.subset_of_obs(obs_p, fnctrl, cruises=cruises, wo_surface=wo_surface)
    [MAEs_task_Pad, MAEs_task_Thd, MAEs_task_Pap, MAEs_task_Thp] = f.calc_all_MAEs(this_obs_d_ave, this_obs_p_ave, 
                                                                                   modelrunIDs=modelrunIDs, modeldir=modeldir,
                                                                                   ensemble=ensemble, weighted_vol=weighted_vol, 
                                                                                   weighted_unc=weighted_unc, verbose=False)
    MAE_results[key+'_Pad'] = MAEs_task_Pad
    MAE_results[key+'_Thd'] = MAEs_task_Thd
    MAE_results[key+'_Pap'] = MAEs_task_Pap
    MAE_results[key+'_Thp'] = MAEs_task_Thp
    
    counter += 1
    print('--- FINISHED TASK NUMBER', counter, ' ---')

print('...FINISHED!')
# if ERROR: cruises objects are defined a few cells above



######### COMPUTE ALL DESIRED RMSEs: TAKES A LONG WHILE ###########
                            # for 150 runs x 16 tasks x 2 (Pad,Thd) = 4800 it took 30 minutes so ca. 0.4 second per RMSE per run
                            # but really depends on overhead of loading observations again for each task

model_run_nrs = all_run_nrs[ensemble]    # all; takes ca. 1 hour for 1TU and 10 mins for 2TU;  MULTIPLIED WITH number of cruises_restricted below

if ensemble == 'PAR':
    modelrunIDs = model_run_nrs
else:
    modelrunIDs = np.asarray(model_run_nrs) + 1000  # must be ID   

print('--- NUMBER OF TASKS: GOING TO COMPUTE', len(RMSE_tasks), 'different versions of RMSEs (actually x 4 for Pad,Thd,Pap,Thp).')

RMSE_results = {}
counter = 0
for key in RMSE_tasks:
    (cruises, wo_surface, weighted_vol, weighted_unc) = RMSE_tasks[key]
    
    [junk, this_obs_d_ave, junk2] = f.subset_of_obs(obs_d, fnctrl, cruises=cruises, wo_surface=wo_surface)
    [junk, this_obs_p_ave, junk2] = f.subset_of_obs(obs_p, fnctrl, cruises=cruises, wo_surface=wo_surface)
    
    [RMSEs_task_Pad, RMSEs_task_Thd, RMSEs_task_Pap, RMSEs_task_Thp] = f.calc_all_RMSEs(this_obs_d_ave, this_obs_p_ave, 
                                                                                        modelrunIDs=modelrunIDs, modeldir=modeldir, 
                                                                                        ensemble=ensemble, weighted_vol=weighted_vol, 
                                                                                        weighted_unc=weighted_unc, verbose=False)
    RMSE_results[key+'_Pad'] = RMSEs_task_Pad
    RMSE_results[key+'_Thd'] = RMSEs_task_Thd
    RMSE_results[key+'_Pap'] = RMSEs_task_Pap
    RMSE_results[key+'_Thp'] = RMSEs_task_Thp
    
    counter += 1
    print('--- FINISHED TASK NUMBER', counter, ' ---')
    
print('...FINISHED COMPUTATION!')


################################
# EXPORT RESULTS
################################

print('STARTING EXPORT... ')

# add last results to total_table
col = 8  # start inserting from column 8, because the last column inserted above is at position 7

for key in MAE_results:
    total_table.insert(col, key, MAE_results[key])
    col += 1

# continue with RMSE at col where MAE finished
for key in RMSE_results:
    total_table.insert(col, key, RMSE_results[key])
    col += 1
    
# export table to csv
fn = savedir / ('total_table_ensemble_' + ensemble + '.csv')
total_table.to_csv(fn, index=False)
    
print('...FINISHED EXPORT TO ', fn, '! THANK YOU FOR WAITING :)')

