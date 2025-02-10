#!/usr/bin/python3

# Run this script via   $python 0_methods_B_tuning_1_sample.py file_name_without_.csv
# for example           $python 0_methods_B_tuning_1_sample.py parameters_2TU

# This script reads in desired parameter ranges and settings from a file xx.csv 
# and writes 'nsets' number of generated parameter sets to xx_sampled.csv

# This script uses lhs_helper.py
# Latin hypercube sampling (lhs) is the method used to spread out parameters over the parameter space

# Author:      Marco Steinacher
# Adapted by:  Jeemijn Scheen

# This script was published in the zenodo repository https://doi.org/10.5281/zenodo.10622403 along with the paper:
# Jeemijn Scheen, Jörg Lippold, Frerk Pöppelmeier, Finn Süfke and Thomas F. Stocker. Promising regions for detecting 
# the overturning circulation in Atlantic 231Pa/230Th: a model-data comparison. Paleoceanography and Paleoclimatology, 2025.

# load packages
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import csv
import sys
import lhs_helper
import math
import netCDF4 as nc
import argparse
from pathlib import Path         # Path objects to avoid inter-platform trouble in file paths

# save argument that is given when running script
parser = argparse.ArgumentParser()
parser.add_argument('parameter_filename', help="Name of csv parameter file (without extension)")  # must be located in in_dir
args = parser.parse_args()

################
## FILE PATHS ##
################
# KEEP THE Path() FUNCTION AND USE FORWARD SLASHES '/' ON EVERY OPERATING SYSTEM
in_dir = Path('./0_methods_B_tuning_files/')    # folder containing input files parameters_XXX.csv
out_dir = Path('./figures/')                    # folder for saving output files parameters_XXX_sampled.csv


#############
## OPTIONS ##
#############

## Sample parameters and write output to <args.parameter_filename>_sampled.csv
do_sampling = 1
nsets       = 3000      # Number of samples
new_seed    = False     # Generate new random seed?
seed = 69064187         # Seed to use if new_seed = False

## Plot individual samples
do_plot_samples = 0

## Run statistical checks on sampled distributions
do_check = 0

## Generate plots
do_plot  = 0
do_plot_uniform = 0     # Include uniform distributions in plots

## Plot posterior distribution
do_plot_posterior   = 0
posterior_scorefile = 'post_process/output/out_cases.nc'

#################
## END OPTIONS ##
#################

if do_sampling:
    # Open output file
    outfile = open(out_dir / (args.parameter_filename+'_sampled.csv'),'w')

if do_plot_posterior:
    scorefile    = nc.NetCDFFile(posterior_scorefile,'r')
    constr_score = scorefile.variables['constr_score'].getValue() # (set,grp1,grp2,grp3,grp4)
    param_values = scorefile.variables['param_values'].getValue() # (set, param)

# Read parameter list
params = csv.reader(open(in_dir / (args.parameter_filename+'.csv'),'r'))
dist = []
dist_uni = []
param_name = []
param_units = []
param_min = []
param_max = []
param_std = []
for row in params:
    # Skip commented lines
    if row[0][0] == "#" or row[0][1] == "#":
        continue
    pname = row[0]+'_'+row[1]
    punits = row[2]
    pstd  = float(row[3])
    pmin  = float(row[4])
    pmax  = float(row[5])
    pdist = row[6]
    pclip1 = float(row[7])   # truncnorm: left clip   lognorm: shape parameter for stddev = (max-min)/4
    pclip2 = float(row[8])   # truncnorm: right clip  lognorm: shape parameter for [min,max] = 95% c.i.
    param_name.append(pname)
    param_units.append(punits)
    param_std.append(pstd)
    param_min.append(pmin)
    param_max.append(pmax)
    info = '# Adding parameter '+pname+'['+punits+']: '

    dist_uni.append(st.uniform(loc=pmin,scale=pmax-pmin))

    stddev = (pmax-pmin)/4.0

    if pdist == 'uni':
        # Uniform distribution [min,max]
        info = info + 'uniform('+str(pmin)+';'+str(pmax)+')'
        dist.append(st.uniform(loc=pmin,scale=pmax-pmin))
    elif pdist == 'norm':
        # Normal distribution around standard value with stddev=(max-min)/4
        info = info + 'norm('+str(pstd)+';'+str(stddev)+')'
        dist.append(st.norm(pstd,scale=stddev))
    elif pdist == 'truncnorm':
        # Truncated normal distribution: as norm but truncated to interval [param1,param2]
        info = info + 'truncnorm('+str(pstd)+';'+str(stddev)+';['+str(pclip1)+':'+str(pclip2)+'])'
        dist.append(st.truncnorm((pclip1-pstd)/stddev,(pclip2-pstd)/stddev,loc=pstd,scale=stddev))
    elif pdist == 'lognorm':
        # Log-normal distribution with shape=param1 & location=param2 
        # (see Steinacher et al. 2013, Nature, Supplement page 4)
        info = info + 'lognorm('+str(pstd)+';'+str(pclip1)+';loc='+str(pclip2)+')'
        dist.append(st.lognorm(pclip1,scale=pstd-pclip2,loc=pclip2))
    else:
        print(info+"ERROR: Unknown distribution: "+pdist)
        sys.exit()

    print(info)
    if do_sampling:
        outfile.write(info+"\n")

if do_sampling:
    # Do Latin hypercube sampling (lhs)
    if new_seed == True:
        seed = np.random.randint(sys.maxsize)
        outfile.write("# Generated new random seed: "+str(seed)+" (maxsize = "+str(sys.maxsize)+")"+'\n')
    else:
        outfile.write("# Using predefined random seed: "+str(seed)+'\n')
    np.random.default_rng(seed)  # Set random seed
    ret = lhs_helper.lhs(nsets,dist)

    # Output in CSV format
    outfile.write("# setID")
    for j,parval in enumerate(ret[0]):
        outfile.write(",")
        outfile.write(param_name[j])
    outfile.write("\n")
    for i,parset in enumerate(ret):
        outfile.write(str(i))
        for j,parval in enumerate(parset):
            outfile.write(",")
            if param_name[j]=='EBM_climsens':
                f = np.poly1d(np.array([-1.76740301e-03, 5.20924594e-02,
                    -5.57474413e-01, 2.73354234e+00, -5.17440559e+00]))
                outfile.write(str(f(parval)))
            else:
                outfile.write(str(parval))
        outfile.write("\n")
    # Flush output
    outfile.flush()

if do_plot_samples:
    #  Read parameters/results and plot
    samples_x = []
    samples_y = []
    samples_color = []
    samples_size = []
    count = []
    result = csv.reader(open(args.parameter_filename+'_sampled.csv','r'))
    i = 0
    print('Reading '+args.parameter_filename+'_sampled.csv')
    for parset in result:
        print(parset)
        # Skip commented lines
        if parset[0][0] == "#":
            continue
        for jj,parval in enumerate(parset):
            j = jj-1
            print('j = '+str(j), i)
            print(parval)
            if i == 0:
                samples_x.append([])
                samples_y.append([])
                count.append(0)
            if j >=0 and j < len(param_name):
                ## Only calculate y-value for parameter values, not results
                samples_x[j].append(float(parval))
                samples_y[j].append((0.1+count[j]*1.0/nsets)*0.003)
                count[j] = count[j] + 1
            elif j == len(param_name):
                ## Run status
                print("status = "+parval)
                if float(parval) == 1:
                    # OK
                    samples_color.append(0.0)
                    samples_size.append(3)
                elif float(parval) == 5 or float(parval) == 10:
                    # CPOOL2 || C2LSOM
                    samples_color.append(1.0)
                    samples_size.append(7)
                elif float(parval) == 9:
                    # PHOTO
                    samples_color.append(2.0)
                    samples_size.append(10)
                else:
                    # REST
                    samples_color.append(3.0)
                    samples_size.append(13)
        i = i + 1

if do_plot:
    ## Plotting
    pp = PdfPages(args.parameter_filename+'_plots.pdf')
    fig = plt.figure(figsize=(15,10))
    page = 0
    results_offset = len(param_name)
    for i,name in enumerate(param_name):
        if i == 12+page*12:
            fig.savefig(pp,format='pdf')
            fig = plt.figure(figsize=(15,10))
            page = page + 1
        delta = param_max[i] - param_min[i]
        if do_plot_samples:
            x = np.linspace(min(param_min[i]-0.5*delta,min(samples_x[i])),max(param_max[i]+0.5*delta,max(samples_x[i])),1000)
        else:
            x = np.linspace(param_min[i]-0.5*delta,param_max[i]+0.5*delta,1000)
        fig.subplots_adjust(hspace=0.3,wspace=0.3)
        p = fig.add_subplot(4,3,i-page*12+1)


        if do_plot_samples:
            p.plot(samples_x[i],samples_y[i],marker='+',markersize=2,linestyle='None',color='blue')
        p.plot(x,dist[i].pdf(x)/sum(dist[i].pdf(x)),color='blue')
        if do_plot_uniform:
            p.plot(x,dist_uni[i].pdf(x)/sum(dist_uni[i].pdf(x)),color='green')

        if do_plot_posterior:
            nbins = 40
            edge_delta = (max(x)-min(x))/(nbins-2)
            post_min = min(x-edge_delta)
            post_max = max(x+edge_delta)
            binsize = (post_max - post_min)/nbins
            bins_x = []
            bins_y = []
            bins_count = np.zeros(nbins)
            for j in range(0,nbins):
                bins_x.append((j+0.5)*binsize + post_min)
                bins_y.append(0.0)
            for model in range(0,nsets):
                if constr_score[model,0,0,0,0] > 0.0 and constr_score[model,0,0,0,0] < 1000.0:
                    score = math.exp(-0.5*constr_score[model,0,0,0,0])
                else:
                    continue
                bin_i = int(np.floor((param_values[model,i+1]-post_min)/binsize))
                if bin_i <= -1:
                    bin_i = 0
                if bin_i >= nbins:
                    bin_i = nbins-1
                bins_y[bin_i] = bins_y[bin_i] + score
                bins_count[bin_i] = bins_count[bin_i] + 1

            ## Interpolate binned values and plot
            posterior = np.interp(x,bins_x,bins_y)
            p.plot(x,posterior/sum(posterior),color='red')

            prior     = np.interp(x,bins_x,bins_count)
            p.plot(x,prior/sum(prior),color='green')

        p.set_title(name+" ["+param_units[i]+"]",size=8)
        plt.setp(p.get_xticklabels(), fontsize=8)
        plt.setp(p.get_yticklabels(), fontsize=8)

    fig.savefig(pp,format='pdf')
    pp.close()


## Check statistical properties
if do_check:
    info = '#'
    info = info + '\n' + '# Check distributions:'
    info = info + '\n' + '# --------------------'
    for i,name in enumerate(param_name):
        median = dist[i].ppf(0.5)
        t = dist[i].stats()
        mean   = t[0]
        stddev = t[1]**0.5
        min_cdf = dist[i].cdf(param_min[i])
        max_cdf = dist[i].cdf(param_max[i])

        info = info + '\n' + '# '+ name+': [Min,Max] = ['+str(param_min[i])+','+str(param_max[i])+'] -> ['+str(min_cdf*100.0)+'%,'+str(max_cdf*100.0)+'%] ('+str((max_cdf-min_cdf)*100.0)+"% c.i.)"
        info = info + '\n' + '#   mean    = '+str(mean)
        info = info + '\n' + '#   median  = '+str(median) + ' stddev = '+str(stddev)
        info = info + '\n' + '#   default = '+str(param_std[i]) + ' minmax4 = '+str((param_max[i]-param_min[i])/4.0)

    print(info)
    if do_sampling:
        outfile.write(info+'\n')

if do_sampling:
    outfile.close()

#EOF
