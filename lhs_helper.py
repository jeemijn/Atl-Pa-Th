#!/usr/bin/python3

# This script is used for the tuning procedure in 0_methods_B_tuning_1_sample.py
# Latin hypercube sampling (lhs) is the method used to spread out parameters over the parameter space

## Author:      Marco Steinacher

# This script was published in the zenodo repository https://doi.org/10.5281/zenodo.10622403 along with the paper:
# Jeemijn Scheen, Jörg Lippold, Frerk Pöppelmeier, Finn Süfke and Thomas F. Stocker. Promising regions for detecting 
# the overturning circulation in Atlantic 231Pa/230Th: a model-data comparison. Paleoceanography and Paleoclimatology, 2025.


import numpy as np

## Latin hypercube sampling
def lhs(nsample,dist):

    nvar = len(dist)

    # Get matrix of uniformly distributed random numbers [0,1)
    ran = np.random.random_sample([nsample,nvar])
 
    s = np.zeros((nsample,nvar))
    dists = dist
    for j,d in enumerate(dists):
        # Get random permutation of samples for variable j
        idx = np.random.permutation(nsample)

        # Cum. prob. of random value inside each interval
        P = (idx + ran[:,j])/nsample

        # Call Inverse cdf for each P
        s[:,j] = d.ppf(P)

    return s
