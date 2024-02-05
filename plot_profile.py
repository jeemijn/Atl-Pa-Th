#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""author: Gunnar Jansen, gunnar.jansen@unibe.ch

A class that produces profiles through the Atlantic, Pacific, Indian and
Southern Ocean. This class works only for the comparison of two models.
"""

import matplotlib.pyplot as plt

from plot_helpers import area_mean_z

class Profile(object):
    def __init__(self, dataset0=None, dataset1=None, variable=None, time=-1, scale=1, label='', title=''):
        self.dataset0 = dataset0
        self.dataset1 = dataset1
        self.variable = variable
        self.time = time
        self.scale = scale
        self.label = label
        self.title = title

    def plot(self):
        fig, ax = plt.subplots(1, 4, figsize=(14, 6))#figsize=(8,5))

        # define variable of interest here already, otherwise where() gives MemoryError
        obj0 = self.dataset0[self.variable].sel(time=self.dataset0.time[self.time])

        obj1 = self.dataset1[self.variable].sel(time=self.dataset1.time[self.time])

        # compute
        obj_atl0 = obj0.where(self.dataset0.mask == 1) # Atl excluding S.O.
        obj_pac0 = obj0.where(self.dataset0.mask == 2) # Pac excluding S.O.
        obj_ind0 = obj0.where(self.dataset0.mask == 3) # Ind excluding S.O.
        obj_so0 = obj0.where(self.dataset0.mask == 4)  # S.O.

        obj_atl1 = obj1.where(self.dataset1.mask == 1) # Atl excluding S.O.
        obj_pac1 = obj1.where(self.dataset1.mask == 2) # Pac excluding S.O.
        obj_ind1 = obj1.where(self.dataset1.mask == 3) # Ind excluding S.O.
        obj_so1 = obj1.where(self.dataset1.mask == 4)  # S.O.

        area_atl = self.dataset0.area.where(self.dataset0.mask.isel(z_t=0) == 1)
        area_pac = self.dataset0.area.where(self.dataset0.mask.isel(z_t=0) == 2)
        area_ind = self.dataset0.area.where(self.dataset0.mask.isel(z_t=0) == 3)
        area_so = self.dataset0.area.where(self.dataset0.mask.isel(z_t=0) == 4)

        atl0 = self.scale*area_mean_z(obj=obj_atl0, area=area_atl)
        pac0 = self.scale*area_mean_z(obj=obj_pac0, area=area_pac)
        ind0 = self.scale*area_mean_z(obj=obj_ind0, area=area_ind)
        so0 = self.scale*area_mean_z(obj=obj_so0, area=area_so)

        atl1 = self.scale*area_mean_z(obj=obj_atl1, area=area_atl)
        pac1 = self.scale*area_mean_z(obj=obj_pac1, area=area_pac)
        ind1 = self.scale*area_mean_z(obj=obj_ind1, area=area_ind)
        so1 = self.scale*area_mean_z(obj=obj_so1, area=area_so)

        # plots
        z = self.dataset0.z_t
        ax[0].set_ylabel("depth [m]")
        ax[0].plot(atl0, z, label='baseline')
        ax[0].plot(atl1, z, label='candidate')
        ax[0].set_title("Atlantic")
        ax[1].plot(pac0, z, label='pac')
        ax[1].plot(pac1, z, label='pac')
        ax[1].set_title("Pacific")
        ax[2].plot(ind0, z, label='ind')
        ax[2].plot(ind1, z, label='ind')
        ax[2].set_title("Indian")
        ax[3].plot(so0, z, label='so')
        ax[3].plot(so1, z, label='so')
        ax[3].set_title("Southern Ocean")
        ax[0].legend()                      # for axes 1-3: adjust labels or no legend

        for axis in ax:
            axis.invert_yaxis()
            axis.set_ylim([5000, 0])

        plt.tight_layout()
