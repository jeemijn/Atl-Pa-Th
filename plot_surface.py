#!/usr/bin/env python3
# -*- coding: utf-8 -*-   
"""
A class to generate surface plots for the Bern3D model

Author: Gunnar Jansen, gunnar.jansen@unibe.ch
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmp

from plot_helpers import extend

class Surface(object):
    """Author: Gunnar Jansen, gunnar.jansen@unibe.ch"""
    def __init__(self, dataset=None, variable=None, time=-1, scale=1, landmask=None,
                 cmap=cmp.coolwarm, clabel='', cmin=None, cmax=None, title=''):
        self.dataset = dataset
        self.variable = variable
        self.time = time
        self.scale = scale
        self.landmask = landmask
        self.cmap = cmap
        self.clabel = clabel
        self.cmin = cmin
        self.cmax = cmax
        self.title = title

    def __get_coords(self, data):
        xlist = self.dataset.coords['lon_u'].values
        ylist = self.dataset.coords['lat_u'].values
        X, Y = np.meshgrid(xlist, ylist)

        return X, Y

    def plot(self):
        """
        Creates the surface plot of the requested variable.
        Returns a figure and axis handle.
        Author: Gunnar Jansen, gunnar.jansen@unibe.ch
        """
        fig, ax = plt.subplots(1, figsize=(7, 4))

        if 'z_t' in self.dataset[self.variable].dims:
            data = self.dataset[self.variable].isel(z_t=0, time=self.time)
        else: # no z-coordinate for variables like sea ice
            data = self.dataset[self.variable].isel(time=self.time)

        X, Y = self.__get_coords(data)
        Z = self.scale*data.values

        c = ax.pcolor(X, Y, Z, cmap=self.cmap, vmin=self.cmin, vmax=self.cmax)

        [xu, yu, land_mask_surf] = self.landmask

        ax.pcolor(X, Y, extend(land_mask_surf + 1.0),
                  cmap='Greys', vmin=-0.5, vmax=0.5, alpha=1.0) # add black land

        fig.colorbar(c, ax=ax, label=r'$\bf{'+ self.clabel + '}$')
        ax.set_title(self.title)
        ax.set_xticks([120, 180, 240, 300, 360, 420])
        ax.set_xticklabels(['120°E', '180°W', '120°W', '60°W', '0°', '60°E'])
        ax.set_yticks([80, 40, 0, -40, -80])
        ax.set_yticklabels(['80°N', '40°N', 'EQ', '40°S', '80°S'])

        plt.tight_layout()

        return fig, ax
