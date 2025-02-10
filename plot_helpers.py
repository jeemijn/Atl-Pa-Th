#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
A collection of small functions that help to make Bern3D evaluation plots

Authors: Gunnar Jansen, Jeemijn Scheen, Raphael Roth
"""

import io
import base64
import numpy as np
from IPython.display import HTML

# get_landmask() was moved away from here to functions.py

# FUNCTIONS
def load_data(file_path, spinup_yr=1765.0, kyr=False):
    """"author: J. Scheen, jeemijn.scheen@nioz.nl
    
    Input:
    - file_path to output files (without the _timeseries_ave.nc ending)
    - spinup_yr needs to be given if spinup is not done for the year 1765
    - if kyr, then convert time from years to kiloyears

    Output:
    - [data, data_full] containing 2 xarray DataArray objects

    Explanation of output:
    1) data = data from timeseries_ave.nc output file
    2) data_full = data from full_ave.nc output file

    For all: the year axis is changed from simulation years to years they represent in A.D."""
    from xarray import open_dataset

    data = open_dataset(file_path + '_timeseries_ave.nc', decode_times=False)
    # here 'time' var is in years (no date) so can't be decoded
    data_full = open_dataset(file_path + '_full_ave.nc', decode_times=False)

    ## change years to simulation years
    subtract_yrs = spinup_yr
    data['time'] -= subtract_yrs
    data_full['time'] -= subtract_yrs

    if kyr:
        data['time'] = data['time'] / 1000.0
        data_full['time'] = data_full['time'] / 1000.0

    return [data, data_full]


def extend(var):
    """
    author: R.Roth

    adds one element of rank-2 matrices to be plotted by pcolor
    """
    [a,b]=var.shape
    field = np.ma.masked_invalid(np.ones((a+1, b+1))*np.nan)
    field[0:a, 0:b]=var
    return field


def area_mean_z(obj, area):
    '''author: J. Scheen, jeemijn.scheen@nioz.nl
    
    Takes area-weighted horizontal average of a certain data_var but keeps the z direction.
    - obj must be the data_var wanted e.g. data_full.TEMP. Assuming it has these vars in this order:
    z = 0, lat_t = 1, lon_t = 2
    - area must contain the grid-cell area of the region in obj e.g. data_full.area.
    Must have the same shape and mask/selection as obj. '''

    weights_ext = np.tile(area, (len(obj.z_t), 1, 1)) # copy weights repeatingly along z direction

    # handle NaN values (where land or masked out)
    weights_ext[np.isnan(obj.values)] = 0         # not needed, only for extra safety
    obj.values[np.isnan(obj.values)] = 0          # needed, otherwise still nan * 0 = nan

    return np.average(obj, axis=(1, 2), weights=weights_ext)
                                             # take weighted average over 1 = lat, 2 = lon

def cmap_discretize(cmap, N, minmax=None, white_center=False):
    """
    author: R.Roth

    Returns a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        minmax: [min, max] where min>0 and max<1 (cmap coverage), or None
        N: number of colors.

    usefull for e.g. pcolor-plots, not necessary for contourf-plots

    Example
        newmap=cmap_discretize(plt.cm.jet,16)
    """
    import matplotlib.colors as cols

    if minmax is not None:
        colors_i = np.concatenate((np.linspace(minmax[0], minmax[1], N), (0., 0., 0., 0.)))
    else:
        colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for k, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1, k], colors_rgba[i, k]) for i in range(N+1)]

    if white_center:
        # "white out" the bands closest to the middle
        num_middle_bands = 2 - (N % 2)
        middle_band_start_idx = (N - num_middle_bands) // 2
        for middle_band_idx in range(middle_band_start_idx,
                                     middle_band_start_idx + num_middle_bands):
            for key in cdict.keys():
                old = cdict[key][middle_band_idx]
                cdict[key][middle_band_idx] = old[:2] + (1.,)
                old = cdict[key][middle_band_idx + 1]
                cdict[key][middle_band_idx + 1] = old[:1] + (1.,) + old[2:]

    # Return colormap object.
    if isinstance(cmap.name, str):
        name = cmap.name
    else:
        name = "dummy"

    return cols.LinearSegmentedColormap(name+"_%d"%N, cdict, 1024)



class FlowLayout(object):
    '''Author: Gunnar Jansen, gunnar.jansen@unibe.ch
    
    A class / object to display plots in a horizontal / flow layout below a cell '''
    def __init__(self):
        # string buffer for the HTML: initially some CSS; images to be appended
        self.sHtml = """
        <style>
        .floating-box {
        display: inline-block;
        margin: 10px;
        border: 0px solid #888888;  
        }
        </style>
        """

    def add_plot_fig(self, oAxes):
        ''' Saves a PNG representation of a Matplotlib Axes object '''
        Bio = io.BytesIO() # bytes buffer for the plot
        fig = oAxes#.get_figure()
        fig.savefig(Bio, dpi=fig.dpi, bbox_inches='tight')
        #fig.canvas.print_png(Bio, bbox_inches='tight') # make a png of the plot in the buffer

        # encode the bytes as string using base 64
        sB64Img = base64.b64encode(Bio.getvalue()).decode()
        self.sHtml += (
            '<div class="floating-box">'+
            '<img src="data:image/png;base64,{}\n">'.format(sB64Img)+
            '</div>')

    def add_plot_ax(self, oAxes):
        ''' Saves a PNG representation of a Matplotlib Axes object '''
        Bio = io.BytesIO() # bytes buffer for the plot
        fig = oAxes.get_figure()
        fig.savefig(Bio, dpi=fig.dpi, bbox_inches='tight')
        #fig.canvas.print_png(Bio) # make a png of the plot in the buffer

        # encode the bytes as string using base 64
        sB64Img = base64.b64encode(Bio.getvalue()).decode()
        self.sHtml += (
            '<div class="floating-box">'+
            '<img src="data:image/png;base64,{}\n">'.format(sB64Img)+
            '</div>')

    def PassHtmlToCell(self):
        ''' Final step - display the accumulated HTML '''
        display(HTML(self.sHtml))
