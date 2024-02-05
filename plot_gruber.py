#!/usr/bin/env python3
# -*- coding: utf-8 -*-   
"""
A class to plot a Gruber section for the Bern3D model.

Most functions in this file were made by
Author: Gunnar Jansen, gunnar.jansen@unibe.ch
"""

import plot_helpers as hlp
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cmp           # colormaps
import matplotlib.colors as cols
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import seaborn as sns

import functions as f

class Gruber(object):
    def __init__(self, dataset=None, variable=None, time=-1, scale=1, title='', cmap=cmp.coolwarm, clabel='',
                 cmin=None, cmax=None, levels=10,
                 section_lat=[[-58, 67.5], [-58], [-58, 67.5]],
                 section_lon=[[335.5], [205, 328.5], [195]],
                 section_title=['Atlantic', 'Southern Ocean', 'Pacific'],
                 section_xticks=[[60, 30, 0, -30, -60], [200, 260, 320], [60, 30, 0, -30, -60]]):
        """author: G. Jansen, gunnar.jansen@climate.unibe.ch
        
        Inputs:
          * dataset - xarray dataset object
          * variable - string of variable name from dataset
          * time - timestep to inspect in Gruber plot
          * scale - factor to scale values in plot (e.g. unit conversions)
          * title - Title for the Gruber plot
          * cmap - matplotlib colormap object
          * clabel - string for colorbar label
          * cmin - Minimum value for color axis (float)
          * cmax - Maximum value for color axis (float)
          * levels - number of countour levels to plot
          * section_lat - [3, x] array of section latitudes on T-grid
          * section_lon - [3, x] array of section longitudes on T-grid
          * section_xticks - [3, x] array of section x-ticks

        Important note: The standard setup reproduces the standard Gruber plot.
        Other sections are possible by providing section_lat, section_lon and
        section_xticks. Note that the sections must be defined similar to the
        pre-defined setup. Every section has either:
        - 2 lat and 1 lon coordinate  => latitudinal section
        - 1 lat and 2 lon coordinates => longitudinal section
        - 2 lat and 2 lon coordinates => diagonal section

        The Gruber plot with default sections is (in T-grid coordinates):
        Atlantic @335.5 lon_t, from 67.5 to -58 lat_t
        S.O.     @-58 lat_t, from 328.5 to 205 lon_t
        Pacific  @195 lon_t, from -58 to 67.5 lat_t
        """
        self.dataset = dataset
        self.variable = variable
        self.time = time
        self.scale = scale
        self.title = title
        self.cmap = cmap
        self.clabel = clabel
        self.cmin = cmin
        self.cmax = cmax
        self.levels = levels
        self.section_lat = section_lat
        self.section_lon = section_lon
        self.section_title = section_title
        self.section_xticks = section_xticks

        

    def set_xticklabels(self, axes, trajectory_map=False):
        """author: G. Jansen, gunnar.jansen@climate.unibe.ch, 
        Adapted by jeemijn.scheen@climate.unibe.ch
        
        Sets and calculates appropriate x-axis ticks and labels.
        Includes automatic conversion from Bern3D coordinates to standard
        [-180, 180] and [0, 360] coordinates.
        - set trajectory_map=True for a (lon,lat) plot (trajectory plot) s.t. y ticks also changed
        """
        for ax in axes:

            if trajectory_map:
                # x axis (lon => needs W/S)
                ticks = ax.get_xticks()
                ticks = np.asarray([lon-360 if lon >= 180 else lon for lon in ticks])  # convert Bern3D grid [100,460] to [-180,180]
                labels = [s + r'$^{\circ}$' for s in abs(ticks).astype('str')]
                suffix = ['W' if s == -1 else 'E' if s == 1 else '' for s in np.sign(ticks)]
                ## old working version: gives all in W (also 270W)
                # ticks = ticks -360 +360*((ticks-360) < 0).astype(int)                
                # labels = [s + r'$^{\circ}$' for s in abs(ticks-360).astype('str')]
                # suffix = ['W' if s == 1 else 'E' if s == -1 else '' for s in np.sign(ticks)]
                xlabels = [l+n for l, n in zip(labels, suffix)]
                ax.set_xticklabels(xlabels)

                # y axis (lat => needs N/S)
                ticks = ax.get_yticks()
                labels = [s + r'$^{\circ}$' for s in abs(ticks).astype('str')]
                suffix = ['N' if s == 1 else 'S' if s == -1 else '' for s in np.sign(ticks)]
                ylabels = [l+n for l, n in zip(labels, suffix)]
                ax.set_yticklabels(ylabels)
            else:
                # Gruber section plot: 
                # x axis (lon or lat): infer from xticks which is needed 
                # i.e. if section is going in zonal or meridional direction
                ticks = ax.get_xticks()
                if np.all(abs(ticks) < 100):
                    labels = [s + r'$^{\circ}$' for s in abs(ticks).astype('str')]
                    suffix = ['N' if s == 1 else 'S' if s == -1 else '' for s in np.sign(ticks)]
                    xlabels = [l+n for l, n in zip(labels, suffix)]
                else:
                    ticks = np.asarray([lon-360 if lon >= 180 else lon for lon in ticks])  # convert Bern3D grid [100,460] to [-180,180]
                    labels = [s + r'$^{\circ}$' for s in abs(ticks).astype('str')]
                    suffix = ['W' if s == -1 else 'E' if s == 1 else '' for s in np.sign(ticks)]
                    xlabels = [l+n for l, n in zip(labels, suffix)]

                ax.set_xticklabels(xlabels)


    def get_cticks(self, lo, hi, levels, con_ticks=None):
        """author: G. Jansen, gunnar.jansen@climate.unibe.ch
        
        Sets and calculates appropriate color-axis ticks and labels.
        Includes ticks and labels for contour lines as well.
        """
        levels = self.levels

        ticks = np.linspace(lo, hi, (levels+1))[::2]
        ticks = [float(str("%.3g"%i)) for i in np.nditer(ticks)]

        tickslabel = []
        for el in ticks:
            #remove tailing 0 after decimal point
            s = str(el)
            tickslabel.append(s.rstrip('0').rstrip('.') if '.' in s else s)

        #ticks for contour
        if con_ticks is None:
            conticks = np.linspace(lo, hi, (levels+1))[::1]
        else:
            conticks = np.linspace(lo, hi, (con_ticks+1))[::1]

        #remove 0 from contour-ticks (done separately)
        conticks = list(conticks)
        conticks = [x for x in conticks if x != 0.0]
        conticks = np.array(conticks)

        conticks = [float(str("%.3g"%i)) for i in np.nditer(conticks)]

        cfmt = {}
        for el in conticks:
            #remove trailing 0 after decimal point
            s = str(el)
            cfmt[el] = s.rstrip('0').rstrip('.') if '.' in s else s

        return ticks, tickslabel, conticks, cfmt



    def add_colorbar(self, fig, data, cbar_below=True, extend_both=False):
        """author: G. Jansen, gunnar.jansen@climate.unibe.ch
        
        Adds colorbar to the Gruber plot
        """
        if self.cmin is None:
            cmin = self.scale*np.nanmin(data)
        else:
            cmin = self.cmin
        if self.cmax is None:
            cmax = self.scale*np.nanmax(data)
        else:
            cmax = self.cmax

        if extend_both:
            ext = 'both'
        else:
            ext = 'neither'
            if np.nanmin(data) < cmin and np.nanmax(data) > cmax:
                ext = 'both'
            elif np.nanmin(data) < cmin and (not np.nanmax(data) > cmax):
                ext = 'min'
            elif (not np.nanmin(data) < cmin) and np.nanmax(data) > cmax:
                ext = 'max'

        ticks, tickslabel, conticks, cfmt = self.get_cticks(cmin, cmax, self.levels)

        norm = cols.Normalize(vmin=cmin, vmax=cmax)

        if cbar_below:
            #position =       [left, bottom, width, height]
            cax = fig.add_axes([0.27, -0.065, 0.52, 0.045])
            cb=mpl.colorbar.ColorbarBase(cax,norm=norm,cmap=self.cmap,extend=ext,ticks=ticks,orientation='horizontal')
            cb.ax.set_xticklabels(tickslabel)
            cax.set_ylabel(self.clabel, fontsize=16, rotation=0)
            cax.yaxis.set_label_coords(-0.2, -0.8)
        else: # plot cbar on right hand side
            #position =       [left, bottom, width, height]
            cax = fig.add_axes([0.93, 0.23, 0.03, 0.65])
            cb=mpl.colorbar.ColorbarBase(cax,norm=norm,cmap=self.cmap,extend=ext,ticks=ticks,orientation='vertical')
            cb.ax.set_yticklabels(tickslabel)
            cax.set_ylabel(self.clabel, fontsize=14, rotation=0)
            # cax.yaxis.set_label_coords(2.8, -0.05)  # if label is like 'Pad [uBq/kg]'
            cax.yaxis.set_label_coords(1.8, -0.05)  # if label is like '[uBq/kg]'


    def get_section_Tcoords(self, lat_t, lon_t):
        """ Finds section coordinates on T-grid:
        Input lat_t, lon_t gives start and end point of section in coord T-grid values.
        Output [lat_t_range, lon_t_range] are Dataarrays containing all coord T-grid values,
        of end points and in between.
        Direction is implicitly defined by the shapes of lat_t, lon_t:
        - VERTICAL   if len(lat_t)=2, len(lon_t)=1 
        - HORIZONTAL if len(lat_t)=1, len(lon_t)=2
        - DIAGONAL   if len(lat_t)=2, len(lon_t)=2  """
        
        if len(lat_t)==1:
            if len(lon_t)==2:
                vertical = False  # i.e. horizontal
                diagonal = False
            else:
                raise Exception("lengths of lat_t, lon_t do not form a meaningful combination.")      
        elif len(lat_t)==2:
            if len(lon_t)==1:
                vertical = True  
                diagonal = False
            elif len(lon_t)==2:
                vertical = False # not used,
                diagonal = True  # because diagonal boolean overpowers vertical boolean
            else:
                raise Exception("lengths of lat_t, lon_t do not form a meaningful combination.")
        else:
            raise Exception("lengths of lat_t, lon_t do not form a meaningful combination.")       
        
        if diagonal: # overwrites status of 'vertical'
            # convert degrees lat/lon to index lat/lon
            ilatmin_t = self.dataset.lat_t.values.tolist().index(lat_t[0])  # T-grid
            ilatmax_t = self.dataset.lat_t.values.tolist().index(lat_t[1])  # T-grid
            ilonmin_t = self.dataset.lon_t.values.tolist().index(lon_t[0])  # T-grid
            ilonmax_t = self.dataset.lon_t.values.tolist().index(lon_t[1])  # T-grid

            # find pixels on diagonal line with Bresenham algorithm (needs T-grid at 0.5, 1.5, etc.) 
            # NB here Bresenham is implemented/called as if grid were regular, because this 
            # keeps things simple and is more intuitive to readers of papers
            row, col = self.Bresenham(x0=ilonmin_t+0.5, y0=ilatmin_t+0.5, x1=ilonmax_t+0.5, y1=ilatmax_t+0.5)
            # convert back to Bern3D coordinates
            row = (row - 0.5).astype(int)
            col = (col - 0.5).astype(int)
            # convert coord index to coord value
            lat_t_range = np.asarray([self.dataset.lat_t[i].item() for i in col])
            lon_t_range = np.asarray([self.dataset.lon_t[i].item() for i in row])
        else:
            if vertical:
                ilatmin_t = self.dataset.lat_t.values.tolist().index(lat_t[0])  # T-grid
                ilatmax_t = self.dataset.lat_t.values.tolist().index(lat_t[1])  
                # slicing excludes last index so +1:
                lat_t_range = self.dataset.isel(lat_t=slice(ilatmin_t, ilatmax_t+1)).lat_t.values 
                lon_t_range = np.repeat(lon_t, (abs(ilatmax_t - ilatmin_t) + 1)) # repeat lon_t value along lat_t axis
            else:  # horizontal
                ilonmin_t = self.dataset.lon_t.values.tolist().index(lon_t[0])  # T-grid
                ilonmax_t = self.dataset.lon_t.values.tolist().index(lon_t[1])  
                # slicing excludes last index so +1:
                lon_t_range = self.dataset.isel(lon_t=slice(ilonmin_t, ilonmax_t+1)).lon_t.values 
                lat_t_range = np.repeat(lat_t, (abs(ilonmax_t - ilonmin_t) + 1)) # repeat lat_t value along lon_t axis

        return lat_t_range, lon_t_range


    def get_section_data(self, lat_t, lon_t, vertical=True, diagonal=False, input_all_cells=False):
        """author: G. Jansen, gunnar.jansen@climate.unibe.ch
        
        Extract the correct section data from dataset.
        Output: var (the data on T grid), x (corresponding slice of lat_u or lon_u axis).

        Important: Either lat_t or lon_t needs to be a [2, 1] list in ascending order.
                   The other needs to be a of shape [1, 1].
        Exception: For a diagonal section both need to be a [2,1] list (in any order) and
                   'diagonal' needs to be True (overwrites 'vertical')
                   
        - input_all_cells gives the option to directly prescribe all coordinates
        i.e. with section_lon = [lon1, lon2, lon3, ..., lon_end] in any shape as opposed to the 
        default (input_all_cells=False) section_lon = [[lon_start,lon_end], [lon_start2, lon_end2]] etc
        NB if True then the options vertical and diagonal are ignored.
        """
        
        if input_all_cells:
            if len(lat_t) != len(lon_t):
                raise TypeError('For input_all_cells=True lat_t and lon_t need the same length.')
            var = xr.concat([self.dataset[self.variable].sel(lat_t=lat_t[i], lon_t=lon_t[i]).isel(time=-1) 
                             for i in range(len(lat_t))],
                             dim='n').transpose()  # 'n' is name of new dimension that lists index of x=(lon,lat) coords

            # this would be useful administration for ticks but plotting routine can't handle (lon,lat)
            # x = [(lon, lat) for lat, lon in zip(lon_t, lat_t)]
            
            # take var.n but make it 1 longer and also add 0.5 s.t. entire number ticks in middle of water column
            x = np.asarray(list(var.n.values) + [var.n[-1].item()+1]) + 0.5  

            # This outputs an index like [0.5, 1.5 etc] of the correct length. Can check by printing:
            # print('x:', x)
            # Why? we want to start water column count from 1 but this variable x is on the u-grid so gives the first edge at 0.5. 
            
            return var, x

        if diagonal: # overwrites status of 'vertical'
            try:
                if np.shape(lat_t) != (2,) or np.shape(lon_t) != (2,):
                    raise TypeError('You need length 2 of lat_t and lon_t for a diagonal section.')
                    # for diagonal, inputs don't need to be in ascending order (not always possible)

                lat_range, lon_range = self.get_section_Tcoords(lat_t=lat_t, lon_t=lon_t)

                # for this diagonal case, lat_t[0,1] are now start and end points but not necessarily minm, maxm
                # (ascending order not obligatory since not always possible for both coords simulataneously)
                lat_t_min = min(lat_t[0], lat_t[1])
                lat_t_max = max(lat_t[0], lat_t[1])
                lon_t_min = min(lon_t[0], lon_t[1])
                lon_t_max = max(lon_t[0], lon_t[1])              
                
                # construct corresponding u-grid coordinates for x
                ilatmin = self.dataset.lat_t.values.tolist().index(lat_t_min)   # u-grid index on LHS equals T-grid index
                ilatmax = self.dataset.lat_t.values.tolist().index(lat_t_max)+1 # u-grid index on RHS equals T index plus 1
                ilonmin = self.dataset.lon_t.values.tolist().index(lon_t_min)
                ilonmax = self.dataset.lon_t.values.tolist().index(lon_t_max)+1

                dy = abs(ilatmax - ilatmin)
                dx = abs(ilonmax - ilonmin)

                # For diagonal sections, both lat and lon increase. Which one to pick as coord x? 
                # We choose x=lat if slope is 45 degrees (dy=dx, most common use case) or if
                # mainly vertical (dy > dx). We choose x=lon if mainly horizontal (dy < dx).
                if dy >= dx: # pick lat
                    x = self.dataset.isel(lat_u=slice(ilatmin, ilatmax+1)).lat_u.values
        
                    # construct required part of datavar            
                    var = self.dataset[self.variable].isel(time=self.time)
                    # list all diagonal cells (actually water columns) one by one:
                    diag_cells = [var.sel(lon_t=i, lat_t=j) for i,j in zip(lon_range,lat_range)] 
                    var = xr.concat(diag_cells, dim='lat_t') # lon_t stays on as non-iterable coordinate for reference
                    var = var.transpose('z_t', 'lat_t')   # set dimensions in correct order
                else:  # pick lon                           
                    x = self.dataset.isel(lon_u=slice(ilonmin, ilonmax+1)).lon_u.values
                    
                    var = self.dataset[self.variable].isel(time=self.time)
                    diag_cells = [var.sel(lon_t=i, lat_t=j) for i,j in zip(lon_range,lat_range)] 
                    var = xr.concat(diag_cells, dim='lon_t') # lat_t stays on as non-iterable coordinate for reference
                    var = var.transpose('z_t', 'lon_t')
            except:
                raise Exception('Dimensions of lat_t and/or lon_t are not consistent for diagonal section.')
        else:
            if vertical:
                try:
                    if lat_t[0] > lat_t[1]:
                        raise TypeError('Inputs must be in ascending order.')
                    if np.shape(lon_t) != (1,):
                        raise TypeError('You need len(lon_t)=1 for a vertical section.')
                    var = self.dataset[self.variable].sel(lat_t=slice(lat_t[0],lat_t[1])).sel(lon_t=lon_t[0]).isel(time=-1)
                    ilatmin = self.dataset.lat_t.values.tolist().index(lat_t[0])    # u-grid index on LHS equals T-grid index
                    ilatmax = self.dataset.lat_t.values.tolist().index(lat_t[1])+1  # u-grid index on RHS equals T index plus 1
                    x = self.dataset.isel(lat_u=slice(ilatmin, ilatmax+1)).lat_u.values  
                except:
                    raise Exception('Dimensions of lat_t and/or lon_t are not consistent for vertical section.')
            else:
                try:
                    if np.shape(lat_t) != (1,):
                        raise TypeError('You need len(lat_t)=1 for a horizontal section.')
                    if lon_t[0] > lon_t[1]:
                        raise TypeError('Inputs must be in ascending order.')
                    var = self.dataset[self.variable].sel(lon_t=slice(lon_t[0],lon_t[1]),lat_t=lat_t[0]).isel(time=self.time)
                    ilonmin = self.dataset.lon_t.values.tolist().index(lon_t[0])
                    ilonmax = self.dataset.lon_t.values.tolist().index(lon_t[1])+1
                    x = self.dataset.isel(lon_u=slice(ilonmin, ilonmax+1)).lon_u.values
                except:
                    raise Exception('Dimensions of lat_t and/or lon_t are not consistent for horizontal section.')
        return var, x


    def add_section(self, ax_upper, ax_lower, lat_t, lon_t, title='',
                    vertical=True, diagonal=False, invert_x=True, invert_y=True, 
                    input_all_cells=False, avoid_negative=False):
        """author: G. Jansen, gunnar.jansen@climate.unibe.ch
        
        Adds a section to the Gruber plot.
        This method takes two axes as well as the section coordinates as input.
        Sections can be either vertical or horizontal depending on the boolean
        vertical, or they can be diagonal (boolean diagonal ignores boolean vertical).
        If this section is run in the direction of decreasing lat_t or decreasing lon_t,
        then invert_x needs to be true. Invert_y makes depth downwards (typically always true).
        
        - input_all_cells gives the option to directly prescribe all coordinates
        i.e. with section_lon = [lon1, lon2, lon3, ..., lon_end] in any shape as opposed to the 
        default (input_all_cells=False) section_lon = [[lon_start,lon_end], [lon_start2, lon_end2]] etc
        NB if True then the options vertical, diagonal, invert_x and invert_y are ignored.
        """

        var, x = self.get_section_data(lat_t, lon_t, vertical, diagonal, input_all_cells=input_all_cells)
        
        if self.cmin is None:
            cmin = np.nanmin(self.scale*var.values)
        else:
            cmin = self.cmin
        if self.cmax is None:
            cmax = np.nanmax(self.scale*var.values)
        else:
            cmax = self.cmax

        ticks, tickslabel, conticks, cfmt = self.get_cticks(cmin, cmax, self.levels)

        # scale z from m to km (if not already done in input data):
        if self.dataset.z_w.max() > 4500.0 and self.dataset.z_t.max() > 4500.0: 
            scale_z = 1.0e-3  # convert m to km
        elif self.dataset.z_w.max() < 5.1 and self.dataset.z_t.max() < 5.1:
            scale_z = 1.0     # input already in km
        else:
            raise Exception("coords z_w and z_t of input data need same unit (either m or km).")

        if avoid_negative:
            # avoid simulated negative concentrations in colorbar by setting them to zero:
            # this only occurs rarely, namely in Atl cruises only for GIPY05 for Pap, Thp in a few water columns
            var = xr.where(var < 0.0, 0.0, var)

        # plot twice, for upper kilometer and lower kilometers (1000-5000m):
        for axis in [ax_upper, ax_lower]:
            axis.pcolormesh(x, scale_z*self.dataset.z_w.values, self.scale*var.values, 
                            vmin=cmin, vmax=cmax, cmap=self.cmap, edgecolor='face')
            
            # Set x limits for more stability e.g. limits shouldn't shift when plotting observations on top
            axis.set_xlim(x[0],x[-1])

            if not input_all_cells: # for input_all_cells we leave out contours
                if diagonal: # overwrites vertical
                    # get T grid coordinates
                    lat_t_range, lon_t_range = self.get_section_Tcoords(lat_t=lat_t, lon_t=lon_t)
                    if 'lon_t' in var.dims:   # need lon_t
                        c = axis.contour(lon_t_range, scale_z*self.dataset.z_t.values, self.scale*var.values,
                                         levels=conticks, colors=('k'), linestyles='-')                    
                    elif 'lat_t' in var.dims: # need lat_t
                        c = axis.contour(lat_t_range, scale_z*self.dataset.z_t.values, self.scale*var.values,
                                         levels=conticks, colors=('k'), linestyles='-')
                    else:
                        raise Exception("ERROR: cannot find coordinates for diagonal section.")
                elif vertical is True:
                    c = axis.contour(var.lat_t, scale_z*self.dataset.z_t.values, self.scale*var.values,
                                     levels=conticks, colors=('k'), linestyles='-')
                else:
                    c = axis.contour(var.lon_t, scale_z*self.dataset.z_t.values, self.scale*var.values,
                                     levels=conticks, colors=('k'), linestyles='-')

        # axis-specific settings:
        ax_upper.set_ylim([0, 1])
        ax_lower.set_ylim([1, 5])

        ax_upper.set_yticks([0, 0.5])
        ax_upper.set_yticklabels(['0', '0.5']) # explicit to avoid '0.0'
        ax_upper.set_title(title)

        if invert_x and not input_all_cells:
            ax_upper.invert_xaxis()
            ax_lower.invert_xaxis()
        if invert_y:
            ax_upper.invert_yaxis()
            ax_lower.invert_yaxis()

        plt.setp(ax_upper.get_xticklabels(), visible=False)

        return var
    
    def plot_trajectory(self, section_lon, section_lat, verbose=False, alpha=0.85, input_all_cells=False):
        """Produces a figure with a map of the transect trajectory. 
        
        Input:
        - section_lon array should be the exact section_lon as used in the transect plotting fnc under
        consideration (i.e. fnc in this file) (length=nr of sections; each element is a 1 or 2D array)
        - section_lat array should be the exact section_lat as used in the transect plotting function
        - if verbose, then coordinates of each lat,lon are printed 
        - alpha is transparency of grid cells (e.g. 1.0, or lower to notice unwanted overlaps)
        - input_all_cells gives the option to directly prescribe all coordinates
          i.e. with section_lon = [lon1, lon2, lon3, ..., lon_end] in any shape as opposed to the 
          default (input_all_cells=False) section_lon = [[lon_start,lon_end], [lon_start2, lon_end2]] etc
        Output:
        - figure handle of trajectory figure"""
       
        sns.set_style('ticks')
        trajectory_fig, ax = plt.subplots(1)

        if len(section_lon) != len(section_lat):
            raise Exception("plot_trajectory(): section_lon and section_lat should have the same length") 

        # coordinates of total section
        if input_all_cells:
            sec_lat = section_lat
            sec_lon = section_lon
        else:
            # get_section_Tcoords extends [start,end] lat/lon to [start, start+1, ..., end]
            sec_lat = [self.get_section_Tcoords(lon_t=section_lon[i], lat_t=section_lat[i])[0] 
                       for i in range(len(section_lat))]
            sec_lon = [self.get_section_Tcoords(lon_t=section_lon[i], lat_t=section_lat[i])[1] 
                       for i in range(len(section_lon))]

            sec_lat = np.concatenate([sec_lat[i] for i in range(len(sec_lat))])
            sec_lon = np.concatenate([sec_lon[i] for i in range(len(sec_lon))])
        
        if len(sec_lon) != len(sec_lat):
            raise Exception("plot_trajectory(): something went wrong; sec_lon and sec_lat different length") 
        
        land_and_traject = f.get_landmask(self.dataset)[2].copy(deep=True) 
        # copy because don't want to change original
        
        # give grid cells of section a value s.t. will be coloured
        for this_lon, this_lat in zip(sec_lon, sec_lat):
            land_and_traject.loc[dict(lon_t=this_lon, lat_t=this_lat)] = 0.5

        X,Y = np.meshgrid(self.dataset.lon_u, self.dataset.lat_u) # u grid because pcolor needs left bottom corner
        Z = land_and_traject.values
        ax.pcolor(X,Y,Z, cmap='Dark2', alpha=alpha) # nice cmaps: Dark2, Set2, Accent, Spectral_r, copper 
        ax.set_yticks(range(-90,100,30))
        ax.set_xticks(range(180,460,90))
        self.set_xticklabels(plt.gcf().get_axes(), trajectory_map=True) # add degree N, S, W
        
        # annotate cruise order with number of water column
        if verbose:
            print('i:   lon_t:     lat_t:')
        counter = 1
        for this_lon, this_lat in zip(sec_lon, sec_lat):
            ax.text(this_lon,this_lat, str(counter), transform=ax.transData, color='white', 
                    fontsize=5, ha='center', va='center')
            if verbose:
                print(counter, "  ", this_lon, "    ", this_lat)
            counter += 1
        
        return trajectory_fig

    
    def plot(self, trajectory_info=False, input_all_cells=False, cruise='', 
             cbar_extend_both=False, avoid_negative=False):
        """Produces the complete Gruber section plot. Returns a matplotlib figure handle
        - default output: figure handle
        - if trajectory_info: outputs [fig, section_lat, section_lon] to use for a direct
          call of: fig = self.plot_trajectory(section_lat, section_lon)
        - input_all_cells gives the option to directly prescribe all coordinates
          i.e. with section_lon = [lon1, lon2, lon3, ..., lon_end] in any shape as opposed to the 
          default (input_all_cells=False) section_lon = [[lon_start,lon_end], [lon_start2, lon_end2]] etc
        - cruise gives the option to clarify cruise name in xlabel
        - cbar_extend_both [default False] forces a colorbar with arrows in both directions
        - avoid_negative [default False] plots (rare) negative concentrations as 0
          """

        if len(self.dataset.lat_t) != 40 or len(self.dataset.lon_t) != 41:
            raise Exception("This function is only implemented for the new 40x41 grid.")

        fig = plt.figure(figsize=(8, 4))

        if input_all_cells:
            # use only 1 subplot & 1 section as all cells are given in directly (instead of different sub-trajectories)
            
            # set height ratios for subplots
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
            ax0 = plt.subplot(gs[0])  # upper ocean
            ax1 = plt.subplot(gs[1])  # deep ocean

            sec1 = self.add_section(ax0, ax1, self.section_lat, self.section_lon,  # using this title because only 1 section
                                    title=self.title, input_all_cells=True, avoid_negative=avoid_negative)
            
            # Cosmetics: Remove spaces between subplots
            plt.subplots_adjust(hspace=.1)
            
            # Cosmetics: Set x-axis label
            if cruise == '':
                ax1.set_xlabel('Water column index', loc='center')
            else:
                if cruise in ['ng', 'deng', 'pavia']:
                    cruise_label = "transect by " + cruise.capitalize() + " et al." # more respectful naming
                else:
                    cruise_label = cruise
                ax1.set_xlabel('Water column index of ' + cruise_label, loc='center')
            
            ## Cosmetics: Set x-axis ticks
            water_column_ticks = self.section_xticks[0]  # indices of water columns, from 1

            # deep ocean: show water column index as xticks
            ax1.set_xticks(water_column_ticks)

            # # upper ocean: (A) EITHER show xticks without labels
            # ax0.set_xticks(water_column_ticks)
            # ax0.tick_params(top=True, labeltop=False, bottom=True, labelbottom=False)

            # upper ocean: (B) OR show lon or lat as xticks
            ax0.set_xticks(water_column_ticks)  # to place ticks at correct x position
            ax0.tick_params(top=True, labeltop=True, bottom=True, labelbottom=False)
            if len(np.unique(self.section_lon)) >= len(np.unique(self.section_lat)):
                # cruise goes more West-East than North-South  
                # => show lon coords
                new_xticklabels = [self.section_lon[i-1] for i in water_column_ticks]
                x_is_lon = True
            else:
                # cruise goes more North-South than West-East  
                # => show lat coords
                new_xticklabels = [self.section_lat[i-1] for i in water_column_ticks]
                x_is_lon = False
            # add '°W' etc
            ax0 = f.convert_ticks_of_section_plot(ax0, x_is_lon=x_is_lon, Bern3D_grid=True, 
                                                  explicit_xticklabels=new_xticklabels)
            if cruise == 'GA02':
                # omit 1 lat xticklabel because too crowded (printed on top of each other)
                ticklabels = ax0.get_xticklabels()
                ax0.set_xticklabels(ticklabels[:-2] + [""] + ticklabels[-1:]) # empty # -2

            # Colorbar: for this case (input_all_cells) we add a vertical colorbar on the right
            # to avoid confusion and cramped space with the now present xlabel
            self.add_colorbar(fig, sec1, cbar_below=False, extend_both=cbar_extend_both)
        else:
            # set height ratios for subplots
            gs = gridspec.GridSpec(2, 3, width_ratios=[3, 2, 3], height_ratios=[1, 2])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax2 = plt.subplot(gs[2])
            ax3 = plt.subplot(gs[3])
            ax4 = plt.subplot(gs[4])
            ax5 = plt.subplot(gs[5])

            sec1 = self.add_section(ax0, ax3, self.section_lat[0], self.section_lon[0],
                                    title=self.section_title[0], vertical=True, diagonal=False,
                                    invert_x=True, invert_y=True, avoid_negative=avoid_negative)
            sec2 = self.add_section(ax1, ax4, self.section_lat[1], self.section_lon[1],
                                    title=self.section_title[1], vertical=False, diagonal=False, 
                                    invert_x=True, invert_y=True, avoid_negative=avoid_negative)
            sec3 = self.add_section(ax2, ax5, self.section_lat[2], self.section_lon[2],
                                    title=self.section_title[2], vertical=True, diagonal=False,
                                    invert_x=False, invert_y=True, avoid_negative=avoid_negative)

            # Cosmetics: Remove spaces between subplots
            plt.subplots_adjust(wspace=.0)
            plt.subplots_adjust(hspace=.1)

            # Cosmetics: Set x-axis ticks
            # Goal: north/south ticks on bottom of ax_lower
            # east/west ticks on top of ax_upper
            # all ticks are repeated (w/o label) at the 1 km boundary
            ax0.set_xticks(self.section_xticks[0])
            ax1.xaxis.tick_top() # add ticks on top because this section is longitudinal
            ax1.xaxis.set_ticks_position('both') # also keep bottom ticks at 1 km
            ax1.set_xticks(self.section_xticks[1])
            ax2.set_xticks(self.section_xticks[2])
            ax3.set_xticks(self.section_xticks[0])
            ax4.set_xticks([])  # no ticks because they are in upper ax1
            ax5.set_xticks(self.section_xticks[2])
            self.set_xticklabels(plt.gcf().get_axes()) # add degree N, S, W

            # Cosmetics: Turn off y-axis tick labels on all sections except first
            plt.setp(ax1.get_yticklabels(), visible=False)
            plt.setp(ax2.get_yticklabels(), visible=False)
            plt.setp(ax4.get_yticklabels(), visible=False)
            plt.setp(ax5.get_yticklabels(), visible=False)

            # Colorbar
            self.add_colorbar(fig, np.hstack([sec1, sec2, sec3]), extend_both=cbar_extend_both)

            # Cosmetics: Set overall figure title
            #fig.suptitle(self.title, fontsize=14, ha='left')
            fig.text(0.1, 1.0, self.title, ha='center', va='center',
                     fontsize=14, bbox=dict(facecolor='none', edgecolor='black'))

        # Cosmetics: Set overall y-Label
        fig.text(0.06, 0.5, 'Depth (km)', ha='center', va='center',
                 rotation='vertical', fontsize=14)

        # Cosmetics: Turn ocean topography black
        for ax_nr,ax in enumerate(plt.gcf().get_axes()):
            if ax_nr % 3 != 2: # excluding colorbars (assuming every 3rd axis) because gives ugly black behind extension arrow
                ax.set_facecolor('k')

        if trajectory_info:
            return [fig, self.section_lat, self.section_lon] 
            # were already present from initialization, but this function confirms these were actually used
        else:
            return fig


    def plot_GA03_GA02N(self, old_grid=True, trajectory_info=False, avoid_negative=False, cbar_extend_both=False):
        """This function is based on the standard Gruber plot() function, but
        represents another transect: geotraces GA03 Leg 2, continuing on GA02N
        
        - old_grid: this plot works for the old 36x36 grid and the new 40x41 grid
        - default output: figure handle
        - if trajectory_info: outputs [fig, section_lat, section_lon] to use for a direct
          call of: fig = self.plot_trajectory(section_lat, section_lon)

        This section is given in rempfer17epsl, supplementary Fig A4. Note that rempfer17epsl
        wrongly names this total transect GA03 instead of GA03 Leg 2 + GA02N.
        Source: https://www.bodc.ac.uk/geotraces/cruises/section_maps/atlantic_ocean

        This function produces the complete section plot. Returns a matplotlib figure handle
        
        IF ONE SECTION OF THE PLOT STAYS BLACK, CHECK IF THE XTICKS OF THAT SECTION ARE ON LAT RESP LON GRID.
        For diagonal sections: lat ticks are needed if dx=dy (45 degrees) or dx<dy; otherwise lon.
        """
       
        fig = plt.figure(figsize=(8, 4))
        # set height ratios for subplots
        gs = gridspec.GridSpec(2, 4, width_ratios=[2, 1, 1, 1], height_ratios=[1, 2])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax2 = plt.subplot(gs[2])
        ax3 = plt.subplot(gs[3])
        ax4 = plt.subplot(gs[4])
        ax5 = plt.subplot(gs[5])
        ax6 = plt.subplot(gs[6])
        ax7 = plt.subplot(gs[7])

        # change default section_lat and section_lon to GA03 + GA02N transect:
        # consists of 4 subsections, starting from Western Africa: 
        
        if old_grid:
            if len(self.dataset.lat_t) != 36 or len(self.dataset.lon_t) != 36:
                raise Exception("ERROR: old_grid is set to True but grid is not the old 36x36 grid.")
            # These sections reproduce the transect in rempfer17epsl Fig. A4, (wrongly) named GA03
            # diagonal 6 cells; vertical 3 cells; diagonal 3 cells; vertical 2 cells
            self.section_lon = [[345.0, 295.0], [295.0], [295.0, 315.0], [315.0]]
            self.section_lat = [[7.983556145555406, 24.624318352164067], [28.178642740529867, 35.68533471265205], 
                                [39.70901655306836, 48.590377890729144], [53.663942485386045, 59.441568214065114]]
            self.section_title=['NW: Senegal to USA', 'to N', 'to NE', 'N: to Greenland']
            self.section_xticks=[[10, 20], [30], [40, 50], [60]] # all are lat in this case           
        else: # new grid
            if len(self.dataset.lat_t) != 40 or len(self.dataset.lon_t) != 41:
                raise Exception("ERROR: old_grid is set to False but grid is not the new 40x41 grid.")
            # These sections convert the transect closest to rempfer17epsl Fig. A4, (wrongly) named GA03,
            # to the new grid. Purpose: comparison to rempfer17epsl, Fig. 2
            # 4 sections have resp. 9 diagonal cells; 3 vertical; 2 diagonal; 2 vertical
            self.section_lon = [[349.5, 293.5], [293.5], [300.5, 307.5], [314.5]]
            self.section_lat = [[7.5, 22.5], [27.5, 37.5], [42.5, 47.5], [52.5, 57.5]]
            self.section_title=['NW: Senegal to USA', 'to N', 'to NE', 'N: to Greenland']
            self.section_xticks=[[300, 320, 340], [30], [40, 50], [60]] # sec1 is lon; rest is lat in this case
            # note that for diagonal sections, lat ticks are needed if dx=dy (45 degrees) or dx<dy; otherwise lon
            
        sec1 = self.add_section(ax0, ax4, self.section_lat[0], self.section_lon[0],
                                title=self.section_title[0], diagonal=True, 
                                invert_x=False, invert_y=True, avoid_negative=avoid_negative)
        sec2 = self.add_section(ax1, ax5, self.section_lat[1], self.section_lon[1],
                                title=self.section_title[1], vertical=True,
                                invert_x=False, invert_y=True, avoid_negative=avoid_negative)
        sec3 = self.add_section(ax2, ax6, self.section_lat[2], self.section_lon[2],
                                title=self.section_title[2], diagonal=True, 
                                invert_x=False, invert_y=True, avoid_negative=avoid_negative)
        sec4 = self.add_section(ax3, ax7, self.section_lat[3], self.section_lon[3],
                                title=self.section_title[3], vertical=True,
                                invert_x=False, invert_y=True, avoid_negative=avoid_negative)

        # Cosmetics: Remove spaces between subplots
        plt.subplots_adjust(wspace=.0)
        plt.subplots_adjust(hspace=.1)

        # Cosmetics: Set overall figure title
        #fig.suptitle(self.title, fontsize=14, ha='left')
        fig.text(0.1, 1.0, self.title, ha='center', va='center',
                 fontsize=14, bbox=dict(facecolor='none', edgecolor='black'))

        # Cosmetics: Set overall y-Label
        fig.text(0.06, 0.5, 'Depth (km)', ha='center', va='center',
                 rotation='vertical', fontsize=14)

        # Cosmetics: Set x-axis ticks
        # Goal: north/south ticks on bottom of ax_lower
        # east/west ticks on top of ax_upper
        # all ticks are repeated (w/o label) at the 1 km boundary
        ax0.set_xticks(self.section_xticks[0])
        ax1.set_xticks(self.section_xticks[1])
        ax2.set_xticks(self.section_xticks[2])
        ax3.set_xticks(self.section_xticks[3])
        ax4.set_xticks(self.section_xticks[0])
        ax5.set_xticks(self.section_xticks[1])
        ax6.set_xticks(self.section_xticks[2])
        ax7.set_xticks(self.section_xticks[3])
        self.set_xticklabels(plt.gcf().get_axes()) # add degree N,W,S,E
       
        # Cosmetics: Turn off y-axis tick labels on all sections except first
        plt.setp(ax1.get_yticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        plt.setp(ax5.get_yticklabels(), visible=False)
        plt.setp(ax6.get_yticklabels(), visible=False)
        plt.setp(ax7.get_yticklabels(), visible=False)

        # Cosmetics: Turn ocean topography black
        for ax in plt.gcf().get_axes():
            ax.set_facecolor('k')

        # Colorbar
        self.add_colorbar(fig, np.hstack([sec1, sec2, sec3, sec4]))

        if trajectory_info:
            return [fig, self.section_lat, self.section_lon] # were set above depending on old_grid
        else:
            return fig
        
        
    def plot_GA02(self, trajectory_info=False, avoid_negative=False, cbar_extend_both=False):
        """This function is based on the standard Gruber plot() function, but
        represents another transect: geotraces GA02 (all 3 legs)
        
        - default output: figure handle
        - if trajectory_info: outputs [fig, section_lat, section_lon] to use for a direct
          call of: fig = self.plot_trajectory(section_lat, section_lon)

        Source: https://www.bodc.ac.uk/geotraces/cruises/section_maps/atlantic_ocean and cruise reports therein

        This function produces the complete section plot. Returns a matplotlib figure handle
        
        IF ONE SECTION OF THE PLOT STAYS BLACK, CHECK IF THE XTICKS OF THAT SECTION ARE ON LAT RESP LON GRID.
        For diagonal sections: lat ticks are needed if dx=dy (45 degrees) or dx<dy; otherwise lon.
        """
       
        fig = plt.figure(figsize=(8, 4))
        # set height ratios for subplots
        gs = gridspec.GridSpec(2, 7, width_ratios=[3, 2, 3, 1, 1, 2, 1], height_ratios=[1, 2])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax2 = plt.subplot(gs[2])
        ax3 = plt.subplot(gs[3])
        ax4 = plt.subplot(gs[4])
        ax5 = plt.subplot(gs[5])
        ax6 = plt.subplot(gs[6])
        ax7 = plt.subplot(gs[7])
        ax8 = plt.subplot(gs[8])
        ax9 = plt.subplot(gs[9])
        ax10 = plt.subplot(gs[10])
        ax11 = plt.subplot(gs[11])
        ax12 = plt.subplot(gs[12])
        ax13 = plt.subplot(gs[13])

        # change default section_lat and section_lon to GA02 transect:
        # consists of 7 subsections, starting from South Argentina through the Western Atlantic to Greenland:
        # diagonal cells; diagonal; diagonal; 2 vertical; 2 horizontal; diagonal; diagonal 
        #   with nr of diagonal cells depending on how angle is (Bresenham algorithm has chosen automatically)
        
        if len(self.dataset.lat_t) != 40 or len(self.dataset.lon_t) != 41:
            raise Exception("ERROR: grid is not 40x41.")
        self.section_lon = [[307.5,328.5],[328.5,335.5],[328.5,300.5],[293.5],[293.5,300.5],[307.5,321.5],[314.5,328.5]]
        self.section_lat = [[-50.0,-22.5],[-17.5,-1.5],[-1.5,17.5],[22.5,27.5],[32.5],[32.5,47.5],[52.5,62.5]]
        self.section_title=['NE from South Argentina','','NW to Bermuda','','','NE to Greenland','']
        self.section_xticks=[[-40],[-20],[0],[20],[300],[40],[60]] # sec5 is lon; rest is lat in this case
        # note that for diagonal sections, lat ticks are needed if dx=dy (45 degrees) or dx<dy; otherwise lon
            
        sec1 = self.add_section(ax0, ax7, self.section_lat[0], self.section_lon[0],
                                title=self.section_title[0], diagonal=True, 
                                invert_x=False, invert_y=True)
        sec2 = self.add_section(ax1, ax8, self.section_lat[1], self.section_lon[1],
                                title=self.section_title[1], diagonal=True,
                                invert_x=False, invert_y=True)
        sec3 = self.add_section(ax2, ax9, self.section_lat[2], self.section_lon[2],
                                title=self.section_title[2], diagonal=True, 
                                invert_x=False, invert_y=True)
        sec4 = self.add_section(ax3, ax10, self.section_lat[3], self.section_lon[3],
                                title=self.section_title[3], vertical=True,
                                invert_x=False, invert_y=True)
        sec5 = self.add_section(ax4, ax11, self.section_lat[4], self.section_lon[4],
                                title=self.section_title[4], vertical=False,
                                invert_x=False, invert_y=True)
        sec6 = self.add_section(ax5, ax12, self.section_lat[5], self.section_lon[5],
                                title=self.section_title[5], diagonal=True,
                                invert_x=False, invert_y=True)
        sec7 = self.add_section(ax6, ax13, self.section_lat[6], self.section_lon[6],
                                title=self.section_title[6], diagonal=True,
                                invert_x=False, invert_y=True)

        # Cosmetics: Remove spaces between subplots
        plt.subplots_adjust(wspace=.0)
        plt.subplots_adjust(hspace=.1)

        # Cosmetics: Set overall figure title
        #fig.suptitle(self.title, fontsize=14, ha='left')
        fig.text(0.1, 1.0, self.title, ha='center', va='center',
                 fontsize=14, bbox=dict(facecolor='none', edgecolor='black'))

        # Cosmetics: Set overall y-Label
        fig.text(0.06, 0.5, 'Depth (km)', ha='center', va='center',
                 rotation='vertical', fontsize=14)

        # Cosmetics: Set x-axis ticks
        # Goal: north/south ticks on bottom of ax_lower
        # east/west ticks on top of ax_upper
        # all ticks are repeated (w/o label) at the 1 km boundary
        ax0.set_xticks(self.section_xticks[0])
        ax1.set_xticks(self.section_xticks[1])
        ax2.set_xticks(self.section_xticks[2])
        ax3.set_xticks(self.section_xticks[3])
        ax4.set_xticks(self.section_xticks[4])
        ax5.set_xticks(self.section_xticks[5])
        ax6.set_xticks(self.section_xticks[6])
        ax7.set_xticks(self.section_xticks[0])
        ax8.set_xticks(self.section_xticks[1])
        ax9.set_xticks(self.section_xticks[2])
        ax10.set_xticks(self.section_xticks[3])
        ax11.set_xticks(self.section_xticks[4])
        ax12.set_xticks(self.section_xticks[5])
        ax13.set_xticks(self.section_xticks[6])
        self.set_xticklabels(plt.gcf().get_axes()) # add degree N,W,S,E

        # Cosmetics: Turn off y-axis tick labels on all sections except first (upper+lower ax)
        for axis in [ax1, ax2, ax3, ax4, ax5, ax6, ax8, ax9, ax10, ax11, ax12, ax13]:            
            plt.setp(axis.get_yticklabels(), visible=False)

        # Cosmetics: Turn ocean topography black
        for ax in plt.gcf().get_axes():
            ax.set_facecolor('k')

        # Colorbar
        self.add_colorbar(fig, np.hstack([sec1, sec2, sec3, sec4, sec5, sec6, sec7]))

        if trajectory_info:
            return [fig, self.section_lat, self.section_lon]
        else:
            return fig


    def Bresenham(self, x0, y0, x1, y1):
        """Bresenham line algorithm to convert a diagonal line to i,j pixels
        Finds indices of the dominant grid points that intersect the diagonal line
        See https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm#Method for more information
        This function uses the grid convention as in this wikipedia article:
        - (0,0) on top left and y coordinate goes downward (also works if y coord input goes upward)
        - u-grid/v-grid represented by whole-numbered indices 0, 1, 2, ...
        - T-grid represented by half-numbered indices 0.5, 1.5, 2.5, ...
        - algorithm should still work if T-grid is not centered, e.g. 0.4, 1.7, etc. (not tested)

        Input:
        - float x0, y0: start of the section as T-grid indices
        - float x1, y1: end of the section as T-grid indices
        NB this algorithm works in every direction, no need for x1 > x0 and y1 > y0.

        Output:
        - [row, col]: arrays with T-grid x,y indices of resulting section
        """

        for coord in [x0, y0, x1, y1]:
            if coord%1 < 1e-4: 
                raise Exception('ERROR: x0,y0,x1,y1 should lie on T-grid (usually indices 0.5, 1.5, etc)')

        dx = x1 - x0
        dy = y1 - y0

        ## DEFINE INCREMENTS FOR X AND Y DEPENDING ON DIRECTION OF LINE
        if dx == 0:      # vertical line
            x_inc = 0                
        elif x1 > x0:    # to the right
            x_inc = 1
        else:            # to the left
            x_inc = -1
        if dy == 0:      # horizontal line
            y_inc = 0
        elif y1 > y0:    # down (if (0,0) at left upper corner)
            y_inc = 1
        else:            # up ( ""  "" )
            y_inc = -1

        ## 3 CASES, DEPENDING ON THE SLOPE (are valid for each possible line direction):
        error = 0.0
        row = []
        col = []
        x = x0
        y = y0
        # CASE A). abs(slope) = 1
        if int(round(abs(dx))) == int(round(abs(dy))):
            n_steps = int(round(abs(dx)))
            for i in range(0, n_steps+1): # 1 more coordinate to save than n_steps
                row = np.append(row, x)
                col = np.append(col, y)
                # always go diagonal: increment both x and y
                # no decisions needed, so no error needed
                x = x + x_inc
                y = y + y_inc

        # CASE B). abs(slope) < 1
        elif abs(dx) > abs(dy):
            # for every x increment, the error=distance(x_on_Tgrid, real_x_on_line). derive by similar triangles:
            derror = abs(dy/float(dx)) # delta error
            # loop over x to find exactly 1 match for every x
            for x in np.arange(x0, x1+0.5*x_inc, x_inc): # +0.5*x_inc to avoid last value being excluded
                row = np.append(row, x)
                col = np.append(col, y)
                error += derror # because we always increment x
                if error > 0.5: # then increment y in addition, so this step is diagonal
                # N.B. > 0.5 vs >= 0.5 is a random choice (influences result in certain cases)
                    y = y + y_inc
                    error = error - 1.0 # knowing that incrementing y goes in the direction of lowering the error

        # CASE C). abs(slope) > 1
        elif abs(dx) < abs(dy):
            # for every y increment, the error=distance(y_on_Tgrid, real_y_on_line). derive by similar triangles:
            derror = abs(dx/float(dy)) 
            # loop over y to find exactly 1 match for every y
            for y in np.arange(y0, y1+0.5*y_inc, y_inc): 
                row = np.append(row, x)
                col = np.append(col, y)
                error += derror # because we always increment y
                if error > 0.5: # then increment x in addition, so this step is diagonal
                    x = x + x_inc
                    error = error - 1.0 # knowing that incrementing x goes in the direction of lowering the error

        ## TESTING
        ## reproducing the figure on wikipedia (slope < 1): 
        # row,col = Bresenham(x0=1.5,y0=1.5,x1=11.5,y1=5.5)  # down to the right with slope < 1 WORKS
        # row,col = Bresenham(x1=1.5,y1=1.5,x0=11.5,y0=5.5)  # up to the left with slope < 1 WORKS
        # row,col = Bresenham(x0=11.5,y0=1.5,x1=1.5,y1=5.5)  # mirrored: down to the left with slope < 1 WORKS
        # row,col = Bresenham(x1=11.5,y1=1.5,x0=1.5,y0=5.5)  # mirrored: up to the right with slope < 1 WORKS
        ## comparable tests were done with slope > 1 and slope = 1
        # fig, ax = plt.subplots(1)
        # ax.plot(row, col)
        # ax.set_xticks(np.arange(0, 12, 1))
        # ax.set_yticks(np.arange(0, 12, 1))
        # ax.invert_yaxis()
        # plt.grid()

        return row, col