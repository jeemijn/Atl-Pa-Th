#!/usr/bin/env python3
# coding: utf-8

"""This file contains all user-defined functions (in random order). Authors are attributed in the docstring per function."""

def drop_most_vars(dataset, vars_to_keep):
   """Drops all other variables in xarray dataset

   Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
   all_vars = list(dataset.variables.keys())

   vars_to_drop = [var for var in all_vars if var not in vars_to_keep]
    
   return dataset.drop(vars_to_drop)

def plot_regions_map(ax, full_obj, ymax=5.0, verbose=False, title='Defined regions',
                     regions={1: [[25,26], [31,33]], 2: [[20,22], [28,30]],
                     3: [[23,25], [20,25]], 4: [[24,26], [13,16]]},
                     colors=['darkred', 'orangered', 'sandybrown','peachpuff'], lw=2,
                     Bern3D_grid=False):
   """Plots regions map in a subpanel.
   Input: 
   - axis of subplot to use for the map
   - full_obj is data_fulls[run] of a certain run (doesn't matter; only grid is used)
   - float ymax in km
   - verbose prints out the boundaries of the regions
   - title
   - regions dict with lon,lat python boundary indices per region. 
   - colors list for region rectangles
   - lw linewidth (default 2)
   - Bern3D_grid adds model grid lines
   
   Output
   - axis object given back; plot already made on it
   
   Example usage:
      fig, ax = plt.subplots(1,1, figsize=(4,5))
      plot_regions_map(ax, data_fulls[runs[0]], verbose=True)
      plt.tight_layout()
      plt.savefig(savedir / ('regions_map.pdf'))

    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl
   """

   # Define small regions of interest in the Atlantic; 
   # just 4-6 water columns s.t. not drawing conclusions from 1 water column/grid cell
   # form: [[lon_start, lon_end],[lat_start,lat_end]] with indices as Bern3D map i.e. starting from 1

   # The default describes regions of similar size for: 
   # Region 1: NE of Bermuda Rise / south of GRL
   # Region 2: Bermuda Rise
   # Region 3: off Venezuela & northern Brasil
   # Region 4: off southern Brasil
   
   from numpy import meshgrid
   from matplotlib import ticker
   from matplotlib.patches import Rectangle

   # preparations
   lon_u = full_obj.lon_u.values
   lat_u = full_obj.lat_u.values

   if verbose:
      print('regions setting:', regions, '\n')
      # print region info in T coordinates
      lon_t = full_obj.lon_t.values
      lat_t = full_obj.lat_t.values
      for region_nr, [lons,lats] in regions.items():
         print('Region', region_nr, 'is ',lon_t[lons[0]-1],'-',lon_t[lons[1]-1],
               'lon and',lat_t[lats[0]-1],'-',lat_t[lats[1]-1],'N (T-grid)')
      print('\n')

   land_mask = get_landmask(full_obj)
   X,Y = meshgrid(lon_u, lat_u) # u grid because pcolor needs left bottom corner

   # plot map
   ax.pcolormesh(X, Y, land_mask[2].values, cmap='Dark2')  # green land
   ax.set_title(title)

   if Bern3D_grid:
      # add lines according to Bern3D model grid
      # place minor ticks on every grid point such that they appear in grid
      ax.xaxis.set_major_locator(ticker.FixedLocator([270,300,330,360]))
      ax.set_xticklabels([-90,-60,-30,0])
      ax.xaxis.set_minor_locator(ticker.FixedLocator(full_obj.lon_u.values + 1e-2)) # adding an epsilon s.t. minor and major ticks don't coincide because ...
      # ... I use minor ticks for grid lines & major ticks for tick labels. However, matplotlib suppresses minor ticks when they coincide with major ticks.
      ax.yaxis.set_major_locator(ticker.FixedLocator(range(-60,70,30)))
      ax.yaxis.set_minor_locator(ticker.FixedLocator(full_obj.lat_u.values + 1e-2))
      ax.grid("on", which='minor', lw=0.5)
   else:
      ax.set_yticks(range(-60,70,30))
      ax.set_xticks([270,300,330,360])
      ax.set_xticklabels([-90,-60,-30,0])
      ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
      ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
   ax.set_xlim(258,382)

   # draw rectangles
   # note: xticklabels were overwritten, xticks go actually from 270 to 360
   # estimates to center text in the region rectangles
   fontsize_in_lon = 4.0
   fontsize_in_lat = 6.0
   for region_nr, [lons,lats] in regions.items():
      # convert t-grid to u-grid and fortran to python indices:
      lon_min = lon_u[lons[0]-1]
      lon_max = lon_u[lons[1]]
      lat_min = lat_u[lats[0]-1]
      lat_max = lat_u[lats[1]]   
      rect = Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min, 
                        linewidth=lw, facecolor='none', 
                        edgecolor=colors[region_nr-1], zorder=3)
      ax.add_patch(rect)
      ax.text(lon_min+(lon_max-lon_min)/2-fontsize_in_lon/2, 
              lat_min+(lat_max-lat_min)/2-fontsize_in_lat/2, 
              str(region_nr), color='k', fontstyle='italic', fontsize=14, zorder=3)
              
   return ax
   
   
def find_Atl_mask_on_lat_u(full_obj):
    """Input:
    - xarray DataSet data_fulls of any run; must contain 'masks' variable & z_t, lon_t, lat_u coords
    Output:
    - xarray DataArray Atl_mask_lat_u_coord is a boolean mask of the Atlantic basin (including its S.O.
    sector) with coordinates (z_t, lon_t, lat_u), i.e. suitable for variable v.
    
    ATTENTION all Atlantic z_t coords are True, also inside sediment. 
    So in the usage the var upon which you use it should be nan or 0 in the sediment.
    
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    from numpy import full, insert, asarray
    from xarray import DataArray

    ## make boolean Atl basin mask suitable for v => i.e. on (lon_t, lat_u, z_t) grid
    z_t = full_obj.z_t.data
    lon_t = full_obj.lon_t.data
    lat_u = full_obj.lat_u.data
    Atl_mask_lat_t_coord = (full_obj.masks == 1.0).copy(deep=True)

    # 1) initialize with False
    Atl_mask_lat_u_coord = DataArray(data=full((32,41,41), False),
                                     dims=['z_t', 'lat_u', 'lon_t'],
                                     coords={'z_t': (['z_t'], z_t,
                                               {'units': "m", 'long_name': "Depth", 
                                                'standard_name': "depth", 'axis': "Z"}),
                                            'lat_u': (['lat_u'], lat_u,
                                               {'units': "degrees_north", 'long_name': "Latitude", 
                                                'standard_name': "latitude", 'axis': "Y"}),
                                            'lon_t': (['lon_t'], lon_t,
                                               {'units': "degrees_east", 'long_name': "Longitude", 
                                                'standard_name': "longitude", 'axis': "X"})},
                                    attrs={'long_name': 'Atlantic basin mask including its Southern Ocean sector',
                                           'units': 'True for Atl basin (also in sediment!); False elsewhere'})
    # 2) construct with lat_u from the existing one with lat_t; TAKING SAME VALUE FOR ALL Z
    for i,this_lon in enumerate(lon_t):
        # consider a fixed lon band and :
        lon_band_mask = Atl_mask_lat_t_coord.sel(lon_t=this_lon).isel(z_t=0).values
        if len(lon_band_mask.nonzero()[0]) > 0:
            highest_true_index = lon_band_mask.nonzero()[0][-1]
            # make the array 1 longer (lon_t => lon_u) by inserting extra True s.t. True at both boundaries
            lon_band_mask = insert(lon_band_mask, highest_true_index+1, True)
        else: # lon band lies entirely outside Atl
            # make the array 1 longer, also needed here
            lon_band_mask = insert(lon_band_mask, -1, False)

        Atl_mask_lat_u_coord[:,:,i] = asarray([lon_band_mask]*32)  # copying lon_band_mask for all 32 depth layers

    # test: these 2 look now slightly different around GRL, as expected (plot for u grid is wrong / a bit shifted)
    # Atl_mask_lat_u_coord.isel(z_t=14).plot()
    # Atl_mask_lat_t_coord.isel(z_t=0).plot()

    # # test 2: these plots are identical indeed! 
    # # Except an extra point of 0 is added for Atl_mask_lat_u_coord because we did not mask out z_t values; however they are/should be 0 in v so fine.
    # full_obj.v.where(Atl_mask_lat_u_coord).sel(lat_u=lat_bnd).isel(z_t=0).plot()  # m/s; v lives on (lon_t, lat_u, z_t) coords
    # full_obj.v.sel(lon_t=slice(324.0,375.0), lat_u=lat_bnd).isel(z_t=0).plot()  # m/s; v lives on (lon_t, lat_u, z_t) coords
    
    return Atl_mask_lat_u_coord


def find_Atl_mask_on_lon_u(full_obj):
    """  ANALAGOUS TO FUNCTION find_Atl_mask_on_lat_u
    Input:
    - xarray DataSet data_fulls of any run; must contain 'masks' variable & z_t, lon_t, lat_u coords
    Output:
    - xarray DataArray Atl_mask_lon_u_coord is a boolean mask of the Atlantic basin (including its S.O.
    sector) with coordinates (z_t, lat_t, lon_u), i.e. suitable for variable u.
    
    ATTENTION all Atlantic z_t coords are True, also inside sediment. 
    So in the usage the var upon which you use it should be nan or 0 in the sediment.
    
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    from numpy import full, insert, asarray
    from xarray import DataArray

    ## make boolean Atl basin mask suitable for u => i.e. on (lon_u, lat_t, z_t) grid
    z_t = full_obj.z_t.data
    lat_t = full_obj.lat_t.data
    lon_u = full_obj.lon_u.data
    Atl_mask_lon_t_coord = (full_obj.masks == 1.0).copy(deep=True)

    # 1) initialize with False
    Atl_mask_lon_u_coord = DataArray(data=full((32,40,42), False),
                                     dims=['z_t', 'lat_t', 'lon_u'],
                                     coords={'z_t': (['z_t'], z_t,
                                               {'units': "m", 'long_name': "Depth", 
                                                'standard_name': "depth", 'axis': "Z"}),
                                            'lat_t': (['lat_t'], lat_t,
                                               {'units': "degrees_north", 'long_name': "Latitude", 
                                                'standard_name': "latitude", 'axis': "Y"}),
                                            'lon_u': (['lon_u'], lon_u,
                                               {'units': "degrees_east", 'long_name': "Longitude", 
                                                'standard_name': "longitude", 'axis': "X"})},
                                    attrs={'long_name': 'Atlantic basin mask including its Southern Ocean sector',
                                           'units': 'True for Atl basin (also in sediment!); False elsewhere'})
    # 2) construct with lon_u from the existing one with lon_t; TAKING SAME VALUE FOR ALL Z
    for i,this_lat in enumerate(lat_t):
        # consider a fixed lon band and :
        lat_band_mask = Atl_mask_lon_t_coord.sel(lat_t=this_lat).isel(z_t=0).values
        if len(lat_band_mask.nonzero()[0]) > 0:
            highest_true_index = lat_band_mask.nonzero()[0][-1]
            # make the array 1 longer (lat_t => lat_u) by inserting extra True s.t. True at both boundaries
            lat_band_mask = insert(lat_band_mask, highest_true_index+1, True)
        else: # lat band lies entirely outside Atl
            # make the array 1 longer, also needed here
            lat_band_mask = insert(lat_band_mask, -1, False)

        Atl_mask_lon_u_coord[:,i,:] = asarray([lat_band_mask]*32)  # copying lat_band_mask for all 32 depth layers

    return Atl_mask_lon_u_coord


def deg_to_rad(deg):
    from numpy import pi
    return deg / 360.0 * 2 * pi


def compute_transport_over_1_cell(full_obj, inside_cell, outside_cell, flux_type):
    """Computes the advection of Pad and Thd across 1 certain cell.
    Function is a simplified & edited copy of compute_merid_transport_across_section()
    It is a separated function s.t. it can be used outside Atlantic basin e.g. for 
    the flux into the Mediterranean.
        
    Input:
    - xarray DataSet data_fulls of the run; must contain 'Pad', 'Thd' & the usual coords
    - inside_cell is tuple with python indices (lon_t,lat_t)
    - outside_cell is tuple with python indices (lon_t,lat_t)
    - str flux_type: 'u' or 'v'
    Output:
    - [merid_transport_Pad, merid_transport_Thd] in uBq/s (summed over water column); 
      SIGN: positive if directed from inside_cell to outside_cell

    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    # u/v include advection and diffusion (also GM) of water itself
    # Here we compute meridional advection of Pa, Th with the water. 
    # Thus diffusion of Pa, Th (due to gradient fields of Pa, Th) is not taken into account.
    
    from numpy import diff, asarray, cos
    
    assert asarray(inside_cell).shape == (2,), "compute_transport_over_1_cell() expects only 1 inside_cell tuple (no [] around it)"
    assert asarray(outside_cell).shape == (2,), "compute_transport_over_1_cell() expects only 1 outside_cell tuple (no [] around it)"
    assert len(flux_type)==1, "compute_transport_over_1_cell() expects only 1 flux_type"
    
    lat_t = full_obj.lat_t
    lat_u = full_obj.lat_u
    lon_t = full_obj.lon_t
    lon_u = full_obj.lon_u

    dz = diff(full_obj.z_w) * 1000           # km to m
    R_earth = 6378137                        # m
    
    if flux_type == 'v':
        assert inside_cell[0] == outside_cell[0], "for flux_type v, lon should be constant"
        assert abs(inside_cell[1]-outside_cell[1]) == 1, "for flux_type v, lat_indices should differ by 1"

        # v at this lat,lon water column:
        # need to search for lat_u_bnd because lat on v-grid not always in middle of T-grid
        if lat_t[inside_cell[1]] < lat_t[outside_cell[1]]:
            # inside_cell lies south of outside_cell
            inside_south_of_outside = True
            # v-grid index equals highest T-grid index of neighbour cells e.g. |  T-grid 11  |v-grid 12|  T-grid 12  |
            lat_u_bnd = lat_u[outside_cell[1]].item()
        else:
            # inside_cell lies north of outside_cell
            inside_south_of_outside = False
            lat_u_bnd = lat_u[inside_cell[1]].item()
            
        this_v = full_obj.v.sel(lat_u=lat_u_bnd).isel(lon_t=inside_cell[0])  # m/s

        # area of (lon,z) surface at this lat,lon:
        lon_u_diff = (lon_u[inside_cell[0]+1] - lon_u[inside_cell[0]]).item()     # delta_lon in rad for this 1 grid cell
        # find bow length: multiply with cos(theta=lat) because 'horizontal' circumference of sphere is cos(theta) * 2 pi R 
        dlon = R_earth * deg_to_rad(lon_u_diff) * cos(deg_to_rad(lat_u_bnd))   # delta_lon in m
        this_area = dz * dlon                                                     # in m2 on v grid

        # concentration; convert concs of outside_cell and inside_cell on T grid to v grid via weighted avg:
        this_pad = ( ( full_obj.Pad.isel(lon_t=inside_cell[0],lat_t=inside_cell[1]) 
                      * abs(lat_u_bnd-lat_t[outside_cell[1]])
                      + full_obj.Pad.isel(lon_t=outside_cell[0],lat_t=outside_cell[1]) 
                      * abs(lat_u_bnd-lat_t[inside_cell[1]]) ) 
                    / abs(lat_t[inside_cell[1]] - lat_t[outside_cell[1]]) )
        this_thd = ( ( full_obj.Thd.isel(lon_t=inside_cell[0],lat_t=inside_cell[1]) 
                      * abs(lat_u_bnd-lat_t[outside_cell[1]])
                      + full_obj.Thd.isel(lon_t=outside_cell[0],lat_t=outside_cell[1]) 
                      * abs(lat_u_bnd-lat_t[inside_cell[1]]) ) 
                    / abs(lat_t[inside_cell[1]] - lat_t[outside_cell[1]]) )

        # resulting flux at this edge (summed over this water column):
        if inside_south_of_outside:
            # sign of v already in direction of inside to outside
            merid_transp_Pa = (this_pad * this_area * this_v).sum().item()
            merid_transp_Th = (this_thd * this_area * this_v).sum().item()
        else:
            # swap sign because inside to outside vector is 180 degrees opposite to v                
            merid_transp_Pa = (this_pad * this_area * -1 * this_v).sum().item()
            merid_transp_Th = (this_thd * this_area * -1 * this_v).sum().item()                
    elif flux_type == 'u':
        assert inside_cell[1] == outside_cell[1], "for flux_type u, lat should be constant"
        assert abs(inside_cell[0]-outside_cell[0]) == 1, "for flux_type u, lon_indices should differ by 1"

        # u at this lat,lon water column:
        # need to search for lon_u_bnd because lon on u-grid not always in middle of T-grid
        if lon_t[inside_cell[0]] < lon_t[outside_cell[0]]:
            # inside_cell lies west of outside_cell
            inside_west_of_outside = True
            # u-grid index equals highest T-grid index of neighbour cells e.g. |  T-grid 11  |u-grid 12|  T-grid 12  |
            lon_u_bnd = lon_u[outside_cell[0]].item()
        else:
            # inside_cell lies east of outside_cell
            inside_west_of_outside = False
            lon_u_bnd = lon_u[inside_cell[0]].item()
        this_u = full_obj.u.sel(lon_u=lon_u_bnd).isel(lat_t=inside_cell[1])  # m/s

        # area of (lat,z) surface at this lat,lon:
        lat_u_diff = lat_u[inside_cell[1]+1] - lat_u[inside_cell[1]] # delta_lat in rad for this 1 grid cell
        dlat = R_earth * deg_to_rad(lat_u_diff.item())               # delta_lat in m
        this_area = dz * dlat

        # concentration; convert concs of outside_cell and inside_cell on T grid to u grid via weighted avg:
        this_pad = ( ( full_obj.Pad.isel(lon_t=inside_cell[0],lat_t=inside_cell[1]) 
                      * abs(lon_u_bnd-lon_t[outside_cell[0]])
                      + full_obj.Pad.isel(lon_t=outside_cell[0],lat_t=outside_cell[1]) 
                      * abs(lon_u_bnd-lon_t[inside_cell[0]]) )
                    / abs(lon_t[inside_cell[0]] - lon_t[outside_cell[0]]) )
        this_thd = ( ( full_obj.Thd.isel(lon_t=inside_cell[0],lat_t=inside_cell[1]) 
                      * abs(lon_u_bnd-lon_t[outside_cell[0]])
                      + full_obj.Thd.isel(lon_t=outside_cell[0],lat_t=outside_cell[1]) 
                      * abs(lon_u_bnd-lon_t[inside_cell[0]]) )
                    / abs(lon_t[inside_cell[0]] - lon_t[outside_cell[0]]) )

        # resulting flux at this edge (summed over this water column):
        if inside_west_of_outside:
            # sign of u already in direction of inside to outside
            merid_transp_Pa = (this_pad * this_area * this_u).sum().item()
            merid_transp_Th = (this_thd * this_area * this_u).sum().item()
        else:
            # swap sign because inside to outside vector is 180 degrees opposite to u
            merid_transp_Pa = (this_pad * this_area * -1 * this_u).sum().item()
            merid_transp_Th = (this_thd * this_area * -1 * this_u).sum().item()                
    else:
        raise Exception("Unknown flux_type", flux_type)

    merid_transp_Pa = merid_transp_Pa / 60 * 1e6  # dpm/s to uBq/s
    merid_transp_Th = merid_transp_Th / 60 * 1e6

    return [merid_transp_Pa, merid_transp_Th]  # in uBq/s; positive means from inside_cell to outside_cell


def compute_merid_transport_across_section(full_obj, cell_inside_bnd_Tgrid, 
                                           cell_outside_bnd_Tgrid, flux_type, verbose=False):
    """Computes the meridional advection of Pad and Thd across a given section (straight line or also u-component parts)
    Formula: v * A(dlon-dz surface) * [Pad]  and  u * A(dlat-dz surface) * [Pad]
    Sign of the result: positive corresponds to transport from cells_inside_bnd towards cells_outside_bnd

    Input:
    - xarray DataSet data_fulls of the run; must contain 'masks', 'Pad', 'Thd' & the usual coords
    - array cell_inside_bnd_Tgrid of tuples with python indices (lon_t,lat_t) walking along the boundary/section on the inside.
      Put in cells double if multiple fluxes are involved (i.e. if a u and a v flux cross the section for this cell)
    - array cell_outside_bnd_Tgrid of tuples with python indices (lon_t,lat_t) of where to cell_inside_bnd_Tgrid is pointing
      N.B. function is equivalent under swapping cell_inside_bnd_Tgrid & cell_outside_bnd_Tgrid
    - str array flux_type: fill with 'u' or 'v' in same order
    - if verbose, then print the merid transport contributions per water column; signs also from inside to outside
    
    Output:
    - [merid_transport_Pad, merid_transport_Thd] in uBq/s directed from cells_inside_bnd to cells_outside_bnd

    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    # u/v include advection and diffusion (also GM) of water itself
    # Here we compute meridional advection of Pa, Th with the water. 
    # Thus diffusion of Pa, Th (due to gradient fields of Pa, Th) is not taken into account.
    
    # we have cell-edges on v grid (pure merid. transport) as well as on u grid (zonal transport in/outside our transect)

    from numpy import diff, asarray, cos
    
    lat_t = full_obj.lat_t
    lat_u = full_obj.lat_u
    lon_t = full_obj.lon_t
    lon_u = full_obj.lon_u
    
    ## == A). Prepare concentration ==
    pad_Atl = full_obj.Pad.where(full_obj.masks == 1.0)
    thd_Atl = full_obj.Thd.where(full_obj.masks == 1.0)

    ## == B). Prepare vertical area of grid cells where water flows through ==
    dz = diff(full_obj.z_w) * 1000           # km to m
    R_earth = 6378137                        # m

    # = for u transport = (lat,z) edge:
    dlat = R_earth * deg_to_rad(diff(lat_u))             # delta_lat in m
    area_lat_z = asarray([dz * this_dlat for this_dlat in dlat])  # global (lat_t, z_t)

    # = for v transport = (lon,z) edge:
    dlon_rad = deg_to_rad(diff(lon_u))   # delta_lon in radians
    # find bow length: multiply with cos(theta=lat) because 'horizontal' circumference of sphere is cos(theta) * 2 pi R 
    # BUT for now we omit cos(theta) to reuse this object when walking over multiple latitudes theta
    area_lon_z_dcostheta = asarray([dz * this_dlon * R_earth for this_dlon in dlon_rad]) # global (lon_t, z_t)
    # usage: area_on_lon_z = area_lon_z_dcostheta * np.cos(deg_to_rad(this_lat_u))  # in m2 on v grid

    ## == C). prepare u and v in Atl basin ==
    Atl_mask_lat_u_coord = find_Atl_mask_on_lat_u(full_obj)
    Atl_mask_lon_u_coord = find_Atl_mask_on_lon_u(full_obj)

    # === walk over transect via cell-to-cell fluxes ===
    assert len(cell_inside_bnd_Tgrid)==len(cell_outside_bnd_Tgrid), "cell_inside_bnd_Tgrid and cell_outside_bnd_Tgrid unequal length"
    assert len(cell_inside_bnd_Tgrid)==len(flux_type), "cell_inside_bnd_Tgrid and flux_type unequal length"

    merid_transp_Pa = []  # array walking over transect containing contribution per water column; sum=total
    merid_transp_Th = []
    for n in range(len(cell_inside_bnd_Tgrid)):
        inside_cell = cell_inside_bnd_Tgrid[n]    # contains [lon_index, lat_index]
        outside_cell = cell_outside_bnd_Tgrid[n]  # contains [lon_index, lat_index]
        if flux_type[n] == 'v':
            assert inside_cell[0] == outside_cell[0], "for flux_type v, lon should be constant at n="+str(n)
            assert abs(inside_cell[1]-outside_cell[1]) == 1, "for flux_type v, lat_indices should differ by 1 at n="+str(n)

            # v at this lat,lon water column:
            # need to search for lat_u_bnd because lat on v-grid not always in middle of T-grid
            if lat_t[inside_cell[1]] < lat_t[outside_cell[1]]:
                # inside_cell lies south of outside_cell
                inside_south_of_outside = True
                # v-grid index equals highest T-grid index of neighbour cells e.g. |  T-grid 11  |v-grid 12|  T-grid 12  |
                lat_u_bnd = lat_u[outside_cell[1]].item()
            else:
                # inside_cell lies north of outside_cell
                inside_south_of_outside = False
                lat_u_bnd = lat_u[inside_cell[1]].item()
            this_v = full_obj.v.where(Atl_mask_lat_u_coord).sel(lat_u=lat_u_bnd).isel(lon_t=inside_cell[0])  # m/s

            # area of (lon,z) surface at this lat,lon:
            this_area = area_lon_z_dcostheta[inside_cell[0],:] * cos(deg_to_rad(lat_u_bnd))

            # concentration; convert concs of outside_cell and inside_cell on T grid to v grid via weighted avg:
            this_pad = ( ( pad_Atl.isel(lon_t=inside_cell[0],lat_t=inside_cell[1]) 
                          * abs(lat_u_bnd-lat_t[outside_cell[1]])
                          + pad_Atl.isel(lon_t=outside_cell[0],lat_t=outside_cell[1]) 
                          * abs(lat_u_bnd-lat_t[inside_cell[1]]) ) 
                        / abs(lat_t[inside_cell[1]] - lat_t[outside_cell[1]]) )
            this_thd = ( ( thd_Atl.isel(lon_t=inside_cell[0],lat_t=inside_cell[1]) 
                          * abs(lat_u_bnd-lat_t[outside_cell[1]])
                          + thd_Atl.isel(lon_t=outside_cell[0],lat_t=outside_cell[1]) 
                          * abs(lat_u_bnd-lat_t[inside_cell[1]]) ) 
                        / abs(lat_t[inside_cell[1]] - lat_t[outside_cell[1]]) )

            # resulting flux at this edge (summed over this water column):
            if inside_south_of_outside:
                # sign of v already in direction of inside to outside                
                this_sign = 1.0
            else:
                # swap sign because inside to outside vector is 180 degrees opposite to v                
                this_sign = -1.0                
            # save contribution of this step; sum over z; convert dpm/s to uBq/s
            merid_transp_Pa.append((this_sign * this_pad * this_area * this_v).sum().item() / 60 * 1e6)
            merid_transp_Th.append((this_sign * this_thd * this_area * this_v).sum().item() / 60 * 1e6)
            
        elif flux_type[n] == 'u':
            assert inside_cell[1] == outside_cell[1], "for flux_type u, lat should be constant at n="+str(n)
            assert abs(inside_cell[0]-outside_cell[0]) == 1, "for flux_type u, lon_indices should differ by 1 at n="+str(n)
            
            # u at this lat,lon water column:
            # need to search for lon_u_bnd because lon on u-grid not always in middle of T-grid
            if lon_t[inside_cell[0]] < lon_t[outside_cell[0]]:
                # inside_cell lies west of outside_cell
                inside_west_of_outside = True
                # u-grid index equals highest T-grid index of neighbour cells e.g. |  T-grid 11  |u-grid 12|  T-grid 12  |
                lon_u_bnd = lon_u[outside_cell[0]].item()
            else:
                # inside_cell lies east of outside_cell
                inside_west_of_outside = False
                lon_u_bnd = lon_u[inside_cell[0]].item()
            this_u = full_obj.u.where(Atl_mask_lon_u_coord).sel(lon_u=lon_u_bnd).isel(lat_t=inside_cell[1])  # m/s

            # area of (lat,z) surface at this lat,lon:
            this_area = area_lat_z[inside_cell[1],:]
            
            # concentration; convert concs of outside_cell and inside_cell on T grid to u grid via weighted avg:
            this_pad = ( ( pad_Atl.isel(lon_t=inside_cell[0],lat_t=inside_cell[1]) 
                          * abs(lon_u_bnd-lon_t[outside_cell[0]])
                          + pad_Atl.isel(lon_t=outside_cell[0],lat_t=outside_cell[1]) 
                          * abs(lon_u_bnd-lon_t[inside_cell[0]]) )
                        / abs(lon_t[inside_cell[0]] - lon_t[outside_cell[0]]) )
            this_thd = ( ( thd_Atl.isel(lon_t=inside_cell[0],lat_t=inside_cell[1]) 
                          * abs(lon_u_bnd-lon_t[outside_cell[0]])
                          + thd_Atl.isel(lon_t=outside_cell[0],lat_t=outside_cell[1]) 
                          * abs(lon_u_bnd-lon_t[inside_cell[0]]) )
                        / abs(lon_t[inside_cell[0]] - lon_t[outside_cell[0]]) )
            
            # resulting flux at this edge (summed over this water column):
            if inside_west_of_outside:
                # sign of u already in direction of inside to outside
                this_sign = 1.0
            else:
                # swap sign because inside to outside vector is 180 degrees opposite to u
                this_sign = -1.0                
            # save contribution of this step; sum over z; convert dpm/s to uBq/s
            merid_transp_Pa.append((this_sign * this_pad * this_area * this_u).sum().item() / 60 * 1e6)
            merid_transp_Th.append((this_sign * this_thd * this_area * this_u).sum().item() / 60 * 1e6)
            
        else:
            raise Exception("Unknown flux_type", flux_type)

    if verbose:
        print('Pad transport per water column [uBq/s] (from inside_cells to outside_cells):\n', 
              ["{:.2e}".format(x) for x in merid_transp_Pa])
        print('Thd transport per water column [uBq/s] (from inside_cells to outside_cells):\n', 
              ["{:.2e}".format(x) for x in merid_transp_Th])
                
    # in uBq/s; positive means from cells_inside_bnd to cells_outside_bnd
    return [asarray(merid_transp_Pa).sum(), asarray(merid_transp_Th).sum()]


def remin_curve_val(z, particle):
    """Input: 
    - z in km
    - particle in 'POC', 'CaCO3', 'opal', 'dust' or 'neph'

    Output: 
    - fraction of particle between 0 and 1 still present (not remineralized) at this depth
    
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    from math import exp

    assert particle in ['POC', 'CaCO3', 'opal', 'dust', 'neph'], "Enter valid particle type"
    assert z == 0 or z > 0.075, "z [km] needs to be below euphotic zone (0-75 m)"
    
    if z == 0:
        return 1.0
    
    zcomp = 0.075 # km    
    alpha_POC = 0.83
    rem_scale_ca = 5.066 # km
    rem_scale_op = 10.0 # km
    
    if particle == 'POC':
        return (z/zcomp)**(-alpha_POC)
    if particle == 'CaCO3':
        return exp(-(z-zcomp)/rem_scale_ca)
    if particle == 'opal':
        return exp(-(z-zcomp)/rem_scale_op)
    if particle == 'dust' or particle == 'neph':
        return 1.0


def interpolate_linear_segments(X, Y, nr_data_points):
    """Linearly interpolate a short array of X,Y points into a longer array.
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
   
    from numpy import arange

    assert len(X) == len(Y), "X and Y need to have the same length"
    assert sorted(X) == X, "X should be sorted"
    assert nr_data_points > len(X), "nr_data_points not large enough"

    dx = (X[-1]-X[0])/nr_data_points

    # loop over segments
    X_new = []
    Y_new = []
    for i in range(len(X)-1):
        this_xmin = X[i]
        this_xmax = X[i+1]
        this_y0 = Y[i]
        this_y1 = Y[i+1]
        
        this_X_new = arange(this_xmin, this_xmax, dx)
        this_Y_new = [( (this_xmax-x)*this_y0 + (x-this_xmin)*this_y1 ) / (this_xmax-this_xmin) 
                        for x in this_X_new]

        X_new += list(this_X_new)
        Y_new += list(this_Y_new)

    return[X_new, Y_new]


def runname(nr, ensemble, ID=False):
    """Input: 
    - int nr is the run number from 0 to n
    - ensemble name, which is needed to find runname/filename. e.g. '1TU' or '2TU'
    - [optional] if ID=True, then nr is instead interpreted as run ID from 1000 to n+1000

    Output: 
    - str runname corresponding to it (10 chars)
    
    N.B. technically speaking: with runname we mean parameterfilesname here. We dont mean executablename, 
    since the executable is the same for the entire ensemble.
    
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    import numpy as np
    
    ## only used for ensemble '3TU':
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
                   
    assert ensemble in all_run_nrs, "runname(): ensemble " + str(ensemble) + " not implemented yet."

    if ensemble == 'PAR':
        assert type(nr) == str, "Expecting a string 'nr' for ensemble=PAR"
        assert nr in all_run_nrs[ensemble], "'nr' does not lie in allowed list for ensemble PAR."
        return 'PAR' + nr  # ignoring ID boolean and rest of function

    # same prefix and suffix system for all ensembles so far except PAR
    run_prefix = ensemble
    run_suffix = '_PI'     # phase name: pre-industrial

    if ID:
        assert nr-1000 in all_run_nrs[ensemble], "ID nr" + str(nr) + " does not lie in interval from 1000 to ... To interpret as run nr from 0 to nr_runs, use ID=False."
        return run_prefix + str(nr) + run_suffix
    else:
        assert nr in all_run_nrs[ensemble], "nr should lie in list all_run_nrs[ensemble]"
        return run_prefix + str(nr+1000) + run_suffix


def get_obs_geotraces(fnobs, dissolved_type='BOTTLE', p_type='all', drop_meta_data=False, good_quality=True): 
    """Gets observations as a pandas dataframe and does first processing 
    (columns renamed and shuffled, dropping empty rows and rows with only history comments).
    A multi-index (lat,lon,z) is added to the dataframe.
    Suitable for dissolved or particle-bound observations (selects based on filename fnobs).
    
    Columns are dropped (depending on dissolved_type and drop_meta_data) and 
    rows are selected (depending on dissolved_type resp. p_type and good_quality).
    
    Input:
    - str fnobs is path (use path variable) + filename of observations (a .txt file)
    - str dissolved_type can be 'ALL'/'all' or 'BOTTLE', 'BOAT_PUMP','SUBICE_PUMP','FISH', or 'UWAY'(ship's underway).
      For 'ALL' all these columns are kept. For any specific type, only this type is kept and their rows renamed to 'Pad', 'Thd'.
      NB is ignored if obs are of particle-bound type
    - str p_type can be 'all' or 'combined'. 
      For 'all', all particle-bound types (TP=no-filter, SPT=small particles and LPT=large particles) are kept as columns
      For 'combined', these columns are merged into Pap and Thp and a column Pap_type, Thp_type are added.
      NB is ignored if obs are of dissolved type
    - bool drop_meta_data drops data such as with datetime, ship name, etc. Cruise name stays.
    - bool good_quality; if true then only obs with quality variable 'good' are retained.
    
    Output:
    - obs: original observations with the closest model coords lat_sim, lon_sim, z_sim added.
      Data is also processed and a subset taken, as explained above.
   
    Authors: Sebastian Lienert, Jeemijn Scheen (jeemijn.scheen@nioz.nl)"""
    
    import pandas as pd
    import xarray as xr
    import numpy as np
    
    assert 'IDP' in str(fnobs), "use fnc get_obs_other() instead for non-geotraces files"

    # load observation file
    extension = str(fnobs).split('.')[-1]
    assert extension == 'txt', "File extension of fnobs should be .txt (GEOTRACES ascii files)."
            
    if 'Pad_Thd' in str(fnobs):
        # DISSOLVED OBSERVATIONS     #########################################################################
        obs = pd.read_csv(fnobs, sep='\t', header=39, dtype={'Cruise Aliases':'str'})
        ## NOTES ON WARNING ABOUT COLUMN 21:
        # column 21 contains 'QV:SEADATANET.2': the quality control variable for var 3=Pad.
        # This qc column contains floats as well as occurences of 'Q', which means 'value below limit of quantification'.
        # So it is fine to keep this column mixed.
        print('Column 21 has mixed types as it contains quality control variables, which can be int or string. Fine.')

        obs = obs.rename(columns={'Cruise': 'cruise', 'Station' : 'station', 'Longitude [degrees_east]' : 'lon', 
                                  'Latitude [degrees_north]' : 'lat', 'Bot. Depth [m]' : 'z_bottom', 
                                  'CTDPRS_T_VALUE_SENSOR [dbar]' : 'pressure', 'QV:SEADATANET' : 'pressure_qc',
                                  'DEPTH [m]' : 'z', 'QV:SEADATANET.1' : 'z_qc', 
                                  'STANDARD_DEV' : 'Pad_BOTTLE_err', 'QV:SEADATANET.2':'Pad_BOTTLE_qc', 
                                  'STANDARD_DEV.1':'Pad_FISH_err', 'QV:SEADATANET.3':'Pad_FISH_qc',
                                  'STANDARD_DEV.2':'Pad_UWAY_err', 'QV:SEADATANET.4':'Pad_UWAY_qc',
                                  'STANDARD_DEV.3':'Pad_BOAT_PUMP_err', 'QV:SEADATANET.5':'Pad_BOAT_PUMP_qc',
                                  'STANDARD_DEV.4':'Pad_SUBICE_PUMP_err', 'QV:SEADATANET.6':'Pad_SUBICE_PUMP_qc',
                                  'STANDARD_DEV.5':'Thd_BOTTLE_err', 'QV:SEADATANET.7':'Thd_BOTTLE_qc', 
                                  'STANDARD_DEV.6':'Thd_FISH_err', 'QV:SEADATANET.8':'Thd_FISH_qc',
                                  'STANDARD_DEV.7':'Thd_UWAY_err', 'QV:SEADATANET.9':'Thd_UWAY_qc',
                                  'STANDARD_DEV.8':'Thd_BOAT_PUMP_err', 'QV:SEADATANET.10':'Thd_BOAT_PUMP_qc',
                                  'STANDARD_DEV.9':'Thd_SUBICE_PUMP_err', 'QV:SEADATANET.11':'Thd_SUBICE_PUMP_qc',
                                 })
        d_types = ['BOTTLE','BOAT_PUMP','SUBICE_PUMP','FISH','UWAY']
        obs = obs.rename(columns={'Pa_231_D_CONC_'+t+' [uBq/kg]':'Pad_'+t for t in d_types})
        obs = obs.rename(columns={'Th_230_D_CONC_'+t+' [uBq/kg]':'Thd_'+t for t in d_types})        
        
        if dissolved_type in d_types:
            # rename our dissolved_type of interest to the default: 'Pad' and 'Thd'
            obs = obs.rename(columns={'Pad_'+dissolved_type:'Pad', 'Pad_'+dissolved_type+'_err':'Pad_err',
                                      'Thd_'+dissolved_type:'Thd', 'Thd_'+dissolved_type+'_err':'Thd_err',
                                      'Pad_'+dissolved_type+'_qc':'Pad_qc', 'Thd_'+dissolved_type+'_qc':'Thd_qc'})
            # drop other forms of Pad, Thd we don't use, including their error estimate and quality variables
            all_other_types_all_cols = np.hstack([['Pad_'+t, 'Pad_'+t+'_err', 'Pad_'+t+'_qc',
                                                   'Thd_'+t, 'Thd_'+t+'_err', 'Thd_'+t+'_qc',] 
                                                    for t in d_types if t is not dissolved_type])
            obs.drop(all_other_types_all_cols, axis=1, inplace=True)
            # obs.drop() drops the columns indicated; later on we will drop the rows that became entirely nan
        elif dissolved_type.lower() != 'all':
            # for dissolved_type='all' we do nothing; columns were already renamed appropriately
            raise Exception("dissolved_type, ", dissolved_type, ", should be 'ALL'/'all' or in ", d_types)

    elif 'Pap_Thp' in str(fnobs):
        # PARTICLE-BOUND OBSERVATIONS     #################################################################
        assert p_type.lower() == 'all' or p_type == 'combined', "Set a valid p_type out of ['all', 'combined']."

        obs = pd.read_csv(fnobs, sep='\t', header=35, dtype={'Cruise Aliases':'str'})

        obs = obs.rename(columns={'Cruise': 'cruise', 'Station' : 'station', 'Longitude [degrees_east]' : 'lon', 
                                  'Latitude [degrees_north]' : 'lat', 'Bot. Depth [m]' : 'z_bottom', 
                                  'CTDPRS_T_VALUE_SENSOR [dbar]' : 'pressure', 'QV:SEADATANET' : 'pressure_qc',
                                  'DEPTH [m]' : 'z', 'QV:SEADATANET.1' : 'z_qc',
                                  'STANDARD_DEV' : 'Pap_TP_err', 'QV:SEADATANET.2':'Pap_TP_qc', 
                                  'STANDARD_DEV.1' : 'Pap_SPT_err', 'QV:SEADATANET.3':'Pap_SPT_qc', 
                                  'STANDARD_DEV.2' : 'Pap_LPT_err', 'QV:SEADATANET.4':'Pap_LPT_qc', 
                                  'STANDARD_DEV.3' : 'Thp_TP_err', 'QV:SEADATANET.5':'Thp_TP_qc', 
                                  'STANDARD_DEV.4' : 'Thp_SPT_err', 'QV:SEADATANET.6':'Thp_SPT_qc', 
                                  'STANDARD_DEV.5' : 'Thp_LPT_err', 'QV:SEADATANET.7':'Thp_LPT_qc'
                                 })

        # if p_type == 'all' we are finished and keep all columns TP, SPT and LPT
        if p_type == 'combined':
            # already delete nan rows to increase speed (is repeated at the end)
            obs = take_data_subset_nonnan(obs, verbose=True)

            ## COMBINE 3 PBOUND TYPES WITH A MERGE
            # make 3 copies containing only each of the 3 pbound_types
            obs_TP_only = obs.drop(['Pa_231_SPT_CONC_PUMP [uBq/kg]', 'Pap_SPT_err', 'Pap_SPT_qc', 
                                    'Pa_231_LPT_CONC_PUMP [uBq/kg]', 'Pap_LPT_err', 'Pap_LPT_qc',
                                    'Th_230_SPT_CONC_PUMP [uBq/kg]', 'Thp_SPT_err', 'Thp_SPT_qc', 
                                    'Th_230_LPT_CONC_PUMP [uBq/kg]', 'Thp_LPT_err', 'Thp_LPT_qc']
                                   , axis=1, inplace=False).rename(columns=
                                    {'Pa_231_TP_CONC_PUMP [uBq/kg]' : 'Pap', 
                                     'Pap_TP_err' : 'Pap_err', 'Pap_TP_qc' : 'Pap_qc',
                                     'Th_230_TP_CONC_PUMP [uBq/kg]' : 'Thp', 
                                     'Thp_TP_err' : 'Thp_err', 'Thp_TP_qc' : 'Thp_qc'})
            obs_SPT_only = obs.drop(['Pa_231_TP_CONC_PUMP [uBq/kg]', 'Pap_TP_err', 'Pap_TP_qc', 
                                    'Pa_231_LPT_CONC_PUMP [uBq/kg]', 'Pap_LPT_err', 'Pap_LPT_qc',
                                    'Th_230_TP_CONC_PUMP [uBq/kg]', 'Thp_TP_err', 'Thp_TP_qc', 
                                    'Th_230_LPT_CONC_PUMP [uBq/kg]', 'Thp_LPT_err', 'Thp_LPT_qc']
                                   , axis=1, inplace=False).rename(columns=
                                    {'Pa_231_SPT_CONC_PUMP [uBq/kg]' : 'Pap', 
                                     'Pap_SPT_err' : 'Pap_err', 'Pap_SPT_qc' : 'Pap_qc',
                                     'Th_230_SPT_CONC_PUMP [uBq/kg]' : 'Thp', 
                                     'Thp_SPT_err' : 'Thp_err', 'Thp_SPT_qc' : 'Thp_qc'})
            obs_LPT_only = obs.drop(['Pa_231_TP_CONC_PUMP [uBq/kg]', 'Pap_TP_err', 'Pap_TP_qc', 
                                    'Pa_231_SPT_CONC_PUMP [uBq/kg]', 'Pap_SPT_err', 'Pap_SPT_qc',
                                    'Th_230_TP_CONC_PUMP [uBq/kg]', 'Thp_TP_err', 'Thp_TP_qc', 
                                    'Th_230_SPT_CONC_PUMP [uBq/kg]', 'Thp_SPT_err', 'Thp_SPT_qc']
                                   , axis=1, inplace=False).rename(columns=
                                    {'Pa_231_LPT_CONC_PUMP [uBq/kg]' : 'Pap', 
                                     'Pap_LPT_err' : 'Pap_err', 'Pap_LPT_qc' : 'Pap_qc',
                                     'Th_230_LPT_CONC_PUMP [uBq/kg]' : 'Thp', 
                                     'Thp_LPT_err' : 'Thp_err', 'Thp_LPT_qc' : 'Thp_qc'})
            # add administrational column per type
            obs_TP_only.insert(22, 'Pap_type', ['TP_CONC_PUMP'] * len(obs_TP_only))
            obs_TP_only.insert(26, 'Thp_type', ['TP_CONC_PUMP'] * len(obs_TP_only))
            obs_SPT_only.insert(22, 'Pap_type', ['SPT_CONC_PUMP'] * len(obs_SPT_only))
            obs_SPT_only.insert(26, 'Thp_type', ['SPT_CONC_PUMP'] * len(obs_SPT_only))
            obs_LPT_only.insert(22, 'Pap_type', ['LPT_CONC_PUMP'] * len(obs_LPT_only))
            obs_LPT_only.insert(26, 'Thp_type', ['LPT_CONC_PUMP'] * len(obs_LPT_only))

            # drop newly made nan rows
            obs_TP_only = obs_TP_only[np.isnan(obs_TP_only['Thp']) == False]
            obs_SPT_only = obs_SPT_only[np.isnan(obs_SPT_only['Thp']) == False]
            obs_LPT_only = obs_LPT_only[np.isnan(obs_LPT_only['Thp']) == False]

            # merge all 3 (including the administrational columns Pap_type and Thp_type)
            obs = obs_TP_only.merge(obs_SPT_only, how='outer').merge(obs_LPT_only, how='outer')

    ######################################################################################    
    # FROM HERE ON CODE WORKS FOR BOTH DISSOLVED AND PARTICLE-BOUND OBSERVATIONS
    if drop_meta_data:
        obs.drop(["Type", "yyyy-mm-ddThh:mm:ss.sss", "Operator's Cruise Name", 'Ship Name', 'Period',
                  'Chief Scientist', 'GEOTRACES Scientist', 'Cruise Aliases',
                  'Cruise Information Link', 'BODC Cruise Number'], axis=1, inplace=True)
        # keeping 'QV:ODV:SAMPLE'

    ## delete rows with history comments (contain no data)
    bad_rows = list(obs[obs.cruise.str.startswith('//<History>')].index)
    obs.drop(index=bad_rows, inplace=True)

    ## set multi-index
    obs.set_index(['lon', 'lat', 'z'], inplace=True)

    # rearrange column order via position integers to be a bit more logical
    # order up to now: 
    # cruise, station, lon, lat, z_bottom, pressure, pressure_qc, z, z_qc, ...
    old_len = len(obs.columns)
    column_order = [5,2,0,1,3,4] + list(range(6,len(obs.columns)))
    obs = obs[obs.columns[column_order]]
    assert len(obs.columns) == old_len, "ERROR: some columns disappeared"
    obs.sort_index(inplace=True) # needed to be able to use .loc or slice in multi-index
    
    if good_quality:
        obs = restrict_obs_to_good_quality(obs)

    ## The number of data points seems misleadingly large, because not at all stations or all dephts Pad/Thd is measured
    # However at all stations and depth, the columns z_bottom, pressure etc are filled and sometimes var_qc.
    # => data set contains rows that are all NaN in Pad and Thd (in all 5 dissolved_types) and 
    #    a .dropna(how='all') doesnt help is not sufficient but we can use this:
    obs = take_data_subset_nonnan(obs, verbose=True)
    # This drastically reduces the number of data rows (numbers are printed if verbose=True)
    # Note: only the dissolved_type under consideration is kept and even more rows are dropped        
        
    return obs


def get_obs_other(fnobs): 
    """Gets observations other than geotraces as a pandas dataframe.
    A multi-index (lat,lon,z) is added to the dataframe.
    Suitable for dissolved observations only.
        
    Input:
    - str fnobs is path (use path variable) + filename of observations (a .txt file)
    
    Output:
    - obs: original observations with the closest model coords lat_sim, lon_sim, z_sim added.
   
    Authors: Sebastian Lienert, Jeemijn Scheen (jeemijn.scheen@nioz.nl)"""
    
    import pandas as pd
    import xarray as xr
    import numpy as np
    
    assert 'IDP' not in str(fnobs), 'this function is not for geotraces'

    # load observation file for dissolved
    extension = str(fnobs).split('.')[-1]
    assert extension == 'csv', "File extension of fnobs should be .csv because this function is for files other than GEOTRACES."
    
    if 'Pad_Thd' in str(fnobs):
        obs = pd.read_csv(fnobs, sep=',', header=0)
    elif 'Pap_Thp' in str(fnobs):
        raise Exception('this function is not for geotraces; p-bound not implemented/needed yet')
    else:
        raise Exception("file name should contain 'Pad_Thd'.")
    
    ## delete empty rows (if any)
    obs = take_data_subset_nonnan(obs, verbose=True)
    
    ## change longitude from [-180,180] to [0,360]
    obs['lon'] = convert_minus_180_plus_180_lon_to_0_360_lon(obs['lon'].values)

    ## set multi-index
    obs.set_index(['lon', 'lat', 'z'], inplace=True)

    return obs  



def take_data_subset_nonnan(obs, verbose=True):
    """Can take a minute.
    Input:
    - obs pandas dataframe containing Pad and Thd or Pap and Thp variables. 
      The Pad and Thd may still be of the form(s) Pad_BOTTLE etc.

    Output:
    - obs_subset pandas_dataframe with only rows that have at least 1 non-Nan values in the (max. 10) 
      columns with Pad, Thd concentrations
    
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""

    # taking only concentrations themselves; not the _qc or _err (usually _qc is 9 for missing values so is always present)
    # for p-bound: concentration column names are recognized because still ending in ']'
    col_nrs_with_data = [(a or b or c or d or e or f or g or h or i or j or k or l or m or n or o) 
                         for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o in zip(obs.columns == 'Pad', obs.columns == 'Thd', 
                                              obs.columns == 'Pad_BOTTLE', obs.columns == 'Thd_BOTTLE', 
                                              obs.columns == 'Pad_BOAT_PUMP', obs.columns == 'Thd_BOAT_PUMP', 
                                              obs.columns == 'Pad_SUBICE_PUMP', obs.columns == 'Thd_SUBICE_PUMP', 
                                              obs.columns == 'Pad_FISH', obs.columns == 'Thd_FISH', 
                                              obs.columns == 'Pad_UWAY', obs.columns == 'Thd_UWAY', 
                                              obs.columns.str.endswith(']'), 
                                              obs.columns == 'Pap', obs.columns == 'Thp')]
    def row_has_data(row):
        """Returns True if at least 1 out of all 10 Pad, Thd variables has data in this row (resp. out of all 6 Pap, Thp variables)."""
        return not row.loc[(col_nrs_with_data)].isnull().values.all()

    obs_subset = obs[obs.apply(row_has_data, axis=1)]
    if verbose:
        print('Reduced dataset from ', len(obs),' rows to ', len(obs_subset), ' rows after deleting nans.')
    return obs_subset


def restrict_obs_to_good_quality(obs):
    """Input: obs dataframe 
    Output: restricted_obs dataframe, where values of variables are changed to NaN 
    whenever their respective var_qc is not 'good quality' (flag = 1).
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    import numpy as np

    obs_restricted = obs.copy(deep=True)

    vars_with_qc = [convert_varqc_to_var(col) for col in obs_restricted.columns if col[-3:]=='_qc']
    # we ignore z_qc and pressure_qc, which are 0='no quality control' everywhere anyway
    vars_with_qc = set(vars_with_qc)-set(['z','pressure'])

    for var in vars_with_qc:
        var_column_not_good_quality = [x != 1.0 and x != '1.0' and x != '1' 
                                       for x in obs[convert_var_to_varqc(var)].values]
        # literally change value of var to NaN if var_qc is not 1=good quality
        obs_restricted.loc[var_column_not_good_quality, var] = np.nan

    return obs_restricted


def convert_varqc_to_var(var_qc_str):
    """To properly convert column names from column name of var_qc to column name of var. 
    This got a bit complicated because of my usage of abbreviations like Pad etc (but they are useful so I keep them).
    Works for both dissolved and particle-bound observations.
    Note: the opposite function convert_var_to_varqc() is also available. 
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    if var_qc_str[-3:] != '_qc':
        raise Exception("convert_varqc_to_var(): need a var_qc string that ends in '_qc'. Now we got var_qc_str=", var_qc_str)
    
    # abbreviated var names
    if var_qc_str in ['Pad_qc', 'Thd_qc', 'Pap_qc', 'Thp_qc']:
        var_str = var_qc_str[:-3]  # just omit '_qc'
    # dissolved var names:
    elif var_qc_str[0:4] == 'Pad_':
        var_str = 'Pa_231_D_' + var_qc_str[4:-3] + ' [uBq/kg]'
    elif var_qc_str[0:4] == 'Thd_':
        var_str = 'Th_230_D_' + var_qc_str[4:-3] + ' [uBq/kg]'
    # particulate var names (half abbreviated):
    elif var_qc_str[0:4] == 'Pap_':
        var_str = 'Pa_231_' + var_qc_str[4:-3] + '_CONC_PUMP [uBq/kg]'
    elif var_qc_str[0:4] == 'Thp_':
        var_str = 'Th_230_' + var_qc_str[4:-3] + '_CONC_PUMP [uBq/kg]'
    else:
        var_str = var_qc_str[:-3]  # just omit '_qc' ; will maybe not always work
    
    return var_str


def convert_var_to_varqc(var_str):
    """Opposite conversion to function convert_varqc_to_var().
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    if var_str[-3:] == '_qc':
        raise Exception("convert_var_to_varqc(): need a var string that does NOT end in '_qc'. Now we got var_str=", var_str)

    # abbreviated var names
    if var_str in ['Pad', 'Thd', 'Pap', 'Thp']:
        var_qc_str = var_str + '_qc'
    # dissolved var names; are as 'Pa_231_D_XXXXX [uBq/kg]' for some XXXXX
    elif var_str[0:8] == 'Pa_231_D':
        var_qc_str = 'Pad_' + var_str[9:-9] + '_qc'
    elif var_str[0:8] == 'Th_230_D':
        var_qc_str = 'Thd_' + var_str[9:-9] + '_qc'
    # particulate var names; are as 'Pa_231_XXX_CONC_PUMP [uBq/kg]' for some XXX
    elif var_str[0:7] == 'Pa_231_':
        var_qc_str = 'Pap_' + var_str[7:-19] + '_qc'
    elif var_str[0:7] == 'Th_230_':
        var_qc_str = 'Thp_' + var_str[7:-19] + '_qc'
    else:
        var_qc_str = var_str + '_qc'  # will maybe not always work
    
    return var_qc_str


def obs_to_model_grid(obs, fnctrl, weight_avg_by_uncertainty=True):
    """Transfers observation dataframe to model grid. In model grid cells with multiple observations, these 
    observations are averaged (if desired: with 1/err as weight) and the uncertainty is propagated.
    
    Input:
    - obs pandas dataframe with multi-index (lat, lon, z) = location of the measurements
    - str fnctrl is path (use path variable) + filename of ctrl model run; used only for model grid definition
    - weight_avg_by_uncertainty determines whether the average of observations is weighted by the observational uncertainties.
         if True, then 'var_err' columns are needed in obs for vars Pad,Thd OR Pap,Thp.

    Outputs 3 dataframes:
    - obs pandas dataframe as original input, now with columns lat_sim, lon_sim, z_sim added [for testing purposes]
      => contains all original columns
    - obs_ave pandas dataframe (index: lat_sim, lon_sim, z_sim) after average if >1 obs per model grid cell
      => only columns of var names + their errors are kept e.g. Pad, Pad_err, Thd, Thd_err; and cruise
    - obs_ave_num (index: lat_sim, lon_sim, z_sim) is counter of the averaging that was done above
      => only columns of var names + their errors are kept e.g. Pad, Pad_err, Thd, Thd_err; and cruise
    
    Author: Sebastian Lienert
    Edited by Jeemijn Scheen (jeemijn.scheen@nioz.nl)"""
    
    import pandas as pd
    import xarray as xr
    import numpy as np

    if 'Pap' in obs.columns and 'path_ratio_p' in obs.columns:
        print("obs_to_model_grid(): found 'Pap' as well as 'path_ratio_p' in obs.columns; continuing with Pap.")
    if ('Pad' in obs.columns and 'Pap' in obs.columns) or ('Pad' in obs.columns and 'path_ratio_p' in obs.columns):
        raise Exception('obs_to_model_grid(): not able to infer whether obs has dissolved or particle-bound forms')
    elif 'Pad' in obs.columns:
        var_names = ['Pad','Thd'] # seawater
    elif 'Pap' in obs.columns:
        var_names = ['Pap','Thp'] # seawater
    elif 'path_ratio_p' in obs.columns:
        var_names = ['path_ratio_p'] # sediment or seawater
    else:
        raise Exception('obs_to_model_grid(): not able to infer whether obs has dissolved or particle-bound forms')
    
    ###################################################################################################
    ## PART 1). transfer observations to model grid

    # if axis 'lat_sim' already present, then this fnc was already called once on the object and we skip part 1). entirely
    if 'lat_sim' not in obs.columns:
        # look for corresponding lat/lon/z coordinates of obs in control file
        ctrl = xr.open_dataset(str(fnctrl) + '.0001765_full_ave.nc', decode_times=False)

        # generate extra columns with nearest lat,lon,z of the model grid
        lon_obs = np.vstack(obs.index)[:,0]
        lat_obs = np.vstack(obs.index)[:,1]
        z_obs = np.vstack(obs.index)[:,2]

        print('Starting to match obs to nearest model coords... ')
        # model lon (ctrl.lon_t) goes from 100 to 460; convert here first from 0 to 360 (as obs are)
        list_ctrl_lon_t_0_to_360 = [lon if lon < 360.0 else (lon - 360.0) for lon in ctrl.lon_t]
        list_ctrl_lon_t_0_to_360.sort()
        # convert to DataArray in order to use .sel(method='nearest')
        ctrl_lon_t_0_to_360 = xr.DataArray(
            data=list_ctrl_lon_t_0_to_360,
            coords={'lon_t': list_ctrl_lon_t_0_to_360},
            dims=['lon_t'],
            attrs={'long_name': "Longitude (T grid) from 0 to 360", 'units': "degrees_east"})
        # matching obs to model coords
        lon_sim_0_to_360 = [
            float(ctrl_lon_t_0_to_360.sel(lon_t=this_lon, method='nearest'))
            for this_lon in lon_obs
        ]
        # we dont need to apply boundary conditions explicitly because the model grid is regular around 360 degrees
        # (model grid points at 355.0 and 5.0 have the boundary of 0=360 exactly in their middle)

        lat_sim = [
            float(ctrl.lat_t.sel(lat_t=this_lat, method='nearest'))
            for this_lat in lat_obs
        ]

        z_sim = [
            float(ctrl.z_t.sel(z_t=this_z, method='nearest'))
            for this_z in z_obs
        ]
        print('... finished matching.')

        obs.insert(2, 'lon_sim_0_to_360', lon_sim_0_to_360) # add new column after 2nd column 

        # now convert our intuitive lon (0 to 360) to model coord lon (100 to 460)
        # for clarity we add both lon columns
        lon_sim_100_to_460 = [lon if lon > 100.0 else (lon + 360.0) for lon in lon_sim_0_to_360]
        # do not sort because order reflects order of obs!
        obs.insert(3, 'lon_sim_100_to_460', lon_sim_100_to_460)

        obs.insert(4, 'lat_sim', lat_sim)
        obs.insert(5, 'z_sim', z_sim)

    ###################################################################################################
    # PART 2). average the observations that lie in the same model grid cell:
    
    # old nice solution: worked but only if non-weighted average AND without error propagation
    # obs_ave = obs.groupby(['lat_sim', 'lon_sim_100_to_460', 'z_sim']).mean()
    # obs_ave_num = obs.groupby(['lat_sim', 'lon_sim_100_to_460', 'z_sim']).count()
    
    for this_var in var_names:
        assert this_var+'_err' in obs.columns, "variable " + this_var + " has no error column."

    if weight_avg_by_uncertainty:
        print('Performing a weighted average (weights=1/error) in cells with >1 obs, with error propagation.')
    else:
        print('Performing a non-weighted average in cells with >1 obs, with error propagation.')
        
    # change multi-index from obs coords to model coords
    obs_index_model = obs.reset_index().set_index(['lat_sim','lon_sim_100_to_460','z_sim'], inplace=False)
    obs_index_model.sort_index(inplace=True)
    if 'cruise' in obs.columns:   # then obs is seawater data
        relevant_cols_wo_info = var_names + [var+'_err' for var in var_names]
        relevant_cols_with_info = var_names + [var+'_err' for var in var_names] + ['cruise']
    elif 'path_ratio_p' in obs.columns:  # then obs is sediment data (since no cruise column)
        relevant_cols_wo_info = ['path_ratio_p', 'path_ratio_p_err']
        relevant_cols_with_info = ['#','path_ratio_p', 'path_ratio_p_err', 'region', 'age_ka']
    else:
        raise Exception("Something wrong; can't determine relevant_cols_wo_info and relevant_cols_with_info.")
    # multi-index in obs_index_model not unique yet: some indices have multiple rows (where multiple obs)
    index_unique = obs_index_model.index.unique()
       
    # initialize obs_ave with less columns than obs, because not meaningful to average metadata columns
    obs_ave = pd.DataFrame(index=index_unique, columns=relevant_cols_with_info)
    obs_ave_num = pd.DataFrame(index=index_unique, dtype='int', columns=relevant_cols_wo_info)
    # loop over index = naive implementation
    # loop over unique version of index, such that values are only adjusted once
    for this_index in index_unique:
        if len(obs_index_model.loc[this_index, :]) == 0:
            raise Exception("No row present at this index. Something went wrong.")
        elif len(obs_index_model.loc[this_index, :]) == 1:
            # only 1 observation here: just copy over
            for var in var_names:
                obs_ave_num.loc[this_index, (var)] = 1
                obs_ave.loc[this_index, (var)] = obs_index_model.loc[this_index, (var)].values[0]
                
                obs_ave_num.loc[this_index, (var+'_err')] = 1
                obs_ave.loc[this_index, (var+'_err')] = obs_index_model.loc[this_index, (var+'_err')].values[0]
        else:
            # multiple observations/rows present at this index
            # obs_sed: ATTENTION if they have different ages at this same index, we only keep 'Hol_avg' etc
            for var in var_names:
                these_obs = obs_index_model.loc[this_index, var].values
                obs_ave_num.loc[this_index, (var)] = len(these_obs)
                
                obs_ave_num.loc[this_index, (var+'_err')] = len(these_obs)
                these_obs_err = obs_index_model.loc[this_index, var+'_err'].values

                # boolean 'weight_avg_by_uncertainty' determines here whether weighted or normal average:
                obs_ave.loc[this_index, (var, var+'_err')] = avg_obs_and_error_of_1_grid_cell(
                    few_obs=these_obs, few_obs_err=these_obs_err, 
                    weight_avg_by_uncertainty=weight_avg_by_uncertainty)
            
        # copy over additional columns with important information
        for extra_var in ['cruise', '#', 'region', 'age_ka']:
            # cruise is for seawater obs (d and p); other extra_vars for sediment obs
            if extra_var in obs.columns:
                # taking the 1st value: ASSUMES all values are identical for obs in same grid cell
                obs_ave.loc[this_index, extra_var] = obs_index_model.loc[this_index, extra_var].values[0]
                
        if 'z' in obs_ave.columns:
            # it is confusing to have a obs. z in the obs_ave. From which obs is this the z?
            obs_ave.drop('z', axis=1, inplace=True)

    for var in var_names:            
        assert len(obs_index_model) == obs_ave_num[var].sum(), "Total of counts is not equal to original for var "+var
    print('Function obs_to_model_grid(): done.')

    return [obs, obs_ave, obs_ave_num]



def avg_obs_and_error_of_1_grid_cell(few_obs, few_obs_err, weight_avg_by_uncertainty=True):
    """ Averages observations of a certain variable that lie in the same grid cell. 
    Also the corresponding propagated error is computed.
    Only give in the couple of values that lie within the same fixed grid cell.
    
    Input:
    - float arr few_obs is a few occurrences of a certain var, which all have the same lat_sim, lon_sim and z_sim 
    - float arr few_obs_err are the corresponding few uncertainties
    - bool weight_avg_by_uncertainty determines whether the average of observations is weighted by the observational uncertainties.
    
    Output:
    - float obs_avg: an average is taken (weighted or not)
    - float obs_err_avg: resulting error computed via propagation of 
      measurement uncertainty (also taking into account weights if applicable).
    
    ASSUMPTION: observational errors (within 1 grid cell) are random and independent
    
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    import numpy as np
    
    # FORMULAS FOR ERROR PROPAGATION IN (NON-)WEIGHTED MEAN:
    # based on 'an introduction to Error Analysis, John R. Taylor, chapter 3' I found,
    # for variables (observations) x_k for k=1,...,N and independent and random errors e_k:
    
    # NORMAL AVERAGE:
    # mean = 1/N * sum_k x_k
    # error_mean = 1/N sqrt( sum_k (e_k)^2 )
    
    # WEIGHTED AVERAGE with weights w_k:
    # mean = ( sum_k w_k * x_k ) / ( sum_k w_k )
    # error_mean = sqrt( sum_k (w_k)^2 * (e_k)^2 ) / ( sum_k w_k )
    # We use w_k=1/e_k so to avoid confusion, we don't write it out but keep w_k in a separate var
    # UPDATE: because I defined w_k = 1/e_k, by design always sum_k (w_k)^2 * (e_k)^2 ) = N 
    # so we can simplify
    # error_mean = sqrt(N) / ( sum_k w_k )
    
    # This formula was confirmed here: 
    # https://math.stackexchange.com/questions/123276/error-propagation-on-weighted-mean
    
    import numpy as np
    
    assert len(few_obs) == len(few_obs_err), "few_obs and few_obs_err should have same length"
    assert len(few_obs) > 1, "few_obs needs len>1"
           
    few_obs = np.asarray(few_obs)
    few_obs_err = np.asarray(few_obs_err)

    if weight_avg_by_uncertainty:
        # take as weights 1/err
        few_weights = 1/few_obs_err  # if changed: remove simplification below!
        
        # take weighted average (formulas above)
        obs_avg = np.average(few_obs, weights=few_weights)
        # obs_err_avg = np.sqrt( (few_weights**2 * few_obs_err**2).sum() ) / few_weights.sum()
        # UPDATE: we can simplify this (see comment above) to:
        obs_err_avg = np.sqrt(len(few_obs_err)) / few_weights.sum()
    else:
        # take normal average (formulas above)
        obs_avg = np.average(few_obs)
        obs_err_avg = 1/len(few_obs_err) * np.sqrt( (few_obs_err**2).sum() )
        
    return [obs_avg, obs_err_avg]


def subset_of_obs(obs, fnctrl, cruises='all', wo_surface=False):
    """ Take a subset of the observations, selecting certain cruises (also implies basin) and with or without surface ocean.
    Input: 
    - obs pandas dataframe
    - cruises can be a str array of cruises names or 'all' (keeping all cruises that are inside the given 'obs')
    - bool wo_surface indicates whether to ignore the surface ocean. The limit of surface ocean is hardcoded below.
    Output:
    - [obs, obs_ave, obs_ave_num], where obs is now the subset of obs and obs_ave(num) corresponding. 
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    surface_layer = 500  # meter
    
    # restrict to cruises
    if type(cruises)==str:
        assert cruises.lower() == 'all', "Unknown cruises object. Should equal 'all' or array of cruise names."
        # for cruises='all': do nothing, keeping all cruises
    elif (len(cruises)==1 and cruises[0].lower() != 'all') or len(cruises)>1:
        # for cruises=['all']: do nothing, keeping all cruises
        obs = obs.where(obs.cruise.isin(cruises)).dropna(how='all')
    
    # restrict to surface ocean
    if wo_surface:
        obs = obs.loc[(slice(None), slice(None), slice(surface_layer,6000)),:]

    return obs_to_model_grid(obs, fnctrl)   # returns [obs, obs_ave, obs_ave_num]


def prepare_stations_z_plot(this_obs):
    """Input: 
    - this_obs pandas dataframe with the observations, which can be the entire obs or a sliced part with only certain observations.
    Output:
    - list str stations with station names
    - list stations_lon with longitude of stations in order of stations list
    - list stations_lat with latitude of stations in order of stations list
    - list stations_z_bottom with bottom depth [m] in order of stations list
    - list of arrays stations_z with per station an array of z values measured at this station
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    import numpy as np

    ## prepare plotting objects
    cruises = np.unique(this_obs.cruise)

    stations = []      # will list all stations (can have double names for multiple cruises)
    stations_lon = []  # will list lons in order of arr stations
    stations_lat = [] 
    stations_z_bottom = []
    stations_z = []    # array of arrays: will list per station an array of z values appearing
    # Note that station names are not unique! E.g. GN01 has a station '1' but GA02 also
    for this_cruise in cruises:
        this_cruise_obs = this_obs[this_obs.cruise == this_cruise].dropna(how='all')

        cruise_stations = np.unique(this_cruise_obs.station)  # stations applicable for this cruise
        for this_station in cruise_stations:
            station_all_coords = this_cruise_obs[this_cruise_obs.station == this_station].index

            ## prepare left panel: get lats and lon of stations 
            stations.append(this_station)
            # 1st row takes first station (all equivalent for lon,lat); index 0/1 takes lon/lat
            stations_lon.append(station_all_coords[0][0])
            stations_lat.append(station_all_coords[0][1])
            
            # # THIS CHECK + COUNTS OF THE TOTAL SHOWED THAT THE ASSUMPTION DIRECTLY ABOVE IS INVALID (ONLY) IN THE CASE OF 
            # # GA03 FOR STATIONS 1, 3, 10 AND 12. NAMELY FOR THESE STATIONS, 2 LAT-LON PAIRS EXIST INSTEAD OF 1. SO THEY ARE IN FACT 2 STATIONS.
            # # PLOT BY E.G. this_cruise_obs[this_cruise_obs.station == 'Station 1']
            # this_lon = station_all_coords[0][0]
            # this_lat = station_all_coords[0][1]
            # for z_i, meas in enumerate(station_all_coords):
            #     assert meas[0] == this_lon and meas[1] == this_lat, "station " + this_station + " not consistent at coords" + str(meas) + "z_i" + str(z_i)

            stations_z_bottom.append(this_cruise_obs[this_cruise_obs.station == this_station].z_bottom.values[0])
            # [0] takes row 1; all rows equivalent for z_bottom
            
            ## prepare right panel: get depths per station
            # keep all rows of station and get the 3rd index column (=z)
            stations_z.append(np.vstack(station_all_coords)[:,2])

    assert len(stations) == len(stations_lon), "smt is wrong in lon"
    assert len(stations) == len(stations_lat), "smt is wrong in lat"
    assert len(stations) == len(stations_z_bottom), "smt is wrong in z_bottom"
    assert len(stations) == len(stations_z), "smt is wrong in z"
    
    return [stations, stations_lon, stations_lat, stations_z_bottom, stations_z]


def get_spinup_yr_str(spinup_yr, run_nr=None):
    """Convert int or int array spinup_yr to string spinup_yr as it occurs in filenames.
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    from numpy import ndarray
    
    if isinstance(spinup_yr, (list, tuple, ndarray)):  # test if spinup_yr is a list/tuple/array
        assert run_nr > -2, "run_nr must be given because spinup_yr is a list"
        spinup_yr = spinup_yr[run_nr]

    if spinup_yr == 0:      # will give problems with cases of spinup 0 C.E. to 999 C.E.
        spinup_yr_str = '0000'

    return str(spinup_yr)


def load_var_multiple_runs(variables, file_type, folder, runs, spinup_yr=1765):
    """ Loads 1 or more specific variables (to save memory) from multiple runs
    
    Input: 
    - str or str arr variables is either 1 variable name 'var1' or multiple ['var1','var2'] etc
    - str file_type in ['full_ave', 'full_inst', 'timeseries_ave', 'timeseries_inst'] is the file in which this var is located
    - folder must be a pathlib.Path object
    - str array runs contains runnames in this folder
    - spinup_yr [optional; if other than 1765] is an int if all simulations have equal spinup; otherwise int array
      (needed because it is part of the output file names)
    
    Output:
    - dictionary with keys: runnames; values: xarray datasets of the variable (e.g. in (lat,lon,z,time) dimensions)
    
    Assumptions:
    - assuming that all runs have the same list of output variables (i.e. ran with the same diagnosis flags)

    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    from xarray import open_dataset
    from numpy import ndarray
    import netCDF4
    
    assert file_type in ['full_ave', 'full_inst', 'timeseries_ave', 'timeseries_inst'], "ERROR: Invalid file_type"
    
    ## determine variables to drop later (all variables except the desired one(s))
    # ASSUMPTION HERE: all runs have the same list of output variables (i.e. ran with the same diagnosis flags)
    # this assumption can be avoided if this step is moved to inside the for loop; however that will be slower
    file_1 = runs[0] + '.000' + get_spinup_yr_str(spinup_yr, run_nr=0)   # load first run to see which variables are inside
    ds_test = netCDF4.Dataset(folder / (file_1 + '_' + file_type + '.nc'))
    all_vars = list(ds_test.variables.keys())
    vars_to_drop = [var for var in all_vars if var not in variables]
    vars_keep = ['time','z_t','z_w','lat_t','lat_u','lat_rho',
                 'lon_t','lon_u','lon_rho','yearstep_oc','yearstep_atm','month']  # other vars to such as coords or masks
    vars_to_drop = [var for var in all_vars if var not in variables+vars_keep]
    
    var_datasets = {}           # empty dict for output xarray DataSets
    for nr, run_name in enumerate(runs):
        ## create dataset
        # using / because folder is a pathlib.Path object (works on every operating system)
        file = run_name + '.000' + get_spinup_yr_str(spinup_yr, run_nr=nr)
        var_datasets[run_name] = open_dataset(folder / (file + '_' + file_type + '.nc'), decode_times=False, drop_variables=vars_to_drop)
        # change years to simulation years
        var_datasets[run_name].coords['time'] = var_datasets[run_name]['time'] - spinup_yr

# OLD PART USING CHUNKS / DUSK ARRAYS; USEFUL IF STILL LOW ON MEMORY
#         # create data_fulls_inst using dask arrays (chunks) because of large file sizes
#             data_fulls_inst[runname] = open_dataset(folder / (file + '_full_inst.nc'), 
#                                                     decode_times=False, chunks={'yearstep_oc': 20})
#             data_fulls_inst[runname].coords['time'] = data_fulls_inst[runname]['time'] - subtract_yrs

    return var_datasets


def get_sim(memberid, path, obs_ave, ensemble, convert_unit_to_obs=True, no_res_entire=False):
    """Get simulation Pad,Thd resp. Pap,Thp output of a member run, for model grid cells with obs present. 
    ASSUMPTION: takes last time step if multiple time steps present
        
    Spinup year is hardcoded here as 1765 (PI).
    Takes a few seconds.
    
    For PAR ensemble: input just runname ending string (as in all_run_nrs) as memberid; quickfix
    Hack in general for other runnames: use memberid=runname (entirely) and ensemble='FREE'
    
    Input:
    - int memberid = run_ID of the run under consideration
    - Path object path to folder with simulation results
    - obs_ave is needed. The data itself is not used, only obs_ave.index and the column names
    - ensemble name, which is needed to find runname/filename. e.g. '1TU' or '2TU'
    - if convert_unit_to_obs, the model unit of Pad and Thd is converted from dpm/m3 to uBq/kg (as used in obs).
    - if no_res_entire then only res_table is outputted (for performance).
    
    Output:
    For Pad, Thd columns present in obs_ave: Pad and Thd values of model output of memberid. 
    For Pap, Thp columns present in obs_ave: Pap and Thp values of model output of memberid. 
    For path_ratio_p column present in obs_ave: Pap and Thp values of model output of memberid. 
    This output is provided in 2 ways:
    - res_table is a pandas dataframe (table) with the model result of Pad,Thd resp. Pap,Thp only in the grid cells where 
      obs are present. Unit depends on convert_unit_to_obs (see above)
    - res_entire is an xarray dataarray with the entire model result (more variables; all grid cells)
      Unit depends on convert_unit_to_obs (see above)
    
    Author: Sebastian Lienert
    Edited by Jeemijn Scheen (jeemijn.scheen@nioz.nl)"""
    
    import pandas as pd
    import xarray as xr
    
    if 'Pap' in obs_ave.columns and 'path_ratio_p' in obs_ave.columns:
        print("get_sim(): found 'Pap' as well as 'path_ratio_p' in obs_ave.columns; continuing with Pap.")
    if ('Pad' in obs_ave.columns and 'Pap' in obs_ave.columns) or ('Pad' in obs_ave.columns and 'path_ratio_p' in obs_ave.columns):
        raise Exception('get_sim(): not able to infer whether obs_ave has dissolved or particle-bound forms')
    elif 'Pad' in obs_ave.columns:
        form = 'd'
    elif 'Pap' in obs_ave.columns or 'path_ratio_p' in obs_ave.columns:
        form = 'p'
    else:
        raise Exception('get_sim(): not able to infer whether obs_ave has dissolved or particle-bound forms')
        
    obs_index = obs_ave.index
    
    if ensemble == 'FREE':
        fnmember = path / (memberid + '.0001765_full_ave.nc')
    else:
        fnmember = path / (runname(memberid, ID=True, ensemble=ensemble) + '.0001765_full_ave.nc')
    res_entire = xr.open_dataset(fnmember, decode_times=False)
    # we keep all vars;
    # vars of interest have the form:
    # float Pad(time, lat_t, lon_t, z_t)
    # they can be assessed via res_entire.Pad
    
    # change years to simulation years
    res_entire.coords['time'] = res_entire.coords['time'] - 1765

    # could apply a time constraint => not needed since time only has 1 value
    # res_entire = res_entire.sel(time=slice(4000,5000)).mean(dim='time')
    
    if convert_unit_to_obs:
        rho = res_entire.rho_SI.isel(time=-1) # rho is available in model output itself
        if form == 'd':
            res_entire['Pad'] = xr.Variable(data=model_to_sw_unit(val=res_entire.Pad, rho_model=rho),
                                            dims=('time', 'z_t', 'lat_t', 'lon_t'),
                                            attrs={'long_name' : 'dissolved Protactinium',
                                                  'units' : 'uBq/kg', 
                                                  'time_rep' : 'averaged'})
            res_entire['Thd'] = xr.Variable(data=model_to_sw_unit(val=res_entire.Thd, rho_model=rho),
                                            dims=('time', 'z_t', 'lat_t', 'lon_t'),
                                            attrs={'long_name' : 'dissolved Thorium',
                                                  'units' : 'uBq/kg', 
                                                  'time_rep' : 'averaged'})
        if form == 'p':
            res_entire['Pap'] = xr.Variable(data=model_to_sw_unit(val=res_entire.Pap, rho_model=rho),
                                            dims=('time', 'z_t', 'lat_t', 'lon_t'),
                                            attrs={'long_name' : 'particle-bound Protactinium',
                                                  'units' : 'uBq/kg', 
                                                  'time_rep' : 'averaged'})
            res_entire['Thp'] = xr.Variable(data=model_to_sw_unit(val=res_entire.Thp, rho_model=rho),
                                            dims=('time', 'z_t', 'lat_t', 'lon_t'),
                                            attrs={'long_name' : 'particle-bound Thorium',
                                                  'units' : 'uBq/kg', 
                                                  'time_rep' : 'averaged'})

    # Panda DataFrame with obs_ave index
    if form == 'd':
        res_table = pd.DataFrame(index=obs_index, dtype='float64', columns=['Pad', 'Thd'])
        for (this_lat, this_lon, this_z) in obs_index:
            res_table.loc[(this_lat, this_lon, this_z),('Pad','Thd',)] = [float(
                res_entire.Pad.sel(lat_t=this_lat, lon_t=this_lon, z_t=this_z).isel(time=-1)),
                                                                         float(
                res_entire.Thd.sel(lat_t=this_lat, lon_t=this_lon, z_t=this_z).isel(time=-1))]
    if form == 'p':
        res_table = pd.DataFrame(index=obs_index, dtype='float64', columns=['Pap', 'Thp'])
        res_table.sort_index(inplace=True)
        for (this_lat, this_lon, this_z) in obs_index:
            res_table.loc[(this_lat, this_lon, this_z),('Pap','Thp',)] = [float(
                res_entire.Pap.sel(lat_t=this_lat, lon_t=this_lon, z_t=this_z).isel(time=-1)),
                                                                         float(
                res_entire.Thp.sel(lat_t=this_lat, lon_t=this_lon, z_t=this_z).isel(time=-1))]    
        
    # # a few of the rows of res_table can be NaN due to observations being present at grid cells/depths where
    # # the model is land or sediment
    # res_table = res_table.dropna(how='all')
    # not needed because in calc_rmse and calc_mae a .drop() is performed
        
    if no_res_entire:
        return res_table
    else:
        return [res_table, res_entire]
    
    
def convert_sim_to_sed_obs_grid(path_ratio_p_bottom, obs_ave):
    """Convert simulation path_ratio_p_bottom output to only on model grid cells with sed_obs present. 
    
    This function is related to get_sim, which is used for obs_d and obs_p; this function works for obs_sed.
    It is a different fnc because model output does not contain path_ratio_p_bottom by itself; this is computed in the
    notebook and then given into this function.
        
    Input:
    - datarray path_ratio_p_bottom of a run with coords (lat,lon); after computation present in data_fulls[runs[n]].path_ratio_p_bottom
    - obs_ave must correspond to obs_sed_ave. The data itself is not used, only obs_ave.index and the column names
    
    Output:
    - res_table is a pandas dataframe (table) with the model result of path_ratio_p_bottom only in the grid cells where 
      obs are present (unitless).
    
    Author: Sebastian Lienert
    Edited by Jeemijn Scheen (jeemijn.scheen@nioz.nl)"""
    
    import pandas as pd
    import xarray as xr
    
    assert 'path_ratio_p' in obs_ave.columns, 'needs path_ratio_p in obs_ave'
    assert 'z_t' not in path_ratio_p_bottom.coords, 'path_ratio_p_bottom cannot depend on z_t'
    assert 'time' not in path_ratio_p_bottom.coords, 'path_ratio_p_bottom cannot depend on time'
        
    obs_index = obs_ave.reset_index().set_index(['lat_sim','lon_sim_100_to_460']).index  # remove z coordinate from index
    
    # Panda DataFrame with obs_ave index
    res_table = pd.DataFrame(index=obs_index, dtype='float64', columns=['path_ratio_p'])
    for (this_lat, this_lon) in obs_index:
        res_table.loc[(this_lat, this_lon),('path_ratio_p',)] = [float(
            path_ratio_p_bottom.sel(lat_t=this_lat, lon_t=this_lon))]
        
    # 1 of the sed obs. (core C2 PC-2121009) lies where the model is land
    # res_table = res_table.dropna(how='all')  # dropping nan not needed anymore as this core was manually shifted into the ocean in obs_sed_ave

    return res_table


def model_to_sw_unit(val, rho_model):
    """Convert model unit dpm/m3 to microBq/kg = uBq/kg (geotraces seawater data)
    Input:
    - value in dpm/m3: SCALAR VALUE OR DATASET WITH SAME COORDS AS rho_model
    - rho_model is the density of the current grid cell in kg/m3: SCALAR VALUE OR DATASET
                                        => if dataset: take data_full.rho_SI.isel(time=-1)
    Output:
    - value in uBq/kg
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    # using: 1 Bq = 60 dpm (disintegrations per minute); m = rho * V
    
    # conversion factor in total: A_mod = 1e6 / (60 * rho) * A_meas
    # explanation: 1/60 for dpm to Bq;  1e6 for Bq to uBq;  / rho for /m3 to /kg
    return val / 60 * 1e6 / rho_model


def sw_to_model_unit(val):
    """Convert microBq/kg (geotraces seawater data) to model unit dpm/m3
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    # explanation: see model_to_sw_unit()

    # assuming rho=1.03e3 kg/m3
    import warnings
    warnings.warn("WARNING: density in sw_unit_to_model_unit() is not realistic")
    return val * 60 / 1e6 * 1.03e3 # 1.03e3 should be rho_measured but not (easily) available
    # EXPLANATION: the "correct" way for the conversion is to calculate density 
    # from T, S that is measured at all stations and consider this per station
    # IT IS EASIER TO CONVERT MODEL OUTPUT (WHICH HAS RHO) TO SW UNITS SO THIS
    # FUNCTION IS NOT IN USE ANYMORE


def convert_0_360_lon_to_model_lon(lon_column):
    """Convert longitudes of data from [0,360] degrees East (360 at Greenwhich)
    to Bern3D model coords [100,460].
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    assert min(lon_column) >= 0.0 and max(lon_column) <= 360.0, "Input lon should be [0,360]"
    
    # from [0,360] Greenwhich at 360 to [100,460] Greenwhich at 360
    res = [lon+360.0 if lon <= 100.0 else lon for lon in lon_column]

    assert min(res) >= 100.0 and max(res) <= 460.0, "Something wrong with output lon."
    return res


def convert_minus_180_plus_180_lon_to_0_360_lon(lon_column):
    """Convert longitudes of data from [-180,180] degrees East (0 at Greenwhich) 
    to the longitude coords [0,360] (360 at Greenwhich).
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
        
    assert min(lon_column) >= -180.0 and max(lon_column) <= 180.0, "Input lon should be [-180,180]"
    # if assertion error: check for nan values
    
    # from [-180,180] Greenwhich at 0 to [0,360] Greenwhich at 360
    res = [lon+360.0 if lon <= 0.0 else lon for lon in lon_column]
    
    assert min(res) >= 0.0 and max(res) <= 360.0, "Something wrong with output lon."
    return res


def calc_rmse(model, observation, weights=None, verbose=False):
    """Returns the root mean square error RMSE of model-observation for 1 var.
    ASSUMPTION: takes last time step if multiple time steps present
    
    Usage: for vars Pad and Thd, call this function twice, with 
           once model=model.Pad, observation=observation.Pad, weights=weights_Pad; repeat for Thd
    
    Input needs 3 pandas Series with the same index & 1 boolean
    NOTE THAT MODEL AND OBSERVATION NEED TO HAVE THE SAME UNIT!
    - model is a pandas Series containing Pad and Thd variables 
      e.g. from [a, junk] = get_sim() take a.Pad
    - observation is a pandas Series containing obs of Pad, Thd averaged to model grid
      e.g. from [junk, obs_ave, junk2] = obs_to_model_grid() take obs_ave.Pad
    - optional: weights is a pandas Series containing the desired weights. Don't need to be normalized.
      e.g. 1/observational uncertainty or boxvol or a combination of those.
    - verbose makes a print statement of the result (1 line per function call)
    
    Output:
    - float RMSE = sqrt( (sum_{k=1}^N w_k * (sim_k - obs_k)^2 / (sum_{k=1}^N w_k) )
      or if weights=None, we have trivially w_k = 1/N so 
            RMSE = sqrt( 1/N * sum_{k=1}^N (sim_k - obs_k)^2 )

    Author: Sebastian Lienert
    Edited by Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    import pandas as pd
    import numpy as np
    
    assert len(model) == len(observation), "model and observation must have equal length"
    
    if isinstance(model, pd.core.series.Series) and isinstance(
            observation, pd.core.series.Series):
        # 'error' is the misfit between model output and observations
        error = (model - observation).dropna()  # nan if model or obs has a value and the other doesn't (bathymetry)
        if weights is None:
            se = error**2
            mse = se.mean()
        else:
            if isinstance(weights, pd.core.series.Series):
                # commented out: these assertions make the function very slow, especially the 3rd 1
                # assert len(weights) == len(observation), "weights and observation must have equal length"
                # assert weights.sum() > 0, "sum of weights must be positive"  # avoid division by zero
                # assert np.asarray([np.isnan(w) or w >= 0 for w in weights]).all(), "negative weights are not permitted"

                numerator = (error**2 * weights).sum()
                denominator = weights.sum()
                mse = float(numerator / denominator)
            else: 
                raise ValueError('weights is not a pandas series')
                
        rmse = np.sqrt(mse)
        if verbose:
            print('RMSE =', np.round(rmse,3), '[uBq/kg]')
        return rmse
    else:
        raise ValueError('model and/or observations are not a pandas series')

        
# based on a copy of calc_rmse
def calc_mae(model, observation, weights=None, verbose=False):
    """Returns the mean abolute error MAE of model-observation for 1 var.
    ASSUMPTION: takes last time step if multiple time steps present
    
    Usage: for vars Pad and Thd, call this function twice, with 
           once model=model.Pad, observation=observation.Pad, weights=weights_Pad; repeat for Thd
    
    Input needs 3 pandas Series with the same index & 1 boolean
    NOTE THAT MODEL AND OBSERVATION NEED TO HAVE THE SAME UNIT!
    - model is a pandas Series containing Pad and Thd variables 
      e.g. from [a, junk] = get_sim() take a.Pad
    - observation is a pandas Series containing obs of Pad, Thd averaged to model grid
      e.g. from [junk, obs_ave, junk2] = obs_to_model_grid() take obs_ave.Pad
    - optional: weights is a pandas Series containing the desired weights. Don't need to be normalized.
      e.g. 1/observational uncertainty or boxvol or a combination of those.
    - verbose makes a print statement of the result (1 line per function call)
    
    Output:
    - float MAE = (sum_{k=1}^N w_k * |sim_k - obs_k| ) / (sum_{k=1}^N w_k)
      or if weights=None, we have trivially w_k = 1/N so 
            MAE = 1/N * sum_{k=1}^N |sim_k - obs_k|
            
    Author: Sebastian Lienert"""
    
    import pandas as pd
    import numpy as np
    
    assert len(model) == len(observation), "model and observation must have equal length"
    
    if isinstance(model, pd.core.series.Series) and isinstance(
            observation, pd.core.series.Series):
        # 'error' is the misfit between model output and observations
        error = (model - observation).dropna()  # nan if model or obs has a value and the other doesn't (bathymetry)
        if weights is None:
            ae = abs(error)
            mae = ae.mean()
        else:
            if isinstance(weights, pd.core.series.Series):
                # commented out: these assertions make the function very slow, especially the 3rd 1
                # assert len(weights) == len(observation), "weights and observation must have equal length"
                # assert weights.sum() > 0, "sum of weights must be positive"  # avoid division by zero
                # assert np.asarray([np.isnan(w) or w >= 0 for w in weights]).all(), "negative weights are not permitted"

                numerator = (abs(error) * weights).sum()
                denominator = weights.sum()
                mae = float(numerator / denominator)
            else: 
                raise ValueError('weights is not a pandas series')
        
        if verbose:
            print('MAE =', np.round(mae,4), '[uBq/kg]')
        return mae
    else:
        raise ValueError('model and/or observations are not a pandas series')


def calc_all_RMSEs(obs_d_ave, obs_p_ave, modelrunIDs, modeldir, ensemble, weighted_vol=True, weighted_unc=True, verbose=False):
    """This function calculates all RMSEs between many model runs and given obs. 
    CAN TAKE LONG (DEPENDING ON NR. OF RUNS)
    
    For PAR ensemble:  need to input just modelrunIDs=all_run_nrs as quick fix for wrong ensemble names
    Hack in general for other runnames: use memberid=runname (entirely) and ensemble='FREE'
    
    Input:
    - obs_d_ave is obs_ave object of (Pad,Thd); already the wanted subset of obs (if applicable)
    - obs_p_ave is obs_ave object of (Pap,Thp); already the wanted subset of obs (if applicable)
    - int arr modelrunIDs is list of memberids = run_IDs of the model runs
    - path variable modeldir is directory where model results are saved
    - str ensemble is ensemble name to which model runs belong. Needed to construct filenames.
    - [bool weighted_vol & weighted_unc] 
            if one or both True: the weights are computed within this fnc based on boxvol and/or observational error
            if both False: a non-weighted RMSE is computed
    - [bool verbose] prints every found RMSE
    
    Output:
    - float arr of RMSE_Pad in order of modelrunIDs
    - float arr of RMSE_Thd in order of modelrunIDs
    - float arr of RMSE_Pap in order of modelrunIDs
    - float arr of RMSE_Thp in order of modelrunIDs
    
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    import numpy as np
    
    print('Going to compute RMSE_Pad, RMSE_Thd, RMSE_Pap, RMSE_Thp for', len(modelrunIDs),'model runs.')
    
    if weighted_vol or weighted_unc:
        ## load in example run to get model grid (res_table not needed => junk)
        [junk, res_entire] = get_sim(memberid=modelrunIDs[0], path=modeldir, obs_ave=obs_d_ave, # which obs_ave doesn't matter because only using res_entire
                                     ensemble=ensemble, convert_unit_to_obs=True) 
        # redefining weights; especially important if obs are a subset of the observations (only certain cruises, w/o surface, etcc):
        # ( although weights dont need to be normalized, the below is always needed s.t. len(obs_ave) = len(weights) )
        [weights_Pad, weights_Thd] = find_weights(obs_d_ave, res_entire, weighted_vol, weighted_unc)
        [weights_Pap, weights_Thp] = find_weights(obs_p_ave, res_entire, weighted_vol, weighted_unc)
    
    # compute RMSEs
    if verbose:
        if weighted_vol and weighted_unc:
            print('Going to compute and print RMSEs (4 per run: for Pad, Thd, Pap & Thp) weighted by grid cell volume and observational uncertainty.')
        else:
            if weighted_vol:
                print('Going to compute and print RMSEs (4 per run: for Pad, Thd, Pap & Thp) weighted by grid cell volume only.')
            elif weighted_unc:
                print('Going to compute and print RMSEs (4 per run: for Pad, Thd, Pap & Thp) weighted by observational uncertainty only.')
            else:
                print('Going to compute and print a non-weighted RMSEs (4 per run: for Pad, Thd, Pap & Thp).')
    RMSEs_Pad = []
    RMSEs_Thd = []
    RMSEs_Pap = []
    RMSEs_Thp = []
    # loop over model runs
    for n,runid in enumerate(modelrunIDs):
        # load model run
        res_d_table = get_sim(memberid=runid, path=modeldir, obs_ave=obs_d_ave, ensemble=ensemble, 
                              convert_unit_to_obs=True, no_res_entire=True)
        res_p_table = get_sim(memberid=runid, path=modeldir, obs_ave=obs_p_ave, ensemble=ensemble, 
                              convert_unit_to_obs=True, no_res_entire=True)

        # calc RMSEs of this model run, depending on weights
        if verbose:
            print('Run',n,', runid',runid, ':')
        if weighted_vol or weighted_unc:
            RMSEs_Pad.append(calc_rmse(model=res_d_table.Pad, observation=obs_d_ave.Pad, weights=weights_Pad, verbose=verbose))
            RMSEs_Thd.append(calc_rmse(model=res_d_table.Thd, observation=obs_d_ave.Thd, weights=weights_Thd, verbose=verbose))
            RMSEs_Pap.append(calc_rmse(model=res_p_table.Pap, observation=obs_p_ave.Pap, weights=weights_Pap, verbose=verbose))
            RMSEs_Thp.append(calc_rmse(model=res_p_table.Thp, observation=obs_p_ave.Thp, weights=weights_Thp, verbose=verbose))
        else:
            # non-weighted RMSE
            RMSEs_Pad.append(calc_rmse(model=res_d_table.Pad, observation=obs_d_ave.Pad, weights=None, verbose=verbose))
            RMSEs_Thd.append(calc_rmse(model=res_d_table.Thd, observation=obs_d_ave.Thd, weights=None, verbose=verbose))
            RMSEs_Pap.append(calc_rmse(model=res_p_table.Pap, observation=obs_p_ave.Pap, weights=None, verbose=verbose))
            RMSEs_Thp.append(calc_rmse(model=res_p_table.Thp, observation=obs_p_ave.Thp, weights=None, verbose=verbose))
            
    if verbose:
        print('Finished.')
                
    return [np.array(RMSEs_Pad), np.array(RMSEs_Thd), np.array(RMSEs_Pap), np.array(RMSEs_Thp)]


# based on a copy of calc_all_RMSEs
def calc_all_MAEs(obs_d_ave, obs_p_ave, modelrunIDs, modeldir, ensemble, weighted_vol=True, weighted_unc=True, verbose=False):
    """This function calculates all MAEs between many model runs and given obs.
    CAN TAKE LONG (DEPENDING ON NR. OF RUNS)
    
    For PAR ensemble:  need to input just modelrunIDs=all_run_nrs as quick fix for wrong ensemble names
    Hack in general for other runnames: use memberid=runname (entirely) and ensemble='FREE'
    
    Input:
    - obs_d_ave is obs_ave object of (Pad,Thd); already the wanted subset of obs (if applicable)
    - obs_p_ave is obs_ave object of (Pap,Thp); already the wanted subset of obs (if applicable)
    - int arr modelrunIDs is list of memberids = run_IDs of the model runs
    - path variable modeldir is directory where model results are saved
    - str ensemble is ensemble name to which model runs belong. Needed to construct filenames.
    - [bool weighted_vol & weighted_unc] 
            if one or both True: the weights are computed within this fnc based on boxvol and/or observational error
            if both False: a non-weighted MAE is computed
    - [bool verbose] prints every found MAE
    
    Output:
    - float arr of MAE_Pad in order of modelrunIDs
    - float arr of MAE_Thd in order of modelrunIDs
    - float arr of MAE_Pap in order of modelrunIDs
    - float arr of MAE_Thp in order of modelrunIDs
    
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    import numpy as np
    
    print('Going to compute MAE_Pad, MAE_Thd, MAE_Pap, MAE_Thp for', len(modelrunIDs),'model runs.')
    
    # compute weights
    if weighted_vol or weighted_unc:
        ## load in example run to get model grid (res_table not needed => junk)
        [junk, res_entire] = get_sim(memberid=modelrunIDs[0], path=modeldir, obs_ave=obs_d_ave, # which obs_ave doesn't matter because only using res_entire
                                     ensemble=ensemble, convert_unit_to_obs=True)
        # redefining weights; especially important if obs are a subset of the observations (only certain cruises, w/o surface, etcc):
        # ( although weights dont need to be normalized, the below is always needed s.t. len(obs_ave) = len(weights) )
        [weights_Pad, weights_Thd] = find_weights(obs_d_ave, res_entire, weighted_vol, weighted_unc)
        [weights_Pap, weights_Thp] = find_weights(obs_p_ave, res_entire, weighted_vol, weighted_unc)
    
    # compute MAEs
    if verbose:
        if weighted_vol and weighted_unc:
            print('Going to compute and print MAEs (4 per run: for Pad, Thd, Pap & Thp) weighted by grid cell volume and observational uncertainty.')
        else:
            if weighted_vol:
                print('Going to compute and print MAEs (4 per run: for Pad, Thd, Pap & Thp) weighted by grid cell volume only.')
            elif weighted_unc:
                print('Going to compute and print MAEs (4 per run: for Pad, Thd, Pap & Thp) weighted by observational uncertainty only.')
            else:
                print('Going to compute and print non-weighted MAEs (4 per run: for Pad, Thd, Pap & Thp).')
    MAEs_Pad = []
    MAEs_Thd = []
    MAEs_Pap = []
    MAEs_Thp = []
    # loop over model runs
    for n,runid in enumerate(modelrunIDs):
        # load model run
        res_d_table = get_sim(memberid=runid, path=modeldir, obs_ave=obs_d_ave, ensemble=ensemble, 
                              convert_unit_to_obs=True, no_res_entire=True)
        res_p_table = get_sim(memberid=runid, path=modeldir, obs_ave=obs_p_ave, ensemble=ensemble, 
                              convert_unit_to_obs=True, no_res_entire=True)

        # calc MAEs of this model run, depending on weights
        if verbose:
            print('Run',n,', runid',runid, ':')
        if weighted_vol or weighted_unc:
            MAEs_Pad.append(calc_mae(model=res_d_table.Pad, observation=obs_d_ave.Pad, weights=weights_Pad, verbose=verbose))
            MAEs_Thd.append(calc_mae(model=res_d_table.Thd, observation=obs_d_ave.Thd, weights=weights_Thd, verbose=verbose))
            MAEs_Pap.append(calc_mae(model=res_p_table.Pap, observation=obs_p_ave.Pap, weights=weights_Pap, verbose=verbose))
            MAEs_Thp.append(calc_mae(model=res_p_table.Thp, observation=obs_p_ave.Thp, weights=weights_Thp, verbose=verbose))
        else:
            # non-weighted RMSE
            MAEs_Pad.append(calc_mae(model=res_d_table.Pad, observation=obs_d_ave.Pad, weights=None, verbose=verbose))
            MAEs_Thd.append(calc_mae(model=res_d_table.Thd, observation=obs_d_ave.Thd, weights=None, verbose=verbose))
            MAEs_Pap.append(calc_mae(model=res_p_table.Pap, observation=obs_p_ave.Pap, weights=None, verbose=verbose))
            MAEs_Thp.append(calc_mae(model=res_p_table.Thp, observation=obs_p_ave.Thp, weights=None, verbose=verbose))

    if verbose:
        print('Finished.')
                
    return [np.array(MAEs_Pad), np.array(MAEs_Thd), np.array(MAEs_Pap), np.array(MAEs_Thp)]


def combine_mean_errors(MEs_4types, obs_d_ave, obs_p_ave):
    """This function combines different mean errors MEs (either RMSEs or MAEs) into 1 total skill measure.
    The four MEs for Pad, Thd, Pap and Thp are combined.
    The combination is just the sum with a normalization by dividing by the respective average of observations, as taken
    from obs_ave (can be a subset, such as the mean errors as hand can have been computed on a subset).
    Input:
    - MEs_4types=[ME_Pad, ME_Thd, ME_Pap, ME_Thp] arrays of arrays with the results for ME=RMSE or ME=MAE
    - obs_d_ave to use for normalization of dissolved
    - obs_p_ave to use for normalization of particle-bound.
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl """
    
    assert len(MEs_4types)==4, "MEs_4types must have length 4 in the order [ME_Pad, ME_Thd, ME_Pap, ME_Thp] "
        
    [ME_Pad, ME_Thd, ME_Pap, ME_Thp] = MEs_4types
        
    return ME_Pad / obs_d_ave.Pad.mean() + ME_Thd / obs_d_ave.Thd.mean() + ME_Pap / obs_p_ave.Pap.mean() + ME_Thp / obs_p_ave.Thp.mean()


def find_weights(obs_ave, res_entire, volume=True, uncertainty=True):
    """Finds weights based on obs. uncertainty and/or grid cell volume
    Formula: 
    - if volume & uncertainty: w_k = vol_k / err_k
    - if volume only: w_k = vol_k
    - if uncertainty only: w_k = 1 / err_k
    
    Input:
    - obs_ave
    - res_entire (an example model run output to use its box volume)
    - volume: is taken into account or not (see formula above)
    - uncertainty: is taken into account or not (see formula above)
    
    Output:
    If Pad, Thd columns detected in obs_ave input:
    - weights_Pad
    - weights_Thd
    If Pap, Thp columns detected in obs_ave input:
    - weights_Pap
    - weights_Thp
    If path_ratio_p column detected in obs_ave input:
    - weights_path_ratio_p

    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    import pandas as pd
    assert volume or uncertainty, "Needs at least volume or uncertainty to determine weights."
    
    if volume:
        # find grid cell volumes of relevant cells only (i.e. where obs are present)
        cell_volume = pd.Series(index=obs_ave.index, dtype='float')  # obs_ave.index = model.index
        for (this_lat, this_lon, this_z) in cell_volume.index:
            # using 'res_entire' model result from any run
            cell_volume.loc[(this_lat, this_lon, this_z)] = float(res_entire.boxvol.sel(lat_t=this_lat, lon_t=this_lon, z_t=this_z))
        if not uncertainty:
            return [cell_volume, cell_volume]

    if uncertainty:
        if 'Pap' in obs_ave.columns and 'path_ratio_p' in obs_ave.columns:
            print("find_weights(): found 'Pap' as well as 'path_ratio_p' in obs_ave.columns; continuing with Pap.")
        if ('Pad' in obs_ave.columns and 'Pap' in obs_ave.columns) or ('Pad' in obs_ave.columns and 'path_ratio_p' in obs_ave.columns):
            raise Exception('find_weights(): not able to infer whether obs_ave has dissolved or particle-bound forms')    
        elif 'Pad' in obs_ave.columns:
            # determine weights via the formula above for dissolved form
            if volume:
                weights_Pad = cell_volume / obs_ave.Pad_err
                weights_Thd = cell_volume / obs_ave.Thd_err
            else:
                weights_Pad = 1 / obs_ave.Pad_err
                weights_Thd = 1 / obs_ave.Thd_err
            return [weights_Pad, weights_Thd]
        elif 'Pap' in obs_ave.columns:
            # determine weights via the formula above for particle-bound form
            if volume:
                weights_Pap = cell_volume / obs_ave.Pap_err
                weights_Thp = cell_volume / obs_ave.Thp_err
            else:
                weights_Pap = 1 / obs_ave.Pap_err
                weights_Thp = 1 / obs_ave.Thp_err                
            return [weights_Pap, weights_Thp]
        elif 'path_ratio_p' in obs_ave.columns:
            # determine weights via the formula above
            if volume:
                weights_path_ratio_p = cell_volume / obs_ave.path_ratio_p_err
            else:
                weights_path_ratio_p = 1 / obs_ave.path_ratio_p_err
            return weights_path_ratio_p       
        else:
            raise Exception('find_weights(): not able to infer whether obs_ave has dissolved or particle-bound forms')


def remin_curve_val(z, particle):
    """Input: z in meters; particle in 'POC', 'CaCO3', 'opal', 'dust' or 'neph'
    Output: fraction of particle between 0 and 1 that is still present (not remineralized) at this depth
    
    APPROXIMATION: if z in euphotic zone, output = 1 (neglecting that grid cells lie partially in it).
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    import math
    
    assert particle in ['POC', 'CaCO3', 'opal', 'dust', 'neph'], "Enter valid particle type"
    assert z >= 0, "z [meter] must be positive"
    # assert z == 0 or z > 75, "z [meter] needs to be below euphotic zone"
    # APPROXIMATION: IF z IN EUPHOTIC ZONE, WE TAKE 1 (neglecting that grid cells lie partially in it)
    if z < 75:
        # in euphotic zone
        return 1.0
    
    zcomp = 75 # meter    
    alpha_POC = 0.83
    rem_scale_ca = 5066 # m
    rem_scale_op = 10000 # m
    
    if particle == 'POC':
        return (z/zcomp)**(-alpha_POC)
    if particle == 'CaCO3':
        return math.exp(-(z-zcomp)/rem_scale_ca)
    if particle == 'opal':
        return math.exp(-(z-zcomp)/rem_scale_op)
    if particle == 'dust' or particle == 'neph':
        return 1.0

    
def add_line(axis, val, horizontal=True, c='grey', alpha=1.0, ls='solid'):
    """Adds line to a subplot (horizontal or vertical).
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    [[xmin, xmax], [ymin, ymax]] = [axis.get_xlim(), axis.get_ylim()]
    if horizontal:
        axis.plot([xmin,xmax],[val, val], c, alpha=alpha, linestyle=ls)
    else:
        axis.plot([val, val],[ymin,ymax], c, alpha=alpha, linestyle=ls)
    return axis
    
    
def add_rectangle(axis, x_bnds=None, y_bnds=None, c='grey', alpha=0.2):
    """Adds rectangle to a subplot. 
    x_bnds and y_bnds can be given in; if 1 is omitted the entire width/height is taken
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl."""
    
    import matplotlib.patches as patches
    
    if x_bnds is None and y_bnds is None:
        return axis

    [[xmin, xmax], [ymin, ymax]] = [axis.get_xlim(), axis.get_ylim()]
    if x_bnds is None:
        x_bnds = [xmin, xmax]
    if y_bnds is None:
        y_bnds = [ymin, ymax]
    rect = patches.Rectangle((x_bnds[0], y_bnds[0]), x_bnds[1]-x_bnds[0], y_bnds[1]-y_bnds[0], 
                             edgecolor='none', facecolor=c, alpha=alpha) # xy, width, height
    axis.add_patch(rect)
    
    # this function can mess up legend, so first call legend then add rectangles
    return axis
    
    
def load_cruise_stations_other(fnobs): 
    """This function loads dissolved seawater observations from additional studies other than geotraces.
    GOAL: to determine cruise tracks on the Bern3D grid (together with other functions).

    Input:
    - str fnobs is path (use path variable) + filename of observations (a .csv file)
    
    Output:
    - obs with only columns kept that give coords of cruise track & additional cruise information.
   
    Authors: Sebastian Lienert, Jeemijn Scheen (jeemijn.scheen@nioz.nl)"""

    import pandas as pd    
    assert 'IDP' not in str(fnobs), 'this function is not for geotraces'

    # load observation file for dissolved
    extension = str(fnobs).split('.')[-1]
    assert extension == 'csv', "File extension of fnobs should be .csv because this function is for files other than GEOTRACES."
    
    if 'Pad_Thd' in str(fnobs):
        obs = pd.read_csv(fnobs, sep=',', header=0)
    else:
        raise Exception("file name should contain 'Pad_Thd'.")
    
    ## change longitude from [-180,180] to [0,360]
    obs['lon'] = convert_minus_180_plus_180_lon_to_0_360_lon(obs['lon'].values)
    
    # rename columns
    obs = obs.rename(columns={'lon': 'lon_obs', 'lat' : 'lat_obs'})

    # drop other columns        
    obs.drop(['z'] + 
             [s for s in obs.columns if 'Pa' in s or 'Th' in s], axis=1, inplace=True)

    # many duplicate rows exist because we deleted the z axis
    obs.drop_duplicates(inplace=True)

    return obs  


def load_cruise_stations_geotraces(fnobs): 
    """This function loads observations from geotraces dissolved OR particle-bound seawater (IDP2021).
    GOAL: to determine cruise tracks on the Bern3D grid (together with other functions).

    Input:
    - str fnobs is path (use path variable) + filename of observations (a .txt file)
    
    Output:
    - obs with only columns kept that give coords of cruise track & additional cruise information.
   
    Authors: Sebastian Lienert, Jeemijn Scheen (jeemijn.scheen@nioz.nl)"""

    import pandas as pd    
    assert 'IDP' in str(fnobs), "use fnc get_all_obs_other() instead for non-geotraces files"

    # load observation file
    extension = str(fnobs).split('.')[-1]
    assert extension == 'txt', "File extension of fnobs should be .txt (GEOTRACES ascii files)."
            
    # DISSOLVED OBSERVATIONS     #########################################################################
    if 'Pad_Thd' in str(fnobs):
        obs = pd.read_csv(fnobs, sep='\t', header=39, dtype={'Cruise Aliases':'str'})
        
        ## NOTES ON WARNING ABOUT COLUMN 21:
        # column 21 contains 'QV:SEADATANET.2': the quality control variable for var 3=Pad.
        # This qc column contains floats as well as occurences of 'Q', which means 'value below limit of quantification'.
        # So it is fine to keep this column mixed.
        print('Column 21 has mixed types as it contains quality control variables, which can be int or string. Fine.')

    # PARTICLE-BOUND OBSERVATIONS     #################################################################
    elif 'Pap_Thp' in str(fnobs):
        obs = pd.read_csv(fnobs, sep='\t', header=35, dtype={'Cruise Aliases':'str'})

    # FROM HERE ON CODE WORKS FOR BOTH DISSOLVED AND PARTICLE-BOUND OBSERVATIONS    ###################

    # rename columns that we want to keep
    obs = obs.rename(columns={'Cruise': 'cruise', 'Station' : 'station', 'Period' : 'period',
                              'Longitude [degrees_east]' : 'lon_obs', 'Latitude [degrees_north]' : 'lat_obs', 
                              "Operator's Cruise Name" : 'cruise_name_by_operator', 'BODC Cruise Number' : 'BODC_cruise_number',
                              'Cruise Aliases' : 'cruise_aliases', 'Cruise Information Link' : 'cruise_info_link'})

    # drop other columns        
    obs.drop(['Type', "yyyy-mm-ddThh:mm:ss.sss", 'Bot. Depth [m]', 'CTDPRS_T_VALUE_SENSOR [dbar]', 
              'DEPTH [m]', 'Ship Name', 'Chief Scientist', 'GEOTRACES Scientist', 'QV:ODV:SAMPLE'] + 
             [s for s in obs.columns if 'STANDARD_DEV' in s 
                                         or 'QV:SEADATANET' in s
                                         or 'Pa_' in s or 'Th_' in s], axis=1, inplace=True)

    # delete rows with history comments (contain no data)
    bad_rows = list(obs[obs.cruise.str.startswith('//<History>')].index)
    obs.drop(index=bad_rows, inplace=True)
    
    # many duplicate rows exist because we deleted the z axis
    obs.drop_duplicates(inplace=True)

    return obs


def add_model_coords_2D(obs, fnctrl):
    """Transfers a 2D (lon_obs, lat_obs) dataframe with cruise locations to model grid.   
    This function is a simplified version of obs_to_model_grid(), without the hassle of 
    taking care of observations.
        
    GOAL: to determine cruise tracks on the Bern3D grid (together with other functions).

    Input:
    - obs is dataframe that contains lon_obs and lat_obs coordinates (no z; no multi-index set).
      e.g. the output of load_cruise_stations_geotraces() or load_cruise_stations_other()
    - fnctrl file of control run; only model grid is used
    
    Output:
    - same obs with added columns lon_sim_0_to_360, lon_sim_100_to_460 and lat_sim
    
    Author: Sebastian Lienert
    Edited by Jeemijn Scheen (jeemijn.scheen@nioz.nl)"""
    
    import pandas as pd
    import xarray as xr
    assert obs.lon_obs.min() >= 0.0 and obs.lon_obs.max() <= 360.0, "longitude of obs must be in [0,360]"
    
    # look for corresponding lat/lon coordinates of obs in control file
    ctrl = xr.open_dataset(str(fnctrl) + '.0001765_full_ave.nc', decode_times=False)

    # model lon (ctrl.lon_t) goes from 100 to 460; convert here first from 0 to 360 (as obs are)
    list_ctrl_lon_t_0_to_360 = [lon if lon < 360.0 else (lon - 360.0) for lon in ctrl.lon_t]
    list_ctrl_lon_t_0_to_360.sort()
    # convert to DataArray in order to use .sel(method='nearest')
    ctrl_lon_t_0_to_360 = xr.DataArray(
      data=list_ctrl_lon_t_0_to_360,
      coords={'lon_t': list_ctrl_lon_t_0_to_360},
      dims=['lon_t'],
      attrs={'long_name': "Longitude (T grid) from 0 to 360", 'units': "degrees_east"})
    # matching obs to model coords
    lon_sim_0_to_360 = [
        float(ctrl_lon_t_0_to_360.sel(lon_t=this_lon, method='nearest'))
        for this_lon in obs.lon_obs
    ]
    # we dont need to apply boundary conditions explicitly because the model grid is regular around 360 degrees
    # (model grid points at 355.0 and 5.0 have the boundary of 0=360 exactly in their middle)

    lat_sim = [
        float(ctrl.lat_t.sel(lat_t=this_lat, method='nearest'))
        for this_lat in obs.lat_obs
    ]

    obs.insert(2, 'lon_sim_0_to_360', lon_sim_0_to_360) # add new column after 2nd column 

    # now convert our intuitive lon (0 to 360) to model coord lon (100 to 460)
    # for clarity we add both lon columns
    lon_sim_100_to_460 = [lon if lon > 100.0 else (lon + 360.0) for lon in lon_sim_0_to_360]
    # do not sort because order reflects order of obs!
    obs.insert(3, 'lon_sim_100_to_460', lon_sim_100_to_460)
    obs.insert(4, 'lat_sim', lat_sim)
    
    return obs


def get_all_cruise_coords(obsdir, fnctrl):
    """This function loads and combines cruise tracks of the different datasets to be used later for section plots. 
    It keeps all cruise coordinates, independent of which measurements were done.
    For dissolved these datasets are loaded: geotraces, deng, ng, pavia.
    For particle-bound: geotraces.
    
    Input:
    - obsdir: folder with datasets (filenames are hardcoded)
    - fnctrl: filename of control simulation (used for Bern3D grid only)

    Output:
    - coords_all: object with all cruise coordinates regridded to the Bern3D
    
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    from pandas import concat

    ## 1A). Get coords of cruises with dissolved seawater obs. ##########################
    coords_d_geotraces = load_cruise_stations_geotraces(fnobs=obsdir / 'Pad_Thd_IDP2021.txt')
    coords_d_deng = load_cruise_stations_other(obsdir / 'Deng2018Pad_Thd_formatted_uBq_per_kg.csv') # is geovide
    coords_d_ng = load_cruise_stations_other(obsdir / 'Ng2020Pad_Thd_formatted_dpm_per_1000kg.csv')
    coords_d_pavia = load_cruise_stations_other(obsdir / 'Pavia2020Pad_Thd_formatted_uBq_per_kg.csv') 
    # combine all coords from dissolved data:
    coords_d = concat([coords_d_geotraces, coords_d_deng, coords_d_ng, coords_d_pavia], join='outer')

    ## 1B). Get coords of cruises with particle-bound seawater obs. ####################
    coords_p = load_cruise_stations_geotraces(fnobs=obsdir / 'Pap_Thp_IDP2021.txt')

    ## 1C). Combine all coords ####################
    coords_all = concat([coords_d, coords_p], join='outer')
    # this created duplicates because p-bound cruises are usually also in dissolved cruises
    coords_all.drop_duplicates(inplace=True)

    ## 2). Convert coords to model grid ####################
    coords_all = add_model_coords_2D(coords_all, fnctrl)
    
    return coords_all


def sort_coords_per_cruise(coords_all, drop_non_relevant_side_branches, sort_extra_manual=True):
    """"Organizes and sorts coordinates of each cruise in a dict 'model_coords_per_cruise'
    Input:
    - coords_all object (output of get_all_cruise_coords() )
    - boolean drop_non_relevant_side_branches
    - boolean sort_extra_manual for a more logical order where default sorting by lat/lon not sufficient
    Output:
    - model_coords_per_cruise with keys cruise strings and values dataset with all coordinates
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    from numpy import unique, asarray

    if (not drop_non_relevant_side_branches) and sort_extra_manual:
        raise Exception('for sort_extra_manual=True you need drop_non_relevant_side_branches=True currently')
    
    model_coords_per_cruise = {}
    for this_cruise in unique(coords_all.cruise):
        res = coords_all[coords_all.cruise == this_cruise].loc[:,('lon_sim_100_to_460', 'lat_sim')].drop_duplicates()

        # sort resulting coords such that cruise 'walks' in a somewhat logical way
        if len(unique(res.lon_sim_100_to_460)) >= len(unique(res.lat_sim)):
            # cruise goes more West-East than North-South
            res = res.sort_values(['lon_sim_100_to_460', 'lat_sim'])  # main sort is on lon
        else:
            # cruise goes more North-South than West-East
            res = res.sort_values(['lat_sim','lon_sim_100_to_460'])  # main sort is on lat
        res.reset_index(inplace=True, drop=True)

        # # hard-coded adjustments to cruise tracks:
        if drop_non_relevant_side_branches:
            if this_cruise == 'GA02':
                # original water column indices of GA02's coords-to-drop: 14, 20, 22, 23, 25, 38, 41, 42, 43 (count from 1)
                coords_to_drop_GA02 = [[335.5, -10.5],[328.5, 4.5],[328.5, 7.5],[335.5, 7.5],[335.5, 10.5],
                                    [328.5, 42.5],[335.5, 47.5],[349.5, 47.5],[356.5, 47.5]]

                # find indices of side branches to drop from res (=where equal to coords_to_drop_GA02)
                indices_to_drop = [] 
                for (drop_this_lon, drop_this_lat) in coords_to_drop_GA02:
                    for i, row in res.iterrows():
                        if row.lon_sim_100_to_460 == drop_this_lon and row.lat_sim == drop_this_lat:
                            indices_to_drop.append(i)

                print('GA02 has', len(res), 'indices before dropping side branches.')
                res = res.drop(indices_to_drop)
                print('GA02 has', len(res), 'indices after dropping side branches.')
            elif this_cruise == 'GA03':
                # original water column indices of GA03's coords-to-drop: 1, 4, 8, 9, 12, 14, 19 (count from 1)
                coords_to_drop_GA03 = [[278.0, 32.5],[293.5, 22.5],[300.5, 17.5],[300.5, 22.5],
                                       [307.5, 17.5],[314.5, 17.5],[328.5, 22.5]]  # (lon,lat)

                # find indices of side branches to drop from res (=where equal to coords_to_drop_GA03)
                indices_to_drop = [] 
                for (drop_this_lon, drop_this_lat) in coords_to_drop_GA03:
                    for i, row in res.iterrows():
                        if row.lon_sim_100_to_460 == drop_this_lon and row.lat_sim == drop_this_lat:
                            indices_to_drop.append(i)

                print('GA03 has', len(res), 'indices before dropping side branches.')
                res = res.drop(indices_to_drop)
                print('GA03 has', len(res), 'indices after dropping side branches.')

        # switch from pandas to numpy array
        res = res.values

        if sort_extra_manual:
            coords_orig = res.copy()
            if this_cruise == 'GA10':
                # swap water column indices 2 and 3 on map (python indices 1 and 2)
                res[1] = coords_orig[2]
                res[2] = coords_orig[1]
            elif this_cruise == 'GIPY05':
                # totally change order
                new_order = [2,1,4,3,6,5,8,7,9,10,11,13,12,14,15,20,16,
                             21,17,22,18,23,19,24,25,26,27,28,29,30]
                res = asarray([list(coords_orig[i-1])  # -1 goes from index on map to python index
                               for i in new_order])
            elif this_cruise == 'GA02':
                # swap water column indices 23 and 24 on map (is python indices 22 and 23)
                res[22] = coords_orig[23]
                res[23] = coords_orig[22]
            elif this_cruise == 'GA03':
                # change order a lot: swap 3&5; 6&7; 10&11; rename 17 to 14 and update 14,15 etc. +1
                new_order = [1,2,5,4,3,7,6,8,9,11,10,12,13,17,14,15,16,18,19,20]
                res = asarray([list(coords_orig[i-1])
                               for i in new_order])
            assert len(coords_orig) == len(res), "something wrong after manual sorting "+this_cruise

        model_coords_per_cruise[this_cruise] = res
        
    return model_coords_per_cruise
    
    
def add_sw_obs(fig, var, obs_ave, cruise, model_coords_of_cruise, cmap, vmin, vmax, verbose=False):
    """Adds observations of Pa or Th from seawater data to section plot.
    Input:
    - figure handle of corresponding trajectory plot
    - string var is 'Pad_Bq', 'Thd_Bq', 'Pap_Bq' or 'Thp_Bq'
    - obs_ave must correspond to seawater observations averaged to grid cells: 
      obs_d_ave or obs_p_ave (generated via obs_to_model_grid()), as appropriate
    - string cruise is a valid cruise occurring in the obs_obj data
    - model_coords_of_cruise is a 2xn array describing the plotted water columns of this cruise;
        n = number of water columns; order is as plotted; each row is [lon, lat] in model coordinates
        Typically this is from the object model_coords_per_cruise[cruise]
    - string cmap
    - float vmin of colourbar of this var
    - float vmax
    - bool verbose: prints the plotted data
    Output:
    - new fig object
    Example usage:
    fig = add_sw_obs(fig, var='Pad_Bq', obs_ave=obs_d_ave, cruise='GA02', cmap=cmap, vmin=this_vmin, vmax=this_vmax,
                    model_coords_of_cruise=model_coords_per_cruise['GA02'])
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    from numpy import unique, where
    
    if var[-3:] == '_Bq':
        var = var[:-3]  # remove '_Bq'
    if not var in ['Pad', 'Thd', 'Pap', 'Thp', 'path_ratio_p']:
        raise Exception("Unvalid var.")
        
    axlist = fig.axes
    assert len(axlist)==3, "Expect 3 axes obj as we expect a section plot with input_all_cells=True (surface+deep+cbar)"
    
    assert cruise in unique(obs_ave.cruise), "Cruise "+ cruise+" not present in the obs_obj given in. For p-bound only GA03;GIPY04;GIPY05;GN01;GP16 have measurements."
    
    data_to_plot = obs_ave[obs_ave.cruise==cruise].dropna(how='all')

    if len(data_to_plot[var].dropna(how='all'))>0:
        # only proceed if data are not empty

        # Add a new water_column_index to data in the correct order
        # np.where(A) finds all indices where A is True
        # [0][0] takes the result 3 out of e.g. (array[3],)
        # +1 because want to count water columns from 1, like in plot_gruber.py: plot() > add_section() > get_section_data() > x
        water_column_index = [where((model_coords_of_cruise[:,0] == lon_sim) & (model_coords_of_cruise[:,1] == lat_sim))[0][0] + 1
                              for (lat_sim,lon_sim,z_sim) in data_to_plot.index]  # orig: works but hard to debug

        # IF THE ABOVE LINE GIVES AN ERROR "IndexError: index 0 is out of bounds for axis 0 with size 0"
        # THEN YOU DEFINED A TRAJECTORY THAT DOES NOT GO OVER ALL WATER COLUMNS WITH OBS
        # s.t. "where(a & b)" doesn't find a match and outputs (array[],) instead of e.g. (array[3],)

        # includes redundancy of z coordinates in same water column so has the form e.g. [0,0,0,0,1,2,2,3]
        data_to_plot['water_column_index'] = water_column_index
        
        for axis in axlist[0:2]:  # exclude 3rd axis = colorbar  
            axis.scatter(x=data_to_plot.water_column_index.values, y=data_to_plot.index.get_level_values('z_sim').values/1000.0, 
                         c=data_to_plot[var].values, 
                         marker='o', s=40, lw=0.5, edgecolor='k', cmap=cmap, vmin=vmin, vmax=vmax)

        if verbose:
            print(f'{(data_to_plot.water_column_index.values)=}')         
            # print(f'{(data_to_plot.index.get_level_values("z_sim").values/1000.0)=}') 
            # print(f'{(data_to_plot[var].values)=}')

    return fig


def load_data_multiple_runs(folder, runs, spinup_yr=1765, full=True, full_inst=False, z_in_km=False, add_more_PaTh_vars=False):
    """Input: 
    - folder must be a pathlib.Path object
    - runs is list of runname strings in this folder. This can also be 1 run; it will then end up alone in the usual dicts (see Output)
    - spinup_yr [optional; if other than 1765] is an int if all simulations have equal spinup; otherwise int array
      N.B. needed since file_path = folder + runname + spinup_yr 
    - if z_in_km: convert z coordinates from m to km
    - if add_more_PaTh_vars: compute extra vars relevant for Pa and Th, based on output
      => this adds the vars: path_ratio_p (=Pap/Thp), path_ratio_d (=Pad/Thd), 
                             & Pad_Bq, Thd_Bq, Pap_Bq, Thp_Bq (=Pad, Thd, Pap, Thp converted to unit uBq/kg)
    - full [optional] if you want (no) full_ave.nc file (e.g. not generated for runs with output every time step)
    - full_inst [optional] if you want full_inst.nc file as well (for special runs diagnosing convection or seasonal cycle)
    
    Output:
    - [datas, data_fulls (optional), data_full_inst(optional)] 
       contains 1 to 3 dictionaries with runs; depending on full & full_inst setting
        NB default setting gives:
       [datas, data_fulls] 

    Explanation of output:
    1) data = timeseries data (2D) from timeseries_ave.nc output file
    2) data_full = 3D data from full_ave.nc output file 
    3) data_full_inst = data from full_inst.nc output file (not yearly averaged but specific time step at end of year)

    For all 3: the year axis is changed from simulation years to years in C.E.
    
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    from xarray import open_dataset
    from numpy import ndarray

    assert not (add_more_PaTh_vars and full_inst), "data_fulls_inst is not implemented yet together with option add_more_PaTh_vars"
    
    datas = {}           # empty dict for timeseries output DataSets
    if full:
        data_fulls = {}  # empty dict for full output DataSets
    if full_inst:
        data_fulls_inst = {} 
        
    # change years to simulation years
    subtract_yrs = spinup_yr 
    
    for nr, runname in enumerate(runs):
        # set up path
        if spinup_yr == 0:  # ignoring all problems with cases of spinup 0 C.E. to 999 C.E.
            spinup_yr_str = '0000'
        else:
            if isinstance(spinup_yr, (list, tuple, ndarray)):  # test if spinup_yr is a list/tuple/array
                spinup_yr = spinup_yr[nr]
            spinup_yr_str = str(spinup_yr)
        file = runname + '.000' + spinup_yr_str                # first part of file name; same for all
        # create datas
        # using / because folder is a pathlib.Path object (works on every operating system)
        datas[runname] = open_dataset(folder / (file + '_timeseries_ave.nc'), decode_times=False) 
        datas[runname].coords['time'] = datas[runname]['time'] - subtract_yrs
        # create data_fulls
        if full:
            data_fulls[runname] = open_dataset(folder / (file + '_full_ave.nc'), decode_times=False)
            data_fulls[runname].coords['time'] = data_fulls[runname]['time'] - subtract_yrs
            if add_more_PaTh_vars:
                ## add ratios Pa/Th as a variable because then plotting routines can read it
                # add variable to dataset
                data_fulls[runname]["path_ratio_p"]=(['time', 'z_t', 'lat_t', 'lon_t'],  
                                                    (data_fulls[runname].Pap / data_fulls[runname].Thp).data) 
                # repeat for ratio of dissolved types
                data_fulls[runname]["path_ratio_d"]=(['time', 'z_t', 'lat_t', 'lon_t'],  
                                                    (data_fulls[runname].Pad / data_fulls[runname].Thd).data)

                # add Pa, Th in sw units of uBq/kg s.t. Gruber plot can read it
                # (if the scaling were a constant, one could use the Gruber 'scale'param, but it scales with rho(i,j,k))
                for var in ['Pad', 'Pap', 'Thd', 'Thp']:
                    data_fulls[runname][var + "_Bq"]=(['time', 'z_t', 'lat_t', 'lon_t'],  
                                                        (model_to_sw_unit(data_fulls[runname][var], 
                                                                          data_fulls[runname].rho_SI.isel(time=-1))).data)
        # create data_fulls_inst using dask arrays (chunks) because of large file sizes
        if full_inst:
            data_fulls_inst[runname] = open_dataset(folder / (file + '_full_inst.nc'), 
                                                    decode_times=False, chunks={'yearstep_oc': 20})
            data_fulls_inst[runname].coords['time'] = data_fulls_inst[runname]['time'] - subtract_yrs
    
    if z_in_km:
        if len(runs)>1 and runs[0]==runs[1]:
            if len(runs)==2:
                # we have 2 equal runs and no other runs => convert z once instead of for loop over runs
                these_runs = [runs[0]]
            else:
                raise Exception("cannot apply z_in_km because some runs occur twice and >2 runs")
        else:
            # the normal situation: the runs are different
            these_runs = runs

        for runname in these_runs:
            for z_coord in ['z_t', 'z_w']:
                datas[runname][z_coord] = datas[runname][z_coord] / 1000.0             # convert depth to km
                if full:
                    data_fulls[runname][z_coord] = data_fulls[runname][z_coord] / 1000.0
                if full_inst:
                    data_fulls_inst[runname][z_coord] = data_fulls_inst[runname][z_coord] / 1000.0

    # generate returned list in correct order
    res = [datas]
    if full:
        res.append(data_fulls)
    if full_inst:
        res.append(data_fulls_inst)
    return res
        
def area_mean(obj, obj_with_data_var, keep_lat=False, keep_lon=False, basin=""):
    '''Takes horizontal area-weighted average of a certain data_var. 
    
    - obj must be a DataSet with data variable 'area' and coordinates 'lat_t', 'lon_t'
    - obj_with_data_var must contain the data_var wanted e.g. data_full.TEMP
    - basin can be set; otherwise the result will be too small by a fixed factor.   
    options: 'pac' and 'atl' (mask 2 and 1, resp.) and 'pacso' and 'atlso' (masks 2 and 1, resp.)
    - if keep_lat is True then latitude is kept as a variable and the area_weight is only done over longitude.       
    - if keep_lon is True then area_weight is only done over latitude.
    
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl'''       
        
    if keep_lat and keep_lon:
        raise Exception("not possible to average when both keep_lat and keep_lon.")
        
    weighted_data = obj_with_data_var * obj.area       # element-wise mult 
    
    if 'z_t' in obj_with_data_var.dims: 
        mask = obj.mask
        masks = obj.masks
    else: # alternative if no z_t in obj_with_data_var, otherwise a z_t dimension would be added to obj_with_data_var 
        mask = obj.mask.isel(z_t=0)
        masks = obj.masks.isel(z_t=0)
    
    ## find area weights:
    # first we construct a data_var_no0, only used for correct shape and mask of required area object
    # replace 0 by 1 since we don't want to replace 0 with nan or divide by 0 in next step: 
    if basin == 'pac':
        data_var_no0 = obj_with_data_var.where(mask==2).where(obj_with_data_var != 0.00000000, 1.0)         
    elif basin == 'atl':
        data_var_no0 = obj_with_data_var.where(mask==1).where(obj_with_data_var != 0.00000000, 1.0)
    elif basin == 'so':
        data_var_no0 = obj_with_data_var.where(mask==4).where(obj_with_data_var != 0.00000000, 1.0)
    elif basin == 'pacso':
        data_var_no0 = obj_with_data_var.where(masks==2).where(obj_with_data_var != 0.00000000, 1.0) 
    elif basin == 'atlso':
        data_var_no0 = obj_with_data_var.where(masks==1).where(obj_with_data_var != 0.00000000, 1.0)
    elif basin == "":
        data_var_no0 = obj_with_data_var.where(obj_with_data_var != 0.00000000, 1.0) 
    else:
        raise Exception("basin should be empty '' or one out of: 'pac', 'atl', 'pacso', 'atlso', 'so'.")
    area = obj.area * data_var_no0 / data_var_no0    # expand area to dimensions & MASK of data_var
    # important; otherwise sum of weights still includes land (global avg SST will be 4 degrees too low)

    if keep_lat:
        weights = area.sum(dim='lon_t')            # one value for each z_t,time
        return weighted_data.sum(dim='lon_t') / weights.where(weights != 0) # weighted average
    elif keep_lon:
        weights = area.sum(dim='lat_t')            # one value for each z_t,time
        return weighted_data.sum(dim='lat_t') / weights.where(weights != 0) # weighted average
    else:
        weights = area.sum(dim='lat_t').sum(dim='lon_t')            # one value for each z_t,time
        return weighted_data.sum(dim='lon_t').sum(dim='lat_t') / weights.where(weights != 0) # weighted average


def convert_ticks_of_map(ax, Bern3D_grid=True, font=None):
    """Usage: for a map (e.g. trajectory plot) i.e. longitude on x-axis & latitude on y-axis.
    Makes ticklabels nicer e.g. -60 => 60°W.

    Input:
    - ax object (longitude on x-axis & latitude on y-axis)
    - if Bern3D_grid, longitude values are in [100,460] but ticks displayed as e.g. 30°W, 60°E etc.
    - font can be set
    Output:
    Ticks are changed on ax object and returns:
    - new ax object

    Author: Gunnar Jansen (gunnar.jansen@unibe.ch)
    Edited by Jeemijn Scheen (jeemijn.scheen@nioz.nl)"""

    # get current ticks
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    
    # find new labels
    xlabels = convert_lon_ticks(xticks, Bern3D_grid=Bern3D_grid)
    ylabels = convert_lat_ticks(yticks)
    
    # set new labels
    if font is None:
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)
    else:
        ax.set_xticklabels(xlabels, font=font)
        ax.set_yticklabels(ylabels, font=font)
    
    return ax


def convert_ticks_of_section_plot(ax, x_is_lon, Bern3D_grid=True, 
                                  font=None, explicit_xticklabels=None):
    """Usage: to convert x-axis tick only, which can be lat OR lon (e.g. a section plot).
    Makes ticklabels nicer e.g. -60 => 60°W.
    
    Input:
    - ax object (lon OR lat on x-axis & a variable on y-axis)
    - boolean x_is_lon: set True if x-axis is longitude, False if latitude
    - if Bern3D_grid, longitude values are in [100,460] but ticks displayed as e.g. 30°W, 60°E etc.
    - font can be set
    - explicit_xticklabels (list of floats) can be set instead of inferring from ax

    Output:
    Ticks are changed on ax object and returns:
    - new ax object

    Author: Gunnar Jansen, gunnar.jansen@unibe.ch"""

    from numpy import asarray
    
    # get current ticks
    if explicit_xticklabels is None:
        xticks = ax.get_xticks()
    else:
        xticks = asarray(explicit_xticklabels)
    
    # find new x labels (lon OR lat)
    if x_is_lon:
        xlabels = convert_lon_ticks(xticks, Bern3D_grid=Bern3D_grid)
    else:
        xlabels = convert_lat_ticks(xticks)
    
    # set new labels
    if font is None:
        ax.set_xticklabels(xlabels)
    else:
        ax.set_xticklabels(xlabels, font=font)
    
    return ax


def convert_lat_ticks(lat_ticks):
    """Make latitude ticks nicer
    Input:
    - list lat_ticks with floats
    Output:
    - list lat_ticklabels rounded to integers e.g. ['30°S', '0°', '30°N']
    Author: Gunnar Jansen, gunnar.jansen@unibe.ch"""

    from numpy import sign
    
    # int() rounds ° to entire integers e.g. 53° instead of 52.5°
    labels = [str(round(s)) + r'$^{\circ}$' for s in abs(lat_ticks)]
    suffix = ['N' if s == 1 else 'S' if s == -1 else '' for s in sign(lat_ticks)]
    lat_ticklabels = [l+n for l, n in zip(labels, suffix)]    

    return lat_ticklabels


def convert_lon_ticks(lon_ticks, Bern3D_grid=True):
    """Make longitude ticks nicer
    Input:
    - list lon_ticks with floats
    - if Bern3D_grid convert from lon [100,460] to [-180,180]
    Output:
    - list lon_ticklabels rounded to integers e.g. ['30°W', '0°', '30°E']
    Author: Gunnar Jansen, gunnar.jansen@unibe.ch"""

    from numpy import asarray, sign

    if Bern3D_grid: # convert [100,460] grid to [-180,180]
        lon_ticks = asarray([lon-360 if lon >= 180 else lon for lon in lon_ticks])
    
    labels = [str(round(s)) + r'$^{\circ}$' for s in abs(lon_ticks)]
    # round() rounds ° to entire integers where applicable e.g. 53° instead of 52.5°
    suffix = ['W' if s == -1 else 'E' if s == 1 else '' for s in sign(lon_ticks)]
    lon_ticklabels = [l+n for l, n in zip(labels, suffix)]

    return lon_ticklabels


def plot_surface(fig, ax, x, y, z, title="", grid='T', cbar=True, cbar_label='', cmap=None, 
                 vmin=None, vmax=None, ticklabels=False, bern3d_grid=True):
    """Makes a (lat,lon) plot using pcolormesh. 
    Input:
    - fig and ax must be given
    - x and y are either obj.lon_u and obj.lat_u (if var on T grid) or obj.lon_t and obj.lat_t (if var on U grid)
    - z must be a lat x lon array
    Optional input:
    - title
    - grid can be 'T' [default] (if z values on lon_t x lat_t) or 'U' (if on lon_u x lat_u)
    - cbar determines whether a cbar is plotted [default True]
    - cbar_label can be set
    - cmap gives colormap to use
    - vmin, vmax give min and max of colorbar
    - ticklabels prints the tick labels of lat/lon
    - if bern3d_grid is False, then less checks are possible as dimensions of lat, lon are unknown
    Output:
    - [ax, cbar] axis and cbar object are given back.
    Example usage:
    ax[0] = plot_surface(fig, ax[0], obj.lon_u, obj.lat_u, obj.TEMP.isel(z_t=0, time=0).values) 
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""
    
    if grid == 'T':
        if bern3d_grid:
            if (len(x) != 42) or (len(y) != 41) or (z.shape != (40,41)):
                raise Exception("x,y must be on u-grid and z on T-grid (if var is not on T-grid, then set: grid='U')")
        if cmap is None:
            cpf = ax.pcolormesh(x, y, extend(z), vmin=vmin, vmax=vmax, shading='nearest') # pcolor takes default for vmin=None
            # about shading: in python 3.8 I got a deprecationwarning for default shading='flat' when X and Y have the same dimensions as Z
            # so I added shading='nearest'. source: https://matplotlib.org/stable/gallery/images_contours_and_fields/pcolormesh_grids.html
        else:
            cpf = ax.pcolormesh(x, y, extend(z), cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
    elif grid == 'U':
        # U grid implemented analogously but not tested yet
        if bern3d_grid:
            if (len(x) != 41) or (len(y) != 40) or (z.shape != (41,42)):
                raise Exception("x,y must be on T-grid and z on U-grid (if var is not on U-grid, then set: grid='T')")
        if cmap is None:
            cpf = ax.pcolormesh(x, y, z, vmin=vmin, vmax=vmax)        
        else:
            cpf = ax.pcolormesh(x, y, extend(z), cmap=cmap, vmin=vmin, vmax=vmax)            
    else:
        raise Exception("Set grid to 'T' or 'U'.")
    
    ax.set_title(title)
    if not ticklabels:
        ax.tick_params(labelbottom=False, labelleft=False)  # remove tick labels with lat/lon values
    if cbar:
        cbar = fig.colorbar(cpf, ax=ax, label=cbar_label)
        return [ax, cbar]
    else:
        return [ax, cpf]
    

def extend(var):
    """
    Adds one element of rank-2 matrices to be plotted by pcolor
    Author: Raphael Roth
    """
    
    from numpy import ma as ma
    from numpy import ones, nan
    
    [a,b] = var.shape
    field = ma.masked_invalid(ones((a+1,b+1))*nan)
    field[0:a,0:b] = var
    return field

def create_land_mask(obj, data_full):
    """Creates land mask suitable for a (lat,z)-plot or (lon,lat)-plot.
    Input:
    - obj from whose nan values to create land mask e.g. any var masked to atlantic basin 
    obj should either have lat_t and z_t coord OR lat_u and z_w coord OR lon_t and lat_t coord
    - data_full: xarray data set that contains respective coords on u,v,w grid
    Output: 
    - [mask, cmap_land]
    NB cmap_land is independent of the land mask itself but just needed for plotting
    Usage example for (lat,z)-plot:
    - plot land by:
    X, Y = np.meshgrid(data_full.lat_u.values, data_full.z_w.values)
    if obj has z_t, lat_t coord:
        ax[i].pcolormesh(X,Y,extend(mask), cmap = cmap_land, vmin = -0.5, vmax = 0.5)
    if obj has z_w, lat_u coord: 
        identical but without extend()"""

    from numpy import isnan, ma, unique
    from matplotlib import pyplot as plt
    
    if 'time' in obj.dims:
        obj = obj.isel(time = 0)
    if not( ('lat_t' in obj.dims and 'z_t' in obj.dims) or ('lat_u' in obj.dims and 'z_w' in obj.dims) 
          or ('lon_t' in obj.dims and 'lat_t' in obj.dims)):
        raise Exception("obj should have either (z_t,lat_t) or (z_w,lat_u) or (lon_t,lat_t) coord")
    if unique(isnan(obj)).any() == False:
        raise Exception("obj has no nan values. Change 0 values to nan values first (if appropriate).")
    
    mask = isnan(obj) 
    mask = ma.masked_where(mask == False, mask) # change False values to nan such that cmap.set_bad picks them
    cmap_land = plt.cm.Greys.copy() 
    # cmap_land.set_bad(color='white') # where there is still ocean but no contour interpolation
    cmap_land.set_bad(color='#FF000000') # transparent

    return [mask, cmap_land]


def get_landmask(dataset):
    """creates a surface land mask suitable for plotting T-grid variables
    Input:
    - dataset (xarray containing .mask variable)
    Output:
    - [xu, yu, land_mask_surf (xarray containing surface land mask)]
    """

    land_mask = dataset.mask.where(dataset.mask == 0)
    land_mask_surf = land_mask.sel(z_t=land_mask.z_t[0])
    xu = dataset.lon_u # use u/v grid because pcolormesh() needs corners of the wished (T) grid
    yu = dataset.lat_u
    return [xu, yu, land_mask_surf]


def generate_1_section_plot_fig(dataset, variable, cruise, basic_data, title='', this_vmin=0.0,
                                vmaxs={'Pad_Bq' : 5, 'Thd_Bq' : 26, 'Pap_Bq' : 0.2, 'Thp_Bq' : 3, 'path_ratio_d' : 1, 'path_ratio_p' : 0.2},
                                levels=10, cbar_extend_both=False, obs=True, avoid_negative=False, verbose=False):
    """Input:
    A) relevant settings:
    - dataset=data_fulls[runs[n]]
    - variable must be Pad [if in dpm/m3 wanted] etc or Pad_Bq or path_ratio_d or path_ratio_p
    - cruise e.g. 'GA02' etc.
    - title can be given in. Usually enter: title=var_label[this_var]+' ('+labels[runs[n]]+')'
    - float this_vmin [default 0] is lower boundary for all cbars
    - dict vmaxs [default values] is upper boundary for all cbars
    - levels [default 10] is nr of contour levels
    - cbar_extend_both [default False] forces a colorbar with arrows in both directions
    - obs [default True] is whether to plot obs data
    - avoid_negative [default False] plots (rare) negative concentrations as 0
    - verbose [default False] is whether to print lon/lat coordinates
    
    B) also give in things generally available in the notebook:
    
    basic_data = [time, model_coords_per_cruise, cmap, obs_d_ave, obs_p_ave]
    
    - t is time index (usually -1)
    - model_coords_per_cruise=model_coords_per_cruise
    - cmap=cmap
    - obs_d_ave=obs_d_ave
    - obs_p_ave=obs_p_ave
    
    Output: 
    - fig handle (to be used with pdfjam)
    """
    
    ## add clarifying labels per cruise which give the rough direction
    cruise_directions = {'GIPY05' : ['West','East'], 'GA10' : ['West','East'], 'GA03' : ['West','East'], 'deng' : ['West','East'],
                         'GPc01' : ['West','East'], 'GP16' : ['West','East'], 'GN02' : ['West','East'],
                         'GIPY04' : ['South', 'North'], 'GA02' : ['South', 'North'], 'GAc02' : ['South', 'North']}

    from plot_gruber import Gruber
    import matplotlib.pyplot as plt
    
    assert len(basic_data) == 5, "smt wrong with basic_data"
    [t, model_coords_per_cruise, cmap, obs_d_ave, obs_p_ave] = basic_data

    ## plot section for this var
    # create nice variable labels
    if variable[:-1] == 'path_ratio_':
        var_label = ''
    elif variable[-3:] == '_Bq':
        var_label = '[$\mu$Bq/kg]'
    else:
        var_label = '[$dpm/m$^3$]'
    # prepare xticks
    nr_water_columns = len(model_coords_per_cruise[cruise])
    if nr_water_columns < 10:
        # tick at every integer
        xticks = range(1, nr_water_columns+1, 1)
    else:
        # ticks in steps of 5 + start + end
        xticks = [1] + list(range(5, nr_water_columns, 5)) + [nr_water_columns]
    # generate plot object
    var_section = Gruber(dataset=dataset, variable=variable, time=t, title=title,
                         cmap=cmap, clabel=var_label, cmin=this_vmin, cmax=vmaxs[variable], levels=levels,
                         section_lat=model_coords_per_cruise[cruise][:,1],
                         section_lon=model_coords_per_cruise[cruise][:,0],
                         section_xticks=[xticks]) # xticks needs to be list inside list
    # plot
    fig = var_section.plot(trajectory_info=False, input_all_cells=True, cruise=cruise, 
                           cbar_extend_both=cbar_extend_both, avoid_negative=avoid_negative)
    # add obs
    if obs:
        # for path_ratio_p/d to work, var path_ratio_p/d should be added to the obs_p/d_ave object. 
        if variable[2] == 'd' or variable == 'path_ratio_d':  # dissolved
            fig = add_sw_obs(fig, var=variable, obs_ave=obs_d_ave, cruise=cruise, cmap=cmap, 
                             vmin=this_vmin, vmax=vmaxs[variable], 
                             model_coords_of_cruise=model_coords_per_cruise[cruise], verbose=verbose)
        elif variable[2] == 'p' or variable == 'path_ratio_p': # p-bound
            fig = add_sw_obs(fig, var=variable, obs_ave=obs_p_ave, cruise=cruise, cmap=cmap, 
                             vmin=this_vmin, vmax=vmaxs[variable], 
                             model_coords_of_cruise=model_coords_per_cruise[cruise], verbose=verbose)
        else:
            raise Exception("variable "+var+" unknown")
    
    if cruise in cruise_directions.keys():
        axlist = plt.gcf().get_axes()
        ax_deep_ocean = axlist[1]  # 2nd subplot is deep ocean; 3th is cbar
        ax_deep_ocean.text(0.0, -0.26, cruise_directions[cruise][0], transform=ax_deep_ocean.transAxes, 
                           size=14, fontstyle='italic', ha='left')
        ax_deep_ocean.text(1.0, -0.26, cruise_directions[cruise][1], transform=ax_deep_ocean.transAxes, 
                           size=14, fontstyle='italic', ha='right')

    return fig

def generate_trajectory_fig(cruise, basic_data, verbose=False, title=''):
    """Input:
    A) relevant settings:
    - cruise e.g. 'GA02' etc.
    
    B) also give in things generally available in the notebook:
    
    basic_data = [dataset, time, model_coords_per_cruise, cmap]
    
    - dataset=data_fulls[runs[n]] (which run does not matter; using only grid)
    - t is time index (usually -1)
    - model_coords_per_cruise=model_coords_per_cruise
    - cmap=cmap
    - verbose prints coordinates
    - title can be given in (default title is cruise name)
    
    Output: 
    - fig handle (to be used with pdfjam)
    """
    from plot_gruber import Gruber

    if cruise in ['GA02', 'GAc02','GA03', 'GA10', 'GIPY04', 'GIPY05', 'deng', 'ng']:
        zoom_atl = True
    else:
        zoom_atl = False
    if cruise in ['GP16', 'GPc01', 'GSc02', 'pavia']:
        zoom_pac = True
    else:
        zoom_pac = False

    assert len(basic_data) == 4, "smt wrong with basic_data"
    [dataset, t, model_coords_per_cruise, cmap] = basic_data
    
    if title == '': 
        # default title is cruise name
        if cruise in ['ng', 'deng', 'pavia']:
            this_title = cruise.capitalize() + " et al."
        else:
            this_title = cruise
    else:
        this_title = title

    # test for temperature; use cruise as title here because it is for a trajectory plot
    test_section = Gruber(dataset=dataset, variable='TEMP', time=t, title=this_title, cmap=cmap, 
                          clabel=r'Temperature @'+str(round(dataset.time[t].item()))+'yr',
                          cmin=-5, cmax=30, levels=10,
                          # the below does nothing for plot_trajectory() at the moment but it does for the other plotting fnc's
                          section_lat=model_coords_per_cruise[cruise][:,1],
                          section_lon=model_coords_per_cruise[cruise][:,0])
    fig = test_section.plot_trajectory(section_lat=model_coords_per_cruise[cruise][:,1],
                                       section_lon=model_coords_per_cruise[cruise][:,0], 
                                       verbose=verbose, input_all_cells=True)

    axis = fig.axes[0]           # expect only 1 axis
    axis.set_title(this_title) # title not working from Gruber above because that one is meant for Gruber plot
    if zoom_atl:
        axis.set_xlim(258,382)
        # change size of existing figure
        fig.set_size_inches(2.5, 4)
        # overwrite xticks
        axis.set_xticks([270,300,330,360,390])
        # add '°W' etc
        axis = convert_ticks_of_map(axis, Bern3D_grid=True)
        axis.tick_params(axis='x', labelsize=11)
        axis.tick_params(axis='y', labelsize=11)
    if zoom_pac:
        axis.set_xlim(100,304)
        # change size of existing figure
        fig.set_size_inches(2.5, 4)  # still as for Atl
        # overwrite xticks
        axis.set_xticks([135,180,225,270])  # every 45 degrees
        # add '°W' etc
        axis = convert_ticks_of_map(axis, Bern3D_grid=True)
        axis.tick_params(axis='x', labelsize=11)
        axis.tick_params(axis='y', labelsize=11)
    
    return fig

def plot_contour(obj, fig, ax, var='OPSI', levels=None, hi=None, lo=None, cbar=True, title='', 
                 cmap=None, add_perc=False, extend=None):
    """Makes a contour plot with x=lat, y=depth and for colors 3 variables are possible:
    1) var = 'OPSI' then colors = OPSI (overturning psi, stream function)
    2) var = 'TEMP' then a lat-lon plot is made e.g. for a temperature, which must be in cK.
    3) var = 'CONC' idem as OPSI but plots a concentration so no dotted streamline contours etc

    Input (required):
    - obj needs to be set to an xarray DataArray containing coords lat and z_t (values eg OPSI+GM_OPSI)
    - fig needs to be given (in order to be able to plot colorbar);  from e.g. fig,ax=plt.subplots(1,2)
    - ax needs to be set to an axis object;   use e.g. ax or ax[2] if multiple subplots
    
    Input (optional):
    - var: see 3 options above
    - levels, hi and lo can be set to nr of contours, highest and lowest value, respectively.
      NB if levels = None, automatic contour levels are used and the colorbar will not be nicely centered.
      NB if var='CONC', colorbar ticks are hardcoded to maximal 6 ticks
    - cbar can be plotted or not
    - title of the plot can be set
    - cmap sets colormap  (inspiration: Oranges, Blues, Purples, PuOr_r, viridis) [default: coolwarm]
    - add_perc adds a '%' after each colorbar tick label (=unit for dye concentrations)
    - extend tells the colorbar to extend: 'both', 'neither', 'upper' or 'lower' (default: automatic)
    
    Output:
    - plot is made on axis object
    - if cbar: cbar object is returned (allows you to change the cbar ticks)
    - else: cpf object is returned (allows you to make a cbar)
    
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""  
    
    from numpy import meshgrid, arange, ceil, concatenate, floor, asarray, unique, sort
    from matplotlib import ticker, colors
    
    # avoid the somewhat cryptic TypeError: "Input z must be at least a 2x2 array."
    if obj.shape[0] < 2:
        raise Exception("object needs to have at least 2 depth steps.")
    elif obj.shape[1] < 2:
        raise Exception("object needs to have at least 2 steps on the x axis of contour plot.")

    if cmap is None:
        cmap = 'coolwarm'
            
    # make x,y,z arrays for contour plot
    if var == 'OPSI':
        xlist = obj.lat_u.values
        ylist = obj.z_w.values
        unit = 'Sv'      # unit for colorbar label
    elif var == 'TEMP':
        xlist = obj.lon_t.values
        ylist = obj.lat_t.values
        unit = '[cK]'    # assuming obj is in centi-Kelvin
    elif var == 'CONC':  # only difference wrt OPSI: need T-grid
        xlist = obj.lat_t.values 
        ylist = obj.z_t.values
        unit = ''
    else:
        raise Exception('var must be equal to OPSI or TEMP or CONC')
    Z = obj.values 
    X, Y = meshgrid(xlist, ylist)
    
    # prepare array for contour levels:
    if levels != None: # use levels, hi, lo
        # define contour levels
        step = (hi-lo) / float(levels)         # go to float to avoid integer rounding (can => division by zero)  
        level_arr = arange(lo, hi+step, step) 
        level_arr[abs(level_arr) < 1e-4] = 0.  # to avoid contour level labels called '-0.0' 

        # set nr of decimals for labels of contour lines
        if asarray(level_arr).max() >= 10.0:
            fmt = '%1.0f' 
        else:
            fmt = '%1.1f'
        
        # PLOT CONTOUR LINES ACCORDING TO levels, hi, lo PARAMETERS
        if var == 'OPSI':
            # contour lines
            cp_neg = ax.contour(X,Y,Z, level_arr[level_arr<0], colors='k', linestyles='dashed', linewidths=0.5) 
            cp_pos = ax.contour(X,Y,Z, level_arr[level_arr>0], colors='k', linestyles='-', linewidths=0.5) 
            cp0 = ax.contour(X,Y,Z, level_arr[abs(level_arr)<1e-4], colors='k', linestyles='-', linewidths=1.5)
            # contour labels
            for cp in [cp_neg, cp_pos, cp0]:    # add level values to contour lines
                ax.clabel(cp, inline=True, fontsize=12, colors='k', use_clabeltext=True, fmt=fmt) 
                # NB fmt formats text (standard %1.3f); use_clabeltext sets labels parallel to contour line
        elif var == 'TEMP':
            # hardcoded in order to resemble Gebbie & Huybers 2019, Fig 2
            # => contourf on the far bottom will plot for every 2.5 cK
            # but for contour lines:
            # keep small steps within [-10,10] but go to steps of 10 outside that interval
            this_level_arr = concatenate((level_arr[abs(level_arr) <= 10.0], level_arr[level_arr%10==0]))
            this_level_arr = unique(sort(this_level_arr))

            cp = ax.contour(X, Y, Z, this_level_arr, colors='k', linestyles='-', linewidths=0.5) 
            # contour labels via dict comprehension 
            # round to integer if levels are not like -7.5,-2.5,2.5,7.5 s.t. no '.0' appears
            fmt_dict = {x : str(x) if x-floor(x) == 0.5 else str(int(round(x))) for x in this_level_arr}
            ax.clabel(cp, inline=True, fontsize=12, colors='k', use_clabeltext=True, fmt=fmt_dict) 
        elif var == 'CONC':             
            # contour lines
            cp_non0 = ax.contour(X, Y, Z, level_arr[abs(level_arr)>1e-4], colors='k', linestyles='-', linewidths=0.5) 
            cp0 = ax.contour(X,Y,Z, level_arr[abs(level_arr)<1e-4], colors='k', linestyles='-', linewidths=1.5) 
            # contour labels
            for cp in [cp_non0, cp0]:    # add level values to contour lines
                ax.clabel(cp, inline=True, fontsize=12, colors='k', use_clabeltext=True, fmt=fmt) 
                
    else: # PLOT CONTOUR LINES WITH AUTOMATIC LEVELS (colorbar ugly/not centered)
        cpf = ax.contourf(X, Y, Z, cmap=cmap) # contour fill
        cp = ax.contour(X, Y, Z, colors='k', linestyles='-', linewidths=0.5) # contour lines
        # contour labels
        ax.clabel(cp, inline=True, fontsize=12, colors='k', use_clabeltext=True) # labels
        
    # PLOT CONTOUR FILL
    if var == 'TEMP':
        zorder=0 # makes things on top possible, here: grid lines
    else:
        zorder=1
    if extend is None:
        cpf = ax.contourf(X, Y, Z, level_arr, cmap=cmap, zorder=zorder) 
    else:
        cpf = ax.contourf(X, Y, Z, level_arr, cmap=cmap, extend=extend, zorder=zorder)
        
    # COLOUR BAR
    if cbar:
        if var == 'TEMP':
            # hardcoded in order to resemble Gebbie & Huybers 2019 science, Fig 2
            cbar_obj = fig.colorbar(cpf, ax=ax, label=unit, orientation='horizontal', pad=0.15)
        else:
            cbar_obj = fig.colorbar(cpf, ax=ax, label=unit)  
        # set cbar ticks and labels
        if var == 'CONC':
            tick_locator = ticker.MaxNLocator(nbins=6) # to force 0-100% cbar to reasonable ticks without manually
            cbar_obj.locator = tick_locator
            cbar_obj.update_ticks()
        if add_perc:
            perc = "%"
        else:
            perc = ""
        cticks = cbar_obj.get_ticks()
        ctick_labels = ['0'+perc if abs(x) < 1e-4
                        else str(int(round(x)))+perc if hi >= 10.0 
                        else str(round(x,1))+perc if hi > 5.0 
                        else str(round(x,2))+perc for x in cticks]

        # gives ERROR: cant find it for now. Complains nr of FixedLocator (21) not equal to ticks to set (7)
        # cbar_obj.ax.set_yticklabels(ctick_labels)  
        
        # print(cticks)
        # print(ctick_labels)
        # print(cbar_obj.ax.get_yticklabels())          

    # LABELS AND TICKS
    ax.set_title(title)
    if var != 'TEMP':
        # for TEMP do nothing; the axes (lon, lat) are obvious from world map
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Depth [km]')
        ax.set_ylim(0,5)
        ax.invert_yaxis()    # depth goes down on y axis    
        ax.set_yticks(range(0,6,1))
    
    if cbar:
        return cbar_obj  # return already plotted cbar object s.t. it can be adjusted
    else:
        return cpf       # return object with which plotting cbar is possible
        
        
def plot_overturning(data, data_full, times, time_avg=False, atl=True, pac=False, sozoom=False, 
                     levels=None, lo=None, hi=None, land=True, all_anoms=False):
    """Plots figure of overturning stream function panels at certain time steps and basins.
    Columns:
    - a column for every t in times array
    Rows: 
    - if atl: overturning as measured only in Atlantic basin
    - if pac: overturning as measured only in Pacific basin
    - if sozoom: Southern Ocean sector of global overturning
    - global overturning (always plotted)
    
    Input:
    - data and data_full xarray datasets with depth in kilometers
    - times array with time indices, e.g., 50 stands for data_full.time[50]
    - time_avg [default False] plots a 30 year average around the selected time steps instead of the 1 annual value 
      NB for t=0 a 15 year average on the future side is taken
    - atl, pac and/or sozoom basins (rows; see above)
    - levels, lo and hi set the number of colour levels and min resp. max boundaries
    - land [optional] prints black land on top
    - all_anoms [optional] plots all values as anomalies w.r.t. t1 except the first column (t1)
      NB anomaly plots have a hardcoded colorbar between -2 and 2 Sv
    
    Output:
    - returns [fig, ax]: a figure and axis handle
        
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""

    # SETTINGS:
    so_bnd = -80  # S.O. southern boundary e.g. -90 or -80
    # color of land: black when (vmin,vmax) = (-0.5,0.5) and grey when (0.5,1.5) and light grey when (0.8,1.5)
    vmin = 0.8
    vmax = 1.5  
    
    from matplotlib.pyplot import subplots, suptitle, tight_layout
    from numpy import zeros, ceil, sum, nan, meshgrid
    
    row_nr = 1 + sum([atl, pac, sozoom]) # np.sum() gives nr of True values 
    col_nr = len(times)
        
    if all_anoms:
        # anomaly plots have a hardcoded colorbar between -2 and 2 Sv
        hi_anom = 2.0
        lo_anom = -2.0
        levels_anom = 10
           
    opsi_all_t = data_full.OPSI + data_full.GMOPSI      # total global overturning; still for all times
    opsi_a_all_t = data_full.OPSIA + data_full.GMOPSIA  # atlantic overturning
    opsi_p_all_t = data_full.OPSIP + data_full.GMOPSIP  # pacific overturning
       
    if land:     
        # in this case we want to replace 0 values (land) by nan values
        # such that the opsi variables are not plotted on land & the nan values are needed to plot land
        opsi_all_t = opsi_all_t.where(opsi_all_t != 0.0, nan)
        opsi_a_all_t = opsi_a_all_t.where(opsi_a_all_t != 0.0, nan)
        opsi_p_all_t = opsi_p_all_t.where(opsi_p_all_t != 0.0, nan)
        [mask_gl, cmap_land_gl]   = create_land_mask(opsi_all_t, data_full) 
        [mask_atl, cmap_land_atl] = create_land_mask(opsi_a_all_t, data_full) 
        [mask_pac, cmap_land_pac] = create_land_mask(opsi_p_all_t, data_full) 
        X, Y = meshgrid(data_full.lat_u.values, data_full.z_w.values) # same for all subplots    
    
    fig, ax = subplots(nrows=row_nr, ncols=col_nr, figsize=(14, 3*row_nr)) 
        
    for i in range(0,row_nr):
        for j in range(0,col_nr):
            ax[i,j].set_xticks(range(-75,80,25))
            
    ## now we go through the rows/basins one by one (if they exist); 
    ## within each we have a for loop over cols/times 
    ## where we save plotted objects for all times in opsi{} s.t. we can take the anomaly between them
            
    this_row = 0
    if atl:    
        opsi = {}  # keys=cols, vals=opsi plot object for that col/timestep
        for n,t in enumerate(times):
            # pick correct ax object
            if row_nr == 1: 
                this_ax = ax[n]
            else: 
                this_ax = ax[this_row,n] 

            # select correct data for time step and row in t_slice:
            if time_avg:
                # over 30 yrs e.g. 1750 now becomes average over [1735, 1765]
                opsi[n] = opsi_a_all_t.sel(time=slice(t-16,t+16)).mean(dim='time')
            else:
                opsi[n] = opsi_a_all_t.sel(time=t) 
            this_title = "Atlantic overturning"
            this_ax.set_xlim([-50,90]) # exclude S.O. 
            
            # plot
            if land:  
                this_ax.pcolormesh(X, Y, mask_atl, cmap=cmap_land_atl, vmin=vmin, vmax=vmax) 
            if all_anoms and n != 0:  # first row is never anomaly
                opsi_diff = opsi[n] - opsi[0]
                plot_contour(opsi_diff, fig, ax=this_ax, levels=levels_anom, lo=lo_anom, hi=hi_anom, 
                             var='OPSI', extend='both', title=this_title + ' anomaly')             
            else:
                plot_contour(opsi[n], fig, ax=this_ax, levels=levels, lo=lo, hi=hi, 
                             var='OPSI', extend='both', title=this_title)
        this_row += 1

    if pac:
        opsi = {}  # keys=cols, vals=opsi plot object for that col/timestep
        for n,t in enumerate(times):
            # pick correct ax object
            if row_nr == 1: 
                this_ax = ax[n]
            else: 
                this_ax = ax[this_row,n] 

            # select correct data for time step and row in t_slice:
            if time_avg:
                opsi[n] = opsi_p_all_t.sel(time=slice(t-16,t+16)).mean(dim='time')
            else:
                opsi[n] = opsi_p_all_t.sel(time=t) 
            this_title = "Indo-Pacific overturning"
            this_ax.set_xlim([-50,90]) # exclude S.O.
            
            # plot
            if land:  
                this_ax.pcolormesh(X, Y, mask_pac, cmap=cmap_land_pac, vmin=vmin, vmax=vmax) 
            if all_anoms and n != 0:  # first row is never anomaly
                opsi_diff = opsi[n] - opsi[0]
                plot_contour(opsi_diff, fig, ax=this_ax, levels=levels_anom, lo=lo_anom, hi=hi_anom, 
                             var='OPSI', extend='both', title=this_title + ' anomaly')             
            else:
                plot_contour(opsi[n], fig, ax=this_ax, levels=levels, lo=lo, hi=hi, 
                             var='OPSI', extend='both', title=this_title)                
        this_row += 1

    if sozoom:
        opsi = {}  # keys=cols, vals=opsi plot object for that col/timestep
        for n,t in enumerate(times):
            # pick correct ax object
            if row_nr == 1: 
                this_ax = ax[n]
            else: 
                this_ax = ax[this_row,n] 

            # select correct data for time step and row in t_slice:
            if time_avg:
                opsi[n] = opsi_all_t.sel(lat_u=slice(so_bnd, -50), time=slice(t-16,t+16)).mean(dim='time')
            else:
                opsi[n] = opsi_all_t.sel(lat_u=slice(so_bnd, -50), time=t)
            this_title = "Southern Ocean overturning"
            # make exception for SO (otherwise only 1 tick at -75)
            this_ax.set_xticks(range(-90,-45,10)) 
            
            # plot
            if land:  
                this_ax.pcolormesh(X, Y, mask_gl, cmap=cmap_land_gl, vmin=vmin, vmax=vmax) 
                this_ax.set_xlim([so_bnd,-50]) # ticks change to automatic but that seems fine
            if all_anoms and n != 0:  # first row is never anomaly
                opsi_diff = opsi[n] - opsi[0]
                plot_contour(opsi_diff, fig, ax=this_ax, levels=levels_anom, lo=lo_anom, hi=hi_anom, 
                             var='OPSI', extend='both', title=this_title + ' anomaly')             
            else:
                plot_contour(opsi[n], fig, ax=this_ax, levels=levels, lo=lo, hi=hi, 
                             var='OPSI', extend='both', title=this_title)
        this_row += 1

    ## always plot global overturning in last row
    opsi = {}  # keys=cols, vals=opsi plot object for that col/timestep
    for n,t in enumerate(times):
        # pick correct ax object
        if row_nr == 1: 
            this_ax = ax[n]
        else: 
            this_ax = ax[this_row,n] 

        # select correct data for time step and row in t_slice:
        if time_avg:
            opsi[n] = opsi_all_t.sel(time=slice(t-16,t+16)).mean(dim='time')
        else:
            opsi[n] = opsi_all_t.sel(time=t)
        this_title = "Global overturning"
        
        # plot
        if land:  
            this_ax.pcolormesh(X, Y, mask_gl, cmap=cmap_land_gl, vmin=vmin, vmax=vmax) 
        if all_anoms and n != 0:  # first row is never anomaly
            opsi_diff = opsi[n] - opsi[0]
            plot_contour(opsi_diff, fig, ax=this_ax, levels=levels_anom, lo=lo_anom, hi=hi_anom, 
                         var='OPSI', extend='both', title=this_title + ' anomaly')             
        else:
            plot_contour(opsi[n], fig, ax=this_ax, levels=levels, lo=lo, hi=hi, 
                         var='OPSI', extend='both', title=this_title)
        # print info on minimum and maximum
        if time_avg:
            print('@%1.0f CE: global MOC min=%1.2f Sv, AMOC max=%1.2f Sv' 
                  %(ceil(times[n]), data.OPSI_min.sel(time=slice(t-16,t+16)).mean(dim='time'),
                    data.OPSIA_max.sel(time=slice(t-16,t+16)).mean(dim='time')))
        else:
            print('@%1.0f CE: global MOC min=%1.2f Sv, AMOC max=%1.2f Sv' 
                  %(ceil(times[n]), data.OPSI_min.sel(time=t).item(), data.OPSIA_max.sel(time=t).item()))

    tight_layout()    
    
    return fig, ax


def plot_overturning_multiple_runs(runs, datas, data_fulls, time_step=-1, time_avg=True, 
                                   atl=True, pac=False, sozoom=False, 
                                   levels=None, lo=None, hi=None, land=True, all_anoms=False):
    """Plots figure of overturning stream function panels for multiple runs and basins.
    Columns:
    - a column for every run
    Rows: 
    - if atl: overturning as measured only in Atlantic basin
    - if pac: overturning as measured only in Pacific basin
    - if sozoom: Southern Ocean sector of global overturning
    - global overturning (always plotted)
    
    Input:
    - runs: list of runnames to plot
    - datas and data_fulls dictionaries of xarray datasets with (2D and 3D) model output (keys are runnames), with depth in kilometers
      NB these may include more runs than plotted; only the runnames in 'runs' list are used
    - time_step (index!) of time to plot
    - time_avg [default False] plots a 30 year average around the selected time steps instead of the 1 annual value 
      NB for t=0 a 15 year average on the future side is taken
    - atl, pac and/or sozoom basins (rows; see above)
    - levels, lo and hi set the number of colour levels and min resp. max boundaries
    - land [optional] prints black land on top
    - all_anoms [optional] plots all values as anomalies w.r.t. first run except the first run itself
      NB anomaly plots have a hardcoded colorbar between -2 and 2 Sv
    
    Output:
    - returns [fig, ax]: a figure and axis handle
        
    Author: Jeemijn Scheen, jeemijn.scheen@nioz.nl"""

    # SETTINGS:
    so_bnd = -80  # S.O. southern boundary e.g. -90 or -80
    # color of land: black when (vmin,vmax) = (-0.5,0.5) and grey when (0.5,1.5) and light grey when (0.8,1.5)
    vmin = 0.8
    vmax = 1.5  
    
    from matplotlib.pyplot import subplots, suptitle, tight_layout
    from numpy import zeros, ceil, sum, nan, meshgrid
    
    row_nr = 1 + sum([atl, pac, sozoom]) # np.sum() gives nr of True values 
    col_nr = len(runs)
        
    if all_anoms:
        # anomaly plots have a hardcoded colorbar between -2 and 2 Sv
        hi_anom = 2.0
        lo_anom = -2.0
        levels_anom = 10

    fig, ax = subplots(nrows=row_nr, ncols=col_nr, figsize=(14, 3*row_nr)) 

    for i in range(0,row_nr):
        for j in range(0,col_nr):
            ax[i,j].set_xticks(range(-75,80,25))

    if land:
        X, Y = meshgrid(data_fulls[runs[0]].lat_u.values, data_fulls[runs[0]].z_w.values) # same for all subplots    

    ## now we go through the rows/basins one by one (if they exist); 
    ## within each we have a for loop over cols/runs 
    ## results are saved in opsi dict in between to be able to take anomalies between runs
    this_row = 0
    if atl:    
        opsi = {}  # keys=cols, vals=opsi plot object for that col/run
        for n,run in enumerate(runs):
            data_full = data_fulls[run]
            # pick correct ax object
            if row_nr == 1: 
                this_ax = ax[n]
            else: 
                this_ax = ax[this_row,n] 

            # select correct data
            opsi_a_all_t = data_full.OPSIA + data_full.GMOPSIA  # atlantic overturning
            if land:     
                # in this case we want to replace 0 values (land) by nan values
                # such that the opsi variables are not plotted on land & the nan values are needed to plot land
                opsi_a_all_t = opsi_a_all_t.where(opsi_a_all_t != 0.0, nan)
                [mask_atl, cmap_land_atl] = create_land_mask(opsi_a_all_t, data_full) 

            if time_avg:
                # over 30 yrs e.g. 1750 now becomes average over [1735, 1765]
                opsi[n] = opsi_a_all_t.sel(time=slice(time_step-16,time_step+16)).mean(dim='time')
            else:
                opsi[n] = opsi_a_all_t.isel(time=time_step) 
            this_title = "Atlantic overturning"
            this_ax.set_xlim([-50,90]) # exclude S.O. 
            
            # plot
            if land:  
                this_ax.pcolormesh(X, Y, mask_atl, cmap=cmap_land_atl, vmin=vmin, vmax=vmax) 
            if all_anoms and n != 0:  # first row is never anomaly
                opsi_diff = opsi[n] - opsi[0]
                plot_contour(opsi_diff, fig, ax=this_ax, levels=levels_anom, lo=lo_anom, hi=hi_anom, 
                             var='OPSI', extend='both', title=this_title + ' anomaly')             
            else:
                plot_contour(opsi[n], fig, ax=this_ax, levels=levels, lo=lo, hi=hi, 
                             var='OPSI', extend='both', title=this_title)
        this_row += 1

    if pac:
        opsi = {}  # keys=cols, vals=opsi plot object for that col/run
        for n,run in enumerate(runs):
            data_full = data_fulls[run]
            # pick correct ax object
            if row_nr == 1: 
                this_ax = ax[n]
            else: 
                this_ax = ax[this_row,n] 

            # select correct data 
            opsi_p_all_t = data_full.OPSIP + data_full.GMOPSIP  # pacific overturning
            if land:     
                # in this case we want to replace 0 values (land) by nan values
                # such that the opsi variables are not plotted on land & the nan values are needed to plot land
                opsi_p_all_t = opsi_p_all_t.where(opsi_p_all_t != 0.0, nan)
                [mask_pac, cmap_land_pac] = create_land_mask(opsi_p_all_t, data_full) 
            if time_avg:
                opsi[n] = opsi_p_all_t.sel(time=slice(time_step-16,time_step+16)).mean(dim='time')
            else:
                opsi[n] = opsi_p_all_t.isel(time=time_step) 
            this_title = "Indo-Pacific overturning"
            this_ax.set_xlim([-50,90]) # exclude S.O.
            
            # plot
            if land:  
                this_ax.pcolormesh(X, Y, mask_pac, cmap=cmap_land_pac, vmin=vmin, vmax=vmax) 
            if all_anoms and n != 0:  # first row is never anomaly
                opsi_diff = opsi[n] - opsi[0]
                plot_contour(opsi_diff, fig, ax=this_ax, levels=levels_anom, lo=lo_anom, hi=hi_anom, 
                             var='OPSI', extend='both', title=this_title + ' anomaly')             
            else:
                plot_contour(opsi[n], fig, ax=this_ax, levels=levels, lo=lo, hi=hi, 
                             var='OPSI', extend='both', title=this_title)                
        this_row += 1

    if sozoom:
        opsi = {}  # keys=cols, vals=opsi plot object for that col/run
        for n,run in enumerate(runs):
            data_full = data_fulls[run]
            # pick correct ax object
            if row_nr == 1: 
                this_ax = ax[n]
            else: 
                this_ax = ax[this_row,n] 

            # select correct data 
            opsi_all_t = data_full.OPSI + data_full.GMOPSI      # total global overturning; still for all times       
            if land:     
                # in this case we want to replace 0 values (land) by nan values
                # such that the opsi variables are not plotted on land & the nan values are needed to plot land
                opsi_all_t = opsi_all_t.where(opsi_all_t != 0.0, nan)
                [mask_gl, cmap_land_gl] = create_land_mask(opsi_all_t, data_full) 
            if time_avg:
                opsi[n] = opsi_all_t.sel(lat_u=slice(so_bnd, -50), time=slice(time_step-16,time_step+16)).mean(dim='time')
            else:
                opsi[n] = opsi_all_t.isel(lat_u=slice(so_bnd, -50), time=time_step)
            this_title = "Southern Ocean overturning"
            # make exception for SO (otherwise only 1 tick at -75)
            this_ax.set_xticks(range(-90,-45,10)) 
            
            # plot
            if land:  
                this_ax.pcolormesh(X, Y, mask_gl, cmap=cmap_land_gl, vmin=vmin, vmax=vmax) 
                this_ax.set_xlim([so_bnd,-50]) # ticks change to automatic but that seems fine
            if all_anoms and n != 0:  # first row is never anomaly
                opsi_diff = opsi[n] - opsi[0]
                plot_contour(opsi_diff, fig, ax=this_ax, levels=levels_anom, lo=lo_anom, hi=hi_anom, 
                             var='OPSI', extend='both', title=this_title + ' anomaly')             
            else:
                plot_contour(opsi[n], fig, ax=this_ax, levels=levels, lo=lo, hi=hi, 
                             var='OPSI', extend='both', title=this_title)
        this_row += 1

    ## always plot global overturning in last row
    opsi = {}  # keys=cols, vals=opsi plot object for that col/run
    for n,run in enumerate(runs):
        data = datas[run]
        data_full = data_fulls[run]
        # pick correct ax object
        if row_nr == 1: 
            this_ax = ax[n]
        else: 
            this_ax = ax[this_row,n] 

        # select correct data
        opsi_all_t = data_full.OPSI + data_full.GMOPSI      # total global overturning; still for all times       
        if land:     
            # in this case we want to replace 0 values (land) by nan values
            # such that the opsi variables are not plotted on land & the nan values are needed to plot land
            opsi_all_t = opsi_all_t.where(opsi_all_t != 0.0, nan)
            [mask_gl, cmap_land_gl] = create_land_mask(opsi_all_t, data_full) 
        if time_avg:
            opsi[n] = opsi_all_t.sel(time=slice(time_step-16,time_step+16)).mean(dim='time')
        else:
            opsi[n] = opsi_all_t.isel(time=time_step)
        this_title = "Global overturning"
        
        # plot
        if land:  
            this_ax.pcolormesh(X, Y, mask_gl, cmap=cmap_land_gl, vmin=vmin, vmax=vmax) 
        if all_anoms and n != 0:  # first row is never anomaly
            opsi_diff = opsi[n] - opsi[0]
            plot_contour(opsi_diff, fig, ax=this_ax, levels=levels_anom, lo=lo_anom, hi=hi_anom, 
                         var='OPSI', extend='both', title=this_title + ' anomaly')             
        else:
            plot_contour(opsi[n], fig, ax=this_ax, levels=levels, lo=lo, hi=hi, 
                         var='OPSI', extend='both', title=this_title)
        
        # print info on minimum and maximum
        if time_avg:
            print('15-30 yr average around %1.0f CE: global MOC min=%1.2f Sv, AMOC max=%1.2f Sv' 
                  %(ceil(data_full.time[time_step]),
                    data.OPSI_min.sel(time=slice(data_full.time[time_step]-16,data_full.time[time_step]+16)).mean(dim='time'),
                    data.OPSIA_max.sel(time=slice(data_full.time[time_step]-16,data_full.time[time_step]+16)).mean(dim='time')))
        else:
            print('@%1.0f CE: global MOC min=%1.2f Sv, AMOC max=%1.2f Sv' 
                  %(ceil(data_full.time[time_step]),
                    data.OPSI_min.sel(time=data_full.time[time_step]).item(), 
                    data.OPSIA_max.sel(time=data_full.time[time_step]).item()))

    tight_layout()    
    
    return fig, ax
