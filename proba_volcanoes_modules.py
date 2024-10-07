import proba_modules as pm
import numpy as np
from pyrocko import moment_tensor as mtm
from importlib import reload
import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from tqdm import tqdm

# Example latitude-dependent velocity function
def vlat_func(latitude):
    # Example: vlat decreases as latitude increases
    return -4*np.sign(latitude) *np.exp(-(latitude/35)**2)/110.
    #return -0.4*np.exp(-(latitude/35)**2)

def wrap_latitude(lat_rad):
    # Wrap latitude values correctly after crossing the poles
    lat_wrapped = np.arcsin(np.sin(lat_rad))
    return lat_wrapped

def wrap_longitude(lon_rad):
    # Wrap longitude values within the range [-pi, pi]
    lon_wrapped = (lon_rad + np.pi) % (2 * np.pi) - np.pi
    return lon_wrapped

def compute_positions_vectorized_w_interpolator(lat0, lon0, wind_direction_interpolator, wind_strength_interpolator, times, R0):
    # Convert initial latitude and longitude from degrees to radians
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    
    # Initialize arrays to store results
    lat_rad = lat0_rad[:,0]
    lon_rad = lon0_rad[:,0]
    new_lat_rad = np.zeros_like(times, dtype=float)
    new_lon_rad = np.zeros_like(times, dtype=float)
    
    # 
    # Compute positions for each time step
    for i, time in tqdm(enumerate(times[0,:]), total=times.shape[1]):
        
        # Calculate the latitude-dependent latitude velocity
        #vlat = vlat_func(np.degrees(lat_rad))
        az = wind_direction_interpolator.ev(np.degrees(lat_rad), np.degrees(lon_rad))
        w_strength = wind_direction_interpolator.ev(np.degrees(lat_rad), np.degrees(lon_rad))
        vlon = np.cos(lon_rad)*w_strength
        vlat = np.sin(lon_rad)*w_strength
        
        # Calculate the distance traveled in latitude and longitude
        dlat = vlat * time
        dlon = vlon * time
        
        # Convert distances into angular displacements (in radians)
        delta_lat_rad = dlat / R0
        delta_lon_rad = dlon / (R0 * np.cos(lat_rad))
        
        # Calculate the new latitude and longitude in radians
        lat_rad += delta_lat_rad
        lon_rad += delta_lon_rad
        
        # Wrap the latitude and longitude values
        lat_rad = wrap_latitude(lat_rad)
        lon_rad = wrap_longitude(lon_rad)
        
        # Store the results
        new_lat_rad[:,i] = lat_rad
        new_lon_rad[:,i] = lon_rad
    
    # Convert the final latitude and longitude back to degrees
    new_lat = np.degrees(new_lat_rad)
    new_lon = np.degrees(new_lon_rad)
    
    return new_lat.ravel(), new_lon.ravel()

def compute_positions_vectorized(lat0, lon0, vlat_func, vlon, times, R0):
    # Convert initial latitude and longitude from degrees to radians
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    
    # Initialize arrays to store results
    lat_rad = lat0_rad[:,0]
    lon_rad = lon0_rad[:,0]
    new_lat_rad = np.zeros_like(times, dtype=float)
    new_lon_rad = np.zeros_like(times, dtype=float)
    
    # Compute positions for each time step
    for i, time in tqdm(enumerate(times[0,:]), total=times.shape[1]):
        
        # Calculate the latitude-dependent latitude velocity
        vlat = vlat_func(np.degrees(lat_rad))
        
        # Calculate the distance traveled in latitude and longitude
        dlat = vlat * time
        dlon = vlon * time
        
        # Convert distances into angular displacements (in radians)
        delta_lat_rad = dlat / R0
        delta_lon_rad = dlon / (R0 * np.cos(lat_rad))
        
        # Calculate the new latitude and longitude in radians
        lat_rad += delta_lat_rad
        lon_rad += delta_lon_rad
        
        # Wrap the latitude and longitude values
        lat_rad = wrap_latitude(lat_rad)
        lon_rad = wrap_longitude(lon_rad)
        
        # Store the results
        new_lat_rad[:,i] = lat_rad
        new_lon_rad[:,i] = lon_rad
    
    # Convert the final latitude and longitude back to degrees
    new_lat = np.degrees(new_lat_rad)
    new_lon = np.degrees(new_lon_rad)
    
    return new_lat.ravel(), new_lon.ravel()

def haversine_distance(lat1, lon1, lat2, lon2, R):
    # Convert latitudes and longitudes from degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    
    return distance/1e3

def get_amps_at_baloons(T0s_offset, LAT_offset, ID_LAT0, TIMES, times, shape_TIMES, all_times, all_mags, distances, TL_new, arrival_time, batch_size):

    #batch_size = LAT_offset.size//2500
    ids_start = np.arange(batch_size, LAT_offset.size+batch_size, batch_size)
    mags_ev, amps_ev, mask = np.zeros((all_times.size, LAT_offset.size)), np.zeros((all_times.size, LAT_offset.size)), np.zeros((all_times.size, LAT_offset.size), dtype=bool)
    id_start = 0
    for id_end in tqdm(ids_start[:]):
        current_batch_size = np.arange(LAT_offset.size)
        current_batch_size = current_batch_size[(current_batch_size>=id_start)&(current_batch_size<id_end)].size
        loc_idx = np.where((ID_LAT0>=id_start)&(ID_LAT0<id_end))
        ID_TIMES, ID_ALL_TIMES = np.meshgrid(np.arange(TIMES.size)[loc_idx], np.arange(all_times.size))
        arrival_times = arrival_time(distances[ID_TIMES], 50., (all_times[ID_ALL_TIMES])*365*24*3600)/(365*24*3600)
        #all_shape = (all_times.size,)+shape_TIMES # ev x balloon init loc/t0 x balloon flight time 
        all_shape = (all_times.size, current_batch_size, shape_TIMES[1]) # ev x balloon init loc/t0 x balloon flight time 
        balloon_times = T0s_offset[ID_LAT0[ID_TIMES]]+TIMES[ID_TIMES]/(365*24*3600)
        offset_okay = times[1]/(365*24*3600)
        mask_loc = (balloon_times>=arrival_times-offset_okay)&(balloon_times<=arrival_times+offset_okay)
        
        id_event = abs(balloon_times-arrival_times).reshape(all_shape).argmin(axis=-1)[:,:,None]
        mask[:,id_start:id_end] = np.take_along_axis(mask_loc.reshape(all_shape), id_event, axis=-1)[:,:,0]
        arrival_times_ev = np.take_along_axis(arrival_times.reshape(all_shape), id_event, axis=-1)[:,:,0]
        distances_ev = np.take_along_axis(distances[ID_TIMES].reshape(all_shape), id_event, axis=-1)[:,:,0]
        
        #print(all_shape, (all_times.size,)+shape_TIMES, balloon_times.shape, TIMES[ID_TIMES].shape, distances_ev.shape)
        mags_ev[:,id_start:id_end] = np.take_along_axis(all_mags[ID_ALL_TIMES].reshape(all_shape), id_event, axis=-1)[:,:,0]
        amps_ev[:,id_start:id_end] = TL_new(distances_ev, mags_ev[:,id_start:id_end])
        id_start = id_end

    return mags_ev, amps_ev, mask


import matplotlib.colors as colors

def add_cbar(ax, sc, label, fontsize=12.):

    axins = inset_axes(ax, width="70%", height="2.5%", loc='lower left', 
                    bbox_to_anchor=(0.15, 1.01, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False, labelrotation=90.)
    cbar = plt.colorbar(sc, cax=axins, orientation="horizontal")  
    cbar.ax.xaxis.set_ticks_position('top') 
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.tick_top()
    cbar.ax.set_xlabel(label, labelpad=2, fontsize=fontsize) 

def plot_proba_sequence(amps_ev, all_times, all_mags, TL_new, lat_vol, t0s_offset, lat_offset, LAT_offset_shape, mask, noise_level = 0.01, factor = (np.log10(2.)+4.)/4., snr_threshold=1, fontsize=12.):

    dists = np.logspace(0, 4.*factor, 100)
    
    cmap = plt.cm.coolwarm  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    idamp = (amps_ev*(mask)).argmax(axis=0)[None,:]  
    number_over_snr = ((amps_ev*(mask)/noise_level)>snr_threshold).sum(axis=0).reshape(LAT_offset_shape)
    #print(amps_ev*(mask) + 1e-10*(~mask), amps_ev.shape, mask.shape, idamp.shape, LAT_offset_shape, idamp)
    amps_ev_reshaped = np.take_along_axis(amps_ev*(mask) + 1e-10*(~mask), idamp, axis=0)[0].reshape(LAT_offset_shape) # lon x lat x t0
    #distances_ev_reshaped = np.take_along_axis(distances_ev*(mask), idamp, axis=0)[0].reshape(LAT_offset_shape) # lon x lat x t0
    #mags_ev_reshaped = np.take_along_axis(mags_ev*(mask), idamp, axis=0)[0].reshape(LAT_offset_shape) # lon x lat x t0
    ID_TIMES_EV, DISTS = np.meshgrid(np.arange(all_times.size), dists)

    fig = plt.figure(figsize=(11,11))
    grid = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(grid[0,0])
    ax.scatter(all_times, all_mags, c=all_mags, cmap='magma', vmin=1., vmax=all_mags.max())
    ax.set_xlabel('Time (years since main shock)')
    ax.set_ylabel('Magnitude (Mw)')

    ax = fig.add_subplot(grid[0,1], sharex=ax)
    ZTL = TL_new(DISTS.T, all_mags[ID_TIMES_EV].T)/noise_level
    #sc = ax.contourf(all_times[ID_TIMES_EV], DISTS, TL_new(DISTS, all_mags[ID_TIMES_EV])/noise_level, levels=[0.1, 0.5, 1, 5, 10], locator=ticker.LogLocator(), cmap='Blues', )
    sc = ax.pcolormesh(all_times, dists, ZTL.T, norm=colors.LogNorm(vmin=0.1, vmax=10), cmap=cmap)
    ax.set_xlabel('Time (years since main shock)', fontsize=fontsize)
    ax.set_ylabel('Distance (km)', fontsize=fontsize)
    ax.set_yscale('log')
    add_cbar(ax, sc, f'Peak SNR', fontsize=fontsize)
    ax.set_facecolor(cmaplist[0])

    ax = fig.add_subplot(grid[1,1], sharex=ax)
    Z = amps_ev_reshaped.max(axis=0)/noise_level
    sc = ax.pcolormesh(t0s_offset, lat_offset+lat_vol, Z, norm=colors.LogNorm(vmin=0.1, vmax=10), cmap=cmap)
    #sc = plt.pcolormesh(t0s_offset, lat_offset, amps_ev_reshaped.max(axis=0))
    #sc = plt.pcolormesh(t0s_offset, lat_offset, mags_ev_reshaped.max(axis=0))
    #sc = plt.pcolormesh(lon_offset, lat_offset, np.log(amps_ev_reshaped.mean(axis=-1)).T)
    #sc = plt.pcolormesh(t0s_offset, lat_offset, distances_ev_reshaped.min(axis=0))
    ax.set_xlabel('Balloon start time (years since main shock)', fontsize=fontsize)
    ax.set_ylabel('Balloon start latitude (deg)', fontsize=fontsize)
    add_cbar(ax, sc, f'Peak SNR', fontsize=fontsize)
    ax.set_facecolor(cmaplist[0])
    ax.axhline(lat_vol, color='black', linestyle='--', alpha=0.5)

    cmap = plt.cm.Greens  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    #cmaplist[0] = (.5, .5, .5, 1.0) # force the first color entry to be grey
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.arange(7)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax = fig.add_subplot(grid[1,0], sharex=ax)
    Z = number_over_snr.max(axis=0)
    sc = ax.pcolormesh(t0s_offset, lat_offset+lat_vol, Z, cmap='Greens', norm=norm)
    ax.set_xlabel('Balloon start time (years since main shock)', fontsize=fontsize)
    ax.set_ylabel('Balloon start latitude (deg)', fontsize=fontsize)
    add_cbar(ax, sc, f'Number of events with SNR > {snr_threshold:.0f}', fontsize=fontsize)
    ax.axhline(lat_vol, color='black', linestyle='--', alpha=0.5)

    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    ax.set_facecolor(cmaplist[0])