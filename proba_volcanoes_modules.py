import proba_modules as pm
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from tqdm import tqdm
import matplotlib.dates as mdates
from obspy.core.utcdatetime import UTCDateTime

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

from pyproj import Geod
def compute_positions_vectorized_w_interpolator(lat0, lon0, wind_direction_interpolator, wind_strength_interpolator, times, R0):

    g = Geod(ellps='WGS84') # Use Clarke 1866 ellipsoid.

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
    time_prev = times[0,0]
    new_lat_rad[:,0] = lat_rad
    new_lon_rad[:,0] = lon_rad
    for i, time in tqdm(enumerate(times[0,1:]), total=times.shape[1]-1):
        
        time_diff = time-time_prev
        time_prev = time

        az = np.radians(wind_direction_interpolator.ev(np.degrees(lat_rad), np.degrees(lon_rad)))
        w_strength = wind_strength_interpolator.ev(np.degrees(lat_rad), np.degrees(lon_rad))

        use_geod = True
        if not use_geod:
            vlon_mpers = np.sin(az)*w_strength
            vlat_mpers = np.cos(az)*w_strength
            
            # Calculate the distance traveled in latitude and longitude
            dlat_m = vlat_mpers * time_diff
            dlon_m = vlon_mpers * time_diff
            
            # Convert distances into angular displacements (in radians)
            delta_lat_rad = dlat_m / R0
            delta_lon_rad = dlon_m / (R0 * np.cos(lat_rad))
            
            # Calculate the new latitude and longitude in radians
            lat_rad += delta_lat_rad
            lon_rad += delta_lon_rad

            # Wrap the latitude and longitude values
            lat_rad = wrap_latitude(lat_rad)
            lon_rad = wrap_longitude(lon_rad)

        else:
            #print('--------------')
            #print(np.degrees(lon_rad[0]), np.degrees(lat_rad[0]))
            #print(w_strength[0]*time_diff, w_strength[0], time_diff, np.degrees(az[0]), )
            lon_rad, lat_rad, _ = g.fwd(np.degrees(lon_rad), np.degrees(lat_rad), np.degrees(az), w_strength*time_diff)
            #print(lon_rad[0], lat_rad[0])
            lon_rad, lat_rad = np.radians(lon_rad), np.radians(lat_rad)
        
        # Store the results
        new_lat_rad[:,i+1] = lat_rad
        new_lon_rad[:,i+1] = lon_rad
    
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
        #arrival_times_ev = np.take_along_axis(arrival_times.reshape(all_shape), id_event, axis=-1)[:,:,0]
        distances_ev = np.take_along_axis(distances[ID_TIMES].reshape(all_shape), id_event, axis=-1)[:,:,0]
        
        #print(all_shape, (all_times.size,)+shape_TIMES, balloon_times.shape, TIMES[ID_TIMES].shape, distances_ev.shape)
        mags_ev[:,id_start:id_end] = np.take_along_axis(all_mags[ID_ALL_TIMES].reshape(all_shape), id_event, axis=-1)[:,:,0]
        amps_ev[:,id_start:id_end] = TL_new(distances_ev, mags_ev[:,id_start:id_end])
        id_start = id_end

    return mags_ev, amps_ev, mask


import matplotlib.colors as colors

def add_cbar(ax, sc, label, fontsize=12., bbox_to_anchor=(0.15, 1.01, 1, 1.), color='black'):

    axins = inset_axes(ax, width="70%", height="2.5%", loc='lower left', 
                    bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False, labelrotation=90.)
    cbar = plt.colorbar(sc, cax=axins, orientation="horizontal")  
    cbar.ax.xaxis.set_ticks_position('top') 
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.tick_top()
    cbar.ax.set_xlabel(label, labelpad=2, fontsize=fontsize, color=color) 

    return cbar

def plot_seq(ax_first, loc_catalog, cmap, color, type_ev=None, fontsize_text=12., xpad=0., max_val=7., maxval_annot=None, str_annot='', vmax=7.):

    ax_first.scatter(loc_catalog.UTC, loc_catalog.mag, c=loc_catalog.mag, cmap=cmap, vmin=1., vmax=vmax, s=3)
    ax_first.plot([loc_catalog.UTC.min(), loc_catalog.UTC.max()], [max_val, max_val], lw=4., color=color, clip_on=False, zorder=100)
    mid_point = UTCDateTime(loc_catalog.UTC.min()) + (UTCDateTime(loc_catalog.UTC.max()) - UTCDateTime(loc_catalog.UTC.min()))/2.
    if type_ev is None:
        type_ev = loc_catalog.iloc[0].type_ev
    ax_first.text(mid_point.datetime+pd.Timedelta(days=xpad), max_val+0.2, type_ev, ha='center', va='bottom', fontsize=fontsize_text, clip_on=False, color=color)

    if maxval_annot is not None:
        ax_first.annotate(
            str_annot,  # No text for annotation, just the arrow
            xy=(mid_point, maxval_annot),  # Point where the arrowhead will be
            xytext=(mid_point, maxval_annot+0.6),  # Starting point of the arrow (slightly above xy)
            arrowprops=dict(arrowstyle='->', lw=2, color=color),
            ha='center',
            color=color
        )

def plot_sequence_events(fig, ax_first, ax_zoom, catalog_hawai, max_val=7., fontsize=12., fontsize_label=20., color_labels='black'):

    time_collapse = UTCDateTime('2018-05-01')
    time_collapse_end = UTCDateTime('2018-08-05')
    time_collapse_seq_start = UTCDateTime('2018-05-11')
    time_collapse_seq_end = UTCDateTime('2018-08-05')
    time_cratercollapse_seq_start = UTCDateTime('2018-05-1')
    time_cratercollapse_seq_end = UTCDateTime('2018-05-04')
    #time_magma_start = UTCDateTime('2020-12-20')
    time_magma_start = time_collapse_end
    time_maunaloa_seq_start = UTCDateTime('2022-11-27')
    time_maunaloa_seq_end = UTCDateTime('2022-12-13')
    mag_min_collapses = 4.8

    catalog_hawai['type_ev'] = 'regular'
    catalog_hawai.loc[catalog_hawai.UTC<=time_collapse.datetime, 'type_ev'] = 'Low-level eruptive activity'
    catalog_hawai.loc[(catalog_hawai.UTC>=time_collapse.datetime)&(catalog_hawai.UTC<time_collapse_end.datetime), 'type_ev'] = 'Other during collapse'
    catalog_hawai.loc[(catalog_hawai.UTC>=time_collapse_seq_start.datetime)&(catalog_hawai.UTC<time_collapse_seq_end.datetime)&(catalog_hawai.mag>mag_min_collapses),'type_ev'] = 'Kilauea Collapse'
    catalog_hawai.loc[catalog_hawai.mag==catalog_hawai.mag.max(),'type_ev'] = 'Slumping'
    catalog_hawai.loc[(catalog_hawai.UTC>=time_cratercollapse_seq_start.datetime)&(catalog_hawai.UTC<time_cratercollapse_seq_end.datetime),'type_ev'] = 'Pu`u`O`o crater collapse'
    catalog_hawai.loc[catalog_hawai.UTC>=time_collapse_end.datetime, 'type_ev'] = 'Aftershocks collapse'
    catalog_hawai.loc[(catalog_hawai.UTC>=time_magma_start.datetime),'type_ev'] = 'Magma influx and small eruptions'
    catalog_hawai.loc[(catalog_hawai.UTC>=time_maunaloa_seq_start.datetime)&(catalog_hawai.UTC<=time_maunaloa_seq_end.datetime),'type_ev'] = 'Mauna Loa eruption'
    catalog_hawai['unique_id'] = pd.factorize(catalog_hawai['type_ev'])[0]

    #fig = plt.figure(figsize=(10,5))
    #grid = fig.add_gridspec(3, 6)

    #ax_first = fig.add_subplot(grid[:2,:3])
    catalog_hawai_before = catalog_hawai.loc[catalog_hawai.type_ev.isin(['Low-level eruptive activity'])]
    plot_seq(ax_first, catalog_hawai_before, 'Reds', 'tab:red', type_ev='Low-level\neruptive activity', vmax=catalog_hawai.mag.max())

    catalog_hawai_collapse = catalog_hawai.loc[~catalog_hawai.type_ev.isin(['Low-level eruptive activity', 'Aftershocks collapse', 'Magma influx and small eruptions', 'Mauna Loa eruption'])]
    plot_seq(ax_first, catalog_hawai_collapse, 'Greens', 'tab:green', type_ev='', maxval_annot=5.5, str_annot='Kilauea\nCollapses', vmax=catalog_hawai.mag.max())

    #catalog_hawai_later = catalog_hawai.loc[catalog_hawai.type_ev.isin(['Aftershocks collapse',])]
    #plot_seq(ax_first, catalog_hawai_later, 'Blues', 'tab:blue', type_ev='After\nshocks', xpad=-800., vmax=catalog_hawai.mag.max())

    catalog_hawai_magma = catalog_hawai.loc[catalog_hawai.type_ev.isin(['Magma influx and small eruptions',])]
    plot_seq(ax_first, catalog_hawai_magma, 'Oranges', 'tab:orange', type_ev='Magma\ninflux', xpad=200., vmax=catalog_hawai.mag.max())

    catalog_hawai_manuloa = catalog_hawai.loc[catalog_hawai.type_ev.isin(['Mauna Loa eruption',])]
    plot_seq(ax_first, catalog_hawai_manuloa, 'Purples', 'tab:purple', type_ev='', maxval_annot=5., str_annot='Mauna Loa\neruption', vmax=catalog_hawai.mag.max())

    ax_first.set_ylabel('Magnitude (Mw)', fontsize=fontsize, color=color_labels)
    ax_first.text(-0., 1., 'a)', fontsize=fontsize_label, ha='left', va='bottom', transform=ax_first.transAxes)
    ax_first.set_xlim([catalog_hawai.UTC.min(), catalog_hawai.UTC.max()])
    ax_first.set_ylim([catalog_hawai.mag.min(), max_val])
    ax_zoom.tick_params(axis='both', labelsize=fontsize, rotation=90., colors=color_labels)
    ax_first.tick_params(axis='both', labelsize=fontsize, colors=color_labels)
    rect = plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax_first.transAxes,  # Use axes coordinates
        color='white',
        zorder=-100,  # Place it below all other elements
    )
    ax_first.add_patch(rect)

    #ax_zoom = fig.add_subplot(grid[:2,3:], sharey=ax_first)

    catalog_loc = catalog_hawai.loc[catalog_hawai.type_ev.isin(['Kilauea Collapse'])]
    sc = ax_zoom.scatter(catalog_loc.UTC, catalog_loc.mag, c='limegreen', alpha=0.3, s=100, label=catalog_loc.iloc[0].type_ev)
    sc.set_edgecolor("none")

    catalog_loc = catalog_hawai.loc[catalog_hawai.type_ev.isin(['Slumping'])]
    sc = ax_zoom.scatter(catalog_loc.UTC, catalog_loc.mag, c='tab:blue', alpha=0.3, s=100, label=catalog_loc.iloc[0].type_ev)
    sc.set_edgecolor("none")

    catalog_loc = catalog_hawai.loc[catalog_hawai.type_ev.isin(['Pu`u`O`o crater collapse'])]
    sc = ax_zoom.scatter(catalog_loc.UTC, catalog_loc.mag, c='tab:red', alpha=0.3, s=100, label=catalog_loc.iloc[0].type_ev)
    sc.set_edgecolor("none")

    ax_zoom.scatter(catalog_hawai_collapse.UTC, catalog_hawai_collapse.mag, c=catalog_hawai_collapse.unique_id, cmap='Greens', s=3)

    catalog_loc = catalog_hawai.loc[catalog_hawai.type_ev.isin(['Pu`u`O`o crater collapse'])]
    ax_zoom.scatter(catalog_loc.UTC, catalog_loc.mag, c='tab:green', alpha=0.3, cmap='Greens', s=3)

    ax_zoom.legend(frameon=False)
    ax_zoom.tick_params(axis='both', labelleft=False)
    date_format = mdates.DateFormatter('%m-%d')  # Format: YYYY-MM-DD
    ax_zoom.xaxis.set_major_formatter(date_format)
    ax_zoom.set_title(f'Collapse events\nin 2018', color=color_labels)
    ax_zoom.set_xlim([catalog_hawai_collapse.UTC.min(), catalog_hawai_collapse.UTC.max()])
    ax_zoom.text(-0., 1., 'b)', fontsize=fontsize_label, ha='left', va='bottom', transform=ax_zoom.transAxes)

    rect = plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax_zoom.transAxes,  # Use axes coordinates
        color='white',
        zorder=-100,  # Place it below all other elements
    )
    ax_zoom.add_patch(rect)

def plot_proba_sequence_small(catalog_hawai, amps_ev, t0s_offset, LAT_offset_shape, mask, snrs, noise_level = 0.01, fontsize=12., number_over_snr=None, idamp=None, amps_ev_reshaped=None, color_labels='black', fontsize_label=20.):
    
    cmap = plt.cm.coolwarm  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]

    if idamp is None:
        idamp = (amps_ev*(mask)).argmax(axis=0)[None,:]  

    if number_over_snr is None:
        #snrs = np.arange(0.5, 5., 0.5)
        number_over_snr = np.zeros((snrs.size,) + LAT_offset_shape)
        for isnr,  snr in tqdm(enumerate(snrs), total=snrs.size):
            number_over_snr[isnr,:] = ((amps_ev*(mask)/noise_level)>snr).sum(axis=0).reshape(LAT_offset_shape)

    if amps_ev_reshaped is None:
        amps_ev_reshaped = np.take_along_axis(amps_ev*(mask) + 1e-10*(~mask), idamp, axis=0)[0].reshape(LAT_offset_shape) # lon x lat x t0

    fig = plt.figure(figsize=(10,6))
    grid = fig.add_gridspec(2, 6)

    ax_first = fig.add_subplot(grid[:1,:4])
    ax_zoom = fig.add_subplot(grid[:1,4:], sharey=ax_first)

    ## Plotting time distribution of events and labels
    plot_sequence_events(fig, ax_first, ax_zoom, catalog_hawai, max_val=7., fontsize=12., fontsize_label=20., color_labels=color_labels)

    ## SNR total distribution
    ax = fig.add_subplot(grid[1,4:],)
    field = amps_ev_reshaped.ravel()/noise_level
    bins_orig = np.logspace(np.log10(0.1),np.log10(10.), 50)
    bins = np.r_[0, bins_orig, 100]
    #bins = np.linspace(0.1,10., 50)
    hist = ax.hist(field, bins=bins, log=True, density=True, orientation='horizontal')

    vals = []
    ax_proba = ax.twiny()
    for threshold in hist[1][:-1]:
        idx = hist[1][:-1] >= threshold
        #integral = np.trapz(hist[0][idx], x=hist[1][:-1][idx])/np.trapz(hist[0][:], x=hist[1][:-1][:])
        integral = np.trapz(hist[0][idx], x=hist[1][:-1][idx])/np.trapz(hist[0][:], x=hist[1][:-1][:])
        vals.append( integral )
    ax_proba.plot(vals, hist[1][:-1], color='tab:red', label='P(>SNR)')
    #print(hist[0], hist[1])
    ax_proba.legend(frameon=False)
    ax_proba.set_xscale('log')
    ax_proba.set_yscale('log')
    ax_proba.tick_params(axis='x', labelcolor='tab:red')
    ax_proba.set_xlabel('Probability', fontsize=fontsize, color='tab:red')
    ax_proba.set_xlim([1e-2, 1e1])
    ax_proba.tick_params(axis='both', labelsize=fontsize)
    ax_proba.set_xticks([0.1, 0.5, ])
    ax_proba.set_xticklabels([f'10%', f'50%',])
    #ax.set_xlabel('SNR', fontsize=fontsize, color=color_labels)
    ax.set_xlabel('Probability Density Function', fontsize=fontsize, color=color_labels)
    ax.set_yscale('log')
    ax.set_xlim([1e-3, 1])
    ax.set_ylim([bins_orig.min(), bins_orig.max()])
    ax.tick_params(axis='both', labelsize=fontsize, labelleft=False, colors=color_labels)
    ax.text(-0., 1., 'd)', fontsize=fontsize_label, ha='left', va='bottom', transform=ax.transAxes)

    ##
    ## Plotting number of events > SNR
    ##
    cmap = plt.cm.Greens  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    #cmaplist[0] = (.5, .5, .5, 1.0) # force the first color entry to be grey
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.arange(7)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    from matplotlib.colors import TwoSlopeNorm

    number_over_snr_mean = np.mean(number_over_snr, axis=(1,2))
    cmap = plt.cm.coolwarm
    norm = TwoSlopeNorm(vmin=number_over_snr_mean.min(), vmax=2., vcenter=1.)  # Midpoint at 0
    datetimes = [catalog_hawai.UTC.min() + pd.Timedelta(days=year * 365.25) for year in t0s_offset]
    ax = fig.add_subplot(grid[1,:4], sharex=ax_first, sharey=ax)
    sc = ax.pcolormesh(datetimes, snrs, number_over_snr_mean, cmap=cmap, norm=norm)
    add_cbar(ax, sc, f'Average number of events detected', fontsize=fontsize, color=color_labels)
    ax.set_ylabel(f'SNR', fontsize=fontsize, color=color_labels)
    ax.tick_params(axis='both', labelsize=fontsize, colors=color_labels)
    ax.xaxis_date()
    ax.text(-0., 1., 'c)', fontsize=fontsize_label, ha='left', va='bottom', transform=ax.transAxes)
    rect = plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,  # Use axes coordinates
        color='white',
        zorder=-100,  # Place it below all other elements
    )
    ax.add_patch(rect)

    fig.align_ylabels()
    fig.align_xlabels()
    fig.subplots_adjust(hspace=0.7, wspace=0.3)

    return fig

def plot_proba_sequence(catalog_hawai, amps_ev, all_times, all_mags, TL_new, lat_vol, t0s_offset, lat_offset, LAT_offset_shape, mask, noise_level = 0.01, factor = (np.log10(2.)+4.)/4., snr_threshold=1, plot_SNR_distrib=True, fontsize=12., number_over_snr=None, amps_ev_reshaped=None, color_labels='black'):

    dists = np.logspace(0, 4.*factor, 100)
    fontsize_label = 20.
    
    cmap = plt.cm.coolwarm  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    idamp = (amps_ev*(mask)).argmax(axis=0)[None,:]  
    if number_over_snr is None:
        number_over_snr = ((amps_ev*(mask)/noise_level)>snr_threshold).sum(axis=0).reshape(LAT_offset_shape)
    if amps_ev_reshaped is None:
        amps_ev_reshaped = np.take_along_axis(amps_ev*(mask) + 1e-10*(~mask), idamp, axis=0)[0].reshape(LAT_offset_shape) # lon x lat x t0
    #distances_ev_reshaped = np.take_along_axis(distances_ev*(mask), idamp, axis=0)[0].reshape(LAT_offset_shape) # lon x lat x t0
    #mags_ev_reshaped = np.take_along_axis(mags_ev*(mask), idamp, axis=0)[0].reshape(LAT_offset_shape) # lon x lat x t0
    ID_TIMES_EV, DISTS = np.meshgrid(np.arange(all_times.size), dists)

    fig = plt.figure(figsize=(10,11))
    grid = fig.add_gridspec(7, 6)
    plt.rcParams['text.usetex'] = False

    ax_first = fig.add_subplot(grid[:2,:4])
    ax_zoom = fig.add_subplot(grid[:2,4:], sharey=ax_first)

    ## Plotting time distribution of events and labels
    plot_sequence_events(fig, ax_first, ax_zoom, catalog_hawai, max_val=7., fontsize=12., fontsize_label=20., color_labels=color_labels)

    if plot_SNR_distrib:
        ax = fig.add_subplot(grid[2:4,4:],)
        field = amps_ev_reshaped.ravel()/noise_level
        bins = np.logspace(np.log10(0.1),np.log10(10.), 50)
        #bins = np.linspace(0.1,10., 50)
        hist = ax.hist(field, bins=bins, log=True, density=True)

        vals = []
        ax_proba = ax.twinx()
        for threshold in hist[1][:-1]:
            idx = hist[1][:-1] >= threshold
            integral = np.trapz(hist[0][idx], x=hist[1][:-1][idx])/np.trapz(hist[0][:], x=hist[1][:-1][:])
            vals.append( integral )
        ax_proba.plot(hist[1][:-1], vals, color='tab:red', label='P(>SNR)')
        ax_proba.legend(frameon=False)
        ax_proba.set_xscale('log')
        ax_proba.set_yscale('log')
        ax_proba.tick_params(axis='y', labelcolor='tab:red')
        ax_proba.set_ylim([1e-2, 1e1])
        ax_proba.tick_params(axis='both', labelsize=fontsize)
        ax.set_xlabel('SNR', fontsize=fontsize, color=color_labels)
        ax_proba.set_ylabel('PDF or Probability', fontsize=fontsize, color=color_labels)
        ax.set_xscale('log')
        ax.set_ylim([1e-2, 1e1])
        ax.tick_params(axis='both', labelsize=fontsize, labelleft=False, colors=color_labels)

    else:
        ax = fig.add_subplot(grid[2:4,4:],)
        ZTL = TL_new(DISTS.T, all_mags[ID_TIMES_EV].T)/noise_level
        #sc = ax.contourf(all_times[ID_TIMES_EV], DISTS, TL_new(DISTS, all_mags[ID_TIMES_EV])/noise_level, levels=[0.1, 0.5, 1, 5, 10], locator=ticker.LogLocator(), cmap='Blues', )
        #sc = ax.pcolormesh(all_times, dists, ZTL.T, norm=colors.LogNorm(vmin=0.1, vmax=10), cmap=cmap)
        sc = ax.pcolormesh(all_times, dists, ZTL.T, vmin=1., vmax=10, cmap=cmap)
        ax.set_xlabel('Time\n(years since main shock)', fontsize=fontsize, color=color_labels)
        ax.set_ylabel('Distance (km)', fontsize=fontsize, color=color_labels)
        ax.set_yscale('log')
        add_cbar(ax, sc, f'Peak SNR', fontsize=fontsize, color=color_labels)
        ax.set_facecolor(cmaplist[0])
    ax.text(-0., 1., 'f)', fontsize=fontsize_label, ha='left', va='bottom', transform=ax.transAxes)
    rect = plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,  # Use axes coordinates
        color='white',
        zorder=-100,  # Place it below all other elements
    )
    ax.add_patch(rect)

    ##
    ## Plotting peak SNR
    ##
    ax = fig.add_subplot(grid[5:,:4], sharex=ax_first)
    Z = amps_ev_reshaped.max(axis=0)/noise_level
    datetimes = [catalog_hawai.UTC.min() + pd.Timedelta(days=year * 365.25) for year in t0s_offset]
    #sc = ax.pcolormesh(datetimes, lat_offset+lat_vol, Z, norm=colors.LogNorm(vmin=0.1, vmax=10), cmap=cmap)
    sc = ax.pcolormesh(datetimes, lat_offset+lat_vol, Z, vmin=1., vmax=10, cmap=cmap)
    #sc = plt.pcolormesh(t0s_offset, lat_offset, amps_ev_reshaped.max(axis=0))
    #sc = plt.pcolormesh(t0s_offset, lat_offset, mags_ev_reshaped.max(axis=0))
    #sc = plt.pcolormesh(lon_offset, lat_offset, np.log(amps_ev_reshaped.mean(axis=-1)).T)
    #sc = plt.pcolormesh(t0s_offset, lat_offset, distances_ev_reshaped.min(axis=0))
    ax.set_xlabel('Balloon start time (years since main shock)', fontsize=fontsize, color=color_labels)
    ax.set_ylabel('Balloon\nstart latitude (deg)', fontsize=fontsize, color=color_labels)
    cbar = add_cbar(ax, sc, f'Peak SNR', fontsize=fontsize, color=color_labels)
    ax.set_facecolor(cmaplist[0])
    ax.axhline(lat_vol, color='black', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', labelsize=fontsize, colors=color_labels)
    ax.text(-0., 1., 'e)', fontsize=fontsize_label, ha='left', va='bottom', transform=ax.transAxes)
    rect = plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,  # Use axes coordinates
        color='white',
        zorder=-100,  # Place it below all other elements
    )
    ax.add_patch(rect)

    ##
    ## Plotting number of events > SNR
    ##
    cmap = plt.cm.Greens  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    #cmaplist[0] = (.5, .5, .5, 1.0) # force the first color entry to be grey
    cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.arange(7)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax = fig.add_subplot(grid[2:4,:4], sharex=ax)
    Z = number_over_snr.mean(axis=0)
    sc = ax.pcolormesh(datetimes, lat_offset+lat_vol, Z, cmap='Greens', norm=norm)
    #ax.set_xlabel('Balloon start time\n(years since main shock)', fontsize=fontsize)
    ax.set_ylabel('Balloon\nstart latitude (deg)', fontsize=fontsize, color=color_labels)
    cbar2 = add_cbar(ax, sc, f'Number of events with SNR > {snr_threshold:.0f}', fontsize=fontsize, color=color_labels)
    ax.axhline(lat_vol, color='black', linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', labelsize=fontsize, labelbottom=False, colors=color_labels)
    ax.text(-0., 1., 'c)', fontsize=fontsize_label, ha='left', va='bottom', transform=ax.transAxes)
    ax.set_facecolor(cmaplist[0])
    rect = plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,  # Use axes coordinates
        color='white',
        zorder=-100,  # Place it below all other elements
    )
    ax.add_patch(rect)

    import datetime
    interval = datetime.timedelta(days=30*6) # 6 months
    #ax = fig.add_subplot(grid[4,:4], sharex=ax)
    bbox_to_anchor=(0., -0.65, 1, 1.)
    ax_dis = inset_axes(ax, width="100%", height="35%", loc='lower left', bbox_to_anchor=bbox_to_anchor, bbox_transform=ax.transAxes, borderpad=0)
    ax_dis.sharex(ax)
    ax_dis.bar(datetimes, number_over_snr.mean(axis=(0,1)), log=True, color='tab:green', width=interval,)# label=f'Number events per flight with SNR > {snr_threshold:.0f}')
    ax_dis.axhline(1., linestyle='--', color='black')
    ax_dis.set_ylim([1e-2, 6])
    ax_dis.xaxis_date()
    ax_dis.set_ylabel(f'Number events per flight\n with SNR > {snr_threshold:.0f}', fontsize=fontsize, color=color_labels)
    ax_dis.tick_params(axis='both', labelsize=fontsize, colors=color_labels)
    ax_dis.legend(frameon=False, loc='upper left')
    ax.text(-0., -0.25, 'd)', fontsize=fontsize_label, ha='left', va='bottom', transform=ax.transAxes)
    rect = plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,  # Use axes coordinates
        color='white',
        zorder=-100,  # Place it below all other elements
    )
    ax.add_patch(rect)

    fig.align_ylabels()
    fig.align_xlabels()
    fig.subplots_adjust(hspace=1.5, wspace=0.3)

    return fig