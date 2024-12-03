import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import pyproj
from tqdm import tqdm
from matplotlib.patches import Polygon as Polygon_mpl
import geopandas as gpd
from functools import partial
from multiprocessing import get_context
import pandas as pd 
from pdb import set_trace as bp
from scipy import interpolate
import proba_modules as pm

def compute_surfaces_CPUs(thresholds, LATS, LONS, lats_stations, lons_stations, polys, plot, R0, max_dist, nb_CPU=10):

    partial_compute_surfaces = partial(compute_surfaces, thresholds, LATS, LONS, lats_stations, lons_stations, polys, plot, R0)
    nb_chunks = max_dist.shape[0]
    idx_start_all = np.arange(nb_chunks)
    
    N = min(nb_CPU, nb_chunks)
    ## If one CPU requested, no need for deployment
    if N == 1:
        print('Running serial')
        gdf = partial_compute_surfaces((0, idx_start_all, max_dist))

    ## Otherwise, we pool the processes
    else:
    
        step_idx =  nb_chunks//N
        list_of_lists = []
        idxs = []
        for i in range(N):
            idx = np.arange(i*step_idx, (i+1)*step_idx)
            if i == N-1:
                idx = np.arange(i*step_idx, nb_chunks)
            idxs.append(idx_start_all[idx][0])
            list_of_lists.append( (i, idx_start_all[idx], max_dist[idx_start_all[idx],:]) )

        with get_context("spawn").Pool(processes = N) as p:
            print(f'Running across {N} CPU')
            gdfs = p.map(partial_compute_surfaces, list_of_lists)
            p.close()
            p.join()

        gdf = gpd.GeoDataFrame()
        for idx, gdf_loc in zip(idxs, gdfs):
            if not 'iscenario' in gdf_loc.columns:
                continue
            gdf_loc['iscenario'] += idx
            gdf = pd.concat([gdf, gdf_loc], ignore_index=True)

    return gdf

def compute_surfaces(thresholds, LATS, LONS, lats_stations, lons_stations, polys, plot, R0, inputs):

    debug = False

    iCPU, all_idx, max_dist = inputs

    proj = pyproj.Proj(proj='robin', lat_0=0., lon_0=0., a=R0, b=R0)
    x, y = proj(LONS, LATS)

    gdf = []
    #for iscenario in tqdm(range(500,503)):
    for iscenario_loc in tqdm(range(max_dist.shape[0]), disable=not iCPU == 0):

        iscenario = all_idx[iscenario_loc]
        if debug:
            if not iscenario == 160:
                continue

        max_map = max_dist[iscenario_loc,:]

        dict_scenario = {'iscenario': iscenario}
        lats_loc = lats_stations[iscenario,:]
        lons_loc = lons_stations[iscenario,:]
        for istat in range(lats_stations.shape[1]):
            dict_scenario[f'lat_{istat}'] = lats_loc[istat]
            dict_scenario[f'lon_{istat}'] = lons_loc[istat]

        inds_last = []
        first_pass_done = False
        threshold_offset = 80.
        cpt_threshold = 0
        for ithreshold, threshold in enumerate(thresholds[:]):

            #print(f'threshold {first_pass_done} - {threshold}')
            inds = np.where(max_map/1e3<=threshold+threshold_offset)[0] 
            if inds.size == 0:
                continue

            cpt_threshold += 1

            inds_to_process = np.setdiff1d(inds, inds_last)
            polys_processed_temp = unary_union([polys[ii] for ii in inds_to_process])
            if first_pass_done:
                polys_processed_temp = unary_union([polys_processed, polys_processed_temp])
            
            polys_processed = polys_processed_temp
            inds_last = inds
            
            dict_loc = dict(geometry=polys_processed, distance=threshold,)
            dict_loc.update(dict_scenario)

            gdf.append(dict_loc)

            if plot and cpt_threshold < 5 and debug:
                plt.figure()
                ax = plt.gca()

                #print(f'Threshold {threshold}')
                if polys_processed.geom_type == 'MultiPolygon':
                    polys_loc = polys_processed.geoms
                else:
                    polys_loc = [polys_processed]

                for poly in polys_loc:
                    coords = np.array(poly.exterior.coords)
                    p = Polygon_mpl(coords, facecolor = 'tab:green')
                    #plt.scatter(x[inds], y[inds], s=1, label=threshold)
                    ax.add_patch(p)
                    for interior in poly.interiors:
                        interior_coords = np.array(interior.coords)
                        interior_patch = Polygon_mpl(interior_coords, facecolor='white')  # Set hole facecolor to white
                        ax.add_patch(interior_patch)

                x, y = proj(lons_stations[iscenario, :], lats_stations[iscenario, :])
                ax.scatter(x, y, marker='x', s=100, color='red')
                #ax.legend()
                
                ax.set_title(f'{threshold} - iscenario: {iscenario}')
                print(f'Saving')
                #plt.savefig(f'./test_threshold{cpt_threshold}_CPU{iCPU}_iscenario.png')

            first_pass_done = True

        if debug:
            break

    gdf = gpd.GeoDataFrame(gdf)
    return gdf

def haversine_distance(lon1, lat1, lons, lats, RADIUS_VENUS=6051.8e3):
    """
    Vectorized Haversine formula to compute distances on Venus.

    lon1, lat1 : scalar
        Reference longitude and latitude in degrees.
    lons, lats : arrays
        Arrays of longitudes and latitudes in degrees.

    Returns
    -------
    distances : array
        Array of distances from (lon1, lat1) to each point in (lons, lats) in meters.
    """
    # Convert degrees to radians
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lons_rad = np.radians(lons)
    lats_rad = np.radians(lats)
    
    # Haversine formula
    dlon = lons_rad - lon1_rad
    dlat = lats_rad - lat1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lats_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Distance in meters
    distances = RADIUS_VENUS * c
    
    return distances

def compute_alphas(f_rho, f_VER, periods, alts, times, tau=0.5*1e4, surface_amplitude=1., c=200, z0=90.):

    ALTS, TIMES = np.meshgrid(alts, times)

    #tau = 0.5*1e4 # s, after eq. 23 in Lognonne, 2016
    #surface_amplitude = 1. # m/s, at airglow altitude
    #c = 200 # km/s
    #z0 = 90.
    amplification = np.sqrt(f_rho(z0)/f_rho(alts))
    Az = surface_amplitude*amplification
    #dzAz = np.r_[0., np.diff(Az) / dz]
    dzAz = np.gradient(Az, alts)
    dzAz = interpolate.interp1d(alts, dzAz, kind='quadratic', bounds_error=False, fill_value=0.)

    alphas = []
    #plt.figure()
    for period in tqdm(periods):
        std_t = period/2.
        t0 = 3*std_t
        f0 = -2*(1/std_t)*((times-t0)/std_t)*np.exp(-((times-t0)/std_t)**2)
        f0 /= abs(f0).max()
        f0 = interpolate.interp1d(times, f0, kind='quadratic', bounds_error=False, fill_value=0.)
        #plt.plot(times, f0(times), label=period)
        df0dt = np.gradient(f0(times), times)
        df0dt = interpolate.interp1d(times, df0dt, kind='quadratic', bounds_error=False, fill_value=0.)
        dVER = -(tau/(1+1*(2*np.pi/period)*tau)) * f_VER(ALTS) * (-(1/c)*df0dt(TIMES-(ALTS-alts.min())*1e3/c)*Az + dzAz(ALTS)*f0(TIMES-(ALTS-alts.min())*1e3/c))
        sig = np.trapz((dVER), x=alts, axis=1)
        max_val_SNR = abs(sig).max()
        alphas.append(max_val_SNR)
    #plt.legend()
    return alphas

def compute_airglow_SNR(TL_new_v, distances, beta, periods, alphas, photons_airglow, f_VER=None, alts=np.linspace(90., 120., 1000), m0=7., file_airglow=''):

    if isinstance(TL_new_v, dict):
        TL_new_v_freqs = []
        l_freqs = np.array([freq for freq in TL_new_v.keys()])
        for period in periods:
            i_TL = np.argmin(abs(1./l_freqs-period))
            TL_new_v_freqs.append( TL_new_v[l_freqs[i_TL]] )
    else:
        TL_new_v_freqs = [TL_new_v for _ in periods]

    sigma_airglow = 1./(np.sqrt(photons_airglow))
    if f_VER is not None:
        sigma_airglow *= np.trapz(f_VER(alts), x=alts)

    airglow_scaling_all = pd.DataFrame()
    for period, alpha, TL_new_v_loc in zip(periods, alphas, TL_new_v_freqs):
        scaling = (beta*alpha)*TL_new_v_loc(distances, m0)/sigma_airglow
        airglow_scaling = pd.DataFrame(np.c_[distances, scaling], columns=['distance', 'SNR'])
        airglow_scaling['period'] = period
        airglow_scaling_all = pd.concat([airglow_scaling_all, airglow_scaling])
    airglow_scaling_all.reset_index(drop=True, inplace=True)

    if file_airglow:
        airglow_scaling_all.to_csv(file_airglow, header=True, index=False)

    return airglow_scaling_all

def get_f_VER(file_airglow):

    #file_airglow = './data/VER_profile_scaled.csv'
    VER = pd.read_csv(file_airglow)
    VER.columns=['VER', 'alt']
    #f_VER = interpolate.interp1d(VER.alt, VER.VER, kind='quadratic', bounds_error=False, fill_value=(VER.VER.iloc[0], VER.VER.iloc[-1]))
    f_VER = interpolate.interp1d(VER.alt, VER.VER, kind='quadratic', bounds_error=False, fill_value=0.)
    
    return f_VER

def get_airglow_scaling_from_TL(TL_new_p, scaling_in, period, R0=6052000, sigma_balloon=1e-2, boost_SNR=1., m0=7., adjust_indexes=False):

    #scaling = pd.read_csv(file_scaling, header=[0])
    scaling = scaling_in.copy()
    if 'period' in scaling.columns:
        diff = abs(scaling.period-period)
        scaling = scaling.loc[diff==diff.min()]
    f_airglow = interpolate.interp1d(scaling.distance, scaling.SNR, kind='nearest', bounds_error=False, fill_value=(scaling.SNR.iloc[0], scaling.SNR.iloc[-1]))

    distances = np.linspace(0., np.pi*R0/1.001, 300)/1e3
    distances_r0 = np.linspace(-np.pi*R0/1.001, np.pi*R0/1.001, 300)/1e3
    DIST, DIST_R0 = np.meshgrid(distances, distances_r0)

    diff = abs(TL_new_p(DIST+DIST_R0, m0)/sigma_balloon-boost_SNR*f_airglow(DIST))
    flipped_diff = np.flip(diff, axis=0)
    flipped_indices = flipped_diff.argmin(axis=0) ## In order to get argmin to return the largest index if multiple minima
    original_indices = diff.shape[0] - 1 - flipped_indices

    coefs = distances_r0[original_indices]
    if adjust_indexes:
        # Determine the dominant direction
        differences = np.diff(original_indices)
        if np.sum(differences > 0) >= np.sum(differences < 0):
            dominant_direction = "increasing"
        else:
            dominant_direction = "decreasing"

        # Adjust indices to follow the dominant direction
        adjusted_indices = original_indices.copy()
        for i in range(1, len(adjusted_indices)):
            if dominant_direction == "increasing" and adjusted_indices[i] < adjusted_indices[i - 1]:
                adjusted_indices[i] = adjusted_indices[i - 1]  # Force monotonic increase
            elif dominant_direction == "decreasing" and adjusted_indices[i] > adjusted_indices[i - 1]:
                adjusted_indices[i] = adjusted_indices[i - 1]  # Force monotonic decrease

        coefs = distances_r0[adjusted_indices]

    coefs[0] = 0.

    """
    plt.figure()
    plt.plot(distances, distances_r0[original_indices], label='orig')
    plt.plot(distances, distances_r0[adjusted_indices])
    plt.legend()
    """

    f_alt_scaling = interpolate.interp1d(distances, coefs, bounds_error=False, fill_value=(coefs[0], coefs[-1]))

    return f_alt_scaling

def get_airglow_SNR(file_curve, freq, file_atmos='./data/profile_VCD_for_scaling_pd.csv', file_nightglow='./data/VER_profile_scaled.csv', file_dayglow='./data/VER_profile_dayglow.csv', R0=6052000, photons_dayglow=3.5e5, alpha_dayglow=1e-5, photons_nightglow=2e4, beta=1., TL_new_v=None, TL_new_p=None, m0 = 7.):

    ## Standard Inputs
    R0 = 6052000
    distances = np.linspace(0., np.pi*R0/1.001, 300)/1e3
    #file_nightglow = './data/VER_profile_scaled.csv'
    f_VER_nightglow = get_f_VER(file_nightglow)
    #file_dayglow = './data/VER_profile_dayglow.csv'
    f_VER_dayglow = get_f_VER(file_dayglow)
    alts_dayglow = np.linspace(90., 150., 1000)
    alts_nightglow = np.linspace(90., 120., 1000)
    times = np.linspace(0., 2000., 2000)

    ## Load Venus data
    profile = pd.read_csv(file_atmos)
    f_rho = interpolate.interp1d(profile.altitude/1e3, profile.rho, kind='quadratic')
    f_c = interpolate.interp1d(profile.altitude/1e3, profile.c, kind='quadratic')

    ## Load frequency dependent TL curves
    #file_curve = './data/GF_data/GF_Dirac_1Hz_all_wfreq.csv'
    #freq = [0.01, 0.1, 1.]
    dict_TL = dict(dist_min=100., rho0=f_rho(0.), rhob=f_rho(90.), cb=f_c(90.), use_savgol_filter=True, plot=False, scalar_moment=10e6, return_dataframe=False)
    if TL_new_v is None:
        TL_new_v, _, _ = pm.get_TL_curves(file_curve, freq, unknown='velocity', **dict_TL)
    if TL_new_p is None:
        TL_new_p, _, _ = pm.get_TL_curves(file_curve, freq, unknown='pressure', **dict_TL)

    ## Compute nightglow scaling which is period dependent unlike dayglow
    ## Dayglow scaling is period independent because dominated by advection
    periods = np.array([1./freq for freq in TL_new_v.keys()])
    alphas_nightglow = compute_alphas(f_rho, f_VER_nightglow, periods, alts_nightglow, times,)
    alphas_dayglow = [alpha_dayglow for _ in periods]

    ## Compute SNR from velocity TLs
    dayglow_scaling = compute_airglow_SNR(TL_new_v, distances, beta, periods, alphas_dayglow, photons_dayglow, f_VER=f_VER_dayglow, alts=alts_dayglow, file_airglow='', m0=m0)
    nightglow_scaling = compute_airglow_SNR(TL_new_v, distances, beta, periods, alphas_nightglow, photons_nightglow, f_VER=f_VER_nightglow, alts=alts_nightglow, file_airglow='', m0=m0)

    return dayglow_scaling, nightglow_scaling, TL_new_v, TL_new_p

def get_airglow_scaling(file_curve, freq, file_atmos='./data/profile_VCD_for_scaling_pd.csv', file_nightglow='./data/VER_profile_scaled.csv', file_dayglow='./data/VER_profile_dayglow.csv', R0=6052000, sigma_balloon=1e-2, boost_SNR=1., photons_dayglow=3.5e5, alpha_dayglow=1e-5, photons_nightglow=2e4, beta=1., TL_new_v=None, TL_new_p=None, m0 = 7.):

    ## Standard Inputs
    R0 = 6052000
    distances = np.linspace(0., np.pi*R0/1.001, 300)/1e3
    #file_nightglow = './data/VER_profile_scaled.csv'
    f_VER_nightglow = get_f_VER(file_nightglow)
    #file_dayglow = './data/VER_profile_dayglow.csv'
    f_VER_dayglow = get_f_VER(file_dayglow)
    alts_dayglow = np.linspace(90., 150., 1000)
    alts_nightglow = np.linspace(90., 120., 1000)
    times = np.linspace(0., 2000., 2000)

    ## Load Venus data
    profile = pd.read_csv(file_atmos)
    f_rho = interpolate.interp1d(profile.altitude/1e3, profile.rho, kind='quadratic')
    f_c = interpolate.interp1d(profile.altitude/1e3, profile.c, kind='quadratic')

    ## Load frequency dependent TL curves
    #file_curve = './data/GF_data/GF_Dirac_1Hz_all_wfreq.csv'
    #freq = [0.01, 0.1, 1.]
    dict_TL = dict(dist_min=100., rho0=f_rho(0.), rhob=f_rho(90.), cb=f_c(90.), use_savgol_filter=True, plot=False, scalar_moment=10e6, return_dataframe=False)
    if TL_new_v is None:
        TL_new_v, _, _ = pm.get_TL_curves(file_curve, freq, unknown='velocity', **dict_TL)
    if TL_new_p is None:
        TL_new_p, _, _ = pm.get_TL_curves(file_curve, freq, unknown='pressure', **dict_TL)

    ## Compute nightglow scaling which is period dependent unlike dayglow
    ## Dayglow scaling is period independent because dominated by advection
    periods = np.array([1./freq for freq in TL_new_v.keys()])
    alphas_nightglow = compute_alphas(f_rho, f_VER_nightglow, periods, alts_nightglow, times,)
    alphas_dayglow = [alpha_dayglow for _ in periods]

    ## Compute SNR from velocity TLs
    dayglow_scaling = compute_airglow_SNR(TL_new_v, distances, beta, periods, alphas_dayglow, photons_dayglow, f_VER=f_VER_dayglow, alts=alts_dayglow, file_airglow='', m0=m0)
    nightglow_scaling = compute_airglow_SNR(TL_new_v, distances, beta, periods, alphas_nightglow, photons_nightglow, f_VER=f_VER_nightglow, alts=alts_nightglow, file_airglow='', m0=m0)

    ## Compute frequency dependent scaling function
    opt_scaling = dict(R0=R0, sigma_balloon=sigma_balloon, m0=m0, adjust_indexes=True)
    f_alt_scaling_dayglow, f_alt_scaling_nightglow = dict(), dict()
    for period, scaling in dayglow_scaling.groupby('period'):
        TL = TL_new_p[1./period]
        f_alt_scaling_dayglow[period] = get_airglow_scaling_from_TL(TL, scaling, period, boost_SNR=boost_SNR['dayglow'], **opt_scaling)

    for period, scaling in nightglow_scaling.groupby('period'):
        TL = TL_new_p[1./period]
        f_alt_scaling_nightglow[period] = get_airglow_scaling_from_TL(TL, scaling, period, boost_SNR=boost_SNR['nightglow'], **opt_scaling)

    return f_alt_scaling_dayglow, f_alt_scaling_nightglow, TL_new_v, TL_new_p

def compute_intersections(C1, L1, C2, L2):
    """
    Compute the intersection center and half-length for multiple cases.

    Parameters:
    C1, L1 : ndarray
        Centers and half-lengths of the first set of lines.
    C2, L2 : ndarray
        Centers and half-lengths of the second set of lines.

    Returns:
    intersection_centers : ndarray
        Centers of the intersections.
    intersection_half_lengths : ndarray
        Half-lengths of the intersections.
    """
    # Calculate endpoints for both sets of lines
    start1, end1 = C1 - L1, C1 + L1
    start2, end2 = C2 - L2, C2 + L2

    # Intersection endpoints
    left = np.maximum(start1, start2)
    right = np.minimum(end1, end2)

    # Check for intersections and compute results
    valid = left <= right  # Boolean mask for valid intersections
    intersection_centers = np.where(valid, (left + right) / 2, 0.)
    intersection_half_lengths = np.where(valid, (right - left) / 2, -1.)

    return intersection_centers, intersection_half_lengths

def get_max_dist(lats_stations, lons_stations, LATS, LONS, id_scenario, id_stat, s_cluster=100, use_airglow=False, which_stat_is_airglow=1, f_alt_scaling=None, lon_0_airglow=dict(nightglow=180., dayglow=0.), radius_airglow=dict(nightglow=60., dayglow=70.), radius_view=60., airglow_considered=['dayglow', 'nightglow']):

    if use_airglow and f_alt_scaling is None:
        print('ERROR: Cannot have airglow and not altitude scaling functional')
        return

    ind_without_airglow = id_stat
    only_airglow = False
    if use_airglow:
        ind_without_airglow = np.setdiff1d(id_stat, np.array([which_stat_is_airglow]))
        only_airglow = False if ind_without_airglow.size > 0 else True
        ind_with_airglow = np.array([which_stat_is_airglow])

    max_dist = np.zeros((lats_stations.shape[0], LONS.size))
    clusters = np.arange(0, lats_stations.shape[0]-1, s_cluster)
    for _, i_cluster in tqdm(enumerate(clusters), total=clusters.size):
        
        inds_loc = np.arange(i_cluster, min(i_cluster+s_cluster,lats_stations.shape[0]))
        if not only_airglow:
            id_ref_quake, all_id_scenarios, all_id_stat = np.meshgrid(np.arange(LONS.size), id_scenario[inds_loc], ind_without_airglow)
            shape_init_ref = id_ref_quake.shape
            id_ref_quake, all_id_scenarios, all_id_stat = id_ref_quake.ravel(), all_id_scenarios.ravel(), all_id_stat.ravel()

            max_dist_loc = haversine_distance(LONS[id_ref_quake], LATS[id_ref_quake], lons_stations[all_id_scenarios, all_id_stat].ravel(), lats_stations[all_id_scenarios, all_id_stat].ravel())
            max_dist[inds_loc,:] = max_dist_loc.reshape(shape_init_ref).max(axis=-1)

        if use_airglow:
            id_ref_quake, all_id_scenarios, all_id_stat = np.meshgrid(np.arange(LONS.size), id_scenario[inds_loc], ind_with_airglow)
            shape_init_ref = id_ref_quake.shape
            id_ref_quake, all_id_scenarios, all_id_stat = id_ref_quake.ravel(), all_id_scenarios.ravel(), all_id_stat.ravel()
            
            ## Computing maximum scaled distances for both types of airglow
            max_dist_airglow_all = np.zeros_like(LONS[id_ref_quake])+1e10
            for type_airglow in airglow_considered:
                new_lon, new_radius = compute_intersections(all_id_scenarios*0+lon_0_airglow[type_airglow], radius_airglow[type_airglow], lons_stations[all_id_scenarios, all_id_stat].ravel(), radius_view)
                max_dist_airglow = haversine_distance(LONS[id_ref_quake], LATS[id_ref_quake], new_lon, LONS[id_ref_quake]*0.,) 
                max_dist_airglow -= new_radius*1e2*1e3
                max_dist_airglow[max_dist_airglow<0] = 0.
                max_dist_airglow[new_radius<0] = 1e10
                max_dist_airglow += f_alt_scaling[type_airglow](max_dist_airglow/1e3)*1e3
                max_dist_airglow[max_dist_airglow<0] = 0.
                max_dist_airglow[max_dist_airglow/1e3>19000.] = 19000.*1e3
                max_dist_airglow_all = np.min(np.stack((max_dist_airglow_all, max_dist_airglow), axis=-1), axis=-1)
                
            max_dist_airglow_all = max_dist_airglow_all.reshape(shape_init_ref).max(axis=-1)
            max_dist[inds_loc,:] = np.max(np.stack((max_dist[inds_loc,:], max_dist_airglow_all), axis=-1), axis=-1)

    
    return max_dist

def get_polys(LONS, LATS, dlon, R0=6052000):
    
    proj = pyproj.Proj(proj='robin', lat_0=0., lon_0=0., a=R0, b=R0)

    offsets = np.array([[0.,0.], [0.,dlon/2.], [dlon,dlon/2.], [dlon,0.]])
    polys = []
    for lon, lat in zip(LONS, LATS):
        coords = offsets.copy()
        coords[:,0] += lon
        coords[:,1] += lat
        x, y = proj(coords[:,0], coords[:,1])
        poly = Polygon(np.c_[x, y])
        polys.append(poly)

    return polys

def get_grid(lats, lons):

    #dlon = 2.
    #lats, lons = np.arange(-90, 89, dlon/2.), np.arange(-180, 179, dlon)
    LATS, LONS = np.meshgrid(lats, lons)
    shape_init = LATS.shape
    LATS, LONS = LATS.ravel(), LONS.ravel()

    return LATS, LONS, shape_init

def wrap_longitude(lon_start, L):
    
    return ((lon_start + L + 180) % 360) - 180

def get_stations(lats, lons, offsets=[[2., 4.], [10., 20.]], use_airglow=False, use_only_airglow=False, fixed_stations=dict(scenario_1=[[0., 0.]]), add_velocity=False, vel_baloon=0.2, vel_imager=2.6):

    LATS, LONS = np.meshgrid(lats, lons)
    LATS, LONS = LATS.ravel(), LONS.ravel()

    vel_baloon_lon = vel_baloon*1e-2
    vel_imager_lon = vel_imager*1e-2
    rel_imager_lon = vel_imager_lon-vel_baloon_lon
    
    if not offsets and not fixed_stations:
        lats_stations = np.array([LATS]).T
        lons_stations = np.array([LONS]).T

    if not offsets and use_airglow:
        if use_only_airglow:
            lats_stations = np.array([lons*0.]).T
            lons_stations = np.array([lons]).T
        else:
            if add_velocity:
                X = np.zeros_like(lons)
                lon_initial = lons[0]
                dlon = abs(lons[1]-lons[0])
                dlon_mod = dlon*(1+rel_imager_lon/vel_baloon_lon)
                X[0] = lon_initial
                for ilon in range(1,X.size):
                    X[ilon] = wrap_longitude(X[ilon-1], -dlon_mod)

                LATS_airglow, X = np.meshgrid(lats*0., X)
                LATS_airglow, X = LATS_airglow.ravel(), X.ravel()

            else:
                X = lons_stations
                LATS_airglow = lons_stations*0.
            lats_stations = np.c_[lats_stations, LATS_airglow]
            lons_stations = np.c_[lons_stations, X]
    
    for ioffset, offsets_network in enumerate(offsets):
        lats_other_stations = np.array([LATS]).T
        lons_other_stations = np.array([LONS]).T
        #print(f'offsets_network: {offsets_network}')
        for one_offset in offsets_network:
            lon1 = LONS+one_offset
            lon1[lon1>180] = -180 + abs(lon1[lon1>180]-180.)
            lats_other_stations = np.c_[lats_other_stations, LATS]
            lons_other_stations = np.c_[lons_other_stations, lon1]

        if use_airglow:
            if add_velocity:
                X = np.zeros_like(lons)
                lon_initial = lons[0]
                dlon = abs(lons[1]-lons[0])
                X[0] = lon_initial
                dlon_mod = dlon*(1+rel_imager_lon/vel_baloon_lon)
                for ilon in range(1,X.size):
                    X[ilon] = wrap_longitude(X[ilon-1], -dlon_mod)

                LATS_airglow, X = np.meshgrid(lats*0, X)
                LATS_airglow, X = LATS_airglow.ravel(), X.ravel()
            else:
                X = lons_stations
            lats_other_stations = np.c_[lats_other_stations, LATS_airglow]
            lons_other_stations = np.c_[lons_other_stations, X]

        if ioffset == 0:
            lats_stations = lats_other_stations.copy()
            lons_stations = lons_other_stations.copy()
        else:
            lats_stations = np.r_[lats_stations, lats_other_stations]
            lons_stations = np.r_[lons_stations, lons_other_stations]
        #print(lats_stations.shape)

    for iscenario, (scenario_id, scenario) in enumerate(fixed_stations.items()):
        lats_other_stations = np.array([LATS]).T
        lons_other_stations = np.array([LONS]).T
        for istation, station in enumerate(scenario):

            lats_other_stations = np.c_[lats_other_stations, LATS*0.+station[0]]
            lons_other_stations = np.c_[lons_other_stations, LATS*0.+station[1]]

        if iscenario == 0:
            lats_stations = lats_other_stations.copy()
            lons_stations = lons_other_stations.copy()
        else:
            lats_stations = np.r_[lats_stations, lats_other_stations]
            lons_stations = np.r_[lons_stations, lons_other_stations]
        
    #id_scenario = np.arange(LONS.size)
    id_scenario = np.arange(lons_stations.shape[0])
    id_stat = np.arange(lons_stations.shape[1])

    return lats_stations, lons_stations, id_scenario, id_stat

##########################
if __name__ == '__main__':

    bp()