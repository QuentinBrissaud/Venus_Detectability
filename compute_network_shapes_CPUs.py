import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import pyproj
from tqdm import tqdm
import time
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
        gdf = partial_compute_surfaces((0, max_dist))

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
            list_of_lists.append( (i, max_dist[idx_start_all[idx],:]) )

        with get_context("spawn").Pool(processes = N) as p:
            print(f'Running across {N} CPU')
            gdfs = p.map(partial_compute_surfaces, list_of_lists)
            p.close()
            p.join()

        gdf = gpd.GeoDataFrame()
        for idx, gdf_loc in zip(idxs, gdfs):
            gdf_loc['iscenario'] += idx
            gdf = pd.concat([gdf, gdf_loc], ignore_index=True)

    return gdf

def compute_surfaces(thresholds, LATS, LONS, lats_stations, lons_stations, polys, plot, R0, inputs):

    iCPU, max_dist = inputs

    shape_init = LATS.shape
    proj = pyproj.Proj(proj='robin', lat_0=0., lon_0=0., a=R0, b=R0)
    x, y = proj(LONS, LATS)

    gdf = []
    #for iscenario in tqdm(range(500,503)):
    for iscenario in tqdm(range(max_dist.shape[0]), disable=not iCPU == 0):

        #print(f'Scenario {iscenario}')
        max_map = max_dist[iscenario,:].reshape(shape_init)

        dict_scenario = dict(iscenario=iscenario)
        lats_loc = lats_stations[iscenario,:]
        lons_loc = lons_stations[iscenario,:]
        for istat in range(lats_stations.shape[1]):
            dict_scenario[f'lat_station_{istat}'] = lats_loc[istat]
            dict_scenario[f'lon_station_{istat}'] = lons_loc[istat]

        inds_last = []
        first_pass_done = False
        for threshold in thresholds[:]:

            #print(f'threshold {first_pass_done} - {threshold}')
            inds = np.where(max_map.ravel()/1e3<=threshold)[0] 
            if inds.size == 0:
                continue

            inds_to_process = np.setdiff1d(inds, inds_last)
            polys_processed_temp = unary_union([polys[ii] for ii in inds_to_process])
            if first_pass_done:
                #print('-->', polys_loc)
                polys_processed_temp = unary_union([polys_processed, polys_processed_temp])
            else:
                first_pass_done = True
            #print(polys_loc)
            #print(polys_loc_temp)
            polys_processed = polys_processed_temp
            inds_last = inds
            
            dict_loc = dict(geometry=polys_processed, distance=threshold,)
            dict_loc.update(dict_scenario)

            gdf.append(dict_loc)

            if plot:
                plt.figure()
                ax = plt.gca()

                #print(f'Threshold {threshold}')
                if polys_processed.geom_type == 'MultiPolygon':
                    polys_loc = polys_processed.geoms
                else:
                    polys_loc = [polys_processed]

                #print(polys_loc)
                for poly in polys_loc:
                    #print(poly)
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
                ax.set_title(threshold)

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

def get_airglow_scaling(TL_new, period, file_scaling, R0=6052000, sigma_balloon=1e-2, boost_SNR=1., m0=7.,):

    scaling = pd.read_csv(file_scaling, header=[0])
    if 'period' in scaling.columns:
        diff = abs(scaling.period-period)
        scaling = scaling.loc[diff==diff.min()]
    f_airglow = interpolate.interp1d(scaling.distance, scaling.SNR, kind='nearest', bounds_error=False, fill_value=(scaling.SNR.iloc[0], scaling.SNR.iloc[-1]))

    distances = np.linspace(0., np.pi*R0/1.001, 300)/1e3
    distances_r0 = np.linspace(-np.pi*R0/1.001, np.pi*R0/1.001, 300)/1e3
    DIST, DIST_R0 = np.meshgrid(distances, distances_r0)

    diff = abs(TL_new(DIST+DIST_R0, m0)/sigma_balloon-boost_SNR*f_airglow(DIST))
    flipped_diff = np.flip(diff, axis=0)
    flipped_indices = flipped_diff.argmin(axis=0) ## In order to get argmin to return the largest index if multiple minima
    original_indices = diff.shape[0] - 1 - flipped_indices
    #original_indices = diff.argmin(axis=0)
    coefs = distances_r0[original_indices]
    f_alt_scaling = interpolate.interp1d(distances, coefs, bounds_error=False, fill_value=(coefs[0], coefs[-1]))

    return f_alt_scaling

def get_max_dist(lats_stations, lons_stations, LATS, LONS, id_scenario, id_stat, s_cluster=100, use_airglow=False, which_stat_is_airglow=1, R_airglow=5000., f_alt_scaling=None):

    ind_without_airglow = id_stat
    if use_airglow:
        ind_without_airglow = np.setdiff1d(id_stat, np.array([which_stat_is_airglow]))
        ind_with_airglow = np.array([which_stat_is_airglow])

    max_dist = np.zeros((lats_stations.shape[0], LONS.size))
    clusters = np.arange(0, lats_stations.shape[0]-1, s_cluster)
    for _, i_cluster in tqdm(enumerate(clusters), total=clusters.size):
        
        inds_loc = np.arange(i_cluster, min(i_cluster+s_cluster,lats_stations.shape[0]))
        id_ref_quake, all_id_scenarios, all_id_stat = np.meshgrid(np.arange(LONS.size), id_scenario[inds_loc], ind_without_airglow)
        shape_init_ref = id_ref_quake.shape
        #print(shape_init_ref)
        id_ref_quake, all_id_scenarios, all_id_stat = id_ref_quake.ravel(), all_id_scenarios.ravel(), all_id_stat.ravel()

        max_dist_loc = haversine_distance(LONS[id_ref_quake], LATS[id_ref_quake], lons_stations[all_id_scenarios, all_id_stat].ravel(), lats_stations[all_id_scenarios, all_id_stat].ravel())
        max_dist[inds_loc,:] = max_dist_loc.reshape(shape_init_ref).max(axis=-1)

        if use_airglow:
            id_ref_quake, all_id_scenarios, all_id_stat = np.meshgrid(np.arange(LONS.size), id_scenario[inds_loc], ind_with_airglow)
            shape_init_ref = id_ref_quake.shape
            id_ref_quake, all_id_scenarios, all_id_stat = id_ref_quake.ravel(), all_id_scenarios.ravel(), all_id_stat.ravel()
            max_dist_airglow = haversine_distance(LONS[id_ref_quake], LATS[id_ref_quake], lons_stations[all_id_scenarios, all_id_stat].ravel(), lats_stations[all_id_scenarios, all_id_stat].ravel())
            max_dist_airglow -= R_airglow*1e3 ## Accounting for large field of view
            max_dist_airglow[max_dist_airglow<0] = 0.
            max_dist_airglow += f_alt_scaling(max_dist_airglow/1e3)*1e3
            max_dist_airglow[max_dist_airglow<0] = 0.
            max_dist_airglow = max_dist_airglow.reshape(shape_init_ref).max(axis=-1)
            #print(max_dist_airglow.shape, max_dist[inds_loc,:].shape, shape_init_ref)
            
            max_dist[inds_loc,:] = np.max(np.stack((max_dist[inds_loc,:], max_dist_airglow), axis=-1), axis=-1)

    
    return max_dist

def get_max_dist_airglow(lats_stations, lons_stations, LATS, LONS, which_stat_is_airglow=1, R_airglow=5000.*1e3,):

    max_dist = np.zeros((lats_stations.shape[0], LONS.size))
    for istation, (lat, lon) in tqdm(enumerate(zip(lats_stations, lons_stations)), total=lons_stations.shape[0]):

        lats_loc, lons_loc = np.repeat(lat, LATS.size), np.repeat(lon, LONS.size)

        max_dist[istation,:] = haversine_distance(LONS, LATS, lons_loc, lats_loc)
        max_dist[istation,:] -= R_airglow
        max_dist[istation, max_dist[istation,:]<0] = 0.
    
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

def get_stations(lats, lons, offsets=[[2., 4.], [10., 20.]], fixed_stations=dict(scenario_1=[[0., 0.]])):

    LATS, LONS = np.meshgrid(lats, lons)
    LATS, LONS = LATS.ravel(), LONS.ravel()

    if not offsets and not fixed_stations:
        lats_stations = np.array([LATS]).T
        lons_stations = np.array([LONS]).T
    
    for ioffset, offsets_network in enumerate(offsets):

        lats_other_stations = np.array([LATS]).T
        lons_other_stations = np.array([LONS]).T
        for one_offset in offsets_network:
            lon1 = LONS+one_offset
            lon1[lon1>180] = -180 + abs(lon1[lon1>180]-180.)
            lats_other_stations = np.c_[lats_other_stations, LATS]
            lons_other_stations = np.c_[lons_other_stations, lon1]

        if ioffset == 0:
            lats_stations = lats_other_stations.copy()
            lons_stations = lons_other_stations.copy()
        else:
            lats_stations = np.r_[lats_stations, lats_other_stations]
            lons_stations = np.r_[lons_stations, lons_other_stations]

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

    dlon = 2.
    lats, lons = np.arange(-90, 89, dlon/2.), np.arange(-180, 179, dlon)
    LATS, LONS, shape_init = get_grid(lats, lons)

    polys = get_polys(LONS, LATS, dlon, R0=6052000)

    dlon_stations = 5.
    lats, lons = np.arange(-90, 89, dlon/2.), np.arange(-180, 179, dlon)
    lats_stations, lons_stations, id_scenario, id_stat = get_stations(lats, lons, offsets=[[2., 4.]], fixed_stations=dict())

    file_max_dist = './data/max_dist/max_dist_2_4.npy'
    max_dist = np.load(file_max_dist, mmap_mode='r')

    plot = False
    R0 = 6052000
    thresholds = np.arange(10000, np.pi*R0/1.001, 5e5)[:]/1e3
    #gdf = compute_surfaces(LATS, LONS, lats_stations, lons_stations, polys, plot, R0, max_dist)
    gdf = compute_surfaces_CPUs(thresholds, LATS, LONS, lats_stations, lons_stations, polys, plot, R0, max_dist, nb_CPU=1)
    gdf.to_file("./data/network_shp/network_2_4.shp")
    bp()