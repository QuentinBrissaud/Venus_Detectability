import numpy as np
from pdb import set_trace as bp
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('./Venus_Detectability/')

import compute_network_shapes_CPUs as cns

def plot_one_max_dist(id, period, max_dist_loc, use_airglow=False, vmin=1e3, vmax=19e3):

    plt.figure()
    sc = plt.scatter(LONS, LATS, c=max_dist_loc[id,:]/1e3, s=1, vmin=vmin, vmax=vmax)
    #max_dist_loc_threshold = max_dist_loc[id,:].copy()
    #max_dist_loc_threshold
    plt.colorbar(sc)
    plt.scatter(lons_stations[id,:], lats_stations[id,:], c='red', marker='^')

    if use_airglow:
        circle = plt.Circle((lons_stations[id,-1], lats_stations[id,-1]), opt_dist['radius_view'], edgecolor='red', ls='--', facecolor='none', linewidth=2, label='FOV')
        plt.gca().add_artist(circle)

        colors = dict(dayglow='tab:blue', nightglow='tab:pink')
        for airglow, lon0 in opt_dist['lon_0_airglow'].items():
            radius = opt_dist['radius_airglow'][airglow]
            plt.scatter(lon0, 0., marker='o', label=airglow, color=colors[airglow])
            circle = plt.Circle((lon0, 0.), radius, edgecolor=colors[airglow], ls='--', facecolor='none', linewidth=2,)
            plt.gca().add_artist(circle)

    plt.title(f'Period = {period} s')
    plt.legend(frameon=False)

##########################
if __name__ == '__main__':

    main_dir = '/staff/quentin/Documents/Projects/2024_Venus_Detectability/'
    use_airglow_max_dist = True
    airglow_and_balloon = True
    type_detection = 'same_event'
    #type_detection = 'any_event'
    offsets = [[20., 50.]]
    #offsets = []
    boost_SNR=dict(dayglow=1, nightglow=10.)
    boost_SNR=dict(dayglow=1, nightglow=1.)
    airglow_considered=['nightglow',]
    airglow_considered=['dayglow',]
    model_subsurface = 'Cold100'
    airglow_considered=['nightglow', 'dayglow',]
       
    #################
    ## Discretization
    dlon = 2.
    lats, lons = np.arange(-90, 89, dlon/2.), np.arange(-180, 179, dlon)
    LATS, LONS, shape_init = cns.get_grid(lats, lons)

    polys = cns.get_polys(LONS, LATS, dlon, R0=6052000)

    ##########
    ## Airglow
    #file_curve = f'{main_dir}data/GF_Dirac_1Hz_all_wfreq.csv'
    #file_precomputed_scaling = f'{main_dir}data/data_airglow_scaling.csv'
    dir_GF = '/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/'
    file_curve = f'{dir_GF}GF_Dirac_1Hz_combined.csv'
    file_precomputed_scaling = f'{main_dir}data/data_airglow_scaling_from_seismo.csv'
    dict_scaling = dict(file_precomputed_scaling=file_precomputed_scaling, R0=6052000, sigma_balloon=1e-2, boost_SNR=boost_SNR, photons_dayglow=3.5e5, alpha_dayglow=1e-5, photons_nightglow=2e4, beta=1., file_atmos=f'{main_dir}Venus_Detectability/data/profile_VCD_for_scaling_pd.csv', file_nightglow=f'{main_dir}Venus_Detectability/data/VER_profile_scaled.csv', file_dayglow=f'{main_dir}Venus_Detectability/data/VER_profile_dayglow.csv', model_subsurface=model_subsurface)
    freq = [1e-2, 1e-1, 1.]
    if use_airglow_max_dist:
        f_alt_scaling_dayglow, f_alt_scaling_nightglow, TL_new_v, TL_new_p = cns.get_airglow_scaling(file_curve, freq, **dict_scaling)
    else:
        f_alt_scaling_nightglow = {0.: None}

    ############################
    ## Compute station locations
    dlon_stations = 5.
    lats, lons = np.arange(-90, 89, dlon_stations/2.), np.arange(-180, 179, dlon_stations)
    if use_airglow_max_dist:
        offsets = []
        dict_stations = dict(offsets=offsets, use_airglow=True, use_only_airglow=not airglow_and_balloon, fixed_stations=dict(), add_velocity=True, vel_baloon=0.1, vel_imager=0.33)
    else:
        dict_stations = dict(offsets=offsets, use_airglow=False, use_only_airglow=False, fixed_stations=dict(), add_velocity=True, vel_baloon=0.1, vel_imager=0.33)
    lats_stations, lons_stations, id_scenario, id_stat = cns.get_stations(lats, lons, **dict_stations)

    ############################
    ## Compute maximum distances
    if use_airglow_max_dist: ## Airglow
        which_stat_is_airglow = 0 if not airglow_and_balloon else 1
        opt_dist = dict(s_cluster=100, use_airglow=dict_stations['use_airglow'], which_stat_is_airglow=which_stat_is_airglow, lon_0_airglow=dict(nightglow=180., dayglow=0.), radius_airglow=dict(nightglow=60., dayglow=70.), radius_view=60., airglow_considered=airglow_considered, type_detection=type_detection)
        max_dist = dict()
        for period, f_alt_scaling_nightglow_period in f_alt_scaling_nightglow.items():
            f_alt_scaling = dict(nightglow=f_alt_scaling_nightglow_period, dayglow=f_alt_scaling_dayglow[period])
            #f_alt_scaling = dict(nightglow=f_alt_scaling_nightglow_period, dayglow=f_alt_scaling_nightglow_period)
            max_dist_loc = cns.get_max_dist(lats_stations, lons_stations, LATS, LONS, id_scenario, id_stat, f_alt_scaling=f_alt_scaling, **opt_dist)
            max_dist[period] = max_dist_loc.copy()
    else: ## Only balloons
        opt_dist = dict(s_cluster=100, use_airglow=False, type_detection=type_detection)
        max_dist = {1.: cns.get_max_dist(lats_stations, lons_stations, LATS, LONS, id_scenario, id_stat, **opt_dist)}

    ################
    ## Save max dist
    save_max_dist = False
    str_offsets = '-'.join(['_'.join(np.array(l_offsets).astype(str)) for l_offsets in dict_stations['offsets']])
    if save_max_dist:
        #max_dist = np.load('./max_dist.npy', mmap_mode='r')
        for period, max_dist_loc in max_dist.items():
            file = f'{main_dir}data/max_dist/max_dist_{str_offsets}_{period:.0f}s.npy'
            with open(file, 'wb') as f:
                np.save(f, max_dist_loc)

    #################
    ## Compute shapes
    plot = False
    R0 = 6052000
    thresholds = np.arange(10000, np.pi*R0/1.001, 5e5)[:]/1e3
    gdf_all = pd.DataFrame()
    for period, max_dist_loc in max_dist.items():
        print(f'Period {period}')
        gdf_loc = cns.compute_surfaces_CPUs(thresholds, LATS, LONS, lats_stations, lons_stations, polys, plot, R0, max_dist_loc, nb_CPU=10)
        gdf_loc['period'] = period
        gdf_all = pd.concat([gdf_all, gdf_loc])
    gdf_all.reset_index(drop=True, inplace=True)

    ##############
    ## Save shapes

    ext = 'shp'
    ext = 'gpkg'

    """
    base_name = '_'.join(airglow_considered)
    file = f"./data/airglow_shp/{base_name}_SNRnight{dict_scaling['boost_SNR']['nightglow']}_SNRday{dict_scaling['boost_SNR']['dayglow']}_{type_detection}_m{model_subsurface}.{ext}"
    """

    base_name = '_'.join(airglow_considered)
    file = f"./data/airglow_shp/{base_name}_SNRnight{dict_scaling['boost_SNR']['nightglow']}_SNRday{dict_scaling['boost_SNR']['dayglow']}_{type_detection}_m{model_subsurface}_wballoons.{ext}"
    
    """
    base_name = 'one_balloon'
    base_name = 'network_balloon'
    file = f"{main_dir}data/network_shp/{base_name}_{str_offsets}_{type_detection}.{ext}"
    """
    """
    base_name = f'{"_".join(airglow_considered)}_one_balloon'
    file = f"./data/airglow_shp/{base_name}_SNRnight{dict_scaling['boost_SNR']['nightglow']}_SNRday{dict_scaling['boost_SNR']['dayglow']}_{str_offsets}_{type_detection}{ext}"
    """
    if ext == 'shp':
        gdf_all.to_file(file)
    elif ext == 'gpkg':
        gdf_all.to_file(file, driver="GPKG")
    else:
        print(f'Not saved with extension {ext}')

    bp()