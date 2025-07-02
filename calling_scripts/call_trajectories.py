import numpy as np
from pdb import set_trace as bp
from scipy import interpolate
import pandas as pd
import os

import sys
sys.path.append('./Venus_Detectability/')

import VCD_trajectory_modules as VCD
import proba_modules as pm

def merge_and_fix_surface_ratio_region(pattern, regions=['corona', 'rift', 'ridge', 'intraplate'], set_minradius_to_zero=True, write=False):

    ## e.g., pattern = './data/surface_ratios_{region}_active.csv'

    all_data = pd.DataFrame()
    for region in regions:
        data = pd.read_csv(pattern.format(region), header=[0])
        data['region'] = region
        all_data = pd.concat([all_data, data])
    all_data.reset_index(drop=True, inplace=True)

    iloc = -1
    for _, group in all_data.groupby(['lon', 'lat']):
        iloc += 1
        all_data.loc[all_data.index.isin(group.index), 'iloc'] = iloc

    if set_minradius_to_zero:
        all_data.loc[all_data.iradius==0, 'ratio'] = 0.
        all_data.loc[all_data.iradius==0, 'ratio_map'] = 0.

    all_data.loc[all_data.lon < 0, 'lon'] += 360.

    if write:
        all_data.to_csv(pattern.format('all'), header=True, index=False)

    return all_data

##########################
if __name__ == '__main__':

    base_data_folder = f'/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/'

    #file_ratio = f'{base_data_folder}surface_ratios/surface_ratios_active_fixed.csv'
    #file_ratio = f'{base_data_folder}surface_ratios/surface_ratios_active_fixed.csv'
    file_atmos = f'{base_data_folder}profile_VCD_for_scaling_pd.csv'
    file_slopes = f'{base_data_folder}Venus_data/distribution_venus_per_mw.csv'

    profile = pd.read_csv(file_atmos)
    f_rho = interpolate.interp1d(profile.altitude/1e3, profile.rho, kind='quadratic')
    f_t = interpolate.interp1d(profile.altitude/1e3, profile.t, kind='quadratic')
    f_gamma = interpolate.interp1d(profile.altitude/1e3, profile.gamma, kind='quadratic')
    f_c = interpolate.interp1d(profile.altitude/1e3, profile.c, kind='quadratic')

    #file_curve = './data/GF_Dirac_1Hz_all_wfreq.csv'
    freq = [0.01, 0.1, 1.]
    alt_balloons = [55., 65., 45.]
    scenarios = ['active_low_min', 'active_low_max', 'active_high_min', 'active_high_max']

    freq = [0.01, 0.1, 1.]
    alt_balloons = [50., 55., 60., 65.]
    scenarios = ['active_low_min',]

    noise_level = 1e-2

    pd_slopes = pm.get_slopes(file_slopes)

    ## proc7
    type_detection = 'same_event'
    subsurface_models = ['Cold100', 'Hot25', 'Hot10', 'Hot40']

    ## proc3
    #type_detection = 'any_event'
    #subsurface_models = ['Cold100', 'Hot25', 'Hot10', 'Hot40']

    ## proc1
    #type_detection = 'any_event' ## One balloon so dummy value
    #subsurface_models = ['Cold100', 'Hot25', 'Hot10', 'Hot40']

    ## saturn
    #type_detection = 'same_event'
    #subsurface_models = ['Cold100']

    ## Balloon drop off locations
    lats = np.linspace(-65., 65., 25)
    lons = np.linspace(-180., 179., 30)
    LATS, LONS = np.meshgrid(lats, lons)
    LATS, LONS = LATS.ravel(), LONS.ravel()
    mission_durations = [30., 60., 90., 120., 150., 180.]
    max_number_months = np.max(mission_durations)/30.

    ## Sensor surface ratios
    type_sensors = dict(
        #network='/staff/quentin/Documents/Projects/2024_Venus_Detectability/data/surface_ratios/surface_ratios_network_balloon_20.0_50.0_any_event_{}_active.csv',
        network='/staff/quentin/Documents/Projects/2024_Venus_Detectability/data/surface_ratios/surface_ratios_network_balloon_20.0_50.0_same_event_{}_active.csv',
        #dayglow_up='/staff/quentin/Documents/Projects/2024_Venus_Detectability/data/surface_ratios/surface_ratios_dayglow_SNRnight1.0_SNRday1_same_event_m'+subsurface_model+'_{}_active.csv'
        #nightglow_dayglow_up='/staff/quentin/Documents/Projects/2024_Venus_Detectability/data/surface_ratios/surface_ratios_nightglow_dayglow_SNRnight1.0_SNRday1_same_event_m'+subsurface_model+'_{}_active.csv'
        #airglow_one_balloon='/staff/quentin/Documents/Projects/2024_Venus_Detectability/data/surface_ratios/surface_ratios_nightglow_dayglow_one_balloon_SNRnight10.0_SNRday1__same_event_corrected_{}_active.csv',
        #one_balloon='/staff/quentin/Documents/Projects/2024_Venus_Detectability/data/surface_ratios/surface_ratios_one_balloon_{}_active.csv'
    )
    
    surface_ratios = dict()
    for name_sensor, pattern in type_sensors.items():
        surface_ratios[name_sensor] = merge_and_fix_surface_ratio_region(pattern, regions=['corona', 'rift', 'ridge', 'intraplate'], write=False)

    for alt_balloon in alt_balloons: 

        print('------------------------------------')
        print(f'- Alt balloon: {alt_balloon}')

        ## Check if all files for this altitude already exist
        all_available = True
        for subsurface_model in subsurface_models:
            for name_sensor, pattern in type_sensors.items():
                for scenario in scenarios:
                    dir_GF = '/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/'
                    file_curve = f'{dir_GF}GF_Dirac_1Hz_combined_wHot40.csv'
                    TL_new, TL_new_qmin, TL_new_qmax = pm.get_TL_curves_precomputed(file_curve, rho0=f_rho(0.), rhob=f_rho(alt_balloon), cb=f_c(alt_balloon), unknown='pressure', model=subsurface_model)
                    for period in TL_new.keys():
                        file_proba = f'/staff/quentin/Documents/Projects/2024_Venus_Detectability/data/final_probas/final_probas_{name_sensor}_{type_detection}_{scenario}_n{noise_level}_alt{alt_balloon:.0f}_{1./period:.0f}s_{subsurface_model}_19.06.2025.csv'
                        if not os.path.isfile(file_proba) and all_available:
                            print('- Files needed to be computed for this balloon altitude')
                            all_available = False
                            break

        if all_available:
            print('- All files existing for this altitude, skipping')
            continue

        ## Wind interpolators
        print('-- build interpolators')
        file_atmos = f'{base_data_folder}VCD_atmos_globe_new.dat'
        winds = VCD.get_winds(file_atmos, alt_balloon*1e3)
        wind_direction_interpolator, wind_strength_interpolator, _ = VCD.get_winds_interpolator(None, alt_balloon*1e3, winds=winds)

        for subsurface_model in subsurface_models:

            print(f'- Subsurface model: {subsurface_model}')

            #TL_new, TL_new_qmin, TL_new_qmax = pm.get_TL_curves(file_curve, freq, dist_min=100., rho0=f_rho(0.), rhob=f_rho(alt_balloon), use_savgol_filter=True, plot=False, scalar_moment=10e6, unknown='pressure', return_dataframe=False)
            dir_GF = '/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/'
            file_curve = f'{dir_GF}GF_Dirac_1Hz_combined_wHot40.csv'
            TL_new, TL_new_qmin, TL_new_qmax = pm.get_TL_curves_precomputed(file_curve, rho0=f_rho(0.), rhob=f_rho(alt_balloon), cb=f_c(alt_balloon), unknown='pressure', model=subsurface_model)

            for name_sensor, pattern in type_sensors.items():

                #surface_ratios = merge_and_fix_surface_ratio_region(pattern, regions=['corona', 'rift', 'ridge', 'intraplate'], write=False)

                for scenario in scenarios:

                    for period in TL_new.keys():
                        
                        print(f'- period: {period}')

                        #file_proba = f'/staff/quentin/Documents/Projects/2024_Venus_Detectability/data/final_probas_network_{scenario}_n{noise_level}_alt{alt_balloon:.0f}_{1./period:.0f}s.csv'
                        file_proba = f'/staff/quentin/Documents/Projects/2024_Venus_Detectability/data/final_probas/final_probas_{name_sensor}_{type_detection}_{scenario}_n{noise_level}_alt{alt_balloon:.0f}_{1./period:.0f}s_{subsurface_model}_19.06.2025.csv'
                        #file_proba = f'/staff/quentin/Documents/Projects/2024_Venus_Detectability/data/final_probas_{name_sensor}_{scenario}_n{noise_level}_alt{alt_balloon:.0f}_{1./period:.0f}s.csv'
                        if os.path.isfile(file_proba):
                            print(f'Skipping already existing file: {file_proba}')
                            continue

                        print('- Compute hourly detectability map')
                        dlat = 5.
                        r_venus = 6052
                        opt_model = dict(
                            scenario = scenario, # Iris' seismicity scenario
                            dists = np.arange(10., np.pi*r_venus, 200), # Low discretization will lead to terrible not unit integrals
                            M0s = np.linspace(3., 8., 30), # Low discretization will lead to terrible not unit integrals
                            SNR_thresholds = np.linspace(0.1, 10., 50),
                            noise_level = noise_level, # noise level in Pa
                            duration = 1./(365.*24.), # (1/mission_duration)
                            all_lats = np.arange(-89, 90, dlat),
                            all_lons = np.arange(0, 359, dlat*2),
                            #all_lats = np.arange(-90., 90.+dlat, dlat),
                            #all_lons = np.arange(-180, 180+dlat*2, dlat*2),
                            homogeneous_ratios = False,
                            m_min = 3.,
                            r_venus = r_venus,
                            nb_CPU=8
                        )

                        #proba_model = pm.proba_model(pd_slopes, surface_ratios[name_sensor], TL_new[period], TL_new_qmin[period], TL_new_qmax[period])
                        #proba_model.compute_proba_map(**opt_model)
                        apply_fc_correction = None
                        proba_model = pm.proba_model_CPUs(pd_slopes, surface_ratios[name_sensor], TL_new[period], TL_new_qmin[period], TL_new_qmax[period], apply_fc_correction=apply_fc_correction)
                        proba_model.compute_scores_across_CPUs(**opt_model)

                        print('--------------------------')
                        print('Integrate along trajectory')
                        #pd_final_probas = pm.compute_multiple_trajectories_CPUs(proba_model, winds, LATS, LONS, mission_durations, max_number_months, nb_CPU=10)
                        pd_final_probas = pm.compute_multiple_trajectories_vectorized_CPUs(proba_model, wind_direction_interpolator, wind_strength_interpolator, LATS, LONS, mission_durations, nb_CPU=10)
                        pd_final_probas.to_csv(file_proba, header=True, index=False)