import numpy as np
from pdb import set_trace as bp
from scipy import interpolate
import pandas as pd
from tqdm import tqdm
import pyproj
import os

import sys
sys.path.append('./Venus_Detectability/')

import proba_modules as pm

def merge_and_fix_surface_ratio_region(pattern, regions=['corona', 'rift', 'ridge', 'intraplate'], set_minradius_to_zero=True, write=False, set_final_radius_to_prev=False):

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

    if set_final_radius_to_prev:
        all_data.reset_index(drop=True, inplace=True)
        for grp_name, group in all_data.groupby(['iloc', 'period', 'region']):
            idx_prev = group.loc[group.iradius==group.iradius.max()-1].iloc[0].name
            idx = group.loc[group.iradius==group.iradius.max()].iloc[0].name
            ratio = group.loc[group.index==idx_prev, 'ratio'].iloc[0]
            ratio_map = group.loc[group.index==idx_prev, 'ratio_map'].iloc[0]
            all_data.loc[all_data.index==idx, 'ratio'] = ratio
            all_data.loc[all_data.index==idx, 'ratio_map'] = ratio_map

    if write:
        all_data.to_csv(pattern.format('all'), header=True, index=False)


    return all_data

def compute_proba_cte_velocity_traj(lons, snrs, proba_profile, vel_horiz, duration_months=6, init_location=(0,0), azimuth=90., R0=6052000):
    
    g = pyproj.Geod(proj='robin', lat_0=0., lon_0=0., a=R0, b=R0)   
    duration_hours = np.arange(duration_months*30*24)
    distances = duration_hours*vel_horiz*1e3*3600
    n_distances = len(distances)
    lat, lon = np.repeat(init_location[0], n_distances), np.repeat(init_location[1], n_distances)
    angles = np.tile(azimuth, n_distances)
    endlon, _, _ = g.fwd(lon, lat, angles, distances)
    endlon[endlon<0] += 360.
    
    probas_traj = np.zeros((snrs.size, duration_hours.size))
    for isnr, snr in tqdm(enumerate(snrs), total=snrs.size):
        f_proba_vs_lon = interpolate.interp1d(lons, proba_profile[isnr,:], kind='quadratic', bounds_error=False, fill_value=(proba_profile[isnr,0], proba_profile[isnr,-1]))
        
        probas_traj_loc = f_proba_vs_lon(endlon)
        probas_traj[isnr,:] = 1. - np.cumprod(1. - probas_traj_loc)
        
    return duration_hours, endlon, probas_traj

##########################
if __name__ == '__main__':

    base_data_folder = f'/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/'

    file_atmos = f'{base_data_folder}profile_VCD_for_scaling_pd.csv'
    dir_GF = '/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/'
    dir_final_probas = './data/final_probas/'
    file_curve = f'{dir_GF}GF_Dirac_1Hz_combined_wHot40_lognormal_updated_minthick.csv'
    set_final_radius_to_prev = True

    profile = pd.read_csv(file_atmos)
    f_rho = interpolate.interp1d(profile.altitude/1e3, profile.rho, kind='quadratic')
    f_t = interpolate.interp1d(profile.altitude/1e3, profile.t, kind='quadratic')
    f_gamma = interpolate.interp1d(profile.altitude/1e3, profile.gamma, kind='quadratic')
    f_c = interpolate.interp1d(profile.altitude/1e3, profile.c, kind='quadratic')

    SNRs = [1., 75.]
    #SNRs = [150.]

    #file_curve = './data/GF_Dirac_1Hz_all_wfreq.csv'
    freq = [0.01, 0.1, 1.]
    scenarios = ['active_low_min', 'active_low_max', 'active_high_min']
    scenarios = ['active_low_max',]

    brunes = [True, False]

    noise_level = 1e-2
    duration_days = 30*24.
    vel_horiz = 0.3 # Project saatellite velocity

    subsurface_models = ['Cold100', 'Hot25', 'Hot10', 'Hot40']
    subsurface_models = ['Cold100',]

    for SNR in SNRs:

        for subsurface_model in subsurface_models:

            print(f'- Subsurface model: {subsurface_model}')

            folder_sr = '/staff/quentin/Documents/Projects/2024_Venus_Detectability/data/surface_ratios/'
            pattern_file_dayglow = f'dayglow_SNRnight{SNR}_SNRday{SNR}_same_event_m{subsurface_model}_20.12.2025'
            file_sr_fmt = f'{folder_sr}surface_ratios_{pattern_file_dayglow}'
            pattern = file_sr_fmt + '_{}_active_20.12.2025.csv'
            surface_ratios_periods_dayglow = merge_and_fix_surface_ratio_region(pattern, regions=['corona', 'rift', 'ridge', 'intraplate'], write=False, set_final_radius_to_prev=set_final_radius_to_prev)

            pattern_file_nightglow = f'nightglow_SNRnight{SNR}_SNRday{SNR}_same_event_m{subsurface_model}_20.12.2025'
            file_sr_fmt = f'{folder_sr}surface_ratios_{pattern_file_nightglow}'
            pattern = file_sr_fmt + '_{}_active_20.12.2025.csv'
            surface_ratios_periods_nightglow = merge_and_fix_surface_ratio_region(pattern, regions=['corona', 'rift', 'ridge', 'intraplate'], write=False, set_final_radius_to_prev=set_final_radius_to_prev)

            alt = surface_ratios_periods_dayglow.alt_ref_balloon.iloc[0]

            file_slopes = f'{base_data_folder}Venus_data/distribution_venus_per_mw_geotherms_{subsurface_model}_f1.0_updated.csv'
            pd_slopes = pm.get_slopes(file_slopes)

            opt_TL = dict(
                rho0=f_rho(0.), 
                rhob=f_rho(alt), 
                c0=f_c(0), 
                cb=f_c(alt), 
                unknown='pressure', 
                model=subsurface_model
            )
            
            shape_new, scale_new = pm.get_lognormal_precomputed(file_curve, **opt_TL,)

            for scenario in scenarios:

                periods = shape_new.keys()
                #periods = [0.01, 0.1]
                for period in periods:

                    for use_brune in brunes:
                        
                        pattern_proba_nightglow = f'probas_{pattern_file_nightglow}_{period}Hz_brune{use_brune}_{scenario}_20.12.2025_test.csv'
                        pattern_proba_dayglow = f'probas_{pattern_file_dayglow}_{period}Hz_brune{use_brune}_{scenario}_20.12.2025_test.csv'

                        if os.path.exists(f'{dir_final_probas}{pattern_proba_nightglow}') and os.path.exists(f'{dir_final_probas}{pattern_proba_dayglow}'):
                            print(f'Skip files like: {dir_final_probas}{pattern_proba_dayglow}')
                            #continue
                    
                        print(f'- period: {period}')

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
                        )

                        apply_fc_correction = None if not use_brune else period
                        which_TL_distribution = 'lognormal'
                        amplitude_model = scale_new[period], shape_new[period]
                        #amplitude_model = scale_new[0.01], shape_new[0.01] # DEBUG remove to make sense

                        #proba_model = pm.proba_model_CPUs(pd_slopes, surface_ratios, amplitude_model, apply_fc_correction=apply_fc_correction, which_TL_distribution=which_TL_distribution)
                        #proba_model.compute_scores_across_CPUs(**opt_model)

                        opt_init = dict(
                            apply_fc_correction=None, 
                            which_TL_distribution='lognormal',
                            use_v_scaler=False,  
                            photons_dayglow=3.5e5, 
                            photons_nightglow=2e4, 
                            #data_scaling=data_scaling.loc[(data_scaling.f1<=freq)&(data_scaling.f2>=freq)].iloc[0]*1
                        )

                        surface_ratios = surface_ratios_periods_dayglow.loc[surface_ratios_periods_dayglow.period==1./period]
                        proba_model_dayglow = pm.proba_model_airglow(pd_slopes, surface_ratios, amplitude_model, type_airglow='dayglow', **opt_init)
                        proba_model_dayglow.compute_proba_map(**opt_model)

                        surface_ratios = surface_ratios_periods_nightglow.loc[surface_ratios_periods_nightglow.period==1./period]
                        proba_model_nightglow = pm.proba_model_airglow(pd_slopes, surface_ratios, amplitude_model, type_airglow='nightglow', **opt_init)
                        proba_model_nightglow.compute_proba_map(**opt_model)

                        lons = proba_model_dayglow.all_lons
                        snrs = proba_model_dayglow.SNR_thresholds

                        opt_traj = dict(
                            duration_months=duration_days/30., 
                            init_location=(0,0), 
                            azimuth=90.
                        )

                        ilat = np.argmin(abs(opt_model['all_lats']))
                        proba_profile = proba_model_dayglow.proba_all[:,ilat,:]
                        duration_hours, endlon, probas_traj_dayglow = compute_proba_cte_velocity_traj(lons, snrs, proba_profile, vel_horiz, **opt_traj)

                        proba_profile = proba_model_nightglow.proba_all[:,ilat,:]
                        duration_hours, endlon, probas_traj_nightglow = compute_proba_cte_velocity_traj(lons, snrs, proba_profile, vel_horiz, **opt_traj)

                        SNRS, DUR = np.meshgrid(snrs, duration_hours)
                        probas_airglow = pd.DataFrame(np.c_[SNRS.ravel(), DUR.ravel(), (probas_traj_nightglow.T).ravel()], columns=['SNR', 'hour', 'proba'])
                        probas_airglow.to_csv(f'{dir_final_probas}{pattern_proba_nightglow}', header=True, index=False)

                        probas_airglow = pd.DataFrame(np.c_[SNRS.ravel(), DUR.ravel(), (probas_traj_dayglow.T).ravel()], columns=['SNR', 'hour', 'proba'])
                        probas_airglow.to_csv(f'{dir_final_probas}{pattern_proba_dayglow}', header=True, index=False)

    bp()