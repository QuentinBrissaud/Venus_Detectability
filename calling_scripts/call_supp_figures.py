import numpy as np
from pdb import set_trace as bp
from scipy import interpolate
import pandas as pd
import os

import sys
sys.path.append('./Venus_Detectability/')

import VCD_trajectory_modules as VCD
import proba_modules as pm

##########################
if __name__ == '__main__':

    base_data_folder = f'/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/'

    file_atmos = f'{base_data_folder}profile_VCD_for_scaling_pd.csv'
    dir_GF = '/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/'
    file_curve = f'{dir_GF}GF_Dirac_1Hz_combined_wHot40_lognormal_updated_minthick.csv'

    profile = pd.read_csv(file_atmos)
    f_rho = interpolate.interp1d(profile.altitude/1e3, profile.rho, kind='quadratic')
    f_t = interpolate.interp1d(profile.altitude/1e3, profile.t, kind='quadratic')
    f_gamma = interpolate.interp1d(profile.altitude/1e3, profile.gamma, kind='quadratic')
    f_c = interpolate.interp1d(profile.altitude/1e3, profile.c, kind='quadratic')

    #file_curve = './data/GF_Dirac_1Hz_all_wfreq.csv'
    freq = [0.01, 0.1, 1.]
    alt_balloons = [50.]
    #alt_balloons = [60., 65.]
    #scenarios = ['active_low_min', 'active_low_max', 'active_high_min']
    scenarios = ['active_low_max',]
    brunes = [True, False]

    noise_level = 1e-2
    duration_days = 60

    start_locations = [[-45.,0.], [45.,0.], [-45.,45.], [85.,0.]]
    #start_locations = [[-45.,0.],]
    subsurface_models = ['Cold100', 'Hot25', 'Hot10', 'Hot40']

    file_ratio = '/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/surface_ratios/surface_ratios_active_fixed.csv'
    surface_ratios = pm.get_surface_ratios(file_ratio)

    

    for alt_balloon in alt_balloons: 

        file_atmos = '/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/VCD_atmos_globe_new.dat'
        altitude = alt_balloon*1e3
        winds = VCD.get_winds(file_atmos, altitude)

        opt_TL = dict(
            rho0=f_rho(0.), 
            rhob=f_rho(alt_balloon), 
            c0=f_c(0), 
            cb=f_c(alt_balloon), 
            unknown='pressure', 
        )
    
        for subsurface_model in subsurface_models:

            print(f'- Subsurface model: {subsurface_model}')

            file_slopes = f'{base_data_folder}Venus_data/distribution_venus_per_mw_geotherms_{subsurface_model}_f1.0_updated.csv'
            pd_slopes = pm.get_slopes(file_slopes)
            
            shape_new, scale_new = pm.get_lognormal_precomputed(file_curve, **opt_TL, model=subsurface_model)

            for scenario in scenarios:

                for period in shape_new.keys():

                    for use_brune in brunes:
                    
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
                            nb_CPU=8
                        )

                        apply_fc_correction = None if not use_brune else period

                        which_TL_distribution = 'lognormal'
                        amplitude_model = scale_new[period], shape_new[period]
                        proba_model = pm.proba_model_CPUs(pd_slopes, surface_ratios, amplitude_model, apply_fc_correction=apply_fc_correction, which_TL_distribution=which_TL_distribution)
                        proba_model.compute_scores_across_CPUs(**opt_model)

                        ## Tracjectories
                        snrs = proba_model.SNR_thresholds
                        lats, lons = proba_model.all_lats, proba_model.all_lons

                        for start_location in start_locations:

                            opt_trajectory = dict(
                                time_max=3600*24*duration_days,
                                save_trajectory=False,
                                folder = './data/',
                            )
                            trajectory = VCD.compute_trajectory(winds, start_location, **opt_trajectory)

                            ####### WRINKLE RIDGES #######
                            #probas = proba_model_wrinkles.proba_all.copy()

                            ####### TECTONIC #######
                            probas = proba_model.proba_all.copy() # SNR x lats x lons

                            new_trajectories = pm.compute_proba_one_trajectory(trajectory, snrs, lats, lons, probas, norm_factor_time=3600.)

                            folder_traj = './data/trajectories_data/'
                            file = f'traj_a{alt_balloon}_m{subsurface_model}_s{scenario}_p{period}_l{start_location[0]}_{start_location[1]}_brune{use_brune}'
                            filepath = f'{folder_traj}{file}.csv'
                            new_trajectories.to_csv(filepath, header=True, index=False)

                            opt_visualization = dict(
                                VENUS=None,#pm.get_regions('../../../Venus_data/')
                                snr=1., 
                                n_colors=10, 
                                c_cbar='black', 
                                fontsize=15., 
                                ylim=[0., 70.],
                                plot_time=True,
                                plot_volcanoes=False,
                                n_colors_proba = 10, 
                                n_colors_winds = 7,
                                file=f'{folder_traj}Figure_3_balloon_{file}_14.11.2025.pdf'
                            )

                            new_trajectories_total = new_trajectories.copy()
                            new_trajectories_total['seismicity'] = 'low'

                            ####### WRINKLE RIDGES #######
                            #pm.plot_trajectory(new_trajectories_total, proba_model_wrinkles, winds, **opt_visualization)

                            ####### TECTONIC #######
                            pm.plot_trajectory(new_trajectories_total, proba_model, winds, **opt_visualization)