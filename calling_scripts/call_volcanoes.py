import numpy as np
from pdb import set_trace as bp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate

import sys
sys.path.append('./Venus_Detectability/')

import VCD_trajectory_modules as VCD
import proba_volcanoes_modules as pvm
import proba_modules as pm

def get_trajectories(mission_duration, dt, dt_new, lat_vol, lon_vol, lat_offset, lon_offset, t0s_offset, R0):

    LAT_offset, LON_offset = np.meshgrid(lat_offset, lon_offset)
    LAT_offset_shape_init = LAT_offset.shape
    LAT_offset, LON_offset = LAT_offset.ravel(), LON_offset.ravel()
    lat0 = lat_vol+LAT_offset  # Initial latitude in degrees
    lon0 = lon_vol+LON_offset  # Initial longitude in degrees

    times = np.arange(0, mission_duration, dt) 
    TIMES, ID_LAT0 = np.meshgrid(times, np.arange(LAT_offset.size))
    shape_TIMES = TIMES.shape # balloon init loc/t0 x balloon flight time 
    TIMES, ID_LAT0 = TIMES.ravel(), ID_LAT0.ravel() 

    idx_dt_step = int(dt_new/dt)
    times_downsampled = times[::idx_dt_step]
    latitudes, longitudes = pvm.compute_positions_vectorized_w_interpolator(lat0[ID_LAT0].reshape(shape_TIMES), lon0[ID_LAT0].reshape(shape_TIMES), wind_direction_interpolator, wind_strength_interpolator, TIMES.reshape(shape_TIMES), R0)
    latitudes, longitudes = latitudes.reshape(shape_TIMES)[:,::idx_dt_step], longitudes.reshape(shape_TIMES)[:,::idx_dt_step]

    distances = pvm.haversine_distance(latitudes, longitudes, lat_vol, lon_vol, R0)

    LAT_offset, LON_offset, T0s_offset = np.meshgrid(lat_offset, lon_offset, t0s_offset)
    LAT_offset_shape = LAT_offset.shape
    LAT_offset, LON_offset, T0s_offset = LAT_offset.ravel(), LON_offset.ravel(), T0s_offset.ravel()
    lat0 = lat_vol+LAT_offset  # Initial latitude in degrees
    lon0 = lon_vol+LON_offset  # Initial longitude in degrees

    TIMES, ID_LAT0 = np.meshgrid(times_downsampled, np.arange(LAT_offset.size))
    shape_TIMES = TIMES.shape # balloon init loc/t0 x balloon flight time 
    TIMES, ID_LAT0 = TIMES.ravel(), ID_LAT0.ravel() 

    dt_new = mission_duration/100.
    idx_dt_step = int(dt_new/dt)
    times_downsampled = times[::idx_dt_step]
    latitudes, longitudes = pvm.compute_positions_vectorized_w_interpolator(lat0[ID_LAT0].reshape(shape_TIMES), lon0[ID_LAT0].reshape(shape_TIMES), wind_direction_interpolator, wind_strength_interpolator, TIMES.reshape(shape_TIMES), R0)
    latitudes, longitudes = latitudes.reshape(shape_TIMES)[:,::idx_dt_step], longitudes.reshape(shape_TIMES)[:,::idx_dt_step]

    latitudes, longitudes = None, None

    distances_repeated = np.zeros(shape_TIMES)
    for itime in tqdm(range(shape_TIMES[-1])):
        distances_repeated[:,itime] = np.repeat(distances[:,0].reshape(LAT_offset_shape_init)[:,:,None], t0s_offset.size, axis=-1).ravel()
    
    distances = None

    distances_repeated = distances_repeated.ravel()

    return T0s_offset, LAT_offset, ID_LAT0, TIMES, times_downsampled, shape_TIMES,distances_repeated, LAT_offset_shape

##########################
if __name__ == '__main__':


    dir_GF = '/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/'
    dir_data = '/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/'
    #model_names = ['Cold100', 'Hot10']
    model_names = ['Hot40', 'Hot25']
    alt_balloons = [50., 60.]
    lat_vol, lon_vol = 8.687137, -52.476376
    mission_duration = (6./12.)*365*24*3600
    dt = 1*3600/6.
    dt_new = mission_duration/100.
    R0 = 6371000  # Earth's radius in meters
    batch_size = 100
    freqs = [0.01, 0.1, 1.]
    noise_level=1e-2
    snr_threshold=1
    snrs = np.logspace(np.log10(0.1),np.log10(10.), 50)
    arrival_time = lambda dist, h_balloon, t0: t0 + (h_balloon/0.35 + dist/3.5)
    lat_offset, lon_offset = np.linspace(-50., 50., 80), np.linspace(-50., 50., 40)
    #lat_offset, lon_offset = np.linspace(-50., 50., 8), np.linspace(-50., 50., 4) # Debug
    dt_between_events = 6/12

    ###################
    ## Event catalog ##
    catalog_hawai = pd.read_csv(f'{dir_data}hawai_catalog_since_1983.csv', header=[0])
    catalog_hawai.loc[:,'UTC'] = pd.to_datetime(catalog_hawai.UTC)
    catalog_hawai = catalog_hawai.loc[catalog_hawai.mag>=3.]

    all_mags = catalog_hawai.mag.values
    all_times = catalog_hawai['time (years)'].values
    t0s_offset = np.arange(all_times.min()-2./12, all_times.max()-2./12, dt_between_events) # in years

    #######################
    ## Atmospheric model ##
    file_atmos = f'{dir_data}profile_VCD_for_scaling_pd.csv'
    profile = pd.read_csv(file_atmos)
    f_rho = interpolate.interp1d(profile.altitude/1e3, profile.rho, kind='quadratic')
    f_t = interpolate.interp1d(profile.altitude/1e3, profile.t, kind='quadratic')
    f_p = interpolate.interp1d(profile.altitude/1e3, profile.p, kind='quadratic')
    f_gamma = interpolate.interp1d(profile.altitude/1e3, profile.gamma, kind='quadratic')
    f_c = interpolate.interp1d(profile.altitude/1e3, profile.c, kind='quadratic')

    ## Loop over input parameters
    for alt_balloon in alt_balloons:

        file_atmos = f'{dir_data}VCD_atmos_globe_new.dat'
        winds = VCD.get_winds(file_atmos, alt_balloon*1e3)
        wind_direction_interpolator, wind_strength_interpolator, _ = VCD.get_winds_interpolator(file_atmos, alt_balloon*1e3, winds=winds)

        for model_name in model_names:


            ################
            ## Amplitudes ##
            opt_TL = dict(
                rho0=f_rho(0.), 
                rhob=f_rho(alt_balloon), 
                c0=f_c(0), 
                cb=f_c(alt_balloon), 
                unknown='pressure', 
                model=model_name
            )

            file_curve = f'{dir_GF}GF_Dirac_1Hz_combined_wHot40_lognormal_updated_minthick.csv'
            shape_new, scale_new = pm.get_lognormal_precomputed(file_curve, **opt_TL)

            for freq in freqs:
            
                file_fig = f'./figures/Figure_5_volcanoes_small_1983_{1./freq:.0f}s_{alt_balloon:.1f}km_{model_name}.pdf'

                ##################
                ## Trajectories ##
                T0s_offset, LAT_offset, ID_LAT0, TIMES, times_downsampled, shape_TIMES, distances_repeated, LAT_offset_shape = get_trajectories(mission_duration, dt, dt_new, lat_vol, lon_vol, lat_offset, lon_offset, t0s_offset, R0)
                
                ###########################
                ## Event characteristics ##
                file_amps = f'./data/volcanoes/amps_ev_{1./freq:.0f}s_1983_10.11.2025.npy'
                file_mask = f'./data/volcanoes/mask_{1./freq:.0f}s_1983_10.11.2025.npy'
                import os
                #bp()
                if not os.path.exists(file_amps):
                    mags_ev, amps_ev, mask = pvm.get_amps_at_baloons(T0s_offset, LAT_offset, ID_LAT0, TIMES, times_downsampled, shape_TIMES, all_times, all_mags, distances_repeated, scale_new[freq], arrival_time, batch_size)
                
                    with open(file_amps, 'wb') as f:
                        np.save(f, amps_ev)

                    with open(file_mask, 'wb') as f:
                        np.save(f, mask)
                else:
                    amps_ev = np.load(file_amps)
                    mask = np.load(file_mask)

                ########################
                ## Derived quantities ##
                #snr_threshold=1
                #number_over_snr = ((amps_ev*(mask)/noise_level)>snr_threshold).sum(axis=0).reshape(LAT_offset_shape)
                idamp = (amps_ev*(mask)).argmax(axis=0)[None,:]  
                amps_ev_reshaped = np.take_along_axis(amps_ev*(mask) + 1e-10*(~mask), idamp, axis=0)[0].reshape(LAT_offset_shape) # lon x lat x t0

                number_over_snr = np.zeros((snrs.size,) + LAT_offset_shape)
                for isnr,  snr in tqdm(enumerate(snrs), total=snrs.size):
                    number_over_snr[isnr,:] = ((amps_ev*(mask)/noise_level)>snr).sum(axis=0).reshape(LAT_offset_shape)

                ##############
                ## Plotting ##
                opt_figure = dict(
                    noise_level=noise_level, 
                    fontsize=12., 
                    number_over_snr=number_over_snr, 
                    idamp=idamp, 
                    amps_ev_reshaped=amps_ev_reshaped, 
                    color_labels='black', 
                    fontsize_label=20.,
                )
                fig = pvm.plot_proba_sequence_small(catalog_hawai, amps_ev, t0s_offset, LAT_offset_shape, mask, snrs, **opt_figure)

                
                fig.savefig(file_fig)