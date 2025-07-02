import numpy as np
import pandas as pd
from scipy import interpolate
from pdb import set_trace as bp

import sys
sys.path.append('/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability')
import proba_modules as pm

##########################
if __name__ == '__main__':

    ## Atmosphere
    file_atmos = '/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/profile_VCD_for_scaling_pd.csv'
    profile = pd.read_csv(file_atmos)
    f_rho = interpolate.interp1d(profile.altitude/1e3, profile.rho, kind='quadratic')
    f_t = interpolate.interp1d(profile.altitude/1e3, profile.t, kind='quadratic')
    f_gamma = interpolate.interp1d(profile.altitude/1e3, profile.gamma, kind='quadratic')
    f_c = interpolate.interp1d(profile.altitude/1e3, profile.c, kind='quadratic')

    ## TL
    dir_GF = '/projects/restricted/infrasound/data/infrasound/2023_Venus_inversion/'
    file_curve = f'{dir_GF}GF_Dirac_1Hz_combined_wHot40.csv'
    alt_balloon = 50.
    opt_TL = dict(
        rho0=f_rho(0.), 
        rhob=f_rho(alt_balloon), 
        cb=f_c(alt_balloon), 
        unknown='pressure', 
        model='Cold100'
    )
    TL_new, TL_new_qmin, TL_new_qmax = pm.get_TL_curves_precomputed(file_curve, **opt_TL)

    ## Moment rate
    file_slopes = '/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/Venus_data/distribution_venus_per_mw.csv'
    pd_slopes = pm.get_slopes(file_slopes)

    ## Surface ratios
    file_ratio = '/staff/quentin/Documents/Projects/2024_Venus_Detectability/Venus_Detectability/data/surface_ratios/surface_ratios_active_fixed.csv'
    surface_ratios = pm.get_surface_ratios(file_ratio)

    ## Probas
    dlat = 5.
    r_venus = 6052
    opt_model = dict(
        scenario = 'active_low_min', # Iris' seismicity scenario
        dists = np.arange(10., np.pi*r_venus, 200), # Low discretization will lead to terrible not unit integrals
        M0s = np.linspace(3., 8., 30), # Low discretization will lead to terrible not unit integrals
        SNR_thresholds = np.linspace(0.1, 10., 50),
        noise_level = 1e-2, # noise level in Pa
        duration = 1./(365.*24.), # (1/mission_duration)
        all_lats = np.arange(-89, 90, dlat),
        all_lons = np.arange(0, 359, dlat*2),
        #all_lats = np.arange(-90., 90.+dlat, dlat),
        #all_lons = np.arange(-180, 180+dlat*2, dlat*2),
        homogeneous_ratios = False,
        m_min = 3.,
        r_venus = r_venus,
        #verbose=False
        nb_CPU=8
    )

    freq = 1. # Or None
    apply_fc_correction = None
    #proba_model = pm.proba_model(pd_slopes, surface_ratios, TL_new[freq], TL_new_qmin[freq], TL_new_qmax[freq], apply_fc_correction=apply_fc_correction)
    #proba_model.compute_proba_map(**opt_model)
    proba_model = pm.proba_model_CPUs(pd_slopes, surface_ratios, TL_new[freq], TL_new_qmin[freq], TL_new_qmax[freq], apply_fc_correction=apply_fc_correction)
    proba_model.compute_scores_across_CPUs(**opt_model)

    bp()