from pyrocko import gf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyrocko import moment_tensor as pmt
from scipy import signal
from tqdm import tqdm 

def filter_wave(waveform, f1, f2, dt):

    #b, a = signal.butter(N=10, Wn=[f1, f2], btype='bandpass', analog=False, fs=1./dt, output='ba')
    #y_tf = signal.lfilter(b, a, dirac)
    sos = signal.butter(N=10, Wn=[f1, f2], btype='bandpass', analog=False, fs=1./dt, output='sos')
    return signal.sosfilt(sos, waveform)

def show_bounds(store):

    # Get the spatial bounds for receivers
    distance_min = store.config.distance_min
    distance_max = store.config.distance_max

    # Get the spatial bounds for sources
    depth_min = store.config.source_depth_min
    depth_max = store.config.source_depth_max

    # Print the bounds
    print(f"Receiver distance range: {distance_min} km to {distance_max} km")
    print(f"Source depth range: {depth_min} km to {depth_max} km")

def prepare_mts(strikes):
    
    scalar_moment = 10e6
    strike, dip, rake = 0., 90., 180.
    mt_strike = pmt.MomentTensor(strike=strike, dip=dip, rake=rake, scalar_moment=scalar_moment).m6()
    strike, dip, rake = 0., 90., 270.
    mt_normal = pmt.MomentTensor(strike=strike, dip=dip, rake=rake, scalar_moment=scalar_moment).m6()
    
    types = []
    mts = []
    for strike in strikes:
        types += ['normal', 'strike_slip', ]
        mts.append( dict(mnn=mt_normal[0]+strike, mee=mt_normal[1], mdd=mt_normal[2], mne=mt_normal[3], mnd=mt_normal[4], med=mt_normal[5]) )
        mts.append( dict(mnn=mt_strike[0]+strike, mee=mt_strike[1], mdd=mt_strike[2], mne=mt_strike[3], mnd=mt_strike[4], med=mt_strike[5],) )
        
    return types, mts
    
def prepare_waveforms(dist, azimuths, ref_location):
    
    azimuths_rad = np.radians(azimuths)
    id_dist, id_azimuths = np.arange(dist.size), np.arange(len(azimuths))
    id_dist, id_azimuths = np.meshgrid(id_dist, id_azimuths)
    id_dist, id_azimuths = id_dist.ravel(), id_azimuths.ravel()
    north_shifts = dist[id_dist] * np.cos(azimuths_rad[id_azimuths])
    east_shifts = dist[id_dist] * np.sin(azimuths_rad[id_azimuths])
    
    waveform_targets = [
        gf.Target(
            quantity='velocity',
            lat = ref_location[0],
            lon = ref_location[1],
            north_shift=north_shift,
            east_shift=east_shift,
            store_id=store_id,
            interpolation='multilinear',
            codes=('NET', 'STA', 'LOC', 'Z'))
        for north_shift, east_shift in tqdm(zip(north_shifts, east_shifts), total=east_shifts.size)
        ]
    
    return waveform_targets, dist[id_dist], azimuths_rad[id_azimuths]
    
def build_amps_and_traces(dists, depths, base_folder, store_id, f_targets, stf=None):

    engine = gf.LocalEngine(store_dirs=[f'{base_folder}{store_id}/'])
    store = engine.get_store(store_id)
    show_bounds(store)
    
    ref_location = [0., 0.]
    
    delta_az = 25.
    azimuths = np.arange(0., 360., delta_az)
    #azimuths = [0.]
    waveform_targets, dists_waveform, az_waveform = prepare_waveforms(dists, azimuths, ref_location)
    
    strikes = [0.]
    types, mts = prepare_mts(strikes)
    
    id_mts, id_depths = np.meshgrid(np.arange(len(mts)), np.arange(len(depths)))
    id_mts, id_depths = np.meshgrid(id_mts, id_depths)
    id_mts, id_depths = id_mts.ravel(), id_depths.ravel()
    
    all_amps_RW = pd.DataFrame()
    all_amps_S = pd.DataFrame()
    #factor = 1.
    for id_mt, id_depth in tqdm(zip(id_mts, id_depths), total=id_depths.size):
        
        mt = mts[id_mt]
        depth = depths[id_depth]
        type_mt = types[id_mt]

        stf_add = {}
        if stf is not None:
            stf_add['stf'] = stf
        #print(stf_add)
        mt_source = gf.MTSource(lat=ref_location[0], lon=ref_location[1], depth=depth, **mt, **stf_add)

        # The computation is performed by calling process on the engine
        try:
            response = engine.process(mt_source, waveform_targets)
        except:
            dict_RW = pd.DataFrame(np.c_[-np.ones_like(dists_waveform), -np.ones_like(dists_waveform), dists_waveform, az_waveform], columns=['amp_RW', 'amp_S', 'dist', 'az'])
            dict_RW['type_mt'] = type_mt
            dict_RW['depth'] = depth
            all_amps_RW = pd.concat([all_amps_RW, dict_RW])

            dict_RW = pd.DataFrame(np.c_[-np.ones_like(dists_waveform), -np.ones_like(dists_waveform), dists_waveform, az_waveform], columns=['amp_RW', 'amp_S', 'dist', 'az'])
            dict_RW['type_mt'] = type_mt
            dict_RW['depth'] = depth
            all_amps_RW = pd.concat([all_amps_RW, dict_RW])
            
        # convert results in response to Pyrocko traces
        synthetic_traces = response.pyrocko_traces()

        f_targets_mod = f_targets + [['dirac']]
        for f_targets_loc in f_targets_mod:

            amps_RW = []
            amps_S = []
            for t, waveform in zip(waveform_targets, synthetic_traces):
                dist = t.distance_to(mt_source)
                depth = mt_source.depth
                
                compute_S = False
                if compute_S:
                    t_s = store.t('s', (depth, dist))
                    t_S = store.t('S', (depth, dist))
                    arrival_time = t_s
                    if (t_s is None):
                        arrival_time = t_S
                    elif t_S is not None:
                        arrival_time = min(t_S, t_s)
                    if (t_S is None):
                        arrival_time = t_s

                    if arrival_time is None:
                        v_RW = (1./0.95)*dist/waveform.get_xdata()[waveform.get_ydata().argmax()]
                        t_RW = dist/v_RW
                        arrival_time = t_RW-20.
                    else:
                        arrival_time += 10.

                dt = waveform.get_xdata()[1]-waveform.get_xdata()[0]
                waveform_processed = waveform.get_ydata()
                if len(f_targets_loc) > 1:
                    waveform_processed = filter_wave(waveform_processed, f_targets_loc[0], f_targets_loc[1], dt)

                max_RW = abs(waveform_processed).max()
                amps_RW.append(max_RW)
                if compute_S:
                    iS = np.argmin(abs(waveform.get_xdata()-(arrival_time)))
                    max_S = abs(waveform_processed)[:iS].max()
                    amps_S.append(max_S)

            if compute_S:
                dict_RW = pd.DataFrame(np.c_[amps_RW, amps_S, dists_waveform, az_waveform], columns=['amp_RW', 'amp_S', 'dist', 'az'])
            else:
                dict_RW = pd.DataFrame(np.c_[amps_RW, dists_waveform, az_waveform], columns=['amp_RW', 'dist', 'az'])
            dict_RW['type_mt'] = type_mt
            dict_RW['depth'] = depth
            fmin, fmax = 0., 1.
            if len(f_targets_loc) > 1:
                fmin = f_targets_loc[0] if f_targets_loc[0] is not None else 0.
                fmax = f_targets_loc[1] if f_targets_loc[1] is not None else 1.
                #print(fmin, fmax, f_targets_loc)
            dict_RW['fmin'] = fmin
            dict_RW['fmax'] = fmax
            all_amps_RW = pd.concat([all_amps_RW, dict_RW])
    
    return all_amps_RW, all_amps_S

def get_all_amps(base_folder, stores_id, dists, depths, f_targets, stf):

    all_amps_RW, all_amps_S = pd.DataFrame(), pd.DataFrame()
    for istore, (store_id, dist) in enumerate(zip(stores_id, dists)):
        all_amps_RW_loc, all_amps_S_loc = build_amps_and_traces(dist, depths, base_folder, store_id, f_targets, stf=stf)
        all_amps_RW_loc['store'] = store_id
        #all_amps_S_loc['store'] = store_id
        
        all_amps_RW = pd.concat([all_amps_RW, all_amps_RW_loc])
        #all_amps_S = pd.concat([all_amps_S, all_amps_S_loc]

    return all_amps_RW

##########################
if __name__ == '__main__':

    ## Discretization
    delta_dist = 50e3
    epsilon = 5e3
    dists = []
    dists.append(np.arange(0., 50.e3, 5e3)) # in km
    #dists.append(np.arange(50.e3+epsilon, 8000.e3+epsilon, delta_dist)) # in km
    #dists.append(np.arange(8000.e3+epsilon, 16000.e3+epsilon, delta_dist)) # in km
    #dists.append(np.arange(0., 50.e3, 5e3)) # in km
    #dists.append(np.arange(50.e3+epsilon, 8000.e3+epsilon, delta_dist)) # in km
    #dists.append(np.arange(8000.e3+epsilon, 16000.e3+epsilon, delta_dist)) # in km
    delta_depth = 5e3
    depths = np.arange(5e3, 50e3+delta_depth, delta_depth)

    ## STF
    #period = 1./1e-1
    #stf = gf.BoxcarSTF(period, anchor=0.)
    #stf = gf.BoxcarSTF(period, anchor=0.)
    #stf = gf.TriangularSTF(effective_duration=period)
    stf = None

    f_bins = np.logspace(np.log10(1e-2), np.log10(0.5), 4)
    f_targets = []
    for binleft, binright in zip(f_bins[:-1], f_bins[1:]):
        f_targets += [[binleft, binright]]

    ## Greens functions STORES
    base_folder = '/projects/infrasound/data/infrasound/2023_Venus_inversion/'
    stores_id = []

    stores_id.append('GF_venus_qssp_nearfield')
    #stores_id.append('GF_venus_qssp')
    #stores_id.append('GF_venus_qssp_8000km')

    #stores_id.append('GF_venus_qssp_nearfield_c50km')
    #stores_id.append('GF_venus_qssp_c50km')
    #stores_id.append('GF_venus_qssp_8000km_c50km')

    all_amps_RW = get_all_amps(base_folder, stores_id, dists, depths, f_targets, stf)
    all_amps_RW.to_csv('./GF_Dirac_1Hz_all_wfreq.csv', header=True, index=False)