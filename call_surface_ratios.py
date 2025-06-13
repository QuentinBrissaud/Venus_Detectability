import numpy as np
import os
from pdb import set_trace as bp
import pyproj
import geopandas as gpd

import sys
sys.path.append('./Venus_Detectability/')
import surface_ratios_module as srm
import compute_network_shapes_CPUs as cns

def compute_map(lat_0=-89., lon_0=0., R0=6052000):
    proj = pyproj.Proj(proj='robin', lat_0=lat_0, lon_0=lon_0, a=R0, b=R0)
    polygon_map, _ = srm.compute_whole_map_region(lon_0, proj, dlon=1., dlat=1)
    return polygon_map

##########################
if __name__ == '__main__':

    

    """
    dlon_stations = 5.
    lats, lons = np.arange(-90, 89, dlon_stations/2.), np.arange(-180, 179, dlon_stations)
    LATS, LONS = np.meshgrid(lats, lons)
    LATS, LONS = LATS.ravel(), LONS.ravel()
    l_points = np.c_[LONS, LATS]
    """
    use_airglow_max_dist = True
    airglow_and_balloon = False
    offsets = [[20., 50.]]
    dir_outputs = './data/surface_ratios/'

    """
    folder_shp = 'network_shp'
    network = 'network_balloon_20.0_50.0_any_event'
    ext = 'shp'
    """
    """
    folder_shp = 'network_shp'
    network = 'network_balloon_20.0_50.0_same_event'
    ext = 'gpkg'
    """
    """
    folder_shp = 'airglow_shp'
    network = 'nightglow_dayglow_SNRnight10.0_SNRday1_same_event'
    ext = 'shp'
    """
    """
    folder_shp = 'airglow_shp'
    network = 'nightglow_dayglow_one_balloon_SNRnight10.0_SNRday1__same_event'
    ext = 'gpkg'
    """
    
    folder_shp = 'airglow_shp'
    network = 'nightglow_dayglow_SNRnight1.0_SNRday1_same_event_mCold100'
    ext = 'shp'
    """
    folder_shp = 'airglow_shp'
    network = 'nightglow_SNRnight1.0_SNRday1_any_event_mCold100'
    ext = 'shp'
    """
    file_shp = f'./data/{folder_shp}/{network}.{ext}'

    dlon_stations = 5.
    lats, lons = np.arange(-90, 89, dlon_stations/2.), np.arange(-180, 179, dlon_stations)
    if use_airglow_max_dist:
        offsets = []
        dict_stations = dict(offsets=offsets, use_airglow=True, use_only_airglow=not airglow_and_balloon, fixed_stations=dict(), add_velocity=True, vel_baloon=0.2, vel_imager=0.36)
    else:
        dict_stations = dict(offsets=offsets, use_airglow=False, use_only_airglow=False, fixed_stations=dict(), add_velocity=True, vel_baloon=0.2, vel_imager=0.36)
    lats_stations, lons_stations, id_scenario, id_stat = cns.get_stations(lats, lons, **dict_stations)

    LATS, LONS, _, _ = cns.get_stations(lats, lons, **dict_stations)
    l_points = np.c_[LONS[:,0], LATS[:,0]]

    #l_points = l_points
    R0 = 6052000
    proj = pyproj.Proj(proj='robin', lat_0=-89., lon_0=0., a=R0, b=R0)
    l_radius = np.arange(10000, np.pi*R0/1.001, 5e5)[:]
    lon_0, lat_0 = 0., -89.

    opt_maps = dict(
        #folder_TL_data='./data/TL_data/', 
        lat_0=lat_0, 
        lon_0=lon_0, 
        R0=R0
    )
    #polygon_map, surface1_lon, surface1_lat, n_subshapes = srm.compute_map_and_TL(**opt_maps)
    surface1_lon, surface1_lat, n_subshapes = None, None, None
    polygon_map = srm.compute_map(**opt_maps)

    PATH_VENUS_DATA = os.path.join("./Venus_Detectability/data/")
    #find_active_corona_only=False
    #l_unioned_linestring = load_wrinkles_shp(PATH_VENUS_DATA, epsilon=0.5)
    find_active_corona_only=True
    l_unioned_linestring = srm.load_tectonic_iris_shp(PATH_VENUS_DATA, polygon_map, epsilon=1e-1, lat_0=-89., lon_0=0., R0=6052000, find_active_corona_only=find_active_corona_only)

    #gdf = gpd.read_file(f"./data/airglow_shp/nightglow_dayglow_SNRnight10.0_SNRday1.shp")
    #gdf = gpd.read_file(f"./data/network_shp/network_balloon_20.0_50.0.shp")
    gdf = gpd.read_file(file_shp)
    #gdf = gpd.read_file(f"./data/airglow_shp/nightglow_dayglow_one_balloon_SNRnight10.0_SNRday1_.shp")
    #gdf = gpd.read_file(f"./data/network_shp/one_balloon_.shp")
    
    for region, unioned_linestring in l_unioned_linestring.items():
        opt_surface = dict(
            lon_0=lon_0,
            l_radius=l_radius, 
            proj=proj, 
            polygon_map=polygon_map, 
            polygon_region=unioned_linestring, 
            subsample_db=5, 
            buffer_line=120000, 
            threshold_neighbor_pts=20e6, 
            random_state=0,
            n_subshapes=n_subshapes, 
            l_points=l_points, 
            surface1_lon=surface1_lon, 
            surface1_lat=surface1_lat,
            gdf=gdf,
            filter_gdf_before_CPU=True,
            nb_CPU=10,
        )
        
        ratio_df = srm.compute_surface_ratios_wrinkles_across_CPU(**opt_surface)
        region_str = region
        if find_active_corona_only:
            region_str = f'{region_str}_active'

        #plt.figure(); plt.plot(ratio_df.radius.iloc[:30], ratio_df.ratio_map.iloc[:30]); plt.savefig('./test2.png')
        ratio_df['region'] = region_str

        #ratio_df.to_csv(f'./data/surface_ratios/surface_ratios_airglow_{region}.csv', index=False, header=True)
        #ratio_df.to_csv(f'./data/surface_ratios/surface_ratios_one_balloon_{region}.csv', index=False, header=True)
        #ratio_df.to_csv(f'./data/surface_ratios/surface_ratios_network_balloon_any_event_{region}.csv', index=False, header=True)
        #ratio_df.to_csv(f'./data/surface_ratios/surface_ratios_airglow_one_balloon_{region_str}.csv', index=False, header=True)
        ratio_df.to_csv(f'{dir_outputs}surface_ratios_{network}_{region_str}.csv', index=False, header=True)

    bp()
