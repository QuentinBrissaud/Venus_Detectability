import pyproj
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from sklearn.cluster import DBSCAN
from shapely.validation import make_valid
from tqdm import tqdm
import pandas as pd
from pyproj import Geod
import numpy as np
import geopandas as gpd 
import os
import seaborn as sns
from matplotlib.patches import Polygon as Polygon_mpl
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
import matplotlib.colors as mcol
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

def spherical_cap_boundary(lat_0, lon_0, radius, R0, num_points=100):
    """
    Compute a list of latitude, longitudes corresponding to the boundary of a spherical cap.

    Parameters:
    lat_0, lon_0: Center of the spherical cap in degrees.
    radius: Radius of the spherical cap in meters.
    R0: Radius of the sphere in meters.
    num_points: Number of points to generate along the boundary.

    Returns:
    A list of (lat, lon) tuples representing the boundary of the spherical cap.
    """
    # Convert the center coordinates to radians
    lat_0_rad = np.radians(lat_0)
    lon_0_rad = np.radians(lon_0)

    # Compute the angular radius of the cap
    angular_radius = radius / R0

    # Generate a list of angles
    angles = np.linspace(0, 2*np.pi, num_points)

    # Compute the latitude and longitude of each point on the boundary
    lat_rad = np.arcsin(np.sin(lat_0_rad)*np.cos(angular_radius) + np.cos(lat_0_rad)*np.sin(angular_radius)*np.cos(angles))
    lon_rad = lon_0_rad + np.arctan2(np.sin(angles)*np.sin(angular_radius)*np.cos(lat_0_rad), np.cos(angular_radius) - np.sin(lat_0_rad)*np.sin(lat_rad))

    # Convert the coordinates back to degrees
    lat = np.degrees(lat_rad)
    lon = np.degrees(lon_rad)
    #lon[lon < 0] += 360.

    # Combine the latitude and longitude into a list of tuples
    boundary = list(zip(lon, lat))

    return boundary

def spherical_cap_boundary_v2(g, lat_0, lon_0, radius, num_points=100):

    n_radius = len(l_radius)
    lat, lon = np.repeat(lat_0, n_radius*num_points), np.repeat(lon_0, n_radius*num_points)
    angles = np.linspace(1., 359., n_radius*num_points)
    R = np.repeat(l_radius, num_points)
    endlon, endlat, _ = g.fwd(lon, lat, angles, R)

    return list(zip(endlon, endlat))

def spherical_cap_boundary_v3(g, l_lat_0, l_lon_0, l_radius, proj, num_points=100):

    n_loc = l_lat_0.size
    n_radius = len(l_radius)
    lat, lon = np.repeat(l_lat_0, n_radius*num_points), np.repeat(l_lon_0, n_radius*num_points)
    angles = np.tile(np.linspace(1., 359., num_points), n_radius*n_loc)
    R = np.repeat(l_radius, num_points*n_loc)
    endlon, endlat, _ = g.fwd(lon, lat, angles, R)
    endlon, endlat = proj(endlon, endlat)

    return np.c_[endlon, endlat]
        
def compute_region_venus(polygon_map, ext_points, proj, show_bar=True):

    # ext_points: e.g., VENUS['rift'].exterior
    polygon2 = Polygon()
    for ext in tqdm(ext_points, disable=show_bar):
        if ext is None:
            continue
        surface2 = list(ext.coords)
        coords = np.array([proj(lon, lat) for lon, lat in surface2])
        clustering = DBSCAN(eps=500000, min_samples=5).fit(coords)
        lines = []
        #print(f'labels: {np.unique(clustering.labels_)}')
        for label in np.unique(clustering.labels_):
            #print(f'- size: {coords[clustering.labels_==label].shape[0]}')
            if coords[clustering.labels_==label].shape[0] > 1:
                lines.append( LineString(coords[clustering.labels_==label]).buffer(200000) )
        lines = MultiPolygon(lines)
        
        diff_map = polygon_map.difference(make_valid(lines))
        if diff_map.geom_type == 'MultiPolygon':
            ipolymax = -1
            max_val = -1
            for ipoly, sub_poly in enumerate(diff_map.geoms):
                if sub_poly.area > max_val:
                    ipolymax = ipoly
                    max_val = sub_poly.area

            for ipoly, sub_poly in enumerate(diff_map.geoms):
                if not ipoly == ipolymax:
                    continue 
                diff_map = polygon_map.difference(sub_poly)

            diff_map = make_valid(diff_map)
            if diff_map.geom_type == 'MultiPolygon': 
                for ipoly, sub_poly in enumerate(diff_map.geoms):
                    #if ipoly == ipolymax:
                    #    continue 
                    if sub_poly.geom_type == 'Polygon':
                        polygon2 = polygon2.union( sub_poly.buffer(10) )
                    #exts.append(iext)
            else:
                polygon2 = polygon2.union( diff_map )
                
    polygon2 = make_valid(polygon2)
    return polygon2

def compute_whole_map_region(lon_0, proj, dlon=10., dlat=10.): 

    left_lon  = -180.01+lon_0 if lon_0 > 0 else 180.01+lon_0
    right_lon = -179.01+lon_0 if lon_0 > 0 else 179.01+lon_0
    bottom_lat = -89.9#+lat_0 if lat_0 > 0 else 90+lat_0
    top_lat = 89.9#+lat_0 if lat_0 < 0 else -90+lat_0
    
    coords_map = [proj(lon, bottom_lat) for lon in np.arange(left_lon, right_lon, dlon)]
    coords_map += [proj(right_lon, lat) for lat in np.arange(bottom_lat, top_lat, dlat)]
    coords_map += [proj(lon, top_lat) for lon in np.arange(right_lon, left_lon, -dlon)]
    coords_map += [proj(left_lon, lat) for lat in np.arange(top_lat, bottom_lat, -dlat)]
    polygon_map = Polygon(coords_map)
    return polygon_map, coords_map

def compute_TL_region(polygon_map, coords_poly1, proj, lon_0, lon_0_current, sensor, db, subsample_db=1):
    
    proj_sensor = proj(lon_0_current, 0.)
    proj_sensor_ref = proj(lon_0, 0.)
    offset = abs(proj_sensor_ref[0]-proj_sensor[0])

    #t = time()

    #coords_poly1 = np.array([proj(lon, lat) for lon, lat in surface1])[:-1]

    #print(f'#1-0 Elapsed {time()-t} s')
    #t = time()
    
    #coords_poly1 = np.roll(coords_poly1, np.argmax(abs(coords_poly1[:,0]-proj_lon_0))-1, axis=0)
    coords_poly1 = np.roll(coords_poly1, -np.argmax(abs(coords_poly1[:,0]-offset))-1, axis=0)

    #print(f'#1-1 Elapsed {time()-t} s')
    #t = time()

    clustering = db.fit(coords_poly1)

    #print(f'#1-2 Elapsed {time()-t} s')
    #t = time()

    #lines = []
    polygon1 = Polygon()
    for label in np.unique(clustering.labels_):
        polygon1 = polygon1.union( LineString(coords_poly1[clustering.labels_==label][:-1]).buffer(200000) )
    #polygon1 = MultiPolygon(lines)
    diff_poly = polygon_map.difference(polygon1.buffer(10)) # The buffer here is to avoid errors like TopologyException: Input geom 1 is invalid: Self-intersection
    
    #print(f'#1-3 Elapsed {time()-t} s')
    #t = time()
    if diff_poly.geom_type == 'MultiPolygon':
        polys = []
        for poly in diff_poly.geoms:
            polys.append(poly)
            if poly.contains(sensor):
                polygon1 = poly
                break
    else:
        if not polygon1.contains(sensor):
            polygon1 = diff_poly
            
    #print(f'#1-4 Elapsed {time()-t} s')

    return polygon1

def compute_surface_area_ratio(intersection, polygon_map):
    return intersection.area/polygon_map.area

def compute_region_intraplate(VENUS, polygon_map, proj):

    if 'intraplate' in VENUS:
        return VENUS['intraplate']

    polygon2 = Polygon()
    for region in VENUS:
        polygon2 = polygon2.union( compute_region_venus(polygon_map, VENUS[region].exterior, proj, show_bar=False) )

    return polygon_map.difference(polygon2)

def plot_one_TL(polygon_map, polygon1, polygon2, intersection, lon_0_current, lat_0_current, proj):

    plt.figure()

    plt.plot(*polygon_map.exterior.xy, label='TL', color='tab:blue')
    plt.plot(*polygon1.exterior.xy, label='map', color='tab:red', linestyle=':')
    for poly in polygon2.geoms:
        if poly.geom_type == 'LineString':
            plt.plot(*poly.xy, label='venus', color='orange')
        else:
            plt.plot(*poly.exterior.xy, label='venus', color='orange')
    if intersection.geom_type == 'MultiPolygon':
        for poly in intersection.geoms:
            if poly.geom_type == 'LineString':
                plt.scatter(*poly.xy, label='inter', color='black', alpha=0.5, s=50, )
            else:
                plt.scatter(*poly.exterior.xy, label='inter', color='black', alpha=0.5, s=10, )
    x, y = proj(lon_0_current, lat_0_current)
    plt.scatter(x, y, marker='^', edgecolor='black', color='red', s=200, zorder=1000)
    plt.title(f'Surface area ratio: {intersection.area/polygon_map.area:.2e}')

def compute_ratios(VENUS, l_lon, l_lat, output_file, R0=6052000, num_points=100, ratio_df=pd.DataFrame()):

    g = pyproj.Geod(proj='robin', lat_0=0., lon_0=0., a=R0, b=R0)    
    LONS, LATS = np.meshgrid(l_lon, l_lat)
    LONS, LATS = LONS.ravel(), LATS.ravel()
    l_points = list(zip(LONS, LATS))
    l_radius = np.arange(10000, np.pi*R0/1.01, 100000)
    proj = pyproj.Proj(proj='robin', lat_0=l_points[0][1], lon_0=l_points[0][0], a=R0, b=R0)
    surface1 = spherical_cap_boundary_v3(g, LATS, LONS, l_radius, proj, num_points=num_points)
    polygon_map, _ = compute_whole_map_region(l_points[0][0], proj, dlon=1., dlat=1)
    VENUS['intraplate'] = compute_region_intraplate(VENUS, polygon_map, proj)
    db = DBSCAN(eps=200000, min_samples=20)
    
    polygon1s = []
    for iregion, region in enumerate(VENUS):
        first_loop = True
        iloc = -1
        print(f'- {region} -')
        itotal = -1
        for lon_0_current, lat_0_current in tqdm(l_points):

            iloc += 1
            surface1_loc = surface1[iloc*num_points*len(l_radius):(iloc+1)*num_points*len(l_radius), :]

            if first_loop:
                lon_0 = lon_0_current
                polygon_map, _ = compute_whole_map_region(lon_0, proj, dlon=1., dlat=1)
                if region == 'intraplate':
                    polygon2 = VENUS[region]
                else:
                    polygon2 = compute_region_venus(polygon_map, VENUS[region].exterior, proj, show_bar=False)
                first_loop = False
            sensor = Point(proj(lon_0_current, lat_0_current))

            for iradius, radius in enumerate(l_radius):

                itotal += 1
                
                #t = time()
                #surface1 = spherical_cap_boundary_v2(g, lat_0_current, lon_0_current, [radius], num_points=1000)
                #iradius = 0
                #print(f'#0 Elapsed {time()-t} s')
                #t = time()
                if iregion == 0:
                    polygon1s.append( compute_TL_region(polygon_map, surface1_loc[iradius*num_points:(iradius+1)*num_points, :], proj, lon_0, lon_0_current, sensor, db, subsample_db=5) )
                #print(f'#1 Elapsed {time()-t} s')

                #t = time()
                intersection = polygon2.intersection(polygon1s[itotal])
                #polygon1 = None
                #print(f'#2 Elapsed {time()-t} s')
                ratio = compute_surface_area_ratio(intersection, polygon_map)
                loc_dict = {'region': region, 'iloc': iloc, 'lon': lon_0_current, 'lat': lat_0_current, 'iradius': iradius, 'radius': radius, 'ratio': ratio,}
                ratio_df = ratio_df.append([loc_dict])

    ratio_df.reset_index(drop=True, inplace=True)
    ratio_df.to_csv(output_file, index=False, header=True)

###############################
## PROBABILISTIC MODEL BELOW ##
###############################

from scipy import interpolate
from pyrocko import moment_tensor as pmt
from obspy.geodetics import degrees2kilometers
from tqdm import tqdm
from scipy import special
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pyrocko import moment_tensor as mtm
from tqdm import tqdm
import geopandas as gpd 
from scipy.special import erf

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import RectBivariateSpline
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
import matplotlib.colors as mcol
from matplotlib.patches import Polygon  as Polygon_mpl
from sklearn.cluster import DBSCAN
import matplotlib.patches as mpatches

def get_regions(dir_venus_data):

    PATH_VENUS_DATA = os.path.join(dir_venus_data)
    PATH_VENUS = os.path.join(f"{PATH_VENUS_DATA}tectonic_settings_Venus")
    VENUS = {
        'corona': gpd.read_file(f"{PATH_VENUS}/corona.shp"),
        'rift': gpd.read_file(f"{PATH_VENUS}/rifts.shp"),
        'ridge': gpd.read_file(f"{PATH_VENUS}/ridges.shp")
    }
    return VENUS

def get_volcano_locations(file_volcano):
    volcanoes = pd.read_csv(file_volcano, header=[0])
    return volcanoes        

def get_slopes(file_slopes):
    pd_slopes = pd.read_csv(file_slopes, header=[0])
    return pd_slopes        

# Define the integrand function
def integrand(t, sigma):
    return 1 + erf(t / sigma)

# Perform the integration
def integral_f(Tmax, sigma, times=np.linspace(0., 1., 100)):
    
    init_shape = Tmax.shape
    Tmax = Tmax.ravel()
    sigma = sigma.ravel()
    id_TMAX = np.arange(Tmax.size)
    TIMES, IDs = np.meshgrid(times, id_TMAX)
    TIMES *= Tmax[IDs]
    DT = np.diff(TIMES, axis=1)[:,0]
    #print(IDs.shape, IDs.max(), Tmax.shape, SIGMA.shape, DT.shape)
    integral = np.sum(integrand(Tmax[IDs], sigma[IDs]), axis=1)*DT
    return integral.reshape(init_shape)

def compute_EI_from_VEI(alt_balloon=50., rho_air=65., std_shadow_zone=100., vei=np.linspace(0., 6., 100), Tmax=np.linspace(10., 100., 20), r=np.linspace(10., 400., 50)):

    TMAX_total, R_total = np.meshgrid(Tmax, r)
    SIGMA_total = TMAX_total/2.

    R_balloon = lambda R: np.sqrt(R**2+alt_balloon**2)
    acoustic_zone = lambda R: np.exp(-(R/std_shadow_zone)**2)

    factor_GF = (1./(integral_f(TMAX_total, SIGMA_total)))*(rho_air/(8*np.pi*1e3))*(1e9)
    factor_GF_median = np.median(factor_GF)
    factor_GF_q25 = np.quantile(factor_GF, q=0.25)
    factor_GF_q75 = np.quantile(factor_GF, q=0.75)
    TL_new = lambda dist, VEI: acoustic_zone(dist)*factor_GF_median*(1./R_balloon(dist))*10**(VEI-5)
    TL_new_qmin = lambda dist, VEI: acoustic_zone(dist)*factor_GF_q25*(1./R_balloon(dist))*10**(VEI-5)
    TL_new_qmax = lambda dist, VEI: acoustic_zone(dist)*factor_GF_q75*(1./R_balloon(dist))*10**(VEI-5)

    return TL_new, TL_new_qmin, TL_new_qmax

def convert_VEI_to_Mw(vjet = np.linspace(0.1, 0.5, 80), rho_tephra = np.linspace(1., 3.5, 40), vps = np.linspace(2., 5., 20), factor_scaling = np.logspace(-2, 1, 10)):

    FACTOR_conv, VP_conv, RHO_conv, VJET_conv = np.meshgrid(factor_scaling, vps, rho_tephra, vjet)

    factor_mw = lambda factor_scaling, vp, rho_tephra, vjet: -np.log10((factor_scaling*vp*1e3*vjet*1e3*rho_tephra*1e3))-4.
    output = factor_mw(FACTOR_conv, VP_conv, RHO_conv, VJET_conv)
    factor_mw_median = np.quantile(output, q=0.5)
    factor_mw_q25 = np.quantile(output, q=0.25)
    factor_mw_q75 = np.quantile(output, q=0.75)
    VEI_to_Mw_median = lambda vei: (vei - factor_mw_median)*(2./3.) -6.07
    VEI_to_Mw_q25 = lambda vei: (vei - factor_mw_q25)*(2./3.) -6.07
    VEI_to_Mw_q75 = lambda vei: (vei - factor_mw_q75)*(2./3.) -6.07

    return VEI_to_Mw_median, VEI_to_Mw_q25, VEI_to_Mw_q75

def get_TL_curves_with_EI(file_curve, dist_min=100., alt_balloon=50., rho0=50, rhob=1.):

    TL_seismic_new, TL_seismic_new_qmin, TL_seismic_new_qmax = get_TL_curves(file_curve, dist_min)

    VEI_to_Mw_median, VEI_to_Mw_q25, VEI_to_Mw_q75 = convert_VEI_to_Mw()
    TL_seismic_new_VEI = lambda dist, VEI: TL_seismic_new(dist, VEI_to_Mw_median(VEI))
    TL_seismic_new_qmin_VEI = lambda dist, VEI: TL_seismic_new_qmin(dist, VEI_to_Mw_median(VEI))
    TL_seismic_new_qmax_VEI = lambda dist, VEI: TL_seismic_new_qmax(dist, VEI_to_Mw_median(VEI))

    TL_EI_new, TL_EI_new_qmin, TL_EI_new_qmax = compute_EI_from_VEI(alt_balloon=alt_balloon)

    
    density_ratio = np.sqrt(rhob/rho0)

    TL_new = lambda dist, VEI: np.maximum(density_ratio*TL_EI_new(dist, VEI), TL_seismic_new_VEI(dist, VEI))
    TL_new_qmin = lambda dist, VEI: np.maximum(density_ratio*TL_EI_new_qmin(dist, VEI), TL_seismic_new_qmin_VEI(dist, VEI))
    TL_new_qmax = lambda dist, VEI: np.maximum(density_ratio*TL_EI_new_qmax(dist, VEI), TL_seismic_new_qmax_VEI(dist, VEI))

    return TL_new, TL_new_qmin, TL_new_qmax

def plot_TL(file_curve, TL_new, TL_new_qmin, TL_new_qmax, unknown, dists=np.linspace(1., 18e3, 100), mags=[5., 6.], noise_level=1e-2):

    
    plt.figure()
    for imag, mag in enumerate(mags):
        label = dict()
        if imag == len(mags)-1:
            label['label'] = 'uncertainty'
        plt.plot(dists, TL_new(dists, mag)/noise_level, label=mag, linewidth=3.)
        plt.fill_between(dists, TL_new_qmin(dists, mag)/noise_level, TL_new_qmax(dists, mag)/noise_level, color='grey', alpha=0.3, **label)

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Distance (km)', fontsize=12.)
    if unknown == 'pressure':
        plt.ylabel('Amplitude (Pa)', fontsize=12.)
    else:
        plt.ylabel('Velocity (m/s)', fontsize=12.)
    plt.legend(frameon=False, fontsize=12.)
    folder_curve_img = '/'.join(file_curve.split('/')[:-1])
    plt.savefig(f'{folder_curve_img}/TL_examples.png', transparent=True)

def get_TL_curves_one_freq(pd_all_amps_in, freq, dist_min, rho0, rhob, cb, use_savgol_filter, scalar_moment, unknown):

    pd_all_amps = pd_all_amps_in.copy()
    if 'fmax' in pd_all_amps.columns:
        diff = abs(pd_all_amps.fmax-freq)
        pd_all_amps = pd_all_amps.loc[diff==diff.min()]
        
    if 'median_rw' in pd_all_amps.columns:
        x = pd_all_amps.dist.values/1e3
        median = pd_all_amps['median_rw'].values
        q25 = pd_all_amps['median_q0.25_rw'].values
        q75 = pd_all_amps['median_q0.75_rw'].values
    else:
        x = pd_all_amps.dist.unique()/1e3
        median = pd_all_amps.groupby(['dist'])['amp_RW'].median().reset_index().amp_RW.values
        q25 = pd_all_amps.groupby(['dist'])['amp_RW'].quantile(q=0.25).reset_index().amp_RW.values
        q75 = pd_all_amps.groupby(['dist'])['amp_RW'].quantile(q=0.75).reset_index().amp_RW.values

    median /= scalar_moment
    q25 /= scalar_moment
    q75 /= scalar_moment

    if use_savgol_filter:
        from scipy.signal import savgol_filter
        
        """
        isep = np.argmin(abs(x-2000))

        fs = []
        for y in [pd_all_amps.median_rw.values, pd_all_amps['median_q0.25_rw'].values, pd_all_amps['median_q0.75_rw'].values]:
            y_smooth = np.zeros_like(y)
            window_size = 5  # Must be odd
            poly_order = 3
            y_smooth[:isep] = savgol_filter(y[:isep], window_size, poly_order)
            window_size = 15  # Must be odd
            poly_order = 3
            y_smooth[isep:] = savgol_filter(y[isep:], window_size, poly_order)
            f = interpolate.interp1d(x, y_smooth, bounds_error=False, fill_value=(y_smooth[0], y_smooth[-1]))
            fs.append(f)
        """
        fs = []
        for y in [median, q25, q75]:
            y_smooth = np.zeros_like(y)
            window_size = 5  # Must be odd
            poly_order = 3
            y_smooth[:] = savgol_filter(y[:], window_size, poly_order)
            #window_size = 15  # Must be odd
            #poly_order = 3
            #y_smooth[isep:] = savgol_filter(y[isep:], window_size, poly_order)
            f = interpolate.interp1d(x, y_smooth, bounds_error=False, fill_value=(y_smooth[0], y_smooth[-1]))
            fs.append(f)

        f_mean = fs[0]
        f_qmin = fs[1]
        f_qmax = fs[2]
    
    else:
        ## Rayleigh waves
        f_mean = interpolate.interp1d(x, median, bounds_error=False, fill_value=(median[0], median[-1]))
        f_qmin = interpolate.interp1d(x, q25, bounds_error=False, fill_value=(q25[0], q25.iloc[-1]))
        f_qmax = interpolate.interp1d(x, q75, bounds_error=False, fill_value=(q75[0], q75[-1]))

        ## Body waves
        """
        f_mean = interpolate.interp1d(degrees2kilometers(pd_all_amps.dist.values), pd_all_amps.median_body.values, bounds_error=False, fill_value=(pd_all_amps.median_body.iloc[0], pd_all_amps.median_body.iloc[-1]))
        f_qmin = interpolate.interp1d(degrees2kilometers(pd_all_amps.dist.values), pd_all_amps['median_q0.25_body'].values, bounds_error=False, fill_value=(pd_all_amps['median_q0.25_body'].iloc[0], pd_all_amps['median_q0.25_body'].iloc[-1]))
        f_qmax = interpolate.interp1d(degrees2kilometers(pd_all_amps.dist.values), pd_all_amps['median_q0.75_body'].values, bounds_error=False, fill_value=(pd_all_amps['median_q0.75_body'].iloc[0], pd_all_amps['median_q0.75_body'].iloc[-1]))
        """

    TL_base_seismic = lambda dist, m0: pmt.magnitude_to_moment(m0)*f_mean(dist)
    #TL_base_seismic_std = lambda dist, m0: pmt.magnitude_to_moment(m0)*f_std(dist)
    TL_base_seismic_qmin = lambda dist, m0: pmt.magnitude_to_moment(m0)*f_qmin(dist)
    TL_base_seismic_qmax = lambda dist, m0: pmt.magnitude_to_moment(m0)*f_qmax(dist)

    ## Mag vs amp relationship -> https://gfzpublic.gfz-potsdam.de/rest/items/item_65142/component/file_292577/content
    """
    TL_base_seismic_disp = lambda dist, m0: 10**(m0 -1.66*np.log10(kilometers2degrees(dist)) -3.3)*period # eq. 3
    TL_base_seismic = lambda dist, m0: 1e-6*TL_base_seismic_disp(dist, m0)*2*np.pi/period
    """

    if unknown == 'pressure':
        density_ratio = rhob*cb*np.sqrt(rho0/(rhob))
    else:
        density_ratio = np.sqrt(rho0/(rhob))

    #TL_base = lambda dist, m0: density_ratio*(TL_base_seismic(dist,m0)*1e-6)/(2*np.pi*period) # Raphael
    TL_base = lambda dist, m0: density_ratio*(TL_base_seismic(dist,m0))
    TL_base_qmin = lambda dist, m0: density_ratio*(TL_base_seismic_qmin(dist,m0))
    TL_base_qmax = lambda dist, m0: density_ratio*(TL_base_seismic_qmax(dist,m0))
    
    TL_new = lambda dist, m0: TL_base(dist, m0)*(dist>=dist_min) + TL_base(dist_min, m0)*(dist<dist_min)
    TL_new_qmin = lambda dist, m0: TL_base_qmin(dist, m0)*(dist>=dist_min) + TL_base_qmin(dist_min, m0)*(dist<dist_min)
    TL_new_qmax = lambda dist, m0: TL_base_qmax(dist, m0)*(dist>=dist_min) + TL_base_qmax(dist_min, m0)*(dist<dist_min)

    return TL_new, TL_new_qmin, TL_new_qmax

def get_TL_curves(file_curve, freq, dist_min = 100., rho0=50., rhob=1., cb=250., use_savgol_filter=False, plot=False, scalar_moment=1, unknown='pressure', return_dataframe=False):

    only_one_TL = False
    if isinstance(freq, float):
        freq = [freq]
        only_one_TL = True
    elif not isinstance(freq, list):
        print("The variable is neither a float nor a list.")
        return None, None, None

    pd_all_amps = pd.read_csv(file_curve, header=[0])
    
    TL_new, TL_new_qmin, TL_new_qmax = dict(), dict(), dict()
    for one_freq in freq:
        TL_new_loc, TL_new_qmin_loc, TL_new_qmax_loc = get_TL_curves_one_freq(pd_all_amps, one_freq, dist_min, rho0, rhob, cb, use_savgol_filter, scalar_moment, unknown)
        TL_new[one_freq] = TL_new_loc
        TL_new_qmin[one_freq] = TL_new_qmin_loc
        TL_new_qmax[one_freq] = TL_new_qmax_loc

        if plot:
            plot_TL(file_curve, TL_new_loc, TL_new_qmin_loc, TL_new_qmax_loc, unknown)

    if only_one_TL:
        TL_new, TL_new_qmin, TL_new_qmax = TL_new_loc, TL_new_qmin_loc, TL_new_qmax_loc

    if return_dataframe:
        return TL_new, TL_new_qmin, TL_new_qmax, pd_all_amps
    else:
        return TL_new, TL_new_qmin, TL_new_qmax

def get_surface_ratios(file_ratio):
    surface_ratios = pd.read_csv(file_ratio)
    surface_ratios.loc[surface_ratios.radius==surface_ratios.radius.min(), 'ratio_map'] = 0.
    surface_ratios.loc[surface_ratios.radius==surface_ratios.radius.min(), 'ratio'] = 0.
    return surface_ratios

#######################
## PROBA CLASS MODEL ##
#######################

class proba_model:

    def __init__(self, pd_slopes, surface_ratios, TL, TL_qmin, TL_qmax):
        
        self.pd_slopes = pd_slopes
        self.surface_ratios = surface_ratios
        self.TL = TL
        self.TL_qmin = TL_qmin
        self.TL_qmax = TL_qmax

    @staticmethod
    def return_number_per_cat_and_setting(mw, pd_slopes, cat_quake, setting):

        slope = pd_slopes.loc[(pd_slopes.type_setting==setting)&(pd_slopes.type_unknown=='slope'), cat_quake].iloc[0]
        intercept = pd_slopes.loc[(pd_slopes.type_setting==setting)&(pd_slopes.type_unknown=='intercept'), cat_quake].iloc[0]
        func = lambda mw: 10**(np.log10(mtm.magnitude_to_moment(mw))*slope+intercept)
        
        return func(mw)

    def compute_ratemag(self, region):
        
        Nquake_over_mag = self.return_number_per_cat_and_setting(self.M0s, self.pd_slopes, region, self.scenario)
        Nquake_over_mag_min = self.return_number_per_cat_and_setting(self.m_min, self.pd_slopes, region, self.scenario)
        Nquake_mag = (Nquake_over_mag_min-Nquake_over_mag)/Nquake_over_mag_min
        f_mag = np.gradient(Nquake_mag, self.dproba_M0s, edge_order=2)
        F_MAGS = np.zeros(self.shape_init)
        F_MAGS += f_mag[:,np.newaxis,np.newaxis,np.newaxis]
        return F_MAGS

    def compute_ratio_vs_loc(self, lats, lons, locations, region):
        
        surface_ratios_region = self.surface_ratios.loc[self.surface_ratios.region==region]

        #l_radius = surface_ratios_region.radius.unique()
        l_iloc = surface_ratios_region['iloc'].unique()
        l_lats = [surface_ratios_region.loc[surface_ratios_region['iloc']==iloc, 'lat'].iloc[0] for iloc in l_iloc]
        l_lons = [surface_ratios_region.loc[surface_ratios_region['iloc']==iloc, 'lon'].iloc[0] for iloc in l_iloc]
        #ratios = surface_ratios_region['ratio'].values.reshape((l_iloc.size, l_radius.size)) # iloc x radius
        RATIOs = np.zeros(self.shape_init)
        
        for iloc in locations:
            
            if self.homogeneous_ratios:
                dist_degree = np.radians((self.dists/(np.pi*self.r_venus))*180.)
                ratios_values_interp = (1-np.cos(dist_degree))/2.
            else:
                isurface = np.argmin(np.sqrt((lats[iloc]-l_lats)**2+(lons[iloc]-l_lons)**2))
                surface_ratios_region_iloc = surface_ratios_region.loc[surface_ratios_region['iloc']==l_iloc[isurface]]
                #print(isurface, region, surface_ratios_region['iloc'].unique())
                radius_values = surface_ratios_region_iloc.radius.values
                ratios_values = surface_ratios_region_iloc.ratio.values
            
                ratios_values_interp = np.interp(self.dists, radius_values/1e3, ratios_values)
            
            RATIOs[:,:,:,iloc] = np.gradient(ratios_values_interp, self.dproba_dists)[np.newaxis,:,np.newaxis]
            
        return RATIOs

    def compute_cum_pdf(self, DISTS, MAG, DETECT_T):

        DISTS_r, MAG_r, DETECT_T_r =  DISTS.reshape(self.shape_init), MAG.reshape(self.shape_init), DETECT_T.reshape(self.shape_init)
        x = DISTS_r[:,:,0:1,0] # M0s x dists x SNR x loc
        mu = self.TL(x, MAG_r[:,:,0:1,0])/self.noise_level
        sigma_qmax = self.TL_qmax(x, MAG_r[:,:,0:1,0])/self.noise_level
        sigma_qmin = self.TL_qmin(x, MAG_r[:,:,0:1,0])/self.noise_level
        
        total_pdf = np.zeros(self.shape_init)
        cums_max = 1-0.5*(1+special.erf( (DETECT_T_r[0:1,0:1,:,0]-mu)/(sigma_qmax*np.sqrt(2.)) ))
        cums_min = 1-0.5*(1+special.erf( (DETECT_T_r[0:1,0:1,:,0]-mu)/(sigma_qmin*np.sqrt(2.)) ))
        
        cums = np.where(DETECT_T_r[0:1,0:1,:,0] < mu, cums_min, cums_max)
        
        total_pdf += cums[:,:,:,np.newaxis]
            
        return total_pdf.ravel()
        
    def integrate_cum_pdf(self, DISTS, MAG, DETECT_T, RATIOs, F_MAGS):
        total_pdf = self.compute_cum_pdf(DISTS, MAG, DETECT_T)
        
        integrated = {}
        for region in RATIOs:
            integrated[region] = (F_MAGS[region]*RATIOs[region]*total_pdf).reshape(self.shape_init).sum(axis=(0,1)) 
            
        return integrated

    def compute_rate(self, DISTS, MAG, DETECT_T, RATIOs, F_MAGS):

        integrals = self.integrate_cum_pdf(DISTS, MAG, DETECT_T, RATIOs, F_MAGS)

        Nquake_over_mag = lambda region: self.return_number_per_cat_and_setting(self.m_min, self.pd_slopes, region, self.scenario)
        if self.homogeneous_ratios:
            region_str = {}
            region_str['coronae'] = 'corona'
            region_str['ridges'] = 'ridge'
            region_str['rifts'] = 'rift'
            region_str['intraplate'] = 'intraplate'
            l_regions = self.pd_slopes.loc[:,~self.pd_slopes.columns.isin(['type_setting', 'type_unknown'])].columns.values
            Nquake_over_mag = lambda region: np.sum([self.surface_ratios.loc[self.surface_ratios['region']==region_str[one_region]].ratio.max()*self.return_number_per_cat_and_setting(self.m_min, self.pd_slopes, one_region, self.scenario) for one_region in l_regions])
            
        return [Nquake_over_mag(region)*self.dproba_dists*self.dproba_M0s*integrals[region] for region in integrals]

    def compute_Poisson(self, DISTS, MAG, DETECT_T, RATIOs, F_MAGS):
        if self.rates_provided is None:
            rates = self.compute_rate(DISTS, MAG, DETECT_T, RATIOs, F_MAGS)
        else:
            rates = [rate for _, rate in self.rates_provided.items()]
        
        proba = 1.-np.prod([np.exp(-self.duration*rate) for rate in rates], axis=0)
        if self.return_rate:
            return proba, rates
        return proba

    def get_ratios_famp(self, lats, lons, locations):

        F_MAGS = {}
        RATIOs = {}
        l_regions = self.pd_slopes.loc[:, ~self.pd_slopes.columns.str.contains('type')].columns.values
        for region in l_regions:

            if (not region=='intraplate') and 'inactive' in self.scenario:
                if self.verbose:
                    print(f'Inactive scenario - Skipping region {region}')
                return None
            
            if self.verbose:
                print(f'Computing rate for region: {region}')
            F_MAGS[region] = self.compute_ratemag(region).ravel() 
            region_str = region
            if region_str in 'coronae':
                region_str = 'corona'
            elif region_str == 'ridges':
                region_str = 'ridge'
            elif region_str == 'rifts':
                region_str = 'rift'
            if (not self.homogeneous_ratios) or (self.homogeneous_ratios and region=='intraplate'): # only need one region if homogeneous distributions of venusquakes
                if self.verbose:
                    print(f'Computing surface ratio for region: {region}')
            RATIOs[region] = self.compute_ratio_vs_loc(lats, lons, locations, region_str).ravel()

        if self.homogeneous_ratios:
            region_str = {}
            region_str['coronae'] = 'corona'
            region_str['ridges'] = 'ridge'
            region_str['rifts'] = 'rift'
            region_str['intraplate'] = 'intraplate'
            F_MAGS['intraplate'] *= self.surface_ratios.loc[self.surface_ratios.region=='intraplate'].ratio.max()
            for region in l_regions:
                if not region=='intraplate':
                    F_MAGS['intraplate'] += F_MAGS[region]*self.surface_ratios.loc[self.surface_ratios.region==region_str[region]].ratio.max()
                    if self.verbose:
                        print(f'Combining rate of region {region} into intraplate')
                    del F_MAGS[region]

        return F_MAGS, RATIOs

    def init_discretization(self):

        self.dproba_dists = self.dists[1]-self.dists[0]
        self.dproba_M0s = self.M0s[1]-self.M0s[0]

    def init_parameter_space(self, lats, lons, locations):

        DISTS, MAG, DETECT_T, LOC = np.meshgrid(self.dists, self.M0s, self.SNR_thresholds, locations) # M0s x dists x SNR x loc
        self.shape_init = DISTS.shape
        DISTS, MAG, DETECT_T, LOC = DISTS.ravel(), MAG.ravel(), DETECT_T.ravel(), LOC.ravel()
        return DISTS, MAG, DETECT_T, LOC

    def compute_proba_map(self, scenario, dists, M0s, SNR_thresholds, noise_level, duration, all_lats, all_lons, homogeneous_ratios, m_min, r_venus, return_rate=False, rates_provided=None, verbose=False):

        self.scenario = scenario
        self.dists = dists
        self.M0s = M0s
        self.SNR_thresholds = SNR_thresholds
        self.noise_level = noise_level
        self.duration = duration
        self.all_lats = all_lats
        self.all_lons = all_lons
        self.homogeneous_ratios = homogeneous_ratios
        self.m_min = m_min
        self.r_venus = r_venus
        self.verbose = verbose
        self.return_rate = return_rate
        self.rates_provided = rates_provided

        self.init_discretization()
        
        self.proba_all = np.zeros((self.SNR_thresholds.size, self.all_lats.size, self.all_lons.size))
        self.rates_all = None
        if self.return_rate:
            l_regions = self.pd_slopes.loc[:, ~self.pd_slopes.columns.str.contains('type')].columns.values
            self.rates_all = {region: np.zeros((self.SNR_thresholds.size, self.all_lats.size, self.all_lons.size)) for region in l_regions}

        for ilon, lon in tqdm(enumerate(self.all_lons), total=len(all_lons)):

            lats_orig, lons_orig = self.all_lats, np.array([lon])
            lats, lons = np.meshgrid(lats_orig, lons_orig)
            lats, lons = lats.ravel(), lons.ravel()
            locations = np.arange(lats.size)

            DISTS, MAG, DETECT_T, LOC = self.init_parameter_space(lats, lons, locations)

            F_MAGS, RATIOs = self.get_ratios_famp(lats, lons, locations)

            self.m_min = self.M0s.min() ## TODO: Check this, why not using mmin?
            proba = self.compute_Poisson(DISTS, MAG, DETECT_T, RATIOs, F_MAGS)
            if self.return_rate:
                proba, rates = proba
            self.proba_all[:,:,ilon] = proba
            if self.return_rate:
                for region, rate in zip(l_regions, rates):
                    self.rates_all[region][:,:,ilon] = rate

###################
## PROBA AIRGLOW ##
###################

class proba_model_airglow(proba_model):

    def __init__(self, pd_slopes, surface_ratios, TL, TL_qmin, TL_qmax):
        super().__init__(pd_slopes, surface_ratios, TL, TL_qmin, TL_qmax)

    @staticmethod
    def return_number_per_cat_and_setting(mw, pd_slopes, cat_quake, setting):

        popt = pd_slopes.loc[(pd_slopes.type_setting==setting), cat_quake].values
        poly1d = np.poly1d(popt)
        func = lambda mw: 10**poly1d(mw)
        
        return func(mw)

##########################
## PROBA MODEL WRINKLES ##
##########################

class proba_model_wrinkles(proba_model):

    def __init__(self, pd_slopes, surface_ratios, TL, TL_qmin, TL_qmax):
        super().__init__(pd_slopes, surface_ratios, TL, TL_qmin, TL_qmax)

    @staticmethod
    def return_number_per_cat_and_setting(mw, pd_slopes, cat_quake, setting):

        popt = pd_slopes.loc[(pd_slopes.type_setting==setting), cat_quake].values
        poly1d = np.poly1d(popt)
        func = lambda mw: 10**poly1d(mw)
        
        return func(mw)

###########################
## PROBA MODEL VOLCANOES ##
###########################

def haversine(lon1, lat1, lon2, lat2, r = 6052.):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Calculate the result
    return c * r

class proba_model_volcano(proba_model):

    def __init__(self, lat_volcanoes, lon_volcanoes, pd_slopes, surface_ratios, TL, TL_qmin, TL_qmax):
        super().__init__(pd_slopes, surface_ratios, TL, TL_qmin, TL_qmax)
        self.lat_volcanoes = lat_volcanoes
        self.lon_volcanoes = lon_volcanoes

    @staticmethod
    def return_number_per_cat_and_setting(mw, pd_slopes, cat_quake, setting):
    
        coefs = pd_slopes.loc[(pd_slopes.type_setting==setting)&(pd_slopes.type_unknown.str.contains('coef_')), cat_quake].values
        p = np.poly1d(coefs)
        func = lambda mw: 10**(p(mw))

        return func(mw)

    def compute_cum_pdf(self, DISTS, MAG, DETECT_T):

        DISTS_r, MAG_r, DETECT_T_r =  DISTS.reshape(self.shape_init), MAG.reshape(self.shape_init), DETECT_T.reshape(self.shape_init)
        x = DISTS_r[:,:,0:1,:] # M0s x dists x SNR x loc
        mu = self.TL(x, MAG_r[:,:,0:1,:])/self.noise_level
        sigma_qmax = self.TL_qmax(x, MAG_r[:,:,0:1,:])/self.noise_level
        sigma_qmin = self.TL_qmin(x, MAG_r[:,:,0:1,:])/self.noise_level
        
        total_pdf = np.zeros(self.shape_init)
        cums_max = 1-0.5*(1+special.erf( (DETECT_T_r[0:1,0:1,:,0:1]-mu)/(sigma_qmax*np.sqrt(2.)) ))
        cums_min = 1-0.5*(1+special.erf( (DETECT_T_r[0:1,0:1,:,0:1]-mu)/(sigma_qmin*np.sqrt(2.)) ))
        
        cums = np.where(DETECT_T_r[0:1,0:1,:,0:1] < mu, cums_min, cums_max)
        
        total_pdf += cums[:,:,:,:]
            
        return total_pdf.ravel()

    def compute_ratio_vs_loc(self, lats, lons, locations, region):
        RATIOs = np.ones(self.shape_init) ## Volcanoes occur at discrete locations, no need for integral along radius
        return RATIOs

    def init_discretization(self):
        self.dproba_dists = 1. ## Volcanoes occur at discrete locations, no need for integral along radius
        self.dproba_M0s = self.M0s[1]-self.M0s[0]

    def init_parameter_space(self, lats, lons, locations):

        idx_dists = np.arange(self.lon_volcanoes.size)
        IDX_DISTS, MAG, DETECT_T, LOC = np.meshgrid(idx_dists, self.M0s, self.SNR_thresholds, locations) # M0s x dists x SNR x loc
        self.shape_init = IDX_DISTS.shape
        IDX_DISTS, MAG, DETECT_T, LOC = IDX_DISTS.ravel(), MAG.ravel(), DETECT_T.ravel(), LOC.ravel()

        DISTS = haversine(lons[LOC], lats[LOC], self.lon_volcanoes[IDX_DISTS], self.lat_volcanoes[IDX_DISTS])

        return DISTS, MAG, DETECT_T, LOC

####################################
## BALLOON TRAJECTORY INTEGRATION ##
####################################

def plot_maps_all_trajectories(pd_final_probas, lons, lats, mission_durations, cmap_bounds=np.linspace(0., 0.8, 15)):

    LONS, LATS = np.meshgrid(lons, lats)

    cmap = cm.get_cmap("Reds", lut=len(cmap_bounds))
    norm = mcol.BoundaryNorm(cmap_bounds, cmap.N)
        
    fig = plt.figure(figsize=(10,10))
    grid = fig.add_gridspec(len(mission_durations), 3)
    iduration = -1
    for duration, one_duration in pd_final_probas.groupby(['duration']):
        iduration += 1
        isnr = -1
        for snr, one_snr in one_duration.groupby(['snr']):
            isnr +=1
            field = one_snr.proba.values.reshape(lons.size, lats.size)
            
            
            ax = fig.add_subplot(grid[iduration, isnr])
            m = Basemap(projection='robin', lon_0=0, ax=ax)
            m.drawmeridians(np.linspace(-180., 180., 5), labels=[0, 0, 0, 1], fontsize=10,)
            m.drawparallels(np.linspace(-90., 90., 5), labels=[1, 0, 0, 0], fontsize=10,)
            x, y = m(LONS.ravel(), LATS.ravel())
            x, y = x.reshape(LONS.shape), y.reshape(LONS.shape)
            sc = m.pcolormesh(x, y, field.T, cmap=cmap, norm=norm)
            #plt.colorbar(sc)
            plt.title(f'SNR {snr} - {duration:.0f} days')

    fmt = lambda x, pos: '{:.2f} %'.format(x*1e2) # 
    axins = inset_axes(ax, width="6%", height="250%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = fig.colorbar(sc, format=FuncFormatter(fmt), cax=axins, orientation='vertical', extend='both', ticks=cmap_bounds[1:],)

def plot_proba_all_trajectories(pd_final_probas, mission_durations, xlim=[0, 1], ylim=[0, 40.]):

    fig = plt.figure(figsize=(14,3))
    grid = fig.add_gridspec(1, len(mission_durations))
    iduration = -1
    for duration, one_duration in pd_final_probas.groupby(['duration']):
        iduration += 1
        ax = fig.add_subplot(grid[0, iduration])
        ax.set_title(f'Mission {duration:.0f} days')
        for snr, one_snr in one_duration.groupby(['snr']):
            #values, bins = np.histogram(one_snr.proba, bins=20)
            #bin_centers = 0.5 * (bins[:-1] + bins[1:])
            #ax.bar(bin_centers, values/, width=(bins[1] - bins[0]), label=snr)
            ax.hist(one_snr.proba, bins=20, label=snr, density=True)
        ax.set_xlabel('Detection Probability')
        if iduration == 0:
            ax.legend(title='SNR', frameon=False)
            ax.set_ylabel('Probability Density Function')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

import VCD_trajectory_modules as VCD
"""
def compute_multiple_trajectories(proba_model, winds, lats, lons):
    
    LATS, LONS = np.meshgrid(lats, lons)
    LATS, LONS = LATS.ravel(), LONS.ravel()
    
    opt_trajectory = dict(
        nstep_max=1000, 
        time_max=3600*24*30*2,
        save_trajectory=False,
        folder = './data/',
    )
    pd_final_probas = pd.DataFrame()
    for lat, lon in tqdm(zip(LATS, LONS), total=LATS.size):
        start_location = [lat, lon] # lat, lon
        trajectory = VCD.compute_trajectory(winds, start_location, **opt_trajectory)
        new_trajectories = compute_proba_one_trajectory(trajectory, proba_model, norm_factor_time=3600., disable_bar=True) ## Venusquake
        pd_final_proba = new_trajectories.groupby('snr').last().reset_index()[['snr', 'proba']]
        pd_final_proba['lat'] = lat
        pd_final_proba['lon'] = lon
        pd_final_probas = pd.concat([pd_final_probas, pd_final_proba])
        
    return pd_final_probas
"""

def compute_proba_one_trajectory(trajectory_in, snrs, lats, lons, probas, snrs_selected=[1.,2.,5.], norm_factor_time=3600., disable_bar=False):
#def compute_proba_one_trajectory(trajectory_in, proba_model, snrs_selected=[1.,2.,5.], norm_factor_time=3600., disable_bar=False):
    
    #snrs = proba_model.SNR_thresholds
    #lats, lons = proba_model.all_lats, proba_model.all_lons
    #probas = proba_model.proba_all.copy() # SNR x lats x lons
    trajectory = trajectory_in.copy()
    isnrs = [np.argmin(abs(snrs-snr)) for snr in snrs_selected]

    g = Geod(ellps='WGS84')

    ## Assign time bins to each entry of the trajectory
    trajectory['dt'] = trajectory.time/norm_factor_time # time in hours
    bins = np.arange(int(trajectory['dt'].max())+1)
    trajectory['bin_dt'] = np.searchsorted(bins, trajectory['dt'].values)
    
    ## Find corresponding probabilities along trajectory
    LATS_STAT, LONS_STAT = np.meshgrid(lats, lons)
    LATS_STAT, LONS_STAT = LATS_STAT.ravel(), LONS_STAT.ravel()

    id_bins, id_map = np.meshgrid(np.arange(trajectory.shape[0]), np.arange(LATS_STAT.size))
    id_bins_shape = id_bins.shape
    id_bins, id_map = id_bins.ravel(), id_map.ravel()
    _, _, dists = g.inv(trajectory.lon.values[id_bins], trajectory.lat.values[id_bins], LONS_STAT[id_map], LATS_STAT[id_map])
    dists = dists.reshape(id_bins_shape)
    iclosest = dists.argmin(axis=0)
    trajectory['id_map'] = iclosest
    trajectory['lat_map'] = LATS_STAT[iclosest]
    trajectory['lon_map'] = LONS_STAT[iclosest]

    delta_each_hour = 1./24.
    f_traj_lat = interpolate.interp1d(trajectory.dt.values, trajectory.lat.values, bounds_error=False, fill_value=trajectory.lat.values[-1])
    f_traj_lon = interpolate.interp1d(trajectory.dt.values, trajectory.lon.values, bounds_error=False, fill_value=trajectory.lon.values[-1])
    new_trajectory = pd.DataFrame()
    for isnr, snr in tqdm(zip(isnrs, snrs_selected), total=len(isnrs), disable=disable_bar):
        probas_snr_ravelled = probas[isnr].ravel()
        time_start = 0.
        proba_not_detecting = 1.
        for _, one_dt in trajectory.groupby('bin_dt'):
            time_end = time_start + 1.
            times = np.arange(time_start, time_end, delta_each_hour)
            lats_hours, lons_hour = f_traj_lat(times), f_traj_lon(times)
            closest_cell = np.array([np.argmin(np.sqrt((lat-lats)**2+(lon-lons)**2)) for lat, lon in zip(lats_hours, lons_hour)])
            proba_not_detecting *= np.prod((1.-probas_snr_ravelled[closest_cell])**delta_each_hour)
            time_start = time_end

            trajectory_loc = trajectory.loc[trajectory.index.isin(one_dt.index)].copy()
            trajectory_loc['proba'] = 1.-proba_not_detecting
            trajectory_loc['snr'] = snr
            new_trajectory = pd.concat([new_trajectory, trajectory_loc])
    new_trajectory.reset_index(drop=True, inplace=True)

    ## Compute total probability
    """
    proba_total = np.ones(probas.shape[0])
    #probas_loc = probas.reshape((probas.shape[0], np.prod(probas.shape[1:])))
    new_trajectory = pd.DataFrame()
    for bin_dt, one_dt in tqdm(trajectory.groupby('bin_dt')):

        ## Take average of probabilities in the bin
        proba_loc = np.zeros(probas.shape[0])
        for isnr in range(proba_loc.size):
            l_probas = []
            for _, one_loc in one_dt.iterrows():
                l_probas.append( probas[isnr].ravel()[int(one_loc.id_map)] )
            proba_loc[isnr] = np.median(l_probas)
        proba_total *= (1.-proba_loc)
        
        for isnr, snr in zip(isnrs, snrs_selected):
            trajectory_loc = trajectory.loc[trajectory.index.isin(one_dt.index)].copy()
            trajectory_loc['proba'] = 1-proba_total[isnr]
            trajectory_loc['snr'] = snr
            new_trajectory = pd.concat([new_trajectory, trajectory_loc])
    """
    return new_trajectory

def compute_multiple_trajectories(snrs, lats, lons, probas, winds, mission_durations, max_number_months, inputs):
#def compute_multiple_trajectories(proba_model, winds, mission_durations, max_number_months, inputs):
    
    #LATS, LONS = np.meshgrid(lats, lons)
    #LATS, LONS = LATS.ravel(), LONS.ravel()
    icpu, LATS, LONS = inputs

    opt_trajectory = dict(
        nstep_max=1000, 
        time_max=3600*24*30*max_number_months,
        save_trajectory=False,
        folder = './data/',
    )
    pd_final_probas = pd.DataFrame()
    for lat, lon in tqdm(zip(LATS, LONS), total=LATS.size, disable=not icpu==0):
        start_location = [lat, lon] # lat, lon
        trajectory = VCD.compute_trajectory(winds, start_location, **opt_trajectory)
        #new_trajectories = compute_proba_one_trajectory(trajectory, proba_model, norm_factor_time=3600., disable_bar=True) ## Venusquake
        new_trajectories = compute_proba_one_trajectory(trajectory, snrs, lats, lons, probas, norm_factor_time=3600., disable_bar=True) ## Venusquake
        
        for target_duration in mission_durations:
            
            days = new_trajectories.time/(3600*24)
            pd_final_proba = new_trajectories.loc[days<=target_duration,:].groupby('snr').last().reset_index()[['snr', 'proba']]
            pd_final_proba['lat'] = lat
            pd_final_proba['lon'] = lon
            pd_final_proba['duration'] = target_duration
            pd_final_probas = pd.concat([pd_final_probas, pd_final_proba])
        
    return pd_final_probas

from functools import partial
from multiprocessing import get_context
def compute_multiple_trajectories_CPUs(proba_model, winds, LATS, LONS, mission_durations, max_number_months=4, nb_CPU=10):

    snrs = proba_model.SNR_thresholds
    lats, lons = proba_model.all_lats, proba_model.all_lons
    probas = proba_model.proba_all.copy() # SNR x lats x lons

    partial_compute_multiple_trajectories = partial(compute_multiple_trajectories, snrs, lats, lons, probas, winds, mission_durations, max_number_months)
    #partial_compute_multiple_trajectories = partial(compute_multiple_trajectories, proba_model, winds, mission_durations, max_number_months)
    nb_chunks = LATS.shape[0]
    idx_start_all = np.arange(nb_chunks)
    
    N = min(nb_CPU, nb_chunks)
    ## If one CPU requested, no need for deployment
    if N == 1:
        print('Running serial')
        pd_final_probas = partial_compute_multiple_trajectories((0, LATS, LONS))

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
            list_of_lists.append( (i, LATS[idx_start_all[idx]], LONS[idx_start_all[idx]]) )

        with get_context("spawn").Pool(processes = N) as p:
            print(f'Running across {N} CPU')
            all_pd_final_probas = p.map(partial_compute_multiple_trajectories, list_of_lists)
            p.close()
            p.join()

        pd_final_probas = gpd.GeoDataFrame()
        for idx, pd_final_probas_loc in zip(idxs, all_pd_final_probas):
            #gdf_loc['iscenario'] += idx
            pd_final_probas = pd.concat([pd_final_probas, pd_final_probas_loc], ignore_index=True)
        pd_final_probas.reset_index(drop=True, inplace=True)

    return pd_final_probas

#############################################
## VISUALIZATION PROBABILISTIC MODEL BELOW ##
#############################################

def draw_screen_poly(xy, ax, legend, color='red'):
    #x, y = m( lons, lats )
    #xy = zip(x,y)
    poly = Polygon_mpl( xy, facecolor=color, alpha=0.4, **legend )
    ax.add_patch(poly)

def extract_coords(gdf):
    points = []
    for geom in gdf.geometry:
        if geom.geom_type == 'LineString':
            points.extend(list(geom.coords))
    return points

def plot_regions(m, ax, VENUS, use_active_corona=False, plot_lines=True):

    if use_active_corona:
        active_corona = gpd.read_file(f"./data/active_corona_shape/active_corona.shp").iloc[0].geometry

    color_dict = {'corona': 'tab:red', 'rift': 'tab:green', 'ridge': 'tab:blue', 'intraplate': 'white', 'wrinkles': 'tab:green'}
    for region in VENUS:

        if region == 'intraplate':
            continue

        print(f'Processing region {region}')
        
        if region == 'wrinkles':
            points = extract_coords(VENUS[region])
            points = np.array(points)
            x_bins = np.linspace(-180., 180., 300)
            y_bins = np.linspace(-90., 90., 150)
            density, x_edges, y_edges = np.histogram2d(points[:,0], points[:,1], bins=[x_bins, y_bins])
            x_edges, y_edges = np.meshgrid(x_edges, y_edges)
            shape_init = x_edges.shape
            x_edges, y_edges = x_edges.ravel(), y_edges.ravel()
            x_edges, y_edges = m(x_edges, y_edges)
            x_edges, y_edges = x_edges.reshape(shape_init), y_edges.reshape(shape_init)
            density[density>1] = 1
            m.pcolormesh(x_edges, y_edges, density.T, alpha=1, cmap='Greens',)
            continue

        for pol_in in tqdm(VENUS[region].explode(index_parts=False).reset_index(drop=True).geometry.values):

            pol = pol_in
            if pol.geom_type == 'LineString':
                pol = pol.buffer(0.01)

            ext_in = pol.exterior
            if ext_in is None:
                continue

            ext_all = [ext_in]
            if use_active_corona and region == 'corona':
                save = ext_all[0]
                ext_all = []
                for active_pol in active_corona.geoms:
                    ext_all_loc = active_pol.intersection(Polygon(save).buffer(0))
                    if ext_all_loc.geom_type == 'MultiPolygon':
                        ext_all_loc = [geo.exterior for geo in ext_all_loc.geoms]
                    else:
                        ext_all_loc = [ext_all_loc.exterior]
                    ext_all += ext_all_loc
            
            #print(m(-180., -40.), m(0., -40.), m(130., -40.), m(300., -40.), m(360., -40.))
            for ext in ext_all:

                surface2 = [m(lon,lat) for lon, lat in list(ext.coords)]
                if len(surface2) == 0:
                    continue
                
                clustering = DBSCAN(eps=500000, min_samples=5).fit(np.array(surface2))
                #lines = []
                for label in np.unique(clustering.labels_):
                    coords = np.array(surface2)[clustering.labels_==label]
                    legend = {}
                    draw_screen_poly(coords, ax, legend, color=color_dict[region])
                
    #if not region == 'wrinkles':
    change_label = {'wrinkles': 'Wrinkle Ridges', 'corona': 'Coronae', 'ridge': 'Ridges', 'rift': 'Rifts'}
    patches = [mpatches.Patch(facecolor=color_dict[region], label=change_label[region], alpha=0.5, edgecolor='black') for region in VENUS]
    if 'ridge' in VENUS: ## Add intraplate
        patches += [mpatches.Patch(facecolor='white', label='Intraplate', alpha=0.5, edgecolor='black')]
    ax.legend(handles=patches, frameon=False, bbox_to_anchor=(1.1, -0.1), ncol=4, bbox_transform=ax.transAxes)
    if plot_lines:
        m.drawmeridians(np.linspace(-180., 180., 5), labels=[0, 0, 0, 1], fontsize=10)
        m.drawparallels(np.linspace(-90., 90., 5), labels=[1, 0, 0, 0], fontsize=10)
    
def interpolate_2d(current_map, lons_in, lats, toplot_in, dnew=1.):

    toplot = toplot_in.copy()
    lons = lons_in.copy()
    if lons_in.max() > 180.:
        lons[lons>=180.] -= 360.
    idx = np.argsort(lons)
    toplot[:,:] = toplot[idx,:]
    lons = lons[idx]

    #lons_new = np.arange(lons.min(), lons.max(), 2*dnew)
    lons_new = np.arange(-180., 180., 2*dnew)
    lats_new = np.arange(lats.min(), lats.max(), dnew)
    
    LAT, LON = np.meshgrid(lats_new, lons_new)
    shape = LON.shape
    x, y = current_map(LON.ravel(), LAT.ravel())
    x, y = x.reshape(shape), y.reshape(shape)
    
    #interpolator = RectBivariateSpline(lons, lats, toplot.T, kx=1, ky=1)
    interpolator = RectBivariateSpline(lons, lats, toplot, kx=1, ky=1)

    #print(lons_new.min(), lons_new.max(), lons.min(), lons.max())

    #LAT, LON = np.meshgrid(lats_new, lons_new)
    #shape_ = LON.shape
    
    toplot = interpolator.ev(LON.ravel(), LAT.ravel()).reshape(shape)
    #print(toplot.shape, LON.min(), LON.max(), LAT.min(), LAT.max(), x.min(), x.max(), y.min(), y.max())
    #plt.pcolormesh(y, x, toplot)


    return x, y, toplot, lons_new, lats_new

def one_map(ax, fig, proba, lats, lons, SNR_thresholds, snr, n_colors, show_title,  c_cbar = 'white', l_snr_to_plot=[], low_cmap=[], high_cmap=[], interpolate=True, proba_all_homo=None):
    
    m = Basemap(projection='robin', lon_0=0, ax=ax)
            
    LAT, LON = np.meshgrid(lats, lons)
    shape = LON.shape
    x, y = m(LON.ravel(), LAT.ravel())
    x, y = x.reshape(shape), y.reshape(shape)

    isnr = np.argmin(abs(SNR_thresholds-snr))
    toplot = proba[isnr,:]*1e2
    if proba_all_homo is not None:
        toplot /= proba_all_homo[isnr,0,0]*1e2
        toplot = 1e2*(1.-toplot)
    
    fmt = lambda x, pos: '{:.2f}'.format(x) # 
    if l_snr_to_plot:
        isnr = np.argmin(abs(SNR_thresholds-l_snr_to_plot[0]))
        toplot = proba[isnr,:]*1e2
        isnr = np.argmin(abs(SNR_thresholds-l_snr_to_plot[1]))
        toplot = 1e2*(1-(toplot - proba[isnr,:]*1e2)/toplot)
        fmt = lambda x, pos: '{:.2f}'.format(x) # 
        
    if show_title:
        if l_snr_to_plot:
            ax.set_title(f'$\%$ of probability decrease')
        else:
            ax.set_title(f'SNR {SNR_thresholds[isnr]:.0f}')
            
    
    if interpolate:
        x, y, toplot, lons_new, lats_new = interpolate_2d(m, lons, lats, toplot, dnew=1.)

    #print(low_cmap)
    if l_snr_to_plot or len(low_cmap)==0:
        cmap_bounds = np.linspace(toplot.min(), toplot.max(), n_colors)
        cmap = cm.get_cmap("Reds", lut=len(cmap_bounds))
    else:
        cmap_bounds = [0] + [c for c in low_cmap] + [c for c in high_cmap] # ISSI
    
        # Create low and high colormaps
        low_cmap_ = cm.get_cmap('Blues_r', len(low_cmap)+1)
        high_cmap_ = cm.get_cmap('Reds', len(high_cmap))

        # Combine the colors from both colormaps
        combined_colors = np.vstack((low_cmap_(np.linspace(0, 1, len(low_cmap)+1)), high_cmap_(np.linspace(0, 1, len(high_cmap)))))
        cmap = mcol.ListedColormap(combined_colors)

    norm = mcol.BoundaryNorm(cmap_bounds, cmap.N)
    
    sc = m.pcolormesh(x, y, toplot, alpha=1, cmap=cmap, norm=norm)
    m.drawmeridians(np.linspace(-180., 180., 5), labels=[0, 0, 0, 1], fontsize=12)
    m.drawparallels(np.linspace(-90., 90., 5), labels=[1, 0, 0, 0], fontsize=12)

    axins = inset_axes(ax, width="3%", height="100%", loc='lower left', bbox_to_anchor=(1.03, 0., 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = fig.colorbar(sc, format=FuncFormatter(fmt), cax=axins, orientation='vertical', extend='both', ticks=cmap_bounds[1:],)
   
    fontsize = 14.
    cbar.ax.tick_params(axis='both', colors=c_cbar, labelsize=fontsize)
    if not l_snr_to_plot:
        cbar.set_label('Detection probability (in %)', rotation=270, labelpad=15, color=c_cbar, fontsize=fontsize)

    return m
    
def plot_map(proba_model, VENUS, l_snr_to_plot=[], c_cbar='white', n_colors=20, show_title=True, low_cmap=[], high_cmap=[], proba_all_homo=None, plot_all_regions=False, plot_volcanoes=False, use_active_corona=False):
    
    lats = proba_model.all_lats
    lons = proba_model.all_lons
    SNR_thresholds = proba_model.SNR_thresholds
    proba = proba_model.proba_all

    #cmap_bounds = [0] + [c for c in np.arange(1.25e-2, 2.2e-2, 0.25e-2)] + [c for c in np.arange(1.2e-1, 1.5e-1, 0.05e-1)] # ISSI
    if not l_snr_to_plot:
        l_snr_to_plot = [1.]
    
    fig = plt.figure(figsize=(15,10))
    grid = fig.add_gridspec(2, len(l_snr_to_plot))
        
    for isnr_grid, snr in enumerate(l_snr_to_plot):
        
        ax = fig.add_subplot(grid[0, isnr_grid])
        m = one_map(ax, fig, proba, lats, lons, SNR_thresholds, snr, n_colors, show_title, c_cbar=c_cbar, low_cmap=low_cmap, high_cmap=high_cmap, proba_all_homo=proba_all_homo)
        if plot_volcanoes:
            x, y = m(proba_model.lon_volcanoes, proba_model.lat_volcanoes)
            m.scatter(x, y, marker='x', color='black', s=30)
        
        
    ax = fig.add_subplot(grid[1, 0])
    _ = one_map(ax, fig, proba, lats, lons, SNR_thresholds, snr, n_colors, show_title, c_cbar=c_cbar, l_snr_to_plot=l_snr_to_plot)
    
    if plot_all_regions and VENUS is not None:
        ax = fig.add_subplot(grid[1, 1])
        m = Basemap(projection='robin', lon_0=0, ax=ax)
        m.scatter(0., 0., latlon=True, s=0.1)
        plot_regions(m, ax, VENUS, use_active_corona=use_active_corona)
        
    fig.subplots_adjust(top=0.8, wspace=0.3)
    #fig.savefig('./test_data_Venus/map_probas.png', dpi=800., transparent=True)

def one_map_traj(fig, ax, lats, lons, new_trajectories_snr, VENUS, n_colors=10, c_cbar='white', fontsize=12., plot_time=False, alpha_traj=0.5, add_cbar=True):
    
    snr = new_trajectories_snr.snr.iloc[0]
    
    m = Basemap(projection='robin', lon_0=0, ax=ax)
            
    LAT, LON = np.meshgrid(lats, lons)
    shape = LON.shape
    x, y = m(LON.ravel(), LAT.ravel())
    x, y = x.reshape(shape), y.reshape(shape)

    fmt = lambda x, pos: '{:.0f}'.format(x) # 
    
    if plot_time:
        cmap_bounds = np.linspace(new_trajectories_snr.time.min()/(24*3600.), new_trajectories_snr.time.max()/(24*3600.), n_colors)
    else:
        cmap_bounds = np.linspace(new_trajectories_snr.proba.min()*1e2, new_trajectories_snr.proba.max()*1e2, n_colors)
    
    cmap = cm.get_cmap("viridis", lut=len(cmap_bounds))
    norm = mcol.BoundaryNorm(cmap_bounds, cmap.N)
    
    color_dict = {'corona': 'tab:red', 'rift': 'tab:green', 'ridge': 'tab:blue', 'intraplate': 'white'}
    if VENUS is not None:
        plot_regions(m, ax, VENUS, color_dict)
            
    #sc = m.pcolormesh(x, y, toplot, alpha=1, cmap=cmap, norm=norm)
    if plot_time:
        sc = m.scatter(new_trajectories_snr.lon, new_trajectories_snr.lat, c=new_trajectories_snr.time/(24*3600.), s=5, cmap=cmap, norm=norm, latlon=True, zorder=10, alpha=alpha_traj)
    else:
        sc = m.scatter(new_trajectories_snr.lon, new_trajectories_snr.lat, c=new_trajectories_snr.proba*1e2, s=5, cmap=cmap, norm=norm, latlon=True, zorder=10, alpha=alpha_traj)
    
    if add_cbar:
        axins = inset_axes(ax, width="80%", height="6%", loc='lower left', bbox_to_anchor=(0.1, -.2, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
        axins.tick_params(axis='both', which='both', labeltop=False, labelleft=False, top=False, left=False)
        cbar = fig.colorbar(sc, format=FuncFormatter(fmt), cax=axins, orientation='horizontal', extend='both', ticks=cmap_bounds[1:],)
        
        if plot_time:
            name_cbar = f'Time (days)'
        else:
            name_cbar = f'Detection probability (%) for SNR={snr:.1f}'
        cbar.set_label(name_cbar, rotation=0, labelpad=10, color=c_cbar, fontsize=fontsize)
        axins.xaxis.set_label_position('bottom')
        axins.xaxis.set_ticks_position('bottom')
        cbar.ax.tick_params(axis='both', colors=c_cbar, labelsize=fontsize-2., )

    if VENUS is not None:
        patches = [mpatches.Patch(facecolor=color_dict[region], label=region, alpha=0.5, edgecolor='black') for region in color_dict]
        ax.legend(handles=patches, frameon=False, bbox_to_anchor=(1., -0.1), columnspacing=0.5, handletextpad=0.25, ncol=4, bbox_transform=ax.transAxes, fontsize=fontsize, labelcolor=c_cbar)
    
    return m
    
def add_vertical_cbar(fig, ax, sc, cmap_bounds, fmt, c_cbar, name_cbar):

    axins = inset_axes(ax, width="3%", height="80%", loc='lower left', bbox_to_anchor=(1.03, 0.1, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar = fig.colorbar(sc, format=FuncFormatter(fmt), cax=axins, orientation='vertical', extend='both', ticks=cmap_bounds[:],)
    cbar.set_label(name_cbar, rotation=90, labelpad=10, color=c_cbar, fontsize=12)
    cbar.ax.tick_params(axis='both', colors=c_cbar, labelsize=12., )

def plot_trajectory(new_trajectories_total, proba_model, winds, VENUS=None, snr=1., n_colors=10, c_cbar='white', fontsize=15., ylim=[0., 20.], plot_time=False, plot_volcanoes=False):
    
    lats, lons = proba_model.all_lats, proba_model.all_lons
    
    fig = plt.figure(figsize=(14,8))
    grid = fig.add_gridspec(2, 2)
        
    ax = fig.add_subplot(grid[1, 1])
    ax_winds = fig.add_subplot(grid[0, 1])
    ax_vs_time = fig.add_subplot(grid[1, 0])
    ax_vs_lon = fig.add_subplot(grid[0, 0], sharex=ax_vs_time)
    
    iseismicity = -1
    linestyles = ['-', '--', ':']
    cmap = sns.color_palette('rocket', n_colors=new_trajectories_total.snr.unique().size,)
    lines_snr = []
    lines_seismicity = []
    for seismicity, new_trajectories in new_trajectories_total.groupby('seismicity'):
    
        iseismicity += 1
        if iseismicity == 0:
            new_trajectories_snr = new_trajectories.loc[new_trajectories.snr==snr]
            m = one_map_traj(fig, ax, lats, lons, new_trajectories_snr, VENUS, n_colors=n_colors, c_cbar=c_cbar, fontsize=fontsize, plot_time=plot_time)
            m_winds = one_map_traj(fig, ax_winds, lats, lons, new_trajectories_snr, None, n_colors=n_colors, c_cbar=c_cbar, fontsize=fontsize, plot_time=plot_time, add_cbar=False)
            
            if VENUS is None:

                n_colors = 10
                fmt = lambda x, pos: '{:.2f} %'.format(x*1e2)
                idx_snr = np.argmin(abs(proba_model.SNR_thresholds-snr))
                x, y, toplot, _, _ = interpolate_2d(m, proba_model.all_lons, proba_model.all_lats, proba_model.proba_all[idx_snr,:,:], dnew=1.)
                cmap_bounds = np.linspace(toplot.min(), toplot.max(), n_colors)
                cmap_p = cm.get_cmap("Reds", lut=len(cmap_bounds))
                norm = mcol.BoundaryNorm(cmap_bounds, cmap_p.N)
                sc_proba = m.pcolormesh(x, y, toplot, zorder=0, cmap=cmap_p, norm=norm)
                m.drawmeridians(np.linspace(-180., 180., 5), labels=[0, 0, 0, 1], fontsize=12)
                m.drawparallels(np.linspace(-90., 90., 5), labels=[1, 0, 0, 0], fontsize=12)
                add_vertical_cbar(fig, ax, sc_proba, cmap_bounds, fmt, c_cbar, 'Hourly probability') 

                n_colors = 7
                fmt = lambda x, pos: '{:.0f}'.format(x)
                unknown = 'wind_direction'
                vmax, vmin = -88, -92
                winds_grp = winds.groupby(['lat', 'lon']).first().reset_index()
                lat_size = winds_grp.lat.unique().size
                lon_size = winds_grp.lon.unique().size
                LON, LAT = np.meshgrid(winds_grp.lon.unique(), winds_grp.lat.unique())
                x, y = m_winds(LON.ravel(), LAT.ravel())
                x, y = x.reshape(lat_size, lon_size), y.reshape(lat_size, lon_size)
                cmap_bounds = np.linspace(vmin, vmax, n_colors)
                cmap_w = cm.get_cmap("Greens", lut=len(cmap_bounds))
                norm = mcol.BoundaryNorm(cmap_bounds, cmap_w.N)
                sc_winds = m_winds.pcolormesh(x, y, winds_grp[unknown].values.reshape(lat_size, lon_size), norm=norm, cmap=cmap_w, alpha=0.8, zorder=5)
                add_vertical_cbar(fig, ax_winds, sc_winds, cmap_bounds, fmt, c_cbar, 'Wind direction')    

                m_winds.drawmeridians(np.linspace(-180., 180., 5), labels=[0, 0, 0, 1], fontsize=12)
                m_winds.drawparallels(np.linspace(-90., 90., 5), labels=[1, 0, 0, 0], fontsize=12)
            
            if plot_volcanoes:
                x, y = m(proba_model.lon_volcanoes, proba_model.lat_volcanoes)
                m.scatter(x, y, marker='x', color='black', s=30)
                m_winds.scatter(x, y, marker='x', color='black', s=30)
            
        isnr = -1
        for snr, new_trajectories_snr in new_trajectories.groupby('snr'):
            isnr += 1
            line, = ax_vs_time.plot(new_trajectories_snr.time/(24*3600.), 1e2*new_trajectories_snr.proba, color=cmap[isnr], label=snr, linestyle=linestyles[iseismicity])
            if iseismicity == 0:
                lines_snr.append(line)
            line, = ax_vs_time.plot(new_trajectories_snr.time/(24*3600.), 1e2*new_trajectories_snr.proba, color=cmap[isnr], label=seismicity, linestyle=linestyles[iseismicity])
            if isnr == 0:
                lines_seismicity.append(line)
            
    ax_vs_lon.plot(new_trajectories_snr.time/(24*3600.), new_trajectories_snr.lon, label='longitude', color='black')
    ax_vs_lon.plot([0., 0.], [0., 0.], label='latitude', color='tab:red')
    ax_vs_lon.legend(loc='upper left', frameon=False, labelcolor=c_cbar, fontsize=fontsize)
    ax_vs_lon.set_ylabel('Longitude', color=c_cbar, fontsize=fontsize)
    ax_vs_lat = ax_vs_lon.twinx()  # instantiate a second Axes that shares the same x-axis
    ax_vs_lat.plot(new_trajectories_snr.time/(24*3600.), new_trajectories_snr.lat, label='latitude', color='tab:red')
    ax_vs_lat.grid(alpha=0.4)
    ax_vs_lon.tick_params(axis='both', colors=c_cbar, labelsize=fontsize)
    ax_vs_lat.tick_params(axis='both', colors=c_cbar, labelsize=fontsize)
    ax_vs_lat.set_ylabel('Latitude', fontsize=fontsize, color='tab:red')
    ax_vs_lat.tick_params(axis='y', labelcolor='tab:red')

    #ax_vs_time.legend(frameon=False, title='SNR')
    ax_vs_time.set_ylabel('Detection probability (%)', color=c_cbar, fontsize=fontsize)
    ax_vs_time.set_xlabel('Time (days)', color=c_cbar, fontsize=fontsize)
    ax_vs_time.set_xlim([0., new_trajectories_snr.time.max()/(24*3600.)])
    ax_vs_time.set_ylim(ylim)
    ax_vs_time.grid(alpha=0.4)
    
    # Creating the first legend
    first_legend = ax_vs_time.legend(handles=lines_snr, loc='upper left', title='SNR', frameon=False, labelcolor=c_cbar, fontsize=fontsize)
    #ax_vs_time.legend(handles=first_legend.legendHandles, labels=[text.get_text() for text in first_legend.get_texts()], loc='upper left', title='SNR')

    # Creating and adding the second legend
    if VENUS is not None:
        second_legend = ax_vs_time.legend(handles=lines_seismicity, loc='upper left', title='Seismicity', bbox_to_anchor=(0.25, 1), frameon=False, labelcolor=c_cbar, fontsize=fontsize)
        #ax_vs_time.legend(handles=second_legend.legendHandles, bbox_to_anchor=(0.5, 1), labels=[text.get_text() for text in second_legend.get_texts()], loc='upper left', title='Seismicity')
        ax_vs_time.add_artist(first_legend)
        ax_vs_time.patch.set_alpha(0.5)
        plt.setp(second_legend.get_title(), color=c_cbar, fontsize=fontsize)
    
    ax_vs_time.tick_params(axis='both', colors=c_cbar, labelsize=fontsize)
    plt.setp(first_legend.get_title(), color=c_cbar, fontsize=fontsize)
    
    fontsize_label = 20.
    ax_vs_lon.text(-0.07, 1., 'a)', fontsize=fontsize_label, ha='right', va='bottom', transform=ax_vs_lon.transAxes)
    ax_winds.text(-0.1, 1., 'b)', fontsize=fontsize_label, ha='right', va='bottom', transform=ax_winds.transAxes)
    ax_vs_time.text(-0.07, 1., 'c)', fontsize=fontsize_label, ha='right', va='bottom', transform=ax_vs_time.transAxes)
    ax.text(-0.1, 1., 'd)', fontsize=fontsize_label, ha='right', va='bottom', transform=ax.transAxes)
    
    fig.align_ylabels() 
    fig.subplots_adjust(wspace=0.15, bottom=0.2, top=0.8)
    fig.patch.set_alpha(0.)
    fig.savefig('./figures/Figure_2_balloon_proba.pdf',)


##########################
if __name__ == '__main__':

    """
    PATH_VENUS_DATA = os.path.join("../../../Venus_data/")
    PATH_VENUS = os.path.join(f"{PATH_VENUS_DATA}tectonic_settings_Venus")
    VENUS = {
        'corona': gpd.read_file(f"{PATH_VENUS}/corona.shp"),
        'rift': gpd.read_file(f"{PATH_VENUS}/rifts.shp"),
        'ridge': gpd.read_file(f"{PATH_VENUS}/ridges.shp")
    }

    ## Below to create surface ratios
    output_file = './test_data_Venus/surface_ratios.csv'
    l_lon = np.arange(-179, -150, 1)
    l_lat = np.arange(-89, 90, 1)
    compute_ratios(VENUS, l_lon, l_lat, output_file, ratio_df=pd.DataFrame())
    """

    file_slopes = '../../../Venus_data/distribution_venus_per_mw.csv'
    pd_slopes = get_slopes(file_slopes)

    file_curve = './test_data_Venus/GF_reverse_fault_1Hz_c15km.csv'
    TL_new, TL_new_qmin, TL_new_qmax = get_TL_curves(file_curve, dist_min = 100., plot=False)

    file_ratio = './test_data_Venus/surface_ratios_fixed.csv'
    surface_ratios = get_surface_ratios(file_ratio)

    dlat = 5.
    r_venus = 6052
    opt_model = dict(
        scenario = 'active_high_min', # Iris' seismicity scenario
        dists = np.arange(10., np.pi*r_venus, 200), # Low discretization will lead to terrible not unit integrals
        M0s = np.linspace(3., 8., 30), # Low discretization will lead to terrible not unit integrals
        SNR_thresholds = np.linspace(0.1, 10., 50),
        noise_level = 5e-2, # noise level in Pa
        duration = 1./(365.*24), # (1/mission_duration)
        all_lats = np.arange(-90., 90.+dlat, dlat),
        all_lons = np.arange(-180, 180+dlat*2, dlat*2),
        homogeneous_ratios = False,
        m_min = 3.,
        r_venus = r_venus,
        
    )

    proba_all_other_high = compute_proba_map(pd_slopes, surface_ratios, TL_new, TL_new_qmin, TL_new_qmax, **opt_model)

    ## Visualization
    # low_cmap, high_cmap = np.arange(1e-2, 5e-2, 5e-3), np.arange(5e-2, 1.25e-1, 1e-2) # 1 hour
    low_cmap, high_cmap = np.arange(5e-1, 1, 1e-1), np.arange(2, 3, 0.25) # 1 day RW
    low_cmap, high_cmap = np.arange(2.5e-1, 1, 1e-1), np.arange(2, 3, 0.25) # 1 day RW low activity
    #low_cmap, high_cmap = np.arange(3e-1, 1, 1e-1), np.arange(1, 3, 0.25) # 1 day body

    plot_map(all_lats, all_lons, proba_all_other_high, SNR_thresholds, VENUS, c_cbar='black', l_snr_to_plot=[1.,5.], n_colors=10, low_cmap=low_cmap, high_cmap=high_cmap,)#low_cmap, high_cmap = np.arange(20, 60, 10), np.arange(60, 80, 10) # 1 day ratio over homogeneous
    #plot_map(all_lats, all_lons, proba_all_other, SNR_thresholds, duration, VENUS, l_snr_to_plot=[1.,5.], n_colors=10, low_cmap=low_cmap, high_cmap=high_cmap, proba_all_homo=proba_all_homo)

