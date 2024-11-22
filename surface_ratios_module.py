import pyproj
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point
from shapely.ops import split
from sklearn.cluster import DBSCAN
from shapely.validation import make_valid
from tqdm import tqdm
from time import time
import numpy as np
import pandas as pd
import geopandas as gpd 
import os
import matplotlib.pyplot as plt
from pdb import set_trace as bp
from functools import partial
from multiprocessing import get_context
from shapely.ops import transform
import pyproj
from shapely.ops import unary_union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.mixture import GaussianMixture

##########################
## PREPROCESSING SHAPES ##
##########################

def load_tectonics_shp(PATH_VENUS_DATA):

    PATH_VENUS = os.path.join(f"{PATH_VENUS_DATA}Venus_data/tectonic_settings_Venus")
    VENUS = {
        'corona': gpd.read_file(f"{PATH_VENUS}/corona.shp"),
        'rift': gpd.read_file(f"{PATH_VENUS}/rifts.shp"),
        'ridge': gpd.read_file(f"{PATH_VENUS}/ridges.shp")
    }

    return VENUS

# Function to apply the projection to a geometry
def trim_polygons(polygon):
    
    antimeridian = LineString([(-180, -90), (-180, 90)])
    meridian = LineString([(180, -90), (180, 90)])
    coords = np.array(polygon.exterior.coords) 
    
    if coords.min() < -180.:
    
        geoms = split(polygon, antimeridian).geoms
        idx_keep_antimeridian = 0
        l_antimeridian = len(geoms)
        for igeom, geom in enumerate(geoms):
            coords = np.array(geom.exterior.coords)
            if coords[:,0].min() >= -180.:
                idx_keep_antimeridian = igeom
                
        polygon = geoms[idx_keep_antimeridian]
                
    elif coords.max() > 180.:
            
        geoms = split(polygon, meridian).geoms
        idx_keep_meridian = 0
        l_meridian = len(geoms)
        for igeom, geom in enumerate(geoms):
            coords = np.array(geom.exterior.coords)
            if coords[:,0].max() <= 180.:
                idx_keep_meridian = igeom
            
        polygon = geoms[idx_keep_meridian]
    
    return polygon

def split_lines(temp, epsilon=1e-3):

    venus_crs_180 = """
    GEOGCS["GCS_Venus",
        DATUM["D_Venus",
            SPHEROID["S_Venus",6051800,0]],
        PRIMEM["Reference_Meridian",180],
        UNIT["degree",0.0174532925199433],
        AXIS["Longitude",EAST],
        AXIS["Latitude",NORTH]]
    """

    venus_crs = 'GEOGCS["GCS_Venus",DATUM["D_Venus",SPHEROID["S_Venus",6051800,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433],AXIS["Longitude",EAST],AXIS["Latitude",NORTH]]'

    antimeridian = LineString([(0, -90), (0, 90)])

    # Set the custom CRS to the GeoDataFrame
    #temp.set_crs(venus_crs, inplace=True, allow_override=True)
    temp.loc[:, 'geometry'] = temp['geometry'].to_crs(venus_crs_180)
    temp.loc[:, 'geometry'] = temp['geometry'].to_crs(venus_crs) ## Trick to convert from 0 to 360 to -180 to 180
    
    # Select rows with MultiPolygon geometries and explode them
    is_multipolygon = temp['geometry'].apply(lambda geom: isinstance(geom, MultiPolygon))
    temp_new = temp[~is_multipolygon]
    if temp[is_multipolygon].shape[0] > 0:
        temp_new = pd.concat([temp_new, temp[is_multipolygon].explode(index_parts=False).reset_index(drop=True)])
    temp_new.reset_index(inplace=True, drop=True)
    #temp_new.loc[:, 'geometry'] = temp_new['geometry'].to_crs(venus_crs_180)
    temp_new = temp_new[['geometry']]

    for ii, geo in temp_new.iterrows():

        it_is_a_polygon = False
        if geo.geometry.geom_type == 'Polygon': ## Tectonic settings
            it_is_a_polygon = True
            coords = np.array(geo.geometry.exterior.coords)
        else: ## Wrinkle ridges
            coords = np.array(geo.geometry.coords)

        if abs(np.diff(coords[:,0])).max() > 180:

            test_loc = temp_new.iloc[ii:ii+1]
            test_loc.loc[:, 'geometry'] = test_loc['geometry'].to_crs(venus_crs_180)
            #print(test_loc)
            split_polygons = split(test_loc.geometry.iloc[0], antimeridian)

            geoms = []
            for geom in split_polygons.geoms: # MultiLineString
                if it_is_a_polygon:
                    coords_loc = np.array(geom.exterior.coords)
                    sign = np.sign(coords_loc[np.argmax(abs(coords_loc[:,0])),0])
                    coords_loc[coords_loc[:,0]==0,0] += sign*epsilon
                    geoms.append( Polygon(coords_loc) )
                    
                else:
                    coords_loc = np.array(geom.coords)
                    idx = np.argmax(abs(coords_loc[:,0]))
                    idx_other = np.argmin(abs(coords_loc[:,0]))
                    sign_found = np.sign(coords_loc[idx,0])
                    coords_loc[idx_other,0] = sign_found*epsilon
                    geoms.append( LineString(coords_loc) )

            if it_is_a_polygon:
                geoms = MultiPolygon(geoms)
            else:
                geoms = MultiLineString(geoms)

            test_loc.loc[:,'geometry'] = geoms
            test_loc.set_crs(venus_crs_180, inplace=True, allow_override=True)
            test_loc.loc[:, 'geometry'] = test_loc['geometry'].to_crs(venus_crs)

            temp_new.loc[temp_new.index == ii,'geometry'] = test_loc.geometry
            
    return temp_new.explode(ignore_index=True).reset_index(drop=True)

def split_lines_deprecated(temp, epsilon=1e-3):

    venus_crs_180 = """
    GEOGCS["GCS_Venus",
        DATUM["D_Venus",
            SPHEROID["S_Venus",6051800,0]],
        PRIMEM["Reference_Meridian",180],
        UNIT["degree",0.0174532925199433],
        AXIS["Longitude",EAST],
        AXIS["Latitude",NORTH]]
    """

    venus_crs = 'GEOGCS["GCS_Venus",DATUM["D_Venus",SPHEROID["S_Venus",6051800,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433],AXIS["Longitude",EAST],AXIS["Latitude",NORTH]]'

    antimeridian = LineString([(0, -90), (0, 90)])
    #antimeridian = LineString([(180, -90), (180, 90)])
    
    # Select rows with MultiPolygon geometries and explode them
    is_multipolygon = temp['geometry'].apply(lambda geom: isinstance(geom, MultiPolygon))
    temp_new = temp[~is_multipolygon]
    if temp[is_multipolygon].shape[0] > 0:
        temp_new = pd.concat([temp_new, temp[is_multipolygon].explode(index_parts=False).reset_index(drop=True)])
    temp_new.reset_index(inplace=True, drop=True)
    temp_new.loc[:, 'geometry'] = temp_new['geometry'].to_crs(venus_crs_180)
    temp_new = temp_new[['geometry']]

    for ii, geo in temp_new.iterrows():

        it_is_a_polygon = False
        if geo.geometry.geom_type == 'Polygon': ## Tectonic settings
            it_is_a_polygon = True
            coords = np.array(geo.geometry.exterior.coords)
            #print(coords)
        else: ## Wrinkle ridges
            coords = np.array(geo.geometry.coords)

        #if abs(np.diff(coords[:,0])[0]) > 180:
        if abs(np.diff(coords[:,0])).max() > 180:
            #print(coords)
            #test_loc = temp_new.iloc[ii:ii+1].geometry.to_crs(venus_crs_180).iloc[0]
            test_loc = temp_new.iloc[ii:ii+1].geometry.iloc[0]
            split_polygons = split(test_loc, antimeridian)

            geoms = []
            for geom in split_polygons.geoms: # MultiLineString
                if it_is_a_polygon:
                    coords_loc = np.array(geom.exterior.coords)
                    idx_middle = np.where(abs(coords_loc[:,0])>epsilon)[0]
                    geoms.append( Polygon(coords_loc[idx_middle,:]) )
                    
                else:
                    coords_loc = np.array(geom.coords)
                    idx = np.argmax(abs(coords_loc[:,0]))
                    idx_other = np.argmin(abs(coords_loc[:,0]))
                    sign_found = np.sign(coords_loc[idx,0])
                    coords_loc[idx_other,0] = sign_found*epsilon
                    geoms.append( LineString(coords_loc) )

            if it_is_a_polygon:
                geoms = MultiPolygon(geoms)
            else:
                geoms = MultiLineString(geoms)
            
            #temp_loc = temp_new.loc[temp_new.index.isin([ii])].copy().reset_index(drop=True)
            #temp_loc.loc[:,'geometry'] = geoms

            temp_new.loc[temp_new.index.isin([ii]),'geometry'] = geoms
            #temp_new.loc[temp_new.index.isin([ii]),'geometry'] = temp_new.loc[temp_new.index.isin([ii]),'geometry'].to_crs(venus_crs)
            
    temp_new.loc[:, 'geometry'] = temp_new['geometry'].to_crs(venus_crs) ## Very important to convert the coordinates back into the right coordinate system

    return temp_new.explode(ignore_index=True).reset_index(drop=True)

# Function to apply the projection to a geometry
def project_geometry(geometry, diff_x, project):

    coords_init = np.array(geometry.exterior.coords)
    poly = transform(project, geometry)
    coords = np.array(poly.exterior.coords)
    if ~poly.is_valid:
        poly = make_valid(poly)
        
    if coords_init[:,0].min() < -180. or coords_init[:,0].max() > 180.:
        print('less than -180', coords_init[:,0].min(), coords_init[:,0].max())
        
    diff_poly = abs(np.diff(coords[:,0])).max()
    if diff_poly > diff_x:
        print(f'diff too large {diff_poly:.2f} > {diff_x:.2f}')
        print(coords_init[:,0])
        print(coords_init[:,1])
        
    return poly

def preprocess_one_region(wrinkle_ridges, epsilon, lat_0, lon_0, R0, proj=None):

    # Define the custom CRS using the WKT string
    venus_crs = """
    GEOGCS["GCS_Venus",
        DATUM["D_Venus",
            SPHEROID["S_Venus",6051800,0]],
        PRIMEM["Reference_Meridian",0],
        UNIT["degree",0.0174532925199433],
        AXIS["Longitude",EAST],
        AXIS["Latitude",NORTH]]
    """

    # Set the custom CRS to the GeoDataFrame
    wrinkle_ridges.set_crs(venus_crs, inplace=True)

    print('Split linestrings crossing antimeridian')
    wrinkle_ridges = split_lines(wrinkle_ridges.copy())
    
    print('Buffering linestrings into polygons')
    wrinkle_ridges['geometry'] = wrinkle_ridges.geometry.buffer(epsilon)
    wrinkle_ridges = wrinkle_ridges.explode(ignore_index=True).reset_index(drop=True)

    # Apply the projection to each geometry in the GeoDataFrame
    print('Trim polygons with values beyond -180, 180')
    temp = wrinkle_ridges['geometry'].apply(lambda geom: trim_polygons(geom))

    print('Project polygons to 2d map')
    if proj == None:
        proj = pyproj.Proj(proj='robin', lat_0=lat_0, lon_0=lon_0, a=R0, b=R0)
    project = pyproj.Transformer.from_proj(pyproj.Proj(proj='latlong'), proj, always_xy=True).transform
    lons = np.linspace(-180., 180., 100000)
    x, _ = proj(lons, np.zeros_like(lons))
    diff_x = abs(x.max()-x.min())
    temp2 = temp.apply(lambda geom: project_geometry(geom, diff_x, project),)

    print('Converting list of polygons into multipolygon')
    unioned_linestring = unary_union(temp2.geometry)

    return unioned_linestring

def load_tectonic_iris_shp(PATH_VENUS_DATA, polygon_map, epsilon=1e-1, lat_0=-89., lon_0=0., R0=6052000, find_active_corona_only=False):

    proj = pyproj.Proj(proj='robin', lat_0=lat_0, lon_0=lon_0, a=R0, b=R0)
    PATH_VENUS = os.path.join(f"{PATH_VENUS_DATA}Venus_data/tectonic_settings_Venus")
    VENUS = {
        'corona': gpd.read_file(f"{PATH_VENUS}/corona.shp"),
        'rift': gpd.read_file(f"{PATH_VENUS}/rifts.shp"),
        'ridge': gpd.read_file(f"{PATH_VENUS}/ridges.shp"),
    }

    l_regions = dict()
    for region, region_shp in VENUS.items():
        print(f'Processing {region}')
        l_regions[region] = preprocess_one_region(region_shp, epsilon, lat_0, lon_0, R0, proj=proj)
        #plt.figure(); [plt.plot(np.array(list(geo.exterior.coords))[:,0], np.array(list(geo.exterior.coords))[:,1]) for geo in l_regions[region].geoms]; plt.savefig(f'./test_active_{region}.png')

    if find_active_corona_only:
        active_corona = gpd.read_file(f"{PATH_VENUS}/../../active_corona_shape/active_corona.shp")
        active_corona = preprocess_one_region(active_corona, epsilon, lat_0, lon_0, R0, proj=proj)
        #plt.figure(); [plt.scatter(np.array(list(geo.exterior.coords))[:,0], np.array(list(geo.exterior.coords))[:,1]) for geo in l_regions['corona'].geoms]; plt.savefig('./test0.png')
        #plt.figure(); [plt.plot(np.array(list(geo.exterior.coords))[:,0], np.array(list(geo.exterior.coords))[:,1]) for geo in active_corona.geoms]; plt.savefig('./test_active_corona_anna.png')
        #bp()
        l_regions['corona'] = l_regions['corona'].intersection(active_corona)
        #plt.figure(); [plt.scatter(np.array(list(geo.exterior.coords))[:,0], np.array(list(geo.exterior.coords))[:,1]) for geo in l_regions['corona'].geoms]; plt.savefig('./test.png')
        
    l_regions['intraplate'] = compute_region_intraplate(l_regions, proj)

    return l_regions

def load_wrinkles_shp(PATH_VENUS_DATA, epsilon=1e-1, lat_0=-89., lon_0=0., R0=6052000):

    print('Loading shape files')
    PATH_VENUS = os.path.join(f"{PATH_VENUS_DATA}wrinkle_ridges")
    """
    VENUS = {
        'wrinkle_ridges': gpd.read_file(f"{PATH_VENUS}/wrinkle_ridges.shp"),
    }
    """
    wrinkle_ridges = gpd.read_file(f"{PATH_VENUS}/wrinkle_ridges.shp")
    unioned_linestring = preprocess_one_region(wrinkle_ridges, epsilon, lat_0, lon_0, R0)
    
    return dict(wringkles=unioned_linestring)

######################
## COMPUTING RATIOS ##
######################

from scipy import interpolate

def resample_trajectory(proj, shape_init, lon_0, lon_loc, lon_traj, lat_traj, num_points, threshold = 4e6, verbose=False):
    
    n_loc, n_radius = shape_init[1:]
    lon_traj_r, lat_traj_r = lon_traj.reshape(shape_init), lat_traj.reshape(shape_init)
    #lon_loc_r = lon_loc.reshape(shape_init)
    
    n_subshapes = []
    new_coords_lat = np.zeros((num_points,)+lon_traj_r.shape[1:])
    new_coords_lon = np.zeros((num_points,)+lon_traj_r.shape[1:])

    if verbose:
        print('- resampling TL boundaries')

    for iloc in tqdm(range(n_loc), disable=not verbose):
        
        """
        lon_0_current = lon_loc_r[0,iloc,0]
        proj_sensor = proj(lon_0_current, 0.)
        proj_sensor_ref = proj(lon_0, 0.)
        offset = abs(proj_sensor_ref[0]-proj_sensor[0])
        """

        for iradius in range(n_radius):

            trajectory = np.c_[lon_traj_r[:,iloc,iradius], lat_traj_r[:,iloc,iradius]]
            
            #ind = -np.argmax(abs(trajectory[:,0]-offset))-1
            #trajectory = np.roll(trajectory, ind, axis=0)
            
            # Unpack the coordinates
            x, y = trajectory[:,0], trajectory[:,1]

            #plt.figure()
            #plt.scatter(x, y, c=np.arange(x.size))

            # Compute the cumulative distance along the trajectory
            distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
            diff_distance = np.diff(distance)

            num_shapes = 1
            isep = distance.size
            
            loc_maxs = []
            #print(f'- max distance: {np.max(np.diff(distance))} (ratio: {np.max(np.diff(distance))/threshold})')
            if np.max(diff_distance) > threshold:
                loc_maxs = np.where(diff_distance>threshold)[0]
                dist_begin_end = np.sqrt((x[0]-x[-1])**2+(y[0]-y[-1])**2)
                # Check if there is two shapes
                # Either two large distances between neighboring points or beginning and end of trajectory not close to each other
                if (len(loc_maxs) > 1) or ((len(loc_maxs) == 1) and (dist_begin_end > threshold)): 
                    num_shapes += 1
                    isep = np.argmax(diff_distance)+1

                if loc_maxs.size > 1:
                    ind = trajectory.size - np.max(loc_maxs) - 1
                    trajectory = np.roll(trajectory, ind, axis=0)
                    x, y = zip(*trajectory)
                    x, y = x[:-1], y[:-1]
                    distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
                    isep = np.argmax(np.diff(distance))+1
                
            n_subshapes.append(num_shapes)

            #plt.figure()
            for ishape in range(num_shapes):
                
                # Unpack the coordinates
                if ishape == 0:
                    x, y = trajectory[:isep,0], trajectory[:isep,1]
                else:
                    x, y = trajectory[isep:,0], trajectory[isep:,1]

                ind = 0
                distance = np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2)
                diff_distance_bool = distance.max() > threshold
                #print(diff_distance_bool)
                
                #plt.title(iradius)
                if diff_distance_bool:

                    ind = -x.argmax()
                    x = np.roll(x, ind)
                    y = np.roll(y, ind)
                    distance = np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2)
                    diff_distance_bool = distance.max() > threshold
                    i = 0
                    while diff_distance_bool:
                        i += 1
                        x = np.roll(x, -1)
                        y = np.roll(y, -1)
                        #distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
                        distance = np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2)
                        diff_distance_bool = distance.max() > threshold
                        #print(distance.max(), diff_distance_bool)
                        #print((np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2)).max())
                        diff_distance_bool = np.diff(distance).max() > threshold
                        #print(np.diff(distance).max())
                        #plt.plot(x, y, label=i, linewidth=5-i)
                        #plt.plot(x, y, label=f'{i}: {np.diff(distance).max():.2f}', linewidth=(5-i)**4, edgecolor='black')
                        #if i > 19:#x.size+1:
                        #    break
                #plt.legend()
                #plt.scatter(x, y, c=np.arange(x.size))
                #plt.scatter(x, y, c=np.arange(x.size), s=5)

                #continue

                #plt.plot(x, y, )
            
                # Compute the cumulative distance along the trajectory
                #distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
                distance = np.cumsum(distance)
                #print(distance.shape, x.shape)

                # Normalize the distance values to be between 0 and 1
                distance = distance / distance[-1]
                
                # Perform cubic spline interpolation
                spline, u = interpolate.splprep([x, y], s=0, u=distance)

                # Generate the normalized distances of the new points
                num_points_shape = num_points//num_shapes
                u_new = np.linspace(0, 1, num_points_shape)

                # Compute the new coordinates
                new_coords_loc = interpolate.splev(u_new, spline)
                new_coords_lon[num_points_shape*ishape:num_points_shape*(ishape+1),iloc,iradius] = new_coords_loc[0]
                new_coords_lat[num_points_shape*ishape:num_points_shape*(ishape+1),iloc,iradius] = new_coords_loc[1]
        
    return new_coords_lon, new_coords_lat, n_subshapes
        
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
        #print('lines', lines.geom_type)
        #print('polygon2', polygon2.geom_type)
        diff_map = polygon_map.difference(make_valid(lines))
        #print('diff_map', diff_map.geom_type)
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
            
            #print('diff_map2', diff_map.geom_type)
        
            diff_map = make_valid(diff_map)
            #print('diff_map3', diff_map.geom_type)
            if diff_map.geom_type == 'MultiPolygon':
                for ipoly, sub_poly in enumerate(diff_map.geoms):
                    #if ipoly == ipolymax:
                    #    continue 
                    if sub_poly.geom_type == 'Polygon':
                        #print('sub_poly', sub_poly.buffer(10))
                        polygon2 = polygon2.union( sub_poly.buffer(10) )
                    #exts.append(iext)
            else:
                #print('diff_map3 before polygon2', polygon2.geom_type)
                #bef = polygon2.geom_type
                #polygon2_bef = polygon2
                polygon2 = polygon2.union( diff_map.buffer(10) )
                #aft = polygon2.geom_type
                #if bef == 'MultiPolygon' and aft == 'GeometryCollection':
                #    return polygon2, polygon2_bef, diff_map
                #print('diff_map3 after polygon2', polygon2.geom_type)
                
        else:
            polygon2 = polygon2.union(lines)
                
    polygon2 = make_valid(polygon2)
    return polygon2

def compute_whole_map_region(lon_0, proj, dlon=10., dlat=10., apply_proj=True): 

    left_lon  = -180.
    right_lon = 180.
    bottom_lat = -90.#+lat_0 if lat_0 > 0 else 90+lat_0
    top_lat = 90.#+lat_0 if lat_0 < 0 else -90+lat_0
    
    if apply_proj:
        coords_map = [proj(lon, bottom_lat) for lon in np.arange(left_lon, right_lon+dlon, dlon)]
        coords_map += [proj(right_lon, lat) for lat in np.arange(bottom_lat, top_lat+dlat, dlat)]
        coords_map += [proj(lon, top_lat) for lon in np.arange(right_lon, left_lon-dlon, -dlon)]
        coords_map += [proj(left_lon, lat) for lat in np.arange(top_lat, bottom_lat-dlat, -dlat)]
    else:
        coords_map = [(lon, bottom_lat) for lon in np.arange(left_lon, right_lon+dlon, dlon)]
        coords_map += [(right_lon, lat) for lat in np.arange(bottom_lat, top_lat+dlat, dlat)]
        coords_map += [(lon, top_lat) for lon in np.arange(right_lon, left_lon-dlon, -dlon)]
        coords_map += [(left_lon, lat) for lat in np.arange(top_lat, bottom_lat-dlat, -dlat)]

    polygon_map = Polygon(coords_map)
    return polygon_map, coords_map



def compute_TL_region_v2(polygon_map, coords_poly1, proj, lon_0, lon_0_current, sensor, n_subshape, subsample_db=1, buffer_line=300000, buffer_sensor=180000, n_init_max=15, threshold_neighbor_pts = 5e5, random_state=1):
    
    #proj_sensor = proj(lon_0_current, 0.)
    #proj_sensor_ref = proj(lon_0, 0.)
    #offset = abs(proj_sensor_ref[0]-proj_sensor[0])

    #coords_poly1 = np.roll(coords_poly1, -np.argmax(abs(coords_poly1[:,0]-offset))-1, axis=0)
    last_area = 0
    
    #plt.figure()
    #plt.title(n_subshape)
    
    
    scaler = MinMaxScaler()
    coords_poly1_fixed = scaler.fit_transform(coords_poly1)
    
    diff_poly_return = None
    
    polygon1_return = None
    for iin, n_components in enumerate([n_subshape]):
        
        l_n_init = np.arange(1,n_init_max)
        l_covariances = ['full', 'spherical']
        NINIT, COV = np.meshgrid(l_n_init, l_covariances)
        NINIT, COV = NINIT.ravel(), COV.ravel()
        for n_init, cov in zip(NINIT, COV):

            #print(n_init, cov)
            clustering = GaussianMixture(n_components=n_components, n_init=n_init, covariance_type=cov, random_state=random_state).fit_predict(coords_poly1_fixed)
            polygon1 = Polygon()
            max_dx = 0.
            #plt.figure()
            for label in np.unique(clustering):
                max_dx = max(max_dx, abs(np.diff(coords_poly1[clustering==label][:-1,0])).max())
                #print(max_dx, max_dx > threshold_neighbor_pts)
                #plt.plot(coords_poly1[clustering==label][:-1,0], coords_poly1[clustering==label][:-1,1])
                polygon1 = polygon1.union( LineString(coords_poly1[clustering==label][:-1]).buffer(buffer_line) )
                #plt.plot(coords_poly1[clustering==label][:-1,0], coords_poly1[clustering==label][:-1,1])
            
            if max_dx > threshold_neighbor_pts: # Check for potential issues with clustering, i.e., shared points between shapes
                continue

            diff_poly = polygon_map.difference(polygon1.buffer(10)) # The buffer here is to avoid errors like TopologyException: Input geom 1 is invalid: Self-intersection

            if diff_poly.geom_type == 'MultiPolygon':
        
                areas = []
                idx_sensor = []
                for poly in diff_poly.geoms:
                    areas.append(poly.area)
                    #idx = 1 if poly.contains(sensor) else 0
                    idx_sensor.append(poly.buffer(buffer_sensor).contains(sensor))
                areas = np.array(areas)
                idx_sensor = np.array(idx_sensor)

                if np.where(idx_sensor)[0].size == 0: # Check for potential issues with clustering, i.e., sensor not in any of the shapes
                    continue
                    
                if np.where(idx_sensor)[0].size > 1: # If multiple shapes contain the sensors (because of the buffer) we select the smallest shape
                    idx_min = np.argmin(areas)
                    for idx in range(len(idx_sensor)):
                        if not idx == idx_min:
                            idx_sensor[idx] = False

                if np.where(idx_sensor)[0].size == 0:
                    continue

                #scenario = 0
                idx_other = -1
                if len(areas) > 2:
                    if areas[np.where(idx_sensor)[0][0]] < np.max(areas):
                        #scenario = 1 # two small areas with one of them containing the sensor
                        idx_other = np.where(~idx_sensor)[0]
                        idx_other = idx_other[np.argmin(areas[idx_other])]

                polygon1 = MultiPolygon()
                cpt_area = -1
                for poly in diff_poly.geoms:
                    cpt_area += 1

                    if idx_sensor[cpt_area] or (idx_other == cpt_area):
                        polygon1 = polygon1.union(poly)

            else:
                if not polygon1.contains(sensor):
                    polygon1 = diff_poly

            if (polygon1.area-last_area>1e11) or (iin == 0):
                last_area = polygon1.area
                polygon1_return = polygon1
                diff_poly_return = diff_poly
                
            break

    return polygon1_return, diff_poly_return

def compute_surface_area_ratio(intersection, polygon_map):
    return intersection.area/polygon_map.area

def compute_region_intraplate(l_regions, proj):

    polygon_map, _ = compute_whole_map_region(0., proj, dlon=1., dlat=1., apply_proj=True)

    #if 'intraplate' in VENUS:
    #    return VENUS['intraplate']

    polygon2 = Polygon()
    for region, poly in l_regions.items():
        #polygon2 = polygon2.union( compute_region_venus(polygon_map, VENUS[region].exterior, proj, show_bar=False) )
        polygon2 = polygon2.union( poly )

    """
    plt.figure(); [plt.plot(np.array(geo.exterior.coords)[:,0], np.array(geo.exterior.coords)[:,1]) for geo in polygon2.geoms[:1].geoms]; plt.plot(np.array(polygon_map.exterior.coords)[:,0], np.array(polygon_map.exterior.coords)[:,1]);
    plt.savefig('./test_map.png')
    bp()

    plt.figure(); plt.plot(np.array(polygon_map.exterior.coords)[:,0], np.array(polygon_map.exterior.coords)[:,1]); plt.savefig('./test_map2.png')
    plt.figure(); [plt.plot(np.array(geo.exterior.coords)[:,0], np.array(geo.exterior.coords)[:,1]) for geo in polygon_map.difference(polygon2).geoms]; [plt.plot(np.array(geo.interiors[0].coords)[:,0], np.array(geo.interiors[0].coords)[:,1]) for geo in polygon_map.difference(polygon2).geoms if len(geo.interiors) > 0]; plt.savefig('./test_map3.png')

    plt.figure(); [plt.plot(np.array(geo.exterior.coords)[:,0], np.array(geo.exterior.coords)[:,1]) for geo in polygon2.geoms]; [plt.plot(np.array(geo.interiors[0].coords)[:,0], np.array(geo.interiors[0].coords)[:,1], linestyle=':') for geo in polygon2.geoms if len(geo.interiors) > 0]; plt.savefig('./test_map4.png')
    

    polygon_ = polygon_map.difference(polygon2)
    """

    polygon2 = polygon_map.difference(polygon2)

    return polygon2

def compute_map(lat_0=-89., lon_0=0., R0=6052000):
    proj = pyproj.Proj(proj='robin', lat_0=lat_0, lon_0=lon_0, a=R0, b=R0)
    polygon_map, _ = compute_whole_map_region(lon_0, proj, dlon=1., dlat=1)
    return polygon_map

def compute_map_and_TL(folder_TL_data, lat_0=-89., lon_0=0., R0=6052000):
#def compute_map_and_TL(LONS, LATS, l_points, l_radius, num_points=2000, lat_0=-89., lon_0=0., R0=6052000):

    #proj_init = pyproj.Geod(proj='robin', lat_0=0., lon_0=0., a=R0, b=R0)  
    proj = pyproj.Proj(proj='robin', lat_0=lat_0, lon_0=lon_0, a=R0, b=R0)
    #surface1_lon, surface1_lat, n_subshapes, _, _ = spherical_cap_boundary_v3(proj_init, proj, l_points[0][0], LATS, LONS, l_radius, num_points=num_points)
    surface1_lon = np.load(f'{folder_TL_data}coords_lon.npy')
    surface1_lat = np.load(f'{folder_TL_data}coords_lat.npy')
    n_subshapes = np.load(f'{folder_TL_data}n_subshapes.npy')
    polygon_map, _ = compute_whole_map_region(lon_0, proj, dlon=1., dlat=1)

    return polygon_map, surface1_lon, surface1_lat, n_subshapes

def compute_surface_ratios_wrinkles(lon_0, l_radius, proj, polygon_map, polygon_region, subsample_db, buffer_line, threshold_neighbor_pts, random_state, gdf, input):

    use_gdf = False
    periods = [0.]
    if gdf is not None:
        threshold_acceptable =  np.diff(l_radius)[0]/2.
        use_period = False
        if 'period' in gdf.columns:
            use_period = True
            periods = gdf.periods.unique()
        use_gdf = True

    ## polygon2 is a multipolygon
    i_cpu, l_points, surface1_lon, surface1_lat, n_subshapes = input ## TODO: When using geodataframe no need to pass surface1_lon, surface1_lat, n_subshapes
    opt_TL = dict(subsample_db=subsample_db, buffer_line=buffer_line, threshold_neighbor_pts=threshold_neighbor_pts, random_state=random_state)

    if i_cpu == 0:
        print('- Looping over all sensor locations')

    ratio_df = pd.DataFrame()
    for iloc, (lon_0_current, lat_0_current) in tqdm(enumerate(l_points), total=l_points.shape[0], disable=not i_cpu==0):
    
        for period in periods:

            sensor = Point(proj(lon_0_current, lat_0_current))

            for iradius, radius in enumerate(l_radius):

                if use_gdf: ## We already computed the shapes
                    diff_lat = abs(gdf.lon_statio-lon_0_current)
                    diff_lon = abs(gdf.lat_statio-lat_0_current)
                    diff_dist = abs(gdf['distance']*1e3-radius)
                    if diff_dist.min() > threshold_acceptable: ## In case where there is no distance close to radius, i.e., no stations close enough to the source for a network for example we give a zero radius
                        loc_dict = {'iloc': iloc, 'lon': lon_0_current, 'lat': lat_0_current, 'iradius': iradius, 'radius': radius, 'ratio': 0., 'ratio_map': 0.}
                        ratio_df = pd.concat([ratio_df, pd.DataFrame([loc_dict])])
                        continue
                    else:
                        poly = gdf.loc[(diff_lat==diff_lat.min())&(diff_lon==diff_lon.min())&(diff_dist==diff_dist.min())]
                        if use_period:
                            diff_t = abs(gdf.period-period)
                            poly = poly.loc[diff_t==diff_t.min()]
                        poly = poly.geometry.iloc[0]
                else:
                    trajectory = np.c_[surface1_lon[:,iloc,iradius], surface1_lat[:,iloc,iradius]]
                    poly, _ = compute_TL_region_v2(polygon_map, trajectory, proj, lon_0, lon_0_current, sensor, n_subshapes[iloc,iradius], **opt_TL)

                if poly is None:
                    print('Error --> ', iradius, iloc)
                    break

                intersection = polygon_region.intersection(poly)
                
                ratio = compute_surface_area_ratio(intersection, polygon_map)
                ratio_map = compute_surface_area_ratio(intersection, polygon_region)
                loc_dict = {'iloc': iloc, 'lon': lon_0_current, 'lat': lat_0_current, 'iradius': iradius, 'radius': radius, 'ratio': ratio, 'ratio_map': ratio_map, 'period': period}
                ratio_df = pd.concat([ratio_df, pd.DataFrame([loc_dict])])

    ratio_df.reset_index(drop=True, inplace=True)
    return ratio_df

def compute_surface_ratios_wrinkles_across_CPU(lon_0, l_radius, proj, polygon_map, polygon_region, subsample_db, buffer_line, threshold_neighbor_pts, random_state, l_points, n_subshapes=None, surface1_lon=None, surface1_lat=None, gdf=None, nb_CPU=12):

    if n_subshapes is None and gdf is None:
        print(f'Either provide shape file or TL subshapes')
        return

    if n_subshapes is not None:
        n_subshapes = n_subshapes.reshape(surface1_lon.shape[1:])
    nb_chunks = l_points.shape[0]
    partial_compute_scores = partial(compute_surface_ratios_wrinkles, lon_0, l_radius, proj, polygon_map, polygon_region, subsample_db, buffer_line, threshold_neighbor_pts, random_state, gdf)
        
    N = min(nb_CPU, nb_chunks)
    ## If one CPU requested, no need for deployment
    if N == 1:
        print('Running serial')
        ratio_df = partial_compute_scores( (0, l_points, surface1_lon, surface1_lat, n_subshapes) )

    ## Otherwise, we pool the processes
    else:
    
        step_idx =  nb_chunks//N
        list_of_lists = []
        idxs = []
        for i in range(N):
            idx = np.arange(i*step_idx, (i+1)*step_idx)
            if i == N-1:
                idx = np.arange(i*step_idx, nb_chunks)
            idxs.append(idx)

            input_list = (i, l_points[idx,:], None, None, None)
            if surface1_lon is not None:
                input_list = (i, l_points[idx,:], surface1_lon[:, idx,:], surface1_lat[:, idx,:], n_subshapes[idx,:])
            list_of_lists.append( input_list )

        with get_context("spawn").Pool(processes = N) as p:
            print(f'Running across {N} CPU')
            results = p.map(partial_compute_scores, list_of_lists)
            p.close()
            p.join()

        ratio_df = pd.DataFrame()
        for idx, result in zip(idxs, results):
            ratio_df = pd.concat([ratio_df, result])

    iloc = -1
    for _, group in ratio_df.groupby(['lon', 'lat']):
        iloc += 1
        ratio_df.loc[ratio_df.index.isin(group.index), 'iloc'] = iloc

    ratio_df.reset_index(drop=True, inplace=True)
    return ratio_df

def compute_coordinates_TL_one_cluster_network(l_radius, num_points, lon_0, lat_0, R0, threshold, s_cluster, input):

    icpu, l_points = input

    LONS, LATS = l_points[:,0], l_points[:,1]
    num_points_init = num_points*10
    num_points_init = num_points*10
    l_angles = np.linspace(0., 359.9999, num_points_init)
    n_loc = LATS.size
    n_radius = len(l_radius)
    n_angles = len(l_angles)

    proj_init = pyproj.Geod(proj='robin', lat_0=0., lon_0=lon_0, a=R0, b=R0)  
    proj = pyproj.Proj(proj='robin', lat_0=lat_0, lon_0=lon_0, a=R0, b=R0)

    n_clusters = int(np.ceil(n_loc/s_cluster))
    l_cluster = np.arange(s_cluster)
    iloc, iloc_quake, R = np.meshgrid(l_cluster, l_loc_quakes, l_radius)
    shape_init = iloc.shape
    angles, R, iloc = angles.ravel(), R.ravel(), iloc.ravel()

    lats_stations, lons_stations = np.array([-30, -18, 20]), np.array([20, 25, -80])
    ID_quake_wstat, ID_wstat = np.meshgrid(np.arange(iloc.size), np.arange(lats_stations.size))
    shape_wstat_init = ID_quake_wstat.shape
    ID_quake_wstat, ID_wstat = ID_quake_wstat.ravel(), ID_wstat.ravel()
        
    #print('Finding coordinates TLs for each radius and location')
    coords_lon = np.zeros((num_points, n_loc, n_radius))
    coords_lat = np.zeros((num_points, n_loc, n_radius))
    n_subshapes = []
    for icluster in tqdm(range(n_clusters), disable=not icpu==0):
        
        iloc_loc = iloc + icluster*s_cluster
        iloc_loc_3d = l_cluster + icluster*s_cluster
        if icluster == n_clusters-1:
            ind_valid = iloc_loc<n_loc
            lon, lat = LONS[iloc_loc[ind_valid]], LATS[iloc_loc[ind_valid]]
            angles = angles[ind_valid]
            R = R[ind_valid]
            iloc_loc = iloc_loc[ind_valid]
            iloc_loc_3d = iloc_loc_3d[iloc_loc_3d<n_loc]
            shape_init = (n_angles, iloc_loc_3d.size, n_radius)
        else:
            lon, lat = LONS[iloc_loc], LATS[iloc_loc]

        _, _, max_dist = g.inv(lon[ID_quake_wstat], lat[ID_quake_wstat], lons_stations[ID_wstat], lats_stations[ID_wstat])
        max_dist = max_dist.reshape(shape_wstat_init).max(axis=0)

        endlon_deg, endlat_deg, _ = proj_init.fwd(lon, lat, angles, R)
        endlon_deg, endlat_deg = proj(endlon_deg, endlat_deg)
        
        coords_lon[:,iloc_loc_3d,:], coords_lat[:,iloc_loc_3d,:], n_subshapes_loc = resample_trajectory(proj, shape_init, lon_0, lon, endlon_deg, endlat_deg, num_points, threshold=threshold)
        n_subshapes += n_subshapes_loc

    return coords_lon, coords_lat, n_subshapes

def compute_coordinates_TL_one_cluster(l_radius, num_points, lon_0, lat_0, R0, threshold, s_cluster, input):

    icpu, l_points = input

    LONS, LATS = l_points[:,0], l_points[:,1]
    num_points_init = num_points*10
    num_points_init = num_points*10
    l_angles = np.linspace(0., 359.9999, num_points_init)
    n_loc = LATS.size
    n_radius = len(l_radius)
    n_angles = len(l_angles)

    proj_init = pyproj.Geod(proj='robin', lat_0=0., lon_0=lon_0, a=R0, b=R0)  
    proj = pyproj.Proj(proj='robin', lat_0=lat_0, lon_0=lon_0, a=R0, b=R0)

    n_clusters = int(np.ceil(n_loc/s_cluster))
    l_cluster = np.arange(s_cluster)
    iloc, angles, R = np.meshgrid(l_cluster, l_angles, l_radius)
    shape_init = iloc.shape
    angles, R, iloc = angles.ravel(), R.ravel(), iloc.ravel()

    #print('Finding coordinates TLs for each radius and location')
    coords_lon = np.zeros((num_points, n_loc, n_radius))
    coords_lat = np.zeros((num_points, n_loc, n_radius))
    n_subshapes = []
    for icluster in tqdm(range(n_clusters), disable=not icpu==0):
        
        iloc_loc = iloc + icluster*s_cluster
        iloc_loc_3d = l_cluster + icluster*s_cluster
        if icluster == n_clusters-1:
            ind_valid = iloc_loc<n_loc
            lon, lat = LONS[iloc_loc[ind_valid]], LATS[iloc_loc[ind_valid]]
            angles = angles[ind_valid]
            R = R[ind_valid]
            iloc_loc = iloc_loc[ind_valid]
            iloc_loc_3d = iloc_loc_3d[iloc_loc_3d<n_loc]
            shape_init = (n_angles, iloc_loc_3d.size, n_radius)
        else:
            lon, lat = LONS[iloc_loc], LATS[iloc_loc]

        endlon_deg, endlat_deg, _ = proj_init.fwd(lon, lat, angles, R)
        endlon_deg, endlat_deg = proj(endlon_deg, endlat_deg)
        
        bp()

        """
        endlon_deg_source, endlat_deg_source = proj(lon, lat)
        plt.figure()
        plt.scatter(endlon_deg, endlat_deg, c=np.arange(num_points_init))
        plt.scatter(endlon_deg_source, endlat_deg_source, marker='*')
        plt.title(f'{endlon_deg.shape} {np.unique(lon)} {np.unique(lat)}')
        """
        
        coords_lon[:,iloc_loc_3d,:], coords_lat[:,iloc_loc_3d,:], n_subshapes_loc = resample_trajectory(proj, shape_init, lon_0, lon, endlon_deg, endlat_deg, num_points, threshold=threshold)
        #plt.scatter(coords_lon[:,0,0], coords_lat[:,0,0], marker='^')
        n_subshapes += n_subshapes_loc

    return coords_lon, coords_lat, n_subshapes

def compute_coordinates_TL_across_CPUs(l_radius, num_points, lon_0, lat_0, R0, threshold, s_cluster, l_points, nb_CPU=12):

    nb_chunks = l_points.shape[0]
    partial_compute_scores = partial(compute_coordinates_TL_one_cluster, l_radius, num_points, lon_0, lat_0, R0, threshold, s_cluster)
        
    N = min(nb_CPU, nb_chunks)
    ## If one CPU requested, no need for deployment
    if N == 1:
        print('Running serial')
        coords_lon, coords_lat, n_subshapes = partial_compute_scores( (0, l_points) )

    ## Otherwise, we pool the processes
    else:
    
        step_idx =  nb_chunks//N
        list_of_lists = []
        idxs = []
        for i in range(N):
            idx = np.arange(i*step_idx, (i+1)*step_idx)
            if i == N-1:
                idx = np.arange(i*step_idx, nb_chunks)
            idxs.append(idx)
            list_of_lists.append( (i, l_points[idx,:]) )

        with get_context("spawn").Pool(processes = N) as p:
            print(f'Running across {N} CPU')
            results = p.map(partial_compute_scores, list_of_lists)
            p.close()
            p.join()

        n_subshapes = []
        coords_lon = np.zeros((num_points, l_points.shape[0], l_radius.size))
        coords_lat = np.zeros((num_points, l_points.shape[0], l_radius.size))
        for idx, result in zip(idxs, results):
            coords_lon[:,idx,:], coords_lat[:,idx,:], n_subshapes_loc = result
            n_subshapes += n_subshapes_loc

    return  coords_lon, coords_lat, n_subshapes

def merge_and_fix_surface_ratio_region(pattern, regions=['corona', 'rift', 'ridge', 'intraplate']):

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

    all_data.to_csv(pattern.format('all'), header=True, index=False)

##########################
if __name__ == '__main__':

    compute_TL_coords = False
    compute_ratios = True

    dlat = 5
    l_lon = np.arange(0, 359, dlat*2)
    l_lat = np.arange(-89, 90, dlat)
    LONS, LATS = np.meshgrid(l_lon, l_lat)
    LONS, LATS = LONS.ravel(), LATS.ravel()
    l_points = np.c_[LONS, LATS]
    #l_points = l_points
    R0 = 6052000
    proj = pyproj.Proj(proj='robin', lat_0=-89., lon_0=0., a=R0, b=R0)
    l_radius = np.arange(10000, np.pi*R0/1.001, 5e5)[:]
    lon_0, lat_0 = 0., -89.

    """
    if compute_TL_coords:
        
        opt_coords = dict(
            l_radius = l_radius, 
            num_points = 1000,
            lon_0=lon_0, 
            lat_0=lat_0, 
            R0=R0, 
            threshold=4e6, 
            s_cluster=2,
            l_points=l_points,
            nb_CPU=1
        )
        coords_lon, coords_lat, n_subshapes = compute_coordinates_TL_across_CPUs(**opt_coords)

        bp()

        with open('./data/TL_data/coords_lon.npy', 'wb') as f: np.save(f, coords_lon)
        with open('./data/TL_data/coords_lat.npy', 'wb') as f: np.save(f, coords_lat)
        with open('./data/TL_data/n_subshapes.npy', 'wb') as f: np.save(f, np.array(n_subshapes))
    """

    if compute_ratios:

        opt_maps = dict(
            folder_TL_data='./data/TL_data/', 
            lat_0=lat_0, 
            lon_0=lon_0, 
            R0=R0
        )
        polygon_map, surface1_lon, surface1_lat, n_subshapes = compute_map_and_TL(**opt_maps)

        PATH_VENUS_DATA = os.path.join("./data/")
        #find_active_corona_only=False
        #l_unioned_linestring = load_wrinkles_shp(PATH_VENUS_DATA, epsilon=0.5)
        find_active_corona_only=True
        l_unioned_linestring = load_tectonic_iris_shp(PATH_VENUS_DATA, polygon_map, epsilon=1e-1, lat_0=-89., lon_0=0., R0=6052000, find_active_corona_only=find_active_corona_only)

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
                #gdf=gpd.read_file(f"./data/airglow_shp/network_SNRnight10.0_SNRday1.shp"),
                nb_CPU=3,
            )
            #opt_surface = dict(lon_0=0.,l_radius=l_radius,proj=proj, polygon_map=polygon_map, polygon_region=unioned_linestring, subsample_db=5, buffer_line=120000,threshold_neighbor_pts=20e6, random_state=0,n_subshapes=n_subshapes, l_points=l_points, surface1_lon=surface1_lon, surface1_lat=surface1_lat,nb_CPU=20)
            ratio_df = compute_surface_ratios_wrinkles_across_CPU(**opt_surface)
            if find_active_corona_only:
                region = f'{region}_active'

            bp()

            #plt.figure(); plt.plot(ratio_df.radius.iloc[:30], ratio_df.ratio_map.iloc[:30]); plt.savefig('./test2.png')
            #ratio_df['region'] = region
            #ratio_df.to_csv(f'./data/surface_ratios/surface_ratios_airglow_{region}.csv', index=False, header=True)

    bp()