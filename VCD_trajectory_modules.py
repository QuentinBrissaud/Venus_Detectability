import pandas as pd
import seaborn as sns
from pyproj import Geod
import numpy as np
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as clr
from matplotlib import patheffects
path_effects = [patheffects.withStroke(linewidth=3, foreground="w")]

def get_winds(file_atmos, altitude):

    all_data = pd.read_csv(file_atmos, header=[0])

    winds = all_data.loc[(abs(all_data.alt-altitude)==abs(all_data.alt-altitude).min())
                        &(all_data['var'].isin(['W-E wind component (m/s)', 'S-N wind component (m/s)']))]
    winds['gid'] = winds.groupby(['lat', 'lon'])['val'].transform('idxmin')
    winds['wind_strength'] = winds.groupby(['lat', 'lon'])['val'].transform(lambda x: np.sqrt(sum(x**2)))
    winds['wind_direction'] = winds.groupby(['lat', 'lon'])['val'].transform(lambda x: np.degrees(np.arctan2(-x.iloc[0], x.iloc[1])))
    winds.sort_values(by=['lat', 'lon'], inplace=True)

    return winds

def get_winds_interpolator(file_atmos, altitude, winds=None):

    if winds is None:
        winds = get_winds(file_atmos, altitude)

    unknown = 'wind_direction'
    winds_grp = winds.groupby(['lat', 'lon']).first().reset_index()
    lats = winds_grp.lat.unique()
    lons = winds_grp.lon.unique()
    wind_field = winds_grp[unknown].values.reshape(lats.size, lons.size)
    wind_direction_interpolator = RectBivariateSpline(lats, lons, wind_field)

    unknown = 'wind_strength'
    winds_grp = winds.groupby(['lat', 'lon']).first().reset_index()
    lats = winds_grp.lat.unique()
    lons = winds_grp.lon.unique()
    wind_field = winds_grp[unknown].values.reshape(lats.size, lons.size)
    wind_strength_interpolator = RectBivariateSpline(lats, lons, wind_field)

    return wind_direction_interpolator, wind_strength_interpolator, winds

def compute_distance(dist_top, dist_right, dist_bottom, dist_left, angle):
    # Convert the angle to radians
    angle_rad = np.radians(angle)

    # Compute the four possible intersection points
    x_right = dist_right
    y_right = dist_top + dist_right * np.tan(angle_rad)

    x_top = dist_left + dist_top / np.tan(angle_rad)
    y_top = dist_top

    x_left = dist_left
    y_left = dist_top - dist_left * np.tan(angle_rad)

    x_bottom = dist_left - dist_bottom / np.tan(angle_rad)
    y_bottom = dist_bottom

    # Check which intersection points are within the rectangle
    intersections = [(x_right, y_right) if 0 <= y_right <= dist_top else None,
                     (x_top, y_top) if 0 <= x_top <= dist_right else None,
                     (x_left, y_left) if 0 <= y_left <= dist_top else None,
                     (x_bottom, y_bottom) if 0 <= x_bottom <= dist_right else None]

    # Remove None values
    intersections = [point for point in intersections if point is not None]

    # Compute the distances to the intersection points
    distances = [np.sqrt(x**2 + y**2) for x, y in intersections]
    
    print(distances, intersections)

    iclosest = np.argmin(distances)
    # Return the minimum distance
    return distances[iclosest], intersections[iclosest]

def wrap_longitude(start_lon, L):
    
    return ((start_lon + L + 180) % 360) - 180

def compute_trajectory_airglow(start_lon, dlon, velocity_imager=2.6, time_max=864000, save_trajectory=False, folder='./data'):

    velocity_imager_lon = velocity_imager*1e-2
    dt = dlon/abs(velocity_imager_lon)
    times = np.arange(0., time_max, dt)

    L = velocity_imager_lon*times
    lons = wrap_longitude(start_lon, L)

    trajectory = pd.DataFrame()
    trajectory['time'] = times
    trajectory['lat'] = 0.
    trajectory['lon'] = lons
    trajectory.reset_index(drop=True, inplace=True)

    #if save_trajectory:
    #    trajectory.to_csv(f'{folder}trajectory_balloon_lat{start_location[0]:.2f}_lon{start_location[1]:.2f}_{time_max/(3600*24)}days.csv', header=True, index=False)
    
    return trajectory

def compute_trajectory(winds, start_location, time_max=864000, save_trajectory=False, folder='./data'):
    
    g = Geod(ellps='WGS84') # Use Clarke 1866 ellipsoid.

    dlat, dlon = winds.lat.diff().max(), winds.lon.diff().max()
    if not start_location:
        start_location = [np.random.uniform(-90., 90., 1)[0], np.random.uniform(-180., 180., 1)[0]]
    locations = []
    times = []
    vels = []
    wind_dir = []

    ## Initial cell
    dists = np.sqrt((winds.lat - start_location[0])**2+(winds.lon - start_location[1])**2)
    data = winds[dists==dists.min()]
    current_cell_location = data[['lat', 'lon']].iloc[0].values

    #istep = 0
    last_time = 0.
    while (last_time <= time_max):# and (istep <= nstep_max):

        #istep += 1
        """
        dists = np.sqrt((winds.lat - start_location[0])**2+(winds.lon - start_location[1])**2)
        data = winds[dists==dists.min()]
        current_cell_location = data[['lat', 'lon']].iloc[0].values
        """

        wind_direction = data.wind_direction.iloc[0]
        wind_strength = data.wind_strength.iloc[0]
        vels.append( wind_strength )
        wind_dir.append( wind_direction )
        locations.append( start_location )
        times.append( last_time )

        _, _, dist_to_edge_top = g.inv(start_location[1], start_location[0], current_cell_location[1], current_cell_location[0] + dlat/2.)
        _, _, dist_to_edge_bottom = g.inv(start_location[1], start_location[0], current_cell_location[1], current_cell_location[0] - dlat/2.)
        _, _, dist_to_edge_right = g.inv(start_location[1], start_location[0], current_cell_location[1] + dlon/2., current_cell_location[0])
        _, _, dist_to_edge_left = g.inv(start_location[1], start_location[0], current_cell_location[1] - dlon/2., current_cell_location[0])

        min_dist = min(dist_to_edge_top, dist_to_edge_right, dist_to_edge_bottom, dist_to_edge_left)
        max_dist = max(dist_to_edge_top, dist_to_edge_right, dist_to_edge_bottom, dist_to_edge_left)
        #distances = np.linspace(min_dist, np.sqrt(2*(max_dist**2)))
        distances = np.linspace(min_dist/2., max_dist*2., 1000)
        starts = np.repeat(start_location[1], distances.size)
        ends = np.repeat(start_location[0], distances.size)
        wind_directions = np.repeat(wind_direction, distances.size)
        lons, lats, _ = g.fwd(starts, ends, wind_directions, distances)

        corner_top_right = (current_cell_location[1] + dlon/2., current_cell_location[0] + dlat/2.)
        corner_bottom_right = (current_cell_location[1] + dlon/2., current_cell_location[0] - dlat/2.)
        corner_bottom_left = (current_cell_location[1] - dlon/2., current_cell_location[0] - dlat/2.)
        corner_top_left = (current_cell_location[1] - dlon/2., current_cell_location[0] + dlat/2.)
        az_to_corner_top_right, _, _ = g.inv(start_location[1], start_location[0], corner_top_right[0], corner_top_right[1])
        az_to_corner_bottom_right, _, _ = g.inv(start_location[1], start_location[0], corner_bottom_right[0], corner_bottom_right[1])
        az_to_corner_bottom_left, _, _ = g.inv(start_location[1], start_location[0], corner_bottom_left[0], corner_bottom_left[1])
        az_to_corner_top_left, _, _ = g.inv(start_location[1], start_location[0], corner_top_left[0], corner_top_left[1])

        if (wind_direction>=az_to_corner_top_right)&(wind_direction<az_to_corner_bottom_right):
            #print('right', az_to_corner_top_right, az_to_corner_bottom_right)
            iclosest = np.argmin(np.sqrt((corner_top_right[0]-lons)**2))
        elif (wind_direction>=az_to_corner_bottom_right)&(wind_direction<az_to_corner_bottom_left):
            #print('bottom', az_to_corner_bottom_right, az_to_corner_bottom_left)
            iclosest = np.argmin(np.sqrt((corner_bottom_right[1]-lats)**2))
        elif (wind_direction>=az_to_corner_bottom_left)&(wind_direction<az_to_corner_top_left):
            #print('left', az_to_corner_bottom_left, az_to_corner_top_left)
            iclosest = np.argmin(np.sqrt((corner_bottom_left[0]-lons)**2))
        else:
            iclosest = np.argmin(np.sqrt((corner_top_right[1]-lats)**2))
            #print('top', az_to_corner_top_left, az_to_corner_top_right)

        """
        points = []
        points.append( [current_cell_location[1], current_cell_location[0] + dlat/2.] )
        points.append( [current_cell_location[1], current_cell_location[0] - dlat/2.] )
        points.append( [current_cell_location[1] + dlon/2., current_cell_location[0]] )
        points.append( [current_cell_location[1] - dlon/2., current_cell_location[0]] )
        points = np.array(points)
        lons_pts = points[:,0]
        lats_pts = points[:,1]

        id_lons_pts = 0
        if (wind_direction>=45.)&(wind_direction<135.):
            id_lons_pts = 2
        elif (wind_direction>=135.)&(wind_direction<180.):
            id_lons_pts = 1
        else:
            id_lons_pts = 3
        iclosest = np.argmin(np.sqrt((lons_pts[id_lons_pts]-lons)**2 + (lats_pts[id_lons_pts]-lats)**2))
        """

        #print('old', start_location)
        #print('old cell', current_cell_location[1], current_cell_location[0])

        ## Update balloon location
        iclosest = min(iclosest+1, lats.size-1) # Find index of next location in next cell
        start_location = [lats[iclosest], lons[iclosest]]

        ## Update current cell
        dists = np.sqrt((winds.lat - start_location[0])**2+(winds.lon - start_location[1])**2)
        data = winds[dists==dists.min()]
        current_cell_location = data[['lat', 'lon']].iloc[0].values

        """
        print('new', start_location)
        print('new cell', current_cell_location[1], current_cell_location[0])
        print(corner_top_right)
        print(corner_bottom_right)
        print(corner_bottom_left)
        print(corner_top_left)
        print(wind_direction)
        """

        #locations.append( start_location )
        #times.append( last_time + distances[iclosest]/wind_strength )
        last_time = times[-1] + distances[iclosest]/wind_strength 
        #print(distances[iclosest], wind_strength, distances[iclosest]/wind_strength, wind_direction)
    
    vels.append( wind_strength )
    wind_dir.append( wind_direction )
    locations.append( start_location )
    times.append( last_time )

    locations = np.array(locations)
    times = np.array(times)
    wind_dir = np.array(wind_dir)
    vels = np.array(vels)
    
    trajectory = pd.DataFrame()
    trajectory['time'] = times
    trajectory['lat'] = locations[:,0]
    trajectory['lon'] = locations[:,1]
    trajectory['wind_dir'] = wind_dir
    trajectory['wind_strength'] = vels
    trajectory.reset_index(drop=True, inplace=True)

    if save_trajectory:
        trajectory.to_csv(f'{folder}trajectory_balloon_lat{start_location[0]:.2f}_lon{start_location[1]:.2f}_{time_max/(3600*24)}days.csv', header=True, index=False)
    
    return trajectory
    
def plot_one_map(ax, winds_grp, trajectory, unknown, display_legend=True, display_proba=False, display_final_proba=False, vmin=-180., vmax=180., unit='degrees from North'):
    
    m = Basemap(projection='kav7', resolution=None, lat_0=0, lon_0=0, ax=ax)
    
    lat_size = winds_grp.lat.unique().size
    lon_size = winds_grp.lon.unique().size
    
    LON, LAT = np.meshgrid(winds_grp.lon.unique(), winds_grp.lat.unique())
    x, y = m(LON.ravel(), LAT.ravel())
    x, y = x.reshape(lat_size, lon_size), y.reshape(lat_size, lon_size)
    #print(winds_grp.val.values.reshape(lat_size, lon_size))
    #print(x.shape, y.shape)
    #sc_dir = m.pcolormesh(x, y, winds_grp.val.values.reshape(lat_size, lon_size), cmap='cividis', alpha=0.8, zorder=5)
    sc_dir = m.pcolormesh(x, y, winds_grp[unknown].values.reshape(lat_size, lon_size), vmin=vmin, vmax=vmax, cmap='cividis', alpha=0.8, zorder=5)
    #x, y = m(winds_grp.lon.values, winds_grp.lat.values)
    #sc_dir = m.scatter(x, y, c=winds_grp[unknown], cmap='cividis', s=10, alpha=0.3, zorder=5)
    
    if trajectory.shape[0] > 0:
        x, y = m(trajectory.lon.values, trajectory.lat.values)
        if True:
            if display_proba:
                cmap = sns.color_palette('rocket', as_cmap=True)
                sc = m.scatter(x, y, c=trajectory.proba, cmap=cmap, s=5, zorder=10, vmin=0., vmax=1.)
            else:
                VariableLimits = np.arange(12)
                cmap = sns.color_palette('rocket', n_colors=VariableLimits.size, as_cmap=True)
                norm = clr.BoundaryNorm(VariableLimits, ncolors=256)
                sc = m.scatter(x, y, c=trajectory.time.values/(3600.*24), cmap=cmap, s=5, zorder=10, norm=norm)

            if display_legend:

                from matplotlib.ticker import FuncFormatter
                fmt = lambda x, pos: '{:.0%}'.format(x) # 
                axins = inset_axes(ax, width="3%", height="80%", loc='lower left', bbox_to_anchor=(1.03, 0.1, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
                axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
                cbar = plt.colorbar(sc, format=FuncFormatter(fmt), cax=axins, orientation='vertical', extend='both')

                if display_proba:
                    cbar.set_label('Probability', rotation=270, labelpad=10)
                else:
                    cbar.set_label('Time (days)', rotation=270, labelpad=10)

        if display_final_proba:
            m.scatter(x[-1], y[-1], edgecolor='red', facecolor=None, s=5, zorder=20)
            x, y = m(trajectory.lon.values[-1]+4., trajectory.lat.values[-1]-4.)
            ax.text(x, y, f'Final probability: {trajectory.proba.values[-1]*100:.0f}$\%$', zorder=50, path_effects=path_effects, ha='left', va='top', transform=ax.transData)
    
    
    axins_dir = inset_axes(ax, width="80%", height="5%", loc='lower left', bbox_to_anchor=(.1, 1.03, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins_dir.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)
    cbar_dir = plt.colorbar(sc_dir, cax=axins_dir, orientation='horizontal', extend='both')
    cbar_dir.set_label(unknown.replace('_', ' ') + f' ({unit})', labelpad=10)
    cbar_dir.ax.xaxis.tick_top()
    cbar_dir.ax.xaxis.set_label_position('top')
    
def plot_trajectory(winds, trajectory, only_first_plot=True, vmin=-180., vmax=180.):
    
    winds_grp = winds.groupby(['lat', 'lon']).first().reset_index()
    
    fig = plt.figure(figsize=(6,5))
    if only_first_plot:
        grid = fig.add_gridspec(1, 1)
    else:
        grid = fig.add_gridspec(1, 2)
    
    ax = fig.add_subplot(grid[0,0])
    #winds_grp = winds.loc[winds['var'] == 'W-E wind component (m/s)']
    #plot_one_map(ax, winds_grp, 'val', display_days=False)
    plot_one_map(ax, winds_grp, trajectory, 'wind_direction', display_legend=True, display_proba=True, display_final_proba=True, vmin=vmin, vmax=vmax)
    #M0 = trajectory.M0.iloc[0]
    #ax.set_title(f'Detection probabilities for $M_w={M0:.1f}$', y=1.3)
    
    if not only_first_plot:

        ax = fig.add_subplot(grid[0,1])
        #winds_grp = winds.loc[winds['var'] == 'S-N wind component (m/s)']
        #plot_one_map(ax, winds_grp, 'val')
        plot_one_map(ax, winds_grp, trajectory, 'wind_strength', unit='m/s', vmin=0., vmax=100.)

    fig.subplots_adjust(left=0., right=0.8, bottom=0., top=1.,)
    #fig.savefig('./test_data_Venus/balloon_trajectory_proba2.png', transparent=True)