import numpy as np
import math
import json
import time
from datetime import datetime
import os.path
import matplotlib.path as mplPath
from scipy.spatial import cKDTree
import frictionmap

"""
Created by:
Leonhard Hermansdorfer

Created on:
01.12.2018
"""

def generate_friction_map(track_name, initial_mue, cellwidth_m, friction_descriptions, inside_trackbound='left', bool_show_plots=False):
    """
    This script generates a grid respresenting the friction map with specified cellwidth. Additionally, it fills the
    corresponding cells of the friction map with a default mue value.

    Input
    ---
    :param  track_name:             Name of the racetrack. The name must be the same as the name of the folder in the
                                    f1tenth_racetracks folder.
    :param  initial_mue:            mue value which should be used to initialize the friction map (all cells contain this value).
    :param  cellwidth_m:            width of the grid cells of the friction map (cells are quadratic).
    :param  friction_descriptions   description of the frictions along the track.
                                    Parameters:
                                    - percentage_location_s: (array of two floats) Distance in % from the beginning of the trajectory.
                                                            This specifies the region where to apply the friction.
                                                            !!! Important !!! - In the current version the first number must be always lower than the second one.
                                    - change: ('constant' / 'linear') Constant - constant friction in the region
                                                                    Linear - Linear change of the friction in the region.
                                    - friction: (Array of one(for 'constant' change) / two(for 'linear' change) floats) Defines the friction in the region.
    :param  inside_trackbound:      specifies which trackbound is on the inside of the racetrack ('right' or 'left'). This is
                                    only necessary for circuits (closed racetracks).
    :param  bool_show_plots:        boolean which enables plotting of the reference line, the friction map and the corresponding
                                    mue values

    Output
    ---
    Saves the friction map as a grid and the corresponding mue values as a dictionary in the frictionmap/Output folder.
    """
    
    # ----------------------------------------------------------------------------------------------------------------------
    # INITIALIZATION -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # determine names of output files
    datetime_save = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_tpamap = track_name + '_tpamap.csv'  # datetime_save + '_' +
    filename_tpadata = track_name + '_tpadata.json'

    # set paths
    path2module = os.path.dirname(os.path.abspath(__file__))
    path2reftrack_file = os.path.join(path2module, '../f1tenth_racetracks', track_name, track_name + '_centerline.csv')
    path2tpamap_file = os.path.join(path2module, 'Output', track_name, filename_tpamap)
    path2tpadata_file = os.path.join(path2module, 'Output', track_name, filename_tpadata)

    # ----------------------------------------------------------------------------------------------------------------------
    # CALCULATE REFERENCE LINE ---------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    # read reference track file
    reftrack = frictionmap.reftrack_functions.load_reftrack(path2track=path2reftrack_file)

    # check whether reference line is closed (is the race track a circuit or not)
    bool_isclosed_refline = frictionmap.reftrack_functions.check_isclosed_refline(refline=reftrack[:, :2])

    # calculate coordinates of the track boundaries
    reftrackbound_right, reftrackbound_left = frictionmap.reftrack_functions.calc_trackboundaries(reftrack=reftrack)

    # construct array between each two points on the trajectory
    track_point_distances = []
    for i in range(len(reftrack[:, :2]) - 1):
        track_point_distances.append(
            math.sqrt(pow(reftrack[:, :2][i][0] - reftrack[:, :2][i + 1][0], 2.0) + pow(reftrack[:, :2][i][1] - reftrack[:, :2][i + 1][1], 2.0)))

    # if is the trajectory closed add the distance between last and first point as well
    if bool_isclosed_refline:
        track_point_distances.append(math.sqrt(pow(reftrack[:, :2][-1][0] - reftrack[:, :2][0][0], 2.0) + pow(reftrack[:, :2][-1][1] - reftrack[:, :2][0][1], 2.0)))

    # calculate cumulative sum from the distances
    track_cumsum_distances = np.cumsum(track_point_distances)
    # ----------------------------------------------------------------------------------------------------------------------
    # SAMPLE COORDINATES FOR FRICTION MAP ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    timer_start = time.perf_counter()

    # match left/right track boundary to "inside" and "outside" (only necessary for circuits)
    if bool_isclosed_refline and inside_trackbound == 'right':
        trackbound_inside = reftrackbound_right
        trackbound_outside = reftrackbound_left

        sign_trackbound = -1

    else:
        trackbound_inside = reftrackbound_left
        trackbound_outside = reftrackbound_right

        sign_trackbound = 1

    # set a default distance which is added / subtracted from max / min reference line coordinate to ensure that whole
    # racetrack is covered during coordinate point sampling
    default_delta = int(math.ceil(np.amax(reftrack[:, 2]) + np.amax(reftrack[:, 3]) + 5.0))

    # calculate necessary range to cover the whole racetrack with grid xyRange = [x_min, x_max, y_min, y_max]
    xyRange = [int(math.floor(np.amin(reftrack[:, 0]) - default_delta)),
            int(math.ceil(np.amax(reftrack[:, 0]) + default_delta)),
            int(math.floor(np.amin(reftrack[:, 1]) - default_delta)),
            int(math.ceil(np.amax(reftrack[:, 1]) + default_delta))]

    # set-up 2D-grid
    x_grid = np.arange(xyRange[0], xyRange[1] + 0.1, cellwidth_m)
    y_grid = np.arange(xyRange[2], xyRange[3] + 0.1, cellwidth_m)

    # get number of coordinates for array initialization
    size_array = x_grid.shape[0] * y_grid.shape[0]
    coordinates = np.empty((size_array, 2))

    # create coordinate array which contains all coordinate of the defined grid
    i_row = 0

    for x_coord in x_grid:
        coordinates[i_row:i_row + y_grid.shape[0], 0] = np.full((y_grid.shape[0]), x_coord)
        coordinates[i_row:i_row + y_grid.shape[0], 1] = y_grid
        i_row += y_grid.shape[0]

    # set maximum distance between grid cells outside the track and trackboundaries to determine all relevant grid cells
    dist_to_trackbound = cellwidth_m * 1.1

    # distinguish between a closed racetrack (circuit) and an "open" racetrack
    if bool_isclosed_refline:
        bool_isIn_rightBound = mplPath.Path(trackbound_outside). \
            contains_points(coordinates, radius=(dist_to_trackbound * sign_trackbound))
        bool_isIn_leftBound = mplPath.Path(trackbound_inside). \
            contains_points(coordinates, radius=-(dist_to_trackbound * sign_trackbound))
        bool_OnTrack = (bool_isIn_rightBound & ~bool_isIn_leftBound)

    else:
        trackbound = np.vstack((trackbound_inside, np.flipud(trackbound_outside)))
        bool_OnTrack = mplPath.Path(trackbound).contains_points(coordinates, radius=-dist_to_trackbound)

    # generate the friction map with coordinates which are within the trackboundaries or within the defined range outside
    tpa_map = cKDTree(coordinates[bool_OnTrack])

    print('INFO: Time elapsed for tpa_map building: {:.3f}s\nINFO: tpa_map contains {} coordinate points'.format(
        (time.perf_counter() - timer_start), tpa_map.n))

    # ----------------------------------------------------------------------------------------------------------------------
    # SAVE FRICTION MAP ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    timer_start = time.perf_counter()

    # create dictionary filled with default mue value (value as numpy array)
    tpamap_indices = tpa_map.indices
    tpa_data = dict(zip(tpamap_indices, np.full((tpamap_indices.shape[0], 1), initial_mue)))

    print('INFO: Time elapsed for tpa_data dictionary building: {:.3f}s'.format(time.perf_counter() - timer_start))

    # search through all friction coefficients
    for i in range(tpa_map.data.size // 2):
        # find closest point on the trajectory to the current friction point
        closest_point_id = np.argmin(np.sum(np.square(reftrack[:, :2] - tpa_map.data[i]), 1))

        # calculate where on the trajectory is this point [%]
        percentage_distance_from_start = 100.0 / track_cumsum_distances[-1] * track_cumsum_distances[closest_point_id]

        for friction_description in friction_descriptions:
            if friction_description['percentage_location_s'][0] <= percentage_distance_from_start < friction_description['percentage_location_s'][1]:
                if friction_description['change'] == 'constant':
                    tpa_data[i] = np.array([friction_description['friction'][0]])
                elif friction_description['change'] == 'linear':
                    s_0 = friction_description['percentage_location_s'][0]
                    s_1 = friction_description['percentage_location_s'][1]
                    s_current = percentage_distance_from_start
                    f_0 = friction_description['friction'][0]
                    f_1 = friction_description['friction'][1]
                    f_current = (f_1 - f_0) / (s_1 - s_0) * (s_current - s_0) + f_0
                    tpa_data[i] = np.array([f_current])

    output_folder = os.path.join(path2module, 'frictionmap', 'Output', track_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # save friction map (only grid) ('*_tpamap.csv')
    with open(path2tpamap_file, 'wb') as fh:
        np.savetxt(fh, tpa_map.data, fmt='%0.4f', delimiter=';', header='x_m;y_m')

    print('INFO: tpa_map saved successfully!')

    # get tpadata as string to save as a dictionary (as .json file)
    tpa_data_string = {str(k): list(v) for k, v in tpa_data.items()}

    # save friction data ('*_tpadata.json')
    with open(path2tpadata_file, 'w') as fh:
        json.dump(tpa_data_string, fh, separators=(',', ': '))

    print('INFO: tpa_data saved successfully!')

    # ----------------------------------------------------------------------------------------------------------------------
    # CREATE PLOTS ---------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    if bool_show_plots:
        # plot reference line and normal vectors
        frictionmap.reftrack_functions.plot_refline(reftrack=reftrack)

        # plot spatial grid of friction map
        frictionmap.plot_frictionmap_grid. \
            plot_voronoi_fromVariable(tree=tpa_map,
                                    refline=reftrack[:, :2],
                                    trackbound_left=reftrackbound_left,
                                    trackbound_right=reftrackbound_right)

        # plot friction data of friction map
        frictionmap.plot_frictionmap_data. \
            plot_tpamap_fromVariable(tpa_map=tpa_map,
                                    tpa_data=tpa_data,
                                    refline=reftrack[:, :2],
                                    trackbound_left=reftrackbound_left,
                                    trackbound_right=reftrackbound_right)