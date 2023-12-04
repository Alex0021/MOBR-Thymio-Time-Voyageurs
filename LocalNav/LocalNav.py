import os
import sys
import math
from statistics import mean
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.getcwd(), 'LocalNav/sensors'))
sys.path.insert(0, os.path.join(os.getcwd(), 'LocalNav/global_map'))
sys.path.insert(0, os.path.join(os.getcwd(), 'LocalNav/tools'))

from sensors import obstacles_pos
from local_occupancy import sensor_measurements, sensor_distances
from local_occupancy import thymio_coords, sensor_pos_from_center, sensor_angles
from map_global import create_map_global, update_map
from linear_regression import linear_regression
from count_group import count_transitions, count_group


def localNav(abs_pos, goal_position, prox_horizontal, map_global):
    # Get sensors values
    sensor_vals = prox_horizontal

    # Exchange rear elements to represent view from above
    sensor_vals[5], sensor_vals[6] = sensor_vals[6], sensor_vals[5]

    # Compute position of obstacles
    obstacles_positions = obstacles_pos(sensor_vals, sensor_measurements, sensor_distances, sensor_pos_from_center, sensor_angles)


    # Compute linear regression of lines from front and back
    # Front
    data = obstacles_positions[0:5]
    X, y_pred, mask = linear_regression(data)

    # Back
    data_back = obstacles_positions[5:8]
    X_back, y_pred_back, mask_back = linear_regression(data_back)


    # Compute absolute values
    # Rotation matrix
    theta = -abs_pos[0][2]+math.pi/2
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
    # Thymio
    thymio_coords_rotated = np.dot(thymio_coords, rotation_matrix)
    thymio_X = abs_pos[0][0]+thymio_coords_rotated[:,0]
    thymio_y = abs_pos[0][1]+thymio_coords_rotated[:,1]

    # Front obstacles
    front_obst_coords = np.column_stack((X, y_pred))
    front_obst_coords_rotated = np.dot(front_obst_coords, rotation_matrix)
    front_obst_X = front_obst_coords_rotated[:, 0]+abs_pos[0][0]
    front_obst_y = front_obst_coords_rotated[:, 1]+abs_pos[0][1]

    # Back obstacles
    back_obst_coords = np.column_stack((X_back, y_pred_back))
    back_obst_coords_rotated = np.dot(back_obst_coords, rotation_matrix)
    back_obst_X = back_obst_coords_rotated[:, 0]+abs_pos[0][0]
    back_obst_y = back_obst_coords_rotated[:, 1]+abs_pos[0][1]

    # Calculate the angle to reach the goal in a straight line
    delta_x = goal_position[0][0] - abs_pos[0][0]
    delta_y = goal_position[0][1] - abs_pos[0][1]
    angle_to_goal = math.atan2(delta_y, delta_x)-abs_pos[0][2]

    # Filter value to create extra wall, constraints
    # Compute number of group
    nb_grouph, mask, mask_back = count_group(mask, mask_back, X, y_pred, X_back, y_pred_back)

    
    # Update Map
    map_global = update_map(mask, mask_back, map_global, front_obst_X, front_obst_y, back_obst_X, back_obst_y)

    return map_global

