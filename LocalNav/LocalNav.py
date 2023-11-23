import os
import sys
import math
from statistics import mean
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.getcwd(), 'sensors'))
sys.path.insert(0, os.path.join(os.getcwd(), 'global_map'))
sys.path.insert(0, os.path.join(os.getcwd(), 'tools'))
from sensors import obstacles_pos
from local_occupancy import sensor_measurements, sensor_distances
from local_occupancy import thymio_coords, sensor_pos_from_center, sensor_angles
from map_global import create_map_global, update_map, draw_line
from linear_regression import linear_regression
from count_group import count_transitions, count_group
from normalize_angle import normalize_angle

# %matplotlib inline


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
    data_back = obstacles_positions[5:7]
    X_back, y_pred_back, mask_back = linear_regression(data_back)

    # Filter value to create extra wall, constraints, nb of possible exit path
    nb_grouph, mask, mask_back = count_group(mask, mask_back, X, y_pred, X_back, y_pred_back)

    # Add extra Walls
    i=0
    while i < len(data):
        if mask[i] and np.isnan(y_pred[i]):
            X[i] = (X[i-1]+X[i+1])/2
            y_pred[i] = (y_pred[i-1]+y_pred[i+1])/2
        i = i + 1

    # Compute absolute values
    # Rotation matrix
    theta = abs_pos[0][2]
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
    angle_to_goal = math.atan2(delta_y, delta_x)

    # Number of possible path
    if nb_grouph == 1:
        nb_path = 2
    else:
        nb_path = nb_grouph

    # Find best path
    possible_path = []

    # Look if possible in front
    for i in range(len(mask.flatten())):
        if mask[i]==0:
            possible_path.append(i)
            
    # Look if possible in back
    for i in range(len(mask_back.flatten())):
        if mask_back[i]==0:
            possible_path.append(i+5)
            
    # Add edge front
    if mask[0]==1:
        possible_path.append(-1)
    if mask[4]==1:
        possible_path.append(-2)

    # Add edge back
    if mask_back[0]==1:
        possible_path.append(-3)
    if mask_back[1]==1:
        possible_path.append(-4)


    # Convert possible_path to angles
    angle_mapping = {
        6: 140, -4: 60,-1: 90, 0: 40, 1: 40, 2: 0, 3: -40, 4: -60, -2: -90, -3: -90, 5: -140
    }

    possible_path_angles = []

    for i, state in enumerate(possible_path):
        # Check if the state has a corresponding angle in the mapping
        if state in angle_mapping:
            # Convert degrees to radians
            angle_in_radians = math.radians(angle_mapping[state])
            possible_path_angles.append(angle_in_radians)
            
    for angle in possible_path_angles:
        angle = normalize_angle(angle + theta)

    # Choose angle closest to angle_to_goal
    closest_angle = min(possible_path_angles, key=lambda x: abs(x - angle_to_goal))

    # Update Map
    map_global = update_map(mask, mask_back, map_global, front_obst_X, front_obst_y, back_obst_X, back_obst_y)

    return closest_angle, map_global

