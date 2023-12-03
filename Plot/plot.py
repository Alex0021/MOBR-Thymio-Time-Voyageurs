import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
import math

def plot(abs_pos, thymio_coords, goal_position, map_global, closest_angle, ax, best_path=None):
    # Plot
    # Plot Map
    ax.imshow(map_global, cmap='binary', origin='lower')
    ax.set_title("Global Map")
    ax.set_xlim(0, len(map_global[:,1]))
    ax.set_ylim(0, len(map_global[1,:]))
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Y [cm]")
    # Plot goal
    ax.scatter(goal_position[0][0], goal_position[0][1], color='red')
    # Plot thymio
    # Rotation matrix
    theta = abs_pos[0][2] - math.pi/2
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
    # Thymio
    thymio_coords_rotated = np.dot(rotation_matrix, thymio_coords.T)
    thymio_coords_rotated = thymio_coords_rotated.T
    thymio_X = abs_pos[0][0]+thymio_coords_rotated[:,0]
    thymio_y = abs_pos[0][1]+thymio_coords_rotated[:,1]
    ax.plot(thymio_X, thymio_y)
    
    # Plot Direction
    x, y, angle = abs_pos[0][:]
    dx = math.cos(closest_angle)#angle_radians
    dy = math.sin(closest_angle)#closest_angle
    ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=0.1, color='r', width=0.01)

    if best_path is not None:
        ax.plot(best_path[0], best_path[1], marker='o', color='blue')


