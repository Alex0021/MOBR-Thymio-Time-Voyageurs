import numpy as np
import math

def create_map_global():
    # Create a xy map with all zeros
    map_size_x = 100
    map_size_y = 100
    map_global = np.zeros((map_size_x, map_size_y))

    # Set the edges to 1
    map_global[0, :] = 1  # Top edge
    map_global[-1, :] = 1  # Bottom edge
    map_global[:, 0] = 1  # Left edge
    map_global[:, -1] = 1  # Right edge
    
    return map_global


def update_map(abs_pos, map_global, front_obst_X, front_obst_y, back_obst_X, back_obst_y):
    # Init
    rounded_x_indices = []
    rounded_y_indices = []
  
    # Set all value from absolute position
    for i in range(len(front_obst_X)):
        if not ((math.isnan(front_obst_y[i]) and math.isnan(front_obst_y[i])) ):
            rounded_x_indices.append(np.round(abs_pos[-1][0] + front_obst_X[i], 1).astype(int))
            rounded_y_indices.append(np.round(abs_pos[-1][1] + front_obst_y[i], 1).astype(int))
            
    for i in range(len(back_obst_X)):
        if not ((math.isnan(back_obst_y[i]) and math.isnan(back_obst_y[i])) ):
            rounded_x_indices.append(np.round(abs_pos[-1][0] + back_obst_X[i], 1).astype(int))
            rounded_y_indices.append(np.round(abs_pos[-1][1] + back_obst_y[i], 1).astype(int))


    for x, y in zip(rounded_x_indices[1:], rounded_y_indices[1:]):
        map_global[x, y] = 1
                
    return map_global