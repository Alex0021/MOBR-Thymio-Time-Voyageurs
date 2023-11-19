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


def update_map(abs_pos, map_global, obstacles_pos):
    # Set all value from absolute position
    for pos in obstacles_pos:
        if not math.isinf(pos[0] or pos[1]):
            rounded_x_indices = np.round(abs_pos[-1][0] + pos[0], 1).astype(int)
            rounded_y_indices = np.round(abs_pos[-1][1] + pos[1], 1).astype(int)

    for x, y in zip(rounded_x_indices, rounded_y_indices):
        map_global[x, y] = 1
                
    return map_global