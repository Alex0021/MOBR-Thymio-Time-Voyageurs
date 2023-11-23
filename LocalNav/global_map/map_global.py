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

def draw_line(map_array, x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        map_array[y0, x0] = 1
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def update_map(mask, mask_back, map_global, front_obst_X, front_obst_y, back_obst_X, back_obst_y):
    # Init
    rounded_x_indices = []
    rounded_y_indices = []
    
    # Set all values from absolute position
    # If mask is true, draw a line within front_obst_X, front_obst_y
    for i in range(len(front_obst_X)):
        if not (math.isnan(front_obst_X[i]) or math.isnan(front_obst_y[i])):
            rounded_x_indices.append(int(round(front_obst_X[i], 1)))
            rounded_y_indices.append(int(round(front_obst_y[i], 1)))

    for i in range(len(rounded_x_indices) - 1):
        x0, y0 = rounded_x_indices[i], rounded_y_indices[i]
        x1, y1 = rounded_x_indices[i + 1], rounded_y_indices[i + 1]

        if mask[i]:
            draw_line(map_global, x0, y0, x1, y1)
            
    # Reset indices for back points
    rounded_x_indices = []
    rounded_y_indices = []

    # If mask_back is true, draw a line within back_obst_X, back_obst_y
    for i in range(len(back_obst_X)):
        if not (math.isnan(back_obst_X[i]) or math.isnan(back_obst_y[i])):
            rounded_x_indices.append(int(round(back_obst_X[i], 1)))
            rounded_y_indices.append(int(round(back_obst_y[i], 1)))

    for i in range(len(rounded_x_indices) - 1):
        x0, y0 = rounded_x_indices[i], rounded_y_indices[i]
        x1, y1 = rounded_x_indices[i + 1], rounded_y_indices[i + 1]

        if mask_back[i]:
            draw_line(map_global, x0, y0, x1, y1)
                
    return map_global
    
 