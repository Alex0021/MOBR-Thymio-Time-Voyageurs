import numpy as np
import math

# Draw lines between 2 points or more
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
         
# Given actual absolute position of robot, current global map and local obstacles, update map to add local obstacles into global map
def update_map(mask, mask_back, map_global, front_obst_X, front_obst_y, back_obst_X, back_obst_y):
    # Init
    rounded_x_indices = []
    rounded_y_indices = []
    updated_mask = []
    # FRONT
    # If mask is true, draw a point or line within obst_X, obst_y
    # Add obstacles if point
    for i in range(len(front_obst_X)):
        if not (math.isnan(front_obst_X[i]) or math.isnan(front_obst_y[i])):
            rounded_x_indices.append(int(round(front_obst_X[i], 1)))
            rounded_y_indices.append(int(round(front_obst_y[i], 1)))
            updated_mask.append(mask[i])
            map_global[int(round(front_obst_y[i], 1)), int(round(front_obst_X[i], 1))] = 1
            
    # Add obstacles if lines
    for i in range(len(rounded_x_indices) - 1):
        x0, y0 = rounded_x_indices[i], rounded_y_indices[i]
        x1, y1 = rounded_x_indices[i + 1], rounded_y_indices[i + 1]
        
        # If mask is true, draw a line within front_obst_X, front_obst_y
        if updated_mask[i]:
            draw_line(map_global, x0, y0, x1, y1)       
            
    # Reset indices for back points
    rounded_x_indices = []
    rounded_y_indices = []
    updated_mask = []

    # BACK
    # If mask_back is true, draw a point or line within back_obst_X, back_obst_y
    # Add obstacles if point
    for i in range(len(back_obst_X)):
        if not (math.isnan(back_obst_X[i]) or math.isnan(back_obst_y[i])):
            rounded_x_indices.append(int(round(back_obst_X[i], 1)))
            rounded_y_indices.append(int(round(back_obst_y[i], 1)))
            updated_mask.append(mask_back[i])
            map_global[int(round(back_obst_y[i], 1)), int(round(back_obst_X[i], 1))] = 1

    # Add obstacles if lines
    for i in range(len(rounded_x_indices) - 1):
        x0, y0 = rounded_x_indices[i], rounded_y_indices[i]
        x1, y1 = rounded_x_indices[i + 1], rounded_y_indices[i + 1]

        if updated_mask[i]:
            draw_line(map_global, x0, y0, x1, y1)
          
    return map_global
    
 