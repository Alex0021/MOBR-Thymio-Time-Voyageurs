import math

def find_limits(map_global, abs_pos):
    rows, cols = len(map_global), len(map_global[0])
    x = abs_pos[0][0]
    y = abs_pos[0][1]
    theta = abs_pos[0][2]
    
    # Initialize distance and bounds
    top_distance = bottom_distance = left_distance = right_distance = 120
    xmin_bound = xmax_bound = ymin_bound = ymax_bound = 0

    # Calculate the direction vector based on the robot's angle
    direction_vector = (math.cos(theta), math.sin(theta))
     
    # Iterate from the top side
    for i in range(int(y), 0, -1):
        if math.isclose(theta, 0.0, abs_tol=1e-5):
            new_x = int(x)
        else:
            new_x = int(x + (y - i) / math.tan(theta))
        if 0 <= new_x < cols and 0 <= i < rows and map_global[i][new_x] == 1:
            top_distance = y - i
            break

    # Iterate from the bottom side
    for i in range(int(y), rows):
        if math.isclose(theta, 0.0, abs_tol=1e-5):
            new_x = int(x)
        else:
            new_x = int(x + (i - y) / math.tan(theta))
        if 0 <= new_x < cols and 0 <= i < rows and map_global[i][new_x] == 1:
            bottom_distance = i - y
            break

    # Iterate from the left side
    for j in range(int(x), 0, -1):
        if math.isclose(theta, 0.0, abs_tol=1e-5):
            new_y = int(y)
        else:
            new_y = int(y + (x - j) / math.tan(theta))
        if 0 <= new_y < rows and 0 <= j < cols and map_global[new_y][j] == 1:
            left_distance = x - j
            break

    # Iterate from the right side
    for j in range(int(x), cols):
        if math.isclose(theta, 0.0, abs_tol=1e-5):
            new_y = int(y)
        else:
            new_y = int(y + (j - x) / math.tan(theta))
        if 0 <= new_y < rows and 0 <= j < cols and map_global[new_y][j] == 1:
            right_distance = j - x
            break
                
    # Calculate bounds based on distances
    ymin_bound = max(0, y - top_distance)
    ymax_bound = min(rows - 1, y + bottom_distance)
    xmin_bound = max(0, x - left_distance)
    xmax_bound = min(cols - 1, x + right_distance)


    return xmin_bound, xmax_bound, ymin_bound, ymax_bound