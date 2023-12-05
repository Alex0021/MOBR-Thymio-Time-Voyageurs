import math
import numpy as np

## Sensor measurements
sensor_distances = np.array([i for i in range(0,21)])
sensor_measurements = np.array([5150, 4996, 4964, 4935, 4554, 4018, 3624, 3292, 2987, 
              2800, 2580, 2307, 2039, 1575, 1127, 833, 512, 358, 157, 52, 0])

# Thymio outline
center_offset = np.array([5.5,5.5])
thymio_coords = np.array([[0,0], [11,0], [11,8.5], [10.2, 9.3], 
                          [8, 10.4], [5.5,11], [3.1, 10.5], 
                          [0.9, 9.4], [0, 8.5], [0,0]])-center_offset

# Sensor positions and orientations
sensor_pos_from_center = np.array([[0.9,9.4], [3.1,10.5], [5.5,11.0], [8.0,10.4], [10.2,9.3], [8.5,0], [2.5,0]])-center_offset
sensor_angles = np.array([120, 105, 90, 75, 60, -90, -90])*math.pi/180

