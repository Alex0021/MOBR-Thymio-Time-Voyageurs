import math
import numpy as np
from scipy.interpolate import interp1d

## Interpolation from sensor values to distances in cm
def sensor_val(val, sensor_measurements, sensor_distances):# in cm
    """
    Returns the distance corresponding to the sensor value based 
    on the sensor characteristics
    :param val: the sensor value to be convert to a distance
    :return: corresponding distance in cm
    """
    if val == 0:
        return np.inf
    
    f = interp1d(sensor_measurements, sensor_distances)
    return max(f(val).item() + 3, 0)

def obstacles_pos(sensor_vals, sensor_measurements, sensor_distances, sensor_pos_from_center, sensor_angles):
    """
    Returns a list containing the position of the obstacles
    w.r.t the center of the Thymio robot. 
    :param sensor_vals: sensor values provided clockwise starting from the top left sensor.
    :return: numpy.array() that contains the position of the different obstacles
    """
    dist_to_sensor = [sensor_val(x, sensor_measurements, sensor_distances) for x in sensor_vals]
    dx_from_sensor = [d*math.cos(alpha) for (d, alpha) in zip(dist_to_sensor, sensor_angles)]
    dy_from_sensor = [d*math.sin(alpha) for (d, alpha) in zip(dist_to_sensor, sensor_angles)]
    obstacles_pos = [[x[0]+dx, x[1]+dy] for (x,dx,dy) in zip(sensor_pos_from_center,dx_from_sensor,dy_from_sensor )]
    return np.array(obstacles_pos)