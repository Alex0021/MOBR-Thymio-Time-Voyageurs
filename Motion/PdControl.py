import math
import numpy as np

# TUNABLE PARAMETERS
BASE_SPEED = 75 # In thymio units
ANGLE_FACTOR = 100
DIST_FACTOR = 7


def pd_control(pose, target):
    R = np.array([[math.cos(pose[2]), math.sin(pose[2])], [-math.sin(pose[2]), math.cos(pose[2])]])
    target_ref_thymio = R @ (target - pose[:2])
    dist = np.linalg.norm(target_ref_thymio)
    angle = math.atan2(target_ref_thymio[1], target_ref_thymio[0])
    # IF ANGLE TO TARGET IS TOO HIGH, TURN ON ITSELF
    if abs(angle) > math.pi/8:
        angle_sign = np.sign(angle)
        left_m = -1*angle_sign*(BASE_SPEED + abs(angle)*ANGLE_FACTOR/2)
        right_m = angle_sign*(BASE_SPEED + abs(angle)*ANGLE_FACTOR/2)
    # FOLLOW HEADING VECTOR W/ ANGLE PROPORTION
    else:
        left_m = BASE_SPEED + DIST_FACTOR*dist - angle*ANGLE_FACTOR
        right_m = BASE_SPEED + DIST_FACTOR*dist + angle*ANGLE_FACTOR
    return np.clip([left_m, right_m], -400, 400)