import math

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi