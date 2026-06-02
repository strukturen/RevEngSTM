# Reverse Engineering Strut-and-Tie Models
# ================================
# Karin Yu, 2026

# utils, includes some common helper functions

__author__ = 'Karin Yu'

import numpy as np

def getAngle(dx: float, dy: float):
    # get angle in radian from x-axis, based on dx and dy
    if dx == 0:
        if dy > 0:
            return np.pi / 2
        elif dy < 0:
            return np.pi / 2 * 3
        else:
            return 0
    elif dx > 0:
        return np.arctan(dy / dx)
    else:
        return np.arctan(dy / dx) + np.pi

def boundAngle(angle, factor_max_angle = 2):
    # bound angle to between 0 and factor_max_angle * pi
    if angle < 0: angle += 2*np.pi
    if factor_max_angle == 2 and angle >= factor_max_angle*np.pi:
        angle -= 2*np.pi
    while factor_max_angle == 1 and angle >= np.pi:
        angle -= np.pi
    return angle

def LineInPolygon(line, polygon):
    # returns True if Line is inside polygon but not overlapping boundary
    return polygon.intersects(line) and not polygon.touches(line)
