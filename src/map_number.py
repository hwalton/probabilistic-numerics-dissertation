import numpy as np

def map_number(x, x_min, x_max, y_min, y_max):
    initial_output = (x - x_min) * (y_max - y_min) / (x_max - x_min) + y_min
    clipped_output = np.clip(initial_output, min(y_min,y_max), max(y_min,y_max))
    return clipped_output
