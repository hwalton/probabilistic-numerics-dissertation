import numpy as np

out = np.shape(np.ones((100,10)) @ np.ones((10,10)) @ np.ones((10,100)))

print(out)