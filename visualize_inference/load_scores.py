import numpy as np
scores = np.load("trajectory_ddpo/scores.npy")
print([float(x) for x in scores])
