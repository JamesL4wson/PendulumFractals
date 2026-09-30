import numpy as np
import matplotlib.pyplot as plt

divergencesFile = open("3PendulumHighResData.txt")
divergences = [float(line.rstrip()) for line in divergencesFile]

divergGrid = np.reshape(divergences, (1000, 1000))
divergGrid = np.roll(divergGrid, (500, 500), axis=(0, 1))
divergGrid = np.swapaxes(divergGrid, 0, 1)

fig = plt.figure(figsize=(10, 8), dpi=100)

plt.imshow(divergGrid, cmap='viridis', vmin=min(divergences), vmax=max(divergences), norm="symlog")

plt.show()