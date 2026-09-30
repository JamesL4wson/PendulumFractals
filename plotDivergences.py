import numpy as np
import matplotlib.pyplot as plt

divergencesFile = open("divergences.txt")
divergences = [float(line.rstrip()) for line in divergencesFile]

divergGrid = np.reshape(divergences, (100, 50))
divergGrid = np.roll(divergGrid, (25, 50), axis=(0, 1))
divergGrid = np.swapaxes(divergGrid, 0, 1)

fig = plt.figure(figsize=(10, 8), dpi=100)

plt.imshow(divergGrid, cmap='viridis', vmin=min(divergences), vmax=max(divergences), norm="symlog")

plt.show()