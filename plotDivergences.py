import numpy as np
import matplotlib.pyplot as plt

divergencesFile = open("3PendulumHighResData.txt")
divergences = [float(line.rstrip()) for line in divergencesFile]

divergGrid = np.reshape(divergences, (1000, 1000))
divergGrid = np.roll(divergGrid, (500, 500), axis=(0, 1))
divergGrid = np.swapaxes(divergGrid, 0, 1)

fig, ax = plt.subplots(figsize=(1,1))  
plt.axis('off')
ax.set_position([0, 0, 1, 1], which='both')
ax.set_aspect('equal')

plt.imshow(divergGrid, cmap='viridis', vmin=min(divergences), vmax=max(divergences), norm="symlog")

plt.savefig('destination_path.png', format='png', dpi=1000, pad_inches=0)