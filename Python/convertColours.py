import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# =====================================================

oldImageName = 'TestSolver2.png'
newImageName = 'newName.png'
colorMap = 'bone'

compress = False

# =====================================================

im = Image.open(oldImageName)
PILpixelColors = list(im.getdata())

PILpixelColors = [PILpixelColor[0] for PILpixelColor in PILpixelColors]

pixelColors = np.array(PILpixelColors)
pixelColors = np.reshape(pixelColors, im.size)

fig, ax = plt.subplots(figsize=(1,1))  
plt.axis('off')
ax.set_position([0, 0, 1, 1], which='both')
ax.set_aspect('equal')

plt.imshow(-pixelColors, cmap=colorMap)

plt.savefig(newImageName, format='png', dpi=im.size[0], pad_inches=0)

print(max(pixelColors[0]))

# =====================================================
if (compress):
    im = Image.open(newImageName)
    im.save(newImageName, "JPEG", optimize = True, quality = 100)