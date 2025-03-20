# 這是用來生成複數張SEM圖的腳本(跑這個檔案時須搭配在run_nebula.sh裡不可單一使用)

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# # Find out which file to open
if len(sys.argv) < 3:
    print("No output file or image number provided")
    sys.exit()

filename = sys.argv[1]
image_number = sys.argv[2]

if not os.path.exists(filename):
    print("File {} cannot be found".format(filename))
    sys.exit()

# This is a numpy datatype that corresponds to output files
electron_dtype = np.dtype([
    ('x',  '=f'), ('y',  '=f'), ('z',  '=f'), # Position
    ('dx', '=f'), ('dy', '=f'), ('dz', '=f'), # Direction
    ('E',  '=f'),                             # Energy
    ('px', '=i'), ('py', '=i')])              # Pixel index

# Open the output file
data = np.fromfile(filename, dtype=electron_dtype)
print("Number of electrons detected: {}".format(len(data)))


# Make a histogram of pixel indices
xmin = data['px'].min()
xmax = data['px'].max()
ymin = data['py'].min()
ymax = data['py'].max()
H, xedges, yedges = np.histogram2d(data['px'], data['py'],
	bins = [
		np.linspace(xmin-.5, xmax+.5, xmax-xmin+2),
		np.linspace(ymin-.5, ymax+.5, ymax-ymin+2)
	])


# Make a plot
output_image_filename = f"triangles_{image_number}_latest.png"
plt.imsave(output_image_filename, H.T, cmap='gray', format='png')
print("Histogram saved as grey level image to", output_image_filename)