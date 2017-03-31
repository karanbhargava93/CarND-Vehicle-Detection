import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import joblib
import cv2
from lesson_functions import *

dist_pickle = joblib.load('../model/svc.pkl')
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

img = mpimg.imread('../test_images/test4.jpg')

ystart = 400
ystop = 656
scale = 1.5
    
draw_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
plt.imshow(draw_img)

plt.show()