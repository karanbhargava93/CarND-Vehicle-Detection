import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import joblib
import cv2
from lesson_functions import *
from moviepy.editor import VideoFileClip
from IPython.display import HTML

dist_pickle = joblib.load('../model/svc.pkl')
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

img = mpimg.imread('../test_images/img0019.jpg')
img1 = mpimg.imread('../test_images/test4.jpg')

ystart = 400
ystop = 656
scale = 1.5

vehicles = detected_vehicles(img)
vehicles.update_params(ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)


white_output = 'output_video.mp4'
clip1 = VideoFileClip("project_video.mp4", audio = False)
white_clip = clip1.fl_image(vehicles.process_img_heatmap) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

HTML("""<video width="960" height="540" controls> <source src="{0}"> </video>""".format(white_output))

    
# draw_img = vehicles.process_img_heatmap(img)
# plt.figure()
# plt.imshow(draw_img, cmap = 'gray')

# draw_img1 = vehicles.process_img_heatmap(img1)
# plt.figure()
# plt.imshow(draw_img1, cmap = 'gray')

# plt.show()