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

# img = mpimg.imread('../test_images/img0019.jpg')
img1 = mpimg.imread('../test_images/test1.jpg')
img2 = mpimg.imread('../test_images/test2.jpg')
img3 = mpimg.imread('../test_images/test3.jpg')
img4 = mpimg.imread('../test_images/test4.jpg')
img5 = mpimg.imread('../test_images/test5.jpg')
img6 = mpimg.imread('../test_images/test6.jpg')


# ystart = 400
# ystop = 656
# scale = 1.5

ystart1 = 350
ystop1 = 500
scale1 = 1

ystart2 = 400
ystop2 = 600
scale2 = 1.5

ystart3 = 500
ystop3 = 700
scale3 = 2.5

vehicles = detected_vehicles(img1)
vehicles.update_params(ystart1, ystop1, scale1, ystart2, ystop2, scale2, ystart3, ystop3, scale3, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)


white_output = 'test_output_video.mp4'
clip1 = VideoFileClip("project_video.mp4", audio = False)
white_clip = clip1.fl_image(vehicles.process_img_heatmap) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

# HTML("""<video width="960" height="540" controls> <source src="{0}"> </video>""".format(white_output))

    
# draw_img = vehicles.process_img_heatmap(img4)
# plt.figure()
# plt.imshow(draw_img)

# draw_img1 = vehicles.process_img_heatmap(img1)
# draw_img2 = vehicles.process_img_heatmap(img2)
# draw_img3 = vehicles.process_img_heatmap(img3)
# draw_img4 = vehicles.process_img_heatmap(img4)
# draw_img5 = vehicles.process_img_heatmap(img5)
# draw_img6 = vehicles.process_img_heatmap(img6)

# plt.figure()
# plt.subplot(321)
# plt.imshow(draw_img1, cmap = 'gray')
# plt.subplot(322)
# plt.imshow(draw_img2, cmap = 'gray')
# plt.subplot(323)
# plt.imshow(draw_img3, cmap = 'gray')
# plt.subplot(324)
# plt.imshow(draw_img4, cmap = 'gray')
# plt.subplot(325)
# plt.imshow(draw_img5, cmap = 'gray')
# plt.subplot(326)
# plt.imshow(draw_img6, cmap = 'gray')

# plt.show()