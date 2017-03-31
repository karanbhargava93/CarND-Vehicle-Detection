import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from tqdm import tqdm

import joblib

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split


# Read in cars and notcars
images_cars = glob.glob('../data/vehicles/vehicles/**/*.png')
images_notcars = glob.glob('../data/non-vehicles/non-vehicles/**/*.png')
# images_notcars1 = glob.glob('../data/non-vehicles/non-vehicles/Extras/*.png')
cars = []
notcars = []

for image in tqdm(images_cars):
	cars.append(image)

for image in tqdm(images_notcars):
	notcars.append(image)

# for image in tqdm(images_notcars1):
#     notcars.append(image)

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
print('Len of cars:', len(cars))
print('Len of not cars:', len(notcars))

sample_size_cars = 8790
sample_size_notcars = 8965

ind_cars = np.random.permutation(len(cars))
ind_notcars = np.random.permutation(len(notcars))

cars = np.asarray(cars)
notcars = np.asarray(notcars)

cars = cars[ind_cars[0:sample_size_cars]]
notcars = notcars[ind_notcars[0:sample_size_notcars]]

### TODO: Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [360, None] # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

print('Car Feature:', (car_features[0].shape))

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

print('Size of X:', scaled_X.shape)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
print('Size of Y:', y.shape)

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()


# Save the SVM
d = {'svc': svc}
d['color_space'] = color_space
d['orient'] = orient
d['pix_per_cell'] = pix_per_cell
d['cell_per_block'] = cell_per_block
d['hog_channel'] = hog_channel
d['spatial_size'] = spatial_size
d['hist_bins'] = hist_bins
d['hog_feat'] = hog_feat
d['y_start_stop'] = y_start_stop
d['X_scaler'] = X_scaler

s = joblib.dump(d, '../model/svc.pkl')


# image = mpimg.imread('../test_images/test4.jpg') #'bbox-example-image.jpg')
# draw_image = np.copy(image)

# # Uncomment the following line if you extracted training
# # data from .png images (scaled 0 to 1 by mpimg) and the
# # image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255

# windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
#                     xy_window=(96, 96), xy_overlap=(0.5, 0.5))

# hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
#                         spatial_size=spatial_size, hist_bins=hist_bins, 
#                         orient=orient, pix_per_cell=pix_per_cell, 
#                         cell_per_block=cell_per_block, 
#                         hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                         hist_feat=hist_feat, hog_feat=hog_feat)                       

# window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

# plt.imshow(window_img)

# plt.show()