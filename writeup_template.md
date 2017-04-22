# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

[test1]: ./output_images/test1.png
[test2]: ./output_images/test2.png
[test3]: ./output_images/test3.png
[test4]: ./output_images/test4.png
[frames]: ./output_images/frames.png

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines # through # of the file called `train.py` in the code folder.  

I started by reading in all the `vehicle` and `non-vehicle` images.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the following parameters worked best. I also observed that the accuracy increased as I increased the histogram binning but it took a lot more time as well.

```
color_space = 'YCrCb'       # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9                  # HOG orientations
pix_per_cell = 8            # HOG pixels per cell
cell_per_block = 2          # HOG cells per block
hog_channel = 'ALL'         # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)     # Spatial binning dimensions
hist_bins = 32              # Number of histogram bins
spatial_feat = True         # Spatial features on or off
hist_feat = True            # Histogram features on or off
hog_feat = True             # HOG features on or off
y_start_stop = [360, None]  # Min and max in y to search in slide_window()
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG, color and spatial features. The code for this step is contained in lines #67 through #128 of the file called `train.py` in the code folder. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I determined the scale and size of the window through trial an error from the test images. They are encorporated into the `detected_vehicles` class in `lesson_functions.py` in the code folder. The values for the same are set at runtime in the file `runMe.py` in the code folder. The scales and search ranges are given below:
```
ystart1 = 350
ystop1 = 500
scale1 = 1

ystart2 = 400
ystop2 = 600
scale2 = 1.5

ystart3 = 500
ystop3 = 700
scale3 = 2.5
```

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a decent result.  Here are some example images:

![alt text][test1]
![alt text][test2]
![alt text][test3]
![alt text][test4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap. This heatmap is then added to a moving average filter which is implemented in class `detected_vehicles` with `update_heatmap` in lines `#53` to `#58` of `lesson_functions.py` in the code folder. After filtering, I thresholded that map to identify vehicle positions and used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Assuming each blob corresponded to a vehicle, I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding results:

![alt text][frames]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Setting the scale and the search regions for the windows was tough, moreover tuning the parameters for the SVM was also tough to get rid of false positives. The current pipeline fails due to varied illumination conditions, e.g. due to the change in the road. I had tried using HSV colorspace to combat the issue but didn't have much luck with it. I believe a deep learning approach would be much more elegant in this problem however that might not be real time. If the deep learning framework is to operate real time, then the processing power would have to be a lot!

