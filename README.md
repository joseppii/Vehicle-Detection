# Vehicle Detection Project Report

The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/test_images.png
[image2]: ./output_images/hog_features.png
[image3]: ./output_images/norm_features.png
[image4]: ./output_images/test1.jpg
[image5]: ./output_images/test2.jpg
[image6]: ./output_images/test3.jpg
[image7]: ./output_images/test4.jpg
[image8]: ./output_images/test5.jpg
[image9]: ./output_images/test6.jpg
[video1]: ./output_video.mp4

## Implementation

This project was implemented using the `car_detector.ipynb` jupyter notebook . The resulting images can be found under the test images folder. The resulting video can be found under the root folder.

### 1. Trainning data

The data used throught this project is a combination of vehicle and non-vehicle examples from the GTI vehicle image database and the KITTI vision benchmark suite, supplied as part of this project. They are located in the data folder under two separate folders, one per class. Data is loaded into two arrays, car and notcars. Section one of the IPython notebook contains the relevant code. The complete array contains 8792  cars and 8968 non-car images. A sample from the `vehicle` and `non-vehicle` classes can be seen in the figure below.
![alt text][image1]

## 2. Feature extraction

### 2.1 Histogram of Oriented Gradients (HOG)

The code for this step is contained in the second section of the IPython notebook `car_detector.ipynb`. HOG features were estimated using `skimage.hog()` . A wrapper method was created, `get_hog_features()` that returns the estimated features and an image visualization, if specified. The parameters used were deduced after experimentation. The images were converted from to `YCrCb`. An example using the HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` can be seen below:

![alt text][image2]

### 2.2 Binned color features

Binned color features are implemented within `bin_spatial()`

### 2.3 Color histogram features

Color histogram features are implemented within `color_hist()`

### 2.4 Combined features

The methods described in the previous three sections are combined within `extract_features()`. In addition to that, HOG features can be calculated for each channel individually or for all three channels combined. The later was the option that gave better results. In the example seen in the figure below, the extracted features were also normalized. The same parameters were used as in the previous sections. Spatial size and histogram bin were both set to 32.

![alt text][image3]

#### 3. Classifier trainning and optimization

A linear SVM was trainned using the existing data. 20% of the data was used as a trainning set. The data was split using `sklearn.cross_validation.train_test_split()`. The initial test accuracy of the SVM was 0.9885. However, using `sklearn.model_selection.GridSearchCV()` an optimal value of C was obtained. Using this value the accuracy of the SVM was increased to 0.991. The trainned SVM, along with the parameters used was saved into a pickle file (`classifier.p`).

### 4. Sliding Window Search

The sliding window implementation is described in section four of the IPython notebook. In particular, within the `find_cars()` method. This method returns a list of all the bounding boxes detected using the previously described feature extraction method. This method, in addition to the parameters required for feature detection, it also takes a scale parameter and a set of pixel coordinates for the y-axis. The pixel coordinates are used to restict the search area and the scale parameter to scale the search box accordingly.
After experimenting with various values(values from 1.1 to 3 where tested, with a step of 0.1). It was found that greater scales worked better for detecting larger objects which mostly appear at the lower part of the screen. Hence, three search areas were defined, each one with a different scale. The following search vector was specified:

    search_vec = [(350, 500, 1.3), (400, 650, 1.8), (500, 700, 2.5)]

The search was applied on all images found within the test_images folder. For each test image a heatmap was constructed and thresholded. Individual cars were detected using `scipy.ndimage.measurements.label()`. A bounding box was drawn for each detection. The heatmap for each image, as well as the corresponding bounding box can be seen in the figures below:

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]

### 5. Video Pipeline

The video pipeline used consists of three stages:

    * find_cars()
    * filter_bboxes()
    * draw_boxes()

During the first stage, a search was performed on three scales, in three different overlapping regions within each frame, using YCrCb 3-channel HOG features, plus spatially binned color and histograms of color in the feature vector.
During the second stage the previously generated bounding box list was filtered.
During the third stage the bounding boxes were drawn. The pipeline is encapsulated within `car_detector()`. The method takes a single image/frame as an input parameter and returns an image with the bounding boxes overlayed. The resulting video can be seen here:

[![CarDetector](https://i9.ytimg.com/vi/zwdcZViScFE/hqdefault.jpg?sqp=CPiI8dwF&rs=AOn4CLCVJ8B4bJa3IjkbSklIsqdtXTgGHg)](https://www.youtube.com/embed/zwdcZViScFE)

### 6.Discussion

The most obvious problem is related to the resolution of the algorithm. When two cars are very close to each other or overlapping, particularly at the bottom of each frame, were the scale is quite large, are detected as one. One possible way of going around this, is to use a different algorithm for detecting the car and a different for tracking it across frames. The HOG-detector should be used to detect a car or re-detect it, if lost. In this way one or more cars can be detected simultaneously, even if they partially overlap. When a car cannot be tracked, the HOG detector could be re-used. The tracking algorithm should not be a block algorithm.

In addition to that, the algorithm is computationaly very expensive, yielding an anacceptably low framerate for real time applications. Restricting its application to every X number of frames would remedy this problem. Given that an acceptable framerate is 30Fps, X should be set as X = 30 / algorithm_fps.

The method that computes the bounding boxes is currently limited to a video with a vertical resolution of 720. The method should be modified to accept any video size.

In order to combine this detector with the camera calibration project, it should be refactored into a class. The lane coordinates can also be used to restrict the search area in the x-axis, yielding in a higher framerate.

The HOG detector operates on blocks of pixels, therefore it should run significantly faster on a GPU, if such an implementation is available.