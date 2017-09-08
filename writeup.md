
# Vehicle Detection Project

**Author: Igor Passchier**

**email: igor.passchier@tassinternational.com**


## Goals

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## References
* The code for the project can be found in the jupyter [notebook](Vehicle-Detection.ipynb)
* All example images can be found in the [output_images](output_images) folder
* Also the final [video](output_images/result.mp4) can be found in this directory

[//]: # (Image References)
[hogcar]: ./output_images/hogcar.png
[hognocar]: ./output_images/hognocar.png
[hsvcar]: ./output_images/hsvcar.png
[histcar]: ./output_images/histcar.png
[scaledcar]: ./output_images/scaledcar.png
[incorrectcars]: ./output_images/incorrectcars.png

[bbox1]: ./output_images/bbox1.png
[bbox2]: ./output_images/bbox2.png
[bboxed]: ./output_images/bboxed.png
[identified]: ./output_images/identified.png
[heatmap]: ./output_images/heatmap.png
[final]: ./output_images/final.png

[result]: ./output_images/result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points 

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

This document. References to the code are made to the sections in the jupyter [notebook](Vehicle-Detection.ipynb)

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The notebook starts with some imports and helper functions for plotting images. In the second section "Create training data file names" I have a generic function to create filenames based on a regexp. Anything with `non-vehicles` in the name is treated as not being a vehicle, while all others are considered cars. By default, I shuffle the data, and the function also allows to limit the number of images (useful for testing further on in the code with a limited set of images.
The function `read_imnage` actually reads the file, and applies the required scaling in case of a .png file

In most cases, I created seperate blocks with the function definition, followed by a code block to execute the code and to play around with the parameters. In this case, I only print the number of images found, abouth 9000 cars and 9000 non-cars.

In the section "Hog features extraction" I defined a generic function to call skimage.hog(), and return the feature fector and hog_image.

#### 2. Explain how you settled on your final choice of HOG parameters.

I have played around with various values of orientation and pix_per_cell. These have a clear effect on the output hog image. I tried to make them small enough to still see clear features. Increasing them gives more details, but of course also large feature vector, and thus slower feature extraction and vehicle detection with the final pipeline. 

I settled for orient=11 and pix_per_cell=8 as an acceptable comprimize. Later, during training of the classifier I made small variations around those values to see the effect on the speed and quality of the classifier.

I did something similar for the color spaces. But also here, the final decission is only made during training of the classifier.

Below are examples for the Y channel of YCrCb color space, with the above mentioned values for the other parameters, for a car and non-car respectively. The hog images are clearly different, which should be good for the classifier.


![alt text][hogcar]

![alt text][hognocar]


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Spatial binning is implemented in the "Spatial binning" section, while color histogramming is implemented in "Color histogramming". I played around with color spaces and looked visually to the differences in the histograms of the features. YUV color space gave the most pronounced differences, so that is what I decided to use.
Below are the test images for spatial binning and color histogramming, respectively.


![alt text][hsvcar]

![alt text][histcar]

In the final feature set I used for the training, I decided to keep only two types of features: hog, and color histogram.

In the section "Feature extraction" I implement the functions to extract the features from one or more images. `single_img_features` extracts the features of a single image, with all relevant parameters of the algorithm as function parameters. The defaults in the code are the values I finally used.

The function `extract_features` reads all images based the file names and extracts and returns all features.

In the second code block, all features of all training data are determined. In line 10, also the scaler of the features is trained. This is clearly necessay, because the values for the different features fluctuates heavenly, see below.

![alt text][scaledcar]

In the section "Create training data", all steps are brought together:

* line 1-19: set all parameters
* line 26-47: extract scaled features
* line 51-53: splt the data in training and test data
* line 56-61: provide some statistics over the whole process

In the section "Train the model" I have implemented the training functions. I have started with a linear SVC model and used to investigate further the effects of hog parameters and color spaces. Afterwards, when I settled on the feature extraction parameters, I used GridSearchCV with SVC to optimize the parameters. As the parameter space is very large, I started by variying a single parameter in big steps, then with smaller steps around the optimium. Finally, a used a full parameter search for C and gamma in 3 steps each. Initially, I experimented with kernels `rbf` and `linear`, but found that the `rbf` kernel give better results. For the C parameter, I searched between 1-1000, and for gamma between 0.1-10 * 1/2000 (2000 is about the number of features in my feature vector. 1/number of features is the default, so therefore I searched around that value.

I got the best results for 
* kernel = rbf
* C=10
* gamma=0.0005

Resulting in a test accuracy of 0.9977

In the next section, I created a classifier function `classify_image`, taking an input image and returning the classification prediction.

In the section "Inspect incorrectly classified test images", I have inspected the images that remained incorrectly classified. Especially for the incorrectly classified cars it is clear that this are mainly car pictures taken from the rear-left. To fix this, a larger training set with these type of images are required.

![alt text][incorrectcars]



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is split in several function. I have made to versions: first an approach where every window is treated as an individual image, later a second set of functions where the hog feature extraction is mixed with the sliding window search to optimnize the pipeline.

`slide_window` generated windows of a specific size (`xy_window`) for a specific region of an image (image size is `shape`, region is defined by `x_start_stop` and `y_start_stop`. Also the overlap can be specified.

A second function `create_winows` uses this function to generate the windows in different sizes.

In the next section, I implementeda utility function `draw_boxes` to plot bounding boxes on top of an image in specific colors. This is used in the next section to investigate the windows to use. Below are individual images for the three sizes of windows (overlap set to 0, to clearly see the size), and the complete set of windows (here, the overlap has been set to the real value of 0.75.

![][bbox1]

![][bbox2]

For tuning the window sizes, I have created 6 additional snapshots from the training video, with cars at different distances. I also did some tuning by running the full pipelines with different scales and overlaps.

In "Perform the search in all defined windows" a function `search_windows` is implemented, looping over all windows, running the classifier, and storing the identified windows.

The `search_one` function is wrapper, where all parameters are taken from global parameters, which makes it easier to use in the pipeline. Later, I replaced this function in the pipeline by the optimized version where hog_feature extraction and sliding window is mixed.

Below is an example of the identified windows on a sample image

![][bboxed]

Next step is to combine the pixels in all hot windows, apply a threshold, and generate labels for the resulting areas. This is done in section "Heat mapping and labels". The images below show the heatmap, labels, and final boundingbox of the identied cars on a sample image.

![][heatmap]

![][identified]



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In "Optimized function to determine hot windows" I have implemented a function `find_cars` This function scans the image for all windows of a specific scale, by combining feature extraction and classification.

In line 12-23 the relevant area of the image is selected, scaled and color converted.
In line 38-40 the hog features for the complete area are determined.
In line 42-66, the image area is scanned with the window, features are extracted, and classification is performed.

In line 67, I draw the result on the image. I trick the system by setting the `all` parameter to `True`, which provides me an image overlayed with all the windows being scanned. This is useful for debugging, not for the final result.

The `search_one_fast` function uses the find_cars function at different scales and different areas, and combines the hot windows. The function allows to use 4 different windows size, which can be turned on and off via changing the (hard_coded) True and False values (e.g. in line 8, 17, etc.). The `search_one_fast` function is a drop in replacement of the `search_one` function defined earlier, allowing easy switching in the pipeliine between the two.

Development performance was optimized by making a save and restore function for the classifier and global parameter definition, see section "Functions to save and restore the full state". These are based on `pickle.dump` and `pickle.load`

Then, in "Pipeline definition" I implement an initialization function, to (re)set all global parameters used, and a pipeline function.

The pipeline function contains 2 optimizations:
1. To optimize tuning speed, a parameter `ratio` is used. This is the ratio between number of frames in the video, and number of frames analysed. So if ratio=1, all frames are analysed, but if ratio=25, only 1 frame per second (25 Hz video) is analysed. In that way, I can go through the complete video and sample only a subset of the images. This greatly speeds up tuning times
2. The second optimization is to reduce false positives, and to stabalize the detected bounding boxes. The global parameter `roling` determines over how many heatmaps from consequetive frames an integrated heatmap is generated. A threshold `roling_threshold` on the integrated heatmap is specified, similar to the threshold on the heatmap of the individual frames.

The roling average works best if the normal threshold on the heatmap is put to 0 (so no threshold on the individual heatmaps), and a threshold slightly larger than the number of frames averaged in the roling average. In this way, an area needs to be identified in more than one window in at least one frame, and should be identified in the majority of frames at least once. 

If the roling average is done over too many frames, the bounding box starts lagging behind. Therefore, I fixed the roling average over 5 frames, and a threshold of 6.

A example from the final pipeline is shown below


![][final]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's the resulting video [result]

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The heatmapping, labelling, and roling average have been described before.


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found the performance optimization the most difficult part. So may parameters are available, like 

* what features to use, and how many of each
* what classifier to use, and with what parameters. 
* what size and what locations to use for the sliding window

And they all work together, e.g. maybe use less features, but more windows, or the other way around. 

If a vehicle has been identified, this could be used in the next step to only search in that area in the next frame. That could speed up significantly, and also reduce false positives. New vehicles could be searched for then only in the (lower) left and right side of the frame (and then they should be large), or at the horizon (but then they should be small). I did not implement this, as it would require significant tuning, especially if you want to make it stable also for other videos.





