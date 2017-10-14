*this is an assignment from Udacity, the end lesson*

In this project, my goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4),  Check out the writeup for more detials. 


**The goals / steps of this project are the following:**

Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
Optionally, I also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
Note: for those first two steps I normalize the features and randomize a selection for training and testing.
Implement a sliding-window technique and use  trained classifier to search for vehicles in images.
Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
Estimate a bounding box for vehicles detected.
These example images come from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself. 

Some example images for testing are listed in my pipeline on single frames are located in the test_images folder.  The video called project_video.mp4 is the video my pipeline should worked on.

