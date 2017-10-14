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


**Basic Build Instructions**

1.Clone this repo.
2.Make a build directory: mkdir build && cd build
3.Compile: cmake .. && make
4.Run it: ./pid.
5.Tips for setting up your environment can be found.

Tips for setting up the enviroment on Mac:
For most instances of missing packages and messages regarding uWebsockets, refer to Linux and Windows troubleshooting. Below are some common issues and their solutions.

.sh files not recognized on run: Try chmod a+x for example chmod a+x install-mac.sh
missing openssl, libuv, or cmake: install-mac.sh contains the line brew install openssl libuv cmake, which will not execute properly if homebrew is not installed. To determine if homebrew is installed, execute which brew in a terminal. If a path returns it is installed, otherwise you see brew not found. Follow the guidance here to install homebrew, then try running install-mac.sh again.
If the step above does not resolve issues regarding openssl, please try the guidance provided [here], here and (https://github.com/udacity/CarND-PID-Control-Project/issues/2) and here
Issues with rootless mode in recent versions of OSx: Some recent versions of OSx have a rootless mode by default that cause some install script commands to fail, even when running as root or sudo. To disable this reboot in recovery mode (command+R), and execute csrutil disable in a terminal. After this is complete, try running the install script.
After following these steps there may be some messages regarding makefile not found or can't create symbolic link to websockets. There is likely nothing wrong with the installation. Before doing any other troubleshooting make sure that build steps (10 and 11 from Windows and Linux instructions) have been executed from the top level of the project directory, then test the installation using running the code (step 12 from Windows and Linux instructions).
