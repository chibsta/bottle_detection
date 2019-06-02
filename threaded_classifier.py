# USAGE
# python capture_video.py

# import the necessary packages

import numpy as np
import cv2
import imutils
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--record", type=str, default="no",
	help="Record a video of the camera(s)")
args = vars(ap.parse_args())

# Capture the video streams
cam1 = cv2.VideoCapture(1)
cam2 = cv2.VideoCapture(2)
cam3 = cv2.VideoCapture(3)

#set properties for camera 1
cam1.set(cv2.CAP_PROP_CONTRAST, 0.55)
cam1.set(cv2.CAP_PROP_SATURATION, 0.5)
cam1.set(cv2.CAP_PROP_HUE, 0.55)
cam1.set(cv2.CAP_PROP_BRIGHTNESS, 0.65)

#set properties for camera 2
cam2.set(cv2.CAP_PROP_CONTRAST, 0.55)
cam2.set(cv2.CAP_PROP_SATURATION, 0.5)
cam2.set(cv2.CAP_PROP_HUE, 0.55)
cam2.set(cv2.CAP_PROP_BRIGHTNESS, 0.65)

# set properties for camera 3
cam3.set(cv2.CAP_PROP_CONTRAST, 0.55)
cam3.set(cv2.CAP_PROP_SATURATION, 0.5)
cam3.set(cv2.CAP_PROP_HUE, 0.55)
cam3.set(cv2.CAP_PROP_BRIGHTNESS, 0.65)

