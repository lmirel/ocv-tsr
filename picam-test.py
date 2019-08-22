#!/usr/bin/env python

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from datetime import datetime
import time
import cv2
import numpy as np
import sys
from PIL import Image

#from msvcrt import getch
#import curses

print ("#i:initializing camera..")
# initialize the camera and grab a reference to the raw camera capture
picw = 640
pich = 480
camera = PiCamera()
time.sleep (2) #give camera time to warm up
camera.resolution = (picw, pich)
camera.framerate = 30
#camera.rotation = 180
rawCapture = PiRGBArray (camera, size=(picw, pich))

print ("#i:started processing camera stream..")
estop = False
while not estop:
    lower_col1 = np.array ([0,  50,  50])
    upper_col1 = np.array ([10, 255, 255])
    #
    lower_col2 = np.array ([170, 50,  50])
    upper_col2 = np.array ([180, 255, 255])
 
    for frame in camera.capture_continuous (rawCapture, format="bgr", use_video_port=True):
      try:
        #rawCapture.seek(0)
        image = frame.array
        #
        blurred = cv2.GaussianBlur (image, (11, 11), 0)
        hsv = cv2.cvtColor (blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        # lower mask (0-10)
        mask0 = cv2.inRange (hsv, lower_col1, upper_col1)
        # upper mask (170-180)
        mask1 = cv2.inRange (hsv, lower_col2, upper_col2)
        # join my masks
        cmask = mask0 + mask1
        #
        cmask = cv2.erode (cmask, None, iterations=2)
        cmask = cv2.dilate (cmask, None, iterations=2)
        #iname = "./raw/mask-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        #cv2.imwrite (iname, cmask)
        iname = "./raw/image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        #cv2.imwrite (iname, image)
        print ("#i:saving frame {}".format (iname))
        #
        rawCapture.truncate(0)
        #
      except KeyboardInterrupt:
        estop = True
        break
    #time.sleep (0.01)
