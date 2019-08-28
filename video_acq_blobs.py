#
import numpy as np
import cv2
from datetime import datetime

lFps_sec = 0 #current second
lFps_c = 0 #current fps
lFps_k = 0 #current frames
lFps_M = 0 #max fps

c_r_min = 5 #10
c_r_max = 30

# -*- coding: utf-8 -*-
"""
@author: Javier Perez
@email: javier_e_perez21@hotmail.com

"""
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 20;
 
# Filter by Color
params.filterByColor = True
params.blobColor = 255

# Filter by Area.
params.filterByArea = False
params.minArea = 100
params.maxArea = 500
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create (params)
#
def check_red_blobs (im):
 
    # Detect blobs.
    keypoints = detector.detect(im)
 
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints (im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints
ESC=27   
Mm = 0  #max matches
cFk = 0

camera = cv2.VideoCapture ('/home/jetson/Work/dataset/GOPR1415s.mp4')
#camera = cv2.VideoCapture ('/home/jetson/Work/dataset/GP011416s.mp4')

while True:
  ret, imgCamColor = camera.read()
  if ret == False:
    print ("#e: read from camera error: {}".format(ret))
    exit(1)
  #
  cFk = cFk + 1
  if imgCamColor is not None:
    #result = imgCamColor
    result = check_red_blobs (imgCamColor)
    #fps computation
    cFps_sec = datetime.now().second
    lFps_k = lFps_k + 1
    if lFps_sec != cFps_sec:
      lFps_c = lFps_k - 1
      lFps_k = 0
    if lFps_M < lFps_k:
      lFps_M = lFps_k
    lFps_sec = cFps_sec
    #print ("#i:max fps {}".format (lFps_M))
    cfpst = "FPS {}/{} f#{}".format(lFps_M, lFps_c, cFk)
    cv2.putText (result, cfpst, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, 0)
        
    cv2.imshow ('result', result)
  
    key = cv2.waitKey(1)
    if key == ESC:
        break
  else:
    print ("#w:dropping frames {}".format(cFk))

camera.release()
cv2.destroyAllWindows()
