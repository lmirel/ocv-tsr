#
# https://www.design-reuse.com/articles/41154/traffic-sign-recognition-tsr-system.html
# https://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html#sphx-glr-auto-examples-edges-plot-circular-elliptical-hough-transform-py

import numpy as np
import cv2
import time
from datetime import datetime
#from tsrframeocr import TSRFrameOCR

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

#
show_display = True
#
show_fps = True
#
lFps_sec = 0 #current second
lFps_c = 0 #current fps
lFps_k = 0 #current frames
lFps_M = 0 #max fps
lFps_T = 0 #tot
lFps_rS = 0 #running seconds

c_r_min = 12 #10
c_r_max = 50 #50

# define range of white color in HSV
sensitivity = 15
lower_white = np.array([0, 0, 255 - sensitivity])
upper_white = np.array([255, sensitivity, 255])
#this is red
lower_col1 = np.array ([0,  50,  50])
upper_col1 = np.array ([10, 255, 255])
#
lower_col2 = np.array ([170, 50,  50])
upper_col2 = np.array ([180, 255, 255])
#
b_th = 70   #black_threshold 
kFot = 0    #count of saved frames
#
def check_red_circles (image):
  hsv = cv2.cvtColor (image, cv2.COLOR_BGR2HSV)
  # construct a mask for the color "green", then perform
  # a series of dilations and erosions to remove any small
  # blobs left in the mask
  # lower mask (0-10)
  #mask0 = cv2.inRange (hsv, lower_white, upper_white)
  mask0 = cv2.inRange (hsv, lower_col1, upper_col1)
  # upper mask (170-180)
  mask1 = cv2.inRange (hsv, lower_col2, upper_col2)
  # join my masks
  cmask = mask0 + mask1
  #
  print("#i:mask shape {}".format(cmask.shape))
  #image_gray = cv2.cvtColor (cmask, cmask, cv2.COLOR_BGR2GRAY)
  #print("#i:gray shape {}".format(image_gray.shape))
  #image_gray = color.hsv2gray (cmask)
  #image_gray = cv2.cvtColor (cmask, cv2.COLOR_HSV2BGR)
  #image_gray = cv2.cvtColor (image_gray, cv2.COLOR_BGR2GRAY)
  # Load picture and detect edges
  edges = canny (cmask, sigma=3, low_threshold=10, high_threshold=50)
  
  # Detect two radii
  hough_radii = np.arange (20, 35, 1)
  hough_res = hough_circle (edges, hough_radii)
  
  # Select the most prominent 3 circles
  accums, cx, cy, radii = hough_circle_peaks (hough_res, hough_radii, total_num_peaks=3)
  
  # Draw them
  #image = color.gray2rgb (image)
  for center_y, center_x, radius in zip (cy, cx, radii):
      circy, circx = circle_perimeter (center_y, center_x, radius, shape=image.shape)
      image[circy, circx] = (220, 20, 20)
  return image
  #
#
ESC=27   
Mm = 0  #max matches
cFk = 0

#tsrfocr = TSRFrameOCR ()
#tsrfocr.start ()

camera = cv2.VideoCapture ('/home/jetbot/Work/dataset/GOPR1415s.mp4')
#camera = cv2.VideoCapture ('/home/jetson/Work/dataset/GP011416s.mp4')

while True:
    try:
        ret, imgCamColor = camera.read()
        if ret == False:
            print ("#e: read from camera error: {}".format(ret))
            exit(1)
        #
        cFk = cFk + 1
        if imgCamColor is not None:
            result = imgCamColor
            if cFk == 360:
              result = check_red_circles (imgCamColor)
            #fps computation
            cFps_sec = datetime.now().second
            lFps_k = lFps_k + 1
            if lFps_sec != cFps_sec:
              lFps_c = lFps_k - 1
              lFps_k = 0
              lFps_rS = lFps_rS + 1 #increment seconds - we assume we get here every second
            if lFps_M < lFps_k:
              lFps_M = lFps_k
            lFps_sec = cFps_sec
            if show_fps == True:
                #print ("#i:max fps {}".format (lFps_M))
                if lFps_rS > 0:
                  aFps = int (cFk / lFps_rS)
                else:
                  aFps = 1
                cfpst = "FPS M{} / A{} / C{} / T{} p{} s{}".format (lFps_M, aFps, lFps_c, cFk, kFot, lFps_rS)
                #print ("perf {}".format(cfpst))
                #
                if show_display == True:
                    cv2.putText (result, cfpst, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, 0)
                else:
                    if (cFk % 50) == 0:
                        print (cfpst)
            #   
            if show_display == True:
                cv2.imshow ('result', result)
                #
                key = cv2.waitKey(1)
                if key == ESC:
                    break
                if cFk == 360:
                  time.sleep(10)
            #
            #tsrfocr.update()
        else:
            print ("#w:dropping frames {}".format(cFk))
        #
    except KeyboardInterrupt:
        estop = True
        break
#
#print ("#w:dropping {} frames".format (tsrfocr.count()))
#
camera.release()
cv2.destroyAllWindows()
