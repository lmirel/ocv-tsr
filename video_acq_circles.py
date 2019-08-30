#
import numpy as np
import cv2
from datetime import datetime
from tsrframeocr import TSRFrameOCR

lFps_sec = 0 #current second
lFps_c = 0 #current fps
lFps_k = 0 #current frames
lFps_M = 0 #max fps

c_r_min = 10
c_r_max = 50

# define range of white color in HSV
# change it according to your need !
lower_white = np.array([0,0,0])
upper_white = np.array([0,0,255])
#this is red
lower_col1 = np.array ([0,  50,  50])
upper_col1 = np.array ([10, 255, 255])
#
lower_col2 = np.array ([170, 50,  50])
upper_col2 = np.array ([180, 255, 255])
#
def check_red_circles (image):
    #blurred = cv2.GaussianBlur (image, (11, 11), 0)
    hsv = cv2.cvtColor (image, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    # lower mask (0-10)
    #mask0 = cv2.inRange (hsv, lower_white, upper_white)
    mask0 = cv2.inRange (hsv, lower_col1, upper_col1)
    # upper mask (170-180)
    #mask1 = cv2.inRange (hsv, lower_col2, upper_col2)
    # join my masks
    #cmask = mask0 + mask1
    cmask = mask0
    #
    #cmask = cv2.erode (cmask, None, iterations=2)
    #cmask = cv2.dilate (cmask, None, iterations=2)
    #iname = "./raw/mask-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    #cv2.imwrite (iname, cmask)
    #detect circles
    #"""
    circles = cv2.HoughCircles (cmask, cv2.HOUGH_GRADIENT, 1, 60,
              param1=100, param2=20, minRadius=c_r_min, maxRadius=c_r_max)
    #process circles
    c_x = 0
    c_y = 0
    c_r = 0
    if circles is not None:
      #iname = "./raw/image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
      #cv2.imwrite (iname, image)
      #print ("#i:saving frame {}".format (iname))
      #circles = np.uint16 (np.around (circles))
      for i in circles[0,:]:
        c_x = int(i[0])
        c_y = int(i[1])
        c_r = int(i[2])
        #print("#i:detected circle {}x{}r{}".format(c_x, c_y, c_r))
        tsr_img = image.copy()
        tsr_img = tsr_img[c_y - c_r:c_y + c_r, c_x - c_r:c_x + c_r]
        #send to OCR engine for interpretation
        tsrfocr.save (tsr_img)
        # draw the outer circle
        cv2.circle (image, (c_x, c_y), c_r, (0,0,255), 2)
    #"""
#
ESC=27   
Mm = 0  #max matches
cFk = 0

tsrfocr = TSRFrameOCR ()
tsrfocr.start ()

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
    result = imgCamColor
    check_red_circles (imgCamColor)
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
        
    cv2.imshow('result', result)
  
    key = cv2.waitKey(1)
    if key == ESC:
        break
  else:
    print ("#w:dropping frames {}".format(cFk))

camera.release()
cv2.destroyAllWindows()
