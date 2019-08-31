#
import numpy as np
import cv2
from datetime import datetime
from tsrframeocr import TSRFrameOCR

lFps_sec = 0 #current second
lFps_c = 0 #current fps
lFps_k = 0 #current frames
lFps_M = 0 #max fps

c_r_min = 5 #10
c_r_max = 40 #50

# define range of white color in HSV
sensitivity = 20
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([255,sensitivity,255])
#lower_white = np.array([0, 0, 200])
#upper_white = np.array([0, 0, 255])
#this is red
lower_col1 = np.array ([0,  50,  50])
upper_col1 = np.array ([10, 255, 255])
#
lower_col2 = np.array ([170, 50,  50])
upper_col2 = np.array ([180, 255, 255])

ESC = 27

camera = cv2.VideoCapture ('/home/jetson/Work/dataset/GOPR1415s.mp4')
#camera = cv2.VideoCapture ('/home/jetson/Work/dataset/GP011416s.mp4')

while True:
    ret, imgCamColor = camera.read()
    if ret == False:
        print ("#e: read from camera error: {}".format(ret))
        exit(1)
    image = imgCamColor
    #snip
    #create a CLAHE object (Arguments are optional).
    #clahe = cv2.createCLAHE (clipLimit=2.0, tileGridSize=(8,8))
    #im1 = np.uint16(image)
    #cl1 = clahe.apply(im1)
    ##
    #m_img = cv2.medianBlur (image,5)
    #rt, th1 = cv2.threshold (m_img, 180, 255, cv2.THRESH_BINARY)
    #th2 = cv2.inpaint (m_img, th1, 9, cv2.INPAINT_TELEA)
    #rv, th2 = cv2.threshold (image, 12, 255, cv2.THRESH_BINARY)
    #blurred = cv2.GaussianBlur (th2, (11, 11), 0)
    #
    blurred = cv2.GaussianBlur (image, (11, 11), 0)
    hsv = cv2.cvtColor (blurred, cv2.COLOR_BGR2HSV)
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
    #cmask = mask0
    #
    #cmask = cv2.inRange (hsv, lower_col, upper_col)
    cmask = cv2.erode (cmask, None, iterations=2)
    cmask = cv2.dilate (cmask, None, iterations=2)
    #snip
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #color_mask = cv2.inRange(hsv, lower_col, upper_col)
    result = cv2.bitwise_and (image, image, mask=cmask)
    
    cv2.imshow("Camera Output", image)
    #cv2.imshow("Clahe", cl1)
    #cv2.imshow("Threshold", th2)
    #cv2.imshow("EQC", hisimg)
    cv2.imshow("HSV", hsv)
    cv2.imshow("Color Mask", cmask)
    cv2.imshow("Final Result", result)
    
    #key = stdscr.getch()
    key = cv2.waitKey(1)
    if key == ESC:
        break

camera.release()
cv2.destroyAllWindows()
