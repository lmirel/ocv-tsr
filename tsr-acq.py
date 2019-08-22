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

def hisEqulColor(img):
    ycrcb = cv2.cvtColor (img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split (ycrcb)
    #print (len(channels))
    cv2.equalizeHist (channels[0], channels[0])
    #clahe = cv2.createCLAHE (clipLimit = 2.0, tileGridSize = (8, 8))
    #channels[0] = clahe.apply (channels [0])
    cv2.merge (channels, ycrcb)
    cv2.cvtColor (ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

print ("#i:initializing camera..")
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
time.sleep (2) #give camera time to warm up
camera.resolution = (1280, 720)
camera.framerate = 32
camera.rotation = 180
rawCapture = PiRGBArray(camera, size=(1280, 720))

#stdscr = curses.initscr()
#curses.cbreak()
#stdscr.keypad(1)

#stdscr.addstr(0,10,"Hit 'q' to quit")
#stdscr.refresh()

#key = ''

if len(sys.argv) == 1:
   hue_value1 = 1
   hue_value2 = 1
else:
   for a in sys.argv[1:]:
      hue_value1 = (int(a))
      hue_value2 = (int(a))

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
        rawCapture.seek(0)
        image = frame.array
        #mimage = np.fromfile (rawCapture, dtype=np.uint8)
        #print (image)
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
        #hisimg = hisEqulColor (image)
        #image = hisimg
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
        #cmask = cv2.inRange (hsv, lower_col, upper_col)
        cmask = cv2.erode (cmask, None, iterations=2)
        cmask = cv2.dilate (cmask, None, iterations=2)
        #cv2.imwrite ("mask.png", mask)
        #detect circles
        circles = cv2.HoughCircles (cmask, cv2.HOUGH_GRADIENT, 1, 60,
                  param1=100, param2=20, minRadius=30, maxRadius=200)
        #process circles
        c_x = 0
        c_y = 0
        c_r = 0
        cidx = 1
        spdv = 0
        spdf = 0
        if circles is not None:
          #circles = np.uint16 (np.around (circles))
          for i in circles[0,:]:
            # draw the outer circle
            #cv2.circle (image, (i[0], i[1]), i[2], (0,255,0), 2)
            #
            #cv2.imwrite ("frame.png", frame)

            #if i[2] > c_r:
            #c_x = int(i[0])
            #c_y = int(i[1])
            #c_r = int(i[2])
            #uw = int(c_r * 80 / 100)
            #uh = int(c_r * 65 / 100)
            #image = frame
            #image = image[c_y - uh:c_y + uh, c_x - uw:c_x + uw]
            #mask  = mask[c_y - uh:c_y + uh, c_x - uw:c_x + uw]
            #
            #image = cv2.imdecode (frame, 1)
            iname = "./raw/image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
            cv2.imwrite (iname, image)
            print ("#i:saving frame {}".format (iname))
        #snip
        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #color_mask = cv2.inRange(hsv, lower_col, upper_col)
        #result = cv2.bitwise_and (image, image, mask=cmask)
        """
        cv2.imshow("Camera Output", image)
        #cv2.imshow("Clahe", cl1)
        #cv2.imshow("Threshold", th2)
        cv2.imshow("EQC", hisimg)
        cv2.imshow("HSV", hsv)
        cv2.imshow("Color Mask", cmask)
        cv2.imshow("Final Result", result)
        """
        rawCapture.truncate(0)
        """
        #key = stdscr.getch()
        k = cv2.waitKey(5) #& 0xFF
        #print ("key: ", k)
        if "q" == chr(k & 0xff):
            estop = True
            break
        if "p" == chr(k & 0xff):
            hue_value = hue_value - 1
            break
        if "n" == chr(k & 0xff):
            hue_value = hue_value + 1
            break
        """
      except KeyboardInterrupt:
        estop = True
        break
    time.sleep (1)

#curses.endwin()
