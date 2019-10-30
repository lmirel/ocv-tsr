#!/usr/bin/env python

# import the necessary packages
import time
import cv2
import numpy as np
import sys
from jetcam.csi_camera import CSICamera
from jetcam.usb_camera import USBCamera
from datetime import datetime
#from msvcrt import getch
#import curses
cv2_display = 1
cv2_save_frame = 0

lFps_sec = 0 #current second
lFps_c = 0 #current fps
lFps_k = 0 #current frames
lFps_M = 0 #max fps

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

# initialize the camera and grab a reference to the raw camera capture
#camera = cv2.VideoCapture(0)
## CSI Camera
#camera = CSICamera(width=1280, height=720)
camera = USBCamera(width=1280, height=720, capture_device=3)

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

estop = False
while not estop:
    lower_col1 = np.array ([0,  50,  50])
    upper_col1 = np.array ([10, 255, 255])
    #
    lower_col2 = np.array ([170, 50,  50])
    upper_col2 = np.array ([180, 255, 255])

    while not estop:
#      ret, frame = camera.read()
      frame = camera.read()
      try:
        image = frame
        """
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
        hisimg = hisEqulColor (image.copy())
        image = hisimg
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
        #snip
        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #color_mask = cv2.inRange(hsv, lower_col, upper_col)
        result = cv2.bitwise_and (image, image, mask=cmask)
        #
        """
        iname = "./raw/image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
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
        cfpst = "FPS {}/{}".format(lFps_M, lFps_c)
        cv2.putText (image, cfpst, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        #
        #print ("#i:saving frame {} sec {} knt {} max {}".format (iname, cFps_sec, lFps_k, lFps_M))
        #
        if cv2_save_frame == 1:
            cv2.imwrite (iname, image)
        #
        if cv2_display == 1:
            cv2.imshow("Camera Output", image)
            #cv2.imshow("Clahe", cl1)
            #cv2.imshow("Threshold", th2)
            #cv2.imshow("EQC", hisimg)
            #cv2.imshow("HSV", hsv)
            #cv2.imshow("Color Mask", cmask)
            #cv2.imshow("Final Result", result)
     
            #rawCapture.truncate(0)
    
            #key = stdscr.getch()
            k = cv2.waitKey(1) #& 0xFF
            #print ("key: ", k)
            if "q" == chr(k & 0xff):
                estop = True
                break
      except KeyboardInterrupt:
        estop = True
        break
    #time.sleep (1)

#curses.endwin()
camera.release()
if cv2_display == 1:
    cv2.destroyAllWindows()
