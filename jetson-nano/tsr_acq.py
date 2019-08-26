#!/usr/bin/env python

# import the necessary packages
import time
import cv2
import numpy as np
import sys
from jetcam.csi_camera import CSICamera
from datetime import datetime
#from msvcrt import getch
#import curses
cv2_display = 1
cv2_save_frame = 0

picW = 1280
picH = 720

lFps_sec = 0
lFps_k = 0
lFps_M = 0

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
camera = CSICamera(width=picW, height=picH)

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
        #detect circles
        circles = cv2.HoughCircles (cmask, cv2.HOUGH_GRADIENT, 1, 60,
                  param1=100, param2=20, minRadius=30, maxRadius=200)
        #process circles
        c_x = 0
        c_y = 0
        c_r = 0
        if circles is not None:
          #iname = "./raw/image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
          #cv2.imwrite (iname, image)
          #print ("#i:saving frame {}".format (iname))
          #circles = np.uint16 (np.around (circles))
          cidx = 1
          for i in circles[0,:]:
            image = frame.copy()
            c_x = int(i[0])
            c_y = int(i[1])
            c_r = int(i[2])
            #print("#i:detected circle {}x{}r{}".format(c_x, c_y, c_r))
            # draw the outer circle
            cv2.circle (frame, (c_x, c_y), c_r, (0,255,0), 2)
            #
            uw = c_r #int(c_r * 80 / 100)
            #uh = int(c_r * 65 / 100)
            #save image section - full height
            image = image[0:picH, c_x - uw:c_x + uw]
            #mask  = mask[c_y - uh:c_y + uh, c_x - uw:c_x + uw]
            #
            iname = "./raw/image-{}-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"), cidx)
            #cv2.imwrite (iname, image)
            #print ("#i:saving frame {}".format (iname))
            cidx = cidx + 1
            #
        #fps computation
        cFps_sec = datetime.now().second
        lFps_k = lFps_k + 1
        if lFps_sec != cFps_sec:
            lFps_k = 0
        lFps_sec = cFps_sec
        if lFps_M < lFps_k:
            lFps_M = lFps_k
            print ("#i:max fps {}".format (lFps_M))
        #
        #print ("#i:saving frame {} sec {} knt {} max {}".format (iname, cFps_sec, lFps_k, lFps_M))
        #
        
        if cv2_save_frame == 1:
            cv2.imwrite (iname, image)
        #
        if cv2_display == 1:
            cv2.imshow("Camera Output", frame)
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
