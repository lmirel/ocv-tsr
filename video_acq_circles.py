#
# https://www.design-reuse.com/articles/41154/traffic-sign-recognition-tsr-system.html

import numpy as np
import cv2
from datetime import datetime
from tsrframeocr import TSRFrameOCR

#
show_display = True
#
show_fps = True
#
lFps_sec = 0 #current second
lFps_c = 0 #current fps
lFps_k = 0 #current frames
lFps_M = 0 #max fps

c_r_min = 10 #10
c_r_max = 60 #50

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
    circles = cv2.HoughCircles (cmask, cv2.HOUGH_GRADIENT, 1, 
                60, param1=100, param2=20, minRadius=c_r_min, maxRadius=c_r_max)
    #            60, param1=100, param2=20, minRadius=c_r_min, maxRadius=c_r_max)
    #process circles
    c_x = 0
    c_y = 0
    c_r = 0
    if circles is not None:
      #iname = "./raw/image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
      #cv2.imwrite (iname, image)
      #print ("#i:saving frame {}".format (iname))
      #circles = np.uint16 (np.around (circles))
      kTS = "{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
      for i in circles[0,:]:
        c_x = int(i[0])
        c_y = int(i[1])
        c_r = int(i[2])
        #print("#i:detected circle {}x{}r{}".format(c_x, c_y, c_r))
        if c_x > c_r and c_y > c_r:
            c_r = int(i[2]) - 4 #'crop' the red part
            #crop the image area containing the circle
            tsr_img = image.copy()
            tsr_img = tsr_img[c_y - c_r:c_y + c_r, c_x - c_r:c_x + c_r]
            #
            #print("#i:circle size {} {}x{}r{}".format (tsr_img.shape, c_x, c_y, c_r))
            if tsr_img.shape[0] == tsr_img.shape[1]:
                # draw mask
                mask = np.full ((c_r*2, c_r*2), 0, dtype=np.uint8)  # mask is only
                cv2.circle (mask, (c_r, c_r), c_r, (255, 255, 255), -1)
                # get first masked value (foreground)
                fg = cv2.bitwise_or (tsr_img, tsr_img, mask = mask)
                # get second masked value (background) mask must be inverted
                mask = cv2.bitwise_not (mask)
                bg = np.full (tsr_img.shape, 255, dtype=np.uint8)
                bk = cv2.bitwise_or (bg, bg, mask = mask)
                # combine foreground+background
                final = cv2.bitwise_or (fg, bk)
                """
                #gray out image
                gray = cv2.cvtColor (tsr_img, cv2.COLOR_BGR2GRAY)
                #gray = tsr_img
                ret, gray = cv2.threshold (gray, b_th, 255, cv2.THRESH_BINARY)
                wpk = cv2.countNonZero (gray)
                tpk = tsr_img.shape[0]*tsr_img.shape[1]
                print ("#i:white pixels {} in {}".format(wpk, tpk))
                """
                """
                #send to OCR engine for interpretation
                #tsrfocr.save (gray)
                #crop the MASK area containing the circle
                tsr_msk = cmask.copy()
                tsr_msk = tsr_msk[c_y - c_r:c_y + c_r, c_x - c_r:c_x + c_r]
                #tsr_img.copyTo (tsr_msk, tsr_msk)
                """
                #cv2.imwrite (iname, final)
                gray = cv2.cvtColor (final, cv2.COLOR_BGR2GRAY)
                #gray = tsr_img
                ret, gray = cv2.threshold (gray, b_th, 255, cv2.THRESH_BINARY)
                #cv2.imwrite (iname, gray)
                wpk = cv2.countNonZero (gray)
                tpk = tsr_img.shape[0]*tsr_img.shape[1]
                if wpk > tpk * 70 / 100 and wpk < tpk * 80 / 100:
                    #print ("#i:white pixels {} in {}".format(wpk, tpk))
                    global kFot
                    kFot = kFot + 1
                    iname = "./raw/thd-image-{}_{}.png".format (kTS, kFot)
                    #print ("#i:saved {}".format (iname))
                    #cv2.imwrite (iname, gray)
                    #send to OCR engine for interpretation
                    tsrfocr.save (tsr_img)
                #iname = "./raw/thd-gray-{}_{}.png".format (kTS, kFot)
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
    try:
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
            if show_fps == True:
                #print ("#i:max fps {}".format (lFps_M))
                cfpst = "FPS {}/{} f#{}".format(lFps_M, lFps_c, cFk)
                #print ("perf {}".format(cfpst))
                #
                if show_display == True:
                    cv2.putText (result, cfpst, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, 0)
            #   
            if show_display == True:
                cv2.imshow('result', result)
                #
                key = cv2.waitKey(1)
                if key == ESC:
                    break
            #
            #tsrfocr.update()
        else:
            print ("#w:dropping frames {}".format(cFk))
        #
    except KeyboardInterrupt:
        estop = True
        break
#
print ("#w:dropping {} frames".format (tsrfocr.count()))
#
camera.release()
cv2.destroyAllWindows()
