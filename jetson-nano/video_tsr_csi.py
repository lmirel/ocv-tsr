#
# https://www.design-reuse.com/articles/41154/traffic-sign-recognition-tsr-system.html
#
# OS needs:
# apt install libleptonica-dev libtesseract-dev tesseract-ocr
# pip3 install tesserocr
#

import numpy as np
import cv2

from datetime import datetime
#from tsrframeocr import TSRFrameOCR

import jetson.utils
import jetson.inference

import argparse
import sys
# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                           formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage())

parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)
#
show_display = True
#
show_fps = True
show_fps = True
#
lFps_sec = 0  #current second
lFps_c = 0    #current fps
lFps_k = 0    #current frames
lFps_M = 0    #max fps
lFps_T = 0    #tot
lFps_rS = 0   #running seconds
cFk = 0       #frame count

c_r_min = 5 #5 #10
c_r_max = 30 #25 #50

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
b_th = 170 #170 #70   #black_threshold 
kFot = 0    #count of saved frames
#
def check_red_circles (image):
    blurred = cv2.blur (image, (5, 5))
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
    #cmask = mask1
    result = image
    #
    #cmask = cv2.erode (cmask, None, iterations=2)
    #cmask = cv2.dilate (cmask, None, iterations=2)
    #iname = "./raw/mask-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    #cv2.imwrite (iname, cmask)
    #detect circles
    #"""
    circles = cv2.HoughCircles (cmask, cv2.HOUGH_GRADIENT, 1, 
                200, param1=100, param2=20, minRadius=c_r_min, maxRadius=c_r_max)
    #            60, param1=100, param2=20, minRadius=c_r_min, maxRadius=c_r_max)
    #process circles
    c_x = 0
    c_y = 0
    c_r = 0
    if circles is not None:
      kTS = "{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
      #iname = "/mnt/raw/img-{}-frame.png".format (kTS)
      #circles = np.uint16 (np.around (circles))
      for i in circles[0,:]:
        c_x = int(i[0])
        c_y = int(i[1])
        c_r = int(i[2]) #autocrop the 'red' circle
        #print("#i:detected circle {}x{}r{}".format(c_x, c_y, c_r))
        #print ("#i:saving frame {}".format (iname))
        #cv2.imwrite (iname, image)
        if c_r > 6 and c_x > c_r and c_y > c_r:
            #crop the image area containing the circle
            tsr_img = image.copy()
            tsr_img = tsr_img[c_y - c_r:c_y + c_r, c_x - c_r:c_x + c_r]
            #
            global kFot
            kFot = kFot + 1
            #cv2.imwrite (iname, final)
            #iname = "./raw/thd-image-{}_{}.png".format (kTS, kFot)
            #print ("#i:saved {}".format (iname))
            #cv2.imwrite (iname, gray)
            # send to OCR engine for interpretation
            tsr_imga = cv2.cvtColor (tsr_img, cv2.COLOR_BGR2RGBA)
            cuda_mem = jetson.utils.cudaFromNumpy (tsr_imga)
            #print (cuda_mem)
            class_idx, confidence = net.Classify (cuda_mem, tsr_img.shape[0], tsr_img.shape[1])
            if class_idx >= 0: # or confidence * 100) > 60:
                confi = int (confidence * 1000)
                if confi > 899:
                    # find the object description
                    class_desc = net.GetClassDesc (class_idx)
                    # save as image
                    iname = "/mnt/raw/img-{}_{}-cuda-{}_{}.jpg".format (kTS, kFot, class_desc, confi)
                    jetson.utils.saveImageRGBA (iname, cuda_mem, tsr_img.shape[0], tsr_img.shape[1])
                    # overlay the result on the image
                    print ("found sign {} {:s} - net {} fps {}".format (confi, class_desc, net.GetNetworkName (), net.GetNetworkFPS ()))
                    #iname = "/mnt/raw/img-{}_{:.0f}_{}-gray.png".format (kTS, pwp, kFot)
                    #cv2.imwrite (iname, gray)
                    iname = "/mnt/raw/img-{}_{}-ori.jpg".format (kTS, kFot)
                    cv2.imwrite (iname, tsr_img)
                    iname = "/mnt/raw/img-{}_{}-frame.jpg".format (kTS, kFot)
                    cv2.imwrite (iname, image)
                    #iname = "./raw/thd-gray-{}_{}.png".format (kTS, kFot)
        # draw the outer circle
        cv2.circle (result, (c_x, c_y), c_r, (0,0,255), 2)
    #"""
    return result
#
ESC=27   
Mm = 0  #max matches

#tsrfocr = TSRFrameOCR ()
#tsrfocr.start ()

# load the recognition network
net = jetson.inference.imageNet (opt.network, sys.argv)

camera = cv2.VideoCapture ('/mnt/cv2video-720p-20191101-140122-097784.avi')
#camera = cv2.VideoCapture ('/mnt/Work/dataset/GOPR1415s.mp4')
#camera = cv2.VideoCapture ('/home/jetson/Work/dataset/GP011416s.mp4')

s_fm = 450  #start frame

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
            #if cFk % 5 == 0:
            if cFk > s_fm:
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
            #
            #tsrfocr.update()
        else:
            print ("#w:dropping frames {}".format(cFk))
        #
    except KeyboardInterrupt:
        estop = True
        break
#
#tsrfocr.stop()
#print ("#w:dropping {} frames".format (tsrfocr.count()))
#
camera.release()
cv2.destroyAllWindows()