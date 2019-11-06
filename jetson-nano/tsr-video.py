#!/usr/bin/python3
#
#tsr-video.py --model=resnet18_e34.onnx --input_blob=input_0 --output_blob=output_0 --labels=labels.txt --video=1
#tsr-video.py --model=resnet18_e34.onnx --input_blob=input_0 --output_blob=output_0 --labels=labels.txt --video=1 --display=1

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
# pip3 install pyserial
import serial

import jetson.utils
import jetson.inference

import argparse
import sys
# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                           formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage())
#
parser.add_argument("--video", type=str, help="filename of the video to process")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")
parser.add_argument("--display", type=int, default=0, help="render stream to DISPLAY")
#
try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)
#
try:
    ser = serial.Serial ('/dev/ttyUSB0', 9600, timeout=1)
except:
    print("")
    print ('!serial port NOT accessible')
    ser = None
    #sys.exit(0)
#
# open the serial port
if ser is not None and ser.isOpen ():
    print (ser.name + ' is open...')
    #
    st = 'v'
    ser.write (st.encode ())
    st = '0000'
    ser.write (st.encode ())
    st = '    '
    ser.write (st.encode ())
#
show_display = False
if opt.display == 1:
    show_display = True
#
show_fps = True
#
lFps_sec = 0  #current second
lFps_c = 0    #current fps
lFps_k = 0    #current frames
lFps_M = 0    #max fps
lFps_T = 0    #tot
lFps_rS = 0   #running seconds
cFk = 0       #frame count
#
cs_sec = 0
cs_spd = 0
# circles identification
c_r_min = 5 #5 #10
c_r_max = 30 #30 #25 #50
# define range of red color in HSV
lower_col1 = np.array ([0,  50,  50])
upper_col1 = np.array ([10, 255, 255])
#
lower_col2 = np.array ([170, 50,  50])
upper_col2 = np.array ([180, 255, 255])
#
kFot = 0    #count of saved frames
#POV definition
# source 1280x720
# of interest povs
# xx >= 640 - right half
# yy <= 360 - top half
# xx: 320..640..960
# yy: 180..360..540
c_xx = (640 + 160) # horizontal mid-point
c_yy = 360 #c_ry # vertical mid-point
c_rx = int (360 / 2) # horizontal width (half): x-rx, x+rx
c_ry = int (360 / 2) # vertical width (half):   y-ry, y+ry
#
def write_to_7seg (val):
    if ser is None:
        return
    #never access the serial twice for the same value
    if write_to_7seg._mval == val:
        return
    #
    write_to_7seg._mval = val
    if val == -1:
        st = '    '
    else:
        st = '{:4d}'.format (val)
    #
    ser.write (st.encode ())
#
def img_subrange (img):
    #crop the image area containing the circle
    # subtract the interesting frame
    global c_xx, c_yy, c_rx, c_ry 
    return img.copy()[c_yy - c_ry:c_yy + c_ry, c_xx - c_rx:c_xx + c_rx]
#
def do_detect (cuda_mem, width, height):
    # detect objects in the image (with overlay)
    return detnet.Detect (cuda_mem, width, height, "box,labels,conf")
#
def do_ai (tsr_img, kTS, kFot, sub_img, dfy, cfy):
    width = tsr_img.shape[0]
    height = tsr_img.shape[1]
    confi = 0
    #cv2.imwrite (iname, final)
    #iname = "./raw/thd-image-{}_{}.png".format (kTS, kFot)
    tsr_imga = cv2.cvtColor (tsr_img, cv2.COLOR_BGR2RGBA)
    cuda_mem = jetson.utils.cudaFromNumpy (tsr_imga)
    # print the detections
    if dfy == True:
        detections = detnet.Detect (cuda_mem, width, height, "box,labels,conf")
        if len (detections) > 0:
            #print("detected {:d} objects in image".format(len(detections)))
            iname = "/mnt/_tsr/raw/_objs/img-{}_{}-cuda-o{}.jpg".format (kTS, kFot, 0)
            #jetson.utils.saveImageRGBA (iname, cuda_mem, width, height)
            #for detection in detections:
            #    print(detection)
            # print out timing info
            #net.PrintProfilerTimes()
            #print (cuda_mem)
        #
    if cfy == True:
        class_idx, confidence = imgnet.Classify (cuda_mem, width, height)
        confi = int (confidence * 1000)
        if class_idx >= 0 and confi > 800: # or confidence * 100) > 60:
            # find the object description
            class_desc = imgnet.GetClassDesc (class_idx)
            #print ("found sign {:d} {:s} on {:d}".format (confi, class_desc, kFot))
            # save images
            iname = "/mnt/_tsr/raw/{}/img-{}_{}-cuda-c{}.jpg".format (class_desc, kTS, kFot, confi)
            #jetson.utils.saveImageRGBA (iname, cuda_mem, width, height)
            iname = "/mnt/_tsr/raw/{}/img-{}_{}-ori.jpg".format (class_desc, kTS, kFot)
            #cv2.imwrite (iname, tsr_img)
            if sub_img is not None:
                iname = "/mnt/_tsr/raw/{}/img-{}_{}-frame.jpg".format (class_desc, kTS, kFot)
                #cv2.imwrite (iname, sub_img)
            # overlay the result on the image
            if confi > 990: # over 99% confidence
                #print ("found sign {} {:s} fps {}".format (confi, class_desc, net.GetNetworkFPS ()))
                # update the indicator
                global cs_spd   
                if class_idx == 0:#kph20    
                    cs_spd = 20 
                if class_idx == 1:#kph30    
                    cs_spd = 30 
                if class_idx == 2:#kph50    
                    cs_spd = 50 
                if class_idx == 3:#kph60    
                    cs_spd = 60 
                if class_idx == 4:#kph70    
                    cs_spd = 70 
                if class_idx == 5:#kph80    
                    cs_spd = 80 
                if class_idx == 6:#kph100   
                    cs_spd = 100    
                if class_idx == 7:#kph120   
                    cs_spd = 120    
                global cs_sec   
                cs_sec = datetime.now().second
    return confi
#
def check_red_circles (image, kTS):
    sub_img = img_subrange (image)
    # process subimage
    blurred = cv2.blur (sub_img, (5, 5))
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
    #
    result = sub_img
    #cmask = cv2.erode (cmask, None, iterations=2)
    #cmask = cv2.dilate (cmask, None, iterations=2)
    #iname = "./raw/mask-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    #cv2.imwrite (iname, cmask)
    #detect circles
    circles = cv2.HoughCircles (cmask, cv2.HOUGH_GRADIENT, 1, 
                200, param1=100, param2=20, minRadius=c_r_min, maxRadius=c_r_max)
    #            60, param1=100, param2=20, minRadius=c_r_min, maxRadius=c_r_max)
    #process circles
    c_x = 0
    c_y = 0
    c_r = 0
    if circles is not None:
      #kTS = "{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
      #iname = "/mnt/raw/img-{}-frame.png".format (kTS)
      #circles = np.uint16 (np.around (circles))
      for i in circles[0,:]:
        c_x = int(i[0])
        c_y = int(i[1])
        c_r = int(i[2]) #autocrop the 'red' circle
        #print("#i:detected circle {}x{}r{}".format(c_x, c_y, c_r))
        if True: #c_r > 6 and c_x > c_r and c_y > c_r:
            tsr_img = sub_img.copy()
            tsr_img = tsr_img[c_y - c_r:c_y + c_r, c_x - c_r:c_x + c_r]
            #
            global kFot
            kFot = kFot + 1
            #
            do_ai (tsr_img, kTS, kFot, sub_img, False, True)
            #
        cv2.circle (result, (c_x, c_y), c_r, (0,0,255), 2)
        #.for
    #.if circles is not None:
    #
    #return tsr_img
    return result
#
ESC=27   
Mm = 0  #max matches
#
camera = cv2.VideoCapture (opt.video)
#camera = cv2.VideoCapture ('/mnt/.tsr/cv2video-720p-20191101-140122-097784.avi')
#camera = cv2.VideoCapture ('/mnt/.tsr/GOPR1415s.mp4')
#camera = cv2.VideoCapture ('/mnt/.tsr/GP011416s.mp4')
#
# load the recognition network
imgnet = jetson.inference.imageNet (opt.network, sys.argv)
# load the object detection network
detnet = jetson.inference.detectNet ("ssd-mobilenet-v2", threshold=0.5)
#
s_fm = 0 #450  #start frame
write_to_7seg._mval = -2

while True:
    try:
        ret, imgCamColor = camera.read()
        if ret == False:
            print ("#e: read from camera error: {}".format(ret))
            exit(1)
        #
        cFk = cFk + 1
        if imgCamColor is not None:
            kTS = "{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
            result = imgCamColor
            #if cFk % 5 == 0:
            if cFk > s_fm:
                # on 10watt nvpmodel -m0 && jetson_clocks:
                # img_subrange 60fps
                # check_red_circles 60fps
                # subrange + classify 38fps
                # red detect + classify: approx.30fps-60fps
                ###
                # subrange + classify 1 or 91 17fps
                result = check_red_circles (imgCamColor, kTS) #img_subrange (imgCamColor) #check_red_circles (imgCamColor, kTS)
                #
                #if result is not None:
                #   do_ai (result, kTS, 1, None, True, False)
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
            #turn off sign display
            if cs_sec > 0:
                # flash the indicator 
                if cFps_sec % 2 == 0:
                    write_to_7seg (-1)
                else:
                    write_to_7seg (cs_spd)
                # stop flashing and show the last speed
                if cs_sec + 5 < cFps_sec:
                    cs_sec = 0
                    write_to_7seg (cs_spd)
            #
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
                key = cv2.waitKey (1)
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
write_to_7seg (-1)

camera.release()
if show_display == True:
    cv2.destroyAllWindows()
#
