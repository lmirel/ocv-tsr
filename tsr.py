# USAGE
# python ocr.py --image images/example_01.png 
# python ocr.py --image images/example_02.png  --preprocess blur

# import the necessary packages
from PIL import Image
import tesserocr
import argparse
import cv2
import os
import numpy as np
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

#this is RED!
lower_col1 = np.array ([0,  50,  50])
upper_col1 = np.array ([10, 255, 255])
#
lower_col2 = np.array ([170, 50,  50])
upper_col2 = np.array ([180, 255, 255])

# resize the frame, blur it, and convert it to the HSV
# color space
frame = imutils.resize (image, width=600)
image = frame.copy ()
#frame = imutils.rotate(frame, angle=180)
blurred = cv2.GaussianBlur(frame, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
# lower mask (0-10)
mask0 = cv2.inRange (hsv, lower_col1, upper_col1)
# upper mask (170-180)
mask1 = cv2.inRange (hsv, lower_col2, upper_col2)
# join my masks
mask = mask0 + mask1
# construct a mask for the color "green", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
#mask = cv2.inRange(hsv, greenLower, greenUpper)
mask = cv2.erode (mask, None, iterations=2)
mask = cv2.dilate (mask, None, iterations=2)

cv2.imwrite ("mask.png", mask)
#detect circles
circles = cv2.HoughCircles (mask, cv2.HOUGH_GRADIENT, 1, 40,
    param1=100, param2=20, minRadius=20, maxRadius=200)
#process circles
c_x = 0
c_y = 0
c_r = 0
if circles is not None:
  #circles = np.uint16 (np.around (circles))
  for i in circles[0,:]:
    # draw the outer circle
    cv2.circle (frame,(i[0], i[1]), i[2],(0,255,0),2)
    #
    cv2.imwrite ("frame.png", frame)
    
    #if i[2] > c_r:
    c_x = int(i[0])
    c_y = int(i[1])
    c_r = int(i[2])

#if c_r >= 34:
# draw the outer circle
#cv2.circle (frame,(c_x, c_y), c_r,(0,255,0),2)
# draw the center of the circle
#cv2.circle (frame,(c_x, c_y),2,(0,0,255),3)
# draw the outer circle
#cv2.circle (mask,(c_x, c_y),c_r,(0,255,0),2)
# draw the center of the circle
#cv2.circle (mask,(c_x, c_y),2,(0,0,255),3)
#save cropped image
    uw = int(c_r/2 + c_r/6)
    uh = int(c_r/2)
    print ("max circle at {},{}r{} / image size: {}x{}".format(c_x, c_y, c_r, uw*2, uh*2))
    image = image[c_y - uh:c_y + uh, c_x - uw:c_x + uw]
    #mask  = mask[c_y - uh:c_y + uh, c_x - uw:c_x + uw]
    try:
      if image[0].size > 0:
        cv2.imwrite ("image.png", image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #mask_inv = cv2.bitwise_not(mask)
        #gray = cv2.bitwise_and (gray, gray, mask = mask_inv)
        cv2.imwrite ("gray.png", gray)
        #cv2.imwrite ("mask.png", mask)
        #use tesserocr
        print (tesserocr.image_to_text (Image.fromarray(gray)))  # print ocr text from image
    except:
      print ("OCR failed due to zero image size")
# show the output images
# cv2.imshow("Image", image)
#cv2.imshow("Output", gray)
#cv2.waitKey(0)
