# USAGE
# python ocr.py --image images/example_01.png 
# python ocr.py --image images/example_02.png  --preprocess blur

# import the necessary packages
from PIL import Image
import pytesseract
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
frame = imutils.resize(image, width=600)
image = frame.copy()
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

#detect circles
circles = cv2.HoughCircles (mask, cv2.HOUGH_GRADIENT, 1, 40,
    param1=100, param2=20, minRadius=10, maxRadius=0)
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
      if i[2] > c_r:
        c_x = int(i[0])
        c_y = int(i[1])
        c_r = int(i[2])

if c_r > 0:
  print("found max circle at {},{}r{}".format(c_x, c_y, c_r))
  #
  # draw the outer circle
  cv2.circle (frame,(c_x, c_y), c_r,(0,255,0),2)
  # draw the center of the circle
  cv2.circle (frame,(c_x, c_y),2,(0,0,255),3)
  # draw the outer circle
  cv2.circle (mask,(c_x, c_y),c_r,(0,255,0),2)
  # draw the center of the circle
  cv2.circle (mask,(c_x, c_y),2,(0,0,255),3)
  #save cropped image
  uw = int(c_r/2 + c_r/5)
  uh = int(c_r/2 + c_r/5)
  image = image[c_y - uh:c_y + uh, c_x - uw:c_x + uw]
  cv2.imwrite ("image.png", image)

# load the example image and convert it to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Image", gray)

# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
	gray = cv2.medianBlur(gray, 3)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# show the output images
# cv2.imshow("Image", image)
#cv2.imshow("Output", gray)
#cv2.waitKey(0)

cv2.imwrite ("frame.png", frame)
cv2.imwrite ("mask.png", mask)
