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

#import random as rng
#rng.seed(12345)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

def captch_ex(img):
    img = img
    img_final = img.copy()
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    # to manipulate the orientation of dilution ,  large x means horizonatally dilating  more,
    #    large y means vertically dilating more
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (15, 3))
    dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

    # for cv2.x.x
    # findContours returns 3 variables for getting contours
    _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # for cv3.x.x comment above line and uncomment line below

    #image, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg' 
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
    # write original image with added contours to disk
    #cv2.imshow('captcha_result', img)
    #cv2.waitKey()
    cv2.imwrite ("captcha.png", img)

#this is RED!
lower_col1 = np.array ([0,  50,  50])
upper_col1 = np.array ([10, 255, 255])
#
lower_col2 = np.array ([170, 50,  50])
upper_col2 = np.array ([180, 255, 255])

# resize the frame, blur it, and convert it to the HSV
# color space
frame = imutils.resize (image, width=600)
ori_image = frame.copy ()
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

#cv2.imwrite ("mask.png", mask)
#detect circles
circles = cv2.HoughCircles (mask, cv2.HOUGH_GRADIENT, 1, 60,
    param1=100, param2=20, minRadius=30, maxRadius=200)
#process circles
c_x = 0
c_y = 0
c_r = 0
cidx = 1
if circles is not None:
  #circles = np.uint16 (np.around (circles))
  for i in circles[0,:]:
    # draw the outer circle
    cv2.circle (frame, (i[0], i[1]), i[2], (0,255,0), 2)
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
    #crop image
    uw = int(c_r * 85 / 100)
    uh = int(c_r * 65 / 100)
    print ("max circle at {},{}r{} / image size: {}x{}".format(c_x, c_y, c_r, uw*2, uh*2))
    image = ori_image.copy()
    image = image[c_y - uh:c_y + uh, c_x - uw:c_x + uw]
    #mask  = mask[c_y - uh:c_y + uh, c_x - uw:c_x + uw]
    iname = "image-{}.png".format(cidx)
    cv2.imwrite (iname, image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #
    """#
    #--- performing Otsu threshold ---
    ret,thresh1 = cv2.threshold (gray, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    #--- choosing the right kernel
    #--- kernel size of 3 rows (to join dots above letters 'i' and 'j')
    #--- and 10 columns to join neighboring letters in words and neighboring words
    rect_kernel = cv2.getStructuringElement (cv2.MORPH_RECT, (5, 1))
    dilation = cv2.dilate (thresh1, rect_kernel, iterations = 1)
    #---Finding contours ---
    _, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    im2 = gray.copy()
    for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    iname = "contours-{}.png".format(cidx)
    cv2.imwrite (iname, im2)
    """#
    #gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #mask_inv = cv2.bitwise_not(mask)
    #gray = cv2.bitwise_and (gray, gray, mask = mask_inv)
    iname = "gray-{}.png".format(cidx)
    cv2.imwrite (iname, gray)
    #cv2.imwrite ("mask.png", mask)
    cidx = cidx + 1
    #use tesserocr
    print (tesserocr.image_to_text (Image.fromarray(gray)))  # print ocr text from image
# show the output images
# cv2.imshow("Image", image)
#cv2.imshow("Output", gray)
#cv2.waitKey(0)
