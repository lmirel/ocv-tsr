from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import time
from tsrframeocr import TSRFrameOCR
#
max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
tb_radius = 'Radius'
tb_scale = 'Scale'
window_name = 'Threshold Demo'
#
def Threshold_Demo(val):
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    threshold_type = cv.getTrackbarPos (trackbar_type, window_name)
    threshold_value = cv.getTrackbarPos (trackbar_value, window_name)
    #
    c_r = cv.getTrackbarPos (tb_radius, window_name) #int(src.shape[0]/2)
    o_r = int (src.shape[0]/2)
    #print ("rad {} vs {}".format (c_r, o_r))
    #print ("src {}".format (src.shape))
    # draw mask
    mask = np.full ((o_r*2, o_r*2), 0, dtype=np.uint8)  # mask is only
    cv.circle (mask, (o_r, o_r), c_r, (255, 255, 255), -1)
    # get first masked value (foreground)
    fg = cv.bitwise_or (src, src, mask = mask)
    # get second masked value (background) mask must be inverted
    mask = cv.bitwise_not (mask)
    bg = np.full (src.shape, 255, dtype=np.uint8)
    bk = cv.bitwise_or (bg, bg, mask = mask)
    # combine foreground+background
    final = cv.bitwise_or (fg, bk)
    src_gray = cv.cvtColor (final, cv.COLOR_BGR2GRAY)
    #src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    #
    _, dst = cv.threshold (src_gray, threshold_value, max_binary_value, threshold_type )
    #cv.imshow (window_name, src)
    cv.imshow (window_name, cv.resize (dst,(240, 240)))
    wpk = cv.countNonZero (dst)
    tpk = dst.shape[0] * dst.shape[1]
    pwp = (wpk/tpk) * 100
    #print ("pwp {:.2f} tpk {} wpk {}".format(pwp, tpk, wpk))
    tsrfocr.save (dst)
#
###
parser = argparse.ArgumentParser(description='Code for Basic Thresholding Operations tutorial.')
parser.add_argument('--input', help='Path to input image.', default='stuff.png')
args = parser.parse_args()
src = cv.imread(args.input)
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
#
tsrfocr = TSRFrameOCR ()
tsrfocr.start ()
#
print ("source: {}".format (src.shape))
#
cv.namedWindow (window_name)
cv.createTrackbar (trackbar_type, window_name , 3, max_type, Threshold_Demo)
# Create Trackbar to choose Threshold value
cv.createTrackbar (trackbar_value, window_name , 0, max_value, Threshold_Demo)
# Call the function to initialize
o_r = int (src.shape[0]/2)
cv.createTrackbar (tb_radius, window_name , o_r, o_r, Threshold_Demo)
cv.createTrackbar (tb_scale, window_name , o_r, o_r*2, Threshold_Demo)
#
Threshold_Demo(0)
# Wait until user finishes program
cv.waitKey()
tsrfocr.stop()
print ("#w:dropping {} frames".format (tsrfocr.count()))
time.sleep(.5)
