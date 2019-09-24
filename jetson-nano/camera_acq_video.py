import cv2
import numpy as np
 
# Create a VideoCapture object
from jetcam.csi_camera import CSICamera                                                                                                                                                                                                                                         

#capture = cv2.VideoCapture(0)
capture = CSICamera (width=1280, height=720)                                                                                                                                                                                                                                     

# Check if camera opened successfully
#if (capture.isOpened() == False): 
#  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
w=1280 #int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH ))
h=720  #int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT ))
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.VideoWriter_fourcc() does not exist
video_writer = cv2.VideoWriter("output.avi", fourcc, 25, (w, h))
# record video
while True:
    frame = capture.read()
    video_writer.write(frame)
    cv2.imshow('Video Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
video_writer.release()
cv2.destroyAllWindows()
