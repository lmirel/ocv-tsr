# import the necessary packages
from threading import Thread
import cv2
from datetime import datetime
import tesserocr
from PIL import Image
import time

"""
apt install libleptonica-dev libtesseract-dev tesseract-ocr
pip3 install tesserocr
"""

b_th = 70 #black_threshold 

class TSRFrameOCR:
	def __init__(self, **kwargs):
		self.stopped = False
		self.frame_list = []
		self.speed = 0	#current speed value
		self.kFin = 0		#frames in - counter
		self.kFot = 0		#frames out - counter

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread (target=self.update, args= ())
		t.daemon = True
		t.start ()
		return self

	def update(self):
		print("#i:start OCR thread")
		while self.stopped == False:
			if len(self.frame_list) > 0:
				fs = self.frame_list.pop (0)
				if fs is not None:
					self.kFot = self.kFot + 1
					#kTS = "{}_{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"), self.kFot)
					#print("#i:OCRth:process frame {}x{}r{} name {}".format (c_x, c_y, c_r, iname))
					spd = tesserocr.image_to_text (Image.fromarray (fs)).strip("\n\r")
					if spd.isnumeric ():
						cspeed = int (spd)
						if (cspeed % 5) == 0:	#speed value should be multiple of 5kph
							#clear list if we found 2 similar speeds
							if self.speed == cspeed: #did we just validate the same speed?
								#let's clear the work list
								self.kFot = self.kFot + len(self.frame_list)
								self.frame_list.clear ()
							#new speed is current speed
							self.speed = cspeed
						#
						#iname = "./raw/spd-image-{}.png".format (kTS)
						#cv2.imwrite (iname, fs)
						print ("speed: {}kph with {} on {} shape {}".format (self.speed, spd, iname, fs.shape))  # print ocr text from image
			#break
			time.sleep(0.0001)
		# if the thread indicator variable is set, stop the thread
		#if self.stopped:
		#	return
		print("#i:end OCR thread")

	def stop(self):
		# indicate that the thread should be stopped
		print("#i:stop OCR thread")
		self.stopped = True

	def count(self):
		# indicate that the thread should be stopped
		return len(self.frame_list)
		
	def speed(self):
		# indicate that the thread should be stopped
		return self.speed
		
	def save(self, frm):
		# return the frame most recently read
		self.frame_list.append(frm)
		self.kFin = self.kFin + 1
		#kTS = "{}_{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"), self.kFin)
		#print ("#i:OCRth:stored frame {}".format (kTS))  # print ocr text from image
