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
		self.speed = 0
		self.kFin = 0
		self.kFot = 0

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
					kTS = "{}_{}".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"), self.kFot)
					# keep looping infinitely until the thread is stopped
					c_r = int (fs.shape[0] / 2)
					c_x = c_r
					c_y = c_r
					#print("#i:OCRth:process frame {}x{}r{} name {}".format (c_x, c_y, c_r, iname))
					spd = tesserocr.image_to_text (Image.fromarray (fs)).strip("\n\r")
					if spd.isnumeric ():
						self.speed = int (spd)
						#
						iname = "./raw/thd-image-{}.png".format (kTS)
						cv2.imwrite (iname, fs)
						print ("speed: {}kph on {}".format (spd, iname))  # print ocr text from image
					 	#exit if we found 2 similar speeds
						#if self.speed > 0 and self.speed == int (spd):
						#	break
					"""
					#turn image gray for OCR
					gray = cv2.cvtColor (fs, cv2.COLOR_BGR2GRAY)
					iname = "./raw/thd-image-{}-grey.png".format (kTS)
					ret, gray = cv2.threshold (gray, b_th, 255, 0)
					#cv2.imwrite (iname, gray)
					#also get the MASK
					fs = self.frame_list.pop (0)
					iname = "./raw/thd-image-{}-mask.png".format (kTS)
					cv2.imwrite (iname, fs)
					#define image segments % width
					irange = [55, 60, 65, 70, 75, 80, 85]
					uw = int(c_r * 80 / 100)
					uh = int(c_r * 65 / 100)
					for irg in irange:
						uw = int(c_r * irg / 100)
						#print ("max circle at {},{}r{} / image size: {}x{}".format(c_x, c_y, c_r, uw*2, uh*2))
						image = gray.copy()
						image = image[c_y - uh:c_y + uh, c_x - uw:c_x + uw]
						#mask  = mask[c_y - uh:c_y + uh, c_x - uw:c_x + uw]
						#iname = "./raw/thd-image-{}.png".format (datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
						#cv2.imwrite (iname, image)
						#
						#iname = "image-{}.png".format(cidx)
						#cv2.imwrite (iname, image)
						#
						#use tesserocr
						spd = tesserocr.image_to_text (Image.fromarray (image)).strip("\n\r")
						if spd.isnumeric ():
							print ("speed: {}kph on {}".format (spd, iname))  # print ocr text from image
						 	#exit if we found 2 similar speeds
							if self.speed > 0 and self.speed == int (spd):
								break
							self.speed = int (spd)
					"""
			#break
			time.sleep(0.0001)
		# if the thread indicator variable is set, stop the thread
		#if self.stopped:
		#	return
		print("#i:end save thread")

	def stop(self):
		# indicate that the thread should be stopped
		print("#i:stop save thread")
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
