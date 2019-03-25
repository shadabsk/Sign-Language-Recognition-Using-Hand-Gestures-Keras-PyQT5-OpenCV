from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from scipy.ndimage import imread
from PyQt5.QtCore import QTimer,Qt 
from PyQt5 import QtGui
from tkinter import filedialog
from tkinter import * 
import tkinter as tk
import sys
import os
import runpy
import cv2
import numpy as np
import qimage2ndarray
import win32api
import winGuiAuto
import win32gui
import win32con
import keyboard



def nothing(x):
	pass

image_x, image_y = 64,64

from keras.models import load_model
classifier = load_model('ASLModel.h5')

fileEntry=[]
for file in os.listdir("SampleGestures"):
    if file.endswith(".png"):
    	fileEntry.append(file)

def clearfunc(cam):
	cam.release()
	cv2.destroyAllWindows()

def saveBuff(cam,finalBuffer):
	cam.release()
	cv2.destroyAllWindows()
	f=open("temp.txt","w")
	for i in finalBuffer:
		f.write(i)
	f.close()

def capture_images(cam,saveimg,mask):
	cam.release()
	cv2.destroyAllWindows()
	if not os.path.exists('./SampleGestures'):
		os.mkdir('./SampleGestures')

	gesname=saveimg[-1]
	img_name = "./SampleGestures/"+"{}.png".format(str(gesname))
	save_img = cv2.resize(mask, (image_x, image_y))
	cv2.imwrite(img_name, save_img)


def controlTimer(self):
	# if timer is stopped
	self.timer.isActive()
		# create video capture
	self.cam = cv2.VideoCapture(0)
		# start timer
	self.timer.start(20)
            

def predictor():
	import numpy as np
	from keras.preprocessing import image
	test_image = image.load_img('1.png', target_size=(64, 64))
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier.predict(test_image)
	gesname=''

	image_to_compare = cv2.imread("./SampleGestures/"+fileEntry[0])
	original = cv2.imread("1.png")
	sift = cv2.xfeatures2d.SIFT_create()
	kp_1, desc_1 = sift.detectAndCompute(original, None)
	kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

	index_params = dict(algorithm=0, trees=5)
	search_params = dict()
	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(desc_1, desc_2, k=2)

	good_points = []
	ratio = 0.6
	for m, n in matches:
	      if m.distance < ratio*n.distance:
	             good_points.append(m)
	#print(fileEntry[i])
	if(abs(len(good_points)+len(matches))>20):
		gesname=fileEntry[0]
		gesname=gesname.replace('.png','')
		return gesname

	elif result[0][0] == 1:
		  return 'A'
	elif result[0][1] == 1:
		  return 'B'
	elif result[0][2] == 1:
		  return 'C'
	elif result[0][3] == 1:
		  return 'D'
	elif result[0][4] == 1:
		  return 'E'
	elif result[0][5] == 1:
		  return 'F'
	elif result[0][6] == 1:
		  return 'G'
	elif result[0][7] == 1:
		  return 'H'
	elif result[0][8] == 1:
		  return 'I'
	elif result[0][9] == 1:
		  return 'J'
	elif result[0][10] == 1:
		  return 'K'
	elif result[0][11] == 1:
		  return 'L'
	elif result[0][12] == 1:
		  return 'M'
	elif result[0][13] == 1:
		  return 'N'
	elif result[0][14] == 1:
		  return 'O'
	elif result[0][15] == 1:
		  return 'P'
	elif result[0][16] == 1:
		  return 'Q'
	elif result[0][17] == 1:
		  return 'R'
	elif result[0][18] == 1:
		  return 'S'
	elif result[0][19] == 1:
		  return 'T'
	elif result[0][20] == 1:
		  return 'U'
	elif result[0][21] == 1:
		  return 'V'
	elif result[0][22] == 1:
		  return 'W'
	elif result[0][23] == 1:
		  return 'X'
	elif result[0][24] == 1:
		  return 'Y'
	elif result[0][25] == 1:
		  return 'Z'
	#else:
		#return 'None'
			   

def checkFile():
	checkfile=os.path.isfile('temp.txt')
	if(checkfile==True):
		fr=open("temp.txt","r")
		content=fr.read()
		fr.close()
	else:
		content="File Not Found"
	return content

class Dashboard(QtWidgets.QMainWindow):
	def __init__(self):
		super(Dashboard, self).__init__()
		uic.loadUi('UI_Files/dash.ui', self)
		self.timer = QTimer()
		self.create.clicked.connect(self.createGest)
		self.exp2.clicked.connect(self.exportFile)
		self.scan_sen.clicked.connect(self.scanSent)
		if(self.scan_sinlge.clicked.connect(self.scanSingle)==True):
			self.timer.timeout.connect(self.scanSingle)
		
		#controlTimer(self)
		
	def createGest(self):
		try:
			clearfunc(self.cam)
		except:
			pass
		#runpy.run_path("CreateGest.py")
		uic.loadUi('UI_Files/create_gest.ui', self)
		self.create.clicked.connect(self.createGest)
		self.exp2.clicked.connect(self.exportFile)
		if(self.scan_sen.clicked.connect(self.scanSent)):
			controlTimer(self)
		self.scan_sinlge.clicked.connect(self.scanSingle)	
		self.pushButton_2.clicked.connect(lambda:clearfunc(self.cam))
		self.plainTextEdit.setPlaceholderText("Enter Gesture Name Here") 
		img_text = ''
		saveimg=[]
		while True:
			ret, frame = self.cam.read()
			frame = cv2.flip(frame,1)
			try:
				frame=cv2.resize(frame,(321,270))
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				img2 = cv2.rectangle(frame, (150,50),(300,200), (0,255,0), thickness=2, lineType=8, shift=0)
			except:
				keyboard.press_and_release('esc')

			height2, width2, channel2 = img2.shape
			step2 = channel2 * width2
        # create QImage from image
			qImg2 = QImage(img2.data, width2, height2, step2, QImage.Format_RGB888)
        # show image in img_label
			try:
				self.label_3.setPixmap(QPixmap.fromImage(qImg2))
				slider2=self.trackbar.value()
			except:
				pass

			lower_blue = np.array([0, 0, 0])
			upper_blue = np.array([179, 255, slider2])
			
			imcrop = img2[52:198, 152:298]
			hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
			mask = cv2.inRange(hsv, lower_blue, upper_blue)
			
			cv2.namedWindow("Image", cv2.WINDOW_NORMAL )
			image = cv2.imread('template.png')
			cv2.imshow("Image",image)
			cv2.setWindowProperty("Image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
			cv2.resizeWindow("Image",298,430)
			cv2.moveWindow("Image", 1052,214)


			cv2.namedWindow("mask", cv2.WINDOW_NORMAL )
			cv2.imshow("mask", mask)
			cv2.setWindowProperty("mask",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
			cv2.resizeWindow("mask",140,160)
			cv2.moveWindow("mask", 764,271)

			hwnd = winGuiAuto.findTopWindow("mask")
			win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

			try:
				ges_name = self.plainTextEdit.toPlainText()
			except:
				pass

			if(len(ges_name)>1):
				saveimg.append(ges_name)
				ges_name=''
			else:
				saveimg.append(ges_name)
				ges_name=''


			try:
				self.pushButton.clicked.connect(lambda:capture_images(self.cam,saveimg,mask))
			except:
				pass			
				
			if cv2.waitKey(1) == 27:
			    break


		self.cam.release()
		cv2.destroyAllWindows()
		if not os.path.exists('./SampleGestures'):
			os.mkdir('./SampleGestures')

		gesname=saveimg[-1]
		img_name = "./SampleGestures/"+"{}.png".format(str(gesname))
		save_img = cv2.resize(mask, (image_x, image_y))
		cv2.imwrite(img_name, save_img)

		#self.close()

	def exportFile(self):
		try:
			clearfunc(self.cam)
		except:
			pass
		uic.loadUi('UI_Files/export.ui', self)
		self.create.clicked.connect(self.createGest)
		self.exp2.clicked.connect(self.exportFile)
		self.scan_sen.clicked.connect(self.scanSent)
		self.scan_sinlge.clicked.connect(self.scanSingle)
		content=checkFile()
		self.textBrowser.setText("		 "+content)
		if(content=="File Not Found"):
			self.pushButton_2.setEnabled(False)
		else:
			self.pushButton_2.clicked.connect(self.on_click)

	def on_click(self):
		content=checkFile()
		root=Tk()
		root.withdraw()
		root.filename =  filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("text files","*.txt"),("all files","*.*")))
		name=root.filename
		#fr.close()
		fw=open(name+".txt","w")
		fw.write(content)
		os.remove("temp.txt")
		fw.close()
		root.destroy()


	def scanSent(self):
		try:
			clearfunc(self.cam)
		except:
			pass
		uic.loadUi('UI_Files/scan_sent.ui', self)
		self.create.clicked.connect(self.createGest)
		self.exp2.clicked.connect(self.exportFile)
		if(self.scan_sen.clicked.connect(self.scanSent)):
			controlTimer(self)
		self.scan_sinlge.clicked.connect(self.scanSingle)	
		self.pushButton_2.clicked.connect(lambda:clearfunc(self.cam))
		img_text = ''
		img_text1 = ''
		append_text=''
		new_text=''
		finalBuffer=[]
		x=0
		y=0
		y1=y+10
		z=0

		while True:
			ret, frame =self.cam.read()
			frame = cv2.flip(frame,1)
			try:
				frame=cv2.resize(frame,(331,310))
			
			
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				img = cv2.rectangle(frame, (150,50),(300,200), (0,255,0), thickness=2, lineType=8, shift=0)
			except:
				keyboard.press_and_release('esc')
				keyboard.press_and_release('esc')
				keyboard.press_and_release('esc')

			height, width, channel = img.shape
			step = channel * width
        # create QImage from image
			qImg = QImage(img.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
			try:
				self.label_3.setPixmap(QPixmap.fromImage(qImg))
				slider=self.trackbar.value()
			except:
				pass
				
			lower_blue = np.array([0, 0, 0])
			upper_blue = np.array([179, 255, slider])

			imcrop = img[52:198, 152:298]
			hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
			mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
			
			image = cv2.imread('template.png')
			cv2.imshow("Image",image)
			cv2.moveWindow("Image", 1030,150)
			
			cv2.namedWindow("mask", cv2.WINDOW_NORMAL )
			cv2.imshow("mask", mask1)
			cv2.setWindowProperty("mask",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
			cv2.resizeWindow("mask",118,108)
			cv2.moveWindow("mask", 905,271)

			hwnd = winGuiAuto.findTopWindow("mask")
			win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

			
			self.textBrowser.setText(img_text)
			img_name = "1.png"
			save_img = cv2.resize(mask1, (image_x, image_y))
			cv2.imwrite(img_name, save_img)
			img_text=predictor()
			if cv2.waitKey(1) == ord('c'):
					try:
						append_text+=img_text
						new_text+=img_text
					except:
						append_text+=''
					self.textBrowser_4.setText(new_text)
					'''try:
						y=0
						for x in range(len(finalBuffer)):
							self.textBrowser_4.setText(finalBuffer[x])
							y+=20
					except:
						pass'''
					
					if(len(append_text)>1):
						finalBuffer.append(append_text)
						append_text=''
					else:
						finalBuffer.append(append_text)
						append_text=''
					 
			try:
				self.pushButton.clicked.connect(lambda:saveBuff(self.cam,finalBuffer))
			except:
				pass
			if cv2.waitKey(1) == 27:
				f=open("temp.txt","w")
				for i in finalBuffer:
					f.write(i)
				f.close()
				
				break
				
		self.cam.release()
		cv2.destroyAllWindows()

	def scanSingle(self):
		try:
			clearfunc(self.cam)
		except:
			pass
		uic.loadUi('UI_Files/scan_single.ui', self)
		self.create.clicked.connect(self.createGest)
		self.exp2.clicked.connect(self.exportFile)
		self.scan_sen.clicked.connect(self.scanSent)
		if(self.scan_sinlge.clicked.connect(self.scanSingle)):
			controlTimer(self)
		self.pushButton_2.clicked.connect(lambda:clearfunc(self.cam))
		img_text = ''
		while True:
			ret, frame = self.cam.read()
			frame = cv2.flip(frame,1)
			try:
				frame=cv2.resize(frame,(321,270))
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				img1 = cv2.rectangle(frame, (150,50),(300,200), (0,255,0), thickness=2, lineType=8, shift=0)
			except:
				keyboard.press_and_release('esc')

			height1, width1, channel1 = img1.shape
			step1 = channel1 * width1
        # create QImage from image
			qImg1 = QImage(img1.data, width1, height1, step1, QImage.Format_RGB888)
        # show image in img_label
			try:
				self.label_3.setPixmap(QPixmap.fromImage(qImg1))
				slider1=self.trackbar.value()
			except:
				pass

			lower_blue = np.array([0, 0, 0])
			upper_blue = np.array([179, 255, slider1])
			
			imcrop = img1[52:198, 152:298]
			hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
			mask = cv2.inRange(hsv, lower_blue, upper_blue)
			
			image = cv2.imread('template.png')
			cv2.imshow("Image",image)
			cv2.moveWindow("Image", 1030,150)
			
			cv2.namedWindow("mask", cv2.WINDOW_NORMAL )
			cv2.imshow("mask", mask)
			cv2.setWindowProperty("mask",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
			cv2.resizeWindow("mask",118,108)
			cv2.moveWindow("mask", 894,271)

			hwnd = winGuiAuto.findTopWindow("mask")
			win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
			
			self.textBrowser.setText("\n\n\t"+str(img_text))
			img_name = "1.png"
			save_img = cv2.resize(mask, (image_x, image_y))
			cv2.imwrite(img_name, save_img)
			img_text = predictor()
				

			if cv2.waitKey(1) == 27:
			    break


		self.cam.release()
		cv2.destroyAllWindows()


app = QtWidgets.QApplication([])
win = Dashboard()
win.show()
sys.exit(app.exec())
