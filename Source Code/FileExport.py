from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QUrl 
import os
from tkinter import filedialog
from tkinter import * 
import tkinter as tk
import sys
import runpy

checkfile=os.path.isfile('temp.txt')
if(checkfile==True):
	fr=open("temp.txt","r")
	content=fr.read()
else:
	content="File Not Found"
class FileExport(QtWidgets.QMainWindow):
	def __init__(self):
		super(FileExport, self).__init__()
		uic.loadUi('UI_Files/export.ui', self)
		self.create.clicked.connect(self.createGest)
		self.export_2.clicked.connect(self.exportFile)
		self.scan_sen.clicked.connect(self.scanSent)
		self.scan_sinlge.clicked.connect(self.scanSingle)        
		self.textBrowser.setText("         "+content)
		if(content=="File Not Found"):
			self.pushButton_2.setEnabled(False)
		else:
			self.pushButton_2.clicked.connect(self.on_click)
	def on_click(self):
		root=Tk()
		root.filename =  filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("text files","*.txt"),("all files","*.*")))
		name=root.filename
		fr.close()
		fw=open(name+".txt","w")
		fw.write(content)
		os.remove("temp.txt")
		fw.close()
		root.destroy()
	def createGest(self):
		runpy.run_path("CreateGest.py")
		self.close()
	def scanSent(self):
		runpy.run_path("ScanSent.py")
		self.close()		
	def scanSingle(self):
		runpy.run_path("ScanSingle.py")
		self.close()
	def exportFile(self):
		runpy.run_path("FileExport.py")
		self.close()
		
app = QtWidgets.QApplication([])
win1 = FileExport()
win1.show()


