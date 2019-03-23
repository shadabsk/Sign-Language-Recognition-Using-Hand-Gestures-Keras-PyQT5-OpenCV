from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QUrl 
import sys
import runpy

class ScanSingle(QtWidgets.QMainWindow):
	def __init__(self):
		super(ScanSingle, self).__init__()
		uic.loadUi('UI_Files/scan_single.ui', self)
		self.create.clicked.connect(self.createGest)
		self.export_2.clicked.connect(self.exportFile)
		self.scan_sen.clicked.connect(self.scanSent)
		self.scan_sinlge.clicked.connect(self.scanSingle)
	def createGest(self):
		runpy.run_path("CreateGest.py")
		self.close()
	def exportFile(self):
		runpy.run_path("FileExport.py")
		self.close()
	def scanSent(self):
		runpy.run_path("ScanSent.py")
		self.close()		
	def scanSingle(self):
		runpy.run_path("ScanSingle.py")
		self.close()
app = QtWidgets.QApplication([])
win = ScanSingle()
win.show()

