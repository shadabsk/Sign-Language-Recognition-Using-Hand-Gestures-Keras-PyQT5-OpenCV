from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QUrl 
import sys
import os
import runpy
class Dashboard(QtWidgets.QMainWindow):
	def __init__(self):
		super(Dashboard, self).__init__()
		uic.loadUi('UI_Files/dash.ui', self)
		self.create.clicked.connect(self.createGest)
		self.exp2.clicked.connect(self.exportFile)
		self.scan_sen.clicked.connect(self.scanSent)
		self.scan_sinlge.clicked.connect(self.scanSingle)
	def createGest(self):
		#runpy.run_path("CreateGest.py")
		uic.loadUi('UI_Files/create_gest.ui', self)
		self.create.clicked.connect(self.createGest)
		self.exp2.clicked.connect(self.exportFile)
		self.scan_sen.clicked.connect(self.scanSent)
		self.scan_sinlge.clicked.connect(self.scanSingle)
		#self.close()
	def exportFile(self):
		uic.loadUi('UI_Files/export.ui', self)
		self.create.clicked.connect(self.createGest)
		self.exp2.clicked.connect(self.exportFile)
		self.scan_sen.clicked.connect(self.scanSent)
		self.scan_sinlge.clicked.connect(self.scanSingle)
		#runpy.run_path("FileExport.py")
		#self.close()
	def scanSent(self):
		uic.loadUi('UI_Files/scan_sent.ui', self)
		self.create.clicked.connect(self.createGest)
		self.exp2.clicked.connect(self.exportFile)
		self.scan_sen.clicked.connect(self.scanSent)
		self.scan_sinlge.clicked.connect(self.scanSingle)
		#runpy.run_path("ScanSent.py")
		#self.close()		
	def scanSingle(self):
		uic.loadUi('UI_Files/scan_single.ui', self)
		self.create.clicked.connect(self.createGest)
		self.exp2.clicked.connect(self.exportFile)
		self.scan_sen.clicked.connect(self.scanSent)
		self.scan_sinlge.clicked.connect(self.scanSingle)
		#runpy.run_path("ScanSingle.py")
		#self.close()
app = QtWidgets.QApplication([])
win = Dashboard()
win.show()
sys.exit(app.exec())

 
