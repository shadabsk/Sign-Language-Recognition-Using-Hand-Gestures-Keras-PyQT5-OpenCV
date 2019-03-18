from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QUrl 
import os
from tkinter import filedialog
from tkinter import * 
import tkinter as tk
import sys

checkfile=os.path.isfile('temp.txt')
if(checkfile==True):
	fr=open("temp.txt","r")
	content=fr.read()
else:
	content="File Not Found"

 
class Example(QtWidgets.QMainWindow):
    def __init__(self):
        super(Example, self).__init__()
        uic.loadUi('UI_Files/FileExport.ui', self)
        self.textBrowser.setText(content)
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
           
app = QtWidgets.QApplication([])
win = Example()
win.show()
sys.exit(app.exec())