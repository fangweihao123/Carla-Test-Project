#这个文件用pyqt5写了一个界面用来进行静态测试
#实际上就是将一个文件夹的照片通过yolo网络进行目标检测
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel ,QDialog
from PyQt5.QtGui import QPixmap
from mainwindow import *
from dialog import *
import os
from PIL import Image
from PIL.ImageQt import ImageQt
import time
from yolo import YOLO


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, net, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.currentFile = None
        self.directory = None
        self.net = net


    def showImg(self,pix):
        self.label.setStyleSheet("border: 2px solid red")
        self.label.setPixmap(pix)
        self.label.setScaledContents(True)
        self.label_2.setText('DONE')
        QApplication.processEvents()
        #非常关键的API 可以在当前行对所有时间进行更新 也就不会卡了
        time.sleep(1)


    def yoloDetect(self):
        #需要进行detect
        if self.currentFile is None:
            return
        img_result, box_, score_, class_, input_shape_yrj, boxes_yrj, boxes_yrj_score, yrj_yolo_output \
            = self.net.detect_image(Image.open(self.currentFile))
        # img_result.show()
        qim = ImageQt(img_result)
        pix = QPixmap.fromImage(qim)
        self.showImg(pix)

    def findFile(self):
        fileName, filetype = QFileDialog.getOpenFileName(self,
                                                          "选取文件",
                                                          "./",
                                                          "All Files (*);;Text Files (*.txt)")  # 设置文件扩展名过滤,注意用双分号间隔
        self.currentFile = fileName
        pix = QPixmap(self.currentFile)
        self.showImg(pix)

    def findDir(self):
        self.directory = QFileDialog.getExistingDirectory(self,
                                                      "选取文件夹",
                                                      "./")  # 起始路径
        print(self.directory)


    def batchDetect(self):
        if self.directory is None:
            return
        filelist = os.listdir(self.directory)
        for filename in filelist:
            print(filename)
            secondSeg = os.path.splitext(filename)[-1]
            if secondSeg != '.jpg' and secondSeg != '.png' and secondSeg != '.bmp':
                continue
            #yolodetect函数需要提前设置currentfile参数
            self.currentFile = self.directory + '/' + filename
            self.yoloDetect()

class ChildWindow(QDialog,Ui_Dialog):
    def __init__(self, net, parent=None):
        super(ChildWindow, self).__init__(parent)
        self.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mNet = YOLO()
    myWin = MyWindow(net = mNet)
    myWin.show()
    sys.exit(app.exec_())
