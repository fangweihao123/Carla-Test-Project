#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:08:09 2019

@author: sse
"""

from yolo import YOLO
from PIL import Image
import numpy as np
from es import OpenES
import tensorflow as tf
import os
import cv2 as cv
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from mainwindow import *
import sys

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

    def yoloDetect(self):
        print('detect')

    def findFile(self):
        fileName, filetype = QFileDialog.getOpenFileName(self,
                                                          "选取文件",
                                                          "./",
                                                          "All Files (*);;Text Files (*.txt)")  # 设置文件扩展名过滤,注意用双分号间隔
        # directory1 = QFileDialog.getExistingDirectory(self,
        #                                               "选取文件夹",
        #                                               "./")  # 起始路径
        # print(directory1)
        self.currentFile = fileName
        pix = QPixmap(fileName)
        self.label.setStyleSheet("border: 2px solid red")
        self.label.setPixmap(pix)
        self.label.setScaledContents(True)

#Optimizer中间采用了演化算法的思想 通过演化算法进行对抗样本攻击
class Optimizer:
    def __init__(self,box,index):
        self.anchorList = []       #存放当前所有bounding boxes的中心点
        self.fwh_boxes = []
        self.indexList = []
        print(box[0])
        print(box[1])
        self.minY = box[0]         #y1
        self.minX = box[1]         #x1
        self.maxY = box[2]         #y2
        self.maxX = box[3]         #x2
        self.addAnchor(box)
        self.fwh_boxes.append(box)
        self.indexList.append(index)
        self.maxAnchorDistance = 10.0

    def calDistance(self,anchorY1,anchorX1,anchorY2,anchorX2):
        return np.sqrt(np.square(anchorX1-anchorX2) + np.square(anchorY1-anchorY2))

    #self相当于cpp中的self指针
    def check_if_box_belong_same_object(self,box):
        cnt = 0
        for eachAnchor in self.anchorList:
            if self.calDistance(eachAnchor[0],eachAnchor[1],(box[0] + box[2])/2,(box[1] + box[3])/2) < self.maxAnchorDistance:
                #如果该box的中心点距离当前所有的中心点的距离大于30
                cnt+=1
        return True if cnt == len(self.anchorList) else False

    #通过下面两个函数 进行数学计算来大致确定一共有多少个物体
    def addAnchor(self,box):
        temp = []
        temp.append((box[0] + box[2]) / 2)          # 第一个是y
        temp.append((box[1] + box[3]) / 2)          # 第二个是x
        self.anchorList.append(temp)

    def addBox(self,box,index):
        #添加box的流程 第一步 需要更新anchorlist 其次 需要更新最大最小的值 后期需要
        self.addAnchor(box)
        if box[0]<self.minY:
            self.minY = box[0]
        if box[1]<self.minX:
            self.minX = box[1]
        if box[2]>self.maxY:
            self.maxY = box[2]
        if box[3]>self.maxX:
            self.maxX = box[3]
        self.fwh_boxes.append(box)
        self.indexList.append(index)

    #演化算法初始化
    def initOpenES(self):
        self.minY = int(self.minY)
        self.minX = int(self.minX)
        self.maxY = int(self.maxY)
        self.maxX = int(self.maxX)
        if self.minY < 0:
            self.maxY -= self.minY
            self.minY = 0
        if self.maxY > img_height:
            self.minY -= (self.maxY-img_height)
            self.maxY = img_height
        if self.minX < 0:
            self.maxX -= self.minX
            self.minX = 0
        if self.maxX > img_width:
            self.minX -= (self.maxX - img_width)
            self.maxX = img_width

        self.maxHeight = self.maxY - self.minY
        self.maxWidth = self.maxX - self.minX
        # height * width
        self.es = OpenES(num_params=(self.maxHeight * self.maxWidth * 3), sigma_init=0.01, popsize=50)



#交叉熵
def cal_loss(_score,_label):
    loss=sess.run(cross_entropy,feed_dict={y_:_label,res_:_score
                        })
    return loss

def cal_loss_1(_score,_label):
    loss=(1-_score[2])**2
    return loss

def read_from_allboxes_use_index(allboxes, index):
    if isinstance(index, list):
        result = []
        for i in range(len(index)):
            num_layers = index[i][0]
            j, k, l = index[i][2], index[i][3], index[i][4]
            #        print(allboxes[num_layers][0,j,k,l])
            result.append(allboxes[num_layers][0, j, k, l])
    return result

# 通过单个index获取信息
def read_from_allboxes_use_loc(allboxes, loc):
    a, b, c, d = loc[0], loc[2], loc[3], loc[4]

    return allboxes[a][0, b, c, d]

#检测车辆的函数
def detect_Car(net_, img_):
    img_result, box_, score_, class_, input_shape_yrj, boxes_yrj, boxes_yrj_score, yrj_yolo_output \
        = net_.detect_image(img_)
    index = []
    for box_index in range(len(box_)):
        # 2是car这个类在coco_classes.txt文件中car类的下标 下标从0开始
        if class_[box_index] == 2:
            for i in range(len(boxes_yrj)):
                a, b, c = boxes_yrj[i].shape[1], boxes_yrj[i].shape[2], boxes_yrj[i].shape[3]
                for j in range(a):
                    for k in range(b):
                        for l in range(c):
                            if (boxes_yrj[i][0, j, k, l] == box_[box_index]).all():
                                index.append([i, 0, j, k, l])

    result_box = read_from_allboxes_use_index(boxes_yrj, index)
    result_score = read_from_allboxes_use_index(boxes_yrj_score, index)

    #    print(result_box)
    #    print(result_score)
    return img_result, index, result_box, result_score, boxes_yrj_score

#神经网络初始化 mian 函数开始
net=YOLO()
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
y_=tf.placeholder(dtype=tf.float32,shape=[80])
res_=tf.placeholder(dtype=tf.float32,shape=[80])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(res_)))
originAccuracyPath = 'origin.txt'
openESAccuracyPath = 'current.txt'
filePath = './WWH/'
filelist = os.listdir(filePath)
trainIteration = 5
#初始化框架
# app = QApplication(sys.argv)
# myWin = MyWindow()
# myWin.show()
# sys.exit(app.exec_())

#对文件夹下面的文件进行对抗样本攻击
for filename in filelist:
    print(filename)
    img_path = filePath + filename
    image_test=Image.open(img_path)
    print(image_test)
    a=np.array(image_test)

    img_height = len(a)
    img_width = len(a[0])
    #四通道转化为三通道
    a=a[:,:,0:3]
    print('-----------------------------------')
    print(a.shape)
    #test_
    #test
    a = Image.fromarray(np.uint8(a))
    a.show()

    #通过index-list来确定信息
    #将image作为输入....
    img_result,index,box_,score_,all_box=detect_Car(net,image_test)
    print(index)
    print(box_)
    #box里面的数据内容是[y1 x1 y2 x2]
    #print(score_)
    img_result.show()
    #img_result.save('ori_result.bmp')

    #从a数组转化而来
    ord_img=np.array(a,dtype='float32')
    ord_img/=255.0

    optimizerList = []
    for loopIndex in range(len(box_)):
        flag = False
        for optimizer_item in optimizerList:
            #此时的optimizer_item都是初始化过了的
            flag = optimizer_item.check_if_box_belong_same_object(box_[loopIndex])
            if flag is True:
                optimizer_item.addBox(box_[loopIndex],index[loopIndex])
                break
        if flag is False:
            optimizerList.append(Optimizer(box_[loopIndex],index[loopIndex]))

    print(optimizerList)
    label_=np.zeros([80])
    label_[2]=1             #LABEL2就是车辆的标志
    print(label_)

    for optimizer_item in optimizerList:
        #引入cnt变量计算迭代次数
        cnt = 0
        resultScore = 0.0
        #这里需要向原文件写入准确率
        for i in range(len(optimizer_item.indexList)):
            tmpResScore = read_from_allboxes_use_loc(all_box, optimizer_item.indexList[i])
            resultScore += tmpResScore[2]
        with open(originAccuracyPath,'a+') as f:
            f.write(str(resultScore / len(optimizer_item.indexList)))
            f.write('\n')
        optimizer_item.initOpenES()
        target_y1 = optimizer_item.minY
        target_y2 = optimizer_item.minY + optimizer_item.maxHeight
        target_x1 = optimizer_item.minX
        target_x2 = optimizer_item.minX + optimizer_item.maxWidth
        while True:
            solutions = optimizer_item.es.ask()
            loss_list = []
            #这里乘以255是为了变成图片
            cnt += 1
            ord_=Image.fromarray(np.uint8(ord_img*255.0))
            for x in (solutions):
                #width 160 height 100
                x=np.reshape(x,[optimizer_item.maxHeight,optimizer_item.maxWidth,3])
                new_img=ord_img+0.

                new_img[target_y1:target_y2,target_x1:target_x2,:]+=x

                #限制数组里面的值 因为前一步是加上了噪音区域
                new_img=np.clip(new_img,0.0,1.0)

                newimg=new_img*255.
                newimg=Image.fromarray(np.uint8(newimg))
                # newimg.show()

                img_result_1,index,box_,score_,all_box_score=detect_Car(net,newimg)
                # img_result_1.show()
                total_loss = 0
                #有多少index就有多少次操作 单个index进行搜索
                for i in range(len(optimizer_item.indexList)):
                    tmpScore = read_from_allboxes_use_loc(all_box_score,optimizer_item.indexList[i])
                    tmpLoss = cal_loss_1(tmpScore,label_)
                    total_loss += tmpLoss
                loss_list.append(total_loss)

                # count_2+=1
                #记录添加噪声的次数
                # print(count_1,',',count_2,':', total_loss)
        #        print(count_1,',',count_2,':', loss_2)
                # print(count_1,',',count_2,':', loss_3)

            optimizer_item.es.tell(loss_list)

            # print(optimizer_item.es.best_reward)

            best_noise=optimizer_item.es.best_mu
            best_noise=np.reshape(best_noise,[optimizer_item.maxHeight,optimizer_item.maxWidth,3])

            temp=np.reshape(ord_img,[-1])
            temp=[x for x in temp]
            temp=np.array(temp)
            #height 与 width
            adv_img=np.reshape(temp,[img_height,img_width,3])
        #    adv_img = temp
            adv_img[target_y1:target_y2,target_x1:target_x2,:]+=best_noise

            adv_img = np.clip(adv_img,0.0,1.0)

            # ord_img=adv_img

            adv_img*=255.0
            adv_img=Image.fromarray(np.uint8(adv_img))

            ord_img[target_y1:target_y2,target_x1:target_x2,:]+=best_noise
            img_result,index,box_,score_,all_box_score=detect_Car(net,adv_img)
            # img_result.show()
            #如果有多个框 需要修改 获得多个值
            breakFlag = True
            resultScore = 0.0
            for i in range(len(optimizer_item.indexList)):
                tmpResScore = read_from_allboxes_use_loc(all_box_score,optimizer_item.indexList[i])
                if tmpResScore[2]>0.65:
                    breakFlag = False
                resultScore += tmpResScore[2]

            if cnt > trainIteration:
                with open(openESAccuracyPath,'a+') as f:
                    f.write(str(resultScore / len(optimizer_item.indexList)))
                    f.write('\n')
                break
            # if breakFlag is True:
            #     break
        temp_noise = np.zeros([img_height,img_width,3])
        best_noise = temp_noise[target_y1:target_y2,target_x1:target_x2,:]+best_noise
        best_noise = best_noise*255.0
        noise_=Image.fromarray(np.uint8(best_noise))
        # noise_.save('noise.jpg')
        adv_img.show()
        # adv_img.save('after.jpg')