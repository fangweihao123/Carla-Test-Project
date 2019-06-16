## 基于CARLA的测试方法实现


### 环境要求
* github上面的keras-yolo3
https://github.com/qqwweee/keras-yolo3
* tensorflow
* 相对应的cuda和cudnn
* staticTest需要在python环境中安装pyqt5
* 将对应的文件放到keras的文件夹即可
<br>
<br>
<br>
* 静态测试
###### 静态测试是对于一组图片进行的目标测试，效果如下
![](https://github.com/fangweihao123/Photo-Repo/raw/master/statictest.png)
###### 其主要思路就是用pyqt5作为界面设计，将CARLA中获得的图片逐张进行目标检测，对应的文件为staticTest.py
<br>
<br>

* 动态测试
###### 动态测试是将将CARLA中的图片取出进行处理，再进行目标检测，具体的效果如下
![](https://github.com/fangweihao123/Photo-Repo/raw/master/video.png)
###### 其中主要的思路就是讲获得的图片数据放入到数组中，按照顺序进行展现，其中比较关键的代码如下,可以较快的加快buffer数据转化为图像的速度，对应的文件是videoTest.py
```
array = np.frombuffer(cur.raw_data, dtype=np.dtype("uint8"))
array = np.reshape(array, (cur.height, cur.width, 4))
```
<br>
<br>

* 对抗样本测试
###### 在测试中引入了对抗样本，具体效果如下
![](https://raw.githubusercontent.com/fangweihao123/Photo-Repo/master/adversialTest1.png)
![](https://raw.githubusercontent.com/fangweihao123/Photo-Repo/master/adversialTest2.png)
###### 其主要思路是用演化算法的对抗样本生成手法，在图像中加入噪声，使得目标识别网络失灵，从而检测目标算法鲁棒性，对应的文件为adversialTest.py