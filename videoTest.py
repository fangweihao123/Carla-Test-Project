#这个文件主要将CARLA环境中的图像取出加以处理并且以视频流的形式显示出来
import glob
import os
import sys

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import requests

import argparse
import random
import time
# import skvideo.io
import numpy as np
import math
import asyncio
from socket import*
import json
from io import StringIO
import pickle
import cv2
from yolo import YOLO
from PIL import Image
from es import OpenES
import tensorflow as tf
from carla import ColorConverter as cc
# from wwh import detect_Car,cal_loss_1,read_from_allboxes_use_index,read_from_allboxes_use_loc

#这边就作为客户端
HOST = '127.0.0.1'    # The remote host
PORT = 7555                 # The same port as used by the server
s = None
BUFSIZE = 1024
ADDR = (HOST, PORT)
myflag = False

#Carla的车辆控制对象 第一个是空车 第二个是停车
vControlStart = carla.VehicleControl(brake=0.0)
vControlStop = carla.VehicleControl(brake=1000.0)

def main():
    #file.read 返回到是字符串
    #keras网络初始化
    net = YOLO()

    # sess = tf.Session(config=config)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    y_ = tf.placeholder(dtype=tf.float32, shape=[80])
    res_ = tf.placeholder(dtype=tf.float32, shape=[80])
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(res_)))

    #carla客户端初始化设置
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='number of vehicles (default: 10)')
    argparser.add_argument(
        '-d', '--delay',
        metavar='D',
        default=2.0,
        type=float,
        help='delay in seconds between spawns (default: 2.0)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='avoid spawning vehicles prone to accidents')
    args = argparser.parse_args()

    try:

        actor_list = []
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        #r = requests.get('https://127.0.0.1:8888')

        world = client.get_world()
        vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')
        #sensor_blueprints = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')     #获得色彩镜头的蓝图
        #re存储所有得到的图像
        re = []
        def convet_image_to_np(image):
            w = image.width
            h = image.height
            result = np.zeros((h,w,3))          #height在前面..
            for i in range(h):
                for j in range(w):
                    color = image.__getitem__(i*w + j)
                    #blue green red?
                    result[i][j][0] = color.b
                    result[i][j][1] = color.g
                    result[i][j][2] = color.r
                    # result[i][j][3] = color.a
            return result


        def test_callback(image):
            re.append(convet_image_to_np(image).astype(np.uint8))
            print("test")
            #print(convet_image_to_np(image).astype(np.uint8))

        def store_Image2(image):
            time.sleep(0.1)
            if len(re) == 0:
                re.append(image)
            else:
                if image.frame_number > re[len(re)-1].frame_number and len(re)<= 5000:
                    re.append(image)

        #获取车辆的速度
        def get_speed(vehicle):
            """
            Compute speed of a vehicle in Kmh
            :param vehicle: the vehicle for which speed is calculated
            :return: speed as a float in Kmh
            """
            vel = vehicle.get_velocity()
            return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


        #在transform的位置产生车辆
        def try_spawn_random_vehicle_at(transform):
            blueprint = random.choice(vehicle_blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            #这边设置了自动驾驶
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                actor_list.append(vehicle)
                vehicle.set_autopilot()
                print('spawned %r at %s' % (vehicle.type_id, transform.location))
                return vehicle
            return None

        #生成一个附在交通工具的镜头 并且把照相机获取的照片放在_out文件夹中
        def try_spawn_attach_camera_to_vehicle(mVehicle,transform):
            print(camera_bp)
            camera = world.spawn_actor(camera_bp,transform,attach_to=mVehicle)
            #每一帧会生成一个图片 但是生成的速度过快
            # camera.listen(lambda image:image.save_to_disk(
            #     '_out/%06d.png' % image.frame_number
            # ))
            camera.listen(lambda image: store_Image2(image))
            return camera


        my_vehicle = try_spawn_random_vehicle_at(carla.Transform(carla.Location(x=10, y=10, z=5), carla.Rotation(pitch=0, yaw=0, roll=0)))
        print(my_vehicle)
        my_camera = try_spawn_attach_camera_to_vehicle(my_vehicle,carla.Transform(carla.Location(x=1, y=0, z=1.5), carla.Rotation(pitch=0, yaw=0, roll=0)))
        print('main')
        flag = False
        current = 0
        while True:
            # test文件是作为动态检测的存在
            # print(my_vehicle.get_location())
            # print(my_camera.get_location())
            state = 0
            carsCnt = 0
            if len(re) > 0 and current < len(re):
                print(current)
                print(len(re))
                cur = re[current]
                current = len(re) - 1
                # current += 1
                # cur.convert(cc.Raw)
                array = np.frombuffer(cur.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (cur.height, cur.width, 4))
                #关键API 将buffer中的数据转化为图片 该操作可以极大地加快数据转化速度
                array = array[:, :, :3]
                # array.size = (cur.width,cur.height)
                # array = array[:, :, ::-1]
                # temp = convet_image_to_np(cur).astype(np.uint8)
                # cv2.imshow("result", array)
                # cv2.waitKey(1)
                #用yolo检测 将image对象作为输入
                img_result, box_, score_, class_, input_shape_yrj, boxes_yrj, boxes_yrj_score, yrj_yolo_output \
                    = net.detect_image(Image.fromarray(array))
                img_result = np.array(img_result)
                speed = str(get_speed(my_vehicle))[:6]
                cv2.putText(img_result, "speed:" + speed, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 1)

                # 将原图片和添加文字后的图片拼接起来
                # res = np.hstack([src, AddText])

                for box_index in range(len(box_)):
                    if class_[box_index] == 2:
                        carsCnt += 1
                if carsCnt > 0:
                    my_vehicle.set_autopilot(enabled=False)
                else:
                    my_vehicle.set_autopilot(enabled=True)


                #
                # plt.imshow(img_result)
                # plt.pause(0.01)

                # img_result,index,box_,score_,all_box=detect_Car(net,temp)
                cv2.imshow("result",img_result)
                cv2.waitKey(1)
    finally:
        print('\ndestroying %d actors' % len(actor_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        #结束进程的时候删除生成的物体


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
