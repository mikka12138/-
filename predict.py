#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo import YOLO
from PIL import Image
import pandas as pd
from tqdm import tqdm
import cv2
import os
import time


# f = open("./VOCdevkit/VOC2007/ImageSets/Main/test.txt", "r")   # 设置文件对象
# data = f.readlines()  # 直接将文件中按行读到list里，效果与方法2一样
# f.close()
# data = ['./VOCdevkit/VOC2007/JPEGImages/' + i[:-1] + '.jpg' for i in data]

data_dir = './test_data/ship/'
data = os.listdir(data_dir)
# print(data)
yolo = YOLO()

output_dir = './test_data/ship_result/'
start = time.time()
i = 1
for img in tqdm(data):
    image = Image.open(data_dir + img)
    # image.show()
    # print(input())
    r_image = yolo.detect_image(image)
    # print(r_image)
    # r_image.show()
    # r_image.save('test_results/tank_warship_person/%s.jpg' % i)
    r_image.save(output_dir + 'data_result%s' % img)
    i += 1

print('cost :  %s seconds' % int(time.time() - start))