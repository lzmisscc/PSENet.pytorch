import json
import os
from os.path import join
import sys
import tqdm
import base64
import cv2
import numpy as np
import uuid


def points2l_t_r_b(points):
    points = [
        min(points[0][0], points[1][0]), min(points[1][1], points[0][1]),
        max(points[0][0], points[1][0]), max(points[1][1], points[0][1])
    ]
    return list(map(int, points))


def two2four(points):
    return [points[0], points[1], points[2], points[1], points[2], points[3], points[0], points[3]]


def StrToImg(str, name):
    img_b64decode = base64.b64decode(str)  # base64解码
    img_array = np.fromstring(img_b64decode, np.uint8)  # 转换np序列
    img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)  # 转换Opencv格式
    # cv2.imwrite(name, img)
    return img


tmp = 0
import glob

for path in glob.glob('path/*'):
    print(path)
    save_path = path.replace('allZhongben', 'allZhongbenpse')
    os.makedirs(save_path, True)
    all_dir = [name for name in os.listdir(path) if '.json' in name]
    for name in tqdm.tqdm(all_dir):
        try:
            with open(join(path, name), 'r', encoding='gbk') as f:
                x = json.load(f)
        except Exception as e:
            try:
                with open(join(path, name), 'r', encoding='utf8') as f:
                    x = json.load(f)
            except Exception as e:
                print(name)
                continue
        str_img = x['imageData']
        img_name = x['imagePath']
        img = StrToImg(str_img, os.path.join(path, img_name))
        points = []
        sfz_0_point = []
        sfz_1_point = []

        for i in x['shapes']:
            if i['label'] == '0':
                sfz_0_point = points2l_t_r_b(i['points'])
            elif i['label'] == '1':
                sfz_1_point = points2l_t_r_b(i['points'])
        if not (sfz_0_point and sfz_1_point):
            tmp += 1
            print(img_name, str(tmp))
            continue
        sfz_0 = []
        sfz_1 = []
        # find points
        for i in x['shapes']:
            if i['label'] != '0' and i['label'] != '1':
                points = points2l_t_r_b(i['points'])
                if points[0] >= sfz_0_point[0] and points[1] >= sfz_0_point[1] and points[2] <= sfz_0_point[2] and \
                        points[3] <= sfz_0_point[3]:
                    sfz_0.append([points[0] - sfz_0_point[0], points[1] - sfz_0_point[1],
                                  points[2] - sfz_0_point[0], points[3] - sfz_0_point[1], ])
                else:
                    sfz_1.append([points[0] - sfz_1_point[0], points[1] - sfz_1_point[1],
                                  points[2] - sfz_1_point[0], points[3] - sfz_1_point[1], ])
        sfz_0_image_name = uuid.uuid4().hex + '.jpg'
        sfz_1_image_name = uuid.uuid4().hex + '.jpg'
        # save image
        shape = img.shape
        if len(shape) == 3:
            cv2.imwrite(join(save_path, sfz_0_image_name), img[
                                                           sfz_0_point[1]:sfz_0_point[3], sfz_0_point[0]:sfz_0_point[2],
                                                           :
                                                           ])
            cv2.imwrite(join(save_path, sfz_1_image_name), img[
                                                           sfz_1_point[1]:sfz_1_point[3], sfz_1_point[0]:sfz_1_point[2],
                                                           :
                                                           ])
        else:
            cv2.imwrite(join(save_path, sfz_0_image_name), img[
                                                           sfz_0_point[1]:sfz_0_point[3], sfz_0_point[0]:sfz_0_point[2],
                                                           ])
            cv2.imwrite(join(save_path, sfz_1_image_name), img[
                                                           sfz_1_point[1]:sfz_1_point[3], sfz_1_point[0]:sfz_1_point[2],
                                                           ])
        # save txt
        with open(join(save_path, sfz_0_image_name.replace('jpg', 'txt')), 'w') as f:
            f.writelines(
                [','.join(map(str, two2four(i))) + '\n' for i in sfz_0]
            )
        with open(join(save_path, sfz_1_image_name.replace('jpg', 'txt')), 'w') as f:
            f.writelines(
                [','.join(map(str, two2four(i))) + '\n' for i in sfz_1]
            )
