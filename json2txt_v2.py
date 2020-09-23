import json
import os
from os.path import join
import sys
import tqdm
import base64
import cv2
import numpy as np


def StrToImg(str, name):
    img_b64decode = base64.b64decode(str)  # base64解码
    img_array = np.fromstring(img_b64decode, np.uint8)  # 转换np序列
    img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)  # 转换Opencv格式
    cv2.imwrite(name, img)
    return True


# 输入路径，字典中对应的json路径与图片路径
for p in ["path"]:
    path = p
    print(path)
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
        # else:
        #     continue
        # str_img = x['imageData']
        img_name = x['imagePath']
        # StrToImg(str_img, os.path.join(path, img_name))
        points = []
        for i in x['shapes']:
            # if i['label'] == '0':
            #     continue
            if len(points) == 4:
                point = list(map(str, map(round, [
                    i['points'][0][0], i['points'][0][1],
                    i['points'][1][0], i['points'][0][1],
                    i['points'][1][0], i['points'][1][1],
                    i['points'][0][0], i['points'][1][1]
                ])))
            else:
                point = list(map(str, map(round, [
                    i['points'][0][0], i['points'][0][1],
                    i['points'][1][0], i['points'][1][1],
                    i['points'][2][0], i['points'][2][1],
                    i['points'][3][0], i['points'][3][1]
                ])))
            points.append(','.join(point) + '\n')
        with open(join(p, name.replace('.json', '.txt')), 'w') as f:
            f.writelines(points)
