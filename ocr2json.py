import os
from os.path import join
import json
from PIL import Image
import base64
import cv2
import numpy as np
import mxnet as mx
from cnocr import CnOcr
import time
import math
import matplotlib.pyplot as plt

ocr = CnOcr()


def imageToStr(image):
    with open(image, 'rb') as f:
        image_byte = base64.b64encode(f.read())
        # print(type(image_byte))
    image_str = image_byte.decode('ascii')  # byte类型转换为str
    # print(type(image_str))
    return image_str


with open('1.json', 'r') as f:
    L = json.load(f)

templates = {
    "label": "1",
    "line_color": None,
    "fill_color": None,
    "points": [
        [
            158.21739130434776,
            130.6086956521739
        ],
        [
            598.4347826086955,
            177.3478260869565
        ]
    ],
    "shape_type": "rectangle",
    "flags": {}
}
gen_json_path = 'result/_scale4/result'
save_json_path = './ocr_result'
img_pathss = []

for i in [gen_json_path, ]:
    for root, sub_root, n in os.walk(i):
        img_pathss.append(root)

for path in img_pathss:
    for img in os.listdir(path):
        if '.jpg' not in img and '.png' not in img:
            continue
        print(img)
        image = Image.open(join(path, img))
        image_np = np.array(image, np.uint8)
        with open(join(path, img[:-4] + '.txt'), 'r') as f:
            txt = f.readlines()

        points = []
        for line in txt:
            line = list(map(float, line.strip('\n').split(',')[0:8]))
            x = [line[i] for i in range(0, 8, 2)]
            y = [line[i] for i in range(1, 8, 2)]
            x_min, y_min, x_max, y_max = min(x), min(y), max(x), max(y)
            point = [[x_min, y_min], [x_max, y_max]]
            templates["points"] = point
            crop_image = np.array(image.crop([x_min, y_min, x_max, y_max]), np.uint8)
            templates["label"] = ''.join(ocr.ocr_for_single_line(
                crop_image
            ))
            points += [templates.copy()]
        L["shapes"] = points
        L["imagePath"] = img
        L["imageData"] = imageToStr(join(path, img))
        L["imageWidth"], L["imageHeight"] = Image.open(join(path, img)).size

        image.save(join(save_json_path, img[:-4] + '.jpg'))
        with open(join(save_json_path, img[:-4] + '.json'), 'w') as f:
            json.dump(L, f)
