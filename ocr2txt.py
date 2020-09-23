import os
from os.path import join
import json
from PIL import Image
import base64
import cv2
import numpy as np


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


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
p = '/home/lz/立案识别模板/立案识别模板/jin_data/立案图片分类/'
p2 = '/home/lz/立案识别模板/立案识别模板/jin_data/终本图片分类/'

for path in [p, p2]:
    for img in os.listdir(path):
        if '.jpg' not in img and '.png' not in img:
            continue
        print(img)
        image = Image.open(join(path, img))
        with open(join(path, img[:-4] + '.txt'), 'r') as f:
            txt = f.readlines()

        points = []
        labels = []
        for line in txt:
            line = list(map(float, line.strip('\n').split(',')[0:8]))
            x = [line[i] for i in range(0, 8, 2)]
            y = [line[i] for i in range(1, 8, 2)]
            x_min, y_min, x_max, y_max = min(x), min(y), max(x), max(y)
            point = [[x_min, y_min], [x_max, y_max]]
            labels.append('label' + '\n')

        with open(join(path, img[:-4] + '.txt'), 'w') as f:
            f.writelines(labels)
