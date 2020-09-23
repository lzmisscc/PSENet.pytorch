import os
from os.path import join
import json
from PIL import Image
import base64
# import cv2
import numpy as np
from ocr.model import predict2 as ocr


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


def txt2json(img_path, save_name, lines, crop_imgs):
    # image = Image.open(img_path)

    points = []
    for line, crop_image in zip(lines, crop_imgs):
        x = [line[i] for i in range(0, 8, 2)]
        y = [line[i] for i in range(1, 8, 2)]
        x_min, y_min, x_max, y_max = min(x), min(y), max(x), max(y)
        point = [[x_min, y_min], [x_max, y_max]]
        templates["points"] = point
        # crop_image = image.crop([x_min, y_min, x_max, y_max])
        templates["label"] = ''.join(ocr(
            crop_image
        ))
        points += [templates.copy()]
    L["shapes"] = points
    L["imagePath"] = os.path.basename(img_path)
    L["imageData"] = imageToStr(img_path)
    L["imageWidth"], L["imageHeight"] = Image.open(img_path).size

    with open(save_name, 'w') as f:
        json.dump(L, f)
