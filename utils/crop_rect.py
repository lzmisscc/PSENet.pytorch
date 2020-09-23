# -*- coding: utf-8 -*-
# @Date    : 2020/6/15 22:19
# @Author  : LiuZhuang

import cv2
import numpy as np


def translate(image, x, y):
    """from https://www.programcreek.com/python/example/89459/cv2.getRotationMatrix2D:
        Example 29
    """
    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # 返回转换后的图像
    return shifted


def crop_rect(img, rect, alph=0.05):
    """
    adapted from https://github.com/ouyanghuiyu/chineseocr_lite/blob/e959b6dbf3/utils.py
    从图片中按框截取出图片patch。
    """
    center, sizes, angle = rect[0], rect[1], rect[2]
    sizes = (int(sizes[0] * (1 + alph)), int(sizes[1]))
    center = (int(center[0]), int(center[1]))

    if 1.5 * sizes[0] < sizes[1]:
        sizes = (sizes[1], sizes[0])
        angle += 90
    elif angle < -45 and (0.66 < sizes[0] / (1e-6 + sizes[1]) < 1.5):
        sizes = (sizes[1], sizes[0])
        angle -= 270

    height, width = img.shape[0], img.shape[1]
    # 先把中心点平移到图片中心，然后再旋转就不会截断图片了
    img = translate(img, width // 2 - center[0], height // 2 - center[1])
    center = (width // 2, height // 2)

    # FIXME 如果遇到一个贯穿整个图片对角线的文字行，旋转还是会有边缘被截断的情况
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, sizes, center)
    # cv2.imwrite("img_translate.jpg", img)
    # cv2.imwrite("img_rot.jpg", img_rot)
    # cv2.imwrite("img_crop_rot.jpg", img_crop)
    # import pdb; pdb.set_trace()
    return img_crop
