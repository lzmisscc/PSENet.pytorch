# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import torch
import shutil
import numpy as np
import config
import os
import cv2
from tqdm import tqdm
from models import PSENet
from predict import Pytorch_model
# from cal_recall.script import cal_recall_precison_f1
from utils import draw_bbox, txt2json
from utils.crop_rect import crop_rect

torch.backends.cudnn.benchmark = True


def main(model_path, backbone, scale, path, save_path, gpu_id):
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_paths = [os.path.join(path, x) for x in os.listdir(path)]
    net = PSENet(backbone=backbone, pretrained=False, result_num=config.n)
    model = Pytorch_model(model_path, net=net, scale=scale, gpu_id=gpu_id)
    total_frame = 0.0
    total_time = 0.0
    for img_path in tqdm(img_paths):
        print(img_path)
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_path, str(img_name) + '.json')
        _, boxes_list, t, rects, img = model.predict(img_path, )
        total_frame += 1
        total_time += t
        boxes_list = boxes_list.reshape(-1, 8)

        cropped_imgs = []
        for idx, rect in enumerate(rects):
            cropped_img = crop_rect(img, rect, alph=0.05)
            cropped_imgs.append(cropped_img)
        txt2json.txt2json(img_path, save_name, boxes_list, cropped_imgs)
        cv2.imwrite(os.path.join(save_path, '{}.jpg'.format(img_name)),
                    cv2.imread(img_path))


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = ['0', '1']
    backbone = 'resnet50'
    scale = 4
    model_path = 'pse.pth'
    gpu_id = 1
    import glob
    for data_path in glob.glob(''):
        print(data_path)
        save_path = data_path.replace('allZhongben', 'allZhongbenpse')
        os.makedirs(save_path, True)
        print('backbone:{},scale:{},model_path:{}'.format(backbone, scale, model_path))
        main(
            model_path, backbone, scale, data_path, save_path, gpu_id=gpu_id
        )
