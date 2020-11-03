# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import torch
import shutil
import numpy as np
from torch.serialization import save
import config
import os
import cv2
from tqdm import tqdm
from models import PSENet
from predict import Pytorch_model
from cal_recall.script import cal_recall_precison_f1
from utils import draw_bbox
import time

torch.backends.cudnn.benchmark = True


def main(model_path, backbone, scale, path, save_path, gpu_id, gt_path=None):
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_img_folder = os.path.join(save_path, 'img')
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
    save_txt_folder = os.path.join(save_path, 'result')
    if not os.path.exists(save_txt_folder):
        os.makedirs(save_txt_folder)
    img_paths = [os.path.join(path, x) for x in os.listdir(path)]
    net = PSENet(backbone=backbone, pretrained=False, result_num=config.n)
    model = Pytorch_model(model_path, net=net, scale=scale, gpu_id=gpu_id)
    total_frame = 0.0
    total_time = 0.0
    for img_path in img_paths:
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_txt_folder, str(img_name) + '.txt')

        start_time = time.time()
        _, boxes_list, t, _, _ = model.predict(img_path)
        print(img_name, time.time() - start_time)

        total_frame += 1
        total_time += t
        img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
        cv2.imwrite(os.path.join(save_img_folder,
                                 '{}.jpg'.format(img_name)), img)
        # add filter
        if gt_path:
            gt_bbox = np.loadtxt(
                os.path.join(gt_path, os.path.basename(save_name)),
                delimiter=','
            )
            search_bbox(pre=boxes_list, gt=gt_bbox)
        np.savetxt(save_name, boxes_list.reshape(-1, 8),
                   delimiter=',', fmt='%d')
    print('fps:{}'.format(total_frame / total_time))
    return save_txt_folder


def fun(pre: np.array):
    pre = pre.reshape(2, 4)
    pre = np.array([np.min(pre[0, :]), np.min(pre[1, :]),
                    np.max(pre[0, :]), np.max(pre[1, :])],
                   np.float32)
    return pre


def iou(pre, gt):
    pre = fun(pre)
    gt = fun(gt)
    h = min(pre[2], gt[2]) - max(pre[0], gt[0])
    w = min(pre[3], gt[3]) - max(pre[1], gt[1])
    if h <= 0 or w <= 0:
        return 0
    else:
        return w*h / ((pre[3]-pre[1])*(pre[2]-pre[0])+(gt[3]-gt[1])*(gt[2]-gt[0])-w*h)


def search_bbox(pre, gt):
    save_bbox = []
    for gt_bbox in gt:
        result = [iou(pre_bbox, gt_bbox) for pre_bbox in pre]
        index = np.argmax(result)
        save_bbox.append(pre[index])
    return np.array(save_bbox)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    backbone = 'resnet18'
    scale = 1
    model_path = 'output/PSENet_599.pth'
    # model_path = 'pruner.pth'
    # data_path = '/home/cc/eval/img'
    # gt_path = '/home/cc/eval/txt'
    # data_path = './sfz-txt/gt-img'
    # gt_path = './sfz-txt/gt-txt'
    # data_path = '/home/cc/20200730jixu/pse/财产反馈汇总表/img'
    # gt_path = '/home/cc/20200730jixu/pse/财产反馈汇总表/txt'
    # data_path = '/home/cc/20200730jixu/pse/结案审批表/img'
    # gt_path = '/home/cc/20200730jixu/pse/结案审批表/txt'
    # data_path = '/home/cc/20200730jixu/pse/结案审批表/img'
    # gt_path = '/home/cc/20200730jixu/pse/结案审批表/txt'
    # data_path = '/home/cc/yyzz/123456/pse/yyzz/lsz_res/img'
    # gt_path = '/home/cc/yyzz/123456/pse/yyzz/lsz_res/txt'
    # data_path = '/home/cc/yyzz/123456/pse/yyzz/yyzz_res/img'
    # gt_path = '/home/cc/yyzz/123456/pse/yyzz/yyzz_res/txt'
    data_path = '1009/img'
    gt_path = '1009/txt'
    save_path = './result/'
    gpu_id = 0
    print('backbone:{},scale:{},model_path:{}'.format(
        backbone, scale, model_path))
    save_path = main(model_path, backbone, scale, data_path,
                     save_path, gpu_id=gpu_id, gt_path=gt_path)
    result = cal_recall_precison_f1(gt_path=gt_path, result_path=save_path)
    print(result)
