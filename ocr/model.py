# -*- coding:utf-8 -*-
import os
import numpy as np
from imp import reload
from PIL import Image, ImageOps

from keras.layers import Input
from keras.models import Model
# import keras.backend as K

from . import keys
from . import densenet

reload(densenet)

characters = keys.alphabet[:]
# characters=keys.alphabet3220[:]
characters = characters[1:] + u'卍'
nclass = len(characters)

input = Input(shape=(32, None, 1), name='the_input')
y_pred = densenet.dense_cnn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)

modelPath = 'ocr.h5'
if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)
else:
    raise modelPath


def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and (
                (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)


def pad_image(image, target_size):
    iw, ih = image.size  # 原始图像的尺寸
    w, h = target_size  # 目标图像的尺寸
    scale = min(float(w) / float(iw), float(h) / float(ih))  # 转换的最小比例

    # 保证长或宽，至少一个符合目标图像的尺寸
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)  # 缩小图像
    # image.show()
    new_image = Image.new('RGB', target_size, (255, 255, 255))  # 生成灰色图像
    # // 为整数除法，计算图像的位置
    new_image.paste(image, (0, (h - nh) // 2))  # 将图像填充为中间图像，两侧为灰色的样式
    # plt.imshow(new_image)
    # plt.savefig("image.jpg")
    return new_image


def predict(img):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)

    # img = img.resize([width, 32], Image.ANTIALIAS)
    img = pad_image(img, (width, 32)).convert('L')

    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    img = np.array(img).astype(np.float32) / 255.0 - 0.5

    X = img.reshape([1, 32, width, 1])

    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, :, :]

    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
    # out = u''.join([characters[x] for x in out[0]])
    out = decode(y_pred)

    return out


def predict2(img):
    img = Image.fromarray(img).convert('L')
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale) if scale != 0 else 1
    if width < 16:
        return ''
    # print(width)
    img = img.resize([width, 32], Image.ANTIALIAS)

    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''

    img = np.array(img).astype(np.float32) / 255.0 - 0.5

    X = img.reshape([1, 32, width, 1])

    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, :, :]

    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
    # out = u''.join([characters[x] for x in out[0]])
    out = decode(y_pred)

    return out
