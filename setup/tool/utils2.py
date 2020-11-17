import time
from PIL import Image, ImageDraw
import torch.nn as nn
import torch
import numpy as np
import os
import sys
from PIL import Image
from torchvision import transforms

g_transform = transforms.Compose([
    transforms.ToTensor()
])


def makedir(path):
    '''

    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def img2Tensor(imgpath) -> torch.Tensor:
    '''

    :param imgpath:
    :return:
    '''
    with Image.open(imgpath) as img:
        img = img.convert("L")
        return g_transform(img)


def formattime(start_ms, end_ms):
    ms = end_ms - start_ms
    m_end, s_end = divmod(ms, 60)
    h_end, m_end = divmod(m_end, 60)
    time_data = "%02d:%02d:%02d" % (h_end, m_end, s_end)
    return time_data




def toTensor(data):
    '''
    :param data:
    :return:
    '''
    if isinstance(data, torch.FloatTensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    elif isinstance(data, (list, tuple)):
        return torch.tensor(list(data)).float()  # 针对列表和元组，注意避免list里是tensor的情况
    elif isinstance(data, torch.Tensor):
        return data.float()
    return


def toNumpy(data):
    '''
    :param data:
    :return:
    '''
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, (list, tuple)):
        return np.array(list(data))  # 针对列表和元组
    return


def toList(data):
    '''
    :param data:
    :return:
    '''
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, torch.Tensor):
        return data.numpy().tolist()
    elif isinstance(data, (list, tuple)):
        return list(data)  # 针对列表和元组
    return


def isBox(box):
    '''
    判断是否是box
    :param box:
    :return:
    '''
    box = toNumpy(box)
    if box.ndim == 1 and box.shape == (4,) and np.less(box[0], box[2]) and np.less(box[1], box[3]):
        return True
    return False


def isBoxes(boxes):
    '''
    判断是否是boxes
    :param boxes:
    :return:
    '''
    boxes = toNumpy(boxes)
    if boxes.ndim == 2 and boxes.shape[1] == 4:
        if np.less(boxes[:, 0], boxes[:, 2]).all() and np.less(boxes[:, 1], boxes[:, 3]).all():
            return True
    return False


def area(box):
    '''

    :param box:
    :return:
    '''
    return torch.mul((box[2] - box[0]), (box[3] - box[1]))


def areas(boxes):
    '''

    :param boxes:
    :return:
    '''
    return torch.mul((boxes[:, 2] - boxes[:, 0]), (boxes[:, 3] - boxes[:, 1]))

# iou


def iou(box, boxes, isMin=False):
    '''
    define iou function
    :param box:
    :param boxes:
    :param isMin:
    :return:
    '''

    box = toTensor(box)

    boxes = toTensor(boxes)  # 注意boxes为二维数组

    # 如果boxes为一维，升维
    if boxes.ndimension() == 1:
        boxes = torch.unsqueeze(boxes, dim=0)

    # box_area = torch.mul((box[2] - box[0]), (box[3] - box[1]))  # the area of the first row
    # boxes_area = torch.mul((boxes[:, 2] - boxes[:, 0]), (boxes[:, 3] - boxes[:, 1]))  # the area of other row

    box_area = area(box)
    boxes_area = areas(boxes)
    xx1 = torch.max(box[0], boxes[:, 0])
    yy1 = torch.max(box[1], boxes[:, 1])
    xx2 = torch.min(box[2], boxes[:, 2])
    yy2 = torch.min(box[3], boxes[:, 3])

    inter = torch.mul(torch.max(
        (xx2 - xx1), torch.Tensor([0, ])), torch.max((yy2 - yy1), torch.Tensor([0, ])))
    # print("inter",inter.shape, box_area.shape, boxes_area.shape, box_area)

    if (isMin == True):
        # intersection divided by union
        over = torch.div(inter, torch.min(box_area, boxes_area))
    else:
        # intersection divided by union
        over = torch.div(inter, (box_area + boxes_area - inter))
    return over


def nms(boxes_input, threshold=0.3, isMin=False):
    '''
    define nms function
    :param boxes_input:
    :param isMin:
    :param threshold:
    :return:
    '''
    # print("aaa",boxes_input[:,:4].shape)
    if isBoxes(boxes_input[:, :4]):
        '''split Tensor'''
        boxes = toTensor(boxes_input)

        boxes = boxes[torch.argsort(-boxes[:, 4])]

        r_box = []
        while (boxes.size(0) > 1):
            r_box.append(boxes[0])
            mask = torch.lt(iou(boxes[0], boxes[1:], isMin), threshold)
            boxes = boxes[1:][mask]  # the other row of Tensor
            '''mask 不能直接放进来,会报IndexError'''
        if (boxes.size(0) > 0):
            r_box.append(boxes[0])
        if r_box:
            return torch.stack(r_box)  # 绝对不能转整数，要不然置信度就变成0
    elif isBox(boxes_input):
        return toTensor(boxes_input)
    return torch.Tensor([])
    # return torch.stack(r_box).long()



