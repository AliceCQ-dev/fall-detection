#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import math
import copy
np.set_printoptions(suppress=True)
import pandas as pd
from glob import glob
import os,sys,gc
import argparse
import json,time
import matplotlib.pyplot as plt
import torch
from torch import from_numpy
import subprocess
from pathlib import Path
from typing import NamedTuple
from math import ceil, floor
import ffmpeg
from src import model
from src import util_add_fun as util
from src.body import Body
from torch import nn


# In[ ]:


def get_bbox(keypoints):
    found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
    found_kpt_id = 0
    for kpt_id in range(18):
        if keypoints[kpt_id, 0] == -1:
            continue
        found_keypoints[found_kpt_id] = keypoints[kpt_id]
        found_kpt_id += 1
    bbox = cv2.boundingRect(found_keypoints)
    return bbox


class VideoReader(object):
    def __init__(self, file_name, code_name, frame_start=300, frame_interval=2):
        self.file_name = file_name
        self.code_name = str(code_name)
        self.frame_interval = frame_interval  # 帧间隔，2表示每隔2帧取1帧
        self.frame_start = frame_start
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        self.frame_index = self.frame_start  # 初始化起始帧

        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        while True:
            was_read, img = self.cap.read()
            if not was_read:
                raise StopIteration
            
            # cv2.putText(img, self.code_name, (5, 35),
            #                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            self.frame_index += self.frame_interval

            return img

def get_keypoints(candidate):

    tt = np.array(candidate)

    result = []

    # 获取所有索引值并转换为整数
    idx = tt[:, 3].astype(int)

    for i in range(18):
        # 找到当前索引i在tt中的位置
        positions = np.where(idx == i)[0]
        
        if len(positions) > 0:
            # 如果找到，提取第一和第二列
            result.append(tt[positions[0], 0:2].astype(int).tolist())
        else:
            # 如果没找到，用[-1, -1]补充
            result.append([-1, -1])

    return np.array(result)


def draw_bodypose(canvas, candidate, subset):
    # 初始化一个全黑的画布（原图尺寸）
    black_canvas = np.zeros_like(canvas)
    
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]
    
    # 1. 只画骨架（不画关键点圆圈）
    for i in range(17):  # 遍历所有骨架连接
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            
            # 获取两个关键点坐标
            Y = candidate[index.astype(int), 0]  # x坐标
            X = candidate[index.astype(int), 1]  # y坐标
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            
            # 用白色绘制骨架
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(black_canvas, polygon, (255, 255, 255))  # 白色骨架
    
    # 2. 调整分辨率为 (128, 128)
    resized_canvas = cv2.resize(black_canvas, (128, 128), interpolation=cv2.INTER_NEAREST)
    
    return resized_canvas


# In[ ]:


class CNN(nn.Module):
    def __init__(self):
        # 调用父类（nn.Module）的构造函数，确保模型继承并初始化
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(16384, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.sequential(x)


def put_text_with_background(img, text, position=(3, 15)):
    """
    在黑底图像上显示清晰文字
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    color = (0, 0, 255)  # 红色文字
    bg_color = (255, 255, 255)  # 白色背景
    
    # 获取文字尺寸
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 绘制背景矩形
    x, y = position
    # cv2.rectangle(img, (x-2, y - text_height - 2), 
    #               (x + text_width + 2, y + 2), bg_color, -1)
    cv2.rectangle(img, (int(x-1.5), int(y - text_height - 1.5)), 
                  (int(x + text_width + 1.5), int(y + 1.5)), bg_color, -1)
    
    # 绘制文字
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
    return img

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)


# openpose = Body('/home/bhennelly/Documents/QIN/thesis_project/pytorch-openpose-direct-v2/weights/body_pose_model.pth')
openpose = Body(r'D:\projects\thesis_project\pytorch-openpose-direct-v2\weights\body_pose_model.pth')


# data_path = '/home/bhennelly/Documents/QIN/thesis_project/FLIR_fall'
data_path = r'D:\projects\thesis_project\FLIR_fall'


cnn = CNN()

# weight_path = "/home/bhennelly/Documents/QIN/thesis_project/pytorch-openpose-direct-v2/weights"
weight_path = r'D:\projects\thesis_project\pytorch-openpose-direct-v2\weights'
checkpoint_name = 'FLIR_best_checkpoint.pth'

cnn.load_state_dict(torch.load(os.path.join(weight_path, checkpoint_name), map_location=torch.device(DEVICE)))
cnn.to(DEVICE)
cnn.eval()

print(cnn)


# In[ ]:


frame_reader = VideoReader(os.path.join(data_path, 'FALL_26.avi'), 'demo', frame_start = 0, frame_interval=2)


for img in frame_reader:
    
    candidate, subset = openpose(img)
    if len(candidate)>=3:
        skeleton_img = draw_bodypose(img, candidate, subset)

        # skeleton_img = cv2.imread(skeleton_img, cv2.IMREAD_GRAYSCALE) #以灰度图形式读数据,但是必须放入的是img的地址
        skeleton_img = cv2.cvtColor(skeleton_img, cv2.COLOR_BGR2GRAY) # 用这种方式转成灰度图

        test_img = skeleton_img.copy()

        skeleton_img = skeleton_img.reshape(-1)  # 展平 (128,128) -> (16384,)
        skeleton_img = skeleton_img / 255.0  # 归一化

        keypoints = get_keypoints(candidate)
        pose_bbox = get_bbox(keypoints)

        crown_proportion = pose_bbox[2]/pose_bbox[3] #宽高比

        # 预测后的再处理
        skeleton_tensor = torch.tensor(skeleton_img).float().unsqueeze(0).to(DEVICE)  # 明确指定为float类型
        with torch.no_grad():
            predict = cnn(skeleton_tensor)

        print('predicted fall probability: ', predict[:,0])
        possible_rate = predict[:,0].detach().cpu().numpy()[0]
        
        # possible_rate = possible_rate.detach().cpu().numpy()[0]

        if possible_rate >= 0.50:
            pose_action = 'fall'
        else:
            pose_action = 'normal'
        
        test_img = put_text_with_background(test_img, pose_action, (3, 15))
        
        cv2.imshow('DEMO', test_img)

        cv2.waitKey(0)  # 等待任意键按下，才会继续执行下一行

cv2.destroyAllWindows()



