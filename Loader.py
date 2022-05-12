from torch.utils.data.dataset import Dataset
from copy import deepcopy
import math
import numpy as np
import torch
import torch.torch_version 
from torch import nn
import cv2
import csv
from skimage import morphology

class MnistDataSet(Dataset):
  def __init__(self,rootDir) -> None:
    super().__init__()
    self.FileList = []
    with open(f"{rootDir}/chinese_mnist.csv",mode="r",encoding="utf8") as csvFile:
      reader = csv.reader(csvFile)
      next(reader)            #读掉表头
      for row in reader:
        self.FileList.append((f"{rootDir}/input_{row[0]}_{row[1]}_{row[2]}.jpg",int(row[2])-1))

  def __imgPrepoocess(self,img,target):
    t = 32
    img = np.where(img <= t, 0, img)  # 这里丢弃低亮度的像素，消除噪点
    # 矩形框裁剪
    edge = [0, 0, 0, 0]
    for i in range(0, img.shape[0]):
        if (img[i].max() > t):
            edge[0] = max(0, i)
            break
    for i in range(img.shape[0] - 1, -1, -1):
        if (img[i].max() > t):
            edge[1] = min(img.shape[0], i + 1)
            break
    for i in range(0, img.shape[1]):
        if (img[:, i].max() > t):
            edge[2] = max(0, i)
            break
    for i in range(img.shape[1] - 1, -1, -1):
        if (img[:, i].max() > t):
            edge[3] = min(img.shape[1], i + 1)
            break
    if (target == 1):  # 对1做特殊处理，否则会被拉伸
        wd = edge[3] - edge[2]
        mid = (edge[0] + edge[1]) / 2
        edge[0] = math.floor(mid - wd / 2)
        edge[1] = math.floor(mid + wd / 2)
    img = img[edge[0]:edge[1], edge[2]:edge[3]]  # 裁切

    size = 56
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)  # 缩放
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)  # 规格化灰度
    img = cv2.equalizeHist(img)  # 直方图均衡化
    img = cv2.copyMakeBorder(img,4,4,4,4,borderType=cv2.BORDER_CONSTANT,value=0)#填充边缘
    img = cv2.threshold(img, 64, 1, cv2.THRESH_BINARY)[1]  # 阈值
    img = morphology.skeletonize(img)  # 提取骨架，由scikit-image提供的算法

    img = torch.from_numpy(img)
    img = img.float()
    return img
    
  def __getitem__(self, index):
    img = cv2.imread(self.FileList[index][0],flags=cv2.IMREAD_GRAYSCALE)
    img = self.__imgPrepoocess(img,self.FileList[index][1])
    img = img.unsqueeze(0)
    img = img.repeat_interleave(3,0)

    return img, self.FileList[index][1]
      
  def __len__(self):
    return len(self.FileList)