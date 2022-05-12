from copy import deepcopy
import random
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from MyModel import MnistModel
from Loader import *
from matplotlib import pyplot

if __name__ == '__main__':
  #部署GPU
  device = torch.device("cuda")
  #数据
  testSet = MnistDataSet("Verify")
  testLoader = DataLoader(testSet,100,shuffle=False,drop_last=False)
  testLen = len(testSet)

  #模型
  #myModel = MnistModel()
  myModel = torchvision.models.vgg16()
  myModel.classifier[6] = nn.Linear(4096,15)
  print(myModel)
  myModel.load_state_dict(torch.load("Models/vgg16.pth"))  #移除dropout
  myModel.train(False)
  myModel = myModel.to(device)
  
  #代价函数
  lossFn = nn.CrossEntropyLoss()
  lossFn = lossFn.to(device)

  #参数
  tset_step = 0
  testLoss = 0
  rightCount = 0

  #错误散点图
  wrongScatter = [[],[]]
  #错误图
  wrongImgs = [[]for i in range(15)]

  logWriter = SummaryWriter("TestLogs")
  with torch.no_grad():
    for data in testLoader:
      imgs,targets = data
      imgs = imgs.to(device)
      targets = targets.to(device)
      outputs = myModel(imgs)
      result = torch.argmax(outputs,1)
      rightCount += torch.sum(result==targets)

      #写入错误图片和概率分布
      for i in range(len(result)):
        if result[i] != targets[i]:
          wrongScatter[0].append(result[i].item()+(random.random()-0.5)*0.3)
          wrongScatter[1].append(targets[i].item()+(random.random()-0.5)*0.3)
          wrongImgs[targets[i]].append(imgs[i].cpu().numpy().tolist())
      loss = lossFn(outputs,targets)
      testLoss+=loss
    rightRate = rightCount/testLen*100
  print(f"测试损失:{testLoss}，正确率：{rightRate:.2f}%")

  for i in range(len(wrongImgs)):
    imgs = torch.from_numpy(np.asarray(wrongImgs[i]))
    if(len(imgs)!=0):
      logWriter.add_images(f"target:{i}",imgs)
# pyplot.cla()
# pyplot.scatter(x=wrongScatter[0],y=wrongScatter[1])
# pyplot.show()
  logWriter.close()