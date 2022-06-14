"""
File: adv_examples.py -- Adversarial Examples Creation
Authors: Apurva Gandhi and Shomik Jain
Date: 2/02/2020
"""

MODELS_DIR = ''

MODEL_NAMES = [''] 
VAL_DIR = ""

SOFTMAX = False
# Only use > 1 for carlini wagner
BATCH_SIZE = 1
DEVICE_STR='0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import cv2
from torchvision.utils import save_image
import regex as re
#from cw_attack import L2Adversary
from deepfake_detector.pretrained_mods import xception
import torch.nn.functional as F
#from albumentations import Resize
import torch.backends.cudnn as cudnn
import math
from sklearn.metrics import confusion_matrix
from torch import linalg as LA

class ImageFolderWithPaths(datasets.ImageFolder):
    #Custom dataset that includes image file paths. Extends
    #torchvision.datasets.ImageFolder
    

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


transformation = transforms.Compose([transforms.Resize((299, 299)),transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

val_data = ImageFolderWithPaths(root=VAL_DIR, transform=transformation)
dataloaders = {} 
dataloaders['val'] = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)

dataset_sizes = {}
dataset_sizes['val'] = len(val_data)
print(dataset_sizes)

class_names = val_data.classes
print(class_names)
print(val_data.class_to_idx)

device = torch.device("cuda:"+DEVICE_STR if torch.cuda.is_available() else "cpu")

if SOFTMAX:
  criterion = nn.CrossEntropyLoss().to(device)
else: 
  criterion = nn.BCEWithLogitsLoss().to(device)

for mn in MODEL_NAMES:
  print(mn)
  model = xception.imagenet_pretrained_xception()
  model = model.cuda()
  model = nn.DataParallel(model).to(device)
  #model_params = torch.load(MODELS_DIR+mn, map_location=device)
  #model.load_state_dict(model_params)
  model_params = torch.load(MODELS_DIR+mn)
  model.load_state_dict(model_params)

  cudnn.benchmark = True
  running_corrects = 0
  total = 0
  model.eval()

  #FC layer 1 version
  for imgs, labels, paths in tqdm(dataloaders['val']):
      epoch_running_acc = 0
      imgs = imgs.to(device) 
      labels = labels.to(device)
      outputs = model(imgs)
      label = torch.ones_like(labels)
      thresh_preds = torch.round(torch.sigmoid(outputs))
      print(thresh_preds)
      running_corrects += torch.sum(thresh_preds==label.unsqueeze(1))
      epoch_running_acc += torch.sum(thresh_preds == label.unsqueeze(1))
      #print(epoch_running_acc / 32) 
      total += imgs.size(0)

print(running_corrects.double()/total*100)
