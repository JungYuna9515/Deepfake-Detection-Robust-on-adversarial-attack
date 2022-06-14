"""
File: adv_examples.py -- Adversarial Examples Creation
Authors: Apurva Gandhi and Shomik Jain
Date: 2/02/2020
"""

MODELS_DIR = ''
MODEL_NAMES = ['xception.pth']

ATTACK = 'fgsm'

VAL_DIR = ""

OUTPUT_DIR = ""

SOFTMAX = False
# Only use > 1 for carlini wagner
BATCH_SIZE = 1
DEVICE_STR='0'

from distutils.log import error
import torch
import gc
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
from cw_attack import L2Adversary
from pretrained_mods import xception
from albumentations import Resize
from advertorch.attacks import PGDAttack
from advertorch.attacks.utils import rand_init_delta
from advertorch.utils import clamp
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import batch_l1_proj


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def fgsm(model, loss, eps, softmax=False):
	def attack(img, label):
		output = model(img)
		if softmax: 
			error = loss(output, label)
		else:
			error = loss(output, label.unsqueeze(1).float())
		error.backward()
		perturbed_img = torch.clamp(img + eps*img.grad.data.sign(), 0, 1).detach()
		img.grad.zero_()
		return perturbed_img
	return attack

class PGD:
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, images, labels, eps=2 / 255, alpha=0.01, iterations=40):
        images = images.to(device)
        labels = labels.to(device)

        ori_images = images.data

        for i in range(iterations):
            images.requires_grad = True
            outputs = self.model(images)

            self.model.zero_grad()
            cost = self.loss(outputs.squeeze(-1), labels.type_as(outputs)).to(device)
            cost.backward()

            adv_images = images + alpha * images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        return torch.clamp(ori_images + eta, min=0, max=1).detach_()

def save_batch(model, imgs, labels, paths, criterion, class_names, device, batch_size=BATCH_SIZE):
  for i in range(batch_size):
    label = labels[i]
    orig_path = paths[i]
    img_num = orig_path.split('/')[-1].split('.')[0]
    img_out = model_out + class_names[label] + '/' + img_num + '.jpg'
    save_image(imgs[i],img_out)

def generate_adversarial_examples(data_loader, attack, criterion, class_names, device, softmax=False):
	adversarial_examples = list()
	labels = list()
	paths = list()
 
	for img, label, path in tqdm(data_loader):
		img = img.to(device) 
		label = label.to(device)
		img.requires_grad = True
		perturbed_img = attack(img, label)
		img.requires_grad = False

		adversarial_examples.append(perturbed_img)
		labels.append(label)
		paths.append(path)
		save_batch(model, perturbed_img, label, path, criterion, class_names, device)

	return adversarial_examples, labels, paths

def test(inputs, labels, model, device):
    running_corrects = 0
    num = 0
    for inputs, labels in tqdm(zip(inputs, labels)):
        inputs = inputs.to(device)
        inputs = inputs.float() / 255.0
        transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        inputs = transform(inputs)
        num += inputs.shape[0]
        labels = labels.to(device)
        outputs = model(inputs)
        #_, preds = torch.max(outputs, 1)
        preds = torch.sigmoid(outputs)
        preds = torch.round(preds)
        running_corrects += torch.sum(preds == labels.unsqueeze(1))
    print(running_corrects)
    print(num)
    print(running_corrects / 10296 * 100)
    return (running_corrects.double()/num * 100)

def test_softmax(inputs, labels, model, device):
    num_correct = 0
    for img, label in tqdm(zip(inputs, labels)):
        img = img.to(device)
        label = label.to(device)
        _, pred = torch.max(model(img), 1)
        if(pred == 1):
            if(label == 1):
                num_correct += 1
        else:
            if(label == 0):
                num_correct += 1

    return num_correct/len(inputs)

def test_softmax_batch(inputs, labels, model, device):
    running_corrects = 0
    for inputs, labels in tqdm(zip(inputs, labels)):
        inputs = inputs.to(device)
        inputs = inputs.float() / 255.0
        transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        inputs = transform(inputs)
        num += inputs.shape[0]
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels)
    return (running_corrects.double()/10296*100)

transformation = transforms.Compose([transforms.Resize((299, 299)),transforms.ToTensor()])

val_data = ImageFolderWithPaths(root=VAL_DIR, transform=transformation)
dataloaders = {} 
dataloaders['val'] = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

dataset_sizes = {}
dataset_sizes['val'] = len(val_data)
print(dataset_sizes)

class_names = val_data.classes
print(class_names)
print(val_data.class_to_idx)

device = torch.device("cuda:"+DEVICE_STR if torch.cuda.is_available() else "cpu")

if SOFTMAX:
  criterion = nn.CrossEntropyLoss()
else: 
  criterion = nn.BCEWithLogitsLoss()

for mn in MODEL_NAMES:
  print(mn)
  model = xception.imagenet_pretrained_xception()
  model = nn.DataParallel(model)
  model = model.cuda()
  model_params = torch.load(MODELS_DIR+mn)
  model.load_state_dict(model_params)
  #model.load_state_dict(model_params.module.state_dict())
  """
  x = torch.randn(4,3,299,299,device=device)
  logits = model(x)
  print(logits)
  preds = torch.sigmoid(logits)
  print(preds)
  thresh_preds = torch.round(preds)
  print(thresh_preds)
  """
  # Create Directories to Output Examples
  mn = mn.split('.')[0]
  model_out = OUTPUT_DIR + mn + '_' + ATTACK + '/'
  os.mkdir(model_out)

  os.mkdir(model_out + class_names[0]+'/')
  os.mkdir(model_out + class_names[1]+'/')
  regex = re.compile(r'\d+')

  # ATTACK: save images after each batch
  if 'fgsm' in ATTACK:
    adv_examples, labels, paths = generate_adversarial_examples(dataloaders["val"], fgsm(model, criterion, 1/255, softmax=SOFTMAX), criterion, class_names, device)

  elif 'pgd' in ATTACK:
    adversary = PGD(model,device)
    adv_examples, labels, paths = generate_adversarial_examples(dataloaders["val"], adversary, criterion, class_names, device)

  print(len(adv_examples))
  print(len(labels))
  # Evaluate Adversarial Examples
  if BATCH_SIZE > 1:
    print("Acc on adversarial_examples (before saving):", test_softmax_batch(adv_examples, labels, model, device))
  else:
    print("Acc on adversarial_examples (before saving):", test(adv_examples, labels, model, device) if not SOFTMAX else test_softmax(adv_examples, labels, model, device))

    