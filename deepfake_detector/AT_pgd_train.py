MODELS_DIR = '/home/ansible/yuna/adversarial_deepfakes/pretrained_mods/weights/'
MODEL_NAMES = ['xception_celebdf(originalweak).pth']

SOFTMAX = False

BATCH_SIZE = 32
EPOCHS = 3
DEVICE_STR='0'

USE_HHRELU = False
USE_REG= True
REG_STRENGTH = 0

USE_NOISE = False
NOISE_TYPE = ''
NOISE_1 = 0
NOISE_2 = 0

TRAIN_DIR = "/home/ansible/yuna/dataset/Celeb-DF-v2/train"

OUTPUT_DIR = "/home/ansible/yuna/adversarial_deepfakes/regularization_weights/"


import argparse

import gc

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset


import datasets
import timm
import metrics
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from facedetector.retinaface import df_retinaface

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, lr_scheduler
import numpy as np
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from torch.autograd import Variable
from pretrained_mods import xception
import torch.backends.cudnn as cudnn
from advertorch.attacks import PGDAttack
from advertorch.attacks.utils import rand_init_delta
from advertorch.utils import clamp
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import batch_l1_proj

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()

def lipshitz_regularization(images, model, z=2, train=True, psi=1000, use_softmax=SOFTMAX, num_classes = 2):
    if use_softmax:
      repeated_images = images.repeat(num_classes, 1, 1, 1, 1)
      repeated_output = torch.stack([model(repeated_images[0]).sum(axis=0), model(repeated_images[1]).sum(axis=0)])
      grads = torch.autograd.grad(repeated_output, repeated_images, grad_outputs=torch.eye(num_classes).to(device), create_graph=train)[0]
    else:
      grads = torch.autograd.grad(model(images).sum(), images, create_graph=train)[0]
    return psi*grads.abs().pow(z).mean()

def add_noise(input, noise_type=NOISE_TYPE, noise1=NOISE_1, noise2=NOISE_2):
  if noise_type == 'Gaussian':
    noise = Variable(input.data.new(input.size()).normal_(noise1, noise2))
    return torch.clamp(input + noise, 0, 1)  
  elif noise_type == 'Cauchy':
    noise = Variable(input.data.new(input.size()).cauchy_(noise1, noise2))
    return torch.clamp(input + noise, 0, 1)  
  elif noise_type == 'Uniform': 
    noise = Variable(input.data.new(input.size()).uniform_(noise1, noise2))
    return torch.clamp(input + noise, 0, 1)  
  return input

device = "cuda" if torch.cuda.is_available() else "cpu"

class PGD:
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, images, labels, eps=1 / 255, alpha=0.01, iterations=40):
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

def train(dataset, data, method, normalization, augmentations, img_size,
          folds=1, epochs=1, batch_size=32, lr=0.001, fulltrain=False, load_model_path=None, return_best=False
          ):
    """
    Train a DNN for a number of epochs.

    # parts from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # adapted by: Christopher Otto
    """
    training_time = time.time()
    # use gpu for calculations if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    average_auc = []
    average_loss = []
    average_acc = []
    average_ap = []
    average_one_rec = []
    average_five_rec = []
    average_nine_rec = []

    # k-fold cross-val if folds > 1
    for fold in range(folds):
        if folds > 1:
            # doing k-fold cross-validation
            print(f"Fold: {fold}")

        best_acc = 0.0
        best_loss = 100
        current_acc = 0.0
        current_loss = 100.0
        best_auc = 0.0
        best_ap = 0.0

        # get train and val indices
        if fulltrain == False:
            if folds > 1:
                # cross validation
                train_idx, val_idx = kfold_cross_val(method, fold, data)
            else:
                _, _, _, _, train_idx, val_idx = holdout_val(
                    method, fold, data)
        # prepare training and validation data
        if fulltrain == True:
            train_dataset, train_loader = prepare_fulltrain_datasets(
                dataset, method, data, img_size, normalization, augmentations, batch_size)
        else:
            train_dataset, train_loader, val_dataset, val_loader = prepare_train_val(
                dataset, method, data, img_size, normalization, augmentations, batch_size, train_idx, val_idx)
        if load_model_path is None:
            # train model from pretrained imagenet or mesonet or noisy student weights
            if method == 'xception':
                # load the xception model
                #model = xception.imagenet_pretrained_xception()
                model = xception.xception(num_classes=1,pretrained=False)
            elif method == 'efficientnetb7':
                model = timm.create_model(
                    'tf_efficientnet_b7_ns', pretrained=True)
                # binary classification output
                model.classifier = nn.Linear(2560, 1)
            elif method == 'efficientnetb4':
                model = timm.create_model(
                    'tf_efficientnet_b4_ns', pretrained=True)
                # binary classification output
                model.classifier = nn.Linear(1792, 1)
            elif method == 'efficientnetb3':
                model = timm.create_model(
                    'tf_efficientnet_b3_ns', pretrained=True)
                # binary classification output
                model.classifier = nn.Linear(1536, 1)
            elif method == 'mesonet':
                # load MesoInception4 model
                model = mesonet.MesoInception4()
                # load mesonet weights that were pretrained on the mesonet dataset from https://github.com/DariusAf/MesoNet
                model.load_state_dict(torch.load(
                    "./deepfake_detector/pretrained_mods/weights/mesonet_pretrain.pth"))
        else:
            # continue to train model from custom checkpoint
            model = torch.load(load_model_path)

        if return_best:
            best_model_state = copy.deepcopy(model.state_dict())

        # put model on gpu
        model = model.cuda()
        model = nn.DataParallel(model).to(device)
        # binary cross-entropy loss
        loss_func = nn.BCEWithLogitsLoss()
        #loss_func = nn.CrossEntropyLoss()
        lr = lr
        # adam optimizer
        optimizer = Adam(model.parameters(), lr=lr)
        
        # cosine annealing scheduler
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=0.000001, last_epoch=-1)

        for e in range(epochs):
            print('#' * 20)
            print(f"Epoch {e}/{epochs}")
            print('#' * 20)
            # training and validation loop
            for phase in ["train", "val"]:
                if phase == "train":
                    # put layers in training mode
                    model.train()
                else:
                    # turn batchnorm and dropout layers to eval mode
                    model.eval()
                running_loss = 0.0
                running_corrects = 0.0
                running_auc_labels = []
                running_ap_labels = []
                running_auc_preds = []
                running_ap_preds = []
                adversary = PGD(model,device)

                if phase == "train":
                    # then load training data
                    for imgs, labels in tqdm(train_loader):
                        # put calculations on gpu
                        imgs = imgs.to(device)
                        labels = labels.to(device)
                        # set accumulated gradients to zero
                        optimizer.zero_grad()
                        perturbed_imgs = adversary(imgs,labels)
                        # forward pass of inputs and turn on gradient computation during train
                        with torch.set_grad_enabled(phase == "train"):
                            predictions = model(perturbed_imgs)
                            sig = torch.sigmoid(predictions)
                            # predictions for acc calculation; classification thresh 0.5
                            thresh_preds = torch.round(torch.sigmoid(predictions))
                            loss = loss_func(predictions.squeeze(-1), labels.type_as(predictions))
                            if USE_REG:
                                perturbed_imgs.requires_grad = True
                                reg = lipshitz_regularization(perturbed_imgs,model)
                                perturbed_imgs.requires_grad = False
                                loss = loss + reg
                            if phase == "train":
                                # backpropagate gradients
                                loss.backward()
                                # update parameters
                                optimizer.step()

                        running_loss += loss.item() * imgs.size(0)
                        # calc accuracy
                        running_corrects += torch.sum(thresh_preds ==
                                                      labels.unsqueeze(1))
                        running_auc_labels.extend(
                            labels.detach().cpu().numpy())
                        running_auc_preds.extend(
                            sig.detach().cpu().numpy().flatten().tolist())
                        running_ap_labels.extend(labels.detach().cpu().numpy())
                        running_ap_preds.extend(
                            sig.detach().cpu().numpy().flatten().tolist())
                    if phase == 'train':
                        # update lr
                        scheduler.step()
                    epoch_loss = running_loss / len(train_dataset)
                    epoch_acc = running_corrects / len(train_dataset)
                    try:
                        epoch_auc = roc_auc_score(running_auc_labels, running_auc_preds)
                    except ValueError:
                        epoch_auc = 0
                        pass
                    epoch_ap = average_precision_score(
                        running_ap_labels, running_ap_preds)
                    print(
                        f"{phase} Loss: {epoch_loss}, Acc: {epoch_acc}, AUC: {epoch_auc}, AP: {epoch_ap}")
                        
                    if fulltrain == True and e+1 == epochs:
                        # save model if epochs reached
                        print("Save fulltrain model.")
                        torch.save(
                            model.state_dict(), os.getcwd() + f'/{method}_{dataset}.pth')

                else:
                    if fulltrain == True:
                        continue
                    # get valitation data
                    for imgs, labels in tqdm(val_loader):
                        # put calculations on gpu
                        imgs = imgs.to(device)
                        labels = labels.to(device)
                        # set accumulated gradients to zero
                        optimizer.zero_grad()

                        perturbed_imgs = adversary(imgs,labels)                     
                        # forward pass of inputs and turn on gradient computation during train
                        with torch.set_grad_enabled(phase == "train"):
                            predictions = model(perturbed_imgs)
                            sig = torch.sigmoid(predictions)
                            # predictions for acc calculation; classification thresh 0.5
                            #thresh_preds : 
                            thresh_preds = torch.round(torch.sigmoid(predictions))
                            loss = loss_func(predictions.squeeze(-1), labels.type_as(predictions))

                        running_loss += loss.item() * imgs.size(0)
                        # calc accuracy
                        running_corrects += torch.sum(thresh_preds ==
                                                      labels.unsqueeze(1))

                        running_auc_labels.extend(
                            labels.detach().cpu().numpy())
                        running_auc_preds.extend(
                            sig.detach().cpu().numpy().flatten().tolist())
                        running_ap_labels.extend(labels.detach().cpu().numpy())
                        running_ap_preds.extend(
                            sig.detach().cpu().numpy().flatten().tolist())

                    epoch_loss = running_loss / len(val_dataset)
                    epoch_acc = running_corrects / len(val_dataset)
                    try:
                        epoch_auc = roc_auc_score(running_auc_labels, running_auc_preds)
                    except ValueError:
                        epoch_auc = 0
                        pass
                    epoch_ap = average_precision_score(
                        running_ap_labels, running_ap_preds)
                    print(
                        f"{phase} Loss: {epoch_loss}, Acc: {epoch_acc}, AUC: {epoch_auc}")

                    # save model if acc better than best acc
                    if epoch_acc > best_acc:
                        print("Found a better model.")
                        one_rec, five_rec, nine_rec = metrics.prec_rec(
                            running_auc_labels, running_auc_preds, method, alpha=100, plot=False)
                        best_acc = epoch_acc
                        best_auc = epoch_auc
                        best_loss = epoch_loss
                        best_ap = epoch_ap
                        if folds > 1:
                            torch.save(model.state_dict(), os.getcwd() + f'/{method}_best_acc_model_fold{fold}.pth')
                            if return_best:
                                best_model_state = copy.deepcopy(
                                    model.state_dict())
                        else:
                            print(best_acc)
                            print(best_auc)
                            print(best_loss)
                            print(best_ap)
                            with open("Output.txt", "w") as text_file:
                                print(f"Acc: {best_acc}", file=text_file)
                                print(f"AUC: {best_auc}", file=text_file)
                                print(f"Loss: {best_loss}", file=text_file)
                                print(f"AP: {best_ap}", file=text_file)
                                print(f"Epoch: {e+1}", file=text_file)
                            torch.save(model.state_dict(), os.getcwd() + f'/{method}_ep{e}_{best_acc}_{best_auc}_{best_ap}_{best_loss}.pth')
                            if return_best:
                                best_model_state = copy.deepcopy(
                                    model.state_dict())
                    # if loss is lower, but accuracy equal, take that model as best new model
                    # e.g. used with small datasets when accuracy goes to 1.0 quickly
                    elif epoch_acc == best_acc and epoch_loss < best_loss:
                        print("Found a better model.")
                        one_rec, five_rec, nine_rec = metrics.prec_rec(
                            running_auc_labels, running_auc_preds, method, alpha=100, plot=False)
                        best_acc = epoch_acc
                        best_auc = epoch_auc
                        best_loss = epoch_loss
                        best_ap = epoch_ap
                        if folds > 1:
                            torch.save(model.state_dict(), os.getcwd() + f'/{method}_best_acc_model_fold{fold}.pth')
                            if return_best:
                                best_model_state = copy.deepcopy(
                                    model.state_dict())
                        else:
                            print(best_acc)
                            print(best_auc)
                            print(best_loss)
                            print(best_ap)
                            torch.save(model.state_dict(), os.getcwd() + f'/{method}_ep{e}_{best_acc}_{best_auc}_{best_ap}_{best_loss}.pth')
                            if return_best:
                                best_model_state = copy.deepcopy(
                                    model.state_dict())

            gc.collect()
            torch.cuda.empty_cache()
        average_auc.append(best_auc)
        average_ap.append(best_ap)
        average_acc.append(best_acc)
        average_loss.append(best_loss)
        if not fulltrain:
            average_one_rec.append(one_rec)
            average_five_rec.append(five_rec)
            average_nine_rec.append(nine_rec)
        else:
            # only saved model is returned
            return model, 0, 0, 0, 0
    # average the best results of all folds
    average_auc = np.array(average_auc).mean()
    average_ap = np.array(average_ap).mean()
    average_acc = np.mean(np.asarray(
        [entry.cpu().numpy() for entry in average_acc]))
    average_loss = np.mean(np.asarray([entry for entry in average_loss]))
    average_one_rec = np.array(average_one_rec).mean()
    average_five_rec = np.array(average_five_rec).mean()
    average_nine_rec = np.array(average_nine_rec).mean()

    if folds > 1:
        print(f"Average AUC: {average_auc}")
        print(f"Average AP: {average_ap}")
        print(f"Average Acc: {average_acc}")
        print(f"Average Loss: {average_loss}")
        print()
        print("Average Cost (best possible cost is 0.0):")
        print(f"{average_one_rec} cost for 0.1 recall.")
        print(f"{average_five_rec} cost for 0.5 recall.")
        print(f"{average_nine_rec} cost for 0.9 recall.")
        print(
            f"Duration: {(time.time() - training_time) // 60} min and {(time.time() - training_time) % 60} sec.")
    else:
        print()
        print("Best models metrics:")
        print(f"Acc: {average_acc}")
        print(f"AUC: {average_auc}")
        print(f"AP: {average_ap}")
        print(f"Loss: {average_loss}")
        print()
        print("Cost (best possible cost is 0.0):")
        print(f"{average_one_rec} cost for 0.1 recall.")
        print(f"{average_five_rec} cost for 0.5 recall.")
        print(f"{average_nine_rec} cost for 0.9 recall.")
        print(
            f"Duration: {(time.time() - training_time) // 60} min and {(time.time() - training_time) % 60} sec.")

    if return_best:
        # load best model params
        model.load_state_dict(best_model_state)
    return model, average_auc, average_ap, average_acc, average_loss



def kfold_cross_val(method, fold, df):
    """
    Return train and val indices for 5 folds of 5-fold cross validation.
    """
    if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
        X = df['original'].values
        y = df['label'].values
    else:
        X = df['video'].values
        y = df['label'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=24)
    train = []
    val = []
    for train_index, val_index in kf.split(X):
        train.append(train_index)
        val.append(val_index)

    # return indices for fold
    return list(train[fold]), list(val[fold])


def holdout_val(method, fold, df):
    """
        Return training and validation data in a holdout split.
    """
    if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
        X = df['original'].values
        y = df['label'].values
        indices = df.index.values.tolist()
    else:
        X = df['video'].values
        y = df['label'].values
        indices = df.index.values.tolist()

    X_train, X_test, y_train, y_test, train_idx, val_idx = train_test_split(
        X, y, indices, test_size=0.2, random_state=24)
    return X_train, X_test, y_train, y_test, train_idx, val_idx


def prepare_fulltrain_datasets(dataset, method, data, img_size, normalization, augmentations, batch_size):
    """
    Prepare datasets for training with all data.
    """
    if dataset == 'celebdf':
        train_dataset = datasets.CelebDFDataset(
            data, img_size, method=method,  normalization=normalization, augmentations=augmentations)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataset, train_loader
    


def prepare_train_val(dataset, method, data, img_size, normalization, augmentations, batch_size, train_idx, val_idx):
    """
    Prepare training and validation dataset.
    """
    if dataset == 'celebdf':
        train_dataset = datasets.CelebDFDataset(
            data.iloc[train_idx], img_size, method=method, normalization=normalization, augmentations=augmentations)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_dataset = datasets.CelebDFDataset(
            data.iloc[val_idx], img_size, method=method, normalization=normalization, augmentations=None)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
    return train_dataset, train_loader, val_dataset, val_loader
