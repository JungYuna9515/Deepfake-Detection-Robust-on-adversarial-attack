import argparse
import copy
import os
import shutil
import test
import time
import zipfile
import timm
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import metrics
import matplotlib.pyplot as plt
import cv2
import datasets
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import train
import pgd_train
import utils
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from albumentations import (
    Compose, FancyPCA, GaussianBlur, GaussNoise, HorizontalFlip,
    HueSaturationValue, ImageCompression, OneOf, PadIfNeeded,
    RandomBrightnessContrast, Resize, ShiftScaleRotate, ToGray)
from pretrained_mods import xception
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from facedetector.retinaface import df_retinaface
from pretrained_mods import efficientnetb1lstm
from pretrained_mods import mesonet
from pretrained_mods import resnetlstm
from utils import vidtimit_setup_real_videos



parser = argparse.ArgumentParser(
    description='Start deepfake detection.')
parser.add_argument('--detect_single', default=False,
                    type=bool, help='Choose for single prediction.')
parser.add_argument('--benchmark', default=False, type=bool,
                    help='Choose for benchmarking.')
parser.add_argument('--train', default=False, type=bool,
                    help='Choose for training.')
parser.add_argument('--path_to_vid', default=None,
                    type=str, help='Choose video path.')
parser.add_argument('--path_to_img', default=None,
                    type=str, help='Choose image path.')
parser.add_argument('--detection_method', default="xception_clebdf",
                    type=str, help='Choose detection method.')
parser.add_argument('--data_path', default=None, type=str,
                    help='Specify path to dataset.')
parser.add_argument('--dataset', default="celebdf", type=str,
                    help='Specify the name of the dataset.')
parser.add_argument('--cmd', default="True", type=str,
                    help='True if executed via command line.')
parser.add_argument('--model_type', default="xception",
                    type=str, help='Choose detection model type for training.')
parser.add_argument('--epochs', default=20,
                    type=int, help='Choose number of training epochs.')
parser.add_argument('--batch_size', default=32,
                    type=int, help='Choose the minibatch size.')
parser.add_argument('--lr', default=0.0001,
                    type=int, help='Choose the minibatch size.')
parser.add_argument('--folds', default=1,
                    type=int, help='Choose validation folds.')
parser.add_argument('--augs', default="weak",
                    type=str, help='Choose augmentation strength.')
parser.add_argument('--fulltrain', default=False,
                    type=bool, help='Choose whether to train with the full dataset and no validation set.')
parser.add_argument('--facecrops_available', default=False,
                    type=bool, help='Choose whether videos are already preprocessed.')
parser.add_argument('--face_margin', default=0.3,
                    type=float, help='Choose the face margin.')
parser.add_argument('--seed', default=24,
                    type=int, help='Choose the random seed.')
parser.add_argument('--save_path', default=None,
                    type=str, help='Choose the path where face crops shall be saved.')                       
                                                             
                                      
                                      


class DFDetector():
    """
    The Deepfake Detector. 
    It can detect on a single video, 
    benchmark several methods on benchmark datasets
    and train detectors on several datasets.
    """

    def __init__(self):
        pass

    @classmethod
    def detect_single(cls, video_path=None, image_path=None, label=None, method="xception_uadfv", cmd=False):
        """Perform deepfake detection on a single video with a chosen method."""
        # prepare the method of choice
        sequence_model = False
        if method == "xception_celebdf":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "Xception_CELEB-DF"
        elif method == "efficientnetb3_celebdf":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B3_CELEB-DF"
        elif method == "efficientnetb4_celebdf":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B4_CELEB-DF"
        elif method == "efficientnetb7_celebdf":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B7_CELEB-DF"
        elif method == "mesonet_celebdf":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "MesoNet_CELEB-DF"
        if video_path:
            if not method == 'dfdcrank90_celebdf' and not method == 'dfdcrank90_dfdc':
                data = [[1, video_path]]
                df = pd.DataFrame(data, columns=['label', 'video'])
                loss = test.inference(
                    model, df, img_size, normalization, dataset=None, method=method, face_margin=0.3, sequence_model=sequence_model, num_frames=20, single=True, cmd=cmd)

            if round(loss) == 1:
                result = "Deepfake detected."
                print("Deepfake detected.")
                return used, result
            else:
                result = "This is a real video."
                print("This is a real video.")
                return used, result

    @classmethod
    def benchmark(cls, dataset=None, data_path=None, method="xception_celebdf", seed=24):
        """Benchmark deepfake detection methods against popular deepfake datasets.
           The methods are already pretrained on the datasets. 
           Methods get benchmarked against a test set that is distinct from the training data.
        # Arguments:
            dataset: The dataset that the method is tested against.
            data_path: The path to the test videos.
            method: The deepfake detection method that is used.
        # Implementation: Christopher Otto
        """
        # seed numpy and pytorch for reproducibility
        reproducibility_seed(seed)
        if method not in ['xception_celebdf','xception_dfdc', 'efficientnetb3_celebdf', 'efficientnetb4_celebdf', 'efficientnetb7_celebdf','efficientnetb7_dfdc', 'mesonet_celebdf','mesonet_dfdc', 'resnet_lstm_celebdf', 'resnet_lstm_dfdc','efficientnetb1_lstm_celebdf', 'efficientnetb1_lstm_dfdc','dfdcrank90_celebdf', 'dfdcrank90_dfdc']:
            raise ValueError("Method is not available for benchmarking.")
        else:
            # method exists
            cls.dataset = dataset
            cls.data_path = data_path
            cls.method = method
            if method in []:
                face_margin = 0.0
            else:
                face_margin = 0.3
        if cls.dataset == 'celebdf':
            num_frames = 20
            setup_celebdf_benchmark(cls.data_path, cls.method)
        else:
            raise ValueError(f"{cls.dataset} does not exist.")
        # get test labels for metric evaluation
        df = label_data(dataset_path=cls.data_path,
                        dataset=cls.dataset, test_data=True)
        # prepare the method of choice
        if cls.method == 'xception_celebdf':
            model, img_size, normalization = prepare_method(
                method=cls.method, dataset=cls.dataset, mode='test')
        elif cls.method == 'efficientnetb7_celebdf':
            model, img_size, normalization = prepare_method(
                method=cls.method, dataset=cls.dataset, mode='test')
        elif cls.method == 'efficientnetb4_celebdf':
            model, img_size, normalization = prepare_method(
                method=cls.method, dataset=cls.dataset, mode='test')
        elif cls.method == 'efficientnetb3_celebdf':
            model, img_size, normalization = prepare_method(
                method=cls.method, dataset=cls.dataset, mode='test')
        elif cls.method == 'mesonet_celebdf':
            model, img_size, normalization = prepare_method(
                method=cls.method, dataset=cls.dataset, mode='test')

        print(f"Detecting deepfakes with \033[1m{cls.method}\033[0m ...")
        # benchmarking
        auc, ap, loss, acc = test.inference(
            model, df, img_size, normalization, dataset=cls.dataset, method=cls.method, face_margin=face_margin, num_frames=num_frames)
        return [auc, ap, loss, acc]

    @classmethod
    def train_method(cls, dataset=None, data_path=None, method="xception", img_save_path=None, epochs=1, batch_size=32,
                     lr=0.001, folds=1, augmentation_strength='weak', fulltrain=False, faces_available=False, face_margin=0, seed=24):
        """Train a deepfake detection method on a dataset."""
        if img_save_path is None:
            raise ValueError(
                "Need a path to save extracted images for training.")
        cls.dataset = dataset
        print(f"Training on {cls.dataset} dataset.")
        cls.data_path = data_path
        cls.method = method
        cls.epochs = epochs
        cls.batch_size = batch_size
        cls.lr = lr
        cls.augmentations = augmentation_strength
        # no k-fold cross val if folds == 1
        cls.folds = folds
        # whether to train on the entire training data (without val sets)
        cls.fulltrain = fulltrain
        cls.faces_available = faces_available
        cls.face_margin = face_margin
        print(f"Training on {cls.dataset} dataset with {cls.method}.")
        # seed numpy and pytorch for reproducibility
        reproducibility_seed(seed)
        #folder_count = 35
        _, img_size, normalization = prepare_method(
            cls.method, dataset=cls.dataset, mode='train')
        # # get video train data and labels
        df = label_data(dataset_path=cls.data_path,
                        dataset=cls.dataset, test_data=False, fulltrain=cls.fulltrain)
        # detect and extract faces if they are not available already
        if not cls.faces_available:
            if cls.dataset == 'celebdf':
                addon_path = '/facecrops/'
                # check if all folders are available
                if not os.path.exists(img_save_path + '/Celeb-real/'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"Celeb-real\" folder is missing.")
                if not os.path.exists(img_save_path + '/Celeb-synthesis/'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"Celeb-synthesis\" folder is missing.")
                if not os.path.exists(img_save_path + '/YouTube-real/'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"YouTube-real\" folder is missing.")
                if not os.path.exists(img_save_path + '/List_of_testing_videos.txt'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"List_of_testing_videos.txt\" file is missing.")
                if not os.path.exists(img_save_path + '/facecrops/'):
                    # create directory in save path for face crops
                    os.mkdir(img_save_path + addon_path)
                    os.mkdir(img_save_path + '/facecrops/real/')
                    os.mkdir(img_save_path + '/facecrops/fake/')
                #else:
                    # delete create again if it already exists with old files
                    #shutil.rmtree(img_save_path + '/facecrops/')
                    #os.mkdir(img_save_path + addon_path)
                    #os.mkdir(img_save_path + '/facecrops/real/')
                    #os.mkdir(img_save_path + '/facecrops/fake/')
 
            if cls.dataset == 'dfdc':
                num_frames = 5
            else:
                num_frames = 25
            print(
                f"Detect and save {num_frames} faces from each video for training.")
            if cls.face_margin > 0.0:
                print(
                    f"Apply {cls.face_margin*100}% margin to each side of the face crop.")
            else:
                print("Apply no margin to the face crop.")
            # load retinaface face detector
            net, cfg = df_retinaface.load_face_detector()
            for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
                video = row.loc['video']
                label = row.loc['label']
                vid = os.path.join(video)
                if cls.dataset == 'celebdf':
                    vid_name = row.loc['video_name']
                    if label == 1:
                        video = vid_name
                        save_dir = os.path.join(
                            img_save_path + '/facecrops/fake/')
                    else:
                        video = vid_name
                        save_dir = os.path.join(
                            img_save_path + '/facecrops/real/')
                # detect faces, add margin, crop, upsample to same size, save to images
                faces = df_retinaface.detect_faces(
                    net, vid, cfg, num_frames=num_frames)
                # save frames to directory
                vid_frames = df_retinaface.extract_frames(
                    faces, video, save_to=save_dir, face_margin=cls.face_margin, num_frames=num_frames, test=False)

        # put all face images in dataframe
        df_faces = label_data(dataset_path=cls.data_path,
                              dataset=cls.dataset, method=cls.method, face_crops=True, test_data=False, fulltrain=cls.fulltrain)
        # choose augmentation strength
        augs = df_augmentations(img_size, strength=cls.augmentations)
        # start method training
        
        model, average_auc, average_ap, average_acc, average_loss = pgd_train.train(dataset=cls.dataset, data=df_faces,
                                                                                method=cls.method, img_size=img_size, normalization=normalization, augmentations=augs,
                                                                                folds=cls.folds, epochs=cls.epochs, batch_size=cls.batch_size, lr=cls.lr, fulltrain=cls.fulltrain
                                                                                )
        return model, average_auc, average_ap, average_acc, average_loss


def prepare_method(method, dataset, mode='train'):
    """Prepares the method that will be used for training or benchmarking."""
    if method == 'xception' or method == 'xception_celebdf':
        img_size = 299
        normalization = 'xception'
        if mode == 'test':
            model = xception.imagenet_pretrained_xception()
            # load the xception model that was pretrained on the respective datasets training data
            if method == 'xception_celebdf':
                model_params = torch.load(
                    os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                print(os.getcwd(
                ) + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params,strict=False)
            return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    elif method == 'efficientnetb7' or method == 'efficientnetb7_celebdf':
        # 380 image size as introduced here https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721
        img_size = 380
        normalization = 'imagenet'
        if mode == 'test':
            if method == 'efficientnetb7_celebdf':
                # successfully used by https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721 (noisy student weights)
                model = timm.create_model(
                    'tf_efficientnet_b7_ns', pretrained=True)
                model.classifier = nn.Linear(2560, 1)
                # load the efficientnet model that was pretrained on the uadfv training data
                model_params = torch.load(
                    os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params,strict=False)
            return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    elif method == 'efficientnetb4' or method == 'efficientnetb4_celebdf':
        # 380 image size as introduced here https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721
        img_size = 380
        normalization = 'imagenet'
        if mode == 'test':
            if method == 'efficientnetb4_celebdf':
                # successfully used by https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721 (noisy student weights)
                model = timm.create_model(
                    'tf_efficientnet_b4_ns', pretrained=True)
                model.classifier = nn.Linear(1792, 1)
                # load the efficientnet model that was pretrained on the uadfv training data
                model_params = torch.load(
                    os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params,strict=False)
            return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    elif method == 'efficientnetb3' or method == 'efficientnetb3_celebdf':
        # 380 image size as introduced here https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721
        img_size = 380
        normalization = 'imagenet'
        if mode == 'test':
            if method == 'efficientnetb3_celebdf':
                # successfully used by https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721 (noisy student weights)
                model = timm.create_model(
                    'tf_efficientnet_b3_ns', pretrained=True)
                model.classifier = nn.Linear(1536, 1)
                # load the efficientnet model that was pretrained on the uadfv training data
                model_params = torch.load(
                    os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params,strict=False)
            return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    elif method == 'mesonet' or method == 'mesonet_celebdf':
        # 256 image size as proposed in the MesoNet paper (https://arxiv.org/abs/1809.00888)
        img_size = 256
        # use [0.5,0.5,0.5] normalization scheme, because no imagenet pretraining
        normalization = 'xception'
        if mode == 'test':
            if method == 'mesonet_celebdf':
                # load MesoInception4 model
                model = mesonet.MesoInception4()
                # load the mesonet model that was pretrained on the uadfv training data
                model_params = torch.load(
                    os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                print(os.getcwd(
                ) + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params)
                return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    else:
        raise ValueError(
            f"{method} is not available. Please use one of the available methods.")


def label_data(dataset_path=None, dataset='celebdf', method='xception', face_crops=False, test_data=False, fulltrain=False):
    """
    Label the data.
    # Arguments:
        dataset_path: path to data
        test_data: binary choice that indicates whether data is for testing or not.
    # Implementation: Christopher Otto
    """
    # structure data from folder in data frame for loading
    if dataset_path is None:
        raise ValueError("Please specify a dataset path.")
    if not test_data:
        if dataset == 'celebdf':
            # prepare celebdf training data by
            # reading in the testing data first
            df_test = pd.read_csv(
                dataset_path + '/List_of_testing_videos.txt', sep=" ", header=None)
            df_test.columns = ["label", "video"]
            # switch labels so that fake label is 1
            df_test['label'] = df_test['label'].apply(switch_one_zero)
            df_test['video'] = dataset_path + '/' + df_test['video']
            # structure data from folder in data frame for loading
            if not face_crops:
                video_path_real = os.path.join(dataset_path + "/Celeb-real/")
                video_path_youtube_real = os.path.join(dataset_path + "/YouTube-real/")
                video_path_fake = os.path.join(
                    dataset_path + "/Celeb-synthesis/")
                real_list = []
                for _, _, videos in os.walk(video_path_real):
                    for video in tqdm(videos):
                        # label 0 for real image
                        real_list.append({'label': 0, 'video': video})
                youtube_list = []
                for _, _, videos in os.walk(video_path_youtube_real):
                    for video in tqdm(videos):
                        # label 0 for real image
                        youtube_list.append({'label': 0, 'video': video})

                fake_list = []
                for _, _, videos in os.walk(video_path_fake):
                    for video in tqdm(videos):
                        # label 1 for deepfake image
                        fake_list.append({'label': 1, 'video': video})

                # put data into dataframe
                df_real = pd.DataFrame(data=real_list)
                df_youtube_real = pd.DataFrame(data=youtube_list)
                df_fake = pd.DataFrame(data=fake_list)
                # add real and fake path to video file name
                df_real['video_name'] = df_real['video']
                df_youtube_real['video_name'] = df_youtube_real['video']
                df_fake['video_name'] = df_fake['video']
                df_real['video'] = video_path_real + df_real['video']
                df_youtube_real['video'] = video_path_youtube_real + df_youtube_real['video']
                df_fake['video'] = video_path_fake + df_fake['video']
                # put testing vids in list
                testing_vids = list(df_test['video'])
                # remove testing videos from training videos
                df_real = df_real[~df_real['video'].isin(testing_vids)]
                print(len(df_real))
                df_youtube_real = df_youtube_real[~df_youtube_real['video'].isin(testing_vids)]
                print(len(df_youtube_real)," = ")
                df_fake = df_fake[~df_fake['video'].isin(testing_vids)]
                df_real = pd.concat([df_real,df_youtube_real],ignore_index=True)
                # undersampling strategy to ensure class balance of 50/50
                df_fake_sample = df_fake.sample(
                    n=len(df_real), random_state=24).reset_index(drop=True)
                # concatenate both dataframes to get full training data (964 training videos with 50/50 class balance)
                df = pd.concat([df_real, df_fake_sample], ignore_index=True)
            else:
                # if face crops available go to path with face crops
                video_path_crops_real = os.path.join(
                    dataset_path + "/facecrops/real/")
                video_path_crops_fake = os.path.join(
                    dataset_path + "/facecrops/fake/")
                # add labels to videos
                data_list = []
                for _, _, videos in os.walk(video_path_crops_real):
                    for video in tqdm(videos):
                        # label 0 for real video
                        data_list.append(
                            {'label': 0, 'video': video_path_crops_real + video})

                for _, _, videos in os.walk(video_path_crops_fake):
                    for video in tqdm(videos):
                        # label 1 for deepfake video
                        data_list.append(
                            {'label': 1, 'video': video_path_crops_fake + video})
                # put data into dataframe
                df = pd.DataFrame(data=data_list)
                if len(df) == 0:
                    raise ValueError(
                        "No faces available. Please set faces_available=False.")
    else:
        if dataset == 'celebdf':
            # reading in the celebdf testing data
            df_test = pd.read_csv(
                dataset_path + '/List_of_testing_videos.txt', sep=" ", header=None)
            df_test.columns = ["label", "video"]
            # switch labels so that fake label is 1
            df_test['label'] = df_test['label'].apply(switch_one_zero)
            df_test['video'] = dataset_path + '/' + df_test['video']
            print(f"{len(df_test)} test videos.")
            return df_test
        # put data into dataframe
        df = pd.DataFrame(data=data_list)

    if test_data:
        print(f"{len(df)} test videos.")
    else:
        if face_crops:
            print(f"Lead to: {len(df)} face crops.")
        else:
            print(f"{len(df)} train videos.")
    print()
    return df


def df_augmentations(img_size, strength="weak"):
    """
    Augmentations with the albumentations package.
    # Arguments:
        strength: strong or weak augmentations

    # Implementation: Christopher Otto
    """
    if strength == "weak":
        print("Weak augmentations.")
        augs = Compose([
            # hflip with prob 0.5
            HorizontalFlip(p=0.5),
            # adjust image to DNN input size
            Resize(width=img_size, height=img_size)
            #Resize(width=img_size,height=img_size),
            #HorizontalFlip(p=0.5),
            #ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            #GaussNoise(p=0.1),
            #GaussianBlur(blur_limit=3, p=0.05),
            #PadIfNeeded(min_height=img_size, min_width=img_size,
                        #border_mode=cv2.BORDER_CONSTANT)
            ])
        return augs
    elif strength == "strong":
        print("Strong augmentations.")
        # augmentations via albumentations package
        # augmentations adapted from Selim Seferbekov's 3rd place private leaderboard solution from
        # https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721
        augs = Compose([
            # hflip with prob 0.5
            HorizontalFlip(p=0.5),
            ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            GaussNoise(p=0.1),
            GaussianBlur(blur_limit=3, p=0.05),
            # adjust image to DNN input size
            Resize(width=img_size, height=img_size)
        ])
        return augs
    else:
        raise ValueError(
            "This augmentation option does not exist. Choose \"weak\" or \"strong\".")


def reproducibility_seed(seed):
    print(f"The random seed is set to {seed}.")
    # set numpy random seed
    np.random.seed(seed)
    # set pytorch random seed for cpu and gpu
    torch.manual_seed(seed)
    # get deterministic behavior
    torch.backends.cudnn.deterministic = True


def switch_one_zero(num):
    """Switch label 1 to 0 and 0 to 1
        so that fake videos have label 1.
    """
    if num == 1:
        num = 0
    else:
        num = 1
    return num

def setup_celebdf_benchmark(data_path, method):
    """
    Setup the folder structure of the Celeb-DF Dataset.
    """
    if data_path is None:
        raise ValueError("""Please go to https://github.com/danmohaha/celeb-deepfakeforensics
                                and scroll down to the dataset section.
                                Click on the link \"this form\" and download the dataset. 
                                Extract the files and organize the folders follwing this folder structure:
                                ./celebdf/
                                        Celeb-real/
                                        Celeb-synthesis/
                                        YouTube-real/
                                        List_of_testing_videos.txt
                                """)
    if data_path.endswith("Celeb-DF-v2"):
        print(
            f"Benchmarking \033[1m{method}\033[0m on the \033[1m Celeb-DF \033[0m dataset with ...")
    else:
        raise ValueError("""Please organize the dataset directory in this way:
                            ./Celeb-DF-v2/
                                    Celeb-real/
                                    Celeb-synthesis/
                                    YouTube-real/
                                    List_of_testing_videos.txt
                        """)


def main():
    # parse arguments
    args = parser.parse_args()
    # initialize the deepfake detector with the desired task
    if args.detect_single:
        print(f"Detecting with {args.detection_method}.")
        DFDetector.detect_single(
            video_path=args.path_to_vid, image_path=args.path_to_img, method=args.detection_method, cmd=args.cmd)
    elif args.benchmark:
        print(args)
        DFDetector.benchmark(
            dataset=args.dataset, data_path=args.data_path, method=args.detection_method)
    elif args.train:
        print(args)
        print(args.facecrops_available)
        DFDetector.train_method(dataset=args.dataset, data_path=args.data_path, method=args.model_type, img_save_path=args.save_path, epochs=args.epochs, batch_size=args.batch_size,
                     lr=args.lr, folds=args.folds, augmentation_strength=args.augs, fulltrain=args.fulltrain,  face_margin=args.face_margin, faces_available=args.facecrops_available, seed=args.seed)
    else:
        print("Please choose one of the three modes: detect_single, benchmark, or train.")


if __name__ == '__main__':
    main()
