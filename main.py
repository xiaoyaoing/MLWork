import logging.config
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from configuration import config
from utils.CIFAR100 import CIFAR100
from utils.ImageFolder import ImageFolder
from utils.data_loader import get_statistics
from methods.finetune import Finetune
from collections import defaultdict
from dataset import get_datalist, get_classlist
from utils.sampler import RandomIdentitySampler

args = config.base_parser()


def getData():    # Data
    print('==> Preparing data..')

    if args.data == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        root = 'DataSet'
        traindir = root + '/cifar'
        num_classes = 100

    if args.data == 'cub':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values),
        ])
        root = 'DataSet/CUB_200_2011'
        traindir = os.path.join(root, 'train')
        testdir = os.path.join(root, 'test')

        num_classes = 200

    if args.data == 'car':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values),
        ])
        root = 'DataSet/Car196'
        traindir = os.path.join(root, 'train')
        testdir = os.path.join(root, 'test')

        num_classes = 196

    if args.data == 'flower':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values),
        ])
        root = 'DataSet/flowers'
        traindir = os.path.join(root, 'train')
        testdir = os.path.join(root, 'test')

        num_classes = 102

    if args.data == 'imagenet_sub' or args.data == 'imagenet_full':
        mean_values = [0.485, 0.456, 0.406]
        std_values = [0.229, 0.224, 0.225]
        transform_train = transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                 std=std_values)
        ])
        root = '/datatmp/datasets/ILSVRC12_256'
        traindir = os.path.join(root, 'train')
        num_classes = 100
    return transform_train,traindir,num_classes


def main():

    # logging.config.fileConfig("./configuration/logging.conf")
    # logger = logging.getLogger()
    #
    # os.makedirs(f"logs/{args.dataset}", exist_ok=True)
    # save_path = "tmp"
    # fileHandler = logging.FileHandler("logs/{}.log".format(save_path), mode="w")
    # formatter = logging.Formatter(
    #     "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    # )
    # fileHandler.setFormatter(formatter)
    # logger.addHandler(fileHandler)
    #
    #
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    #
    #     logger.info(f"Set the device ({device})")

    transform_train,traindir,num_classes=getData()

    num_task = args.task
    num_class_per_task = (num_classes - args.base) // (num_task - 1)

    np.random.seed(args.seed)
    random_perm = np.random.permutation(num_classes)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    fisher = {}
    prototype = {}
    for i in range(num_task):
        if i == 0:
            class_index = random_perm[:args.base]
        else:
            class_index = random_perm[args.base + (i - 1) * num_class_per_task:args.base + i * num_class_per_task]
        if args.data == 'cifar100':
            trainfolder = CIFAR100(
                root=traindir, train=True, download=True, transform=transform_train, index=class_index)
        else:
            trainfolder = ImageFolder(
                traindir, transform_train, index=class_index)

        train_loader = torch.utils.data.DataLoader(
            trainfolder, batch_size=args.BatchSize,
            sampler=RandomIdentitySampler(
                trainfolder, num_instances=args.num_instances),
            drop_last=True, num_workers=args.nThreads)

        feat_loader = torch.utils.data.DataLoader(
            trainfolder, batch_size=1, shuffle=True, drop_last=False)
        # Fix the random seed to be sure we have the same permutation for one experiment


        # 已有 train_loader, feat_loader

        #这里调用训练的接口

        # if args.method == 'EWC' or args.method == 'MAS':
        #     fisher = train_fun(args, train_loader,
        #                        feat_loader, i, fisher=fisher)
        # else:
        #     train_fun(args, train_loader, feat_loader, i)



if __name__ == "__main__":
    main()
