from torchvision import transforms
import torch
import torch.optim as optim
# import torch.backends.cudnn as cudnn
import argparse
from torch import nn
from torchvision import datasets
from data.config import cfg_newnasmodel as cfg
from tensorboardX import SummaryWriter
import numpy as np
from feature.random_seed import set_seed_cpu
import matplotlib.pyplot as plt

class DatasetHandler():
    def __init__(self, trainDataSetFolder, cfg, seed=10):
        self.seed = seed
        self.augmentDatasetList = []
        self.trainDataSetFolder = trainDataSetFolder
        self.normalize = self.resize(cfg["image_size"])
        self.originalData = None
        self.originalTrainDataset = None
        self.originalValDataset = None
        self.getTrainDataset()
        self.getValDataset()
    def __split_data(self, all_data, ratio=0.2):
        n = len(all_data)  # total number of examples
        n_val = int(ratio * n)  # take ~10% for val
        set_seed_cpu(self.seed)
        train_data, val_data = torch.utils.data.random_split(all_data, [(n - n_val), n_val])
        return train_data, val_data
    def resize(self, img_dim):
        return transforms.Compose([transforms.Resize(img_dim),
                                                transforms.CenterCrop(img_dim),
                                                transforms.ToTensor(),
                                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])
    def addAugmentDataset(self, augmentF):
        # newAugmentData = augmentF(self.originalTrainData)
        
        try:
            newAugmentDataset = datasets.ImageFolder(self.trainDataSetFolder, transform=transforms.Compose([
                self.normalize,
                augmentF
            ]))
            newAugmentTrainDataset, _ = self.__split_data(newAugmentDataset, 0.2)
        except Exception as e:
            print("Fail to load data set from: ",  self.trainDataSetFolder)
            print(e)
            exit()
        # self.augmentDatasetList.append(newAugmentData)
        # print("self.trainDataset", type(self.trainDataset))
        # print("newAugmentTrainDataset", type(newAugmentTrainDataset))
        # print("before concate", len(self.originalTrainDataset))
        self.originalTrainDataset = torch.utils.data.ConcatDataset([self.originalTrainDataset, newAugmentTrainDataset])
        # print("after concate", len(self.originalTrainDataset))
        # exit()
    def getTrainDataset(self):
        if self.originalTrainDataset==None:
            self.originalData = datasets.ImageFolder(self.trainDataSetFolder, transform=transforms.Compose([
                    self.normalize,
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
            self.originalTrainDataset, self.originalValDataset = self.__split_data(self.originalData, 0.2)
            self.augmentDatasetList.append(self.originalTrainDataset)
        # print("trainDataSetFolder", self.trainDataSetFolder)
        # print("tatal number of train images: ", len(self.augmentDatasetList))
        return self.originalTrainDataset
    
    def getValDataset(self):
        if self.originalValDataset==None:
            self.originalData = datasets.ImageFolder(self.trainDataSetFolder, transform=transforms.Compose([
                    self.normalize,
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
            self.originalTrainDataset, self.originalValDataset = self.__split_data(self.originalData, 0.2)
            self.augmentDatasetList.append(self.originalTrainDataset)
        # print("trainDataSetFolder", self.trainDataSetFolder)
        # print("tatal number of val images: ", len(self.originalValDataset))
        return self.originalValDataset
    def getTestDataset(self):
        self.testDataset = datasets.ImageFolder(self.trainDataSetFolder, transform=transforms.Compose([
                    self.normalize,
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
        # print("tatal number of test images: ", len(self.testDataset))
        return self.testDataset
    @staticmethod
    def getOriginalDataset(trainDataSetFolder, cfg, seed=10):
        datsetHandle = DatasetHandler(trainDataSetFolder, cfg, seed)
        return datsetHandle.originalTrainDataset
    def getLength(self):
        return len(self.trainDataset)
    def __getitem__(self, index):
        return self.trainDataset[index]
    def getClassToIndex(self)->dict:
        return self.originalData.class_to_idx
    def getIndexToClass(self)->dict:
        # key of return object is Int type
        IndexToClass = {}
        for key in self.originalData.class_to_idx:
            IndexToClass[self.originalData.class_to_idx[key]] = key
        return IndexToClass
def printImage(train_data, index):

    fig, axes = plt.subplots(len(train_data)//5, 5)
    for i in range(len(train_data)):
        img, label = train_data[i]
        # print(i//5, i%5)

        if len(train_data)//5==1:
            axes[i%5].imshow(img.permute(1, 2, 0))
        else:
            axes[i//5, i%5].imshow(img.permute(1, 2, 0))
        # if i%5==4:
    plt.savefig('foo{}.png'.format(index))   
    

    
if __name__ == "__main__":
    trainDataSetFolder = "../datasetPractice/train"
    datasetHandler = DatasetHandler(trainDataSetFolder, cfg, 10)
    print(datasetHandler.getLength())
    printImage(datasetHandler.getTrainDataset(), "0")

    datasetHandler.addAugmentDataset(transforms.RandomHorizontalFlip(p=1))
    print(datasetHandler.getLength())
    printImage(datasetHandler.getTrainDataset(), "1")
    
    datasetHandler.addAugmentDataset(transforms.RandomGrayscale(p=1))
    print(datasetHandler.getLength())
    printImage(datasetHandler.getTrainDataset(), "2")
    
    # train_data = datasetHandler.getTrainDataset()


    
    
    exit()
    set_seed_cpu(20)
    train_data, val_data = prepareDataSet()
    set_seed_cpu(20)
    train_data2, val_data2 = prepareDataSet()
    # print(len(train_data), len(val_data))
    # figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    fig, axes = plt.subplots(2, 5)
    
    
    
    
    print(len(train_data))
    for i in range(len(train_data)+len(train_data2)):
        if i//5==0:
            img, label = train_data[i-1]
            print(i//5, i%5)
            axes[i//5, i%5].imshow(img.permute(1, 2, 0))
        else:
            img, label = train_data2[(i-1)%5]
            print(i//5, i%5)
            axes[i//5, i%5].imshow(img.permute(1, 2, 0))
        # print(type(axes))
        # axes[0][i].add_image(img.numpy())
        # plt.axis("off")
        # plt.savefig('foo.png')
        # plt.imshow(img.permute(1, 2, 0), cmap="gray")
    # plt.show()
    plt.savefig('foo.png')
        