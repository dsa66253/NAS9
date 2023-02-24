import os
import torch
import torch.optim as optim
from test import TestController
# import torch.backends.cudnn as cudnn
import argparse
from torch import nn
from data.config import cfg_newnasmodel, trainDataSetFolder, seed
import numpy as np
from data.config import folder
from feature.make_dir import makeDir
from feature.random_seed import set_seed_cpu
from tqdm import tqdm
from models.retrainModel import NewNasModel
from utility.AccLossMonitor import AccLossMonitor
from feature.utility import setStdoutToFile, setStdoutToDefault
from feature.utility import getCurrentTime, accelerateByGpuAlgo, get_device
from utility.DatasetHandler import DatasetHandler
from  utility.DatasetReviewer import DatasetReviewer
import json 
from utility.HistDrawer import HistDrawer
from models.initWeight import initialize_weights
from utility.ValController import ValController
stdoutTofile = True
accelerateButUndetermine = cfg_newnasmodel["cuddbenchMark"]
recover = False
def printNetGrad(net):
    for name, para in net.named_parameters():
        print("grad", name, "\n", para)
        break
def parse_args(k=0):
    parser = argparse.ArgumentParser(description='imagenet nas Training')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    args = parser.parse_args()
    return args


def saveAccLoss(kth, lossRecord, accRecord):
    print("save record to ", folder["accLossDir"])
    try:
        np.save(os.path.join(folder["accLossDir"], "retrainTrainLoss_"+str(kth)), lossRecord["train"])
        np.save(os.path.join(folder["accLossDir"], "retrainValnLoss_"+str(kth)), lossRecord["val"])
        np.save(os.path.join(folder["accLossDir"], "retrainTrainAcc_"+str(kth)), accRecord["train"])
        np.save(os.path.join(folder["accLossDir"], "retrainValAcc_"+str(kth)), accRecord["val"])
    except Exception as e:
        print("Fail to save acc and loss")
        print(e)
def prepareDataSet():
    #info prepare dataset
    datasetHandler = DatasetHandler(trainDataSetFolder, cfg, seed_weight)
    # datasetHandler.addAugmentDataset(transforms.RandomHorizontalFlip(p=1))
    # datasetHandler.addAugmentDataset(transforms.RandomRotation(degrees=10))
    print("dataset:", trainDataSetFolder)
    print("training dataset set size:", len(datasetHandler.getTrainDataset()))
    print("val dataset set size:", len(datasetHandler.getValDataset()))
    print("class_to_idx", datasetHandler.getClassToIndex())
    return datasetHandler.getTrainDataset(), datasetHandler.getValDataset()

def prepareDataLoader(trainData, valData):
    #info prepare dataloader
    train_loader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valData, batch_size=batch_size, num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True)
    return train_loader, val_loader

def prepareLossFunction():
    #info prepare loss function
    print('Preparing loss function...')
    return  nn.CrossEntropyLoss()

def prepareModel(kth):
    #info load decode json
    filePath = os.path.join(folder["decode"], "{}th_decode.json".format(kth))
    f = open(filePath)
    archDict = json.load(f)
        
    #info prepare model
    print("Preparing model...")
    set_seed_cpu(seed_weight)
    net = NewNasModel(cellArch=archDict)
    net.train()
    net = net.to(device)
    print("net.cellArch:", net.cellArch)
    print("net", net)
    initialize_weights(net, seed_weight)
    # tmpF(net)
    return net
def preparedTransferModel(kth):
    #info load decode json
    filePath = os.path.join(folder["decode"], "{}th_decode.json".format(kth))
    f = open(filePath)
    archDict = json.load(f)
        
    #info prepare model
    print("Preparing model...")
    set_seed_cpu(seed_weight)
    net = NewNasModel(cellArch=archDict)
    net.train()
    net = net.to(device)
    print("net.cellArch:", net.cellArch)
    print("net", net)
    initialize_weights(net, seed_weight)

    #info prepare pretrain weight
    f = open("./curExperiment.json")
    exp = json.load(f)
    f.close()
    targetExpName = "1223.brutL0L1"
    for key in exp:
        expName = targetExpName+"."+key.split(".")[2]
    
    modelLoadPath = os.path.join("./log/1223.brutL0L1", expName, folder["retrainSavedModel"], "NewNasModel{}_Final.pt".format(kth) )
    print("modelLoadPath", modelLoadPath)
    tmpModelWeight = torch.load( modelLoadPath )
    net.load_state_dict(tmpModelWeight)
    # exit()
    # tmpF(net)
    return net
def prepareOpt(net):
    return optim.SGD(net.getWeight(), lr=initial_lr, momentum=momentum,
                    weight_decay=weight_decay)  # 是否采取 weight_decay

def saveCheckPoint(kth, epoch, optimizer, net, lossRecord, accReocrd):
    makeDir(folder["savedCheckPoint"])
    print("save check point kth {} epoch {}".format(kth, epoch))
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': lossRecord,
            'acc': accReocrd
            }, 
            os.path.join(folder["savedCheckPoint"], "{}_{}_{}.pt".format(args.network, kth, epoch)))
    except Exception as e:
        print("Failt to save check point")
        print(e)
        
def recoverFromCheckPoint(model, optimizer):
    pass
    checkpoint = torch.load(folder["savedCheckPoint"])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

def printNetWeight(net):
    for name, para in net.named_parameters():
        print(name, "\n", para.data)
        
def makeAllDir():
    for folderName in folder:
        print("making folder ", folder[folderName])
        makeDir(folder[folderName])
        
def compareNet(alexnet, net):
    # _, alexPara = alexnet.named_parameters()
    # _, nasPara = net.named_parameters()
    # print(alexPara)
    for alexnet, nasNet in zip(alexnet.named_parameters(), net.named_parameters()):
        alexnetLayerName, alexnetLayerPara = alexnet
        nasNetLayerName , nasNetLyaerPara = nasNet
        print(alexnetLayerName, nasNetLayerName)
        print(alexnetLayerPara.data.sum(), nasNetLyaerPara.data.sum())
        
        
def tmpF(net):
    print("tmpF")
    for netLayerName , netLyaerPara in net.named_parameters():
        print(netLayerName)
        print(netLyaerPara.data)
        
def weightCount(net):
    count = 0
    for netLayerName , netLyaerPara in net.named_parameters():
        print(netLyaerPara.device)
        shape = netLyaerPara.shape
        dim=1
        for e in shape:
            dim = e*dim
        count = count + dim
    return count
def gradCount(net):
    count = 0
    for netLayerName , netLyaerPara in net.named_parameters():
        if netLyaerPara.grad!=None:
            shape = netLyaerPara.grad.shape
            dim=1
            for e in shape:
                dim = e*dim
            count = count + dim
    return count

def reviewDatasetAcc(k, net):
    datasets = ["../dataset/test", "../dataset2/test", "../dataset3/test"] 
    DatasetReviewer(kth=str(k), allData=None, device=device, cfg=cfg)
    DatasetReviewer.reviewEachDataset(net, datasets)
    

def myTrain(kth, trainData, trainDataLoader, valDataLoader, net, model_optimizer, criterion, writer):
    record_train_loss = np.array([])
    record_val_loss = np.array([])
    record_train_acc = np.array([])
    record_val_acc = np.array([])
    record_test_acc = np.array([])
    for epoch in tqdm(range(cfg["epoch"]), unit =" iter on {}".format(kth)):
        print("start epoch", epoch)
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        # tqdm(range(start_iter, max_iter), unit =" iter on {}".format(kth))
        net.train() # set the model to training mode
        for i, data in enumerate(trainDataLoader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            model_optimizer.zero_grad() 
            outputs = net(inputs) 

            batch_loss = criterion(outputs, labels)
            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            batch_loss.backward() 
            model_optimizer.step() 
            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()
            
        record_train_acc = np.append(record_train_acc, train_acc/len(trainData)*100)
        record_train_loss = np.append(record_train_loss, train_loss/len(trainDataLoader))
        
        valAcc = valC.val(net)
        record_val_acc = np.append(record_val_acc, valAcc)
        record_val_loss = np.append(record_val_loss, torch.Tensor([0]))
        # print("epoch", epoch)
        # print("record_val_acc", record_val_acc)
        # print("record_train_acc", record_train_acc)
        #info test set
        testAcc = testC.test(net)
        record_test_acc = np.append(record_test_acc, testAcc)
        
    last_epoch_val_acc = valC.val(net)
    lossRecord = {"train": record_train_loss, "val": record_val_loss}
    accRecord = {"train": record_train_acc, "val": record_val_acc, "test": record_test_acc}
    print("start test model before save model")
    testAcc = testC.test(net)
    # print(record_val_acc)
    # print(record_train_acc)
    # testC.printAllModule(net)
    torch.save(net.state_dict(), os.path.join(folder["retrainSavedModel"], cfg['name'] + str(kth) + '_Final.pt'))
    
    return last_epoch_val_acc, lossRecord, accRecord


if __name__ == '__main__':
    device = get_device()
    torch.device(device)
    print("running on device: {}".format(device))
    torch.set_printoptions(precision=6, sci_mode=False, threshold=1000)
    torch.set_default_dtype(torch.float32) #* torch.float will slow the training speed
    valList = []
    cfg = cfg_newnasmodel   
    for k in range(0, cfg["numOfKth"]):
        #info handle stdout to a file
        if stdoutTofile:
            f = setStdoutToFile(folder["log"]+"/retrain_5cell_{}th.txt".format(str(k)))
        
        print("working directory ", os.getcwd())
        #info set seed
        seed_weight = seed[str(k)]
            
        args = parse_args(str(k))
            
        accelerateByGpuAlgo(cfg["cuddbenchMark"])
        set_seed_cpu(seed_weight)
        #! test same initial weight
        
        makeAllDir()
            
        batch_size = cfg['batch_size']

        #todo find what do these stuff do
        num_workers = args.num_workers
        momentum = args.momentum
        weight_decay = args.weight_decay
        initial_lr = args.lr
        gamma = args.gamma

        

        
        
        # writer = SummaryWriter(log_dir=folder["tensorboard_retrain"], comment="{}th".format(str(k)))
        
        print("seed_weight{} start at ".format(seed_weight), getCurrentTime())
        print("cfg", cfg)
        
        #info training process 
        trainData, valData = prepareDataSet()
        trainDataLoader, valDataLoader = prepareDataLoader(trainData, valData)
    
        criterion = prepareLossFunction()
        net = prepareModel(k)
        # net = preparedTransferModel(k)
        histDrawer = HistDrawer(folder["pltSavedDir"])
        histDrawer.drawNetConvWeight(net, tag="ori_{}".format(str(k)))
        model_optimizer = prepareOpt(net)
        #info test controller
        testC = TestController(cfg, device)
        testC.printAllModule(net)
        #info validation controller
        valC = ValController(cfg, device, valDataLoader, criterion)
        # info training loop
        last_epoch_val_acc, lossRecord, accRecord = myTrain(k, trainData, trainDataLoader, valDataLoader, net, model_optimizer, criterion, writer=None)  # 進入model訓練

        #info exam each dataset accuracy respectively
        # reviewDatasetAcc(k)

        histDrawer.drawNetConvWeight(net, tag="trained_{}".format(str(k)))
        #info record training processs
        alMonitor = AccLossMonitor(k, folder["pltSavedDir"], folder["accLossDir"], trainType="retrain")
        alMonitor.plotAccLineChart(accRecord)
        alMonitor.plotLossLineChart(lossRecord)
        alMonitor.saveAccLossNp(accRecord, lossRecord)
        testC.saveDatasetAcc(kth=k)
        valList.append(last_epoch_val_acc)
        print('retrain validate accuracy:')
        print(valList)
        # writer.close()
        #info handle output file
        if stdoutTofile:
            setStdoutToDefault(f)
            
    print('retrain validate accuracy:')
    print(valList)



