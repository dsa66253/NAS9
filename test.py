import argparse
from tqdm import tqdm
import os
import copy
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import Subset
from torchvision import transforms, datasets
from data.config import  cfg_alexnet, folder, seed, cfg_newnasmodel as cfg, testDataSetFolder, dataset1Name, dataset2Name, dataset3Name
from models.retrainModel import NewNasModel
# from model import Model
from TsengCode.alexnet import Baseline
from feature.normalize import normalize
from feature.make_dir import makeDir
from feature.utility import  setStdoutToDefault, setStdoutToFile, accelerateByGpuAlgo, get_device
from feature.random_seed import set_seed_cpu
import json
from utility.DatasetHandler import DatasetHandler


stdoutTofile = True
accelerateButUndetermine = True
targetExpName = "0101.brutL0L1"
targetTestSet = "../dataset23/test"
def parse_args(i):
    parser = argparse.ArgumentParser(description='imagenet nas Training')
    parser.add_argument('-m', '--trained_model',
                        default='./weights_retrain_pdarts/NewNasModel' + str(i) + '_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='newnasmodel', help='alexnet or newnasmodel')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

    parser.add_argument('--genotype_file', type=str, default='genotype_' + str(i) + '.npy',
                        help='put decode file')

    args = parser.parse_args()
    return args
def saveAccLoss(kth, accRecord):
    print("save record to ", folder["accLossDir"])
    try:
        np.save(os.path.join(folder["accLossDir"], "testAcc_"+str(kth)), accRecord["testAcc"])
    except Exception as e:
        print("Fail to save acc")
        print(e)
def printNetWeight(net):
    for name, para in net.named_parameters():
        print(name, para)
def prepareData():
    print("preparing test data set")
    PATH_test = testDataSetFolder
    test = Path(PATH_test)
    test_transforms = normalize(seed_cpu, img_dim)

    # choose the training datasets
    test_data = datasets.ImageFolder(test, transform=test_transforms)
    print("test_data.class_to_idx", test_data.class_to_idx)
    return test_data

def prepareDataLoader(test_data):
    print("preparing data loader")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return test_loader
def prepareChekcPointModel(num_classes, kth, epoch):
    try:
        #info prepare model
        genotype_filename = os.path.join(folder["decode_folder"], args.genotype_file)
        cell_arch = np.load(genotype_filename)
        net = NewNasModel(cell_arch)
        # print("net.alphas", net.alphas)

        modelLoadPath = os.path.join( folder["savedCheckPoint"], "newnasmodel_{}_{}.pt".format(kth, epoch) )
        net.load_state_dict( torch.load( modelLoadPath )['model_state_dict'] )
        net = net.to(device)
        net.eval()
        print("Loading model from ", modelLoadPath)
        return net
    # todo test.py go wrong, check pytorch document how to test model
    except Exception as e:
        print("Fail to load model from ", modelLoadPath)
        print(e)
        exit()
def prepareAvgModel(num_classes, kth):
    #info preparing trained NAS model
    numOfModel = 3
    if args.network == "newnasmodel":
        try :
            #info prepare architecture
            #info load decode json
            filePath = os.path.join(folder["decode"], "{}th_decode.json".format(kth))
            f = open(filePath)
            archDict = json.load(f)
            
            print("archDict", archDict)
        except:
            print("Fail to load architecture from ", filePath)
            exit()
            
        try:
            #info prepare model
            net = NewNasModel(cellArch=archDict)
            toModelWeight = None
            for i in range(numOfModel):
                print("preparing model ", "NewNasModel{}_Final.pt".format(kth+i))
                modelLoadPath = os.path.join( folder["retrainSavedModel"], "NewNasModel{}_Final.pt".format(kth+i) )
                tmpModelWeight = torch.load( modelLoadPath )
                # for k in tmpModelWeight:
                #     if "conv_5x5.op.0.weight" in k:
                #         print("tmpModelWeight[k]", tmpModelWeight[k])
                #         break
                for k in tmpModelWeight:
                    if toModelWeight==None:
                        toModelWeight = copy.deepcopy(tmpModelWeight)
                        break
                    else:
                        toModelWeight[k] = toModelWeight[k] + tmpModelWeight[k]
            for k in toModelWeight:
                toModelWeight[k] = toModelWeight[k] / numOfModel
            for k in toModelWeight:
                if "conv_5x5.op.0.weight" in k:
                    # print("toModelWeight[k]", toModelWeight[k])
                    break
            # modelLoadPath = os.path.join( folder["retrainSavedModel"], "NewNasModel{}_Final.pt".format(kth) )
            net.load_state_dict( toModelWeight )
            # print("torch.load( modelLoadPath )", torch.load( modelLoadPath ).keys())
            # for k, v in net.named_parameters():
            #     if "conv_5x5.op.0.weight" in k:
            #         print("v", v)
            net = net.to(device)
            net.eval()
            print("Loading model from ", modelLoadPath)
            return net
        # todo test.py go wrong, check pytorch document how to test model
        except Exception as e:
            print("Fail to load model from ", modelLoadPath)
            print(e)
            exit()
def preparedTransferModel(kth):
    #info load decode json
    filePath = os.path.join(folder["decode"], "{}th_decode.json".format(kth))
    f = open(filePath)
    archDict = json.load(f)
        
    #info prepare model
    print("Preparing model...")
    net = NewNasModel(cellArch=archDict)
    net.train()
    net = net.to(device)
    print("net.cellArch:", net.cellArch)
    print("net", net)

    #info prepare pretrain weight
    f = open("./curExperiment.json")
    exp = json.load(f)
    f.close()
    # targetExpName = "1218.brutL0L1"
    for key in exp:
        expName = targetExpName+"."+key.split(".")[2]
    
    modelLoadPath = os.path.join("./log", targetExpName, expName, folder["retrainSavedModel"], "NewNasModel{}_Final.pt".format(kth) )
    print("modelLoadPath", modelLoadPath)
    tmpModelWeight = torch.load( modelLoadPath )
    net.load_state_dict(tmpModelWeight)
    # exit()
    # tmpF(net)
    return net
def prepareModel(num_classes, kth):
    print("preparing model: ", args.network)
    #info preparing alexnet model
    if args.network == "alexnet":
        try:
            pass
            # net = Baseline(num_classes)
            # modelLoadPath = os.path.join(folder["retrainSavedModel"], "alexnet{}_Final.pth".format(kth))
            # net.load_state_dict(torch.load( modelLoadPath ))
            # net = net.to(device)
            # net.eval()
            # print("Loading model from ", modelLoadPath)
            
            # return net
        except:
            print("Fail to load model from ", modelLoadPath)
            exit()
            
    #info preparing trained NAS model
    if args.network == "newnasmodel":
        try :
            #info prepare architecture
            #info load decode json
            filePath = os.path.join(folder["decode"], "{}th_decode.json".format(kth))
            f = open(filePath)
            archDict = json.load(f)
            
            print("archDict", archDict)
        except:
            print("Fail to load architecture from ", filePath)
            exit()
            
        try:
            #info prepare model
            net = NewNasModel(cellArch=archDict)
            print("net ", net)
            modelLoadPath = os.path.join( folder["retrainSavedModel"], "NewNasModel{}_Final.pt".format(kth) )
            modelWeight = torch.load( modelLoadPath )
            # print("torch.load( modelLoadPath )", torch.load( modelLoadPath ).keys())
            net.load_state_dict( modelWeight )
            net = net.to(device)
            net.eval()
            print("Loading model from ", modelLoadPath)
            return net
        # todo test.py go wrong, check pytorch document how to test model
        except Exception as e:
            print("Fail to load model from ", modelLoadPath)
            print(e)
            exit()
def testCheckPointModel(test_loader, net, kth, epoch):
    print("start testing kth{} epoch{} ".format( kth, epoch))
    confusion_matrix_torch = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(tqdm(test_loader, 0, unit ="{}th_{}epoch".format(kth, epoch))):
            images, labels = data
            labels = labels.to(device)
            images = images.to(device)
            outputs = net(images)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum()
            for t, p in zip(labels.view(-1), predict.view(-1)):
                confusion_matrix_torch[t.long(), p.long()] += 1

        acc = correct / total
    print('Accuracy of the network on the 1500 test images at epoch {}:{}'.format(epoch, acc))
    return acc

def test(test_loader, net):
    print("start testing")
    set_seed_cpu(20)
    confusion_matrix_torch = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(tqdm(test_loader, 0)):
            images, labels = data

            labels = Variable(labels.cuda())
            images = Variable(images.cuda())
            outputs = net(images)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum()
            for t, p in zip(labels.view(-1), predict.view(-1)):
                confusion_matrix_torch[t.long(), p.long()] += 1

        acc = correct / total
    print('Accuracy of the network on the 1500 test images: %f %%' % (100 * acc))
    return acc.cpu()    

#! the next step print model's weight and bias
class TestController:
    def __init__(self, cfg, device, seed=20, testDataSetFolder=testDataSetFolder):
        self.cfg = cfg
        self.testDataSetFolder = testDataSetFolder
        self.testSetHandler = self.prepareData(seed, testDataSetFolder)
        # self.oriTestSetHandler = self.prepareData(seed, targetTestSet)
        # self.curToOriIndex = self.makeTrainformIndex()
        self.IndexToClass = self.testSetHandler.getIndexToClass()
        # print("self.curToOriIndex", self.curToOriIndex)
        # print("tatal number of test images: ", len(self.testSetHandler.getTestDataset()))
        print("testDataSetFolder", testDataSetFolder)
        self.testDataLoader = self.prepareDataLoader(self.testSetHandler.getTestDataset())
        # self.oriTestDataLoader = self.prepareDataLoader(self.oriTestSetHandler.getTestDataset())
        self.num_classes = cfg["numOfClasses"]
        self.device = device
        self.statics = {
            "dataset1":{
                "correct":0,
                "total":0,
                "accList":[]
                
            },
            "dataset2":{
                "correct":0,
                "total":0,
                "accList":[]
            },
            "dataset3":{
                "correct":0,
                "total":0,
                "accList":[]
            },
        }
    def printAllModule(self, net):
        print("printAllModule()")
        for k, v in net.named_parameters():
            if v.requires_grad:
                print (k, v.data.sum())
    def makeTrainformIndex(self):
        # info transform testDataset index to oriTestDataset index
        trans = {}
        testDic = self.testSetHandler.getClassToIndex()
        oriTestDic = self.oriTestSetHandler.getClassToIndex()
        for key in testDic:
            value = None
            for key2 in oriTestDic:
                if key2==key:
                    value = oriTestDic[key2]
            trans[testDic[key]] = value
        return trans
    def transformIndex(self, labels):
        transPredict = labels.detach().clone()
        for i in range(len(labels)):
            transPredict[i] = self.curToOriIndex[labels[i].item()]
        return transPredict
    def test(self, net, showOutput=False):
        # print("self.testSet.getClassToIndex()", self.testSetHandler.getClassToIndex())
        # print("self.oriTestSet.getClassToIndex()", self.oriTestSetHandler.getClassToIndex())
        self.refreshStatics()
        net.eval()
        # print(net)
        with torch.no_grad():
            correct = 0
            total = 0
            for i, data in enumerate(self.testDataLoader):
                images, labels = data
                labels = labels.to(self.device)
                images = images.to(self.device)
                outputs = net(images)
                _, predict = torch.max(outputs.data, 1)
                total += labels.size(0)
                # total = total + 1
                # print("predict", predict.shape, predict)
                # print("labels", labels.shape, labels)
                # info transform testset index to targetExp index
                # labels = self.transformIndex(labels)
                # print("labels", labels.shape, labels)
                checkedSheet = predict == labels
                correct += (checkedSheet).sum().item()
                
                for i in range(len(labels)):
                    className = self.IndexToClass[labels[i].item()]
                    if className in dataset1Name:
                        self.statics["dataset1"]["total"] = self.statics["dataset1"]["total"] + 1
                        if checkedSheet[i]==True:
                            self.statics["dataset1"]["correct"] = self.statics["dataset1"]["correct"] + 1
                        
                    elif(className in dataset2Name):
                        self.statics["dataset2"]["total"] = self.statics["dataset2"]["total"] + 1
                        if checkedSheet[i]==True:
                            self.statics["dataset2"]["correct"] = self.statics["dataset2"]["correct"] + 1
                        
                    elif(className in dataset3Name):
                        self.statics["dataset3"]["total"] = self.statics["dataset3"]["total"] + 1
                        if checkedSheet[i]==True:
                            self.statics["dataset3"]["correct"] = self.statics["dataset3"]["correct"] + 1
            for dataset in ["dataset1", "dataset2", "dataset3"]:
                try:
                    self.statics[dataset]["accList"].append(self.statics[dataset]["correct"]/self.statics[dataset]["total"])
                except:
                    self.statics[dataset]["accList"].append(0)
            print(self.statics)
                # print("=================================")
                # for t, p in zip(labels.view(-1), predict.view(-1)):
                #     confusion_matrix_torch[t.long(), p.long()] += 1
                # print("outputs ", outputs)
                # print("predict", predict)
                # print("labels", labels)
                # print("total {}, correct {}".format(total, correct))
            acc = correct / total

        net.train()
        return acc * 100
    def refreshStatics(self):
        for datasetName in self.statics:
            self.statics[datasetName]["correct"] = 0
            self.statics[datasetName]["total"] = 0
    def saveDatasetAcc(self, kth):
        np.save(os.path.join(folder["accLossDir"], "{}_test_acc_{}_{}".format("retrain", "dataset1", str(kth))), self.statics["dataset1"]["accList"])
        np.save(os.path.join(folder["accLossDir"], "{}_test_acc_{}_{}".format("retrain", "dataset2", str(kth))), self.statics["dataset2"]["accList"])
        np.save(os.path.join(folder["accLossDir"], "{}_test_acc_{}_{}".format("retrain", "dataset3", str(kth))), self.statics["dataset3"]["accList"])
    def prepareData(self, seed, testDataSetFolder):
        datasetHandler = DatasetHandler(testDataSetFolder, cfg, seed)
        return datasetHandler
    def prepareDataLoader(self, test_data):
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.cfg["batch_size"], num_workers=0, shuffle=False)
        return test_loader

def saveAcc(saveNp):
    np.save(os.path.join(folder["accLossDir"], "{}_test_acc_{}".format("retrain", str(kth))), saveNp)
if __name__ == '__main__':
    # print("main fucntion")
    torch.set_printoptions(precision=6, sci_mode=False, threshold=1000)
    device = get_device()
    valList = []
    for kth in range(cfg["numOfKth"]):
        #info handle stdout to a file
        if stdoutTofile:
            trainLogDir = "./log"
            makeDir(trainLogDir)
            f = setStdoutToFile(trainLogDir+"/test_{}.txt".format(str(kth)))
        
        args = parse_args(str(kth))
        # makeDir(args.save_folder, args.log_dir)

        accelerateByGpuAlgo(cfg["cuddbenchMark"])
        num_classes = cfg["numOfClasses"]
        
        img_dim = cfg['image_size']
        num_gpu = cfg['ngpu']
        batch_size = cfg['batch_size']
        max_epoch = cfg['epoch']
        gpu_train = cfg['gpu_train']

        num_workers = args.num_workers
        momentum = args.momentum
        weight_decay = args.weight_decay
        initial_lr = args.lr
        gamma = args.gamma
        
        seed_cpu = seed[str(kth)]
        set_seed_cpu(seed_cpu)
        # testSet = prepareData()
        # testDataLoader = prepareDataLoader(testSet)
        


        
        testC = TestController(cfg, "cuda")

        #info test checkpoint model
        # accRecord = {"testAcc":np.array([])}
        
        # for epoch in range(cfg["epoch"]):
        #     net = prepareChekcPointModel(num_classes, kth, epoch)
        #     # accAtkthAtEpoch = testCheckPointModel( testDataLoader, net, kth, epoch)
        #     accAtkthAtEpoch = testC.test(net)
        #     accRecord["testAcc"] = np.append(accRecord["testAcc"], accAtkthAtEpoch.cpu())
        # saveAccLoss(kth, accRecord)
        
        #info test final model
        net = prepareModel(num_classes, kth)
        # net = preparedTransferModel(kth)
        
        last_epoch_val_acc = testC.test(net)
        # saveAcc([last_epoch_val_acc])

        # print("normal model", last_epoch_val_acc)
        # net = prepareAvgModel(num_classes, kth)
        # last_epoch_val_acc = testC.test(net)
        # print("Avg model", last_epoch_val_acc)
        # testC.printAllModule(net)
        # exit()
        valList.append(last_epoch_val_acc)
        print('test validate accuracy:')
        print(valList)
        if stdoutTofile:
            setStdoutToDefault(f)
        # exit()
