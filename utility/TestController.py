
import torch
from data.config import  cfg_newnasmodel as cfg, testDataSetFolder
from utility.DatasetHandler import DatasetHandler

class TestController:
    def __init__(self, cfg, device, seed=20, testDataSetFolder=testDataSetFolder):
        self.cfg = cfg
        self.testDataSetFolder = testDataSetFolder
        self.testSetHandler = self.prepareData(seed, testDataSetFolder)
        # self.oriTestSetHandler = self.prepareData(seed, targetTestSet)
        # self.curToOriIndex = self.makeTrainformIndex()
        # print("self.curToOriIndex", self.curToOriIndex)
        # print("tatal number of test images: ", len(self.testSetHandler.getTestDataset()))
        print("testDataSetFolder", testDataSetFolder)
        self.testDataLoader = self.prepareDataLoader(self.testSetHandler.getTestDataset())
        # self.oriTestDataLoader = self.prepareDataLoader(self.oriTestSetHandler.getTestDataset())
        self.num_classes = cfg["numOfClasses"]
        self.device = device
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
        
        confusion_matrix_torch = torch.zeros(self.num_classes, self.num_classes)
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
                correct += (predict == labels).sum().item()
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
    def prepareData(self, seed, testDataSetFolder):
        datasetHandler = DatasetHandler(testDataSetFolder, cfg, seed)
        return datasetHandler
    def prepareDataLoader(self, test_data):
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.cfg["batch_size"], num_workers=0, shuffle=False)
        return test_loader