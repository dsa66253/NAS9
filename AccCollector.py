import numpy as np
import torch
import csv
import os
import matplotlib.pyplot as plt
from utility.HistDrawer import HistDrawer
class AccCollector():
    def __init__(self, baseDir = "1027_brutL3L4", fileNameTag=""):
        self.fileNameTag = fileNameTag
        self.baseDir = baseDir
        self.saveFolder="./tmp"
        self.title = ""
        self.ymax = 85
        self.ymin = 65
    def addExp(self, baseDir, color="red", dataset="val", title=""):
        self.title = self.title +"."+ title + color
        a = []
        labels = []
        numOfOp = 5
        for i in range(numOfOp):
            for j in range(numOfOp):
                # for l in range(numOfOp):
                expAcc = "{}.{}_{}".format(baseDir, i, j)
                labels.append(expAcc)
                data = []
                for k in range(10):
                    # base = os.walk(baseDir)
                    #* get last epoch acc
                    loadPath = "./log/{}/{}.{}_{}/accLoss/retrain_{}_acc_{}.npy".format(baseDir, baseDir, str(i), str(j), dataset, str(k)) 
                    # print(loadPath)
                    # print(np.load(loadPath))
                    acc = round(np.load(loadPath)[-1], 2)
                    #* get test acc by correspoding max val acc
                    # acc = self.__getAccByMaxVal(i, j, k, baseDir)
                    data.append(acc)
                    # self.a.append([expAcc, k , acc])
                a.append(data)
        if hasattr(self, "axs"):
            pass
        else:
            self.fig, self.axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        # ax = fig.add_axes([0, 0, 1, 1])
        # print(baseDir, "a", a)
        self.axs.boxplot(a, labels=labels,  showmeans=False,  boxprops=dict(color=color), meanprops=dict(color=color))
        self.axs.yaxis.grid()
        self.axs.xaxis.grid()
        self.axs.set_title(self.title)
        # self.axs.set_ylim([self.ymin, self.ymax])
        self.axs.set_yticks(np.arange(self.ymin, self.ymax, 1))
        plt.xticks(rotation=90)
    def getDynamicXLabel(self, numOfTotalLayer, numOfOp):
        label = []
        for i in range(numOfTotalLayer):
            for j in range(numOfTotalLayer):
                label.append("")
    def addExp2(self, baseDirList, color="red", dataset="val", title=""):
        self.title = self.title +"."+ ", ".join(baseDirList)
        a = []
        for baseDir in baseDirList:
            # info load npy
            data = []
            numOfkth=10
            for kth in range(numOfkth):
                loadPath = "./log/{}/accLoss/retrain_{}_acc_{}.npy".format(baseDir, dataset, str(kth)) 
                acc = round(np.load(loadPath)[-1], 2)
                #* get test acc by correspoding max val acc
                # acc = self.__getAccByMaxVal(i, j, k, baseDir)
                data.append(acc)
            if hasattr(self, "axs"):
                pass
            else:
                self.fig, self.axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
            a.append(data)
        print(a, baseDirList)
        self.axs.boxplot(a, labels=baseDirList,  showmeans=False,  boxprops=dict(color=color), meanprops=dict(color=color))
        self.axs.yaxis.grid()
        self.axs.xaxis.grid()
        self.axs.set_title(self.title)
        # self.axs.set_ylim([self.ymin, self.ymax])
        self.axs.set_yticks(np.arange(self.ymin, self.ymax, 1))
        plt.xticks(rotation=90)
    def savePlt(self, dataset):
        saveName = os.path.join("./log", self.baseDir, "box_"+dataset+self.fileNameTag+".png")
        print("save to ", saveName)
        plt.savefig(saveName)
        plt.close()
    # def boxPlot(self, dataset="val"):

    #     a = []
    #     labels = []
    #     for i in range(5):
    #         for j in range(5):
    #             expAcc = "{}.{}_{}".format(self.baseDir, i, j )
    #             labels.append(expAcc)
    #             data = []
    #             for k in range(10):
    #                 # base = os.walk(baseDir)
    #                 acc = round(np.load("./log/{}/{}.{}_{}/accLoss/retrain_{}_acc_{}.npy".format(self.baseDir, self.baseDir, str(i), str(j), dataset, str(k)) )[-1], 2)
    #                 data.append(acc)
    #                 # self.a.append([expAcc, k , acc])
    #             a.append(data)
    #     fig, axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    #     # ax = fig.add_axes([0, 0, 1, 1])
    #     c = "red"
    #     axs.boxplot(a, labels=labels,  showmeans=True,  boxprops=dict(color=c),)
    #     axs.yaxis.grid()
    #     axs.xaxis.grid()
    #     axs.set_title("box_"+dataset)
    #     plt.xticks(rotation=90)
    #     saveName = os.path.join("./log", self.baseDir, "box_"+dataset+self.fileNameTag)
    #     print("save to ", saveName)
    #     plt.savefig(saveName)
    #     plt.close()
        # plt.savefig("plot.png")
    def __getAccByMaxVal(self, i, j, k, baseDir):
        valAcc = np.load( "./log/{}/{}.{}_{}/accLoss/retrain_val_acc_{}.npy".format(baseDir, baseDir, str(i), str(j), str(k)) )
        testAcc = np.load("./log/{}/{}.{}_{}/accLoss/retrain_test_acc_{}.npy".format(baseDir, baseDir, str(i), str(j), str(k)) )
        valIndex = np.argmax(valAcc)
        return round(testAcc[valIndex], 2)
    def saveCsv(self, dataset):
        self.a = []
        total = 0
        hit = 0
        loss = 0
        for i in range(5):
            for j in range(5):
                expAcc = "{}.{}_{}".format(self.baseDir, i, j )
                tmp = [expAcc]

                for k in range(10):
                    # base = os.walk(baseDir)
                    testAcc = np.load("./log/{}/{}.{}_{}/accLoss/retrain_test_acc_{}.npy".format(self.baseDir, self.baseDir, str(i), str(j), str(k)) )
                    # acc = round(np.load("./log/{}/{}.{}_{}/accLoss/retrain_test_acc_{}.npy".format(self.baseDir, self.baseDir, str(i), str(j), str(k)) )[-1], 2)
                    # tmp.append(acc)
                    testIndex = np.argmax(testAcc)
                    valAcc = np.load("./log/{}/{}.{}_{}/accLoss/retrain_val_acc_{}.npy".format(self.baseDir, self.baseDir, str(i), str(j), str(k)) )
                    valIndex = np.argmax(valAcc)
                    
                    # print(rawAcc)π
                    # print(index, rawAcc[index])
                    total = total + 1
                    if testIndex==valIndex:
                        hit = hit + 1
                    else:
                        loss = loss + (abs(testAcc[testIndex] - valAcc[valIndex]))
                        print(testAcc[testIndex] - valAcc[valIndex])
                    
        print(hit, total, hit/total)
        print(loss/(hit/total))
    def calDiffValTest(self, dataset, expName):
        histDrawer = HistDrawer("./tmp")
        toHistList = []
        self.a = []
        total = 0
        hit = 0
        loss = 0
        for i in range(5):
            for j in range(5):
                for l in range(5):
                    expAcc = "{}.{}_{}_{}".format(self.baseDir, i, j, l)
                    tmp = [expAcc]

                    for k in range(10):
                        testAcc = np.load("./log/{}/{}.{}_{}_{}/accLoss/retrain_test_acc_{}.npy".format(self.baseDir, self.baseDir, str(i), str(j), str(l), str(k)) )
                        testIndex = np.argmax(testAcc)
                        valAcc = np.load("./log/{}/{}.{}_{}_{}/accLoss/retrain_val_acc_{}.npy".format(self.baseDir, self.baseDir, str(i), str(j), str(l), str(k)) )
                        valIndex = np.argmax(valAcc)
                        firstNegMaIndex = self.getFirstNegMaIndex(valAcc, startEpoch=20)
                        # print(firstNegMaIndex)
                        # print(valAcc[:firstNegMaIndex])
                        # print(testAcc[:firstNegMaIndex])
                        valIndex = np.argmax(valAcc[:firstNegMaIndex])
                        
                        # if testIndex<20:
                        #     print(expName, i, j, k)
                        #     print("index", testIndex, valIndex, "acc", testAcc[testIndex], testAcc[valIndex])
                        toHistList.append(testIndex)
                        total = total + 1
                        if testIndex==valIndex:
                            hit = hit + 1
                        else:
                            loss = loss + (abs(testAcc[testIndex] - testAcc[valIndex]))
        histDrawer.drawHist(torch.tensor(toHistList, dtype=torch.float32), fileName="maxTestAcc", tag=expName)
        print(hit, total, "hit rate", hit/total)
        print("Avg loss ", loss/total)
    def getFirstNegMaIndex(self, ma, startEpoch=10):
        stopEpoch = 10
        derivativeMa = [0] #have no previous acc at first epoch
        
        #info calculate deravitive by dy=0
        for i in range(1, len(ma)):
            dx = ma[i]-ma[i-1]
            dy = 1
            derivativeMa.append(dx/dy)
        # print(ma)
        # print(derivativeMa)
        # exit()
        #info find index by condition
        firstNegMaIndex=0
        for i in range(len(derivativeMa)):
            if derivativeMa[i]<0 and i>startEpoch:
                firstNegMaIndex=i
                break
        if firstNegMaIndex==0:
            firstNegMaIndex=startEpoch
        # print(derivativeMa[firstNegMaIndex-1:firstNegMaIndex+2])
        # exit()
        return firstNegMaIndex
    # def boxPlotCsv(self):
    #     self.a = []
    #     for i in range(5):
    #         for j in range(5):
    #             expAcc = "{}.{}_{}".format(self.baseDir, i, j )
    #             for k in range(10):
    #                 # base = os.walk(baseDir)
    #                 acc = round(np.load("./log/{}/{}.{}_{}/accLoss/retrain_test_acc_{}.npy".format(self.baseDir, self.baseDir, str(i), str(j), str(k)) )[-1], 2)

    #                 self.a.append([expAcc, k , acc])
    #     with open('./boxPlot.csv', 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',')
    #         for row in self.a:
    #             writer.writerow(row)
    def createMAvg(input):
        howMany = 5
        ma = np.copy(input)
        for i in range(0, len(input)):
            window = []
            for j in range(-2, -2+howMany):
                if i+j>=0 and i+j<len(input):
                    # print(i+j)
                    window.append(input[i+j])
            # print(input)
            ma[i] = np.mean(window)
            # print(i, window, ma[i])
        return ma
def getLoss():
    expList = ["1027_brutL3L4", "1028_2brutL3L4", "1029_2brutL3L4", "1029_brutL3L4", "1103_brutL3L4", "1111_2brutL0L1"]
    expList = ["1122_2.brutL0L1"]
    for exp in expList:
        print(exp)
        accC = AccCollector(exp, fileNameTag="")
        accC.calDiffValTest("test", expName=exp)
if __name__=="__main__":
    np.set_printoptions(precision=2)
    accC = AccCollector("0226", fileNameTag="_0302_2")
    testOrVal = "test"
    baseDirList = ["0226"]
    for i in range(2, 10):
        baseDirList.append("0226_{}".format(str(i)))
    baseDirList.append("0227")
    for i in range(2, 13):
        baseDirList.append("0227_{}".format(str(i)))
    print(baseDirList)
    accC.addExp2(baseDirList=baseDirList, color="red", dataset=testOrVal, title="0302")
    # accC.addExp("0226_3", color="green", dataset=testOrVal, title="0226_3")
    # accC.addExp("0226_6", color="blue", dataset=testOrVal, title="0226_6")
    # accC.addExp("0226", color="black", dataset=testOrVal, title="0226")
    # getLoss()
    # accC.addExp("1027_brutL3L4", color="red", dataset="test", title="1027_brutL3L4")
    # accC.addExp("1029_2brutL3L4", color="green", dataset="test", title="1029_2brutL3L4")
    # accC.addExp("1103_brutL3L4", color="blue", dataset="test", title="1103_brutL3L4")
    accC.savePlt(dataset="test")
    # accC.boxPlot("val")
    # accC.boxPlot("test")
    # accC.saveCsv("val")
    
    # accC.saveCsv("test")
    
    