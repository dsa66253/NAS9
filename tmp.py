import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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
        for i in range(5):
            for j in range(5):
                expAcc = "{}.{}_{}".format(baseDir, i, j )
                labels.append(expAcc)
                data = []
                for k in range(10):
                    # base = os.walk(baseDir)
                    #* get last epoch acc
                    loadPath = "./log/{}/{}.{}_{}/accLoss/retrain_{}_acc_{}.npy".format(baseDir, baseDir, str(i), str(j), dataset, str(k)) 
                    # print(loadPath)
                    # acc = round(np.load(loadPath)[-1], 2)
                    #* get test acc by correspoding max val acc
                    acc = self.__getAccByMaxVal(i, j, k, baseDir)
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

def plot_acc_curve(accRecord, title='default', saveFolder="./"):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    fig, ax = plt.subplot()
    ax.plot(accRecord['train'], c='tab:red', label='train')
    ax.plot(accRecord['val'], c='tab:cyan', label='val')
    try:
        ax.plot(accRecord['test'], c='tab:brown', label='test')
    except Exception as e:
        print("null accRecord['test']", e)
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    ax.set_title(format(title))
    ax.legend()
    # plt.show()
    plt.savefig(os.path.join(saveFolder, title)) 
def plot_acc_curves(accRecord, ax, title='default', saveFolder="./"):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    totalEpoch = len(accRecord["train"])
    ax.plot(accRecord['train'], c='tab:red', label='train')
    ax.plot(accRecord['val'], c='tab:cyan', label='val')
    
    try:
        ax.plot(accRecord['test'], c='tab:brown', label='test')
    except Exception as e:
        print("null accRecord['test']", e)
    ax.yaxis.grid()
    ax.xaxis.grid()
    ax.set_yticks(range(0, 110, 10))
    ax.set_xticks(range(0, totalEpoch, 10))
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    ax.set_title(format(title))
    ax.legend()
def plot_combined_acc(folder = "./accLoss", title='combine', saveFolder="./plot", trainType="Nas"):
    numOfAx = 3
    indexOfAx = 0
    numOfFig = cfg["numOfKth"] // numOfAx
    indexOfFig = 0
    
    for i in range(numOfFig):
        fig, axs = plt.subplots(numOfAx, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        for kth in range(numOfAx):
            trainNasTrainAccFile = os.path.join(folder, "{}_train_acc_{}.npy".format(trainType, str(indexOfAx)) )
            trainNasnValAccFile = os.path.join( folder,"{}_val_acc_{}.npy".format(trainType, str(indexOfAx)) )
            testAccFile = os.path.join( folder,"{}_test_acc_{}.npy".format(trainType, str(indexOfAx)) )
            # testAccFile = os.path.join(folder, "trainNasTestAcc_{}.npy".format(trainType, str(kth)) )
            try:
                accRecord = {
                    "train": np.load(trainNasTrainAccFile),
                    "val": np.load(trainNasnValAccFile),
                    "test": np.load(testAccFile)
                }
            except:
                accRecord = {
                    "train": np.load(trainNasTrainAccFile),
                    "val": np.load(trainNasnValAccFile),
                    # "test": np.load(testAccFile)
                }
            plot_acc_curves(accRecord, axs[kth], "acc_"+str(indexOfAx), "./plot")
            indexOfAx = indexOfAx + 1
        indexOfFig = indexOfFig + 1
        fileName = trainType+"_"+  str(indexOfFig)
        print("save png to ", os.path.join(saveFolder, fileName))
        plt.savefig(os.path.join(saveFolder, fileName))
        

        
class AccDrawer():
    def __init__(self, expName="", baseDir="./") -> None:
        self.expName = expName
        self.baseDir = baseDir
    def createMAvg(self, input):
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
    def plot_combined_acc(self, sourceFolder = "accLoss", title='combine', saveFolder="plot", trainType="Nas"):
        numOfAx = 3
        indexOfAx = 0
        numOfFig = cfg["numOfKth"] // numOfAx
        indexOfFig = 0
        
        for i in range(numOfFig):
            fig, axs = plt.subplots(numOfAx, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
            for kth in range(numOfAx):
                trainNasTrainAccFile = os.path.join(sourceFolder, "{}_train_acc_{}.npy".format(trainType, str(indexOfAx)) )
                trainNasnValAccFile = os.path.join( sourceFolder,"{}_val_acc_{}.npy".format(trainType, str(indexOfAx)) )
                testAccFile = os.path.join( sourceFolder,"{}_test_acc_{}.npy".format(trainType, str(indexOfAx)) )
                # testAccFile = os.path.join(folder, "trainNasTestAcc_{}.npy".format(trainType, str(kth)) )
                try:
                    accRecord = {
                        "train": np.load(trainNasTrainAccFile),
                        "val": np.load(trainNasnValAccFile),
                        "test": np.load(testAccFile)
                    }
                except:
                    accRecord = {
                        "train": np.load(trainNasTrainAccFile),
                        "val": np.load(trainNasnValAccFile),
                        # "test": np.load(testAccFile)
                    }
                self.plot_acc_curves(accRecord, axs[kth], "acc_"+str(indexOfAx))
                indexOfAx = indexOfAx + 1
            indexOfFig = indexOfFig + 1
            fileName = trainType+"_"+  str(indexOfFig)
            print("save png to ", os.path.join(saveFolder, fileName))
            plt.savefig(os.path.join(saveFolder, fileName))
        plt.cla()
        plt.close("all")
    def plot_acc_curves(self, accRecord, ax, title='default'):
        totalEpoch = len(accRecord["train"])
        ax.plot(accRecord['train'], c='tab:red', label='train', marker = '.')
        ax.plot(accRecord['val'], c='tab:cyan', label='val', marker = '.')
        ma = self.createMAvg(accRecord['val'])
        ax.plot(ma, c='y', label='valMA', marker = '.')
        
        try:
            ax.plot(accRecord['test'], c='tab:brown', label='test', marker = '.')
        except Exception as e:
            print("null accRecord['test']", e)
        ax.yaxis.grid()
        ax.xaxis.grid()
        ax.set_yticks(range(0, 110, 10))
        ax.set_xticks(range(0, totalEpoch, 10))
        ax.set_xlabel('epoch')
        ax.set_ylabel('acc')
        ax.set_title(format(title))
        ax.legend()
class testParent():
    def __init__(self) -> None:
        self.y=8
    
class test(testParent):
    def __init__(self):
        testParent.__init__(self)
        a = 0
    def foo(self):
        print(self.y)
class Mother(object):
    def __init__(self):
        self._haircolor = "Brown"

class Child(Mother):
    def __init__(self): 
        Mother.__init__(self)   
    def print_haircolor(self):
        print (self._haircolor)

if __name__=="__main__":
    # plot_combined_acc(trainType="Nas")
    # c = Child()
    # c.print_haircolor()
    # t = test()
    # t.foo()
    exit()
    expList = ["1202_3.brutL1L2", "1204.brutL2L3", "1206.brutL3L4"]
    for expName in expList:
        for i in range(5):
            for j in range(5):
                # expName="1201.brutL0L1"
                baseDir=os.path.join("./log", expName, "{}.{}_{}".format(expName, str(i), str(j)))
                accD = AccDrawer(expName=expName, baseDir=baseDir)
                sourceFolder = os.path.join(baseDir, "accLoss")
                saveFolder = os.path.join(baseDir, "plot")
                accD.plot_combined_acc(sourceFolder=sourceFolder, saveFolder=saveFolder, trainType="retrain")
        
    # net = "alexnet"
    # folder = "./accLoss" 
    # title='combine_'+net
    # saveFolder="./plot"
    # fig, axs = plt.subplots(1, figsize=(10, 8), sharex=True, constrained_layout=True)
    # for kth in range(1):
    #     trainNasTrainAccFile = os.path.join(folder, "trainNasTrainAcc_{}.npy".format(str(kth)) )
    #     trainNasnValAccFile = os.path.join( folder,"trainNasValAcc_{}.npy".format(str(kth)) )
    #     testAccFile = os.path.join(folder, "trainNasTestAcc_{}.npy".format(str(kth)) )
        
        
    #     accRecord = {
    #         "train": np.load(trainNasTrainAccFile)*100,
    #         "val": np.load(trainNasnValAccFile)*100,
    #         "test": np.load(testAccFile)*100
    #         }
    #     plot_acc_curves(accRecord, axs, "acc_"+str(kth), "./plot")
    # # plt.show()
    # print("save png to ", os.path.join(saveFolder, title))
    # plt.savefig(os.path.join(saveFolder, title))
    exit()
    folder = "./accLoss"
    for kth in range(3):
        trainNasTrainAccFile = os.path.join(folder, "trainNasTrainAcc_{}.npy".format(str(kth)) )
        trainNasnValAccFile = os.path.join( folder,"trainNasValAcc_{}.npy".format(str(kth)) )
        testAccFile = os.path.join(folder, "testAcc_{}.npy".format(str(kth)) )
        
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
    accC = AccCollector("1118_2.brutL0L1", fileNameTag="_1208_2")
    testOrVal = "test"
    accC.addExp("1118_2.brutL0L1", color="red", dataset=testOrVal, title="1118_2.brutL0L1")
    accC.addExp("1202_3.brutL1L2", color="green", dataset=testOrVal, title="1202_3.brutL1L2")
    accC.addExp("1204.brutL2L3", color="blue", dataset=testOrVal, title="1204.brutL2L3")
    # accC.addExp("1111_brutL0L1", color="black", dataset=testOrVal, title="1111_brutL0L1")
    accC.savePlt(dataset=testOrVal)
    # getLoss()
    # accC.addExp("1027_brutL3L4", color="red", dataset="test", title="1027_brutL3L4")
    # accC.addExp("1029_2brutL3L4", color="green", dataset="test", title="1029_2brutL3L4")
    # accC.addExp("1103_brutL3L4", color="blue", dataset="test", title="1103_brutL3L4")
    # accC.savePlt(dataset="test")
    # accC.boxPlot("val")
    # accC.boxPlot("test")
    # accC.saveCsv("val")
    
    # accC.saveCsv("test")
    
    