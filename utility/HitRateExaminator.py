from typing import Union, List
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
from utility.HistDrawer import HistDrawer
from data.config import cfg_nasmodel as cfg
class HitRateExaminator():
    def __init__(self, baseDir = "1027_brutL3L4", fileNameTag="") -> None:
        self.fileNameTag = fileNameTag
        self.baseDir = baseDir
        self.saveFolder="./tmp"
        self.title = ""
        self.ymax = 85
        self.ymin = 30
    def getHLByExp(self, expPath) -> List[float]:
        # return hit rate and loss of the expName
        total = 0
        valHit = 0
        loss = 0
        lastHit = 0
        toHistList = []
        for k in range(cfg["numOfKth"]):
            testAcc = np.load( path.join(expPath, "accLoss", "retrain_test_acc_{}.npy".format(str(k))) )
            testIndex = np.argmax(testAcc)
            valAcc = np.load( path.join(expPath, "accLoss", "retrain_val_acc_{}.npy".format(str(k))) )
            valIndex = np.argmax(valAcc)
            print("testAcc",testAcc)
            print("valAcc", valAcc)
            toHistList.append(testIndex)
            total = total + 1
            if testIndex==(len(testAcc)-1):
                lastHit = lastHit + 1
            if testIndex==valIndex:
                valHit = valHit + 1
            else:
                loss = loss + (abs(testAcc[-1] - testAcc[valIndex]))

        return [valHit/total, lastHit/total, loss/total]
    #todo line chart for each sub experiment
    #todo histogram for parent experiment
    #todo data: 2 layer and 3 layer brute force

    def plotHitRateLineChart(self, expName, color="red", dataset="val", title="") -> None:
        a = []
        xlabel = []
        hitRateList = []
        lossList = []
        lastHitList = []
        numOfOp = 3
        fig, ax = plt.subplots(figsize=(10, 8), sharex=True, constrained_layout=True)
        for i in range(numOfOp):
            for j in range(numOfOp):
                for k in range(numOfOp):
                    xlabel.append( "{}_{}_{}".format(str(i), str(j), str(k)) )
                    subExpName = "{}.{}_{}_{}".format(expName, str(i), str(j), str(k) )
                    subExpPath = path.join("./log", expName, subExpName)
                    hL = self.getHLByExp(subExpPath)
                    hitRateList.append(hL[0]*100)
                    lastHitList.append(hL[1]*100)
                    lossList.append(hL[2])
        print("loss accuracy %", lossList)
        ax.plot(xlabel, hitRateList, c="black", linestyle="solid", marker='o')
        ax.plot(xlabel, lastHitList, c="black", linestyle="dashed", marker='x')
        # ax.plot(xlabel, lossList, c="black", linestyle="dotted")
        ax.set_xlabel('architecture')
        ax.set_ylabel('hit rate %')
        ax.yaxis.grid()
        ax.xaxis.grid()
        colorsLabel = ["max val epoch hit rate", "last epoch hit rate"]
        ax.legend(colorsLabel , loc='lower right')
        savePath =path.join("./log", self.baseDir, "hitrate"+self.fileNameTag)
        plt.savefig(savePath)
        print("save png to ", savePath)
        plt.close()
    def hitRateHis(self):
        fig, ax = plt.subplots(2, 1, figsize=(5, 2.7), layout='constrained')
        # ax[1].hist(np.reshape(tensor.data.cpu().numpy(), (-1)), bins=100)
    def plotAccLineChart(self, expName):
        dataDic = {
            "train": None,
            "val": None,
            "test": None
        }
        xlabel = []
        numOfOp = 1
        fig, ax = plt.subplots(figsize=(10, 8), sharex=True, constrained_layout=True)
        for i in range(numOfOp):
            for j in range(numOfOp):
                for k in range(numOfOp):
                    subExpName = "{}.{}_{}_{}".format(expName, str(i), str(j), str(k))
                    subExpPath = path.join("./log", expName, subExpName)
                    accFolder = path.join(subExpPath, "accLoss")
                    self.plot_combined_acc(folder=accFolder, saveFolder="./log/{}".format(self.baseDir), trainType="retrain")
                
    def __plot_acc_curves(self, accRecord, ax, title='default', saveFolder="./"):
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
        ax.set_ylabel('accuracy %')
        ax.set_title(format(title))
        ax.legend()
    def plot_combined_acc(self, folder = "./accLoss", title='combine', saveFolder="./plot", trainType="Nas"):
        numOfAx = 3
        indexOfAx = 0
        numOfFig = cfg["numOfKth"] // numOfAx
        indexOfFig = 0
        
        for i in range(numOfFig):
            fig, axs = plt.subplots(numOfAx, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
            for kth in range(numOfAx):
                trainNasTrainAccFile = path.join(folder, "{}_train_acc_{}.npy".format(trainType, str(indexOfAx)) )
                trainNasnValAccFile = path.join( folder,"{}_val_acc_{}.npy".format(trainType, str(indexOfAx)) )
                testAccFile = path.join( folder,"{}_test_acc_{}.npy".format(trainType, str(indexOfAx)) )
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
                self.__plot_acc_curves(accRecord, axs[kth], str(indexOfAx)+" kth", "./plot")
                indexOfAx = indexOfAx + 1
            indexOfFig = indexOfFig + 1
            fileName = trainType+"_"+  str(indexOfFig)
            print("save png to ", path.join(self.baseDir, fileName+self.fileNameTag))
            plt.savefig(path.join(saveFolder, fileName+self.fileNameTag))
if __name__=="__main__":

    np.set_printoptions(precision=2)
    ANASList = []
    # for i in range(1, 25):
    #     ANASList.append("0327_"+str(i))
    ExpName = "0202_2.brutL0L1L2"
    hitRateE = HitRateExaminator(ExpName, fileNameTag="_0202_2_brutL0L1L2")
    # for i in range(5):
    #     for j in range(5):
    #         expName = "1129_4.brutL0L1.{}_{}".format(i, j)
    hitRateE.plotHitRateLineChart(ExpName)
    hitRateE.plotAccLineChart(ExpName)
    
