import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os
from data.config import cfg_newnasmodel as cfg
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
def plot_loss_curve(lossRecord, title='default', saveFolder="./"):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    figure(figsize=(6, 4))
    plt.plot(lossRecord['train'], c='tab:red', label='train')
    plt.plot(lossRecord['val'], c='tab:cyan', label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss of {}'.format(title))
    plt.legend()
    
    plt.savefig(os.path.join(saveFolder, title))

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
        
        accRecord = {"train": np.load(trainNasTrainAccFile),
            "val": np.load(trainNasnValAccFile),
            "test": np.load(testAccFile)
            }
        plot_acc_curve(accRecord, "acc_"+str(kth), "./plot")
