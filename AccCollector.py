import numpy as np
import csv
import os
import matplotlib.pyplot as plt
class AccCollector():
    def __init__(self, baseDir):
        self.baseDir = baseDir
        self.saveFolder="./tmp"
    def addExp(self, baseDir, color="red", dataset="val", title=""):
        a = []
        labels = []
        for i in range(5):
            for j in range(5):
                expAcc = "{}.{}_{}".format(baseDir, i, j )
                labels.append(expAcc)
                data = []
                for k in range(10):
                    # base = os.walk(baseDir)
                    acc = round(np.load("./log/{}/{}.{}_{}/accLoss/retrain_{}_acc_{}.npy".format(baseDir, baseDir, str(i), str(j), dataset, str(k)) )[-1], 2)
                    data.append(acc)
                    # self.a.append([expAcc, k , acc])
                a.append(data)
        if hasattr(self, "axs"):
            pass
        else:
            self.fig, self.axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        # ax = fig.add_axes([0, 0, 1, 1])
        self.axs.boxplot(a, labels=labels,  showmeans=False,  boxprops=dict(color=color), meanprops=dict(color=color))
        self.axs.yaxis.grid()
        self.axs.xaxis.grid()
        self.axs.set_ylim([50, 85])
        self.axs.set_title(title)
        plt.xticks(rotation=90)

    
    def savePlt(self, dataset, ):
        saveName = os.path.join("./log", self.baseDir, "box_"+dataset)
        print("save to ", saveName)
        plt.savefig(saveName)
    def boxPlot(self, dataset="val"):
        a = []
        labels = []
        for i in range(5):
            for j in range(5):
                expAcc = "{}.{}_{}".format(self.baseDir, i, j )
                labels.append(expAcc)
                data = []
                for k in range(10):
                    # base = os.walk(baseDir)
                    acc = round(np.load("./log/{}/{}.{}_{}/accLoss/retrain_{}_acc_{}.npy".format(self.baseDir, self.baseDir, str(i), str(j), dataset, str(k)) )[-1], 2)
                    data.append(acc)
                    # self.a.append([expAcc, k , acc])
                a.append(data)
        fig, axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        # ax = fig.add_axes([0, 0, 1, 1])
        c = "red"
        axs.boxplot(a, labels=labels,  showmeans=True,  boxprops=dict(color=c),)
        axs.yaxis.grid()
        axs.xaxis.grid()
        axs.set_title("box_"+dataset)
        plt.xticks(rotation=90)
        
        saveName = os.path.join("./log", self.baseDir, "box_"+dataset)
        print("save to ", saveName)
        plt.savefig(saveName)
        # plt.savefig("plot.png")
        
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
        #         self.a.append(tmp)
        #         tmp = [expAcc]
        #         for k in range(10):
        #             # base = os.walk(baseDir)
        #             acc = round(np.load("./log/{}/{}.{}_{}/accLoss/retrain_val_acc_{}.npy".format(self.baseDir, self.baseDir, str(i), str(j), str(k)) )[-1], 2)
        #             tmp.append(acc)
        #         self.a.append(tmp)
        # with open('./acc.csv', 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile, delimiter=',')
        #     for row in self.a:
        #         writer.writerow(row)
    def calDiffValTest(self, dataset):
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
                        # print(testAcc[testIndex] - valAcc[valIndex])
                    
        print(hit, total, "hit rate", hit/total)
        print("average loss ", loss/total)
        
    def boxPlotCsv(self):
        self.a = []
        for i in range(5):
            for j in range(5):
                expAcc = "{}.{}_{}".format(self.baseDir, i, j )
                for k in range(10):
                    # base = os.walk(baseDir)
                    acc = round(np.load("./log/{}/{}.{}_{}/accLoss/retrain_test_acc_{}.npy".format(self.baseDir, self.baseDir, str(i), str(j), str(k)) )[-1], 2)

                    self.a.append([expAcc, k , acc])
        with open('./boxPlot.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in self.a:
                writer.writerow(row)
if __name__=="__main__":
    np.set_printoptions(precision=2)
    accC = AccCollector("1028_2brutL3L4")
    accC.addExp("1028_2brutL3L4", color="red", dataset="test", title="1028_2brutL3L4")
    accC.savePlt(dataset="test")
    # accC.addExp("1028_2brutL3L4", color="blue", dataset="test", title="1028_2brutL3L4")
    
    # accC.boxPlot("val")
    # accC.boxPlot("test")
    # accC.saveCsv("val")
    # accC.calDiffValTest("test")
    # accC.saveCsv("test")
    
    