import numpy as np
import csv
import os
import matplotlib.pyplot as plt
class AccCollector():
    def __init__(self):
        self.baseDir = "1029_2brutL3L4"
        self.saveFolder="./plot"
        # self.baseDir = "./log/1024_brutL3L4/1024_brutL3L4.0_0/accLoss" 
        # base = os.walk(self.baseDir)

        # print(len(a))
        # for i in a:
        #     print(i)
        # for i in base:
        #     print(i)
    def boxPlot(self, dataset="val"):
        # todo sort by layername
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
        axs.boxplot(a, labels=labels,  showmeans=True)
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
        for i in range(5):
            for j in range(5):
                expAcc = "{}.{}_{}".format(self.baseDir, i, j )
                tmp = [expAcc]
                for k in range(10):
                    # base = os.walk(baseDir)
                    acc = round(np.load("./log/{}/{}.{}_{}/accLoss/retrain_test_acc_{}.npy".format(self.baseDir, self.baseDir, str(i), str(j), str(k)) )[-1], 2)
                    tmp.append(acc)
                self.a.append(tmp)
                tmp = [expAcc]
                for k in range(10):
                    # base = os.walk(baseDir)
                    acc = round(np.load("./log/{}/{}.{}_{}/accLoss/retrain_val_acc_{}.npy".format(self.baseDir, self.baseDir, str(i), str(j), str(k)) )[-1], 2)
                    tmp.append(acc)
                self.a.append(tmp)
        with open('./acc.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in self.a:
                writer.writerow(row)
    
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
    accC = AccCollector()
    accC.boxPlot("val")
    accC.boxPlot("test")
    accC.saveCsv("val")
    # accC.saveCsv("test")
    
    