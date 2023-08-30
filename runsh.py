import subprocess
import json, os, sys, copy
from os.path import isfile, join
from data.config import folder, cfg_newnasmodel as cfg
def makeAllDir():
    for folderName in folder:
        print("making folder ", folder[folderName])
        makeDir(folder[folderName])
def makeDir(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
def setStdoutToFile(filePath):
    f = open(filePath, 'w')
    sys.stdout = f
    return f
def setStdoutToDefault(f):
    f.close()
    sys.stdout = sys.__stdout__
def doExpBasedExperiments():
    conti = True
    while conti:
        filePath = os.path.join("./experiments.json")
        f = open(filePath)
        exp = json.load(f)
        finishCount = 0
        for expName in exp:
            if exp[expName]==0:
                # exp[expName]=1
                # f = setStdoutToFile("./experiments.json")
                # print(json.dumps(exp, indent=4)) #* make ndarray to list
                # setStdoutToDefault(f)
                subprocess.call('./train.sh')
                break
            finishCount = finishCount + 1
            if finishCount==len(exp):
                exit()
        print("finish trina.sh")
def brutNas():
    # this funciion also handle decode job
    initiManualAssign = {
        "layer_0_1": [
            0,
            0,
            0,
            0,
            0
        ],
        "layer_1_2": [
            1,
            0,
            0,
            0,
            0
        ],
        "layer_2_5": [
            0,
            0,
            0,
            0,
            0
        ],
    }
    # brutally train all possible arch of first two layers
    numOfOp = 3
    count = 0
    for i in range(numOfOp):
        # for fisrt layer
        for j in range(numOfOp):
            # for second layeer
            for k in range(numOfOp):
                count = count + 1
                if count <= 2:
                    continue
                manualAssign = copy.deepcopy(initiManualAssign)
                
                manualAssign["layer_0_1"][i] = 1
                manualAssign["layer_1_2"][j] = 1
                manualAssign["layer_2_5"][k] = 1
                f = setStdoutToFile("./curExperiment.json")
                curExpName = "0723.brutL0L1L2.{}_{}_{}".format(i, j, k)
                desDir = join("./log", curExpName)
                print(json.dumps({curExpName:1}, indent=4))
                setStdoutToDefault(f)

<<<<<<< HEAD
                makeDir(desDir)
                makeAllDir()
                #info handle decode job
                for kth in range(cfg["numOfKth"]):
                    filePath = "./decode/{}th_decode.json".format(kth)
                    f = setStdoutToFile(filePath)
                    print(json.dumps(manualAssign, indent=4)) #* make ndarray to list
                    setStdoutToDefault(f)   
                
                subprocess.call('./train.sh')
=======
<<<<<<< HEAD
            makeDir(desDir)
            makeAllDir()
            #info handle decode job
            for kth in range(cfg["numOfKth"]):
                filePath = "./decode/{}th_decode.json".format(kth)
                f = setStdoutToFile(filePath)
                print(json.dumps(manualAssign, indent=4)) #* make ndarray to list
                setStdoutToDefault(f)   
            
            subprocess.call('./train.sh')
>>>>>>> 4f7e3e0ae41a4292f6c540add2b81ef0f1d542b7
            
            # exit()
def brutInit():
    # this funciion also handle decode job
    initiManualAssign = {
        "layer_0_1": [
            0,
            0,
            0,
            0,
            1
        ],
        "layer_1_2": [
            0,
            1,
            0,
            0,
            0
        ],
        "layer_2_3": [
            1,
            0,
            0,
            0,
            0
        ],
        "layer_3_4": [
            1,
            0,
            0,
            0,
            0
        ],
        "layer_4_5": [
            1,
            0,
            0,
            0,
            0
        ],
    }
    # brutally train all possible arch of first two layers
    expNameList = [
    # "0227",
    # "0227_2",
    # "0227_3",
    # "0227_4",
    # "0227_5",
    # "0227_6",
    # "0227_7",
    # "0227_6",
    # "0227_7",
    # "0227_8",
    # "0227_9",
    # "0227_10",
    # "0227_11",
    # "0227_12",
    "0620_3"
    ]
    count = 0
    for expName in expNameList:
        manualAssign = copy.deepcopy(initiManualAssign)
        f = setStdoutToFile("./curExperiment.json")
        curExpName = expName
        desDir = join("./log", curExpName)
        print(json.dumps({curExpName:1}, indent=4))
        setStdoutToDefault(f)

        makeDir(desDir)
        makeAllDir()
        #info handle decode job
        for kth in range(cfg["numOfKth"]):
            filePath = "./decode/{}th_decode.json".format(kth)
            f = setStdoutToFile(filePath)
            print(json.dumps(manualAssign, indent=4)) #* make ndarray to list
            setStdoutToDefault(f)   
        
        subprocess.call('./train.sh')
        
        # exit()
=======
                makeDir(desDir)
                makeAllDir()
                #info handle decode job
                for kth in range(cfg["numOfKth"]):
                    filePath = "./decode/{}th_decode.json".format(kth)
                    f = setStdoutToFile(filePath)
                    print(json.dumps(manualAssign, indent=4)) #* make ndarray to list
                    setStdoutToDefault(f)   
                
                subprocess.call('./train.sh')
                
                # exit()
            
>>>>>>> aa1cde33ea23406543be284d0f9b55e7ec89629f

if __name__=="__main__":
    # brutNas()
    brutInit()

    

