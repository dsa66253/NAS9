import torch
import torch.nn as nn
import random
import numpy as np
import json, os
def set_seed_cpu(seed):
    # print("set_seed_cpu seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def openCurExp():
    filePath = os.path.join("./curExperiment.json")
    f = open(filePath)
    exp = json.load(f)
    for key in exp:
        return key 

exp2IniFunc = {
    # "r0920_3": lambda weight: torch.nn.init.normal_(weight, 0.025, 0.025/2),
    # "r0920": lambda weight: torch.nn.init.normal_(weight, 0.025, 0.025/2),
    # "r0919_3": lambda weight: torch.nn.init.normal_(weight, 0.0125, 0.0125/2),
    # "r0919": lambda weight: torch.nn.init.normal_(weight, 0.025, 0.025/2),
    # "r0918_3": lambda weight: torch.nn.init.normal_(weight, 0.00625, 0.00625/2),
    # "r0918": lambda weight: torch.nn.init.uniform_(weight, 0, 0.025/2),
    # "r0924": lambda weight: torch.nn.init.normal_(weight, 0.01, 0.01/2),
    # "r0916_4": lambda weight: torch.nn.init.uniform_(weight, -0.05, 0.05),
    # "r0916_2": lambda weight: torch.nn.init.uniform_(weight, -0.05/2, 0.05/2),
    # "0925": lambda weight: torch.nn.init.uniform_(weight, -0.05, 0.0),
    # "0925_3": lambda weight: torch.nn.init.uniform_(weight, 0.0, 0.0025),
    # "0926": 0,
    # "0927": lambda weight: torch.nn.init.uniform_(weight, 0.025/4, 0.025/4),
    # "0927_3": lambda weight: torch.nn.init.uniform_(weight, 0.025/8, 0.025/8),
    # "0928_2": lambda weight: torch.nn.init.kaiming_normal_(weight),
    # "0928_3": lambda weight: torch.nn.init.uniform_(weight, -0.05, 0.05),
    # "0928_6": lambda weight: torch.nn.init.uniform_(weight, -0.05/2, 0.05/2),
    "0228": lambda weight: torch.nn.init.uniform_(weight, -4, 4),
    "0228_2": lambda weight: torch.nn.init.uniform_(weight, -2, 2),
    "0228_3": lambda weight: torch.nn.init.uniform_(weight, -2/2, 2/2),
    "0228_4": lambda weight: torch.nn.init.uniform_(weight, -2/4, 2/4),
    # "0228_5": lambda weight: torch.nn.init.uniform_(weight, -0.005, 0.005) ,
    # "0228_6": lambda weight: torch.nn.init.uniform_(weight, -0.005/2, 0.005/2),
    # "0228_7": lambda weight: torch.nn.init.uniform_(weight, -0.005/4, 0.005/4),
    # "0228_8": lambda weight: torch.nn.init.uniform_(weight, -0.005/8, 0.005/8),
    # "0228_9": lambda weight: torch.nn.init.uniform_(weight, -0.005/16, 0.005/16),
    "0227": lambda weight: torch.nn.init.normal_(weight, 0.0, 1),
    "0227_2": lambda weight: torch.nn.init.normal_(weight, 0.0, 1.0/2),
    "0227_3": lambda weight: torch.nn.init.normal_(weight, 0.0, 1.0/4),
    "0227_4": lambda weight: torch.nn.init.normal_(weight, 0.0, 1.0/8),
    "0227_5": lambda weight: torch.nn.init.normal_(weight, 0.0, 1.0/32),
    "0227_6": lambda weight: torch.nn.init.normal_(weight, 0.0, 1.0/64),
    "0227_7": lambda weight: torch.nn.init.normal_(weight, 0.0, 1.0/128),
    "0227_8": lambda weight: torch.nn.init.normal_(weight, 0.0, 1.0/256),
    "0227_9": lambda weight: torch.nn.init.normal_(weight, 0.01, 0.01/2),
    "0227_10": lambda weight: torch.nn.init.normal_(weight, 0.02, 0.02/2),
    "0227_11": lambda weight: torch.nn.init.normal_(weight, -0.01, 0.01/2),
    "0227_12": lambda weight: torch.nn.init.normal_(weight, -0.02, 0.02/2),
}
def TsengInitializeWeights(model, seed):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            set_seed_cpu(seed)
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def initialize_weights(model, seed):
    print("set initialize weight with seed ", seed)
    curExp = openCurExp()
    print("cuurent experiment", curExp)
    # TsengInitializeWeights(model, seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            set_seed_cpu(seed)
            # exp2IniFunc[curExp](m.weight)
            # torch.nn.init.kaiming_normal_(m.weight)
            # m.weight = torch.abs(m.weight)
            # exp2IniFunc[curExp](m.weight)
            torch.nn.init.uniform_(m.weight, -0.005/2, 0.005/2)
            # m.weight.data.fill_(0)
            # setTensorPositive(m.weight)
            # torch.nn.init.normal_(m.weight, 0.00625, 0.00625/2)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 1)
        elif isinstance(m, nn.Linear):
            set_seed_cpu(seed)
            # torch.nn.init.uniform_(m.weight, -0.005/2, 0.005/2)
            # exp2IniFunc[curExp](m.weight)
            # torch.nn.init.kaiming_normal_(m.weight)
            # setTensorPositive(m.weight.data)
            # torch.nn.init.uniform_(m.weight, 0, 0.025/2)
            # m.weight.data.fill_(0)
            # nn.init.constant_(m.bias, 0)
            # torch.nn.init.normal_(m.weight, 0.00625, 0.00625/2)
            pass
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
def setTensorPositive(tensor):
    tmp = torch.zeros(tensor.shape)
    tmp = torch.nn.init.kaiming_normal_(tmp)
    tmp = torch.abs(tmp)
    with torch.no_grad():
        tensor*=0
        tensor+= tmp
