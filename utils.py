import numpy as np 
import random 

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 


def LoadData(filepath):
    data = []
    target = []
    tmp = []
    shuffled_idces = []
    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            tmp.append(line)
    random.shuffle(tmp)
    for i, s in enumerate(tmp):
        shuffled_idces.append(s.split(",")[0])
        dataline = s.split(",")[1:]
        data.append(list(map(float, dataline[:-1])))
        target.append(int(dataline[-1]))
    return data, target, shuffled_idces

def Preprocessing(data, target):
    assert len(data) == len(target), "size doesn't match "
    processeddata = scale(np.array(data),-1,1)
    categories = []
    for i,v in enumerate(target):
        if (v not in categories):
            categories.append(v)
    categories.sort()
    # print(categories) 
    nc = len(categories)
    nt = len(target)
    processedtarget = np.zeros((nc, nt))
    for i,v in enumerate(target):
        processedtarget[categories.index(v),i] = 1
    return processeddata.T, processedtarget, categories


def sigmoid(x):
    return 1 / (1 + np.exp(-x))





if __name__ == "__main__":
    data, target, _ = LoadData("./GlassData.csv")
    print(len(data), len(target))
    print(data[0], target[0])
    pdata, pt, c = Preprocessing(data, target)
    print(pdata.shape)
    print(pdata[:,0])
    print(pt.shape)
    print(pt[:,0])
    print(c)


    # x = np.random.random((2,3))
    # print(sigmoid(x))
