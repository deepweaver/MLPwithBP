import numpy as np 
from BPN_momentum import BPN 
from utils import *
bpn = BPN()
bpn.w1 = np.load("resources/weight1.npy")
bpn.w2 = np.load('resources/weight2.npy')
bpn.w3 = np.load('resources/weight3.npy')
# categories = [1, 2, 3, 5, 6, 7] 
data, target, idces = LoadData("./GlassData.csv")
pdata, ptarget, categories = Preprocessing(data, target)


listidx = []

for i in range(pdata.shape[1]):
    outs = ''
    out = bpn.forward(pdata[:,i:i+1]) 
    outs += ',' + ','.join([str(data[i][j]) for j in range(pdata.shape[0])]) + ','
    outs += str(categories[np.argmax(ptarget[:,i:i+1])]) + ',' + str(categories[np.argmax(out)]) + '\n'
    listidx.append([int(idces[i]), outs])
listidx = sorted(listidx, key=lambda x:x[0])
with open("./GlassData2.csv", 'w') as file:
    file.write('ID,Refractive_Index,Sodium,Magnesium,Aluminium,Silicon,Potassium,Calcium,Barium,Iron,Glass_Type,classified label')
    for i in range(pdata.shape[1]):
        file.write(str(listidx[i][0]))  
        file.write(listidx[i][1])
    

