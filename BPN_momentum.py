
import numpy as np 
import random 
from utils import *
import matplotlib.pyplot as plt 
random.seed(0)
np.random.seed(0)

class BPN:
    def __init__(self, input_size=9, output_size=6, hidden_size1=90, hidden_size2=60, learning_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.w1 = np.random.random((self.hidden_size1, self.input_size+1))*0.3
        self.w2 = np.random.random((self.hidden_size2, self.hidden_size1+1))*0.3
        self.w3 = np.random.random((self.output_size, self.hidden_size2+1))*0.3

        self.lr = learning_rate
        self.losses = []

    def forward(self, x):
        bias = np.ones((1,x.shape[1]), dtype=x.dtype)
        self.x1 = np.r_[bias,x] 
        assert self.x1.shape[0] == self.w1.shape[1], "w1 size don't match!"
        self.a1 = np.dot(self.w1,self.x1)
        self.y1 = sigmoid(self.a1)
        bias = np.ones((1,self.y1.shape[1]), dtype=x.dtype)
        self.x2 = np.r_[bias,self.y1] 
        assert self.x2.shape[0] == self.w2.shape[1], "w2 size don't match!"
        self.a2 = np.dot(self.w2,self.x2)
        self.y2 = sigmoid(self.a2)
        bias = np.ones((1,self.y2.shape[1]), dtype=x.dtype)
        self.x3 = np.r_[bias,self.y2]
        assert self.x3.shape[0] == self.w3.shape[1], "w2 size don't match!"
        self.a3 =  np.dot(self.w3, self.x3) 
        self.y3 = sigmoid(self.a3) 

        return self.y3

    def backward(self, t, learning_rate=None):
        if learning_rate != None:
            self.lr = learning_rate

        dEde = t - self.y3 
        self.loss = np.sum(np.abs(dEde)) 
        self.losses.append(self.loss)
        dedy3 = -1 
        dy3da3 = self.y3 * ( 1 - self.y3) 
        da3dw3 = self.x3.T 
        tmp3 = self.lr * dEde * dedy3 * dy3da3
        self.dw3 = tmp3 * da3dw3 

        da3dx3 = self.w3 
        da3dy2 = da3dx3[:,1:]
        dy2da2 = self.y2 * (1 - self.y2) 
        da2dw2 = self.x2.T 
        tmp2 = np.sum( tmp3 * da3dy2, axis=0, keepdims=True).T * dy2da2
        self.dw2 = tmp2 * da2dw2 

        da2dx2 = self.w2 
        da2dy1 = da2dx2[:,1:] 
        dy1da1 = self.y1 * (1 - self.y1) 
        da1dw1 = self.x1.T 
        self.dw1 = np.sum( tmp2 * da2dy1, axis=0, keepdims=True).T * dy1da1 * da1dw1 


        

        # self.dw1 = np.sum(  (self.lr * dEde * dedy2 * dy2da2) * da2dy1 , axis=0, keepdims=True).T * dy1da1 * da1dw1
                           #---------------------------------
        # w1.shape = (13,10)
        # print(self.dw1.shape) 

        self.w1 -= self.dw1
        self.w2 -= self.dw2 
        self.w3 -= self.dw3 




        # self.w2 -= self.dw2 

        



if __name__ == "__main__":
    bpn = BPN(9,6,learning_rate=0.1)
    # x = np.array([1.52101,13.64,4.49,1.1,71.78,0.06,8.75,0,0], dtype=np.float64).reshape(-1,1)
    # print(bpn.forward(x))
    # t = np.array([0,1,0,0,0,0]).reshape(-1,1)
    # bpn.backward(t)
        
    epochs = 1000
    data, target, _ = LoadData("./GlassData.csv")
    pdata, ptarget, categories = Preprocessing(data, target)
    total_n = pdata.shape[1]
    traininput = pdata[:,:int(0.7*total_n)]
    traintarget = ptarget[:,:int(0.7*total_n)]
    validationinput = pdata[:,int(0.7*total_n):int(0.85*total_n)]
    validationtarget = ptarget[:,int(0.7*total_n):int(0.85*total_n)]
    testinput = pdata[:,int(0.85*total_n):]
    testtarget = ptarget[:,int(0.85*total_n):]
    errors = []
    accuracies_of_vali = []
    for e in range(epochs):
        correct_cnt_train = 0
        for i in range(traininput.shape[1]):

            out = bpn.forward(traininput[:,i:i+1])
            bpn.backward(traintarget[:,i:i+1], learning_rate=1-0.1*e/epochs)
            if np.argmax(out) == np.argmax(traintarget[:,i:i+1]):
                correct_cnt_train += 1
        print("accuracy of trainset = {}".format(correct_cnt_train/traininput.shape[1]))
        error = 0
        correct_cnt = 0
        for i in range(validationinput.shape[1]):
            out = bpn.forward(validationinput[:,i:i+1])
            error += np.sum(np.abs(out-validationtarget[:,i:i+1]) )
            if np.argmax(out) == np.argmax(validationtarget[:,i:i+1]):
                correct_cnt += 1
        # print("error = {}".format(error))
        errors.append(error)
        accuracy = correct_cnt/testinput.shape[1]
        print("accuracy of validationset = {}".format(accuracy))
        accuracies_of_vali.append(accuracy) 
        # if accuracy > 0.78:
        #     break
    test_correct_cnt = 0
    for i in range(testinput.shape[1]):
        out = bpn.forward(testinput[:,i:i+1])
        if np.argmax(out) == np.argmax(testtarget[:,i:i+1]):
            print(np.argmax(out))
            test_correct_cnt += 1
    print("testset accuracy is = {}".format(test_correct_cnt/testinput.shape[1])) 
    print("max accuracy of validation set is {}".format(max(accuracies_of_vali)))
    x = range(len(accuracies_of_vali))
    y = accuracies_of_vali
    print(min(y))
    plt.plot(x,y)
    plt.show()

    # uncomment the following to save the weights
    # np.save("./resources/weight1.npy", bpn.w1)
    # np.save("./resources/weight2.npy", bpn.w2)
    # np.save("./resources/weight3.npy", bpn.w3) 





    # with open("./weights.txt", 'w') as file:
    #     for i in range(bpn.w1.shape[0]):
    #         for j in range(bpn.w1.shape[1]):
    #             file.write(str(bpn.w1[i,j]) + ' ')
    #         file.write('\n')
    #     file.write('\n')
    #     for i in range(bpn.w2.shape[0]):
    #         for j in range(bpn.w2.shape[1]):
    #             file.write(str(bpn.w2[i,j]) + ' ')
    #         file.write('\n')
    #     file.write('\n')
    #     for i in range(bpn.w3.shape[0]):
    #         for j in range(bpn.w3.shape[1]):
    #             file.write(str(bpn.w3[i,j]) + ' ')
    #         file.write('\n') 















