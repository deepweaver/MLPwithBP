
import numpy as np 
import random 
from utils import *
import matplotlib.pyplot as plt 
# random.seed(1)
# np.random.seed(1)

class BPN:
    def __init__(self, input_size=9, output_size=6, hidden_size=13, learning_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        # self.w1 = np.zeros((self.hidden_size, self.input_size+1))
        self.w1 = np.random.random((self.hidden_size, self.input_size+1))*0.3
        # self.w2 = np.zeros((self.output_size, self.hidden_size+1))
        self.w2 = np.random.random((self.output_size, self.hidden_size+1))*0.3

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
        return self.y2 

    def backward(self, t, learning_rate=None):
        if learning_rate != None:
            self.lr = learning_rate
        dEde = t - self.y2 
        self.loss = np.sum(np.abs(dEde)) 
        self.losses.append(self.loss)
        # print("dEde's shape is {}".format(dEde.shape))
        dedy2 = -1
        dy2da2 = self.y2 * (1-self.y2)
        # print("dy2da2's shape is {}".format(dy2da2.shape))

        da2dw2 = self.x2.T
        # print("da2dw2's shape is {}".format(da2dw2.shape))
        # print(self.w2.shape)
        self.dw2 = (self.lr * dEde * dedy2 * dy2da2) * da2dw2
                  #---------------------------------
        da2dx2 = self.w2
        # print(da2dx2.shape)
        da2dy1 = da2dx2[:,1:] 
        # print(da2dy1.shape)
        dy1da1 = self.y1 * (1-self.y1)
        # print(dy1da1.shape)
        da1dw1 = self.x1.T 
        # print(da1dw1.shape)
        # print(self.)* (dy1da1 * da1dw1 (13,10))
        self.dw1 = np.sum(  (self.lr * dEde * dedy2 * dy2da2) * da2dy1 , axis=0, keepdims=True).T * dy1da1 * da1dw1
                           #---------------------------------
        # w1.shape = (13,10)
        # print(self.dw1.shape) 

        self.w1 -= self.dw1
        self.w2 -= self.dw2 





        # self.w2 -= self.dw2 

        



if __name__ == "__main__":
    bpn = BPN(9,6,30,learning_rate=0.1)
    # x = np.array([1.52101,13.64,4.49,1.1,71.78,0.06,8.75,0,0], dtype=np.float64).reshape(-1,1)
    # print(bpn.forward(x))
    # t = np.array([0,1,0,0,0,0]).reshape(-1,1)
    # bpn.backward(t)
        
    epochs = 1000
    pdata, ptarget, categories = Preprocessing(*LoadData("./GlassData.csv"))
    total_n = pdata.shape[1]
    traininput = pdata[:,:int(0.8*total_n)]
    traintarget = ptarget[:,:int(0.8*total_n)]

    testinput = pdata[:,int(0.8*total_n):]
    testtarget = ptarget[:,int(0.8*total_n):]
    errors = []

    for e in range(epochs):
        for i in range(traininput.shape[1]):

            bpn.forward(traininput[:,i:i+1])
            bpn.backward(traintarget[:,i:i+1], learning_rate=1-0.2*e/epochs)
        error = 0
        correct_cnt = 0
        for i in range(testinput.shape[1]):
            out = bpn.forward(testinput[:,i:i+1])
            error += np.sum(np.abs(out-testtarget[:,i:i+1]) )
            if np.argmax(out) == np.argmax(testtarget[:,i:i+1]):
                correct_cnt += 1
        print("error = {}".format(error))
        errors.append(error)
        accuracy = correct_cnt/testinput.shape[1]
        print("accuracy = {}".format(accuracy))

    x = range(len(errors))
    y = errors
    print(min(y))
    plt.plot(x,y)
    plt.show()














