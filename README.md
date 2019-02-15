### run
- run a network with one hidden layer, execute `python3 ./BPN.py`
- run a network with tow hidden layer, execute `python3 ./BPN_momentum.py`
  
### result
command `python3 ./BPN_momentum.py` generated result can be found in `network_output.txt`
```
accuracy of trainset = 0.98%
accuracy of validationset = 0.76%
testset accuracy is = 0.73%
```


### explanation
## data preprocessing
in `utils.py`, function `Preprocessing(data, target):` , basicly, `def LoadData(filepath):` loads the csv file and convert it (randomly) into a data list and a test list. Then feed the return value, which is `data` and `target` into `Preprocessing(data, target):`, in which I use `processeddata = scale(np.array(data),-1,1)` to scale all the input data into range [-1,1]. I also in this function, convert the target list into one-hot vectors.
without random shuffling, the accuracy is only 42%

## validating and testing
I devide the dataset into 70% training, 15% validation, 15% testing. Specifically, I use the following code:
```python
    total_n = pdata.shape[1]
    traininput = pdata[:,:int(0.7*total_n)]
    traintarget = ptarget[:,:int(0.7*total_n)]
    validationinput = pdata[:,int(0.7*total_n):int(0.85*total_n)]
    validationtarget = ptarget[:,int(0.7*total_n):int(0.85*total_n)]
    testinput = pdata[:,int(0.85*total_n):]
    testtarget = ptarget[:,int(0.85*total_n):]
```
the pdata stands for "processed data", I split it using slicing of numpy 2d array.

## weight initialization
```python
    self.w1 = np.random.random((self.hidden_size1, self.input_size+1))*0.3
    self.w2 = np.random.random((self.hidden_size2, self.hidden_size1+1))*0.3
    self.w3 = np.random.random((self.output_size, self.hidden_size2+1))*0.3
```
I tested using zero weight as initial weights but later found random initialization is better. Theoratically a smaller weight is even better so I multiplied the random weight with a factor 0.3(I tested other numbers like 0.2,0.1,0.5 and found this 0.3 is the best choice)

## learning rate
```python
class BPN:
    def __init__(self, input_size=9, output_size=6, hidden_size1=90, hidden_size2=60, learning_rate=0.1):
```
When you initialize the bpn, you can set the learning rate and the default is 0.1. I also added `def backward(self, t, learning_rate=None):` learning rate adjustment in the update period, my intuition is that with the epoch getting larger, the learning rate should get smaller and smaller. So each backward in each epoch, I set `            bpn.backward(traintarget[:,i:i+1], learning_rate=1-e/epochs)` the learning rate with 1-e/epochs. When the e is approaching epoch, the learning rate should be aproaching zero.

## network structure
I tested network with one hidden layer with different layer sizes. However, the accuracy is below 0.7 . Then I decided to add another layer. According to paper `Do Deep Nets Really Need to be Deep?`, deeper layers is more ofter than not perform better than simply adding more hidden neurals in a shallow net. So I set the first hidden layer 90 neurons and the second with 60 neurons.

## momentum and regularization
I tested momentum with alpha=0.1 and alpha=0.2 but it did not improve the result. I guess considering my learning rate is decreasing and the training epochs are large, it's not necessary to use momentum.

## Final weight vectors
all three weight vectors are in file "./weights.txt"

## generated class label
`testbpn.py` generates class label and output to file `GlassData2.csv`, the classified label is the last column

## generate comfusion matrix
run `compute_confusion_mat.py` to generate confusion matrix
but first you should `pip3 install pycm==1.8` install pycm

Predict          1     2     3     5     6     7
Actual
1                66    3     1     0     0     0

2                5     69    1     0     0     1

3                4     1     12    0     0     0

5                0     1     0     11    0     1

6                0     0     0     0     9     0

7                2     0     0     0     0     27

Precision
0.85714                 0.93243                 0.85714                 1.0                     1.0              0.93103

Recall
0.94286                 0.90789                 0.70588                 0.84615                 1.0              0.93103
(for further info, look below)





Overall Statistics :

95% CI                                                           (0.86754,0.94554)
AUNP                                                             0.93321
AUNU                                                             0.93335
Bennett S                                                        0.88785
CBA                                                              0.87468
Chi-Squared                                                      855.92716
Chi-Squared DF                                                   25
Conditional Entropy                                              0.49593
Cramer V                                                         0.89439
Cross Entropy                                                    2.18211
Gwet AC1                                                         0.89055
Hamming Loss                                                     0.09346
Joint Entropy                                                    2.67246
KL Divergence                                                    0.00558
Kappa                                                            0.87216
Kappa 95% CI                                                     (0.81881,0.92551)
Kappa No Prevalence                                              0.81308
Kappa Standard Error                                             0.02722
Kappa Unbiased                                                   0.8721
Lambda A                                                         0.85507
Lambda B                                                         0.85401
Mutual Information                                               1.62493
NIR                                                              0.35514
Overall ACC                                                      0.90654
Overall CEN                                                      0.1472
Overall J                                                        (5.01537,0.83589)
Overall MCC                                                      0.87305
Overall MCEN                                                     0.22564
Overall RACC                                                     0.26895
Overall RACCU                                                    0.26931
P-Value                                                          0.0
PPV Macro                                                        0.92963
PPV Micro                                                        0.90654
Phi-Squared                                                      3.99966
RCI                                                              0.74657
RR                                                               35.66667
Reference Entropy                                                2.17653
Response Entropy                                                 2.12086
SOA1(Landis & Koch)                                              Almost Perfect
SOA2(Fleiss)                                                     Excellent
SOA3(Altman)                                                     Very Good
SOA4(Cicchetti)                                                  Excellent
Scott PI                                                         0.8721
Standard Error                                                   0.0199
TPR Macro                                                        0.88897
TPR Micro                                                        0.90654
Zero-one Loss                                                    20

Class Statistics :

Classes                                                          1                       2                       3                       5                       6              7
ACC(Accuracy)                                                    0.92991                 0.94393                 0.96729                 0.99065                 1.0              0.98131
AUC(Area under the roc curve)                                    0.93323                 0.93583                 0.84787                 0.92308                 1.0              0.96011
AUCI(Auc value interpretation)                                   Excellent               Excellent               Very Good               Excellent               Excellent              Excellent
BM(Informedness or bookmaker informedness)                       0.86647                 0.87166                 0.69573                 0.84615                 1.0              0.92022
CEN(Confusion entropy)                                           0.16716                 0.14125                 0.25907                 0.11502                 0              0.11124
DOR(Diagnostic odds ratio)                                       199.5                   262.2                   234.0                   None                    None              1235.25
DP(Discriminant power)                                           1.26802                 1.33346                 1.30622                 None                    None              1.70457
DPI(Discriminant power interpretation)                           Limited                 Limited                 Limited                 None                    None              Limited
ERR(Error rate)                                                  0.07009                 0.05607                 0.03271                 0.00935                 0.0              0.01869
F0.5(F0.5 score)                                                 0.87302                 0.92742                 0.82192                 0.96491                 1.0              0.93103
F1(F1 score - harmonic mean of precision and sensitivity)        0.89796                 0.92                    0.77419                 0.91667                 1.0              0.93103
F2(F2 score)                                                     0.92437                 0.9127                  0.73171                 0.87302                 1.0              0.93103
FDR(False discovery rate)                                        0.14286                 0.06757                 0.14286                 0.0                     0.0              0.06897
FN(False negative/miss/type 2 error)                             4                       7                       5                       2                       0              2
FNR(Miss rate or false negative rate)                            0.05714                 0.09211                 0.29412                 0.15385                 0.0              0.06897
FOR(False omission rate)                                         0.0292                  0.05                    0.025                   0.00985                 0.0              0.01081
FP(False positive/type 1 error/false alarm)                      11                      5                       2                       0                       0              2
FPR(Fall-out or false positive rate)                             0.07639                 0.03623                 0.01015                 0.0                     0.0              0.01081
G(G-measure geometric mean of precision and sensitivity)         0.89898                 0.92008                 0.77784                 0.91987                 1.0              0.93103
GI(Gini index)                                                   0.86647                 0.87166                 0.69573                 0.84615                 1.0              0.92022
IS(Information score)                                            1.38979                 1.39261                 3.43161                 4.04103                 4.57154              2.78039
J(Jaccard index)                                                 0.81481                 0.85185                 0.63158                 0.84615                 1.0              0.87097
LS(Lift score)                                                   2.62041                 2.62553                 10.78992                16.46154                23.77778              6.87039
MCC(Matthews correlation coefficient)                            0.84699                 0.87703                 0.76089                 0.91532                 1.0              0.92022
MCEN(Modified confusion entropy)                                 0.25544                 0.22192                 0.34437                 0.17138                 0              0.17301
MK(Markedness)                                                   0.82795                 0.88243                 0.83214                 0.99015                 1.0              0.92022
N(Condition negative)                                            144                     138                     197                     201                     205              185
NLR(Negative likelihood ratio)                                   0.06187                 0.09557                 0.29713                 0.15385                 0.0              0.06972
NPV(Negative predictive value)                                   0.9708                  0.95                    0.975                   0.99015                 1.0              0.98919
P(Condition positive or support)                                 70                      76                      17                      13                      9              29
PLR(Positive likelihood ratio)                                   12.34286                25.05789                69.52941                None                    None              86.12069
PLRI(Positive likelihood ratio interpretation)                   Good                    Good                    Good                    None                    None              Good
POP(Population)                                                  214                     214                     214                     214                     214              214
PPV(Precision or positive predictive value)                      0.85714                 0.93243                 0.85714                 1.0                     1.0              0.93103
PRE(Prevalence)                                                  0.3271                  0.35514                 0.07944                 0.06075                 0.04206              0.13551
RACC(Random accuracy)                                            0.1177                  0.12281                 0.0052                  0.00312                 0.00177              0.01836
RACCU(Random accuracy unbiased)                                  0.11796                 0.12283                 0.00525                 0.00314                 0.00177              0.01836
TN(True negative/correct rejection)                              133                     133                     195                     201                     205              183
TNR(Specificity or true negative rate)                           0.92361                 0.96377                 0.98985                 1.0                     1.0              0.98919
TON(Test outcome negative)                                       137                     140                     200                     203                     205              185
TOP(Test outcome positive)                                       77                      74                      14                      11                      9              29
TP(True positive/hit)                                            66                      69                      12                      11                      9              27
TPR(Sensitivity, recall, hit rate, or true positive rate)        0.94286                 0.90789                 0.70588                 0.84615                 1.0              0.93103
Y(Youden index)                                                  0.86647                 0.87166                 0.69573                 0.84615                 1.0              0.92022
dInd(Distance index)                                             0.0954                  0.09898                 0.29429                 0.15385                 0.0              0.06981
sInd(Similarity index)                                           0.93254                 0.93001                 0.7919                  0.89121                 1.0              0.95064


