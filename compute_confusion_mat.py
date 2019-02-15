from pycm import *
y_actu = []
y_pred = []
with open("./GlassData2.csv", 'r') as file:
    for line in file:
        a, b = line[:-1].split(',')[-2:]
        y_actu.append(a)
        y_pred.append(b)
cm = ConfusionMatrix(actual_vector=y_actu, predict_vector=y_pred) # Create CM From Data
print(y_actu)
print(y_pred)
print(cm.classes)
print(cm.table)
print(cm)
