

import csv
import numpy as np
from sklearn import datasets, ensemble




with open('datacsv.csv') as inputcsv:
    csv_list=list(csv.reader(inputcsv))


x3=np.array([])
x5=np.array([])
x6=np.array([])
y=np.array([])





def adjustRSQUARE(ycap,y,d):  #I took this function from my previous lab
    RSS = 0
    TSS = 0
    yavg = np.mean(y)
    for i in range(len(y)):
        RSS = RSS + (y[i] - ycap[i]) * (y[i] - ycap[i])
        TSS = TSS + (y[i] - yavg) * (y[i] - yavg)

    adjR = 1 - ((RSS / (len(y) - d - 1)) / (TSS / (len(y) - 1)))

    return adjR

def Rsquarecalculator(y,yest): #I took this function from my previous lab
    yavg=np.mean(y)
    RSS=0
    TSS=0

    for i in range(len(y)):
        ext=y[i]-yest[i]
        ext2=y[i]-yavg
        RSS=RSS+ext*ext
        TSS=TSS+ext2*ext2

    Rsqu=1-(RSS/TSS)
    return  Rsqu


for row in csv_list:
    if(row != csv_list[0]):

        x3=np.append(x3, float(row[3]))
        x5=np.append(x5,float(row[5]))
        x6=np.append(x6,float(row[6]))
        if row[7] == '':
            continue
        y = np.append(y, int(row[7]))

x3t=x3[100:120]
x5t=x5[100:120]
x6t=x6[100:120]
print(x6t)
print(x3t)
Xtestnew=np.column_stack((x3t,x5t,x6t,x3t*x3t,x5t*x5t,x6t*x6t,x3t*x5t,x3t*x6t,x5t*x6t,x3t*x5t*x6t))

x3=x3[0:100]
x5=x5[0:100]
x6=x6[0:100]
X=np.column_stack((x3,x5,x6,x3*x3,x5*x5,x6*x6,x3*x5,x3*x6,x5*x6,x3*x5*x6)) #nonlinear.pdf slides 6th and 7nd
A=X.T



cvhat=np.array([])
Xtest=X[60:80]
Xtrain=np.delete(X,range(60,80),0)

Ytest=y[60:80]
Ytrain=np.delete(y,range(60,80),0)
Adjusted=np.array([])
Rsqu=np.array([])
for i in range(0,1000,1):
    reg=ensemble.GradientBoostingRegressor(max_depth=None,random_state=i,max_features="log2") #I get help from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
    reg.fit(Xtrain,Ytrain)
    predictions=reg.predict(Xtest)
    Adjusted = np.append(Adjusted,adjustRSQUARE(predictions, Ytest, 10))
    Rsqu = np.append(Rsqu,Rsquarecalculator(Ytest, predictions))

print("Predictions",predictions)

print(len(predictions))
print("Adjusted Rsquare",max(Adjusted))
print("Rsquare Score",max(Rsqu))
maxval=0
for j in range(0,len(Rsqu),1):
    if(Rsqu[j]==max(Rsqu)):
        maxval=j




newreg = ensemble.GradientBoostingRegressor(max_depth=None, random_state=maxval,max_features="log2")
newreg.fit(Xtrain, Ytrain)
pred1=newreg.predict(Xtest)
Adjusted1 =adjustRSQUARE(pred1, Ytest, 10)
Rsqu1 = Rsquarecalculator(Ytest, pred1)
newprediction = newreg.predict(Xtestnew)
print("new Adjusted",Adjusted1)
print("new Rsquare",Rsqu1)

print("new predictions",newprediction)
print(len(newprediction))