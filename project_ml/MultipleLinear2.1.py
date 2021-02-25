import numpy as np
import matplotlib as plt
import csv

with open('datacsv.csv') as inputcsv:
    csv_list=list(csv.reader(inputcsv))

x1=np.array([])
x2=np.array([])
x3=np.array([])
x4=np.array([])
x5=np.array([])
x6=np.array([])
y=np.array([])

#Multiple Linear Regression for x3 x5 and x6
for row in csv_list:
    if(row != csv_list[0]):

        x3=np.append(x3, float(row[3]))
        x5=np.append(x5,float(row[5]))
        x6=np.append(x6,float(row[6]))
        if row[7] == '':
            continue
        y = np.append(y, int(row[7]))

print(x3)
ones=np.ones((100))
X=np.column_stack((ones,x3[0:100],x5[0:100],x6[0:100]))
A=X.T
Y=y

def coef(transpose,X,y):
    coef = np.linalg.inv(transpose@X)@transpose@y
    return coef

def adjustRSQUARE(ycap,y,d):
    RSS = 0
    TSS = 0
    yavg = np.mean(y)
    for i in range(len(y)):
        RSS = RSS + (y[i] - ycap[i]) * (y[i] - ycap[i])
        TSS = TSS + (y[i] - yavg) * (y[i] - yavg)

    adjR = 1 - ((RSS / (len(y) - d - 1)) / (TSS / (len(y) - 1)))

    return adjR

def Rsquarecalculator(y,yest):
    yavg=np.mean(y)
    RSS=0
    TSS=0

    for i in range(len(y)):
        ext=y[i]-yest[i]
        ext2=y[i]-yavg
        RSS=RSS+ext*ext
        TSS=TSS+ext2*ext2

    Rsqu=1-(RSS/TSS)
    print("R^2 value ",Rsqu)

def yest(X,coef):
    yest = X @ coef
    return yest


coefs=coef(A,X,Y)
yests=yest(X,coefs)

Rsquarecalculator(Y,yests)
print("Adjusted Rsquare",adjustRSQUARE(yests,Y,3)) #The result are nearly same with Maximum Adjusted R Square score at the previous step.
