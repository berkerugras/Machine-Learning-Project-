import numpy as np
import csv

with open('datacsv.csv') as inputcsv:
    csv_list=list(csv.reader(inputcsv))


x3=np.array([])
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

ones=np.ones((100))
X=np.column_stack((ones,x3[0:100],x5[0:100],x6[0:100]))
A=X.T
Y=y

def cvest(X,coef): 
    cvest=X@coef
    return cvest

def coef(transpose, X, y): 
    coef = np.linalg.inv(transpose @ X) @ transpose @ y
    return coef

def MSE(yest,y): #I took this function from my previous lab
    sum=0
    for i in range(0,len(y),1):
        sum+=(yest[i]-y[i])*(yest[i]-y[i])

    MSE=sum/len(y)
    return MSE

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
    return  Rsqu

def kfoldfunction(X,y,k):
    CVMSE=np.array([])
    foldsize = int(np.round((len(X) / k)))
    for i in range(0, len(X) , int(foldsize)):
        test = X[i:i + int(foldsize)]
        train = np.delete(X, range(i, i + int(foldsize)), 0)
        yTest = y[i:i + int(foldsize)]
        ytrain = np.delete(y, range(i, i + int(foldsize)), 0)
        trainspose = train.T
        coef1 = coef(trainspose, train, ytrain)
        cvhat = cvest(test, coef1)
        mseval=MSE(cvhat,yTest)
        CVMSE=np.append(CVMSE,mseval)


    minumum=np.min(CVMSE)

    for j in range(len(CVMSE)):
        if(CVMSE[j]==minumum):
            index=j
    print("x length",len(X))

    xnewtest=X[index*foldsize:index*foldsize+foldsize]
    xnewtrain=np.delete(X,range(index*foldsize,index*foldsize+foldsize),0)

    xtrainspose=xnewtrain.T

    ynewtest=Y[index*foldsize:index*foldsize+foldsize]
    ynewtrain=np.delete(Y,range(index*foldsize,index*foldsize+foldsize),0)



    coefnew=coef(xtrainspose,xnewtrain,ynewtrain)
    print("x train length", len(xnewtrain))
    print("x test length", len(xnewtest))
    newest=cvest(xnewtest,coefnew)
    rsqnew=Rsquarecalculator(ynewtest,newest)
    adjstrsq=adjustRSQUARE(newest,ynewtest,3)
    print("rsquare",rsqnew)
    print("adjusted rsq",adjstrsq)


kfoldfunction(X,Y,10)
