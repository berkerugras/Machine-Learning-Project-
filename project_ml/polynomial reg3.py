
import csv
import numpy as np


with open('datacsv.csv') as inputcsv:
    csv_list=list(csv.reader(inputcsv))


x3=np.array([])
x5=np.array([])
x6=np.array([])
y=np.array([])

for row in csv_list:
    if(row != csv_list[0]):

        x3=np.append(x3, float(row[3]))
        x5=np.append(x5,float(row[5]))
        x6=np.append(x6,float(row[6]))
        if row[7] == '':
            continue
        y = np.append(y, int(row[7]))

ones=np.ones((100))


def coef(transpose,X,y): #I took this function from my previous lab
    coef = np.linalg.pinv(transpose@X)@transpose@y
    return coef

def rsquare(y,yest): #I took this function from my previous lab
    yavg = np.mean(y)
    RSS = 0
    TSS = 0

    for i in range(len(y)):
        ext = y[i] - yest[i]
        ext2 = y[i] - yavg
        RSS = RSS + ext * ext
        TSS = TSS + ext2 * ext2

    Rsqu = 1 - (RSS / TSS)
    return Rsqu

def ycap(X,coef): #I took this function from my previous lab
    cvest=X@coef
    return cvest

def adjustRSQUARE(ycap,y,d): #I took this function from my previous lab
    RSS = 0
    TSS = 0
    yavg = np.mean(y)
    for i in range(len(y)):
        RSS = RSS + (y[i] - ycap[i]) * (y[i] - ycap[i])
        TSS = TSS + (y[i] - yavg) * (y[i] - yavg)

    adjR = 1 - ((RSS / (len(y) - d - 1)) / (TSS / (len(y) - 1)))

    return adjR



x3=x3[0:100]
x5=x5[0:100]
x6=x6[0:100]
X=np.column_stack((x3,x5,x6,x3*x3,x5*x5,x6*x6,x3*x5,x3*x6,x5*x6,x3*x5*x6)) 
A=X.T
coefficients=coef(A,X,y)
estimation=ycap(X,coefficients)
Rsquare=rsquare(y,estimation)
adjstrsq=adjustRSQUARE(estimation,y,10)
print("RSQUARE VALUE",Rsquare)
print("ADJUSTED RSQUARE VALUE",adjstrsq)
