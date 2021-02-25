import numpy as np
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
for row in csv_list:
    if(row != csv_list[0]):
        x1=np.append(x1, float(row[1]))
        x2=np.append(x2, float(row[2]))
        x3=np.append(x3, float(row[3]))
        x4=np.append(x4, float(row[4]))
        x5=np.append(x5, float(row[5]))
        x6=np.append(x6, float(row[6]))
        if row[7] == '':
            continue
        y = np.append(y, int(row[7]))

ones=np.ones((100))
X=np.column_stack((ones, x1[0:100], x2[0:100], x3[0:100],x4[0:100],x5[0:100],x6[0:100]))
Y=y
A=X.T
def ycap(X,coef): #I got it from my previous lab.
    cvest=X@coef
    return cvest

def coef(transpose,X,y): #I got it from my previous lab.
    coef = np.linalg.pinv(transpose@X)@transpose@y
    return coef

def adjustedR(ycap,ysalary,d): #I got it from my previous lab.
    RSS=0
    TSS=0
    yavg=np.mean(ysalary)
    for i in range(len(ysalary)):
        RSS=RSS+(ysalary[i]-ycap[i])*(ysalary[i]-ycap[i])
        TSS=TSS+(ysalary[i]-yavg)*(ysalary[i]-yavg)


    adjR=1-((RSS/(len(ysalary)-d-1))/(TSS/(len(ysalary)-1)))

    return adjR

def rsquare(y,yest): #I got it from my previous lab.
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



def backwardselection(A,X,y): #I got this function from my lab 6.
    M6cap = np.array([])
    M6coef = coef(A, X, y)
    M6cap = np.append(M6cap, ycap(X, M6coef))
    M6adjr = adjustedR(M6cap, y, len(X[0] - 1))

    adjRsquare = np.array([])
    deleted = np.array([])
    adjRsquare = np.append(adjRsquare, M6adjr)
    while (len(X[0]) > 1):

        RSStank = np.array([])
        for j in range(1, len(X[0]), 1):
            Matrix = X
            Matrix = np.delete(X, j, axis=1)
            Coef = coef(Matrix.T, Matrix, y)
            Cap = ycap(Matrix, Coef)
            RSStank = np.append(RSStank, rsquare(y, Cap))

        tmpr = np.sort(RSStank)

        for i in range(0, len(RSStank), 1):
            if (tmpr[len(RSStank) - 1] == RSStank[i]):
                deleted = np.append(deleted, i + 1)
                X = np.delete(X, i + 1, axis=1)
                print(i+1)
                break
        print(X) # I printed new metrices here
        Thelastcoef = coef(X.T, X, y)
        Thelastcap = ycap(X, Thelastcoef)
        adjRsquare = np.append(adjRsquare, adjustedR(Thelastcap, y, len(X[0])-1))
    print("Models",adjRsquare) # I look at the sequence of which adjusted R Squared score is maximum, then I looked at the new X matrix results to understand which matrix gives that score

#As a result of backward selection I understand that if I remove x1,x2 and x4 predictors I will get better Rsquare scores.
backwardselection(A,X,Y)

