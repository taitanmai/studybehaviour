import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import ks_2samp
import math

def footprintForActivity(dfSubLogActivityVariance,activity, activityList):
    cols = []
    check = {}
    N,S,A = 0.0,0.0,0.0
    for a in activityList:
        check[a] = 0
    for index, row in dfSubLogActivityVariance.iterrows():
        for a in activityList:
            col = activity + '-' + a
            if row[col] > 0:
                check[a] = check[a] + 1
    for c in check:
        if check[c] == 0:
            N = N + 1
        elif check[c] < len(dfSubLogActivityVariance):
            S = S + 1
        else:
            A = A + 1
    RE = 0
    for i in [A,S,N]:
        if i > 0:
            RE = RE - (i/len(activityList))*math.log2(i/len(activityList))             
    return [A,S,N], RE

def extractRE(dfSubLogActivityVariance, subLogSize, activityList):    
    i = 0
    j = subLogSize - 1
    n = len(dfSubLogActivityVariance)
    extractRE = {}
    while(j < n):
        for a in activityList:
            rc, re = footprintForActivity(dfSubLogActivityVariance.loc[i:j,:], a, activityList)
            if a not in extractRE:
                extractRE.update({ a : [[dfSubLogActivityVariance.loc[i,'startDate'],dfSubLogActivityVariance.loc[j-1,'startDate'], re]]})
            else:
                extractRE[a].append([dfSubLogActivityVariance.loc[i,'startDate'],dfSubLogActivityVariance.loc[j-1,'startDate'], re])        
        i = j
        j = j + subLogSize        
    return extractRE

def extractREbyDay(dfSubLogActivityVariance, activityList):
    return 

def KSTest(popSize, extractREactivity, iterrate = 0 ):
    pValueList = []
    j = popSize
    extractREactivity = np.array(extractREactivity)
    reValues = extractREactivity[:,2]
    n = len(extractREactivity)
    if iterrate == 1:
        while(j < n):
            i = 0
            k = i + j
            if k + j < n:
                while(k < n):
                    if k + j < n:
                        x = reValues[i:k]           
                        y = reValues[k:(k+j)]
                        pVal = ks_2samp(x, y)[1]
                        pValueList.append([j,str(extractREactivity[i][0]), str(extractREactivity[k-1][1]), str(extractREactivity[k][0]), str(extractREactivity[k+j-1][1]),i,k,pVal])
                        i = k
                        k = i + j
                    else:
                        break
                j = j + 1
            else:
                break
    else:
        i = 0
        k = i + j
        if k + j < n:
            while(k < n):
                if k + j < n:
                    x = extractREactivity[:,2][i:k]           
                    y = extractREactivity[:,2][k:(k+j)]
                    pVal = ks_2samp(x, y)[1]
                    pValueList.append([j,str(extractREactivity[i][0]),str(extractREactivity[k-1][1]), str(extractREactivity[k][0]), str(extractREactivity[k+j-1][1]),i,k,pVal])
                    i = k
                    k = i + j
                else:
                    break
            j = j + 1
    return pValueList

def KSTestForManyActivity(popSize, activityList, extractedRE):
    pValList = {}
    for a in activityList:
        pVal = KSTest(popSize,extractedRE[a])
        pValNp = np.array(pVal)
        pValList.update({a : pValNp})
    return pValList

def KSTestAveragePValForManyActivity(KSTestForManyActivityList):
    pValueAverage = []
    numberOfActivity = len(KSTestForManyActivityList)
    KSTestForManyActivityList1 = KSTestForManyActivityList.copy()
    for activity in KSTestForManyActivityList:
        if len(pValueAverage) == 0: #initialise average result set
            numberOfTests = len(KSTestForManyActivityList[activity][:,7])
            pValueAverage = np.copy(KSTestForManyActivityList[activity]) #when pValueAverage change, the main numpy array will not change
            pValueAverage[:,7] = np.zeros(numberOfTests)

        pValueAverage[:,7] = np.add(pValueAverage[:,7].astype(float),KSTestForManyActivityList[activity][:,7].astype(float))
    print(pValueAverage[:,7][0])
    print(numberOfActivity)
    pValueAverage[:,7] = pValueAverage[:,7].astype(float)/float(numberOfActivity)
    
    KSTestForManyActivityList1.update({'average' : pValueAverage}) 
    return KSTestForManyActivityList1

    
    






