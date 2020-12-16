import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import stats
import math
import seaborn as sns
%matplotlib inline

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






