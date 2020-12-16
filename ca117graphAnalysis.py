import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import ks_2samp
from scipy.stats.stats import pearsonr
import math
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import dataProcessing
import FCAMiner
# import pyRMT
import libRMT
import graphLearning
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.stats import kurtosis
import warnings
import time
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import networkx as nx
import PredictionResult
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
# sns.set()
# sns.set_style("whitegrid", {"axes.facecolor": ".9"})
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


eventLog_ca117 = pd.read_csv('ca117_eventLog_nonfixed.csv')
# eventLog_ca117 = eventLog_ca117.drop([1160345])
eventLog_ca117['time:timestamp'] = pd.to_datetime(eventLog_ca117['time:timestamp'])
eventLog_ca117 = eventLog_ca117.loc[:, ~eventLog_ca117.columns.str.contains('^Unnamed')]
# materials = eventLog_ca116.loc[:,['org:resource','concept:name','description']]
weeksEventLog = [g for n, g in eventLog_ca117.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
weeksEventLog = weeksEventLog[14:26]

lectureList = dataProcessing.getLectureList(eventLog_ca117,['html|py'])
eventLog_ca117_filtered = eventLog_ca117.loc[eventLog_ca117['description'].str.contains('|'.join(lectureList))]

# eventLog_ca116_filtered = eventLog_ca116_filtered.drop(eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('http|report|ex|dashboard|graphs.html')].index)
eventLog_ca117_filtered['pageName'] = eventLog_ca117_filtered['description'].str.extract(r'([^\/][\S]+.html)', expand=False)
eventLog_ca117_filtered['pageName'] = eventLog_ca117_filtered['pageName'].fillna('General')
eventLog_ca117_filtered['pageName'] = eventLog_ca117_filtered['pageName'].str.replace('.web','')
eventLog_ca117_filtered = eventLog_ca117_filtered.drop(eventLog_ca117_filtered.loc[eventLog_ca117_filtered['concept:name'].isin(['click-0','click-1','click-2','click-3'])].index)

eventLog_ca117_filtered.loc[eventLog_ca117_filtered['description'].str.contains('correct|incorrect'),'pageName'] = 'Practice'
eventLog_ca117_filtered.loc[eventLog_ca117_filtered['description'].str.contains('^\/einstein\/$'),'pageName'] = 'Practice'

eventLog_ca117_filtered.rename(columns={'concept:instance':'concept:instance1',
                                   'concept:name':'concept:name1',
                                   'case:concept:name' : 'case:concept:name1'},  inplace=True)
eventLog_ca117_filtered['concept:instance'] = eventLog_ca117_filtered['pageName']
eventLog_ca117_filtered['concept:name'] = eventLog_ca117_filtered['pageName']
eventLog_ca117_filtered['date'] = eventLog_ca117_filtered['time:timestamp'].dt.date

eventLog_ca117_filtered['case:concept:name'] = eventLog_ca117_filtered['date'].astype(str) + '-' + eventLog_ca117_filtered['org:resource'].astype(str)

eventLog_ca117_filtered['concept:name'] = eventLog_ca117_filtered['pageName'] + '*' + eventLog_ca117_filtered['concept:name1']
eventLog_ca117_filtered['concept:instance'] = eventLog_ca117_filtered['pageName'] + '*' + eventLog_ca117_filtered['concept:instance1']

weeksEventLog_filtered = [g for n, g in eventLog_ca117_filtered.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
weeksEventLog_filtered = weeksEventLog_filtered[14:26]

a = weeksEventLog_filtered[0]

#descriptive analysis:
listMaterials = []
for w in range(0,12):
    listMaterials.append(weeksEventLog_filtered[w]['pageName'].value_counts().rename('week' + str(w+1)))
materialAccessedByWeek = pd.concat(listMaterials, axis=1)

materialAccessedByWeek['ofWeek'] = ''
materialAccessedByWeek['pageType'] = ''
materialAccessedByWeek = materialAccessedByWeek.fillna(0)
materialAccessedByWeek.to_csv('materialAccessedByWeek_ca117_2018.csv')

materialAccessedByWeek['sumOfpageActivity'] = materialAccessedByWeek.sum(axis = 1, skipna = True)
accessedPageSummary = materialAccessedByWeek.loc[:,['pageType','sumOfpageActivity']].groupby([pd.Grouper('pageType')]).sum()
accessedPageSummary['perc']= accessedPageSummary['sumOfpageActivity']/accessedPageSummary['sumOfpageActivity'].sum()


#data upload processs
dataUpload = pd.read_csv('ca117_uploads.csv')
dataUpload['date'] = pd.to_datetime(dataUpload.date)


nonExUpload = dataUpload.drop(dataUpload.loc[dataUpload['task'].str.match('ex')].index)
nonExUploadByWeek = [g for n, g in nonExUpload.groupby(pd.Grouper(key='date',freq='W'))]
nonExUploadByWeek = nonExUploadByWeek[14:26]


#practice results
workingWeekExcercise = []
cummulativeExerciseWeeks = []
for week in range(0,12):  
        
    workingWeekExcercise.append(nonExUploadByWeek[week])
    practiceResult = pd.concat(workingWeekExcercise) #nonExUploadByWeek[week] 

    #adjust number of correct: For each task, number of correct submission/number of submission for that task
    practiceResultSum = practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
    practiceResultSum['correct_adjusted'] = practiceResultSum['correct']/practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).count()['correct']
    cummulativeResult = practiceResultSum.reset_index().groupby([pd.Grouper(key='user')]).sum()
    
    # cummulativeResult = practiceResultSum.groupby([pd.Grouper(key='user')]).sum()
    cummulativeResult['cumm_practice'] = cummulativeResult['correct']/practiceResult.groupby([pd.Grouper(key='user')]).count()['date']
    cummulativeResult['successPassedRate'] = cummulativeResult['passed']/(cummulativeResult['passed'] + cummulativeResult['failed'])
    # cummulativeResult = graphLearning.mapNewLabel(cummulativeResult, reLabelIndex)
    cummulativeExerciseWeeks.append(cummulativeResult)
    
#activity data matrix construction
weeksEventLog_filtered_pageType = []
for w in range(0,12):
    tmp = weeksEventLog_filtered[w].merge(materialAccessedByWeek.loc[:,['pageType']], left_on=weeksEventLog_filtered[w].pageName, 
                                    right_on=materialAccessedByWeek.loc[:,['pageType']].index)
    weeksEventLog_filtered_pageType.append(tmp)
    
workingWeekLog = []
activityDataMatrixWeeks_pageType = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    workingWeekLog.append(weeksEventLog_filtered_pageType[w])
    LogPageactivityCountByUser =  pd.concat(workingWeekLog) #weeksEventLog_filtered[w]
    LogPageactivityCountByUser = FCAMiner.activityDataMatrixContruct(LogPageactivityCountByUser,'pageType')
    LogPageactivityCountByUser = LogPageactivityCountByUser.fillna(0)
    # LogPageactivityCountByUser = FCAMiner.activityDataMatrixPercentage(LogPageactivityCountByUser)
    # LogPageactivityCountByUser = graphLearning.mapNewLabel(LogPageactivityCountByUser,reLabelIndex)
    activityDataMatrixWeeks_pageType.append(LogPageactivityCountByUser)
    
for w in range(0,12):
    temp = activityDataMatrixWeeks_pageType[w].merge(cummulativeExerciseWeeks[w].loc[:,:], left_on=activityDataMatrixWeeks_pageType[w].index, 
                                                     right_on=cummulativeExerciseWeeks[w].index)    
    temp = temp.set_index(['key_0'])
    activityDataMatrixWeeks_pageType[w] = temp

a =  activityDataMatrixWeeks_pageType[11].corr()

w = 5
for col1 in ['Practice','General','Lecture','Labsheet']:
    for col2 in ['correct']:
        p = stats.pearsonr(activityDataMatrixWeeks_pageType[w][col1].values,activityDataMatrixWeeks_pageType[w][col2].values)
        print(col1 + '-' + col2 + ':' + str(p[0]) + ' - ' +str(p[1]))