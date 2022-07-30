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

import mlfinlab as ml
from mlfinlab.networks.mst import MST
from mlfinlab.networks.dash_graph import DashGraph
from networkx.algorithms import community
import itertools
from plotly.offline import plot
import plotly.graph_objects as go

basePath = 'E:\\Data\\extractedData\\'

#extract event log 
eventLog_ca177 = pd.read_csv(basePath + 'ca177_eventLog_nonfixed.csv')
eventLog_ca177 =eventLog_ca177.loc[eventLog_ca177['time:timestamp'] != ' ']
eventLog_ca177['time:timestamp'] = pd.to_datetime(eventLog_ca177['time:timestamp'])
eventLog_ca177 = eventLog_ca177.loc[:, ~eventLog_ca177.columns.str.contains('^Unnamed')]
# materials = eventLog_ca116.loc[:,['org:resource','concept:name','description']]
weeksEventLog = [g for n, g in eventLog_ca177.groupby(pd.Grouper(key='time:timestamp',freq='W'))]

#process for new activity

# lectureList = dataProcessing.getLectureList(eventLog_ca116,['html|py'])
# eventLog_ca116_filtered = eventLog_ca116.loc[eventLog_ca116['description'].str.contains('|'.join(lectureList))]
eventLog_ca177_filtered = eventLog_ca177.loc[eventLog_ca177['description'].str.contains('.html|.py|#|/einstein/')]
eventLog_ca177_filtered['pageName'] = eventLog_ca177_filtered['description'].str.extract(r'([^\/][\S]+.html)', expand=False)
eventLog_ca177_filtered['pageName'] = eventLog_ca177_filtered['pageName'].str.replace('.web','')
eventLog_ca177_filtered = eventLog_ca177_filtered.drop(eventLog_ca177_filtered.loc[eventLog_ca177_filtered['concept:name'].isin(['click-0','click-1','click-2','click-3'])].index)

eventLog_ca177_filtered.loc[eventLog_ca177_filtered['description'].str.contains('correct|incorrect'),'pageName'] = 'Practice'
eventLog_ca177_filtered.loc[eventLog_ca177_filtered['description'].str.contains('^\/einstein\/'),'pageName'] = 'Practice'
eventLog_ca177_filtered['pageName'] = eventLog_ca177_filtered['pageName'].fillna('General')

a = eventLog_ca177_filtered.loc[eventLog_ca177_filtered['pageName'] == 'Practice']

eventLog_ca177_filtered.rename(columns={'concept:instance':'concept:instance1',
                                   'concept:name':'concept:name1',
                                   'case:concept:name' : 'case:concept:name1'},  inplace=True)
eventLog_ca177_filtered['concept:instance'] = eventLog_ca177_filtered['pageName']
eventLog_ca177_filtered['concept:name'] = eventLog_ca177_filtered['pageName']
eventLog_ca177_filtered['date'] = eventLog_ca177_filtered['time:timestamp'].dt.date

eventLog_ca177_filtered['case:concept:name'] = eventLog_ca177_filtered['date'].astype(str) + '-' + eventLog_ca177_filtered['org:resource'].astype(str)

# eventLog_ca116_filtered['concept:name'] = eventLog_ca116_filtered['pageName'] + '*' + eventLog_ca116_filtered['concept:name1']
# eventLog_ca116_filtered['concept:instance'] = eventLog_ca116_filtered['pageName'] + '*' + eventLog_ca116_filtered['concept:instance1']


weeksEventLog_filtered = [g for n, g in eventLog_ca177_filtered.groupby(pd.Grouper(key='time:timestamp',freq='W'))]

for i in weeksEventLog_filtered:
    print(len(i))

a =  weeksEventLog_filtered[19]   

weeksEventLog_filtered = weeksEventLog_filtered[8:20]



#------------- Extract number of activities and assign pageType dictionary for later use
#--------- descriptive analysis
listMaterials = []
for w in range(0,12):
    listMaterials.append(weeksEventLog_filtered[w]['pageName'].value_counts().rename('week' + str(w+1)))
materialAccessedByWeek = pd.concat(listMaterials, axis=1)

materialAccessedByWeek['ofWeek'] = ''
materialAccessedByWeek['pageType'] = ''
materialAccessedByWeek = materialAccessedByWeek.fillna(0)
materialAccessedByWeek.to_csv(basePath + 'materialAccessedByWeek_ca177_2018.csv')

materialAccessedByWeek = pd.read_csv(basePath + 'materialAccessedByWeek_ca177_2018.csv', index_col=0)

materialAccessedByWeek['sumOfpageActivity'] = materialAccessedByWeek.sum(axis = 1, skipna = True)
accessedPageSummary = materialAccessedByWeek.loc[:,['pageType','sumOfpageActivity']].groupby([pd.Grouper('pageType')]).sum()
accessedPageSummary['perc']= accessedPageSummary['sumOfpageActivity']/accessedPageSummary['sumOfpageActivity'].sum()

weeksEventLog_filtered_pageType = []
for w in range(0,12):
    tmp = weeksEventLog_filtered[w].merge(materialAccessedByWeek.loc[:,['pageType','ofWeek']], left_on=weeksEventLog_filtered[w].pageName, 
                                    right_on=materialAccessedByWeek.loc[:,['pageType']].index)    
    tmp.loc[tmp['pageName'] == 'Practice',['ofWeek']] = w
    tmp['pageTypeWeek'] = tmp['pageType'] + '_' + tmp['ofWeek'].astype(str)
    tmp['concept:name'] = tmp['pageTypeWeek'] + '*' + tmp['concept:instance1']
    tmp['concept:instance'] = tmp['pageTypeWeek'] + '*' + tmp['concept:instance1']
    weeksEventLog_filtered_pageType.append(tmp)
    
a = weeksEventLog_filtered_pageType[6].sort_values(['case:concept:name','time:timestamp'])

dataUpload = pd.read_csv(basePath + 'ca177_uploads.csv')
dataUpload['date'] = pd.to_datetime(dataUpload.date)

exUpload = dataUpload.loc[dataUpload['task'].str.match('ex')]

ex1 = exUpload.loc[exUpload['task'].str.match('ex1')]
ex1 = ex1.sort_values(by=['user','task'])
ex1 = ex1.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
ex2 = exUpload.loc[exUpload['task'].str.match('ex2')]
ex2 = ex2.sort_values(by=['user','task'])
ex2 = ex2.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()

assessment1A = dataProcessing.assessmentConstruction(ex1,4)
assessment1A['adjustedPerformance'] = (assessment1A['perCorrect'] + assessment1A['perPassed'])/2
assessment2A = dataProcessing.assessmentConstruction(ex2,4)
assessment2A['adjustedPerformance'] = (assessment2A['perCorrect'] + assessment2A['perPassed'])/2


assessment1A.rename(columns={'correct':'correct1A',
                          'perCorrect':'perCorrect1A',
                          'failed':'failed1A',
                            'passed':'passed1A',
                            'perPassed':'perPassed1A',
                            'testSubmitted':'testSubmitted1A',
                            'adjustedPerformance':'adjustedPerformance1A'}, 
                  inplace=True)
assessment2A.rename(columns={'correct':'correct2A',
                          'perCorrect':'perCorrect2A',
                          'failed':'failed2A',
                            'passed':'passed2A',
                            'perPassed':'perPassed2A',
                            'testSubmitted':'testSubmitted2A',
                            'adjustedPerformance':'adjustedPerformance2A'}, 
                  inplace=True)

assessment1A = assessment1A.set_index(['user'])
assessment2A = assessment2A.set_index(['user'])


assessment = pd.concat([assessment1A,assessment2A], axis=1)
assessment = assessment.fillna(0)

assessment['grade'] = (assessment['perCorrect1A']+assessment['perCorrect2A'])/3
assessment['perPassed'] = (assessment['passed1A'] + assessment['passed2A'])/(assessment['passed1A'] + assessment['passed2A'] +
                        + assessment['failed1A']+ assessment['failed2A'])

ex1_excellent = assessment1A.loc[(assessment1A['perCorrect1A'] <= 1) & (assessment1A['perCorrect1A'] >= 0.6)]
ex1_weak = assessment1A.loc[(assessment1A['perCorrect1A'] >= 0) & (assessment1A['perCorrect1A'] < 0.6)]

ex2_excellent = assessment2A.loc[(assessment2A['perCorrect2A'] <= 1)&(assessment2A['perCorrect2A'] >= 0.5)]
ex2_weak = assessment2A.loc[(assessment2A['perCorrect2A'] >= 0) & (assessment2A['perCorrect2A'] < 0.5)]


nonExUpload = dataUpload.drop(dataUpload.loc[dataUpload['task'].str.match('ex')].index)
nonExUploadByWeek = [g for n, g in nonExUpload.groupby(pd.Grouper(key='date',freq='W'))]

nonExUpload['version'].unique()
#merge exam result with transition data matrix:
reLabelIndex = dataProcessing.reLabelStudentId(assessment.index)


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
    
    
assessment_label = assessment.copy()
assessment_label = graphLearning.mapNewLabel(assessment, reLabelIndex)

assessment_label1A = assessment1A.copy()
assessment_label1A = graphLearning.mapNewLabel(assessment1A, reLabelIndex)
assessment_label2A = assessment2A.copy()
assessment_label2A = graphLearning.mapNewLabel(assessment2A, reLabelIndex)

#activity data matrix construction with newPAgeTYpe
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
    temp = activityDataMatrixWeeks_pageType[w].merge(cummulativeExerciseWeeks[w].loc[:,:], left_on=activityDataMatrixWeeks_pageType[w].index, right_on=cummulativeExerciseWeeks[w].index)
    temp = temp.set_index(['key_0'])
    if w in [0,1,2,3,4,5]:
        studentResult = assessment_label1A
    elif w in [6,7,8,9,10,11]:
        studentResult = assessment_label2A
    temp = temp.merge(studentResult, left_on=temp.index, right_on=studentResult.index)
    temp = temp.set_index(['key_0'])
    if -1 in temp.index:
        temp = temp.drop([-1])
    activityDataMatrixWeeks_pageType[w] = temp

a =  activityDataMatrixWeeks_pageType[11].corr()

workingWeekLog = []
activityDataMatrixWeeks_pageTypeWeek = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    workingWeekLog.append(weeksEventLog_filtered_pageType[w])
    LogPageactivityCountByUser =  pd.concat(workingWeekLog) #weeksEventLog_filtered[w]
    LogPageactivityCountByUser = FCAMiner.activityDataMatrixContruct(LogPageactivityCountByUser,'pageTypeWeek')
    LogPageactivityCountByUser = LogPageactivityCountByUser.fillna(0)
    # LogPageactivityCountByUser = FCAMiner.activityDataMatrixPercentage(LogPageactivityCountByUser)
    # LogPageactivityCountByUser = graphLearning.mapNewLabel(LogPageactivityCountByUser,reLabelIndex)
    activityDataMatrixWeeks_pageTypeWeek.append(LogPageactivityCountByUser)

a = activityDataMatrixWeeks_pageTypeWeek[11].columns

#eliminate zero column    
for w in range(0,12):
    activityDataMatrixWeeks_pageTypeWeek[w] = activityDataMatrixWeeks_pageTypeWeek[w].loc[:, (activityDataMatrixWeeks_pageTypeWeek[w] != 0).any(axis=0)]
    activityDataMatrixWeeks_pageTypeWeek[w] = activityDataMatrixWeeks_pageTypeWeek[w].loc[:, ~activityDataMatrixWeeks_pageTypeWeek[w].columns.str.contains('Exam')]
    
for w in range(0,12):
    activityDataMatrixWeeks_pageTypeWeek[w].to_csv(basePath + 'transitionMatrixStorage_new/ca1162020_activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv',index=True)

for w in range(0,12):
    if w in [0,1,2,3,4,5]:
        studentResult = assessment1A
    elif w in [6,7,8,9,10,11]:
        studentResult = assessment2A
    activityDataMatrixWeeks_pageTypeWeek[w] = activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w].index.isin(studentResult.index)]


#read data page activity 2019 to combine
activityDataMatrixWeeks_pageTypeWeek_2019 = []
for w in range(0,12):
    activityDataMatrixWeeks_pageTypeWeek_2019.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1772019_activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv',index_col=0))

ex2_excellent_2019 = pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1772019_ex2_excellent.csv',index_col=0)
ex2_weak_2019 = pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1772019_ex2_weak.csv',index_col=0)

ex2_excellent = pd.concat([ex2_excellent, ex2_excellent_2019])
ex2_weak = pd.concat([ex2_weak,ex2_weak_2019])

#combind data 2018 and 2019
for w in range(0,12):
    activityDataMatrixWeeks_pageTypeWeek[w] = pd.concat([activityDataMatrixWeeks_pageTypeWeek[w], activityDataMatrixWeeks_pageTypeWeek_2019[w]])

for w in range(0,12):
    activityDataMatrixWeeks_pageTypeWeek[w] = activityDataMatrixWeeks_pageTypeWeek[w].dropna(1)



    
activityExamData = []   
for w in range(0,12):
    temp = activityDataMatrixWeeks_pageTypeWeek[w].merge(cummulativeExerciseWeeks[w].loc[:,:], left_on=activityDataMatrixWeeks_pageTypeWeek[w].index.astype(str), right_on=cummulativeExerciseWeeks[w].index)
    temp = temp.set_index(['key_0'])
    if w in [0,1,2,3,4,5]:
        studentResult = assessment1A
    elif w in [6,7,8,9,10,11]:
        studentResult = assessment2A
    temp = temp.merge(studentResult, left_on=temp.index, right_on=studentResult.index)
    temp = temp.set_index(['key_0'])
    if -1 in temp.index:
        temp = temp.drop([-1])
    activityExamData.append(temp)

a =  activityExamData[11].corr()
a1 = a['perCorrect2A'].sort_values()

a =  activityExamData[5].corr()
a2 = a['perCorrect1A'].sort_values()


examCorrelation = pd.concat([a2,a1], axis=1)
pageTypeWeekList = pd.concat([weeksEventLog_filtered_pageType[i] for i in range(0,12)])['pageTypeWeek'].unique()
examCorrelation = examCorrelation.loc[examCorrelation.index.isin(pageTypeWeekList)]

examCorrelation = examCorrelation.sort_index()
examCorrelation.rename(columns={'perCorrect1A':'Lab Exam 1',
                          'perCorrect2A':'Lab Exam 2'}, 
                  inplace=True)
sns.heatmap(examCorrelation, cmap='RdYlGn')

#correlation analysis between exam, practice and activity
w = 7
for col1 in pageTypeWeekList:
    if col1 in activityDataMatrixWeeks_pageTypeWeek[w]:
        for col2 in ['perCorrect2A']:
            p = stats.pearsonr(activityDataMatrixWeeks_pageTypeWeek[w][col1].values,activityDataMatrixWeeks_pageTypeWeek[w][col2].values)
            print(col1 + '-' + col2 + ':' + str(p[0]) + ' - ' +str(p[1]))
        
for week in range(0,12):
    if week in [0,1,2,3,4,5,6]:
        exellent = ex1_excellent.index
        weak = ex1_weak.index
    elif week in [6,7,8,9,10,11]:
        exellent = ex2_excellent.index
        w = ex2_weak.index        
    activityDataMatrixWeeks_pageType[week]['result'] = 2
    activityDataMatrixWeeks_pageType[week].loc[activityDataMatrixWeeks_pageType[week].index.isin(exellent),['result']] = 1
    activityDataMatrixWeeks_pageType[week].loc[activityDataMatrixWeeks_pageType[week].index.isin(weak),['result']] = 0

for week in range(0,12):    
    print('Week: ' + str(week) + '...')
    for col1 in ['Practice','General','Lecture','Solution','Labsheet']:
            p = stats.ttest_ind(activityDataMatrixWeeks_pageType[week].loc[activityDataMatrixWeeks_pageType[week]['result']==1,[col1]],
                                activityDataMatrixWeeks_pageType[week].loc[activityDataMatrixWeeks_pageType[week]['result']==0,[col1]], equal_var=False)
            print(col1 + ':' + str(p[0]) + ' - ' +str(p[1]))


#-------------------------------------------------------------------------------------
#IPR
   
pca_result = []
pcaDataWeeks = []
columnsReturn2 = []
for w in range(0,12):
    # tempData = transitionDataMatrixWeeks[w].loc[:,columns]
    tempData = activityDataMatrixWeeks_pageTypeWeek[w]
    # tempData = tempData.merge(prediction_transition[w+1]['data']['successPassedRate'], left_on = tempData.index, right_on=prediction_transition[w+1]['data']['successPassedRate'].index).set_index('key_0')
    temp = FCAMiner.PCAcohortToValue(tempData)
    temp1 = temp[1]
    pcaResult = temp[0]
    # temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
    pcaDataWeeks.append(temp1)
    pca_result.append(pcaResult)
    columnsReturn2.append(temp[2])

   
for w in range(0,12):
    pcaDataWeeks[w]['result_exam_1'] = 0
    if w in [0,1,2,3,4,5]:
        pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ex1_excellent.index),['result_exam_1']] = 1
    elif w in [6,7,8,9,10,11]:
        pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ex2_excellent.index),['result_exam_1']] = 1

fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0
num_bins = 100
for w in range(0,12):
    ax = fig.add_subplot(3,4,w+1)
    graph.append(ax)
    graph[countGraph].set_xlabel('Eigenvalues', fontsize = 15)
    graph[countGraph].set_ylabel('IPR', fontsize = 15)
    graph[countGraph].set_title('Inverse Participation Ratio week ' + str(w+1), fontsize = 20)
    graph[countGraph].grid()
    # graph[countGraph].axhline(y=0, color='k')
    # graph[countGraph].axvline(x=0, color='k')
    eigenValueList = pca_result[w].explained_variance_
    eigenVectorList = pca_result[w].components_
    IPRlist = libRMT.IPRarray(eigenValueList,eigenVectorList)
    graph[countGraph].axhline(y=IPRlist['IPR'].mean(), color='k', label='mean value of IPR') 
    graph[countGraph].plot(IPRlist['eigenvalue'], IPRlist['IPR'], '-', color ='blue', label='IPR')
    graph[countGraph].legend(loc='upper right')
    countGraph = countGraph + 1           
plt.show()

#draw one graph only:
fig = plt.figure(figsize=(20,15),dpi=120)
ax = fig.subplots()
ax.set_xlabel('Eigenvalues', fontsize = 30)
ax.set_ylabel('IPR', fontsize = 30)
ax.tick_params(axis='both', which='major', labelsize=25)
ax.tick_params(axis='both', which='minor', labelsize=25)
# ax.set_title('Inverse Participation Ratio week ' + str(w+1), fontsize = 30)
ax.grid()
# graph[countGraph].axhline(y=0, color='k')
# graph[countGraph].axvline(x=0, color='k')
eigenValueList = pca_result[9].explained_variance_
eigenVectorList = pca_result[9].components_
IPRlist = libRMT.IPRarray(eigenValueList,eigenVectorList)
ax.axhline(y=IPRlist['IPR'].mean(), color='k', label='mean value of IPR') 
ax.plot(IPRlist['eigenvalue'], IPRlist['IPR'], '-', color ='blue', label='IPR')
ax.legend(loc='upper right', fontsize=20)
plt.show()

#outbounce select
a = libRMT.selectOutboundComponents(pcaDataWeeks[11],pca_result[11].explained_variance_)

#eigenvalues
fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0
num_bins = 100
for w in range(0,10):
    ax = fig.add_subplot(3,4,w+1)
    graph.append(ax)
    graph[countGraph].set_xlabel('eigenvalue λ', fontsize = 15)
    graph[countGraph].set_ylabel('P(λ)', fontsize = 15)
    graph[countGraph].set_title('Week ' + str(w+1), fontsize = 20)
    graph[countGraph].grid()
    # graph[countGraph].axhline(y=0, color='k')
    # graph[countGraph].axvline(x=0, color='k')
    eigenValueList_graph = pca_result[w].explained_variance_
    
    n, bins, patches = graph[countGraph].hist(eigenValueList_graph, num_bins, 
                           density = 1,  
                           color ='blue',  
                           alpha = 0.7, label='Sampled') 
    densityArray = libRMT.marcenkoPastur(len(pcaDataWeeks[w]),len(pcaDataWeeks[w].columns),bins)
    density = densityArray[0]
    graph[countGraph].plot(bins, density, '-', color ='black',label='RMT') 
    graph[countGraph].legend(loc='upper right')
    min_lambda = densityArray[1]
    max_lambda = densityArray[2]
    graph[countGraph].text( min_lambda*1.1, -0.08, 'λ-', color = 'black', ha = 'center', va = 'center',fontsize=15)
    graph[countGraph].text( max_lambda*1.1, -0.08, 'λ+', color = 'black', ha = 'center', va = 'center',fontsize=15)
    countGraph = countGraph + 1           
plt.show()

#week 12 only
w=11
fig = plt.figure(figsize=(10,10),dpi=120)
ax = fig.add_subplot(1,1,1)

ax.set_xlabel('eigenvalue λ', fontsize = 15)
ax.set_ylabel('P(λ)', fontsize = 15)
# ax.set_title('Empirical eigenvalue distribution vs RMT prediction', fontsize = 20)
ax.grid()
# graph[countGraph].axhline(y=0, color='k')
# graph[countGraph].axvline(x=0, color='k')
eigenValueList_graph = pca_result[w].explained_variance_

n, bins, patches = ax.hist(eigenValueList_graph, num_bins, 
                       density = 1,  
                       color ='blue',  
                       alpha = 0.7, label='Empirical') 
densityArray = libRMT.marcenkoPastur(len(pcaDataWeeks[w]),len(pcaDataWeeks[w].columns),bins)
density = densityArray[0]
ax.plot(bins, density, '-', color ='black',label='RMT') 
ax.legend(loc='upper right')
min_lambda = densityArray[1]
max_lambda = densityArray[2]
ax.text( min_lambda*1.1, -0.08, 'λ-', color = 'black', ha = 'center', va = 'center',fontsize=15)
ax.text( max_lambda*1.1, -0.08, 'λ+', color = 'black', ha = 'center', va = 'center',fontsize=15)
countGraph = countGraph + 1           
plt.show()


w = 11 
libRMT.biplot(pcaDataWeeks[w],
       np.transpose(pca_result[w].components_[1:3, :]),
       pcaDataWeeks[w].loc[:,['result_exam_1']], activityDataMatrixWeeks_pageTypeWeek[w].columns, 'pc2','pc3', 15,  title=['Lower performing students','Higher performing students'])

#plot loadings

import libRMT    
libRMT.plotLoadings(1,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  
libRMT.plotLoadings(7,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  
libRMT.plotLoadings(10,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  

pc = 3
from scipy import stats
for w in range(0,12):
    if 'pc'+str(pc) in pcaDataWeeks[w].columns:
        a1 = pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 1,['pc'+str(pc)]]
        b1 = pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 0,['pc' + str(pc)]]
        t1, p1 = stats.ttest_ind(a1,b1)
    
        
        print('Week ' + str(w) + ':')
        print('--PC' + str(pc) + ': ' + 't-value: ' + str(t1) + ' p-value: ' + str(p1))
        print('-- Excellent: ' + str(a1.mean()[0]))
        print('-- Weak: ' + str(b1.mean()[0]))  
        


# Community Analysis
workingWeekLog = []
transitionDataMatrixWeeks = []
full_transitionDataMatrixWeeks = []
for week in range(0,12):
    print('Week: ' + str(week) + '...')
    workingWeekLog.append(weeksEventLog_filtered_pageType[week])
    Log = pd.concat(workingWeekLog) 
    tempTransition = FCAMiner.transitionDataMatrixConstruct_directFollow(Log,'pageTypeWeek',[]).fillna(0)
    full_transitionDataMatrixWeeks.append(tempTransition)   
    tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()         
    transitionDataMatrixWeeks.append(tempTransition)

#eliminate zero column    
for w in range(0,12):
    transitionDataMatrixWeeks[w] = transitionDataMatrixWeeks[w].loc[:, (transitionDataMatrixWeeks[w] != 0).any(axis=0)]
    transitionDataMatrixWeeks[w] = transitionDataMatrixWeeks[w].loc[:, ~transitionDataMatrixWeeks[w].columns.str.contains('Exam')]
    
for w in range(0,12):
    transitionDataMatrixWeeks[w].to_csv(basePath + 'transitionMatrixStorage_new/ca1772018_transitionDataMatrixWeeks_direct_accumulated_pageTypeWeek_manyPractice_w'+str(w)+'.csv',index=True)

#read csv iff neeeded
transitionDataMatrixWeeks = []
for w in range(0,12):
    temp = pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1772018_transitionDataMatrixWeeks_direct_accumulated_pageTypeWeek_manyPractice_w' + str(w) + '.csv', index_col=0)
    if w in [0,1,2,3,4,5]:
        studentList = assessment1A.index
    elif w in [6,7,8,9,10,11]:
        studentList = assessment2A.index 
    temp = temp.loc[temp.index.isin(studentList)]
    # temp = graphLearning.mapNewLabel(temp, reLabelIndex)
    # temp = temp.drop(['Practice_0-Practice_0'],axis=1)
    # if w == 1:
    #     temp = temp.drop([8])
    transitionDataMatrixWeeks.append(temp) 
    

############################################
#################### try k-means
from sklearn.cluster import KMeans
from sklearn import preprocessing

w = 11
transitionDataMatrixWeeks_directFollow_standardised11 = preprocessing.normalize(transitionDataMatrixWeeks[w], norm='l2')
transitionDataMatrixWeeks_directFollow_standardised11 = pd.DataFrame(transitionDataMatrixWeeks_directFollow_standardised11, index = transitionDataMatrixWeeks[w].index, columns = transitionDataMatrixWeeks[w].columns)

numberOfClusters = []
mixedCommunityRate = []

for n in range(2, 11):
    print('Number of group: ' + str(n))
    kmeans = KMeans(n_clusters=n, random_state=0).fit(transitionDataMatrixWeeks_directFollow_standardised11)
    kmeans_result = kmeans.labels_
    kmeans_result_df = pd.DataFrame(kmeans_result, index=transitionDataMatrixWeeks_directFollow_standardised11.index,columns=['group'])
    numberOfClusters.append(n)
    numberOfMixedCommunity = 0
    for i in range(0,n):
        kmeans_result_df_group0 = kmeans_result_df.loc[kmeans_result_df['group'] == i,:]
        a = assessment2A.loc[assessment2A.index.isin(kmeans_result_df_group0.index),['perCorrect2A']]
        print(i)
        print(a.mean())
        print("Number of students in the group: " + str(len(a)))
        excellentRate = len(a.loc[a.index.isin(ex2_excellent.index),:])/len(a)
        if (excellentRate >= 0.35) and (excellentRate <= 0.65):
            print(excellentRate)
            numberOfMixedCommunity += 1    
    mixedCommunityRate.append(numberOfMixedCommunity/n)



# end k means #######################


transitionDataMatrixWeeks_directFollow_standardised = []    
for w in range(0,12):
    transitionDataMatrixWeeks_directFollow_standardised.append(dataProcessing.normaliseData(transitionDataMatrixWeeks[w].T))

transitionDataMatrixWeeks_directFollow_normalised = []    
for w in range(0,12):
    transitionDataMatrixWeeks_directFollow_normalised.append(dataProcessing.normaliseData(transitionDataMatrixWeeks[w], 'normalised'))


#transpose transition data matrix
for w in range(0,12):
    transitionDataMatrixWeeks[w] = transitionDataMatrixWeeks[w].T
    transitionDataMatrixWeeks_directFollow_normalised[w] = transitionDataMatrixWeeks_directFollow_normalised[w].T
    transitionDataMatrixWeeks_directFollow_standardised[w] = transitionDataMatrixWeeks_directFollow_standardised[w].T

columns = transitionDataMatrixWeeks[11].columns[10:20]
pd.plotting.scatter_matrix(transitionDataMatrixWeeks[11].loc[:,columns], alpha=0.2, ax=ax)



transitionDataMatrixWeeks_directFollow_normalised[11].plot(x = 0 , y= 1, kind="scatter")

#clean dataset directly

import FCAMiner
pca_result = []
pcaDataWeeks = []
columnsReturn2 = []
for w in range(0,12):
    # tempData = transitionDataMatrixWeeks[w].loc[:,columns]
    tempData = transitionDataMatrixWeeks_directFollow_standardised[w]
    # tempData = tempData.merge(prediction_transition[w+1]['data']['successPassedRate'], left_on = tempData.index, right_on=prediction_transition[w+1]['data']['successPassedRate'].index).set_index('key_0')
    temp = FCAMiner.PCAcohortToValue(tempData)
    temp1 = temp[1]
    pcaResult = temp[0]
    # temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
    pcaDataWeeks.append(temp1)
    pca_result.append(pcaResult)
    columnsReturn2.append(temp[2])    
    
    
import libRMT

transitionDataMatrixWeeks_directFollow_standardised_outbound = []
for week in range(0,12):
    transitionDataMatrixWeeks_directFollow_standardised_outbound.append(libRMT.selectOutboundComponents(pcaDataWeeks[week],pca_result[week].explained_variance_, mode="upper"))

transitionDataMatrixWeeks_directFollow_standardised_cleaned = []    
for week in range(0,12):
    outBoundComponents = transitionDataMatrixWeeks_directFollow_standardised_outbound[week].columns
    componentToClean = ['pc1']
    for c in pcaDataWeeks[week].columns:
        if c not in outBoundComponents:
            componentToClean.append(c)
    transitionDataMatrixWeeks_directFollow_standardised_cleaned.append(libRMT.cleanEigenvectorEffect(transitionDataMatrixWeeks_directFollow_standardised[week],pcaDataWeeks[week], 
                                                           componentToClean, pca_result[week].components_,0.8,0.5)) 



# correlation processing    
# correlation processing    
corrList_cleaned = []
# corrDistanceList = []
for w in range(0,12):
    corrTemp = transitionDataMatrixWeeks_directFollow_standardised_cleaned[w].corr()
    corrList_cleaned.append(corrTemp)
    # corrDistance = (0.5*(1 - corrTemp)).apply(np.sqrt)
    # corrDistanceList.append(corrDistance)

    
#correlation processing    
corrList_original = []
# corrDistanceList_dataNormalised = []
for w in range(0,12):
    corrTemp = transitionDataMatrixWeeks_directFollow_standardised[w].corr()
    corrList_original.append(corrTemp)
    # corrDistance = (0.5*(1 - corrTemp)).apply(np.sqrt)
    # corrDistanceList_dataNormalised.append(corrDistance)

corrList_cleaned_correlation = []
for w in range(0,12):
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    tn_relation = transitionDataMatrixWeeks_directFollow_standardised[w].shape[0] / transitionDataMatrixWeeks_directFollow_standardised[w].shape[1]
    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01
    matrix = corrList_original[w]
    coreTemp = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=False)
    corrList_cleaned_correlation.append(coreTemp)

    
graph_all_weeks = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList_original[w]
    # risk_estimators = ml.portfolio_optimization.RiskEstimators()
    # tn_relation = transitionDataMatrixWeeks_directFollow_standardised[w].shape[0] / transitionDataMatrixWeeks_directFollow_standardised[w].shape[1]
    # The bandwidth of the KDE kernel
    # kde_bwidth = 0.01
    # Finding the Вe-noised Сovariance matrix
    # denoised_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth)
    # denoised_matrix_byLib = pd.DataFrame(denoised_matrix_byLib, index=matrix.index, columns=matrix.columns) denoise_method='target_shrink',
    
    detoned_matrix_byLib = matrix #risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=True)
    # detoned_matrix_byLib = matrix #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)
    distance_matrix = (0.5*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    graphBuild = MST(distance_matrix, 'distance')
    # graphBuild = nx.from_pandas_adjacency(distance_matrix)   
    graph_all_weeks.append(graphBuild)
    
graph_all_weeks_cleaned = []
for w in range(1,12):
    print('Week ' + str(w) + '...')
    matrix = corrList_cleaned[w]
    # risk_estimators = ml.portfolio_optimization.RiskEstimators()
    # tn_relation = transitionDataMatrixWeeks_directFollow_standardised[w].T.shape[0] / transitionDataMatrixWeeks_directFollow_standardised[w].T.shape[1]
    # The bandwidth of the KDE kernel
    # kde_bwidth = 0.01
    # Finding the Вe-noised Сovariance matrix
    # denoised_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth)
    # denoised_matrix_byLib = pd.DataFrame(denoised_matrix_byLib, index=matrix.index, columns=matrix.columns)
    
    # detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, denoise_method='target_shrink', detone=True)
    detoned_matrix_byLib = matrix #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)
    distance_matrix = (0.5*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    graphBuild = MST(distance_matrix, 'distance')
    # graphBuild = nx.from_pandas_adjacency(distance_matrix)   
    graph_all_weeks_cleaned.append(graphBuild)
    
graph_all_weeks_cleaned_correlation = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList_original[w]
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    tn_relation = transitionDataMatrixWeeks_directFollow_standardised[w].shape[0] / transitionDataMatrixWeeks_directFollow_standardised[w].shape[1]
    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01
    # Finding the Вe-noised Сovariance matrix
    # denoised_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth)
    # denoised_matrix_byLib = pd.DataFrame(denoised_matrix_byLib, index=matrix.index, columns=matrix.columns)
    
    detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=True)
    # detoned_matrix_byLib = matrix #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)
    distance_matrix = (0.5*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    # distance_matrix = distance_matrix.fillna(0)
    graphBuild = MST(distance_matrix, 'distance')
    # graphBuild = nx.from_pandas_adjacency(distance_matrix)   
    graph_all_weeks_cleaned_correlation.append(graphBuild)    
    
graph_all_weeks_fullyConnected = []
corrDistance_detoned = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList_original[w]
    # risk_estimators = ml.portfolio_optimization.RiskEstimators()
    # tn_relation = transitionDataMatrixWeeks_directFollow_normalised_transposed[w].shape[0] / transitionDataMatrixWeeks_directFollow_normalised_transposed[w].shape[1]
    # The bandwidth of the KDE kernel
    # kde_bwidth = 0.01
    # detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=True)
    detoned_matrix_byLib = matrix #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)
    distance_matrix = (0.5*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    g = graphLearning.createGraphFromCorrDistance(distance_matrix)
    graph_all_weeks_fullyConnected.append(g)

import graphLearning
graphLearning.checkIfUniqueEdges(graph_all_weeks_fullyConnected[11])

import community as community_louvain
partition = community_louvain.best_partition(graph_all_weeks_cleaned_correlation[11].graph,resolution=1)
girvan_newman_cluster = graphLearning.convertGNcommunityToFlattenList(communityListWeeks_cleaned_correlation[11][7], list(partition.keys()))

partition = community_louvain.best_partition(graph_all_weeks[11].graph,resolution=1)
girvan_newman_cluster = graphLearning.convertGNcommunityToFlattenList(communityListWeeks[11][7], list(partition.keys()))

from sklearn import metrics
metrics.homogeneity_score(np.array(list(partition.values())), np.array(list(girvan_newman_cluster.values())))
metrics.completeness_score(np.array(list(partition.values())), np.array(list(girvan_newman_cluster.values())))
metrics.homogeneity_completeness_v_measure(np.array(list(partition.values())), np.array(list(girvan_newman_cluster.values())), beta=1.0)


partitionConverted = graphLearning.convertFlattenListToCommunity(partition)
partitionConverted =  [v for k, v in partitionConverted.items()]
partitionConverted = tuple(partitionConverted)
partitionConvertedList = [partitionConverted,partitionConverted,partitionConverted,partitionConverted,partitionConverted,partitionConverted,
                          partitionConverted,partitionConverted,partitionConverted,partitionConverted,partitionConverted,partitionConverted,
                          partitionConverted,partitionConverted,partitionConverted]  
     


#community detection
communityListWeeks = []
for w in range(0,12):
    print('Week ' + str(w) + '...')      
    num_comms = len(graph_all_weeks[w].graph._node)
    communityListWeeks.append(graphLearning.community_dection_graph(graph_all_weeks[w], num_comms=num_comms))

communityListWeeks_not_cleaned = []
for w in range(0,12):
    print('Week ' + str(w) + '...')      
    num_comms = len(graph_all_weeks_cleaned[w].graph._node)
    communityListWeeks_not_cleaned.append(graphLearning.community_dection_graph(graph_all_weeks_cleaned[w],  num_comms=num_comms))
    
communityListWeeks_cleaned_correlation = []
for w in range(0,12):
    print('Week ' + str(w) + '...')      
    num_comms = len(graph_all_weeks_cleaned_correlation[w].graph._node)
    communityListWeeks_cleaned_correlation.append(graphLearning.community_dection_graph(graph_all_weeks_cleaned_correlation[w], num_comms=num_comms))

#dendrogram visualisation
graphLearning.girvanNewManDendrogram(graph_all_weeks_cleaned_correlation[11], ex2_excellent.index, ex2_weak.index, num_comms=num_comms, color_threshold = 16)


import graphLearning
import scikit_posthocs as sp
pd.set_option("display.max_rows", None, "display.max_columns", None)
aw10 = graphLearning.extractAssessmentResultOfCommunities(communityListWeeks_cleaned_correlation[11], assessment2A, 'perCorrect2A')
aw10t = sp.posthoc_conover(aw10[1][5])


aw6 = graphLearning.extractAssessmentResultOfCommunities(communityListWeeks[6], assessment1A, 'perCorrect1A')
aw6t = sp.posthoc_conover(aw6[3][5])

a = graphLearning.findTogetherMembers(aw10[18][5],aw11[18][5], aw10[18][1],aw11[18][1])
len(set(assessment2A.index).intersection(set(assessment3A.index)))

a = graphLearning.findTogetherMembers(aw7[18][5],aw11[18][5], aw7[18][1],aw11[18][1])
for i in range(0,8):
    for j in range(0,8):
        if len(a[i][j]) > 1:
            print(a[i][j])

good = 4
bad = 7

goodCommunity = aw10[7][2][good]
badCommunity = aw10[7][2][bad]
w = 11
extractGoodBadCommunity = activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w].index.astype(str).isin(goodCommunity.index) | 
                                                                      activityDataMatrixWeeks_pageTypeWeek[w].index.astype(str).isin(badCommunity.index)]
extractGoodBadCommunity['group'] = 0
extractGoodBadCommunity.loc[extractGoodBadCommunity.index.astype(str).isin(goodCommunity.index),['group']] = good
extractGoodBadCommunity.loc[extractGoodBadCommunity.index.astype(str).isin(badCommunity.index),['group']] = bad

columnListStatsSig = []
for c in extractGoodBadCommunity.columns:
    t1 = stats.normaltest(extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == 1, [c]])[1][0]
    t2 = stats.normaltest(extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == 0, [c]])[1][0]
    if t1 <= 0.1 and t2 <= 0.1:
        columnListStatsSig.append(c)
        
extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == 2, ['Lecture_4']].hist(bins=80)

compareMean = []
for c in extractGoodBadCommunity.columns:
    # arr1 = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == good, [c]]
    # arr2 = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == bad, [c]]
    # test, pvalue = stats.mannwhitneyu(arr1,arr2)

    if c!= 'group':
        meanGood = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == good, [c]].mean()[0]
        stdGood = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == good, [c]].std()[0]
        meanBad = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == bad, [c]].mean()[0]
        stdBad = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == bad, [c]].std()[0]
        compareMean.append([c, meanGood, stdGood, meanBad, stdBad])
        # print(c + ': ' + str(test) + ' Good Community: ' + str(meanGood) + ' -- ' + 'Bad Community: ' + str(meanBad))
compareMeanDf = pd.DataFrame(compareMean, columns=['Material','Mean Good', 'SD Good', 'Mean Bad', 'SD Bad'])
compareMeanDfnewCol = compareMeanDf['Material'].str.split('_', expand = True)
compareMeanDf['week'] = compareMeanDfnewCol[1].astype(str)
compareMeanDf['MaterialType'] = compareMeanDfnewCol[0]
compareMeanDf = compareMeanDf.sort_values(['MaterialType','week'])

compareMeanDf.to_csv(basePath + 'ca1772018_compareMeanDf.csv',index=True)


# visualise community detection results

import matplotlib.cm as cm  

G = graph_all_weeks_cleaned_correlation[11].graph


edgeList = G.edges()

for n in G.nodes:
    nodelist.append(n)
    if n in ex2_excellent.index:
        node_color.append('blue')
    else:
        node_color.append('red')


orderCommunityList = [aw10[8][5][4],aw10[8][5][6],aw10[8][5][9],aw10[8][5][8],aw10[8][5][3],aw10[8][5][0],aw10[8][5][5],aw10[8][5][1],aw10[8][5][2],aw10[8][5][7]]    

node_color = []
nodelist = []
for m in orderCommunityList:
    for n in m.index:
        nodelist.append(n)
        if n in ex2_excellent.index:
            node_color.append('blue')
        else:
            node_color.append('red')

G1 = nx.Graph()
G1.add_nodes_from(nodelist)
G1.add_edges_from(edgeList)

sns.reset_orig

plt.figure(figsize=(20,20))
# pos = nx.planar_layout(G)
pos = nx.circular_layout(G1)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', 3)
nx.draw_networkx_nodes(G1, pos, nodelist,  node_size=100, cmap=cmap, node_color=node_color)
nx.draw_networkx_edges(G1, pos, alpha=1)
plt.grid(False)
# nx.draw_networkx_labels(G, pos, font_size=12)
ax=plt.axes()
ax.set_facecolor('white')
plt.show()


#draw horizontal barchart for compare mean activity
# create plot
fig, ax = plt.subplots(figsize=(30,20), dpi=150)
index = np.arange(len(compareMeanDf.index))
bar_width = 0.4
opacity = 1

rects1 = plt.barh(index, compareMeanDf['Best Group'], bar_width, alpha=opacity, color='b', label='Best Group')
rects2 = plt.barh(index + bar_width, compareMeanDf['Worst Group'], bar_width, alpha=opacity, color='g', label='Worst Group')

plt.ylabel('Course Materials', fontsize=20)
plt.xlabel('Average number of activities', fontsize=20)
plt.title('')
plt.yticks(index + bar_width, compareMeanDf.Material, fontsize=18)
plt.xticks(fontsize=20)
plt.legend(fontsize=25)

fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0    
for w in range(0,12): 
    if w in [0,1,2,3,4,5]:
        excellent = ex1_excellent.index
        weak = ex1_weak.index
    elif w in [6,7,8,9,10,11]:
        excellent = ex2_excellent.index
        weak = ex2_weak.index        
    # excellent = exellentPractice[w]
    # weak = weakPractice[w]
    
    excellentLine = []
    weakLine = []
    mixedLine = []
    
    excellentLine_not_cleaned = []
    weakLine_not_cleaned = []
    mixedLine_not_cleaned = []
    noOfCommunities = []
    upper = 0.70
    lower = 0.30
    for i in range(0,len(graph_all_weeks[w].graph._node)-1):
        a1 = graphLearning.identifyCommunitiesType(communityListWeeks[w][i], excellent, weak)
        excellentLine.append(len(a1.loc[a1['excellentRate'] >= upper]))
        weakLine.append(len(a1.loc[a1['excellentRate'] < lower]))
        mixedLine.append(len(a1.loc[(a1['excellentRate'] <upper) & (a1['excellentRate'] >= lower)]))
        
        a2 = graphLearning.identifyCommunitiesType(communityListWeeks_not_cleaned[w][i], excellent, weak)
        excellentLine_not_cleaned.append(len(a2.loc[a2['excellentRate'] >= upper]))
        weakLine_not_cleaned.append(len(a2.loc[a2['excellentRate'] < lower]))
        mixedLine_not_cleaned.append(len(a2.loc[(a2['excellentRate'] < upper) & (a2['excellentRate'] >= lower)]))
        
        noOfCommunities.append(i+2)
    
    ax = fig.add_subplot(3,4,w+1)
    graph.append(ax)
    graph[countGraph].set_xlabel('Number of communities', fontsize = 15)
    graph[countGraph].set_ylabel('Number of the communities in each group', fontsize = 15)
    graph[countGraph].set_title('Week' + str(w+1), fontsize = 20)
    graph[countGraph].grid()
    # graph[countGraph].plot(noOfCommunities, excellentLine, label = "No of excellent communities")    
    # graph[countGraph].plot(noOfCommunities, weakLine, label = "No of weak communities")  
    graph[countGraph].plot(noOfCommunities, mixedLine, label = "No of mixed communities") 
    
    # graph[countGraph].plot(noOfCommunities, excellentLine_not_cleaned, label = "No of excellent communities not cleaned data")    
    # graph[countGraph].plot(noOfCommunities, weakLine_not_cleaned, label = "No of weak communities not cleaned data")  
    graph[countGraph].plot(noOfCommunities, mixedLine_not_cleaned, label = "No of mixed communities not cleaned data") 
    
    graph[countGraph].legend(loc='upper left')
    countGraph = countGraph + 1               
plt.show()   

#only show in a certain week
w = 11

if w in [0,1,2,3,4,5]:
    excellent = ex1_excellent.index
    weak = ex1_weak.index
elif w in [6,7,8,9,10,11]:
    excellent = ex2_excellent.index
    weak = ex2_weak.index        

# excellent = exellentPractice[w]
# weak = weakPractice[w]

w = 11
excellentLine = []
weakLine = []
mixedLine = []
mixedProportionLine = []

excellentLine_not_cleaned = []
weakLine_not_cleaned = []
mixedLine_not_cleaned = []
mixedProportionLine_not_cleaned = []

# excellentLine_not_cleaned_super = []
# weakLine_not_cleaned_super = []
# mixedLine_not_cleaned_super = []
# mixedProportionLine_not_cleaned_super = []

noOfCommunities = []
upper = 0.65
lower = 0.35
for i in range(0, 9):     #num_comms-1):
    a1 = graphLearning.identifyCommunitiesType(communityListWeeks[w][i], excellent, weak)
    excellentLine.append(len(a1.loc[a1['excellentRate'] >= upper]))
    weakLine.append(len(a1.loc[a1['excellentRate'] < lower]))
    mixedLine.append(len(a1.loc[(a1['excellentRate'] <upper) & (a1['excellentRate'] >= lower)]))
    mixedProportionLine.append(len(a1.loc[(a1['excellentRate'] <upper) & (a1['excellentRate'] >= lower)])/(len(a1)))
    
    a2 = graphLearning.identifyCommunitiesType(communityListWeeks_cleaned_correlation[w][i], excellent, weak)
    excellentLine_not_cleaned.append(len(a2.loc[a2['excellentRate'] >= upper]))
    weakLine_not_cleaned.append(len(a2.loc[a2['excellentRate'] < lower]))
    mixedLine_not_cleaned.append(len(a2.loc[(a2['excellentRate'] < upper) & (a2['excellentRate'] >= lower)]))
    mixedProportionLine_not_cleaned.append(len(a2.loc[(a2['excellentRate'] <upper) & (a2['excellentRate'] >= lower)])/(len(a2)))
    
    # a3 = graphLearning.identifyCommunitiesType(partitionConvertedList[i], excellent, weak)
    # excellentLine_not_cleaned_super.append(len(a3.loc[a3['excellentRate'] >= upper]))
    # weakLine_not_cleaned_super.append(len(a3.loc[a3['excellentRate'] < lower]))
    # mixedLine_not_cleaned_super.append(len(a3.loc[(a3['excellentRate'] < upper) & (a3['excellentRate'] >= lower)]))
    # mixedProportionLine_not_cleaned_super.append(len(a3.loc[(a3['excellentRate'] <upper) & (a3['excellentRate'] >= lower)])/(len(a3)))
    
    
    noOfCommunities.append(i+2)
    
fig = plt.figure(figsize=(15,10),dpi=120)
ax = fig.add_subplot()
ax.set_xlabel('Number of detected communities in the corresponding community structures', fontsize = 15)
ax.set_ylabel('Mixed community rates', fontsize = 15)
ax.set_title('', fontsize = 20)
ax.set_xticks(np.arange(1, 21,1))
ax.grid()
# graph[countGraph].plot(noOfCommunities, excellentLine, label = "No of excellent communities")    
# graph[countGraph].plot(noOfCommunities, weakLine, label = "No of weak communities")  
# graph[countGraph].plot(noOfCommunities, mixedLine, label = "No of mixed communities") 
ax.plot(noOfCommunities, mixedProportionLine, label = "Original data") 

# graph[countGraph].plot(noOfCommunities, excellentLine_not_cleaned, label = "No of excellent communities not cleaned data")    
# graph[countGraph].plot(noOfCommunities, weakLine_not_cleaned, label = "No of weak communities not cleaned data")  
ax.plot(noOfCommunities, mixedProportionLine_not_cleaned, label = "Cleaned data") 
# ax.plot(noOfCommunities, mixedProportionLine_not_cleaned_super, label = "Louvain") 

ax.legend(loc='upper right',  fontsize = 15)
          
plt.show()    



#compare Louvain cleaned data vs original data
#comparing using Louvain

import graphLearning

edges1 = list(graph_all_weeks_fullyConnected[11].edges(data=True))
edges1[1][2]['weight']
len(graph_all_weeks_fullyConnected[11].nodes)

graphLearning.checkIfUniqueEdges(graph_all_weeks_fullyConnected[9])
A = nx.to_numpy_matrix(graph_all_weeks_fullyConnected[11])
np.savetxt(basePath + "Course1-2020.csv", A, delimiter=",")

list(partition.keys())

import community as community_louvain
partition = community_louvain.best_partition(graph_all_weeks_cleaned_correlation[11].graph,resolution=1)
girvan_newman_cluster = graphLearning.convertGNcommunityToFlattenList(communityListWeeks_cleaned_correlation[11][5], list(partition.keys()))


partition = community_louvain.best_partition(graph_all_weeks[11].graph,resolution=1)
girvan_newman_cluster = graphLearning.convertGNcommunityToFlattenList(communityListWeeks[11][5], list(partition.keys()))


from sklearn import metrics
metrics.homogeneity_score(np.array(list(partition.values())), np.array(list(girvan_newman_cluster.values())))
metrics.completeness_score(np.array(list(partition.values())), np.array(list(girvan_newman_cluster.values())))
metrics.homogeneity_completeness_v_measure(np.array(list(partition.values())), np.array(list(girvan_newman_cluster.values())), beta=1.0)
metrics.adjusted_rand_score(np.array(list(partition.values())), np.array(list(girvan_newman_cluster.values())))

import scipy.stats as stats
x1 = np.array(list(partition.values()))
x2 = np.array(list(girvan_newman_cluster.values()))
stats.kendalltau(x1, x2)

partitionConverted = graphLearning.convertFlattenListToCommunity(partition)
partitionConverted =  [v for k, v in partitionConverted.items()]
partitionConverted = tuple(partitionConverted)
partitionConvertedList = [partitionConverted,partitionConverted,partitionConverted,partitionConverted,partitionConverted,partitionConverted,
                          partitionConverted,partitionConverted,partitionConverted,partitionConverted,partitionConverted,partitionConverted,
                          partitionConverted,partitionConverted,partitionConverted]



graph_all_weeks_fullyConnected = []
corrDistance_detoned = []
for w in range(0,10):
    print('Week ' + str(w) + '...')
    matrix = corrList_original[w]
    # risk_estimators = ml.portfolio_optimization.RiskEstimators()
    # tn_relation = transitionDataMatrixWeeks_directFollow_normalised_transposed[w].shape[0] / transitionDataMatrixWeeks_directFollow_normalised_transposed[w].shape[1]
    # The bandwidth of the KDE kernel
    # kde_bwidth = 0.01
    # detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=True)
    detoned_matrix_byLib = matrix #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)
    distance_matrix = (0.5*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    g = graphLearning.createGraphFromCorrDistance(distance_matrix)
    graph_all_weeks_fullyConnected.append(g)


partition_fullyGraph = community_louvain.best_partition(graph_all_weeks_fullyConnected[11],resolution=0.99)
partition_fullyGraphConverted = graphLearning.convertFlattenListToCommunity(partition_fullyGraph)
partition_fullyGraphConverted =  [v for k, v in partition_fullyGraphConverted.items()]
partition_fullyGraphConverted = tuple(partition_fullyGraphConverted)
partition_fullyGraphConvertedList = [partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,
                          partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,
                          partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted]


partition_not_cleaned = community_louvain.best_partition(graph_all_weeks[11].graph)
partitionConverted_not_cleaned = graphLearning.convertFlattenListToCommunity(partition_not_cleaned)
partitionConverted_not_cleaned = [v for k, v in partitionConverted_not_cleaned.items()]
partitionConverted_not_cleaned = tuple(partitionConverted_not_cleaned)


upper = 0.6
lower = 0.4

a2 = graphLearning.identifyCommunitiesType(partitionConverted_not_cleaned, excellent, weak)
len(a2.loc[(a2['excellentRate'] <upper) & (a2['excellentRate'] >= lower)])/(len(a2))

a3 = graphLearning.identifyCommunitiesType(partitionConverted, excellent, weak)
len(a3.loc[(a3['excellentRate'] <upper) & (a3['excellentRate'] >= lower)])/(len(a3))
   
    
#--------------------------------
exellentPractice = []
weakPractice = []
for w in range(0,12):
    extract = graphLearning.extractActiveStudentsOnActivity(cummulativeExerciseWeeks[w], 'correct', 0.6)
    exellentPractice.append(extract[0])
    weakPractice.append(extract[1])
    
# excellentList = exellentPractice[w]
# weakList = weakPractice[w]
# excellentList = labsheetActiveWeeks[w]
# weakList = labsheetLessActiveWeeks[w]

ex1_excellent = graphLearning.mapNewLabel(ex1_excellent, reLabelIndex)
ex1_weak = graphLearning.mapNewLabel(ex1_weak, reLabelIndex)
ex2_excellent = graphLearning.mapNewLabel(ex2_excellent, reLabelIndex)
ex2_weak = graphLearning.mapNewLabel(ex2_weak, reLabelIndex)

excellentList = ex2_excellent.index
weakList = ex2_weak.index
graphLearning.visualiseMSTGraph(graph_all_weeks[11], excellentList, weakList , reLabelIndex)  



#----------------------------------------------
#Node embedding analysis    
#----------------------------------------------
    
node_embeddings_weeks = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    node2vec = Node2Vec(graph_all_weeks[w].graph, dimensions=64, walk_length=8, num_walks=15, p=0.1, q=1)
    model = node2vec.fit(window=8, min_count=1)    
    nodeList = model.wv.index2word
    node_embeddings = [list(model.wv.get_vector(n)) for n in nodeList] # numpy.ndarray of size number of nodes times embeddings dimensionality        
    nodeList = list(map(int,model.wv.index2word)) #convert string node to int node
    node_embeddings = pd.DataFrame(node_embeddings, index = nodeList)
    node_embeddings = node_embeddings.merge(cummulativeExerciseWeeks[w]['correct'],left_on=node_embeddings.index,
                                            right_on=cummulativeExerciseWeeks[w]['correct'].index).set_index('key_0')
    # scaler = StandardScaler()
    # node_embeddings = pd.DataFrame(scaler.fit_transform(node_embeddings), index=node_embeddings.index)
    node_embeddings_weeks.append(node_embeddings)


node_embeddings_2d_df_weeks = []
for w in range(0,12):
    if w in [0,1,2,3,4,5,6]:
        excellent = ex1_excellent.index
        weak = ex1_weak.index
    elif w in [6,7,8,9,10,11]:
        excellent = ex2_excellent.index
        weak = ex2_weak.index     
    
    # excellent = list(map(str,exellentPractice[w]))
    # weak = list(map(str,weakPractice[w]))

    
    # tsne = TSNE(n_components=2)
    # node_embeddings_2d = tsne.fit_transform(node_embeddings_weeks[w])
    pca = PCA(n_components=2)
    node_embeddings_2d = pca.fit_transform(node_embeddings_weeks[w])
    node_embeddings_2d_df = pd.DataFrame(node_embeddings_2d, index = node_embeddings_weeks[w].index)
    node_embeddings_2d_df['result_exam_1'] = '2'
    
    node_embeddings_2d_df.loc[node_embeddings_2d_df.index.isin(excellent),['result_exam_1']] = 1
    node_embeddings_2d_df.loc[node_embeddings_2d_df.index.isin(weak),['result_exam_1']] = 0
    node_embeddings_2d_df_weeks.append(node_embeddings_2d_df)



fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0
for w in range(0,12):
    ax = fig.add_subplot(3,4,w+1)
    graph.append(ax)
    graph[countGraph].set_xlabel('Dimension 1', fontsize = 15)
    graph[countGraph].set_ylabel('Dimension 2', fontsize = 15)
    graph[countGraph].set_title('Week' + str(w+1), fontsize = 20)
    graph[countGraph].grid()
    graph[countGraph].axhline(y=0, color='k')
    graph[countGraph].axvline(x=0, color='k')
    graph[countGraph].scatter(node_embeddings_2d_df_weeks[w].loc[node_embeddings_2d_df_weeks[w]['result_exam_1'] == 1,0]
                           ,node_embeddings_2d_df_weeks[w].loc[node_embeddings_2d_df_weeks[w]['result_exam_1'] == 1,1]
                           , c = 'r'
                           , s = 30, label='Excellent')
    graph[countGraph].scatter(node_embeddings_2d_df_weeks[w].loc[node_embeddings_2d_df_weeks[w]['result_exam_1'] == 0,0]
                           ,node_embeddings_2d_df_weeks[w].loc[node_embeddings_2d_df_weeks[w]['result_exam_1'] == 0,1]
                           , c = 'b'
                           , s = 30, label='Weak')
    graph[countGraph].legend(loc='upper right')
    countGraph = countGraph + 1
               
plt.show()



#--------------- PREDCTION --------------------------------------------------#
#---------------------------------------------------------------------------------#

#data preparation for prediction
#node_embeddings week

workingWeekExcercise = []
prediction_transition = []

for week in range(0,12):
    print('Predicting for Week ...' + str(week))
    if week in [0,1,2,3,4,5]:
        excellent = ex2_excellent.index
        weak = ex2_weak.index
    elif week in [6,7,8,9,10,11]:
        excellent = ex2_excellent.index
        weak = ex2_weak.index                 
   
    cummulativeResult = []
    
    dataForPrediction = activityDataMatrixWeeks_pageTypeWeek #node_embeddings_weeks
    predictionResult = PredictionResult.predict_proba_all_algorithms_data_ready(dataForPrediction[week], excellent, weak, cummulativeResult)
    prediction_transition.append(predictionResult)

reportArray_transition = []
for w in range(0,12):
    for algorithm in prediction_transition[w]:
        if algorithm != 'data':
            reportArray_transition.append([w,algorithm, 
                                  prediction_transition[w][algorithm][0]['accuracy_score'][0],
                                  prediction_transition[w][algorithm][0]['f1_score'][0],
                                  prediction_transition[w][algorithm][0]['precision_score'][0],
                                  prediction_transition[w][algorithm][0]['recall_score'][0],
                                  prediction_transition[w][algorithm][0]['roc_auc'],
                                  prediction_transition[w][algorithm][4].mean(),
                                  prediction_transition[w][algorithm][7].mean(),
                                  prediction_transition[w][algorithm][8].mean()
                                  ])
        
predictionReport_transition = pd.DataFrame(reportArray_transition,columns=['week','algorithm','accuraccy',
                                                     'f1_score','precision','recall',
                                                     'roc_auc','cv mean','cv_mean_f1','cv_mean_recall']) 

title_transition = 'Activity Data Matrix data'
algorithmList = []
# algorithmList = []
PredictionResult.algorithmComparisonGraph('cv mean',predictionReport_transition,algorithmList, title_transition)

community.greedy_modularity_communities(graph_all_weeks[11].graph)


#---------------------------------------------------------------
#------------ EXPLORATORY ANALYSIS --------------------------
#-----------------------------------------------------------------

eventLog_ca177 = pd.read_csv(basePath + 'ca177_eventLog_nonfixed.csv')
eventLog_ca177 =eventLog_ca177.loc[eventLog_ca177['time:timestamp'] != ' ']
eventLog_ca177['time:timestamp'] = pd.to_datetime(eventLog_ca177['time:timestamp'])
eventLog_ca177 = eventLog_ca177.loc[:, ~eventLog_ca177.columns.str.contains('^Unnamed')]
# materials = eventLog_ca116.loc[:,['org:resource','concept:name','description']]
weeksEventLog = [g for n, g in eventLog_ca177.groupby(pd.Grouper(key='time:timestamp',freq='W'))]

#process for new activity

# lectureList = dataProcessing.getLectureList(eventLog_ca116,['html|py'])
# eventLog_ca116_filtered = eventLog_ca116.loc[eventLog_ca116['description'].str.contains('|'.join(lectureList))]
eventLog_ca177_filtered = eventLog_ca177.loc[eventLog_ca177['description'].str.contains('.html|.py|#|/einstein/')]
eventLog_ca177_filtered['pageName'] = eventLog_ca177_filtered['description'].str.extract(r'([^\/][\S]+.html)', expand=False)
eventLog_ca177_filtered['pageName'] = eventLog_ca177_filtered['pageName'].str.replace('.web','')
eventLog_ca177_filtered = eventLog_ca177_filtered.drop(eventLog_ca177_filtered.loc[eventLog_ca177_filtered['concept:name'].isin(['click-0','click-1','click-2','click-3'])].index)

eventLog_ca177_filtered.loc[eventLog_ca177_filtered['description'].str.contains('correct|incorrect'),'pageName'] = 'Practice'
eventLog_ca177_filtered.loc[eventLog_ca177_filtered['description'].str.contains('^\/einstein\/'),'pageName'] = 'Practice'
eventLog_ca177_filtered['pageName'] = eventLog_ca177_filtered['pageName'].fillna('General')

a = eventLog_ca177_filtered.loc[eventLog_ca177_filtered['pageName'] == 'Practice']

eventLog_ca177_filtered.rename(columns={'concept:instance':'concept:instance1',
                                   'concept:name':'concept:name1',
                                   'case:concept:name' : 'case:concept:name1'},  inplace=True)
eventLog_ca177_filtered['concept:instance'] = eventLog_ca177_filtered['pageName']
eventLog_ca177_filtered['concept:name'] = eventLog_ca177_filtered['pageName']
eventLog_ca177_filtered['date'] = eventLog_ca177_filtered['time:timestamp'].dt.date

eventLog_ca177_filtered['case:concept:name'] = eventLog_ca177_filtered['date'].astype(str) + '-' + eventLog_ca177_filtered['org:resource'].astype(str)



weeksEventLog_filtered = [g for n, g in eventLog_ca177_filtered.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
weeksEventLog_filtered = weeksEventLog_filtered[8:20]

numberOfActivitiesByWeek_df = pd.DataFrame()
for w in range(0,12):
    totalActivitiesByDate  =  [g for n, g in weeksEventLog_filtered[w].groupby(pd.Grouper(key='time:timestamp',freq='D'))]
    weekActivities = []
    print(len(totalActivitiesByDate))
    for d in range(0,len(totalActivitiesByDate)):
        weekActivities.append(len(totalActivitiesByDate[d]))
    if len(totalActivitiesByDate) < 7:
        for i in range(0, -len(totalActivitiesByDate) + 7):
            weekActivities.append(0)
    numberOfActivitiesByWeek_df['week'+ str(w+1)] = weekActivities

numberOfActivitiesByWeek_df['day'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
numberOfActivitiesByWeek_df = numberOfActivitiesByWeek_df.set_index('day')

color = ['rosybrown','lightcoral','indianred','brown','firebrick','maroon','slateblue','darkslateblue','mediumpurple','rebeccapurple','blueviolet','indigo']
lines = numberOfActivitiesByWeek_df.plot.bar(figsize=(15,10), color=color)
lines.grid()

# stats of activities of each students
numberOfActivitiesByWeekByStudent_list = []
for w in range(0,12):
    studentActivitiesByDate = weeksEventLog_filtered[w].groupby([ pd.Grouper(key='org:resource'),pd.Grouper(key='time:timestamp',freq='D')]).count()
    studentActivitiesByDate = studentActivitiesByDate.reset_index()
    studentActivitiesByDate['Day'] = studentActivitiesByDate['time:timestamp'].dt.day_name()
    studentActivitiesByDate = studentActivitiesByDate.loc[:,['Day','org:resource','case:concept:name1']]
    numberOfActivitiesByWeekByStudent_list.append(studentActivitiesByDate)

numberOfActivitiesByWeekByStudent_df = pd.concat(numberOfActivitiesByWeekByStudent_list)


numberOfActivitiesByWeekByStudent_df = numberOfActivitiesByWeekByStudent_df.groupby([pd.Grouper('org:resource'),pd.Grouper('Day')]).sum()
numberOfActivitiesByWeekByStudent_df = numberOfActivitiesByWeekByStudent_df.reset_index()

import seaborn as sns

fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(x="Day", y="case:concept:name1", ax=ax, data=numberOfActivitiesByWeekByStudent_df)
ax.grid()


numberOfActivitiesByWeekByStudent_df.boxplot(by='Day', figsize=(15,10), layout=(1,1))

# stats of activities of each students by activity type


weeksEventLog_filtered_pageType = []
for w in range(0,12):
    tmp = weeksEventLog_filtered[w].merge(materialAccessedByWeek.loc[:,['pageType','ofWeek']], left_on=weeksEventLog_filtered[w].pageName, 
                                    right_on=materialAccessedByWeek.loc[:,['pageType']].index)    
    tmp.loc[tmp['pageName'] == 'Practice',['ofWeek']] = w
    tmp['pageTypeWeek'] = tmp['pageType'] + '_' + tmp['ofWeek'].astype(str)
    tmp['concept:name'] = tmp['pageTypeWeek'] + '*' + tmp['concept:instance1']
    tmp['concept:instance'] = tmp['pageTypeWeek'] + '*' + tmp['concept:instance1']
    weeksEventLog_filtered_pageType.append(tmp)
    
    
pageName = 'Labsheet'   
numberOfActivitiesByWeekByStudent_list = []
for w in range(0,12):
    tmp = weeksEventLog_filtered_pageType[w].loc[weeksEventLog_filtered_pageType[w]['pageType'] == pageName,:]
    studentActivitiesByDate = tmp.groupby([ pd.Grouper(key='org:resource'),pd.Grouper(key='time:timestamp',freq='D')]).count()
    studentActivitiesByDate = studentActivitiesByDate.reset_index()
    studentActivitiesByDate['Day'] = studentActivitiesByDate['time:timestamp'].dt.day_name()
    studentActivitiesByDate = studentActivitiesByDate.loc[:,['Day','org:resource','case:concept:name1']]
    numberOfActivitiesByWeekByStudent_list.append(studentActivitiesByDate)

numberOfActivitiesByWeekByStudent_df = pd.concat(numberOfActivitiesByWeekByStudent_list)


numberOfActivitiesByWeekByStudent_df = numberOfActivitiesByWeekByStudent_df.groupby([pd.Grouper('org:resource'),pd.Grouper('Day')]).sum()
numberOfActivitiesByWeekByStudent_df = numberOfActivitiesByWeekByStudent_df.reset_index()

import seaborn as sns

fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(x="Day", y="case:concept:name1", ax=ax, data=numberOfActivitiesByWeekByStudent_df)
ax.grid()

numberOfActivitiesByWeekByStudent_df['result'] = 'NA'
numberOfActivitiesByWeekByStudent_df.loc[numberOfActivitiesByWeekByStudent_df['org:resource'].isin(ex2_excellent.index),['result']] = 'Higher Performing'
numberOfActivitiesByWeekByStudent_df.loc[numberOfActivitiesByWeekByStudent_df['org:resource'].isin(ex2_weak.index),['result']] = 'Lower Performing'
numberOfActivitiesByWeekByStudent_df = numberOfActivitiesByWeekByStudent_df.drop(numberOfActivitiesByWeekByStudent_df.loc[numberOfActivitiesByWeekByStudent_df['result'].isin(['NA'])].index)
numberOfActivitiesByWeekByStudent_df['Ordercustom'] = 0
numberOfActivitiesByWeekByStudent_df.loc[numberOfActivitiesByWeekByStudent_df['Day'] == 'Monday',['Ordercustom']] = 0
numberOfActivitiesByWeekByStudent_df.loc[numberOfActivitiesByWeekByStudent_df['Day'] == 'Tuesday',['Ordercustom']] = 1
numberOfActivitiesByWeekByStudent_df.loc[numberOfActivitiesByWeekByStudent_df['Day'] == 'Wednesday',['Ordercustom']] = 2
numberOfActivitiesByWeekByStudent_df.loc[numberOfActivitiesByWeekByStudent_df['Day'] == 'Thursday',['Ordercustom']] = 3
numberOfActivitiesByWeekByStudent_df.loc[numberOfActivitiesByWeekByStudent_df['Day'] == 'Friday',['Ordercustom']] = 4
numberOfActivitiesByWeekByStudent_df.loc[numberOfActivitiesByWeekByStudent_df['Day'] == 'Saturday',['Ordercustom']] = 5
numberOfActivitiesByWeekByStudent_df.loc[numberOfActivitiesByWeekByStudent_df['Day'] == 'Sunday',['Ordercustom']] = 6
numberOfActivitiesByWeekByStudent_df = numberOfActivitiesByWeekByStudent_df.sort_values(['org:resource','Ordercustom'])
#numberOfActivitiesByWeekByStudent_df = numberOfActivitiesByWeekByStudent_df.drop(['Ordercustom'], axis = 1)

ax = numberOfActivitiesByWeekByStudent_df.boxplot(by=['Ordercustom','result'], figsize=(15,10),  vert=0, showmeans=True)
ax.set_title(pageName)
ax.set_yticklabels(['Mon - Higher','Mon - Lower', 'Tue - Higher', 'Tue - Lower', 'Wed - Higher', 'Wed - Lower', 'Thu - Higher', 'Thu - Lower', 'Fri - Higher', 'Fri - Lower', 'Sat - Higher', 'Sat - Lower', 'Sun - Higher', 'Sun - Lower'])

# data by page type only
numberOfActivitiesByStudentByPageType_list = []
for w in range(0,12):
    tmp = weeksEventLog_filtered_pageType[w]
    studentActivitiesBypageType = tmp.groupby([ pd.Grouper(key='org:resource'),pd.Grouper(key='pageType')]).count()
    studentActivitiesBypageType = studentActivitiesBypageType.reset_index()
    studentActivitiesBypageType = studentActivitiesBypageType.loc[:,['org:resource','pageType','case:concept:name1']]
    numberOfActivitiesByStudentByPageType_list.append(studentActivitiesBypageType)

numberOfActivitiesByStudentByPageType_df = pd.concat(numberOfActivitiesByStudentByPageType_list)
numberOfActivitiesByStudentByPageType_df = numberOfActivitiesByStudentByPageType_df.groupby([ pd.Grouper(key='org:resource'),pd.Grouper(key='pageType')]).sum()
numberOfActivitiesByStudentByPageType_df = numberOfActivitiesByStudentByPageType_df.reset_index()
numberOfActivitiesByStudentByPageType_df['result'] = 'NA'
numberOfActivitiesByStudentByPageType_df.loc[numberOfActivitiesByStudentByPageType_df['org:resource'].isin(ex2_excellent.index),['result']] = 'Higher Performing'
numberOfActivitiesByStudentByPageType_df.loc[numberOfActivitiesByStudentByPageType_df['org:resource'].isin(ex2_weak.index),['result']] = 'Lower Performing'
numberOfActivitiesByStudentByPageType_df = numberOfActivitiesByStudentByPageType_df.drop(numberOfActivitiesByStudentByPageType_df.loc[numberOfActivitiesByStudentByPageType_df['result'].isin(['NA'])].index)

ax = numberOfActivitiesByStudentByPageType_df.boxplot(by=['pageType','result'], figsize=(15,15),  vert=0, showmeans=True, fontsize=20)
ax.set_xlim(0, 5000)
# ax.set_title('Student activities by course learning material type')
# ax.set_yticklabels(['Mon - Higher','Mon - Lower', 'Tue - Higher', 'Tue - Lower', 'Wed - Higher', 'Wed - Lower', 'Thu - Higher', 'Thu - Lower', 'Fri - Higher', 'Fri - Lower', 'Sat - Higher', 'Sat - Lower', 'Sun - Higher', 'Sun - Lower'])


#community compare
excellentCommunity = numberOfActivitiesByStudentByPageType_df.loc[numberOfActivitiesByStudentByPageType_df['org:resource'].isin(aw10[5][5][5].index)]
excellentCommunity['CommunityType'] = 'Higher Performing'
badCommunity = numberOfActivitiesByStudentByPageType_df.loc[numberOfActivitiesByStudentByPageType_df['org:resource'].isin(aw10[5][5][1].index)]
badCommunity['CommunityType'] = 'Lower Performing'
ax  = pd.concat([excellentCommunity, badCommunity]).boxplot(by=['pageType','CommunityType'], figsize=(15,15),  vert=0, showmeans=True, fontsize=20)
ax.set_xlim(0, 5000)

