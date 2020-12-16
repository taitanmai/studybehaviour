import pandas as pd
import numpy as np
import scipy as sp
import time
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
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.stats import kurtosis
import snap
# sns.set()
# sns.set_style("whitegrid", {"axes.facecolor": ".9"})
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

eventLog_ca116 = pd.read_csv('ca116_eventLog_nonfixed.csv')
eventLog_ca116 = eventLog_ca116.drop([1160345])
eventLog_ca116['time:timestamp'] = pd.to_datetime(eventLog_ca116['time:timestamp'])
eventLog_ca116 = eventLog_ca116.loc[:, ~eventLog_ca116.columns.str.contains('^Unnamed')]

lectureList = dataProcessing.getLectureList(eventLog_ca116,['html|py'])
eventLog_ca116_filtered = eventLog_ca116.loc[eventLog_ca116['description'].str.contains('|'.join(lectureList))]
# ex1_personal_log_1 = dataProcessing.addConceptPageToLog(ex1_personal_log_1)

# eventLog_ca116_filtered = eventLog_ca116_filtered.drop(eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('http|report|ex|dashboard|graphs.html')].index)
eventLog_ca116_filtered = eventLog_ca116_filtered.drop(eventLog_ca116_filtered.loc[eventLog_ca116_filtered['concept:name'].isin(['click-0','click-1','click-2'])].index)
eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('.html|.web'),'pageType'] = 'Read_Lecture_Note' 
eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('correct|incorrect'),'pageType'] = 'Exercise'
eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('labsheet|instructions'),'pageType'] = 'Read_Labsheet'
eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('solution'),'pageType'] = 'Check_solution'
eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('http|report|dashboard|graphs.html'),'pageType'] = 'Admin_page'
eventLog_ca116_filtered['pageType'] = eventLog_ca116_filtered['pageType'] .fillna('Other')
eventLog_ca116_filtered = eventLog_ca116_filtered.drop(eventLog_ca116_filtered.loc[eventLog_ca116_filtered['pageType'] == 'Other'].index)


eventLog_ca116_filtered['pageType'].unique()

eventLog_ca116_filtered.rename(columns={'concept:instance':'concept:instance1',
                                   'concept:name':'concept:name1',
                                   'case:concept:name' : 'case:concept:name1'}, 
                  inplace=True)
eventLog_ca116_filtered['concept:instance'] = eventLog_ca116_filtered['pageType']
eventLog_ca116_filtered['concept:name'] = eventLog_ca116_filtered['pageType']
eventLog_ca116_filtered['date'] = eventLog_ca116_filtered['time:timestamp'].dt.date

eventLog_ca116_filtered['case:concept:name'] = eventLog_ca116_filtered['date'].astype(str) + '-' + eventLog_ca116_filtered['org:resource'].astype(str)
eventLog_ca116_filtered['concept:name'] = eventLog_ca116_filtered['pageType'] + '*' + eventLog_ca116_filtered['concept:name1']
eventLog_ca116_filtered['concept:instance'] = eventLog_ca116_filtered['pageType'] + '*' + eventLog_ca116_filtered['concept:instance1']

weeksEventLog_filtered = [g for n, g in eventLog_ca116_filtered.groupby(pd.Grouper(key='time:timestamp',freq='W'))]

workingWeekLog = []
transitionDataMatrixWeeks = []
for week in range(1,13):
    print('Week: ' + str(week) + '...')
    workingWeekLog.append(weeksEventLog_filtered[week])
    Log = pd.concat(workingWeekLog)
    tempTransition = FCAMiner.transitionDataMatrixConstruct_for_prediction(Log).fillna(0)
    tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()            
    transitionDataMatrixWeeks.append(tempTransition)


activityList = []
pageList = ['Read_Lecture_Note','Exercise','Read_Labsheet','Check_solution','Admin_page']
actionList = ['load','scroll','blur','focus','unload','upload','hashchange','selection']
for p in pageList:
    for a in actionList:
        activityList.append(p+'*'+a)

activityCodeList = libRMT.assignNodeNumber(activityList)

graphCharacteristicsWeeks = []
for week in range(0,12):
    print('Week ' + str(week) + '...')
    resultWeekValues = []
    count = 0
    for index, row in transitionDataMatrixWeeks[week].iterrows():
        resultTemp = libRMT.girvin_neuman_profile_extract(row,activityCodeList,index,week)
        resultWeekValues.append(resultTemp)
        count = count + 1
        percentage = (count*100)/float(len(transitionDataMatrixWeeks[week]))
        print('{:.2f}%'.format(percentage))
    graphCharacteristicsWeeks.append(pd.DataFrame(resultWeekValues,columns = ['user','modularity','noOfCluster','CmtyV']))
    

#get week and excellent list
dataUpload = pd.read_csv('ca116_uploads.csv')
dataUpload['date'] = pd.to_datetime(dataUpload.date)

exUpload = dataUpload.loc[dataUpload['task'].str.match('ex')]

ex1 = exUpload.loc[exUpload['task'].str.match('ex1')]
ex1 = ex1.sort_values(by=['user','task'])
ex1 = ex1.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
ex2 = exUpload.loc[exUpload['task'].str.match('ex2')]
ex2 = ex2.sort_values(by=['user','task'])
ex2 = ex2.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
ex3 = exUpload.loc[exUpload['task'].str.match('ex3')]
ex3 = ex3.sort_values(by=['user','task'])
ex3 = ex3.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()

assessment1A = dataProcessing.assessmentConstruction(ex1,4)
assessment1A['adjustedPerformance'] = (assessment1A['perCorrect'] + assessment1A['perPassed'])/2
assessment2A = dataProcessing.assessmentConstruction(ex2,4)
assessment2A['adjustedPerformance'] = (assessment2A['perCorrect'] + assessment2A['perPassed'])/2
assessment3A = dataProcessing.assessmentConstruction(ex3,4)
assessment3A['adjustedPerformance'] = (assessment3A['perCorrect'] + assessment3A['perPassed'])/2

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
assessment3A.rename(columns={'correct':'correct3A',
                          'perCorrect':'perCorrect3A',
                            'failed':'failed3A',
                            'passed':'passed3A',
                            'perPassed':'perPassed3A',
                            'testSubmitted':'testSubmitted3A',
                            'adjustedPerformance':'adjustedPerformance3A'}, 
                  inplace=True)
assessment1A = assessment1A.set_index(['user'])
assessment2A = assessment2A.set_index(['user'])
assessment3A = assessment3A.set_index(['user'])

assessment = pd.concat([assessment1A,assessment2A,assessment3A], axis=1)
assessment = assessment.fillna(0)

assessment['grade'] = (assessment['perCorrect1A']+assessment['perCorrect2A']+assessment['perCorrect3A'])/3
assessment['perPassed'] = (assessment['passed1A'] + assessment['passed2A'] + assessment['passed3A'])/(assessment['passed1A'] + assessment['passed2A'] + assessment['passed3A'] 
                        + assessment['failed1A']+ assessment['failed2A']+ assessment['failed3A'])

for week in range(0,12):
    graphCharacteristicsWeeks[week] = graphCharacteristicsWeeks[week].set_index(['user'])
    if week in [0,1,2,3]:
        graphCharacteristicsWeeks[week] = graphCharacteristicsWeeks[week].merge(assessment1A.loc[:,['perCorrect1A']],
                                                                                left_on=graphCharacteristicsWeeks[week].index,
                                                                                right_on=assessment1A.loc[:,['perCorrect1A']].index)
        graphCharacteristicsWeeks[week].rename(columns={'perCorrect1A':'grade'},inplace=True)
    elif week in [4,5,6,7]:
        graphCharacteristicsWeeks[week] = graphCharacteristicsWeeks[week].merge(assessment2A.loc[:,['perCorrect2A']],
                                                                                left_on=graphCharacteristicsWeeks[week].index,
                                                                            right_on=assessment2A.loc[:,['perCorrect2A']].index)
        graphCharacteristicsWeeks[week].rename(columns={'perCorrect2A':'grade'},inplace=True)
    else:
        graphCharacteristicsWeeks[week] = graphCharacteristicsWeeks[week].merge(assessment3A.loc[:,['perCorrect3A']],
                                                                                left_on=graphCharacteristicsWeeks[week].index,
                                                                            right_on=assessment3A.loc[:,['perCorrect3A']].index)
        graphCharacteristicsWeeks[week].rename(columns={'perCorrect3A':'grade'},inplace=True)
       
            
        
        
fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0
for w in range(0,12):
    ax = fig.add_subplot(3,4,w+1)
    graph.append(ax)
    graph[countGraph].set_xlabel('Grade', fontsize = 15)
    graph[countGraph].set_ylabel('noOfCluster', fontsize = 15)
    graph[countGraph].set_title('Week' + str(w+1), fontsize = 20)
    graph[countGraph].grid()
    graph[countGraph].axhline(y=0, color='k')
    graph[countGraph].axvline(x=0, color='k')
    graph[countGraph].scatter(graphCharacteristicsWeeks[w]['grade']
                           ,graphCharacteristicsWeeks[w]['modularity']
                           , c = 'r'
                           , s = 30, label='Successful')
    graph[countGraph].legend(loc='upper right')
    countGraph = countGraph + 1
               
plt.show()      
        
libRMT.visualiseGraph(transitionDataMatrixWeeks[10].loc['u-13fec06c93690caf5612445fac3691864386423d'],activityCodeList,'Graph2','Graph2',True)
        
a = graphCharacteristicsWeeks[10].loc[graphCharacteristicsWeeks[10].user == 'u-13fec06c93690caf5612445fac3691864386423d',['CmtyV']]        
