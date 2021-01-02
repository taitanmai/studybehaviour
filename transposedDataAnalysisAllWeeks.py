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

#event log processed to get the frequency of activity
activityList = ['load','scroll','focus','blur','unload','hashchange','selection']
basePath = 'G:\\Dataset\\PhD\\'

#extract event log 
eventLog_ca116 = pd.read_csv(basePath + 'ca116_eventLog_nonfixed.csv')
eventLog_ca116 = eventLog_ca116.drop([1160345])
eventLog_ca116['time:timestamp'] = pd.to_datetime(eventLog_ca116['time:timestamp'])
eventLog_ca116 = eventLog_ca116.loc[:, ~eventLog_ca116.columns.str.contains('^Unnamed')]
# materials = eventLog_ca116.loc[:,['org:resource','concept:name','description']]
weeksEventLog = [g for n, g in eventLog_ca116.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
a = weeksEventLog[1]
#process for new activity

# lectureList = dataProcessing.getLectureList(eventLog_ca116,['html|py'])
# eventLog_ca116_filtered = eventLog_ca116.loc[eventLog_ca116['description'].str.contains('|'.join(lectureList))]
eventLog_ca116_filtered = eventLog_ca116.loc[eventLog_ca116['description'].str.contains('.html|.py|einstein|#')]

# eventLog_ca116_filtered = eventLog_ca116_filtered.drop(eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('http|report|ex|dashboard|graphs.html')].index)
eventLog_ca116_filtered = eventLog_ca116_filtered.drop(eventLog_ca116_filtered.loc[eventLog_ca116_filtered['concept:name'].isin(['click-0','click-1','click-2','click-3'])].index)
eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('.html|.web'),'pageType'] = 'Lecture' 
eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('correct|incorrect'),'pageType'] = 'Practice'
eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('labsheet|instructions'),'pageType'] = 'Labsheet'
eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('solution'),'pageType'] = 'Practice'
eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('http|report|dashboard|graphs|log.html'),'pageType'] = 'General'
eventLog_ca116_filtered['pageType'] = eventLog_ca116_filtered['pageType'] .fillna('Other')


a = eventLog_ca116_filtered.loc[eventLog_ca116_filtered['pageType'] == 'Lecture']

eventLog_ca116_filtered['pageType'].unique()
eventLog_ca116_filtered['pageType'].value_counts()
# eventLog_ca116_filtered = eventLog_ca116_filtered.drop(eventLog_ca116_filtered.loc[eventLog_ca116_filtered['pageType'] == 'Other'].index)


eventLog_ca116_filtered['concept:name'].unique()

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


#convert to transition data matrix week

workingWeekLog = []
transitionDataMatrixWeeks = []
full_transitionDataMatrixWeeks = []
for week in range(1,13):
    print('Week: ' + str(week) + '...')
    workingWeekLog.append(weeksEventLog_filtered[week])
    Log = pd.concat(workingWeekLog) 
    tempTransition = FCAMiner.transitionDataMatrixConstruct_directFollow(Log,[]).fillna(0)
    full_transitionDataMatrixWeeks.append(tempTransition)   
    tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()         
    transitionDataMatrixWeeks.append(tempTransition)
    
for w in range(0,12):
    transitionDataMatrixWeeks[w] = transitionDataMatrixWeeks[w].loc[:, (transitionDataMatrixWeeks[w] != 0).any(axis=0)]
    
for w in range(0,12):
    transitionDataMatrixWeeks[w].to_csv('transitionMatrixStorage_new/transitionDataMatrixWeeks_distance_accumulated_w'+str(w)+'.csv',index=True)


transitionDataMatrixWeeks_not_accumulated = []
full_transitionDataMatrixWeeks_not_accumulated = []
for week in range(1,13):
    print('Week: ' + str(week) + '...')
    Log = weeksEventLog_filtered[week]
    tempTransition = FCAMiner.transitionDataMatrixConstruct_directFollow(Log,[]).fillna(0)
    full_transitionDataMatrixWeeks_not_accumulated.append(tempTransition)   
    tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()         
    transitionDataMatrixWeeks_not_accumulated.append(tempTransition)
    
for w in range(0,12):
    transitionDataMatrixWeeks_not_accumulated[w] = transitionDataMatrixWeeks_not_accumulated[w].loc[:, (transitionDataMatrixWeeks_not_accumulated[w] != 0).any(axis=0)]
   
#import data


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

ex1_excellent = assessment1A.loc[(assessment1A['perCorrect1A'] <= 1) & (assessment1A['perCorrect1A'] >= 0.4)]
ex1_weak = assessment1A.loc[(assessment1A['perCorrect1A'] >= 0) & (assessment1A['perCorrect1A'] < 0.4)]

ex2_excellent = assessment2A.loc[(assessment2A['perCorrect2A'] <= 1)&(assessment2A['perCorrect2A'] >= 0.4)]
ex2_weak = assessment2A.loc[(assessment2A['perCorrect2A'] >= 0) & (assessment2A['perCorrect2A'] < 0.4)]

ex3_excellent = assessment3A.loc[(assessment3A['perCorrect3A'] <= 1)&(assessment3A['perCorrect3A'] >= 0.4)]
ex3_weak = assessment3A.loc[(assessment3A['perCorrect3A'] >= 0) & (assessment3A['perCorrect3A'] < 0.4)]

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
    cummulativeResult = graphLearning.mapNewLabel(cummulativeResult, reLabelIndex)
    cummulativeExerciseWeeks.append(cummulativeResult)

a = cummulativeExerciseWeeks[11].corr()

  
transitionDataMatrixWeeks = []
for w in range(0,12):
    temp = pd.read_csv('transitionMatrixStorage_new/transitionDataMatrixWeeks_distance_accumulated_w' + str(w) + '.csv', index_col=0)
    if w in [0,1,2,3]:
        studentList = assessment1A.index
    elif w in [4,5,6,7]:
        studentList = assessment2A.index     
    else:
        studentList = assessment3A.index
    temp = temp.loc[temp.index.isin(studentList)]
    temp = graphLearning.mapNewLabel(temp, reLabelIndex)
    # if w == 1:
    #     temp = temp.drop([8])
    transitionDataMatrixWeeks.append(temp) 


#re label for easier reading

ex1_excellent = graphLearning.mapNewLabel(ex1_excellent, reLabelIndex)
ex1_weak = graphLearning.mapNewLabel(ex1_weak, reLabelIndex)
ex2_excellent = graphLearning.mapNewLabel(ex2_excellent, reLabelIndex)
ex2_weak = graphLearning.mapNewLabel(ex2_weak, reLabelIndex)
ex3_excellent = graphLearning.mapNewLabel(ex3_excellent, reLabelIndex)
ex3_weak = graphLearning.mapNewLabel(ex3_weak, reLabelIndex)

assessment_label = assessment.copy()
assessment_label = graphLearning.mapNewLabel(assessment, reLabelIndex)

assessment_label1A = assessment1A.copy()
assessment_label1A = graphLearning.mapNewLabel(assessment1A, reLabelIndex)
assessment_label2A = assessment2A.copy()
assessment_label2A = graphLearning.mapNewLabel(assessment2A, reLabelIndex)
assessment_label3A = assessment3A.copy()
assessment_label3A = graphLearning.mapNewLabel(assessment3A, reLabelIndex)

#activity data matrix construction
workingWeekLog = []
activityDataMatrixWeeks = []
for w in range(1,13):
    print('Week ' + str(w) + '...')
    workingWeekLog.append(weeksEventLog_filtered[w])
    LogPageactivityCountByUser =  pd.concat(workingWeekLog) #weeksEventLog_filtered[w]
    LogPageactivityCountByUser = FCAMiner.activityDataMatrixContruct(LogPageactivityCountByUser,'pageType')
    LogPageactivityCountByUser = LogPageactivityCountByUser.fillna(0)
    # LogPageactivityCountByUser = FCAMiner.activityDataMatrixPercentage(LogPageactivityCountByUser)
    LogPageactivityCountByUser = graphLearning.mapNewLabel(LogPageactivityCountByUser,reLabelIndex)
    activityDataMatrixWeeks.append(LogPageactivityCountByUser)
    
for w in range(0,12):
    temp = activityDataMatrixWeeks[w].merge(cummulativeExerciseWeeks[w].loc[:,:], left_on=activityDataMatrixWeeks[w].index, right_on=cummulativeExerciseWeeks[w].index)
    temp = temp.set_index(['key_0'])
    if w in [0,1,2,3]:
        studentResult = assessment_label1A
    elif w in [4,5,6,7]:
        studentResult = assessment_label2A
    else:
        studentResult = assessment_label3A
    temp = temp.merge(studentResult, left_on=temp.index, right_on=studentResult.index)
    temp = temp.set_index(['key_0'])
    if -1 in temp.index:
        temp = temp.drop([-1])
    activityDataMatrixWeeks[w] = temp

a =  activityDataMatrixWeeks[11].corr()
b = a['correct'].sort_values()
    
#get active and less active data 
practiceActiveWeeks = []
practiceLessActiveWeeks = []
for w in range(0,12):
    extract = graphLearning.extractActiveStudentsOnActivity(activityDataMatrixWeeks[w], 'Practice', 0.5)
    practiceActiveWeeks.append(extract[0])
    practiceLessActiveWeeks.append(extract[1])
      
lectureActiveWeeks = []
lectureLessActiveWeeks = []
for w in range(0,12):
    extract = graphLearning.extractActiveStudentsOnActivity(activityDataMatrixWeeks[w], 'Read_Lecture_Note', 0.5)
    lectureActiveWeeks.append(extract[0])
    lectureLessActiveWeeks.append(extract[1])

adminActiveWeeks = []
adminLessActiveWeeks = []
for w in range(0,12):
    extract = graphLearning.extractActiveStudentsOnActivity(activityDataMatrixWeeks[w], 'Admin', 0.5)
    adminActiveWeeks.append(extract[0])
    adminLessActiveWeeks.append(extract[1])

checkSolutionActiveWeeks = []
checkSolutionLessActiveWeeks = []
for w in range(0,12):
    extract = graphLearning.extractActiveStudentsOnActivity(activityDataMatrixWeeks[w], 'Check_solution', 0.5)
    checkSolutionActiveWeeks.append(extract[0])
    checkSolutionLessActiveWeeks.append(extract[1])
 
exellentPractice = []
weakPractice = []
for w in range(0,12):
    extract = graphLearning.extractActiveStudentsOnActivity(cummulativeExerciseWeeks[w], 'correct', 0.6)
    exellentPractice.append(extract[0])
    weakPractice.append(extract[1])

    
activityDataMatrixWeeks[11].quantile(0.7)['Read_Lecture_Note']
activityDataMatrixWeeks[11].describe()
a = pd.merge(activityDataMatrixWeeks[11],assessment_label.loc[:,['perCorrect3A']],left_on=activityDataMatrixWeeks[11].index, right_on=assessment_label.loc[:,['perCorrect3A']].index)
a = a.set_index(['key_0'])

# a = activityDataMatrixWeeks[11]
cmap = sns.cm.rocket_r

sns.heatmap(a.corr(), annot=True, center=0.8, yticklabels=True, xticklabels=True, cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()


#transpose transition data matrix
for w in range(0,12):
    transitionDataMatrixWeeks[w] = transitionDataMatrixWeeks[w].T

transitionDataMatrixWeeks_directFollow_normalised = []    
for w in range(0,12):
    transitionDataMatrixWeeks_directFollow_normalised.append(dataProcessing.normaliseData(transitionDataMatrixWeeks[w]))


# correlation processing    
corrList = []
corrDistanceList = []
for w in range(0,12):
    corrTemp = transitionDataMatrixWeeks[w].corr()
    corrList.append(corrTemp)
    corrDistance = (0.5*(1 - corrTemp)).apply(np.sqrt)
    corrDistanceList.append(corrDistance)

# cmap = sns.cm.rocket_r

# sns.heatmap(corrList[3], annot=False, center=0.8, yticklabels=False, xticklabels=False, cmap='coolwarm')
# plt.title('Correlation heatmap week ' + '3')
# plt.show()

# cmap = sns.cm.rocket_r

# sns.heatmap(corrDistanceList[3], annot=False, center=0, yticklabels=False, xticklabels=False, cmap='coolwarm')
# plt.title('Distance correlation heatmap week ' + '3')
# plt.show()


#correlation processing    
corrList_dataNormalised = []
# corrDistanceList_dataNormalised = []
for w in range(0,12):
    corrTemp = transitionDataMatrixWeeks_directFollow_normalised[w].corr()
    corrList_dataNormalised.append(corrTemp)
    # corrDistance = (0.5*(1 - corrTemp)).apply(np.sqrt)
    # corrDistanceList_dataNormalised.append(corrDistance)



graph_all_weeks = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList[w]
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    tn_relation = transitionDataMatrixWeeks_directFollow_normalised[w].shape[0] / transitionDataMatrixWeeks_directFollow_normalised[w].shape[1]
    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01
    # Finding the Вe-noised Сovariance matrix
    # denoised_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth)
    # denoised_matrix_byLib = pd.DataFrame(denoised_matrix_byLib, index=matrix.index, columns=matrix.columns) denoise_method='target_shrink',
    
    detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth,  detone=True)
    # detoned_matrix_byLib = matrix #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)
    distance_matrix = (2*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    graphBuild = MST(distance_matrix, 'distance')
    # graphBuild = nx.from_pandas_adjacency(distance_matrix)   
    graph_all_weeks.append(graphBuild)
    
graph_all_weeks_not_cleaned = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList[w]
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    tn_relation = transitionDataMatrixWeeks_directFollow_normalised[w].shape[0] / transitionDataMatrixWeeks_directFollow_normalised[w].shape[1]
    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01
    # Finding the Вe-noised Сovariance matrix
    # denoised_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth)
    # denoised_matrix_byLib = pd.DataFrame(denoised_matrix_byLib, index=matrix.index, columns=matrix.columns)
    
    # detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, denoise_method='target_shrink', detone=True)
    detoned_matrix_byLib = matrix #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)
    distance_matrix = (2*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    graphBuild = MST(distance_matrix, 'distance')
    # graphBuild = nx.from_pandas_adjacency(distance_matrix)   
    graph_all_weeks_not_cleaned.append(graphBuild)


w = 11    
# excellentList = ex3_excellent.index
# weakList = ex3_weak.index

excellentList = exellentPractice[w]
weakList = weakPractice[w]
# excellentList = labsheetActiveWeeks[w]
# weakList = labsheetLessActiveWeeks[w]

graphLearning.visualiseMSTGraph(graph_all_weeks[w], excellentList, weakList , reLabelIndex)    

#----------------------------------------------
#Communnity Detection analysis    
#----------------------------------------------   



#community detection
communityListWeeks = []
for w in range(0,12):
    print('Week ' + str(w) + '...')      
    num_comms = len(graph_all_weeks[w].graph._node)
    communityListWeeks.append(graphLearning.community_dection_graph(graph_all_weeks[w], most_valuable_edge=graphLearning.most_central_edge, num_comms=num_comms))

communityListWeeks_not_cleaned = []
for w in range(0,12):
    print('Week ' + str(w) + '...')      
    num_comms = len(graph_all_weeks_not_cleaned[w].graph._node)
    communityListWeeks_not_cleaned.append(graphLearning.community_dection_graph(graph_all_weeks_not_cleaned[w], most_valuable_edge=graphLearning.most_central_edge, num_comms=num_comms))

fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0    
for w in range(0,12): 
    # if w in [0,1,2,3]:
    #     excellent = ex1_excellent.index
    #     weak = ex1_weak.index
    # elif w in [4,5,6,7]:
    #     excellent = ex2_excellent.index
    #     weak = ex2_weak.index        
    # else:
    #     excellent = ex3_excellent.index
    #     weak = ex3_weak.index 
    excellent = exellentPractice[w]
    weak = weakPractice[w]
    
    excellentLine = []
    weakLine = []
    mixedLine = []
    
    excellentLine_not_cleaned = []
    weakLine_not_cleaned = []
    mixedLine_not_cleaned = []
    noOfCommunities = []
    upper = 0.70
    lower = 0.30
    for i in range(0,num_comms-1):
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

#community moving flow analysis  

classifyStudentGroupOverCommunitiesWeeks = []
for w in range(0,12):
    excellent = practiceActiveWeeks[w]
    weak = practiceLessActiveWeeks[w]
    
    studentList = transitionDataMatrixWeeks_directFollow_normalised[w].columns
    noOfCommunities = 80
    temp1 = graphLearning.labelCommunity(studentList, communityListWeeks[w], noOfCommunities, excellent, weak, transitionDataMatrixWeeks_directFollow_normalised[w])
    classifyStudentGroupOverCommunitiesWeeks.append(temp1)
    
   
classifyStudentGroupOverCommunitiesWeeks_lecture = []
for w in range(0,12):
    excellent = lectureActiveWeeks[w]
    weak = lectureLessActiveWeeks[w]
    
    studentList = transitionDataMatrixWeeks_directFollow_normalised[w].columns
    noOfCommunities = 80
    temp1 = graphLearning.labelCommunity(studentList, communityListWeeks[w], noOfCommunities, excellent, weak)
    classifyStudentGroupOverCommunitiesWeeks_lecture.append(temp1)

classifyStudentGroupOverCommunitiesWeeks_admin = []
for w in range(0,12):
    excellent = adminActiveWeeks[w]
    weak = adminLessActiveWeeks[w]
    
    studentList = transitionDataMatrixWeeks_directFollow_normalised[w].columns
    noOfCommunities = 80
    temp1 = graphLearning.labelCommunity(studentList, communityListWeeks[w], noOfCommunities, excellent, weak)
    classifyStudentGroupOverCommunitiesWeeks_admin.append(temp1)
    
classifyStudentGroupOverCommunitiesWeeks_check = []
for w in range(0,12):
    excellent = checkSolutionActiveWeeks[w]
    weak = checkSolutionLessActiveWeeks[w]
    
    studentList = transitionDataMatrixWeeks_directFollow_normalised[w].columns
    noOfCommunities = 80
    temp1 = graphLearning.labelCommunity(studentList, communityListWeeks[w], noOfCommunities, excellent, weak)
    classifyStudentGroupOverCommunitiesWeeks_check.append(temp1)

classifyStudentGroupOverCommunitiesWeeks_correct = []
for w in range(0,12):
    excellent = exellentPractice[w]
    weak = weakPractice[w]
    
    studentList = transitionDataMatrixWeeks_directFollow_normalised[w].columns
    noOfCommunities = 20
    temp1 = graphLearning.labelCommunity(studentList, communityListWeeks[w], noOfCommunities, excellent, weak, transitionDataMatrixWeeks_directFollow_normalised[w])
    classifyStudentGroupOverCommunitiesWeeks_correct.append(temp1)

groupMoving_activity = graphLearning.getStudentMovingAtLevelOfCommunityDetection(list(reLabelIndex.values()), classifyStudentGroupOverCommunitiesWeeks_correct,2)
a = graphLearning.sankeyDataGenerator(groupMoving_activity, 8,11, [], [ex3_excellent.index, ex3_weak.index])
graphLearning.sankeyVisualise(a)

   
            

#community flow monitoring

e1 = assessment_label3A.loc[assessment_label3A.index.isin(communityListWeeks[9][0][0])]
e2 = assessment_label3A.loc[assessment_label3A.index.isin(communityListWeeks[9][0][1])]

e1['perCorrect3A'].mean()
e2['perCorrect3A'].mean()
stats.ttest_ind(e1['perCorrect3A'],e2['perCorrect3A'], equal_var = False)


# n = [x for x in range(0, len(node_embeddings_2d))]

# for i, txt in enumerate(n):
#     ax.annotate(txt, (node_embeddings_2d[i][0],node_embeddings_2d[i][1]))


#----------------------------------------------
#Node embedding analysis    
#----------------------------------------------
    
node_embeddings_weeks = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    node2vec = Node2Vec(graph_all_weeks[w].graph, dimensions=32, walk_length=5, num_walks=10)
    model = node2vec.fit(window=5, min_count=1)    
    nodeList = model.wv.index2word
    node_embeddings = [list(model.wv.get_vector(n)) for n in nodeList] # numpy.ndarray of size number of nodes times embeddings dimensionality        
    nodeList = list(map(int,model.wv.index2word)) #convert string node to int node
    node_embeddings = pd.DataFrame(node_embeddings, index = nodeList)
    # node_embeddings = node_embeddings.merge(cummulativeExerciseWeeks[w]['correct'],left_on=node_embeddings.index,
    #                                         right_on=cummulativeExerciseWeeks[w]['correct'].index).set_index('key_0')
    scaler = StandardScaler()
    node_embeddings = pd.DataFrame(scaler.fit_transform(node_embeddings), index=node_embeddings.index)
    node_embeddings_weeks.append(node_embeddings)


node_embeddings_2d_df_weeks = []
for w in range(0,12):
    if w in [0,1,2,3]:
        excellent = ex1_excellent.index
        weak = ex1_weak.index
    elif w in [4,5,6,7]:
        excellent = ex2_excellent.index
        weak = ex2_weak.index        
    else:
        excellent = ex3_excellent.index
        weak = ex3_weak.index        
    
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




workingWeekExcercise = []
prediction_transition = []

for week in range(0,12):
    print('Predicting for Week ...' + str(week))
    if week in [0,1,2,3]:
        excellent = ex1_excellent.index
        weak = ex1_weak.index
    elif week in [4,5,6,7]:
        excellent = ex2_excellent.index
        weak = ex2_weak.index        
    else:
        excellent = ex3_excellent.index
        weak = ex3_weak.index   
        
    # workingWeekExcercise.append(nonExUploadByWeek[week])
    # practiceResult = pd.concat(workingWeekExcercise)

    # #adjust number of correct: For each task, number of correct submission/number of submission for that task
    # practiceResultSum = practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
    # practiceResultSum['correct_adjusted'] = practiceResultSum['correct']/practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).count()['correct']
    # cummulativeResult = practiceResultSum.reset_index().groupby([pd.Grouper(key='user')]).sum()
    
    # # cummulativeResult = practiceResultSum.groupby([pd.Grouper(key='user')]).sum()
    # cummulativeResult['cumm_practice'] = cummulativeResult['correct']/practiceResult.groupby([pd.Grouper(key='user')]).count()['date']
    # cummulativeResult['successPassedRate'] = cummulativeResult['passed']/(cummulativeResult['passed'] + cummulativeResult['failed'])
    cummulativeResult = []
    predictionResult = PredictionResult.predict_proba_all_algorithms_data_ready(node_embeddings_weeks[week], excellent, weak, cummulativeResult)
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

title_transition = 'Graph embeddings - Node2Vec - accumulated data - Sum - 1 as passed, 0 as failed'
algorithmList = []
# algorithmList = []
PredictionResult.algorithmComparisonGraph('accuraccy',predictionReport_transition,algorithmList, title_transition)

#descriptive 
#visualise number of activity
activityDataWeek12 = activityDataMatrixWeeks[11].loc[:,['Practice','Read_Lecture_Note','Admin','Check_solution']]
activityDataWeek12['result'] = 



