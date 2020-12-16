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
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.stats import kurtosis
import warnings
import time
warnings.filterwarnings("ignore")
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


eventLog_ca116_filtered['concept:instance'].unique()

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


# eventLog_ca116_filtered.to_csv("eventLog_ca116_filtered_2018.csv", index=False)
weeksEventLog_filtered = [g for n, g in eventLog_ca116_filtered.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
a = weeksEventLog_filtered[1]
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

ex1_excellent = assessment1A.loc[(assessment1A['perCorrect1A'] <= 1) & (assessment1A['perCorrect1A'] >= 0.4)]
ex1_weak = assessment1A.loc[(assessment1A['perCorrect1A'] >= 0) & (assessment1A['perCorrect1A'] < 0.4)]

ex2_excellent = assessment2A.loc[(assessment2A['perCorrect2A'] <= 1)&(assessment2A['perCorrect2A'] >= 0.4)]
ex2_weak = assessment2A.loc[(assessment2A['perCorrect2A'] >= 0) & (assessment2A['perCorrect2A'] < 0.4)]

ex3_excellent = assessment3A.loc[(assessment3A['perCorrect3A'] <= 1)&(assessment3A['perCorrect3A'] >= 0.4)]
ex3_weak = assessment3A.loc[(assessment3A['perCorrect3A'] >= 0) & (assessment3A['perCorrect3A'] < 0.4)]

nonExUpload = dataUpload.drop(dataUpload.loc[dataUpload['task'].str.match('ex')].index)
nonExUploadByWeek = [g for n, g in nonExUpload.groupby(pd.Grouper(key='date',freq='W'))]

#time calculating
transitionDataTimeByWeeks = []
transitionDataFrequencyByWeeks = []
for week in range(1,13):
    transitionDataTimeByWeeks.append(FCAMiner.transitionDataMatrixConstruct_time(weeksEventLog_filtered[week])[0])
    transitionDataFrequencyByWeeks.append(FCAMiner.transitionDataMatrixConstruct_time(weeksEventLog_filtered[week])[1])
for w in range(0,12):
    transitionDataTimeByWeeks[w] = transitionDataTimeByWeeks[w].fillna(0).groupby([pd.Grouper(key='user')]).sum()
    # transitionDataFrequencyByWeeks[w] = transitionDataFrequencyByWeeks[w].fillna(0).groupby([pd.Grouper(key='user')]).sum()
  
#convert data for PCA - from eventlog to transition data matrix
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

# transitionDataMatrixWeeks_time_eventually = []
# full_transitionDataMatrixWeeks_time_eventually = []
# for week in range(1,13):
#     print('Week: ' + str(week) + '...')
#     workingWeekLog.append(weeksEventLog_filtered[week])
#     Log = weeksEventLog_filtered[week] # pd.concat(workingWeekLog) #
#     tempTransition = FCAMiner.transitionDataMatrixConstruct_eventuallyFollow(Log,[],'time').fillna(0)
#     full_transitionDataMatrixWeeks_time_eventually.append(tempTransition)  
#     tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()            
#     transitionDataMatrixWeeks_time_eventually.append(tempTransition)

# transitionDataMatrixWeeks_count_eventually = []
# full_transitionDataMatrixWeeks_count_eventually = []
# for week in range(1,13):
#     print('Week: ' + str(week) + '...')
#     workingWeekLog.append(weeksEventLog_filtered[week])
#     Log = weeksEventLog_filtered[week] # pd.concat(workingWeekLog) #
#     tempTransition = FCAMiner.transitionDataMatrixConstruct_eventuallyFollow(Log,[],'count').fillna(0)
#     full_transitionDataMatrixWeeks_count_eventually.append(tempTransition)  
#     tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()            
#     transitionDataMatrixWeeks_count_eventually.append(tempTransition)
    
# for w in range(0,12):
#     transitionDataMatrixWeeks_distance_eventually[w].to_csv('transitionMatrixStorage/transitionDataMatrixWeeks_distance_eventually_w'+str(w)+'.csv',index=False)
#     full_transitionDataMatrixWeeks_distance_eventually[w].to_csv('transitionMatrixStorage/full_transitionDataMatrixWeeks_distance_eventually_w'+str(w)+'.csv',index=False)
# for w in range(0,12):
#     transitionDataMatrixWeeks_time_eventually[w].to_csv('transitionMatrixStorage/transitionDataMatrixWeeks_time_eventually_w'+str(w)+'.csv',index=False)
#     full_transitionDataMatrixWeeks_time_eventually[w].to_csv('transitionMatrixStorage/full_transitionDataMatrixWeeks_time_eventually_w'+str(w)+'.csv',index=False)
# for w in range(0,12):
#     transitionDataMatrixWeeks_distance_eventually[w].to_csv('transitionMatrixStorage/transitionDataMatrixWeeks_count_eventually_w'+str(w)+'.csv',index=False)
#     full_transitionDataMatrixWeeks_distance_eventually[w].to_csv('transitionMatrixStorage/full_transitionDataMatrixWeeks_count_eventually_w'+str(w)+'.csv',index=False)

transitionDataMatrixWeeks = []
for w in range(0,12):
    transitionDataMatrixWeeks.append(pd.read_csv('transitionMatrixStorage/transitionDataMatrixWeeks_direct_follow_accumulated_w'+str(w)+'.csv', header=0,index_col=0))

#transpose all transition data matrix
for w in range(0,12):
    transitionDataMatrixWeeks[w] = transitionDataMatrixWeeks[w].T

#pca Convert    
pcaDataWeeks = []
pca_result = []
columnsReturn2 = []

# originalElements = []
# pageList = ['Read_Lecture_Note','Exercise','Read_Labsheet','Check_solution','Admin_page']
# actionList = ['load','scroll','blur','focus','unload','upload','hashchange','selection']
# for p in pageList:
#     for a in actionList:
#         originalElements.append(p+'*'+a)

# columns = []
# for i in originalElements:
#     for j in originalElements:
#         # if i != j:
#         txt = i + '-' + j
#         columns.append(txt)
# columns = list(dict.fromkeys(columns))

for w in range(0,12):
    # tempData = transitionDataMatrixWeeks[w].loc[:,columns]
    tempData = transitionDataMatrixWeeks[w]
    # tempData = tempData.merge(prediction_transition[w+1]['data']['successPassedRate'], left_on = tempData.index, right_on=prediction_transition[w+1]['data']['successPassedRate'].index).set_index('key_0')
    temp = FCAMiner.PCAcohortToValue(tempData)
    temp1 = temp[1]
    pcaResult = temp[0]
    # temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
    pcaDataWeeks.append(temp1)
    pca_result.append(pcaResult)
    columnsReturn2.append(temp[2])
    
#get normalise data
transitonDataMAtrixWeeks_normailise = dataProcessing.normaliseWeeklyData(transitionDataMatrixWeeks)
transitonDataMAtrixWeeks_normailise[2]['Read_Lecture_Note-Read_Lecture_Note'].std()

#cleaning data 
import libRMT
transitionDataMatrixWeeks_normalised_cleaned = []
for w in range(0,12):
    transitionDataMatrixWeeks_normalised_cleaned.append(libRMT.regressionToCleanEigenvectorEffect(transitonDataMAtrixWeeks_normailise[w],pcaDataWeeks[w],1))

#pca Convert to cleaned data
pcaDataWeeks_cleanedData = []
pca_result_cleanedData = []
columnsReturn2 = []

# originalElements = ['Read_Lecture_Note','Read_Labsheet','Exercise','Check_solution','Admin_page']
# columns = []
# for i in originalElements:
#     for j in originalElements:
#         # if i != j:
#         txt = i + '-' + j
#         columns.append(txt)
# columns = list(dict.fromkeys(columns))
for w in range(0,12):
    # tempData = transitionDataMatrixWeeks[w].loc[:,columns]
    tempData = transitionDataMatrixWeeks_normalised_cleaned[w].loc[:,:]
    # tempData = tempData.merge(prediction_transition[w+1]['data']['successPassedRate'], left_on = tempData.index, right_on=prediction_transition[w+1]['data']['successPassedRate'].index).set_index('key_0')
    temp = FCAMiner.PCAcohortToValue(tempData)
    temp1 = temp[1]
    pcaResult_cleanedData = temp[0]
    # temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
    pcaDataWeeks_cleanedData.append(temp1)
    pca_result_cleanedData.append(pcaResult_cleanedData)
    columnsReturn2.append(temp[2])
    
#get eigenvalue list
eigenValueList = []
eigenValueList_cleanedData = []
for w in range(0,12):
    eigenValueList.append(pca_result[w].explained_variance_)
    eigenValueList_cleanedData.append(pca_result_cleanedData[w].explained_variance_)

#check if eigenvector is corrrect :((()))    
# test = transitionDataMatrixWeeks[0].loc[:,columns]
# scaler = StandardScaler()
# x = test.values    
# #x_adjust = x - np.mean(x)    
# scaler.fit(x)
# x = scaler.transform(x)

# from numpy.linalg import eig
# values, vectors = eig(c)

# b = np.dot(pca_result[0].components_,x.T)

#test eigenvalue distribution
# eigenValueList = pca_result[11].explained_variance_

# num_bins = 50
# fig, ax = plt.subplots() 
  
# n, bins, patches = ax.hist(eigenValueList, num_bins, 
#                            density = 1,  
#                            color ='blue',  
#                            alpha = 0.7) 
# density = libRMT.marcenkoPastur(126,25,bins)  
# ax.plot(bins, density, '-', color ='black') 
# ax.set_xlabel('P(eigenvalue)') 
# ax.set_ylabel('Eigenvalue') 
# ax.grid()  
# ax.set_title('Eigenvalue distribution') 
# plt.show() 

# Create plot
fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0
num_bins = 100
for w in range(0,12):
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

#IPR draw
# eigenValueList = pca_result[11].explained_variance_
# eigenVectorList = pca_result[11].components_

# IPRlist = libRMT.IPRarray(eigenValueList,eigenVectorList)

# fig, ax = plt.subplots()  
# ax.plot(IPRlist['eigenvalue'], IPRlist['IPR'], '-', color ='blue') 
# ax.set_xlabel('P(eigenvalue)') 
# ax.set_ylabel('IPR')
# ax.axhline(y=1/float(25), color='k') 
# ax.grid()  
# ax.set_title('IPR') 
# plt.show() 

fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0
num_bins = 50
for w in range(0,12):
    ax = fig.add_subplot(3,4,w+1)
    graph.append(ax)
    graph[countGraph].set_xlabel('P(λ)', fontsize = 15)
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


# fig, ax = plt.subplots()  
# num_bins = 25
# for eigenValLabel, eVec in zip(['λ1','λ2'],eigenVectorList[0:2]):    
#     n, bins, patch = ax.hist(eVec, num_bins, density = 1, alpha = 0.5, label=eigenValLabel)
#     # density = (1/np.sqrt(2*np.pi*np.std(eVec)))*np.exp(-(bins-np.mean(eVec))**2)/(2*np.std(eVec))
#     # ax.plot(bins,density, color='black')
# eVec = eigenVectorList[10]
# ax.hist(eVec, num_bins, density = 1, alpha = 0.5, label='λ10')
# ax.set_xlabel('Eigenvector components') 
# ax.set_ylabel('p(Eigenvector components)')
# ax.legend(loc='upper left')

# ax.grid()  
# ax.set_title('Eigenvector components distribution') 
# plt.show() 

fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0
num_bins = 50
for w in range(0,12):
    ax = fig.add_subplot(3,4,w+1)
    graph.append(ax)
    graph[countGraph].set_xlabel('Eigenvector component', fontsize = 15)
    graph[countGraph].set_ylabel('p(eigvenvector)', fontsize = 15)
    graph[countGraph].set_title('Week ' + str(w+1), fontsize = 20)
    graph[countGraph].grid()
    
    graph[countGraph].axvline(x=0, color='k')
    # graph[countGraph].axhline(y=0, color='k')
    # graph[countGraph].axvline(x=0, color='k')
    eigenValueList = pca_result[w].explained_variance_
    eigenVectorList = pca_result[w].components_
    for eigenValLabel, eVec in zip(['λ1','λ2'],eigenVectorList[0:2]):    
        n, bins, patch = ax.hist(eVec, num_bins, density = 1, alpha = 0.5, label=eigenValLabel + 'kurtosis:' + str(kurtosis(eVec)))
        # density = (1/np.sqrt(2*np.pi*np.std(eVec)))*np.exp(-(bins-np.mean(eVec))**2)/(2*np.std(eVec))
        # ax.plot(bins,density, color='black')
    eVec = eigenVectorList[10]
    ax.hist(eVec, num_bins, density = 1, alpha = 0.5, label='λ10' + 'kurtosis:' + str(kurtosis(eVec)))
    graph[countGraph].legend(loc='upper left')
    countGraph = countGraph + 1           
plt.show()

# #Analyss with activity (not transition)
# #convert data for PCA - from eventlog to activity data matrix
# workingWeekLog = []
# activityDataMatrixWeeks = []
# for week in range(1,13):
#     print('Week: ' + str(week) + '...')
#     workingWeekLog.append(weeksEventLog_filtered[week])
#     Log = pd.concat(workingWeekLog)
#     tempTransition = FCAMiner.activityDataMatrixContruct(Log).fillna(0)
#     # tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()            
#     activityDataMatrixWeeks.append(tempTransition)
    
# #pca Convert for activity data - it does not work because there are only 5 features.  
# pcaDataWeeks_activity = []
# pca_result_activity = []
# columnsReturn_activity = []

# columns = ['Read_Lecture_Note','Read_Labsheet','Exercise','Check_solution','Admin_page']

# for w in range(0,12):
#     tempData = activityDataMatrixWeeks[w].loc[:,columns]
#     # tempData = tempData.merge(prediction_transition[w+1]['data']['successPassedRate'], left_on = tempData.index, right_on=prediction_transition[w+1]['data']['successPassedRate'].index).set_index('key_0')
#     temp = FCAMiner.PCAcohortToValue(tempData)
#     # temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
#     pcaDataWeeks_activity.append(temp[1])
#     pca_result_activity.append(temp[0])
#     columnsReturn_activity.append(temp[2])
    
# #eigenvalue analysis
# # Create plot
# fig = plt.figure(figsize=(40,30),dpi=240)
# graph = []
# countGraph = 0
# num_bins = 50
# for w in range(0,12):
#     ax = fig.add_subplot(3,4,w+1)
#     graph.append(ax)
#     graph[countGraph].set_xlabel('λ', fontsize = 15)
#     graph[countGraph].set_ylabel('P(λ)', fontsize = 15)
#     graph[countGraph].set_title('Eigenvalue distribution week ' + str(w+1), fontsize = 20)
#     graph[countGraph].grid()
#     # graph[countGraph].axhline(y=0, color='k')
#     # graph[countGraph].axvline(x=0, color='k')
#     eigenValueList = pca_result_activity[w].explained_variance_
    
#     n, bins, patches = graph[countGraph].hist(eigenValueList, num_bins, 
#                            density = 1,  
#                            color ='blue',  
#                            alpha = 0.7) 
#     density = libRMT.marcenkoPastur(len(pcaDataWeeks_activity[w]),len(pcaDataWeeks_activity[w].columns),bins)
#     graph[countGraph].plot(bins, density, '-', color ='black') 
#     countGraph = countGraph + 1           
# plt.show()

# #Analyss with activity (not transition)
# #convert data for PCA - from eventlog to activity data matrix
# workingWeekLog = []
# timeDataMatrixWeeks = []
# for week in range(1,13):
#     print('Week: ' + str(week) + '...')
#     workingWeekLog.append(weeksEventLog_filtered[week])
#     Log = pd.concat(workingWeekLog)
#     tempTransition = FCAMiner.activityDataMatrixContruct(Log).fillna(0)
#     # tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()            
#     activityDataMatrixWeeks.append(tempTransition)
    
# #pca Convert for activity data - it does not work because there are only 5 features.  
# pcaDataWeeks_activity = []
# pca_result_activity = []
# columnsReturn_activity = []

# columns = ['Read_Lecture_Note','Read_Labsheet','Exercise','Check_solution','Admin_page']

# for w in range(0,12):
#     tempData = activityDataMatrixWeeks[w].loc[:,columns]
#     # tempData = tempData.merge(prediction_transition[w+1]['data']['successPassedRate'], left_on = tempData.index, right_on=prediction_transition[w+1]['data']['successPassedRate'].index).set_index('key_0')
#     temp = FCAMiner.PCAcohortToValue(tempData)
#     # temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
#     pcaDataWeeks_activity.append(temp[1])
#     pca_result_activity.append(temp[0])
#     columnsReturn_activity.append(temp[2])
    
# #eigenvalue analysis
# # Create plot
# fig = plt.figure(figsize=(40,30),dpi=240)
# graph = []
# countGraph = 0
# num_bins = 50
# for w in range(0,12):
#     ax = fig.add_subplot(3,4,w+1)
#     graph.append(ax)
#     graph[countGraph].set_xlabel('P(λ)', fontsize = 15)
#     graph[countGraph].set_ylabel('λ', fontsize = 15)
#     graph[countGraph].set_title('Eigenvalue distribution week ' + str(w+1), fontsize = 20)
#     graph[countGraph].grid()
#     # graph[countGraph].axhline(y=0, color='k')
#     # graph[countGraph].axvline(x=0, color='k')
#     eigenValueList = pca_result_activity[w].explained_variance_
    
#     n, bins, patches = graph[countGraph].hist(eigenValueList, num_bins, 
#                            density = 1,  
#                            color ='blue',  
#                            alpha = 0.7) 
#     density = libRMT.marcenkoPastur(len(pcaDataWeeks_activity[w]),len(pcaDataWeeks_activity[w].columns),bins)
#     graph[countGraph].plot(bins, density, '-', color ='black') 
#     countGraph = countGraph + 1           
# plt.show()

# a = weeksEventLog_filtered[10].loc[:,['case:concept:name','concept:instance','concept:name','time:timestamp','org:resource']]
# a.columns
# np.dot(pca_result[5].components_[0],pca_result[5].components_[2])

def plotLoadings(week,pca_result,columnsReturn1):
    loadings = pd.DataFrame(pca_result[week].components_[0:2, :], 
                            columns=columnsReturn1[week])
    maxPC = 1.01 * np.max(np.max(np.abs(loadings.loc[0:2, :])))
    f, axes = plt.subplots(1, 2, figsize=(5, 5), sharey=True)
    for i, ax in enumerate(axes):
        pc_loadings = loadings.loc[i, :]
        colors = ['C0' if l > 0 else 'C1' for l in pc_loadings]
        ax.axvline(color='#888888')
        ax.axvline(x=0.1, color='#888888')
        ax.axvline(x=-0.1, color='#888888')
        pc_loadings.plot.barh(ax=ax, color=colors)
        ax.set_xlabel(f'PC{i+1}')
        ax.set_xlim(-maxPC, maxPC)
    plt.title('Week '+str(week+1))
    
plotLoadings(1,pca_result_cleanedData,columnsReturn2)  
plotLoadings(7,pca_result_cleanedData,columnsReturn2)  
plotLoadings(11,pca_result,columnsReturn2)  


loadings = pd.DataFrame(pca_result[11].components_[0:2, :], 
                        columns=columnsReturn2[11])
loadings = loadings.T
loadings = loadings.merge(assessment3A.loc[:,['perCorrect3A']], left_on=loadings.index, 
                          right_on = assessment3A.loc[:,['perCorrect3A']].index)

loadings['sign'] = 0
loadings['result_exam'] = 1
loadings.loc[loadings[1] >= 0,['sign']] = 1
loadings.loc[loadings[1] < 0,['sign']] = 0
loadings.loc[loadings['perCorrect3A'] >= 0.4,['result_exam']] = 1
loadings.loc[loadings['perCorrect3A'] < 0.4,['result_exam']] = 0

data_crosstab = pd.crosstab(loadings['sign'], 
                            loadings['result_exam'],  
                               margins = True) 

loadings.corr()

from scipy.stats import chi2_contingency
chi2_contingency(data_crosstab)

pcaDataWeeks1_cleanedData = []
for w in range(0,12):
    if w in [0,1,2,3]:
        column = 'perCorrect1A'
    elif w in [4,5,6,7]:
        column = 'perCorrect2A'
    else:
        column = 'perCorrect3A'
    pcaDataWeeks1_cleanedData.append(pcaDataWeeks_cleanedData[w].merge(assessment[column], left_on = pcaDataWeeks_cleanedData[w].index, right_on = assessment[column].index).set_index('key_0'))
    pcaDataWeeks1_cleanedData[w].rename(columns={column:'result_exam_1'}, 
                  inplace=True)
    
for w in range(0,12):
    a1 = pcaDataWeeks1_cleanedData[w].loc[pcaDataWeeks1_cleanedData[w]['result_exam_1'] == 1,['pc1']]
    b1 = pcaDataWeeks1_cleanedData[w].loc[pcaDataWeeks1_cleanedData[w]['result_exam_1'] == 0,['pc1']]
    t1, p1 = stats.ttest_ind(a1,b1)
    a2 = pcaDataWeeks1_cleanedData[w].loc[pcaDataWeeks1_cleanedData[w]['result_exam_1'] == 1,['pc2']]
    b2 = pcaDataWeeks1_cleanedData[w].loc[pcaDataWeeks1_cleanedData[w]['result_exam_1'] == 0,['pc2']]
    t2, p2 = stats.ttest_ind(a2,b2)
    
    print('Week ' + str(w) + ':')
    print('--PC1: ' + 't-value: ' + str(t1) + ' p-value: ' + str(p1))
    print('-- Excellent: ' + str(a1.mean()[0]))
    print('-- Weak: ' + str(b1.mean()[0]))
    print('--PC2: ' + 't-value: ' + str(t2) + ' p-value: ' + str(p2))
    print('-- Excellent: ' + str(a2.mean()[0]))
    print('-- Weak: ' + str(b2.mean()[0]))      

# Create plot
fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0
for w in range(0,12):
    ax = fig.add_subplot(3,4,w+1)
    graph.append(ax)
    graph[countGraph].set_xlabel('PC 1', fontsize = 15)
    graph[countGraph].set_ylabel('PC 2', fontsize = 15)
    graph[countGraph].set_title('Week' + str(w+1), fontsize = 20)
    graph[countGraph].grid()
    graph[countGraph].axhline(y=0, color='k')
    graph[countGraph].axvline(x=0, color='k')
    graph[countGraph].scatter(pcaDataWeeks1_cleanedData[w].loc[pcaDataWeeks1_cleanedData[w]['result_exam_1'] == 1,['pc1']]
                           ,pcaDataWeeks1_cleanedData[w].loc[pcaDataWeeks1_cleanedData[w]['result_exam_1'] == 1,['pc2']]
                           , c = 'r'
                           , s = 30, label='Successful')
    graph[countGraph].scatter(pcaDataWeeks1_cleanedData[w].loc[pcaDataWeeks1_cleanedData[w]['result_exam_1'] == 0,['pc1']]
                           ,pcaDataWeeks1_cleanedData[w].loc[pcaDataWeeks1_cleanedData[w]['result_exam_1'] == 0,['pc2']]
                           , c = 'b'
                           , s = 30, label='Successful')
    graph[countGraph].legend(loc='upper right')
    countGraph = countGraph + 1
               
plt.show()

#correlation heatmap
pcaDataWeeks1_cleanedData[9].loc[:,['pc1','pc2','pc3','pc4','result_exam_1']].corr()


#prediction
#Prediction for each week
import PredictionResult
workingWeekExcercise = []
# prediction = {}
# prediction_cumm_practice = {}
# prediction_cumm_practice = {} #store transition matrix for prediction 
prediction_transition1 = {}
prediction_transition2 = {}
prediction_transition3 = {}
prediction_transition4 = {}
prediction_transition5 = {}
prediction_transition6 = {}
timePerformance = []
overall_prediction_transition = {}
for week in range(0,12):
    print('Week: ' + str(week) + '...')   

    if week in [0,1,2,3]:

        workingWeekExcercise.append(nonExUploadByWeek[week])
        excellent = ex1_excellent.index
        weak = ex1_weak.index
    elif week in [4,5,6,7]:

        workingWeekExcercise.append(nonExUploadByWeek[week])
        excellent = ex2_excellent.index
        weak = ex2_weak.index
    else:

        workingWeekExcercise.append(nonExUploadByWeek[week])
        excellent = ex3_excellent.index
        weak = ex3_weak.index
    
    # overall_excellent = overall_pass.index
    # overall_weak = overall_failed.index    

    practiceResult = pd.concat(workingWeekExcercise)
    
    #adjust number of correct: For each task, number of correct submission/number of submission for that task
    practiceResultSum = practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
    practiceResultSum['correct_adjusted'] = practiceResultSum['correct']/practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).count()['correct']
    cummulativeResult = practiceResultSum.reset_index().groupby([pd.Grouper(key='user')]).sum()

    # cummulativeResult = practiceResultSum.groupby([pd.Grouper(key='user')]).sum()
    cummulativeResult['cumm_practice'] = cummulativeResult['correct']/practiceResult.groupby([pd.Grouper(key='user')]).count()['date']
    cummulativeResult['successPassedRate'] = cummulativeResult['passed']/(cummulativeResult['passed'] + cummulativeResult['failed'])
    
   
    pcaData1 = transitionDataMatrixWeeks[week] #original data - scenario 1
    pcaData2 = libRMT.selectOutboundComponents(pcaDataWeeks[week],eigenValueList[week]) #filtered pca data to original data - scenario 2 #testing with eigenvalue > 1
    pcaData3 = pcaDataWeeks[week]  #full pca data - scenario 3
    pcaData4 = transitionDataMatrixWeeks_normalised_cleaned[week] #clean data - scenario 4
    pcaData5 = pcaDataWeeks_cleanedData[week] #pca clean data - scenario 5
    pcaData6 = libRMT.selectOutboundComponents(pcaDataWeeks_cleanedData[week],eigenValueList_cleanedData[week]) #pca filtered clean data - scenario 6
    # print(pcaData.columns)

    mode = 'transition'
    
    tic = time.time()
    test1 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData1,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario1',week,toc-tic])
    
    tic = time.time()
    test2 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData2,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario2',week,toc-tic])
    
    tic = time.time()
    test3 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData3,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario3',week,toc-tic])
    
    tic = time.time()
    test4 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData4,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario4',week,toc-tic])    
    
    tic = time.time()
    test5 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData5,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario5',week,toc-tic])    
    
    tic = time.time()
    test6 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData6,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario6',week,toc-tic])    

    # test1 = PredictionResult.predict_proba_all_algorithms(Log,overall_excellent,overall_weak,cummulativeResult,lectureList,mode)
    
    # prediction_cumm_practice.update({ week : test })
    prediction_transition1.update({ week : test1 })
    prediction_transition2.update({ week : test2 })
    prediction_transition3.update({ week : test3 })
    prediction_transition4.update({ week : test4 })
    prediction_transition5.update({ week : test5 })
    prediction_transition6.update({ week : test6 })
    # overall_prediction_transition.update({week : test1})

#################### Visualise for transition matrix data
reportArray_transition = []
prediction_transition = prediction_transition6
for w in range(0,12):
    for algorithm in prediction_transition[w]:
        if algorithm != 'data':
            reportArray_transition.append([w,algorithm, 
                                  prediction_transition[w][algorithm][0]['accuracy_score'][0],
                                  prediction_transition[w][algorithm][0]['f1_score'][0],
                                  prediction_transition[w][algorithm][0]['precision_score'][0],
                                  prediction_transition[w][algorithm][0]['recall_score'][0],
                                  prediction_transition[w][algorithm][0]['roc_auc'],
                                  prediction_transition[w][algorithm][4].mean()
                                  ])
        
predictionReport_transition = pd.DataFrame(reportArray_transition,columns=['week','algorithm','accuraccy',
                                                     'f1_score','precision','recall',
                                                     'roc_auc','cv mean']) 

title_transition = 'pca cleaned data filtered outbound components'
algorithmList = []
# algorithmList = []
PredictionResult.algorithmComparisonGraph('f1_score',predictionReport_transition,algorithmList, title_transition)


#report

predictionReport_transition1 = PredictionResult.reportPredictiveResult(prediction_transition1)
predictionReport_transition2 = PredictionResult.reportPredictiveResult(prediction_transition2)
predictionReport_transition3 = PredictionResult.reportPredictiveResult(prediction_transition3)
predictionReport_transition4 = PredictionResult.reportPredictiveResult(prediction_transition4)
predictionReport_transition5 = PredictionResult.reportPredictiveResult(prediction_transition5)
predictionReport_transition6 = PredictionResult.reportPredictiveResult(prediction_transition6)

predictionReport_transition1['scenario'] = 1
predictionReport_transition2['scenario'] = 2
predictionReport_transition3['scenario'] = 3
predictionReport_transition4['scenario'] = 4
predictionReport_transition5['scenario'] = 5
predictionReport_transition6['scenario'] = 6

predictionReport_transition = pd.concat([predictionReport_transition1,predictionReport_transition2,predictionReport_transition3
                                         ,predictionReport_transition4,predictionReport_transition5,predictionReport_transition6])

selectedBestResult = []
score_type = 'roc_auc'
for w in range(0,12):
    for s in range(1,7):
        selectedBestResult.append(PredictionResult.getBestAlgorithmInAWeek(predictionReport_transition,score_type, s, w))

selectedBestResult = pd.DataFrame(selectedBestResult,columns = ['Week','Scenario','Best score', 'Best Algorithm'])
selectedBestResult.to_csv('BestResultScore.csv',index=False)

#RMT Classified

import libRMT

for week in range(0,12):
    print('Week: ' + str(week) + '...')   
    excellentData = []
    weakData = []
    if week in [0,1,2,3]:
        excellentData = transitionDataMatrixWeeks_normalised_cleaned[week].loc[transitionDataMatrixWeeks_normalised_cleaned[week].index.isin(ex1_excellent.index)]
        weakData = transitionDataMatrixWeeks_normalised_cleaned[week].loc[transitionDataMatrixWeeks_normalised_cleaned[week].index.isin(ex1_weak.index)]
    elif week in [4,5,6,7]:
        excellentData = transitionDataMatrixWeeks_normalised_cleaned[week].loc[transitionDataMatrixWeeks_normalised_cleaned[week].index.isin(ex2_excellent.index)]
        weakData = transitionDataMatrixWeeks_normalised_cleaned[week].loc[transitionDataMatrixWeeks_normalised_cleaned[week].index.isin(ex2_weak.index)]
    else:
        excellentData = transitionDataMatrixWeeks_normalised_cleaned[week].loc[transitionDataMatrixWeeks_normalised_cleaned[week].index.isin(ex3_excellent.index)]
        weakData = transitionDataMatrixWeeks_normalised_cleaned[week].loc[transitionDataMatrixWeeks_normalised_cleaned[week].index.isin(ex3_weak.index)]
    
    # workingWeekExcercise.append(nonExUploadByWeek[week])
    # practiceResult = pd.concat(workingWeekExcercise)
    
    # #adjust number of correct: For each task, number of correct submission/number of submission for that task
    # practiceResultSum = practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
    # practiceResultSum['correct_adjusted'] = practiceResultSum['correct']/practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).count()['correct']
    # cummulativeResult = practiceResultSum.reset_index().groupby([pd.Grouper(key='user')]).sum()

    # # cummulativeResult = practiceResultSum.groupby([pd.Grouper(key='user')]).sum()
    # cummulativeResult['cumm_practice'] = cummulativeResult['correct']/practiceResult.groupby([pd.Grouper(key='user')]).count()['date']
    # cummulativeResult['successPassedRate'] = cummulativeResult['passed']/(cummulativeResult['passed'] + cummulativeResult['failed'])
    # cum_practice_col = ['successPassedRate'] #,'correct_adjusted,cumm_practice','successPassedRate'
    # excellentData = excellentData.merge(cummulativeResult.loc[:,cum_practice_col], 
    #                                         left_on=excellentData.index, right_on=cummulativeResult.index)    
    # excellentData = excellentData.set_index('key_0')
    
    # weakData = weakData.merge(cummulativeResult.loc[:,cum_practice_col], 
    #                                         left_on=weakData.index, right_on=cummulativeResult.index)    
    # weakData = weakData.set_index('key_0')
    
    excellentDataValues = excellentData.values
    weakDataValues = weakData.values
    
    scaler = StandardScaler()
    excellentDataValues = scaler.fit_transform(excellentDataValues)
    scaler = StandardScaler()
    weakDataValues = scaler.fit_transform(weakDataValues)
    
    excellentData['result_exam_1'] = 1
    weakData['result_exam_1'] = 0
    X_train, X_test, y_train, y_test = train_test_split(excellentDataValues, excellentData['result_exam_1'], random_state=5,test_size=0.20)
    clf = libRMT.RMTClassifier()
    clf.fit(X_train)
    test_preds = clf.predict(X_test)
    decoy_preds = clf.predict(weakDataValues)
    
    print('TP: ' + str(np.mean(test_preds))) 
    print('FP: ' + str(np.mean(decoy_preds))) 


#export processed data
ex1_excellent.to_csv('ex1_excellent_2018.csv',index=True)
ex1_weak.to_csv('ex1_weak_2018.csv',index=True)
ex2_excellent.to_csv('ex2_excellent_2018.csv',index=True)
ex2_weak.to_csv('ex2_weak_2018.csv',index=True)
ex3_excellent.to_csv('ex3_excellent_2018.csv',index=True)
ex3_weak.to_csv('ex3_weak_2018.csv',index=True)

nonExUpload.to_csv('nonExUpload_2018.csv',index=False)
