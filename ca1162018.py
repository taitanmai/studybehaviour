#add libraries

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
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import shapiro
from scipy.stats import normaltest
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

# activityList = ['load','scroll','focus','blur','unload','hashchange','selection']

basePath = 'E:\\Data\\extractedData\\'

eventLog_ca116 = pd.read_csv(basePath + 'ca116_eventLog_nonfixed.csv')
# eventLog_ca116 = eventLog_ca116.drop([1160345])
eventLog_ca116 =eventLog_ca116.loc[eventLog_ca116['time:timestamp'] != ' ']
eventLog_ca116['time:timestamp'] = pd.to_datetime(eventLog_ca116['time:timestamp'])
eventLog_ca116 = eventLog_ca116.loc[:, ~eventLog_ca116.columns.str.contains('^Unnamed')]
# materials = eventLog_ca116.loc[:,['org:resource','concept:name','description']]
weeksEventLog = [g for n, g in eventLog_ca116.groupby(pd.Grouper(key='time:timestamp',freq='W'))]

eventLog_ca116_filtered = eventLog_ca116.loc[eventLog_ca116['description'].str.contains('.html|.py|#|/einstein/')]
eventLog_ca116_filtered['pageName'] = eventLog_ca116_filtered['description'].str.extract(r'([^\/][\S]+.html)', expand=False)
eventLog_ca116_filtered['pageName'] = eventLog_ca116_filtered['pageName'].str.replace('.web','')
eventLog_ca116_filtered = eventLog_ca116_filtered.drop(eventLog_ca116_filtered.loc[eventLog_ca116_filtered['concept:name'].isin(['click-0','click-1','click-2','click-3'])].index)

eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('correct|incorrect'),'pageName'] = 'Practice'
eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('^\/einstein\/'),'pageName'] = 'Practice'
eventLog_ca116_filtered['pageName'] = eventLog_ca116_filtered['pageName'].fillna('General')

a = eventLog_ca116_filtered.loc[eventLog_ca116_filtered['pageName'] == 'General']

eventLog_ca116_filtered.rename(columns={'concept:instance':'concept:instance1',
                                   'concept:name':'concept:name1',
                                   'case:concept:name' : 'case:concept:name1'},  inplace=True)
eventLog_ca116_filtered['concept:instance'] = eventLog_ca116_filtered['pageName']
eventLog_ca116_filtered['concept:name'] = eventLog_ca116_filtered['pageName']
eventLog_ca116_filtered['date'] = eventLog_ca116_filtered['time:timestamp'].dt.date

eventLog_ca116_filtered['case:concept:name'] = eventLog_ca116_filtered['date'].astype(str) + '-' + eventLog_ca116_filtered['org:resource'].astype(str)


# eventLog_ca116_filtered['concept:name'] = eventLog_ca116_filtered['pageName'] + '*' + eventLog_ca116_filtered['concept:name1']
# eventLog_ca116_filtered['concept:instance'] = eventLog_ca116_filtered['pageName'] + '*' + eventLog_ca116_filtered['concept:instance1']


weeksEventLog_filtered = [g for n, g in eventLog_ca116_filtered.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
weeksEventLog_filtered = weeksEventLog_filtered[1:13]
a = weeksEventLog_filtered[1].loc[weeksEventLog_filtered[1]['pageName'] == 'Practice']

#------------- Extract number of activities and assign pageType dictionary for later use
#--------- descriptive analysis
listMaterials = []
for w in range(0,12):
    listMaterials.append(weeksEventLog_filtered[w]['pageName'].value_counts().rename('week' + str(w)))
materialAccessedByWeek = pd.concat(listMaterials, axis=1)

materialAccessedByWeek['ofWeek'] = ''
materialAccessedByWeek['pageType'] = ''
materialAccessedByWeek = materialAccessedByWeek.fillna(0)
materialAccessedByWeek.to_csv(basePath + 'materialAccessedByWeek_ca116_2018.csv')

materialAccessedByWeek = pd.read_csv(basePath + 'materialAccessedByWeek_ca116_2018.csv', index_col=0)

materialAccessedByWeek['sumOfpageActivity'] = materialAccessedByWeek.sum(axis = 1, skipna = True)
accessedPageSummary = materialAccessedByWeek.loc[:,['pageType','sumOfpageActivity','ofWeek']].groupby([pd.Grouper('pageType'),pd.Grouper('ofWeek')]).sum()
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
    
a = weeksEventLog_filtered_pageType[11].sort_values(['case:concept:name','time:timestamp'])

workingWeekLog = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    workingWeekLog.append(weeksEventLog_filtered_pageType[w].loc[:,['case:concept:name','concept:instance','concept:name','time:timestamp','org:resource','lifecycle:transition','pageTypeWeek']])

newLog = pd.concat(workingWeekLog)
newLog.to_csv(basePath + 'newEventLog_byWeek_ca116_2018.csv', index = False)

    

dataUpload = pd.read_csv(basePath + 'ca116_uploads.csv')
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

ex1_excellent.to_csv(basePath + 'ca1162018_ex1_excellent.csv',index=True)
ex1_weak.to_csv(basePath + 'ca1162018_ex1_weak.csv',index=True)

ex2_excellent = assessment2A.loc[(assessment2A['perCorrect2A'] <= 1)&(assessment2A['perCorrect2A'] >= 0.4)]
ex2_weak = assessment2A.loc[(assessment2A['perCorrect2A'] >= 0) & (assessment2A['perCorrect2A'] < 0.4)]

ex2_excellent.to_csv(basePath + 'ca1162018_ex2_excellent.csv',index=True)
ex2_weak.to_csv(basePath + 'ca1162018_ex2_weak.csv',index=True)

ex3_excellent = assessment3A.loc[(assessment3A['perCorrect3A'] <= 1)&(assessment3A['perCorrect3A'] >= 0.4)]
ex3_weak = assessment3A.loc[(assessment3A['perCorrect3A'] >= 0) & (assessment3A['perCorrect3A'] < 0.4)]

ex3_excellent.to_csv(basePath + 'ca1162018_ex3_excellent.csv',index=True)
ex3_weak.to_csv(basePath + 'ca1162018_ex3_weak.csv',index=True)

nonExUpload = dataUpload.drop(dataUpload.loc[dataUpload['task'].str.match('ex')].index)
nonExUploadByWeek = [g for n, g in nonExUpload.groupby(pd.Grouper(key='date',freq='W'))]

nonExUpload['version'].unique()
#merge exam result with transition data matrix:
reLabelIndex = dataProcessing.reLabelStudentId(assessment.index)

#exam result


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

for w in range(0,12):
    cummulativeExerciseWeeks[w] = graphLearning.mapNewLabel(cummulativeExerciseWeeks[w],reLabelIndex)
    
assessment_label = assessment.copy()
assessment_label = graphLearning.mapNewLabel(assessment, reLabelIndex)

assessment_label1A = assessment1A.copy()
assessment_label1A = graphLearning.mapNewLabel(assessment1A, reLabelIndex)
assessment_label2A = assessment2A.copy()
assessment_label2A = graphLearning.mapNewLabel(assessment2A, reLabelIndex)
assessment_label3A = assessment3A.copy()
assessment_label3A = graphLearning.mapNewLabel(assessment3A, reLabelIndex)

#activity data matrix construction with newPAgeTYpe
workingWeekLog = []
activityDataMatrixWeeks_pageType = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    workingWeekLog.append(weeksEventLog_filtered_pageType[w])
    LogPageactivityCountByUser = pd.concat(workingWeekLog) #weeksEventLog_filtered_pageType[w]  #pd.concat(workingWeekLog) #weeksEventLog_filtered_pageType[w]
    LogPageactivityCountByUser = FCAMiner.activityDataMatrixContruct(LogPageactivityCountByUser,'pageType')
    LogPageactivityCountByUser = LogPageactivityCountByUser.fillna(0)
    # LogPageactivityCountByUser = FCAMiner.activityDataMatrixPercentage(LogPageactivityCountByUser)
    # LogPageactivityCountByUser = graphLearning.mapNewLabel(LogPageactivityCountByUser,reLabelIndex)
    activityDataMatrixWeeks_pageType.append(LogPageactivityCountByUser)
    
for w in range(0,12):
    temp = activityDataMatrixWeeks_pageType[w].merge(cummulativeExerciseWeeks[w].loc[:,:], left_on=activityDataMatrixWeeks_pageType[w].index, right_on=cummulativeExerciseWeeks[w].index)
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
    activityDataMatrixWeeks_pageType[w] = temp

a =  activityDataMatrixWeeks_pageType[7].corr()

workingWeekLog = []
activityDataMatrixWeeks_pageTypeWeek = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    workingWeekLog.append(weeksEventLog_filtered_pageType[w])
    LogPageactivityCountByUser = pd.concat(workingWeekLog) #weeksEventLog_filtered_pageType[w]
    LogPageactivityCountByUser = FCAMiner.activityDataMatrixContruct(LogPageactivityCountByUser,'pageTypeWeek')
    LogPageactivityCountByUser = LogPageactivityCountByUser.fillna(0)
    # LogPageactivityCountByUser = FCAMiner.activityDataMatrixPercentage(LogPageactivityCountByUser)
    # LogPageactivityCountByUser = graphLearning.mapNewLabel(LogPageactivityCountByUser,reLabelIndex)
    activityDataMatrixWeeks_pageTypeWeek.append(LogPageactivityCountByUser)

for w in range(0,12):
    activityDataMatrixWeeks_pageTypeWeek[w] = graphLearning.mapNewLabel(activityDataMatrixWeeks_pageTypeWeek[w] ,reLabelIndex)

for w in range(0,12):
    if w in [0,1,2,3]:
        studentResult = assessment1A
    elif w in [4,5,6,7]:
        studentResult = assessment2A
    else:
        studentResult = assessment3A
    activityDataMatrixWeeks_pageTypeWeek[w] = activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w].index.isin(studentResult.index)]
    

g = sns.pairplot(activityDataMatrixWeeks_pageTypeWeek[11][activityDataMatrixWeeks_pageTypeWeek[11].columns[:10]])
g.fig.set_size_inches(20,20)

    
#save data page activity
for w in range(0,12):
    activityDataMatrixWeeks_pageTypeWeek[w].to_csv(basePath + 'transitionMatrixStorage_new/activityDataMatrixWeeks_pageTypeWeek_newPractice_new_w'+str(w)+'.csv',index=True)

#read data page activity
activityDataMatrixWeeks_pageTypeWeek = []
for w in range(0,12):
    activityDataMatrixWeeks_pageTypeWeek.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv',index_col=0))

#check normal distribution
df = transitionDataMatrixWeeks_directFollow_standardised[10]
for i in df.columns:
    stat, p = shapiro(df[i])
    print(i + ': ' + 'Statistics=%.3f, p=%.3f' % (stat, p))

plt.hist(df[df.columns[0]], 100, alpha=0.5, label='Col_1')
plt.hist(df[df.columns[1]], 100, alpha=0.5, label='Col_1')
plt.hist(df[df.columns[2]], 100, alpha=0.5, label='Col_1')
plt.legend(loc='upper right')
plt.show()


activityExamData = []   
for w in range(0,12):
    temp = activityDataMatrixWeeks_pageTypeWeek[w].merge(cummulativeExerciseWeeks[w].loc[:,:], left_on=activityDataMatrixWeeks_pageTypeWeek[w].index.astype(str), right_on=cummulativeExerciseWeeks[w].index)
    temp = temp.set_index(['key_0'])
    if w in [0,1,2,3]:
        studentResult = assessment1A
    elif w in [4,5,6,7]:
        studentResult = assessment2A
    else:
        studentResult = assessment3A
    temp = temp.merge(studentResult, left_on=temp.index, right_on=studentResult.index)
    temp = temp.set_index(['key_0'])
    if -1 in temp.index:
        temp = temp.drop([-1])
    activityExamData.append(temp)

a =  activityExamData[11].corr()
a1 = a['perCorrect3A'].sort_values()

a =  activityExamData[7].corr()
a2 = a['perCorrect2A'].sort_values()

a =  activityExamData[3].corr()
a3 = a['perCorrect1A'].sort_values()


examCorrelation = pd.concat([a3,a2,a1], axis=1)
pageTypeWeekList = pd.concat([weeksEventLog_filtered_pageType[i] for i in range(0,12)])['pageTypeWeek'].unique()
examCorrelation = examCorrelation.loc[examCorrelation.index.isin(pageTypeWeekList)]
examCorrelation = examCorrelation.sort_index()
examCorrelation.rename(columns={'perCorrect1A':'Lab Exam 1',
                          'perCorrect2A':'Lab Exam 2',
                          'perCorrect3A':'Lab Exam 3'}, 
                  inplace=True)
sns.heatmap(examCorrelation, cmap='RdYlGn')

for w in range(0,12):
    if w in [0,1,2,3]:
        studentResult = assessment1A
    elif w in [4,5,6,7]:
        studentResult = assessment2A
    else:
        studentResult = assessment3A

    activityDataMatrixWeeks_pageTypeWeek[w] = activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w].index.isin(studentResult.index)]
#correlation analysis between exam, practice and activity
w = 7
for col1 in pageTypeWeekList:
    if col1 in activityDataMatrixWeeks_pageTypeWeek[w]
    for col2 in ['perCorrect2A']:
        p = stats.pearsonr(activityDataMatrixWeeks_pageTypeWeek[w][col1].values,activityDataMatrixWeeks_pageTypeWeek[w][col2].values)
        print(col1 + '-' + col2 + ':' + str(p[0]) + ' - ' +str(p[1]))
        
for week in range(0,12):
    if week in [0,1,2,3]:
        exellent = ex1_excellent.index
        weak = ex1_weak.index
    elif week in [4,5,6,7]:
        exellent = ex2_excellent.index
        w = ex2_weak.index        
    else:
        exellent = ex3_excellent.index
        weak = ex3_weak.index  
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
    transitionDataMatrixWeeks[w].to_csv(basePath + 'transitionMatrixStorage_new/ca1162018_transitionDataMatrixWeeks_direct_accumulated_pageTypeWeek_manyPractice_w'+str(w)+'.csv',index=True)

for w in range(0,12):    
    transitionDataMatrixWeeks[w] = graphLearning.mapNewLabel(transitionDataMatrixWeeks[w], reLabelIndex)

    
#read csv iff neeeded
transitionDataMatrixWeeks = []
for w in range(0,12):
    temp = pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1162018_transitionDataMatrixWeeks_direct_accumulated_pageTypeWeek_manyPractice_w' + str(w) + '.csv', index_col=0)
    if w in [0,1,2,3]:
        studentList = assessment1A.index.astype(str)
    elif w in [4,5,6,7]:
        studentList = assessment2A.index.astype(str)     
    else:
        studentList = assessment3A.index.astype(str)
    temp = temp.loc[temp.index.isin(studentList)]
    # temp = graphLearning.mapNewLabel(temp, reLabelIndex)
    # temp = temp.drop(['Practice_0-Practice_0'],axis=1)
    # if w == 1:
    #     temp = temp.drop([8])
    transitionDataMatrixWeeks.append(temp) 
    
for w in range(0,12):
    transitionDataMatrixWeeks[w].index = transitionDataMatrixWeeks[w].index + ['-2018']

assessment1A.index = assessment1A.index + '-2018'
assessment2A.index = assessment2A.index + '-2018'
assessment3A.index = assessment3A.index + '-2018'

############################################
#################### try k-means
from sklearn.cluster import KMeans
from sklearn import preprocessing

w = 10
transitionDataMatrixWeeks_directFollow_standardised11 = preprocessing.normalize(transitionDataMatrixWeeks[w], norm='l2')
transitionDataMatrixWeeks_directFollow_standardised11 = pd.DataFrame(transitionDataMatrixWeeks_directFollow_standardised11, index = transitionDataMatrixWeeks[w].index, columns = transitionDataMatrixWeeks[w].columns )

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
        a = assessment3A.loc[assessment3A.index.isin(kmeans_result_df_group0.index),['perCorrect3A']]
        print(i)
        print(a.mean())
        print(len(a))
        excellentRate = len(a.loc[a.index.isin(ex3_excellent.index),:])/len(a)
        if (excellentRate > 0.35) and (excellentRate < 0.65):
            numberOfMixedCommunity += 1    
    mixedCommunityRate.append(numberOfMixedCommunity/n)



# end k means #######################


transitionDataMatrixWeeks_directFollow_standardised = []    
for w in range(0,12):
    transitionDataMatrixWeeks_directFollow_standardised.append(dataProcessing.normaliseData(transitionDataMatrixWeeks[w].T))

transitionDataMatrixWeeks_directFollow_normalised = []    
for w in range(0,12):
    transitionDataMatrixWeeks_directFollow_normalised.append(dataProcessing.normaliseData(transitionDataMatrixWeeks[w].T, 'normalised'))




transitionDataMatrixWeeks_directFollow_standardised[11].loc[:,transitionDataMatrixWeeks_directFollow_standardised[11].columns[0]].hist(bins=200)
transitionDataMatrixWeeks_directFollow_standardised[11].loc[:,transitionDataMatrixWeeks_directFollow_standardised[11].columns[1]].hist()

fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(111)

ax1.scatter(transitionDataMatrixWeeks_directFollow_standardised[11].loc[:,transitionDataMatrixWeeks_directFollow_standardised[11].columns[0]], transitionDataMatrixWeeks_directFollow_standardised[11].loc[:,transitionDataMatrixWeeks_directFollow_standardised[11].columns[3]], s=30, c='b', marker="s", label='first')

#transitionDataMatrixWeeks_directFollow_standardised[11].loc[:,['u-5eb79c2f761495b10c559c1ce780cdb1a2d46df8']]

plt.legend(loc='upper left');
plt.show()


#correlation heatmap with exam
w = 11
a = transitionDataMatrixWeeks[w].merge(assessment3A.loc[:,['perCorrect3A']], left_on=transitionDataMatrixWeeks[w].index, right_on=assessment3A.index).set_index('key_0')
b = a.corr()
b1 = b['perCorrect3A'].sort_values()
ax = sns.heatmap(a.corr(),xticklabels=False, yticklabels=False)    

#transpose transition data matrix
transitionDataMatrixWeeks_transposed = []
transitionDataMatrixWeeks_directFollow_normalised_transposed = []
transitionDataMatrixWeeks_directFollow_standardised_transposed = []
for w in range(0,12):
    transitionDataMatrixWeeks_transposed.append(transitionDataMatrixWeeks[w].T)
    transitionDataMatrixWeeks_directFollow_normalised_transposed.append(transitionDataMatrixWeeks_directFollow_normalised[w].T)
    transitionDataMatrixWeeks_directFollow_standardised_transposed.append(transitionDataMatrixWeeks_directFollow_standardised[w].T)

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
                                                           componentToClean, pca_result[week].components_,0.5,0.5)) 

# correlation processing    
corrList_cleaned = []
# corrDistanceList = []
for w in range(0,12):
    corrTemp = transitionDataMatrixWeeks_directFollow_standardised_cleaned[w].corr()
    corrList_cleaned.append(corrTemp)
    # corrDistance = (0.5*(1 - corrTemp)).apply(np.sqrt)
    # corrDistanceList.append(corrDistance)
fig, ax = plt.subplots(1, 1, figsize = (20, 15), dpi=120)
sns.set(font_scale=2)
matrixMask = np.triu(corrList_original[11], k = 1)
sns.heatmap(corrList_original[11], cmap='RdYlGn', yticklabels=False, xticklabels=False, mask=matrixMask)
plt.xlabel("Students", fontsize=22)
plt.ylabel("Students", fontsize=22)


fig, ax = plt.subplots(1, 1, figsize = (20, 15), dpi=120)
sns.set(font_scale=2)
matrixMask = np.triu(detoned_matrix_byLib)
sns.heatmap(detoned_matrix_byLib, cmap='RdYlGn', yticklabels=False, xticklabels=False, mask=matrixMask)
plt.xlabel("Students", fontsize=22)
plt.ylabel("Students", fontsize=22)

fig, ax = plt.subplots(1, 1, figsize = (20, 15), dpi=120)
sns.set(font_scale=2)
matrixMask = np.triu(distance_matrix, k = -1)
sns.heatmap(distance_matrix, cmap='RdYlGn', yticklabels=False, xticklabels=False, mask=matrixMask)
plt.xlabel("Students", fontsize=22)
plt.ylabel("Students", fontsize=22)

transitionDataMatrixWeeks_directFollow_standardised_cleaned[1]  
#correlation processing    
corrList_original = []
# corrDistanceList_dataNormalised = []
for w in range(0,12):
    corrTemp = transitionDataMatrixWeeks_directFollow_standardised[w].corr()
    corrList_original.append(corrTemp)
    # corrDistance = (0.5*(1 - corrTemp)).apply(np.sqrt)
    # corrDistanceList_dataNormalised.append(corrDistance)

corrList_original_of_nonStandardised = []
# corrDistanceList_dataNormalised = []
for w in range(0,12):
    corrTemp = transitionDataMatrixWeeks[w].T.corr()
    corrList_original_of_nonStandardised.append(corrTemp)

    
corrList_cleaned_correlation = []
for w in range(0,12):
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    tn_relation = transitionDataMatrixWeeks_directFollow_standardised[w].shape[0] / transitionDataMatrixWeeks_directFollow_standardised[w].shape[1]
    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01
    matrix = corrList_original[w]
    coreTemp = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=True)
    coreTemp = pd.DataFrame(coreTemp, index = matrix.index, columns = matrix.columns)
    corrList_cleaned_correlation.append(coreTemp)

#draw eigenvalues distribution
w=11
df = transitionDataMatrixWeeks_directFollow_standardised[w]

temp = FCAMiner.PCAcohortToValue(df)
temp1 = temp[1]
pcaResult = temp[0]


fig = plt.figure(figsize=(30,10),dpi=120)
ax = fig.add_subplot(1,1,1)

ax.set_xlabel('eigenvalue λ', fontsize = 15)
ax.set_ylabel('P(λ)', fontsize = 15)
# ax.set_title('Empirical eigenvalue distribution vs RMT prediction', fontsize = 20)
ax.grid()
# graph[countGraph].axhline(y=0, color='k')
# graph[countGraph].axvline(x=0, color='k')
eigenValueList_graph = pcaResult.explained_variance_
num_bins = 150
n, bins, patches = ax.hist(eigenValueList_graph, num_bins, 
                       density = 1,  
                       color ='blue',  
                       alpha = 0.7, label='Empirical') 
densityArray = libRMT.marcenkoPastur(len(df),len(df.columns),bins)
density = densityArray[0]
ax.plot(bins, density, '-', color ='black',label='RMT') 
ax.legend(loc='upper right')
min_lambda = densityArray[1]
max_lambda = densityArray[2]
ax.text( min_lambda*1.1, -0.08, 'λ-', color = 'black', ha = 'center', va = 'center',fontsize=15)
ax.text( max_lambda*1.1, -0.08, 'λ+', color = 'black', ha = 'center', va = 'center',fontsize=15)
plt.show()


distaneMatrixList_cleaned_correlation = []
for w in range(0,12):
    matrix = corrList_original[w]
    coreTemp = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=True)
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
for w in range(0,12):
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
    distance_matrix = distance_matrix.fillna(0)
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
    


import matplotlib.cm as cm  

G = graph_all_weeks_cleaned_correlation[10].graph

G = graph_all_weeks_cleaned_correlation[10].graph

edgeList = G.edges()

for n in G.nodes:
    nodelist.append(n)
    if n in ex3_excellent.index:
        node_color.append('blue')
    else:
        node_color.append('red')


orderCommunityList = [aw10[6][5][2],aw10[6][5][1],aw10[6][5][5],aw10[6][5][0],aw10[6][5][4],aw10[6][5][7],aw10[6][5][6],aw10[6][5][3]]    

node_color = []
nodelist = []
for m in orderCommunityList:
    for n in m.index:
        nodelist.append(n)
        if n in ex3_excellent.index:
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

communityWeek12 = graphLearning.community_dection_graph(graph_all_weeks_cleaned_correlation[11], num_comms=len(graph_all_weeks_cleaned_correlation[11].graph._node))   

columnsList = []
communitySeparation = communityWeek12[6]
for i in range(len(communitySeparation)):
    for j in range(len(communitySeparation[i])):
        if communitySeparation[i][j] in ex3_excellent.index:
            k = 'excellent'
        else:
            k = 'weak'
        columnsList.append([communitySeparation[i][j],i,k])
columnsListDf = pd.DataFrame(columnsList)        
columnsListDf = columnsListDf.sort_values(by = [1,2])

columns = list(columnsListDf.loc[:,[0]][0])
transitionDataMatrixWeeks_directFollow_standardised[11] = transitionDataMatrixWeeks_directFollow_standardised[11].loc[:,columns]

    
#community detection
communityListWeeks = []
for w in range(0,12):
    print('Week ' + str(w) + '...')      
    num_comms = len(graph_all_weeks[w].graph._node)
    communityListWeeks.append(graphLearning.community_dection_graph(graph_all_weeks[w], num_comms=num_comms))

communityListWeeks_cleaned = []
for w in range(0,12):
    print('Week ' + str(w) + '...')      
    num_comms = len(graph_all_weeks_cleaned[w].graph._node)
    communityListWeeks_cleaned.append(graphLearning.community_dection_graph(graph_all_weeks_cleaned[w], num_comms=num_comms))

communityListWeeks_cleaned_correlation = []
for w in range(0,12):
    print('Week ' + str(w) + '...')      
    num_comms = len(graph_all_weeks_cleaned_correlation[w].graph._node)
    communityListWeeks_cleaned_correlation.append(graphLearning.community_dection_graph(graph_all_weeks_cleaned_correlation[w], num_comms=num_comms))


communityListWeeks_fullConnected = []
for w in range(11,12):
    print('Week ' + str(w) + '...')      
    num_comms = len(graph_all_weeks_fullyConnected[w]._node)
    communityListWeeks_fullConnected.append(graphLearning.community_dection_graph(graph_all_weeks_fullyConnected[w], num_comms=num_comms, mst=False))


#dendrogram visualisation
graphLearning.girvanNewManDendrogram(graph_all_weeks_cleaned_correlation[10], ex3_excellent.index, ex3_weak.index, num_comms=num_comms)


import graphLearning
import scikit_posthocs as sp
pd.set_option("display.max_rows", None, "display.max_columns", None)
aw10 = graphLearning.extractAssessmentResultOfCommunities(communityListWeeks_cleaned_correlation[10], assessment3A, 'perCorrect3A')
aw10t = sp.posthoc_conover(aw10[6][5])

aw9 = graphLearning.extractAssessmentResultOfCommunities(communityListWeeks_cleaned_correlation[9], assessment3A, 'perCorrect3A')
aw9t = sp.posthoc_conover(aw9[18][5])

aw10 = graphLearning.extractAssessmentResultOfCommunities(communityListWeeks_cleaned_correlation[10], assessment3A, 'perCorrect3A')
aw10t = sp.posthoc_conover(aw10[18][5])

aw7 = graphLearning.extractAssessmentResultOfCommunities(communityListWeeks_cleaned_correlation[7], assessment2A, 'perCorrect2A')
aw7t = sp.posthoc_conover(aw7[18][5])

a = graphLearning.findTogetherMembers(aw10[18][5],aw11[18][5], aw10[18][1],aw11[18][1])
len(set(assessment2A.index).intersection(set(assessment3A.index)))

a = graphLearning.findTogetherMembers(aw7[18][5],aw11[18][5], aw7[18][1],aw11[18][1])
for i in range(0,8):
    for j in range(0,8):
        if len(a[i][j]) > 1:
            print(a[i][j])

good = 2
bad = 3

goodCommunity = aw10[6][5][good]
badCommunity = aw10[6][5][bad]

goodStudentEventLog = []
workingWeekLog = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    workingWeekLog.append(weeksEventLog_filtered_pageType[w].loc[weeksEventLog_filtered_pageType[w]['org:resource'].isin(ex3_excellent.index)])
goodStudentEventLog =  pd.concat(workingWeekLog)
goodStudentEventLog.loc[:,['case:concept:name','pageTypeWeek','time:timestamp','org:resource','lifecycle:transition']].to_csv(basePath + 'ca1162018_goodStudentsLog_withTest.csv', index=False)

a = goodStudentEventLog.loc[:,['case:concept:name','pageTypeWeek','time:timestamp','org:resource','lifecycle:transition']]
a1 = a.loc[a['org:resource'] == 'u-19bad38d8231fe7f2fc601254aee8354cace7a43']

badStudentEventLog = []
workingWeekLog = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    workingWeekLog.append(weeksEventLog_filtered_pageType[w].loc[weeksEventLog_filtered_pageType[w]['org:resource'].isin(ex3_weak.index)])
badStudentEventLog =  pd.concat(workingWeekLog)
badStudentEventLog.loc[:,['case:concept:name','pageTypeWeek','time:timestamp','org:resource','lifecycle:transition']].to_csv(basePath + 'ca1162018_badStudentsLog_withTest.csv', index=False)

weeksEventLog_filtered_pageType[w].columns

w = 10
extractGoodBadCommunity = activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w].index.astype(str).isin(goodCommunity.index) | 
                                                                      activityDataMatrixWeeks_pageTypeWeek[w].index.astype(str).isin(badCommunity.index)]
extractGoodBadCommunity['group'] = 0
extractGoodBadCommunity.loc[extractGoodBadCommunity.index.astype(str).isin(goodCommunity.index),['group']] = good
extractGoodBadCommunity.loc[extractGoodBadCommunity.index.astype(str).isin(badCommunity.index),['group']] = bad

columnListStatsSig = []
for c in extractGoodBadCommunity.columns:
    t1 = stats.normaltest(extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == good, [c]])[1][0]
    t2 = stats.normaltest(extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == bad, [c]])[1][0]
    if t1 <= 0.1 and t2 <= 0.1:
        columnListStatsSig.append(c)
        
extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == 1, ['Lecture_4']].hist(bins=80)

compareMean = []
for c in extractGoodBadCommunity.columns:
    arr1 = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == good, [c]]
    arr2 = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == bad, [c]]
    test, pvalue = stats.mannwhitneyu(arr1,arr2)

    if c!= 'group':
        meanGood = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == good, [c]].mean()[0]
        stdGood = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == good, [c]].std()[0]
        meanBad = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == bad, [c]].mean()[0]
        stdBad = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == bad, [c]].std()[0]
        compareMean.append([c, meanGood, stdGood, meanBad, stdBad, test, pvalue])
        # print(c + ': ' + str(test) + ' Good Community: ' + str(meanGood) + ' -- ' + 'Bad Community: ' + str(meanBad))
compareMeanDf = pd.DataFrame(compareMean, columns=['Material','Mean Good', 'SD Good', 'Mean Bad', 'SD Bad', 'Test','p-value'])
compareMeanDfnewCol = compareMeanDf['Material'].str.split('_', expand = True)
compareMeanDf['week'] = compareMeanDfnewCol[1].astype(str)
compareMeanDf['MaterialType'] = compareMeanDfnewCol[0]
compareMeanDf = compareMeanDf.sort_values(['MaterialType','week'])

compareMeanDf.to_csv(basePath + 'ca1162018_compareMeanDf.csv',index=True)

extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == good, ['Lecture_11']]
#detect transition

w = 10
extractGoodBadCommunityTransition = transitionDataMatrixWeeks[w].loc[transitionDataMatrixWeeks[w].index.astype(str).isin(goodCommunity.index) | transitionDataMatrixWeeks[w].index.astype(str).isin(badCommunity.index)]
extractGoodBadCommunityTransition['group'] = 0
extractGoodBadCommunityTransition.loc[extractGoodBadCommunityTransition.index.astype(str).isin(goodCommunity.index),['group']] = 2
extractGoodBadCommunityTransition.loc[extractGoodBadCommunityTransition.index.astype(str).isin(badCommunity.index),['group']] = 3

compareMeanTransition = []
for c in extractGoodBadCommunityTransition.columns:
    arr1 = extractGoodBadCommunityTransition.loc[extractGoodBadCommunity['group'] == 3, [c]]
    arr2 = extractGoodBadCommunityTransition.loc[extractGoodBadCommunity['group'] == 2, [c]]
    try:
        test = stats.mannwhitneyu(arr1,arr2)[1]
        if test <= 0.05:
            if c!= 'group':
                if c.split('-')[0] != c.split('-')[1]:
                    meanGood = extractGoodBadCommunityTransition.loc[extractGoodBadCommunityTransition['group'] == 2, [c]].mean()[0]
                    meanBad = extractGoodBadCommunityTransition.loc[extractGoodBadCommunityTransition['group'] == 3, [c]].mean()[0]
                    compareMeanTransition.append([c, meanGood, meanBad])
    except:
        continue
        # print(c + ': ' + str(test) + ' Good Community: ' + str(meanGood) + ' -- ' + 'Bad Community: ' + str(meanBad))
compareMeanTransitionDf = pd.DataFrame(compareMeanTransition, columns=['Transition','Best Group', 'Worst Group'])

#------------------------------- t tesst with original data
w = 10
count = 0
for c in activityDataMatrixWeeks_pageTypeWeek[w].columns:
    a1 = activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w].index.isin(ex3_excellent.index),[c]]
    b1 = activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w].index.isin(ex3_weak.index),[c]]
    t1, p1 = stats.mannwhitneyu(a1,b1) 
    # print('Week ' + str(w) + ':')
    if p1 <= 0.05:
        print(c + ': ' + 't-value: ' + str(t1) + ' p-value: ' + str(p1))
        print('-- Excellent: ' + str(a1.mean()[0]))
        print('-- Weak: ' + str(b1.mean()[0]))
        count = count + 1
print(count)

#draw horizontal barchart for compare mean activity
# create plot
fig, ax = plt.subplots(figsize=(30,20), dpi=150)
index = np.arange(len(compareMeanDf.index))
bar_width = 0.4
opacity = 1

rects1 = plt.barh(index, compareMeanDf['Mean Good'], bar_width, alpha=opacity, color='b', label='Highest performing community')
rects2 = plt.barh(index + bar_width, compareMeanDf['Mean Bad'], bar_width, alpha=opacity, color='g', label='Lowest performing community')

plt.ylabel('Course Materials', fontsize=20)
plt.xlabel('Average number of activities', fontsize=20)
plt.title('')
plt.yticks(index + bar_width, compareMeanDf.Material, fontsize=18)
plt.xticks(fontsize=20)
plt.legend(fontsize=25)

# plt.tight_layout()
plt.show()
     
import graphLearning       
        
classifyStudentGroupOverCommunitiesWeeks = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    if w in [0,1,2,3]:
        excellent = ex1_excellent.index
        weak = ex1_weak.index
    elif w in [4,5,6,7]:
        excellent = ex2_excellent.index
        weak = ex2_weak.index
    else:
        excellent = ex3_excellent.index
        weak = ex3_weak.index
    
    studentList = transitionDataMatrixWeeks_directFollow_standardised[w].columns
    noOfCommunities = 20
    temp1 = graphLearning.labelCommunity(studentList, communityListWeeks[w], noOfCommunities, excellent, weak, transitionDataMatrixWeeks_directFollow_standardised[w])
    classifyStudentGroupOverCommunitiesWeeks.append(temp1)

for w in range(0,12):
    classifyStudentGroupOverCommunitiesWeeks[w].to_csv(basePath + 'transitionMatrixStorage_new/relabel_assessment_ca1162018_w' + str(w) + '.csv', index=True)

fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0    
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
    # excellent = exellentPractice[w]
    # weak = weakPractice[w]
    
    excellentLine = []
    weakLine = []
    mixedLine = []
    mixedProportionLine = []
    
    excellentLine_not_cleaned = []
    weakLine_not_cleaned = []
    mixedLine_not_cleaned = []
    mixedProportionLine_not_cleaned = []
    noOfCommunities = []
    upper = 0.75
    lower = 0.25
    for i in range(0,num_comms-1):
        a1 = graphLearning.identifyCommunitiesType(communityListWeeks[w][i], excellent, weak)
        excellentLine.append(len(a1.loc[a1['excellentRate'] >= upper]))
        weakLine.append(len(a1.loc[a1['excellentRate'] < lower]))
        mixedLine.append(len(a1.loc[(a1['excellentRate'] <upper) & (a1['excellentRate'] >= lower)]))
        mixedProportionLine.append(len(a1.loc[(a1['excellentRate'] <upper) & (a1['excellentRate'] >= lower)])/(len(a1)))
        
        a2 = graphLearning.identifyCommunitiesType(communityListWeeks_not_cleaned[w][i], excellent, weak)
        excellentLine_not_cleaned.append(len(a2.loc[a2['excellentRate'] >= upper]))
        weakLine_not_cleaned.append(len(a2.loc[a2['excellentRate'] < lower]))
        mixedLine_not_cleaned.append(len(a2.loc[(a2['excellentRate'] < upper) & (a2['excellentRate'] >= lower)]))
        mixedProportionLine_not_cleaned.append(len(a2.loc[(a2['excellentRate'] <upper) & (a2['excellentRate'] >= lower)])/(len(a2)))
        
        noOfCommunities.append(i+2)
    
    ax = fig.add_subplot(3,4,w+1)
    graph.append(ax)
    graph[countGraph].set_xlabel('Number of communities', fontsize = 15)
    graph[countGraph].set_ylabel('Proportion of mixed Communities over all communities', fontsize = 15)
    graph[countGraph].set_title('Week' + str(w+1), fontsize = 20)
    graph[countGraph].grid()
    # graph[countGraph].plot(noOfCommunities, excellentLine, label = "No of excellent communities")    
    # graph[countGraph].plot(noOfCommunities, weakLine, label = "No of weak communities")  
    # graph[countGraph].plot(noOfCommunities, mixedLine, label = "No of mixed communities") 
    graph[countGraph].plot(noOfCommunities, mixedProportionLine, label = "Proportion of mixed communities") 
    
    # graph[countGraph].plot(noOfCommunities, excellentLine_not_cleaned, label = "No of excellent communities not cleaned data")    
    # graph[countGraph].plot(noOfCommunities, weakLine_not_cleaned, label = "No of weak communities not cleaned data")  
    graph[countGraph].plot(noOfCommunities, mixedProportionLine_not_cleaned, label = "Proportion of mixed communities not cleaned data") 
    
    graph[countGraph].legend(loc='upper left')
    countGraph = countGraph + 1               
plt.show()      


#only show in a certain week
sns.reset_orig()
w = 10

if w in [0,1,2,3]:
    excellent = ex1_excellent.index
    weak = ex1_weak.index
elif w in [4,5,6,7]:
    excellent = ex2_excellent.index
    weak = ex2_weak.index        
else:
    excellent = ex3_excellent.index
    weak = ex3_weak.index 
# excellent = exellentPractice[w]
# weak = weakPractice[w]

w = 10
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
    
    # a3 = graphLearning.identifyCommunitiesType(partition_fullyGraphConvertedList[i], excellent, weak)
    # excellentLine_not_cleaned_super.append(len(a3.loc[a3['excellentRate'] >= upper]))
    # weakLine_not_cleaned_super.append(len(a3.loc[a3['excellentRate'] < lower]))
    # mixedLine_not_cleaned_super.append(len(a3.loc[(a3['excellentRate'] < upper) & (a3['excellentRate'] >= lower)]))
    # mixedProportionLine_not_cleaned_super.append(len(a3.loc[(a3['excellentRate'] <upper) & (a3['excellentRate'] >= lower)])/(len(a3)))
       
    
    noOfCommunities.append(i+2)
    
fig = plt.figure(figsize=(15,10),dpi=240)
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

ax.plot(noOfCommunities, mixedCommunityRate, label = "Kmeans with normalised data") 
# ax.plot(noOfCommunities, mixedProportionLine_not_cleaned_super, label = "Louvain community fully graph") 

ax.legend(loc='upper right',  fontsize = 15)
          
plt.show() 

#comparing using Louvain

import graphLearning

edges1 = list(graph_all_weeks_fullyConnected[11].edges(data=True))
edges1[1][2]['weight']
len(graph_all_weeks_fullyConnected[11].nodes)

graphLearning.checkIfUniqueEdges(graph_all_weeks_fullyConnected[11])
A = nx.to_numpy_matrix(graph_all_weeks_fullyConnected[11])
np.savetxt(basePath + "Course1-2018.csv", A, delimiter=",")

list(partition.keys())

import community as community_louvain
partition = community_louvain.best_partition(graph_all_weeks_cleaned_correlation[10].graph,resolution=1)
girvan_newman_cluster = graphLearning.convertGNcommunityToFlattenList(communityListWeeks_cleaned_correlation[11][6], list(partition.keys()))


partition = community_louvain.best_partition(graph_all_weeks[10].graph,resolution=1)
girvan_newman_cluster = graphLearning.convertGNcommunityToFlattenList(communityListWeeks[10][6], list(partition.keys()))


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


partition_fullyGraph = community_louvain.best_partition(graph_all_weeks_fullyConnected[11],resolution=0.99)
partition_fullyGraphConverted = graphLearning.convertFlattenListToCommunity(partition_fullyGraph)
partition_fullyGraphConverted =  [v for k, v in partition_fullyGraphConverted.items()]
partition_fullyGraphConverted = tuple(partition_fullyGraphConverted)
partition_fullyGraphConvertedList = [partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,
                          partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted,
                          partition_fullyGraphConverted,partition_fullyGraphConverted,partition_fullyGraphConverted]


partition_not_cleaned = community_louvain.best_partition(graph_all_weeks[10].graph)
partitionConverted_not_cleaned = graphLearning.convertFlattenListToCommunity(partition_not_cleaned)
partitionConverted_not_cleaned = [v for k, v in partitionConverted_not_cleaned.items()]
partitionConverted_not_cleaned = tuple(partitionConverted_not_cleaned)

a2 = graphLearning.identifyCommunitiesType(partitionConverted_not_cleaned, excellent, weak)
len(a2.loc[(a2['excellentRate'] <upper) & (a2['excellentRate'] >= lower)])/(len(a2))

a3 = graphLearning.identifyCommunitiesType(partitionConverted, excellent, weak)
len(a3.loc[(a3['excellentRate'] <upper) & (a3['excellentRate'] >= lower)])/(len(a3))
   
    
#visualise MST
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
ex3_excellent = graphLearning.mapNewLabel(ex3_excellent, reLabelIndex)
ex3_weak = graphLearning.mapNewLabel(ex3_weak, reLabelIndex)

excellentList = ex3_excellent.index
weakList = ex3_weak.index
graphLearning.visualiseMSTGraph(graph_all_weeks_cleaned_correlation[10], {'Higher performing students': excellentList, 'Lower performing students': weakList},  reLabelIndex)  



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
    # node_embeddings = node_embeddings.merge(cummulativeExerciseWeeks[w]['correct'],left_on=node_embeddings.index,
    #                                         right_on=cummulativeExerciseWeeks[w]['correct'].index).set_index('key_0')
    # scaler = StandardScaler()
    # node_embeddings = pd.DataFrame(scaler.fit_transform(node_embeddings), index=node_embeddings.index)
    node_embeddings_weeks.append(node_embeddings)

for w in range(0,12):
    node_embeddings_weeks[w] = node_embeddings_weeks[w].merge(cummulativeExerciseWeeks[w]['correct'],left_on=node_embeddings_weeks[w].index,
                                        right_on=cummulativeExerciseWeeks[w]['correct'].index).set_index('key_0')

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
    
    node_embeddings_2d_df.loc[node_embeddings_2d_df.index.astype(int).isin(excellent),['result_exam_1']] = 1
    node_embeddings_2d_df.loc[node_embeddings_2d_df.index.astype(int).isin(weak),['result_exam_1']] = 0
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
filteredData = []
for week in range(0,12):
    if week in [0,1,2,3]:
        excellent = ex1_excellent.index
        weak = ex1_weak.index
    elif week in [4,5,6,7]:
        excellent = ex2_excellent.index
        weak = ex2_weak.index        
    else:
        excellent = ex3_excellent.index
        weak = ex3_weak.index   
    #dataForPrediction =  node_embeddings_weeks 
    filteredData.append(activityDataMatrixWeeks_pageTypeWeek[week].loc[activityDataMatrixWeeks_pageTypeWeek[week].index.isin(excellent.union(weak))])

#PCA, denoised and detrend data before prediction
import FCAMiner
pca_result = []
pcaDataWeeks = []
columnsReturn2 = []
for w in range(0,12):
    # tempData = transitionDataMatrixWeeks[w].loc[:,columns]
    tempData = filteredData[w]
    # tempData = tempData.merge(prediction_transition[w+1]['data']['successPassedRate'], left_on = tempData.index, right_on=prediction_transition[w+1]['data']['successPassedRate'].index).set_index('key_0')
    temp = FCAMiner.PCAcohortToValue(tempData)
    temp1 = temp[1]
    pcaResult = temp[0]
    # temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
    pcaDataWeeks.append(temp1)
    pca_result.append(pcaResult)
    columnsReturn2.append(temp[2])    

dataForPredictionOriginal =  filteredData

#filtered PCA data
dataForPrediction_outbound = []
for week in range(0,12):
    dataForPrediction_outbound.append(libRMT.selectOutboundComponents(pcaDataWeeks[week],pca_result[week].explained_variance_))

#clean first eigen effect data with regression
dataForPrediction = []    
for week in range(0,12):
    # outBoundComponents = []
    # for c in dataForPrediction_outbound[week].columns:
    #     outBoundComponents.append(int(c[-1]))
    
    componentToClean = [1]
    # for c in range(1,len(filteredData[week].columns)+1):
    #     if c not in outBoundComponents:
    #         componentToClean.append(c)
    dataForPrediction.append(libRMT.regressionToCleanEigenvectorEffect(filteredData[week], pcaDataWeeks[week],componentToClean)) 

#clean first eigen effect data (not regression, just reconstruct X_hat from outbound components)
import libRMT
dataForPrediction = []    
for week in range(0,12):
    outBoundComponents = dataForPrediction_outbound[week].columns
    componentToClean = ['pc1']
    # for c in pcaDataWeeks[week].columns:
    #     if c not in outBoundComponents:
    #         componentToClean.append(c)
    dataForPrediction.append(libRMT.cleanEigenvectorEffect(filteredData[week],pcaDataWeeks[week], componentToClean, pca_result[week].components_,0.5,0.8)) 


#prediction
import PredictionResult
prediction_transition = []
cummulativeResult = []
for week in range(0,12):
    print('Predicting for Week ...' + str(week))
    # if week in [0,1,2,3]:
    #     excellent = ex1_excellent.index
    #     weak = ex1_weak.index
    # elif week in [4,5,6,7]:
    #     excellent = ex2_excellent.index
    #     weak = ex2_weak.index        
    # else:
    excellent = ex3_excellent.index
    weak = ex3_weak.index  
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

title_transition = 'Prediction with PCA outbound data - exams 3'#'Outbound (both > lambda_max and < lambda_min) eigenvalues only by RMT'
algorithmList = []
# algorithmList = []
PredictionResult.algorithmComparisonGraph('roc_auc',predictionReport_transition,algorithmList, title_transition)
PredictionResult.algorithmComparisonGraph('cv mean',predictionReport_transition,algorithmList, title_transition)

community.greedy_modularity_communities(graph_all_weeks[11].graph)
sum(pca_result[11].components_[0]**2)

#--------------- PREDCTION with activity data --------------------------------------------------#
#---------------------------------------------------------------------------------#

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
    if w in [0,1,2,3]:
        pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ex1_excellent.index),['result_exam_1']] = 1
    elif w in [4,5,6,7]:
        pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ex2_excellent.index),['result_exam_1']] = 1
    else:
        pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ex3_excellent.index),['result_exam_1']] = 1

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
fig = plt.figure(figsize=(20,15))
ax = fig.subplots()
ax.set_xlabel('Eigenvalues', fontsize = 30)
ax.set_ylabel('IPR', fontsize = 30)
ax.tick_params(axis='both', which='major', labelsize=25)
ax.tick_params(axis='both', which='minor', labelsize=25)
ax.set_title('Inverse Participation Ratio week ' + str(w+1), fontsize = 30)
ax.grid()
# graph[countGraph].axhline(y=0, color='k')
# graph[countGraph].axvline(x=0, color='k')
eigenValueList = pca_result[11].explained_variance_
eigenVectorList = pca_result[11].components_
IPRlist = libRMT.IPRarray(eigenValueList,eigenVectorList)
ax.axhline(y=IPRlist['IPR'].mean(), color='k', label='mean value of IPR') 
ax.plot(IPRlist['eigenvalue'], IPRlist['IPR'], '-', color ='blue', label='IPR')
ax.legend(loc='upper right', fontsize=20)
plt.show()

#outbounce select
a = libRMT.selectOutboundComponents(pcaDataWeeks[11],pca_result[11].explained_variance_)

w = 11    
libRMT.biplot(pcaDataWeeks[w],
       np.transpose(pca_result[w].components_[1:3, :]),
       pcaDataWeeks[w].loc[:,['result_exam_1']], activityDataMatrixWeeks_pageTypeWeek[w].columns, 'pc2','pc3', title=['Lower performing students','Higher performing students'])

  
libRMT.plotLoadings(1,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  
libRMT.plotLoadings(7,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  
libRMT.plotLoadings(11,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  

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
    
################################ Time analysis
#time calculating
transitionDataTimeByWeeks = []
workingWeekLog = []
for week in range(0,12):
    workingWeekLog.append(weeksEventLog_filtered_pageType[week])
    LogPageactivityCountByUser =  pd.concat(workingWeekLog)
    transitionDataTimeByWeeks.append(FCAMiner.transitionDataMatrixConstruct_time(LogPageactivityCountByUser, activityColumn = 'pageTypeWeek')[0])
for w in range(0,12):
    transitionDataTimeByWeeks[w] = transitionDataTimeByWeeks[w].fillna(0).groupby([pd.Grouper(key='user')]).sum()
    
#eliminate zero column    
for w in range(0,12):
    transitionDataTimeByWeeks[w] = transitionDataTimeByWeeks[w].loc[:, (transitionDataTimeByWeeks[w] != 0).any(axis=0)]

#read data page activity
for w in range(0,12):
    transitionDataTimeByWeeks[w].to_csv(basePath + 'transitionMatrixStorage_new/transitionDataTimeByWeeks_w'+str(w)+'.csv',index=True)

    
activityDataTimeByWeeks = []
workingWeekLog = []
for w in range(0,12):
    workingWeekLog.append(weeksEventLog_filtered_pageType[w])
    LogPageactivityCountByUser =  pd.concat(workingWeekLog)
    activityList = LogPageactivityCountByUser['pageTypeWeek'].unique()
    transitionList = transitionDataTimeByWeeks[w].columns
    for i in activityList:
        transitionDataTimeByWeeks[w][i] = 0
        for t in transitionList:
            if t.split('-')[0] == i:
                transitionDataTimeByWeeks[w][i] = transitionDataTimeByWeeks[w][i] + transitionDataTimeByWeeks[w][t]
    activityDataTimeByWeeks.append(transitionDataTimeByWeeks[w].loc[:,activityList])
            
#correlation analysis -> not good  
activityExamData = []   
for w in range(0,12):
    temp = activityDataTimeByWeeks[w].merge(cummulativeExerciseWeeks[w].loc[:,:], left_on=activityDataTimeByWeeks[w].index.astype(str), right_on=cummulativeExerciseWeeks[w].index)
    temp = temp.set_index(['key_0'])
    if w in [0,1,2,3]:
        studentResult = assessment1A
    elif w in [4,5,6,7]:
        studentResult = assessment2A
    else:
        studentResult = assessment3A
    temp = temp.merge(studentResult, left_on=temp.index, right_on=studentResult.index)
    temp = temp.set_index(['key_0'])
    if -1 in temp.index:
        temp = temp.drop([-1])
    activityExamData.append(temp)

a =  activityExamData[11].corr()
a1 = a['perCorrect3A'].sort_values()

a =  activityExamData[7].corr()
a2 = a['perCorrect2A'].sort_values()

a =  activityExamData[3].corr()
a3 = a['perCorrect1A'].sort_values()


examCorrelation = pd.concat([a3,a2,a1], axis=1)
pageTypeWeekList = pd.concat([weeksEventLog_filtered_pageType[i] for i in range(0,12)])['pageTypeWeek'].unique()
examCorrelation = examCorrelation.loc[examCorrelation.index.isin(pageTypeWeekList)]
examCorrelation = examCorrelation.sort_index()
examCorrelation.rename(columns={'perCorrect1A':'Lab Exam 1',
                          'perCorrect2A':'Lab Exam 2',
                          'perCorrect3A':'Lab Exam 3'}, 
                  inplace=True)
sns.heatmap(examCorrelation, cmap='RdYlGn')    

#IPR
pca_result = []
pcaDataWeeks = []
columnsReturn2 = []
for w in range(0,12):
    # tempData = transitionDataMatrixWeeks[w].loc[:,columns]
    tempData = activityDataTimeByWeeks[w]
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
    if w in [0,1,2,3]:
        pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ex1_excellent.index),['result_exam_1']] = 1
    elif w in [4,5,6,7]:
        pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ex2_excellent.index),['result_exam_1']] = 1
    else:
        pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ex3_excellent.index),['result_exam_1']] = 1

fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0
num_bins = 50
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
fig = plt.figure(figsize=(30,20),dpi=120)
ax = fig.subplots()
ax.set_xlabel('Eigenvalues', fontsize = 30)
ax.set_ylabel('IPR', fontsize = 30)
ax.tick_params(axis='both', which='major', labelsize=25)
ax.tick_params(axis='both', which='minor', labelsize=25)
ax.set_title('Inverse Participation Ratio week ' + str(w+1), fontsize = 30)
ax.grid()
# graph[countGraph].axhline(y=0, color='k')
# graph[countGraph].axvline(x=0, color='k')
eigenValueList = pca_result[11].explained_variance_
eigenVectorList = pca_result[11].components_
IPRlist = libRMT.IPRarray(eigenValueList,eigenVectorList)
ax.axhline(y=IPRlist['IPR'].mean(), color='k', label='mean value of IPR') 
ax.plot(IPRlist['eigenvalue'], IPRlist['IPR'], '-', color ='blue', label='IPR')
ax.legend(loc='upper right')
plt.show()

#outbounce select
a = libRMT.selectOutboundComponents(pcaDataWeeks[11],pca_result[11].explained_variance_)

#eigenvalues
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

import libRMT
libRMT.plotLoadings(1,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  
libRMT.plotLoadings(4,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  
libRMT.plotLoadings(11,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  

pc = 1
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
        
#--------------------------------------------------------------
#======== Code analysis -------------------------
#------------------------------------------
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# model= Doc2Vec.load(basePath + "ca116_2vecSize50.model")
model= Doc2Vec.load(basePath + "ca116_2vecSize50DM1.model")

a = model.docvecs[0]
a = np.array(list(a) + list(a))
taskList = dataUpload['task'].unique()
transitionDataMatrixWeeks[10].index[1]

def similarityBetweenTwoStudent(studentId1, studentId2, doc2vecModel, taskList):   
    vectorStudent1 = []
    vectorStudent2 = []
    for t in taskList:
        key1 = studentId1+'*'+t
        key2 = studentId2+'*'+t
        if (key1 in doc2vecModel.docvecs.index2entity) and (key2 in doc2vecModel.docvecs.index2entity):
            vectorStudent1 = vectorStudent1 + list(doc2vecModel.docvecs[key1])
            vectorStudent2 = vectorStudent2 + list(doc2vecModel.docvecs[key2])
    
    return cosine_similarity([vectorStudent1],[vectorStudent2])[0][0]

w = 10
studentIdList  = transitionDataMatrixWeeks[w].index
communities = aw10[3][5]  

similarityBetweenTwoStudent('u-efe2fecfc722d35c2f64de01db5c615051aca17f-2018', 'u-f36d0dcfcb3b05807c144548e64d26b8c2c5ece1-2018', model, taskList)

def getAllSimilarities(studentIdList, doc2vecModel, taskList):      
    allCodeDistance =  []
    count = 1
    for i in range(0,len(studentIdList)):  
        for j in range(i, len(studentIdList)):
           print(str(count)) 
           allCodeDistance.append(similarityBetweenTwoStudent(studentIdList[i], studentIdList[j], doc2vecModel, taskList))
           count = count + 1
    return allCodeDistance   
        

def compareCodeSimilarityScoreMean(communities, populationMeanList, model, taskList):
    result = []
    for c in communities:
        groupSimilarities =  getAllSimilarities(c.index, model, taskList) 
        ttest = stats.ttest_ind(allSimilariies,groupSimilarities)
        result.append((np.mean(groupSimilarities), ttest[0], ttest[1]))
    return result


        
allSimilariies =   getAllSimilarities(studentIdList, model, taskList) 

a = compareCodeSimilarityScoreMean(aw10[6][2], allSimilariies, model, taskList)

group3Similarities =  getAllSimilarities(communities[3].index, model, taskList) 
group2Similarities =  getAllSimilarities(communities[2].index, model, taskList) 
group1Similarities =  getAllSimilarities(communities[1].index, model, taskList) 
group4Similarities =  getAllSimilarities(communities[4].index, model, taskList) 
group0Similarities =  getAllSimilarities(communities[0].index, model, taskList) 

np.mean(allSimilariies)
np.mean(group3Similarities)
np.mean(group2Similarities)
np.mean(group1Similarities)
np.mean(group4Similarities)
np.mean(group0Similarities)

stats.ttest_ind(allSimilariies,group3Similarities)
stats.ttest_ind(allSimilariies,group2Similarities)
stats.ttest_ind(allSimilariies,group1Similarities)
stats.ttest_ind(allSimilariies,group4Similarities)
stats.ttest_ind(allSimilariies,group0Similarities)

#-------------------------------
w = 11
for c in activityDataMatrixWeeks_pageTypeWeek[w].columns:
    a1 = activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w].index.isin(ex3_excellent.index),[c]]
    b1 = activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w].index.isin(ex3_weak.index),[c]]
    t1, p1 = stats.ttest_ind(a1,b1)        
    # print('Week ' + str(w) + ':')
    if p1 <= 0.05:
        print(c + ': ' + 't-value: ' + str(t1) + ' p-value: ' + str(p1))
        print('-- Excellent: ' + str(a1.mean()[0]))
        print('-- Weak: ' + str(b1.mean()[0]))
        
#------------------------------------------------------
#===== find center students analysis -----------------
#-------------------------------------------
import graphLearning
centerNodeWeeks = []
for w in range(0,12):
    centerNodeWeeks.append(graphLearning.findAllCenterNodeinCommunities(communityListWeeks_cleaned_correlation[w][6], 
                                                                        graph_all_weeks_cleaned_correlation[w].graph ))

centerNodeWeeks[11]
weightsList = graphLearning.getAllWeights(graph_all_weeks_cleaned_correlation[11].graph)
sum(weightsList.values())/len(weightsList.values())

import community as community_louvain
partition = community_louvain.best_partition(graph_all_weeks_cleaned_correlation[11].graph)



#--------------------------------------------------------
#--------------- multy layer analysis with my own algorithm approach with network x------------
#----------------------------------------------------------------

#extract community 
clustersOverWeekList = []
for w in range(0,12):
    clustersOverWeekList.append(communityListWeeks[w][6])

nodeList = assessment.index    
superGraph = graphLearning.superGraphGeneration(clustersOverWeekList = clustersOverWeekList, weekWeights = [1,2,3,4,5,6,7,8,9,10,11,12], nodeList = nodeList)
a = superGraph.to_numpy().flatten()
plt.hist(a, bins=50)

np.percentile(a, 75)

superGraphInverse = 1./superGraph
superGraphInverse.replace([np.inf, -np.inf], 0, inplace=True)

G = nx.from_pandas_adjacency(superGraphInverse)
T=nx.minimum_spanning_tree(G)
print(sorted(T.edges(data=True)))


plt.figure(3,figsize=(30,30)) 
nx.draw(T, pos=nx.circular_layout(T))
plt.show()

communityDetectionSuperGraph = graphLearning.community_dection_graph(T, num_comms = len(T._node), mst=False)



#--------------------------------------------------------
#--------------- multy layer analysis with iGraph------------
#----------------------------------------------------------------
import graphLearning
iGraphsListWeeks = []
corrDistance_detoned = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList_cleaned_correlation[w]
    # risk_estimators = ml.portfolio_optimization.RiskEstimators()
    # tn_relation = transitionDataMatrixWeeks_directFollow_normalised_transposed[w].shape[0] / transitionDataMatrixWeeks_directFollow_normalised_transposed[w].shape[1]
    # The bandwidth of the KDE kernel
    # kde_bwidth = 0.01
    # detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=True)
    #detoned_matrix_byLib = matrix #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(matrix, index=matrix.index, columns=matrix.columns)
    distance_matrix = (0.5*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    g = graphLearning.createGraphFromCorrDistance_iGraph(distance_matrix)
    iGraphsListWeeks.append(g)

iGraphsListWeeks[11].vs()["labels"]

edge_weight = [w for w in iGraphsListWeeks[11].es["weight"]]

g = iGraphsListWeeks[11].spanning_tree(weights=edge_weight)
g.degree()
g.vs["label"]


graphLabels = [l for l in g.vs()["label"]]

fig, ax = plt.subplots(figsize=(15,15))
layout = g.layout("kk")
plot(g, target=ax, layout=layout, vertex_label = graphLabels)

# calculate dendrogram
dendrogram = g.community_edge_betweenness()
# convert it into a flat clustering
clusters = dendrogram.as_clustering()
# get the membership vector
membership = clusters.membership


import louvain

edge_weight11 = [w for w in iGraphsListWeeks[11].es["weight"]]
g11 = iGraphsListWeeks[11].spanning_tree(weights=edge_weight11)

edge_weight10 = [w for w in iGraphsListWeeks[10].es["weight"]]
g10 = iGraphsListWeeks[10].spanning_tree(weights=edge_weight10)

edge_weight9 = [w for w in iGraphsListWeeks[9].es["weight"]]
g09 = iGraphsListWeeks[9].spanning_tree(weights=edge_weight9)

membership, improvement = louvain.find_partition_temporal(
                             [g09,g10,g11],
                             louvain.CPMVertexPartition,
                             interslice_weight=0.1,
                            resolution_parameter=1)


#--------------- PROCESS ANALYSIS ------------------------------ #
#----------------------------------------------------------------------------#

import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
import pm4py

log_path = basePath + 'newEventLog_byWeek_ca116_2018.csv'
log_path = basePath + 'ca1162018_goodStudentsLog.csv'
log_path = basePath + 'ca1162018_goodStudentsLog_withTest.csv'
log_path = basePath + 'ca1162018_badStudentsLog.csv'
log_path = basePath + 'ca1162018_badStudentsLog_withTest.csv'

log_csv = pd.read_csv(log_path)
log_csv['pageTypeWeek1'] = log_csv['pageTypeWeek']
log_csv.columns
log_csv = log_csv.loc[:,['case:concept:name','pageTypeWeek','pageTypeWeek1','time:timestamp','org:resource','lifecycle:transition']]

log_csv.rename(columns={'pageTypeWeek':'concept:instance',
                          'pageTypeWeek1':'concept:name'
                         }, 
                  inplace=True)

log_csv = badStudentEventLog
log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
log_csv = log_csv.sort_values('time:timestamp')
event_log = log_converter.apply(log_csv)

# pm4py.view_events_distribution_graph(event_log, distr_type="days_week", format="svg")
# pm4py.view_dotted_chart(event_log, format="svg")
# pm4py.view_heuristics_net(heu_net)

heu_net = heuristics_miner.apply_heu(event_log, parameters={heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5})

gviz = hn_visualizer.apply(heu_net, variant=hn_visualizer.Variants.PYDOTPLUS)
hn_visualizer.view(gviz)

#---------------------------------------------------------------
#------------ EXPLORATORY ANALYSIS --------------------------
#-----------------------------------------------------------------

eventLog_ca116 = pd.read_csv(basePath + 'ca116_eventLog_nonfixed.csv')
# eventLog_ca116 = eventLog_ca116.drop([1160345])
eventLog_ca116 =eventLog_ca116.loc[eventLog_ca116['time:timestamp'] != ' ']
eventLog_ca116['time:timestamp'] = pd.to_datetime(eventLog_ca116['time:timestamp'])
eventLog_ca116 = eventLog_ca116.loc[:, ~eventLog_ca116.columns.str.contains('^Unnamed')]
# materials = eventLog_ca116.loc[:,['org:resource','concept:name','description']]
weeksEventLog = [g for n, g in eventLog_ca116.groupby(pd.Grouper(key='time:timestamp',freq='W'))]

eventLog_ca116_filtered = eventLog_ca116.loc[eventLog_ca116['description'].str.contains('.html|.py|#|/einstein/')]
eventLog_ca116_filtered['pageName'] = eventLog_ca116_filtered['description'].str.extract(r'([^\/][\S]+.html)', expand=False)
eventLog_ca116_filtered['pageName'] = eventLog_ca116_filtered['pageName'].str.replace('.web','')
eventLog_ca116_filtered = eventLog_ca116_filtered.drop(eventLog_ca116_filtered.loc[eventLog_ca116_filtered['concept:name'].isin(['click-0','click-1','click-2','click-3'])].index)

eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('correct|incorrect'),'pageName'] = 'Practice'
eventLog_ca116_filtered.loc[eventLog_ca116_filtered['description'].str.contains('^\/einstein\/'),'pageName'] = 'Practice'
eventLog_ca116_filtered['pageName'] = eventLog_ca116_filtered['pageName'].fillna('General')

a = eventLog_ca116_filtered.loc[eventLog_ca116_filtered['pageName'] == 'General']

eventLog_ca116_filtered.rename(columns={'concept:instance':'concept:instance1',
                                   'concept:name':'concept:name1',
                                   'case:concept:name' : 'case:concept:name1'},  inplace=True)
eventLog_ca116_filtered['concept:instance'] = eventLog_ca116_filtered['pageName']
eventLog_ca116_filtered['concept:name'] = eventLog_ca116_filtered['pageName']
eventLog_ca116_filtered['date'] = eventLog_ca116_filtered['time:timestamp'].dt.date

eventLog_ca116_filtered['case:concept:name'] = eventLog_ca116_filtered['date'].astype(str) + '-' + eventLog_ca116_filtered['org:resource'].astype(str)


# eventLog_ca116_filtered['concept:name'] = eventLog_ca116_filtered['pageName'] + '*' + eventLog_ca116_filtered['concept:name1']
# eventLog_ca116_filtered['concept:instance'] = eventLog_ca116_filtered['pageName'] + '*' + eventLog_ca116_filtered['concept:instance1']


weeksEventLog_filtered = [g for n, g in eventLog_ca116_filtered.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
weeksEventLog_filtered = weeksEventLog_filtered[1:13]

a = weeksEventLog_filtered[11].groupby(pd.Grouper(key='time:timestamp',freq='D'), pd.Grouper(key='org:resource'))

a = [g for n, g in weeksEventLog_filtered[11].groupby(pd.Grouper(key='time:timestamp',freq='D'))]
a1 = a[0].groupby(pd.Grouper(key='org:resource')).count()

a2 = weeksEventLog_filtered[11].groupby([pd.Grouper(key='time:timestamp',freq='D'), pd.Grouper(key='org:resource'),]).count()

a2 = a2.reset_index()
a2['Day'] = a2['time:timestamp'].dt.day_name()


# stats of all activities of the whole class

numberOfActivitiesByWeek_df = pd.DataFrame()
for w in range(0,12):
    weeksEventLog_filtered_typeOfstudent = weeksEventLog_filtered[w].loc[weeksEventLog_filtered[w]['org:resource'].isin(ex3_excellent.index),:]
    totalActivitiesByDate  =  [g for n, g in weeksEventLog_filtered_typeOfstudent.groupby(pd.Grouper(key='time:timestamp',freq='D'))]
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

color = ['lightcoral','indianred','brown','firebrick','forestgreen','limegreen','green','darkgreen','mediumpurple','rebeccapurple','blueviolet','indigo']
lines = numberOfActivitiesByWeek_df.plot.bar(figsize=(15,10), color=color)
lines.grid()
lines.set_ylim(0,40000)


# stats of activities of each students


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
numberOfActivitiesByWeekByStudent_df.loc[numberOfActivitiesByWeekByStudent_df['org:resource'].isin(ex3_excellent.index),['result']] = 'Higher Performing'
numberOfActivitiesByWeekByStudent_df.loc[numberOfActivitiesByWeekByStudent_df['org:resource'].isin(ex3_weak.index),['result']] = 'Lower Performing'
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

ax = numberOfActivitiesByWeekByStudent_df.boxplot(by=['Ordercustom','result'], figsize=(15,10), layout=(1,1), vert=0, showmeans=True)
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
numberOfActivitiesByStudentByPageType_df.loc[numberOfActivitiesByStudentByPageType_df['org:resource'].isin(ex3_excellent.index),['result']] = 'Higher Performing'
numberOfActivitiesByStudentByPageType_df.loc[numberOfActivitiesByStudentByPageType_df['org:resource'].isin(ex3_weak.index),['result']] = 'Lower Performing'
numberOfActivitiesByStudentByPageType_df = numberOfActivitiesByStudentByPageType_df.drop(numberOfActivitiesByStudentByPageType_df.loc[numberOfActivitiesByStudentByPageType_df['result'].isin(['NA'])].index)

ax = numberOfActivitiesByStudentByPageType_df.boxplot(by=['pageType','result'], figsize=(15,15),  vert=0, showmeans=True, fontsize=20)
ax.set_xlim(0, 10000)
#ax.set_title('Student activities by course learning material type')
#ax.set_yticklabels(['Mon - Higher','Mon - Lower', 'Tue - Higher', 'Tue - Lower', 'Wed - Higher', 'Wed - Lower', 'Thu - Higher', 'Thu - Lower', 'Fri - Higher', 'Fri - Lower', 'Sat - Higher', 'Sat - Lower', 'Sun - Higher', 'Sun - Lower'])

pageType = 'General'

stats.ttest_ind(numberOfActivitiesByStudentByPageType_df.loc[(numberOfActivitiesByStudentByPageType_df['result']=='Higher Performing')&(numberOfActivitiesByStudentByPageType_df['pageType']==pageType),['case:concept:name1']],
                    numberOfActivitiesByStudentByPageType_df.loc[(numberOfActivitiesByStudentByPageType_df['result']=='Lower Performing')&(numberOfActivitiesByStudentByPageType_df['pageType']==pageType),['case:concept:name1']], equal_var=False)


#community compare
excellentCommunity = numberOfActivitiesByStudentByPageType_df.loc[numberOfActivitiesByStudentByPageType_df['org:resource'].isin(list(aw10[6][5][2].index)+ list(aw10[6][5][1].index) +list(aw10[6][5][5].index) + list(aw10[6][5][0].index))]
excellentCommunity['CommunityType'] = 'Higher Performing'

badCommunity = numberOfActivitiesByStudentByPageType_df.loc[numberOfActivitiesByStudentByPageType_df['org:resource'].isin(list(aw10[6][5][3].index) + list(aw10[6][5][6].index) +list(aw10[6][5][7].index) + list(aw10[6][5][4].index))]
badCommunity['CommunityType'] = 'Lower Performing'


ax  = pd.concat([excellentCommunity, badCommunity]).boxplot(by=['pageType','CommunityType'], figsize=(15,15),  vert=0, showmeans=True, fontsize=20)
ax.set_xlim(0, 10000)



stats.ttest_ind(excellentCommunity.loc[excellentCommunity['pageType']==pageType,['case:concept:name1']],badCommunity.loc[badCommunity['pageType']==pageType,['case:concept:name1']], equal_var=False)


len(numberOfActivitiesByStudentByPageType_df.loc[(numberOfActivitiesByStudentByPageType_df['result']=='Lower Performing')&(numberOfActivitiesByStudentByPageType_df['pageType']==pageType),['case:concept:name1']])

len(excellentCommunity.loc[excellentCommunity['pageType']==pageType,['case:concept:name1']])

#eventlog extraction
temp = pd.concat(weeksEventLog_filtered_pageType)
temp['concept:instance'] = temp['pageTypeWeek']
temp['concept:name'] = temp['pageTypeWeek']
goodStudentEventLog = temp.loc[temp['org:resource'].isin(aw10[6][5][2].index),['case:concept:name','org:resource','lifecycle:transition','concept:instance','concept:name','time:timestamp']]

badStudentEventLog = temp.loc[temp['org:resource'].isin(aw10[6][5][3].index),['case:concept:name','org:resource','lifecycle:transition','concept:instance','concept:name','time:timestamp']]


eventLog_ca116_filtered.columns
a = pd.concat(weeksEventLog_filtered_pageType).head(5)


#transition data extraction
dfgGoodStudent = transitionDataMatrixWeeks[11].loc[transitionDataMatrixWeeks[11].index.isin(aw10[6][5][2].index)]
dfgGoodStudent = dfgGoodStudent.loc[:, (dfgGoodStudent != 0).any(axis=0)]


#material items and week day
pageName = 'Practice'
weeksEventLog_filtered_pageType = []
for w in range(0,12):
    tmp = weeksEventLog_filtered[w].merge(materialAccessedByWeek.loc[:,['pageType','ofWeek']], left_on=weeksEventLog_filtered[w].pageName, 
                                    right_on=materialAccessedByWeek.loc[:,['pageType']].index)    
    tmp.loc[tmp['pageName'] == 'Practice',['ofWeek']] = w
    tmp['pageTypeWeek'] = tmp['pageType'] + '_' + tmp['ofWeek'].astype(str)
    tmp['concept:name'] = tmp['pageTypeWeek'] + '*' + tmp['concept:instance1']
    tmp['concept:instance'] = tmp['pageTypeWeek'] + '*' + tmp['concept:instance1']
    weeksEventLog_filtered_pageType.append(tmp)
    
a = weeksEventLog_filtered_pageType[11]
a1 = weeksEventLog_filtered_pageType[11].groupby([ pd.Grouper(key='pageType'),pd.Grouper(key='time:timestamp',freq='D')]).count()
a1 = a1.reset_index()


fullLogpageType = pd.concat(weeksEventLog_filtered_pageType)
fullLogpageType = fullLogpageType.loc[:,['time:timestamp','pageType','case:concept:name']]
fullLogpageType['Day'] = fullLogpageType['time:timestamp'].dt.day_name()
fullLogpageType = fullLogpageType.groupby([ pd.Grouper(key='pageType'),pd.Grouper(key='time:timestamp',freq='D')]).agg({'case:concept:name':'count', 'Day':'first'})
fullLogpageType = fullLogpageType.reset_index()
fullLogpageType = fullLogpageType.loc[:,['pageType','Day','case:concept:name']]
fullLogpageType = fullLogpageType.groupby([pd.Grouper(key='Day'), pd.Grouper(key='pageType')]).sum()
fullLogpageType = fullLogpageType.unstack()

color = ['lightcoral','indianred','brown','firebrick','forestgreen','limegreen','green','darkgreen','mediumpurple','rebeccapurple','blueviolet','indigo']
lines = fullLogpageType.plot.bar(figsize=(15,10))
lines.grid()
    
numberOfActivitiesByWeekByItems_list = []
for w in range(0,12):
    itemActivitiesByDate = weeksEventLog_filtered_pageType[w].groupby([ pd.Grouper(key='pageType'),pd.Grouper(key='time:timestamp',freq='D')]).count()
    itemActivitiesByDate = itemActivitiesByDate.reset_index()
    itemActivitiesByDate['Day'] = itemActivitiesByDate['time:timestamp'].dt.day_name()
    itemActivitiesByDate = itemActivitiesByDate.loc[:,['Day','pageType','case:concept:name1']]
    numberOfActivitiesByWeekByItems_list.append(itemActivitiesByDate)


pageType = 'Labsheet'
numberOfActivitiesByWeekByItems_df = pd.DataFrame()
for w in range(0,12):
    tmp = numberOfActivitiesByWeekByItems_list[w].loc[numberOfActivitiesByWeekByItems_list[w]['pageType'] == pageType,:]
    tmp1 = tmp['case:concept:name1'].to_list()
    if len(tmp1) < 7:
        for i in range(0, -len(tmp1) + 7):
            tmp1.append(0)
    
    numberOfActivitiesByWeekByItems_df['week' + str(w+1)] = tmp1
    
numberOfActivitiesByWeekByItems_df['day'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
numberOfActivitiesByWeekByItems_df = numberOfActivitiesByWeekByItems_df.set_index('day')

color = ['lightcoral','indianred','brown','firebrick','forestgreen','limegreen','green','darkgreen','mediumpurple','rebeccapurple','blueviolet','indigo']
lines = numberOfActivitiesByWeekByItems_df.plot.bar(figsize=(15,10), color=color, title=pageType)
lines.grid()


import seaborn as sns

fig, ax = plt.subplots(figsize=(15,10))
sns.barplot(x="Day", y="case:concept:name1", ax=ax, data=numberOfActivitiesByWeekByItems_df, title=pageType)
ax.grid()


numberOfActivitiesByWeekByItems_df.boxplot(by='Day', figsize=(15,10), layout=(1,1))

#for pageType
for week in range(0,12):
    if week in [0,1,2,3]:
        exellent = ex1_excellent.index
        weak = ex1_weak.index
    elif week in [4,5,6,7]:
        exellent = ex2_excellent.index
        w = ex2_weak.index        
    else:
        exellent = ex3_excellent.index
        weak = ex3_weak.index  
    activityDataMatrixWeeks_pageType[week]['result'] = 'NA'
    activityDataMatrixWeeks_pageType[week].loc[activityDataMatrixWeeks_pageType[week].index.isin(exellent),['result']] = 'Higher performing'
    activityDataMatrixWeeks_pageType[week].loc[activityDataMatrixWeeks_pageType[week].index.isin(weak),['result']] = 'Lower performing'
    activityDataMatrixWeeks_pageType[week] = activityDataMatrixWeeks_pageType[week].drop(activityDataMatrixWeeks_pageType[week][activityDataMatrixWeeks_pageType[week]['result'] == 'NA'].index)

for week in range(0,12):    
    print('Week: ' + str(week) + '...')
    for col1 in ['Practice','General','Lecture','Labsheet']:
            p = stats.ttest_ind(activityDataMatrixWeeks_pageType[week].loc[activityDataMatrixWeeks_pageType[week]['result']=='Higher performing',[col1]],
                                activityDataMatrixWeeks_pageType[week].loc[activityDataMatrixWeeks_pageType[week]['result']=='Lower performing',[col1]], equal_var=False)
            print(col1 + ':' + str(p[0]) + ' - ' +str(p[1]))

#accumulate page type
for week in range(0,12):
    if week in [0,1,2,3]:
        exellent = ex1_excellent.index
        weak = ex1_weak.index
    elif week in [4,5,6,7]:
        exellent = ex2_excellent.index
        w = ex2_weak.index        
    else:
        exellent = ex3_excellent.index
        weak = ex3_weak.index  
    activityDataMatrixWeeks_pageType[week]['result'] = 'NA'
    activityDataMatrixWeeks_pageType[week].loc[activityDataMatrixWeeks_pageType[week].index.isin(exellent),['result']] = 'Higher performing'
    activityDataMatrixWeeks_pageType[week].loc[activityDataMatrixWeeks_pageType[week].index.isin(weak),['result']] = 'Lower performing'

for week in range(0,12):    
    print('Week: ' + str(week) + '...')
    for col1 in ['Practice','General','Lecture','Labsheet']:
            p = stats.ttest_ind(activityDataMatrixWeeks_pageType[week].loc[activityDataMatrixWeeks_pageType[week]['result']==1,[col1]],
                                activityDataMatrixWeeks_pageType[week].loc[activityDataMatrixWeeks_pageType[week]['result']==0,[col1]], equal_var=False)
            print(col1 + ':' + str(p[0]) + ' - ' +str(p[1]))


#for pageType
for week in range(0,12):
    if week in [0,1,2,3]:
        exellent = ex1_excellent.index
        weak = ex1_weak.index
    elif week in [4,5,6,7]:
        exellent = ex2_excellent.index
        w = ex2_weak.index        
    else:
        exellent = ex3_excellent.index
        weak = ex3_weak.index  
    activityDataMatrixWeeks_pageTypeWeek[week]['result'] = 'NA'
    activityDataMatrixWeeks_pageTypeWeek[week].loc[activityDataMatrixWeeks_pageTypeWeek[week].index.isin(exellent),['result']] = 'Higher performing'
    activityDataMatrixWeeks_pageTypeWeek[week].loc[activityDataMatrixWeeks_pageTypeWeek[week].index.isin(weak),['result']] = 'Lower performing'
    activityDataMatrixWeeks_pageTypeWeek[week] = activityDataMatrixWeeks_pageTypeWeek[week].drop(activityDataMatrixWeeks_pageTypeWeek[week][activityDataMatrixWeeks_pageTypeWeek[week]['result'] == 'NA'].index)

for week in range(0,12):    
    print('Week: ' + str(week) + '...')
    for col1 in ['Practice','General','Lecture','Labsheet']:
            p = stats.ttest_ind(activityDataMatrixWeeks_pageTypeWeek[week].loc[activityDataMatrixWeeks_pageTypeWeek[week]['result']=='Higher performing',[col1]],
                                activityDataMatrixWeeks_pageTypeWeek[week].loc[activityDataMatrixWeeks_pageTypeWeek[week]['result']=='Lower performing',[col1]], equal_var=False)
            print(col1 + ':' + str(p[0]) + ' - ' +str(p[1]))

#accumulate page type week
for week in range(0,12):
    if week in [0,1,2,3]:
        exellent = ex1_excellent.index
        weak = ex1_weak.index
    elif week in [4,5,6,7]:
        exellent = ex2_excellent.index
        w = ex2_weak.index        
    else:
        exellent = ex3_excellent.index
        weak = ex3_weak.index  
    activityDataMatrixWeeks_pageTypeWeek[week]['result'] = 'NA'
    activityDataMatrixWeeks_pageTypeWeek[week].loc[activityDataMatrixWeeks_pageTypeWeek[week].index.isin(exellent),['result']] = 'Higher performing'
    activityDataMatrixWeeks_pageTypeWeek[week].loc[activityDataMatrixWeeks_pageTypeWeek[week].index.isin(weak),['result']] = 'Lower performing'

for week in range(0,12):    
    print('Week: ' + str(week) + '...')
    for col1 in ['Practice','General','Lecture','Labsheet']:
            p = stats.ttest_ind(activityDataMatrixWeeks_pageTypeWeek[week].loc[activityDataMatrixWeeks_pageTypeWeek[week]['result']=='Higher performing',[col1]],
                                activityDataMatrixWeeks_pageTypeWeek[week].loc[activityDataMatrixWeeks_pageTypeWeek[week]['result']=='Lower performing',[col1]], equal_var=False)
            print(col1 + ':' + str(p[0]) + ' - ' +str(p[1]))



boxplot = activityDataMatrixWeeks_pageTypeWeek[11].boxplot(by='result', figsize=(26,5), layout=(1,1))

boxplot = activityDataMatrixWeeks_pageType[11].boxplot(by='result', figsize=(20,5), layout=(1,4))




























