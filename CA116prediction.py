import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import dataProcessing
import FCAMiner
# import pyRMT
import libRMT
import graphLearning
import warnings
from node2vec import Node2Vec
import PredictionResult
warnings.filterwarnings("ignore")
# sns.set()
# sns.set_style("whitegrid", {"axes.facecolor": ".9"})
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import mlfinlab as ml
from mlfinlab.networks.mst import MST
basePath = 'D:\\Dataset\\PhD\\'


#read ca116 2018 data
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

ex2_excellent = assessment2A.loc[(assessment2A['perCorrect2A'] <= 1)&(assessment2A['perCorrect2A'] >= 0.4)]
ex2_weak = assessment2A.loc[(assessment2A['perCorrect2A'] >= 0) & (assessment2A['perCorrect2A'] < 0.4)]

ex3_excellent = assessment3A.loc[(assessment3A['perCorrect3A'] <= 1)&(assessment3A['perCorrect3A'] >= 0.4)]
ex3_weak = assessment3A.loc[(assessment3A['perCorrect3A'] >= 0) & (assessment3A['perCorrect3A'] < 0.4)]

nonExUpload = dataUpload.drop(dataUpload.loc[dataUpload['task'].str.match('ex')].index)
nonExUploadByWeek = [g for n, g in nonExUpload.groupby(pd.Grouper(key='date',freq='W'))]

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
    cummulativeExerciseWeeks.append(cummulativeResult)

transitionDataMatrixWeeks = []
for w in range(0,12):
    temp = pd.read_csv(basePath + 'transitionMatrixStorage_new/transitionDataMatrixWeeks_direct_accumulated_pageTypeWeekAction_w' + str(w) + '.csv', index_col=0)
    if w in [0,1,2,3]:
        studentList = assessment1A.index
    elif w in [4,5,6,7]:
        studentList = assessment2A.index     
    else:
        studentList = assessment3A.index
    temp = temp.loc[temp.index.isin(studentList)]
    # temp = temp.drop(['Practice_0-Practice_0'],axis=1)
    # if w == 1:
    #     temp = temp.drop([8])
    transitionDataMatrixWeeks.append(temp) 


activityDataMatrixWeeks_pageTypeWeek = []    
for w in range(0,12):
    temp = pd.read_csv(basePath + 'transitionMatrixStorage_new/activityDataMatrixWeeks_pageTypeWeek_newPractice_w' + str(w) + '.csv', index_col=0)
    if w in [0,1,2,3]:
        studentList = assessment1A.index
    elif w in [4,5,6,7]:
        studentList = assessment2A.index     
    else:
        studentList = assessment3A.index
    temp = temp.loc[temp.index.isin(studentList)]
    # temp = temp.drop(['Practice_0-Practice_0'],axis=1)
    # if w == 1:
    #     temp = temp.drop([8])
    activityDataMatrixWeeks_pageTypeWeek.append(temp) 


activityDataMatrixWeeks_pageTypeWeek[11].hist(bins=50,density=True)

    
#import data for ca1162019:
dataUpload_2019 = pd.read_csv(basePath + 'ca116_uploads_2019.csv')
dataUpload_2019['date'] = pd.to_datetime(dataUpload_2019.date)

exUpload_2019 = dataUpload_2019.loc[dataUpload_2019['task'].str.match('ex')]

ex1_2019 = exUpload_2019.loc[exUpload_2019['task'].str.match('ex1')]
ex1_2019 = ex1_2019.sort_values(by=['user','task'])
ex1_2019 = ex1_2019.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
ex2_2019 = exUpload_2019.loc[exUpload_2019['task'].str.match('ex2')]
ex2_2019 = ex2_2019.sort_values(by=['user','task'])
ex2_2019 = ex2_2019.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
ex3_2019 = exUpload_2019.loc[exUpload_2019['task'].str.match('ex3')]
ex3_2019 = ex3_2019.sort_values(by=['user','task'])
ex3_2019 = ex3_2019.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()

assessment1A_2019 = dataProcessing.assessmentConstruction(ex1_2019,4)
assessment1A_2019['adjustedPerformance'] = (assessment1A_2019['perCorrect'] + assessment1A_2019['perPassed'])/2
assessment2A_2019 = dataProcessing.assessmentConstruction(ex2_2019,4)
assessment2A_2019['adjustedPerformance'] = (assessment2A_2019['perCorrect'] + assessment2A_2019['perPassed'])/2
assessment3A_2019 = dataProcessing.assessmentConstruction(ex3_2019,4)
assessment3A_2019['adjustedPerformance'] = (assessment3A_2019['perCorrect'] + assessment3A_2019['perPassed'])/2

assessment1A_2019.rename(columns={'correct':'correct1A',
                          'perCorrect':'perCorrect1A',
                          'failed':'failed1A',
                            'passed':'passed1A',
                            'perPassed':'perPassed1A',
                            'testSubmitted':'testSubmitted1A',
                            'adjustedPerformance':'adjustedPerformance1A'}, 
                  inplace=True)
assessment2A_2019.rename(columns={'correct':'correct2A',
                          'perCorrect':'perCorrect2A',
                          'failed':'failed2A',
                            'passed':'passed2A',
                            'perPassed':'perPassed2A',
                            'testSubmitted':'testSubmitted2A',
                            'adjustedPerformance':'adjustedPerformance2A'}, 
                  inplace=True)
assessment3A_2019.rename(columns={'correct':'correct3A',
                          'perCorrect':'perCorrect3A',
                            'failed':'failed3A',
                            'passed':'passed3A',
                            'perPassed':'perPassed3A',
                            'testSubmitted':'testSubmitted3A',
                            'adjustedPerformance':'adjustedPerformance3A'}, 
                  inplace=True)
assessment1A_2019 = assessment1A_2019.set_index(['user'])
assessment2A_2019 = assessment2A_2019.set_index(['user'])
assessment3A_2019 = assessment3A_2019.set_index(['user'])

assessment_2019 = pd.concat([assessment1A_2019,assessment2A_2019,assessment3A_2019], axis=1)
assessment_2019 = assessment_2019.fillna(0)

assessment_2019['grade'] = (assessment_2019['perCorrect1A']+assessment_2019['perCorrect2A']+assessment_2019['perCorrect3A'])/3
assessment_2019['perPassed'] = (assessment_2019['passed1A'] + assessment_2019['passed2A'] + assessment_2019['passed3A'])/(assessment_2019['passed1A'] + assessment_2019['passed2A'] + assessment_2019['passed3A'] 
                        + assessment_2019['failed1A']+ assessment_2019['failed2A']+ assessment_2019['failed3A'])

ex1_excellent_2019 = assessment1A_2019.loc[(assessment1A_2019['perCorrect1A'] <= 1) & (assessment1A_2019['perCorrect1A'] >= 0.4)]
ex1_weak_2019 = assessment1A_2019.loc[(assessment1A_2019['perCorrect1A'] >= 0) & (assessment1A_2019['perCorrect1A'] < 0.4)]

ex2_excellent_2019 = assessment2A_2019.loc[(assessment2A_2019['perCorrect2A'] <= 1)&(assessment2A_2019['perCorrect2A'] >= 0.4)]
ex2_weak_2019 = assessment2A_2019.loc[(assessment2A_2019['perCorrect2A'] >= 0) & (assessment2A_2019['perCorrect2A'] < 0.4)]

ex3_excellent_2019 = assessment3A_2019.loc[(assessment3A_2019['perCorrect3A'] <= 1)&(assessment3A_2019['perCorrect3A'] >= 0.4)]
ex3_weak_2019 = assessment3A_2019.loc[(assessment3A_2019['perCorrect3A'] >= 0) & (assessment3A_2019['perCorrect3A'] < 0.4)]

nonExUpload_2019 = dataUpload_2019.drop(dataUpload_2019.loc[dataUpload_2019['task'].str.match('ex')].index)
nonExUploadByWeek_2019 = [g for n, g in nonExUpload_2019.groupby(pd.Grouper(key='date',freq='W'))]

workingWeekExcercise = []
cummulativeExerciseWeeks_2019 = []
for week in range(0,12):       
    workingWeekExcercise.append(nonExUploadByWeek_2019[week])
    practiceResult = pd.concat(workingWeekExcercise) #nonExUploadByWeek[week] 

    #adjust number of correct: For each task, number of correct submission/number of submission for that task
    practiceResultSum = practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
    practiceResultSum['correct_adjusted'] = practiceResultSum['correct']/practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).count()['correct']
    cummulativeResult = practiceResultSum.reset_index().groupby([pd.Grouper(key='user')]).sum()
    
    # cummulativeResult = practiceResultSum.groupby([pd.Grouper(key='user')]).sum()
    cummulativeResult['cumm_practice'] = cummulativeResult['correct']/practiceResult.groupby([pd.Grouper(key='user')]).count()['date']
    cummulativeResult['successPassedRate'] = cummulativeResult['passed']/(cummulativeResult['passed'] + cummulativeResult['failed'])
    cummulativeExerciseWeeks_2019.append(cummulativeResult)

transitionDataMatrixWeeks_2019 = []
for w in range(0,12):
    temp = pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1162019_transitionDataMatrixWeeks_direct_accumulated_pageTypeWeekAction_w' + str(w) + '.csv', index_col=0)
    if w in [0,1,2,3]:
        studentList = assessment1A_2019.index
    elif w in [4,5,6,7]:
        studentList = assessment2A_2019.index     
    else:
        studentList = assessment3A_2019.index
    temp = temp.loc[temp.index.isin(studentList)]
    # temp = temp.drop(['Practice_0-Practice_0'],axis=1)
    # if w == 1:
    #     temp = temp.drop([8])
    transitionDataMatrixWeeks_2019.append(temp) 
    
activityDataMatrixWeeks_pageTypeWeek_2019 = []    
for w in range(0,12):
    temp = pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1162019_activityDataMatrixWeeks_pageTypeWeek_newPractice_w' + str(w) + '.csv', index_col=0)
    if w in [0,1,2,3]:
        studentList = assessment1A_2019.index
    elif w in [4,5,6,7]:
        studentList = assessment2A_2019.index     
    else:
        studentList = assessment3A_2019.index
    temp = temp.loc[temp.index.isin(studentList)]
    # temp = temp.drop(['Practice_0-Practice_0'],axis=1)
    # if w == 1:
    #     temp = temp.drop([8])
    activityDataMatrixWeeks_pageTypeWeek_2019.append(temp)     
    
#merge data
transitionDataMatrixWeeks_20182019 = []
for w in range(0,12):
    transitionDataMatrixWeeks_20182019.append(pd.concat([transitionDataMatrixWeeks[w],transitionDataMatrixWeeks_2019[w]]))
    transitionDataMatrixWeeks_20182019[w] = transitionDataMatrixWeeks_20182019[w].fillna(0)


for w in range(0,12):
    transitionDataMatrixWeeks_20182019[w] = transitionDataMatrixWeeks_20182019[w].loc[:, (transitionDataMatrixWeeks_20182019[w] != 0).any(axis=0)]

    
cummulativeExerciseWeeks_20182019 = []
for w in range(0,12):
    cummulativeExerciseWeeks_20182019.append(pd.concat([cummulativeExerciseWeeks[w],cummulativeExerciseWeeks_2019[w]]))

ex1_excellent_20182019 = pd.concat([ex1_excellent, ex1_excellent_2019])
ex1_weak_20182019 = pd.concat([ex1_weak, ex1_weak_2019])
ex2_excellent_20182019 = pd.concat([ex2_excellent, ex2_excellent_2019])
ex2_weak_20182019 = pd.concat([ex2_weak, ex2_weak_2019])
ex3_excellent_20182019 = pd.concat([ex3_excellent, ex3_excellent_2019])
ex3_weak_20182019 = pd.concat([ex3_weak, ex3_weak_2019])
    
#normalised and standardised
transitionDataMatrixWeeks_20182019_directFollow_standardised = []    
for w in range(0,12):
    transitionDataMatrixWeeks_20182019_directFollow_standardised.append(dataProcessing.normaliseData(transitionDataMatrixWeeks_20182019[w].T))

transitionDataMatrixWeeks_20182019_directFollow_normalised = []    
for w in range(0,12):
    transitionDataMatrixWeeks_20182019_directFollow_normalised.append(dataProcessing.normaliseData(transitionDataMatrixWeeks_20182019[w].T, 'normalised'))



#transpose transition data matrix
transitionDataMatrixWeeks_20182019_transposed = []
# transitionDataMatrixWeeks_20182019_directFollow_normalised_transposed = []
# transitionDataMatrixWeeks_20182019_directFollow_standardised_transposed = []
for w in range(0,12):
    transitionDataMatrixWeeks_20182019_transposed.append(transitionDataMatrixWeeks_20182019[w].T)
    # transitionDataMatrixWeeks_20182019_directFollow_normalised_transposed.append(transitionDataMatrixWeeks_20182019_directFollow_normalised[w].T)
    # transitionDataMatrixWeeks_20182019_directFollow_standardised_transposed.append(transitionDataMatrixWeeks_20182019_directFollow_standardised[w].T)  

#merge data 2018 2019 for activity data matrix
activityDataMatrixWeeks_20182019 = []
for w in range(0,12):
    activityDataMatrixWeeks_20182019.append(pd.concat([activityDataMatrixWeeks_pageTypeWeek[w],activityDataMatrixWeeks_pageTypeWeek_2019[w]]))
    activityDataMatrixWeeks_20182019[w] = activityDataMatrixWeeks_20182019[w].fillna(0)

activityDataMatrixWeeks_20182019_standardised = []    
for w in range(0,12):
    activityDataMatrixWeeks_20182019_standardised.append(dataProcessing.normaliseData(activityDataMatrixWeeks_20182019[w]))

activityDataMatrixWeeks_20182019_normalised = []    
for w in range(0,12):
    activityDataMatrixWeeks_20182019_normalised.append(dataProcessing.normaliseData(activityDataMatrixWeeks_20182019[w], 'normalised'))

activityDataMatrixWeeks_20182019_standardised[11].hist(bins=50)

#transpose activity data matrix
activityDataMatrixWeeks_20182019_transposed = []
activityDataMatrixWeeks_20182019_normalised_transposed = []
activityDataMatrixWeeks_20182019_standardised_transposed = []
for w in range(0,12):
    activityDataMatrixWeeks_20182019_transposed.append(activityDataMatrixWeeks_20182019[w].T)
    activityDataMatrixWeeks_20182019_normalised_transposed.append(activityDataMatrixWeeks_20182019_normalised[w].T)
    activityDataMatrixWeeks_20182019_standardised_transposed.append(activityDataMatrixWeeks_20182019_standardised[w].T)  


    
# correlation processing    
corrList = []
corrDistanceList = []
for w in range(0,12):
    corrTemp = transitionDataMatrixWeeks_20182019_directFollow_standardised[w].corr()
    corrList.append(corrTemp)
    corrDistance = (2*(1 - corrTemp)).apply(np.sqrt)
    corrDistanceList.append(corrDistance)

transitionDataMatrixWeeks_20182019_transposed[w].loc[:,transitionDataMatrixWeeks_20182019_transposed[w].columns[0:2]].hist()
    
#correlation processing    
corrList_dataNormalised = []
# corrDistanceList_dataNormalised = []
for w in range(0,12):
    corrTemp = transitionDataMatrixWeeks_20182019_directFollow_normalised[w].corr()
    corrList_dataNormalised.append(corrTemp)
    # corrDistance = (0.5*(1 - corrTemp)).apply(np.sqrt)
    # corrDistanceList_dataNormalised.append(corrDistance)
    
    
graph_all_weeks = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList[w]
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    tn_relation = transitionDataMatrixWeeks_20182019_transposed[w].shape[0] / transitionDataMatrixWeeks_20182019_transposed[w].shape[1]
    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01
    # Finding the Вe-noised Сovariance matrix
    # denoised_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth)
    # denoised_matrix_byLib = pd.DataFrame(denoised_matrix_byLib, index=matrix.index, columns=matrix.columns) denoise_method='target_shrink',
    
    detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=True)
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
    tn_relation = transitionDataMatrixWeeks_20182019_directFollow_normalised_transposed[w].shape[0] / transitionDataMatrixWeeks_20182019_directFollow_normalised_transposed[w].shape[1]
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
    


graph_all_weeks_msf = []
corrDistance_detoned = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList_dataNormalised[w]
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    tn_relation = transitionDataMatrixWeeks_20182019_directFollow_normalised_transposed[w].shape[0] / transitionDataMatrixWeeks_20182019_directFollow_normalised_transposed[w].shape[1]
    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01
    detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=True)
    # detoned_matrix_byLib = matrix #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)
    distance_matrix = (2*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    corrDistance_detoned.append(distance_matrix)
    g = graphLearning.createGraphFromCorrDistance(distance_matrix)
    graph_all_weeks_msf.append(g)
    
graph_all_weeks_msf_corrList = []
corrDistance_detoned_corrList = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList[w]
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    tn_relation = transitionDataMatrixWeeks_20182019_transposed[w].shape[0] / transitionDataMatrixWeeks_20182019_transposed[w].shape[1]
    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01
    detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=True)
    # detoned_matrix_byLib = matrix #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)
    distance_matrix = (2*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    corrDistance_detoned_corrList.append(distance_matrix)
    g = graphLearning.createGraphFromCorrDistance(distance_matrix)
    graph_all_weeks_msf_corrList.append(g)

graph_all_weeks_msf_corrList = []
corrDistance_detoned_corrList = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList[w]
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    tn_relation = transitionDataMatrixWeeks_20182019_transposed[w].shape[0] / transitionDataMatrixWeeks_20182019_transposed[w].shape[1]
    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01
    detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=True)
    # detoned_matrix_byLib = matrix #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)
    distance_matrix = (2*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    corrDistance_detoned_corrList.append(distance_matrix)
    g = graphLearning.createGraphFromCorrDistance(distance_matrix)
    graph_all_weeks_msf_corrList.append(g)
    
graph_all_weeks_msf_corrList_notClean = []
corrDistance_detoned_corrList_notClean = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList[w]
    
    distance_matrix = (2*(1 - matrix)).apply(np.sqrt)
    corrDistance_detoned_corrList_notClean.append(distance_matrix)
    g = graphLearning.createGraphFromCorrDistance(distance_matrix)
    graph_all_weeks_msf_corrList_notClean.append(g)


#graph for activity data matrix

# correlation for activity matrix
corrList_activityDataMatrix = []
corrDistanceList_activityDataMatrix = []
for w in range(0,12):
    corrTemp = activityDataMatrixWeeks_20182019_transposed[w].corr()
    corrList_activityDataMatrix.append(corrTemp)
    corrDistance = (2*(1 - corrTemp)).apply(np.sqrt)
    corrDistanceList_activityDataMatrix.append(corrDistance)
    
graph_all_weeks = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList_activityDataMatrix[w]
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    tn_relation = activityDataMatrixWeeks_20182019_transposed[w].shape[0] / activityDataMatrixWeeks_20182019_transposed[w].shape[1]
    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01
    # Finding the Вe-noised Сovariance matrix
    # denoised_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth)
    # denoised_matrix_byLib = pd.DataFrame(denoised_matrix_byLib, index=matrix.index, columns=matrix.columns) denoise_method='target_shrink',
    
    detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=True)
    # detoned_matrix_byLib = matrix #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)
    distance_matrix = (2*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    graphBuild = MST(distance_matrix, 'distance')
    # graphBuild = nx.from_pandas_adjacency(distance_matrix)   
    graph_all_weeks.append(graphBuild)

graph_all_weeks_not_cleaned = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList_activityDataMatrix[w]
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    tn_relation = transitionDataMatrixWeeks_20182019_directFollow_normalised_transposed[w].shape[0] / transitionDataMatrixWeeks_20182019_directFollow_normalised_transposed[w].shape[1]
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


#----------------------------------------------
#Node embedding analysis    
#----------------------------------------------
    
node_embeddings_weeks = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    node2vec = Node2Vec(graph_all_weeks[w].graph, dimensions=128, walk_length=10, num_walks=20, p=0.5, q=1)
    model = node2vec.fit(window=8, min_count=1)    
    nodeList = model.wv.index2word
    node_embeddings = [list(model.wv.get_vector(n)) for n in nodeList] # numpy.ndarray of size number of nodes times embeddings dimensionality        
    nodeList = list(map(str,model.wv.index2word)) #convert string node to int node
    node_embeddings = pd.DataFrame(node_embeddings, index = nodeList)

    # scaler = StandardScaler()
    # node_embeddings = pd.DataFrame(scaler.fit_transform(node_embeddings), index=node_embeddings.index)
    node_embeddings_weeks.append(node_embeddings)

for w in range(0,12):
    node_embeddings_weeks[w] = node_embeddings_weeks[w].merge(cummulativeExerciseWeeks_20182019[w]['correct'],left_on=node_embeddings_weeks[w].index,
                                            right_on=cummulativeExerciseWeeks_20182019[w]['correct'].index).set_index('key_0')

node_embeddings_2d_df_weeks = []
for w in range(0,12):
    if w in [0,1,2,3]:
        excellent = ex1_excellent_20182019.index
        weak = ex1_weak_20182019.index
    elif w in [4,5,6,7]:
        excellent = ex2_excellent_20182019.index
        weak = ex2_weak_20182019.index        
    else:
        excellent = ex3_excellent_20182019.index
        weak = ex3_weak_20182019.index        
    
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

#----------------------------------------------
#Node embedding analysis uncleaned data
#----------------------------------------------
    
node_embeddings_weeks_uncleaned = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    node2vec = Node2Vec(graph_all_weeks_not_cleaned[w].graph, dimensions=64, walk_length=8, num_walks=15, p=0.1, q=1)
    model = node2vec.fit(window=8, min_count=1)    
    nodeList = model.wv.index2word
    node_embeddings = [list(model.wv.get_vector(n)) for n in nodeList] # numpy.ndarray of size number of nodes times embeddings dimensionality        
    nodeList = list(map(str,model.wv.index2word)) #convert string node to int node
    node_embeddings = pd.DataFrame(node_embeddings, index = nodeList)

    # scaler = StandardScaler()
    # node_embeddings = pd.DataFrame(scaler.fit_transform(node_embeddings), index=node_embeddings.index)
    node_embeddings_weeks_uncleaned.append(node_embeddings)

for w in range(0,12):
    node_embeddings_weeks_uncleaned[w] = node_embeddings_weeks_uncleaned[w].merge(cummulativeExerciseWeeks_20182019[w]['correct'],left_on=node_embeddings_weeks_uncleaned[w].index,
                                            right_on=cummulativeExerciseWeeks_20182019[w]['correct'].index).set_index('key_0')

node_embeddings_2d_df_weeks = []
for w in range(0,12):
    if w in [0,1,2,3]:
        excellent = ex1_excellent_20182019.index
        weak = ex1_weak_20182019.index
    elif w in [4,5,6,7]:
        excellent = ex2_excellent_20182019.index
        weak = ex2_weak_20182019.index        
    else:
        excellent = ex3_excellent_20182019.index
        weak = ex3_weak_20182019.index        
    
    # excellent = list(map(str,exellentPractice[w]))
    # weak = list(map(str,weakPractice[w]))

    
    # tsne = TSNE(n_components=2)
    # node_embeddings_2d = tsne.fit_transform(node_embeddings_weeks[w])
    pca = PCA(n_components=2)
    node_embeddings_2d = pca.fit_transform(node_embeddings_weeks_uncleaned[w])
    node_embeddings_2d_df = pd.DataFrame(node_embeddings_2d, index = node_embeddings_weeks_uncleaned[w].index)
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


#--------------- PREDCTION for node embeddings --------------------------------------------------#
#---------------------------------------------------------------------------------#

#data preparation for prediction
#node_embeddings week

workingWeekExcercise = []
prediction_transition = []

for week in range(0,12):
    print('Predicting for Week ...' + str(week))
    if week in [0,1,2,3]:
        excellent = ex1_excellent_20182019.index
        weak = ex1_weak_20182019.index
    elif week in [4,5,6,7]:
        excellent = ex2_excellent_20182019.index
        weak = ex2_weak_20182019.index        
    else:
        excellent = ex3_excellent_20182019.index
        weak = ex3_weak_20182019.index           
   
    cummulativeResult = []
    
    dataForPrediction = node_embeddings_weeks
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

title_transition = 'Graph embeddings - Node2Vec - 2018 and 2019 by weeks data - Sum - 1 as passed, 0 as failed with practice data'
algorithmList = []
# algorithmList = []
PredictionResult.algorithmComparisonGraph('recall',predictionReport_transition,algorithmList, title_transition)

#--------------- PREDCTION with activity data--------------------------------------------------#
#---------------------------------------------------------------------------------#


import FCAMiner
pca_result = []
pcaDataWeeks = []
columnsReturn2 = []
for w in range(0,12):
    # tempData = transitionDataMatrixWeeks[w].loc[:,columns]
    tempData = activityDataMatrixWeeks_20182019[w]
    # tempData = tempData.merge(prediction_transition[w+1]['data']['successPassedRate'], left_on = tempData.index, right_on=prediction_transition[w+1]['data']['successPassedRate'].index).set_index('key_0')
    temp = FCAMiner.PCAcohortToValue(tempData)
    temp1 = temp[1]
    pcaResult = temp[0]
    # temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
    pcaDataWeeks.append(temp1)
    pca_result.append(pcaResult)
    columnsReturn2.append(temp[2])    

#filtered PCA data
dataForPrediction_outbound = []
for week in range(0,12):
    dataForPrediction_outbound.append(libRMT.selectOutboundComponents(pcaDataWeeks[week],pca_result[week].explained_variance_))

dataForPrediction_outbound_upper = []
for week in range(0,12):
    dataForPrediction_outbound_upper.append(libRMT.selectOutboundComponents(pcaDataWeeks[week],pca_result[week].explained_variance_,'upper'))


#clean first eigen effect data with regression
# dataForPrediction = []    
# for week in range(0,12):
#     # outBoundComponents = []
#     # for c in dataForPrediction_outbound[week].columns:
#     #     outBoundComponents.append(int(c[-1]))
    
#     componentToClean = [1]
#     # for c in range(1,len(filteredData[week].columns)+1):
#     #     if c not in outBoundComponents:
#     #         componentToClean.append(c)
#     dataForPrediction.append(libRMT.regressionToCleanEigenvectorEffect(filteredData[week], pcaDataWeeks[week],componentToClean)) 

#clean first eigen effect data (not regression, just reconstruct X_hat from outbound components)
import libRMT
dataDenoiseDetrend_partly = []    
for week in range(0,12):
    outBoundComponents = dataForPrediction_outbound[week].columns
    componentToClean = ['pc1']
    for c in pcaDataWeeks[week].columns:
        if c not in outBoundComponents:
            componentToClean.append(c)
    dataDenoiseDetrend_partly.append(libRMT.cleanEigenvectorEffect(activityDataMatrixWeeks_20182019[week],pcaDataWeeks[week], 
                                                           componentToClean, pca_result[week].components_,0.5,0.8)) 

dataDenoiseDetrend_fully = []    
for week in range(0,12):
    outBoundComponents = dataForPrediction_outbound[week].columns
    componentToClean = ['pc1']
    for c in pcaDataWeeks[week].columns:
        if c not in outBoundComponents:
            componentToClean.append(c)
    dataDenoiseDetrend_fully.append(libRMT.cleanEigenvectorEffect(activityDataMatrixWeeks_20182019[week],pcaDataWeeks[week], 
                                                           componentToClean, pca_result[week].components_,1,1)) 

pca_result_fully_cleanData = []
pcaDataWeeks_fully_cleanData = []
columnsReturn2_fully_cleanData = []
for w in range(0,12):
    # tempData = transitionDataMatrixWeeks[w].loc[:,columns]
    tempData = dataDenoiseDetrend_fully[w]
    # tempData = tempData.merge(prediction_transition[w+1]['data']['successPassedRate'], left_on = tempData.index, right_on=prediction_transition[w+1]['data']['successPassedRate'].index).set_index('key_0')
    temp = FCAMiner.PCAcohortToValue(tempData)
    temp1 = temp[1]
    pcaResult = temp[0]
    # temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
    pcaDataWeeks_fully_cleanData.append(temp1)
    pca_result_fully_cleanData.append(pcaResult)
    columnsReturn2_fully_cleanData.append(temp[2])   

dataForPrediction_outbound_upper_on_fully_cleanData = []
for week in range(0,12):
    dataForPrediction_outbound_upper_on_fully_cleanData.append(libRMT.selectOutboundComponents(pcaDataWeeks_fully_cleanData[week],pca_result_fully_cleanData[week].explained_variance_,'upper'))

    

dataDenoiseDetrend_partly_upper = []    
for week in range(0,12):
    outBoundComponents = dataForPrediction_outbound[week].columns
    componentToClean = ['pc1']
    for c in pcaDataWeeks[week].columns:
        if c not in outBoundComponents:
            componentToClean.append(c)
    dataDenoiseDetrend_partly_upper.append(libRMT.cleanEigenvectorEffect(activityDataMatrixWeeks_20182019[week],pcaDataWeeks[week], 
                                                           componentToClean, pca_result[week].components_,0.5,0.8)) 

dataDenoiseDetrend_fully_upper = []    
for week in range(0,12):
    outBoundComponents = dataForPrediction_outbound[week].columns
    componentToClean = ['pc1']
    for c in pcaDataWeeks[week].columns:
        if c not in outBoundComponents:
            componentToClean.append(c)
    dataDenoiseDetrend_fully_upper.append(libRMT.cleanEigenvectorEffect(activityDataMatrixWeeks_20182019[week],pcaDataWeeks[week], 
                                                           componentToClean, pca_result[week].components_,1,1))  
    

denoiseData = []    
for week in range(0,12):
    outBoundComponents = dataForPrediction_outbound[week].columns
    # componentToClean = ['pc1']
    for c in pcaDataWeeks[week].columns:
        if c not in outBoundComponents:
            componentToClean.append(c)
    denoiseData.append(libRMT.cleanEigenvectorEffect(activityDataMatrixWeeks_20182019[week],pcaDataWeeks[week], 
                                                           componentToClean, pca_result[week].components_,1,0)) 

detrendData = []    
for week in range(0,12):
    outBoundComponents = dataForPrediction_outbound[week].columns
    componentToClean = ['pc1']
    for c in pcaDataWeeks[week].columns:
        if c not in outBoundComponents:
            componentToClean.append(c)
    detrendData.append(libRMT.cleanEigenvectorEffect(activityDataMatrixWeeks_20182019[week],pcaDataWeeks[week],
                                                     componentToClean, pca_result[week].components_,0,1)) 

#prediction
import PredictionResult


def prediction_implementation(data):
    prediction_transition = []
    for week in range(0,12):
        print('Predicting for Week ...' + str(week))
        if week in [0,1,2,3]:
            excellent = ex1_excellent_20182019.index
            weak = ex1_weak_20182019.index
        elif week in [4,5,6,7]:
            excellent = ex2_excellent_20182019.index
            weak = ex2_weak_20182019.index        
        else:
            excellent = ex3_excellent_20182019.index
            weak = ex3_weak_20182019.index  
        cummulativeResult = cummulativeExerciseWeeks_20182019[week].copy()
        predictionResult = PredictionResult.predict_proba_all_algorithms_data_ready(data[week], excellent, weak, cummulativeResult)
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
                                      prediction_transition[w][algorithm][8].mean(),
                                      prediction_transition[w][algorithm][8].mean()
                                      ])
            
    predictionReport_transition = pd.DataFrame(reportArray_transition,columns=['week','algorithm','accuraccy',
                                                         'f1_score','precision','recall',
                                                         'roc_auc','cv_mean_auc','cv_mean_f1','cv_mean_recall', 'cv_mean_accuracy']) 
    return prediction_transition, predictionReport_transition

predictionDetail_original, predictionReport_original = prediction_implementation(activityDataMatrixWeeks_20182019)
predictionDetail_pca, predictionReport_pca = prediction_implementation(pcaDataWeeks)
predictionDetail_RMTfilter, predictionReport_RMTfilter = prediction_implementation(dataForPrediction_outbound)
predictionDetail_partlyDenoiseDetrend, predictionReport_partlyDenoiseDetrend = prediction_implementation(dataDenoiseDetrend_partly)
# predictionDetail_denoise, predictionReport_denoise = prediction_implementation(denoiseData)
# predictionDetail_detrend, predictionReport_detrend = prediction_implementation(detrendData)
predictionDetail_fullyDenoiseDetrend, predictionReport_fullyDenoiseDetrend = prediction_implementation(dataDenoiseDetrend_fully)
predictionDetail_partlyDenoiseDetrend_upper, predictionReport_partlyDenoiseDetrend_upper = prediction_implementation(dataDenoiseDetrend_partly_upper)
predictionDetail_fullyDenoiseDetrend_upper, predictionReport_fullyDenoiseDetrend_upper = prediction_implementation(dataDenoiseDetrend_fully_upper)


predictionDetail_pca_onFullyClean, predictionReport_pca_onFullyClean = prediction_implementation(pcaDataWeeks_fully_cleanData)
predictionDetail_RMTfilter_onFullyClean, predictionReport_RMTfilter_onFullyClean = prediction_implementation(dataForPrediction_outbound_upper_on_fully_cleanData)


predictionReport_original['scenario'] = 'Original'
predictionReport_pca['scenario'] = 'PCA'
predictionReport_RMTfilter['scenario'] = 'RMT filter'
predictionReport_partlyDenoiseDetrend['scenario'] = 'Denoise_detrend_partly'
# predictionReport_denoise['scenario'] = 'Denoise'
# predictionReport_detrend['scenario'] = 'Detrend'
predictionReport_fullyDenoiseDetrend['scenario'] = 'Denoise_detrend_fully'
predictionReport_partlyDenoiseDetrend_upper['scenario'] = 'Denoise_detrend_partly_upper'
predictionReport_fullyDenoiseDetrend_upper['scenario'] = 'Denoise_detrend_fully_upper'
predictionReport_pca_onFullyClean['scenario'] = 'PCA_onFullyClean'
predictionReport_RMTfilter_onFullyClean['scenario'] = 'RMTfilter_onFullyClean'

predictionReport = pd.concat([predictionReport_original,
                              predictionReport_pca,
                              predictionReport_RMTfilter,
                              predictionReport_partlyDenoiseDetrend,
                               # predictionReport_denoise, 
                               # predictionReport_detrend, 
                              predictionReport_fullyDenoiseDetrend,
                              # predictionReport_partlyDenoiseDetrend_upper,
                              # predictionReport_fullyDenoiseDetrend_upper,
                              predictionReport_pca_onFullyClean,
                              predictionReport_RMTfilter_onFullyClean                              
                              ])

predictionReport.to_csv('PredictionReportCA116_1.csv')
title_transition = ''#'Outbound (both > lambda_max and < lambda_min) eigenvalues only by RMT'
algorithmList = ['Gradient Boosting','KNN','Logistic Regression','SVM','XGBoost']#['KNN','Logistic Regression','Random Forest','SVM','XGBoost']
# algorithmList = []

predictionReport_fullyDenoiseDetrend = predictionReport.loc[predictionReport['scenario'] == 'Denoise_detrend_fully']
predictionReport_partlyDenoiseDetrend = predictionReport.loc[predictionReport['scenario'] == 'Denoise_detrend_partly']

PredictionResult.algorithmComparisonGraph('cv_mean_auc',predictionReport_fullyDenoiseDetrend,algorithmList, title_transition, 'cross-validation roc_auc mean')

PredictionResult.algorithmComparisonGraph('accuraccy',predictionReport_RMTfilter_onFullyClean,algorithmList, title_transition)
PredictionResult.algorithmComparisonGraph('cv_mean_auc',predictionReport_RMTfilter_onFullyClean,algorithmList, title_transition)

#statistic test prediction performance between data processing strategise
x1 = predictionReport_detrend.loc[predictionReport_detrend['algorithm'] == 'XGBoost']['cv mean']
x2 = predictionReport_original[predictionReport_original['algorithm'] == 'XGBoost']['cv mean']

x1.hist(bins=10)
x2.hist(bins=10)

x1.mean()
x2.mean()

stats.mannwhitneyu(x1,x2)
stats.ttest_ind(x1,x2)

import matplotlib.pyplot as plt
def plotPredictionReport(dataToPlot, title, metricName): #input as an dictonary where each element is a label -> series

    plt.figure(figsize=(20,15))
    colorList = ['b','g', 'r', 'c', 'm', 'y', 'k', 'purple']
    for dataLabel,color in zip(dataToPlot,colorList):
        data = dataToPlot[dataLabel] 
        plt.plot(range(1,len(data)+1), data, 'o-', color=color, label=dataLabel, markersize=10)

        # plt.plot(predictionReport['week'], predictionReport['recall'],
        #           'o-', color='orange', label='recall_score', markersize=10)
    plt.title(title,fontsize=20)  
    plt.xlim(0, 13)
    plt.ylim(0.4, 0.85)
    plt.xticks(np.arange(0, 13), fontsize=20)
    plt.yticks(np.arange(0.4, 1, 0.1), fontsize=20)
    plt.xlabel("Week", fontsize=20)
    plt.ylabel(metricName + ' Scores', fontsize=20)
    plt.grid()
    plt.legend(loc="lower right", fontsize=18)
    plt.show()

metricName = 'roc_auc'
algorithm = 'XGBoost'
dataToPlot = {
              'Original' : predictionReport_original.loc[predictionReport_original['algorithm'] == algorithm][metricName],             
                'PCA' : predictionReport_pca.loc[predictionReport_pca['algorithm'] == algorithm][metricName],
                'RMT filter' : predictionReport_RMTfilter.loc[predictionReport_RMTfilter['algorithm'] == algorithm][metricName],
                 'Denoise_detrend_partly' : predictionReport_partlyDenoiseDetrend.loc[predictionReport_partlyDenoiseDetrend['algorithm'] == algorithm][metricName],
              # 'Denoise' : predictionReport_denoise.loc[predictionReport_denoise['algorithm'] == algorithm][metricName],
              # 'Detrend' : predictionReport_detrend.loc[predictionReport_detrend['algorithm'] == algorithm][metricName],
              'Denoise_detrend_fully': predictionReport_fullyDenoiseDetrend.loc[predictionReport_fullyDenoiseDetrend['algorithm'] == algorithm][metricName],
               # 'Denoise_detrend_partly_upper': predictionReport_partlyDenoiseDetrend_upper.loc[predictionReport_partlyDenoiseDetrend_upper['algorithm'] == algorithm][metricName],
                # 'Denoise_detrend_fully_upper': predictionReport_fullyDenoiseDetrend_upper.loc[predictionReport_fullyDenoiseDetrend_upper['algorithm'] == algorithm][metricName]
               'PCA_onFullyClean': predictionReport_pca_onFullyClean.loc[predictionReport_pca_onFullyClean['algorithm'] == algorithm][metricName],
               'RMTfilter_onFullyClean': predictionReport_RMTfilter_onFullyClean.loc[predictionReport_RMTfilter_onFullyClean['algorithm'] == algorithm][metricName]
              } 

plotPredictionReport(dataToPlot, 'Predicting performance among data processing strategies - ' + algorithm ,metricName)   

def statsTest(x1, x2):
    return
    
    
#plot barchar comparison
predictionReport = pd.read_csv('PredictionReportCA116_1.csv', index_col=0)

algorithm = ['XGBoost','SVM', 'KNN','Logistic Regression', 'Gradient Boosting']
metricName = 'roc_auc'
dataToBarchart = predictionReport.loc[(predictionReport['week'] == 11) & (predictionReport['algorithm'].isin(algorithm)),['algorithm','scenario',metricName]]
dataToBarchart = dataToBarchart.set_index(['algorithm','scenario'])[metricName]
dataToBarchart = dataToBarchart.unstack(level=-1)
dataToBarchart = dataToBarchart.loc[:,['Original','PCA','RMT filter','Denoise_detrend_fully','Denoise_detrend_partly']]

plt.figure(figsize=(15,10))
plt.ylim(0.5,0.9)
colorList = ['b','g', 'r', 'c', 'm'] #, 'y', 'k', 'purple']
markerList = ['o','v','s','*','D']
label = {'Original' : 'Original data',
         'PCA' : 'PCA data',
         'RMT filter' : 'PCA data with top principal components',
         'Denoise_detrend_fully' : 'Fully cleaned data',
         'Denoise_detrend_partly' : 'Partly cleaned data'
         }
for c, color, marker in zip(dataToBarchart.columns, colorList, markerList):
    plt.plot(dataToBarchart.index, dataToBarchart[c], color=color, marker=marker, markersize=10, label=label[c])
plt.title('', fontsize=14)
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
# plt.xlabel('Prediction Models', fontsize=20)
plt.ylabel('ROC AUC', fontsize=20)
plt.legend(loc="lower right", prop={'size': 16})
plt.grid(True)
plt.show()

dataToBarchart.plot.line(figsize=(15,10), ylim=(0.5,1), title=metricName, sort_columns=True, grid=True)


#----------------- Train 2018 - Predict 2019 -----------------------------------------------
#-----------------------------------------------------------------------------------------------#
X_train_weeks = []
y_train_weeks = []
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
    columnsList = []
    for c1 in activityDataMatrixWeeks_pageTypeWeek[w].columns:
        if c1 != 'result_exam_1':
            if c1 in activityDataMatrixWeeks_pageTypeWeek_2019[w].columns:
                columnsList.append(c1)
    trainingData = activityDataMatrixWeeks_pageTypeWeek[w].loc[:,columnsList] #node_embeddings_weeks[w].loc[node_embeddings_weeks[w].index.isin(excellent.union(weak))]
    columns = trainingData.columns
    trainingData['result_exam_1'] = 2
    trainingData.loc[trainingData.index.isin(excellent),['result_exam_1']] = 1
    trainingData.loc[trainingData.index.isin(weak),['result_exam_1']] = 0
    X_train_weeks.append(trainingData.loc[:,columns])
    y_train_weeks.append(trainingData['result_exam_1'])

for w in range(0,12):
    X_train_weeks[w] = X_train_weeks[w].merge(cummulativeExerciseWeeks[w].loc[:,['correct']], left_on=X_train_weeks[w].index , 
                                              right_on=cummulativeExerciseWeeks[w].loc[:,['correct']].index).set_index('key_0')
# y_train_weeks = []
for w in range(4,12):
    temp = pd.read_csv(basePath + 'transitionMatrixStorage_new/relabel_assessment_ca1162018_w' + str(w) + '.csv', index_col=0)
    temp = X_train_weeks[w].merge(temp, left_on=X_train_weeks[w].index, right_on=temp.index).set_index('key_0')
    temp['result_exam_1'] = y_train_weeks[w]
    temp.loc[temp['2'].isin(['0','1']),['result_exam_1']] = temp['2']
    y_train_weeks[w] = temp['result_exam_1']
    
# a = trainingData.corr()
    
X_test_weeks = []
y_test_weeks = []
for w in range(0,12):
    if w in [0,1,2,3]:
        excellent = ex1_excellent_2019.index
        weak = ex1_weak_2019.index
    elif w in [4,5,6,7]:
        excellent = ex2_excellent_2019.index
        weak = ex2_weak_2019.index        
    else:
        excellent = ex3_excellent_2019.index
        weak = ex3_weak_2019.index        
    columnsList = []
    for c1 in activityDataMatrixWeeks_pageTypeWeek[w].columns:
        if c1 != 'result_exam_1':
            if c1 in activityDataMatrixWeeks_pageTypeWeek_2019[w].columns:
                columnsList.append(c1)
    testData = activityDataMatrixWeeks_pageTypeWeek_2019[w].loc[:,columnsList] #node_embeddings_weeks[w].loc[node_embeddings_weeks[w].index.isin(excellent.union(weak))]
    columns = testData.columns
    testData['result_exam_1'] = 2
    testData.loc[testData.index.isin(excellent),['result_exam_1']] = 1
    testData.loc[testData.index.isin(weak),['result_exam_1']] = 0
    X_test_weeks.append(testData.loc[:,columns])
    y_test_weeks.append(testData['result_exam_1'])

for w in range(0,12):
    X_test_weeks[w] = X_test_weeks[w].merge(cummulativeExerciseWeeks_2019[w].loc[:,['correct']], left_on=X_test_weeks[w].index , 
                                              right_on=cummulativeExerciseWeeks_2019[w].loc[:,['correct']].index).set_index('key_0')
    
trainModels = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    trainDataStandardised = dataProcessing.normaliseData(X_train_weeks[w])
    y_train_weeks[w] = y_train_weeks[w].loc[y_train_weeks[w].index.isin(trainDataStandardised.index)]
    trainModels.append(PredictionResult.trainModel(trainDataStandardised, y_train_weeks[w]))
    
evalModels = []
for w in range(0,12):
    prediction = {}
    for algorithm in trainModels[w]:
        if algorithm != 'data':
            testData = X_test_weeks[w].loc[:,:]
            testDataStandardised = dataProcessing.normaliseData(testData)
            y_test_weeks[w] = y_test_weeks[w].loc[y_test_weeks[w].index.isin(testDataStandardised.index)]
            prediction[algorithm] = PredictionResult.evaluateTestData(trainModels[w][algorithm][1],testDataStandardised, y_test_weeks[w])    
    evalModels.append(prediction)
    
reportArray_transition = []
for w in range(0,12):
    for algorithm in evalModels[w]:
        if algorithm != 'data':
            reportArray_transition.append([w,algorithm, 
                                  evalModels[w][algorithm][0]['accuracy_score'][0],
                                  evalModels[w][algorithm][0]['f1_score'][0],
                                  evalModels[w][algorithm][0]['precision_score'][0],
                                  evalModels[w][algorithm][0]['recall_score'][0],
                                  evalModels[w][algorithm][0]['roc_auc']
                                  ])
        
predictionReport_transition = pd.DataFrame(reportArray_transition,columns=['week','algorithm','accuraccy',
                                                     'f1_score','precision','recall',
                                                     'roc_auc']) 

title_transition = '2018 and 2019 by weeks data - Evaluate 2018 model with 2019 data - activity data matrix - standardised'
algorithmList = []
# algorithmList = []
PredictionResult.algorithmComparisonGraph('roc_auc',predictionReport_transition,algorithmList, title_transition)

#----------------- Train 2018 - Predict 2019 uncleaned data -----------------------------------------------
#-----------------------------------------------------------------------------------------------#
X_train_weeks_uncleaned = []
y_train_weeks_uncleaned = []
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
    
    trainingData = node_embeddings_weeks_uncleaned[w].loc[node_embeddings_weeks_uncleaned[w].index.isin(excellent.union(weak))]
    columns = trainingData.columns
    trainingData['result_exam_1'] = 2
    trainingData.loc[trainingData.index.isin(excellent),['result_exam_1']] = 1
    trainingData.loc[trainingData.index.isin(weak),['result_exam_1']] = 0
    X_train_weeks_uncleaned.append(trainingData.loc[:,columns])
    # y_train_weeks_uncleaned.append(trainingData['result_exam_1'])


    
X_test_weeks_uncleaned = []
y_test_weeks_uncleaned = []
for w in range(0,12):
    if w in [0,1,2,3]:
        excellent = ex1_excellent_2019.index
        weak = ex1_weak_2019.index
    elif w in [4,5,6,7]:
        excellent = ex2_excellent_2019.index
        weak = ex2_weak_2019.index        
    else:
        excellent = ex3_excellent_2019.index
        weak = ex3_weak_2019.index        
    
    testData = node_embeddings_weeks_uncleaned[w].loc[node_embeddings_weeks_uncleaned[w].index.isin(excellent.union(weak))]
    columns = testData.columns
    testData['result_exam_1'] = 2
    testData.loc[testData.index.isin(excellent),['result_exam_1']] = 1
    testData.loc[testData.index.isin(weak),['result_exam_1']] = 0
    X_test_weeks_uncleaned.append(testData.loc[:,columns])
    y_test_weeks_uncleaned.append(testData['result_exam_1'])
    
trainModels_uncleaned = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    trainModels_uncleaned.append(PredictionResult.trainModel(X_train_weeks_uncleaned[w], y_train_weeks_uncleaned[w]))
    
evalModels_uncleaned = []
for w in range(0,12):
    prediction = {}
    for algorithm in trainModels_uncleaned[w]:
        if algorithm != 'data':
            prediction[algorithm] = PredictionResult.evaluateTestData(trainModels_uncleaned[w][algorithm][1],X_test_weeks_uncleaned[w], y_test_weeks_uncleaned[w])    
    evalModels_uncleaned.append(prediction)
    
reportArray_transition_uncleaned = []
for w in range(0,12):
    for algorithm in evalModels_uncleaned[w]:
        if algorithm != 'data':
            reportArray_transition_uncleaned.append([w,algorithm, 
                                  evalModels_uncleaned[w][algorithm][0]['accuracy_score'][0],
                                  evalModels_uncleaned[w][algorithm][0]['f1_score'][0],
                                  evalModels_uncleaned[w][algorithm][0]['precision_score'][0],
                                  evalModels_uncleaned[w][algorithm][0]['recall_score'][0],
                                  evalModels_uncleaned[w][algorithm][0]['roc_auc']
                                  ])
        
predictionReport_transition_uncleaned = pd.DataFrame(reportArray_transition_uncleaned,columns=['week','algorithm','accuraccy',
                                                     'f1_score','precision','recall',
                                                     'roc_auc']) 

title_transition = 'Graph embeddings - Node2Vec - 2018 and 2019 by weeks uncleaned data - Evaluate 2018 uncleaned data model with 2019 data - No exercise data'
algorithmList = []
# algorithmList = []
PredictionResult.algorithmComparisonGraph('roc_auc',predictionReport_transition_uncleaned,algorithmList, title_transition)

#----------------- Train 2018 - Predict 2019 - Not embeddings but use transition or activity data directly, not use graph yet -----------------------------------------------
#-----------------------------------------------------------------------------------------------#
X_train_weeks = []
y_train_weeks = []
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
    
    trainingData = activityDataMatrixWeeks_20182019_normalised[w].loc[activityDataMatrixWeeks_20182019_normalised[w].index.isin(excellent.union(weak))]
    trainingData = trainingData.merge(cummulativeExerciseWeeks_20182019[w]['correct'],left_on=trainingData.index,
                                            right_on=cummulativeExerciseWeeks_20182019[w]['correct'].index).set_index('key_0')
    columns = trainingData.columns
    trainingData['result_exam_1'] = 2
    trainingData.loc[trainingData.index.isin(excellent),['result_exam_1']] = 1
    trainingData.loc[trainingData.index.isin(weak),['result_exam_1']] = 0
    X_train_weeks.append(trainingData.loc[:,columns])
    y_train_weeks.append(trainingData['result_exam_1'])

# a = trainingData.corr()
    
X_test_weeks = []
y_test_weeks = []
for w in range(0,12):
    if w in [0,1,2,3]:
        excellent = ex1_excellent_2019.index
        weak = ex1_weak_2019.index
    elif w in [4,5,6,7]:
        excellent = ex2_excellent_2019.index
        weak = ex2_weak_2019.index        
    else:
        excellent = ex3_excellent_2019.index
        weak = ex3_weak_2019.index        
    
    testData = activityDataMatrixWeeks_20182019_normalised[w].loc[activityDataMatrixWeeks_20182019_normalised[w].index.isin(excellent.union(weak))]
    testData = testData.merge(cummulativeExerciseWeeks_20182019[w]['correct'],left_on=testData.index,
                                            right_on=cummulativeExerciseWeeks_20182019[w]['correct'].index).set_index('key_0')
    columns = testData.columns
    testData['result_exam_1'] = 2
    testData.loc[testData.index.isin(excellent),['result_exam_1']] = 1
    testData.loc[testData.index.isin(weak),['result_exam_1']] = 0
    X_test_weeks.append(testData.loc[:,columns])
    y_test_weeks.append(testData['result_exam_1'])
    
trainModels = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    trainModels.append(PredictionResult.trainModel(X_train_weeks[w], y_train_weeks[w]))
    
evalModels = []
for w in range(0,12):
    prediction = {}
    for algorithm in trainModels[w]:
        if algorithm != 'data':
            prediction[algorithm] = PredictionResult.evaluateTestData(trainModels[w][algorithm][1],X_test_weeks[w], y_test_weeks[w])    
    evalModels.append(prediction)
    
reportArray_transition = []
for w in range(0,12):
    for algorithm in evalModels[w]:
        if algorithm != 'data':
            reportArray_transition.append([w,algorithm, 
                                  evalModels[w][algorithm][0]['accuracy_score'][0],
                                  evalModels[w][algorithm][0]['f1_score'][0],
                                  evalModels[w][algorithm][0]['precision_score'][0],
                                  evalModels[w][algorithm][0]['recall_score'][0],
                                  evalModels[w][algorithm][0]['roc_auc']
                                  ])
        
predictionReport_transition = pd.DataFrame(reportArray_transition,columns=['week','algorithm','accuraccy',
                                                     'f1_score','precision','recall',
                                                     'roc_auc']) 

title_transition = 'Graph embeddings - 2018 and 2019 by weeks data - Evaluate 2018 model with 2019 data - Activity data normalised - with Exercise data'
algorithmList = []
# algorithmList = []
PredictionResult.algorithmComparisonGraph('roc_auc',predictionReport_transition,algorithmList, title_transition)
