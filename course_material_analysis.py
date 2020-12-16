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
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d, Axes3D
# sns.set()
# sns.set_style("whitegrid", {"axes.facecolor": ".9"})
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#uploads
dataUpload = pd.read_csv('ca116_uploads.csv')
dataUpload['date'] = pd.to_datetime(dataUpload.date)

nonExUpload = dataUpload.drop(dataUpload.loc[dataUpload['task'].str.match('ex')].index)


# cummulativeResultWeek2 = weeks[1].groupby([pd.Grouper(key='user')]).sum()
# cummulativeResultWeek2['result'] = cummulativeResultWeek2['correct']/weeks[1].groupby([pd.Grouper(key='user')]).count()['date']
# cummulativeResultWeek2['successPassedRate'] = cummulativeResultWeek2['passed']/(cummulativeResultWeek2['passed'] + cummulativeResultWeek2['failed'])

# fig=plt.figure()
# ax=fig.add_axes([0,0,1,1])
# ax.scatter(cummulativeResultWeek2['result'], cummulativeResultWeek2['successPassedRate'] , color='r')

# ax.set_xlabel('Result')
# ax.set_ylabel('SuccessPassedRate')
# ax.set_title('scatter plot')
# plt.show()


#process exUploadData
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
                            'failed':'failed1A',
                            'passed':'passed1A',
                            'perPassed':'perPassed1A',
                            'testSubmitted':'testSubmitted1A',
                            'adjustedPerformance':'adjustedPerformance1A'}, 
                  inplace=True)
assessment2A.rename(columns={'correct':'correct2A',
                          'perCorrect':'perCorrect2A',
                          'failed':'failed2A',
                            'failed':'failed2A',
                            'passed':'passed2A',
                            'perPassed':'perPassed2A',
                            'testSubmitted':'testSubmitted2A',
                            'adjustedPerformance':'adjustedPerformance2A'}, 
                  inplace=True)
assessment3A.rename(columns={'correct':'correct3A',
                          'perCorrect':'perCorrect3A',
                          'failed':'failed1A',
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

overall_pass = assessment.loc[assessment['grade'] >= 0.4]
overall_failed = assessment.loc[assessment['grade'] < 0.4]

nonExUploadByWeek = [g for n, g in nonExUpload.groupby(pd.Grouper(key='date',freq='W'))]
testAdjustCorrect = nonExUploadByWeek[10]
testAdjustCorrectSum = testAdjustCorrect.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
testAdjustCorrectSum['correct_adjusted'] = testAdjustCorrectSum['correct']/testAdjustCorrect.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).count()['correct']
testAdjustCorrectByUser = testAdjustCorrectSum.reset_index().groupby([pd.Grouper(key='user')]).sum()
#############################################

activityList = ['load','scroll','focus','blur','unload','hashchange','click-0','selection','click-2','click-1','click-3']


#extract event log 
eventLog_ca116 = pd.read_csv('ca116_eventLog_nonfixed.csv')
eventLog_ca116 = eventLog_ca116.drop([1160345])
eventLog_ca116['time:timestamp'] = pd.to_datetime(eventLog_ca116['time:timestamp'])
eventLog_ca116 = eventLog_ca116.loc[:, ~eventLog_ca116.columns.str.contains('^Unnamed')]
# materials = eventLog_ca116.loc[:,['org:resource','concept:name','description']]
weeksEventLog = [g for n, g in eventLog_ca116.groupby(pd.Grouper(key='time:timestamp',freq='W'))]

#process for new activity

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



weeksEventLog_filtered = [g for n, g in eventLog_ca116_filtered.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
for i in weeksEventLog_filtered:
    print(len(i))

a = weeksEventLog_filtered[2]
len(weeksEventLog_filtered[2])
weeksEventLog_filtered[2]['concept:name'].unique()
a.shape
##

#Prediction for each week
import PredictionResult
workingWeekLog = []
workingWeekExcercise = []
# prediction = {}
# prediction_cumm_practice = {}
# prediction_cumm_practice = {} #store transition matrix for prediction 
prediction_transition = {}
overall_prediction_transition = {}
for week in range(1,13):
    print('Week: ' + str(week) + '...')   

    if week in [1,2,3,4]:
        workingWeekLog.append(weeksEventLog_filtered[week])
        workingWeekExcercise.append(nonExUploadByWeek[week-1])
        excellent = ex1_excellent.index
        weak = ex1_weak.index
    elif week in [5,6,7,8]:
        # if week == 5:
        #     workingWeekLog = []
        #     workingWeekExcercise = []
        workingWeekLog.append(weeksEventLog_filtered[week])
        workingWeekExcercise.append(nonExUploadByWeek[week-1])
        excellent = ex2_excellent.index
        weak = ex2_weak.index
    else:
        # if week == 9:
        #     workingWeekLog = []
        #     workingWeekExcercise = []
        workingWeekLog.append(weeksEventLog_filtered[week])
        workingWeekExcercise.append(nonExUploadByWeek[week-1])
        excellent = ex3_excellent.index
        weak = ex3_weak.index
    
    # overall_excellent = overall_pass.index
    # overall_weak = overall_failed.index
    
    Log = pd.concat(workingWeekLog)
    practiceResult = pd.concat(workingWeekExcercise)
    
    #adjust number of correct: For each task, number of correct submission/number of submission for that task
    practiceResultSum = practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
    practiceResultSum['correct_adjusted'] = practiceResultSum['correct']/practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).count()['correct']
    cummulativeResult = practiceResultSum.reset_index().groupby([pd.Grouper(key='user')]).sum()

    # cummulativeResult = practiceResultSum.groupby([pd.Grouper(key='user')]).sum()
    cummulativeResult['cumm_practice'] = cummulativeResult['correct']/practiceResult.groupby([pd.Grouper(key='user')]).count()['date']
    cummulativeResult['successPassedRate'] = cummulativeResult['passed']/(cummulativeResult['passed'] + cummulativeResult['failed'])
    
    mode = 'transition'
    test = PredictionResult.predict_proba_all_algorithms(Log,excellent,weak,cummulativeResult,lectureList,mode)
    # test1 = PredictionResult.predict_proba_all_algorithms(Log,overall_excellent,overall_weak,cummulativeResult,lectureList,mode)
    
    # prediction_cumm_practice.update({ week : test })
    prediction_transition.update({ week : test })
    # overall_prediction_transition.update({week : test1})

#save model

algorithmList = ['Ada Boost','Decision Tree','Gradient Boosting', 'Linear Discriminant','Logistic Regression','Multi Layer Nets','Random Forest','XGBoost']
import joblib
for w in range(1,13):
    for algorithm in algorithmList:
        joblib.dump(prediction_transition[w][algorithm][2], open('save_model/ca116/xgb_ca116_'+algorithm+'_w'+str(w)+'.model', 'wb'))

#################### Visualise for activity matrix data
reportArray = []
title = 'Scores of weekly prediction for the next exam results - cumulative new logs (with context) and practice no reset, new mark threshold'
for w in range(1,13):
    for algorithm in prediction_cumm_practice[w]:
        if algorithm != 'data':
            reportArray.append([w,algorithm, 
                                  prediction_cumm_practice[w][algorithm][0]['accuracy_score'][0],
                                  prediction_cumm_practice[w][algorithm][0]['f1_score'][0],
                                  prediction_cumm_practice[w][algorithm][0]['precision_score'][0],
                                  prediction_cumm_practice[w][algorithm][0]['recall_score'][0],
                                  prediction_cumm_practice[w][algorithm][0]['roc_auc'],
                                  prediction_cumm_practice[w][algorithm][4].mean(),
                                  prediction_cumm_practice[w][algorithm][7].mean()
                                  ])
        
predictionReport = pd.DataFrame(reportArray,columns=['week','algorithm','accuraccy',
                                                     'f1_score','precision','recall',
                                                     'roc_auc','cv mean auc','cv mean f1']) 
#################### Visualise for transition matrix data
reportArray_transition = []
for w in range(1,13):
    for algorithm in prediction_transition[w]:
        if algorithm != 'data':
            reportArray_transition.append([w,algorithm, 
                                  prediction_transition[w][algorithm][0]['accuracy_score'],
                                  prediction_transition[w][algorithm][0]['f1_score'],
                                  prediction_transition[w][algorithm][0]['precision_score'],
                                  prediction_transition[w][algorithm][0]['recall_score'],
                                  prediction_transition[w][algorithm][0]['roc_auc'],
                                  prediction_transition[w][algorithm][4].mean(),
                                  prediction_transition[w][algorithm][7].mean()
                                  ])
        
predictionReport_transition = pd.DataFrame(reportArray_transition,columns=['week','algorithm','accuraccy',
                                                     'f1_score','precision','recall',
                                                     'roc_auc','cv mean','cv mean f1']) 

        
        # prediction_cumm_practice.to_csv('RandomForestPredictionCA116.csv')
#################### Visualise for transition matrix data for overall
reportArray_transition_overall = []
for w in range(1,13):
    for algorithm in overall_prediction_transition[w]:
        if algorithm != 'data':
            reportArray_transition_overall.append([w,algorithm, 
                                  overall_prediction_transition[w][algorithm][0]['accuracy_score'],
                                  overall_prediction_transition[w][algorithm][0]['f1_score'],
                                  overall_prediction_transition[w][algorithm][0]['precision_score'],
                                  overall_prediction_transition[w][algorithm][0]['recall_score'],
                                  overall_prediction_transition[w][algorithm][0]['roc_auc'],
                                  overall_prediction_transition[w][algorithm][4].mean(),
                                  overall_prediction_transition[w][algorithm][7].mean()
                                  ])
        
predictionReport_transition_overall = pd.DataFrame(reportArray_transition_overall,columns=['week','algorithm','accuraccy',
                                                     'f1_score','precision','recall',
                                                     'roc_auc','cv mean','cv mean f1']) 

title_transition = 'Transition matrix data for weekly result - Scores of weekly prediction for the next exam results - cumulative new logs (with context) and practice no reset, new mark threshold'
algorithmList = ['XGBoost','Random Forest','Multi Layer Nets','Gradient Boosting','Decision Tree']
# algorithmList = []
PredictionResult.algorithmComparisonGraph('roc_auc',predictionReport_transition,algorithmList, title_transition)

#Visualise decision Tree
from xgboost import plot_tree
plot_tree(prediction_transition[8]['XGBoost'][2], num_trees=0)
fig = plt.gcf()
fig.set_size_inches(30, 30)

a = prediction_cumm_practice[1]['XGBoost']
a = prediction_cumm_practice[9]['Multi Layer Nets']

#convert data for PCA - from eventlog to transition data matrix
workingWeekLog = []
transitionDataMatrixWeeks = []
for week in range(1,13):
    print('Week: ' + str(week) + '...')
    workingWeekLog.append(weeksEventLog_filtered[week])
    Log = pd.concat(workingWeekLog)
    tempTransition = FCAMiner.transitionDataMatrixConstruct_for_prediction(Log).fillna(0)
    tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()            
    transitionDataMatrixWeeks.append(tempTransition)

#time calculating
transitionDataTimeByWeeks = []
transitionDataFrequencyByWeeks = []
for week in range(1,13):
    transitionDataTimeByWeeks.append(FCAMiner.transitionDataMatrixConstruct_time(weeksEventLog_filtered[week])[0])
    transitionDataFrequencyByWeeks.append(FCAMiner.transitionDataMatrixConstruct_time(weeksEventLog_filtered[week])[1])
for w in range(0,12):
    transitionDataTimeByWeeks[w] = transitionDataTimeByWeeks[w].fillna(0).groupby([pd.Grouper(key='user')]).sum()
    # transitionDataFrequencyByWeeks[w] = transitionDataFrequencyByWeeks[w].fillna(0).groupby([pd.Grouper(key='user')]).sum()
    
############ get PCA transformed data 12 weeks 
# a = transitionDataMatrixWeeks[10].describe().transpose()
pcaDataWeeks = []
pca_result = []
columnsReturn2 = []

originalElements = ['Read_Lecture_Note','Read_Labsheet','Exercise','Check_solution','Admin_page']
columns = []
for i in originalElements:
    for j in originalElements:
        # if i != j:
        txt = i + '-' + j
        columns.append(txt)
columns = list(dict.fromkeys(columns))
for w in range(0,12):
    tempData = transitionDataMatrixWeeks[w].loc[:,columns]
    # tempData = tempData.merge(prediction_transition[w+1]['data']['successPassedRate'], left_on = tempData.index, right_on=prediction_transition[w+1]['data']['successPassedRate'].index).set_index('key_0')
    temp = FCAMiner.PCAcohortToValue(tempData)
    temp1 = temp[1].loc[:,['pc1','pc2','pc3']]
    pcaResult = temp[0]
    temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
    pcaDataWeeks.append(temp1)
    pca_result.append(pcaResult)
    columnsReturn2.append(temp[2])
    

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
    graph[countGraph].scatter(pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 1,['pc1']]
                           ,pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 1,['pc2']]
                           , c = 'r'
                           , s = 30, label='Successful')
    graph[countGraph].scatter(pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 0,['pc1']]
                           ,pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 0,['pc2']]
                           , c = 'b'
                           , s = 30, label='Successful')
    graph[countGraph].legend(loc='upper right')
    countGraph = countGraph + 1
               
plt.show()

#bibplot
def biplot(score, coeff , y, columns):
    '''
    Author: Serafeim Loukas, serafeim.loukas@epfl.ch
    Inputs:
       score: the projected data
       coeff: the eigenvectors (PCs)
       y: the class labels
   '''
    xs = score.loc[:,['pc1']] # projection on PC1
    ys = score.loc[:,['pc2']] # projection on PC2

    n = coeff.shape[0] # number of variables
    plt.figure(figsize=(10,8), dpi=100)
    classes = np.unique(y)
    colors = ['g','r','y']
    markers=['o','^','x']
    for s,l in enumerate(classes):
        plt.scatter(score.loc[score['result_exam_1'] == l,['pc1']],
                    score.loc[score['result_exam_1'] == l,['pc2']], 
                    c = colors[s], marker=markers[s]) # color based on group

    plt.xlabel("PC{}".format(1), size=14)
    plt.ylabel("PC{}".format(2), size=14)
    limx= int(xs.max()) + 1
    limy= int(ys.max()) + 1
    plt.xlim([-limx,limx])
    plt.ylim([-limy,limy])
    plt.grid()
    plt.tick_params(axis='both', which='both', labelsize=14)
    
    plt.figure(figsize=(10,8), dpi=100)
    for i in range(n):
        #plot as arrows the variable scores (each variable has a score for PC1 and one for PC2)
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'k', alpha = 0.9,linestyle = '-',linewidth = 1.5, overhang=0.2)
        plt.text(coeff[i,0]* 1.05, coeff[i,1] * 1.05, str(columns[i]), color = 'k', ha = 'center', va = 'center',fontsize=9)

    plt.xlabel("PC{}".format(1), size=14)
    plt.ylabel("PC{}".format(2), size=14)
    limx= 0.5
    limy= 0.5
    plt.xlim([-limx,limx])
    plt.ylim([-limy,limy])
    plt.grid()
    plt.tick_params(axis='both', which='both', labelsize=14)
    
biplot(pcaDataWeeks[11],
       np.transpose(pca_result[11].components_[0:2, :]),
       pcaDataWeeks[11].loc[:,['result_exam_1']], columns)

#plot loadings
def plotLoadings(week,pca_result,transitionDataMatrixWeeks, columnsReturn1):
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
    
plotLoadings(1,pca_result,transitionDataMatrixWeeks,columnsReturn2)  
plotLoadings(7,pca_result,transitionDataMatrixWeeks,columnsReturn2)  
plotLoadings(11,pca_result,transitionDataMatrixWeeks,columnsReturn2)  
    
####t-test 
from scipy import stats
for w in range(0,12):
    a1 = pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 1,['pc1']]
    b1 = pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 0,['pc1']]
    t1, p1 = stats.ttest_ind(a1,b1)
    a2 = pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 1,['pc2']]
    b2 = pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 0,['pc2']]
    t2, p2 = stats.ttest_ind(a2,b2)
    
    print('Week ' + str(w) + ':')
    print('--PC1: ' + 't-value: ' + str(t1) + ' p-value: ' + str(p1))
    print('-- Excellent: ' + str(a1.mean()[0]))
    print('-- Weak: ' + str(b1.mean()[0]))
    print('--PC2: ' + 't-value: ' + str(t2) + ' p-value: ' + str(p2))
    print('-- Excellent: ' + str(a2.mean()[0]))
    print('-- Weak: ' + str(b2.mean()[0]))   


#Time analysiss
#construct transition data matrix with time
eventLog_ca116_filtered1 = eventLog_ca116
lectureList = dataProcessing.getLectureList(eventLog_ca116,['html|py'])
eventLog_ca116_filtered1 = eventLog_ca116.loc[eventLog_ca116['description'].str.contains('|'.join(lectureList))]

eventLog_ca116_filtered1.loc[eventLog_ca116_filtered1['description'].str.contains('.html'),'pageType'] = 'Read_Lecture_Note' 
eventLog_ca116_filtered1.loc[eventLog_ca116_filtered1['description'].str.contains('correct|incorrect'),'pageType'] = 'Exercise'
eventLog_ca116_filtered1.loc[eventLog_ca116_filtered1['description'].str.contains('labsheet|instructions'),'pageType'] = 'Read_Labsheet'
eventLog_ca116_filtered1.loc[eventLog_ca116_filtered1['description'].str.contains('solution'),'pageType'] = 'Check_solution'
eventLog_ca116_filtered1.loc[eventLog_ca116_filtered1['description'].str.contains('http|report|ex|dashboard|graphs.html'),'pageType'] = 'Other'
eventLog_ca116_filtered1['pageType'] = eventLog_ca116_filtered1['pageType'] .fillna('Other')
eventLog_ca116_filtered1['pageType'].unique()

eventLog_ca116_filtered1.rename(columns={'concept:instance':'concept:instance1',
                                   'concept:name':'concept:name1',
                                   'case:concept:name' : 'case:concept:name1'}, 
                  inplace=True)
eventLog_ca116_filtered1['concept:instance'] = eventLog_ca116_filtered1['pageType']
eventLog_ca116_filtered1['concept:name'] = eventLog_ca116_filtered1['pageType']
eventLog_ca116_filtered1['date'] = eventLog_ca116_filtered1['time:timestamp'].dt.date

eventLog_ca116_filtered1['case:concept:name'] = eventLog_ca116_filtered1['date'].astype(str) + '-' + eventLog_ca116_filtered1['org:resource'].astype(str)
# eventLog_ca116_filtered1.to_csv('Event_Log_CA116_filtered1.csv')

weeksEventLog_filtered1 = [g for n, g in eventLog_ca116_filtered1.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
a = weeksEventLog_filtered1[1]
transitionDataTimeByWeeks = []
transitionDataFrequencyByWeeks = []
   
for w in range(1,13):
    transitionDataTimeByWeeks[w] = transitionDataTimeByWeeks[w].fillna(0).groupby([pd.Grouper(key='user')]).sum()
    transitionDataFrequencyByWeeks[w] = transitionDataFrequencyByWeeks[w].fillna(0).groupby([pd.Grouper(key='user')]).sum()
    
############ get PCA transformed data 12 weeks 
pcaDataWeeksFrequency = []
pca_resultFrequency = []
columnsReturn = []
for w in range(0,12):
    temp = FCAMiner.PCAcohortToValue(transitionDataFrequencyByWeeks[w])
    temp1 = temp[1].loc[:,['pc1','pc2']]
    pcaResult = temp[0]
    temp1 = temp1.merge(prediction_transition[w]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w]['data']['result_exam_1'].index).set_index('key_0')
    pcaDataWeeksFrequency.append(temp1)
    pca_resultFrequency.append(pcaResult)
    columnsReturn.append(temp[2])
    
plotLoadings(1,pca_resultFrequency,transitionDataFrequencyByWeeks,columnsReturn)  
plotLoadings(7,pca_resultFrequency,transitionDataFrequencyByWeeks,columnsReturn)  
plotLoadings(11,pca_resultFrequency,transitionDataFrequencyByWeeks,columnsReturn)

####t-test 
from scipy import stats
for w in range(0,12):
    a1 = pcaDataWeeksFrequency[w].loc[pcaDataWeeksFrequency[w]['result_exam_1'] == 1,['pc1']]
    b1 = pcaDataWeeksFrequency[w].loc[pcaDataWeeksFrequency[w]['result_exam_1'] == 0,['pc1']]
    t1, p1 = stats.ttest_ind(a1,b1)
    a2 = pcaDataWeeksFrequency[w].loc[pcaDataWeeksFrequency[w]['result_exam_1'] == 1,['pc2']]
    b2 = pcaDataWeeksFrequency[w].loc[pcaDataWeeksFrequency[w]['result_exam_1'] == 0,['pc2']]
    t2, p2 = stats.ttest_ind(a2,b2)
    
    print('Week ' + str(w) + ':')
    print('--PC1: ' + 't-value: ' + str(t1) + ' p-value: ' + str(p1))
    print('-- Excellent: ' + str(a1.mean()[0]))
    print('-- Weak: ' + str(b1.mean()[0]))
    print('--PC2: ' + 't-value: ' + str(t2) + ' p-value: ' + str(p2))
    print('-- Excellent: ' + str(a2.mean()[0]))
    print('-- Weak: ' + str(b2.mean()[0]))  
    
    
#Heuristic Miner
from pm4py.objects.conversion.log import factory as conversion_factory
Log = pd.concat(workingWeekLog)

ex1_personal_log_1_converted = conversion_factory.apply(Log.loc[Log['org:resource'].isin(ex3_excellent.index)])
ex1_personal_log_2_converted = conversion_factory.apply(Log.loc[Log['org:resource'].isin(ex3_weak.index)])



from pm4py.algo.discovery.heuristics import factory as heuristics_miner
from pm4py.visualization.heuristics_net import factory as hn_vis_factory

excellent_heu_net = heuristics_miner.apply_heu(ex1_personal_log_2_converted, parameters={"dependency_thresh": 0.0})
gviz = hn_vis_factory.apply(excellent_heu_net)
hn_vis_factory.view(gviz)

