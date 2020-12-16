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

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'



dataUpload = pd.read_csv('ca117_uploads.csv')
dataUpload['date'] = pd.to_datetime(dataUpload.date)

nonExUpload = dataUpload.drop(dataUpload.loc[dataUpload['task'].str.match('ex')].index)

weeks = [g for n, g in nonExUpload.groupby(pd.Grouper(key='date',freq='W'))]

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

assessment1A = assessment1A.set_index(['user'])
assessment2A = assessment2A.set_index(['user'])

assessment = pd.concat([assessment1A,assessment2A], axis=1)
assessment = assessment.fillna(0)

assessment['grade'] = (assessment['perCorrect1A']+assessment['perCorrect2A'])/2
assessment['perPassed'] = (assessment['passed1A'] + assessment['passed2A'])/(assessment['passed1A'] + assessment['passed2A'] 
                        + assessment['failed1A']+ assessment['failed2A'])

ex1_excellent = assessment1A.loc[(assessment1A['perCorrect1A'] <= 1) & (assessment1A['perCorrect1A'] >= 0.6)]
ex1_weak = assessment1A.loc[(assessment1A['perCorrect1A'] >= 0) & (assessment1A['perCorrect1A'] < 0.6)]

ex2_excellent = assessment2A.loc[(assessment2A['perCorrect2A'] <= 1)&(assessment2A['perCorrect2A'] >= 0.5)]
ex2_weak = assessment2A.loc[(assessment2A['perCorrect2A'] >= 0) & (assessment2A['perCorrect2A'] < 0.5)]

overall_pass = assessment.loc[assessment['grade'] >= 0.4]
overall_failed = assessment.loc[assessment['grade'] < 0.4]

nonExUploadByWeek = [g for n, g in nonExUpload.groupby(pd.Grouper(key='date',freq='W'))]


#extract event log 
eventLog_ca117 = pd.read_csv('ca117_eventLog_nonfixed.csv')
# eventLog_ca117 = eventLog_ca117.drop([89978])
eventLog_ca117['time:timestamp'] = pd.to_datetime(eventLog_ca177['time:timestamp'])
eventLog_ca117 = eventLog_ca177.loc[:, ~eventLog_ca117.columns.str.contains('^Unnamed')]
# eventLog_ca117 = eventLog_ca177.loc[eventLog_ca117['time:timestamp'] >= '2019-01-27']
# materials = eventLog_ca116.loc[:,['org:resource','concept:name','description']]
weeksEventLog = [g for n, g in eventLog_ca117.groupby(pd.Grouper(key='time:timestamp',freq='W'))]

lectureList = dataProcessing.getLectureList(eventLog_ca117,['html|py'])
eventLog_ca117_filtered = eventLog_ca117.loc[eventLog_ca117['description'].str.contains('|'.join(lectureList))]
# ex1_personal_log_1 = dataProcessing.addConceptPageToLog(ex1_personal_log_1)

eventLog_ca117_filtered = eventLog_ca117_filtered.drop(eventLog_ca117_filtered.loc[eventLog_ca117_filtered['description'].str.contains('http|report|ex|dashboard|graphs.html')].index)

eventLog_ca117_filtered.loc[eventLog_ca117_filtered['description'].str.contains('.html'),'pageType'] = 'Read_Lecture_Note' 
eventLog_ca117_filtered.loc[eventLog_ca117_filtered['description'].str.contains('correct|incorrect'),'pageType'] = 'Excercise'
eventLog_ca117_filtered.loc[eventLog_ca117_filtered['description'].str.contains('labsheet|instructions'),'pageType'] = 'Read_Labsheet'
eventLog_ca117_filtered.loc[eventLog_ca117_filtered['description'].str.contains('solution'),'pageType'] = 'Check_solution'
eventLog_ca117_filtered['pageType'] = eventLog_ca117_filtered['pageType'] .fillna('Other')
eventLog_ca117_filtered = eventLog_ca117_filtered.drop(eventLog_ca117_filtered.loc[eventLog_ca117_filtered['pageType'] == 'Other'].index)


eventLog_ca117_filtered['pageType'].unique()

eventLog_ca117_filtered.rename(columns={'concept:instance':'concept:instance1',
                                   'concept:name':'concept:name1',
                                   'case:concept:name' : 'case:concept:name1'}, 
                  inplace=True)
eventLog_ca117_filtered['concept:instance'] = eventLog_ca117_filtered['pageType']
eventLog_ca117_filtered['concept:name'] = eventLog_ca117_filtered['pageType']
eventLog_ca117_filtered['date'] = eventLog_ca117_filtered['time:timestamp'].dt.date

case_log_1 = []
for index, row in eventLog_ca117_filtered.iterrows():
    case_log_1.append(str(row['date']) + '-' + str(row['org:resource']))
eventLog_ca117_filtered['case:concept:name'] = case_log_1

weeksEventLog_filtered = [g for n, g in eventLog_ca117_filtered.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
a = weeksEventLog_filtered[5]

#Prediction for each week
import PredictionResult
workingWeekLog = []
workingWeekExcercise = []
# prediction = {}
# prediction_cumm_practice = {}
prediction_cumm_practice = {} #store transition matrix for prediction 
for week in range(0,12):
    print('Week: ' + str(week) + '...')   

    if week in [0,1,2,3,4,5]:
        workingWeekLog.append(weeksEventLog_filtered[week])
        workingWeekExcercise.append(nonExUploadByWeek[week])
        excellent = overall_pass.index
        weak = overall_failed.index
    elif week in [6,7,8,9,10,11]:
        # if week == 5:
        #     workingWeekLog = []
            # workingWeekExcercise = []
        workingWeekLog.append(weeksEventLog_filtered[week])
        workingWeekExcercise.append(nonExUploadByWeek[week])
        excellent = overall_pass.index
        weak = overall_failed.index

    
    Log = pd.concat(workingWeekLog)
    practiceResult = pd.concat(workingWeekExcercise)
    
    #adjust number of correct: For each task, number of correct submission/number of submission for that task
    practiceResultSum = practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
    practiceResultSum['correct_adjusted'] = practiceResultSum['correct']/practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).count()['correct']
    cummulativeResult = practiceResultSum.reset_index().groupby([pd.Grouper(key='user')]).sum()

    # cummulativeResult = practiceResultSum.groupby([pd.Grouper(key='user')]).sum()
    cummulativeResult['cumm_practice'] = cummulativeResult['correct']/practiceResult.groupby([pd.Grouper(key='user')]).count()['date']
    cummulativeResult['successPassedRate'] = cummulativeResult['passed']/(cummulativeResult['passed'] + cummulativeResult['failed'])
    
    test = PredictionResult.predictionRandomForestProbability(Log,excellent,weak,cummulativeResult,lectureList)
    prediction_cumm_practice.update({ week : test })


reportArray = []
for w in range(0,12):
    reportArray.append([w,prediction_cumm_practice[w][0]['accuracy_score'][0],prediction_cumm_practice[w][0]['f1_score'][0],
                        prediction_cumm_practice[w][0]['precision_score'][0],prediction_cumm_practice[w][0]['recall_score'],
                        prediction_cumm_practice[w][0]['roc_auc']])

predictionReport = pd.DataFrame(reportArray,columns=['week','accuraccy','f1_score','precision','recall','roc_auc']) 
# prediction_cumm_practice.to_csv('RandomForestPredictionCA116.csv')

plt.figure(figsize=(18,10))
plt.plot(predictionReport['week'], predictionReport['f1_score'],
          'o-', color='red', label='f1_score', markersize=10)
plt.plot(predictionReport['week'], predictionReport['accuraccy'],
          'o-', color='blue', label='accuracy_score', markersize=10)
plt.plot(predictionReport['week'], predictionReport['roc_auc'],
          'o-', color='green', label='ROC_AUC', markersize=10)
# plt.plot(predictionReport['week'], predictionReport['recall'],
#           'o-', color='orange', label='recall_score', markersize=10)
plt.title('Random Forest - Scores of weekly prediction for the next exam results - cumulative logs and practice no reset, new mark threshold')  
plt.xticks(np.arange(0, 14), fontsize=20)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=20)
plt.xlabel("Week", fontsize=20)
plt.ylabel('Scores', fontsize=20)
plt.grid()
plt.legend(loc="upper right", fontsize=18)
plt.show()


#Visualise decision Tree
from xgboost import plot_tree
plot_tree(prediction_cumm_practice[6][2], num_trees=1)
fig = plt.gcf()
fig.set_size_inches(30, 30)

from sklearn.tree import export_graphviz
from sklearn import tree

export_graphviz(prediction_cumm_practice[6][2].estimators_[0],
                feature_names=prediction_cumm_practice[6][3].columns[0:4],
                filled=True,
                rounded=True)
fig.savefig('rf_individualtree.png')
