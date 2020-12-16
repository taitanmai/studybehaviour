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



dataUpload = pd.read_csv('ca177_uploads.csv')
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

ex1_excellent = assessment1A.loc[(assessment1A['perCorrect1A'] <= 1) & (assessment1A['perCorrect1A'] >= 0.4)]
ex1_weak = assessment1A.loc[(assessment1A['perCorrect1A'] >= 0) & (assessment1A['perCorrect1A'] < 0.4)]

ex2_excellent = assessment2A.loc[(assessment2A['perCorrect2A'] <= 1)&(assessment2A['perCorrect2A'] >= 0.4)]
ex2_weak = assessment2A.loc[(assessment2A['perCorrect2A'] >= 0) & (assessment2A['perCorrect2A'] < 0.4)]

overall_pass = assessment.loc[assessment['grade'] >= 0.6]
overall_failed = assessment.loc[assessment['grade'] < 0.6]

nonExUploadByWeek = [g for n, g in nonExUpload.groupby(pd.Grouper(key='date',freq='W'))]


#extract event log 
eventLog_ca177 = pd.read_csv('ca177_eventLog_nonfixed.csv')
eventLog_ca177 = eventLog_ca177.drop([89978])
eventLog_ca177['time:timestamp'] = pd.to_datetime(eventLog_ca177['time:timestamp'])
eventLog_ca177 = eventLog_ca177.loc[:, ~eventLog_ca177.columns.str.contains('^Unnamed')]
eventLog_ca177 = eventLog_ca177.loc[eventLog_ca177['time:timestamp'] >= '2019-01-27']
# materials = eventLog_ca116.loc[:,['org:resource','concept:name','description']]
weeksEventLog = [g for n, g in eventLog_ca177.groupby(pd.Grouper(key='time:timestamp',freq='W'))]

lectureList = dataProcessing.getLectureList(eventLog_ca177,['html|py'])
eventLog_ca117_filtered = eventLog_ca177.loc[eventLog_ca177['description'].str.contains('|'.join(lectureList))]
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
import joblib

model = joblib.load(open('xgb_ca116_w5', 'rb'))


import PredictionResult
workingWeekLog = []
workingWeekExcercise = []
# prediction = {}
# prediction_cumm_practice = {}
# prediction_cumm_practice = {} #store transition matrix for prediction 
evaluate_ca116_model = {}
for week in range(0,12):
    print('Week: ' + str(week) + '...')   

    if week in [0,1,2,3,4,5]:
        workingWeekLog.append(weeksEventLog_filtered[week])
        workingWeekExcercise.append(nonExUploadByWeek[week])
        excellent = ex1_excellent.index
        weak = ex1_weak.index
    elif week in [6,7,8,9,10,11]:
        # if week == 5:
        #     workingWeekLog = []
            # workingWeekExcercise = []
        workingWeekLog.append(weeksEventLog_filtered[week])
        workingWeekExcercise.append(nonExUploadByWeek[week])
        excellent = ex2_excellent.index
        weak = ex2_weak.index

    
    Log = pd.concat(workingWeekLog)
    practiceResult = pd.concat(workingWeekExcercise)
    
    #adjust number of correct: For each task, number of correct submission/number of submission for that task
    practiceResultSum = practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
    practiceResultSum['correct_adjusted'] = practiceResultSum['correct']/practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).count()['correct']
    cummulativeResult = practiceResultSum.reset_index().groupby([pd.Grouper(key='user')]).sum()

    # cummulativeResult = practiceResultSum.groupby([pd.Grouper(key='user')]).sum()
    cummulativeResult['cumm_practice'] = cummulativeResult['correct']/practiceResult.groupby([pd.Grouper(key='user')]).count()['date']
    cummulativeResult['successPassedRate'] = cummulativeResult['passed']/(cummulativeResult['passed'] + cummulativeResult['failed'])
    
    test = PredictionResult.predict_proba_all_algorithms(Log,excellent,weak,cummulativeResult,lectureList, mode='transition')
    evaluate_ca116_model.update({ week : test })

reportArray = []
for w in range(0,12):
    for algorithm in evaluate_ca116_model[w]:
        if algorithm != 'data':
            reportArray.append([w,algorithm, 
                                  evaluate_ca116_model[w][algorithm][0]['accuracy_score'][0],
                                  evaluate_ca116_model[w][algorithm][0]['f1_score'][0],
                                  evaluate_ca116_model[w][algorithm][0]['precision_score'][0],
                                  evaluate_ca116_model[w][algorithm][0]['recall_score'][0],
                                  # evaluate_ca116_model[w][algorithm][0]['roc_auc'],
                                  evaluate_ca116_model[w][algorithm][4].mean()
                                  ])
        
predictionReport = pd.DataFrame(reportArray,columns=['week','algorithm','accuraccy',
                                                     'f1_score','precision','recall',
                                                     'cv mean']) 
        

algorithmList = ['XGBoost','Multi Layer Nets','Linear Discriminant','Random Forest']
PredictionResult.algorithmComparisonGraph('f1_score',predictionReport,algorithmList)

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

