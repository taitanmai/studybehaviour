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
import joblib
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

### ca116 log 2019
ca116_eventLog_2019 = pd.read_csv('ca116_eventLog_nonfixed_2019.csv')
ca116_eventLog_2019 = ca116_eventLog_2019.loc[ca116_eventLog_2019['time:timestamp'] != ' ']
ca116_eventLog_2019['time:timestamp'] = pd.to_datetime(ca116_eventLog_2019['time:timestamp'])
weeksEventLog_2019 = [g for n, g in ca116_eventLog_2019.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
weeksEventLog_2019 = weeksEventLog_2019[3:15]

#uploads
dataUpload = pd.read_csv('ca116_uploads_2019.csv')
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

#get exam results from file
examResults1 = pd.read_csv('2020-06-29-einstein/marks/2020-ca116-continuous-assessment/2020-ca116-ex1-final.txt',delimiter=' ',header=None,names=['user','mark'],index_col=0)
ex1_excellent1 = examResults1.loc[examResults1['mark'] >= 40]
ex1_weak1 = examResults1.loc[examResults1['mark']<40]

examResults2 = pd.read_csv('2020-06-29-einstein/marks/2020-ca116-continuous-assessment/2020-ca116-ex2-final.txt',delimiter=' ',header=None,names=['user','mark'],index_col=0)
ex2_excellent1 = examResults2.loc[examResults2['mark'] >= 40]
ex2_weak1 = examResults2.loc[examResults2['mark']<40]

examResults3 = pd.read_csv('2020-06-29-einstein/marks/2020-ca116-continuous-assessment/2020-ca116-ex3-final.txt',delimiter=' ',header=None,names=['user','mark'],index_col=0)
ex3_excellent1 = examResults3.loc[examResults3['mark'] >= 40]
ex3_weak1 = examResults3.loc[examResults3['mark']<40]

#filtered log by activity Type
lectureList = dataProcessing.getLectureList(ca116_eventLog_2019,['html|py'])
eventLog_ca116_filtered = ca116_eventLog_2019.loc[ca116_eventLog_2019['description'].str.contains('|'.join(lectureList))]
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
weeksEventLog_filtered = weeksEventLog_filtered[2:15]
for i in weeksEventLog_filtered:
    print(len(i))

a = weeksEventLog_filtered[1].loc[weeksEventLog_filtered[1]['concept:name'] == 'Admin_page']
#prediction
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
        excellent = ex1_excellent1.index
        weak = ex1_weak1.index
    elif week in [5,6,7,8]:
        # if week == 5:
        #     workingWeekLog = []
        #     workingWeekExcercise = []
        workingWeekLog.append(weeksEventLog_filtered[week])
        workingWeekExcercise.append(nonExUploadByWeek[week-1])
        excellent = ex2_excellent1.index
        weak = ex2_weak1.index
    else:
        # if week == 9:
        #     workingWeekLog = []
        #     workingWeekExcercise = []
        workingWeekLog.append(weeksEventLog_filtered[week])
        workingWeekExcercise.append(nonExUploadByWeek[week-1])
        excellent = ex3_excellent1.index
        weak = ex3_weak1.index
    
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
    algorithm = 'XGBoost'
    # model = joblib.load('save_model/ca116/xgb_ca116_'+algorithm+'_w'+str(week)+'.model')
    # test = PredictionResult.evaluateModelWithData(model,Log,excellent,weak,cummulativeResult,lectureList,mode)
    test = PredictionResult.predict_proba_all_algorithms(Log,excellent,weak,cummulativeResult,lectureList,mode)
    # test = PredictionResult.predict_proba_all_algorithms(Log,overall_excellent,overall_weak,cummulativeResult,lectureList,mode)
    
    # prediction_cumm_practice.update({ week : test })
    prediction_transition.update({ week : test })
    # overall_prediction_transition.update({week : test1})

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

title_transition = 'Transition matrix data for weekly result - Scores of weekly prediction for the next exam results - cumulative new logs (with context) and practice no reset, new mark threshold'
# algorithmList = ['XGBoost','Random Forest','Multi Layer Nets','Gradient Boosting','Decision Tree']
algorithmList = []
PredictionResult.algorithmComparisonGraph('f1_score',predictionReport_transition,algorithmList, title_transition)
