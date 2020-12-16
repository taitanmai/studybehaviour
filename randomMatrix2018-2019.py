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

lectureList = [] #will delete later, fix the prediction function


db1 = pd.read_csv("eventLog_ca116_filtered_2018.csv")
db1['time:timestamp'] = pd.to_datetime(db1['time:timestamp'])

db2 = pd.read_csv("eventLog_ca116_filtered_2019.csv")
db2['time:timestamp'] = pd.to_datetime(db2['time:timestamp'])


eventLog_ca116_filtered = pd.concat([db1,db2])

eventLog_ca116_filtered['time:timestamp'] = pd.to_datetime(eventLog_ca116_filtered['time:timestamp'])

weeksEventLog_filtered_2018 = [g for n, g in db1.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
weeksEventLog_filtered_2018 = weeksEventLog_filtered_2018[1:13]

weeksEventLog_filtered_2019 = [g for n, g in db2.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
weeksEventLog_filtered_2019 = weeksEventLog_filtered_2019[3:15]


weeksEventLog_filtered = []
for w in range(0,12):
    temp = pd.concat([weeksEventLog_filtered_2018[w],weeksEventLog_filtered_2019[w]])
    weeksEventLog_filtered.append(temp)

#uploads
ex1_excellent_2018 = pd.read_csv("ex1_excellent_2018.csv")
ex1_weak_2018 = pd.read_csv("ex1_weak_2018.csv")

ex2_excellent_2018 = pd.read_csv("ex2_excellent_2018.csv")
ex2_weak_2018 = pd.read_csv("ex2_weak_2018.csv")

ex3_excellent_2018 = pd.read_csv("ex3_excellent_2018.csv")
ex3_weak_2018 = pd.read_csv("ex3_weak_2018.csv")

ex1_excellent_2019 = pd.read_csv("ex1_excellent_2019.csv")
ex1_weak_2019 = pd.read_csv("ex1_weak_2019.csv")

ex2_excellent_2019 = pd.read_csv("ex2_excellent_2019.csv")
ex2_weak_2019 = pd.read_csv("ex2_weak_2019.csv")

ex3_excellent_2019 = pd.read_csv("ex3_excellent_2019.csv")
ex3_weak_2019 = pd.read_csv("ex3_weak_2019.csv")

ex1_excellent = pd.concat([ex1_excellent_2018,ex1_excellent_2019])
ex1_excellent = ex1_excellent.set_index('user')
ex1_weak = pd.concat([ex1_weak_2018,ex1_weak_2019])
ex1_weak = ex1_weak.set_index('user')

ex2_excellent = pd.concat([ex2_excellent_2018,ex2_excellent_2019])
ex2_excellent = ex2_excellent.set_index('user')
ex2_weak = pd.concat([ex2_weak_2018,ex2_weak_2019])
ex2_weak = ex2_weak.set_index('user')

ex3_excellent = pd.concat([ex3_excellent_2018,ex3_excellent_2019])
ex3_excellent = ex3_excellent.set_index('user')
ex3_weak = pd.concat([ex3_weak_2018,ex3_weak_2019])
ex3_weak = ex3_weak.set_index('user')

#non exam upload
nonExUpload_2018 = pd.read_csv('nonExUpload_2018.csv')
nonExUpload_2018['date'] = pd.to_datetime(nonExUpload_2018['date'])
nonExUploadByWeek_2018 = [g for n, g in nonExUpload_2018.groupby(pd.Grouper(key='date',freq='W'))]

nonExUpload_2019 = pd.read_csv('nonExUpload_2019.csv')
nonExUpload_2019['date'] = pd.to_datetime(nonExUpload_2019['date'])
nonExUploadByWeek_2019 = [g for n, g in nonExUpload_2019.groupby(pd.Grouper(key='date',freq='W'))]

nonExUploadByWeek = []
for w in range(0,12):
    temp = pd.concat([nonExUploadByWeek_2018[w], nonExUploadByWeek_2019[w]])
    nonExUploadByWeek.append(temp)

#time calculating
transitionDataTimeByWeeks = []
transitionDataFrequencyByWeeks = []
for week in range(0,12):
    transitionDataTimeByWeeks.append(FCAMiner.transitionDataMatrixConstruct_time(weeksEventLog_filtered[week])[0])
    transitionDataFrequencyByWeeks.append(FCAMiner.transitionDataMatrixConstruct_time(weeksEventLog_filtered[week])[1])
for w in range(0,12):
    transitionDataTimeByWeeks[w] = transitionDataTimeByWeeks[w].fillna(0).groupby([pd.Grouper(key='user')]).sum()
    # transitionDataFrequencyByWeeks[w] = transitionDataFrequencyByWeeks[w].fillna(0).groupby([pd.Grouper(key='user')]).sum()
  
#convert data for PCA - from eventlog to transition data matrix
workingWeekLog = []
transitionDataMatrixWeeks = []
for week in range(0,12):
    print('Week: ' + str(week) + '...')
    workingWeekLog.append(weeksEventLog_filtered[week])
    Log = pd.concat(workingWeekLog)
    tempTransition = FCAMiner.transitionDataMatrixConstruct_for_prediction(Log).fillna(0)
    tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()            
    transitionDataMatrixWeeks.append(tempTransition)



#pca Conver    
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
    # tempData = transitionDataMatrixWeeks[w].loc[:,columns]
    # tempData = tempData.merge(prediction_transition[w+1]['data']['successPassedRate'], left_on = tempData.index, right_on=prediction_transition[w+1]['data']['successPassedRate'].index).set_index('key_0')
    temp = FCAMiner.PCAcohortToValue(tempData)
    temp1 = temp[1]
    pcaResult = temp[0]
    # temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
    pcaDataWeeks.append(temp1)
    pca_result.append(pcaResult)
    columnsReturn2.append(temp[2])



transitonDataMAtrixWeeks_normailise = dataProcessing.normaliseWeeklyData(transitionDataMatrixWeeks)
transitonDataMAtrixWeeks_normailise[2]['Read_Lecture_Note-Read_Lecture_Note'].std()

import libRMT
transitionDataMatrixWeeks_normalised_cleaned = []
for w in range(0,12):
    transitionDataMatrixWeeks_normalised_cleaned.append(libRMT.regressionToCleanEigenvectorEffect(transitonDataMAtrixWeeks_normailise[w],pcaDataWeeks[w],1))

#pca Convert to cleaned data
pcaDataWeeks_cleanedData = []
pca_result_cleanedData = []
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
    # tempData = transitionDataMatrixWeeks[w].loc[:,columns]
    tempData = transitionDataMatrixWeeks_normalised_cleaned[w].loc[:,columns]
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
    pcaData2 = libRMT.selectOutboundComponents(pcaDataWeeks[week],eigenValueList[week]) #filtered pca data to original data - scenario 2
    pcaData3 = pcaDataWeeks[week]  #full pca data - scenario 3
    pcaData4 = transitionDataMatrixWeeks_normalised_cleaned[week] #clean data - scenario 4
    pcaData5 = pcaDataWeeks_cleanedData[week] #pca clean data - scenario 5
    pcaData6 = libRMT.selectOutboundComponents(pcaDataWeeks_cleanedData[week],eigenValueList_cleanedData[week]) #pca filtered clean data - scenario 6
    # print(pcaData.columns)

    mode = 'transition'
    
    tic = time.time()
    test1 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData1,excellent,weak,cummulativeResult,lectureList,mode)
    toc = time.time()
    timePerformance.append(['scenario1',week,toc-tic])
    
    tic = time.time()
    test2 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData2,excellent,weak,cummulativeResult,lectureList,mode)
    toc = time.time()
    timePerformance.append(['scenario2',week,toc-tic])
    
    tic = time.time()
    test3 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData3,excellent,weak,cummulativeResult,lectureList,mode)
    toc = time.time()
    timePerformance.append(['scenario3',week,toc-tic])
    
    tic = time.time()
    test4 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData4,excellent,weak,cummulativeResult,lectureList,mode)
    toc = time.time()
    timePerformance.append(['scenario4',week,toc-tic])    
    
    tic = time.time()
    test5 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData5,excellent,weak,cummulativeResult,lectureList,mode)
    toc = time.time()
    timePerformance.append(['scenario5',week,toc-tic])    
    
    tic = time.time()
    test6 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData6,excellent,weak,cummulativeResult,lectureList,mode)
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

selectedBestResult.to_csv("report.csv",index=False)
#################### Visualise for transition matrix data
prediction_transition = prediction_transition6
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
                                  prediction_transition[w][algorithm][4].mean()
                                  ])
        
predictionReport_transition1 = pd.DataFrame(reportArray_transition,columns=['week','algorithm','accuraccy',
                                                     'f1_score','precision','recall',
                                                     'roc_auc','cv mean']) 

title_transition = 'pca cleaned data filtered outbound components'
algorithmList = []
# algorithmList = []
PredictionResult.algorithmComparisonGraph('accuraccy',predictionReport_transition1,algorithmList, title_transition)


import libRMT
import warnings
warnings.filterwarnings("ignore")
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


