import pandas as pd
# from numba import jit, cuda 
import numpy as np

import warnings
import time
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
# sns.set()

def assessmentConstruction(df, numberOfTest):
    assessmentResult = []
    for user, result in df.groupby(level=0):
        newRow = []
        newRow.append(user)
        correct = 0
        failed = 0
        passed = 0
        count = 0
        for task, result1 in result.groupby(level=1):
            if result1.iloc[0]['correct'] >= 1:            
                correct = correct + 1
            failed = failed + result1.iloc[0]['failed']
            passed = passed + result1.iloc[0]['passed']
            count = count + 1
        newRow.append(correct)
        newRow.append(float(correct)/numberOfTest)
        newRow.append(failed)
        newRow.append(passed)
        newRow.append(float(passed)/(float(passed) + float(failed)))
        newRow.append(count)
        assessmentResult.append(newRow)

    columns = ['user','correct', 'perCorrect','failed','passed','perPassed','testSubmitted']
    studentListAssessment = pd.DataFrame(assessmentResult,columns=columns)
    return studentListAssessment

def PCAcohortToValue(dataset):
    scaler = StandardScaler()
    pca = PCA(n_components=min(len(dataset), len(dataset.columns)))
    x = dataset.values    
    #x_adjust = x - np.mean(x)    
    scaler.fit(x)
    x = scaler.transform(x)
    pca.fit(x)
    transformed_value = pca.fit_transform(x)
    columns = []
    temp = min(len(dataset.columns),len(transformed_value))
    for i in range(0,temp):
        columns.append('pc' + str(i+1))
    transformed_value1 = pd.DataFrame(transformed_value,columns=columns, index=dataset.index)
    return [pca,transformed_value1,dataset.columns]

def normaliseWeeklyData(data):
    result = []
    for w in range(len(data)):
        x = data[w].values
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        result.append(pd.DataFrame(x,columns=data[w].columns, index = data[w].index))
    return result
        
def regressionToCleanEigenvectorEffect(originalData, PCAdata, eigenvector_to_clean):
    if len(originalData) != len(PCAdata):
        print('Sample length should be equal!!!')
        return
    selectedComponent = 'pc' + str(eigenvector_to_clean)
    selectedComponentData = PCAdata.loc[:,[selectedComponent]]
    cleanedData = pd.DataFrame(columns = originalData.columns, index=originalData.index)
    for c in originalData.columns:
        x = selectedComponentData
        y = originalData[c]
        regressor = LinearRegression()  
        regressor.fit(x, y) #training the algorithm
        alpha = regressor.intercept_
        beta = regressor.coef_[0]

        epsilon = []
        for s in originalData.index:
            epsilonOfstudentS = originalData.loc[originalData.index == s,[c]].values[0][0] - alpha - beta*selectedComponentData.loc[selectedComponentData.index == s][selectedComponent][0]
            epsilon.append(epsilonOfstudentS)
        cleanedData[c] = epsilon
    return cleanedData

def selectOutboundComponents(datasetPC, eigenvalueList):
    sampleLength = len(datasetPC)
    featuresLength = len(datasetPC.columns)
    
    q = featuresLength / float(sampleLength)

    # lambda_min = (1 - np.sqrt(q))**2
    lambda_max = (1 + np.sqrt(q))**2
    lambda_min = (1 - np.sqrt(q))**2
    pcList = ['pc' + str(i) for i in range(1,26)]
    columnList = []
    for eVal, pc in zip(eigenvalueList, pcList):
        if eVal >= 1: #lambda_max:
            columnList.append(pc)
            
    return datasetPC.loc[:,columnList]

def reportPredictiveResult(prediction_transition):
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
            
    predictionReport_transition = pd.DataFrame(reportArray_transition,columns=['week','algorithm','accuraccy',
                                                         'f1_score','precision','recall',
                                                         'roc_auc','cv mean']) 
    return predictionReport_transition


#weekly report
def getBestAlgorithmInAWeek(predictionReport_transition, score_type, scenario, week):
    temp = predictionReport_transition.loc[(predictionReport_transition['week'] == week) & 
                                                (predictionReport_transition['scenario'] == scenario),
                                                [score_type,'algorithm']]

    selected_row = temp.loc[temp[score_type] == temp[score_type].max()].head(1)
    return [week, scenario, selected_row[score_type].values[0], selected_row['algorithm'].values[0]]

def make_decision(y_pred_proba, threshold_as_zero):
    result = []
    for i in y_pred_proba:
        if i[0] >= threshold_as_zero:
            result.append(0)
        else:
            result.append(1)
    return result

def predict_proba_all_algorithms_data_ready(PCAdata, excellentList, weakList,practice=[], mode = 'activity'):
    # LogPageactivityCountByUser = FCAMiner.activityDataMatrixPercentage(LogPageactivityCountByUser)
    excellent_PCAdata = PCAdata.loc[PCAdata.index.isin(excellentList)]
    weak_PCAdata = PCAdata.loc[PCAdata.index.isin(weakList)]
    
    
    excellent_PCAdata['result_exam_1'] = 1
    weak_PCAdata['result_exam_1'] = 0
    PCAdata = pd.concat([excellent_PCAdata,weak_PCAdata])

    cum_practice_col = ['successPassedRate'] #,'correct_adjusted,cumm_practice','successPassedRate'
    PCAdata = PCAdata.merge(practice.loc[:,cum_practice_col], 
                                            left_on=PCAdata.index, right_on=practice.index)    
    PCAdata = PCAdata.set_index('key_0')
    
    colSelection = []#['correct_adjusted','successPassedRate'] #select only col whose p-value <=0.05 with result exam
    for col in PCAdata.columns:
         if col not in ['result_exam_1']:
        #     corr = pearsonr(LogPageactivityCountByUser['result_exam_1'], LogPageactivityCountByUser[col])
        #     if corr[1] <= pvalue:
        #         colSelection.append(col)    
            colSelection.append(col)
    
    if len(colSelection) == 0:
        return 'No columns correlated with result_exam_1'

    X=PCAdata[colSelection] 
    y=PCAdata['result_exam_1']                                       
    # Split dataset into training set and test set
    
    # sampler = RandomUnderSampler(sampling_strategy='auto')
    # X_rs, y_rs = sampler.fit_sample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5) # 80% training and 20% test
    
    xgb = XGBClassifier(
        learning_rate =0.05,
        n_estimators=1000,
        max_depth=5,
        min_child_weight=5,
        objective= 'binary:logistic'
    )
    lr = LogisticRegression()
    ada = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=5),
            n_estimators=1000
        )
    rf=RandomForestClassifier(n_estimators=1000)
    gb = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.025,
        # max_features=2,
        max_depth=5
    )
    dt = DecisionTreeClassifier()
    lda = LinearDiscriminantAnalysis()
    mlp = MLPClassifier()
    
    classifiers = [('XGBoost',xgb),('Logistic Regression',lr),('Ada Boost',ada),('Random Forest',rf),('Gradient Boosting',gb),
                       ('Decision Tree',dt),('Linear Discriminant',lda),('Multi Layer Nets', mlp)]
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    result = {}
    for name, classifier in classifiers:
        print(name + '...')
        classifier.fit(X_train,y_train)
        y_pred_proba=classifier.predict_proba(X_test)
    
        y_pred = make_decision(y_pred_proba, 0.5)
        
        scores = cross_val_score(classifier, X, y, cv=10, scoring='roc_auc')
        confusion_matrix1=metrics.confusion_matrix(y_test,y_pred)
        
        accuracy_score = metrics.accuracy_score(y_test, y_pred),
        f1_score = metrics.f1_score(y_test, y_pred),
        precision_score = metrics.precision_score(y_test, y_pred),
        recall_score = metrics.recall_score(y_test, y_pred),
        roc_auc = metrics.roc_auc_score(y_test, y_pred)
        classification_report1 = metrics.classification_report(y_test,y_pred)
        
        metricsResult = {
            'accuracy_score' : accuracy_score,
            'f1_score' : f1_score,
            'precision_score' : precision_score,
            'recall_score' : recall_score,
            'roc_auc' : roc_auc
        }
        
        if name not in ['Logistic Regression','Linear Discriminant','Multi Layer Nets']:
            feature_imp = pd.Series(classifier.feature_importances_,index=X.columns).sort_values(ascending=False)
        else:
            feature_imp = 'None'
        
        testingData = pd.DataFrame(columns=['y_test','y_pred','y_pred_proba_0','y_pred_proba_1'])
        testingData['y_test'] = y_test
        testingData['y_pred'] = y_pred
        testingData['y_pred_proba_0'] = y_pred_proba[:,0]
        testingData['y_pred_proba_1'] = y_pred_proba[:,1]
        result.update({name : [metricsResult, feature_imp, classifier,testingData, scores, confusion_matrix1, classification_report1]})
    result.update({'data' : PCAdata})
    return result

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

assessment1A = assessmentConstruction(ex1,4)
assessment1A['adjustedPerformance'] = (assessment1A['perCorrect'] + assessment1A['perPassed'])/2
assessment2A = assessmentConstruction(ex2,4)
assessment2A['adjustedPerformance'] = (assessment2A['perCorrect'] + assessment2A['perPassed'])/2
assessment3A = assessmentConstruction(ex3,4)
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


#read data
transitionDataMatrixWeeks = []
for w in range(0,12):
    transitionDataMatrixWeeks.append(pd.read_csv('transitionMatrixStorage/transitionDataMatrixWeeks_count_eventually_w'+str(w)+'.csv', header=0,index_col=0))
    
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
    temp = PCAcohortToValue(tempData)
    temp1 = temp[1]
    pcaResult = temp[0]
    # temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
    pcaDataWeeks.append(temp1)
    pca_result.append(pcaResult)
    columnsReturn2.append(temp[2])
    
#get normalise data
transitonDataMAtrixWeeks_normailise = normaliseWeeklyData(transitionDataMatrixWeeks)

#cleaning data 

transitionDataMatrixWeeks_normalised_cleaned = []
for w in range(0,12):
    transitionDataMatrixWeeks_normalised_cleaned.append(regressionToCleanEigenvectorEffect(transitonDataMAtrixWeeks_normailise[w],pcaDataWeeks[w],1))

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
    temp = PCAcohortToValue(tempData)
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
    pcaData2 = selectOutboundComponents(pcaDataWeeks[week],eigenValueList[week]) #filtered pca data to original data - scenario 2 #testing with eigenvalue > 1
    pcaData3 = pcaDataWeeks[week]  #full pca data - scenario 3
    pcaData4 = transitionDataMatrixWeeks_normalised_cleaned[week] #clean data - scenario 4
    pcaData5 = pcaDataWeeks_cleanedData[week] #pca clean data - scenario 5
    pcaData6 = selectOutboundComponents(pcaDataWeeks_cleanedData[week],eigenValueList_cleanedData[week]) #pca filtered clean data - scenario 6
    # print(pcaData.columns)

    mode = 'transition'
    
    tic = time.time()
    test1 = predict_proba_all_algorithms_data_ready(pcaData1,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario1',week,toc-tic])
    
    tic = time.time()
    test2 = predict_proba_all_algorithms_data_ready(pcaData2,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario2',week,toc-tic])
    
    tic = time.time()
    test3 = predict_proba_all_algorithms_data_ready(pcaData3,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario3',week,toc-tic])
    
    tic = time.time()
    test4 = predict_proba_all_algorithms_data_ready(pcaData4,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario4',week,toc-tic])    
    
    tic = time.time()
    test5 = predict_proba_all_algorithms_data_ready(pcaData5,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario5',week,toc-tic])    
    
    tic = time.time()
    test6 = predict_proba_all_algorithms_data_ready(pcaData6,excellent,weak,cummulativeResult,mode)
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
    

predictionReport_transition1 = reportPredictiveResult(prediction_transition1)
predictionReport_transition2 = reportPredictiveResult(prediction_transition2)
predictionReport_transition3 = reportPredictiveResult(prediction_transition3)
predictionReport_transition4 = reportPredictiveResult(prediction_transition4)
predictionReport_transition5 = reportPredictiveResult(prediction_transition5)
predictionReport_transition6 = reportPredictiveResult(prediction_transition6)

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
        selectedBestResult.append(getBestAlgorithmInAWeek(predictionReport_transition,score_type, s, w))

selectedBestResult = pd.DataFrame(selectedBestResult,columns = ['Week','Scenario','Best score', 'Best Algorithm'])
selectedBestResult.to_csv('BestResultScore_countData.csv',index=False)
