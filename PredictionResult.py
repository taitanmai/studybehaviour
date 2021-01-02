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
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import dataProcessing
import FCAMiner
import seaborn as sns


#######get proba
def make_decision(y_pred_proba, threshold_as_zero):
    result = []
    for i in y_pred_proba:
        if i[0] >= threshold_as_zero:
            result.append(0)
        else:
            result.append(1)
    return result
def predictionRandomForestProbability(dfActivityMatrixWithResult, excellentList, weakList,practice, lectureList=[], pvalue = 0.05):
    if len(lectureList) == 0:
        lectureList = dataProcessing.getLectureList(dfActivityMatrixWithResult)
        
    logPage =  dfActivityMatrixWithResult.loc[dfActivityMatrixWithResult['description'].str.contains('|'.join(lectureList))] 
    # ex1_LogPageIf = dataProcessing.addCompleteTimeToEventLog(ex1_LogPageIf)
    # LogPageActivityCountByUser = logPage.groupby([pd.Grouper(key='org:resource'),pd.Grouper(key='concept:name')]).count()
    LogPageactivityCountByUser = FCAMiner.activityDataMatrixContruct(logPage)
    LogPageactivityCountByUser = LogPageactivityCountByUser.fillna(0)
    excellent_LogPageActivityCountByUser = LogPageactivityCountByUser.loc[LogPageactivityCountByUser.index.isin(excellentList)]
    weak_LogPageActivityCountByUser = LogPageactivityCountByUser.loc[LogPageactivityCountByUser.index.isin(weakList)]
    
    
    excellent_LogPageActivityCountByUser['result_exam_1'] = 1
    weak_LogPageActivityCountByUser['result_exam_1'] = 0
    LogPageactivityCountByUser = pd.concat([excellent_LogPageActivityCountByUser,weak_LogPageActivityCountByUser])
    # cum_practice_col = ['correct_adjusted','successPassedRate'] #,'cumm_practice','successPassedRate'
    # LogPageactivityCountByUser = LogPageactivityCountByUser.merge(practice.loc[:,cum_practice_col], 
    #                                         left_on=LogPageactivityCountByUser.index, right_on=practice.index)    
    # LogPageactivityCountByUser = LogPageactivityCountByUser.set_index('key_0')
    
    colSelection = []#['correct_adjusted','successPassedRate'] #select only col whose p-value <=0.05 with result exam
    for col in LogPageactivityCountByUser.columns:
         if col not in ['result_exam_1','click-3']:
        #     corr = pearsonr(LogPageactivityCountByUser['result_exam_1'], LogPageactivityCountByUser[col])
        #     if corr[1] <= pvalue:
        #         colSelection.append(col)    
            colSelection.append(col)
    
    if len(colSelection) == 0:
        return 'No columns correlated with result_exam_1'

    X=LogPageactivityCountByUser[colSelection] 
    y=LogPageactivityCountByUser['result_exam_1']                                       
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1) # 70% training and 30% test
    
    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=1000)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)
    y_pred_proba = clf.predict_proba(X_test)
    y_pred = make_decision(y_pred_proba, 0.5)
    
    
    accuracy_score = metrics.accuracy_score(y_test, y_pred),
    f1_score = metrics.f1_score(y_test, y_pred),
    precision_score = metrics.precision_score(y_test, y_pred),
    recall_score = metrics.recall_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    
    metricsResult = {
        'accuracy_score' : accuracy_score,
        'f1_score' : f1_score,
        'precision_score' : precision_score,
        'recall_score' : recall_score,
        'roc_auc' : roc_auc
    }
    
    feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
    testingData = pd.DataFrame(columns=['y_test','y_pred','y_pred_proba_0','y_pred_proba_1'])
    testingData['y_test'] = y_test
    testingData['y_pred'] = y_pred
    testingData['y_pred_proba_0'] = y_pred_proba[:,0]
    testingData['y_pred_proba_1'] = y_pred_proba[:,1]
    return [metricsResult, feature_imp, clf,LogPageactivityCountByUser,testingData]




#All in one prediction
def predict_proba_all_algorithms_data_ready(PCAdata, excellentList, weakList,practice=[], mode = 'activity'):
    # LogPageactivityCountByUser = FCAMiner.activityDataMatrixPercentage(LogPageactivityCountByUser)
    excellent_PCAdata = PCAdata.loc[PCAdata.index.isin(excellentList)]
    weak_PCAdata = PCAdata.loc[PCAdata.index.isin(weakList)]
    
    
    excellent_PCAdata['result_exam_1'] = 1
    weak_PCAdata['result_exam_1'] = 0
    PCAdata = pd.concat([excellent_PCAdata,weak_PCAdata])
    
    
    if len(practice) > 0:
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
        n_estimators=100,
        max_depth=5,
        min_child_weight=5,
        objective= 'binary:logistic'
    )
    
    lr = LogisticRegression()
    
    # ada = AdaBoostClassifier(
    #         DecisionTreeClassifier(max_depth=5),
    #         n_estimators=1000
    #     )
    
    rf=RandomForestClassifier(n_estimators=1000)
    
    gb = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        # max_features=2,
        max_depth=5
    )
    
    svmL = svm.SVC(kernel='rbf', probability=True)
    
    dt = DecisionTreeClassifier()
    # lda = LinearDiscriminantAnalysis()
    # mlp = MLPClassifier()
    
    knn = KNeighborsClassifier(n_neighbors=3)
    
    classifiers = [('XGBoost',xgb),('Logistic Regression',lr), ('Decision Tree',dt),('SVM',svmL),('KNN',knn)]
    
    # classifiers = [('Logistic Regression',lr),('SVM',svmL)]
    #Train the model using the training sets y_pred=clf.predict(X_test)
    result = {}
    for name, classifier in classifiers:
        print(name + '...')
        classifier.fit(X_train,y_train)
        y_pred_proba=classifier.predict_proba(X_test)
    
        y_pred = make_decision(y_pred_proba, 0.5)
        
        scores = cross_val_score(classifier, X, y, cv=10, scoring='roc_auc')
        scores_recall = cross_val_score(classifier, X, y, cv=10, scoring='recall')
        scores_f1 = cross_val_score(classifier, X, y, cv=10, scoring='f1')
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
        
        if name not in ['Logistic Regression','Linear Discriminant','Multi Layer Nets','SVM','KNN']:
            feature_imp = pd.Series(classifier.feature_importances_,index=X.columns).sort_values(ascending=False)
        else:
            feature_imp = 'None'
        
        testingData = pd.DataFrame(columns=['y_test','y_pred','y_pred_proba_0','y_pred_proba_1'])
        testingData['y_test'] = y_test
        testingData['y_pred'] = y_pred
        testingData['y_pred_proba_0'] = y_pred_proba[:,0]
        testingData['y_pred_proba_1'] = y_pred_proba[:,1]
        result.update({name : [metricsResult, feature_imp, classifier,testingData, scores, confusion_matrix1, classification_report1, scores_f1, scores_recall]})
    result.update({'data' : PCAdata})
    return result


def evaluateModelWithData(model,logPage, excellentList, weakList, practice, lectureList = [], mode='transition',originalElements = []):
    if mode == 'activity':
        LogPageactivityCountByUser = FCAMiner.activityDataMatrixContruct(logPage)
        LogPageactivityCountByUser = LogPageactivityCountByUser.fillna(0)
        LogPageactivityCountByUser = FCAMiner.activityDataMatrixPercentage(LogPageactivityCountByUser)
    elif mode == 'transition':
        LogPageactivityCountByUser = FCAMiner.transitionDataMatrixConstruct_for_prediction(logPage, ['Other','Read_Lecture_Note','Check_solution','Exercise','Read_Labsheet'])
        LogPageactivityCountByUser = LogPageactivityCountByUser.fillna(0)
        LogPageactivityCountByUser = LogPageactivityCountByUser.groupby([pd.Grouper(key='user')]).sum()
        LogPageactivityCountByUser = FCAMiner.transitionDataMatrixConstruct_for_prediction_percentage(LogPageactivityCountByUser)
        
    excellent_LogPageActivityCountByUser = LogPageactivityCountByUser.loc[LogPageactivityCountByUser.index.isin(excellentList)]
    weak_LogPageActivityCountByUser = LogPageactivityCountByUser.loc[LogPageactivityCountByUser.index.isin(weakList)]
    
    
    excellent_LogPageActivityCountByUser['result_exam_1'] = 1
    weak_LogPageActivityCountByUser['result_exam_1'] = 0
    LogPageactivityCountByUser = pd.concat([excellent_LogPageActivityCountByUser,weak_LogPageActivityCountByUser])

    cum_practice_col = ['successPassedRate'] #,'correct_adjusted,cumm_practice','successPassedRate'
    LogPageactivityCountByUser = LogPageactivityCountByUser.merge(practice.loc[:,cum_practice_col], 
                                            left_on=LogPageactivityCountByUser.index, right_on=practice.index)    
    LogPageactivityCountByUser = LogPageactivityCountByUser.set_index('key_0')
    
    
    X=LogPageactivityCountByUser.drop(['result_exam_1'], axis = 1)
    y_test=LogPageactivityCountByUser['result_exam_1'] 
    
    y_pred_proba=model.predict_proba(X)    
    y_pred = make_decision(y_pred_proba, 0.5)

    confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
    
    accuracy_score = metrics.accuracy_score(y_test, y_pred),
    f1_score = metrics.f1_score(y_test, y_pred),
    precision_score = metrics.precision_score(y_test, y_pred),
    recall_score = metrics.recall_score(y_test, y_pred),
    roc_auc = metrics.roc_auc_score(y_test, y_pred)
    classification_report = metrics.classification_report(y_test,y_pred)
        
    metricsResult = {
        'accuracy_score' : accuracy_score,
        'f1_score' : f1_score,
        'precision_score' : precision_score,
        'recall_score' : recall_score,
        'roc_auc' : roc_auc
    }
    testingData = pd.DataFrame(columns=['y_test','y_pred','y_pred_proba_0','y_pred_proba_1'])
    testingData['y_test'] = y_test
    testingData['y_pred'] = y_pred
    testingData['y_pred_proba_0'] = y_pred_proba[:,0]
    testingData['y_pred_proba_1'] = y_pred_proba[:,1]
    
    return [metricsResult, testingData, classification_report, confusion_matrix]

def algorithmComparisonGraph(field, predictionReport, algorithmList, title = ''):
    if len(algorithmList) == 0:
        algorithmList = predictionReport['algorithm'].unique()
    plt.figure(figsize=(20,15))
    colorList = ['b','g', 'r', 'c', 'm', 'y', 'k', 'purple']
    for al,color in zip(algorithmList,colorList):
        data_al = predictionReport.loc[predictionReport['algorithm'] == al] 
        plt.plot(data_al['week'], data_al[field],
                  'o-', color=color, label=al, markersize=10)

        # plt.plot(predictionReport['week'], predictionReport['recall'],
        #           'o-', color='orange', label='recall_score', markersize=10)
    plt.title(title,fontsize=20)  
    plt.xticks(np.arange(0, 14), fontsize=20)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=20)
    plt.xlabel("Week", fontsize=20)
    plt.ylabel(field + ' Scores', fontsize=20)
    plt.grid()
    plt.legend(loc="lower right", fontsize=18)
    plt.show()
    
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

def trainModel(X, y):
    xgb = XGBClassifier(
        learning_rate =0.05,
        n_estimators=100,
        max_depth=5,
        min_child_weight=5,
        objective= 'binary:logistic'
    )
    
    lr = LogisticRegression()
    
    # ada = AdaBoostClassifier(
    #         DecisionTreeClassifier(max_depth=5),
    #         n_estimators=1000
    #     )
    
    rf=RandomForestClassifier(n_estimators=1000)
    
    gb = GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        # max_features=2,
        max_depth=5
    )
    
    svmL = svm.SVC(kernel='rbf', probability=True)
    
    dt = DecisionTreeClassifier()
    # lda = LinearDiscriminantAnalysis()
    # mlp = MLPClassifier()
    
    knn = KNeighborsClassifier(n_neighbors=3)
    
    classifiers = [('XGBoost',xgb),('Logistic Regression',lr), ('Decision Tree',dt),('SVM',svmL),('KNN',knn)]
    result = {}
    for name, classifier in classifiers:
        print(name + '...')
        classifier.fit(X,y)
        
        scores = cross_val_score(classifier, X, y, cv=10, scoring='roc_auc')
        scores_recall = cross_val_score(classifier, X, y, cv=10, scoring='recall')
        scores_f1 = cross_val_score(classifier, X, y, cv=10, scoring='f1')

        if name not in ['Logistic Regression','Linear Discriminant','Multi Layer Nets','SVM','KNN']:
            feature_imp = pd.Series(classifier.feature_importances_,index=X.columns).sort_values(ascending=False)
        else:
            feature_imp = 'None'
        
        result.update({name : [feature_imp, classifier, scores, scores_f1, scores_recall]})
    result.update({'data' : X})
    return result
    
def evaluateTestData(model, X_test, y_test):
    
    y_pred_proba=model.predict_proba(X_test)    
    y_pred = make_decision(y_pred_proba, 0.5)
    
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
    
    testingData = pd.DataFrame(columns=['y_test','y_pred','y_pred_proba_0','y_pred_proba_1'])
    testingData['y_test'] = y_test
    testingData['y_pred'] = y_pred
    testingData['y_pred_proba_0'] = y_pred_proba[:,0]
    testingData['y_pred_proba_1'] = y_pred_proba[:,1]
    
    return [metricsResult, testingData, classification_report1, confusion_matrix]
        
    