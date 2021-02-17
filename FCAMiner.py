import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import ks_2samp
import math
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from pm4py.evaluation.replay_fitness import factory as replay_factory
from pm4py.objects.conversion.log import factory as conversion_factory
from pm4py.evaluation.precision import factory as precision_factory

def activityDataMatrixContruct(df, ActivityColumn):
    activityList = df[ActivityColumn].unique()
    result = []    

    for al in activityList:
        tempAct = df.loc[df[ActivityColumn] == al]
        tempAct = tempAct.groupby([pd.Grouper(key='org:resource')]).count()
        tempAct[al] = tempAct[ActivityColumn]
        cols = df.drop(['org:resource'], axis=1).columns
        tempAct.drop(cols, axis=1, inplace=True)    
        if len(result) > 0:
            result = pd.concat([result,tempAct], axis=1)    
        else:
            result = tempAct
    return result

def activityDataMatrixPercentage(activityDataMatrix):
    result = []
    cols = activityDataMatrix.columns
    indexList = []
    for index, row in activityDataMatrix.iterrows():
        rowResult = []
        totalActivity = sum(row)
        indexList.append(index)
        for col in cols:
            if totalActivity > 0:
                rowResult.append(float(row[col])/float(totalActivity))
            else:
                rowResult.append(0)
        result.append(rowResult)
    return pd.DataFrame(result,columns=cols,index=indexList)

def activityDataMatrixRelativeCorrelation(activityDataMatrix):
    result = []
    cols = activityDataMatrix.columns
    cols = ['Correct_Excercise','Incorrect_Excercise','Read_Lecture_Note','Check_solution','Practice']
    indexList = []
    for index, row in activityDataMatrix.iterrows():
        rowResult = []
        maxFrequency = max(row)
        indexList.append(index)
        for col in cols:
            if maxFrequency > 0:
                rowResult.append(float(row[col])/float(maxFrequency))
            else:
                rowResult.append(0)
        result.append(rowResult)
    return pd.DataFrame(result,columns=cols,index=indexList)

def activityTimeDataMatrixContruct(df, ActivityColumn): #Inprogress
    activityList = df[ActivityColumn].unique()
    result = []

        
    for al in activityList:
        tempAct = df.loc[df['concept:name'] == al]
        tempAct = tempAct.groupby([pd.Grouper(key='org:resource')]).count()
        tempAct[al] = tempAct['concept:name']
        cols = df.drop(['org:resource'], axis=1).columns
        tempAct.drop(cols, axis=1, inplace=True)    
        if len(result) > 0:
            result = pd.concat([result,tempAct], axis=1)    
        else:
            result = tempAct
    return result

def transitionDataMatrixConstruct_directFollow(dfEventLog, column, originalElements = [], activityCount = False, mode='count'):
    if len(originalElements) == 0:
        originalElements = dfEventLog[column].unique()
    columns = []
    # columns.append('case')
    # columns.append('startDate')
    # columns.append('endDate')
    columns.append('user')
    for i in originalElements:
        for j in originalElements:
            # if i != j:
            txt = i + '-' + j
            columns.append(txt)
    columns = list(dict.fromkeys(columns))
    
    allRow = []
    dfEventLog['case'] = dfEventLog['case:concept:name']
    dfEventLog['activity'] = dfEventLog['concept:name']
    dfEventLog = dfEventLog.set_index(['case','activity'])
    dfEventLog = dfEventLog.sort_values(by=['case','time:timestamp'])
    indexList = []
    for index,row in dfEventLog.groupby(level=0):
        newRow = {}
        indexList.append(index)
        newRow['user'] = row['org:resource'][0] #splitIndex[3]+'-'+splitIndex[4]
        # newRow['startDate'] = row['time:timestamp'][0]
        # newRow['endDate'] = row['time:timestamp'][len(row)-1]
        # newRow['case'] = index
        for i in range(len(row[column])-1):
            key = row[column][i]+'-'+row[column][i+1]
            if key in columns:
                if mode == 'count':
                    flag = 1
                elif mode == 'time':
                    flag = ((row['time:timestamp'][i+1] - row['time:timestamp'][i])/np.timedelta64(1,'s'))
                else:
                    return 'mode_undefined'
                if key not in newRow:
                    newRow[key] = flag
                else:
                    newRow[key] = newRow[key] + 1               
        allRow.append(newRow)
    activityVariance = pd.DataFrame(allRow,columns=columns,index=indexList)
    
    if activityCount:
        result = []
        for al in originalElements:
            tempAct = dfEventLog.loc[dfEventLog[column] == al]
            tempAct = tempAct.groupby([pd.Grouper(key='case:concept:name')]).agg({column: "count"})
            tempAct[al] = tempAct[column]

            cols = [column] #dfEventLog.drop(['case:concept:name'], axis=1).columns
            tempAct.drop(cols, axis=1, inplace=True)    
            if len(result) > 0:
                result = pd.concat([result,tempAct], axis=1)    
            else:
                result = tempAct        
        activityVariance = pd.concat([result, activityVariance], axis=1)    
    return activityVariance

def transitionDataMatrixConstruct_eventuallyFollow(dfEventLog, originalElements = [], mode = 'count'):
    
    if len(originalElements) == 0:
        originalElements = dfEventLog['concept:name'].unique()
    columns = []
    # columns.append('case')
    # columns.append('startDate')
    # columns.append('endDate')
    columns.append('user')
    for i in originalElements:
        for j in originalElements:
            # if i != j:
            txt = i + '-' + j
            columns.append(txt)
    columns = list(dict.fromkeys(columns))
    
    allRow = []
    dfEventLog['case'] = dfEventLog['case:concept:name']
    dfEventLog['activity'] = dfEventLog['concept:name']
    dfEventLog = dfEventLog.set_index(['case','activity'])
    dfEventLog = dfEventLog.sort_values(by=['case','time:timestamp'])
    indexList = []
    
    for index,row in dfEventLog.groupby(level=0):
        newRow = {}
        indexList.append(index)
        newRow['user'] = row['org:resource'][0] #splitIndex[3]+'-'+splitIndex[4]
        # newRow['startDate'] = row['time:timestamp'][0]
        # newRow['endDate'] = row['time:timestamp'][len(row)-1]
        # newRow['case'] = index
        for i in range(len(row['concept:instance'])-1):
            for j in range(i+1, len(row['concept:instance'])):
                key = row['concept:instance'][i]+'-'+row['concept:instance'][j]
                if key in columns:
                    if mode == 'count':
                        flag = 1
                    elif mode == 'distance':
                        flag = j - i
                    elif mode == 'time':
                        flag = ((row['time:timestamp'][j] - row['time:timestamp'][i])/np.timedelta64(1,'s'))
                    else:
                        return 'mode_undefined'
                    if key not in newRow:
                        newRow[key] = flag
                    else:
                        newRow[key] = newRow[key] + flag               
        allRow.append(newRow)
    activityVariance = pd.DataFrame(allRow,columns=columns,index=indexList)
    return activityVariance

def transitionDataMatrixConstruct_time(dfEventLog, originalElements = [], activityColumn = 'concept:instance'):
    if len(originalElements) == 0:
        originalElements = dfEventLog[activityColumn].unique()
    columns = []
    # columns.append('case')
    # columns.append('startDate')
    # columns.append('endDate')
    columns.append('user')
    for i in originalElements:
        for j in originalElements:
            # if i != j:
            txt = i + '-' + j
            columns.append(txt)
    columns = list(dict.fromkeys(columns))

    allRow = []
    allRow1 = []
    dfEventLog['case'] = dfEventLog['case:concept:name']
    dfEventLog['activity'] = dfEventLog[activityColumn]
    dfEventLog = dfEventLog.set_index(['case','activity'])
    dfEventLog = dfEventLog.sort_values(by=['case','time:timestamp'])
    indexList = []
    for index,row in dfEventLog.groupby(level=0):
        newRow = {}
        newRow1 = {}
        indexList.append(index)
        newRow['user'] = row['org:resource'][0] #splitIndex[3]+'-'+splitIndex[4]
        newRow1['user'] = row['org:resource'][0]
        # newRow['startDate'] = row['time:timestamp'][0]
        # newRow['endDate'] = row['time:timestamp'][len(row)-1]
        # newRow['case'] = index
        for i in range(len(row[activityColumn])-1):
            key = row[activityColumn][i]+'-'+row[activityColumn][i+1]
            tempTime = ((row['time:timestamp'][i+1] - row['time:timestamp'][i])/np.timedelta64(1,'s'))
            if key in columns:
                if key not in newRow:
                    newRow1[key] = 0
                    newRow[key] = tempTime
                else:
                    newRow1[key] = newRow1[key] + 1
                    newRow[key] = newRow[key] + tempTime               
        allRow1.append(newRow1)
        allRow.append(newRow)
    activityVariance = pd.DataFrame(allRow,columns=columns,index=indexList)
    activityVariance1 = pd.DataFrame(allRow1,columns=columns,index=indexList)
    return [activityVariance,activityVariance1] #frequency and time

def transitionDataMatrixConstruct_for_prediction_percentage(activityVariance):
    result = []
    cols = activityVariance.columns
    indexList = []
    for index, row in activityVariance.iterrows():
        rowResult = []
        totalActivity = sum(row)
        indexList.append(index)
        for col in cols:
            if totalActivity > 0:
                rowResult.append(float(row[col])/float(totalActivity))
            else:
                rowResult.append(0)
        result.append(rowResult)
    return pd.DataFrame(result,columns=cols,index=indexList)    
    
def PCAactivity(dataset):
    scaler = StandardScaler()
    result = []
    pca = PCA(n_components=min(len(dataset), len(dataset.columns)))
    x = dataset.values    
    #x_adjust = x - np.mean(x)    
    scaler.fit(x)
    x = scaler.transform(x)
    pca.fit(x)
    transformed_value = pca.fit_transform(x)
    
    # eigenvectors/loadings to array
    count = 1
    for k,l in zip(pca.explained_variance_ratio_,pca.explained_variance_):
        row_list = []

        row_list.append(count)
        row_list.append(k)
        row_list.append(l)
        for j in pca.components_[count-1]:
            row_list.append(j)

        result.append(row_list)
        count = count + 1
    columns2 = []
    columns2.append('pc')
    columns2.append('explained_var_ratio')
    columns2.append('eigenvalues')
    for i in dataset.columns:
        columns2.append(i)
    result1 = pd.DataFrame(result,columns=columns2)
    #result1.drop(['pc'])
    return result1

def PCAactivityValue(dataset):
    scaler = StandardScaler()
    result = []
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
    transformed_value1 = pd.DataFrame(transformed_value,columns=columns)
    
    return transformed_value1

def buildTranstionFromfilteredActivity(a):
    columns2 = []
    for i in a:
        for j in a:
            txt = i + '-' + j
            columns2.append(txt)
    return columns2

def PCAcohort(dataset):
    scaler = StandardScaler()
    result = []
    pca = PCA(n_components=min(len(dataset), len(dataset.columns)))
    x = dataset.values    
    #x_adjust = x - np.mean(x)    
    scaler.fit(x)
    x = scaler.transform(x)
    pca.fit(x)
    transformed_value = pca.fit_transform(x)
    
    # eigenvectors/loadings to array
    count = 1
    for k,l in zip(pca.explained_variance_ratio_,pca.explained_variance_):
        row_list = []

        row_list.append(count)
        row_list.append(k)
        row_list.append(l)
        for j in pca.components_[count-1]:
            row_list.append(j)

        result.append(row_list)
        count = count + 1
    columns2 = []
    columns2.append('pc')
    columns2.append('explained_var_ratio')
    columns2.append('eigenvalues')
    for i in dataset.columns:
        columns2.append(i)
    result1 = pd.DataFrame(result,columns=columns2)
    return result1

def getNumberOfPCs(df,acceptedPercentage):
    temp = df['explained_var_ratio']
    sumContrPer = 0
    for i,j in zip(temp,range(0,len(temp))):
        sumContrPer = sumContrPer + i
        if sumContrPer >= acceptedPercentage:
            break
    return [sumContrPer,j]

def contrCal(df,numOfPCs):
    #get sum of eigenvalues of the first number of PCs in df
    totalContrPCs = sum(df['eigenvalues'].head(numOfPCs))
    #calculate contribution of variables into number of PCs
    temp = pd.DataFrame(columns=df.columns)
    listContr = []
    columns = df.columns
    for c in columns:        
        if c not in ['explained_var_ratio','eigenvalues']:
            temp[c] = df[c]*df[c]*df['eigenvalues']
            temp1 = [c,sum(temp[c].head(numOfPCs))/totalContrPCs]            
            listContr.append(temp1)
    result = pd.DataFrame(listContr,columns=['transition','contr_percentage'])
    result = result.drop(result.index[0])
    return result

def PCAcohortToValue(dataset):
    scaler = StandardScaler()
    result = []
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

def replayStudentPetriNets(studentList, model1, model2, eventLog):
    students = []
    for student in studentList:
        studentEventLog = eventLog.loc[eventLog['org:resource'] == student]
        #studentEventLog = studentEventLog.sort_values(by=['case','time:timestamp'])
        studentEventLog1 = conversion_factory.apply(studentEventLog)
        fitness1 = replay_factory.apply(studentEventLog1, model1[0], model1[1], model1[2])        
        #precision1 = precision_factory.apply(studentEventLog1, model1[0], model1[1], model1[2])
        
        fitness2 = replay_factory.apply(studentEventLog1, model2[0], model2[1], model2[2])
        #precision2 = precision_factory.apply(studentEventLog1, model2[0], model2[1], model2[2])
        
        eval1 = fitness1['log_fitness'] #(fitness1['log_fitness'] + precision1)/2
        eval2 = fitness2['log_fitness'] #(fitness2['log_fitness'] + precision2)/2
        students.append([student,eval1,eval2])
    result = pd.DataFrame(students,columns=['studentId','model1','model2'])
    return result




