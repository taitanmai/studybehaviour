import pandas as pd
from datetime import datetime 
import os
import re
import json
from sklearn.preprocessing import StandardScaler


def get_event_log_by_file(basePath, m,u,d):
    corePath = basePath + 'anon/activity/' #'2020-06-29-einstein/activity/'
    path = corePath + m + '/'
    path += u + '/'
    path += d 
    """+ '-activity.log' """
    if os.path.exists(path):
        # print(path)
        f = open(path,'r',encoding="utf8",errors='ignore')
        temp = []
        for i in f:
            split = i.split(' ',6)
            if len(split)>5:
                # if split[0] != ';;' and split[5] != 'upload' and split[5] != 'is-lab-exam:':
                if split[0] != ';;':
                    temp.append(split)
                
        data = pd.DataFrame(temp)
        data.columns = ["date", "time", "moduleCode", "org:resource", "case", "concept:name", "description"]        
        columns = ["case:concept:name", "concept:instance", "concept:name","time:timestamp", "org:resource", "lifecycle:transition","description"]
        data1 = pd.DataFrame(columns = columns)
        data1["case:concept:name"] = data["date"].apply(str) + '-' + data["org:resource"].apply(str) + '-' + data["case"]
        data1["concept:instance"] = data["concept:name"]
        data1["time:timestamp"] = data["date"].apply(str) + ' ' + data['time'].apply(str)
        data1["lifecycle:transition"] = 'start'
        data1["org:resource"] = data["org:resource"]
        data1["concept:name"] = data["concept:name"]
        data1["description"] = data["description"]
        #data1['time:timestamp'] = pd.to_datetime(data1['time:timestamp'])
        """
        for index, row in data1.iterrows():             
            try:
                row['time:timestamp'] = datetime.strptime(row['time:timestamp'], '%Y-%m-%d %H:%M:%S')
            except:
                pass      
        """                
        return data1
    else:
        return []
    
    
def get_eventlogs_by_condition(basePath, module=[],user=[],date=[]):
    corePath = path =  basePath + 'anon/activity/' #'2020-06-29-einstein/activity/'
    if len(module) == 0:
        path = corePath
        module = os.listdir(path)
    module = list(dict.fromkeys(module)) 
    
    if len(user) == 0:
        for m in module:
              path = corePath + m + '/'
              users = os.listdir(path)
              for u in users:
                  user.append(u)        
    user = list(dict.fromkeys(user))
    if len(date) == 0:
        for m in module:
            for u in user:
                path = corePath + m + '/' + u + '/'                
                dates = os.listdir(path)
                for d in dates:
                    date.append(d)    
    date = list(dict.fromkeys(date))
       
    logs = pd.DataFrame()
    for m in module:        
        for u in user:
            for d in date:                
                log = get_event_log_by_file(basePath, m,u,d)                
                if len(log) > 0:                    
                    logs = pd.concat([logs,log])    
    return logs

def get_event_log_by_file_2019(basePath, m,u,d):
    corePath = basePath + '2020-06-29-einstein/activity/'
    path = corePath + m + '/'
    path += u + '/'
    path += d 
    """+ '-activity.log' """
    if os.path.exists(path):
        print(path)
        f = open(path,'r',encoding="utf8",errors='ignore')
        temp = []
        for i in f:
            split = i.split(' ',6)
            if len(split)>5:
                # if split[0] != ';;' and split[5] != 'upload' and split[5] != 'is-lab-exam:':
                if split[0] != ';;':
                    temp.append(split)
                
        data = pd.DataFrame(temp)
        data.columns = ["date", "time", "moduleCode", "org:resource", "case", "concept:name", "description"]        
        columns = ["case:concept:name", "concept:instance", "concept:name","time:timestamp", "org:resource", "lifecycle:transition","description"]
        data1 = pd.DataFrame(columns = columns)
        data1["case:concept:name"] = data["date"].apply(str) + '-' + data["org:resource"].apply(str) + '-' + data["case"]
        data1["concept:instance"] = data["concept:name"]
        data1["time:timestamp"] = data["date"].apply(str) + ' ' + data['time'].apply(str)
        data1["lifecycle:transition"] = 'start'
        data1["org:resource"] = data["org:resource"]
        data1["concept:name"] = data["concept:name"]
        data1["description"] = data["description"]
        #data1['time:timestamp'] = pd.to_datetime(data1['time:timestamp'])
        """
        for index, row in data1.iterrows():             
            try:
                row['time:timestamp'] = datetime.strptime(row['time:timestamp'], '%Y-%m-%d %H:%M:%S')
            except:
                pass      
        """                
        return data1
    else:
        return []
    
    
def get_eventlogs_by_condition_2019(basePath, module=[],user=[],date=[]):
    corePath = path =  basePath + '2020-06-29-einstein/activity/'
    if len(module) == 0:
        path = corePath
        module = os.listdir(path)
    module = list(dict.fromkeys(module)) 
    
    if len(user) == 0:
        for m in module:
              path = corePath + m + '/'
              users = os.listdir(path)
              for u in users:
                  user.append(u)        
    user = list(dict.fromkeys(user))
    if len(date) == 0:
        for m in module:
            for u in user:
                path = corePath + m + '/' + u + '/'                
                dates = os.listdir(path)
                for d in dates:
                    date.append(d)    
    date = list(dict.fromkeys(date))
       
    logs = pd.DataFrame()
    for m in module:        
        for u in user:
            for d in date:                
                log = get_event_log_by_file_2019(basePath, m,u,d)                
                if len(log) > 0:                    
                    logs = pd.concat([logs,log])    
    return logs

def buildActivityUploadData(activity, allActivities, sourceData):
    actUpload = []
    for i in activity:
        act = allActivities.loc[allActivities['concept:name'] == i]
        act = act.groupby('org:resource').count()
        act.drop(['case:concept:name','concept:instance','time:timestamp','lifecycle:transition','description'], axis=1, inplace=True)
        columnName = i+'count'
        act.columns = [columnName]
        if len(actUpload) > 0:
            actUpload = actUpload.merge(act, left_on=data1.index, right_on=act.index)
        else:
            actUpload = data1.merge(act, left_on=data1.index, right_on=act.index)
        actUpload = actUpload.set_index('key_0')
    return actUpload
def pearsonGeneration(field,columns,data):
    array = []
    for i in field:   
        j = [i]
        j.extend(list(stats.pearsonr(activityUpload1[columns], activityUpload1[i+'count'])))
        array.append(j)
    result = pd.DataFrame(array,columns=['activity','correlation','p-value'])
    return result
""" some other things 
from os import listdir
from os.path import isfile, join
path = '/home/tai/Python_data_analytics/einstein/einstein-anon/anon/activity/'
a = os.listdir(path)
"""

def getUploadReportContent(m,u,d,file, basePath):
    corePath =  basePath + 'anon/uploads/' #'2020-06-29-einstein/uploads/'
    path = corePath + m + '/'
    path += u + '/'
    path += d + '/'
    path += file
    # print(path)
    if os.path.exists(path):
        f = open(path,'r',encoding="utf8",errors='ignore')
        content1 = f.read()
        j = json.loads(content1)
        f.close()
        columns = ["date","module","user","task","language","correct","failed","passed","version","timeout","extension","ip"]
        content = []
        for index, c in enumerate(columns,start=0):
            content.append(j[c])        
        return content
    else:
        return []

def extractUploadsData(basePath, module = [],user=[],date=[]):
    corePath = basePath + 'anon/uploads/'                    

    if len(module) == 0:
        path = corePath
        if os.path.exists(path):
            module = os.listdir(path)
    module = list(dict.fromkeys(module)) 
    
    if len(user) == 0:
        for m in module:
              path = corePath + m + '/'
              if os.path.exists(path):
                  users = os.listdir(path)
                  for u in users:
                      user.append(u)        
    user = list(dict.fromkeys(user))

    if len(date) == 0:
        for m in module:
            for u in user:
                path = corePath + m + '/' + u + '/'
                if os.path.exists(path):                
                    dates = os.listdir(path)
                    for d in dates:
                        date.append(d)    
    date = list(dict.fromkeys(date))
    
    columns = ["date","module","user","task","language","correct","failed","passed","version","timeout","extension","ip"]
    
    resultArray = []
    for m in module:
        for u in user:
            for d in date:
                path = corePath + m + '/' + u + '/' + d + '/'
                if os.path.exists(path):
                    files = os.listdir(path)
                    for f in files:                        
                        if re.search(".report.(20|19|18)\d{2}-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]_[0-9][0-9].json$",f):
                            print(f)
                            temp = getUploadReportContent(m,u,d,f,basePath)
                            resultArray.append(temp)
    # print(resultArray)
    result = pd.DataFrame(resultArray,columns=columns)
    result['date'] = pd.to_datetime(result.date)
    return result  

#extract upload for 2019 data
def getUploadReportContent_2019(basePath, m,u,d,file):
    corePath = basePath + '2020-06-29-einstein/uploads/' #'einstein-anon/anon/uploads/'
    path = corePath + d + '/'
    path += m + '/'
    path += u + '/'
    path += file
    if os.path.exists(path):
        f = open(path,'r',encoding="utf8",errors='ignore')
        content1 = f.read()
        j = json.loads(content1)
        f.close()
        columns = ["date","module","user","task","language","correct","failed","passed","version","timeout","extension","ip"]
        content = []
        for index, c in enumerate(columns,start=0):
            content.append(j[c])  
        print(path)
        return content
    else:
        return []

def extractUploadsData_2019(basePath, module = [],user=[],date=[]):
    corePath = basePath + '2020-06-29-einstein/uploads/' #'einstein-anon/anon/uploads/'                    

    if len(date) == 0:
        path = corePath
        if os.path.exists(path):
            date = os.listdir(path)
    date = list(dict.fromkeys(date)) 
    
    if len(module) == 0:
        for d in date:
              path = corePath + d + '/'
              if os.path.exists(path):
                  modules = os.listdir(path)
                  for m in modules:
                      module.append(m)        
    module = list(dict.fromkeys(module))

    if len(user) == 0:
        for d in date:
            for m in module:
                path = corePath + d + '/' + m + '/'
                if os.path.exists(path):                
                    users = os.listdir(path)
                    for u in users:
                        user.append(u)    
    user = list(dict.fromkeys(user))
    
    columns = ["date","module","user","task","language","correct","failed","passed","version","timeout","extension","ip"]
    
    resultArray = []
    for d in date:
        for m in module:
            for u in user:
                path = corePath + d + '/' + m + '/' + u + '/'
                if os.path.exists(path):
                    files = os.listdir(path)
                    for f in files:                        
                        if re.search(".report.(20|19|18)\d{2}-[0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]_[0-9][0-9].json$",f):
                            # print(f)
                            temp = getUploadReportContent_2019(basePath, m,u,d,f)
                            resultArray.append(temp)
    # print(resultArray)
    result = pd.DataFrame(resultArray,columns=columns)
    result['date'] = pd.to_datetime(result.date)
    return result  

#calculate assessment result
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

def addCompleteTimeToEventLog(dfEventLog):
    newData = []
    if 'case' not in dfEventLog.columns:
        dfEventLog['case'] = dfEventLog['case:concept:name']
    if 'activity' not in dfEventLog.columns:
        dfEventLog['activity'] = dfEventLog['concept:name']
    dfEventLog = dfEventLog.sort_values(by=['case','time:timestamp'])
    dfEventLog = dfEventLog.set_index(['case','activity'])
    for index, row in dfEventLog.groupby(level=0):
        for i in range(0,len(row['concept:instance'])):
            newRaw = []
            newRaw.append(row['case:concept:name'][i])
            newRaw.append(row['concept:instance'][i])
            newRaw.append(row['concept:name'][i])
            newRaw.append(row['time:timestamp'][i])
            newRaw.append(row['org:resource'][i])
            newRaw.append(row['lifecycle:transition'][i])
            newRaw.append(row['description'][i])
            if i < len(row['concept:instance']) - 1:
                newRaw.append(row['time:timestamp'][i+1])
            else:
                newRaw.append(row['time:timestamp'][i])
            newData.append(newRaw)
    result = pd.DataFrame(newData,columns=['case:concept:name','concept:instance',
                                           'concept:name','time:timestamp',
                                           'org:resource',
                                           'lifecycle:transition', 
                                           'description','time:timestamp_complete'])
    return result
            
            
def getLectureList(dfEventLog, fileToGet = ['html']):
    labsheets = dfEventLog.loc[dfEventLog['description'].str.contains('|'.join(fileToGet))]
    # labsheets = labsheets.drop(labsheets.loc[labsheets['description'].str.contains('http')].index)
    # labsheets = labsheets.drop(labsheets.loc[labsheets['description'].str.contains('report.html')].index)
    # labsheets = labsheets.drop(labsheets.loc[labsheets['description'].str.contains('index.html')].index)
    # labsheets = labsheets.drop(labsheets.loc[labsheets['description'].str.contains('dashboard')].index)
    # labsheets = labsheets.drop(labsheets.loc[labsheets['description'].str.contains('graphs.html')].index)
    # labsheets = labsheets.drop(labsheets.loc[labsheets['description'].str.contains('sick')].index)
    # labsheets = labsheets.drop(labsheets.loc[labsheets['description'].str.contains('log.html')].index)
    pages = labsheets['description'].unique()

    pageList = []
    for p in pages:        
        element = p.split(' ',2)
        if element[0].strip() != '/':
            pageList.append(element[0].strip())
        
    pageList1 = []
    for pL in pageList:
        element1 = pL.split('#',2)
        if element1[0].strip() != '/':
            pageList1.append(element1[0].strip())
        
    lectureList = list(set(pageList1))
    return lectureList


def addConceptPageToLog(dfEventLog):
    pages = []
    for index, row in dfEventLog.iterrows():
        if '#' in row['description']:
            pages.append(row['description'].split('#')[0])
        else:
            pages.append(row['description'].split(' ')[0])
    dfEventLog['page'] = pages
    return dfEventLog

# basePath = 'G:\\Dataset\\PhD\\'  
# result = extractUploadsData_2019(basePath, module = ['ca177'])   
# result.to_csv(basePath + "ca177_uploads_2019.csv",index=False)    


# module = ['ca277']
# user = [] #get all users
# date = [] #get all dates


# log1 = get_eventlogs_by_condition_2019(basePath, module,user,date=date)

# # # # # a = log1.loc[(log1['concept:instance'].str.contains('is-lab-exam') | log1['concept:instance'].str.contains('upload')) & (log1['org:resource'] == 'u-114b810c95a5ebd746ed7b4ad73634929caa83d8')]
# # # # # u-114b810c95a5ebd746ed7b4ad73634929caa83d8
# log1.to_csv(basePath + "ca277_eventLog_nonfixed_2019.csv",index=False) 




def normaliseWeeklyData(data):
    result = []
    for w in range(len(data)):
        x = data[w].values
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        result.append(pd.DataFrame(x,columns=data[w].columns, index = data[w].index))
    return result
        
def normaliseData(data):
    x = data.values
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    return pd.DataFrame(x, columns = data.columns, index = data.index)

def reLabelStudentId(studentList):
    result = {}
    for i in range(0, len(studentList)):
        result[studentList[i]] = i
    return result
