
import pandas as pd
# from numba import jit, cuda 
import numpy as np
import json

# import pyRMT

import warnings

warnings.filterwarnings("ignore")


def getLectureList(dfEventLog, fileToGet = ['html']):
    labsheets = dfEventLog.loc[dfEventLog['description'].str.contains('|'.join(fileToGet))]
    labsheets = labsheets.drop(labsheets.loc[labsheets['description'].str.contains('http')].index)
    labsheets = labsheets.drop(labsheets.loc[labsheets['description'].str.contains('report.html')].index)
    labsheets = labsheets.drop(labsheets.loc[labsheets['description'].str.contains('index.html')].index)
    labsheets = labsheets.drop(labsheets.loc[labsheets['description'].str.contains('dashboard')].index)
    labsheets = labsheets.drop(labsheets.loc
                               [labsheets['description'].str.contains('graphs.html')].index)
    labsheets = labsheets.drop(labsheets.loc[labsheets['description'].str.contains('sick')].index)
    labsheets = labsheets.drop(labsheets.loc[labsheets['description'].str.contains('log.html')].index)
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

eventLog_ca116 = pd.read_csv('ca116_eventLog_nonfixed.csv')
eventLog_ca116 = eventLog_ca116.drop([1160345])
eventLog_ca116['time:timestamp'] = pd.to_datetime(eventLog_ca116['time:timestamp'])
eventLog_ca116 = eventLog_ca116.loc[:, ~eventLog_ca116.columns.str.contains('^Unnamed')]

lectureList = getLectureList(eventLog_ca116,['html|py'])
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


eventLog_ca116_filtered['concept:instance'].unique()

eventLog_ca116_filtered.rename(columns={'concept:instance':'concept:instance1',
                                   'concept:name':'concept:name1',
                                   'case:concept:name' : 'case:concept:name1'}, 
                  inplace=True)
eventLog_ca116_filtered['concept:instance'] = eventLog_ca116_filtered['pageType']
eventLog_ca116_filtered['concept:name'] = eventLog_ca116_filtered['pageType']
eventLog_ca116_filtered['date'] = eventLog_ca116_filtered['time:timestamp'].dt.date

eventLog_ca116_filtered['case:concept:name'] = eventLog_ca116_filtered['date'].astype(str) + '-' + eventLog_ca116_filtered['org:resource'].astype(str)


# eventLog_ca116_filtered['concept:name'] = eventLog_ca116_filtered['pageType'] + '*' + eventLog_ca116_filtered['concept:name1']
# eventLog_ca116_filtered['concept:instance'] = eventLog_ca116_filtered['pageType'] + '*' + eventLog_ca116_filtered['concept:instance1']


# eventLog_ca116_filtered.to_csv("eventLog_ca116_filtered_2018.csv", index=False)
weeksEventLog_filtered = [g for n, g in eventLog_ca116_filtered.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
# a = weeksEventLog_filtered[1]


#convert data for PCA - from eventlog to transition data matrix
workingWeekLog = []
transitionDataMatrixWeeks_distance_eventually = []
full_transitionDataMatrixWeeks_distance_eventually = []
for week in range(1,13):
    print('Week: ' + str(week) + '...')
    workingWeekLog.append(weeksEventLog_filtered[week])
    Log = weeksEventLog_filtered[week] # pd.concat(workingWeekLog) #
    tempTransition = transitionDataMatrixConstruct_eventuallyFollow(Log,[],'distance').fillna(0)
    full_transitionDataMatrixWeeks_distance_eventually.append(tempTransition)   
    tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()         
    transitionDataMatrixWeeks_distance_eventually.append(tempTransition)

for w in range(0,12):
    transitionDataMatrixWeeks_distance_eventually[w].to_csv('transitionMatrixStorage_noScroll/transitionDataMatrixWeeks_distance_eventually_w'+str(w)+'.csv',index=True)
    full_transitionDataMatrixWeeks_distance_eventually[w].to_csv('transitionMatrixStorage_noScroll/full_transitionDataMatrixWeeks_distance_eventually_w'+str(w)+'.csv',index=True)
print('Distance based done')   

transitionDataMatrixWeeks_time_eventually = []
full_transitionDataMatrixWeeks_time_eventually = []
for week in range(1,13):
    print('Week: ' + str(week) + '...')
    workingWeekLog.append(weeksEventLog_filtered[week])
    Log = weeksEventLog_filtered[week] # pd.concat(workingWeekLog) #
    tempTransition = transitionDataMatrixConstruct_eventuallyFollow(Log,[],'time').fillna(0)
    full_transitionDataMatrixWeeks_time_eventually.append(tempTransition)  
    tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()            
    transitionDataMatrixWeeks_time_eventually.append(tempTransition)

for w in range(0,12):
    transitionDataMatrixWeeks_time_eventually[w].to_csv('transitionMatrixStorage_noScroll/transitionDataMatrixWeeks_time_eventually_w'+str(w)+'.csv',index=True)
    full_transitionDataMatrixWeeks_time_eventually[w].to_csv('transitionMatrixStorage_noScroll/full_transitionDataMatrixWeeks_time_eventually_w'+str(w)+'.csv',index=True)
print('Time based done')

transitionDataMatrixWeeks_count_eventually = []
full_transitionDataMatrixWeeks_count_eventually = []
for week in range(1,13):
    print('Week: ' + str(week) + '...')
    workingWeekLog.append(weeksEventLog_filtered[week])
    Log = weeksEventLog_filtered[week] # pd.concat(workingWeekLog) #
    tempTransition = transitionDataMatrixConstruct_eventuallyFollow(Log,[],'count').fillna(0)
    full_transitionDataMatrixWeeks_count_eventually.append(tempTransition)  
    tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()            
    transitionDataMatrixWeeks_count_eventually.append(tempTransition)

for w in range(0,12):
    transitionDataMatrixWeeks_distance_eventually[w].to_csv('transitionMatrixStorage_noScroll/transitionDataMatrixWeeks_count_eventually_w'+str(w)+'.csv',index=True)
    full_transitionDataMatrixWeeks_distance_eventually[w].to_csv('transitionMatrixStorage_noScroll/full_transitionDataMatrixWeeks_count_eventually_w'+str(w)+'.csv',index=True)
print('Count based done')  