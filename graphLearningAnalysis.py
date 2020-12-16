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
import graphLearning
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.stats import kurtosis
import warnings
import time
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import networkx as nx
import PredictionResult
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
# sns.set()
# sns.set_style("whitegrid", {"axes.facecolor": ".9"})
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

eventLog_ca116 = pd.read_csv('ca116_eventLog_nonfixed.csv')
eventLog_ca116 = eventLog_ca116.drop([1160345])
eventLog_ca116['time:timestamp'] = pd.to_datetime(eventLog_ca116['time:timestamp'])
eventLog_ca116 = eventLog_ca116.loc[:, ~eventLog_ca116.columns.str.contains('^Unnamed')]

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


eventLog_ca116_filtered['concept:instance'].unique()

eventLog_ca116_filtered.rename(columns={'concept:instance':'concept:instance1',
                                   'concept:name':'concept:name1',
                                   'case:concept:name' : 'case:concept:name1'}, 
                  inplace=True)
eventLog_ca116_filtered['concept:instance'] = eventLog_ca116_filtered['pageType']
eventLog_ca116_filtered['concept:name'] = eventLog_ca116_filtered['pageType']
eventLog_ca116_filtered['date'] = eventLog_ca116_filtered['time:timestamp'].dt.date

eventLog_ca116_filtered['case:concept:name'] = eventLog_ca116_filtered['date'].astype(str) + '-' + eventLog_ca116_filtered['org:resource'].astype(str)
eventLog_ca116_filtered['concept:name'] = eventLog_ca116_filtered['pageType'] + '*' + eventLog_ca116_filtered['concept:name1']
eventLog_ca116_filtered['concept:instance'] = eventLog_ca116_filtered['pageType'] + '*' + eventLog_ca116_filtered['concept:instance1']


# eventLog_ca116_filtered.to_csv("eventLog_ca116_filtered_2018.csv", index=False)
weeksEventLog_filtered = [g for n, g in eventLog_ca116_filtered.groupby(pd.Grouper(key='time:timestamp',freq='W'))]
for i in weeksEventLog_filtered:
    print(len(i))
    
a = eventLog_ca116.head(10000)
    
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

assessment1A = dataProcessing.assessmentConstruction(ex1,4)
assessment1A['adjustedPerformance'] = (assessment1A['perCorrect'] + assessment1A['perPassed'])/2
assessment2A = dataProcessing.assessmentConstruction(ex2,4)
assessment2A['adjustedPerformance'] = (assessment2A['perCorrect'] + assessment2A['perPassed'])/2
assessment3A = dataProcessing.assessmentConstruction(ex3,4)
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

#convert data for PCA - from eventlog to transition data matrix
workingWeekLog = []
transitionDataMatrixWeeks_directFollow = []
full_transitionDataMatrixWeeks_directFollow = []
for week in range(1,13):
    print('Week: ' + str(week) + '...')
    workingWeekLog.append(weeksEventLog_filtered[week])
    Log =  pd.concat(workingWeekLog) # weeksEventLog_filtered[week] #
    tempTransition = FCAMiner.transitionDataMatrixConstruct_directFollow(Log, [], True).fillna(0)
    full_transitionDataMatrixWeeks_directFollow.append(tempTransition)   
    tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()         
    transitionDataMatrixWeeks_directFollow.append(tempTransition)

#get one random student as an example    
a = transitionDataMatrixWeeks_directFollow[0]
b = a.loc[a.index[2],:]

activityList = weeksEventLog_filtered[1].loc[:,['concept:name']]['concept:name'].unique()
activityCodeList = graphLearning.assignNodeNumber(activityList)
graph = graphLearning.graphCreationForSingleStudent(b,activityCodeList)

nodelist = [i for i in graph._node]
A = nx.adjacency_matrix(graph, nodelist=nodelist, weight=None)
a = A.todense()

adj = graph._adj.tolist()
adj = np.array(graph._adj)

Aw = nx.adjacency_matrix(graph, nodelist=nodelist)
aw = Aw.todense()
aw_array = np.asarray(aw) 
aw_array[0][1]

freq =nx.get_node_attributes(graph,'weight')
graph._node

nodeFeatures = []
for v in freq:
    nodeFeatures.append(freq[v])
nodeFeatures = np.array(nodeFeatures)

def convertAdjToEdgeFeature(A):
    edgesFeatures = [] #m x n x p
    nodes = []
    for i in range(0,len(A)):
        nodes = []
        for j in range(0,len(A)):
            features = []
            for k in range(0,len(A)):            
                features.append([A[j][k]])
            nodes.append(features)

    return nodes

edgesFeature = convertAdjToEdgeFeature(aw_array)


edgesFeature_nparray = np.array(edgesFeature)


#test get all students with full graph features data
import graphLearning
studentAdata = []
studentEdata = []
studentXdata = []
for w in range(0,12):
    activityList = weeksEventLog_filtered[w+1].loc[:,['concept:name']]['concept:name'].unique()
    activityCodeList = graphLearning.assignNodeNumber(activityList)
    A,X,E = graphLearning.constructGraphFeatureForAll(transitionDataMatrixWeeks_directFollow[w],activityCodeList,w)
    studentAdata.append(A)
    studentEdata.append(E)
    studentXdata.append(X)




plt.subplot(111)
nx.draw(graph, pos=nx.circular_layout(graph),  node_color='r', edge_color='b',with_labels = True)

node2vec = Node2Vec(graph, dimensions=128, walk_length=16, num_walks=100)
model = node2vec.fit(window=10, min_count=1)
model.wv.save_word2vec_format('embedding.csv')


model.wv.get_vector('3')



node_ids = model.wv.index2word  # list of node IDs
node_embeddings = (
    model.wv.vectors
)  # numpy.ndarray of size number of nodes times embeddings dimensionality

node_embeddings.sum(axis=0)

tsne = TSNE(n_components=2)
node_embeddings_2d = tsne.fit_transform(node_embeddings)

alpha = 0.7
# label_map = {l: i for i, l in enumerate(np.unique(node_targets))}

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(
    node_embeddings_2d[:, 0],
    node_embeddings_2d[:, 1],
    # c=node_colours,
    cmap="jet",
    alpha=alpha,
)

for i, txt in enumerate(node_ids):
    ax.annotate(txt, (node_embeddings_2d[i,0], node_embeddings_2d[i,1]))

#node2vec and aggregate for all students
#assign activityCodeList
# action = []

activityCodeList = graphLearning.generateActivityCodeList()

#export transition data matrix week
for w in range(0,12):
    transitionDataMatrixWeeks_directFollow[w].to_csv('transitionMatrixStorage/transitionDataMatrixWeeks_direct_follow_accumulated_w' + str(w) + '.csv', index=True)







import graphLearning
studentActivityEmbedding = []
for w in range(0,12):
    # print(f'Week {w}')
    # studentActivityEmbedding.append(graphLearning.naiveGraphEmbeddingAllStudentsInAWeek(
    #                                     transitionDataMatrixWeeks_directFollow[w], activityCodeList,w))
    studentActivityEmbedding.append(pd.read_csv('embeddingNode2Vec/GraphEmbeddings_accumulated_sum_time_w_' + str(w) + '.csv', index_col=0))

#visualise entire graph embeddings per week, test
week = 10
tsne = TSNE(n_components=2)
node_embeddings_2d = tsne.fit_transform(studentActivityEmbedding[week])
node_embeddings_2d_df = pd.DataFrame(node_embeddings_2d, index = studentActivityEmbedding[week].index)
node_embeddings_2d_df['classified'] = 0
node_embeddings_2d_df.loc[node_embeddings_2d_df.index.isin(ex2_excellent.index),['classified']] = 1
node_embeddings_2d_df.loc[node_embeddings_2d_df.index.isin(ex2_weak.index),['classified']] = 0

alpha = 0.7
# label_map = {l: i for i, l in enumerate(np.unique(node_targets))}

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(
    node_embeddings_2d_df.loc[:, 0],
    node_embeddings_2d_df.loc[:, 1],
    c=node_embeddings_2d_df['classified'],
    cmap="jet",
    alpha=alpha,
)

#visuallise ussing PCA
pca = PCA(n_components=10)
x = studentActivityEmbedding[week]   
x_adjust = x - np.mean(x) 
scaler = StandardScaler()   
scaler.fit(x)
x = scaler.transform(x)
pca.fit(x)
node_embeddings_2d_pca = pca.fit_transform(x)
node_embeddings_2d_pca_df = pd.DataFrame(node_embeddings_2d_pca, index = studentActivityEmbedding[week].index)
node_embeddings_2d_pca_df['classified'] = 0
node_embeddings_2d_pca_df.loc[node_embeddings_2d_df.index.isin(ex3_excellent.index),['classified']] = 1
node_embeddings_2d_pca_df.loc[node_embeddings_2d_df.index.isin(ex3_weak.index),['classified']] = 0

alpha = 0.7
# label_map = {l: i for i, l in enumerate(np.unique(node_targets))}

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(
    node_embeddings_2d_pca_df.loc[:, 2],
    node_embeddings_2d_pca_df.loc[:, 3],
    c=node_embeddings_2d_pca_df['classified'],
    cmap="jet",
    alpha=alpha,
)

pca.explained_variance_ratio_
    
# b = transitionDataMatrixWeeks_directFollow[0].loc['u-13fec06c93690caf5612445fac3691864386423d']
# a = weeksEventLog_filtered[2].loc[weeksEventLog_filtered[2]['org:resource'] == 'u-13fec06c93690caf5612445fac3691864386423d']
# a = full_transitionDataMatrixWeeks_directFollow[0].loc[full_transitionDataMatrixWeeks_directFollow[0]['user'] == '13fec06c93690caf5612445fac3691864386423d']

#prediction

# studentActivityEmbedding[1] = studentActivityEmbedding[1][studentActivityEmbedding[1].columns[0:64]]

#normalise data

#get normalise data
studentActivityEmbedding_normailise = dataProcessing.normaliseWeeklyData(studentActivityEmbedding)
studentActivityEmbedding_normailise[2]['3'].std()

#convert to pca data
studentEmbedding_pcaDataWeeks = []
pca_result = []
for w in range(0,12):
    tempData = studentActivityEmbedding[w]
    # tempData = tempData.merge(prediction_transition[w+1]['data']['successPassedRate'], left_on = tempData.index, right_on=prediction_transition[w+1]['data']['successPassedRate'].index).set_index('key_0')
    temp = FCAMiner.PCAcohortToValue(tempData)
    temp1 = temp[1]
    pcaResult = temp[0]
    # temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
    studentEmbedding_pcaDataWeeks.append(temp1)
    pca_result.append(pcaResult)
    
pca_result[10].explained_variance_ratio_

#cleaning data 
studentActivityEmbedding_cleaned = []
for w in range(0,12):
    studentActivityEmbedding_cleaned.append(libRMT.regressionToCleanEigenvectorEffect(studentActivityEmbedding[w],studentEmbedding_pcaDataWeeks[w],1))

#predictionu7
workingWeekExcercise = []
# prediction = {}
# prediction_cumm_practice = {}
# prediction_cumm_practice = {} #store transition matrix for prediction 
prediction_transition1 = {}
prediction_transition2 = {}
prediction_transition3 = {}
prediction_transition4 = {}

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
    # cummulativeResult = []
   
    pcaData1 = studentActivityEmbedding[week] #original data - scenario 1
    pcaData2 = studentActivityEmbedding_cleaned[week]
    pcaData3 = studentActivityEmbedding_normailise[week]
    pcaData4 = studentEmbedding_pcaDataWeeks[week]
    # print(pcaData.columns)

    mode = 'transition'
    
    tic = time.time()
    test1 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData1,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario1',week,toc-tic])
    
    tic = time.time()
    test2 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData2,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario2',week,toc-tic])
    
    tic = time.time()
    test3 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData3,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario3',week,toc-tic])
    
    tic = time.time()
    test4 = PredictionResult.predict_proba_all_algorithms_data_ready(pcaData4,excellent,weak,cummulativeResult,mode)
    toc = time.time()
    timePerformance.append(['scenario4',week,toc-tic])

    # test1 = PredictionResult.predict_proba_all_algorithms(Log,overall_excellent,overall_weak,cummulativeResult,lectureList,mode)
    
    # prediction_cumm_practice.update({ week : test })
    prediction_transition1.update({ week : test1 })
    prediction_transition2.update({ week : test1 })
    prediction_transition3.update({ week : test1 })
    prediction_transition4.update({ week : test1 })

    # overall_prediction_transition.update({week : test1})

prediction_transition = prediction_transition1    
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

title_transition = 'Graph embeddings - Node2Vec - accumulated data - Sum - exercise data - 5 activities data'
algorithmList = []
# algorithmList = []
PredictionResult.algorithmComparisonGraph('f1_score',predictionReport_transition,algorithmList, title_transition)
