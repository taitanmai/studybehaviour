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

import mlfinlab as ml
from mlfinlab.networks.mst import MST
from mlfinlab.networks.dash_graph import DashGraph
from networkx.algorithms import community
import itertools
from plotly.offline import plot
import plotly.graph_objects as go

basePath = 'D:\\Dataset\\PhD\\'



dataUpload = pd.read_csv(basePath + 'ca116_uploads.csv')
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

ex3_excellent.to_csv(basePath + 'ca1162018_ex3_excellent.csv',index=True)
ex3_weak.to_csv(basePath + 'ca1162018_ex3_weak.csv',index=True)

nonExUpload = dataUpload.drop(dataUpload.loc[dataUpload['task'].str.match('ex')].index)
nonExUploadByWeek = [g for n, g in nonExUpload.groupby(pd.Grouper(key='date',freq='W'))]

nonExUpload['version'].unique()
#merge exam result with transition data matrix:
reLabelIndex = dataProcessing.reLabelStudentId(assessment.index)

#practice results
workingWeekExcercise = []
cummulativeExerciseWeeks = []
for week in range(0,12):  
        
    workingWeekExcercise.append(nonExUploadByWeek[week])
    practiceResult = pd.concat(workingWeekExcercise) #nonExUploadByWeek[week] 

    #adjust number of correct: For each task, number of correct submission/number of submission for that task
    practiceResultSum = practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
    practiceResultSum['correct_adjusted'] = practiceResultSum['correct']/practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).count()['correct']
    cummulativeResult = practiceResultSum.reset_index().groupby([pd.Grouper(key='user')]).sum()
    
    # cummulativeResult = practiceResultSum.groupby([pd.Grouper(key='user')]).sum()
    cummulativeResult['cumm_practice'] = cummulativeResult['correct']/practiceResult.groupby([pd.Grouper(key='user')]).count()['date']
    cummulativeResult['successPassedRate'] = cummulativeResult['passed']/(cummulativeResult['passed'] + cummulativeResult['failed'])
    # cummulativeResult = graphLearning.mapNewLabel(cummulativeResult, reLabelIndex)
    cummulativeExerciseWeeks.append(cummulativeResult)
    
for w in range(0,12):
    cummulativeExerciseWeeks[w] = graphLearning.mapNewLabel(cummulativeExerciseWeeks[w],reLabelIndex)
    
assessment_label = assessment.copy()
assessment_label = graphLearning.mapNewLabel(assessment, reLabelIndex)

assessment_label1A = assessment1A.copy()
assessment_label1A = graphLearning.mapNewLabel(assessment1A, reLabelIndex)
assessment_label2A = assessment2A.copy()
assessment_label2A = graphLearning.mapNewLabel(assessment2A, reLabelIndex)
assessment_label3A = assessment3A.copy()
assessment_label3A = graphLearning.mapNewLabel(assessment3A, reLabelIndex)

#read csv iff neeeded
transitionDataMatrixWeeks = []
for w in range(0,12):
    temp = pd.read_csv(basePath + 'transitionMatrixStorage_new/transitionDataMatrixWeeks_direct_accumulated_pageTypeWeek_manyPractice_w' + str(w) + '.csv', index_col=0)
    if w in [0,1,2,3]:
        studentList = assessment1A.index.astype(str)
    elif w in [4,5,6,7]:
        studentList = assessment2A.index.astype(str)     
    else:
        studentList = assessment3A.index.astype(str)
    temp = temp.loc[temp.index.isin(studentList)]
    temp = graphLearning.mapNewLabel(temp, reLabelIndex)
    # temp = temp.drop(['Practice_0-Practice_0'],axis=1)
    # if w == 1:
    #     temp = temp.drop([8])
    transitionDataMatrixWeeks.append(temp) 
    
transitionDataMatrixWeeks_directFollow_standardised = []    
for w in range(0,12):
    transitionDataMatrixWeeks_directFollow_standardised.append(dataProcessing.normaliseData(transitionDataMatrixWeeks[w].T))

# correlation processing    
corrList = []
corrDistanceList = []
for w in range(0,12):
    corrTemp = transitionDataMatrixWeeks_directFollow_standardised[w].corr()
    corrList.append(corrTemp)
    corrDistance = (0.5*(1 - corrTemp)).apply(np.sqrt)
    corrDistanceList.append(corrDistance)

graph_all_weeks = []
for w in range(0,12):
    print('Week ' + str(w) + '...')
    matrix = corrList[w]
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    tn_relation = transitionDataMatrixWeeks_directFollow_standardised[w].shape[0] / transitionDataMatrixWeeks_directFollow_standardised[w].shape[1]
    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01
    # Finding the Вe-noised Сovariance matrix
    # denoised_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth)
    # denoised_matrix_byLib = pd.DataFrame(denoised_matrix_byLib, index=matrix.index, columns=matrix.columns) denoise_method='target_shrink',
    
    detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth=kde_bwidth, detone=True)
    # detoned_matrix_byLib = matrix #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)
    distance_matrix = (0.5*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    graphBuild = MST(distance_matrix, 'distance')
    # graphBuild = nx.from_pandas_adjacency(distance_matrix)   
    graph_all_weeks.append(graphBuild)

import community as community_louvain  
import matplotlib.cm as cm  
partition = community_louvain.best_partition(graph_all_weeks[11].graph)

G = graph_all_weeks[11].graph
pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, 
                      cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()

dendogram = community_louvain.generate_dendrogram( G)
community_louvain.modularity(partition, G)

for w in range(0,12):
    G = graph_all_weeks[w].graph
    partition = community_louvain.best_partition(G)
    print('Week ' + str(w + 1) + ': ' + str(community_louvain.modularity(partition, G)), ' -- No of coms: ' + str(max(partition, key = partition.get)))
    
partition.items[0]
