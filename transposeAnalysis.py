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

#import data
transitionDataMatrixWeeks_directFollow = []
for w in range(0,12):
    # print(f'Week {w}')
    # studentActivityEmbedding.append(graphLearning.naiveGraphEmbeddingAllStudentsInAWeek(
    #                                     transitionDataMatrixWeeks_directFollow[w], activityCodeList,w))
    transitionDataMatrixWeeks_directFollow.append(pd.read_csv('transitionMatrixStorage/transitionDataMatrixWeeks_direct_follow_accumulated_w' + str(w) + '.csv', index_col=0))

a = transitionDataMatrixWeeks_directFollow[11].loc[~transitionDataMatrixWeeks_directFollow[11].index.isin(assessment3A.index)]
a = a.T
b = assessment1A.loc[assessment2A.index.isin(a.index)]


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

#merge exam result with transition data matrix:

for w in range(0,12):
    if w in [0,1,2,3]:
        excellentList = ex1_excellent.index
        weakList = ex1_weak.index
    elif w in [4,5,6,7]:
        excellentList = ex2_excellent.index
        weakList = ex2_weak.index        
    else:
        excellentList = ex3_excellent.index
        weakList = ex3_weak.index

    excellentStudents = transitionDataMatrixWeeks_directFollow[w].loc[transitionDataMatrixWeeks_directFollow[w].index.isin(excellentList)]
    weakStudents = transitionDataMatrixWeeks_directFollow[w].loc[transitionDataMatrixWeeks_directFollow[w].index.isin(weakList)]
    excellentStudents['result_exam_1'] = 0
    weakStudents['result_exam_1'] = 1
    transitionDataMatrixWeeks_directFollow[w] = pd.concat([excellentStudents,weakStudents])


#transpose transition data matrix
for w in range(0,12):
    transitionDataMatrixWeeks_directFollow[w] = transitionDataMatrixWeeks_directFollow[w].T

#correlation processing    
# corrList = []
# corrDistanceList = []
# for w in range(0,12):
#     corrTemp = transitionDataMatrixWeeks_directFollow[w].corr()
#     corrList.append(corrTemp)
#     corrDistance = (0.5*(1 - corrTemp)).apply(np.sqrt)
#     corrDistanceList.append(corrDistance)

# cmap = sns.cm.rocket_r

sns.heatmap(corrList[3], annot=False, center=0.8, yticklabels=False, xticklabels=False, cmap='coolwarm')
plt.title('Correlation heatmap week ' + '3')
plt.show()

# cmap = sns.cm.rocket_r

sns.heatmap(corrDistanceList[3], annot=False, center=0, yticklabels=False, xticklabels=False, cmap='coolwarm')
plt.title('Distance correlation heatmap week ' + '3')
plt.show()


#normalised data--------------------------------------------------------------
transitionDataMatrixWeeks_directFollow_normalised = []
for w in range(0,12):
    temp = transitionDataMatrixWeeks_directFollow[w][:-1]
    transitionDataMatrixWeeks_directFollow_normalised.append(dataProcessing.normaliseData(temp))

#correlation processing    
corrList_dataNormalised = []
corrDistanceList_dataNormalised = []
for w in range(0,12):
    corrTemp = transitionDataMatrixWeeks_directFollow_normalised[w].corr()
    corrList_dataNormalised.append(corrTemp)
    corrDistance = (0.5*(1 - corrTemp)).apply(np.sqrt)
    corrDistanceList_dataNormalised.append(corrDistance)


w = 11   
matrix = corrList_dataNormalised[w]

# denoised_matrix = libRMT.denoisedCorr(matrix, transitionDataMatrixWeeks_directFollow_normalised[w].shape[0], transitionDataMatrixWeeks_directFollow_normalised[w].shape[1])

# cmap = sns.cm.rocket_r

# sns.heatmap(denoised_matrix, annot=False, center=0, yticklabels=False, xticklabels=False, cmap='coolwarm')
# plt.title('Correlation heatmap week ' + '3')
# plt.show()

# cmap = sns.cm.rocket_r
# denoised_corrDistance = (0.5*(1 - denoised_matrix)).apply(np.sqrt)
# sns.heatmap(denoised_corrDistance, annot=False, center=0, yticklabels=False, xticklabels=False, cmap='coolwarm')
# plt.title('Distance correlation heatmap week ' + '3')
# plt.show()

#denoised by library
risk_estimators = ml.portfolio_optimization.RiskEstimators()
# Setting the required parameters for de-noising
# Relation of number of observations T to the number of variables N (T/N)
tn_relation = transitionDataMatrixWeeks_directFollow_normalised[w].shape[0] / transitionDataMatrixWeeks_directFollow_normalised[w].shape[1]
# The bandwidth of the KDE kernel
kde_bwidth = 0.25
# Finding the Вe-noised Сovariance matrix
denoised_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth)
denoised_matrix_byLib = pd.DataFrame(denoised_matrix_byLib, index=matrix.index, columns=matrix.columns)

detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth, detone=True)
detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)

# sns.heatmap(detoned_matrix_byLib, annot=False, center=0, yticklabels=False, xticklabels=False)
# plt.title('Correlation heatmap week ' + str(w))
# plt.show()

from mlfinlab.networks.mst import MST
custom_matrix = ml.codependence.get_distance_matrix(detoned_matrix_byLib, distance_metric='angular')
reLabelIndex = custom_matrix.reset_index().loc[:,['user']]
custom_matrix = custom_matrix.reset_index(drop=True)
custom_matrix.columns = np.arange(len(custom_matrix.columns))

# sns.heatmap(custom_matrix, annot=False, center=0, yticklabels=False, xticklabels=False)
# plt.title('Distance Correlation heatmap week ' + str(w))
# plt.show()

graph = MST(custom_matrix, 'custom')
studentCohorts = {"excellent": reLabelIndex.loc[reLabelIndex['user'].isin(ex3_excellent.index)].index, "weak": reLabelIndex.loc[reLabelIndex['user'].isin(ex3_weak.index)].index}
graph.set_node_groups(studentCohorts)
graph.get_graph_plot()

from mlfinlab.networks.dash_graph import DashGraph
dash_graph = DashGraph(graph)
server = dash_graph.get_server()

# Run server
server.run_server()

from networkx.algorithms import community
import itertools
communities_generator = community.girvan_newman(graph.graph)

for communities in itertools.islice(communities_generator, 1):
    print(tuple(sorted(c) for c in communities))  

#embedding graph
node2vec = Node2Vec(graph.graph, dimensions=32, walk_length=5, num_walks=10)
model = node2vec.fit(window=10, min_count=1)

node_embeddings = (
    model.wv.vectors
)  # numpy.ndarray of size number of nodes times embeddings dimensionality    

node_embeddings = pd.DataFrame(node_embeddings, index = custom_matrix.index)

tsne = TSNE(n_components=2)
node_embeddings_2d = tsne.fit_transform(node_embeddings)
node_embeddings_2d_df = pd.DataFrame(node_embeddings_2d, index = node_embeddings.index)
node_embeddings_2d_df['classified'] = 55
node_embeddings_2d_df.loc[node_embeddings_2d_df.index.isin(ex3_excellent.index),['classified']] = 1
node_embeddings_2d_df.loc[node_embeddings_2d_df.index.isin(ex3_weak.index),['classified']] = 0

alpha = 0.7
# label_map = {l: i for i, l in enumerate(np.unique(node_targets))}

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(
    node_embeddings_2d_df.loc[node_embeddings_2d_df['classified'] == 0, 0],
    node_embeddings_2d_df.loc[node_embeddings_2d_df['classified'] == 0, 1],
    label = 'Weak',
    alpha=alpha,
)

ax.scatter(
    node_embeddings_2d_df.loc[node_embeddings_2d_df['classified'] == 1, 0],
    node_embeddings_2d_df.loc[node_embeddings_2d_df['classified'] == 1, 1],
    label = 'Excellent',
    alpha=alpha,
)

node_embeddings_2d[0][1]

n = [x for x in range(0, len(node_embeddings_2d))]

for i, txt in enumerate(n):
    ax.annotate(txt, (node_embeddings_2d[i][0],node_embeddings_2d[i][1]))

ax.legend()
plt.title('Week ' + str(w))
plt.show()

workingWeekExcercise = []
for week in range(0,w):
    workingWeekExcercise.append(nonExUploadByWeek[week])

practiceResult = pd.concat(workingWeekExcercise)

#adjust number of correct: For each task, number of correct submission/number of submission for that task
practiceResultSum = practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).sum()
practiceResultSum['correct_adjusted'] = practiceResultSum['correct']/practiceResult.groupby([pd.Grouper(key='user'),pd.Grouper(key='task')]).count()['correct']
cummulativeResult = practiceResultSum.reset_index().groupby([pd.Grouper(key='user')]).sum()

# cummulativeResult = practiceResultSum.groupby([pd.Grouper(key='user')]).sum()
cummulativeResult['cumm_practice'] = cummulativeResult['correct']/practiceResult.groupby([pd.Grouper(key='user')]).count()['date']
cummulativeResult['successPassedRate'] = cummulativeResult['passed']/(cummulativeResult['passed'] + cummulativeResult['failed'])

predictionResult = PredictionResult.predict_proba_all_algorithms_data_ready(node_embeddings_2d_df.drop(['classified'], axis=1), ex1_excellent.index, ex1_weak.index, cummulativeResult)


#test model for fun


