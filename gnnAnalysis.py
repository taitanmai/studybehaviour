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

# eventLog_ca116_filtered.rename(columns={'concept:instance':'concept:instance1',
#                                    'concept:name':'concept:name1',
#                                    'case:concept:name' : 'case:concept:name1'}, 
#                   inplace=True)
eventLog_ca116_filtered['concept:instance'] = eventLog_ca116_filtered['pageType']
eventLog_ca116_filtered['concept:name'] = eventLog_ca116_filtered['pageType']
eventLog_ca116_filtered['date'] = eventLog_ca116_filtered['time:timestamp'].dt.date

eventLog_ca116_filtered['case:concept:name'] = eventLog_ca116_filtered['date'].astype(str) + '-' + eventLog_ca116_filtered['org:resource'].astype(str)

# eventLog_ca116_filtered['concept:name'] = eventLog_ca116_filtered['pageType'] + '*' + eventLog_ca116_filtered['concept:name1']
# eventLog_ca116_filtered['concept:instance'] = eventLog_ca116_filtered['pageType'] + '*' + eventLog_ca116_filtered['concept:instance1']


# eventLog_ca116_filtered.to_csv("eventLog_ca116_filtered_2018.csv", index=False)
weeksEventLog_filtered = [g for n, g in eventLog_ca116_filtered.groupby(pd.Grouper(key='time:timestamp',freq='W'))]

#get mark
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
                          # 'perCorrect':'perCorrect1A',
                          'failed':'failed1A',
                            'passed':'passed1A',
                            'perPassed':'perPassed1A',
                            'testSubmitted':'testSubmitted1A',
                            'adjustedPerformance':'adjustedPerformance1A'}, 
                  inplace=True)
assessment2A.rename(columns={'correct':'correct2A',
                          # 'perCorrect':'perCorrect2A',
                          'failed':'failed2A',
                            'passed':'passed2A',
                            'perPassed':'perPassed2A',
                            'testSubmitted':'testSubmitted2A',
                            'adjustedPerformance':'adjustedPerformance2A'}, 
                  inplace=True)
assessment3A.rename(columns={'correct':'correct3A',
                          # 'perCorrect':'perCorrect3A',
                            'failed':'failed3A',
                            'passed':'passed3A',
                            'perPassed':'perPassed3A',
                            'testSubmitted':'testSubmitted3A',
                            'adjustedPerformance':'adjustedPerformance3A'}, 
                  inplace=True)
assessment1A = assessment1A.set_index(['user'])
assessment2A = assessment2A.set_index(['user'])
assessment3A = assessment3A.set_index(['user'])

assessment1A['result'] = 0
assessment1A.loc[assessment1A['perCorrect'] >= 0.4,['result']] = 1
assessment2A['result'] = 0
assessment2A.loc[assessment2A['perCorrect'] >= 0.4,['result']] = 1
assessment3A['result'] = 0
assessment3A.loc[assessment3A['perCorrect'] >= 0.4,['result']] = 1

assessment = pd.concat([assessment1A,assessment2A,assessment3A], axis=1)
assessment = assessment.fillna(0)

assessment['grade'] = (assessment['perCorrect1A']+assessment['perCorrect2A']+assessment['perCorrect3A'])/3
assessment['perPassed'] = (assessment['passed1A'] + assessment['passed2A'] + assessment['passed3A'])/(assessment['passed1A'] + assessment['passed2A'] + assessment['passed3A'] 
                        + assessment['failed1A']+ assessment['failed2A']+ assessment['failed3A'])


#convert data for PCA - from eventlog to transition data matrix
workingWeekLog = []
transitionDataMatrixWeeks_directFollow = []
full_transitionDataMatrixWeeks_directFollow = []
for week in range(1,13):
    print('Week: ' + str(week) + '...')
    workingWeekLog.append(weeksEventLog_filtered[week])
    Log =  pd.concat(workingWeekLog) # weeksEventLog_filtered[week] #
    tempTransition = FCAMiner.transitionDataMatrixConstruct_directFollow(Log, [], True, 'time').fillna(0)
    full_transitionDataMatrixWeeks_directFollow.append(tempTransition)   
    tempTransition = tempTransition.groupby([pd.Grouper(key='user')]).sum()         
    transitionDataMatrixWeeks_directFollow.append(tempTransition)
    
for w in range(0,12):
    transitionDataMatrixWeeks_directFollow[w].to_csv('transitionMatrixStorage/transitionDataMatrixWeeks_direct_follow_accumulated_time_w' + str(w) + '.csv')

transitionDataMatrixWeeks_directFollow = []
for w in range(0,12):
    # print(f'Week {w}')
    # studentActivityEmbedding.append(graphLearning.naiveGraphEmbeddingAllStudentsInAWeek(
    #                                     transitionDataMatrixWeeks_directFollow[w], activityCodeList,w))
    transitionDataMatrixWeeks_directFollow.append(pd.read_csv('transitionMatrixStorage/transitionDataMatrixWeeks_direct_follow_accumulated_time_w' + str(w) + '.csv', index_col=0))
 

#test get all students with full graph features data
studentAdata = []
studentEdata = []
studentXdata = []
studentYdata = []
for w in range(0,12):
    if w in [0,1,2,3]:
        get_assessment = assessment1A
    elif w in [4,5,6,7]:
        get_assessment = assessment2A
    else:
        get_assessment = assessment3A
    activityList = weeksEventLog_filtered[w+1].loc[:,['concept:name']]['concept:name'].unique()
    activityCodeList = graphLearning.assignNodeNumber(activityList)
    A,X,E,y = graphLearning.constructGraphFeatureForAll(transitionDataMatrixWeeks_directFollow[w],activityCodeList,w,get_assessment)
    studentAdata.append(A)
    studentEdata.append(E)
    studentXdata.append(X)
    studentYdata.append(y)
    
    
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from spektral.layers import EdgeConditionedConv, GlobalSumPool, GraphAttention,GlobalMaxPool,GlobalAttentionPool,GraphConv
from spektral.utils import label_to_one_hot
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from spektral.utils.convolution import localpooling_filter
################################################################################
# PARAMETERS
################################################################################
learning_rate = 1e-8  # Learning rate
epochs = 200        # Number of training epochs
batch_size = 64  

w = 6
A = studentAdata[w]
E = studentEdata[w]  
X = studentXdata[w]  
y = studentYdata[w]      


X_uniq = np.unique(X)
X_uniq = X_uniq[X_uniq != 0]
E_uniq = np.unique(E)
E_uniq = E_uniq[E_uniq != 0]

X = label_to_one_hot(X, X_uniq)
E = label_to_one_hot(E, E_uniq)

# Parameters
N = X.shape[-2]       # Number of nodes in the graphs
F = X[0].shape[-1]    # Dimension of node features
S = E[0].shape[-1]    # Dimension of edge features
n_out = y.shape[-1]   # Dimension of the target

# Train/test split
A_train, A_test, \
X_train, X_test, \
E_train, E_test, \
y_train, y_test = train_test_split(A, X, E, y, test_size=0.2, random_state=5)

# np.where(E >= np.finfo(np.float64).max)

################################################################################
# BUILD MODEL
################################################################################
X_in = Input(shape=(N, F))
A_in = Input(shape=(N, N))
E_in = Input(shape=(N, N, S))

X_1 = EdgeConditionedConv(32, activation='relu')([X_in, A_in, E_in])
X_2 = EdgeConditionedConv(32, activation='relu')([X_1, A_in, E_in])
X_3 = GlobalSumPool()(X_1)

output = Dense(n_out)(X_3)

# Build model
model = Model(inputs=[X_in, A_in, E_in], outputs=output)
optimizer = Adam(lr=learning_rate)

from sklearn.metrics import roc_auc_score
def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

model.compile(optimizer=optimizer, loss='mse',  metrics=['accuracy', auroc])
model.summary()

################################################################################
# FIT MODEL
################################################################################
model.fit([X_train, A_train, E_train],
          y_train,
          batch_size=batch_size,
          epochs=epochs)

################################################################################
# EVALUATE MODEL
################################################################################


print('Testing model')
model_loss = model.evaluate([X_test, A_test, E_test],
                            y_test,
                            batch_size=batch_size)
print('Done. Test loss: {}'.format(model_loss))

a = []
for layers in model.layers:
    a.append(layers.get_weights())

######### GRAPH ATTENTION
X.shape[-1]
N = X.shape[-2]          # Number of nodes in the graphs
F = X.shape[-1]          # Original feature dimensionality
n_classes = y.shape[-1]  # Number of classes
l2_reg = 5e-4            # Regularization rate for l2
learning_rate = 1e-3     # Learning rate for Adam
epochs = 200           # Number of training epochs
batch_size = 32          # Batch size
es_patience = 200        # Patience fot early stopping

# Train/test split
A_train, A_test, \
x_train, x_test, \
y_train, y_test = train_test_split(A, X, y, test_size=0.2, random_state=5)

# Model definition
X_in = Input(shape=(N, F))
A_in = Input((N, N))

gc1 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in, A_in])
gc2 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([gc1, A_in])
pool = GlobalAttentionPool(128)(gc2)

output = Dense(n_classes, activation='softmax')(pool)

# Build model
model = Model(inputs=[X_in, A_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc', auroc])
model.summary()

# Train model
model.fit([x_train, A_train],
          y_train,
          batch_size=batch_size,
          validation_split=0.1,
          epochs=epochs,
          callbacks=[
              EarlyStopping(patience=es_patience, restore_best_weights=True)
          ])

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([x_test, A_test],
                              y_test,
                              batch_size=batch_size)
print('Done. Test loss: {:.4f}. Test acc: {:.2f}'.format(*eval_results))

a = []
for layers in model.layers:
    a.append(layers.get_weights())

#### GCN
# Parameters
K = 2                   # Degree of propagation
N = X.shape[0]          # Number of nodes in the graph
F = X.shape[1]          # Original size of node features
n_classes = y.shape[1]  # Number of classes
l2_reg = 5e-6           # L2 regularization rate
learning_rate = 0.2     # Learning rate
epochs = 20000          # Number of training epochs
es_patience = 200       # Patience for early stopping

# Preprocessing operations
fltr = localpooling_filter(A).astype('f4')
X = X.toarray()

# Pre-compute propagation
for i in range(K - 1):
    fltr = fltr.dot(fltr)
fltr.sort_indices()

# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)
output = GraphConv(n_classes,
                   activation='softmax',
                   kernel_regularizer=l2(l2_reg),
                   use_bias=False)([X_in, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

# Train model
validation_data = ([X, fltr], y, val_mask)
model.fit([X, fltr],
          y,
          sample_weight=train_mask,
          epochs=epochs,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False,  # Shuffling data means shuffling the whole graph
          callbacks=[
              EarlyStopping(patience=es_patience,  restore_best_weights=True)
          ])

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([X, fltr],
                              y,
                              sample_weight=test_mask,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))