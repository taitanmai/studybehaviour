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

#read data page activity
activityDataMatrixWeeks_pageTypeWeek_2018 = []
for w in range(0,12):
    activityDataMatrixWeeks_pageTypeWeek_2018.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv',index_col=0))

activityDataMatrixWeeks_pageTypeWeek_2019 = []
for w in range(0,12):
    activityDataMatrixWeeks_pageTypeWeek_2019.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1162019_activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv',index_col=0))

activityDataMatrixWeeks_pageTypeWeek = []
for w in range(0,12):
    activityDataMatrixWeeks_pageTypeWeek.append(pd.concat([activityDataMatrixWeeks_pageTypeWeek_2019[w], activityDataMatrixWeeks_pageTypeWeek_2018[w]],join='inner'))


ex3_excellent_2018 = pd.read_csv(basePath + 'ca1162018_ex3_excellent.csv',index_col=0)
ex3_weak_2018 = pd.read_csv(basePath + 'ca1162018_ex3_weak.csv',index_col=0)

ex3_excellent_2019 = pd.read_csv(basePath + 'ca1162019_ex3_excellent.csv',index_col=0)
ex3_weak_2019 = pd.read_csv(basePath + 'ca1162019_ex3_weak.csv',index_col=0)

ex3_excellent = pd.concat([ex3_excellent_2018,ex3_excellent_2019])
ex3_weak = pd.concat([ex3_weak_2018,ex3_weak_2019])

#IPR
pca_result = []
pcaDataWeeks = []
columnsReturn2 = []
for w in range(0,12):
    # tempData = transitionDataMatrixWeeks[w].loc[:,columns]
    tempData = activityDataMatrixWeeks_pageTypeWeek[w]
    # tempData = tempData.merge(prediction_transition[w+1]['data']['successPassedRate'], left_on = tempData.index, right_on=prediction_transition[w+1]['data']['successPassedRate'].index).set_index('key_0')
    temp = FCAMiner.PCAcohortToValue(tempData)
    temp1 = temp[1]
    pcaResult = temp[0]
    # temp1 = temp1.merge(prediction_transition[w+1]['data']['result_exam_1'], left_on = temp1.index, right_on=prediction_transition[w+1]['data']['result_exam_1'].index).set_index('key_0')
    pcaDataWeeks.append(temp1)
    pca_result.append(pcaResult)
    columnsReturn2.append(temp[2])

   
for w in range(0,12):
    pcaDataWeeks[w]['result_exam_1'] = 0
    if w in [0,1,2,3]:
        pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ex3_excellent.index),['result_exam_1']] = 1
    elif w in [4,5,6,7]:
        pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ex3_excellent.index),['result_exam_1']] = 1
    else:
        pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ex3_excellent.index),['result_exam_1']] = 1

for w in range(0,12):
    activityDataMatrixWeeks_pageTypeWeek[w]['result_exam_1'] = 0
    if w in [0,1,2,3]:
        activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w].index.isin(ex3_excellent.index),['result_exam_1']] = 1
    elif w in [4,5,6,7]:
        activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w].index.isin(ex3_excellent.index),['result_exam_1']] = 1
    else:
        activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w].index.isin(ex3_excellent.index),['result_exam_1']] = 1



fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0
num_bins = 50
for w in range(0,12):
    ax = fig.add_subplot(3,4,w+1)
    graph.append(ax)
    graph[countGraph].set_xlabel('Eigenvalues', fontsize = 15)
    graph[countGraph].set_ylabel('IPR', fontsize = 15)
    graph[countGraph].set_title('Inverse Participation Ratio week ' + str(w+1), fontsize = 20)
    graph[countGraph].grid()
    # graph[countGraph].axhline(y=0, color='k')
    # graph[countGraph].axvline(x=0, color='k')
    eigenValueList = pca_result[w].explained_variance_
    eigenVectorList = pca_result[w].components_
    IPRlist = libRMT.IPRarray(eigenValueList,eigenVectorList)
    graph[countGraph].axhline(y=IPRlist['IPR'].mean(), color='k', label='mean value of IPR') 
    graph[countGraph].plot(IPRlist['eigenvalue'], IPRlist['IPR'], '-', color ='blue', label='IPR')
    graph[countGraph].legend(loc='upper right')
    countGraph = countGraph + 1           
plt.show()

#draw one graph only:
fig = plt.figure(figsize=(5,5),dpi=120)
ax = fig.subplots()
ax.set_xlabel('Eigenvalues', fontsize = 10)
ax.set_ylabel('IPR', fontsize = 10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=10)
ax.set_title('', fontsize = 10)
ax.grid()
# graph[countGraph].axhline(y=0, color='k')
# graph[countGraph].axvline(x=0, color='k')
eigenValueList = pca_result[11].explained_variance_
eigenVectorList = pca_result[11].components_
IPRlist = libRMT.IPRarray(eigenValueList,eigenVectorList)
ax.axhline(y=IPRlist['IPR'].mean(), color='k', label='mean value of IPR') 
ax.axvline(x=IPRlist['eigenvalue'][0], color='k', ls='--') 
ax.axvline(x=IPRlist['eigenvalue'][1], color='k', ls='--') 
ax.axvline(x=IPRlist['eigenvalue'][2], color='k', ls='--') 
ax.text( IPRlist['eigenvalue'][0]*1, 0.015, 'λ1', color = 'red', ha = 'center', va = 'center',fontsize=10)
ax.text( IPRlist['eigenvalue'][1]*1,0.015, 'λ2', color = 'red', ha = 'center', va = 'center',fontsize=10)
ax.text( IPRlist['eigenvalue'][2]*1, 0.015, 'λ3', color = 'red', ha = 'center', va = 'center',fontsize=10)
ax.plot(IPRlist['eigenvalue'], IPRlist['IPR'], '-', color ='blue', label='IPR')
ax.legend(loc='upper right')
plt.show()

#outbounce select
a = libRMT.selectOutboundComponents(pcaDataWeeks[11],pca_result[11].explained_variance_)

#eigenvalues
fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0
num_bins = 100
for w in range(0,12):
    ax = fig.add_subplot(3,4,w+1)
    graph.append(ax)
    graph[countGraph].set_xlabel('eigenvalue λ', fontsize = 15)
    graph[countGraph].set_ylabel('P(λ)', fontsize = 15)
    graph[countGraph].set_title('Week ' + str(w+1), fontsize = 20)
    graph[countGraph].grid()
    # graph[countGraph].axhline(y=0, color='k')
    # graph[countGraph].axvline(x=0, color='k')
    eigenValueList_graph = pca_result[w].explained_variance_
    
    n, bins, patches = graph[countGraph].hist(eigenValueList_graph, num_bins, 
                           density = 1,  
                           color ='blue',  
                           alpha = 0.7, label='Sampled') 
    densityArray = libRMT.marcenkoPastur(len(pcaDataWeeks[w]),len(pcaDataWeeks[w].columns),bins)
    density = densityArray[0]
    graph[countGraph].plot(bins, density, '-', color ='black',label='RMT') 
    graph[countGraph].legend(loc='upper right')
    min_lambda = densityArray[1]
    max_lambda = densityArray[2]
    graph[countGraph].text( min_lambda*1.1, -0.08, 'λ-', color = 'black', ha = 'center', va = 'center',fontsize=15)
    graph[countGraph].text( max_lambda*1.1, -0.08, 'λ+', color = 'black', ha = 'center', va = 'center',fontsize=15)
    countGraph = countGraph + 1           
plt.show()

#week 12 only
w=11
num_bins = 100
fig = plt.figure(figsize=(5,5),dpi=120)
ax = fig.add_subplot(1,1,1)

ax.set_xlabel('eigenvalue λ', fontsize = 10)
ax.set_ylabel('P(λ)', fontsize = 10)
ax.set_title('', fontsize = 20)
ax.grid()
# graph[countGraph].axhline(y=0, color='k')
# graph[countGraph].axvline(x=0, color='k')
eigenValueList_graph = pca_result[w].explained_variance_

n, bins, patches = ax.hist(eigenValueList_graph, num_bins, 
                       density = 1,  
                       color ='blue',  
                       alpha = 0.7, label='Empirical eigenvalues distribution') 
densityArray = libRMT.marcenkoPastur(len(pcaDataWeeks[w]),len(pcaDataWeeks[w].columns),bins)
density = densityArray[0]
ax.plot(bins, density, '-', color ='black',label='Theoretical eingenvalue distribution') 
ax.legend(loc='upper right')
min_lambda = densityArray[1]
max_lambda = densityArray[2]
ax.text( min_lambda*1.1, -0.05, 'λ-', color = 'red', ha = 'center', va = 'center',fontsize=10)
ax.text( max_lambda*0.9, -0.05, 'λ+', color = 'red', ha = 'center', va = 'center',fontsize=10)     
ax.text( IPRlist['eigenvalue'][0], -0.05, 'λ1', color = 'red', ha = 'center', va = 'center',fontsize=10)   
ax.text( IPRlist['eigenvalue'][1], -0.05, 'λ2', color = 'red', ha = 'center', va = 'center',fontsize=10)   
ax.text( IPRlist['eigenvalue'][2], -0.05, 'λ3', color = 'red', ha = 'center', va = 'center',fontsize=10)      
plt.show()

#bibplot
def biplot(score, coeff , y, columns, col1, col2):
    '''
    Author: Serafeim Loukas, serafeim.loukas@epfl.ch
    Inputs:
       score: the projected data
       coeff: the eigenvectors (PCs)
       y: the class labels
   '''
    xs = score.loc[:,[col1]] # projection on PC1
    ys = score.loc[:,[col2]] # projection on PC2

    n = coeff.shape[0] # number of variables
    plt.figure(figsize=(10,8), dpi=100)
    classes = np.unique(y)
    colors = ['g','r','y']
    markers=['o','^','x']
    for s,l in enumerate(classes):
        if l == 0:
            label = 'Failed students'
        else:
            label = 'Pass students'
        plt.scatter(score.loc[score['result_exam_1'] == l,[col1]],
                    score.loc[score['result_exam_1'] == l,[col2]], 
                    c = colors[s], marker=markers[s], label=label) # color based on group

    plt.xlabel(col1, size=14)
    plt.ylabel(col2, size=14)
    limx= int(xs.max()) + 1
    limy= int(ys.max()) + 1
    plt.xlim([-limx,limx])
    plt.ylim([-limy,limy])
    plt.grid()
    plt.legend()
    plt.tick_params(axis='both', which='both', labelsize=14)
    
    # plt.figure(figsize=(10,8), dpi=100)
    for i in range(n):
        #plot as arrows the variable scores (each variable has a score for PC1 and one for PC2)
        # plt.scatter(coeff[i,0]*25, coeff[i,1]*25, color = 'blue', marker='x')
        plt.arrow(0, 0, coeff[i,0]*25, coeff[i,1]*25, color = 'blue', alpha = 0.9,linestyle = '-',linewidth = 0.2, overhang=0.05)
        plt.text(coeff[i,0]*25* 1.05, coeff[i,1] *25* 1.05, str(columns[i]), color = 'k', ha = 'center', va = 'center',fontsize=8)

    # plt.xlabel(col1, size=14)
    # plt.ylabel(col2, size=14)
    # limx= 0.5
    # limy= 0.5
    # plt.xlim([-limx,limx])
    # plt.ylim([-limy,limy])
    # plt.grid()
    # plt.tick_params(axis='both', which='both', labelsize=14)

w = 11   
biplot(pcaDataWeeks[w],
       np.transpose(pca_result[w].components_[[1,2], :]),
       pcaDataWeeks[w].loc[:,['result_exam_1']], activityDataMatrixWeeks_pageTypeWeek[w].columns, 'pc2','pc3')

#plot loadings
def plotLoadings(week,pca_result,transitionDataMatrixWeeks, columnsReturn1):
    loadings = pd.DataFrame(pca_result[week].components_[0:3, :], 
                            columns=columnsReturn1[week])
    maxPC = 1.01 * np.max(np.max(np.abs(loadings.loc[0:3, :])))
    f, axes = plt.subplots(1, 3, figsize=(20, 20), sharey=True)
    for i, ax in enumerate(axes):
        pc_loadings = loadings.loc[i, :]
        colors = ['C0' if l > 0 else 'C1' for l in pc_loadings]
        ax.axvline(color='#888888')
        ax.axvline(x=1/float(np.sqrt(37)), color='#888888')
        ax.axvline(x=-1/float(np.sqrt(37)), color='#888888')
        pc_loadings.plot.barh(ax=ax, color=colors)
        ax.set_xlabel(f'Principal Component {i+1}')
        ax.set_xlim(-maxPC, maxPC)
    plt.title('Week '+str(week+1))
    
plotLoadings(1,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  
plotLoadings(7,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  
plotLoadings(11,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  

pc = 3
from scipy import stats
for w in range(0,12):
    if 'pc'+str(pc) in pcaDataWeeks[w].columns:
        a1 = pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 1,['pc'+str(pc)]]
        b1 = pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 0,['pc' + str(pc)]]
        t1, p1 = stats.ttest_ind(a1,b1)
    
        
        print('Week ' + str(w) + ':')
        print('--PC' + str(pc) + ': ' + 't-value: ' + str(t1) + ' p-value: ' + str(p1))
        print('-- Excellent: ' + str(a1.mean()[0]))
        print('-- Weak: ' + str(b1.mean()[0]))
        
prcDataWeeksFirst3Components = pcaDataWeeks[11].loc[:,['pc1','pc2','pc3','result_exam_1']]


prcDataWeeksFirst3Components = prcDataWeeksFirst3Components.rename(columns={'pc1': 'Component 1', 'pc2': 'Component 2', 'pc3': 'Component 3'})
prcDataWeeksFirst3Components['Exam Result'] = np.where(prcDataWeeksFirst3Components['result_exam_1'] == 0, 'Fail', 'Pass')


boxplot = prcDataWeeksFirst3Components.boxplot(column=['Component 1', 'Component 2', 'Component 3'], by='Exam Result', figsize=(10,5), fontsize=20, layout=(1, 3))

#plot histogram pc1

w = 11
x = pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 0, ['pc1']]['pc1']
y = pcaDataWeeks[w].loc[pcaDataWeeks[w]['result_exam_1'] == 1, ['pc1']]['pc1']

bins = 100
plt.figure(figsize=(10,3), dpi=100)
plt.hist(x, bins, alpha=0.5, label='Failed students',density=True)
plt.hist(y, bins, alpha=0.5, label='Pass students',density=True)
plt.legend(loc='upper right')
plt.show()

activityDataMatrixWeeks_pageTypeWeek[11].columns.tolist().sort()
columns = ['General_0',
 'Labsheet_1', 
 'Labsheet_2',
 'Labsheet_3',
 'Labsheet_4',
 'Labsheet_5',
 'Labsheet_6',
 'Labsheet_7',
 'Labsheet_8',
 'Labsheet_9',
 'Labsheet_10',
 'Labsheet_11',
 'Labsheet_12',
 'Practice_1',
 'Practice_2',
 'Practice_3',
 'Practice_4',
 'Practice_5',
 'Practice_6',
 'Practice_7',
 'Practice_8',
 'Practice_9',
 'Practice_10',
 'Practice_11',
 'Practice_12',
 'Lecture_1',
 'Lecture_2',
 'Lecture_3',
 'Lecture_4',
 'Lecture_5',
 'Lecture_6',
 'Lecture_7',
 'Lecture_8',
 'Lecture_9',
 'Lecture_10',
 'Lecture_11',
 'Lecture_12',
 'result_exam_1'    
 ]

newActivityDataMatrixWeeks = activityDataMatrixWeeks_pageTypeWeek[11].loc[:,columns]

a = newActivityDataMatrixWeeks.groupby('result_exam_1').mean().T
a = newActivityDataMatrixWeeks.sum().T

# create plot
fig, ax = plt.subplots(figsize=(30,20), dpi=150)
index = np.arange(len(a.index))
bar_width = 0.4
opacity = 1

rects1 = plt.barh(index, a[0], bar_width, alpha=opacity, color='b')
# rects1 = plt.barh(index, a[0], bar_width, alpha=opacity, color='b', label='Fail students')
# rects2 = plt.barh(index + bar_width, a[1], bar_width, alpha=opacity, color='g', label='Pass students')

plt.ylabel('Course Materials', fontsize=20)
plt.xlabel('Total number of events', fontsize=20)
plt.title('')
plt.yticks(index + bar_width, a.index, fontsize=18)
plt.xticks(fontsize=20)
# plt.legend(fontsize=25)

# plt.tight_layout()
plt.show()

w = 11
for c in columns:
    if c != 'result_exam_1':
        a1 = activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w]['result_exam_1'] == 1,[c]]
        b1 = activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w]['result_exam_1'] == 0,[c]]
        t1, p1 = stats.ttest_ind(a1,b1)
    
        
        print(c + ': ' + 't-value: ' + str(t1) + ' p-value: ' + str(p1))
        print('-- Excellent: ' + str(a1.mean()[0]))
        print('-- Weak: ' + str(b1.mean()[0]))