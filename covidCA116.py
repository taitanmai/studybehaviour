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

basePath = 'E:\\Data\\extractedData\\'

ca1162018_transitionDataMatrixWeeks = []
ca1162019_transitionDataMatrixWeeks = []
ca1162020_transitionDataMatrixWeeks = []

for w in range(0,12):
    ca1162018_transitionDataMatrixWeeks.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/transitionDataMatrixWeeks_direct_accumulated_pageTypeWeek_manyPractice_w'+str(w)+'.csv'))
#    ca1162018_transitionDataMatrixWeeks[w].user = ca1162018_transitionDataMatrixWeeks[w].user + '-2018'
    ca1162018_transitionDataMatrixWeeks[w] = ca1162018_transitionDataMatrixWeeks[w].set_index('user')
      
    ca1162019_transitionDataMatrixWeeks.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1162019_transitionDataMatrixWeeks_direct_accumulated_pageTypeWeek_manyPractice_w'+str(w)+'.csv'))
#    ca1162019_transitionDataMatrixWeeks[w].user = ca1162019_transitionDataMatrixWeeks[w].user + '-2019'
    ca1162019_transitionDataMatrixWeeks[w] = ca1162019_transitionDataMatrixWeeks[w].set_index('user')
    
for w in range(0,10):
    ca1162020_transitionDataMatrixWeeks.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1162020_transitionDataMatrixWeeks_direct_accumulated_pageTypeWeek_manyPractice_w'+str(w)+'.csv'))
#    ca1162020_transitionDataMatrixWeeks[w].user = ca1162020_transitionDataMatrixWeeks[w].user + '-2020'   
    ca1162020_transitionDataMatrixWeeks[w] = ca1162020_transitionDataMatrixWeeks[w].set_index('user')
    
   
transitionDataMatrixWeeks = []
# listIndex = []
for w in range(0,10):
    # ca1162018_transitionDataMatrixWeeks[w]['covid'] = 0
    # ca1162019_transitionDataMatrixWeeks[w]['covid'] = 0
    # ca1162020_transitionDataMatrixWeeks[w]['covid'] = 1
    transitionDataMatrixWeeks.append(pd.concat([ca1162018_transitionDataMatrixWeeks[w],ca1162019_transitionDataMatrixWeeks[w],ca1162020_transitionDataMatrixWeeks[w]], join='inner'))
    # listIndex = listIndex + list(transitionDataMatrixWeeks[w].index)

transitionDataMatrixWeeks_directFollow_standardised = []    
for w in range(0,10):
    transitionDataMatrixWeeks_directFollow_standardised.append(dataProcessing.normaliseData(transitionDataMatrixWeeks[w].T))
  

# listIndex = list(set(listIndex))
# reLabelIndex = dataProcessing.reLabelStudentId(listIndex)
# for w in range(0,10):
#     ca1162018_transitionDataMatrixWeeks[w] = graphLearning.mapNewLabel(ca1162018_transitionDataMatrixWeeks[w], reLabelIndex)
#     ca1162019_transitionDataMatrixWeeks[w] = graphLearning.mapNewLabel(ca1162019_transitionDataMatrixWeeks[w], reLabelIndex)
#     ca1162020_transitionDataMatrixWeeks[w] = graphLearning.mapNewLabel(ca1162020_transitionDataMatrixWeeks[w], reLabelIndex)
#     transitionDataMatrixWeeks[w] = graphLearning.mapNewLabel(transitionDataMatrixWeeks[w], reLabelIndex)

ca1162018_transitionDataMatrixWeeks_directFollow_standardised = []    
for w in range(0,12):
    ca1162018_transitionDataMatrixWeeks_directFollow_standardised.append(dataProcessing.normaliseData(ca1162018_transitionDataMatrixWeeks[w].T))
    
ca1162019_transitionDataMatrixWeeks_directFollow_standardised = []    
for w in range(0,12):
    ca1162019_transitionDataMatrixWeeks_directFollow_standardised.append(dataProcessing.normaliseData(ca1162019_transitionDataMatrixWeeks[w].T))
    
ca1162020_transitionDataMatrixWeeks_directFollow_standardised = []    
for w in range(0,10):
    ca1162020_transitionDataMatrixWeeks_directFollow_standardised.append(dataProcessing.normaliseData(ca1162020_transitionDataMatrixWeeks[w].T))
    
ca1162018_activityDataMatrixWeeks_pageTypeWeek = []
ca1162019_activityDataMatrixWeeks_pageTypeWeek = []
ca1162020_activityDataMatrixWeeks_pageTypeWeek = []

for w in range(0,12):
    ca1162018_activityDataMatrixWeeks_pageTypeWeek.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv', index_col=0))
#    ca1162018_activityDataMatrixWeeks_pageTypeWeek[w].index = ca1162018_activityDataMatrixWeeks_pageTypeWeek[w].index + '-2018'
    
    ca1162019_activityDataMatrixWeeks_pageTypeWeek.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1162019_activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv', index_col=0))
#    ca1162019_activityDataMatrixWeeks_pageTypeWeek[w].index = ca1162019_activityDataMatrixWeeks_pageTypeWeek[w].index + '-2019'

for w in range(0,10):
    ca1162020_activityDataMatrixWeeks_pageTypeWeek.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1162020_activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv', index_col=0))
#    ca1162020_activityDataMatrixWeeks_pageTypeWeek[w].index = ca1162020_activityDataMatrixWeeks_pageTypeWeek[w].index + '-2020'
    

# activityDataMatrixWeeks_pageTypeWeek = []
# for w in range(0,10):
#     activityDataMatrixWeeks_pageTypeWeek.append(pd.concat([ca1162018_activityDataMatrixWeeks_pageTypeWeek[w], ca1162019_activityDataMatrixWeeks_pageTypeWeek[w], ca1162020_activityDataMatrixWeeks_pageTypeWeek[w]],join='inner'))

# activityDataMatrixWeeks_pageTypeWeek = []
# for w in range(0,10):
#     activityDataMatrixWeeks_pageTypeWeek.append(ca1162020_activityDataMatrixWeeks_pageTypeWeek[w])


#score result data
ex3_excellent_2018 = pd.read_csv(basePath + 'ca1162018_ex3_excellent.csv',index_col=0)
#ex3_excellent_2018.index = ex3_excellent_2018.index + '-2018'
ex3_weak_2018 = pd.read_csv(basePath + 'ca1162018_ex3_weak.csv',index_col=0)
#ex3_weak_2018.index = ex3_weak_2018.index + '-2018'

ex3_excellent_2019 = pd.read_csv(basePath + 'ca1162019_ex3_excellent.csv',index_col=0)
#ex3_excellent_2019.index = ex3_excellent_2019.index + '-2019'
ex3_weak_2019 = pd.read_csv(basePath + 'ca1162019_ex3_weak.csv',index_col=0)
#ex3_weak_2019.index = ex3_weak_2019.index + '-2019'

ex3_excellent_2020 = pd.read_csv(basePath + 'ca1162020_ex3_excellent.csv',index_col=0)
#ex3_excellent_2020.index = ex3_excellent_2020.index + '-2020'
ex3_weak_2020 = pd.read_csv(basePath + 'ca1162020_ex3_weak.csv',index_col=0)
#ex3_weak_2020.index = ex3_weak_2020.index + '-2020'

#ex3_excellent = pd.concat([ex3_excellent_2018,ex3_excellent_2019, ex3_excellent_2020])
#ex3_weak = pd.concat([ex3_weak_2018,ex3_weak_2019, ex3_weak_2020])
#assessment3A = pd.concat([ex3_excellent, ex3_weak])
#assessment3A = graphLearning.mapNewLabel(assessment3A, reLabelIndex)

#ex3_excellent_2018 = graphLearning.mapNewLabel(ex3_excellent_2018, reLabelIndex)
#ex3_excellent_2019 = graphLearning.mapNewLabel(ex3_excellent_2019, reLabelIndex)
#ex3_excellent_2020 = graphLearning.mapNewLabel(ex3_excellent_2020, reLabelIndex)

# #shuffle the dataframe to avoid same year students stay together:
# import random
# for w in range(0,10):
#     columns = list(transitionDataMatrixWeeks_directFollow_standardised[w].columns)
#     random.shuffle(columns)    
#     transitionDataMatrixWeeks_directFollow_standardised[w] = transitionDataMatrixWeeks_directFollow_standardised[w].loc[:,columns]
    
# #change student position
# columnsList = []
# for i in range(len(communityWeek10[2])):
#     for j in range(len(communityWeek10[2][i])):
#         if communityWeek10[2][i][j] in student2018_excellent.union(student2018_weak):
#             k = 2018
#         elif communityWeek10[2][i][j] in student2019_excellent.union(student2019_weak):
#             k = 2019
#         else:
#             k = 2020
#         columnsList.append([communityWeek10[2][i][j],i,k])
# columnsListDf = pd.DataFrame(columnsList)        
# columnsListDf = columnsListDf.sort_values(by = [1,2])

# columns = list(columnsListDf.loc[:,[0]][0])
# transitionDataMatrixWeeks_directFollow_standardised[9] = transitionDataMatrixWeeks_directFollow_standardised[9].loc[:,columns]
        

    
  
# correlation processing    
transitionDataMatrixWeeks_directFollow_standardised = ca1162020_transitionDataMatrixWeeks_directFollow_standardised


corrList = []
corrDistanceList = []
for w in range(0,10):
    corrTemp = transitionDataMatrixWeeks_directFollow_standardised[w].corr()
    corrList.append(corrTemp)
    corrDistance = (0.5*(1 - corrTemp)).apply(np.sqrt)
    corrDistanceList.append(corrDistance)
    
graph_all_weeks = []
for w in range(0,10):
    print('Week ' + str(w) + '...')
    matrix1 = corrList[w]
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    tn_relation = transitionDataMatrixWeeks_directFollow_standardised[w].shape[0] / transitionDataMatrixWeeks_directFollow_standardised[w].shape[1]
    # The bandwidth of the KDE kernel
    kde_bwidth = 0.01
    # Finding the Вe-noised Сovariance matrix
    # denoised_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth)
    # denoised_matrix_byLib = pd.DataFrame(denoised_matrix_byLib, index=matrix.index, columns=matrix.columns) denoise_method='target_shrink',
    
    detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix1, tn_relation, kde_bwidth=kde_bwidth,  detone=True)
    # detoned_matrix_byLib = matrix1 #no denoised and detoned
    
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix1.index, columns=matrix1.columns)
    distance_matrix1 = (0.5*(1 - detoned_matrix_byLib)).apply(np.sqrt)
    graphBuild = MST(distance_matrix1, 'distance')
    # graphBuild = nx.from_pandas_adjacency(distance_matrix)   
    graph_all_weeks.append(graphBuild)

len(matrix1.columns)
matrix1.index
nx.Graph(matrix1)
distance_matrix1 = distance_matrix1[distance_matrix1.index]

a = distance_matrix1.columns.duplicated()
b = distance_matrix1.columns

dict(enumerate(distance_matrix1.columns))

a = distance_matrix1.loc[distance_matrix1.index == 'u-42988dbb9daefa40708f349a7c4ee0e81ee44ad1']

a.unique()

fig, ax = plt.subplots(1, 1, figsize = (15, 15), dpi=300)
sns.heatmap(detoned_matrix_byLib, cmap='RdYlGn', yticklabels=False, xticklabels=False)
plt.xlabel("Students")
plt.ylabel("Students")

sns.heatmap(distance_matrix1, cmap='RdYlGn', yticklabels=False, xticklabels=False)
plt.xlabel("Students")
plt.ylabel("Students")

w = 9

student2018_excellent.union(student2019_excellent)

student2018_excellent = ca1162018_transitionDataMatrixWeeks[w].loc[ca1162018_transitionDataMatrixWeeks[w].index.isin(ex3_excellent_2018.index)].index
student2018_weak = ca1162018_transitionDataMatrixWeeks[w].loc[~ca1162018_transitionDataMatrixWeeks[w].index.isin(ex3_excellent_2018.index)].index
student2019_excellent = ca1162019_transitionDataMatrixWeeks[w].loc[ca1162019_transitionDataMatrixWeeks[w].index.isin(ex3_excellent_2019.index)].index
student2019_weak = ca1162019_transitionDataMatrixWeeks[w].loc[~ca1162019_transitionDataMatrixWeeks[w].index.isin(ex3_excellent_2019.index)].index
student2020_excellent = ca1162020_transitionDataMatrixWeeks[w].loc[ca1162020_transitionDataMatrixWeeks[w].index.isin(ex3_excellent_2020.index)].index
student2020_weak = ca1162020_transitionDataMatrixWeeks[w].loc[~ca1162020_transitionDataMatrixWeeks[w].index.isin(ex3_excellent_2020.index)].index

studentCohort =  {"2018 2019 excellent": student2018_excellent.union(student2019_excellent), 
                  "2018 2019 weak": student2018_weak.union(student2019_weak), 
                  "2020 excellent" : student2020_excellent,
                  "2020 weak" : student2020_weak
                  }
graphLearning.visualiseMSTGraph(graph_all_weeks[w], studentCohort, reLabelIndex)  

import matplotlib.cm as cm  

G = graph_all_weeks[9].graph
node_color = []
nodelist = []

for n in G.nodes:
    nodelist.append(n)
    if n in student2018_excellent.union(student2018_weak):
        node_color.append('blue')
    elif n in student2019_excellent.union(student2019_weak):
        node_color.append('red')
    elif n in student2020_excellent.union(student2020_weak):
        node_color.append('green')
    else:
        node_color.append('yellow')

#exellent and weak
G = graph_all_weeks[9].graph
node_color = []
nodelist = []

for n in G.nodes:
    nodelist.append(n)
    if n in ex3_excellent_2020.index:
        node_color.append('blue')
    else:
        node_color.append('red')


    
plt.figure(figsize=(20,20))
# pos = nx.planar_layout(G)
pos = nx.circular_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', 3)
nx.draw_networkx_nodes(G, pos, nodelist,  node_size=100, cmap=cmap, node_color=node_color)
nx.draw_networkx_edges(G, pos, alpha=0.5)
# nx.draw_networkx_labels(G, pos, font_size=12)
plt.show()

#community detection
communityListWeeks = []
for w in range(0,10):
    print('Week ' + str(w) + '...')      
    num_comms = len(graph_all_weeks[w].graph._node)
    communityListWeeks.append(graphLearning.community_dection_graph(graph_all_weeks[w],  num_comms=num_comms))

communityWeek10 = graphLearning.community_dection_graph(graph_all_weeks[9], num_comms=len(graph_all_weeks[9].graph._node))    

def checkNumberOfStudentYearInEachCommunity(community, df1, df2, df3):
    result = []
    for c in community:
        r1 = len(df1.loc[df1.index.isin(c)])/len(df1)
        r2 = len(df2.loc[df2.index.isin(c)])/len(df2)
        r3 = len(df3.loc[df3.index.isin(c)])/len(df3)
        result.append(['number',len(df1.loc[df1.index.isin(c)]),len(df2.loc[df2.index.isin(c)]),len(df3.loc[df3.index.isin(c)]), len(c)])
        result.append(['percent in Dataset',len(df1.loc[df1.index.isin(c)])/len(df1),len(df2.loc[df2.index.isin(c)])/len(df2),len(df3.loc[df3.index.isin(c)])/len(df3), len(c)])
        result.append(['percent in Group', len(df1.loc[df1.index.isin(c)])/len(c),len(df2.loc[df2.index.isin(c)])/len(c),len(df3.loc[df3.index.isin(c)])/len(c), len(c)])
    return pd.DataFrame(result, columns=['category','Class 2018', 'Class 2019', 'Class 2020', 'Total in Group'])

def contigencyTableGroupCohort(community, df1, df2, df3):
    result = []
    for c in community:
        r1 = len(df1.loc[df1.index.isin(c)])
        r2 = len(df2.loc[df2.index.isin(c)])
        r3 = len(df3.loc[df3.index.isin(c)])
        result.append([r1,r2,r3])       
    return pd.DataFrame(result, columns=['Class 2018', 'Class 2019', 'Class 2020'])

checkStudentInCommunity = checkNumberOfStudentYearInEachCommunity(communityWeek10[2], ca1162018_transitionDataMatrixWeeks[9], 
                                                                  ca1162019_transitionDataMatrixWeeks[9],ca1162020_transitionDataMatrixWeeks[9])
contigencyTable = contigencyTableGroupCohort(communityWeek10[2], ca1162018_transitionDataMatrixWeeks[9], 
                                                                  ca1162019_transitionDataMatrixWeeks[9],ca1162020_transitionDataMatrixWeeks[9])

g, p, dof, expctd = stats.chi2_contingency(contigencyTable)
print((g,p))

import graphLearning
import scikit_posthocs as sp
pd.set_option("display.max_rows", None, "display.max_columns", None)
aw10 = graphLearning.extractAssessmentResultOfCommunities(communityWeek10, assessment3A, 'perCorrect3A')
aw10t = sp.posthoc_conover(aw10[3][5])
a = aw10[3]

goodCommunity = ca1162020_transitionDataMatrixWeeks[w].index # aw10[3][5][0]
badCommunity = pd.concat([ca1162019_transitionDataMatrixWeeks[w], ca1162018_transitionDataMatrixWeeks[w]]).index # aw10[3][5][1]
w = 9

for w in range(0,10):
    activityDataMatrixWeeks_pageTypeWeek[w] = graphLearning.mapNewLabel(activityDataMatrixWeeks_pageTypeWeek[w] , reLabelIndex)

extractGoodBadCommunity = activityDataMatrixWeeks_pageTypeWeek[w].loc[activityDataMatrixWeeks_pageTypeWeek[w].index.astype(str).isin(goodCommunity) | activityDataMatrixWeeks_pageTypeWeek[w].index.astype(str).isin(badCommunity)]
extractGoodBadCommunity['group'] = 0
extractGoodBadCommunity.loc[extractGoodBadCommunity.index.astype(str).isin(goodCommunity),['group']] = 0
extractGoodBadCommunity.loc[extractGoodBadCommunity.index.astype(str).isin(badCommunity),['group']] = 1

columnListStatsSig = []
for c in extractGoodBadCommunity.columns:
    t1 = stats.normaltest(extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == 0, [c]])[1][0]
    t2 = stats.normaltest(extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == 1, [c]])[1][0]
    if t1 <= 0.1 and t2 <= 0.1:
        columnListStatsSig.append(c)
        
extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == 2, ['Lecture_4']].hist(bins=80)

compareMean = []
for c in extractGoodBadCommunity.columns:
    arr1 = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == 0, [c]]
    arr2 = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == 1, [c]]
    test = stats.mannwhitneyu(arr1,arr2)[1]
    if test <= 0.05:
        if c!= 'group':
            meanGood = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == 0, [c]].mean()[0]
            meanBad = extractGoodBadCommunity.loc[extractGoodBadCommunity['group'] == 1, [c]].mean()[0]
            compareMean.append([c, meanGood, meanBad])
        # print(c + ': ' + str(test) + ' Good Community: ' + str(meanGood) + ' -- ' + 'Bad Community: ' + str(meanBad))
compareMeanDf = pd.DataFrame(compareMean, columns=['Material','During Covid', 'Pre COVID'])
compareMeanDfnewCol = compareMeanDf['Material'].str.split('_', expand = True)
compareMeanDf['week'] = compareMeanDfnewCol[1].astype(int)
compareMeanDf['MaterialType'] = compareMeanDfnewCol[0]
compareMeanDf = compareMeanDf.sort_values(['MaterialType','week'])

#draw horizontal barchart for compare mean activity
# create plot
fig, ax = plt.subplots(figsize=(30,20), dpi=150)
index = np.arange(len(compareMeanDf.index))
bar_width = 0.4
opacity = 1

rects1 = plt.barh(index, compareMeanDf['During Covid'], bar_width, alpha=opacity, color='b', label='During Covid')
rects2 = plt.barh(index + bar_width, compareMeanDf['Pre COVID'], bar_width, alpha=opacity, color='g', label='Pre COVID')

plt.ylabel('Course Materials', fontsize=20)
plt.xlabel('Average number of activities', fontsize=20)
plt.title('')
plt.yticks(index + bar_width, compareMeanDf.Material, fontsize=18)
plt.xticks(fontsize=20)
plt.legend(fontsize=25)


#IPR
activityDataMatrixWeeks_pageTypeWeek = ca1162018_activityDataMatrixWeeks_pageTypeWeek

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
    pcaDataWeeks[w]['covidResult'] = 0
    pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ca1162018_activityDataMatrixWeeks_pageTypeWeek[w].index), ['covidResult']] = 0    
    pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ca1162019_activityDataMatrixWeeks_pageTypeWeek[w].index), ['covidResult']] = 1
    pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ca1162020_activityDataMatrixWeeks_pageTypeWeek[w].index), ['covidResult']] = 2

    
for w in range(0,10):
    pcaDataWeeks[w] = graphLearning.mapNewLabel(pcaDataWeeks[w], reLabelIndex)

communitySparation = communityWeek10[0]
for w in range(0,10):
    pcaDataWeeks[w]['covidResult'] = 0
    for c in range(len(communitySparation)):
        pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(communitySparation[c]), ['covidResult']] = c    

for w in range(0,12):
    pcaDataWeeks[w]['result'] = 0
    pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ex3_excellent_2018.index), ['result']] = 1
    pcaDataWeeks[w].loc[pcaDataWeeks[w].index.isin(ex3_weak_2018.index), ['result']] = 0

fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0
num_bins = 50
for w in range(0,10):
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
w = 11
fig = plt.figure(figsize=(30,20),dpi=120)
ax = fig.subplots()
ax.set_xlabel('Eigenvalues', fontsize = 30)
ax.set_ylabel('IPR', fontsize = 30)
ax.tick_params(axis='both', which='major', labelsize=25)
ax.tick_params(axis='both', which='minor', labelsize=25)
ax.set_title('Inverse Participation Ratio week ' + str(w+1), fontsize = 30)
ax.grid()
# graph[countGraph].axhline(y=0, color='k')
# graph[countGraph].axvline(x=0, color='k')
eigenValueList = pca_result[w].explained_variance_
eigenVectorList = pca_result[w].components_
IPRlist = libRMT.IPRarray(eigenValueList,eigenVectorList)
ax.axhline(y=IPRlist['IPR'].mean(), color='k', label='mean value of IPR') 
ax.plot(IPRlist['eigenvalue'], IPRlist['IPR'], '-', color ='blue', label='IPR')
ax.legend(loc='upper right')
plt.show()

#outbounce select
a = libRMT.selectOutboundComponents(pcaDataWeeks[11],pca_result[9].explained_variance_)

#eigenvalues
fig = plt.figure(figsize=(40,30),dpi=240)
graph = []
countGraph = 0
num_bins = 100
for w in range(0,10):
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

#week 10 only
w=11
num_bins = 100
fig = plt.figure(figsize=(10,10),dpi=120)
ax = fig.add_subplot(1,1,1)

ax.set_xlabel('eigenvalue λ', fontsize = 15)
ax.set_ylabel('P(λ)', fontsize = 15)
ax.set_title('Empirical eigenvalue distribution vs RMT prediction', fontsize = 20)
ax.grid()
# graph[countGraph].axhline(y=0, color='k')
# graph[countGraph].axvline(x=0, color='k')
eigenValueList_graph = pca_result[w].explained_variance_

n, bins, patches = ax.hist(eigenValueList_graph, num_bins, 
                       density = 1,  
                       color ='blue',  
                       alpha = 0.7, label='Empirical') 
densityArray = libRMT.marcenkoPastur(len(pcaDataWeeks[w]),len(pcaDataWeeks[w].columns),bins)
density = densityArray[0]
ax.plot(bins, density, '-', color ='black',label='RMT') 
ax.legend(loc='upper right')
min_lambda = densityArray[1]
max_lambda = densityArray[2]
ax.text( min_lambda*1.1, -0.08, 'λ-', color = 'black', ha = 'center', va = 'center',fontsize=15)
ax.text( max_lambda*1.1, -0.08, 'λ+', color = 'black', ha = 'center', va = 'center',fontsize=15)
        
plt.show()


w = 11
title = {0 : 'Weak 2020', 1:'Excellent 2020', 2:'Class 2020'} 
libRMT.biplot(pcaDataWeeks[w],
       np.transpose(pca_result[w].components_[[1,2], :]),
       pcaDataWeeks[w].loc[:,['result']], activityDataMatrixWeeks_pageTypeWeek[w].columns, 'pc2','pc3', 15, 'result', title)

#plot loadings
    
libRMT.plotLoadings(1,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  
libRMT.plotLoadings(7,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  
libRMT.plotLoadings(9,pca_result,activityDataMatrixWeeks_pageTypeWeek,columnsReturn2)  

pc = 1
from scipy import stats
for w in range(0,10):
    if 'pc'+str(pc) in pcaDataWeeks[w].columns:
        a1 = pcaDataWeeks[w].loc[pcaDataWeeks[w]['covidResult'] == 2,['pc'+str(pc)]]
        b1 = pcaDataWeeks[w].loc[pcaDataWeeks[w]['covidResult'] == 1,['pc' + str(pc)]]
        t1, p1 = stats.ttest_ind(a1,b1)
    
        
        print('Week ' + str(w) + ':')
        print('--PC' + str(pc) + ': ' + 't-value: ' + str(t1) + ' p-value: ' + str(p1))
        print('-- Excellent: ' + str(a1.mean()[0]))
        print('-- Weak: ' + str(b1.mean()[0]))  
        

#--------- statistical test if during and before covid come from the same distribution
x1 = ca1162019_activityDataMatrixWeeks_pageTypeWeek[11]['Practice_1']
x2 = ca1162020_activityDataMatrixWeeks_pageTypeWeek[9]['Practice_1']

stats.ks_2samp(x1, x2)

ca1162018_activityDataMatrixWeeks_pageTypeWeek_1 = ca1162019_activityDataMatrixWeeks_pageTypeWeek

for w in range(0,12):
    ca1162018_activityDataMatrixWeeks_pageTypeWeek_1[w]['result'] = 0
    ca1162018_activityDataMatrixWeeks_pageTypeWeek_1[w].loc[ca1162018_activityDataMatrixWeeks_pageTypeWeek_1[w].index.isin(ex3_excellent_2019.index), ['result']] = 1
    ca1162018_activityDataMatrixWeeks_pageTypeWeek_1[w].loc[ca1162018_activityDataMatrixWeeks_pageTypeWeek_1[w].index.isin(ex3_weak_2019.index), ['result']] = 0


from scipy import stats

for w in range(0, 12):
    
    print('Week ' + str(w) + ':')
    i = 1
    for c in sorted(ca1162018_activityDataMatrixWeeks_pageTypeWeek_1[w].columns):
        if c != 'result':    
            a1 = ca1162018_activityDataMatrixWeeks_pageTypeWeek_1[w].loc[ca1162018_activityDataMatrixWeeks_pageTypeWeek_1[w]['result'] == 1,[c]]
            b1 = ca1162018_activityDataMatrixWeeks_pageTypeWeek_1[w].loc[ca1162018_activityDataMatrixWeeks_pageTypeWeek_1[w]['result'] == 0,[c]]
            t1, p1 = stats.mannwhitneyu(a1,b1)
        
            
            if p1 <= 0.05:
                print(str(i) + '. ' + c + ': ' + 't-value: ' + str(t1) + ' p-value: ' + str(p1) + '-- Excellent: ' + str(a1.mean()[0]) + '-- Weak: ' + str(b1.mean()[0]))
                #print('-- Excellent: ' + str(a1.mean()[0]))
                #print('-- Weak: ' + str(b1.mean()[0]))
                i += 1











