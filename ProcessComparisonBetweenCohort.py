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
from pm4py.evaluation.replay_fitness import factory as replay_factory
from pm4py.objects.conversion.log import factory as conversion_factory
from pm4py.evaluation.precision import factory as precision_factory
from pm4py.statistics.traces.log import case_statistics
import matplotlib.pyplot as plt
import dataProcessing
import FCAMiner
import seaborn as sns
from pm4py.algo.discovery.alpha import factory as alpha_miner
from pm4py.algo.discovery.inductive import factory as inductive_miner
from pm4py.algo.discovery.heuristics import factory as heuristics_miner
from pm4py.objects.conversion.dfg import factory as dfg_mining_factory
from pm4py.algo.discovery.dfg import factory as dfg_factory
from pm4py.visualization.heuristics_net import factory as hn_vis_factory

from pm4py.evaluation.generalization import factory as generalization_factory
from pm4py.evaluation.precision import factory as precision_factory
from pm4py.evaluation.simplicity import factory as simplicity_factory
from pm4py.evaluation.replay_fitness import factory as replay_factory
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
basePath = 'D:\\Dataset\\PhD\\'
#uploads
dataUpload = pd.read_csv('ca116_uploads.csv')
dataUpload['date'] = pd.to_datetime(dataUpload.date)

nonExUpload = dataUpload.drop(dataUpload.loc[dataUpload['task'].str.match('ex')].index)

weeks = [g for n, g in nonExUpload.groupby(pd.Grouper(key='date',freq='W'))]

# cummulativeResultWeek2 = weeks[1].groupby([pd.Grouper(key='user')]).sum()
# cummulativeResultWeek2['result'] = cummulativeResultWeek2['correct']/weeks[1].groupby([pd.Grouper(key='user')]).count()['date']
# cummulativeResultWeek2['successPassedRate'] = cummulativeResultWeek2['passed']/(cummulativeResultWeek2['passed'] + cummulativeResultWeek2['failed'])

# fig=plt.figure()
# ax=fig.add_axes([0,0,1,1])
# ax.scatter(cummulativeResultWeek2['result'], cummulativeResultWeek2['successPassedRate'] , color='r')

# ax.set_xlabel('Result')
# ax.set_ylabel('SuccessPassedRate')
# ax.set_title('scatter plot')
# plt.show()


#process exUploadData
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
                            'failed':'failed1A',
                            'passed':'passed1A',
                            'perPassed':'perPassed1A',
                            'testSubmitted':'testSubmitted1A',
                            'adjustedPerformance':'adjustedPerformance1A'}, 
                  inplace=True)
assessment2A.rename(columns={'correct':'correct2A',
                          'perCorrect':'perCorrect2A',
                          'failed':'failed2A',
                            'failed':'failed2A',
                            'passed':'passed2A',
                            'perPassed':'perPassed2A',
                            'testSubmitted':'testSubmitted2A',
                            'adjustedPerformance':'adjustedPerformance2A'}, 
                  inplace=True)
assessment3A.rename(columns={'correct':'correct3A',
                          'perCorrect':'perCorrect3A',
                          'failed':'failed1A',
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

ex1_excellent = assessment1A.loc[(assessment1A['perCorrect1A'] <= 1) & (assessment1A['perCorrect1A'] >= 1)]
ex1_weak = assessment1A.loc[(assessment1A['perCorrect1A'] >= 0) & (assessment1A['perCorrect1A'] < 0.4)]

ex2_excellent = assessment2A.loc[(assessment2A['perCorrect2A'] <= 1)&(assessment2A['perCorrect2A'] >= 0.4)]
ex2_weak = assessment2A.loc[(assessment2A['perCorrect2A'] >= 0) & (assessment2A['perCorrect2A'] < 0.4)]

ex3_excellent = assessment3A.loc[(assessment3A['perCorrect3A'] <= 1)&(assessment3A['perCorrect3A'] >= 0.4)]
ex3_weak = assessment3A.loc[(assessment3A['perCorrect3A'] >= 0) & (assessment3A['perCorrect3A'] < 0.4)]


#extract event log 
eventLog_ca116 = pd.read_csv('Event_Log_CA116_filtered.csv')
eventLog_ca116['time:timestamp'] = pd.to_datetime(eventLog_ca116['time:timestamp'])
# materials = eventLog_ca116.loc[:,['org:resource','concept:name','description']]
weeksEventLog = [g for n, g in eventLog_ca116.groupby(pd.Grouper(key='time:timestamp',freq='W'))]


last_4_weeks_eventLogs = pd.concat([weeksEventLog[11]])

ex1_personal_log_1 = pd.read_csv(basePath + 'ca1162019_goodCommunity_eventLog.csv') # last_4_weeks_eventLogs.loc[last_4_weeks_eventLogs['org:resource'].isin(ex3_excellent.index)]

ex1_personal_log_2 = pd.read_csv(basePath + 'ca1162019_badCommunity_eventLog.csv') # last_4_weeks_eventLogs.loc[last_4_weeks_eventLogs['org:resource'].isin(ex3_weak.index)]

len(ex1_personal_log_2['org:resource'].unique())

ex1_personal_log_1.rename(columns={'pageTypeWeek':'concept:name'}, inplace=True)
ex1_personal_log_2.rename(columns={'pageTypeWeek':'concept:name'}, inplace=True)


#Process Discovery

from pm4py.objects.conversion.log import factory as conversion_factory

ex1_personal_log_1_converted = conversion_factory.apply(ex1_personal_log_1)
ex1_personal_log_2_converted = conversion_factory.apply(ex1_personal_log_2)

#Heuristic Miner

from pm4py.algo.discovery.heuristics import factory as heuristics_miner
from pm4py.visualization.heuristics_net import factory as hn_vis_factory

excellent_heu_net = heuristics_miner.apply_heu(ex1_personal_log_1_converted, parameters={"dependency_thresh": 0.1})
gviz = hn_vis_factory.apply(excellent_heu_net)
hn_vis_factory.view(gviz)



weak_heu_net = heuristics_miner.apply_heu(ex1_personal_log_2_converted, parameters={"dependency_thresh": 0.0})
gviz = hn_vis_factory.apply(weak_heu_net)
hn_vis_factory.view(gviz)

excellent_dfg = excellent_heu_net.dfg
weak_dfg = weak_heu_net.dfg

#DFG
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
dfg = dfg_discovery.apply(ex1_personal_log_1_converted)
dfg[list(dfg)[0]]
a = list(dfg)
len(list(dfg))

import graphLearning
graph = graphLearning.createGraphFromCounter(dfg)


import community as community_louvain  
import matplotlib.cm as cm  
import networkx as nx
partition = community_louvain.best_partition(graph)

G = graph
pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, 
                      cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()

community_louvain.modularity(partition, G)

girvannewman = graphLearning.community_dection_graph(graph, mst=False)

from pm4py.visualization.dfg import visualizer as dfg_visualization
gviz = dfg_visualization.apply(dfg, log=ex1_personal_log_1_converted, variant=dfg_visualization.Variants.FREQUENCY)
dfg_visualization.view(gviz)

def fixDfg(dfg, activityList = ['Read_Labsheet','Read_Lecture_Note','Excercise','Check_solution']):
    result = {}
    transitionList = []
    for i in activityList:
        for j in activityList:
            transitionList.append((i,j))
    for t in transitionList:
        if t in dfg:
            result.update({t : dfg[t]})
        else:
            result.update({t : 0})
    return result

def percentageFixDFG(dfg_fix, activityList = ['Read_Labsheet','Read_Lecture_Note','Excercise','Check_solution']):
    result = {}
    for i in activityList:
        count = 0
        for j in activityList:
            count = count + dfg_fix[(i,j)]
        for j in activityList:
            result.update({(i,j) : float(dfg_fix[(i,j)])/float(count)})
    return result

def averagePerPersonFixDFG(dfg_fix,noOfPeople, activityList = ['Read_Labsheet','Read_Lecture_Note','Excercise','Check_solution']):
    result = {}
    for i in activityList:
        for j in activityList:
            result.update({(i,j) : float(dfg_fix[(i,j)])/float(noOfPeople)})
    return result

def diffTwoMatrix(matrix1, matrix2, activityList = ['Read_Labsheet','Read_Lecture_Note','Excercise','Check_solution']):
    result = {}
    for i in activityList:
        for j in activityList:
            result.update({(i,j) : matrix1[(i,j)] - matrix2[(i,j)]})
    return result

excellent_dfg_fixed = fixDfg(excellent_dfg)
weak_dfg_fixed = fixDfg(weak_dfg)

excellent_dfg_fixed_percentage = percentageFixDFG(excellent_dfg_fixed)
weak_dfg_fixed_percentage = percentageFixDFG(weak_dfg_fixed)

excellent_average = averagePerPersonFixDFG(excellent_dfg_fixed,len(ex1_personal_log_1['org:resource'].unique()))
weak_average = averagePerPersonFixDFG(weak_dfg_fixed,len(ex1_personal_log_2['org:resource'].unique()))

diff_absolute = diffTwoMatrix(excellent_average,weak_average)



dfg_miner_time_diff_absolute = diffTwoMatrix(dfg_miner_excellent_dfg,dfg_miner_weak_dfg)


#Inductive Miner
from pm4py.algo.discovery.inductive import factory as inductive_miner

tree = inductive_miner.apply_tree(ex1_personal_log_1_converted)

from pm4py.visualization.process_tree import factory as pt_vis_factory

gviz = pt_vis_factory.apply(tree)
pt_vis_factory.view(gviz)

from pm4py.algo.discovery.inductive import factory as inductive_miner

net, initial_marking, final_marking = inductive_miner.apply(ex1_personal_log_1_converted)
from pm4py.visualization.petrinet import factory as pn_vis_factory

gviz = pn_vis_factory.apply(net, initial_marking, final_marking)
pn_vis_factory.view(gviz)

#variant

from pm4py.statistics.traces.log import case_statistics

var_with_count = case_statistics.get_variant_statistics(ex1_personal_log_1_converted, parameters={"max_variants_to_return": 5})


