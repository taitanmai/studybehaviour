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

ca1162018_activityDataMatrixWeeks_pageTypeWeek = []
ca1162019_activityDataMatrixWeeks_pageTypeWeek = []
# ca1162020_activityDataMatrixWeeks_pageTypeWeek = []

for w in range(0,12):
    ca1162018_activityDataMatrixWeeks_pageTypeWeek.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv', index_col=0))
    ca1162018_activityDataMatrixWeeks_pageTypeWeek[w].index = ca1162018_activityDataMatrixWeeks_pageTypeWeek[w].index + '-2018'
    
    ca1162019_activityDataMatrixWeeks_pageTypeWeek.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1162019_activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv', index_col=0))
    ca1162019_activityDataMatrixWeeks_pageTypeWeek[w].index = ca1162019_activityDataMatrixWeeks_pageTypeWeek[w].index + '-2019'
   
    # ca1162020_activityDataMatrixWeeks_pageTypeWeek.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1162020_activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv', index_col=0))
    # ca1162020_activityDataMatrixWeeks_pageTypeWeek[w].index = ca1162020_activityDataMatrixWeeks_pageTypeWeek[w].index + '-2020'
    

activityDataMatrixWeeks_pageTypeWeek = []
for w in range(0,12):
    activityDataMatrixWeeks_pageTypeWeek.append(pd.concat([ca1162018_activityDataMatrixWeeks_pageTypeWeek[w], ca1162019_activityDataMatrixWeeks_pageTypeWeek[w]],join='inner'))

#score
ex3_excellent_2018 = pd.read_csv(basePath + 'ca1162018_ex3_excellent.csv',index_col=0)
ex3_excellent_2018.index = ex3_excellent_2018.index + '-2018'
ex3_weak_2018 = pd.read_csv(basePath + 'ca1162018_ex3_weak.csv',index_col=0)
ex3_weak_2018.index = ex3_weak_2018.index + '-2018'

ex3_excellent_2019 = pd.read_csv(basePath + 'ca1162019_ex3_excellent.csv',index_col=0)
ex3_excellent_2019.index = ex3_excellent_2019.index + '-2019'
ex3_weak_2019 = pd.read_csv(basePath + 'ca1162019_ex3_weak.csv',index_col=0)
ex3_weak_2019.index = ex3_weak_2019.index + '-2019'

ex2_excellent_2018 = pd.read_csv(basePath + 'ca1162018_ex2_excellent.csv',index_col=0)
ex2_excellent_2018.index = ex2_excellent_2018.index + '-2018'
ex2_weak_2018 = pd.read_csv(basePath + 'ca1162018_ex2_weak.csv',index_col=0)
ex2_weak_2018.index = ex2_weak_2018.index + '-2018'

ex2_excellent_2019 = pd.read_csv(basePath + 'ca1162019_ex2_excellent.csv',index_col=0)
ex2_excellent_2019.index = ex2_excellent_2019.index + '-2019'
ex2_weak_2019 = pd.read_csv(basePath + 'ca1162019_ex2_weak.csv',index_col=0)
ex2_weak_2019.index = ex2_weak_2019.index + '-2019'

# ex3_excellent_2020 = pd.read_csv(basePath + 'ca1162020_ex3_excellent.csv',index_col=0)
# ex3_excellent_2020.index = ex3_excellent_2020.index + '-2020'
# ex3_weak_2020 = pd.read_csv(basePath + 'ca1162020_ex3_weak.csv',index_col=0)
# ex3_weak_2020.index = ex3_weak_2020.index + '-2020'

ex3_excellent = pd.concat([ex3_excellent_2018,ex3_excellent_2019])
ex3_weak = pd.concat([ex3_weak_2018,ex3_weak_2019])
assessment3A = pd.concat([ex3_excellent, ex3_weak])
# assessment3A = graphLearning.mapNewLabel(assessment3A, reLabelIndex)