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

ca1162018_activityDataMatrixWeeks = []
ca1162019_activityDataMatrixWeeks = []
ca1162020_activityDataMatrixWeeks = []

for w in range(0,12):
    ca1162018_activityDataMatrixWeeks.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv'))
#    ca1162018_transitionDataMatrixWeeks[w].user = ca1162018_transitionDataMatrixWeeks[w].user + '-2018'
    ca1162018_activityDataMatrixWeeks[w] = ca1162018_activityDataMatrixWeeks[w].set_index(ca1162018_activityDataMatrixWeeks[w].columns[0])
      
    ca1162019_activityDataMatrixWeeks.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1162019_activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv'))
#    ca1162019_transitionDataMatrixWeeks[w].user = ca1162019_transitionDataMatrixWeeks[w].user + '-2019'
    ca1162019_activityDataMatrixWeeks[w] = ca1162019_activityDataMatrixWeeks[w].set_index(ca1162019_activityDataMatrixWeeks[w].columns[0])
    
for w in range(0,10):
    ca1162020_activityDataMatrixWeeks.append(pd.read_csv(basePath + 'transitionMatrixStorage_new/ca1162020_activityDataMatrixWeeks_pageTypeWeek_newPractice_w'+str(w)+'.csv'))
#    ca1162020_transitionDataMatrixWeeks[w].user = ca1162020_transitionDataMatrixWeeks[w].user + '-2020'   
    ca1162020_activityDataMatrixWeeks[w] = ca1162020_activityDataMatrixWeeks[w].set_index( ca1162020_activityDataMatrixWeeks[w].columns[0])
    
 