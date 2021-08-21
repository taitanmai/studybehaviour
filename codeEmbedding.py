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
import re


basePath = 'D:\\Dataset\\PhD\\'


dataUpload_2020 = pd.read_csv(basePath + 'ca116_uploads_2020.csv')
dataUpload_2020['date'] = pd.to_datetime(dataUpload_2020.date)
dataUpload_2020['user'] = dataUpload_2020['user'] + '-2020'
dataUpload_2020 = dataUpload_2020.loc[:,['date','user','task','upload']]
dataUpload_2020 = dataUpload_2020.sort_values(by=['user','date'])
dataUpload_2020_filtered = dataUpload_2020.groupby([pd.Grouper('user'),pd.Grouper('task')]).last()

dataUpload_2019 = pd.read_csv(basePath + 'ca116_uploads_2019.csv')
dataUpload_2019['date'] = pd.to_datetime(dataUpload_2019.date)
dataUpload_2019['user'] = dataUpload_2019['user'] + '-2019'
dataUpload_2019 = dataUpload_2019.loc[:,['date','user','task','upload']]
dataUpload_2019 = dataUpload_2019.sort_values(by=['user','date'])
dataUpload_2019_filtered = dataUpload_2019.groupby([pd.Grouper('user'),pd.Grouper('task')]).last()

dataUpload_2018 = pd.read_csv(basePath + 'ca116_uploads.csv')
dataUpload_2018['date'] = pd.to_datetime(dataUpload_2018.date)
dataUpload_2018['user'] = dataUpload_2018['user'] + '-2018'
dataUpload_2018 = dataUpload_2018.loc[:,['date','user','task','upload']]
dataUpload_2018 = dataUpload_2018.sort_values(by=['user','date'])
dataUpload_2018_filtered = dataUpload_2018.groupby([pd.Grouper('user'),pd.Grouper('task')]).last()

dataUpload = pd.concat([dataUpload_2018_filtered,dataUpload_2019_filtered,dataUpload_2020_filtered])
dataUpload = dataUpload.dropna()
dataUpload.index[0]

tokenized = []
for i in range(0,len(dataUpload['upload'])):
    try:
        tokenized.append(re.sub(r'\s+',' ',re.sub(r'\n+', ' ', dataUpload['upload'][i])).rstrip().split(' ')[2:])
    except: 
        print(dataUpload.index[i])

a = dataUpload_2019.loc[(dataUpload_2019['user'] == 'u-01f3c8c2fc51e05e12c9759010d1417712cca874-2019') &  
                        (dataUpload_2019['task'] == 'sum-integers-3.py')]     
a = dataUpload.index

#Import packages
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


tagged_data = [TaggedDocument(d, ['*'.join(i)]) for i, d in zip(dataUpload.index, tokenized)]
tagged_data

## Train doc2vec model
model = Doc2Vec(tagged_data, vector_size=50, window=3, min_count=1, workers=4, epochs = 200, dm =1)
# Save trained doc2vec model
model.save(basePath + "ca116_2vecSize50DM1.model")
## Load saved doc2vec model
# model= Doc2Vec.load(basePath + "ca116_2vec.model")
## Print model vocabulary

model2 = Doc2Vec(tagged_data, vector_size=200, window=3, min_count=1, workers=4, epochs = 100)
# Save trained doc2vec model
model2.save(basePath + "ca116_2vec2.model")

test_doc = "That is a good device".split(' ')
model.docvecs.most_similar(positive=[model.infer_vector(test_doc)],topn=5)
a = model2.docvecs.index2entity
model2.docvecs['u-008115f43c2e7de2b413fd2296bdbfffd416329e-2020*add-ten-numbers.py']
a = model.docvecs.doctags[0]
a = pd.DataFrame(model.docvecs.vectors_docs, index = model.docvecs.index2entity)

model.infer_vector(tokenized[0])
model.docvecs[0]

