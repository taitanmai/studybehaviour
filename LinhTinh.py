import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import linalg as LA
import libRMT

data = [[6,8,3,5,7],[34,7,16,4,6],[47,63,9,8,2],[10,4,12,13,32],[5,7,45,31,25],[45,6,12,19,21]]

df = pd.DataFrame(data, columns=['col1','col2', 'col3','col4','col5'])




# scaler = StandardScaler()
pca = PCA(n_components=min(len(df), len(df.columns)))
x = df.values    
x = x - np.mean(x,axis=0)    
# scaler.fit(x)
# x = scaler.transform(x)
pca.fit(x)
transformed_value = pca.fit_transform(x)
transformed_value1 = pd.DataFrame(transformed_value,columns=['pc1','pc2','pc3','pc4','pc5'], index=df.index)

df_standardised = pd.DataFrame(x, columns = df.columns, index = df.index)


df2 = (transformed_value1*pca.components_[:,1]).sum(axis=1)
a = df_standardised['col2'] - df2

pca.components_


(transformed_value1['pc2']*pca.components_[:,0][1])

corr = x.T.dot(x)

w, v = LA.eig(corr)

cleanData = libRMT.cleanEigenvectorEffect(df_standardised,transformed_value1,['pc1','pc3','pc4'],pca.components_)



-10.953962099928008*0.71462712+  3.9543819479993187*0.40647578

import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K
from keras.models import Sequential
import threading
from tensorflow.keras.utils import  Sequence
import numpy

model_resnet = ResNet50(weights='imagenet', include_top=False)

for layer in model_resnet.layers[:-12]:
    # 6 - 12 - 18 have been tried. 12 is the best.
    layer.trainable = False
    
x = model_resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x)
y = Dense(46, activation='softmax', name='img')(x)

print(x)

print(keras.__version__)


class Dense():
    def __init__(self, x,y):
        self.x = x
        self.y = y
    def __call__(self, x):
        return x + self.x + self.y
    

Dense(1,2)(3)

# regular expression for splitting by whitespace

import re
import pandas as pd
splitter = re.compile("\s+")

with open('D:/Dataset/Dao/list_attr_img.txt', 'r') as attr_img_file_1000:
        list_attr_img_1000 = [line.rstrip('\n') for line in attr_img_file_1000][2:]
        list_attr_img_1000 = [splitter.split(line) for line in list_attr_img_1000]
        
a = list_attr_img_1000[0]
pd.Series(a).unique()

a = pd.DataFrame(list_attr_img_1000)
a.to_csv('D:/Dataset/Dao/list_attr_img_extracted.csv')
