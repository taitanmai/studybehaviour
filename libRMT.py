import numpy as np
import pandas as pd
import snap
import os
from sklearn.linear_model import LinearRegression
import mlfinlab as ml
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import networkx as nx

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def pEval(lambda_max, lambda_min, q, eigenValue):
    ret = np.sqrt((lambda_max - eigenValue) * (eigenValue - lambda_min))
    ret /= 2 * np.pi * q * eigenValue
    return ret if lambda_min < eigenValue < lambda_max else 0.0        
        
def marcenkoPastur(sampleLength, featuresLength, X):

    q = featuresLength / float(sampleLength)

    lambda_min = (1 - np.sqrt(q))**2
    lambda_max = (1 + np.sqrt(q))**2
    density = []
    i = lambda_min
    for i in X:
        density.append(pEval(lambda_max,lambda_min,q,i))
        i += 0.01
       
    return [density, lambda_min, lambda_max]
    
def IPRcal(eigenvector):
    ipr = 0
    for ev in eigenvector:
        ipr += ev**4
    return ipr

def IPRarray(eigenvalueList, eigenvectorList):
    result = []
    for eVal,eVec in zip(eigenvalueList,eigenvectorList):
        result.append([eVal,IPRcal(eVec)])
    return pd.DataFrame(result, columns=['eigenvalue','IPR'])

def selectOutboundComponents(datasetPC, eigenvalueList):
    sampleLength = len(datasetPC)
    featuresLength = len(datasetPC.columns)
    
    q = featuresLength / float(sampleLength)

    # lambda_min = (1 - np.sqrt(q))**2
    lambda_max = (1 + np.sqrt(q))**2
    lambda_min = (1 - np.sqrt(q))**2
    pcList = ['pc' + str(i) for i in range(1,26)]
    columnList = []
    for eVal, pc in zip(eigenvalueList, pcList):
        if eVal >= 1: #lambda_max:
            columnList.append(pc)
            
    return datasetPC.loc[:,columnList]

def projection(datapoint, dim_V, p, feature_vecs):
    result = np.zeros(p)
    for idx in range(dim_V):
        dotprod = float(np.vdot(feature_vecs[:,idx],datapoint))
        result = result + dotprod*feature_vecs[:,idx]
        
    return np.array([float(i) for i in result])       
            
class RMTClassifier(object):
    def __init__(self, threshold_multiple=1, train_cutoff=0.95):
        self.threshold_multiple = threshold_multiple
        self.train_cutoff = train_cutoff
        self.epsilon = None
        self.feature_vecs = None
        self.dim_V = None
        self.p = None
    
    def fit(self, trainedData):
        N,p = trainedData.shape
        self.p = p
        gamma = float(p)/N
        threshold = ((1+np.sqrt(gamma))**2)*self.threshold_multiple
        
        C = np.dot(trainedData.T, trainedData)/float(N)
        
        evals, evecs = np.linalg.eig(C)
        idx = evals.argsort()
        idx = idx[::-1]
        evals = evals[idx]
        evecs = evecs[:,idx]
        
        dim_V = evals[evals > threshold].shape[0]
        feature_vecs = evecs[:,:dim_V]
        
        self.dim_V, self.feature_vecs = dim_V, feature_vecs
        
        similarity = []
        for i in range(N):
            row = trainedData[i,:]
            row_proj = projection(row, dim_V, p, feature_vecs)
            similarity.append(np.linalg.norm(row - row_proj))
            
        similarity.sort()
        similarity = np.array(similarity)
        cutoff_idx = int(self.train_cutoff * len(similarity))
        epsilon = similarity[cutoff_idx]
        self.epsilon = epsilon
        
    def predict(self, testData, epsilon_multiple=1):
        test_similarity = []
        for i in range(testData.shape[0]):
            row = testData[i,:]
            row_proj = projection(row, self.dim_V, self.p, self.feature_vecs)
            test_similarity.append(np.linalg.norm(row - row_proj))
        predictions = np.array([1 if x < self.epsilon*epsilon_multiple else 0 for x in test_similarity])
        return predictions
      
def generateTransition(activityCodeList, selfLoop = 1):
    result = []
    for node1 in activityCodeList: 
        for node2 in activityCodeList: 
            if selfLoop == 0:
                if node1[0] == node2[0]:
                    continue
            result.append([(node1[0],node2[0]),node1[1] + '-' + node2[1]])
    return result

def assignNodeNumber(activityList):
    return [(i,a) for i,a in zip(range(1,len(activityList)+1),activityList)]

def girvin_neuman_profile_extract(rowData, activityCodeList, index,week):
    columnList = generateTransition(activityCodeList)
    G1 = snap.TNGraph.New()
    checkActivityList = []
    # for node1 in activityCodeList:
    #     for node2 in activityCodeList:
    #         a = node1[1] + '-' + node2[1]
    #         if a in rowData.index:
    #             if node1[0] not in checkActivityList:
    #                 G1.AddNode(node1[0])
    #                 checkActivityList.append(node1[0])
    #             if node2[0] not in checkActivityList:
    #                 G1.AddNode(node2[0])
    #                 checkActivityList.append(node2[0])
    for i in columnList:
        if i[1] in rowData.index:
            if rowData[i[1]] > 0:
                if i[0][0] not in checkActivityList:
                    G1.AddNode(i[0][0])
                    checkActivityList.append(i[0][0])
                if i[0][1] not in checkActivityList:
                    G1.AddNode(i[0][1])
                    checkActivityList.append(i[0][1])
                G1.AddEdge(i[0][0],i[0][1])
    G1_undirect = snap.ConvertGraph(snap.PUNGraph,G1)
    # snap.DrawGViz(G1_undirect, snap.gvlDot, "graphs/week/" + str(week) + "/" + index + ".png", index)
    CmtyV = snap.TCnComV()
    modularity = snap.CommunityGirvanNewman(G1_undirect, CmtyV)
    noOfCluster = len(CmtyV)
    clusterList = []
    for Cmty in CmtyV:
        community = []
        for NI in Cmty:
            community.append(NI)
        clusterList.append(community)
        
    return [index, modularity, noOfCluster, clusterList]
            
def visualiseGraph(rowData, activityCodeList, fileName, title, undirect_conversion=False):
    columnList = generateTransition(activityCodeList)
    G1 = snap.TNGraph.New()
    checkActivityList = []
    for i in columnList:
        if i[1] in rowData.index:
            if rowData[i[1]] > 0:
                if i[0][0] not in checkActivityList:
                    G1.AddNode(i[0][0])
                    checkActivityList.append(i[0][0])
                if i[0][1] not in checkActivityList:
                    G1.AddNode(i[0][1])
                    checkActivityList.append(i[0][1])
                G1.AddEdge(i[0][0],i[0][1])
        
    if undirect_conversion:
        G1 = snap.ConvertGraph(snap.PUNGraph,G1)
    snap.DrawGViz(G1, snap.gvlDot, "graphs/" + "/" + fileName + ".png", title, True)
    

def regressionToCleanEigenvectorEffect(originalData, PCAdata, eigenvector_to_clean):
    if len(originalData) != len(PCAdata):
        print('Sample length should be equal!!!')
        return
    selectedComponent = 'pc' + str(eigenvector_to_clean)
    selectedComponentData = PCAdata.loc[:,[selectedComponent]]
    cleanedData = pd.DataFrame(columns = originalData.columns, index=originalData.index)
    for c in originalData.columns:
        x = selectedComponentData
        y = originalData[c]
        regressor = LinearRegression()  
        regressor.fit(x, y) #training the algorithm
        alpha = regressor.intercept_
        beta = regressor.coef_[0]

        epsilon = []
        for s in originalData.index:
            # epsilonOfstudentS = originalData.loc[originalData.index == s,[c]].values[0][0] - alpha - beta*selectedComponentData.loc[selectedComponentData.index == s][selectedComponent][0]
            epsilonOfstudentS = originalData.loc[originalData.index == s,[c]].values[0][0] - beta*selectedComponentData.loc[selectedComponentData.index == s][selectedComponent][0] - alpha
            epsilon.append(epsilonOfstudentS)
        cleanedData[c] = epsilon
    return cleanedData

def getPCA(matrix):
    # Get eVal,eVec from a Hermitian matrix
    eVal,eVec=np.linalg.eigh(matrix)
    indices=eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal,eVec=eVal[indices],eVec[:,indices]
    eVal=np.diagflat(eVal)
    return eVal,eVec

def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov/np.outer(std,std)
    corr[corr<-1], corr[corr>1] = -1,1 # numerical error
    return corr

def denoisedCorr(corr, sampleLength, featuresLength):
    # Remove noise from corr by fixing random eigenvalues
    q = featuresLength / float(sampleLength)

    # lambda_min = (1 - np.sqrt(q))**2
    lambda_max = (1 + np.sqrt(q))**2    
    eVal, eVec = getPCA(corr)
    nFacts = len(eVal[eVal > lambda_max])

    eVal = np.diag(eVal).copy()
    eVal[nFacts:] = eVal[nFacts:].sum()/float(eVal.shape[0]-nFacts)
    
    eVal = np.diag(eVal)
    corr1 = np.dot(eVec,eVal).dot(eVec.T)
    corr1 = cov2corr(corr1)
    return pd.DataFrame(corr1, columns = corr.columns, index = corr.index)

def cleanDataWeek(normalisedDataWeek):
    risk_estimators = ml.portfolio_optimization.RiskEstimators()
    # Setting the required parameters for de-noising
    # Relation of number of observations T to the number of variables N (T/N)
    tn_relation = normalisedDataWeek.shape[0] / normalisedDataWeek.shape[1]
    # The bandwidth of the KDE kernel
    kde_bwidth = 0.25
    # Finding the Вe-noised Сovariance matrix
    denoised_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth)
    denoised_matrix_byLib = pd.DataFrame(denoised_matrix_byLib, index=matrix.index, columns=matrix.columns)
    
    detoned_matrix_byLib = risk_estimators.denoise_covariance(matrix, tn_relation, kde_bwidth, detone=True)
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=matrix.index, columns=matrix.columns)

