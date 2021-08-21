import numpy as np
import pandas as pd
import snap
import os
from sklearn.linear_model import LinearRegression
import mlfinlab as ml
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

def selectOutboundComponents(datasetPC, eigenvalueList, mode='upper_lower'):
    sampleLength = len(datasetPC)
    featuresLength = len(datasetPC.columns)
    
    q = featuresLength / float(sampleLength)

    # lambda_min = (1 - np.sqrt(q))**2
    lambda_max = (1 + np.sqrt(q))**2
    lambda_min = (1 - np.sqrt(q))**2
    pcList = ['pc' + str(i) for i in range(1,26)]
    columnList = []
    for eVal, pc in zip(eigenvalueList, pcList):
        if mode == 'upper_lower':
            if (eVal >= lambda_max) or (eVal <= lambda_min):
                columnList.append(pc)
        elif mode == 'upper':
            if (eVal >= lambda_max):
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
    return
    if len(originalData) != len(PCAdata):
        print('Sample length should be equal!!!')
        return
    selectedComponents = []
    for eigv in eigenvector_to_clean:
        selectedComponents.append('pc' + str(eigv))
    selectedComponentData = PCAdata.loc[:,selectedComponents]
    cleanedData = pd.DataFrame(columns = originalData.columns, index=originalData.index)
    
    scaler = StandardScaler()
    scaler.fit(originalData)
    originalData = pd.DataFrame(scaler.transform(originalData),columns = originalData.columns, index=originalData.index)
    
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
            data_to_remove = 0
            for selectedComponent in selectedComponents:
                data_to_remove = data_to_remove + beta*selectedComponentData.loc[selectedComponentData.index == s][selectedComponent][0]
            epsilonOfstudentS = originalData.loc[originalData.index == s,[c]].values[0][0] - data_to_remove
            epsilon.append(epsilonOfstudentS)
        cleanedData[c] = epsilon
    return cleanedData

def cleanEigenvectorEffect(originalData, PCAdata, eigenvector_to_clean, pca_components, alpha, beta):
    if len(originalData) != len(PCAdata):
        print('Sample length should be equal!!!')
        return

    scaler = StandardScaler()
    scaler.fit(originalData)
    cleanedData = pd.DataFrame(scaler.transform(originalData),columns = originalData.columns, index=originalData.index)
    
    
    
    removedPCscores = pd.DataFrame(0,columns = cleanedData.columns, index = cleanedData.index)
    for col, component_features in zip(cleanedData.columns, pca_components.T):
        for pc in range(1,len(component_features)+1):
            if ('pc' + str(pc)) in eigenvector_to_clean:
                if ('pc' + str(pc)) == 'pc1':
                    removedPCscores[col] = removedPCscores[col] + beta*PCAdata['pc' + str(pc)]*component_features[pc-1]
                else:
                    removedPCscores[col] = removedPCscores[col] + alpha*PCAdata['pc' + str(pc)]*component_features[pc-1]
        
    
    for c in cleanedData.columns:
        cleanedData[c] = cleanedData[c] - removedPCscores[c]
    
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
    denoised_matrix_byLib = risk_estimators.denoise_covariance(normalisedDataWeek, tn_relation, kde_bwidth)
    denoised_matrix_byLib = pd.DataFrame(denoised_matrix_byLib, index=normalisedDataWeek.index, columns=normalisedDataWeek.columns)
    
    detoned_matrix_byLib = risk_estimators.denoise_covariance(normalisedDataWeek, tn_relation, kde_bwidth, detone=True)
    detoned_matrix_byLib = pd.DataFrame(detoned_matrix_byLib, index=normalisedDataWeek.index, columns=normalisedDataWeek.columns)

def biplot(score, coeff , y, columns, col1, col2, scaleLoadings=25, classifierCol = 'result_exam_1', title=['Failed students','Pass students']):
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
    colors = ['g','r','y','blue','black','orange']
    markers=['o','^','x','d','p','*']
    for s,l in enumerate(classes):
        label = title[l]
        plt.scatter(score.loc[score[classifierCol] == l,[col1]],
                    score.loc[score[classifierCol] == l,[col2]], 
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
        plt.arrow(0, 0, coeff[i,0]*scaleLoadings, coeff[i,1]*scaleLoadings, color = 'blue', alpha = 0.9,linestyle = '-',linewidth = 0.2, overhang=0.05)
        plt.text(coeff[i,0]*scaleLoadings* 1.05, coeff[i,1] *scaleLoadings* 1.05, str(columns[i]), color = 'k', ha = 'center', va = 'center',fontsize=8)

    # plt.xlabel(col1, size=14)
    # plt.ylabel(col2, size=14)
    # limx= 0.5
    # limy= 0.5
    # plt.xlim([-limx,limx])
    # plt.ylim([-limy,limy])
    # plt.grid()
    # plt.tick_params(axis='both', which='both', labelsize=14)


#plot loadings
def plotLoadings(week,pca_result,transitionDataMatrixWeeks, columnsReturn1):
    loadings = pd.DataFrame(pca_result[week].components_[0:8, :], 
                            columns=columnsReturn1[week])
    maxPC = 1.01 * np.max(np.max(np.abs(loadings.loc[0:8, :])))
    f, axes = plt.subplots(1, 8, figsize=(20, 20), sharey=True)
    for i, ax in enumerate(axes):
        pc_loadings = loadings.loc[i, :]
        colors = ['C0' if l > 0 else 'C1' for l in pc_loadings]
        ax.axvline(color='#888888')
        ax.axvline(x=0.1, color='#888888')
        ax.axvline(x=-0.1, color='#888888')
        pc_loadings.plot.barh(ax=ax, color=colors)
        ax.set_xlabel(f'PC{i+1}')
        ax.set_xlim(-maxPC, maxPC)
    plt.title('Week '+str(week+1))