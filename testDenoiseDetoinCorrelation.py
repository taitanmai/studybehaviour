import numpy as np,pandas as pd
import matplotlib.pyplot as plt
#---------------------------------------------------
def mpPDF(var,q,pts):
    print('Print errPDFs...')
    print(f'var: {0}, q: {1}, pts: {2}', var,q,pts)
    # Marcenko-Pastur pdf
    # q=T/N
    eMin, eMax = var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2
    eVal = np.linspace(eMin, eMax, pts)
    print(eVal)
    pdf = q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
    pdf = pd.Series(pdf, index=eVal)
    return pdf

from sklearn.neighbors.kde import KernelDensity
#---------------------------------------------------
def getPCA(matrix):
    # Get eVal,eVec from a Hermitian matrix
    eVal,eVec=np.linalg.eigh(matrix)
    indices=eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal,eVec=eVal[indices],eVec[:,indices]
    eVal=np.diagflat(eVal)
    return eVal,eVec
#---------------------------------------------------
def fitKDE(obs, bWidth=.25, kernel='gaussian', x = None):
    # Fit kernel to a series of obs, and derive the prob of obs
    # x is the array of values on which the fit KDE will be evaluated
    if len(obs.shape) == 1:
        obs = obs.reshape(-1,1)
    kde = KernelDensity(kernel=kernel, bandwidth = bWidth).fit(obs)
    if x is None:
        x = np.unique(obs).reshape(-1,1)
    if len(x.shape) == 1:
        x = x.reshape(-1,1)
    logProb = kde.score_samples(x) # log(density)
    pdf = pd.Series(np.exp(logProb),index=x.flatten())
    return pdf
#---------------------------------------------------
x = np.random.normal(size=(10000,1000))
eVal0, eVec0 = getPCA(np.corrcoef(x,rowvar=0))
pdf0 = mpPDF(1.,q=x.shape[0]/float(x.shape[1]),pts=1000)
pdf1 = fitKDE(np.diag(eVal0),bWidth=.01) # empirical pdf

a = np.corrcoef(x,rowvar=0)
a = np.diag(eVal0)


def getRndCov(nCols,nFacts):
    w = np.random.normal(size=(nCols,nFacts))
    cov = np.dot(w,w.T) # random cov matrix, however not full rank
    cov += np.diag(np.random.uniform(size=nCols)) # full rank cov
    return cov
#---------------------------------------------------
def cov2corr(cov):
    # Derive the correlation matrix from a covariance matrix
    std = np.sqrt(np.diag(cov))
    corr = cov/np.outer(std,std)
    corr[corr<-1], corr[corr>1] = -1,1 # numerical error
    return corr
#---------------------------------------------------
alpha, nCols, nFact, q = 0.995, 1000, 100, 10
cov = np.cov(np.random.normal(size=(nCols*q, nCols)), rowvar=0)
cov = alpha*cov + (1-alpha)*getRndCov(nCols,nFact) # noise+signal
corr0 = cov2corr(cov)
eVal0, eVec0 = getPCA(corr0)

from scipy.optimize import minimize
#---------------------------------------------------
def errPDFs(var, eVal, q, bWidth, pts=1000):
    # Fit error
    print('Print errPDFs...')
    print(f'var: {0}, q: {1}, pts: {2}', var,q,pts)
    
    pdf0 = mpPDF(var,q,pts) # theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values) # empirical pdf
    sse = np.sum((pdf1-pdf0)**2)
    return sse
#---------------------------------------------------
def findMaxEval(eVal,q,bWidth):
    # Find max random eVal by fitting Marcenko’s dist
    out = minimize(lambda *x: errPDFs(*x), 0.5, args=(eVal,q,bWidth),bounds=((1E-5,1-1E-5),))
    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    eMax = var*(1+(1./q)**0.5)**2
    return eMax,var
#---------------------------------------------------
eMax0,var0 = findMaxEval(np.diag(eVal0),q,bWidth=.01)
nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
pdf0.index.values

def denoisedCorr(eVal,eVec,nFacts):
    # Remove noise from corr by fixing random eigenvalues
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:] = eVal_[nFacts:].sum()/float(eVal_.shape[0]-nFacts)
    eVal_ = np.diag(eVal_)
    corr1 = np.dot(eVec,eVal_).dot(eVec.T)
    corr1 = cov2corr(corr1)
    return corr1
#---------------------------------------------------
corr1=denoisedCorr(eVal0,eVec0,nFacts0)
eVal1,eVec1=getPCA(corr1)