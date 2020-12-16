import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import stats
import math
import seaborn as sns
import model_conceptDriftLib as m_cD

activityVariance = pd.read_csv('ActivityVarianceFixedLog.csv')

activityVarianceResult = pd.read_csv('newActivityVarianceWithResult.csv')
activityVarianceResult = activityVarianceResult.loc[:, ~activityVarianceResult.columns.str.contains('^Unnamed')]

activityVariance = activityVariance.loc[activityVariance['endDate'] <= '2018-12-12']

activityList = ['load','scroll','focus','blur','unload','hashchange','click-0','selection','click-2','click-1','click-3']


activityVarianceResult = activityVarianceResult.sort_values(['startDate'])
activityVarianceResult = activityVarianceResult.reset_index()
activityVarianceResult = activityVarianceResult.drop(['index'],axis=1)

extractRE = m_cD.extractRE(activityVarianceResult,100,activityList)
extractRE1 = extractRE




pValDf = pd.DataFrame(pValList['scroll'], columns=['popSize','startDatePop1', 'endDatePop1', 'startDatePop2','endDatePop2','subLogIndex1','subLogIndex2','p-value'])
pValDf['p-value'] = pValDf['p-value'].astype(float)

pValDfFocus = pd.DataFrame(pValList['focus'], columns=['popSize','startDatePop1', 'endDatePop1', 'startDatePop2','endDatePop2','subLogIndex1','subLogIndex2','p-value'])
pValDfFocus['p-value'] = pValDfFocus['p-value'].astype(float)

plt.plot(pValDf['subLogIndex1'], pValDf['p-value'], '-')
plt.title('Scroll')
plt.show()

plt.plot(pValDfFocus['subLogIndex1'], pValDfFocus['p-value'], '-')
plt.title('Focus')
plt.show()


selectedPvalue = pValDfAverage.loc[pValDfAverage['p-value'] <= 0.4]

pValList = m_cD.KSTestForManyActivity(10, activityList,extractRE1)
pValListAverage = m_cD.KSTestAveragePValForManyActivity(pValList)

pValDfAverage = pd.DataFrame(pValListAverage['average'], columns=['popSize','startDatePop1', 'endDatePop1', 'startDatePop2','endDatePop2','subLogIndex1','subLogIndex2','p-value'])
pValDfAverage['p-value'] = pValDfAverage['p-value'].astype(float)

plt.plot(pValDfAverage['subLogIndex1'], pValDfAverage['p-value'], '-')
plt.title('Average')
plt.show()

activityVarianceResult['perPassed'].describe()

#extract excellent and weak cohort

Excellent = activityVarianceResult.loc[activityVarianceResult['perPassed']>= 0.811594]
Excellent = Excellent.reset_index()
Excellent = Excellent.drop(['index'],axis=1)

Weak = activityVarianceResult.loc[activityVarianceResult['perPassed']<= 0.531250]
Weak = Weak.reset_index()
Weak = Weak.drop(['index'],axis=1)

#extract RE values for each cohort
extractRE_excellent = m_cD.extractRE(Excellent,,activityList)
extractRE1_excellent = extractRE_excellent

extractRE_weak = m_cD.extractRE(Weak,100,activityList)
extractRE1_weak = extractRE_weak

#KS Test, population size= 10
pValList_excellent = m_cD.KSTestForManyActivity(50, activityList,extractRE1_excellent)
pValListAverage_excellent = m_cD.KSTestAveragePValForManyActivity(pValList_excellent)

pValList_weak = m_cD.KSTestForManyActivity(50, activityList,extractRE1_weak)
pValListAverage_weak = m_cD.KSTestAveragePValForManyActivity(pValList_weak)

plt.plot(pValListAverage_excellent['average'][:,5].astype(int), pValListAverage_excellent['average'][:,7].astype(float), 'r-', pValListAverage_weak['average'][:,5].astype(int), pValListAverage_weak['average'][:,7].astype(float), 'b-')
plt.show()

plt.plot(pValListAverage_excellent['average'][:,1].astype(datetime), pValListAverage_excellent['average'][:,7].astype(float), 'r-', pValListAverage_weak['average'][:,1].astype(datetime), pValListAverage_weak['average'][:,7].astype(float), 'b-')
plt.show()
