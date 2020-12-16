#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:24:15 2020

@author: tai
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import stats
import model

#---------------------------
from pm4py.objects.log.importer.xes import factory as xes_import_factory
log = xes_import_factory.apply("running-example.xes")
from pm4py.algo.discovery.alpha import factory as alpha_miner
from pm4py.algo.discovery.inductive import factory as inductive_miner

from pm4py.evaluation.precision import factory as precision_factory

alpha_petri, alpha_initial_marking, alpha_final_marking = alpha_miner.apply(log)
inductive_petri, inductive_initial_marking, inductive_final_marking = inductive_miner.apply(log)

precision_alpha = precision_factory.apply(log, alpha_petri, alpha_initial_marking, alpha_final_marking)
precision_inductive = precision_factory.apply(log, inductive_petri, inductive_initial_marking, inductive_final_marking)

print("precision_alpha=",precision_alpha)
print("precision_inductive=",precision_inductive)


from pm4py.algo.discovery.dfg import factory as dfg_factory
dfg = dfg_factory.apply(log)
#-----------------------------
from collections import Counter 
dfg1 = Counter({('scroll','blur') : 1, 
               ('selection','blur') : 1,  
                ('click-0','scroll') : 1, 
               ('focus','selection') : 1,
               ('click-0','blur') : 1, 
               ('blur','focus') : 1,  
                ('scroll','click-0') : 1, 
               ('focus','blur') : 1,
               ('scroll','selection') : 1, 
               ('focus','scroll') : 1,  
                ('load','click-0') : 1, 
               ('load','scroll') : 1,
               ('load','selection') : 1, 
               ('blur','scroll') : 1,  
                ('selection','scroll') : 1,
                 ('focus','unload') : 1,
                 ('selection','unload') : 1
                
                }) 

#excellentLog1A = pd.read_csv('Excellent1A.csv')

'''
10	scroll-blur	0.056749
28	selection-blur	0.054673
32	click-0-scroll	0.053751
17	focus-selection	0.050912
34	click-0-blur	0.048980
21	blur-focus	0.048315
12	scroll-click-0	0.045465
16	focus-blur	0.044872
11	scroll-selection	0.039763
14	focus-scroll	0.039740
6	load-click-0	0.039533
2	load-scroll	0.034766
5	load-selection	0.033676
20	blur-scroll	0.033074
26	selection-scroll	0.032246
31	click-0-load	0.029780
'''
#-----------------

from pm4py.objects.log.importer.csv import factory as csv_importer
excellentLog1A = csv_importer.import_event_stream('Excellent1A_fixed.csv')
from pm4py.objects.conversion.log import factory as conversion_factory
log1 = conversion_factory.apply(excellentLog1A)

from pm4py.visualization.dfg import factory as dfg_vis_factory

gviz = dfg_vis_factory.apply(dfg1, log=log1, variant="frequency")
dfg_vis_factory.view(gviz)


from pm4py.objects.conversion.dfg import factory as dfg_mining_factory

net, im, fm = dfg_mining_factory.apply(dfg1)

from pm4py.visualization.petrinet import factory as pn_vis_factory

gviz = pn_vis_factory.apply(net, im, fm)
pn_vis_factory.view(gviz)

from pm4py.evaluation.replay_fitness import factory as replay_factory
fitness_alpha = replay_factory.apply(log1, net, im, fm)

from pm4py.algo.conformance.alignments import factory as align_factory

alignments = align_factory.apply(log1, net, im, fm)

print(alignments)


#excellentLog1A = excellentLog1A.sort_values(by=['org:resource','case','time:timestamp'])