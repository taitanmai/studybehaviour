import matplotlib
matplotlib.use('TkAgg')
from pylab import *
import networkx as nx

def initialize():
    global g, nextg
    g = nx.karate_club_graph()
    g.pos = nx.spring_layout(g)
    for i in g.nodes():
        g.nodes[i]['state'] = 1 if random() < 0.5 else 0
    nextg = g.copy()
    
def observe():
    global g, nextg
    cla()
    nx.draw(g, cmap = cm.binary, vmin=0, vmax=1, node_color = [g.nodes[i]['state'] for i in g.nodes()], pos = g.pos)
    
def update():
    global g, nextg
    for i in g.nodes():
        count = g.nodes[i]['state']
        for j in g.neighbors(j):
            count += g.node[j]['state']
        ratio = count/(g.degree(i) + 1.0)
        nextg.nodes[i]['state'] = 1 if ratio > .5 else 0 if ratio < .5 else 1 if random() < .5 else 0
    g,nextg = nextg,g

import pycxsimulator
pycxsimulator.GUI().start(func=[initialize,observe,update])
    