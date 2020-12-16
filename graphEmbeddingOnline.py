import networkx as nx
from node2vec import Node2Vec
import pandas as pd
import numpy as np

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

def generateActivityCodeList():
    originalElements = []
    pageList = ['Read_Lecture_Note','Exercise','Read_Labsheet','Check_solution','Admin_page']
    actionList = ['load','scroll','blur','focus','unload','upload','hashchange','selection']
    for p in pageList:
        for a in actionList:
            originalElements.append(p+'*'+a)
    return assignNodeNumber(pageList)

def graphCreationForSingleStudent(transitionRow, activityCodeList):
    #transitionRow as series
    transitionList = generateTransition(activityCodeList)
    checkActivityList = []
    G = nx.Graph()
    for i in transitionList:
           if i[1] in transitionRow.index:
               if transitionRow[i[1]] > 0:
                   if i[0][0] not in checkActivityList:
                       G.add_node(i[0][0])
                       checkActivityList.append(i[0][0])
                   if i[0][1] not in checkActivityList:
                       G.add_node(i[0][1])
                       checkActivityList.append(i[0][1])
                   G.add_edge(i[0][0],i[0][1], weight = transitionRow[i[1]])            
    return G 

def naiveGraphEmbeddingAllStudentsInAWeek(transitionDataMatrix_directFollow_week,activityCodeList,w):
    result = []
    dimensions = 64
    for i in transitionDataMatrix_directFollow_week.index:
        print(f'Week {w} - student: {i}')
        b = transitionDataMatrix_directFollow_week.loc[i,:]
        graph = graphCreationForSingleStudent(b,activityCodeList)
        if len(graph._node) > 0:
            # continue
            node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=10, num_walks=100)
            model = node2vec.fit(window=10, min_count=1)
        
            node_embeddings = (
                model.wv.vectors
            )  # numpy.ndarray of size number of nodes times embeddings dimensionality    
            result.append(node_embeddings.sum(axis=0))
            
        else:
            result.append(np.zeros(64))
    return pd.DataFrame(result, index=transitionDataMatrix_directFollow_week.index)
                
transitionDataMatrixWeeks_directFollow = []
for w in range(0,12):
    a = pd.read_csv('transitionMatrixStorage/transitionDataMatrixWeeks_direct_follow_accumulated_time_w' + str(w) + '.csv', index_col = 0)
    transitionDataMatrixWeeks_directFollow.append(a)

activityCodeList = generateActivityCodeList()

studentActivityEmbedding = []
for w in range(0,12):
    print(f'Week {w}')
    studentActivityEmbedding.append(naiveGraphEmbeddingAllStudentsInAWeek(
                                        transitionDataMatrixWeeks_directFollow[w], activityCodeList,w))

for w in range(0,12):
    studentActivityEmbedding[w].to_csv('embeddingNode2Vec/GraphEmbeddings_accumulated_sum_time_w_' + str(w) + '.csv')

  