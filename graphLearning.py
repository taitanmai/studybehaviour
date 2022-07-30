import networkx as nx
from node2vec import Node2Vec
import pandas as pd
import numpy as np
from stellargraph import StellarGraph
from networkx.algorithms import community
import itertools
import mlfinlab as ml
from mlfinlab.networks.mst import MST
from mlfinlab.networks.dash_graph import DashGraph
from networkx import edge_betweenness_centrality as betweenness
from plotly.offline import plot
import plotly.graph_objects as go
from scipy.stats import f_oneway
from scipy import stats
import igraph 
from igraph import *

from itertools import chain, combinations
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def generateTransition(activityCodeList, selfLoop = 1):
    result = []
    for node1 in activityCodeList: 
        for node2 in activityCodeList: 
            if selfLoop == 0:
                if node1[0] == node2[0]:
                    continue
            result.append([(node1[0],node2[0]),node1[1] + '-' + node2[1],(node1[1],node2[1])])
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

def graphCreationForSingleStudent(transitionRow, activityCodeList, mode='networkx'):
    #transitionRow as series
    transitionList = generateTransition(activityCodeList)
    checkActivityList = []
    G = nx.Graph()
    for i in transitionList:
           if i[1] in transitionRow.index:
               if transitionRow[i[1]] > 0:
                   if i[0][0] not in checkActivityList:
                       G.add_node(i[0][0], weight=transitionRow[i[2][0]], name=i[2][0])
                       checkActivityList.append(i[0][0])
                   if i[0][1] not in checkActivityList:
                       G.add_node(i[0][1], weight=transitionRow[i[2][1]], name=i[2][1])
                       checkActivityList.append(i[0][1])
                   G.add_edge(i[0][0],i[0][1], weight = transitionRow[i[1]])            
    if mode == 'networkx':
        return G 
    else: 
        return StellarGraph.from_networkx(G)

def naiveGraphEmbeddingAllStudentsInAWeek(transitionDataMatrix_directFollow_week,activityCodeList,w):
    result = []
    dimensions = 128
    for i in transitionDataMatrix_directFollow_week.index:
        print(f'Week {w} - student: {i}')
        b = transitionDataMatrix_directFollow_week.loc[i,:]
        graph = graphCreationForSingleStudent(b,activityCodeList)
        if len(graph._node) > 0:
            # continue
            node2vec = Node2Vec(graph, dimensions=dimensions, walk_length=16, num_walks=10)
            model = node2vec.fit(window=10, min_count=1)
        
            node_embeddings = (
                model.wv.vectors
            )  # numpy.ndarray of size number of nodes times embeddings dimensionality    
            result.append(node_embeddings.sum(axis=0))
            
        else:
            result.append(np.zeros(128))
    return pd.DataFrame(result, index=transitionDataMatrix_directFollow_week.index)

def convertAdjToEdgeFeature(A):
    nodes = []
    for i in range(0,len(A)):
        nodes = []
        for j in range(0,len(A)):
            features = []
            for k in range(0,len(A)):            
                features.append([A[j][k]])
            nodes.append(features)
    return nodes

def constructGraphFeatureForAll(transitionDataMatrix_directFollow_week,activityCodeList,w, assessment):
    A = [] #adjency matrix list of all students
    X = [] #node feature of all students
    E = [] #edge feature of all students
    y = []
    for i in transitionDataMatrix_directFollow_week.index:
        if i in assessment.index:
            print(f'Week {w} - student: {i}')
            b = transitionDataMatrix_directFollow_week.loc[i,:]
            graph = graphCreationForSingleStudent(b,activityCodeList)
    
            # nodelist = [i for i in graph._node]
            nodelist = range(1,len(activityCodeList)+1)        
            #adjency matrix
            
            Astudent = nx.adjacency_matrix(graph, nodelist=nodelist, weight=None)
            Astudent = Astudent.todense()
            A.append(Astudent)
            
            #node features
            nodeFeatures = []
            attributes = nx.get_node_attributes(graph,'weight')
            for n in nodelist:
                if n in attributes:
                    nodeFeatures.append([attributes[n]])
                else:
                    nodeFeatures.append([0])
            X.append(nodeFeatures)
            
            #edge features
            Aw = nx.adjacency_matrix(graph, nodelist=nodelist)
            aw = Aw.todense()
            aw = np.asarray(aw)
            edgesFeature = convertAdjToEdgeFeature(aw)
            E.append(edgesFeature)
            
            #corressponding assessment result - target variable
            y.append([assessment.at[i,'result']])

    return np.array(A).astype(np.float64), np.array(X).astype(np.float64), np.array(E).astype(np.float64), np.array(y).astype(np.float64)
        
        
def mapNewLabel(df, reLabelIndex):
    newIndex = []
    for i in df.index:
        if i in reLabelIndex:
            newIndex.append(reLabelIndex[i])
        else:
            newIndex.append(-1)
    df.index = newIndex
    return df

def extractActiveStudentsOnActivity(activityDf, activityName, qualtile):
    threshold = activityDf.quantile(qualtile)[activityName]
    active = list(activityDf.loc[activityDf[activityName] > threshold].index)
    lessActive = list(activityDf.loc[activityDf[activityName] <= threshold].index)
    return [active, lessActive]



def visualiseMSTGraph(MSTgraph, studentCohorts, reLabelIndex):
    MSTgraph.set_node_groups(studentCohorts)
    # MSTgraph.graph = nx.relabel_nodes(MSTgraph.graph, reLabelIndex) #relabel nodes for easy visualisation
    dash_graph = DashGraph(MSTgraph)
    server = dash_graph.get_server()    
    # Run server
    server.run_server()
    
def most_central_edge(G):
    centrality = betweenness(G, weight="weight")
    return max(centrality, key=centrality.get)


#dendrogram
# building initial dict of node_id to each possible subset:
def createDendrogram(communities, excellent, weak, color_threshold=20):    
    node_id = 0
    init_node2community_dict = {node_id: communities[0][0].union(communities[0][1])}
    for comm in communities:
        for subset in list(comm):
            if subset not in init_node2community_dict.values():
                node_id += 1
                init_node2community_dict[node_id] = subset
    
    # turning this dictionary to the desired format in @mdml's answer
    node_id_to_children = {e: [] for e in init_node2community_dict.keys()}
    for node_id1, node_id2 in combinations(init_node2community_dict.keys(), 2):
        for node_id_parent, group in init_node2community_dict.items():
            if len(init_node2community_dict[node_id1].intersection(init_node2community_dict[node_id2])) == 0 and group == init_node2community_dict[node_id1].union(init_node2community_dict[node_id2]):
                node_id_to_children[node_id_parent].append(node_id1)
                node_id_to_children[node_id_parent].append(node_id2)
    
    # also recording node_labels dict for the correct label for dendrogram leaves
    node_labels = dict()
    for node_id, group in init_node2community_dict.items():
        if len(group) == 1:
            node_labels[node_id] = list(group)[0]
        else:
            node_labels[node_id] = ''
    
    # also needing a subset to rank dict to later know within all k-length merges which came first
    subset_rank_dict = dict()
    rank = 0
    for e in communities[::-1]:
        for p in list(e):
            if tuple(p) not in subset_rank_dict:
                subset_rank_dict[tuple(sorted(p))] = rank
                rank += 1
    subset_rank_dict[tuple(sorted(chain.from_iterable(communities[-1])))] = rank
    
    # my function to get a merge height so that it is unique (probably not that efficient)
    def get_merge_height(sub):
        sub_tuple = tuple(sorted([node_labels[i] for i in sub]))
        n = len(sub_tuple)
        other_same_len_merges = {k: v for k, v in subset_rank_dict.items() if len(k) == n}
        min_rank, max_rank = min(other_same_len_merges.values()), max(other_same_len_merges.values())
        range = (max_rank-min_rank) if max_rank > min_rank else 1
        return float(len(sub)) + 0.8 * (subset_rank_dict[sub_tuple] - min_rank) / range
    
    # finally using @mdml's magic, slightly modified:
    G           = nx.DiGraph(node_id_to_children)
    nodes       = G.nodes()
    leaves      = set( n for n in nodes if G.out_degree(n) == 0 )
    inner_nodes = [ n for n in nodes if G.out_degree(n) > 0 ]
    
    # Compute the size of each subtree
    subtree = dict( (n, [n]) for n in leaves )
    for u in inner_nodes:
        children = set()
        node_list = list(node_id_to_children[u])
        while len(node_list) > 0:
            v = node_list.pop(0)
            children.add( v )
            node_list += node_id_to_children[v]
        subtree[u] = sorted(children & leaves)
    
    inner_nodes.sort(key=lambda n: len(subtree[n])) # <-- order inner nodes ascending by subtree size, root is last
    
    # Construct the linkage matrix
    leaves = sorted(leaves)
    index  = dict( (tuple([n]), i) for i, n in enumerate(leaves) )
    Z = []
    k = len(leaves)
    for i, n in enumerate(inner_nodes):
        children = node_id_to_children[n]
        x = children[0]
        for y in children[1:]:
            z = tuple(sorted(subtree[x] + subtree[y]))
            i, j = index[tuple(sorted(subtree[x]))], index[tuple(sorted(subtree[y]))]
            Z.append([i, j, get_merge_height(subtree[n]), len(z)]) # <-- float is required by the dendrogram function
            index[z] = k
            subtree[z] = list(z)
            x = z
            k += 1
    
    # dendrogram
    plt.figure(figsize=(20,20))
    dendrogram(Z, labels=[node_labels[node_id] for node_id in leaves], leaf_rotation=0, orientation="left", color_threshold=color_threshold, above_threshold_color='grey')
    ax = plt.gca()
    xlbls = ax.get_ymajorticklabels()
    for lbl in xlbls:
        if lbl.get_text() in excellent:
            lbl.set_color('blue')
        else:
            lbl.set_color('red')
    plt.show()



#def community_detection_graph_louvain(MSTgraph):
def community_dection_graph(MSTgraph, num_comms = 20, mst=True):
    if mst:
        communities_generator = community.girvan_newman(MSTgraph.graph, most_valuable_edge=most_central_edge)
    else:
        communities_generator = community.girvan_newman(MSTgraph, most_valuable_edge=most_central_edge)
    result = []
    for communities in itertools.islice(communities_generator, num_comms):
        result.append(tuple(sorted(c) for c in communities))
    return result

def girvanNewManDendrogram(MSTgraph, excellent, weak, color_threshold = 20, num_comms = 20, mst=True):
    if mst:
        communities_generator = community.girvan_newman(MSTgraph.graph, most_valuable_edge=most_central_edge)
    else:
        communities_generator = community.girvan_newman(MSTgraph, most_valuable_edge=most_central_edge)
    createDendrogram(list(communities_generator), excellent, weak, color_threshold = color_threshold)


def convertGNcommunityToFlattenList(community, studentList):
    result = {}
    for s in studentList:
        for c in range(0, len(community)):
            if s in community[c]:
                result[s] = c
    return result 

def convertFlattenListToCommunity(communityDic):
    result = {}
    for c in communityDic:
        if communityDic[c] not in result:
            result[communityDic[c]] = []
        result[communityDic[c]].append(c)
    return result
        

def calculateExcellentRateInCommunity(community, excellentList, weakList):
    noOfExcellent = 0
    for s in community:        
        # noOfWeak = 0
        if s in excellentList:            
            noOfExcellent = noOfExcellent + 1
        # elif c in weakList:
        #     noOfWeak += 1

    return noOfExcellent/float(len(community))

def identifyCommunitiesType(communityList, excellentList, weakList):
    result = []
    for c in communityList:
        excellentRate = calculateExcellentRateInCommunity(c, excellentList, weakList)
        result.append([c,excellentRate])
    return pd.DataFrame(result, columns=['community','excellentRate'])

def identifyGroupOfaStudent(studentId, communityGroup, excellentList, weakList):
    communityGroupDf = identifyCommunitiesType(communityGroup, excellentList, weakList)
    for index in communityGroupDf.index:
        if studentId in communityGroupDf.loc[index]['community']:
            return communityGroupDf.loc[index]['excellentRate']
            
def identifyGroupOfaStudentOverAllCommunities(studentId, communityList, excellent, weak):
    row = []
    for c in communityList:
        exRateOfGroupOfStudent =  identifyGroupOfaStudent(studentId, c, excellent, weak)
        row.append(exRateOfGroupOfStudent)
    return row

def identifyGroupOfAllStudentsOverAllCommunitiesInWeek(studentList, communityList, excellent, weak):
    result = []
    for s in studentList:
        row = identifyGroupOfaStudentOverAllCommunities(s, communityList, excellent, weak)
        result.append(row)
    columns = list(range(2,len(communityList)+2))
    return pd.DataFrame(result, index = studentList, columns = columns)

def getStudentMovingAtLevelOfCommunityDetection(studentList, classifyStudentGroupOverCommunitiesWeeks, numberOfCommunitiesGet):
    result = []
    for s in studentList:
        row = []
        for w in range(0, len(classifyStudentGroupOverCommunitiesWeeks)):
            if s in classifyStudentGroupOverCommunitiesWeeks[w].index:
                row.append(classifyStudentGroupOverCommunitiesWeeks[w].loc[s,[numberOfCommunitiesGet]][numberOfCommunitiesGet])
            else:
                row.append(-1)
        result.append(row)
    return pd.DataFrame(result, index=studentList)

def labelCommunity(studentList, communityList, noOfCommunities, excellent, weak, df):
    communityList1 = [communityList[noOfCommunities-2]]
    temp = identifyGroupOfAllStudentsOverAllCommunitiesInWeek(df.columns,communityList1, excellent, weak)
    temp1 = temp.copy()
    temp1 = temp1.to_numpy()
    temp1 = np.where(temp1>= 0.7, 1, np.where(temp1<= 0.3, 0, 2))
    temp1 = pd.DataFrame(temp1, index = temp.index, columns = temp.columns)
    return temp1

def sankeyDataGenerator(groupMovingData, fromWeek, toWeek, beginPoint=[], endPoint=[]):
    source = []
    target = []
    value = []
    color_link = []
    excellentFlag = 3
    mixedFlag = 2
    weakFlag = 1

    for w in range(fromWeek, toWeek):
        groupMovingTemp = groupMovingData.loc[:,[w,w+1]]
        groupMovingTemp['count'] = 0
        temp = groupMovingTemp.groupby([w,w+1]).count()
        for i in temp.index:
            if (i[0] < 0) or (i[1] < 0):
                continue
            else:
                if i[0] == 1:
                    source.append(weakFlag)
                    color_link.append('rgba(241, 100, 39, 0.35)')
                    if i[1] == 1:
                        target.append(weakFlag + 3) 
                    elif i[1] == 2:
                        target.append(mixedFlag + 3)
                    else:
                        target.append(excellentFlag + 3)
                elif i[0] == 2:
                    source.append(mixedFlag)
                    color_link.append('rgba(241, 211, 39, 0.35)')
                    if i[1] == 1:
                        target.append(weakFlag + 3) 
                    elif i[1] == 2:
                        target.append(mixedFlag + 3)
                    else:
                        target.append(excellentFlag + 3)
                else:
                    source.append(excellentFlag)
                    color_link.append('rgba(39, 241, 46, 0.35)')
                    if i[1] == 1:
                        target.append(weakFlag + 3) 
                    elif i[1] == 2:
                        target.append(mixedFlag + 3)
                    else:
                        target.append(excellentFlag + 3)
                value.append(temp.loc[i]['count'])
        excellentFlag = excellentFlag + 3
        mixedFlag = mixedFlag + 3
        weakFlag = weakFlag + 3
    
    #assignlabel
    maxGroup = max(target)
    label = []
    color_node = []
    for u in range(0,maxGroup+1):
        if u%3 == 0:
            label.append("High active - W" + str(int(u/3)))
            color_node.append('#32CD32')
        elif u%3 == 1:
            label.append("Low active - W" + str(int(u//3+1)))
            color_node.append('#EC7063')
        else:
            label.append("Mid active - W" + str(int(u//3+1)))
            color_node.append('#F7DC6F')
            
    if len(endPoint) > 0:
        lastGroupMoving =  groupMovingData.loc[:,[w+1]]
        lastGroupMoving['result'] = ''
        lastGroupMoving.loc[lastGroupMoving.index.isin(endPoint[0]), ['result']] = 1
        lastGroupMoving.loc[lastGroupMoving.index.isin(endPoint[1]), ['result']] = 0
        lastGroupMoving['count'] = 0
        temp = lastGroupMoving.groupby([w+1,'result']).count()
        for i in temp.index:
            if (i[0] < 0) or (i[1] == ''):
                continue
            else:
                if i[0] == 1:
                    source.append(weakFlag)
                    color_link.append('rgba(241, 100, 39, 0.35)')
                    if i[1] == 0:
                        target.append(weakFlag + 3)
                    else:
                        target.append(mixedFlag + 3)                
                elif i[0] == 2:
                    source.append(mixedFlag)
                    color_link.append('rgba(241, 211, 39, 0.35)')
                    if i[1] == 0:
                        target.append(weakFlag + 3) 
                    else:
                        target.append(mixedFlag + 3)
                else:
                    source.append(excellentFlag)
                    color_link.append('rgba(39, 241, 46, 0.35)')
                    if i[1] == 0:
                        target.append(weakFlag + 3)
                    else:
                        target.append(mixedFlag + 3)
                value.append(temp.loc[i]['count'])
        
        for u in range(maxGroup+1,max(target)+1):
            if u%3 == 0:
                continue
            elif u%3 == 1:
                label.append("Failed" + str(int(u//3+1)))
                color_node.append('#EC7063')
            else:
                label.append("Passed" + str(int(u//3+1)))
                color_node.append('#F7DC6F')

        # label.append("Excellent - W" + str(int(u/3)))
    
    return [source, target, value, label, color_node, color_link]

def sankeyVisualise(sankeyData):
    source = sankeyData[0]   
    target = sankeyData[1]
    value =  sankeyData[2]
    label = sankeyData[3]
    color_node = sankeyData[4]
    color_link = sankeyData[5]
    link = dict(source = source, target = target, value = value, color=color_link)
    node = dict(label = label, pad=50, thickness=5, color=color_node)
    data = go.Sankey(link = link, node=node)
    # plot
    fig = go.Figure(data)
    fig.show()
    plot(fig)
    
def createGraphFromCorrDistance(matrix):
    G = nx.Graph()
    for s in matrix.columns:
        G.add_node(s)
    
    checkCouple = []
    for s in matrix.columns:
        for i in matrix.index:
            if (i,s) not in checkCouple:
                G.add_edge(s,i, weight = matrix.loc[i,s])
                checkCouple.append((i,s))
    
    return G
    
def extractAssessmentResultOfCommunities(community, assessment, column):
    result = []
    for cSize in community:
        extractedResult = []
        groups = []
        # normTest = []
        for c in cSize:
            temp = assessment.loc[assessment.index.isin(c)]
            extractedResult.append((temp[column].mean(),temp[column].std()))
            # if len(cSize) == 8:
                # k2, p = stats.normaltest(temp[column])
                # normTest.append((k2, p))
            groups.append(temp[column])
        if len(groups) == 8:
            f,p = f_oneway(groups[0], groups[1],  groups[2] ,  groups[3] ,  groups[4] , groups[5],  groups[6] ,  groups[7] )# ,  groups[8],  groups[9])
                           # ,groups[10], groups[11],  groups[12],  groups[13],  groups[14] , groups[15],  groups[16],  groups[17],  groups[18],  groups[19])
            L, pL = stats.levene(groups[0], groups[1],  groups[2] ,  groups[3] ,  groups[4]  , groups[5],  groups[6] ,  groups[7] )# ,  groups[8],  groups[9])
                           #,groups[10], groups[11],  groups[12],  groups[13],  groups[14] , groups[15],  groups[16],  groups[17],  groups[18],  groups[19])
            fk, pk = stats.kruskal(groups[0], groups[1],  groups[2] ,  groups[3] ,  groups[4] , groups[5],  groups[6] ,  groups[7] )#  ,  groups[8],  groups[9])
                           #,groups[10], groups[11],  groups[12],  groups[13],  groups[14] , groups[15],  groups[16],  groups[17],  groups[18],  groups[19])
            result.append([len(cSize), extractedResult, (f,p), (L,pL), (fk,pk), groups])
        else:
            result.append([len(cSize), extractedResult, groups])
    return result

def getIntersectionTwoGroup(group1, group2):
    return list(set(group1).intersection(set(group2))) 

def findTogetherMembers(communitiesList1, communitiesList2, meanScoreGroup1, meanScoreGroup2 ): #communitiesList1 as pandas series
    result = []
    for key1, c1 in zip(range(0,len(communitiesList1)),communitiesList1):
        tmp1 = []
        for key2, c2 in zip(range(0,len(communitiesList1)),communitiesList2):
            tmp2 = getIntersectionTwoGroup(c1.index, c2.index)
            commonMemberDetail = []
            for t in tmp2:
                commonMemberDetail.append((t,c1[t],c2[t], meanScoreGroup1[key1][0], meanScoreGroup2[key2][0]))
            commonMemberDetail = pd.DataFrame(commonMemberDetail, columns=['student','assessment2A','assessment3A','groupPoint2A','groupPoint3A']).set_index('student')
            tmp1.append(commonMemberDetail)            
        result.append(tmp1)
    return pd.DataFrame(result)
    
def createGraphFromCounter(dfg, mode = 'networkx'):
    transitionList = list(dfg)
        #transitionRow as series
    checkActivityList = []
    G = nx.DiGraph()
    for i in transitionList:
        # if i[0] != i[1]:
            if i[0] not in checkActivityList:
                G.add_node(i[0], name=i[0])
                checkActivityList.append(i[0])
            if i[1] not in checkActivityList:
                G.add_node(i[1], name=i[1])
                checkActivityList.append(i[1])
            G.add_edge(i[0],i[1], weight = 1)            
    if mode == 'networkx':
        return G 
    else: 
        return StellarGraph.from_networkx(G)
    

def findCenterNodeInOneCommunity(graph, community):
    maxDegree = graph.degree[community[0]]
    maxNode = community[0]
    for i in range(1,len(community)):
        if graph.degree[community[i]] > maxDegree:
            maxDegree = graph.degree[community[i]]
            maxNode = community[i]
    return (maxNode, maxDegree)

def findAllCenterNodeinCommunities(communities, graph):
    result = []
    for c in communities:
        find = findCenterNodeInOneCommunity(graph, c)
        result.append((c, find[0], find[1], float(find[1])/len(c)))
    return result
    
def getAllWeights(graph):
    result = {}
    nodes = graph._node
    for n in nodes:
        for m in nodes:
            if (m,n) not in result:
                tmp = graph.get_edge_data(n,m,0)
                if tmp != 0:
                    result[(n,m)] = tmp['weight']
    return result

def checkIfSameCommunity(node1, node2, community):
    for c in community:
        if (node1 in c) and (node2 in c):
            return True
    return False

def superGraphGeneration(clustersOverWeekList = [], weekWeights = [1,1,1,1,1,1,1,1,1,1,1,1], nodeList = []):
    result = np.zeros((len(nodeList),len(nodeList)))
    bonusWeight = np.zeros((len(nodeList),len(nodeList)))
    for i in range(0, len(nodeList)):
        for j in range(0, len(nodeList)):
            bonusWeight = 0
            for k in  range(0, len(clustersOverWeekList)):
               if checkIfSameCommunity(nodeList[i],nodeList[j],clustersOverWeekList[k]):
                   result[i][j] = result[i][j] + 1*weekWeights[k] + bonusWeight
                   bonusWeight = bonusWeight + 1
               else:
                   bonusWeight = 0
    return pd.DataFrame(result, index = nodeList, columns = nodeList)
                   
                   
def checkIfUniqueEdges(graph):
    edges1 = list(graph.edges(data=True))

    flag = 0
    for e1 in range(0, len(edges1)):
        for e2 in range(e1+1, len(edges1)):
            if (edges1[e1][2]['weight'] != 0) and (edges1[e1][2]['weight'] == edges1[e2][2]['weight']) :
                print(edges1[e1])
                print(edges1[e2])
                flag = 1
    if flag == 0:
        print('No duplicate edges')
    print(len(edges1))
 
    
#work with iGraph
def createGraphFromCorrDistance_iGraph(matrix):
    G = Graph()
    labels = []
    for i in range(0, len(matrix.columns)):
        G.add_vertices(1)
        G.vs[i]['id'] = i
        G.vs[i]['label'] = matrix.columns[i]
    
    checkCouple = []
    weights = []
    for s in range(0, len(matrix.columns)):
        for i in range(0, len(matrix.index)):
            if (i,s) not in checkCouple:
                G.add_edges([(s,i)])
                weights.append(matrix.loc[matrix.index[i],matrix.columns[s]])
                checkCouple.append((i,s))
    
    G.es['weight'] = weights
    G.es['label'] = weights
    return G    
    
    

    
    
    

