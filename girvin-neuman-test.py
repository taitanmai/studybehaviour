import snap

G1 = snap.TNGraph.New()
G1.AddNode(1)
G1.AddNode(5)
G1.AddNode(32)
G1.AddEdge(1,1)
G1.AddEdge(1,5)
G1.AddEdge(5,1)
G1.AddEdge(5,32)
for EI in G1.Edges():
    print("edge (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId()))
    
# get first eigenvector of graph adjacency matrix
EigV = snap.TFltV()
snap.GetEigVec(G1, EigV)
