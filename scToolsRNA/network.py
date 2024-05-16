
import numpy as np
import scanpy as sc
import igraph as ig



# GEPHI IMPORT & EXPORT


def export_to_graphml(adata, filename='test.graphml', directed=None):    

    adjacency = adata.obsp['connectivities']

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shap[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    if g.vcount() != adjacency.shape[0]:
        logg.warn('The constructed graph has only {} nodes. '
                  'Your adjacency matrix contained redundant nodes.'
                  .format(g.vcount()))
    g.write_graphml(filename)


def load_pajek_xy(adata, filename='test.net'):
    
    # first determine the number of graph nodes in *.net file
    with open(filename,'r') as file:
        nNodes = 0
        for ln,line in enumerate(file):
            if line.startswith("*Edges"):
                nNodes = ln-1

    # extract xy coordinates from *.net file
    with open(filename,'r') as file:
        lines=file.readlines()[1:nNodes+1] 
        xy = np.empty((nNodes,2))
        for ln,line in enumerate(lines):
            xy[ln,0]=(float(line.split(' ')[2]))
            xy[ln,1]=(float(line.split(' ')[3]))

    # generate ForceAtlas2 data structures and update coordinates
    sc.tl.draw_graph(adata, layout='fa', iterations=1)
    adata.obsm['X_draw_graph_fa']=xy

    return adata


