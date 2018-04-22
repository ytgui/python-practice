import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


G = nx.Graph()

G.add_edge('a', 'b', weight=1.6)
G.add_edge('a', 'c', weight=0.2)
G.add_edge('c', 'd', weight=0.1)
G.add_edge('c', 'e', weight=2.7)
G.add_edge('c', 'f', weight=0.9)
G.add_edge('a', 'd', weight=0.3)

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for (u, v) in G.edges()])
nx.draw_networkx_labels(G, pos)

plt.show()
