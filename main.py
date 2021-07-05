import csv
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_node(1)
G.add_nodes_from([2, 3])
G.add_nodes_from([
    (4, {"color": "B"}),
    (5, {"color": "green"}),
])

H = nx.path_graph(5)
G.add_nodes_from(H)

G.add_edge(1, 2)
# e = (2, 3)
# G.add_edge(*e)
G.add_edges_from([(1, 3), (1, 4)])
G.add_edges_from(H.edges)
# G.clear()
# G.add_edges_from([(1, 2), (1, 3)])
# G.add_node(1)
# G.add_edge(1, 2)
# G.add_node("spam")        # adds node "spam"
# G.add_nodes_from("spam")  # adds 4 nodes: 's', 'p', 'a', 'm'
# G.add_edge(3, 'm')

print(G.number_of_nodes())

print(G.number_of_edges())

# list(G.nodes)

# list(G.edges)

# list(G.adj[1])  # or list(G.neighbors(1))

# G.degree[1]  # the number of edges incident to 1

# G.edges([2, 'm'])

# G.degree([2, 3])


nx.draw(G, with_labels=True, font_weight='bold')
plt.show()

# class node():
# 	def __init__(self,name, side, length):
# 		self.name = name
# 		self.side = side
# 		self.length = length
# 	def show(self):
# 		print("node",self.name)
# 		print("link",self.side)
# 		print("length",self.length)
# 		print("")
# 	def save(self):
# 		return self.name,self.side,self.length

# grapth = []
# grapth.append(node(1,[2,5],[3,7]))
# grapth.append(node(2,[1,5,3],[2,4,8]))
# grapth.append(node(3,[2,4],[2,1]))
# grapth.append(node(4,[3,5,6],[4,8,5]))
# grapth.append(node(5,[1,2,4],[5,4,8]))
# grapth.append(node(6,[4],[1]))

# for  i in grapth:
# 	i.show()

# with open('output.csv', 'w') as csvfile:
#   writer = csv.writer(csvfile)
#   for i in grapth:
# 	  writer.writerow(i.save())

# print("read grapth")
# grapth2 = []
# with open('output.csv') as csvfile:

#   rows = csv.reader(csvfile)

#   for row in rows:
#     grapth2.append(node(row[0],row[1],row[2]))

# for  i in grapth2:
# 	i.show()
