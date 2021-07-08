import csv
import math
import copy
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def perm(n,begin,end):
    global n_array
    if begin>=end:
        n_array.append(n[:])
        # n_array[-1].append(0)
    else:
        i=begin
        for num in range(begin,end):
          n[num],n[i]=n[i],n[num]
          perm(n,begin+1,end)
          n[num],n[i]=n[i],n[num]

def permutation(n):
    global n_array
    n_array = []
    n = [i for i in range(n)]
    perm(n,0,len(n))

def X_cal(n):
    return 100*min(n)+max(n)

sort_t = 0

def sort(n):
    global sort_t
    try:
        x = n.tolist()
    except:
        x = copy.deepcopy(n)
    flag = True
    for i in range(len(x)):
        if x[i][0]>x[i][1]:
            x[i][1],x[i][0] = x[i][0],x[i][1]
    while flag :
        # print("shorting")
        flag = False
        for i in range(len(x)-1):
            if not((x[i][0]<x[i+1][0]) or ((x[i][0]==x[i+1][0])and(x[i][1]<x[i+1][1]))):
                # print(x[i],x[i+1])
                x[i],x[i+1]=x[i+1],x[i]
                # print(x[i],x[i+1])
                flag = True
                sort_t += 1
    return x

G = nx.Graph()
for i in range(1,6):
    G.add_node(i)
G.add_edge(1,2)
G.add_edge(1,5)
G.add_edge(1,6)
G.add_edge(2,4)
G.add_edge(2,5)
G.add_edge(3,4)
G.add_edge(3,5)
G.add_edge(3,6)
G.add_edge(4,6)

# G.add_edge(6,2)
# G.add_edge(1,5)
# G.add_edge(1,6)
# G.add_edge(2,4)
# G.add_edge(2,5)
# G.add_edge(3,4)
# G.add_edge(3,5)
# G.add_edge(3,6)
# G.add_edge(4,1)


edge_list=[ list(i) for i in list(G.edges)]
# print(edge_list)

edge_list=sort(edge_list)
edge_list_X = [X_cal(i) for i in edge_list]
print(edge_list)
# print(edge_list_X)

permutation(G.number_of_nodes())
iterations = math.factorial(G.number_of_nodes())
print("iterations: %d"%iterations)

result = []
result_z = []

for now_iteration in range(iterations):
    new_edge_list = copy.deepcopy(edge_list)
    for i in range(len(new_edge_list)):
        new_edge_list[i][0]=n_array[now_iteration][new_edge_list[i][0]-1]+1
        new_edge_list[i][1]=n_array[now_iteration][new_edge_list[i][1]-1]+1
    y=1
    for i in range(len(new_edge_list)):
        for j in range(i+1,len(new_edge_list)):
            # y *= float(X_cal(new_edge_list[i])-X_cal(new_edge_list[j]))/(X_cal(edge_list[i])-X_cal(edge_list[j]))
            y *= float(X_cal(new_edge_list[i])-X_cal(new_edge_list[j]))/(edge_list_X[i]-edge_list_X[j])
    result_z.append(round(y,5))
    result.append(sort(new_edge_list))


###############################filter by edages
result = np.array(result)
result_z = np.array(result_z)
# result, indices = np.unique(result, axis=0, return_index=True)
# result_z = result_z[indices]

for i in range(len(result)):
    if i ==  len(result_z):
        break
    same_index = np.where(np.all(result == result[i], axis=(1,2)))[0]
    # print same_index
    result_z = np.delete(result_z,same_index[1:])
    result = np.delete(result,same_index[1:],0)
##############################  filter by Z
# for i in range(len(result)):
#     if i ==  len(result_z):
#         break
#     same_index = np.where(result_z==result_z[i])[0]
#     result_z = np.delete(result_z,same_index[1:])
#     result = np.delete(result,same_index[1:],0)
#     same_index = np.where(result_z==(-result_z[i]))[0]
#     result_z = np.delete(result_z,same_index)
#     result = np.delete(result,same_index,0)

print("filter result: %d"%len(result))
# print("filter result: %d"%len(result_z))
with open("test.txt", "wb") as file:
     pickle.dump(result.tolist(), file)

# for i in range(len(result)):
#     print "%d:"%(i+1),
#     # for j in  range(len(n_array[i])):
#         # print "f(%d)=%d"%(j+1,n_array[i][j]+1),
#     # print ""
#     for j in  result[i]:
#         print "[%d, %d]"%(j[0],j[1]),
#     print result_z[i],
#     print "\n"

#######################################edge switching
# all_new_g =[]
# check_g = 0
# print "for this graph",result[check_g].tolist()
# # print result[check_g].tolist()
# for i in range(len(result[check_g])):
#     edges_on_node =  np.unique(result[check_g,np.where((result[check_g]==result[check_g,i,0])  | (result[check_g]==result[check_g,i,1]))[0]],axis=0)
#     if len(np.unique(edges_on_node))!=6:
#         # print "[%d %d] 5 point edge"%(result[check_g,i,0],result[check_g,i,1])
#         continue
#     del_n = np.where((edges_on_node==result[check_g,i]).all(axis=1))[0]
#     edges_on_node = np.delete(edges_on_node, del_n,axis=0)
#     edges_on_node_a = edges_on_node[np.where(edges_on_node==result[check_g,i,0])[0]]
#     edges_on_node_b = edges_on_node[np.where(edges_on_node==result[check_g,i,1])[0]]
#
#     print "(u, v) ",
#     print result[check_g,i]
#     for j in range(len(edges_on_node_b)):
#         # print result[check_g,i]
#         # print edges_on_node_a[0],
#         # print edges_on_node_b[j]
#         new_g = np.array(result[check_g])
#
#         change_index = np.where((new_g[:,0] == edges_on_node_a[0,0]) & (new_g[:,1]==edges_on_node_a[0,1]))[0][0]
#         new_g[change_index][new_g[change_index]==result[check_g,i,0]] = result[check_g,i,1]
#
#         change_index = np.where((new_g[:,0] == edges_on_node_b[j,0]) & (new_g[:,1]==edges_on_node_b[j,1]))[0][0]
#         new_g[change_index][new_g[change_index]==result[check_g,i,1]] = result[check_g,i,0]
#         # new_g = sort(new_g)
#
#         print "switch edges:",
#         # print edges_on_node_a[0],
#         # print edges_on_node_b[j],
#         if len(np.where(np.all(result == sort(new_g), axis=(1,2)))[0]) == 0:
#             # print "-> new graph!"
#             # print sort(new_g)
#             all_new_g.append(sort(new_g))
#         # else:
#         #     print "-> already exists"
#         # print new_g.tolist()
#     # print ""
#
# print all_new_g

###########################find the same
# check = [[[1, 4], [1, 5], [1, 6], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6]], [[1, 3], [1, 5], [1, 6], [2, 3], [2, 5], [2, 6], [3, 4], [4, 5], [4, 6]], [[1, 3], [1, 4], [1, 6], [2, 3], [2, 4], [2, 6], [3, 5], [4, 5], [5, 6]], [[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5], [3, 6], [4, 6], [5, 6]], [[1, 2], [1, 5], [1, 6], [2, 3], [2, 4], [3, 5], [3, 6], [4, 5], [4, 6]], [[1, 2], [1, 4], [1, 6], [2, 3], [2, 5], [3, 4], [3, 6], [4, 5], [5, 6]], [[1, 2], [1, 4], [1, 5], [2, 3], [2, 6], [3, 4], [3, 5], [4, 6], [5, 6]], [[1, 2], [1, 3], [1, 6], [2, 4], [2, 5], [3, 4], [3, 5], [4, 6], [5, 6]], [[1, 2], [1, 3], [1, 5], [2, 4], [2, 6], [3, 4], [3, 6], [4, 5], [5, 6]], [[1, 2], [1, 3], [1, 4], [2, 5], [2, 6], [3, 5], [3, 6], [4, 5], [4, 6]]]
#
# for i in check:
#     print np.where(np.all(result == i, axis=(1,2)))[0]




# print("")
# print(G.nodes)
# print(G.edges)
print("node",G.number_of_nodes())
print("edges",G.number_of_edges())

# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()
