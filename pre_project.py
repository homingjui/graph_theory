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
    return float(100*min(n)+max(n))

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

###############################################parm
n_orders = 4

##############################################parm-end
# G = nx.Graph()
# for i in range(1,n_orders*2+1):
#     G.add_node(i)
#
# for i in range(1,n_orders*2):
#     G.add_edge(i,i+1)
# G.add_edge(1,n_orders*2)
#
#
# G.add_edge(1,5)
# G.add_edge(2,6)
# G.add_edge(3,7)
# G.add_edge(4,8)

#to list
# edge_list=[ list(i) for i in list(G.edges)]
edge_list_array=[[[1, 2], [1, 5], [1, 8], [2, 3], [2, 6], [3, 4], [3, 7], [4, 5], [4, 8], [5, 6], [6, 7], [7, 8]],
                                    [[1, 2], [1, 5], [1, 8], [2, 3], [2, 6], [3, 4], [3, 7], [4, 5], [1, 6], [2, 5], [3, 8], [4, 7]]]
edge_list=sort(edge_list)
edge_list_X = [X_cal(i) for i in edge_list]
print(edge_list_array)

'''
permutation(G.number_of_nodes())
iterations = math.factorial(G.number_of_nodes())
print("iterations: %d"%iterations)
result = []
result_z_w_i = []

######################################find all combination
for now_iteration in range(iterations):
    new_edge_list = copy.deepcopy(edge_list)
    for i in range(len(new_edge_list)):
        new_edge_list[i][0]=n_array[now_iteration][new_edge_list[i][0]-1]+1
        new_edge_list[i][1]=n_array[now_iteration][new_edge_list[i][1]-1]+1
    y=1
    w=1
    for i in range(len(new_edge_list)):
        i_x_cal = X_cal(new_edge_list[i])
        for j in range(i+1,len(new_edge_list)):

            # y *= float(X_cal(new_edge_list[i])-X_cal(new_edge_list[j]))/(X_cal(edge_list[i])-X_cal(edge_list[j]))
            y *= (i_x_cal-X_cal(new_edge_list[j]))/(edge_list_X[i]-edge_list_X[j])
        w *= i_x_cal/edge_list_X[i]
    # print (y,w,now_iteration)
    result_z_w_i.append((y,(abs(y+1)+abs(w-1)),now_iteration))
    result.append(sort(new_edge_list))

# check_n = 30336
# print result[check_n]
# print result_z_w_i[check_n]
print("len result,result_z_w_i: %d %d"%(len(result),len(result_z_w_i)))

result = np.array(result)
result_z_w_i = np.array(result_z_w_i)
for i in range(len(result_z_w_i)):
    if i ==  len(result_z_w_i):
        break
    same_index = np.where(np.all(result == result[i], axis=(1,2)))[0]
    # print same_index
    result_z_w_i = np.delete(result_z_w_i,same_index[1:],0)
    result = np.delete(result,same_index[1:],0)

print("filter len result,result_z_w_i: %d %d"%(len(result),len(result_z_w_i)))
print np.where(result_z_w_i[:,1]<0.0001)[0]


##########################################save file
path = 'output.txt'
record_file = open(path, 'w')
write_string="A1: "
for i in edge_list:
     write_string += str(i)+", "
record_file.write(write_string[:-2])

for i in range(len(result_z_w_i)):
    write_string="\n{}, \tf=".format(i)
    for k in range(len(n_array[0])):
        write_string+=str(k+1)+":"+str(n_array[int(result_z_w_i[i,2])][k]+1)+", "
    write_string+="Z={:+.20f},\n".format(result_z_w_i[i,0])
    for j in list(result[i]):
        write_string += str(j)+", "
    record_file.write(write_string[:-2])
record_file.write("\n\n")
record_file.close()
'''
#########################################show graph
# print("node",G.number_of_nodes())
# print("edges",G.number_of_edges())
# print(G.nodes)
# print(G.edges)
# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()
