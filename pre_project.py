import csv
import math
import copy
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


def perm(n,begin,end):
    global n_array,side_array
    if begin>=end:
        side = False
        for i in range(0,nodes-1,2):
            if n[i]+1==n[i+1] or n[i]-1==n[i+1]:
                side=True
                break
            if (n[i]==0 and n[i+1]==nodes-1) or (n[i+1]==0 and n[i]==nodes-1):
                side=True
                break
        if not side:
            side_array.append(n[:])
        n_array.append(n[:])
        # n_array[-1].append(0)
    else:
        i=begin
        for num in range(begin,end):

          n[num],n[i]=n[i],n[num]
          perm(n,begin+1,end)
          n[num],n[i]=n[i],n[num]

def permutation(n):
    global n_array,side_array
    n_array = []
    side_array = []
    n = [i for i in range(n)]
    perm(n,0,len(n))

def X_cal(n):
    return float(100*min(n)+max(n))


def sort(n):
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
    return x

###############################################parm
nodes = 8
path = 'output.txt'
record_file = open(path, 'w')
record_file.close()
##############################################parm-end
iterations = math.factorial(nodes)
print("iterations: %d"%iterations)
#############################find all G
circle = []
for i in range(1,nodes):
    circle.append([i,i+1])
circle.append([1,nodes])
circle = np.array([circle])

permutation(nodes)
# print len(n_array)
# print side_array

side_array = np.array(side_array)+1
side_array = np.reshape(side_array,(np.shape(side_array)[0],-1,2))
for i in range(len(side_array)):
    side_array[i]=sort(side_array[i])

side_array =np.unique(side_array,axis=0)
# print np.shape(side_array)

circle = np.repeat(circle,np.shape(side_array)[0],axis=0)
edge_list_array = np.hstack((circle,side_array))
# print edge_list_array
for i in range(len(edge_list_array)):
    edge_list_array[i]=sort(edge_list_array[i])

# edge_list_array=[[[1, 2], [1, 5], [1, 8], [2, 3], [2, 6], [3, 4], [3, 7], [4, 5], [4, 8], [5, 6], [6, 7], [7, 8]],
#                                     [[1, 2], [1, 5], [1, 8], [2, 3], [2, 6], [3, 4], [3, 7], [4, 5], [1, 6], [2, 5], [3, 8], [4, 7]]]

# edge_list=sort(edge_list)


print(np.shape(edge_list_array))
# print(edge_list_X_array)
npresult = np.array([])
G_num = 0
################################################################ do all G
for now_G in range(len(edge_list_array)):
    result = []
    result_z_w = []
    if now_G>0:
        if len(np.where(np.all(npresult == edge_list_array[now_G], axis=(1,2)))[0])>0:
            # print ("%d's G is equivalence"%now_G)
            continue
    print ("\nG(%d)"%G_num)
    G_num += 1
    ######################################find all combination
    x_array = np.array([X_cal(i) for i in edge_list_array[now_G]])

    for now_iteration in range(iterations):
        new_edge_list = copy.deepcopy(edge_list_array[now_G])
        # print new_edge_list
        # print n_array[now_iteration]
        #
        # for i in range(len(new_edge_list)):
        #     new_edge_list[i][0]=n_array[now_iteration][new_edge_list[i][0]-1]+1
        #     new_edge_list[i][1]=n_array[now_iteration][new_edge_list[i][1]-1]+1
        # print new_edge_list

        for i in range(nodes):
            new_edge_list[new_edge_list==i+1]=(n_array[now_iteration][i]+1)*10
        new_edge_list/=10
        new_edge_list_x = np.array([X_cal(i) for i in new_edge_list])
        # print new_edge_list
        # input()
        z=1
        w=1
        # for i in range(len(new_edge_list-1)):
        #     z *= np.prod(new_edge_list_x[i+1:]*(-1)+new_edge_list_x[i])/np.prod(x_array[i+1:]*(-1)+x_array[i])
        w=np.prod(new_edge_list_x)/np.prod(x_array)

        # for i in range(len(new_edge_list)):
        #     i_x_cal = X_cal(new_edge_list[i])
        #     for j in range(i+1,len(new_edge_list)):
        #         z *= (new_edge_list_x[i]-new_edge_list_x[j])/(x_array[i]-x_array[j])
        #     w *=  X_cal(new_edge_list[i])/x_array[i]
        result_z_w.append((z,(abs(z+1)+abs(w-1))))

        result.append(sort(new_edge_list))

    # result = np.array(result)
    # print np.where(np.all(result == result[0], axis=(1,2)))
##########################################filter
    result,result_n =np.unique(result,axis=0,return_index=True)
    # print("len result: %d"%(len(result)))
##########################################save file
    record_file = open(path, 'a')
    path_arr = 'output/AA'+str(G_num)+'.txt'
    record_arr_file = open(path_arr, 'w')
    write_string="A"+str(G_num)+": "
    for i in edge_list_array[now_G]:
         write_string += str(i)+", "
    record_arr_file.write(write_string[:-2])
    record_file.write(write_string[:-2])
    record_file.write("\n")
    record_file.close()

    for n_gaph in range(len(result)):
        write_string="\n\n{}, \tf=".format(n_gaph)
        for i in range(len(n_array[0])):
            write_string+=str(i+1)+":"+str(n_array[result_n[n_gaph]][i]+1)+",  "
        write_string += "\n"
        for i in result[n_gaph]:
            write_string += str(i)+", "
        record_arr_file.write(write_string[:-2])
        write_string = "\tZ="+str(result_z_w[result_n[n_gaph]][0])+", W="+str(result_z_w[result_n[n_gaph]][1])
        record_arr_file.write(write_string)

    record_arr_file.close()
    ###########################################result
    print("filted result num: %d"%(len(result)))

    if now_G==0:
        npresult = np.array(result)
    else:
        npresult= np.vstack((npresult,result))
    print ("all f(G) num: %d"%np.shape(npresult)[0])
    # print np.shape(npresult)

npresult=np.unique(npresult,axis=0)
print ("all f(G) num: %d"%np.shape(npresult)[0])





#########################################show graph
# print("node",G.number_of_nodes())
# print("edges",G.number_of_edges())
# print(G.nodes)
# print(G.edges)
# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()
