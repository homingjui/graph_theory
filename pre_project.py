import csv
import math
import copy
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd
from IPython.display import display

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def perm(n,begin,end):
    global n_array,side_array
    if begin>=end:
        side = False
        for i in range(0,len(n)-1,2):
            if n[i]+1==n[i+1] or n[i]-1==n[i+1]:
                side=True
                break
            if (n[i]==0 and n[i+1]==len(n)-1) or (n[i+1]==0 and n[i]==len(n)-1):
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
#end def#####################################
def permutation(n):
    global n_array,side_array
    n_array = []
    side_array = []
    n = [i for i in range(n)]
    perm(n,0,len(n))
#end def#####################################
def X_cal(n):
    return float(100*min(n)+max(n))
#end def#####################################
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
#end def#####################################
def do_all_G(edge_list_array,G_result,npresult,num_record):
    print "***do permutation!***"
    iterations = len(n_array)
    edge_num = len(edge_list_array[0])
    ############################################### do all G
    for now_G in range(len(edge_list_array)):
        result = []
        result_uni = []
        if len(G_result)>0:
            same_flag = False
            for  i in  npresult:
                if len(np.where(np.all(i == edge_list_array[now_G], axis=(1,2)))[0])>0:
                    same_flag = True
                    break
            if same_flag:
                continue
            # if len(np.where(np.all(npresult == edge_list_array[now_G], axis=(1,2)))[0])>0:
            #     continue
        ######################################find all combination
        x_array = np.array([X_cal(i) for i in edge_list_array[now_G]])
        # start = time.time()
        for now_iteration in range(iterations):
            new_edge_list = copy.deepcopy(edge_list_array[now_G])
            for i in range(edge_num):
                new_edge_list[i][0]=n_array[now_iteration][new_edge_list[i][0]-1]+1
                new_edge_list[i][1]=n_array[now_iteration][new_edge_list[i][1]-1]+1
            result_uni.append(sort(new_edge_list))
            result.append(new_edge_list)
        # print time.time()-start
        # start = time.time()
        # result = np.array(result)
        # print np.where(np.all(result == result[0], axis=(1,2)))
    ##########################################filter
        for i in range(len(result_uni)):
            result_uni[i]=sort(result_uni[i])
        result_uni=np.array(result_uni)
        # result_uni,result_n =np.unique(result_uni,axis=0,return_index=True)
    ##########################################save file
        write_string=""
        #######################################################################not filted
        n_flag = False
        for n_gaph in range(len(result_uni)):
            if n_flag and not savefile:
                break
            ###############################get z,w
            new_edge_list_x = np.array([X_cal(j) for j in result[n_gaph]])
            z=1
            for i in range(edge_num):
                z *= np.prod(new_edge_list_x[i+1:]*(-1)+new_edge_list_x[i])/np.prod(x_array[i+1:]*(-1)+x_array[i])
            w =np.prod(new_edge_list_x)/np.prod(x_array)
            if abs(z+1)+abs(w-1)< 0.0001  and sort(result[n_gaph])==sort(result[0]):
                n_flag = True
            ###############################write record
            if savefile:
                write_string+="\n\n"+str(n_gaph)+", f="
                for i in range(len(n_array[0])):
                    write_string+=str(i+1)+":"+str(n_array[n_gaph][i]+1)+",  "
                write_string += "\n"
                write_string += str(result_uni[n_gaph].tolist())
                    # print n_gaph
                write_string += "\tZ="+str(z)+", W="+str((abs(z+1)+abs(w-1)))
        record_file = open(path, 'a')
        path_arr = 'output/'+G_name
        if n_flag:
            path_arr+="N"
            path_arr+=str(num_record.count(-1))
        else:
            path_arr+=str(num_record.count(1))
        path_arr+='.txt'
        write_head=G_name
        if n_flag:
            write_head+="N"
            write_head+="("+str(num_record.count(-1))+"):"
        else:
            write_head+="("+str(num_record.count(1))+"):"
        write_head += str(edge_list_array[now_G].tolist())
        # for i in edge_list_array[now_G]:
        #      write_head += str(i)+", "
        record_file.write(write_head)
        record_file.write("\n")
        record_file.close()
        if savefile:
            record_arr_file = open(path_arr, 'w')
            record_arr_file.write(write_head)
            record_arr_file.write(write_string)
            record_arr_file.close()
        ############################################write num_record
        if n_flag:
            num_record.append(-1)
        else:
            num_record.append(1)
        ###########################################result
        result_uni = np.unique(result_uni,axis=0)
        if n_flag:
            print ""+G_name+"N"+"("+str(num_record.count(-1))+")  ",
        else:
            print ""+G_name+"("+str(num_record.count(1))+")  ",
        print("num f(G) %d"%(len(result_uni)))
        if len(G_result)==0:
            G_result= np.array([edge_list_array[now_G]])
            # npresult = np.array(result_uni)
            npresult.append(result_uni)
        else:
            # npresult= np.vstack((npresult,result_uni))
            npresult.append(result_uni)
            G_result= np.vstack((G_result,[edge_list_array[now_G]]))
        # print "all f(G) num: %d"%np.shape(npresult)[0],
        # print (", different f(G) ->%d"%np.shape(np.unique(npresult,axis=0))[0])
        # print time.time()-start
    return G_result,npresult,num_record
#end def#####################################
def edge_switch():
    all_new_g = []
    for now_switch_G in range(num_record[0],len(G_result)):
        # print "\nfor this graph",str(G_result[now_switch_G].tolist())
        for switch_n in range(len(G_result[now_switch_G])):
            edges_on_node =  np.unique(G_result[now_switch_G,np.where((G_result[now_switch_G]==G_result[now_switch_G,switch_n,0])  |
                                                                                                                                                  (G_result[now_switch_G]==G_result[now_switch_G,switch_n,1]))[0]],axis=0)
            # if len(np.unique(edges_on_node))!=6:
            #     continue
            del_n = np.where((edges_on_node==G_result[now_switch_G,switch_n]).all(axis=1))[0]
            edges_on_node = np.delete(edges_on_node, del_n,axis=0)
            edges_on_node_a = edges_on_node[np.where(edges_on_node==G_result[now_switch_G,switch_n,0])[0]]
            edges_on_node_b = edges_on_node[np.where(edges_on_node==G_result[now_switch_G,switch_n,1])[0]]

            for change_a in range(len(edges_on_node_a)):
                for change_b in range(len(edges_on_node_b)):
                    # print G_result[now_switch_G,switch_n]
                    new_g = np.array(G_result[now_switch_G])
                    change_index = np.where((new_g[:,0] == edges_on_node_a[change_a,0]) & (new_g[:,1]==edges_on_node_a[change_a,1]))[0][0]
                    new_g[change_index][new_g[change_index]==G_result[now_switch_G,switch_n,0]] = G_result[now_switch_G,switch_n,1]
                    change_index = np.where((new_g[:,0] == edges_on_node_b[change_b,0]) & (new_g[:,1]==edges_on_node_b[change_b,1]))[0][0]
                    new_g[change_index][new_g[change_index]==G_result[now_switch_G,switch_n,1]] = G_result[now_switch_G,switch_n,0]
                    same_flag=False
                    for i in new_g:
                        if (len(new_g[(new_g[:,0]==i[0]) & (new_g[:,1]==i[1])])+
                            len(new_g[(new_g[:,0]==i[1]) & (new_g[:,1]==i[0])]))>1:
                            same_flag=True
                            # break
                    if not same_flag:
                        same_flag = False
                        for  i in  npresult:
                            if len(np.where(np.all(i == sort(new_g), axis=(1,2)))[0]) > 0:
                                same_flag = True
                                break
                        if not same_flag:
                            # print G_name+"("+str(now_switch_G-num_record[-1][1])+"): (u, v) ",
                            # print G_result[now_switch_G,switch_n],
                            # print " switch edges:",
                            # print edges_on_node_a[change_a],
                            # print edges_on_node_b[change_b],
                            # print "-> new graph!"
                            all_new_g.append(sort(new_g))
                        # else:
                        #     print G_name+"("+str(now_switch_G-num_record[-1][1])+"): (u, v) ",
                        #     print "-> already exists"
    return all_new_g
#end def#####################################

########################################################################parm
nodes = 8
path = 'output.txt'
record_file = open(path, 'w')
record_file.close()
savefile = False
##############################################parm-end
print "%d nodes, "%nodes,
print("iterations: %d"%math.factorial(nodes))
#############################################find all G
circle = []
for i in range(1,nodes):
    circle.append([i,i+1])
circle.append([1,nodes])
circle = np.array([circle])
permutation(nodes)
n_array = np.array(n_array)
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

# edge_list_array=edge_list_array[:1]

###################################################do all G
find_done = False
all_npresult = []   ###########for all order result
all_G_result = []  ###########for all order G result
all_num_record = []  ########for all order N
all_remove_g = []
print(np.shape(edge_list_array))
G_result = np.array([])   ###########for all G
npresult = []   ###########for all result filter
num_record = [0]
G_name="A"
while not find_done:
    ########################################permutation
    G_result,npresult,num_record=do_all_G(edge_list_array,G_result,npresult,num_record)
    print str(np.shape(G_result)[0]-num_record[0])+" new G"
    # #######################################edge switching
    print "\n***do edge switch***"
    all_new_g=edge_switch()
    num_record[0]=np.shape(G_result)[0]
    print "find "+str(len(all_new_g))+" new graph to check\n"
    if len(all_new_g)==0:
        find_done=True
    edge_list_array = np.array(all_new_g)


print "### "+str(len(G_result))+" G ###"
# print "and "+str(len(npresult))+" f(G)###"
all_npresult.append(npresult)
all_G_result.append(G_result)
all_num_record.append(num_record)
##########################################remove edge
for n_order in range(1,nodes):
    print "\n\n***remove edge***"
    remove_g_all =[]
    for remove_g_n in range(len(G_result)):
        for remove_g_edge in range(len(G_result[remove_g_n])):
            remove_g =np.array(G_result[remove_g_n])
            remove_g_or=np.array(remove_g)
            remove_g[remove_g_edge,1]=remove_g[remove_g_edge,0]
            remove_g[remove_g==G_result[remove_g_n,remove_g_edge,1]] = G_result[remove_g_n,remove_g_edge,0]
            remove_g[remove_g>G_result[remove_g_n,remove_g_edge,1]] -= 1
            same_flag=False
            for j in remove_g:
                if j[0] != j[1]:
                    if (len(remove_g[(remove_g[:,0]==j[0]) & (remove_g[:,1]==j[1])])+
                        len(remove_g[(remove_g[:,0]==j[1]) & (remove_g[:,1]==j[0])]))>1:
                        same_flag=True
                        break
            if not same_flag:
                # print G_result[remove_g_n,i]
                remove_g_x = np.array([X_cal(i) for i in remove_g])
                remove_g_or_x = np.array([X_cal(i) for i in remove_g_or])
                h=1
                for i in range(len(remove_g)):
                    h*= (np.prod(remove_g_x[remove_g_edge+1:]*(-1)+remove_g_x[remove_g_edge])
                            /np.prod(remove_g_or_x[remove_g_edge+1:]*(-1)+remove_g_or_x[remove_g_edge]))
                # print h
                remove_g = np.delete(remove_g,remove_g_edge,axis=0)
                remove_g_all.append([remove_g_n,remove_g_edge,sort(remove_g),None])
            else :
                remove_g_all.append([remove_g_n,remove_g_edge,None,remove_g.tolist()])
    if len([i[2] for i in remove_g_all if i[2]])==0:
        print "find "+str(len(remove_g_all))+" 2n-"+str(n_order)+" G \nall done!"
        break
    print len(remove_g_all)
    all_remove_g.append(remove_g_all)
    # edge_list_array=np.unique(remove_g_all,axis=0)
    print "find "+str(len(edge_list_array))+" 2n-"+str(n_order)+" G"
    ###################################################do all G
    find_done = False
    print "%d nodes "%(nodes-n_order),
    print("iterations: %d"%math.factorial(nodes-n_order))
    G_name=chr(ord(G_name) + 1)
    permutation(nodes-n_order)
    n_array = np.array(n_array)
    G_result = np.array([])   ###########for all G
    npresult = []   ###########for all result filter
    num_record = [0]
    ########################################permutation
    G_result,npresult,num_record=do_all_G(np.array([i[2] for i in remove_g_all if i[2]]),G_result,npresult,num_record)
    print "### "+str(len(G_result))+" 2n-"+str(n_order)+" G,",
    all_npresult.append(npresult)
    all_G_result.append(G_result)
    all_num_record.append(num_record)

print ""
print all_num_record
for i in all_npresult:
    print len(i),
    print [len(j) for j in i]


table =  [[[] for i in range(len(all_num_record[0])-1)] for i in range(len(all_npresult[1])+1)]
for removed_g in all_remove_g[0]:
    # print removed_g
    if removed_g[2] != None:
        for f_g in range(len(all_npresult[1])):
            # print np.where(np.all(all_npresult[1][f_g] == removed_g[2], axis=(1,2)))[0]
            if len(np.where(np.all(all_npresult[1][f_g] == removed_g[2], axis=(1,2)))[0])>0:
                table[f_g][removed_g[0]].append(removed_g[1])
    else :
        table[len(all_npresult[1])][removed_g[0]].append(removed_g[1])
        # print "n"


columns=[]
N = 0
for i in range(1,len(all_num_record[0])):
    if all_num_record[0][i] == 1:
        columns.append("A"+str(i-N))
    else:
        N +=1
        columns.append("AN"+str(N))

index=[]
N = 0
for i in range(1,len(all_num_record[1])):
    if all_num_record[1][i] == 1:
        index.append("B"+str(i-N))
    else:
        N +=1
        index.append("BN"+str(N))
index.append("0")

df =pd.DataFrame(table,index=index,columns=columns)

print df

















#########################################show graph
# print("node",G.number_of_nodes())
# print("edges",G.number_of_edges())
# print(G.nodes)
# print(G.edges)
# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()
