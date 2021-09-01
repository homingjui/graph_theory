####python3.7 -m pip install pympler
####sudo ln -s /usr/bin/python3 /usr/bin/python
####https://zhung.com.tw/article/install-python3-8-pip-for-ubuntu-linux/
import csv
from math import prod,factorial
import copy
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import pandas as pd
from IPython.display import display
from itertools import permutations
from sys import getsizeof
from multiprocessing import Pool

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# def perm(n,begin,end):
#     global n_array,side_array,now_permutation
#     if begin>=end:
#         side = False
#         for i in range(0,len(n)-1,2):
#             if n[i]+1==n[i+1] or n[i]-1==n[i+1]:
#                 side=True
#                 break
#             if (n[i]==0 and n[i+1]==len(n)-1) or (n[i+1]==0 and n[i]==len(n)-1):
#                 side=True
#                 break
#         if not side:
#             side_array[now_permutation]=n
#         else:
#             side_array[now_permutation]=0
#         n_array[now_permutation]=n
#         now_permutation+=1
#     else:
#         for num in range(begin,end):
#           n[num],n[begin]=n[begin],n[num]
#           perm(n,begin+1,end)
#           n[num],n[begin]=n[begin],n[num]
# #end def#####################################
# def permutation(num):
#     global n_array,side_array,now_permutation
#     n = np.arange(num,dtype=np.uint8)
#     n_array = np.repeat([n],factorial(num),axis=0)
#     side_array = np.repeat([n],factorial(num),axis=0)
#     now_permutation=0
#     perm(n,0,len(n))
def grouped(iterable):
    a = iter(iterable)
    return zip(a, a)
#end def#####################################
def X_cal(x):
    return_list=[]
    for n in grouped(x):
        if n[0]<n[1]:
            return_list.append(float(100*n[0]+n[1]))
        elif n[0]>n[1]:
            return_list.append(float(100*n[1]+n[0]))
        elif n[0]==n[1]:
            return_list.append(0)
    return return_list
#end def#####################################
def sort(n):
    print(n)
    for i in grouped(n):
        print(i)
        if i[0]>i[1]:
            i[1],i[0] = i[0],i[1]
    return sorted(x)
#end def#####################################
def r1(iterations):
    return list(map(lambda i: [iterations[i[0]-1]+1,iterations[i[1]-1]+1] , now_G_do))
def do_all_G(nodes_n,edge_list_array,G_result,num_record,npresult,npresult_z,npresult_n):
    global now_G_do
    print ("***do permutation!***")
    # iterations = len(n_array)
    edge_num = len(edge_list_array[0])/2
    ############################################### do all G
    for now_G in range(len(edge_list_array)):
        write_string=""
        if len(G_result)>0:
            same_flag = False
            for  i in  npresult:
                if len(np.where(np.all(i == edge_list_array[now_G], axis=(1,2)))[0])>0:
                    same_flag = True
                    break
            if same_flag:
                continue
        ######################################find all combination
        x_array = np.array(X_cal(edge_list_array[now_G]))
        x_prod = [prod(item-x_array[index+1:]) for index,item in enumerate(x_array)]
        now_G_do = edge_list_array[now_G]
    ##########################################save file
        n_flag = False
        z_result_uni = []
        result=[]
        start = time.time()
        show=0.001
        for n_gaph in map( r1, permutations(range(nodes_n))):
            # print(n_gaph)
            if len(z_result_uni)>=int(factorial(nodes)*show):
                print(str(int(show*100)+1)+"%  time:"+str(time.time()-start), end = '\r')
                show+=0.001
            ###############################get z,w
            result.append(sum(n_gaph, []))
            new_edge_list_x = list(map(X_cal ,n_gaph))
            z = prod([prod(map(lambda n: new_edge_list_x[i]-n,new_edge_list_x[i+1:]))/x_prod[i] for i in range(edge_num-1)])
            z_result_uni.append(z/abs(z))
            if n_flag==False:
                new_edge_list_x=prod(new_edge_list_x)
                w =new_edge_list_x/x_array
                if abs(z+1)+abs(w-1)< 0.0001:
                    if (new_edge_list_x==x_array):
                        n_flag = True
        print("")
        record_file = open(path, 'a')
        path_arr = 'output/'+G_name
        if n_flag:
            path_arr+="N"
            path_arr+=str(num_record.count(-1)+1)
        else:
            path_arr+=str(num_record.count(1)+1)
        path_arr+='.txt'
        write_head=G_name
        if n_flag:
            write_head+="N"
            write_head+="("+str(num_record.count(-1)+1)+"):"
        else:
            write_head+="("+str(num_record.count(1)+1)+"):"
        write_head += str(edge_list_array[now_G])
        record_file.write(write_head)
        record_file.write("\n")
        record_file.close()
        ############################################write num_record
        if n_flag:
            num_record.append(-1)
        else:
            num_record.append(1)
        ###########################################result
        result,result_n =np.unique(list(map(sort,result)),axis=0,return_index=True)####################################filter!!
        print (time.time()-start)
        if n_flag:
            print (""+G_name+"N"+"("+str(num_record.count(-1))+")  ",end="")
        else:
            print (""+G_name+"("+str(num_record.count(1))+")  ",end="")
        print("num f(G) %d"%(len(result)))
        if len(G_result)==0:
            G_result= np.array([edge_list_array[now_G]])
        else:
            G_result= np.vstack((G_result,[edge_list_array[now_G]]))
        npresult_z.append(z_result_uni)
        npresult_n.append(result_n)
        npresult.append(result)
    return G_result,npresult,num_record,npresult_z,npresult_n
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
                            all_new_g.append(sort(new_g))
    return all_new_g
#end def#####################################

########################################################################parm
nodes =8
path = 'output.txt'
record_file = open(path, 'w')
record_file.close()
savefile = False
##############################################parm-end
total_time = time.time()
print("%d nodes, "%nodes,end="")
print("iterations: %d"%factorial(nodes))
#############################################find all G
circle = []
for i in range(1,nodes):
    circle.append([i,i+1])
circle.append([1,nodes])
for n in permutations(range(nodes)):
    side = False
    for i in range(0,len(n)-1,2):
        if n[i]+1==n[i+1] or n[i]-1==n[i+1]:
            side = True
            break
        if (n[i]==0 and n[i+1]==len(n)-1) or (n[i+1]==0 and n[i]==len(n)-1):
            side = True
            break
    if not side:
        side_arr = n
        break

side_array = np.array(side_arr)+1
edge_list_array=[[*sum(circle,[]),*side_array]]
print (edge_list_array)
# edge_list_array=[[[1, 2], [1, 3], [1, 8], [2, 3], [2, 4], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8]]]
# edge_list_array=[[[1, 2], [1, 3], [1, 10], [2, 3], [2, 4], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7], [6, 9], [7, 8], [8, 9], [8, 10], [9, 10]]]

###################################################do all G
find_done = False
all_npresult = []   ###########for all order result
all_npresult_z = []   ###########order result z
all_npresult_n = []   ###########order result pos
all_G_result = []  ###########for all order G result
all_num_record = []  ########for all order N
all_remove_g = []
print(np.shape(edge_list_array))
G_result = np.array([])   ###########for all G
num_record = [0]
G_name="A"

npresult = []   ###########for all result filter
npresult_z = []   ###########esult z
npresult_n = []   ###########f result filter pos
while not find_done:
    ########################################permutation
    G_result,npresult,num_record,npresult_z,npresult_n=do_all_G(nodes,edge_list_array,G_result,num_record,npresult,npresult_z,npresult_n)
    print (str(np.shape(G_result)[0]-num_record[0])+" new G")
    # #######################################edge switching
    print ("\n***do edge switch***")
    all_new_g=edge_switch()
    num_record[0]=np.shape(G_result)[0]
    print ("find "+str(len(all_new_g))+" new graph to check\n")
    if len(all_new_g)==0:
        find_done=True
    edge_list_array = np.array(all_new_g)
    print (time.time()-total_time)


print ("### "+str(len(G_result))+" G ###")
# print "and "+str(len(npresult))+" f(G)###"
all_npresult.append(npresult)
all_G_result.append(G_result)
all_num_record.append(num_record)
all_npresult_z.append(npresult_z)
all_npresult_n.append(npresult_n)
##########################################remove edge
for n_order in range(1,nodes):
    print ("\n\n***remove edge***")
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
                    h*= prod(remove_g_x[i+1:]*(-1)+remove_g_x[i])/prod(remove_g_or_x[i+1:]*(-1)+remove_g_or_x[i])
                remove_g = np.delete(remove_g,remove_g_edge,axis=0)
                remove_g_all.append([remove_g_n,remove_g_edge,sort(remove_g),int(h/abs(h))])
            else :
                remove_g_all.append([remove_g_n,remove_g_edge,None,remove_g.tolist()])
    if len([i[2] for i in remove_g_all if i[2]])==0:
        print ("find "+str(len(remove_g_all))+" 2n-"+str(n_order)+" G \nall done!")
        break
    print (len(remove_g_all))
    all_remove_g.append(remove_g_all)
    print ("find "+str(len(remove_g_all))+" 2n-"+str(n_order)+" G")
    ###################################################do all G
    find_done = False
    print ("%d nodes "%(nodes-n_order),end="")
    print("iterations: %d"%factorial(nodes-n_order))
    G_name=chr(ord(G_name)+1)
    # permutation(nodes-n_order)
    G_result = np.array([])   ###########for all G
    num_record = [0]
    npresult = []   ###########for all result filter
    npresult_z = []   ###########esult z
    npresult_n = []   ###########f result filter pos
    ########################################permutation
    # G_result,npresult,num_record,npresult_z,npresult_n=do_all_G(edge_list_array,G_result,num_record,npresult,npresult_z,npresult_n)
    G_result,npresult,num_record,npresult_z,npresult_n=do_all_G(nodes-n_order,np.array([i[2] for i in remove_g_all if i[2]]),
                                                                G_result,num_record,npresult,npresult_z,npresult_n)
    print ("### "+str(len(G_result))+" 2n-"+str(n_order)+" G###")
    all_npresult.append(npresult)
    all_G_result.append(G_result)
    all_num_record.append(num_record)
    all_npresult_z.append(npresult_z)
    all_npresult_n.append(npresult_n)
    print (time.time()-total_time)

print ("")
print (all_num_record)
for i in range(len(all_npresult)):
    print (len(all_G_result[i]),len(all_npresult[i]),len(all_npresult_z[i]),len(all_npresult_n[i]))
    print ([len(j) for j in all_npresult[i]])
    print ([len(j) for j in all_npresult_n[i]])
    print ([len(j) for j in all_npresult_z[i]])



for i in all_remove_g:
    print (len(i),end="")
    print (" ",end="")
print ("")

G_name="A"
for order in range(len(all_num_record)-1):
    table =  [[[] for i in range(len(all_num_record[order])-1)] for i in range(len(all_npresult[order+1])+1)]
    for removed_g in all_remove_g[order]:
        # print removed_g
        if removed_g[2] != None:
            for f_g in range(len(all_npresult[order+1])):
                finded =  np.where(np.all(all_npresult[order+1][f_g] == removed_g[2], axis=(1,2)))[0]
                if len(finded)>0:
                    pos= all_npresult_n[order+1][f_g][finded[0]]
                    table[f_g][removed_g[0]].append([removed_g[1]+1,removed_g[3],all_npresult_z[order+1][f_g][pos]])
        else :
            table[len(all_npresult[order+1])][removed_g[0]].append(removed_g[1]+1)

    columns=[]
    N = 0
    for i in range(1,len(all_num_record[order])):
        if all_num_record[order][i] == 1:
            columns.append(chr(ord(G_name)+order)+str(i-N))
        else:
            N +=1
            columns.append(chr(ord(G_name)+order)+"N"+str(N))
    index=[]
    N = 0
    for i in range(1,len(all_num_record[order+1])):
        if all_num_record[order+1][i] == 1:
            index.append(chr(ord(G_name)+order+1)+str(i-N))
        else:
            N +=1
            index.append(chr(ord(G_name)+order+1)+"N"+str(N))
    index.append("0")
    sort_indx = []
    row_table = []
    for g_num in range(len(all_num_record[order+1])-1):
        if all_num_record[order+1][g_num+1] == 1:
            sort_indx.append(index[g_num])
            row_table.append(table[g_num])
    for g_num in range(len(all_num_record[order+1])-1):
        if all_num_record[order+1][g_num+1] == -1:
            sort_indx.append(index[g_num])
            row_table.append(table[g_num])
    sort_indx.append(index[-1])
    row_table.append(table[-1])
    N = 0
    sort_columns = copy.deepcopy(columns)
    sort_table =  [[[] for i in range(len(all_num_record[order])-1)] for i in range(len(all_npresult[order+1])+1)]
    for g_num in range(len(all_num_record[order])-1):
        if all_num_record[order][g_num+1] == 1:
            for i in range(len(sort_table)):
                sort_table[i][N]=row_table[i][g_num]
            sort_columns[N]=columns[g_num]
            N+=1
    for g_num in range(len(all_num_record[order])-1):
        if all_num_record[order][g_num+1] == -1:
            for i in range(len(sort_table)):
                sort_table[i][N]=row_table[i][g_num]
            sort_columns[N]=columns[g_num]
            N+=1
    df =pd.DataFrame(sort_table,index=sort_indx,columns=sort_columns)
    df.to_csv('csv/table'+sort_columns[0][0]+'.csv')



    table =  [[[] for i in range(len(all_num_record[order])-1)] for i in range(len(all_npresult[order+1])+1)]
    for removed_g in all_remove_g[order]:
        # print removed_g
        if removed_g[2] != None:
            for f_g in range(len(all_npresult[order+1])):
                finded =  np.where(np.all(all_npresult[order+1][f_g] == removed_g[2], axis=(1,2)))[0]
                if len(finded)>0:
                    pos= all_npresult_n[order+1][f_g][finded[0]]
                    table[f_g][removed_g[0]].append(int(removed_g[3]*all_npresult_z[order+1][f_g][pos]))
        else :
            table[len(all_npresult[order+1])][removed_g[0]].append(removed_g[1]+1)

    columns=[]
    N = 0
    for i in range(1,len(all_num_record[order])):
        if all_num_record[order][i] == 1:
            columns.append(chr(ord(G_name)+order)+str(i-N))
        else:
            N +=1
            columns.append(chr(ord(G_name)+order)+"N"+str(N))
    index=[]
    N = 0
    for i in range(1,len(all_num_record[order+1])):
        if all_num_record[order+1][i] == 1:
            index.append(chr(ord(G_name)+order+1)+str(i-N))
        else:
            N +=1
            index.append(chr(ord(G_name)+order+1)+"N"+str(N))
    index.append("0")
    sort_indx = []
    row_table = []
    for g_num in range(len(all_num_record[order+1])-1):
        if all_num_record[order+1][g_num+1] == 1:
            sort_indx.append(index[g_num])
            row_table.append(table[g_num])
    for g_num in range(len(all_num_record[order+1])-1):
        if all_num_record[order+1][g_num+1] == -1:
            sort_indx.append(index[g_num])
            row_table.append(table[g_num])
    sort_indx.append(index[-1])
    row_table.append(table[-1])
    N = 0
    sort_columns = copy.deepcopy(columns)
    sort_table =  [[[] for i in range(len(all_num_record[order])-1)] for i in range(len(all_npresult[order+1])+1)]
    for g_num in range(len(all_num_record[order])-1):
        if all_num_record[order][g_num+1] == 1:
            for i in range(len(sort_table)):
                sort_table[i][N]=row_table[i][g_num]
            sort_columns[N]=columns[g_num]
            N+=1
    for g_num in range(len(all_num_record[order])-1):
        if all_num_record[order][g_num+1] == -1:
            for i in range(len(sort_table)):
                sort_table[i][N]=row_table[i][g_num]
            sort_columns[N]=columns[g_num]
            N+=1
    df =pd.DataFrame(sort_table,index=sort_indx,columns=sort_columns)
    print (df)

    for row in sort_table[:-1]:
        for i in range(len(row)):
            if len(row[i])==0:
                row[i]=0
            else:
                row[i]=sum(row[i])

    print ("\nM("+sort_columns[0][0]+", "+sort_indx[0][0]+")")
    matrix = []
    m_columns=[i for i in sort_indx[:-1] if i[1]!="N"]
    m_index=[]
    for col in range(len(sort_columns)):
        if sort_columns[col][1]=="N":
            break
        matrix.append([i[col] for i in sort_table[:len(m_columns)]])
        m_index.append( sort_columns[col])
    if len(matrix)>0 and len(m_columns)>0 :
        df =pd.DataFrame(matrix,index=m_index,columns=m_columns)
        print (df)
        df.to_csv('csv/matrix'+sort_columns[0][0]+'.csv')
    else:
        print ("none")

    print ("\nM("+sort_columns[0][0]+"N, "+sort_indx[0][0]+")")
    matrix = []
    m_columns=[i for i in sort_indx[:-1] if i[1]!="N"]
    m_index=[]
    for col in range(len(sort_columns)):
        if sort_columns[col][1]=="N":
            matrix.append([i[col] for i in sort_table[:len(m_columns)]])
            m_index.append( sort_columns[col])
    if len(matrix)>0 and len(m_columns)>0 :
        df =pd.DataFrame(matrix,index=m_index,columns=m_columns)
        print (df)
        df.to_csv('csv/matrix'+sort_columns[0][0]+'N.csv')
    else:
        print ("none")


print (time.time()-total_time)



#########################################show graph
# print("node",G.number_of_nodes())
# print("edges",G.number_of_edges())
# print(G.nodes)
# print(G.edges)
# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()
