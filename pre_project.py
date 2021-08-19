####python3.7 -m pip install pympler
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
import itertools
from pympler.asizeof import asizeof

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def perm(n,begin,end):
    global n_array,side_array,now_permutation
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
            side_array[now_permutation]=n
        else:
            side_array[now_permutation]=0
        n_array[now_permutation]=n
        now_permutation+=1
    else:
        for num in range(begin,end):
          n[num],n[begin]=n[begin],n[num]
          perm(n,begin+1,end)
          n[num],n[begin]=n[begin],n[num]
#end def#####################################
def permutation(num):
    global n_array,side_array,now_permutation
    n = np.arange(num,dtype=np.uint8)
    n_array = np.repeat([n],math.factorial(num),axis=0)
    side_array = np.repeat([n],math.factorial(num),axis=0)
    now_permutation=0
    perm(n,0,len(n))

#end def#####################################
def X_cal(n):
    if n[0]<n[1]:
        return float(100*n[0]+n[1])
    elif n[0]>n[1]:
        return float(100*n[1]+n[0])
    elif n[0]==n[1]:
        return 0
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
def do_all_G(edge_list_array,G_result,num_record,npresult,npresult_z,npresult_n):
    print ("***do permutation!***")
    iterations = len(n_array)
    edge_num = len(edge_list_array[0])
    ############################################### do all G
    for now_G in range(len(edge_list_array)):
        start = time.time()
        write_string=""
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
        ######################################find all combination
        x_array = np.array([X_cal(i) for i in edge_list_array[now_G]])
        x_prod = [np.prod(x_array[i+1:]*(-1)+x_array[i]) for i in range(edge_num)]
        for now_iteration in range(iterations):
            new_edge_list = copy.deepcopy(edge_list_array[now_G])
            now_n = n_array[now_iteration]
            for i in range(edge_num):
                new_edge_list[i][0]=now_n[new_edge_list[i][0]-1]+1
                new_edge_list[i][1]=now_n[new_edge_list[i][1]-1]+1
            result_uni.append(sort(new_edge_list))
            result.append(new_edge_list)
    ##########################################save file
        print (time.time()-start)
        result_uni,result_n =np.unique(result_uni,axis=0,return_index=True)####################################filter!!
        n_flag = False
        z_result_uni = []
        for n_gaph in range(len(result)):
            ###############################get z,w
            new_edge_list_x = np.array([X_cal(j) for j in result[n_gaph]])
            z = [np.prod(new_edge_list_x[i]-new_edge_list_x[i+1:])/x_prod[i] for i in range(edge_num)]
            z = np.prod(z)
            z_result_uni.append(int(z/abs(z)))
            w =np.prod(new_edge_list_x/x_array)
            if n_flag==False:
                if abs(z+1)+abs(w-1)< 0.0001:
                    if sort(result[n_gaph])==sort(result[0]):
                        n_flag = True
            ###############################write record
            if savefile:
                write_string+="\n\n"+str(n_gaph)+", f="
                for i in range(len(n_array[0])):
                    write_string+=str(i+1)+":"+str(n_array[n_gaph][i]+1)+",  "
                write_string += "\n"
                write_string += str(result[n_gaph].tolist())
                    # print n_gaph
                write_string += "\tZ="+str(z)+", W="+str((abs(z+1)+abs(w-1)))

        print (time.time()-start)
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
        write_head += str(edge_list_array[now_G].tolist())
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
        # start = time.time()
        # print time.time()-start
        if n_flag:
            print (""+G_name+"N"+"("+str(num_record.count(-1))+")  ",end="")
        else:
            print (""+G_name+"("+str(num_record.count(1))+")  ",end="")
        print("num f(G) %d"%(len(result_uni)))
        if len(G_result)==0:
            G_result= np.array([edge_list_array[now_G]])
        else:
            G_result= np.vstack((G_result,[edge_list_array[now_G]]))
        npresult_z.append(z_result_uni)
        npresult_n.append(result_n)
        npresult.append(result_uni)
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
nodes =8
path = 'output.txt'
record_file = open(path, 'w')
record_file.close()
savefile = True
##############################################parm-end
total_time = time.time()
print("%d nodes, "%nodes,end="")
print("iterations: %d"%math.factorial(nodes))
#############################################find all G
circle = []
for i in range(1,nodes):
    circle.append([i,i+1])
circle.append([1,nodes])
permutation(nodes)

print(time.time()-total_time)

# 14.239812135696411
# 25.85783863067627

# 10.722990274429321
# 22.64511013031006

side_array=side_array[np.invert(np.all(side_array==0,axis=1))]+1
side_array = np.reshape(side_array,(len(side_array),-1,2))
for i in range(len(side_array)):
    side_array[i]=sort(side_array[i])
side_array =np.unique(side_array,axis=0)
# print np.shape(side_array)

print(time.time()-total_time)

circle = np.repeat([circle],np.shape(side_array)[0],axis=0)
edge_list_array = np.hstack((circle,side_array))
# print edge_list_array
for i in range(len(edge_list_array)):
    edge_list_array[i]=sort(edge_list_array[i])

print(time.time()-total_time)
edge_list_array=edge_list_array[:1]
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
    G_result,npresult,num_record,npresult_z,npresult_n=do_all_G(edge_list_array,G_result,num_record,npresult,npresult_z,npresult_n)
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
                    h*= np.prod(remove_g_x[i+1:]*(-1)+remove_g_x[i])/np.prod(remove_g_or_x[i+1:]*(-1)+remove_g_or_x[i])
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
    print("iterations: %d"%math.factorial(nodes-n_order))
    G_name=chr(ord(G_name)+1)
    permutation(nodes-n_order)
    G_result = np.array([])   ###########for all G
    num_record = [0]
    npresult = []   ###########for all result filter
    npresult_z = []   ###########esult z
    npresult_n = []   ###########f result filter pos
    ########################################permutation
    # G_result,npresult,num_record,npresult_z,npresult_n=do_all_G(edge_list_array,G_result,num_record,npresult,npresult_z,npresult_n)
    G_result,npresult,num_record,npresult_z,npresult_n=do_all_G(np.array([i[2] for i in remove_g_all if i[2]]),
                                                                G_result,num_record,npresult,npresult_z,npresult_n)
    print ("### "+str(len(G_result))+" 2n-"+str(n_order)+" G,",end="")
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
                    table[f_g][removed_g[0]].append(removed_g[3]*all_npresult_z[order+1][f_g][pos])
                    #table[f_g][removed_g[0]].append([removed_g[1],removed_g[3],all_npresult_z[order+1][f_g][pos]])
                    #table[f_g][removed_g[0]].append([removed_g[1]+1,removed_g[3]])
                    #table[f_g][removed_g[0]].append(removed_g[3])
        else :
            table[len(all_npresult[order+1])][removed_g[0]].append(removed_g[1]+1)
            # print "n"

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
    # df =pd.DataFrame(table,index=index,columns=columns)
    # print df
    sort_indx = []
    row_table = []
    # for order in range(1,len(all_num_record)):
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
    # df =pd.DataFrame(row_table,index=sort_indx,columns=columns)
    # print df
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
        df.to_csv('csv/table'+sort_columns[0][0]+'.csv')
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
        df.to_csv('csv/table'+sort_columns[0][0]+'N.csv')
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
