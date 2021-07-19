import math
import numpy as np

def perm(n,begin,end):
    global n_array,side_array
    if begin>=end:
        side = False
        for i in range(0,nodes-1,2):
            if n[i]+1==n[i+1] or n[i]-1==n[i+1]:
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

nodes = 8
#############################find all G
circle = []
for i in range(1,nodes):
    circle.append([i,i+1])
circle.append([1,nodes])

print circle

permutation(nodes)
print len(n_array)
# print side_array

side_array = np.array(side_array)
print np.shape(side_array)[0]
side_array = np.reshape(side_array,(np.shape(side_array)[0],-1,2))
print np.shape(side_array)

for i in range(len(side_array)):
    side_array[i]=sort(side_array[i])

side_array =np.unique(side_array,axis=0)
print np.shape(side_array)




path = 'output2.txt'
record_file = open(path, 'w')
for i in side_array:
    for j in i:
        record_file.write(str(j+1)+" ")
    record_file.write("\n\n")
record_file.close()
# for i in range(node/2):
    # last_num =
