import math
import numpy as np
from itertools import permutations,islice,zip_longest
from time import time,sleep
from pympler.asizeof import asizeof
from operator import mul
from functools import reduce
from sys import getsizeof
import pickle

def sort(n):
    x=[]
    for i in grouped(n):
        if i[0]>i[1]:
            x.append([i[1],i[0]])
        else:
            x.append([i[0],i[1]])
    return sum(sorted(x),[])

def grouped(iterable):
    a = iter(iterable)
    return list(zip(a, a))

# a = [1, 2, 1, 3, 1, 6, 1, 7, 2, 3, 2, 4, 3, 4, 4, 5, 5, 6, 5, 7, 6, 7]
# b = [1, 2, 1, 3, 1, 7, 2, 3, 2, 4, 3, 4, 4, 5, 4, 6, 5, 6, 5, 7, 6, 7]
# b = [1, 2, 1, 3, 1, 4, 1, 7, 2, 3, 2, 6, 3, 4, 4, 5, 5, 6, 5, 7, 6, 7]

# a_2 = grouped(a)
# b_2 = grouped(b)
# c = grouped(c)

x= np.arange(27).reshape((3,3,3))

print(0/5)
print(0/0)

print(np.average(x,axis=0))

print()

# print(np.array(range(5)))

# f = open('output.txt', 'r').read().split('\n')
# arr = []
# for i in f[:-1]:
#     print(i)
#     i=i.split('[(')[1]
#     i=i.split(')]')[0]
#     i=i.split(', ')
#     row = []
#     for j in i:
#         row.append([int(j[0]),int(j[-1])])
#     arr.append(row)
# print(arr)
# del l
# sleep(5)

# print(pickle.load(open("temp/CN(1).pkl","rb")))

# a=[[[1, 2], [1, 3], [1, 12], [2, 3], [2, 4], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [8, 9], [9, 10], [9, 11], [10, 11], [10, 12], [11, 12]]]
# print(asizeof(a))
# a = sum(a[0], [])
# print(a)
# print(asizeof(a))
# def grouped(iterable):
#     a = iter(iterable)
#     return zip(a, a)
#
#
# for x, y in grouped(a):
#    print (x,y)


# n=4
# a= permutations(range(n))
# print(asizeof(a))
# s=0
# for i in list(permutations(range(n))):
#     print(type(i[0]))
#     s+=1
# print (s)

# start = time()
# x=list(permutations(range(n)))
# print(time()-start)
# print(len(x))
# print(asizeof(x))
#
#
# start = time()
# x=tuple(permutations(range(n)))
#
# print(time()-start)
# print(len(x))
# print(x[0])
# print(asizeof(x))

# remove_g_n = 0
# G_result=np.array([[[1,2],[1,3],[1,5],[2,3],[2,4],[3,6],[4,5],[4,6],[5,6]]])
# for i in range(len(G_result[remove_g_n])):
#     remove_g =np.array(G_result[remove_g_n])
#     remove_g_or=np.array(remove_g)
#     remove_g[i,1]=remove_g[i,0]
#     remove_g[remove_g==G_result[remove_g_n,i,1]] = G_result[remove_g_n,i,0]
#     remove_g[remove_g>G_result[remove_g_n,i,1]] -= 1
#     same_flag=False
#     # print remove_g_or
#     # print remove_g
#
#     for j in remove_g:
#         if j[0] != j[1]:
#             if (len(remove_g[(remove_g[:,0]==j[0]) & (remove_g[:,1]==j[1])])+
#                 len(remove_g[(remove_g[:,0]==j[1]) & (remove_g[:,1]==j[0])]))>1:
#                 same_flag=True
#                 break
#     if not same_flag:
#         print G_result[remove_g_n,i]
#         print remove_g_or
#         print remove_g
#         remove_g_x = np.array([X_cal(i) for i in remove_g])
#         remove_g_or_x = np.array([X_cal(i) for i in remove_g_or])
#         h=1
#         for i in range(len(remove_g)):
#             # for j in range(len(remove_g_x[i+1:])):
#             #     print (remove_g_x[i+1:]*(-1)+remove_g_x[i])[j],
#             #     print "/",
#             #     print (remove_g_or_x[i+1:]*(-1)+remove_g_or_x[i])[j]
#                 # print (remove_g_x[i+1:]*(-1))
#             h*= np.prod(remove_g_x[i+1:]*(-1)+remove_g_x[i])/np.prod(remove_g_or_x[i+1:]*(-1)+remove_g_or_x[i])
#         print remove_g
#         # h *=
#         print h
