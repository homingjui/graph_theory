import math
import numpy as np
import copy
from itertools import permutations,islice,zip_longest
from time import time
from pympler.asizeof import asizeof

def X_cal(n):
    if n[0]==n[1]:
        return 0
    return float(100*min(n)+max(n))


x= [1,2,3,4]
b= [4,8,10,12]

a=[[1, 2], [1, 3], [1, 8], [2, 3], [2, 4], [3, 4], [4, 5], [5, 6], [5, 7], [6, 7], [6, 8], [7, 8]]

print(math.prod(a))

z = math.prod([math.prod(a[i]-a[i+1:])/math.prod(a) for i in range(12)])
print(z)

print(math.prod(math.prod(map(lambda i:i-b[n],b[n+1:])) for n in range(3)))
print(math.prod(map(lambda i:i-b[0],b[1:])))




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
