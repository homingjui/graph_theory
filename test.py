import math
import numpy as np
import copy
from itertools import permutations,islice,zip_longest
from time import time
from pympler.asizeof import asizeof
from operator import mul
from functools import reduce
from sys import getsizeof
import multiprocessing


def worker(procnum, return_dict):
    """worker function"""
    print(str(procnum) + " represent!")
    return_dict[procnum] = procnum


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print(return_dict.values())



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
