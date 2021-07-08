import csv
import math
import copy
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

with open("order4_new_graph_order4_10.txt", "r") as fp:
    orignal = pickle.load(fp)

with open("order4_new_graph_3.txt", "r") as fp:
    compare = pickle.load(fp)

print len(orignal),
print len(orignal[0]),
print len(orignal[0][0])
print len(compare),
print len(compare[0]),
print len(compare[0][0])

orignal = np.array(orignal)
compare = np.array(compare)

for i in range(len(compare)):
    print "%d:"%(i+1),
    # for j in  range(len(n_array[i])):
        # print "f(%d)=%d"%(j+1,n_array[i][j]+1),
    # print ""
    for j in  compare[i]:
        print "[%d, %d]"%(j[0],j[1]),
    if len(np.where(np.all(orignal == compare[i], axis=(1,2)))[0])==0:
        print"not found"
    else:
        print np.where(np.all(orignal == compare[i], axis=(1,2)))[0]
    print "\n"

# for i in check:
#     print np.where(np.all(result == i, axis=(1,2)))[0]
