import csv
import math
import copy
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

with open("test.txt", "r") as fp:
    read = pickle.load(fp)
for i in range(len(read)):
    print "%d:"%(i+1),
    # for j in  range(len(n_array[i])):
        # print "f(%d)=%d"%(j+1,n_array[i][j]+1),
    # print ""
    for j in  read[i]:
        print "[%d, %d]"%(j[0],j[1]),
    print "\n"
