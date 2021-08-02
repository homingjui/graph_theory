import math
import numpy as np

def X_cal(n):
    return float(100*min(n)+max(n))

remove_g_n = 0
G_result=np.array([[[1,2],[1,3],[1,5],[2,3],[2,4],[3,6],[4,5],[4,6],[5,6]]])
for i in range(len(G_result[remove_g_n])):
    edges_on_node =  np.unique(G_result[remove_g_n,np.where((G_result[remove_g_n]==G_result[remove_g_n,i,0])  |
                                                                                                                                (G_result[remove_g_n]==G_result[remove_g_n,i,1]))[0] ],axis=0)
    remove_g =np.array(G_result[remove_g_n])
    remove_g_or = np.delete(remove_g,np.where(np.all(remove_g==G_result[remove_g_n,i],axis=1))[0],axis=0)
    remove_g = np.delete(remove_g,np.where(np.all(remove_g==G_result[remove_g_n,i],axis=1))[0],axis=0)
    remove_g[remove_g==G_result[remove_g_n,i,1]] = G_result[remove_g_n,i,0]
    remove_g[remove_g>G_result[remove_g_n,i,1]] -= 1
    same_flag=False

    for j in remove_g:
        if (len(remove_g[(remove_g[:,0]==j[0]) & (remove_g[:,1]==j[1])])+
            len(remove_g[(remove_g[:,0]==j[1]) & (remove_g[:,1]==j[0])]))>1:
            same_flag=True
            # break
    if not same_flag:
        print G_result[remove_g_n,i]
        print remove_g_or
        print remove_g
        remove_g_x = np.array([X_cal(i) for i in remove_g])
        remove_g_or_x = np.array([X_cal(i) for i in remove_g_or])
        h=1
        for i in range(len(remove_g)):
            h*= np.prod(remove_g_x[i+1:]*(-1)+remove_g_x[i])/np.prod(remove_g_or_x[i+1:]*(-1)+remove_g_or_x[i])
        print h
    else :
        print "!0"
