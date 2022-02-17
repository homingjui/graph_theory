import pandas as pd
import numpy as np

# np.set_printoptions(threshold=np.inf)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)

G_name="A"
matrix = []
for i in range(1,5):
    df = pd.read_csv('csv5/matrix'+chr(ord(G_name)+i)+'.csv')
    matrix.append(df.to_numpy()[:,1:].astype('int32'))
    print ('matrix '+chr(ord(G_name)+i))
    print (np.shape(matrix[-1]))
    print('rank:',np.linalg.matrix_rank(matrix[-1]))
    # print (matrix[-1])
    print()



G_name="A"
for i in range(len(matrix)-1):
    print(chr(ord(G_name)+i+1)+"x"+chr(ord(G_name)+i+2))
    # print (np.shape(np.matmul(matrix[i], matrix[i+1])))
    print(np.matmul(matrix[i], matrix[i+1]))
