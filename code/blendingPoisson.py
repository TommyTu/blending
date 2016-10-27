import numpy as np
import scipy
from scipy.sparse import linalg
import cv2

def legal(x,y,rows,cols):
    if(x>=0 and x<rows and y>=0 and y<cols):
        return True
    else:
        return False

# config & input

Topic = 'lion'
# Topic = 'notebook'

backImageName = Topic + '.png'
foreImageName = Topic + '2.png'
maskName = Topic + '_mask.png'
outputName = Topic + '_out_poisson.png'

backImg = cv2.imread('./poisson/' + backImageName) / 255.0
foreImg = cv2.imread('./poisson/' + foreImageName) / 255.0
mask =    cv2.imread('./poisson/' + maskName,0) / 255.0

rows = backImg.shape[0]
cols = backImg.shape[1]
channels = backImg.shape[2]

alls = rows * cols * channels

# build matrix A and B

I = np.array([])
J = np.array([]) 
S = np.array([])
B = np.array([])
numRowsInA = 0

"""
TODO 5 
Construct matrix A & B

add your code here
"""
S = np.zeros((rows*cols*5))
I = np.zeros((rows*cols*5))
J = np.zeros((rows*cols*5))
B = np.zeros((rows*cols,channels))
print(mask.shape)
numRowsInA = rows*cols

k=0


for i in range(rows):
    for j in range(cols):
        if mask[i][j]==0:
            S[k] = 1
            I[k] = i*cols+j
            J[k] = i*cols+j
            k=k+1
            for ch in range(channels):
                B[i*cols+j][ch] = backImg[i][j][ch]
        else:
            for m,n,q in ((0,0,-4),(0,1,1),(0,-1,1),(1,0,1),(-1,0,1)):
                if(legal(i+m,j+n,rows,cols)==True):
                    S[k] = q
                    I[k] = i*cols+j
                    J[k] = (i+m)*cols+j+n
                    k=k+1
                    for ch in range(channels):
                        B[i*cols+j][ch] = B[i*cols+j][ch] + q * (foreImg[i+m][j+n][ch] + backImg[i+m][j+n][ch])



A = scipy.sparse.coo_matrix((S, (I, J)), shape=(numRowsInA, rows*cols))
Rred,sd = scipy.sparse.linalg.cg(A, B[:,0])
print('done')
Rgreen,sd = scipy.sparse.linalg.cg(A, B[:,1])
Rblue,sd =  scipy.sparse.linalg.cg(A, B[:,2])

Rred = np.reshape(Rred, (rows,cols))
Rgreen = np.reshape(Rgreen, (rows,cols))
Rblue = np.reshape(Rblue, (rows,cols))

R = np.dstack((Rred,Rgreen,Rblue))

cv2.imshow('output', R);
cv2.waitKey(0)
cv2.imwrite(outputName, R * 255);
