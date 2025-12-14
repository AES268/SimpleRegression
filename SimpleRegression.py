import pandas as pd
import numpy as np
df=pd.read_csv('MultipleLR.csv - MultipleLR.csv (1).csv',header=None)
X=df.iloc[:,:3].values
Y=df.iloc[:,3].values
X_n=(X-X.min(0))/(X.max(0)-X.min(0))
Y_n=(Y-Y.min())/(Y.max()-Y.min())
w=np.random.rand(3)
b=np.random.rand()
lr=0.01
for _ in range(1000):
    idx=np.random.permutation(len(Y))
    for i in idx:
        pred=np.dot(X_n[i],w)+b
        err=pred-Y_n[i]
        w-=lr*2*err*X_n[i]
        b-=lr*2*err
print("Weights:",w)
print("Bias:",b)
