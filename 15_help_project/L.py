# Notation
# P1=P(Vi=1|θ)
# P0=1-P1
# Cam_Image(i)=I1 to In
# Syn_Image=I0
# Epsilon=threshold

import math
import numpy as np
import pandas as pd
import sklearn
# import matplotlib.pyplot as plt

# Initialization
Epsilon = 1e-4
Alpha = 0
Beta = 0
m,n,N=10,10,10
Mu1,Mu2,Sigma1,Sigma2 = 0,0,0.1,0.1

B_data = pd.read_csv('B.csv', header=None)
G_data = pd.read_csv('G.csv', header=None)
R_data = pd.read_csv('R.csv', header=None)

E1,E2=0,0

# Estimation
for i in range (1,N):
    data
    k = 0
    for i in range (1,N-1):
        for j in range (i,N):
            dist(k) = np.linalg.norm(Cam_Image(i)- Cam_Image(j))
            distance = max(dist(k))

    P1 = Alpha*np.random.normal(Cam_Image(i),Mu1,Sigma1)/ \
         (Alpha*np.random.normal(Cam_Image(i),Mu1,Sigma1)+(1-Alpha)*(Beta*np.random.normal(Cam_Image(i),Mu2,Sigma2)+(1-Beta)/(distance+1)))
    P0 = 1-P1

    for i in range (1,N):
        for j in range (1,m*n):
            E1=E1+(P1*math.log(Alpha*np.random.normal(Cam_Image(i),Mu1,Sigma1)))+ \
               P0*math.log((1-Alpha)*(Beta*np.random.normal(Cam_Image(i),Mu2,Sigma2)+(1-Beta)/(max(Cam_Image(i)_Difference)+1)))
            E2=E2+E1

    # Maximization
    Num,Den=0,0
    for i in range (1,N):
        Num=Num+P1*Cam_Image(:,i)
        Den=sum(P1)
    Syn_Image = Num/Den
    for i in range(1, N):
        Num = Num+P1*np.linalg.norm(Syn_Image- Cam_Image(i))
        Den= sum(P1)
        Sigma= Num/Den
    Alpha=sum(P1)/N

    # Threshold Check
    Syn_Image
    # Print syn image
