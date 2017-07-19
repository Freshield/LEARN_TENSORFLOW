import math
import numpy as np
import pandas as pd
import sklearn

K = 2
Pxi_x = 10
Pxi_y = 2

def Gaussian_possible(number, mu, sigma):
    result = 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-(number-mu)**2/(2*sigma**2))
    return result

def Gaussian_array(x, mu, sigma):
    #Dimension
    #N1 + N2 = 10
    #x(7500,N1)
    #mu(7500,1)
    #sigma(7500,7500,1)
    D,N = x.shape
    mu_array = np.zeros((D,N))
    for i in range(N):
        mu_array[:,i] = np.reshape(mu,(D))

    x = x.T - mu_array.T



B_data = pd.read_csv('B.csv', header=None)
G_data = pd.read_csv('G.csv', header=None)
R_data = pd.read_csv('R.csv', header=None)

X = R_data

Pxi = np.zeros(Pxi_x, Pxi_y)

while True:
    for i in range(K):
        Pxi[:,i] = Gaussian_array()