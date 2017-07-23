import numpy as np
import matplotlib.pyplot as plt

a = np.arange(4).reshape([2,2])

print a
print a * a
print np.sqrt(np.sum(a * a))


def L2_dist(matrix_a, matrix_b):
    matrix_diff = matrix_b - matrix_a
    matrix_diff = matrix_diff * matrix_diff
    result = np.sqrt(np.sum(matrix_diff))
    return result

b = np.array([1,2,3,4])
c = np.array([5,6,7,8])
print L2_dist(b,c)

def Gaussian_possible(number, mu, sigma):
    result = 1./(np.sqrt(2*np.pi)*sigma)*np.exp(-(number-mu)**2/(2*sigma**2))
    return result

def Gaussian_array(x, mu, sigma):
    #Dimension
    #x(500*375,1)
    #mu(500*375,1)
    #sigma(500*375,500*375)
    D = x.shape[0]
    part1 = 1. / (2 * np.pi) ** (D / 2)
    part2 = 1. / (np.linalg.det(sigma) ** 0.5)
    part3 = np.exp(-0.5 * (x - mu).T * np.linalg.inv(sigma) * (x - mu))
    return part1 * part2 * part3

x = np.ones((10,1))
mu = np.ones((10,1))
sigma = np.eye(10)

print Gaussian_array(x,mu,sigma)

#print Gaussian_possible(0.0)

print np.linalg.det([[1,0],[0,1]])

print np.linalg.inv([[1.,0.,1.],[2.,1.,0.],[-3.,2.,-5.]])

print np.eye(2)