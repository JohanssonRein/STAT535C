import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(123)
# Problem 5

# (a) The Metropolis-Hasting algorithm

#Initial value for z
z_0 = np.array([5,5])
mean = np.array([0,0])
cov = np.array([[1,0.75],[0.75,1]])
Id = np.eye(2)
epsilon = 0.1

def f(z):
    return multivariate_normal.pdf(z,mean=mean, cov=cov) 

def MH(z,n=10000):
    samples = np.zeros((n + 1, 2)) #Create a 2 column array with 0 everywhere, so it will store our estimates
    samples[0]=z
    
    samples_mean = np.zeros((n + 1, 2))
    samples_mean[0] = z
    for i in range(n):
        z_new = multivariate_normal.rvs(mean=z,cov=epsilon * Id)
        alpha = min(1,f(z_new)/f(z) )
        u = np.random.uniform(0,1)
        if u <= alpha:
            z = z_new
        samples[i+1]=z
        samples_mean[i+1]=np.mean(samples[:i+2],axis=0)
        
    return samples, samples_mean

samples, samples_mean = MH(z=[5,5],n=10000)

squared_error = np.sum((mean-samples_mean)**2,axis=1)

iterations = np.arange(len(squared_error))

plt.figure(figsize=(8, 5))
plt.plot(iterations[1:], np.log(squared_error[1:]))

plt.xlabel("Number of iterations")
plt.ylabel("Log-squared Error of the estimate")
plt.title("")
plt.grid(alpha=0.3)

plt.show()


        
# print(MH([5,5],n=10000))  
        
        
        
        




    





