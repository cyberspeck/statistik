import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

np.random.seed(0)

n = 250
mu = [0, 0]
sigma = [1, np.sqrt(10)]
weights = [.7, .3]

def gaussian_mix(sample_size):
    sample = np.empty((sample_size))
    for i in range(sample_size):
        Z = np.random.choice([0,1], p=weights) # latent variable
        sample[i] = (np.random.normal(mu[Z], sigma[Z], 1))
    return sample

X_tot = gaussian_mix(n)

class GM1D:
    def __init__(self,X,iterations):
        self.iterations = iterations
        self.X = X
        self.mu = None
        self.pi = None
        self.var = None
  
    def run(self):
        
        """
        Start with arbitrary mu, pi and var
        """
        self.mu = [-8,5]
        self.pi = [1/2,1/2]
        self.var = [5,1]
        
        
        """
        E-Step
        """
        
        for iter in range(self.iterations):
            """Create and calculate probability for each datapoint x_i to belong to gaussian g"""
            r = np.zeros((len(X_tot),2))  
            for column,g,weight in zip(range(2), [norm(loc=self.mu[0],scale=self.var[0]),
                                                  norm(loc=self.mu[1],scale=self.var[1])], self.pi):
                # Write the probability that x belongs to gaussian c in column c. 
                r[:,column] = weight*g.pdf(X_tot)

            """
            Normalize the probabilities such that each row of r sums to 1
            and weight it by mu_c == the fraction of points belonging to cluster c
            """
            for i in range(len(r)):
                r[i] = r[i]/(np.sum(self.pi)*np.sum(r,axis=1)[i])

            if iter == 0 or iter == self.iterations-1:
                """Plot the data"""
                fig = plt.figure(figsize=(10,10))
                ax0 = fig.add_subplot(111)
                for i in range(len(r)):
                    ax0.scatter(self.X[i],0,s=100) 
                    #ax0.scatter(self.X[i],0,c=np.array([r[i][0],r[i][1],r[i][2]]),s=100) 
                """Plot the gaussians"""
                for g,c in zip([norm(loc=self.mu[0],scale=self.var[0]).pdf(np.linspace(-20,20,num=n)),
                                norm(loc=self.mu[1],scale=self.var[1]).pdf(np.linspace(-20,20,num=n))],['r','b']):
                    ax0.plot(np.linspace(-20,20,num=n),g,c=c)
                plt.show()
            
            
            """M-Step"""
    
            """calculate m_c"""
            m_c = []
            for c in range(len(r[0])):
                m = np.sum(r[:,c])
                m_c.append(m) # For each cluster c, calculate the m_c and add it to the list m_c
            """calculate pi_c"""
            for k in range(len(m_c)):
                self.pi[k] = (m_c[k]/np.sum(m_c)) # For each cluster c, calculate the fraction of points pi_c which belongs to cluster c
            """calculate mu_c"""
            self.mu = np.sum(self.X.reshape(len(self.X),1)*r,axis=0)/m_c
            """calculate var_c"""
            var_c = []
            for c in range(len(r[0])):
                var_c.append( (1/m_c[c]) * np.dot(((np.array(r[:,c]).reshape(n,1)) * (self.X.reshape(len(self.X),1)-self.mu[c])).T,
                                                  (self.X.reshape(len(self.X),1)-self.mu[c])) )
    
GM1D = GM1D(X_tot,20)
GM1D.run()
