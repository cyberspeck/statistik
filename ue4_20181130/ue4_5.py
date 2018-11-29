# -*- coding: utf-8 -*-
##!/usr/bin/python3

'''
sources of inspiration:
https://cseweb.ucsd.edu/~elkan/250Bwinter2011/mixturemodels.pdf
https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php
'''


import time
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib import style


def gaussian_mix(sample_size, mu, sigma, weights):
    sample = np.empty((sample_size))
    for i in range(sample_size):
        Z = np.random.choice([0,1], p=weights) # latent variable
        sample[i] = (np.random.normal(mu[Z], sigma[Z], 1))
    return sample

class gaussian_mix_solver:
    def __init__(self, sample, max_iterations, tolerance):
        self.sample = sample
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        # Start with arbitrary mu, sigma and weights
        self.mu = [0,0]
        self.sigma = [1,10]
        self.weights = [1/2,1/2]

        self.iter_mu = [self.mu]
        self.iter_sigma = [self.mu]
        self.iter_weights = [self.mu]

        self.log_likelihoods = []

    def run(self, plot):

        ### Plot initial ###
        if plot:
            fig = plt.figure(figsize=(10,10))
            ax0 = fig.add_subplot(111)
            """Plot the data"""
            for i in range(n):
                ax0.scatter(self.sample[i],0,s=100)
                #ax0.scatter(self.sample[i],0,c=np.array([membership[i][0],membership[i][1],membership[i][2]]),s=100)
            """Plot the gaussians"""
            for g,c in zip([norm(loc=self.mu[0],scale=self.sigma[0]).pdf(np.linspace(-20,20,num=n)),
                            norm(loc=self.mu[1],scale=self.sigma[1]).pdf(np.linspace(-20,20,num=n))],['r','b']):
                ax0.plot(np.linspace(-20,20,num=n),g,c=c)
            plt.title("estimation after 0 iterations:\n mu={},\n sigma={},\n weights={}".format(self.mu, self.sigma, self.weights))
            plt.show()

            print("\nEstimation at iteration no: 0")
            print(" mu= {}".format(self.mu))
            print(" sigma= {}".format(self.sigma))
            print(" weights= {}".format(self.weights))


        for iter in range(self.max_iterations):

            log_likelihood = np.log(np.sum([k*norm(self.mu[c],self.sigma[c]).pdf(self.sample) for k,c in zip(self.weights,range(2))]))
            self.log_likelihoods.append(log_likelihood)

            ### E-Step ###

            """Calculate probability for each datapoint x_i to belong to gaussian g"""
            membership = np.zeros((n,2))
            for column,g,weight in zip(range(2),
                                       [norm(loc=self.mu[0],scale=self.sigma[0]),
                                        norm(loc=self.mu[1],scale=self.sigma[1])],
                                       self.weights):
                # Write probability in column of membership
                membership[:,column] = weight*g.pdf(self.sample)
            """
            Now, Normalize the probabilities such that each row of membership sums to 1
            and weight it by mu_c == the fraction of points belonging to cluster c
            """
            for i in range(n):
                membership[i] = membership[i]/(np.sum(self.weights)*np.sum(membership,axis=1)[i])


            ### M-Step ###

            # For each cluster c, calculate the mean and add it to the list mean_c
            membership_cumulative = []
            for c in range(2):
                cumulative = np.sum(membership[:,c])
                membership_cumulative.append(cumulative)

            # For each cluster c, calculate the fraction of points which belongs to cluster c
            for c in range(2):
                fraction = membership_cumulative[c] / n
                self.weights[c] = fraction

            # calculate mu_c
            for c in range(2):
                mean  = np.sum(membership[:,c] * self.sample)
                mean /= membership_cumulative[c]
                self.mu[c] = mean

            # calculate var_c
            for c in range(2):
                var  = np.sum(membership[:,c] * np.power((self.sample-self.mu[c]), 2))
                var /= membership_cumulative[c]
                self.sigma[c] = np.sqrt(var)

            self.iter_mu.append(self.mu)
            self.iter_sigma.append(self.sigma)
            self.iter_weights.append(self.weights)


        best_iteration = np.argmax(self.log_likelihoods)
        self.max_mu = self.iter_mu[best_iteration]
        self.max_sigma = self.iter_sigma[best_iteration]
        self.max_weights = self.iter_weights[best_iteration]

        ### Plot final ###
        if plot:
            fig = plt.figure(figsize=(10,10))
            ax0 = fig.add_subplot(111)
            """Plot the data"""
            for i in range(n):
                ax0.scatter(self.sample[i],0,s=100)
                #ax0.scatter(self.sample[i],0,c=np.array([membership[i][0],membership[i][1],membership[i][2]]),s=100)
            """Plot the gaussians"""
            for g,c in zip([norm(loc=self.max_mu[0],scale=self.max_sigma[0]).pdf(np.linspace(-20,20,num=n)),
                            norm(loc=self.max_mu[1],scale=self.max_sigma[1]).pdf(np.linspace(-20,20,num=n))],['r','b']):
                ax0.plot(np.linspace(-20,20,num=n),g,c=c)
            plt.title("after {} iterations:\n mu={},\n sigma={},\n weights={}".format(iter, self.max_mu, self.max_sigma, self.max_weights))
            plt.show()

            print("\nBest result found at teration no: {}".format(best_iteration))
            print(" mu= {}".format(self.max_mu))
            print(" sigma= {}".format(self.max_sigma))
            print(" weights= {}".format(self.max_weights))


if __name__ == '__main__':

    n_samples = 5000
    n_samples = 500
    n = 250

    mu = 0
    var1 = 1
    var2 = 10

    max_iterations = 7
    tolerance = 0.0
    # real parameters for mixed gaussian
    real_mu = [0, 0]
    real_sigma = [1, np.sqrt(10)]
    real_weights = [.7, .3]

    print("---------------------------------------------------------------\n")
    print("            Uebung 4_5 David Blacher, Johannes Kurz            \n")
    print("---------------------------------------------------------------\n")
    print("Ziehe {}-mal Stichproben mit n={} aus Mischung "
          "von Normalverteilungen mit mu={}, var1={} & var2={}"
          .format(n_samples, n, mu,var1,var2))

    estimated_mu = np.empty((n_samples, 2))
    estimated_sigma = np.empty((n_samples, 2))
    estimated_weights = np.empty((n_samples, 2))

    for simulation in range(n_samples):
        entire_sample = gaussian_mix(n, real_mu, real_sigma, real_weights)
        gaussian_model = gaussian_mix_solver(entire_sample,max_iterations, tolerance)
        if simulation == 0:
            print("Beispielhaft wird eine numerische Berechnung grafisch sichtbar gemacht.")
            gaussian_model.run(True)

            plt.plot ( range(len(gaussian_model.log_likelihoods)), gaussian_model.log_likelihoods, label="priorA"  )
            plt.title("log likelyhood maximisation")
            plt.show()

            print("\nBitte geben Sie uns 10 sekunden für den Rest der Simulationen...")
        else:
            gaussian_model.run(False)

        estimated_mu[simulation] = gaussian_model.max_mu
        estimated_sigma[simulation] = gaussian_model.max_sigma
        estimated_weights[simulation] = gaussian_model.max_weights

    overall_mean_mu = np.sum(estimated_mu, axis=0) / n_samples
    overall_mean_sigma = np.sum(estimated_sigma, axis=0) / n_samples
    overall_mean_weights = np.sum(estimated_weights, axis=0) / n_samples

    print("\nNach {} Simulationen, sind dies die gemittelten Ergebnisse:".format(n_samples))
    print(" overall_mu= {}".format(overall_mean_mu))
    print(" overall_sigma= {}".format(overall_mean_sigma))
    print(" overall_weights= {}".format(overall_mean_weights))




