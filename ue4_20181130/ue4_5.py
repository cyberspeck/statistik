# -*- coding: utf-8 -*-
##!/usr/bin/python3

import time
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
import threading

## ziehen aus der normal Verteilung: scipy.stats.norm.rvs

n_threads = 1
n_samples = 5000
n_samples = 10
n = 250

mu = 0
var1 = 1
var2 = 10

mu = [0, 0]
sigma = [1, np.sqrt(10)]
weights = [.7, .3]


class gaussian_mix():
    def __init__(self, mu, sigma, weights):
        self.mu = mu
        self.sigma = sigma
        self.weights = weights

    def draw(self, sample_size):
        sample = np.empty((sample_size))
        for i in range(sample_size):
            Z = np.random.choice([0,1], p=weights) # latent variable
            sample[i] = (np.random.normal(mu[Z], sigma[Z], 1))
        return sample

mixed = gaussian_mix(mu, sigma, weights)

class simulation(threading.Thread):
    def __init__(self, id, rtd):
        threading.Thread.__init__(self)
        self.rtd = rtd
        self.id = id
        self.rtd[self.id] = {}
        self.rtd[self.id]['executed'] = 0
        self.rtd[self.id]['done'] = False
    def run(self):
        for j in range(int(n_samples/n_threads)):

            sample = mixed.draw(n)



            self.rtd[self.id]['executed'] += 1

        self.rtd[self.id]['done'] = True


if __name__ == '__main__':
    print("---------------------------------------------------------------\n")
    print("            Uebung 3_6 David Blacher, Johannes Kurz            \n")
    print("---------------------------------------------------------------\n")
    print("Ziehe {}-mal Stichproben mit n={} aus Mischung "
          "von Normalverteilungen mit mu={}, var1={} & var2={}"
          .format(n_samples, n, mu,var1,var2))
    executed = 0
    RTD = {}
    threads = {}
    for k in range(n_threads):
        threads[k] = simulation(k, RTD)
        threads[k].daemon = True
        threads[k].start()
    running = True
    t_start = time.time()
    while running:
        time.sleep(1)
        done_simulations = 0
        for k in range(n_threads):
            done_simulations += RTD[k]['executed']
        print("durchgeführte Simulationen nach {} Sekunden: {}".format(int(time.time()-t_start),done_simulations))
        if RTD[0]['done']:# and RTD[1]['done'] and RTD[2]['done'] and RTD[3]['done']:
            running = False
    for k in range(n_threads):
        executed += RTD[k]['executed']
    #print(RTD)
