# -*- coding: utf-8 -*-
##!/usr/bin/python3

import time
import numpy as np
import scipy.stats, scipy.special
from matplotlib import pyplot as plt
import threading

## ziehen aus der normal Verteilung: scipy.stats.norm.rvs

n_threads = 1
n_samples = 5000
n_samples = 10
n = 250

mu = 0
p = .7
var1 = 1
var2 = 10


def perf_integral(func, lower, upper, precision = 3):
    step = (1/10**precision)
    start = lower + 0.5*step
    value = 0
    return  value

def expect(f, a,b, exp=1):#not in use
    '''
    returns expected value of function f for range [a,b]
    '''
    integrand = lambda y: f(y)* y**exp
    norm = lambda y: f(y)
    try:
        norm_factor =  scipy.integrate.quad ( norm, a, b )[0]
    except:
        return -1
    try:
        result = scipy.integrate.quad ( integrand, a, b )[0]
    except:
        return -1
    return result / norm_factor

def variance(f, expected_value, a,b):
    '''
    returns expected value of function f for range [a,b]
    '''
    result = expect(f, a,b,2) - expect(f, a,b)**2
    return result

def likelyhood_ex(tau,s,n):
    return 1/(tau ** n) * np.exp(-s / tau)

def normfactor(func, limit):
    try:
        result = 1 / scipy.integrate.quad(func, 0., limit)[0]
    except:
        result = -1
    return result

def mixed_norm():
    random1 = p * scipy.stats.norm.rvs(mu,scale=np.sqrt(var1))
    random2 = (1-p) * scipy.stats.norm.rvs(mu,scale=np.sqrt(var2))
    result = random1 + random2
    return result;


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

            sample = []
            for i in range(n):
                sample.append(mixed_norm())
            sorted_sample = sorted(sample)
            s = sum(sample)
            e = s/n
            m = 0
            if n%2==0:
                m  = sorted_sample[int(n/2)]
                m += sorted_sample[int(n/2)-1]
                m /= 2
            else:
                m = sorted_sample[int(len(sorted_sample)/2)]

            print(" s={}, e={}, m={}".format(s,e,m))
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
        time.sleep(5)
        done_simulations = 0
        for k in range(n_threads):
            done_simulations += RTD[k]['executed']
        print("durchgeführte Simulationen nach {} Sekunden: {}".format(int(time.time()-t_start),done_simulations))
        if RTD[0]['done']:# and RTD[1]['done'] and RTD[2]['done'] and RTD[3]['done']:
            running = False
    for k in range(n_threads):
        executed += RTD[k]['executed']
    #print(RTD)
