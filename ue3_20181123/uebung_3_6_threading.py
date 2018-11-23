# -*- coding: utf-8 -*-
##!/usr/bin/python3

import time
import numpy as np
import scipy.stats, scipy.special
from matplotlib import pyplot as plt
import threading

## ziehen aus der gamma Verteilung: scipy.stats.gamma.rvs
## ziehen aus der Exponentialverteilung: scipy.stats.expon.rvs
## numerisches integrieren von f im Intervall von a bis b:
## scipy.integrate.quad( f,a, b )
## Numerisches Lösen eines Gleichungssystems:
## scipy.optimize.fsolve ( equations )

n_threads = 8
n_samples = 5000
#for testing: (should be dvideble by n_threads)
n_samples = 48
n = 100
a = 4
b = 1
alpha = 0.05

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

def getPosterior(tau, prior, likely, s,n):
    numerator = lambda tau: likely(tau, s,n) * prior(tau)
    nominator = lambda tau: scipy.integrate.quad ( numerator, 0, 30 )[0]
    result = numerator(tau) / nominator(tau)
    return result

def normfactor(func, limit):
    try:
        result = 1 / scipy.integrate.quad(func, 0., limit)[0]
    except:
        result = -1
    return result

def confidence_interval(density, alpha, start):
    func = lambda tau: scipy.integrate.quad ( density, 0., tau )[0] -alpha
    try:
        result = scipy.optimize.fsolve (func, start)
    except:
        result = -1
    return result

class simulation(threading.Thread):
    def __init__(self, id, rtd):
        threading.Thread.__init__(self)
        self.rtd = rtd
        self.id = id
        self.rtd[self.id] = {}
        self.rtd[self.id]['hits'] = 0
        self.rtd[self.id]['misses'] = 0
        self.rtd[self.id]['errors'] = []
        self.rtd[self.id]['done'] = False
    def run(self):
        for j in range(int(n_samples/n_threads)):
            exception = False

            tau = scipy.stats.gamma.rvs(a, scale=b)
            #print(" tau_{}={}".format(j, tau))
            samples = []
            samples.append(0)

            # print("Ziehe n={} Werte aus Exponentialverteilung Ex(tau)".format(n))
            for i in range(n):
                samples[0] += scipy.stats.expon.rvs(scale=tau)
            s = samples[0]
            # print(" s={}".format(s))

            prior_scaled = lambda x: scipy.stats.gamma.pdf(x, a, scale=b)
            prior = lambda x: x ** (a - 1) * np.exp(-x / b) / b ** a
            factor = 1
            posterior = lambda tau: factor * getPosterior(tau, prior, likelyhood_ex, s, n)


            # BayesSchätzer & Erwartungswert
            posterior_expect = s / n
            # print("\n Bayes-Schätzer = {}".format(posterior_expect))
            #posterior_expect_numerical = expect(posterior, 0, 20)
            #if posterior_expect_numerical == -1:
                #self.rtd[self.id]['errors'].append([j, tau])
                #continue
            # print(" E[tau] = {}".format(posterior_expect_numerical))

            # normalize posterior:
            factor = normfactor(posterior, (posterior_expect * 2))
            if factor == -1:
                exception = True
            #posterior = lambda tau: factor * getPosterior(tau, prior, likelyhood_ex, s, n)

            #calculate KI
            if not exception:
                leftLimit = confidence_interval(posterior, 0.025, posterior_expect)
            if leftLimit == -1:
                exception = True
            if not exception:
                rightLimit = confidence_interval(posterior, 0.975, posterior_expect)
            if rightLimit == -1:
                exception = True
            # print("Symmetrisches 95% KI:")
            # print(" [{}, {}]".format(leftLimit, rightLimit))
            if exception:
                self.rtd[self.id]['errors'].append([j, tau])
            else:
                if tau <= rightLimit and tau >= leftLimit:
                    #print("tau liegt im Konfidenzintervall")
                    self.rtd[self.id]['hits'] += 1
                else:
                    # print("tau liegt nicht im Konfidenzitervall")
                    self.rtd[self.id]['misses'] += 1
        self.rtd[self.id]['done'] = True


if __name__ == '__main__':
    print("---------------------------------------------------------------\n")
    print("            Uebung 3_6 David Blacher, Johannes Kurz            \n")
    print("---------------------------------------------------------------\n")
    print("Ziehe tau aus a-priori Gammaverteilung mit a={} & b={}".format(a,b))
    print("Parallelisierung auf {} Threads\n".format(n_threads))
    hits = 0
    misses = 0
    errors = 0
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
            done_simulations += RTD[k]['hits']
            done_simulations += RTD[k]['misses']
            done_simulations += len(RTD[k]['errors'])
        print("durchgeführte Simulationen nach {} Sekunden: {}".format(int(time.time()-t_start),done_simulations))
        #print(RTD[0]['done'])
        #print(RTD[1]['done'])
        #print(RTD[2]['done'])
        #print(RTD[3]['done'])
        running = False
        for j in range(n_threads):
            if not RTD[j]['done']:
                #print("Thread {} arbeitet noch.".format(j))
                running = True
    for k in range(n_threads):
        hits += RTD[k]['hits']
        misses += RTD[k]['misses']
        errors += len(RTD[k]['errors'])
    #print(RTD)
    devisor = n_threads*(int(n_samples/n_threads))
    print("Das 95% Konfidenzintervall wurde {} mal getroffen.".format(hits))
    print("Das enstpricht {}%.".format(round(100*hits/devisor,1)))
    print("Das 95% Konfidenzintervall wurde {} mal verfehlt.".format(misses))
    print("Das enstpricht {}%.".format(round(100*misses / devisor, 1)))
    print("Es gab {} Berechnungsfehler".format(errors))
    print("Das enstpricht {}%.".format(round(100*errors / devisor, 1)))
    #print(times)
    ### plot
    #dataPoints = np.linspace ( 0.001, 15, 1000 )
    #plt.plot ( dataPoints, prior(dataPoints), label="prior"  )
    #plt.plot ( dataPoints, posterior(dataPoints), label="posterior"  )
    #plt.legend()
    #plt.title ( "3.6" )
    #plt.savefig ( "3_6.pdf" )
    #plt.show()
