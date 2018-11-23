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

n_threads = 1
n_samples = 5000
n = 100
a = 4
b = 1
alpha = 0.05



class simulation(threading.Thread):
    def __init__(self, id, rtd):
        threading.Thread.__init__(self)
        self.rtd = rtd
        self.id = id
        self.rtd[self.id] = {}
        self.rtd[self.id]['hits'] = 0
        self.rtd[self.id]['misses'] = 0
        self.rtd[self.id]['done'] = False
    def run(self):
        for j in range(int(n_samples/n_threads)):

            tau = scipy.stats.invgamma.rvs(a, scale=b)

            s = 0
            # print("Ziehe n={} Werte aus Exponentialverteilung Ex(tau)".format(n))
            for i in range(n):
                s += scipy.stats.expon.rvs(scale=tau)
            # print(" s={}".format(s))

            a_posterior = n+2
            b_posterior = s+1

            # BayesSchätzer & Erwartungswert
            posterior_expect = s / n
            # print("\n Bayes-Schätzer = {}".format(posterior_expect))

            #posterior = lambda tau: scipy.stats.invgamma.pdf(tau,
            #        a_posterior, scale=b_posterior)
            #posterior_expect_numerical = expect(posterior, 0, 20)
            # print(" E[tau] = {}".format(posterior_expect_numerical))

            #calculate KI
            leftLimit = scipy.stats.invgamma.cdf(alpha/2,
                    a_posterior, scale=b_posterior)
            rightLimit = scipy.stats.invgamma.cdf(1-alpha/2,
                    a_posterior, scale=b_posterior)
            # print("Symmetrisches 95% KI:")
            # print(" [{}, {}]".format(leftLimit, rightLimit))

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
    print("Ziehe {}-mal tau aus a-priori inversen Gammaverteilung mit a={} & b={}"
            .format(n_samples, a,b))
    print("a-posteriori Verteilung ist wieder eine inverse Gammaverteilung mit a=n+2 & b=s+1\n")
    hits = 0
    misses = 0
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
        print("durchgeführte Simulationen nach {} Sekunden: {}".format(int(time.time()-t_start),done_simulations))
        if RTD[0]['done']:# and RTD[1]['done'] and RTD[2]['done'] and RTD[3]['done']:
            running = False
    for k in range(n_threads):
        hits += RTD[k]['hits']
        misses += RTD[k]['misses']
    #print(RTD)
    print("Das 95% Konfidenzintervall wurde {} mal getroffen.".format(hits))
    print("Das enstpricht {}%.".format(round(100*hits/n_samples,1)))
    print("Das 95% Konfidenzintervall wurde {} mal verfehlt.".format(misses))
    print("Das enstpricht {}%.".format(round(100*misses / n_samples, 1)))
