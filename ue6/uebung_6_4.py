# -*- coding: utf-8 -*-
##!/usr/bin/python3

import time
import math as m
from math import sqrt
import numpy as np
import scipy.stats, scipy.special
from matplotlib import pyplot as plt

a_real=2
n_samples=500
n_repeats=500
n_bins = 10

def density(x, a=a_real):
    #if x < 0:
    #    raise ValueError("x has to be >= 0. It is: {}".format(x))
    upper = (x**(a-1)*m.exp(-x))
    print(scipy.stats.gamma(a))
    result = upper/scipy.stats.gamma(a)
    return result

def abweichungsquadratsumme(points, func):
    differences = []
    for i in range(len(points)):
        x = points[i][0]
        differences.append((abs(func(x)-points[i][1]))**2)
    return sum(differences)

if __name__ == '__main__':
    time_start = time.time()
    print("---------------------------------------------------------------\n")
    print("            Uebung 6_4 David Blacher, Johannes Kurz            \n")
    print("---------------------------------------------------------------")
    samples = []
    for i in range(n_samples):
        samples.append(np.random.gamma(2))
    func_real = lambda y: density(y, a_real)
    #Startpunkt für die optimierung
    a_approx = sum(samples)/ len(samples)
    print("Stichprobenmittel:{}".format(a_approx))
    func_approx = lambda y: density(y, a_approx)
    samples.sort()
    #print(samples)
    max = samples[-1]
    bins = []
    points = []
    binsize = max/n_bins
    for i in range(n_bins):
        bins.append([])
        for j in range(n_samples):
            if samples[j] > i*binsize:
                bins[i].append(samples[j])
                if i > 0:
                    bins[i-1].remove(samples[j])
    for i in range(n_bins):
        points.append([(i+0.5)*binsize,len(bins[i])])
    '''
    Was fehlt:
    
    berechnen des abweichungsquadrats und dann die veränderung des parameters a. (+1 und test ob abwqu kleiner wurde, dann -1).
    Wenn in beide Richtungen die Abweichung zunimmt veränderung des parameters in kleineren Schritten.
    
    Ebenfalls muss auf die normierung rücksicht genommen werden, also eine annäherung auf 2 Achsen.
    '''


    print(abweichungsquadratsumme(points, func_approx))
    #print(points)
    #print(bins)



    #Was zum anschauen:
    plt.hist(samples)
    plt.show()