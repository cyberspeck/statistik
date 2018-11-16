# -*- coding: utf-8 -*-

##!/usr/bin/env python3

""" Skeleton file, das eure Lernkurve beschleunigen soll,
    fuer Uebung 2.5.
"""
import sys
import numpy, scipy.misc, math, scipy.integrate, scipy.stats, scipy.optimize
from matplotlib import pyplot as plt
#import pymc3

n=60
s=124

def likelyhood_funct(l):
    return (l ** s) * numpy.exp(-n * l)

def find_max(func, a, b, precision = 6):
    max = 0
    while a < b:
        if func(a) > max:
            max = func(a)
            pos_max = a
        a += 1/(10**precision)
    return round(pos_max, precision)

def hpd_calc_symmetrical(func, alpha, precision = 0.001, start = 0):
    integral = 0
    if alpha >= 1 or alpha <= 0:
        raise ValueError('alpha must be between 0 and 1. It is {}'.format(alpha))
    limit = (1-alpha)+ (alpha/2)
    hpd = [-1,-1]
    x = start
    while integral < limit:
        integral += func(x)*precision
        if hpd[0] == -1:
            if integral >= alpha/2:
                hpd[0] = round(x,3)
        x += precision
    hpd[1] = round(x, 3)
    return hpd

if __name__ == '__main__':
    print("---------------------------------------------------------------\n")
    print("            Uebung 2_5 David Blacher, Johannes Kurz            \n")
    print("---------------------------------------------------------------")
    print("prior: \\pi(\\lambda) \\propto \\frac{1}{1+0.25*\\lambda}")
    prior = lambda l: (1/(1+0.25*l))
    likelyhood = lambda l: likelyhood_funct(l)
    prior_x_likelyhood = lambda l: likelyhood_funct(l)*prior(l)
    test_func = lambda x: x**2
    denominator = scipy.integrate.quad(prior_x_likelyhood, 0, 305)
    #denominator = scipy.integrate.quad(denominator, 0, numpy.inf)  #führt zu overflow innerhalb der funktion
    posterior = lambda l: prior_x_likelyhood(l)/denominator[0]
    #normieren:
    norm_factor = scipy.integrate.quad(posterior,1 ,300)[0]
    posterior = lambda l: prior_x_likelyhood(l)/(denominator[0]*norm_factor)
    bayes = find_max(posterior, 1.8, 2.2)
    print("Bayesian Expected Value: {}".format(bayes))
    #print(scipy.integrate.quad(posterior, 1, 300))
    print(hpd_calc_symmetrical(posterior, 0.05))
    ### plot
    dataPoints = numpy.linspace(0, 4, 1000)
    plt.plot(dataPoints, prior(dataPoints), label="prior")
    #plt.plot(dataPoints, prior(dataPoints), label="likelihood")
    plt.plot(dataPoints, posterior(dataPoints), label='postrior')
    # plt.plot ( poA_expect_numerical, poA(poA_expect_numerical), 'rx', label="E[posteriorA]"  )
    plt.legend()
    plt.title("Uebung 2.5 Poisson")
    plt.savefig("uebung_2_6.pdf")
    plt.show()





# Verwende scipy.stats.poisson.pmf fuer die Wahrscheinlichkeitsfunktion
# ("probability mass function") von Poisson-verteilten Variablen
# Verwende scipy.integrate.quad(f,a,b) um eine Funktion numerisch
# im Bereich von a bis b zu integrieren.
# Das HPD Interval ist schnell selbst implementiert. Gerne kann
# aber auch pymc3.stats.hpd verwendet werden (ist aber nicht wirklich einfacher).

# Fuers plotten:
# plt.plot, plt.xlabel, plt.legend, plt.savefig

# Gutes Gelingen!