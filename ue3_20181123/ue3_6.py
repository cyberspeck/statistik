# -*- coding: utf-8 -*-

#!/usr/bin/python3


import numpy as np
import scipy.stats, scipy.special
from matplotlib import pyplot as plt

## ziehen aus der gamma Verteilung: scipy.stats.gamma.rvs
## ziehen aus der Exponentialverteilung: scipy.stats.expon.rvs
## numerisches integrieren von f im Intervall von a bis b:
## scipy.integrate.quad( f,a, b )
## Numerisches Lösen eines Gleichungssystems:
## scipy.optimize.fsolve ( equations )

n_samples = 5000
n = 100
a = 4
b = 1
alpha = 0.05

def expect(f, a,b, exp=1):
    '''
    returns expected value of function f for range [a,b]
    '''
    integrand = lambda y: f(y)* y**exp
    norm = lambda y: f(y)
    norm_factor =  scipy.integrate.quad ( norm, a, b )[0]
    result = scipy.integrate.quad ( integrand, a, b )[0]
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

def confidence_interval(density, alpha):
    func = lambda tau: scipy.integrate.quad ( density, 0., tau )[0] -alpha
    result = scipy.optimize.fsolve ( func, x0=1 )
    return result



if __name__ == '__main__':
    print("---------------------------------------------------------------\n")
    print("            Uebung 3_6 David Blacher, Johannes Kurz            \n")
    print("---------------------------------------------------------------")

    print("Ziehe tau aus a-priori Gammaverteilung mit a={} & b={}".format(a,b))
    tau = scipy.stats.gamma.rvs(a, scale=b)
    print(" tau={}".format(tau))

    samples = []
    samples.append(0)
    print("Ziehe n={} Werte aus Exponentialverteilung Ex(tau)".format(n))
    for i in range(n):
        samples[0] += scipy.stats.expon.rvs(scale=tau)

    s = samples[0]
    print(" s={}".format(s))


    prior_scaled = lambda x: scipy.stats.gamma.pdf(x,a,scale=b)
    prior = lambda x: x**(a-1) * np.exp(-x / b) / b**a
    posterior = lambda tau: getPosterior(tau, prior, likelyhood_ex, s,n)

    posterior_expect = s/n
    print("\n Bayes-Schätzer = {}".format(posterior_expect))
    posterior_expect_numerical = expect(posterior, 0,20)
    print(" E[tau] = {}".format(posterior_expect_numerical))

    leftLimit = confidence_interval(posterior, 0.025)
    rightLimit = confidence_interval(posterior, 0.975)
    print("Symmetrisches 95% KI:")
    print(" [{}, {}]".format(leftLimit, rightLimit))

    ### plot
    dataPoints = np.linspace ( 0.001, 15, 1000 )
    plt.plot ( dataPoints, prior(dataPoints), label="prior"  )
    plt.plot ( dataPoints, posterior(dataPoints), label="posterior"  )
    plt.legend()
    plt.title ( "3.6" )
    #plt.savefig ( "3_6.pdf" )
    plt.show()

