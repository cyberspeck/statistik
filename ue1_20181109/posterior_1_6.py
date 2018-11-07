# -*- coding: utf-8 -*-

"""
exercise 1.6.

find posterior for
 a) prior = p^2(1-p)^3
 b) prior = sin^2(pi*p)

"""

import numpy as np
import scipy.misc, scipy.integrate
import matplotlib.pyplot as plt

def expect(f, a,b, exp=1):
    '''
    returns expected value of function f for range [a,b]
    '''
    integrand = lambda y: f(y)* y**exp
    result = scipy.integrate.quad ( integrand, a, b )[0]
    return result

def variance(f, expected_value, a,b):
    '''
    returns expected value of function f for range [a,b]
    '''
    result = expect(f, a,b,2) - expect(f, a,b)**2
    return result

def expect_of_beta(a,b):
    '''
    returns expected value of beta function Be(a,b)
    '''
    return (a)/(a+b)

def variance_of_beta(a,b):
    '''
    returns variance of beta function Be(a,b)
    '''
    return (a*b)/( (a+b)**2 *(a+b+1) )

def likely_bernoulli(p, k,n):
    '''
    returns maximum likelyhood for bernoulli
    '''
    nCk = scipy.special.comb ( n, k )
    be = p**k * ( 1 - p )**(n-k)
    return nCk * be

def posterior(p, prior, likely, k,n):
    numerator = lambda p: likely(p, k,n) * prior(p)
    nominator = lambda p: scipy.integrate.quad ( numerator, 0, 1 )[0]
    result = numerator(p) / nominator(p)
    return result


n=1000
k=343
nCk = scipy.special.comb ( n, k )


print("\nBeispiel 1.6")
print("Ein Bernoulli-Experiment wird n-mal wiederholt mit k Erfolgen")
print("Gesucht ist die a-posteriori Dichte, Bayes-Schätzer und Varianz")
print("---------------------------------------------------------------\n")

print("\na) prior = p^2(1-p)^3 ~ Be(3,4)")
prA_a = 3
prA_b = 4
prA = lambda p: p**(prA_a-1)* ( 1 - p )**(prA_b-1)

poA = lambda p: posterior(p, prA, likely_bernoulli, k,n)
print(" -> posterior ~ Be(k+3,n-k+4)")
poA_a =   k + prA_a
poA_b = n-k + prA_b
print("Erwartungswert E[Be(x,y)] = x/(x+y)")
poA_expect = expect_of_beta(poA_a, poA_b)
print(" E = {}".format(poA_expect))
poA_expect_numerical = expect(poA, 0,1)
print("    Kontrolle mit klassischer Formel\n E = {}".format(poA_expect_numerical))
deltaE = abs(poA_expect - poA_expect_numerical)
print("    ausreichende Übereinstimmung:\n delta = {}".format(deltaE))

print("\nVarianz var[Be(x,y)] = xy/((x+y)^2 * (x+y+1))")
poA_variance = variance_of_beta(poA_a, poA_b)
print(" var = {}".format(poA_variance))
poA_variance_numerical = variance(poA, poA_expect_numerical, 0,1)
print("    Kontrolle mit klassischer Formel\n var = {}".format(poA_variance_numerical))
deltaVar = abs(poA_variance - poA_variance_numerical)
print("    ausreichende Übereinstimmung:\n delta = {}".format(deltaVar))

### plot
dataPoints = np.linspace ( 0, 1, 1000 )
plt.plot ( dataPoints, prA(dataPoints), label="priorA"  )
plt.plot ( dataPoints, poA(dataPoints), label="posteriorA"  )
#plt.plot ( poA_expect_numerical, poA(poA_expect_numerical), 'rx', label="E[posteriorA]"  )
plt.legend()
plt.title ( "1.6 a)" )
plt.savefig ( "1_6_a.pdf" )
plt.show()

# --------------------#

print("\nb) prior = sin^2(pi*p)")
prB = lambda p: (np.sin ( np.pi*p ))**2

poB = lambda p: posterior(p, prB, likely_bernoulli, k,n)
poB_expect_numerical = expect(poB, 0,1)
poB_variance_numerical = variance(poB, poB_expect_numerical, 0,1)
print("Mit klassischer Formeln:\n E = {}".format(poB_expect_numerical))
print(" var = {}".format(poB_variance_numerical))

### plot
plt.plot ( dataPoints, prB(dataPoints), label="priorB"  )
plt.plot ( dataPoints, poB(dataPoints), label="posteriorB"  )
#plt.plot ( poB_expect_numerical, poB(poB_expect_numerical), 'rx', label="E[posteriorB]"  )
plt.legend()
plt.title ( "1.6 b)" )
plt.savefig ( "1_6_b.pdf" )
plt.show()

