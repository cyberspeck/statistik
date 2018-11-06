# -*- coding: utf-8 -*-

""" Skeleton file, das eure Lernkurve beschleunigen soll,
    fuer Uebung 1.6.  
"""

## Importieren der wichtigsten Module fuer die Uebungsaufgabe.
import numpy, scipy.misc, math, scipy.integrate
import matplotlib.pyplot as plt

def expect(a, b, k, n):
    return (k+a)/(a+b+n)

def likely_bernoulli(p, k, n):
    nCk = scipy.special.comb ( n, k )
    be = p**k * ( 1 - p )**(n-k)
    return nCk * be


n=1000
k=343
nCk = scipy.special.comb ( n, k )

# find posterior for priors
# a) = p^2(1-p)^3
# b) = sin^2(pi*p)


# a)
# prior ~ Be(3,4)
pr_a = 3
pr_b = 4
pr = lambda p: p**(pr_a-1)* ( 1 - p )**(pr_b-1)
# posterior ~ Be(k+a,n-k+b)
po_a = k + pr_a
po_b = n-k + pr_b

po_expect = expect(po_a, po_b, k, n)

numerator = lambda p: likely_bernoulli(p, k,n) * p**(pr_a-1)* ( 1 - p )**(pr_b-1)
nominator = lambda p: scipy.integrate.quad ( numerator, 0, 1 )[0]

po = lambda p: numerator(p) / nominator(p)
