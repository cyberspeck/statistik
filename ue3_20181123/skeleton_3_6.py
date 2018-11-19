#!/usr/bin/python3

""" Skeleton file, das eure Lernkurve beschleunigen soll,
    fuer Uebung 3.6.  
"""

import numpy as np
import scipy.stats, scipy.special
from matplotlib import pyplot as plt

## ziehen aus der gamma Verteilung: scipy.stats.gamma.rvs
## ziehen aus der Exponentialverteilung: scipy.stats.expon.rvs
## numerisches integrieren von f im Intervall von a bis b:
## scipy.integrate.quad( f,a, b )
## Numerisches Lösen eines Gleichungssystems:
## scipy.optimize.fsolve ( equations )

## ein paar nuetzliche Konstanten
n_samples = 5000
n = 100
a = 4
b = 1
alpha = 0.05
