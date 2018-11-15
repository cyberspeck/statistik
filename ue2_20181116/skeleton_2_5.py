#!/usr/bin/env python3

""" Skeleton file, das eure Lernkurve beschleunigen soll,
    fuer Uebung 2.5.  
"""

import numpy, scipy.misc, math, scipy.integrate, scipy.stats
from matplotlib import pyplot as plt

## Diese Variablen koennt ihr vermutlich gut gebrauchen
n=60
S=124

## Verwende scipy.stats.poisson.pmf fuer die Wahrscheinlichkeitsfunktion
## ("probability mass function") von Poisson-verteilten Variablen
## Verwende scipy.integrate.quad(f,a,b) um eine Funktion numerisch
## im Bereich von a bis b zu integrieren.
## Das HPD Interval ist schnell selbst implementiert. Gerne kann
## aber auch pymc3.stats.hpd verwendet werden (ist aber nicht wirklich einfacher).

## Fuers plotten:
## plt.plot, plt.xlabel, plt.legend, plt.savefig 

## Gutes Gelingen!
