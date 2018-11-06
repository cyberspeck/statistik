# -*- coding: utf-8 -*-

""" Skeleton file, das eure Lernkurve beschleunigen soll,
    fuer Uebung 1.6.  
"""

## Importieren der wichtigsten Module fuer die Uebungsaufgabe.
import numpy, scipy.misc, math, scipy.integrate

## scipy.misc.comb ( n, k )  ## n ueber k = n! / ( k! (n-k)! )
## math.sin ( math.pi * x ) ## sin ( pi * x )

## lambda Funktionen erlauben Definitionen in einer einzigen Zeile
## prior = lambda p: p**2 * ( 1 - p )**3  ## prior(.5) = .03125

## Integral von f ueber das Intervall (a,b)
## scipy.integrate.quad ( f, a, b ) 

## Teile das Intervall (a,b) in n "bins"
## numpy.linspace ( a, b, n )

## Plotten, siehe https://matplotlib.org/
import matplotlib.pyplot as plt

## Plotte y versus x, als Linie, mit Label "label1"
plt.plot ( [1., 2., 5. ], [3, 1., 4. ], label="label1"  ) 
plt.title ( "titel" )

## Beschriftung der x-Achse
plt.xlabel("x axis")

## Legende
plt.legend()

## Speichere in Bilddatei (pdf, png, ... )
#plt.savefig ( "dummy.pdf" )
plt.show()
