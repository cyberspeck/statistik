# -*- coding: utf-8 -*-
##!/usr/bin/python3

import time
import statistics
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt

data = [[5,9],[10,10],[15,14],[20,18],[25,22],[30,24],[40,29],[50,29]]
#data = [[10,10], [30,30], [50,50]] #testdata für die perfekte korrelation
km = 35 #gefahrene kilometer in tausend

def korl_koeff(data_a, data_b):
    if len(data_a) != len(data_b):
        raise ValueError('both data vectors must have the same length')
    avg_a = statistics.mean(data_a)
    avg_b = statistics.mean(data_b)
    den_a = 0
    den_b = 0
    for i in range(len(data_a)):
        den_a += (data_a[i]-avg_a)**2
        den_b += (data_b[i]-avg_b)**2
    denominator = sqrt(den_a*den_b)
    r = 0
    for i in range(len(data_a)):
        r += (data_a[i]-avg_a)*(data_b[i]-avg_b)
    r = r/denominator
    return r

def korl_koeff_buch(data_a, data_b):
    #der im Buch beschriebene korr koeff unterscheidet sich zum wikipedia eintrag durch diesen Vorfaktor.
    #Allerdings kann mit diesem Vorfaktor ein korrl koeff von 1 nie erreicht werden
    n = len(data_a)
    r = korl_koeff(data_a, data_b)
    r = r*(n-1)/n
    return r


def regr_koeff(data_a, data_b):
    if len(data_a) != len(data_b):
        raise ValueError('both data vectors must have the same length')
    avg_a = statistics.mean(data_a)
    avg_b = statistics.mean(data_b)
    denominator = 0
    for i in range(len(data_a)):
        denominator += (data_a[i]-avg_a)**2
    b_1 = 0
    for i in range(len(data_a)):
        b_1 += (data_a[i]-avg_a)*(data_b[i]-avg_b)
    b_1 = round(b_1/denominator,4)
    b_0 = round(avg_b-b_1*avg_a,4)
    result = [b_0, b_1]
    return result

def regr_koeff_buch(x, y):
    n = len(x)
    if len(y) != n:
        raise ValueError('both data vectors must have the same length')
    avg_x = statistics.mean(x)
    avg_y = statistics.mean(y)
    denominator = 0
    for i in range(n):
        denominator += (x[i]**2)-(n*(avg_x**2))
    beta = 0
    for i in range(n):
        beta += (x[i]*y[i])-(n*avg_x*avg_y)
    beta = beta / denominator
    alpha = avg_y - (beta*avg_x)
    result = [alpha, beta]
    return result

if __name__ == '__main__':
    t_start = time.time()
    print("---------------------------------------------------------------\n")
    print("            Uebung 8_3 David Blacher, Johannes Kurz            \n")
    print("---------------------------------------------------------------")
    data_x = []
    data_y = []
    for i in range(len(data)):
        data_x.append(data[i][0])
        data_y.append(data[i][1])
    #Streudiagramm:
    plt.scatter(data_x, data_y, c='r')
    x = np.linspace(5, 55, 1000)
    b = regr_koeff(data_x,data_y)
    plt.plot(x, x*b[1]+b[0], '-b')
    #greek = regr_koeff_buch(data_x, data_y)
    #

    plt.plot(x, x*greek[1]+greek[0], '-g')
    plt.xlabel("jährliche Fahrleistung in 1.000km")
    plt.ylabel("Schadenshäufigkeit in Promille")
    print("Der empirische Korrelationskoeffizient ist {}".format(korl_koeff(data_x, data_y)))
    print("Die Regressionsgerade hat die Form y = x*b_1+b_0")
    print("b_0 = {}\nb_1 = {}".format(b[0], b[1]))
    print("Ein Fahrerix mit {}.000km hat eine Unfallwahrscheinlichkeit von etwa {} Promille".format(km, round(km*b[1]+b[0])))
    t_fin = time.time()
    print("Finished in {} seconds.".format(t_fin-t_start))
    plt.show()