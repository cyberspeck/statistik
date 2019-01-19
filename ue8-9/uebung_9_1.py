# -*- coding: utf-8 -*-
##!/usr/bin/python3

import time
import statistics
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt

data = [[150,2],[625,250]]

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

def regr_koeff_wiki(data_a, data_b):
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
    b_1 = b_1/denominator
    b_0 = avg_b-b_1*avg_a
    result = [b_0, b_1]
    return result

def regr_koeff_buch(data_a, data_b):
    if len(data_a) != len(data_b):
        raise ValueError('both data vectors must have the same length')
    avg_a = statistics.mean(data_a)
    avg_b = statistics.mean(data_b)
    denominator = 0
    for i in range(len(data_a)):
        denominator += data_a[i]**2 - len(data_a)*(avg_a**2)
    beta = 0
    for i in range(len(data_a)):
        beta += data_a[i]*data_b[i] - len(data_a)*avg_a*avg_b
    beta = beta/denominator
    alpha = avg_b - beta*avg_a
    result = [alpha, beta]
    return result

if __name__ == '__main__':
    t_start = time.time()
    print("---------------------------------------------------------------\n")
    print("            Uebung 9_1 David Blacher, Johannes Kurz            \n")
    print("---------------------------------------------------------------")
    data_x = []
    data_y = []
    for i in range(len(data)):
        data_x.append(data[i][0])
        data_y.append(data[i][1])
    #Streudiagramm:
    plt.scatter(data_x, data_y, c='r')
    x = np.linspace(125, 650, 1000)
    b = regr_koeff_wiki(data_x,data_y)
    greek = regr_koeff_buch(data_x, data_y)
    plt.xlabel("Anzahl Mitarbeiter")
    plt.ylabel("Ausbildungskosten in 1.000€")
    print("Der empirische Korrelationskoeffizient ist {}".format(korl_koeff(data_x, data_y)))
    print("Die Regressionsgerade hat die Form y = x*b_1+b_0")
    print("Berechnung nach Wikipedia:\nb_0 = {}\nb_1 = {}".format(b[0], b[1]))
    print("---------------------------------------------------------------")
    print("Berechnung nach dem Buch:\nb_0 = {}\nb_1 = {}".format(greek[0], greek[1]))
    print("---------------------------------------------------------------\n")

    t_fin = time.time()
    print("Finished in {} seconds.".format(t_fin-t_start))
    plt.show()