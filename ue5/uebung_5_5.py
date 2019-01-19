# -*- coding: utf-8 -*-
##!/usr/bin/python3

import time
import math
from math import sqrt
import numpy as np
import scipy.stats, scipy.special
from matplotlib import pyplot as plt

n_of_samples = 500
n = 400
mu = [2,1]
v = [[0.8, 0.2],[0.2,0.6]]

def mean(list):
    mean = 0
    for i in range(len(list)):
        mean += list[i]
    mean = mean/len(list)
    return mean

def sampling_variance(samples1, samples2=None):
    if not samples2:
        samples2 = samples1
        #print("das war jetzt aber sehr eindimensional")
    if len(samples1) != len(samples2):
        raise ValueError('length of the two sample lists must be the same!')
    mean1 = mean(samples1)
    mean2 = mean(samples2)
    s = 0
    for i in range(len(samples1)):
        s += (samples1[i]-mean1)*(samples2[i]-mean2)
    return s
if __name__ == '__main__':
    time_start = time.time()
    print("---------------------------------------------------------------\n")
    print("            Uebung 5_5 David Blacher, Johannes Kurz            \n")
    print("---------------------------------------------------------------")
    print("Es ist der {}.\nWas für ein schöner Moment um Daten statistisch zu analysieren.".format(time.asctime(time.localtime(time_start))))
    print("Behandelt wird eine bivariante Normalverteilung mit einem Korrelationskoeffizienten != 0")
    print("Es werden {} Stichproben mit einem Umfang von jeweils {} gemacht\n".format(n_of_samples, n))
    korkoef = []
    for i in range(n_of_samples):
        samples_x = []
        samples_y = []
        for i in range(n):
            x, y = np.random.multivariate_normal(mu,v)
            samples_x.append(x)
            samples_y.append(y)
        korkoef.append(sampling_variance(samples_x, samples_y)/sqrt(sampling_variance(samples_x, samples_x)*sampling_variance(samples_y,samples_y)))
    korkoef_mean = mean(korkoef)
    print("Mean of koroef: {}".format(korkoef_mean))
    print("Varianz der korkoeff: {}".format(sampling_variance(korkoef, korkoef)))
    #Histogramm:
    korkoef = np.asarray(korkoef)
#    plt.xlim([min(korkoef) - 0.05, max(korkoef) + 0.05])
    plt.hist(korkoef)
    plt.title('Histogramm für den Korrelationskoeffizienten')
    plt.xlabel('Wert')
    plt.ylabel('absolute Häufigkeit')
    plt.show()

    time_finish = time.time()
    print("This operation took {}s".format(int(time_finish-time_start)))