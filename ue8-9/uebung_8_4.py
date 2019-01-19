# -*- coding: utf-8 -*-
##!/usr/bin/python3

import time
import statistics
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt

data = [[165,54.6],[89,53.3],[55,56.3],[34,49.6],[9,47.1],[30,45.9],[59,48.5],[83,50.1],[109,52.4],[127,52.5],[153,53.2],[112,51.4],[80,46.0],[45,44.6]]

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
    b_1 = b_1/denominator
    b_0 = avg_b-b_1*avg_a
    result = [b_0, b_1]
    return result
################################################################
if __name__ == '__main__':
    t_start = time.time()
    print("---------------------------------------------------------------\n")
    print("            Uebung 8_4 David Blacher, Johannes Kurz            \n")
    print("---------------------------------------------------------------")
    data_x = []
    data_y = []
    for i in range(len(data)):
        data_x.append(data[i][0])
        data_y.append(data[i][1])
    #Streudiagramm:
    plt.scatter(data_x, data_y, c='r')
    x = np.linspace(0, 170, 1000)
    b = regr_koeff(data_x,data_y)
    plt.plot(x, x*b[1]+b[0], '-b')
    plt.xlabel("Anzahl Sonnenflecken")
    plt.ylabel("Anzahl Verkehrstote in 1.000")
    print("Der empirische Korrelationskoeffizient ist {}".format(korl_koeff(data_x, data_y)))
    print("Ein Korrelationskoeffizient von deutlich über 0 bedeutet, dass eine Korrelation durchausgegeben ist. Da\n"
          "Verkehrsunfälle primär auf menschliches Versagen / Fehlverhalten zurückzuführen sind Sonnenfelcken die vom\n"
          "Menschen nicht wahrnehmbar sind, als Kausalitätsfaktor mEn auszuschließen.")
    print("---------------------------------------------------------------\n")
    t_fin = time.time()
    print("Finished in {} seconds.".format(t_fin-t_start))
    plt.show()