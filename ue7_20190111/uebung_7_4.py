# -*- coding: utf-8 -*-
##!/usr/bin/python3

import time
import statistics


import math as m
from math import sqrt
import numpy as np
import scipy.stats, scipy.special
from matplotlib import pyplot as plt

groups = 5
exp_par = 1/3 #lambda not tau!!
accuracy = 0.1 #auf wie viele nachkommastellen der algorythmus suchen soll
step_limit = 0.0000001

def abstandsquadrat(vector1, vector2):
    if len(vector1) != len(vector2):
        print(len(vector1))
        print(len(vector2))
        raise ValueError
    abs_sq = 0
    for i in range(len(vector1)):
        abs_sq += (vector1[i]-vector2[i])**2
    return abs_sq

if __name__ == '__main__':
    t_start = time.time()
    print("---------------------------------------------------------------\n")
    print("            Uebung 7_4 David Blacher, Johannes Kurz            \n")
    print("---------------------------------------------------------------")
    #einlesen der daten
    with open('block7.txt', 'r') as f:
        f_contents = f.read()
    seperator = f_contents[10]
    data = []
    current = ""
    #Teilen der Daten
    for i in range(len(f_contents)):
        if f_contents[i] != seperator:
            current += f_contents[i]
        if f_contents[i] == seperator:
            data.append(current)
            current = ""
        if i == len(f_contents)-1:
            data.append(current)
    #aufbereitung
    for i in range(len(data)):
        data[i] = data[i].strip()
        data[i] = float(data[i])
    data.sort()
    #print(data)
    #print(type(data[1]))
    #print(len(data))
    #In 5 Gruppen aufteilen:
    data_grouped = []
    for i in range(groups):
        data_grouped.append([])
    val_per_group = len(data)/groups
    if val_per_group == int(val_per_group):
        for i in range(int(val_per_group)):
            for j in range(groups):
                data_grouped[j].append(data[i+int(val_per_group*j)])
    #print(data_grouped)
    medians = []
    for i in range(groups):
        medians.append(statistics.median(data_grouped[i]))
    #print(medians)
    quantil_size = 1 / groups
    p = quantil_size/2
    quantils = []
    while p <= 1:
        #print(p)
        quantils.append((-np.log(1-p))/exp_par)
        p += quantil_size
    #print(quantils)
    print("Mediane samt zugehörigen Quantilen:")
    for i in range(groups):
        print("{}, {}".format(medians[i], quantils[i]))
    chi_squart = 0
    for i in range(groups):
        #quantils[i] = float(quantils[i])
        chi_squart += (medians[i]-quantils[i])**2/quantils[i]
    print("Chi-Quadrat Wert ist: {}".format(chi_squart))
    print("---------------------------------------------------------------")
    #methode der kleinsten abstandsquadrate:
    #Startwert: medians[4] (sicher größer als tau)
    tau = medians[groups-1]
    quantils = []
    p = quantil_size/2
    while p <= 1:
        #print(p)
        quantils.append((-np.log(1-p))/(1/tau))
        p += quantil_size
    #print(quantils)
    step = 1
    abs_sq = abstandsquadrat(quantils, medians)
    while abstandsquadrat(quantils, medians) > accuracy:
        tau_last = tau
        abs_sq_last = abs_sq
        tau = tau - step
        p = quantil_size / 2
        quantils = []
        while p <= 1:
            # print(p)
            quantils.append((-np.log(1 - p)) / (1 / tau))
            p += quantil_size
        abs_sq = abstandsquadrat(quantils, medians)
        if abs_sq >= abs_sq_last:
            step = step/10
            tau = tau_last
            abs_sq = abs_sq_last
            print(step)
        print(abstandsquadrat(quantils, medians))
        #time.sleep(0.1)
        if step <= step_limit:
            break
    print("---------------------------------------------------------------")
    print("Der gefundene Parameter tau = {}".format(tau))
    print("Das verbleibende Abstandsquadrat ist: {}".format(abstandsquadrat(quantils,medians)))
    t_fin = time.time()
    print("Finished in {} seconds.".format(t_fin-t_start))