# -*- coding: utf-8 -*-
##!/usr/bin/python3

import time
import numpy as np
import scipy.stats, scipy.special
from matplotlib import pyplot as plt

if __name__ == '__main__':
    print("---------------------------------------------------------------\n")
    print("            Uebung 4_4 David Blacher, Johannes Kurz            \n")
    print("---------------------------------------------------------------")
    dichte = lambda x: np.exp(-x)
    sum = 0
    current = 0
    oben = 0.65
    while current < 0.5:
        sum = scipy.integrate.quad(dichte, 0, oben)
        oben += 0.0000001
        current = float(sum[0])
    print(oben)
    print(sum)
