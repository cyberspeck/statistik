# -*- coding: utf-8 -*-
##!/usr/bin/python3

import time
import statistics
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    t_start = time.time()
    print("-----------------------------------------------------------------\n")
    print("             Uebung 9_2 David Blacher, Johannes Kurz             \n")
    print("-----------------------------------------------------------------")
    print("Es soll ein Ablauf mit folgenden Massenströmen beobachtet werden:")
    print("                                                                 ")
    print("             +-----------+             +-----------+             ")
    print(">--- X_1 --->|           |>--- X_3 --->|           |             ")
    print("             | Process_1 |             | Process_2 |>--- X_5 --->")
    print(">--- X_2 --->|           |<--- X_4 ---<|           |             ")
    print("             +-----------+             +-----------+             ")
    print("                                                                 ")
    print("-----------------------------------------------------------------")

    print("-----------------------------------------------------------------\n")
    t_fin = time.time()
    print("Finished in {} seconds.".format(t_fin-t_start))