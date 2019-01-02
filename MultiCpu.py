# main.py
import multiprocessing
import time
import numpy as np
# from func import writeln
# from calc import calc
import scipy.io as sio

def MultiCpuRun(function,number_cpu):
    pool = multiprocessing.Pool(processes=number_cpu)

    for i in range(number_cpu):
        pool.apply_async(function, (i+1, ))
        # pool.apply_async(())
    pool.close()
    pool.join()

    print("Sub-process(es) done.")
