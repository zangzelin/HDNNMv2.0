from gurobipy import *
import math
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import matplotlib as mpl
import numpy as np


def main(M, R, A, m, n):

    modle = Model("diet")

    var_Choose = []
    var_ChoosewithMachine = []

    for i in range(m*n):
        citem = []
        for o in range(n):
            citem.append(modle.addVar(vtype=GRB.BINARY))
        var_Choose.append(citem)

    for ma in range(m):
        var1 = []
        for i in range(m*n):
            citem = []
            for o in range(n):
                citem.append(modle.addVar(vtype=GRB.BINARY))
            var1.append(citem)

        var_ChoosewithMachine.append(var1)

    for i in range(m*n):
        for o in range(n):
            modle.addConstr(var_Choose[i][o] -
                            var_ChoosewithMachine[R[i]][i][o] == 0)

    for i in range(m*n):
        modle.addConstr(quicksum([var_Choose[i][o] for o in range(n)]) == 1)

    for ma in range(m):
        for o in range(n):
            modle.addConstr(
                quicksum(
                    [var_ChoosewithMachine[ma][i][o]
                     for i in range(n*m)]
                ) == 1
            )

    iando = []
    for i in range(n*m):
        for o in range(n):
            iando.append([i, o])

    modle.setObjective(quicksum(
        [var_Choose[i][o] * M[i][o] for i, o in iando]
    ), GRB.MAXIMIZE)
    modle.setParam( 'OutputFlag', False )
    modle.optimize()

    out = []
    count = 0

    x = np.zeros((m, n,2),dtype=int)
    for i, o in iando:
        var_Choose[i][o] = var_Choose[i][o].x
        if var_Choose[i][o] == 1:
            # print('job {} property {} machine {} order {} aim {}'.format(
            #     i//m, i % m, R[i], o, A[i]))
            if A[i] == o:
                count += 1
            x[R[i], o, 0] = i // m
            x[R[i], o, 1] = i % m

    return x,count/n/m


# if __name__ == "__main__":
#     main()
