import copy
import random
import time
# from __future__ import print_function
from itertools import combinations, permutations

# import guchoose

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Import Python wrapper for or-tools constraint solver.
from ortools.constraint_solver import pywrapcp

import Subproblem


class Problem:
    m = None  # number of the machines
    n = None  # number of the jobs
    solute = None
    time_low = None
    time_high = None
    p = np.array([])  # the processing time of jobs
    r = np.array([])  # the order limit
    x = np.array([])  # the result position mat
    h = np.array([])  # the start time of jobs
    e = np.array([])  # the end time of jobs
    f = np.array([])
    best_x = np.array([])
    opetimalx = None
    number_of_1d_feature = 11
    # p.sum()

    def __init__(self, m, n, time_low, time_high):
        self.m = m
        self.n = n
        self.time_high = time_high
        self.time_low = time_low
        self.solute = 0

        a = list(range(self.time_low, self.time_high))
        p = []
        for k in range(self.n):
            p.append(random.sample(a, self.m))
        self.p = np.array(p)

        a = list(range(self.m))
        r = []
        for k in range(self.n):
            r.append(random.sample(a, self.m))
        self.r = np.array(r)

        sum_time_of_job = np.sum(self.p,axis=1)

        for i in range(n):
            for j in range(i+1, n):
                if sum_time_of_job[i] > sum_time_of_job[j]:
                    a = np.copy(self.p[j,:])
                    self.p[j,:] = self.p[i,:]
                    self.p[i,:] = a
                    sum_time_of_job[i],sum_time_of_job[j] = sum_time_of_job[j],sum_time_of_job[i] 
        
        sum_time_of_mach = [[i,0] for i in range(m)]
        for i in range(n):
            for j in range(m):
                sum_time_of_mach[self.r[i,j]][1] += self.p[i,j]
        
        for i in range(m):
            for j in range(i+1, m):
                if sum_time_of_mach[i][1] > sum_time_of_mach[j][1] :
                    sum_time_of_mach[i], sum_time_of_mach[j] = sum_time_of_mach[j], sum_time_of_mach[i]

        nr = np.zeros((n,m),dtype=int)-1
        for i in range(m):
            nr[self.r==i] =sum_time_of_mach[i][0]

        sum_time_of_mach = [[i,0] for i in range(m)]
        for i in range(n):
            for j in range(m):
                sum_time_of_mach[self.r[i,j]][1] += self.p[i,j]    

        self.r = nr

    def Print_info(self):

        machine_job_p = np.zeros((self.m, self.n))
        machine_job_r = np.zeros((self.m, self.n))

        for job in range(self.n):
            for order in range(self.m):
                machine = self.r[job, order]
                machine_job_p[machine, job] = self.p[job, order]
                machine_job_r[machine, job] = order

        np.savetxt('p.csv', machine_job_p, delimiter=',')
        np.savetxt('r.csv', machine_job_r, delimiter=',')
        self.PlotResult()

    def SaveProblemToFile(self, filepath, index, pool=0):

        filename = '{}/jssp_problem_m={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
            filepath, self.m, self.n, self.time_high, self.time_low, pool)
        f = open(filename, 'a')
        # f.write(str(index))
        # f.write('\nr=\n')
        f.write(str(self.m)+'\n')
        f.write(str(self.n)+'\n')
        f.write(TranslateNpToStr(self.p))
        f.write(TranslateNpToStr(self.r))
        f.close()

    def SavesolutionToFile(self, filepath, index, pool=0):
        f = open('{}/jssp_problem_m={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
            filepath, self.m, self.n, self.time_high, self.time_low, pool), 'a')
        # f.write(str(index)+'\n')
        f.write(TranslateNpToStr(self.x))
        f.write(TranslateNpToStr(self.h))
        f.write(TranslateNpToStr(self.e))
        f.close()

    def SoluteWithBBM(self):

        solver = pywrapcp.Solver('jobshop')
        solver.TimeLimit(1)

        all_machines = range(0, self.m)
        all_jobs = range(0, self.n)

        x = np.zeros((self.m, self.n, 2), dtype=int)
        h = np.zeros((self.m, self.n), dtype=int)
        e = np.zeros((self.m, self.n), dtype=int)

        # processing_times = self.p.tolist()
        # machines = self.r.tolist()

        horizon = int(self.p.sum())
        # Creates jobs.
        all_tasks = {}
        for i in all_jobs:
            for j in range(self.m):
                all_tasks[(i, j)] = solver.FixedDurationIntervalVar(
                    0,  horizon, int(self.p[i, j]), False, 'Job_%i_%i' % (i, j))

        # Creates sequence variables and add disjunctive constraints.
        all_sequences = []
        all_machines_jobs = []
        for i in all_machines:

            machines_jobs = []
            for j in all_jobs:
                for k in range(self.m):
                    if self.r[j, k] == i:
                        machines_jobs.append(all_tasks[(j, k)])
            disj = solver.DisjunctiveConstraint(
                machines_jobs, 'machine %i' % i)
            all_sequences.append(disj.SequenceVar())
            solver.Add(disj)

        # Add conjunctive contraints.
        for i in all_jobs:
            for j in range(0, self.m - 1):
                solver.Add(
                    all_tasks[(i, j + 1)].StartsAfterEnd(all_tasks[(i, j)]))

        # Set the objective.
        obj_var = solver.Max([all_tasks[(i, self.m-1)].EndExpr()
                              for i in all_jobs])
        objective_monitor = solver.Minimize(obj_var, 1)
        # Create search phases.
        sequence_phase = solver.Phase([all_sequences[i] for i in all_machines],
                                      solver.SEQUENCE_DEFAULT)
        vars_phase = solver.Phase([obj_var],
                                  solver.CHOOSE_FIRST_UNBOUND,
                                  solver.ASSIGN_MIN_VALUE)
        main_phase = solver.Compose([sequence_phase, vars_phase])
        # Create the solution collector.
        collector = solver.LastSolutionCollector()


        # Add the interesting variables to the SolutionCollector.
        collector.Add(all_sequences)
        collector.AddObjective(obj_var)

        for i in all_machines:
            sequence = all_sequences[i]
            sequence_count = sequence.Size()
            for j in range(0, sequence_count):
                t = sequence.Interval(j)
                collector.Add(t.StartExpr().Var())
                collector.Add(t.EndExpr().Var())
        # Solve the problem.
        disp_col_width = 10
        if solver.Solve(main_phase, [objective_monitor, collector]):
            # print("\nOptimal Schedule Length:", collector.ObjectiveValue(0), "\n")
            sol_line = ""
            sol_line_tasks = ""
            # print("Optimal Schedule", "\n")

            for i in all_machines:
                seq = all_sequences[i]
                sol_line += "Machine " + str(i) + ": "
                sol_line_tasks += "Machine " + str(i) + ": "
                sequence = collector.ForwardSequence(0, seq)
                seq_size = len(sequence)

                for j in range(0, seq_size):
                    t = seq.Interval(sequence[j])
                    # Add spaces to output to align columns.
                    sol_line_tasks += t.Name() + " " * (disp_col_width - len(t.Name()))
                    x[i, j, 0] = int(t.Name().split('_')[1])
                    x[i, j, 1] = int(t.Name().split('_')[2])

                for j in range(0, seq_size):
                    t = seq.Interval(sequence[j])
                    sol_tmp = "[" + \
                        str(collector.Value(0, t.StartExpr().Var())) + ","
                    sol_tmp += str(collector.Value(0,
                                                   t.EndExpr().Var())) + "] "
                    # Add spaces to output to align columns.
                    sol_line += sol_tmp + " " * (disp_col_width - len(sol_tmp))

                    h[i, j] = collector.Value(0, t.StartExpr().Var())
                    e[i, j] = collector.Value(0, t.EndExpr().Var())

                sol_line += "\n"
                sol_line_tasks += "\n"

        self.x = x
        self.h = h
        self.e = e
        self.best_x = x

    def SoluteWithGA(self):

        pt_tmp = self.p
        ms_tmp = self.r + 1

        dfshape = pt_tmp.shape
        num_mc = dfshape[1]  # number of machines
        num_job = dfshape[0]  # number of jobs
        num_gene = num_mc*num_job  # number of genes in a chromosome

        pt = pt_tmp
        ms = ms_tmp

        population_size = 30
        crossover_rate = 0.8
        mutation_rate = 0.2
        mutation_selection_rate = 0.2
        num_mutation_jobs = round(num_gene*mutation_selection_rate)
        num_iteration = 2000

        start_time = time.time()

        Tbest = 999999999999999
        best_list, best_obj = [], []
        population_list = []
        makespan_record = []
        for i in range(population_size):
            # generate a random permutation of 0 to num_job*num_mc-1
            nxm_random_num = list(np.random.permutation(num_gene))
            # add to the population_list
            population_list.append(nxm_random_num)
            for j in range(num_gene):
                # convert to job number format, every job appears m times
                population_list[i][j] = population_list[i][j] % num_job

        for n in range(num_iteration):
            Tbest_now = 99999999999

            '''-------- two point crossover --------'''
            parent_list = copy.deepcopy(population_list)
            offspring_list = copy.deepcopy(population_list)
            # generate a random sequence to select the parent chromosome to crossover
            S = list(np.random.permutation(population_size))

            for m in range(int(population_size/2)):
                crossover_prob = np.random.rand()
                if crossover_rate >= crossover_prob:
                    parent_1 = population_list[S[2*m]][:]
                    parent_2 = population_list[S[2*m+1]][:]
                    child_1 = parent_1[:]
                    child_2 = parent_2[:]
                    cutpoint = list(np.random.choice(
                        num_gene, 2, replace=False))
                    cutpoint.sort()

                    child_1[cutpoint[0]:cutpoint[1]
                            ] = parent_2[cutpoint[0]:cutpoint[1]]
                    child_2[cutpoint[0]:cutpoint[1]
                            ] = parent_1[cutpoint[0]:cutpoint[1]]
                    offspring_list[S[2*m]] = child_1[:]
                    offspring_list[S[2*m+1]] = child_2[:]

            '''----------repairment-------------'''
            for m in range(population_size):
                job_count = {}
                # 'larger' record jobs appear in the chromosome more than m times, and 'less' records less than m times.
                larger, less = [], []
                for i in range(num_job):
                    if i in offspring_list[m]:
                        count = offspring_list[m].count(i)
                        pos = offspring_list[m].index(i)
                        # store the above two values to the job_count dictionary
                        job_count[i] = [count, pos]
                    else:
                        count = 0
                        job_count[i] = [count, 0]
                    if count > num_mc:
                        larger.append(i)
                    elif count < num_mc:
                        less.append(i)

                for k in range(len(larger)):
                    chg_job = larger[k]
                    while job_count[chg_job][0] > num_mc:
                        for d in range(len(less)):
                            if job_count[less[d]][0] < num_mc:
                                offspring_list[m][job_count[chg_job]
                                                  [1]] = less[d]
                                job_count[chg_job][1] = offspring_list[m].index(
                                    chg_job)
                                job_count[chg_job][0] = job_count[chg_job][0]-1
                                job_count[less[d]][0] = job_count[less[d]][0]+1
                            if job_count[chg_job][0] == num_mc:
                                break

            '''--------mutatuon--------'''
            for m in range(len(offspring_list)):
                mutation_prob = np.random.rand()
                if mutation_rate >= mutation_prob:
                    # chooses the position to mutation
                    m_chg = list(np.random.choice(
                        num_gene, num_mutation_jobs, replace=False))
                    # save the value which is on the first mutation position
                    t_value_last = offspring_list[m][m_chg[0]]
                    for i in range(num_mutation_jobs-1):
                        # displacement
                        offspring_list[m][m_chg[i]
                                          ] = offspring_list[m][m_chg[i+1]]

                    # move the value of the first mutation position to the last mutation position
                    offspring_list[m][m_chg[num_mutation_jobs-1]
                                      ] = t_value_last

            '''--------fitness value(calculate makespan)-------------'''
            total_chromosome = copy.deepcopy(
                parent_list)+copy.deepcopy(offspring_list)  # parent and offspring chromosomes combination
            chrom_fitness, chrom_fit = [], []
            total_fitness = 0
            for m in range(population_size*2):  # for every gene line
                j_keys = [j for j in range(num_job)]
                key_count = {key: 0 for key in j_keys}
                j_count = {key: 0 for key in j_keys}
                m_keys = [j+1 for j in range(num_mc)]
                m_count = {key: 0 for key in m_keys}

                for i in total_chromosome[m]:
                    gen_t = int(pt[i][key_count[i]])
                    gen_m = int(ms[i][key_count[i]])
                    j_count[i] = j_count[i]+gen_t
                    m_count[gen_m] = m_count[gen_m]+gen_t

                    if m_count[gen_m] < j_count[i]:
                        m_count[gen_m] = j_count[i]
                    elif m_count[gen_m] > j_count[i]:
                        j_count[i] = m_count[gen_m]

                    key_count[i] = key_count[i]+1

                makespan = max(j_count.values())
                chrom_fitness.append(1/makespan)
                chrom_fit.append(makespan)
                total_fitness = total_fitness+chrom_fitness[m]

            '''----------selection(roulette wheel approach)----------'''
            pk, qk = [], []

            for i in range(population_size*2):
                pk.append(chrom_fitness[i]/total_fitness)
            for i in range(population_size*2):
                cumulative = 0
                for j in range(0, i+1):
                    cumulative = cumulative+pk[j]
                qk.append(cumulative)

            selection_rand = [np.random.rand() for i in range(population_size)]

            for i in range(population_size):
                if selection_rand[i] <= qk[0]:
                    population_list[i] = copy.deepcopy(total_chromosome[0])
                else:
                    for j in range(0, population_size*2-1):
                        if selection_rand[i] > qk[j] and selection_rand[i] <= qk[j+1]:
                            population_list[i] = copy.deepcopy(
                                total_chromosome[j+1])
                            break
            '''----------comparison----------'''
            for i in range(population_size*2):
                if chrom_fit[i] < Tbest_now:
                    Tbest_now = chrom_fit[i]
                    sequence_now = copy.deepcopy(total_chromosome[i])
            if Tbest_now <= Tbest:
                Tbest = Tbest_now
                sequence_best = copy.deepcopy(sequence_now)

            makespan_record.append(Tbest)
        '''----------result----------'''
        # print("optimal sequence", sequence_best)
        # print("optimal value:%f" % Tbest)
        # print('the elapsed time:%s' % (time.time() - start_time))

        import pandas as pd
        import datetime

        x = np.zeros((self.m, self.n, 2), dtype=int)
        h = np.zeros((self.m, self.n), dtype=int)
        e = np.zeros((self.m, self.n), dtype=int)

        m_keys = [j+1 for j in range(num_mc)]
        j_keys = [j for j in range(num_job)]
        key_count = {key: 0 for key in j_keys}
        j_count = {key: 0 for key in j_keys}
        m_count = {key: 0 for key in m_keys}
        j_record = {}
        for i in sequence_best:
            gen_t = int(pt[i][key_count[i]])  # time
            gen_m = int(ms[i][key_count[i]])  # order
            j_count[i] = j_count[i]+gen_t  # time of job
            m_count[gen_m] = m_count[gen_m]+gen_t  # time of machine

            if m_count[gen_m] < j_count[i]:
                m_count[gen_m] = j_count[i]
            elif m_count[gen_m] > j_count[i]:
                j_count[i] = m_count[gen_m]

            # convert seconds to hours, minutes and seconds
            start_time = int(j_count[i]-pt[i][int(key_count[i])])
            end_time = int(j_count[i])

            j_record[(i, gen_m)] = [start_time, end_time,key_count[i]]

            key_count[i] = key_count[i]+1

        df = []
        for m in m_keys:
            for j in j_keys:
                list_of_start = [j_record[(q, m)][0] for q in j_keys]
                list_of_start.sort()
                order = list_of_start.index(j_record[(j, m)][0])
                h[m-1, order] = j_record[(j, m)][0]
                e[m-1, order] = j_record[(j, m)][1]
                x[m-1, order, 0] = j
                x[m-1, order, 1] = j_record[(j, m)][2]
                df.append(dict(Task='Machine %s' % (m), Start='2018-07-14 %s' % (str(j_record[(
                    j, m)][0])), Finish='2018-07-14 %s' % (str(j_record[(j, m)][1])), Resource='Job %s' % (j+1)))

        self.h = h
        self.e = e
        self.x = x
        self.PlotResult()
        # plt.show()

    def PlotResult(self, num=0):

        colorbox = ['yellow', 'whitesmoke', 'lightyellow',
                    'khaki', 'silver', 'pink', 'lightgreen', 'orange', 'grey', 'r', 'brown']

        for i in range(100):
            colorArr = ['1', '2', '3', '4', '5', '6', '7',
                        '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
            color = ""
            for i in range(6):
                color += colorArr[random.randint(0, 14)]
            colorbox.append("#"+color)

        zzl = plt.figure(figsize=(12, 4))
        for i in range(self.m):
            # number_of_mashine:
            for j in range(self.n):
                # number_of_job:

                # % 数据读写
                mPoint1 = self.h[i, j]  # % 读取开始的点
                mPoint2 = self.e[i, j]  # % 读取结束的点
                mText = i + 1.5  # % 读取机器编号
                PlotRec(mPoint1, mPoint2, mText)  # % 画图函数，（开始点，结束点，高度）
                Word = str(self.x[i, j, 0]+1) +'.'+ str(self.x[i, j, 1]+1)  # % 读取工件编号
                # hold on

                # % 填充
                x1 = mPoint1
                y1 = mText-1
                x2 = mPoint2
                y2 = mText-1
                x3 = mPoint2
                y3 = mText
                x4 = mPoint1
                y4 = mText

                plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4],
                         color=colorbox[self.x[i, j, 0]])

                plt.text(0.5*mPoint1+0.5*mPoint2-3, mText-0.5, Word)
        plt.xlabel('Time')
        plt.ylabel('Machine')
        plt.tight_layout()
        plt.savefig('out.png', dpi=400)

    def SoluteWithBBMAndSaveToFile(self, filepath, index, pool=0):
        self.SoluteWithBBM()
        self.SaveProblemToFile(filepath, index, pool)
        self.SavesolutionToFile(filepath, index, pool)

    def SoluteWithGaAndSaveToFile(self, filepath, index, pool=0):
        self.SoluteWithGA()
        self.SaveProblemToFile(filepath, index, pool)
        self.SavesolutionToFile(filepath, index, pool)

    def LoadProblemWithSolution(self, filepath, index, pool=0):
        f = open('data/jssp_problem_'+filepath, 'r')
        # line_list = [ i for i in range(index * 7, index * 7 + 7 )]
        data = f.readlines()
        data = data[index * 7:index * 7 + 7]
        self.m = int(data[0])
        self.n = int(data[1])
        # print(data[2])
        self.p = np.fromstring(
            data[2][:-1], dtype=int, sep=',').reshape((self.n, self.m))
        self.r = np.fromstring(
            data[3][:-1], dtype=int, sep=',').reshape((self.n, self.m))
        self.x = np.fromstring(
            data[4][:-1], dtype=int, sep=',').reshape((self.m, self.n, 2))
        self.h = np.fromstring(
            data[5][:-1], dtype=int, sep=',').reshape((self.m, self.n))
        self.e = np.fromstring(
            data[6][:-1], dtype=int, sep=',').reshape((self.m, self.n))

    def CalculateSimilarityDegree(self,x = None):

        if x is not None:
            self.x = x
 
        right = 0
        for i in range(self.m):
            for j in range(self.n):
                if self.x[i,j,0] == self.best_x[i,j,0] and self.x[i,j,1] == self.best_x[i,j,1]:
                    right += 1

        return right/self.m/self.n

    def LoadProblemWithoutSolution(self, filepath, index, pool=0):
        f = open('data/jssp_problem_'+filepath, 'r')
        # line_list = [ i for i in range(index * 7, index * 7 + 7 )]
        data = f.readlines()
        data = data[index * 7:index * 7 + 7]
        self.m = int(data[0])
        self.n = int(data[1])
        # print(data[2])
        self.p = np.fromstring(
            data[2][:-1], dtype=int, sep=',').reshape((self.n, self.m))
        self.r = np.fromstring(
            data[3][:-1], dtype=int, sep=',').reshape((self.n, self.m))
        
        # self.x = np.fromstring(
        #     data[4][:-1], dtype=int, sep=',').reshape((self.m, self.n, 2))
        # self.h = np.fromstring(
        #     data[5][:-1], dtype=int, sep=',').reshape((self.m, self.n))
        # self.e = np.fromstring(
        #     data[6][:-1], dtype=int, sep=',').reshape((self.m, self.n))

    def subproblem(self):
        sub_list = []
        for job in range(self.n):
            for procedure in range(self.m):
                sub_p = Subproblem.Subproblem(self, job, procedure)
                sub_list.append(sub_p)
        return sub_list

    def GetFeaturesInTest1D(self):

        F = []
        sub_list = self.subproblem()
        for sub in sub_list:
            F.append(sub.GetFeatures1D())
        F = np.array(F)

        return F
    def Getlables(self):

        L = []
        sub_list = self.subproblem()
        for sub in sub_list:
            L.append(sub.label)
        L = np.array(L)

        return L
    def GetFeaturesInTest1D2D(self):
        len_feature_1d = self.number_of_1d_feature

        F1d = []
        F2d1 = []
        F2d2 = []
        F2d3 = []
        F2d4 = []
        F2d5 = []
        F2d6 = []

        sub_list = self.subproblem()
        for sub in sub_list:
            F1d.append(sub.GetFeatures1D())
            features_2d = sub.GetFeatures2D()
            F2d1.append(features_2d[0])
            F2d2.append(features_2d[1])
            F2d3.append(features_2d[2])
            F2d4.append(features_2d[3])
            F2d5.append(features_2d[4])
            F2d6.append(features_2d[5])

        F1d = np.array(F1d).reshape(
            (-1, len_feature_1d, 1))

        F2d1 = np.array(F2d1).reshape(
            (-1, self.n**2, self.n**2, 1))
        F2d2 = np.array(F2d2).reshape(
            (-1, self.n**2, self.n, 1))
        F2d3 = np.array(F2d3).reshape(
            (-1, self.n**2, len_feature_1d, 1))
        F2d4 = np.array(F2d4).reshape(
            (-1, self.n, self.n, 1))
        F2d5 = np.array(F2d5).reshape(
            (-1, self.n, len_feature_1d, 1))
        F2d6 = np.array(F2d6).reshape(
            (-1, len_feature_1d, len_feature_1d, 1))

        tf = [F1d.reshape((-1, len_feature_1d, 1)), F2d1,
              F2d2, F2d3, F2d4, F2d5, F2d6]

        return tf

    def GetIndexMatrix(self):
        index = np.zeros((self.n, self.m))

        sub = self.subproblem()
        for s in sub:
            index[s.job_id, s.machine_id] = s.SearchInX()
        return index

    def SchedulingSequenceGenerationMethod(self, output):
        np.savetxt('output.csv', output,fmt="%.2f",delimiter=',')
        for i in range(self.m * self.n):
            output[i,:] = output[i,:]/output[i,:].sum()

        h = np.zeros((self.m, self.n))
        e = np.zeros((self.m, self.n))
        x = np.zeros((self.m, self.n,2))

        procedure_job = [0] * self.n
        order_machine = [0] * self.m

        for i in range(self.m*self.n):
            possible_probability = []
            
            for job in range(self.n):
                procedure = procedure_job[job]
                machine = self.r[job,min(4,procedure)]
                order = order_machine[machine]
                if procedure <5 and order < 5:
                    possible_probability.append( [job,procedure,machine,order,output[job*self.m+procedure][order]] )   
                else:
                    machine = -1
                    possible_probability.append( [job,procedure,machine,order,0] )   

            possible_probability = sorted(possible_probability, key=lambda x:x[4] )
            
            bestjob,bestproce,bestmachine,bestorder = possible_probability[-1][:4]
            x[bestmachine,bestorder][:] = [bestjob,bestproce]

            procedure_job[bestjob]+=1
            order_machine[bestmachine] += 1
        
        self.x = x

    def GurobiModelingmethod(self, output):
        np.savetxt('output.csv', output,fmt="%.2f",delimiter=',')
        lables = self.Getlables()
        R = self.r.reshape(self.m*self.n)
        # x ,p = guchoose.main(output,R,lables,self.m,self.n)
        
        self.x = x
        h = np.zeros((self.m, self.n))
        e = np.zeros((self.m, self.n))

        for order in range(self.n):
            timeline_machine = np.zeros((self.m), dtype=int)
            timeline_jobs = np.zeros((self.n), dtype=int)
            index_in_machine = np.zeros((self.m), dtype=int)
            job_finsh = np.zeros((self.n), dtype=int)
            for i in range(self.m*self.n):
                mask = np.zeros((self.m), dtype=int)
                for ma in range(self.m):
                    job, order = x[ma, min(self.n-1, index_in_machine[ma]), :]
                    # if job_finsh[job] == order:
                    mask[ma] = timeline_machine[ma]
                    # else:
                    #     mask[ma] = 10000
                earlyestmachine = np.argmin(mask)

                while index_in_machine[earlyestmachine] == self.n:
                    timeline_machine[earlyestmachine] = 100000
                    earlyestmachine = np.argmin(timeline_machine)
                # while can_do_in_machine[earlyestmachine]

                job, order = x[earlyestmachine,
                               index_in_machine[earlyestmachine], :]
                time_s = max(
                    timeline_machine[earlyestmachine], timeline_jobs[job])
                time_e = time_s + self.p[job, order]
                timeline_machine[earlyestmachine] = time_e
                timeline_jobs[job] = time_e
                h[earlyestmachine, index_in_machine[earlyestmachine]] = time_s
                e[earlyestmachine, index_in_machine[earlyestmachine]] = time_e
                index_in_machine[earlyestmachine] += 1
                job_finsh[job] += 1

        self.e = e
        self.h = h

    def PriorityQueuingMethod(self, output):
        np.savetxt('output.csv', output,fmt="%.2f",delimiter=',')
        lables = self.Getlables()
        R = self.r.reshape(self.m*self.n)
        
        x = [[] for j in range(self.m) ]
        for i in range(self.m*self.n):
            machine = R[i]
            x[machine].append([i//self.m,i%self.m,output[i]])
        
        for m in range(self.m):
            x[m].sort(key = lambda x:x[2])

        xx = np.zeros((self.m,self.n,2),dtype=int)
        for i in range(self.m):
            for j in range(self.n):
                xx[i,j,0] = x[i][j][0]
                xx[i,j,1] = x[i][j][1]
        x =xx 
        self.x = xx
        h = np.zeros((self.m, self.n))
        e = np.zeros((self.m, self.n))

        for order in range(self.n):
            timeline_machine = np.zeros((self.m), dtype=int)
            timeline_jobs = np.zeros((self.n), dtype=int)
            index_in_machine = np.zeros((self.m), dtype=int)
            job_finsh = np.zeros((self.n), dtype=int)
            for i in range(self.m*self.n):
                mask = np.zeros((self.m), dtype=int)
                for ma in range(self.m):
                    job, order = x[ma, min(self.n-1, index_in_machine[ma]), :]
                    if job_finsh[job] == order:
                        mask[ma] = timeline_machine[ma]
                    else:
                        mask[ma] = 10000
                earlyestmachine = np.argmin(mask)

                while index_in_machine[earlyestmachine] == self.n:
                    timeline_machine[earlyestmachine] = 100000
                    earlyestmachine = np.argmin(timeline_machine)
                # while can_do_in_machine[earlyestmachine]

                job, order = x[earlyestmachine,
                               index_in_machine[earlyestmachine], :]
                time_s = max(
                    timeline_machine[earlyestmachine], timeline_jobs[job])
                time_e = time_s + self.p[job, order]
                timeline_machine[earlyestmachine] = time_e
                timeline_jobs[job] = time_e
                h[earlyestmachine, index_in_machine[earlyestmachine]] = time_s
                e[earlyestmachine, index_in_machine[earlyestmachine]] = time_e
                index_in_machine[earlyestmachine] += 1
                job_finsh[job] += 1

        self.e = e
        self.h = h
    # def SchedulingSequenceGenerationMethod(self, output):
    #     np.savetxt('output.csv', output, delimiter=',')
    #     x = np.zeros((self.m, self.n, 2), dtype=int)
    #     for machine in range(self.m):
    #         M = []
    #         index = np.argwhere(self.r == machine)
    #         M = [output[job*self.m + pro] for job, pro in index]
    #         M = np.array(M)

    #         for i in range(self.n):
    #             max_in_M = M.max()
    #             a, order = np.argwhere(M == max_in_M)[0]
    #             x[machine, order, :] = index[a]
    #             M[a, :] = 0
    #             M[:, order] = 0
    #     self.x = x
    #     # x = self.x

    #     h = np.zeros((self.m, self.n))
    #     e = np.zeros((self.m, self.n))

    #     for order in range(self.n):
    #         timeline_machine = np.zeros((self.m), dtype=int)
    #         timeline_jobs = np.zeros((self.n), dtype=int)
    #         index_in_machine = np.zeros((self.m), dtype=int)
    #         job_finsh = np.zeros((self.n), dtype=int)
    #         for i in range(self.m*self.n):
    #             mask = np.zeros((self.m), dtype=int)
    #             for ma in range(self.m):
    #                 job, order = x[ma, min(self.n-1, index_in_machine[ma]), :]
    #                 # if job_finsh[job] == order:
    #                 mask[ma] = timeline_machine[ma]
    #                 # else:
    #                 #     mask[ma] = 10000
    #             earlyestmachine = np.argmin(mask)

    #             while index_in_machine[earlyestmachine] == self.n:
    #                 timeline_machine[earlyestmachine] = 100000
    #                 earlyestmachine = np.argmin(timeline_machine)
    #             # while can_do_in_machine[earlyestmachine]

    #             job, order = x[earlyestmachine,
    #                            index_in_machine[earlyestmachine], :]
    #             time_s = max(
    #                 timeline_machine[earlyestmachine], timeline_jobs[job])
    #             time_e = time_s + self.p[job, order]
    #             timeline_machine[earlyestmachine] = time_e
    #             timeline_jobs[job] = time_e
    #             h[earlyestmachine, index_in_machine[earlyestmachine]] = time_s
    #             e[earlyestmachine, index_in_machine[earlyestmachine]] = time_e
    #             index_in_machine[earlyestmachine] += 1
    #             job_finsh[job] += 1
    #     self.e = e
    #     self.h = h
    #     # print('ddd')

    def GetMakespan(self):
        return self.e.max()


def TranslateNpToStr(m):
    a = m.reshape((-1))
    a = list(a)
    s = ''.join(['{},'.format(round(o,2)) for o in a]) + '\n'
    return s


def PlotRec(mPoint1, mPoint2, mText):

    vPoint = np.zeros((4, 2))
    vPoint[0, :] = [mPoint1, mText-1]
    vPoint[1, :] = [mPoint2, mText-1]
    vPoint[2, :] = [mPoint1, mText]
    vPoint[3, :] = [mPoint2, mText]
    plt.plot([vPoint[0, 0], vPoint[1, 0]], [vPoint[0, 1], vPoint[1, 1]], 'k')
    # hold on
    plt.plot([vPoint[0, 0], vPoint[2, 0]], [vPoint[0, 1], vPoint[2, 1]], 'k')
    plt.plot([vPoint[1, 0], vPoint[3, 0]], [vPoint[1, 1], vPoint[3, 1]], 'k')
    plt.plot([vPoint[2, 0], vPoint[3, 0]], [vPoint[2, 1], vPoint[3, 1]], 'k')


if __name__ == "__main__":
    prob = Problem(2, 8, 10, 30)
    prob.SoluteWithBBMAndSaveToFile('data', 0)
    prob.Print_info()
    sub_list = prob.subproblem()
    sub_list[0].Show2DFeatures()
