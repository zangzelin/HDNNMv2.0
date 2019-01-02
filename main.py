# time : 2019-1-2
# author : zelin zang 
# email : zangzelin@gmail.com
# this is the start of this code.
# just type python3 main.py in ubuntu 16 system.

import matplotlib.pyplot as plt
import numpy as np
import progressbar

import JSSPproblem
import MultiCpu
import NnNetwork


def CreateJssp(number_of_problem, index_cpu, m, n, timehigh, timelow):
    # Create Jobshop problem with ortools and save it to 'bigdata/'
    # problem can be solute with two main method:
    #   1. Branch and bound method (BBM) for small problems
    #   2. Genetic Method(GA) for big problems
    # Input:
    #   number_of_loop: is the number of problems need to create
    #   index_cpu: only used in muti cpu:
    #   m: the number of the machine in jobshop problem
    #   n: the number of the job in the job problem
    #   timehigh: the max producing time of the job's one processing
    #   timelow: the min producing time of the job's one processing

    pbar = progressbar.ProgressBar().start()
    for p in range(number_of_problem):

        pbar.update(int((p / (number_of_problem - 1)) * 100))

        # init one Jobshop problem randomly
        prob = JSSPproblem.Problem(m, n, time_low=timelow, time_high=timehigh)

        # solute the problem with two method, you can change it
        # if you do not the the ortools wheels, choose the GA method
        prob.SoluteWithGaAndSaveToFile('data', 0)
        # prob.SoluteWithBBMAndSaveToFile('data', 0)

        # print the information of the problem and the solution, and save it in to the file 'bigdata/'
        prob.Print_info()
        sub_list = prob.subproblem()
        for i, subp in enumerate(sub_list):
            subp.SaveFeaturesToFile('feature', i)
            info = subp.SaveLablesToFile('feature', i)

    pbar.finish()

    return info


def main_train_ann(index_cpu, m, n, timehigh, timelow, loop):
    # train a traditional ann model to solute this problem
    # Input:
    #   index_cpu: is used in muti cpu
    #   m: the number of the machine in jobshop problem
    #   n: the number of the job in the job problem
    #   timehigh: the max producing time of the job's one processing
    #   timelow: the min producing time of the job's one processing
    #   loop: the max loop of the training, you can change it

    format_of_ann = [11, 22, 220, 220, 220, 220, 220, 220, 50, 22, n]
    ann_model = NnNetwork.NeuralNetwork(timelow, timehigh,
                                        'Ann', format_of_ann, batch_size=25*3, m=m, n=n)

    # Load the data created in function CreateJssp()
    ann_model.LoadData(info)

    # Train the network
    ann_model.TrainAnn(loop)

    # save the ann model into bigdata
    ann_model.SaveModel()


def TestAnnModel(m, n, timehigh, timelow):
    # Test the model, and solute the new problem
    # Input:
    #   m: the number of the machine in jobshop problem
    #   n: the number of the job in the job problem
    #   timehigh: the max producing time of the job's one processing
    #   timelow: the min producing time of the job's one processing

    info = 'm={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
        m, n, timehigh, timelow, pool)

    # init a new ann model 
    format_of_ann = [11, 22, 220, 220, 220, 220, 220, 220, 50, 22, n]
    ann_model = NnNetwork.NeuralNetwork(timelow, timehigh,
                                        'Ann', format_of_ann, batch_size=64, m=m, n=n)
    
    # load the parmater in '\bigdata'
    ann_model.LoadModel('model_type=Ann_m={}_n={}time=0.h5'.format(m, n))

    # test the model with 200 different new jobshop problem 
    best, out = ann_model.TestTheNetworkRandomlyntimes(200)

    print(best.sum()/out.sum())


def main_train_HDNNM(index_cpu, m, n, timehigh, timelow, loop, L1, L2):
    # train a new HDNNM model to solute this problem
    # Input:
    #   index_cpu: is used in muti cpu
    #   m: the number of the machine in jobshop problem
    #   n: the number of the job in the job problem
    #   timehigh: the max producing time of the job's one processing
    #   timelow: the min producing time of the job's one processing
    #   loop: the max loop of the training, you can change it
    #   L1: the parmater use to control the format of the model
    #   L2: the parmater use to control the format of the model

    hdnnm_model = NnNetwork.NeuralNetwork(
        timelow=timelow, timehigh=timehigh, type='HDNNM', number_of_n=[
            11, 22, 22, 22, 22, n], batch_size=64, m=m, n=n, L1=L1, L2=L2
    )

    # Load the data created in function CreateJssp()
    hdnnm_model.LoadData(info)
    # Train the network
    hdnnm_model.TrainAnn(loop)
    # save the ann model into bigdata
    hdnnm_model.SaveModel()


def main_loadmodel_and_predict_HDNNM(m, n, timehigh, timelow, L1, L2):
    # Test the model, and solute the new problem
    # Input:
    #   m: the number of the machine in jobshop problem
    #   n: the number of the job in the job problem
    #   timehigh: the max producing time of the job's one processing
    #   timelow: the min producing time of the job's one processing

    info = 'm={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
        m, n, timehigh, timelow, pool)
    ann_model = NnNetwork.NeuralNetwork(
        timelow=timelow, timehigh=timehigh, type='HDNNM', number_of_n=[
            11, 22, 22, 22, 22, n], batch_size=64, m=m, n=n, L1=L1, L2=L2
    )
    
    ann_model.LoadModel('model_type=HDNNM_m={}_n={}time=0.h5'.format(m, n))
    
    # test the model with 200 different new jobshop problem 
    best, out = ann_model.TestTheNetworkRandomlyntimes(200)

    print(best.sum()/out.sum())

    return best.sum()/out.sum()


def ExampleInPaper():
    prob = JSSPproblem.Problem(m, n, time_low=timelow, time_high=timehigh)
    prob.LoadProblemWithoutSolution(info, 0)
    prob.SoluteWithBBM()
    prob.Print_info()
    prob.PlotResult()

    sub = prob.subproblem()
    for subb in sub:
        subb.Show2DFeatures()
    plt.show()


def TrainAndTestHDNNM(m, n, timehigh, timelow, loop, L1, L2):
    
    main_train_HDNNM(0, m, n, timehigh, timelow, 15, L1, L2)
    result = main_loadmodel_and_predict_HDNNM(
        m, n, timehigh, timelow, L1=L1, L2=L2)
    f = open('result/out.csv', 'a')
    f.writelines('{},{},{},{},{},{},{},{} \n'.format(
        m, n, timehigh, timelow, loop, L1, L2, result))


if __name__ == "__main__":

    m = 5
    n = 5
    timehigh = 30
    timelow = 15
    pool = 0

    # info = CreateJssp(600, pool, m, n, timehigh, timelow)
    # print('finish')
    info = 'm={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
        m, n, timehigh, timelow, pool)
    # main_train_ann(0, m, n, timehigh, timelow, 15)
    # main_loadmodel_and_predict_ann( m, n, timehigh, timelow)

    # main_train_HDNNM(0, m, n, timehigh, timelow, 12, 3, 3)
    # main_loadmodel_and_predict_HDNNM(m, n, timehigh, timelow, L1 = 3, L2 = 3)
    
    
    for L1 in range(1, 10):
        for L2 in range(1, 10):
            TrainAndTestHDNNM(m, n, timehigh, timelow, 15, L1, L2)
            TrainAndTestHDNNM(m, n, timehigh, timelow, 15, L1, L2)
            TrainAndTestHDNNM(m, n, timehigh, timelow, 15, L1, L2)
            TrainAndTestHDNNM(m, n, timehigh, timelow, 15, L1, L2)
            TrainAndTestHDNNM(m, n, timehigh, timelow, 15, L1, L2)
