import JSSPproblem
import matplotlib.pyplot as plt
import MultiCpu
import progressbar
import numpy as np
import NnNetwork


def CreateJssp(number_of_loop, index_cpu, m, n, timehigh, timelow):

    pbar = progressbar.ProgressBar().start()
    for p in range(number_of_loop):

        pbar.update(int((p / (number_of_loop - 1)) * 100))

        prob = JSSPproblem.Problem(m, n, time_low=timelow, time_high=timehigh)
        # prob.SoluteWithGaAndSaveToFile('data', 0)
        prob.SoluteWithBBMAndSaveToFile('data', 0)
        # prob.Print_info()
        sub_list = prob.subproblem()
        for i, subp in enumerate(sub_list):
            subp.SaveFeaturesToFile('feature', i)
            info = subp.SaveLablesToFile('feature', i)
    pbar.finish()

    return info


def main_train_ann(index_cpu, m, n, timehigh, timelow, loop):

    ann_model = NnNetwork.NeuralNetwork(timelow, timehigh,
                                        'Ann', [11, 22, 220, 220, 220, 220, 220, 220, 50, 22, n], batch_size=25*3, m=m, n=n)
    # 'HDNNM', [11, 22, 22, 22, 22, n], batch_size=64, m=m, n=n)
    ann_model.LoadData(info)
    ann_model.TrainAnn(loop)
    ann_model.SaveModel()


def main_loadmodel_and_predict_ann(m, n, timehigh, timelow):

    info = 'm={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
        m, n, timehigh, timelow, pool)
    ann_model = NnNetwork.NeuralNetwork(timelow, timehigh,
                                        'Ann', [11, 22, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 220, 50, 22, n], batch_size=64, m=m, n=n)
    # ann_model.LoadData(info)
    ann_model.LoadModel('model_type=Ann_m={}_n={}time=0.h5'.format(m, n))

    best, out = ann_model.TestTheNetworkRandomlyntimes(200)
    # c = ann_model.GetConfusionMatraxOnce()
    # return c
    print(best.sum()/out.sum())


def main_train_HDNNM(index_cpu, m, n, timehigh, timelow, loop, L1, L2):
    # timelow, timehigh, type, number_of_n, batch_size, m, n, L1=0, L2=0
    ann_model = NnNetwork.NeuralNetwork(
        timelow=timelow, timehigh=timehigh, type='HDNNM', number_of_n=[
            11, 22, 22, 22, 22, n], batch_size=64, m=m, n=n, L1=L1, L2=L2
    )
    ann_model.LoadData(info)
    ann_model.TrainAnn(loop)
    ann_model.SaveModel()


def main_loadmodel_and_predict_HDNNM(m, n, timehigh, timelow, L1, L2):

    info = 'm={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
        m, n, timehigh, timelow, pool)
    ann_model = NnNetwork.NeuralNetwork(
        timelow=timelow, timehigh=timehigh, type='HDNNM', number_of_n=[
            11, 22, 22, 22, 22, n], batch_size=64, m=m, n=n, L1=L1, L2=L2
    )
    # ann_model.LoadData(info)
    ann_model.LoadModel('model_type=HDNNM_m={}_n={}time=0.h5'.format(m, n))

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
        m, n, timehigh, timelow, loop, L1, L2,result))


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
    for L1 in range(1,10):
        for L2 in range(1,10):
            TrainAndTestHDNNM(m, n, timehigh, timelow, 15, L1, L2)
            TrainAndTestHDNNM(m, n, timehigh, timelow, 15, L1, L2)
            TrainAndTestHDNNM(m, n, timehigh, timelow, 15, L1, L2)
            TrainAndTestHDNNM(m, n, timehigh, timelow, 15, L1, L2)
            TrainAndTestHDNNM(m, n, timehigh, timelow, 15, L1, L2)
