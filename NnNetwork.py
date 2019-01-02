import time

import keras
import numpy as np
import tensorflow as tf
from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D,
                          concatenate)
from keras.models import Model
from keras.utils import np_utils, plot_model

import JSSPproblem
import Subproblem
import matplotlib.pyplot as plt


class LossHistory(keras.callbacks.Callback):

    number_of_machines = 7
    number_of_jobs = 7
    timelow = 10
    timehigh = 30
    hismak = []

    def PredictWithnetwork(self, input):
        output = self.model.predict(input)
        return output

    def TestTheNetworkRandomlyOnce(self):

        prob = JSSPproblem.Problem(
            self.number_of_machines, self.number_of_jobs, time_low=self.timelow, time_high=self.timehigh)
        prob.SoluteWithBBM()
        best_mak = prob.GetMakespan()

        featrues = prob.GetFeaturesInTest1D2D()
        output = self.PredictWithnetwork(featrues)
        prob.SchedulingSequenceGenerationMethod(output)
        ann_mak = prob.GetMakespan()
        return best_mak, ann_mak

    def TestTheNetworkRandomlyntimes(self, times):

        best_makspan_list = []
        out_makspan_list = []
        for i in range(times):
            best_makspan, out_makspan = self.TestTheNetworkRandomlyOnce()
            best_makspan_list.append(best_makspan)
            out_makspan_list.append(out_makspan)

        return np.array(best_makspan_list), np.array(out_makspan_list)

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        a, b = self.TestTheNetworkRandomlyntimes(100)
        self.hismak.append(a.sum()/b.sum())

        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        # 创建一个图
        plt.figure()
        # acc
        # plt.plot(x,y)，这个将数据画成曲线
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)  # 设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')  # 给x，y轴加注释
        plt.legend(loc="upper right")  # 设置图例显示位置
        # plt.show()

        acc = np.array(self.val_acc[loss_type])
        loss = np.array(self.val_loss[loss_type])
        hismak = np.array(self.hismak)

        return acc, loss, hismak


class NeuralNetwork:
    number_of_machines = 0
    number_of_jobs = 0
    number_of_jobs_2 = 0
    model = []
    layer = None
    number_of_n = [11, 11, 11, 11, 8]
    type = ''
    L1 = None
    L2 = None
    batch_size = None
    correct = None
    number_of_1D_feature = None
    features_1D_list = []
    timelow = None
    timehigh = None
    len_feature_1d = None
    len_feature_nm = None

    train_feature_1d = []
    train_feature_2d1 = []
    train_feature_2d2 = []
    train_feature_2d3 = []
    train_feature_2d4 = []
    train_feature_2d5 = []
    train_feature_2d6 = []
    train_label = []

    test_feature_1d = []
    test_feature_2d1 = []
    test_feature_2d2 = []
    test_feature_2d3 = []
    test_feature_2d4 = []
    test_feature_2d5 = []
    test_feature_2d6 = []
    test_label = []

    number_of_data = []
    number_of_class = []
    label_list = []
    features_2D_1 = []
    features_2D_2 = []
    features_2D_3 = []
    features_2D_4 = []
    features_2D_5 = []
    features_2D_6 = []

    # epochs = 20

    def __init__(self, timelow, timehigh, type, number_of_n, batch_size, m, n, L1=0, L2=0):
        self.type = type
        self.number_of_machines = m
        self.number_of_jobs = n
        self.number_of_jobs_2 = n*n
        self.len_feature_1d = number_of_n[0]
        self.len_feature_nm = m * n
        self.timehigh = timehigh
        self.timelow = timelow
        # self.history = LossHistory()

        if self.type == 'Ann':
            self.layer = len(number_of_n)-3
            self.number_of_n = number_of_n
            self.batch_size = batch_size

            self.model = self.CreateANN()

        elif self.type == 'HDNNM':

            self.layer = [len(number_of_n)-3, len(number_of_n)-3]
            self.number_of_n = number_of_n
            self.batch_size = batch_size
            self.L1 = L1
            self.L2 = L2
            self.model = self.CreateHDNNM()

        else:
            raise NameError('the type of the network have to be Ann or HDNNM')

    def CreateANN(self):

        input1 = Input(shape=(self.number_of_n[0], ), name='i1')
        mid = Dense(self.number_of_n[1],
                    activation='tanh')(input1)
        mid = Dropout(0.25)(mid)

        for i in range(self.layer):
            mid = Dense(self.number_of_n[2+i],
                        activation='tanh')(mid)
            mid = Dropout(0.35)(mid)

        mid = Dense(self.number_of_n[2+self.layer],
                    activation='tanh')(mid)
        out = Dense(1, activation='tanh',
                    name='out')(mid)

        model = Model(input1, out)
        # sgd = keras.optimizers.SGD(
        #     lr=0.11, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

        model.summary()

        return model

    def CreateHDNNM(self):

        input_1d = Input(shape=(self.number_of_n[0], 1), name='i1d')
        input_2d1 = Input(shape=(self.number_of_jobs_2,
                                 self.number_of_jobs_2, 1), name='i2d1')
        input_2d2 = Input(shape=(self.number_of_jobs_2,
                                 self.number_of_jobs, 1), name='i2d2')
        input_2d3 = Input(shape=(self.number_of_jobs_2,
                                 self.len_feature_1d, 1), name='i2d3')
        input_2d4 = Input(shape=(self.number_of_jobs,
                                 self.number_of_jobs, 1), name='i2d4')
        input_2d5 = Input(shape=(self.number_of_jobs,
                                 self.len_feature_1d, 1), name='i2d5')
        input_2d6 = Input(shape=(self.len_feature_1d,
                                 self.len_feature_1d, 1), name='i2d6')

        mid_1d = Dense(self.number_of_n[1],
                       activation='relu', use_bias=True)(input_1d)
        mid_2d1 = Conv2D(3, (3, 3), padding='same')(input_2d1)
        mid_2d2 = Conv2D(3, (3, 3), padding='same')(input_2d2)
        mid_2d3 = Conv2D(3, (3, 3), padding='same')(input_2d3)
        mid_2d4 = Conv2D(3, (3, 3), padding='same')(input_2d4)
        mid_2d5 = Conv2D(3, (3, 3), padding='same')(input_2d5)
        mid_2d6 = Conv2D(3, (3, 3), padding='same')(input_2d6)

        for i in range(self.L1):
            mid_1d = Dense(self.number_of_n[1],
                           activation='relu', use_bias=True)(mid_1d)
            mid_2d1 = Conv2D(3, (3, 3), padding='same')(mid_2d1)
            mid_2d2 = Conv2D(3, (3, 3), padding='same')(mid_2d2)
            mid_2d3 = Conv2D(3, (3, 3), padding='same')(mid_2d3)
            mid_2d4 = Conv2D(3, (3, 3), padding='same')(mid_2d4)
            mid_2d5 = Conv2D(3, (3, 3), padding='same')(mid_2d5)
            mid_2d6 = Conv2D(3, (3, 3), padding='same')(mid_2d6)

        Fla_mid_1d = Flatten()(mid_1d)
        Fla_mid_2d1 = Flatten()(mid_2d1)
        Fla_mid_2d2 = Flatten()(mid_2d2)
        Fla_mid_2d3 = Flatten()(mid_2d3)
        Fla_mid_2d4 = Flatten()(mid_2d4)
        Fla_mid_2d5 = Flatten()(mid_2d5)
        Fla_mid_2d6 = Flatten()(mid_2d6)

        con = concatenate([Fla_mid_1d,
                           Fla_mid_2d1, Fla_mid_2d2, Fla_mid_2d3, Fla_mid_2d4, Fla_mid_2d5, Fla_mid_2d6])

        mid_a = Dense(100, activation='sigmoid', use_bias=True)(con)

        for i in range(self.L2):
            mid_a = Dense(10, activation='sigmoid', use_bias=True)(mid_a)
            mid_a = Dropout(0.05)(mid_a)

        out = Dense(1, activation='relu',
                    name='out', use_bias=True)(mid_a)

        model = Model([input_1d, input_2d1, input_2d2, input_2d3,
                       input_2d4, input_2d5, input_2d6], out)
        # sgd = keras.optimizers.SGD(
        #     lr=0.11, momentum=0.0, decay=0.0, nesterov=False)

        # model.compile(loss='categorical_crossentropy',  # 对数损失
        #               optimizer='adam',
        #               metrics=['accuracy'])

        sgd1 = keras.optimizers.SGD(
            lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        # model.compile(loss='mean_squared_error',  # 对数损失
        #               optimizer=sgd1,
        #               metrics=['accuracy'])
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        model.summary()

        return model

    def TrainAnn(self, epochs):

        if self.type == 'Ann':
            self.model.fit(
                [self.train_feature_1d], self.train_label, batch_size=self.batch_size, epochs=epochs,
                shuffle=True,
                # callbacks=[self.history]
                verbose=1, validation_data=([self.test_feature_1d], self.test_label),
            )

        elif self.type == 'HDNNM':

            tf1 = [self.train_feature_1d.reshape((-1, self.len_feature_1d, 1)), self.train_feature_2d1, self.train_feature_2d2,
                   self.train_feature_2d3, self.train_feature_2d4, self.train_feature_2d5, self.train_feature_2d6]

            tl = [self.test_feature_1d.reshape((-1, self.len_feature_1d, 1)), self.test_feature_2d1, self.test_feature_2d2,
                  self.test_feature_2d3, self.test_feature_2d4, self.test_feature_2d5, self.test_feature_2d6]

            self.model.fit(
                tf1, self.train_label, batch_size=self.batch_size, epochs=epochs,
                # callbacks=[self.history]
                verbose=1, validation_data=(tl, self.test_label),
            )

        else:
            raise NameError('the type of the network have to be Ann or HDNNM')

        # acc,loss,hismak = self.history.loss_plot('epoch')
        # np.savetxt('acc_L1:{}_L2:{}.csv'.format(self.L1,self.L2), acc, delimiter=',')
        # np.savetxt('loss_L1:{}_L2:{}.csv'.format(self.L1,self.L2), loss, delimiter=',')
        # np.savetxt('hismak_L1:{}_L2:{}.csv'.format(self.L1,self.L2), hismak, delimiter=',')

    def savenetwork(self):
        # This function is used to save the the medol trained above

        time_now = int(time.time())
        time_local = time.localtime(time_now)
        time1 = time.strftime("%Y_%m_%d::%H_%M_%S", time_local)
        savename = 'model/ann_stru={}_time={}_correct={}_L1:{}_L2:{}.h5'.format(
            ''.join(self.number_of_n), time1, self.correct, self.L1, self.L2)
        self.model.save(savename)
        return savename

    def normalize_1d_feature(self):
        # self.features_1D_list
        self.number_of_1D_feature = self.features_1D_list.shape[1]
        for i in range(self.number_of_1D_feature):
            Fitem = self.features_1D_list[:, i]
            self.features_1D_list[:, i] = (
                Fitem - np.min(Fitem))/(np.max(Fitem) - np.min(Fitem))

    def GetConfusionMatraxOnce(self):
        prob = JSSPproblem.Problem(
            self.number_of_machines, self.number_of_jobs, time_low=self.timelow, time_high=self.timehigh)
        prob.SoluteWithBBM()
        X_optimal = prob.GetIndexMatrix()

        if self.type == "Ann":
            featrues = prob.GetFeaturesInTest1D()
        else:
            featrues = prob.GetFeaturesInTest1D2D()

        output = self.PredictWithnetwork(featrues)
        prob.SchedulingSequenceGenerationMethod(output)
        X_solution = prob.GetIndexMatrix()

        confusion_matrix = np.zeros((self.number_of_jobs, self.number_of_jobs))

        for i in range(self.number_of_machines):
            for j in range(self.number_of_jobs):
                right = int(X_optimal[i, j])
                predict = int(X_solution[i, j])
                confusion_matrix[right, predict] += 1

        return confusion_matrix

    def GetConfusionMatraxNTimes(self, times):
        confusion_matrix = np.zeros((self.number_of_jobs, self.number_of_jobs))

        for i in range(times):
            confusion_matrix += self.GetConfusionMatraxOnce()

        return confusion_matrix

    def LoadData(self, info):

        features_1D_list = []
        features_2D_1 = []
        features_2D_2 = []
        features_2D_3 = []
        features_2D_4 = []
        features_2D_5 = []
        features_2D_6 = []
        label_list = []

        name = 'bigdata/feature/jssp_feature_'+info
        with open(name, 'r') as f:
            while True:
                data = [f.readline() for i in range(7)]
                if data[0] != '':

                    len_of_1D_features = np.fromstring(
                        data[0][:-1], dtype=float, sep=',').shape[0]
                    len_of_n_muti_n = int(np.sqrt(np.fromstring(
                        data[1][:-1], dtype=float, sep=',').shape))
                    len_of_n = int(np.sqrt(len_of_n_muti_n))

                    number_of_jobs = len_of_n
                    number_of_jobs_2 = len_of_n_muti_n

                    features_1D = np.fromstring(
                        data[0][:-1], dtype=float, sep=',')
                    featrue_2D = []
                    F1 = np.fromstring(
                        data[1][:-1], dtype=float, sep=',').reshape((len_of_n_muti_n, len_of_n_muti_n))
                    F2 = np.fromstring(
                        data[2][:-1], dtype=float, sep=',').reshape((len_of_n_muti_n, len_of_n))
                    F3 = np.fromstring(
                        data[3][:-1], dtype=float, sep=',').reshape((len_of_n_muti_n, len_of_1D_features))
                    F4 = np.fromstring(
                        data[4][:-1], dtype=float, sep=',').reshape((len_of_n, len_of_n))
                    F5 = np.fromstring(
                        data[5][:-1], dtype=float, sep=',').reshape((len_of_n, len_of_1D_features))
                    F6 = np.fromstring(
                        data[6][:-1], dtype=float, sep=',').reshape((len_of_1D_features, len_of_1D_features))

                    features_1D_list.append(features_1D)
                    features_2D_1.append(F1)
                    features_2D_2.append(F2)
                    features_2D_3.append(F3)
                    features_2D_4.append(F4)
                    features_2D_5.append(F5)
                    features_2D_6.append(F6)
                else:
                    break

        name = 'bigdata/feature/jssp_label_'+info
        with open(name, 'r') as f:
            while True:
                data = f.readline()
                if data != '':
                    label_list.append(int(data))
                else:
                    break

        # one hot

        self.features_1D_list, self.normalize_par_1D = self.Normalize_feature(
            np.array(features_1D_list))
        self.features_2D_1, self.normalize_par_2D_1 = self.Normalize_feature(
            np.array(features_2D_1))
        self.features_2D_2, self.normalize_par_2D_2 = self.Normalize_feature(
            np.array(features_2D_2))
        self.features_2D_3, self.normalize_par_2D_3 = self.Normalize_feature(
            np.array(features_2D_3))
        self.features_2D_4, self.normalize_par_2D_4 = self.Normalize_feature(
            np.array(features_2D_4))
        self.features_2D_5, self.normalize_par_2D_5 = self.Normalize_feature(
            np.array(features_2D_5))
        self.features_2D_6, self.normalize_par_2D_6 = self.Normalize_feature(
            np.array(features_2D_6))

        self.number_of_class = max(label_list) + 1
        self.number_of_data = len(label_list)
        # self.label_list = np.zeros(
        #     (self.number_of_data, self.number_of_class), dtype=float)
        # for i, item in enumerate(label_list):
        #     self.label_list[i, item] = 1.
        self.label_list = label_list
        # self.normalize_1d_feature()

        divide1 = int(0.7*self.number_of_data/self.number_of_machines /
                      self.number_of_jobs)*self.number_of_machines*self.number_of_jobs

        self.train_feature_1d = self.features_1D_list[:divide1, :]
        self.train_feature_2d1 = self.features_2D_1[:divide1, :].reshape(
            (-1, self.number_of_jobs_2, self.number_of_jobs_2, 1))
        self.train_feature_2d2 = self.features_2D_2[:divide1, :].reshape(
            (-1, self.number_of_jobs_2, self.number_of_jobs, 1))
        self.train_feature_2d3 = self.features_2D_3[:divide1, :].reshape(
            (-1, self.number_of_jobs_2, self.len_feature_1d, 1))
        self.train_feature_2d4 = self.features_2D_4[:divide1, :].reshape(
            (-1, self.number_of_jobs, self.number_of_jobs, 1))
        self.train_feature_2d5 = self.features_2D_5[:divide1, :].reshape(
            (-1, self.number_of_jobs, self.len_feature_1d, 1))
        self.train_feature_2d6 = self.features_2D_6[:divide1, :].reshape(
            (-1, self.len_feature_1d, self.len_feature_1d, 1))
        self.train_label = np.array(
            self.label_list[:divide1], dtype=float)/self.number_of_jobs
        self.test_feature_1d = self.features_1D_list[divide1:, :]
        self.test_feature_2d1 = self.features_2D_1[divide1:, :].reshape(
            (-1, self.number_of_jobs_2, self.number_of_jobs_2, 1))
        self.test_feature_2d2 = self.features_2D_2[divide1:, :].reshape(
            (-1, self.number_of_jobs_2, self.number_of_jobs, 1))
        self.test_feature_2d3 = self.features_2D_3[divide1:, :].reshape(
            (-1, self.number_of_jobs_2, self.len_feature_1d, 1))
        self.test_feature_2d4 = self.features_2D_4[divide1:, :].reshape(
            (-1, self.number_of_jobs, self.number_of_jobs, 1))
        self.test_feature_2d5 = self.features_2D_5[divide1:, :].reshape(
            (-1, self.number_of_jobs, self.len_feature_1d, 1))
        self.test_feature_2d6 = self.features_2D_6[divide1:, :].reshape(
            (-1, self.len_feature_1d, self.len_feature_1d, 1))
        self.test_label = np.array(
            self.label_list[divide1:])/self.number_of_jobs

        print('finish load data at {}'.format(name))
        # self.label_list = label_list

    def Normalize_feature(self, data):
        datashape = data.shape
        if len(datashape) == 2:
            normalize_parmter = np.zeros((datashape[1], 2))
            for i in range(datashape[1]):

                normalize_parmter[i, 1] = (
                    np.max(data[:, i])-np.min(data[:, i]))/2
                normalize_parmter[i, 0] = 1 + \
                    np.min(data[:, i])/normalize_parmter[i, 1]
                if normalize_parmter[i, 1] < 0.01:
                    data[:, i] = 0
                else:
                    data[:, i] = data[:, i] / \
                        normalize_parmter[i, 1] - normalize_parmter[i, 0]
                # print("s")
        elif len(datashape) == 3:
            normalize_parmter = np.zeros((datashape[1], datashape[2], 2))
            for i in range(datashape[1]):
                for j in range(datashape[2]):
                    normalize_parmter[i, j, 1] = (
                        np.max(data[:, i, j])-np.min(data[:, i, j]))/2
                    normalize_parmter[i, j, 0] = 1 + \
                        np.min(data[:, i, j])/normalize_parmter[i, j, 1]
                    if normalize_parmter[i, j, 1] < 0.01:
                        data[:, i, j] = 0
                    else:
                        data[:, i, j] = data[:, i, j] / \
                            normalize_parmter[i, j, 1] - \
                            normalize_parmter[i, j, 0]
        return data, normalize_parmter

    def SaveModel(self):

        name = 'model_type={}_m={}_n={}time={}.h5'.format(
            self.type, self.number_of_machines, self.number_of_jobs, 0)
        if self.type == "HDNNM":
            name = 'model_type={}_m={}_n={}time={}L1_{}L2_{}.h5'.format(
                self.type, self.number_of_machines, self.number_of_jobs, 0,self.L1,self.L2)            
        self.model.save('bigdata/model/'+name)
        # save the normalize

        name1 = 'bigdata/model/normalize_parm_type={}_m={}_n={}time={}.txt'.format(
            self.type, self.number_of_machines, self.number_of_jobs, 0)
        f = open(name1, 'a')
        f.write(JSSPproblem.TranslateNpToStr(self.normalize_par_1D))
        f.write(JSSPproblem.TranslateNpToStr(self.normalize_par_2D_1))
        f.write(JSSPproblem.TranslateNpToStr(self.normalize_par_2D_2))
        f.write(JSSPproblem.TranslateNpToStr(self.normalize_par_2D_3))
        f.write(JSSPproblem.TranslateNpToStr(self.normalize_par_2D_4))
        f.write(JSSPproblem.TranslateNpToStr(self.normalize_par_2D_5))
        f.write(JSSPproblem.TranslateNpToStr(self.normalize_par_2D_6))
        f.close()
        print('finish save the model')
        return name

    def LoadModel(self, path):

        name = 'bigdata/model/'+path
        self.model = keras.models.load_model(name)
        name1 = 'bigdata/model/normalize_parm' + path[5:-2] + 'txt'
        f = open(name1, 'r')
        for i in range(7):
            line = f.readline()[:-3]
            item_list = line.split(',')
            parmlist = []
            for index, item in enumerate(item_list):

                num = float(item)
                if index % 2 == 0:
                    box = []
                    box.append(num)
                else:
                    box.append(num)
                    parmlist.append(box)
            if i == 0:
                self.normalize_par_1D = np.array(parmlist)
            if i == 1:
                self.normalize_par_2D_1 = np.array(parmlist).reshape(
                    (self.number_of_jobs_2, self.number_of_jobs_2, 2))
            if i == 2:
                self.normalize_par_2D_2 = np.array(parmlist).reshape(
                    (self.number_of_jobs_2, self.number_of_jobs, 2))
            if i == 3:
                self.normalize_par_2D_3 = np.array(parmlist).reshape(
                    (self.number_of_jobs_2, self.len_feature_1d, 2))
            if i == 4:
                self.normalize_par_2D_4 = np.array(parmlist).reshape(
                    (self.number_of_jobs, self.number_of_jobs, 2))
            if i == 5:
                self.normalize_par_2D_5 = np.array(parmlist).reshape(
                    (self.number_of_jobs, self.len_feature_1d, 2))
            if i == 6:
                self.normalize_par_2D_6 = np.array(parmlist).reshape(
                    (self.len_feature_1d, self.len_feature_1d, 2))
            else:
                pass

        print('successful load model')

    def PredictWithnetwork(self, input):

        if self.type == 'Ann':
            for i in range(11):
                if self.normalize_par_1D[i, 1] < 0.01:
                    input[:, i] = 0
                else:
                    # input[:, i] = (input[:, i]-self.normalize_par_1D[i, 0]
                    #                ) / self.normalize_par_1D[i, 1]
                    input[:, i] = input[:, i] / \
                        self.normalize_par_1D[i, 1]-self.normalize_par_1D[i, 0]
            output = self.model.predict(input)

        elif self.type == 'HDNNM':

            for i in range(11):
                if self.normalize_par_1D[i, 1] < 0.01:
                    input[0][:, i] = 0
                else:
                    input[0][:, i] = input[0][:, i] / \
                        self.normalize_par_1D[i, 1]-self.normalize_par_1D[i, 0]

            input[1] = self.NormalWithParm(self.normalize_par_2D_1, input[1])
            input[2] = self.NormalWithParm(self.normalize_par_2D_2, input[2])
            input[3] = self.NormalWithParm(self.normalize_par_2D_3, input[3])
            input[4] = self.NormalWithParm(self.normalize_par_2D_4, input[4])
            input[5] = self.NormalWithParm(self.normalize_par_2D_5, input[5])
            input[6] = self.NormalWithParm(self.normalize_par_2D_6, input[6])
            output = self.model.predict(input)
        return output

    def NormalWithParm(self, parm, input):
        inputshape = input.shape

        for i in range(inputshape[1]):
            for j in range(inputshape[2]):
                if parm[i, j, 1] < 0.01 or parm[i, j, 1] > 100000:
                    input[:, i, j, 0] = 0
                else:
                    input[:, i, j, 0] = input[:, i, j, 0] / \
                        parm[i, j, 1] - \
                        parm[i, j, 0]

        return input

    def GettestData(self, index):

        start = self.number_of_machines*self.number_of_jobs*index
        end = self.number_of_machines*self.number_of_jobs*(index+1)

        return self.test_feature_1d[start:end, :]

    def TestTheNetworkRandomlyOnce(self):

        prob = JSSPproblem.Problem(
            self.number_of_machines, self.number_of_jobs, time_low=self.timelow, time_high=self.timehigh)
        prob.SoluteWithBBM()
        best_mak = prob.GetMakespan()
        # prob.PlotResult()

        if self.type == "Ann":
            featrues = prob.GetFeaturesInTest1D()
        else:
            featrues = prob.GetFeaturesInTest1D2D()

        output = self.PredictWithnetwork(featrues)
        # prob.SchedulingSequenceGenerationMethod(output)
        # s = prob.CalculateSimilarityDegree()
        # print('SchedulingSequenceGenerationMethod:',s)
        prob.PriorityQueuingMethod(output)
        s = prob.CalculateSimilarityDegree()
        print('similary:', s)
        ann_mak = prob.GetMakespan()
        # prob.PlotResult()
        # plt.show()
        return best_mak, ann_mak

    def TestTheNetworkRandomlyntimes(self, times):

        best_makspan_list = []
        out_makspan_list = []
        for i in range(times):
            best_makspan, out_makspan = self.TestTheNetworkRandomlyOnce()
            best_makspan_list.append(best_makspan)
            out_makspan_list.append(out_makspan)

        return np.array(best_makspan_list), np.array(out_makspan_list)


# def main(dataset='featureandlable_traindata_m=8_n=8_timelow=6_timehight=30_numofloop=1000.csv'):

#     # ann_model = NeuralNetwork('Ann', [11, 22, 22, 22, 22, 4], batch_size=64)
#     ann_model.LoadData('')
#     ann_model.TrainAnn(1000)


if __name__ == '__main__':
    pass
    # main()
