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


class NeuralNetwork:
    number_of_machines = 0
    number_of_jobs = 0
    number_of_class = 0
    model = []
    layer = 3
    number_of_n = [11, 11, 11, 11, 8]
    type = ''
    batch_size = 64
    correct = 0
    number_of_1D_feature = 0
    features_1D_list = []

    len_feature_1d = 0
    len_feature_nm = 0

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

    def __init__(self, type, number_of_n, batch_size, m, n):
        self.type = type
        self.number_of_machines = m
        self.number_of_jobs = n
        self.len_feature_1d = number_of_n[0]
        self.len_feature_nm = m * n
        self.number_of_class = number_of_n[-1]
        
        if self.type == 'Ann':
            self.layer = len(number_of_n)-3
            self.number_of_n = number_of_n
            self.batch_size = batch_size

            self.model = self.CreateANN2()

        elif self.type == 'HDNNM':

            self.layer = [len(number_of_n)-3, len(number_of_n)-3]
            self.number_of_n = number_of_n
            self.batch_size = batch_size
            self.model = self.CreateHDNNM()

        else:
            raise NameError('the type of the network have to be Ann or HDNNM')

    def CreateANN(self):

        input1 = Input(shape=(self.number_of_n[0], ), name='i1')
        mid = Dense(self.number_of_n[1],
                    activation='relu', use_bias=True)(input1)

        for i in range(self.layer):
            mid = Dense(self.number_of_n[2+i],
                        activation='relu', use_bias=True)(mid)
            mid = Dropout(0.25)(mid)

        mid = Dense(self.number_of_n[2+self.layer],
                    activation='relu', use_bias=True)(mid)
        out = Dense(self.number_of_n[-1], activation='softmax',
                    name='out', use_bias=True)(mid)

        model = Model(input1, out)
        # sgd = keras.optimizers.SGD(
        #     lr=0.11, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy',  # 对数损失
                      optimizer='adam',
                      metrics=['accuracy'])

        model.summary()

        return model

    def CreateANN2(self):

        input1 = Input(shape=(self.number_of_n[0] * self.number_of_jobs * self.number_of_machines  , ), name='i1')
        
        mid = Dense(400,
                    activation='relu', use_bias=True)(input1)

        for i in range(self.layer):
            mid = Dense(400,
                        activation='relu', use_bias=True)(mid)
            mid = Dropout(0.25)(mid)

        mid = Dense(200,
                    activation='relu', use_bias=True)(mid)
        out = Dense(self.number_of_n[-1]* self.number_of_jobs * self.number_of_machines, activation='softmax',
                    name='out', use_bias=True)(mid)

        model = Model(input1, out)
        # sgd = keras.optimizers.SGD(
        #     lr=0.11, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy',  # 对数损失
                      optimizer='adam',
                      metrics=['accuracy'])

        model.summary()

        return model

    def CreateHDNNM(self):

        input_1d = Input(shape=(self.number_of_n[0],1), name='i1d')
        input_2d1 = Input(shape=(self.len_feature_nm,self.len_feature_nm,1), name='i2d1')
        input_2d2 = Input(shape=(self.len_feature_nm,self.len_feature_nm,1), name='i2d2')
        input_2d3 = Input(shape=(self.len_feature_nm,self.len_feature_1d,1), name='i2d3')
        input_2d4 = Input(shape=(self.len_feature_nm,self.len_feature_nm,1), name='i2d4')
        input_2d5 = Input(shape=(self.len_feature_nm,self.len_feature_1d,1), name='i2d5')
        input_2d6 = Input(shape=(self.len_feature_1d,self.len_feature_1d,1), name='i2d6')

        mid_1d = Dense(self.number_of_n[1],
                       activation='relu', use_bias=True)(input_1d)
        mid_2d1 = Conv2D(3, (3, 3), padding='same')(input_2d1)
        mid_2d2 = Conv2D(3, (3, 3), padding='same')(input_2d2)
        mid_2d3 = Conv2D(3, (3, 3), padding='same')(input_2d3)
        mid_2d4 = Conv2D(3, (3, 3), padding='same')(input_2d4)
        mid_2d5 = Conv2D(3, (3, 3), padding='same')(input_2d5)
        mid_2d6 = Conv2D(3, (3, 3), padding='same')(input_2d6)

        for i in range(self.layer[0]):
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

        mid_a = Dense(100, activation='relu', use_bias=True)(con)

        for i in range(self.layer[1]):
            mid_a = Dense(10, activation='relu', use_bias=True)(mid_a)
            mid_a = Dropout(0.25)(mid_a)

        out = Dense(self.number_of_n[-1], activation='softmax',
                    name='out', use_bias=True)(mid_a)

        model = Model([input_1d, input_2d1, input_2d2, input_2d3,
                       input_2d4, input_2d5, input_2d6], out)
        # sgd = keras.optimizers.SGD(
        #     lr=0.11, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy',  # 对数损失
                      optimizer='adam',
                      metrics=['accuracy'])

        model.summary()

        return model

    def TrainAnn(self, epochs):

        trf = []
        trl = []
        tef = []
        tel = []
        nmutin = self.number_of_jobs*self.number_of_machines

        if self.type == 'Ann':


            for loop in range(7*len(self.features_1D_list )//(nmutin)//10):
                itemtrf = self.train_feature_1d[loop*nmutin:loop*nmutin+nmutin].reshape((nmutin*self.len_feature_1d))
                itemtrl = self.train_label[loop*nmutin:loop*nmutin+nmutin].reshape((nmutin*self.number_of_class))
                trf.append(itemtrf)
                trl.append(itemtrl)

            for loop in range(len(self.features_1D_list )* 3//(nmutin)//10):
                itemtef = self.test_feature_1d[loop*nmutin:loop*nmutin+nmutin].reshape((nmutin*self.len_feature_1d))
                itemtel = self.test_label[loop*nmutin:loop*nmutin+nmutin].reshape((nmutin*self.number_of_class))


                tef.append(itemtef)
                tel.append(itemtel)
            
            trf = np.array(trf)
            trl = np.array(trl)
            tef = np.array(tef)
            tel = np.array(tel)
            
            self.model.fit(
                trf, trl, batch_size=self.batch_size, epochs=epochs,
                verbose=1, validation_data=(tef, tel)
            )

            # self.model.fit(
            #     [self.train_feature_1d], self.train_label, batch_size=self.batch_size, epochs=epochs,
            #     verbose=1, validation_data=([self.test_feature_1d], self.test_label)
            # )

        elif self.type == 'HDNNM':

            tf = [ self.train_feature_1d.reshape((-1,self.len_feature_1d,1)) ,self.train_feature_2d1,self.train_feature_2d2,self.train_feature_2d3,self.train_feature_2d4,self.train_feature_2d5,self.train_feature_2d6 ]

            tl = [ self.test_feature_1d.reshape((-1,self.len_feature_1d,1)),self.test_feature_2d1,self.test_feature_2d2,self.test_feature_2d3,self.test_feature_2d4,self.test_feature_2d5,self.test_feature_2d6 ]

            self.model.fit(
                tf , self.train_label, batch_size=self.batch_size, epochs=epochs,
                verbose=1, validation_data=(tl, self.test_label)
            )

        else:
            raise NameError('the type of the network have to be Ann or HDNNM')



    def savenetwork(self):
        # This function is used to save the the medol trained above

        time_now = int(time.time())
        time_local = time.localtime(time_now)
        time1 = time.strftime("%Y_%m_%d::%H_%M_%S", time_local)
        savename = 'model/ann_stru={}_time={}_correct={}.h5'.format(
            ''.join(self.number_of_n), time1, self.correct)
        self.model.save(savename)
        return savename

    def normalize_1d_feature(self):
        # self.features_1D_list
        self.number_of_1D_feature = self.features_1D_list.shape[1]
        for i in range(self.number_of_1D_feature):
            Fitem = self.features_1D_list[:, i]
            self.features_1D_list[:, i] = (
                Fitem - np.min(Fitem))/(np.max(Fitem) - np.min(Fitem))

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
                    len_of_m_muti_n = int(np.sqrt(np.fromstring(
                        data[1][:-1], dtype=float, sep=',').shape))

                    features_1D = np.fromstring(
                        data[0][:-1], dtype=float, sep=',')
                    featrue_2D = []
                    F1 = np.fromstring(
                        data[1][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_m_muti_n))
                    F2 = np.fromstring(
                        data[2][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_m_muti_n))
                    F3 = np.fromstring(
                        data[3][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_1D_features))
                    F4 = np.fromstring(
                        data[4][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_m_muti_n))
                    F5 = np.fromstring(
                        data[5][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_1D_features))
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

        self.features_1D_list = np.array(features_1D_list)
        self.features_2D_1 = np.array(features_2D_1)
        self.features_2D_2 = np.array(features_2D_2)
        self.features_2D_3 = np.array(features_2D_3)
        self.features_2D_4 = np.array(features_2D_4)
        self.features_2D_5 = np.array(features_2D_5)
        self.features_2D_6 = np.array(features_2D_6)

        self.number_of_class = max(label_list) + 1
        self.number_of_data = len(label_list)
        self.label_list = np.zeros(
            (self.number_of_data, self.number_of_class), dtype=float)
        for i, item in enumerate(label_list):
            self.label_list[i, item] = 1.
        # self.normalize_1d_feature()

        divide = int(0.7*self.number_of_data/self.number_of_machines /
                     self.number_of_jobs)*self.number_of_machines*self.number_of_jobs
        
        self.train_feature_1d = self.features_1D_list[:divide, :]
        self.train_feature_2d1 = self.features_2D_1[:divide,:].reshape((-1,self.len_feature_nm ,self.len_feature_nm,1))
        self.train_feature_2d2 = self.features_2D_2[:divide,:].reshape((-1,self.len_feature_nm,self.len_feature_nm,1))
        self.train_feature_2d3 = self.features_2D_3[:divide,:].reshape((-1,self.len_feature_nm,self.len_feature_1d,1))
        self.train_feature_2d4 = self.features_2D_4[:divide,:].reshape((-1,self.len_feature_nm,self.len_feature_nm,1))
        self.train_feature_2d5 = self.features_2D_5[:divide,:].reshape((-1,self.len_feature_nm,self.len_feature_1d,1))
        self.train_feature_2d6 = self.features_2D_6[:divide,:].reshape((-1,self.len_feature_1d,self.len_feature_1d,1))
        self.train_label = self.label_list[:divide, :]
        self.test_feature_1d = self.features_1D_list[divide:, :]
        self.test_feature_2d1 = self.features_2D_1[divide:,:].reshape((-1,self.len_feature_nm ,self.len_feature_nm,1))
        self.test_feature_2d2 = self.features_2D_2[divide:,:].reshape((-1,self.len_feature_nm ,self.len_feature_nm,1))
        self.test_feature_2d3 = self.features_2D_3[divide:,:].reshape((-1,self.len_feature_nm ,self.len_feature_1d,1))
        self.test_feature_2d4 = self.features_2D_4[divide:,:].reshape((-1,self.len_feature_nm ,self.len_feature_nm,1))
        self.test_feature_2d5 = self.features_2D_5[divide:,:].reshape((-1,self.len_feature_nm ,self.len_feature_1d,1))
        self.test_feature_2d6 = self.features_2D_6[divide:,:].reshape((-1,self.len_feature_1d ,self.len_feature_1d,1))
        self.test_label = self.label_list[divide:, :]

        print('finish load data at {}'.format(name))
        # self.label_list = label_list

    def SaveModel(self):
        name = 'model.h5'
        self.model.save('model/'+name)
        print('finish save the model')
        return name

    def LoadModel(self):

        self.model = keras.models.load_model('model/model.h5')
        print('successful load model')

    def PredictWithnetwork(self, input):
        if self.type == 'Ann':
            output = self.model.predict(input)
        elif self.type == 'HDNNM':
            output = self.model.predict(input)
        return output

    def GettestData(self, index):

        start = self.number_of_machines*self.number_of_jobs*index
        end = self.number_of_machines*self.number_of_jobs*(index+1)

        return self.test_feature_1d[start:end, :]


def main(dataset='featureandlable_traindata_m=8_n=8_timelow=6_timehight=30_numofloop=1000.csv'):

    ann_model = NeuralNetwork('Ann', [11, 22, 22, 22, 22, 4], batch_size=64)
    ann_model.LoadData('')
    ann_model.TrainAnn(1000)


if __name__ == '__main__':
    main()
