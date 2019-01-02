import numpy as np
import matplotlib.pyplot as plt
import JSSPproblem
import random


class Subproblem:
    father_problem = None
    machine_id = None
    job_id = None
    procedure = None
    time = None
    num_of_machine = None
    num_of_job = None
    number_of_1D_feature = None
    features_1D = None
    features_2D = None
    order_in_machine = None

    label = []

    def __init__(self, fatherproblem, jobs, procedure):
        self.father_problem = fatherproblem
        self.job_id = jobs
        self.procedure = procedure
        self.machine_id = fatherproblem.r[jobs, procedure] - 1
        self.time = self.father_problem.p[self.job_id, self.machine_id]
        self.num_of_job = fatherproblem.p.shape[0]
        self.num_of_machine = fatherproblem.p.shape[1]

        self.label = self.SearchInX()
        self.time2 = self.label
        self.order_in_machine = self.SearchInX()
        self.features_1D = self.GetFeatures1D()
        self.features_2D = self.GetFeatures2D()

    def SearchInX(self):

        for i in range(self.num_of_machine):
            for j in range(self.num_of_job):
                if self.job_id == self.father_problem.x[i, j, 0] and self.procedure == self.father_problem.x[i, j, 1]:
                    self.order_in_machine = j
                    return j
        return 'error'

    def CheckOrderInMachine(self):
        father_problem = self.father_problem
        joblist_in_machine = []
        for job in range(father_problem.n):
            for pro in range(father_problem.m):
                machine = father_problem.r[job, pro]
                if machine - 1 == self.machine_id:
                    joblist_in_machine.append([job, pro])

        joblist_in_machine.sort(key=lambda d: d[1])
        index = joblist_in_machine.index([self.job_id, self.procedure])
        return index

    def GetFeatures1D(self):
        # p [job,order]
        # r [job,order]

        features = np.zeros((11))

        T_total = self.father_problem.p.sum()
        T_machine = self.father_problem.p[:, self.machine_id].sum()
        T_job = self.father_problem.p[self.job_id, :].sum()
        T_item = self.time
        order_of_procedure_in_machine = self.CheckOrderInMachine()

        features[0] = self.procedure  # 加工顺序
        features[1] = T_item  # 当前机器当前工序加工时间

        features[2] = self.father_problem.p[self.job_id, # 当前工件已经加工的时间
                                            :self.procedure].sum() / T_job 
        features[3] = self.father_problem.p[self.job_id, # 当前工件剩余加工时间
                                             self.procedure:].sum() / T_job 

        features[5] = self.job_id
        # features[4] = T_total/1000
        # features[4] = T_item/T_total
        # features[5] = T_item/T_machine
        features[6] = T_item/T_machine
        features[7] = T_item/T_total
        features[8] = T_job/T_machine 
        features[9] = np.sum(
            self.father_problem.r[:, self.procedure] == self.machine_id)/self.father_problem.n
        features[10] = order_of_procedure_in_machine/self.father_problem.n
        # features[4] = self.father_problem.p[self.job_id,
        #                                     :self.procedure].sum()/T_job
        features[4] = self.machine_id
        # features[4] = self.father_problem.e[self.machine_id,
        #                                     self.order_in_machine]

        self.number_of_1D_feature = features.shape[0]
        return features

    def GetPLine(self):
        out = np.zeros((self.father_problem.n*self.father_problem.n, 1))
        sum_time_job = self.father_problem.p.sum(axis=1)
        for i in range(self.father_problem.n):
            for j in range(self.father_problem.n):
                out[i*self.father_problem.n + j,
                    0] = sum_time_job[i]/sum_time_job[j]
        # assert(sum_time_job.shape == self.father_problem.n)
        return out

    def GetELine(self):
        out = np.zeros((self.father_problem.n, 1))
        sum_time_job = self.father_problem.p.sum(axis=1)
        for i in range(self.father_problem.n):
            out[i, 0] = self.time/sum_time_job[i]
        # assert(sum_time_job.shape == self.father_problem.n)
        return out

    def GetFeatures2D(self):
        features = []

        p_line = self.GetPLine()
        E_line = self.GetELine()
        f_line = self.features_1D.reshape((-1, 1))

        features.append(np.dot(p_line, p_line.T))
        features.append(np.dot(p_line, E_line.T))
        features.append(np.dot(p_line, f_line.T))
        features.append(np.dot(E_line, E_line.T))
        features.append(np.dot(E_line, f_line.T))
        features.append(np.dot(f_line, f_line.T))

        return features

    def Show2DFeatures(self):

        plt.figure(figsize=(6, 4))

        plt.subplot(231)
        plt.imshow(self.features_2D[0])
        plt.xlabel(r'$D^{2d}_{P^l,P^l}$')
        plt.subplot(232)
        plt.imshow(self.features_2D[1].T)
        plt.xlabel(r'$D^{2d}_{P^l,E^l}$')
        plt.subplot(233)
        plt.imshow(self.features_2D[2].T)
        plt.xlabel(r'$D^{2d}_{P^l,F^{l}_{ij}}$')
        plt.subplot(234)
        plt.imshow(self.features_2D[3])
        plt.xlabel(r'$D^{2d}_{E^l,E^l}$')
        plt.subplot(235)
        plt.imshow(self.features_2D[4])
        plt.xlabel(r'$D^{2d}_{E^l,F^{l}_{ij}}$')
        plt.subplot(236)
        plt.imshow(self.features_2D[5])
        plt.xlabel(r'$D^{2d}_{F^{l}_{ij},F^{l}_{ij}}$')
        plt.tight_layout()
        plt.savefig('figure/n={}m={}order={}.png'.format(self.machine_id,
                                                         self.job_id, self.procedure), dpi=500)

        plt.close()
        # plt.show()

    def SaveFeaturesToFile(self, filepath, index, pool=0):

        father_problem = self.father_problem
        f = open('{}/jssp_feature_m={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
            filepath, father_problem.m, father_problem.n, father_problem.time_high, father_problem.time_low, pool), 'a')
        f.write(JSSPproblem.TranslateNpToStr(self.features_1D))
        f.write(JSSPproblem.TranslateNpToStr(self.features_2D[0]))
        f.write(JSSPproblem.TranslateNpToStr(self.features_2D[1]))
        f.write(JSSPproblem.TranslateNpToStr(self.features_2D[2]))
        f.write(JSSPproblem.TranslateNpToStr(self.features_2D[3]))
        f.write(JSSPproblem.TranslateNpToStr(self.features_2D[4]))
        f.write(JSSPproblem.TranslateNpToStr(self.features_2D[5]))
        f.close()

    def SaveLablesToFile(self, filepath, index, pool=0):

        father_problem = self.father_problem
        probinfo = 'm={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
            father_problem.m, father_problem.n, father_problem.time_high, father_problem.time_low, pool)

        f = open('{}/jssp_label_m={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
            filepath, father_problem.m, father_problem.n, father_problem.time_high, father_problem.time_low, pool), 'a')
        f.write(str(self.label)+'\n')
        f.close()

        return probinfo

    def LoadFeatures(self, filepath, index, pool=0):

        father_problem = self.father_problem
        f = open('{}/jssp_problem_m={}_n={}_timehigh={}_timelow={}_pool={}.txt'.format(
            filepath, father_problem.m, father_problem.n, father_problem.time_high, father_problem.time_low, pool), 'r')
        data = f.readlines()
        data = data[index * 7:index * 7 + 7]

        len_of_1D_features = np.fromstring(
            data[0][:-1], dtype=float, sep=',').shape[0]
        len_of_m_muti_n = int(np.sqrt(np.fromstring(
            data[1][:-1], dtype=float, sep=',').shape))

        self.features_1D = np.fromstring(data[0][:-1], dtype=float, sep=',')
        featrue_2D = []
        featrue_2D.append(np.fromstring(
            data[1][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_m_muti_n)))
        featrue_2D.append(np.fromstring(
            data[2][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_m_muti_n)))
        featrue_2D.append(np.fromstring(
            data[3][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_1D_features)))
        featrue_2D.append(np.fromstring(
            data[4][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_m_muti_n)))
        featrue_2D.append(np.fromstring(
            data[5][:-1], dtype=float, sep=',').reshape((len_of_m_muti_n, len_of_1D_features)))
        featrue_2D.append(np.fromstring(
            data[6][:-1], dtype=float, sep=',').reshape((len_of_1D_features, len_of_1D_features)))
        f.close()

        self.features_2D = featrue_2D
        return self.features_1D, self.features_2D

    def PrintInfo(self):
        print('machineid is {}, jobid is {}'.format(self.job_id, self.job_id))
        print(self.features_1D)
        print(self.features_2D)


# def main(index_cpu):

#     # for p in range(1000):
#     prob = JSSPproblem.Problem(2, 8, 10, 30)
#     prob.SoluteWithGAAndSaveToFile('data', 0)
#     prob.Print_info()
#     sub_list = prob.subproblem()
#     for i, subp in enumerate(sub_list):
#         subp.SaveFeaturesToFile('feature', i)
#         info = subp.SaveLablesToFile('feature', i)

    # sub_list[0].LoadFeatures('feature',0)


if __name__ == "__main__":
    # MultiCpu.MultiCpuRun(main,4)
    # main(0)
    pass
