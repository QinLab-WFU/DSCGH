from math import gamma
from numpy.lib.function_base import select
import torch
import torch.nn as NN
from scipy.linalg import hadamard, eig
import numpy as np
import random
from scipy.special import comb
import pdb
import itertools
import copy
from tqdm import tqdm
import torch.nn.functional as F
from utils.get_args import get_args
import xlrd
import math


class OurLossZero(NN.Module):
    def __init__(self, bit):
        """
        :param config: in paper, the hyper-parameter lambda is chose 0.0001
        :param bit:
        """
        super(OurLossZero, self).__init__()
        self.config = get_args()
        self.bit = bit
        # self.alpha_pos, self.alpha_neg, self.beta_neg, self.d_min, self.d_max = self.get_margin()
        self.hash_center = self.generate_center()
        # np.save(f"./results/{self.config.dataset}/{'OurLossDzero'}/CSQ_{self.config.output_dim}.npy", self.hash_center.cpu().numpy())
        # self.label_center = torch.from_numpy(
        #     np.eye(self.config.nb_class, dtype=np.float32)[np.array([i for i in range(self.config.nb_class)])]).to(1)

    def forward(self, u1, u2, y):
        return self.cos_pair(u1, u2, y)

    def cos_pair(self, u, u2, y):
        i_cos_loss = self.cos_eps_loss(u, y)
        t_cos_loss = self.cos_eps_loss(u2, y)

        # i_Q_loss = (u.abs() - 1).pow(2).mean()
        # t_Q_loss = (u2.abs() - 1).pow(2).mean()
        # + 0.0001 * (i_Q_loss + t_Q_loss)

        loss = (i_cos_loss + t_cos_loss)
        return loss

    def cos_eps_loss(self, u, y):
        K = self.bit
        P_one_hot = y

        u_norm = F.normalize(u)
        centers_norm = F.normalize(self.hash_center)

        cos_sim = torch.matmul(u_norm, torch.transpose(centers_norm, 0, 1))  # batch x n_class

        sheet = xlrd.open_workbook('/home/admin00/HYD/DCGH/Metric Learning/utils/codetable.xlsx').sheet_by_index(0)
        threshold = sheet.row(K)[math.ceil(math.log(self.config.nb_class, 2))].value

        pos = 1 - cos_sim
        neg = F.relu(cos_sim - threshold)

        P_num = len(P_one_hot.nonzero())
        N_num = len((P_one_hot == 0).nonzero())

        pos_term = torch.where(P_one_hot == 1, pos.to(torch.float32),
                               torch.zeros_like(cos_sim).to(torch.float32)).sum() / P_num
        neg_term = torch.where(P_one_hot == 0, neg.to(torch.float32),
                               torch.zeros_like(cos_sim).to(torch.float32)).sum() / N_num

        # s = (y @ self.label_center.t()).float()  # batch x n_class
        # loss = 0
        # for i in range(u_norm.shape[0]):
        #     pos_pair = cos_sim[i][s[i] == 1]  # 1
        #     neg_pair = cos_sim[i][s[i] == 0]  # n_class - 1
        #     pos_eps = torch.exp(K ** 0.5 * (pos_pair - m))
        #     neg_eps = torch.exp(K ** 0.5 * (neg_pair - m))
        #     batch_loss = torch.log(pos_eps / (pos_eps + torch.sum(neg_eps))) + torch.sum(
        #         torch.log(1 - neg_eps / (pos_eps + torch.sum(neg_eps)))) / (self.config.nb_class - 1)
        #     loss += batch_loss
        # loss /= u_norm.shape[0]

        loss = pos_term + neg_term
        return loss

    def get_margin(self):
        L = self.bit
        n_class = self.config.nb_class
        right = (2 ** L) / n_class
        d_min = 0
        d_max = 0
        for j in range(2 * L + 4):
            dim = j
            sum_1 = 0
            sum_2 = 0
            for i in range((dim - 1) // 2 + 1):
                sum_1 += comb(L, i)
            for i in range((dim) // 2 + 1):
                sum_2 += comb(L, i)
            if sum_1 <= right and sum_2 > right:
                d_min = dim
        for i in range(2 * L + 4):
            dim = i
            sum_1 = 0
            sum_2 = 0
            for j in range(dim):
                sum_1 += comb(L, j)
            for j in range(dim - 1):
                sum_2 += comb(L, j)
            if sum_1 >= right and sum_2 < right:
                d_max = dim
        alpha_neg = L - 2 * d_max
        beta_neg = L - 2 * d_min
        alpha_pos = L
        return alpha_pos, alpha_neg, beta_neg, d_min, d_max

    def generate_center(self):
        hash_centers = np.load(f"./centers/CSQ_init_True_{self.config.dataset}_{self.config.nb_class}_{self.config.output_dim}_L2_alpha1.npy")
        self.evaluate_centers(hash_centers)
        Z = torch.from_numpy(hash_centers).float().to(1)
        return Z

    def evaluate_centers(self, H):
        dist = []
        for i in range(H.shape[0]):
            for j in range(i + 1, H.shape[0]):
                TF = np.sum(H[i] != H[j])
                dist.append(TF)
        dist = np.array(dist)
        st = dist.mean() - dist.var() + dist.min()
        print(f"mean is {dist.mean()}; min is {dist.min()}; var is {dist.var()}; max is {dist.max()}")