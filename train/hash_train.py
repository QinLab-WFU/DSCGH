from torch.nn.modules import loss
from model.hash_model import DCMHT as DCMHT
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as scio


from .base import TrainBase
from model.optimization import BertAdam
from utils import get_args, calc_neighbor, cosine_similarity, euclidean_similarity
from utils.calc_utils import calc_map_k_matrix as calc_map_k
from utils.triplet_loss import TripletAllLoss
from dataset.dataloader import dataloader
from utils.oursLossZero import OurLossZero
import time

from utils.calb import calculate_C



class Trainer(TrainBase):

    def __init__(self,
                rank=1):
        args = get_args()
        super(Trainer, self).__init__(args, rank)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        HashModel = DCMHT
        self.model = HashModel(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                            writer=self.writer, logger=self.logger, is_train=self.args.is_train).to(self.rank)
        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))
        
        self.model.float()
        self.optimizer = BertAdam([
                    {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
                    # {'params':self.model.sw_model.parameters(), 'lr': self.args.clip_lr},
                    {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
                    {'params': self.model.text_hash.parameters(), 'lr': self.args.lr}
                    ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
                    b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
                    weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        self.criterion_tri_cos = TripletAllLoss(dis_metric='cos', reduction='sum')
        self.hashcenterloss = OurLossZero(self.args.output_dim)
        self.total_time = 0

        print(self.model)


    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset.")
        self.args.index_file = os.path.join("./dataset", self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join("./dataset", self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join("./dataset", self.args.dataset, self.args.label_file)
        train_data, query_data, retrieval_data = dataloader(captionFile=self.args.caption_file, 
                                        indexFile=self.args.index_file, 
                                        labelFile=self.args.label_file, 
                                        maxWords=self.args.max_words,
                                        imageResolution=self.args.resolution,
                                        query_num=self.args.query_num,
                                        train_num=self.args.train_num,
                                        seed=self.args.seed)
        self.train_labels = train_data.get_all_label().to(1)
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
        self.train_loader = DataLoader(
                dataset=train_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )
        self.query_loader = DataLoader(
                dataset=query_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )
        self.retrieval_loader = DataLoader(
                dataset=retrieval_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )

    def train_epoch(self, epoch, H_i, H_t, B):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d"%(epoch, self.args.epochs))
        all_loss = 0
        times = 0
        for image, text, label, index in self.train_loader:
            start_time = time.time()
            # self.global_step += 1
            # times += 1
            image.float()

            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True)
            label = label.float()

            # index = index.numpy()

            hash_img, hash_text = self.model(image, text)

            H_i[index, :] = hash_img.data
            H_t[index, :] = hash_text.data
            i_ql = torch.sum(torch.pow(B[index, :] - hash_img, 2))
            t_ql = torch.sum(torch.pow(B[index, :] - hash_text, 2))
            loss_quant = i_ql + t_ql

            loss1 = self.hashcenterloss(hash_img, hash_text, label)


            loss2 = self.criterion_tri_cos(hash_img, label, hash_text, margin=0.4)
            loss3 = self.criterion_tri_cos(hash_text, label, hash_img, margin=0.4)


            loss = loss1 + 1.0 * (loss2 + loss3) + 0.0001 * loss_quant
            all_loss += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.total_time += time.time() - start_time

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, time: {self.total_time}")

    def train(self):
        self.logger.info("Start train.")

        B = torch.randn(self.args.train_num, self.args.output_dim).sign().to(1)
        H_i = torch.randn(self.args.train_num, self.args.output_dim).to(1)
        H_t = torch.randn(self.args.train_num, self.args.output_dim).to(1)
        R = torch.zeros(self.args.nb_class, self.args.output_dim).to(1)
        C = B
        D = 0

        for epoch in range(self.args.epochs):
            self.train_epoch(epoch, H_i, H_t, B)

            R = torch.inverse(self.train_labels.t() @ self.train_labels + 1 * torch.eye(self.args.nb_class, device='cuda:1')) @ self.train_labels.t() @ B
            B = (self.train_labels @ R + self.args.gamma * (C - D / self.args.gamma) + 0.5 * (H_i + H_t)).sign()
            C = calculate_C(B + D / self.args.gamma)
            D = D + self.args.gamma * (B - C)
            self.args.gamma = 1.05 * 0.01
            self.valid(epoch)
            # self.save_model(epoch)

        self.logger.info(f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")


    def get_code(self, data_loader, length: int):

        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        encoder_time = 0
        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            image_hash = self.model.encode_image(image)
            image_hash = torch.sign(image_hash)
            text_hash = self.model.encode_text(text)
            text_hash = torch.sign(text_hash)
            encoder_time = time.time() - start_encoder_time
            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data
        
        return img_buffer, text_buffer, encoder_time


    def test(self, mode_name="i2t"):
        if self.args.pretrained == "":
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")
        self.change_state(mode="valid")
        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)
        query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.args.retrieval_num)
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        # print("map map")
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f">>>>>> MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}")

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(">>>>>> save all data!")


    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.args.retrieval_num)
        # print("get all code")
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        # print("map map")
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, None, self.rank)
        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t")
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i")
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, \
                    MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}, query_encoder_time: {q_encoder_time}, retrieval_encoder_time: {r_encoder_time}")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t"):

        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(f">>>>>> save best {mode_name} data!")