from torch.nn.modules import loss
from model.hash_model_cls import DCMHT as DCMHT
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
import numpy as np
import torch.nn.functional as F


class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()
        self.CELoss = torch.nn.CrossEntropyLoss().cuda()

    def forward(self, probs, labels, w):
        # labels = torch.argmax(labels, -1)
        celoss = self.CELoss(probs, labels)
        Q_loss = (w.abs()-1).pow(2).mean()
        return celoss + Q_loss

def top_k_accuracy(output, target):
    with torch.no_grad():
        target = torch.argmax(target, -1)
        total = target.shape[0]
        _, predicted = torch.max(output.data, 1)
        # predicted = F.one_hot(predicted, num_classes=24)
        total_correct = (predicted == target).sum().item()
    return total_correct, total


def test_val(model, test_loader):
    model.eval()
    acc = 0
    t_acc = 0
    total = 0
    t_total = 0
    with torch.no_grad():
        for img, txt, label, index in tqdm(test_loader):
            img = img.to(1)
            txt = txt.to(1)
            label = label.to(1)
            preds, t_preds = model(img, txt)
            temp_acc,  temp_batch= top_k_accuracy(preds, label)
            t_temp_acc, t_temp_batch = top_k_accuracy(t_preds, label)
            acc += temp_acc
            total += temp_batch
            t_acc += t_temp_acc
            t_total += t_temp_batch
    return 100 * acc / total, 100 * t_acc / t_total


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
        self.model = HashModel(label_size=self.args.nb_class, clipPath=self.args.clip_path,
                               writer=self.writer, logger=self.logger, is_train=self.args.is_train).to(self.rank)
        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))

        self.model.float()
        self.optimizer = BertAdam([
            {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
            # {'params':self.model.sw_model.parameters(), 'lr': self.args.clip_lr},
            {'params': self.model.hash.parameters(), 'lr': self.args.lr},
            # {'params': self.model.text_hash.parameters(), 'lr': self.args.lr}
        ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
            weight_decay=self.args.weight_decay, max_grad_norm=1.0)
        print(self.model)
        self.cross_entropy_loss = myLoss()
        self.best_acc = 0
        self.best_t_acc = 0

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

    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d" % (epoch, self.args.epochs))
        all_loss = 0
        times = 0
        for image, text, label, index in self.train_loader:
            self.global_step += 1
            times += 1
            image.float()

            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True)
            label = label.float()

            index = index.numpy()

            hash_img, hash_text = self.model(image, text)
            i_loss = self.cross_entropy_loss(hash_img, label, self.model.hash.fc.weight)
            t_loss = self.cross_entropy_loss(hash_text, label, self.model.hash.fc.weight)

            loss = i_loss + t_loss
            all_loss += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}")

    def train(self):
        self.logger.info("Start train.")

        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            self.valid(epoch)
            # self.save_model(epoch)

        self.logger.info(f">>>>>> 'Best_i_acc':{self.best_acc}, Best_t_acc: {self.best_t_acc}")

    def valid(self, epoch):

        self.acc, self.t_acc = test_val(self.model, self.query_loader)

        if self.acc + self.t_acc >= self.best_acc + self.best_t_acc:
            self.best_acc = self.acc
            self.best_t_acc = self.t_acc
            self.logger.info(f">>>>>> 'Best_i_acc':{self.best_acc}, Best_t_acc: {self.best_t_acc}, 'Best_epoch: {epoch}")
            torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "model-" + str(epoch) + ".pth"))
            self.model.eval()
            with torch.no_grad():
                W = self.model.hash.fc.weight.cpu().numpy()
                # W_t = self.model.text_hash.fc.weight.cpu().numpy()
            np.save(f'./weight/{self.args.dataset}_class_head.npy', W)
            # np.save(f'./weight/{self.args.dataset}_class_head_t.npy', W_t)