import os
import torch
import logging
import torch.nn as nn
import numpy as np
from typing import Union
from model.swin_model import SwinTransformer
from model.model import build_model
from utils import get_logger, get_summary_writer

# import torchvision
# class Res(nn.Module):
#     def __init__(self):
#         super(Res, self).__init__()
#         self.resnet = torchvision.models.resnet18(pretrained=True)
#         self.resnet.fc = nn.Linear(512,512)
#     def forward(self, x):
#         x = self.resnet(x)
#         # x = self.resnet.fc(x)
#         return x

# class ImgNet(nn.Module):
#     def __init__(self):
#         super(ImgNet, self).__init__()
#         self.alexnet = torchvision.models.alexnet(pretrained=True)
#         self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
#         self.hash_layer = nn.Linear(4096, 512)
#         self.alpha = 1.0
#
#         # self.features = nn.Sequential(
#         #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#         #     nn.ReLU(inplace=True),
#         #     nn.MaxPool2d(kernel_size=3, stride=2),
#         #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
#         #     nn.ReLU(inplace=True),
#         #     nn.MaxPool2d(kernel_size=3, stride=2),
#         #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
#         #     nn.ReLU(inplace=True),
#         #     nn.MaxPool2d(kernel_size=3, stride=2),
#         # )
#         # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         # self.classifier = nn.Sequential(
#         #     nn.Dropout(p=0.5),
#         #     nn.Linear(256 * 6 * 6, 4096),
#         #     nn.ReLU(inplace=True),
#         #     nn.Dropout(p=0.5),
#         #     nn.Linear(4096, 4096),
#         # )
#         # self.hash_layer = nn.Linear(4096, code_len)
#
#     def forward(self, x):
#         x = self.alexnet.features(x)
#         x = x.view(x.size(0), -1)
#         feat = self.alexnet.classifier(x)
#         feat = self.hash_layer(feat)
#         return feat

# class TNET(nn.Module):
#     def __init__(self):
#         super(TNET, self).__init__()
#         self.text_module = nn.Sequential(
#             nn.Linear(1386, 4096),
#             nn.ReLU(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(),
#             nn.Linear(4096, 512))
#     def forward(self, x):
#         x = self.text_module(x)
#         return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class LinearHash(nn.Module):

    def __init__(self, inputDim=2048, outputDim=64):
        super(LinearHash, self).__init__()
        self.fc = nn.Linear(inputDim, outputDim)
        # self.fc.apply(weights_init_kaiming)
        self.drop_out = nn.Dropout(p=0.2)
    
    def forward(self, data):
        result = self.fc(data)
        return torch.tanh(self.drop_out(result))


class DCMHT(nn.Module):

    def __init__(self, 
                outputDim=64, 
                clipPath="./ViT-B-32.pt", 
                writer=None, 
                saveDir="./result/log", 
                logger: logging.Logger=None, 
                is_train=True):
        super(DCMHT, self).__init__()
        os.makedirs(saveDir, exist_ok=True)
        self.logger = logger if logger is not None else get_logger(os.path.join(saveDir, "train.log" if is_train else "test.log"))
        self.writer = writer if writer is not None and is_train else get_summary_writer(os.path.join(saveDir, "tensorboard"))
        embedDim, self.clip = self.load_clip(clipPath)
        # if is_train:
        #     self.clip.eval()
        # print("start freezen")
        # self.freezen()
        self.image_hash = LinearHash(inputDim=embedDim, outputDim=outputDim)
        self.text_hash = LinearHash(inputDim=embedDim, outputDim=outputDim)
        # print(self.image_hash)
        # print(self.text_hash)
        # self.alexnet = ImgNet()
        # self.resnet = Res()
        # self.TNET = TNET()
        # weight = torch.load('./swin_pre.pth')['model']
        # self.sw_model = SwinTransformer(hash_length=outputDim)
        # sw_model_dict = self.sw_model.state_dict()
        # sw_model_dict.update(weight)
        # self.sw_model.load_state_dict(sw_model_dict, strict=False)

    def freezen(self):
        for name, param in self.clip.named_parameters():
            # print(name)
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                                        or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                # print("1")
                continue
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= 12:
                    # print("2")
                    continue
            if name.find("conv2.") == 0:
                # print("3")
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    def load_clip(self, clipPath: str) -> tuple:
        try:
            model = torch.jit.load(clipPath, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(clipPath, map_location="cpu")

        return state_dict["text_projection"].shape[1], build_model(state_dict)

    def encode_image(self, image):

        image_embed = self.clip.encode_image(image) #512
        image_embed = self.image_hash(image_embed)

        # image_embed = self.resnet(image)
        # image_embed = self.alexnet(image)
        # f_x = image_embed / torch.sqrt(torch.sum(image_embed.detach() ** 2))
        #
        # mask_img = torch.sigmoid(5 * f_x.squeeze().mm(feature_map.t()))  # size: (batch, num_label) 64*24
        # mask_f_x = mask_img.mm(feature_map) / mask_img.sum(dim=1).unsqueeze(-1)  # size: (batch, emb_dim) 64*512
        # mask_f_x = 0.8 * f_x + (1 - 0.8) * mask_f_x

        # self.sw_model.to(1)
        # image_embed = self.sw_model(image)
        # image_embed = self.image_hash(image_embed)

        return image_embed
    
    def eval(self):
        self.image_hash.eval()
        self.text_hash.eval()
        # self.clip.eval()

    def train(self):
        self.image_hash.train()
        self.text_hash.train()
    
    def encode_text(self, text):

        # text = torch.tensor(text, dtype=torch.float32)
        # text_embed = self.TNET(text)
        text_embed = self.clip.encode_text(text)

        # f_y = text_embed / torch.sqrt(torch.sum(text_embed.detach() ** 2))
        #
        # mask_txt = torch.sigmoid(5 * f_y.squeeze().mm(feature_map.t()))
        # mask_f_y = mask_txt.mm(feature_map) / mask_txt.sum(dim=1).unsqueeze(-1)
        # mask_f_y = 0.8 * f_y + (1 - 0.8) * mask_f_y

        text_embed = self.text_hash(text_embed)

        return text_embed

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)
        return image_embed, text_embed

