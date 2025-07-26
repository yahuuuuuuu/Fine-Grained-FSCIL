import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet12_encoder import *
from torch.nn import init
from CLIP import clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        self.CLIP, self.process = clip.load("ViT-L/14@336px")
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','mstar']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset in ['cub200']:
            self.encoder = resnet18(True,
                                    args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        if self.args.dataset in ['aircraft100', 'car100','StanfordDog']:
            self.encoder = ResNet12()  # pretrained=False
            self.num_features = 640
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.atten_features = (self.num_features // self.args.token_dim) * self.args.v_dim
        self.fc = nn.Linear(self.num_features, self.args.num_classes * 2, bias=False)
        self.T = self.args.moco_t
        self.projector = nn.Sequential(nn.Linear(self.num_features, self.num_features, bias=False), nn.ReLU(),
                                       nn.Linear(self.num_features, self.args.moco_dim, bias=False))
        self.clip_projector = nn.Sequential(nn.Linear(768, 768, bias=False), nn.ReLU(),
                                       nn.Linear(768, self.num_features, bias=False))
        #self.fc_ali = nn.Linear(self.atten_features, 768, bias=False)
        # self.q = nn.Linear(self.args.token_dim, self.args.q_dim, bias=False)
        # self.k = nn.Linear(self.args.token_dim, self.args.q_dim, bias=False)
        # self.v = nn.Linear(self.args.token_dim, self.args.v_dim, bias=False)
        # self.scale = self.args.q_dim ** -0.5

    def forward_metric(self, x):
        x_c = self.clip_encode(x)
        x = self.encode(x)
        if 'cos' in self.mode:
            # text = F.linear(F.normalize(x[:len(x) // 2], p=2, dim=-1), F.normalize(text, p=2, dim=-1))
            x_c = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(x_c, p=2, dim=-1))
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x
            x_c = self.args.temperature * x_c
            return x, x_c

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    def forward_incremental(self, x,fc):
        fu_x = self.encode(x)
        if 'cos' in self.mode:
            fu_x = F.linear(F.normalize(fu_x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
            fu_x = self.args.temperature * fu_x
            return fu_x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x
    
    def test_forward(self, x):
        fu_x = self.encode(x)
        if 'cos' in self.mode:
            fu_x = F.linear(F.normalize(fu_x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            fu_x = self.args.temperature * fu_x
            return fu_x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    # def selfattention(self, feature):
    #     b, dim = feature.shape
    #     n = self.num_features // self.args.token_dim
    #     feature = feature.view(b, n, self.args.token_dim)
    #     q = self.q(feature)
    #     k = self.k(feature)
    #     v = self.v(feature)
    #     similarity = (q @ k.transpose(-2, -1)) * self.scale
    #     similarity = similarity.softmax(dim=-1)
    #     feature = similarity @ v
    #     return feature.view(b, -1)

    def clip_encode(self, x):
        with torch.no_grad():
            data = F.interpolate(x, size=[336, 336], mode="bilinear", align_corners=False)
            image_feature = self.CLIP.encode_image(data)
            # if (text_inputs != None):
            #     text_feature = self.CLIP.encode_text(text_inputs)
            #     text_feature = text_feature.to(x.dtype)
                #text_feature = self.fc_clip(text_feature)
        image_feature = self.clip_projector(image_feature.to(x.dtype))
        return image_feature
    
    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        # x = self.selfattention(x)
        return x

    def forward(self, im_cla, im_q=None, im_k=None):
        if self.mode != 'encoder':
            if (im_q == None):
                input = self.test_forward(im_cla)
                return input
            else:
                logits_classify, x_c = self.forward_metric(im_cla)
                q = self.encode(im_q)
                q = self.projector(q)
                k = self.encode(im_k)  # keys: bs x dim
                k = self.projector(k)

                x1 = F.linear(F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1))
                x1 = x1 / self.T
                return logits_classify, x1, x_c
        elif self.mode == 'encoder':
            input = self.encode(im_cla)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self, dataloader, class_list, session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            trans_data = self.transform1(data)
            data = torch.cat((data, trans_data))
            trans_label = label * 2 + 1
            label = label * 2
            data = self.encode(data)

            #text_feature = text_feature.detach()
            data = data.detach()

            label = torch.cat((label, trans_label))
        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

    def transform1(self, images):
        k = 1
        trans_image = torch.rot90(images, k, (2, 3))
        return trans_image

    def update_fc_avg(self, data, label, class_list):
        new_fc = []
        for class_index in class_list:
            data_index = (label == class_index * 2).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            # proto = proto * self.args.alpha + text_feature[class_index] * (1 - self.args.alpha)
            new_fc.append(proto)
            self.fc.weight.data[class_index * 2] = proto

            data_index = (label == class_index * 2 + 1).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            # proto = proto * self.args.alpha + text_feature[class_index] * (1 - self.args.alpha)
            new_fc.append(proto)
            self.fc.weight.data[class_index * 2 + 1] = proto
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def get_logits(self, x, fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x, fc)
        elif 'cos' in self.args.new_mode:
            logits = self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
            # x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(x, p=2, dim=-1))
            # x1 = x1/0.07
            return logits  # ,x1

