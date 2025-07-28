# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as tF
from .geomloss import SamplesLoss

class Cost:
    def __init__(self, factor=128) -> None:
        self.factor = factor
        self.box_scale = torch.tensor([2.0, 2.0])  # 用tensor

    def __call__(self, x, y):
        # 假設 x,y shape 都是 (B, N, 2)
        x_col = x.unsqueeze(-2)  # (B, N, 1, 2)
        y_row = y.unsqueeze(-3)  # (B, 1, M, 2)
        
        # 加入 box_scale 縮放，避免距離過大
        scale = self.box_scale.to(x.device)
        diff = (x_col - y_row) / scale  # normalize 距離
        C = torch.sum(diff ** 2, dim=-1)
        return C


per_cost = Cost(factor=128)
eps = 1e-8

class GeneralizedLoss(nn.modules.loss._Loss):
    def __init__(self, factor=1, reduction='mean') -> None:
        super().__init__()
        self.factor = factor
        self.reduction = reduction
        self.tau = 5

        self.cost = per_cost
        self.blur = 0.01
        self.scaling = 0.75
        self.reach = 0.5
        self.p = 1
        self.uot = SamplesLoss(blur=self.blur, scaling=self.scaling, debias=False, backend='tensorized', cost=self.cost, reach=self.reach, p=self.p).to("cpu")
        self.pointLoss = nn.L1Loss(reduction=reduction)
        self.pixelLoss = nn.MSELoss(reduction=reduction)

        self.down = 1

    def forward(self, dens, dots, box_size=None):
        device = dots.device  # 或 seq.device

        bs = dens.size(0)
        point_loss, pixel_loss, emd_loss = 0, 0, 0
        entropy = 0
        for i in range(bs):
            den = dens[i, 0]
            seq = torch.nonzero(dots[i, 0]) # N * 2


            if box_size is not None:
                self.cost.box_scale = torch.sigmoid((box_size[i] - 64) / 32) * 2 + 1
            else:
                self.cost.box_scale = [2, 2]

            if seq.size(0) < 1 or den.sum() < eps:
                point_loss += torch.abs(den).mean()
                pixel_loss += torch.abs(den).mean()
                emd_loss += torch.abs(den).mean()
                print(f"den have nothing")
            else:
                A, A_coord = self.den2coord(den)
                A_coord = A_coord.reshape(1, -1, 2)
                A = A.reshape(1, -1, 1)

                B_coord = seq[None, :, :] # 1 * N * 2
                B = torch.ones(seq.size(0), device=device).float().view(1, -1, 1) * self.factor # 1 * N * 1
                
                A = A / (A.sum() + 1e-8)
                B = B / (B.sum() + 1e-8)

                # print(f"A.sum = {A.sum().item()}")
                # print(f"B.sum = {B.sum().item()}")

                oploss, F, G = self.uot(A, A_coord, B, B_coord)

                print(f"oploss = {oploss}")
                
                C = self.cost(A_coord, B_coord)

                # tmp = (F.view(1, -1, 1) + G.view(1, 1, -1) - C).detach() / (self.blur ** self.p)
                # # print("max exp arg:", tmp.max().item())
                # # print("min exp arg:", tmp.min().item())
                exp_term = ((F.view(1, -1, 1) + G.view(1, 1, -1) - C)) / (self.blur ** self.p)
                exp_term = exp_term.clamp(max=10)  # 防止 torch.exp 爆炸
                print(f"exp_term = {exp_term.max().item()}")
                PI = torch.exp(exp_term) * A * B.view(1, 1, -1)

                
                # PI = torch.exp((F.view(1, -1, 1) + G.view(1, 1, -1) - C).detach() / (self.blur ** self.p)) * A * B.view(1, 1, -1)
                
                PI_clamped = PI.clamp(min=1e-8)  # 避免 log(0)
                entropy += torch.mean(PI_clamped * torch.log(PI_clamped))


                emd_loss += torch.mean(oploss)
                point_loss += self.pointLoss(PI.sum(dim=1).view(1, -1, 1), B)
                pixel_loss += self.pixelLoss(PI.sum(dim=2).view(1, -1, 1), A)

        print(f"emd_loss = {emd_loss}, point_loss={self.tau * point_loss}, pixel_loss={self.tau * pixel_loss}, entropy={self.blur * entropy}")
                
        loss = (emd_loss + self.tau * (point_loss + pixel_loss) + self.blur * entropy) 
        return loss
    
    def den2coord(self, denmap): #只拿出 1，因為 dot map
        assert denmap.dim() == 2, f"denmap.shape = {denmap.shape}, whose dim is not 2"
        coord = torch.nonzero(denmap) # (N, 2) N = H * W, 2 先 Y 再 X 座標
        denval = denmap[coord[:, 0], coord[:, 1]]
        return denval, coord
    
