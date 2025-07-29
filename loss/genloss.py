#@title loss_2.py

import torch
import torch.nn as nn
import torch.nn.functional as tF
from .geomloss import SamplesLoss
import torch.nn.functional as F

# --- 對 B 做 Gaussian blur ---
# kernel_size 建議奇數（如 5），sigma 可調整平滑程度
def gaussian_blur(input, kernel_size=5, sigma=1.0):
    channels = input.shape[1]
    padding = kernel_size // 2
    x = torch.arange(-padding, padding + 1, dtype=torch.float32, device=input.device)
    gauss = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss = gauss / gauss.sum()
    kernel_1d = gauss.view(1, 1, -1)  # (1, 1, K)
    kernel_2d = kernel_1d.T @ kernel_1d  # (K, K)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    return F.conv2d(input, kernel_2d, padding=padding, groups=channels)


def weighted_inverted_huber_loss(pred, target, weights, loss_type, delta=1.0):
    
    

    error = pred - target

    
    abs_error = error.abs()

    # print(f"error_max = {abs_error.max()}, error_min = {abs_error.min()}")


    l1_mask = abs_error <= delta

    l2_mask = ~l1_mask

    l1_loss = abs_error[l1_mask]
    l2_loss = (error[l2_mask] ** 2 + delta ** 2) / (2 * delta)

    # 分開加權
    weighted_l1 = l1_loss.sum() * 0.01
    weighted_l2 = l2_loss.sum() * 1

    loss = weighted_l1 + weighted_l2

    print(f"{loss_type} => weighted_l1 = {weighted_l1} / {l1_loss.numel()}")
    print(f"{loss_type} => weighted_l2 = {weighted_l2} / {l2_loss.numel()}")

    # l1_loss.numel() : 列出這個 tensor 有多少元素

    return loss

class Cost:
    def __init__(self, factor=128) -> None:
        self.factor = factor
        self.normalized_coord = torch.tensor([2.0, 2.0])  # 用tensor

    def __call__(self, x, y):
        # 假設 x,y shape 都是 (B, N, 2)
        x_col = x.unsqueeze(-2)  # (B, N, 1, 2)
        y_row = y.unsqueeze(-3)  # (B, 1, M, 2)

        # 加入 box_scale 縮放，避免距離過大
        scale = self.normalized_coord.to(x.device)
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
        self.blur = 0.1
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
                # print(f"den have nothing")
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

                # print(f"oploss = {oploss}")

                C = self.cost(A_coord, B_coord)

                # tmp = (F.view(1, -1, 1) + G.view(1, 1, -1) - C).detach() / (self.blur ** self.p)
                # # print("max exp arg:", tmp.max().item())
                # # print("min exp arg:", tmp.min().item())
                exp_term = ((F.view(1, -1, 1) + G.view(1, 1, -1) - C)) / (self.blur ** self.p)
                exp_term = exp_term.clamp(max=20)  # 防止 torch.exp 爆炸
                # print(f"exp_term = {exp_term.max().item()}")
                PI = torch.exp(exp_term) * A * B.view(1, 1, -1)


                # PI = torch.exp((F.view(1, -1, 1) + G.view(1, 1, -1) - C).detach() / (self.blur ** self.p)) * A * B.view(1, 1, -1)

                PI_clamped = PI.clamp(min=1e-8)  # 避免 log(0)
                entropy += torch.mean(PI_clamped * torch.log(PI_clamped))


                emd_loss += torch.mean(oploss)

                # --- 對 B 進行 blur ---

                # blur dot map
                dot_map = torch.zeros_like(den)
                dot_map[seq[:, 0], seq[:, 1]] = 1.0
                dot_map = dot_map.unsqueeze(0).unsqueeze(0)

                # blur 後的 map
                dot_map_blurred = gaussian_blur(dot_map, kernel_size=5, sigma=1.0)[0, 0]  # (H, W)

                # 重新找非零點位置（避免 seq 有值但 blur 後為 0）
                blurred_seq = torch.nonzero(dot_map_blurred, as_tuple=False)  # shape (N, 2)

                # 抽出值
                B_blurred = dot_map_blurred[blurred_seq[:, 0], blurred_seq[:, 1]].view(1, -1, 1)

                # 更新 PI & pred/target
                PI_B = torch.exp(exp_term) * A * B_blurred.view(1, 1, -1)
                pred = PI_B.sum(dim=1).view(1, -1, 1)        # shape (1, N, 1)
                target = B_blurred          # shape (1, N, 1)

  



                # 因為要讓 pred 與 target 越近越好，所以這邊用 weighted inverse huber loss

                # point_loss += self.pointLoss(PI.sum(dim=1).view(1, -1, 1), B)
                # pixel_loss += self.pixelLoss(PI.sum(dim=2).view(1, -1, 1), A)

                weights = torch.where(target > 0, 0.0001, 0.000001)

                # 計算加權 loss
                # loss_contrib = weights * torch.abs(pred - target)
                point_loss = point_loss + weighted_inverted_huber_loss(pred, target, weights, "point_loss", delta=0.01)

                pred_a = PI.sum(dim=2).view(1, -1, 1)
                target_a = A.view(1, -1, 1)
                weights_a = torch.where(target > 0, 0.01, 0.0001)  # 或調整你認為合適的值
                pixel_loss = pixel_loss + weighted_inverted_huber_loss(pred_a, target_a, weights_a, "pixel_loss", delta=0.0001)

                # 累加進 scalar point_loss（用 sum 或 mean）
                # point_loss = point_loss + loss_contrib.sum()

        print(f"emd_loss = {emd_loss}, point_loss={self.tau * point_loss}, pixel_loss={self.tau * pixel_loss}, entropy={self.blur * entropy}")
        print("I am gaussian blur + weighted inverted_huber_loss")

        loss = (emd_loss + self.tau * (point_loss + pixel_loss) + self.blur * entropy)


        return loss

    def den2coord(self, denmap): #只拿出 1，因為 dot map
        assert denmap.dim() == 2, f"denmap.shape = {denmap.shape}, whose dim is not 2"
        coord = torch.nonzero(denmap) # (N, 2) N = H * W, 2 先 Y 再 X 座標
        denval = denmap[coord[:, 0], coord[:, 1]]
        return denval, coord

