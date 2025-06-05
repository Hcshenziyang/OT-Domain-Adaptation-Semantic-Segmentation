
# 各种小工具
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops as e
import numpy as np
def distance(args, interp2, S, T):
    # 欧式距离的计算

    x = interp2(S).view(S.shape[1], args.ot_size * args.ot_size).transpose(0, 1)
    y = interp2(T).view(S.shape[1], args.ot_size * args.ot_size).transpose(0, 1)


    distx = torch.reshape(torch.sum(torch.square(x), 1), (-1, 1))
    disty = torch.reshape(torch.sum(torch.square(y), 1), (1, -1))
    dist = distx + disty
    dist = dist - 2.0 * torch.matmul(x, torch.transpose(y, 0, 1))
    return dist

def distance_ssl2(feaT, predT, labelT):
    """
    暂定。

    :param feaT:
    :param predT:
    :param labelT:
    :return:
    """
    T = e.rearrange(feaT, 'b c h w -> b (h w) c').squeeze()
    labelT = e.rearrange(labelT.squeeze(1), 'b h w -> b (h w)').squeeze()
    mask_255 = torch.where(labelT == 255, True, False)
    mask_other = torch.where(labelT != 255, True, False)
    mask_255 = torch.nonzero(mask_255).squeeze()
    mask_other = torch.nonzero(mask_other).squeeze()
    T_255 = torch.index_select(T, 0, mask_255.cuda())
    T_other = torch.index_select(T, 0, mask_other.cuda())


    if T_255.shape[0] >= 10:
        distx = torch.reshape(torch.sum(torch.square(T_255), 1), (-1, 1))
        disty = torch.reshape(torch.sum(torch.square(T_other), 1), (1, -1))
        dist = distx + disty
        dist = dist - 2.0 * torch.matmul(T_255, torch.transpose(T_other, 0, 1))
    else:
        dist = torch.tensor(np.zeros((1600, 1600)))

    return dist


def distance_ssl1(args, pt, labelT):
    '''
    伪标签gamma矩阵计算
    1、提取非255样本
    2、
    '''

    pt = pt.view(pt.shape[1], args.ot_size * args.ot_size).transpose(0, 1).cpu()
    labelT = labelT.view(labelT.shape[1], args.ot_size * args.ot_size).transpose(0, 1).cpu()
    m = torch.where(labelT != 255, True, False).squeeze()
    n = torch.nonzero(m).squeeze()
    pt = torch.index_select(pt, 0, n)
    labelT = torch.index_select(labelT, 0, n)
    # 交叉熵计算
    one_hot = F.one_hot(labelT.transpose(1,0).squeeze().long(),num_classes=6).float()
    C1 = torch.cdist(pt, one_hot, p=2.0) ** 2
    return C1

def calcor(S, T):
    # 输出空间类别相关性计算
    S = S.view(S.shape[1], S.shape[2] * S.shape[3]).transpose(1, 0)
    T = T.view(T.shape[1], T.shape[2] * T.shape[3])
    cor = torch.matmul(S, T)
    cor = torch.sigmoid(cor)
    return cor
