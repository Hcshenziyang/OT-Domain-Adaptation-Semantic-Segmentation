import torch
import numpy as np
import ot
import torch.nn.functional as F
import torch.nn as nn
import einops as e


def image_ot(args, imageS, imageT):
    # 图像空间最优传输简单的线性映射
    with torch.no_grad():
        imageS_temp = imageS.squeeze().numpy()
        imageT_temp = imageT.squeeze().numpy()
        imageS_temp = imageS_temp.reshape((imageS_temp.shape[0], imageS_temp.shape[1] * imageS_temp.shape[2]))
        imageS_temp = np.transpose(imageS_temp, (1, 0))
        imageT_temp = imageT_temp.reshape((imageT_temp.shape[0], imageT_temp.shape[1] * imageT_temp.shape[2]))
        imageT_temp = np.transpose(imageT_temp, (1, 0))

        mapping = ot.da.LinearTransport()
        mapping.fit(Xs=imageS_temp, Xt=imageT_temp)
        imageS_temp = mapping.transform(Xs=imageS_temp)
        #
        imageS_temp = np.transpose(imageS_temp, (1, 0))
        imageS_temp = np.clip(imageS_temp, 0, 255)
        imageS_temp = imageS_temp.reshape((3, args.cut_size, args.cut_size))

        imageS = torch.from_numpy(imageS_temp)

    return imageS


def image_ot_emd(args, imageS, imageT):
    # 图像空间最优传输EMD映射
    with torch.no_grad():
        imageS_temp = imageS.squeeze().numpy()
        imageT_temp = imageT.squeeze().numpy()
        imageS_temp = imageS_temp.reshape((imageS_temp.shape[0], imageS_temp.shape[1] * imageS_temp.shape[2]))
        imageS_temp = np.transpose(imageS_temp, (1, 0))
        imageT_temp = imageT_temp.reshape((imageT_temp.shape[0], imageT_temp.shape[1] * imageT_temp.shape[2]))
        imageT_temp = np.transpose(imageT_temp, (1, 0))
        rng = np.random.RandomState()
        # training samples
        nb = 500
        idx1 = rng.randint(imageS_temp.shape[0], size=(nb,))
        idx2 = rng.randint(imageT_temp.shape[0], size=(nb,))

        Xs = imageS_temp[idx1, :]
        Xt = imageT_temp[idx2, :]

        # EMDTransport
        ot_emd = ot.da.EMDTransport()
        ot_emd.fit(Xs=Xs, Xt=Xt)
        imageS_temp = ot_emd.transform(Xs=imageS_temp)
        transp_Xs_emd = np.transpose(imageS_temp, (1, 0))
        imageS_temp = np.clip(transp_Xs_emd, -127, 127)
        imageS_temp = imageS_temp.reshape((3, args.cut_size, args.cut_size))

        imageS = torch.from_numpy(imageS_temp)

    return imageS


def gamma(args, interp2, S, T):
    # DeepJDOT最优传输
    # 寻找最相近的两个样本，计算其距离，当作损失
    with torch.no_grad():
        gs = interp2(S)
        gt = interp2(T)
        gs = gs.detach().view(S.shape[1], args.ot_size * args.ot_size).transpose(0, 1).cpu()
        gt = gt.detach().view(S.shape[1], args.ot_size * args.ot_size).transpose(0, 1).cpu()

        C0 = torch.cdist(gs, gt, p=2.0) ** 2
        gamma1 = ot.emd(ot.unif(gs.shape[0]),
                        ot.unif(gt.shape[0]),
                        C0.squeeze().numpy())
        gamma1 = torch.tensor(gamma1)

        # gamma1 = ot.sinkhorn(
        #     ot.unif(gs.shape[0]),
        #     ot.unif(gt.shape[0]),
        #     C0.squeeze().numpy(),
        #     reg = 1600
        # )
        gamma1 = torch.tensor(gamma1)
    return gamma1


def gamma1(args, interp2, S, T, ps, pt):
    # 特征空间矩阵计算追加输出空间距离
    with torch.no_grad():
        gs = interp2(S)
        gt = interp2(T)
        ps = interp2(ps)
        pt = interp2(pt)

        gs = gs.detach().view(S.shape[1], args.ot_size * args.ot_size).transpose(0, 1).cpu()
        gt = gt.detach().view(S.shape[1], args.ot_size * args.ot_size).transpose(0, 1).cpu()
        ps = ps.detach().view(ps.shape[1], args.ot_size * args.ot_size).transpose(0, 1).cpu()
        pt = pt.detach().view(ps.shape[1], args.ot_size * args.ot_size).transpose(0, 1).cpu()

        C0 = torch.cdist(gs, gt, p=2.0) ** 2
        C1 = torch.cdist(ps, pt, p=2.0) ** 2
        C = C0+C1
        C = C / C.max()
        gamma1 = ot.sinkhorn(
            ot.unif(gs.shape[0]),
            ot.unif(gt.shape[0]),
            C.squeeze().numpy(),
            reg = 0.005
        )

        # gamma1 = ot.emd(ot.unif(gs.shape[0]),
        #                 ot.unif(gt.shape[0]),
        #                 C.squeeze().numpy(),
        #                 )
        gamma1 = torch.tensor(gamma1)
    return gamma1

def gamma_ssl(args, T, T_att, pt, labelT):
    '''
    伪标签gamma矩阵计算
    1、提取非255样本
    2、
    '''
    with torch.no_grad():
        gt = T
        pt = pt
        gt2 = T_att
        gt2 = gt2.detach().view(T_att.shape[1], args.ot_size * args.ot_size).transpose(0, 1).cpu()
        gt = gt.detach().view(T.shape[1], args.ot_size * args.ot_size).transpose(0, 1).cpu()
        pt = pt.detach().view(pt.shape[1], args.ot_size * args.ot_size).transpose(0, 1).cpu()
        labelT = labelT.detach().view(labelT.shape[1], args.ot_size * args.ot_size).transpose(0, 1).cpu()
        m = torch.where(labelT != 255, True, False).squeeze()
        n = torch.nonzero(m).squeeze()
        gt = torch.index_select(gt, 0, n)
        gt2 = torch.index_select(gt2, 0, n)
        pt = torch.index_select(pt, 0, n)
        labelT = torch.index_select(labelT, 0, n)
        # 交叉熵计算
        one_hot = F.one_hot(labelT.transpose(1,0).squeeze().long(),num_classes=6).float()

        C1 = torch.cdist(pt, one_hot, p=2.0) ** 2
        C0 = torch.cdist(gt, gt2, p=2.0) ** 2


        C = C0 + C1
        # gamma1 = ot.sinkhorn(ot.unif(gt.shape[0]),
        #                      ot.unif(gt.shape[0]),
        #                      C.squeeze().numpy(),
        #                      reg = 1)
        gamma1 = ot.emd(ot.unif(gt.shape[0]),
                             ot.unif(gt.shape[0]),
                             C.squeeze().numpy())
        gamma1 = torch.tensor(gamma1)
    return gamma1


def gamma_ssl2(feaT, predT, labelT):
    '''
    适用于注意力机制伪标签训练。
    提取有伪标签特征和无伪标签特征进行最优传输，请注意，输入的参数需要提前下采样。
    '''
    with torch.no_grad():
        T = e.rearrange(feaT, 'b c h w -> b (h w) c').squeeze()
        pt = e.rearrange(predT, 'b c h w -> b (h w) c').squeeze()
        labelT  = e.rearrange(labelT.squeeze(1), 'b h w -> b (h w)').squeeze()
        mask_255 = torch.where(labelT == 255, True, False)
        mask_other = torch.where(labelT != 255, True, False)
        mask_255 = torch.nonzero(mask_255).squeeze()
        mask_other = torch.nonzero(mask_other).squeeze()

        T_255 = torch.index_select(T, 0, mask_255.cuda())
        pt_255 = torch.index_select(pt, 0, mask_255.cuda())
        T_other = torch.index_select(T, 0, mask_other.cuda())
        pt_other = torch.index_select(pt, 0, mask_other.cuda())

        if T_255.shape[0] < 10:
            gamma1 = np.zeros((1600, 1600))
        else:
            C1 = torch.cdist(T_255, T_other, p=2.0) ** 2
            C0 = torch.cdist(pt_255, pt_other, p=2.0) ** 2
            C = C0 + C1

            gamma1 = ot.emd(ot.unif(T_255.shape[0]),
                                 ot.unif(T_other.shape[0]),
                                 C.cpu().squeeze().numpy())
        gamma1 = torch.tensor(gamma1)
    return gamma1


