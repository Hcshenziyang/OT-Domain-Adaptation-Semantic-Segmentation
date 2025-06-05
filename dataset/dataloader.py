import argparse
import torch
from torch.utils import model_zoo
import torch.optim as optim
from apex import amp
import torch.backends.cudnn as cudnn
import torch.nn as nn
import os.path as osp
import os
from modelOT import DeeplabMultiFeature
from dataset.dataloader import train_dataloader
from test_tool import patch_test_model
import logging
import time
from ot_tool import image_ot_emd,gamma,gamma1
from tools import distance
import torch.nn.functional as F
import numpy as np

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    # 学习率衰减
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def amp_backward(loss, optimizer, retain_graph=False):
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward(retain_graph=retain_graph)


def main(args):

    # ############################################程序设置###########################################################
    cudnn.benchmark = True
    cudnn.enabled = True
    device = torch.device("cuda")
    Iter = args.start_iter

    # ############################################数据载入###########################################################
    trainloaderS = train_dataloader(args, args.data_train_S)
    trainloaderT = train_dataloader(args, args.data_train_T)

    # ############################################模型设置###########################################################
    model = DeeplabMultiFeature(num_classes=args.num_classes)
    new_params = model.state_dict().copy()

    if args.restore_from[:4] == 'http':
        saved_state_dict = model_zoo.load_url(args.restore_from)
        for i in saved_state_dict:
            i_parts = i.split('.')
            if not args.num_classes == 6 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params, strict=False)
    else:
        saved_state_dict = torch.load(args.restore_from)
        model.load_state_dict(saved_state_dict, strict=False)

    if args.continue_train:
        model.load_state_dict(saved_state_dict)
        Iter = args.continue_train_iter

    model.train()
    model.cuda()
    # ##########################################优化器设置################################################
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    optimizer.zero_grad()
    #
    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O2",
        keep_batchnorm_fp32=True, loss_scale="dynamic"
    )

    # #####################################部分训练设置#########################################
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    interp = nn.Upsample(size=(args.cut_size, args.cut_size), mode='bilinear', align_corners=True)
    interp2 = nn.Upsample(size=(args.ot_size, args.ot_size), mode='bilinear', align_corners=True)  # 最优传输下采样
    interp3 = nn.Upsample(size=(args.ot_size, args.ot_size), mode='nearest')

    loss_seg_value = 0
    loss_feature_value = 0
    loss_output_value = 0
    loss_seg2_value = 0

    for i_iter in range(Iter, args.num_steps):
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)  # 学习率衰减

        _, batch = trainloaderS.__next__()
        imageS, labelS, _ = batch
        _, batch = trainloaderT.__next__()
        imageT, _, _ = batch

        imageS = image_ot_emd(args, imageS, imageT.detach()).unsqueeze(0)  # 图像空间最优传输

        labelS = labelS.long().to(device)
        imageS = imageS.type(torch.FloatTensor)

        feaS, predS = model(imageS.cuda())
        feaT, predT = model(imageT.cuda())
        loss_seg = seg_loss(interp(predS), labelS)

        # # ########################################## 计算目标域标签 ###############################################
        with torch.no_grad():
            output = F.softmax(predT.detach(), dim=1)
            output = interp(output).cpu().data[0].numpy()
            output = output.transpose(1, 2, 0)

            labelT, prob = np.argmax(output, axis=2), np.max(output, axis=2)  # 返回索引，返回数值
            predicted_label = labelT.copy()  # 预测标签
            missing_categories1 = [i for i in range(6) if i not in np.unique(predicted_label)]
            missing_categories2 = [i for i in range(6) if i not in np.unique(labelS.cpu().unsqueeze(0).numpy())]


        gam = gamma1(args, interp2, feaS.detach(), feaT.detach(), predS.detach(), predT.detach()).to(device)
        dist1 = distance(args, interp2, feaS, feaT)
        if len(missing_categories1) != 0 or len(missing_categories2) != 0:
            print('hello')
            labelS = interp3(labelS.unsqueeze(0).float()).detach().squeeze().cpu().numpy().reshape(-1)
            predicted_label = interp3(torch.from_numpy(predicted_label).float().unsqueeze(0).unsqueeze(1)).detach().squeeze().cpu().numpy().reshape(-1)
            unique_values = set(missing_categories1).symmetric_difference(set(missing_categories2))
            for i in unique_values:
                indexes_of_value1 = np.where(labelS == i)
                indexes_of_value2 = np.where(predicted_label == i)
                for index in indexes_of_value1:
                    gam[index, :] = 0
                for index in indexes_of_value2:
                    gam[:, index] = 0
            # gam = (gam-gam.min())/(gam.max()-gam.min())
            align1_loss = torch.sum(gam * dist1)  # 特征空间对齐
        else:
            # gam = (gam - gam.min()) / (gam.max() - gam.min())
            align1_loss = torch.sum(gam * dist1)  # 特征空间对齐

        # 计算传输正确率
        gam2 = gamma(args, interp2, predS.detach(), predT.detach()).to(device)
        dist2 = distance(args, interp2, predS, predT)
        if len(missing_categories1) != 0 or len(missing_categories2) != 0:
            for i in unique_values:
                indexes_of_value1 = np.where(labelS == i)
                indexes_of_value2 = np.where(predicted_label == i)
                for index in indexes_of_value1:
                    gam2[index, :] = 0
                for index in indexes_of_value2:
                    gam2[:, index] = 0
            # gam2 = (gam2 - gam2.min()) / (gam2.max() - gam2.min())
            align2_loss = torch.sum(gam2 * dist1)  # 特征空间对齐

        else:
            # gam2 = (gam2 - gam2.min()) / (gam2.max() - gam2.min())
            align2_loss = torch.sum(gam2 * dist2)  # 标签空间对齐


        loss = loss_seg + args.alpha1 * align1_loss + args.alpha2 * align2_loss # + args.alpha3 * loss_seg2

        amp_backward(loss, optimizer)
        loss_seg_value += loss_seg.item()
        loss_feature_value += args.alpha1 * align1_loss.item()
        loss_output_value += args.alpha2 * align2_loss.item()
        # loss_seg2_value += args.alpha3 * loss_seg2.item()
        optimizer.step()

        # ########################################################结果展示与保存########################################################

        text = f'第{i_iter}次训练的损失值为：'+'语义损失:{:.4f};特征OT:{:.4f},{:.4f};'.\
                  format(loss_seg.item(), args.alpha1 * align1_loss.item(), args.alpha2 * align2_loss.item())
        print(text)
        if i_iter % args.loss_show_steps == 0:
            text = f'第{i_iter}次训练的损失值为：' + '语义损失:{:.4f};特征OT:{:.4f},{:.4f};语义损失:{:.4f};'. \
                format(loss_seg_value/args.loss_show_steps, loss_feature_value/args.loss_show_steps, loss_output_value/args.loss_show_steps, loss_seg2_value/args.loss_show_steps)
            logger.info(text)
            loss_seg_value = 0
            loss_feature_value = 0
            loss_output_value = 0
            loss_seg2_value = 0

        if i_iter % args.show_steps == 0:
            print(f'第{i_iter}次训练的模型存储中……')
            torch.save(model.state_dict(),
                       osp.join(args.save_dir, f'{args.save_pot_name}{i_iter}.pth'))

        # 在目标域上最好的语义分割模型的保存
        if i_iter % args.show_steps == 0:
            mIoU = patch_test_model(args, model, logger)
            logger.info(mIoU)
            model.train()


if __name__ == '__main__':
    # #######################  名称  #################################
    DATA_NAME = "ISPRS"  # TODO 数据集名称与后续读取有关
    MOEDL_NAME = "OTALL"
    LOG_NAME = DATA_NAME + MOEDL_NAME
    SAVE_POT_NAME = "two0.1_"  # 模型存储名称
    # #######################   图像路径  #################################
    DATA_TRAIN_S = r"F:\data\remotedata\ISPRS\potsdam_IRRG\train"  # 源域训练
    DATA_TRAIN_T = r"F:\data\remotedata\ISPRS\vaihingen_IRRG\train"  # 目标域训练
    DATA_TEST = r"F:\data\remotedata\ISPRS\vaihingen_IRRG\val"  # 测试
    # ###################   超参数设置    ###################################
    LEARNING_RATE = 2.5e-4  # 学习率
    ALPHA1 = 0.01  # 超参数1
    ALPHA2 = 0.001  # 超参数2
    ALPHA3 = 0.1  # 超参数2
    OT_SIZE = 40  # 最优传输下采样
    BATCH_SIZE = 1  # batch_size
    NUM_WORKERS = 4
    MOMENTUM = 0.9
    POWER = 0.9
    WEIGHT_DECAY = 0.0005
    # #######################   图像设置   #################################
    # IMG_MEAN = (123.675, 116.28, 103.53)  # LOVEDA
    IMG_MEAN = (128.0, 128.0, 128.0)  # LOVEDA
    NUM_CLASSES = 6  # 类别数
    CUT_SIZE = 1000  # 裁切尺寸
    # MAPPING = [[0, 255], [1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6]]  # 标签映射
    MAPPING = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]  # 标签映射
    # ##########################   训练设置   ################################
    NUM_STEPS = 50000  # 总训练次数
    START_ITER = 1  # 开始训练次数（0为了便于测试，1正常训练）
    SHOW_STEPS = 500  # 训练多少次后进行验证
    LOSS_SHOW_STEPS = 100  # 损失多少次进行展示
    # 初始训练或者继续训练权重载入
    # RESTORE_FROM = r'D:\code\OTcode3\train\save\save_pot\two0.1_22500.pth'
    RESTORE_FROM = r'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
    CONTINUE_TRAIN = False  # 是否继续训练
    CONTINUE_TRAIN_ITER = 22501  # 继续训练从多少次开始
    # ##########################  保存设置  f################################
    TEST_SAVE = './save/result'  # 保存测试标签路径
    SAVE_DIR = './save/save_pot'  # 权重保存路径
    LOG_DIR = './save/log'  # 日志路径
    # ######################################################################################################################
    parser = argparse.ArgumentParser(description="ADDA gta city")
    parser.add_argument("--data_name", type=str, default=DATA_NAME)
    parser.add_argument("--save_pot_name", type=str, default=SAVE_POT_NAME)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--alpha1", type=float, default=ALPHA1)
    parser.add_argument("--alpha2", type=float, default=ALPHA2)
    parser.add_argument("--alpha3", type=float, default=ALPHA3)
    parser.add_argument("--img_mean", type=float, default=IMG_MEAN)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--power", type=float, default=POWER)
    parser.add_argument("--mapping", type=int, default=MAPPING)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--data_train_S", type=str, default=DATA_TRAIN_S)
    parser.add_argument("--data_train_T", type=str, default=DATA_TRAIN_T)
    parser.add_argument("--data_test", type=str, default=DATA_TEST)
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--cut_size", type=int, default=CUT_SIZE)
    parser.add_argument("--ot_size", type=int, default=OT_SIZE)
    parser.add_argument("--start_iter", type=int, default=START_ITER)
    parser.add_argument("--test_save", type=str, default=TEST_SAVE)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR)
    parser.add_argument("--log_dir", type=str, default=LOG_DIR)
    parser.add_argument("--continue_train", type=bool, default=CONTINUE_TRAIN)
    parser.add_argument("--continue_train_iter", type=int, default=CONTINUE_TRAIN_ITER)
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--show_steps", type=int, default=SHOW_STEPS)
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM)
    parser.add_argument("--loss_show_steps", type=int, default=LOSS_SHOW_STEPS)
    args = parser.parse_args()
    argsDict = args.__dict__
    # 创建文件夹
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.test_save):
        os.makedirs(args.test_save)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


    def get_console_file_logger(name, level=logging.INFO, logdir='./baseline'):
        logger = logging.Logger(name)
        logger.setLevel(level=level)
        logger.handlers = []
        BASIC_FORMAT = "%(asctime)s, %(levelname)s:%(name)s:%(message)s"
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel(level=level)

        fhlr = logging.FileHandler(os.path.join(logdir, str(time.time()) + '.log'))
        fhlr.setFormatter(formatter)
        logger.addHandler(chlr)
        logger.addHandler(fhlr)

        return logger
    logger = get_console_file_logger(name=LOG_NAME, logdir=args.log_dir)
    for eachArg, value in argsDict.items():
        logger.info(eachArg + ' : ' + str(value))
    logger.info('\n##################### start #######################\n')
    main(args)



