import argparse
import torch
import os
from modelOT import DeeplabMultiFeature
# from modelOT_ssl_att import DeeplabMultiFeature
from test_tool import patch_test_model
import logging
import time



def main(args):

    model = DeeplabMultiFeature(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict, strict=False)

    model.eval()
    model.cuda()
    mIoU = patch_test_model(args, model, com_miou = args.com_miou)


if __name__ == '__main__':
    # #######################  名称  #################################
    DATA_NAME = "ISPRS"
    MOEDL_NAME = "OTALL"
    LOG_NAME = DATA_NAME + MOEDL_NAME
    # #######################   图像路径  #################################
    DATA_TEST = r"F:\data\remotedata\ISPRS\vaihingen_IRRG\val"  # 测试
    # #######################   图像设置   #################################
    NUM_CLASSES = 6  # 类别数
    CUT_SIZE = 1000  # 裁切尺寸
    MAPPING = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [255, 255]]  # 标签映射
    IMG_MEAN = (128.0, 128.0, 128.0)
    RESTORE_FROM = r'D:\code\OTcode3\train\save_ssl\save_pot\11_2419000.pth'

    # ##########################  保存设置  f################################
    TEST_SAVE = './result'  # 保存测试标签路径
    COM_MIOU = True

    parser = argparse.ArgumentParser(description="ADDA gta city")
    parser.add_argument("--data_name", type=str, default=DATA_NAME)
    parser.add_argument("--data_test", type=str, default=DATA_TEST)
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--cut_size", type=int, default=CUT_SIZE)
    parser.add_argument("--img_mean", type=float, default=IMG_MEAN)
    parser.add_argument("--test_save", type=str, default=TEST_SAVE)
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM)
    parser.add_argument("--mapping", type=int, default=MAPPING)
    parser.add_argument("--com_miou", type=bool, default=COM_MIOU)
    args = parser.parse_args()
    argsDict = args.__dict__
    if not os.path.exists(args.test_save):
        os.makedirs(args.test_save)

    main(args)


