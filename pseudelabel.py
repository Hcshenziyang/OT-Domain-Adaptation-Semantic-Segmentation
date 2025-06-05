# 伪标签生成主程序
import argparse
import torch
from modelOT import DeeplabMultiFeature
import torch.backends.cudnn as cudnn
from test_tool import pseude_label_creat
import os

def main(args):

    # ##############################################程序设置#############################################
    cudnn.benchmark = True
    cudnn.enabled = True

    # ##############################################网络设置#############################################
    model = DeeplabMultiFeature(num_classes=args.num_classes)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda()
    mIoU = pseude_label_creat(args, model, overlapping=256)


if __name__ == '__main__':
    # #######################  名称  #################################
    DATA_NAME = "ISPRS"
    MOEDL_NAME = "OTALL"
    LOG_NAME = DATA_NAME + MOEDL_NAME
    # #######################   图像路径  #################################
    DATA_TEST = r"F:\data\remotedata\ISPRS\vaihingen_IRRG\train"  # 测试数据
    # #######################   图像设置   #################################
    IMG_MEAN = (128.0, 128.0, 128.0)
    NUM_CLASSES = 6  # 类别数
    CUT_SIZE = 1000  # 裁切尺寸
    MAPPING = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]  # 标签映射
    # #######################  权重设置   #################################
    RESTORE_FROM = r'D:\result\en_paper\Comparisons\cross_domain_seg\PV\Base\PV.pth'
    # RESTORE_FROM = r'D:\code\OTcode2_1\train2\save_ssl\save_pot\alpha0.015000.pth'
    TEST_SAVE = r'D:\code\OTcode3\train\result_pseudo'  # 保存标签路径

    parser = argparse.ArgumentParser(description="ADDA gta city")
    parser.add_argument("--img_mean", type=float, default=IMG_MEAN)
    parser.add_argument("--data_name", type=str, default=DATA_NAME)
    parser.add_argument("--mapping", type=int, default=MAPPING)
    parser.add_argument("--data_test", type=str, default=DATA_TEST)
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--cut_size", type=int, default=CUT_SIZE)
    parser.add_argument("--test_save", type=str, default=TEST_SAVE)
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM)
    args = parser.parse_args()

    # 创建文件夹
    if not os.path.exists(args.test_save):
        os.makedirs(args.test_save)

    main(args)