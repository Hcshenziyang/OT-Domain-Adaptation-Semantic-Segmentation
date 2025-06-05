# 改进后的iou计算
# 更新时间：2022/11/18，创立文件及相关功能


import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import os
import cv2

def fast_hist(a, b, n):
    # 确保a和b在0~n-1的范围内，k是(HxW,)的True和False数列
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  # bincount统计标签出现频率

def per_class_iu(hist):
    """
    Calculate the IoU(Intersection over Union) for each class
    :param hist: np.ndarray with shape (n, n)
    :return: np.ndarray with shape (n,)
    """
    np.seterr(divide="ignore", invalid="ignore")
    res = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    np.seterr(divide="warn", invalid="warn")
    res[np.isnan(res)] = 0.
    return res


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def compute_mIoU(label_dir, pred_dir, num_classes, mapping=None, logger=None):
    # 分别输入真值路径，预测路径,args
    if mapping == None:
        mapping = [[0, 0], [1, 1], [2, 2], [3, 3],[4,4],[5,5]]
    name_classes = np.array(['不透水面', '建筑物', '低矮植被', '树木', '汽车', '背景'], dtype=np.str_)  # 每个类别名称
    # name_classes = np.array(['背景', '建筑物', '道路', '水', '贫瘠', '森林', '农业'], dtype=np.str_)
    hist = np.zeros((num_classes, num_classes))  # 混淆矩阵

    list = os.listdir(label_dir)

    gt_imgs = [join(label_dir, x) for x in list]  # 这三行代码补充了完整的标签真实路径

    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in list]  # 这两行代码补充完完整的预测标签路径


    # 开始计算
    for ind in range(len(gt_imgs)):
        pred = np.asarray(cv2.imread(pred_imgs[ind], cv2.IMREAD_GRAYSCALE))
        label = np.asarray(cv2.imread(gt_imgs[ind], cv2.IMREAD_GRAYSCALE))

        label = label_mapping(label, mapping)

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  # 累加，10张图输出一次
        if ind > 0 and ind % 100 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))

    # miou计算

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        text = '===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2))
        print(text)
        if logger != None:
            logger.info(text)
    text = '===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2))
    print(text)
    if logger != None:
        logger.info(text)

    # OA计算
    oa = np.diagonal(hist).sum()/hist.sum()  # 对角线之和除以所有像素数
    text = '===> OA: ' + str(oa)
    print(text)
    if logger != None:
        logger.info(text)


    # kappa系数计算
    all_num = hist.sum()
    po = oa
    pe = 0
    for i in range(0,hist.shape[0]):
        pe_temp = hist[i,:].sum()*hist[:,i].sum()
        pe += pe_temp  # 行和乘列和，除以所有像素平方
    pe = pe/(all_num*all_num)

    kappa = (po-pe)/(1-pe)
    text = '===> kappa: ' + str(kappa)
    print(text)
    if logger != None:
        logger.info(text)

    return mIoUs, oa, kappa


def main(args):
    compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str, help='directory which stores CityScapes val gt images')
    parser.add_argument('pred_dir', type=str, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')
    args = parser.parse_args()
    main(args)
