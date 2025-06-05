from dataset.dataloader import test_dataloader
import torch
import torch.nn as nn
from compute_mIoU import compute_mIoU
import PIL.Image as Image
import numpy as np
import cv2
import os
from pseudo_label import pseudo_label


def num(i, j, size, ol):
    if i < size or j < size:
        print('错误！图像太小了，无法满足裁切尺寸需求！')
    i_stop = int((i - size) / (size - ol)) + 2
    j_stop = int((j - size) / (size - ol)) + 2
    return i_stop, j_stop


def patch_test_model(args, model, logger=None, overlapping=0, com_miou = True):
    # 滑块裁切
    testloader = test_dataloader(args, args.data_test)
    model.eval()
    test_interp = nn.Upsample(size=(args.cut_size, args.cut_size), mode='bilinear', align_corners=True)


    for index, batch in enumerate(testloader):
        image, name = batch
        output_all = np.ones((image.size(2), image.size(3)))*255
        # F1_all = np.ones((image.size(2), image.size(3)))*255

        # 判断图像行和列各可以裁切多少次
        i_stop, j_stop = num(image.size(2), image.size(3), args.cut_size, overlapping)

        for i in range(0, i_stop):
            if i == i_stop-1:  # 最后一次
                h_s = image.size(2)-args.cut_size
                h_e = image.size(2)
            else:
                h_s = i * (args.cut_size-overlapping)
                h_e = h_s + args.cut_size
            if i != 0:
                out_h = h_s + int(overlapping / 2)
                h = int(overlapping / 2)
            else:
                out_h = 0
                h = 0
            for j in range(0, j_stop):
                if j == j_stop - 1:  # 最后一次
                    w_s = image.size(3) - args.cut_size
                    w_e = image.size(3)
                else:
                    w_s = j * (args.cut_size-overlapping)
                    w_e = w_s + args.cut_size
                image_temp = image[:, :, h_s:h_e, w_s:w_e]
                # 模型预测
                with torch.no_grad():
                    F1, output = model(image_temp.cuda())

                output = test_interp(output).cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
                if j != 0:
                    out_w = w_s + int(overlapping/2)
                    w = int(overlapping/2)
                else:
                    out_w = 0
                    w = 0
                output_all[out_h:h_e, out_w:w_e] = output[h:, w:]



        name = name[0].split('/')[-1]
        # print(output_all.max())
        # print(output_all.astype(np.uint8).max())

        # cv2.imwrite(os.path.join(args.test_save, name[:-3]+'png'), output_all.astype(np.uint8))  # 记住记住记住记住……opencv保存jpg格式会压缩数据！！！！操TM的傻逼格式
        # pred = np.asarray(cv2.imread(os.path.join(args.test_save, name[:-3]+'png'), cv2.IMREAD_GRAYSCALE))
        # print(pred.max())
        # F1_all[F1_all>50]=50
        cv2.imwrite(os.path.join(args.test_save, name), output_all.astype(np.uint8))  # 记住记住记住记住……opencv保存jpg格式会压缩数据！！！！操TM的傻逼格式
        # cv2.imwrite(os.path.join(args.test_save, 'feature'+name), F1_all.astype(np.float))
        # pred = np.asarray(cv2.imread(os.path.join(args.test_save, name[:-3]), cv2.IMREAD_GRAYSCALE))
    if args.data_name == 'ISPRS':
        label_name = 'labels'
    elif args.data_name == 'LOVEDA':
        label_name = 'masks_png'
    if com_miou:
        mIoUs,OA,KAPPA = compute_mIoU(args.data_test +f'/{label_name}',
                             args.test_save,
                             num_classes=args.num_classes,
                             logger=logger,
                             mapping = args.mapping
                             )
        mIoU = round(np.nanmean(mIoUs) * 100, 2)
    else:
        mIoU = 0

    return mIoU


def pseude_label_creat(args, model, overlapping=0):
    # 滑块裁切
    testloader = test_dataloader(args, args.data_test)
    model.eval()
    test_interp = nn.Upsample(size=(args.cut_size, args.cut_size), mode='bilinear', align_corners=True)


    for index, batch in enumerate(testloader):
        image, name = batch
        output_all = np.ones((image.size(2), image.size(3)))*255

        # 判断图像行和列各可以裁切多少次
        i_stop, j_stop = num(image.size(2), image.size(3), args.cut_size, overlapping)

        for i in range(0, i_stop):
            if i == i_stop-1:  # 最后一次
                h_s = image.size(2)-args.cut_size
                h_e = image.size(2)
            else:
                h_s = i * (args.cut_size-overlapping)
                h_e = h_s + args.cut_size
            if i != 0:
                out_h = h_s + int(overlapping / 2)
                h = int(overlapping / 2)
            else:
                out_h = 0
                h = 0
            for j in range(0, j_stop):
                if j == j_stop - 1:  # 最后一次
                    w_s = image.size(3) - args.cut_size
                    w_e = image.size(3)
                else:
                    w_s = j * (args.cut_size-overlapping)
                    w_e = w_s + args.cut_size
                image_temp = image[:, :, h_s:h_e, w_s:w_e]
                # 模型预测
                with torch.no_grad():
                    _, output = model(image_temp.cuda())
                label = pseudo_label(test_interp, output)
                if j != 0:
                    out_w = w_s + int(overlapping/2)
                    w = int(overlapping/2)
                else:
                    out_w = 0
                    w = 0
                output_all[out_h:h_e, out_w:w_e] = label[h:, w:]

        name = name[0].split('/')[-1]
        cv2.imwrite(os.path.join(args.test_save, name), output_all.astype(np.uint8))

    return 1