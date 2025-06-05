# 伪标签预测
import numpy as np
import torch.nn.functional as F

def pseudo_label(interp, predT):

    output = F.softmax(predT.detach(), dim=1)
    output = interp(output).cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)

    labelT, prob = np.argmax(output, axis=2), np.max(output, axis=2)  # 返回索引，返回数值
    predicted_label = labelT.copy()  # 预测标签
    predicted_prob = prob.copy()  # 预测概率

    thres = []  # 阈值
    for i in range(6):
        x = predicted_prob[predicted_label == i]
        if len(x) == 0:
            thres.append(0)
            continue
        x = np.sort(x)
        thres.append(x[int(np.round(len(x) * 0.5))])  # 取中间值
    thres = np.array(thres)
    thres[thres > 0.95] = 0.95
    for i in range(6):
        labelT[(prob < thres[i]) * (labelT == i)] = 255  # 提取高置信度像素赋予伪标签
    return labelT



def pseudo_label2(interp, predT, args, cls_thresh):
    # 参考论文IAST
    if cls_thresh.any() == 0:
        cls_thresh = np.ones(args.num_classes) * 0.9  # 设置初始阈值[0.9 0.9 0.9 0.9 0.9 0.9]

    output = F.softmax(predT.detach(), dim=1)
    output = interp(output).cpu().data[0].numpy()
    output = output.transpose(1, 2, 0)

    labelT, prob = np.argmax(output, axis=2), np.max(output, axis=2)  # 返回索引，返回数值
    predicted_label = labelT.copy()  # 预测标签1000,1000
    predicted_prob = prob.copy()  # 预测概率1000,1000
 # [1 1 1 1 1 1]
    logits_cls_dict = {c: [cls_thresh[c]] for c in range(args.num_classes)}  # 字典，读取每个类别阈值
    for cls in range(args.num_classes):  # 将每一个类别的所有概率放入字典
        logits_cls_dict[cls].extend(predicted_prob[predicted_label == cls].astype(np.float16))
    # 对每个类别提取百分比
    thresh = np.ones(args.num_classes, dtype=np.float32)
    for idx_cls in range(args.num_classes):
        if logits_cls_dict[idx_cls] != None:  # 防止没有这个类别
            arr = np.array(logits_cls_dict[idx_cls])  # 提取某一个类别，下一个公式就是提取这个类别百分比
            thresh[idx_cls] = np.percentile(arr, 100 * (1 - 0.4 * cls_thresh[idx_cls] ** 1.0))  # 提取百分位数,1,2,19,提取50就是2。
    cls_thresh = 0.9 * cls_thresh + (1 - 0.9) * thresh  # 指数移动平均线阈值
    cls_thresh[cls_thresh >= 1] = 0.999  # 限制范围，防止置信度太大，至此，我们的阈值已经设置完毕

    for i in range(6):
        labelT[(prob < cls_thresh[i]) * (labelT == i)] = 255  # 提取高置信度像素赋予伪标签
    return labelT, cls_thresh  # 为了保存全局阈值
