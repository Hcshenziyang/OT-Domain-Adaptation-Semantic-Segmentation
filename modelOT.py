import torch.nn as nn
import numpy as np  # NumPy 包含大量的各种数学运算的函数，包括三角函数，算术运算的函数，复数处理函数等。
import torch


affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    # nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
    # 此处对应in_planes, out_planes输入输出通道数，kernel_size=3卷积核大小为3，stride步长，padding零填充，bias是否将一个学习到的bias增加到输出中


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False  # 参数固定，也就是这些层中不会发生梯度更新了。作用是减小计算量，保证底层网络稳定。
        padding = dilation

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()  # 可以理解为神经网络的列表，后续可以像列表一样追加网络层
        for dilation, padding in zip(dilation_series, padding_series):
            # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
            # [6, 12, 18, 24], [6, 12, 18, 24]
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))
            # 输入通道数，输出通道数为类别数量，也就是分成num_classes个类别，卷积核3*3，步长1，空洞和padding按照上述设置填入即可
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)  # 权重初始化

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)  # 依次进行卷积，将结果相加。相当于多个比例捕获图像上下文。
        return out


class ResNetMultiFeature(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNetMultiFeature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)  # 输入通道数3，输出64，卷积核7，步长2，空白填充3
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)  # 归一化处理
        for i in self.bn1.parameters():  # 更改了required_grad参数：说明当前量是否需要在计算中保留对应的梯度信息，此处改为不保存
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)  # 激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # 最大池化
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        # 第一个列表对应的6，12，18，24就是空洞卷积的参数，第二个列表对应的就是周边填充数量。
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        # self.layer7 = nn.Conv2d(2048, 2048, kernel_size=3, stride=3, bias=False)
        # self.layer8 = nn.Conv2d(6, 6, kernel_size=3, stride=3, bias=False)
        # 5、6层分别针对输入通道数为1024和2048两种情况



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            # 步长不等于1  或者  输入通道64不等于通道数*4  或者  空洞卷积等于2或者4
            # 换句话说，对2，3，4层进行下采样
            downsample = nn.Sequential(
                # 起到封装的作用，进一步简化代码
                # 下采样卷积核1*1，输出通道数为planes*4，步长为1，不添加偏置参数。
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False  # 不保存梯度信息
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        # 每个layers第一层需要单独处理，因为如图所示，上一层输出通道都是下一层输入通道2倍，需要统一一下，然后继续运算
        self.inplanes = planes * block.expansion  # 更新通道数64-256，256-512，512-1024，1024-2048
        for i in range(1, blocks):  # 第一层上方单独给出，进行下采样，后续按3，4，23，3这个数字，构建网络
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)


        x_feature = self.layer4(x)
        x_pred = self.layer6(x_feature)


        return x_feature, x_pred
        # 返回了特征，分类的类别

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        这个生成器返回网络的所有参数，除了最后一个分类层。 请注意，对于每个batchnorm层，
        在deeplab_resnet.py中requires_grad设置为False，因此该函数不返回任何batchnorm参数
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        该生成器返回网络最后一层的所有参数，将像素分类为类
        """
        b = []
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


def DeeplabMultiFeature(num_classes=6):
    model = ResNetMultiFeature(Bottleneck, [3, 4, 23, 3], num_classes)
    return model
