# mri_model.py

import torch
import torch.nn as nn
from torchvision import models

def get_mri_model(num_classes: int = 2,
                  in_channels: int = 1,
                  pretrained: bool = True,
                  freeze_backbone: bool = False,
                  dropout_p: float = 0.5) -> nn.Module:
    """
    加载并定制一个 ResNet-18，用于单通道 MRI 影像分类。

    Args:
        num_classes (int): 最后输出的类别数，默认为 2（AS vs Healthy）。
        in_channels (int): 输入通道数，MRI 通常为 1。
        pretrained (bool): 是否加载 ImageNet 预训练权重。
        freeze_backbone (bool): 是否冻结骨干网络参数，只训练最后分类头。
        dropout_p (float): 分类头 Dropout 概率，用于正则化。

    Returns:
        torch.nn.Module: 配置好的 ResNet-18 模型实例。
    """
    # 1. 加载 ResNet-18
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    # 2. 替换第一层 conv1 以支持自定义 in_channels
    if in_channels != 3:
        orig = model.conv1
        model.conv1 = nn.Conv2d(in_channels,
                                orig.out_channels,
                                kernel_size=orig.kernel_size,
                                stride=orig.stride,
                                padding=orig.padding,
                                bias=False)
        # 用 Kaiming 正态初始化新 conv1
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')

    # 3. 可选冻结骨干网络
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # 4. 定制分类头：Dropout + Linear
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(num_ftrs, num_classes)
    )
    # 确保分类头可训练，并初始化
    for param in model.fc.parameters():
        param.requires_grad = True
    nn.init.kaiming_normal_(model.fc[1].weight, mode='fan_out', nonlinearity='relu')

    return model


if __name__ == "__main__":
    # 简单测试：打印模型结构
    model = get_mri_model(
        num_classes=2,
        in_channels=1,
        pretrained=True,
        freeze_backbone=False,
        dropout_p=0.5
    )
    print(model)
