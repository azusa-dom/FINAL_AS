import torch.nn as nn
from torchvision import models

def get_mri_model(num_classes=2,
                  in_channels=3,
                  pretrained=True,
                  freeze_backbone=False,
                  dropout_p=0.5) -> nn.Module:
    """
    返回一个可直接用于 MRI 图像的 ResNet-18：
    - in_channels=3 保留 ImageNet 原生 conv1
    - 支持冻结骨干、可自定义 dropout
    """
    # 1) 加载预训练或随机初始化的 ResNet18
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    # 2) 可选冻结骨干
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    # 3) 替换最后分类头
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(num_ftrs, num_classes)
    )
    # 初始化 fc 权重并缩小方差
    nn.init.kaiming_normal_(model.fc[1].weight, mode='fan_out', nonlinearity='relu')
    model.fc[1].weight.data.mul_(0.01)
    # 确保 fc 可训练
    for p in model.fc.parameters():
        p.requires_grad = True

    return model
