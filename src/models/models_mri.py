import torch.nn as nn
import torchvision.models as models

def get_mri_model(arch='resnet18', pretrained=False, num_classes=2):
    """
    构建用于MRI图像分类的模型。

    参数:
    - arch: 模型架构名称，例如 'resnet18'
    - pretrained: 是否使用预训练权重（ImageNet）
    - num_classes: 最终分类类别数量（AS vs Healthy 为 2）

    返回:
    - 构建完成的模型
    """
    if arch == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    elif arch == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    elif arch == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    else:
        raise ValueError(f"Unsupported architecture: {arch}")
