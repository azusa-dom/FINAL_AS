# test_models_mri.py

import torch
from models_mri import get_mri_model

def main():
    # 1) 实例化：不加载预训练权重，方便快速跑通
    model = get_mri_model(
        num_classes=2,     # 输出类别数
        in_channels=1,     # 单通道 MRI
        pretrained=False,  # 随机初始化
        freeze_backbone=False,
        dropout_p=0.5
    )
    model.eval()

    # 2) 构造一个假 batch：batch_size=4, 通道=1, 分辨率=224×224
    x = torch.randn(4, 1, 224, 224)

    # 3) 前向推理
    with torch.no_grad():
        out = model(x)

    # 4) 打印核对
    print(f"输入张量形状：{x.shape}")
    print(f"输出张量形状：{out.shape}")  # 期望 (4, 2)

    # 5) 简单断言，确保维度正确
    assert out.shape == (4, 2), "❌ 输出维度不符合预期！"
    print("✅ Forward 测试通过：输出维度正确！")

if __name__ == "__main__":
    main()
