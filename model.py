import torch
import torch.nn as nn

class YeNet(nn.Module):
    def __init__(self, num_classes=200):
        super(YeNet, self).__init__()
        
        # --- 特征提取部分 (VGG Style: Double Conv) ---
        self.features = nn.Sequential(
            # Block 1: 基础纹理 (3 -> 32 -> 32)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 多卷一次！
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 224 -> 112

            # Block 2: 进阶特征 (32 -> 64 -> 64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 112 -> 56

            # Block 3: 高阶特征 (64 -> 128 -> 128)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 56 -> 28

            # Block 4: 顶级抽象 (128 -> 256 -> 256) <--- 新增的深度！
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 这里不加 Pool 了，保持 28x28，或者加了变成 14x14 也可以
            # 我们直接丢给 AdaptiveAvgPool，所以输出尺寸不重要
        )
        
        # --- 自适应池化 (不管是 28x28 还是 14x14，统统压成 4x4) ---
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # --- 分类头 ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 计算维度: 最后一层通道是 256，池化后是 4x4
            # 所以输入是: 256 * 4 * 4 = 4096
            nn.Linear(4096, 1024), # 稍微加大一点中间层
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# --- 自检代码 ---
if __name__ == "__main__":
    # 假装这是 4090，跑个测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(2, 3, 224, 224).to(device)
    model = YeNet(num_classes=200).to(device)
    out = model(dummy)
    print(f"✅ YeNet V3.0 (4090特供版) 构建成功！")
    print(f"输出形状: {out.shape}")