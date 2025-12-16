import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 定义基础残差块 (Basic Block) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二层卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径 (Shortcut)
        # 如果输入输出维度不一样（比如 64 -> 128），或者图片尺寸变了，Shortcut 也要做变换
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # <--- 核心：残差连接
        out = F.relu(out)
        return out

# --- 2. 组装 YeResNet (基于 ResNet18) ---
class YeResNet(nn.Module):
    def __init__(self, num_classes=200):
        super(YeResNet, self).__init__()
        self.in_channels = 64

        # 初始层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 堆叠残差层 (ResNet18 结构: 2-2-2-2)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1) # 展平
        out = self.fc(out)
        return out

# --- 3. 自测代码 (看看有没有 Bug) ---
if __name__ == "__main__":
    # 假装有一个 4090，造一个假数据 (Batch=2, Channel=3, 224x224)
    dummy_input = torch.randn(2, 3, 224, 224)
    model = YeResNet(num_classes=200)
    output = model(dummy_input)
    print(f"✅ YeResNet 构建成功！")
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape} (预期应该是 [2, 200])")