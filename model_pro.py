import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 定义 SE 模块 (加分神器) ---
# 这是一个完全用基础算子搭建的注意力模块，绝对合规！
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # 全局平均池化：把 (B, C, H, W) 变成 (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 两个全连接层，先降维再升维，学习通道之间的权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 把学到的权重乘回原特征图上 (Excitation)
        return x * y.expand_as(x)

# --- 2. 定义 SE-ResidualBlock (魔改版残差块) ---
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

        # 🔥 核心改进：插入 SE 模块
        self.se = SELayer(out_channels)

        # 捷径 (Shortcut) 处理
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 🔥 Attention! 
        # 在加 Shortcut 之前，让 SE 模块给特征打个分
        out = self.se(out) 
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# --- 3. 组装 YeResNet (Based on ResNet18) ---
class YeResNet(nn.Module):
    def __init__(self, num_classes=200):
        super(YeResNet, self).__init__()
        self.in_channels = 64

        # Stage 0: 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Stage 1-4: 堆叠残差层 (ResNet18 结构: 2-2-2-2)
        # 输入 64 -> 输出 64
        self.layer1 = self._make_layer(64, 2, stride=1)
        # 输入 64 -> 输出 128 (尺寸减半)
        self.layer2 = self._make_layer(128, 2, stride=2)
        # 输入 128 -> 输出 256 (尺寸减半)
        self.layer3 = self._make_layer(256, 2, stride=2)
        # 输入 256 -> 输出 512 (尺寸减半)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 权重初始化 (可以让收敛更快，这也是个 Trick)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        out = out.view(out.size(0), -1) # Flatten
        out = self.fc(out)
        return out

# --- 自检代码 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模拟一个 Batch 的输入
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    model = YeResNet(num_classes=200).to(device)
    output = model(dummy_input)
    print(f"✅ YeResNet (SE-Attention版) 构建成功！")
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape} (预期: [2, 200])")
    
    # 算一下参数量，写报告可以用
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params/1e6:.2f} M")