import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# --- 1. 定义 DenseLayer (基本单元) ---
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        # 瓶颈层 (Bottleneck): 1x1 卷积先降维，节省计算量
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)

        # 核心层: 3x3 卷积提取特征
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate

    def forward(self, x):
        # 1. 先过 BN-ReLU-Conv1x1
        out = self.conv1(self.relu1(self.norm1(x)))
        # 2. 再过 BN-ReLU-Conv3x3
        new_features = self.conv2(self.relu2(self.norm2(out)))

        # 3. Dropout (防止过拟合的神器)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        
        # 4. 核心魔法：将输入 x 和新特征 new_features 拼起来！(Concat)
        return torch.cat([x, new_features], 1)

# --- 2. 定义 DenseBlock (一堆 Layer 的集合) ---
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, # 输入通道数会越变越大
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- 3. 定义 Transition (过渡层) ---
# 因为 DenseBlock 会让通道数爆炸，所以中间需要用 Transition 压缩一下
class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        out = self.pool(out)
        return out

# --- 4. 组装 YeDenseNet (Based on DenseNet-121) ---
class YeDenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=200):
        super(YeDenseNet, self).__init__()

        # --- Stage 0: 初始卷积 (Stem) ---
        # 模仿 ResNet/DenseNet 的标准头：7x7 Conv -> MaxPool
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # --- Stage 1-4: DenseBlocks + Transitions ---
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            
            # 更新通道数：输入 + 层数 * 增长率
            num_features = num_features + num_layers * growth_rate
            
            # 如果不是最后一个 Block，就加一个 Transition 层来压缩
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # --- Final Batch Norm ---
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # --- Classifier ---
        self.classifier = nn.Linear(num_features, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # 全局平均池化
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# --- 自检代码 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DenseNet 显存占用比较大，我们用 AMP 之后应该没问题
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    # 初始化 YeDenseNet-121
    model = YeDenseNet(num_classes=200).to(device)
    
    output = model(dummy_input)
    print(f"✅ YeDenseNet (Ultra版) 构建成功！")
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params/1e6:.2f} M") 
    # DenseNet 参数量通常比 ResNet 少，但计算量不减