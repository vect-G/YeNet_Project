import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ==========================================
# 1. SE 模块 (注意力机制)
# ==========================================
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
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
        # 赋予特征权重：重要的通道变亮，不重要的变暗
        return x * y.expand_as(x)

# ==========================================
# 2. 改进版 DenseLayer (加入 SE)
# ==========================================
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate
        
        # 🔥 升级点：给新生长的特征加 SE 注意力
        self.se = SELayer(growth_rate)

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(out)))
        
        # 🔥 升级点：注意力加权
        new_features = self.se(new_features)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
            
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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

# ==========================================
# 3. YeDenseNet Fusion (SE + Bilinear)
# ==========================================
class YeDenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=200):
        super(YeDenseNet, self).__init__()

        # --- Stage 0: Initial Conv ---
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # --- Stage 1-4: DenseBlocks ---
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
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # ==========================================
        # 🔥 升级点：双线性池化头部 (Bilinear Head)
        # ==========================================
        self.num_features = num_features
        
        # 1. 降维层：把 1024/2048 维降到 128 维，否则双线性矩阵太大显存会爆
        # CUB-200 经验值：128 或 256 比较好
        self.dim_reduction_dim = 128 
        self.reduce_dim = nn.Conv2d(num_features, self.dim_reduction_dim, kernel_size=1)
        
        # 2. 分类器：输入是 reduction_dim * reduction_dim (因为是自己乘自己)
        # 128 * 128 = 16384
        self.classifier = nn.Linear(self.dim_reduction_dim * self.dim_reduction_dim, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 🔥 修复点：先检查 bias 是否存在！
                # 只有当 bias 不是 None 的时候，才去初始化它
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        
        # --- Bilinear Pooling Start ---
        
        # 1. 1x1 卷积降维 [B, 1024, H, W] -> [B, 128, H, W]
        out = self.reduce_dim(out)
        
        b, c, h, w = out.size()
        
        # 2. Reshape成 [B, 128, H*W]
        # 相当于把每个像素点看作一个 128维 的特征向量
        out = out.view(b, c, h * w)
        
        # 3. 矩阵乘法 (Bilinear): [B, 128, N] * [B, N, 128] = [B, 128, 128]
        # 这一步捕捉了通道之间的二阶相关性（纹理特征）
        out = torch.bmm(out, out.transpose(1, 2)) / (h * w)
        
        # 4. Signed Square Root Normalization (双线性池化标配，防止数值过大)
        out = torch.sign(out) * torch.sqrt(torch.abs(out) + 1e-5)
        
        # 5. L2 Normalization (展平前做)
        out = F.normalize(out.view(b, -1), dim=1)
        
        # --- Bilinear Pooling End ---
        
        # 6. 分类
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YeDenseNet(num_classes=200).to(device)
    dummy = torch.randn(2, 3, 448, 448).to(device) # CUB 常用大分辨率
    output = model(dummy)
    print("✅ YeDenseNet Fusion (SE + Bilinear) 构建成功！")
    print(f"输出尺寸: {output.shape}") # 应该还是 [2, 200]
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params/1e6:.2f} M")