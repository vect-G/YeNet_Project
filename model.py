import torch
import torch.nn as nn

class YeNet(nn.Module):
    def __init__(self, num_classes=200):
        super(YeNet, self).__init__()
        
        # --- 特征提取 (不变) ---
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 

            # Layer 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Layer 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # --- 【关键修改】新增：自适应池化层 ---
        # 不管前面输出多大，强行把它压缩成 4x4 的大小
        # 这样特征维度就是：128 * 4 * 4 = 2048 (而不是之前的 10万！)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # --- 分类部分 (参数量大幅减少) ---
        self.classifier = nn.Sequential(
            nn.Flatten(), 
            # 输入维度变成了 2048，这下轻松多了
            nn.Linear(2048, 512), 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(512, num_classes) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x) # <--- 记得这里要经过池化
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    dummy = torch.randn(2, 3, 224, 224)
    model = YeNet(num_classes=200)
    out = model(dummy)
    print(f"✅ YeNet V2 手术成功！输出形状: {out.shape}")