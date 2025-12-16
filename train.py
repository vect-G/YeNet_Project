import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# 引入我们前两步写好的积木
from dataset import CUBDataset
from model import YeNet

def main():
    # --- 1. 配置参数 (超参数) ---
    BATCH_SIZE = 32
    LR = 0.0001          # 学习率 (步子迈多大)
    EPOCHS = 15         # 训练几轮 (把书读几遍)
    
    # 自动检测你的 M4 芯片 (MPS加速)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 检测到 M4 芯片，启用 MPS GPU 加速模式！")
    else:
        device = torch.device("cpu")
        print("🐢 未检测到 GPU，使用 CPU 慢速模式...")

    # --- 2. 准备数据 (Data Pipeline) ---
    # 这一步必须要，把图片变成 Tensor 并缩放到统一大小
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 归一化 (让数据分布在 0 附近，训练更稳)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("🔄 正在加载数据...")
    train_dataset = CUBDataset(root_dir='./data/train', transform=data_transform)
    val_dataset = CUBDataset(root_dir='./data/val', transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ 数据加载完毕！训练集: {len(train_dataset)} 张, 验证集: {len(val_dataset)} 张")

    # --- 3. 初始化模型 ---
    model = YeNet(num_classes=200).to(device) # 把模型搬到 GPU 上

    # --- 4. 定义损失函数和优化器 (重点！) ---
    # 这就是你要找的！它肚子里吞掉了 Softmax
    criterion = nn.CrossEntropyLoss() 
    
    # 优化器：负责更新参数 (这里用 Adam，比 SGD 聪明一点)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # --- 5. 开始循环训练 (Loop) ---
    print("\n🏁 开始训练 YeNet...")
    
    for epoch in range(EPOCHS):
        model.train() # 切换到训练模式 (启用 Dropout)
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            # 把数据搬到 GPU
            images, labels = images.to(device), labels.to(device)

            # A. 清空梯度 (防止上次的残留)
            optimizer.zero_grad()

            # B. 正向传播 (考试)
            outputs = model(images)

            # C. 计算损失 (对答案)
            loss = criterion(outputs, labels)

            # D. 反向传播 (找锅)
            loss.backward()

            # E. 更新参数 (改错)
            optimizer.step()

            running_loss += loss.item()
            
            # 每 10 个 Batch 汇报一次进度
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # --- 每个 Epoch 结束，做一次期中考试 (验证集) ---
        validate(model, val_loader, device)

    # --- 6. 保存模型 ---
    print("\n💾 正在保存模型...")
    torch.save(model.state_dict(), 'ye_net_final.pth')
    print("🎉 大功告成！模型已保存为 ye_net_final.pth")

def validate(model, val_loader, device):
    """验证函数：看看模型在验证集上的表现"""
    model.eval() # 切换到评估模式 (关掉 Dropout)
    correct = 0
    total = 0
    with torch.no_grad(): # 考试时不计算梯度，省显存
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # 选得分最高的那个
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"📊 验证集准确率 (Accuracy): {acc:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    main()