import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau # 引入自动变速箱
from torch.utils.data import DataLoader
from torchvision import transforms

# 引入之前的积木
from dataset import CUBDataset
from model import YeNet

def main():
    # --- 1. 配置参数 ---
    BATCH_SIZE = 32
    LR = 0.001          # 初始学习率
    EPOCHS = 15
    
    # 自动检测 M4 芯片
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 检测到 M4 芯片，Ye_Neural 引擎全速启动！")
    else:
        device = torch.device("cpu")
        print("🐢 未检测到 GPU，使用 CPU...")

    # --- 2. 准备数据 ---
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("🔄 正在加载数据...")
    train_dataset = CUBDataset(root_dir='./data/train', transform=data_transform)
    val_dataset = CUBDataset(root_dir='./data/val', transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ 数据加载完毕！")

    # --- 3. 初始化模型 ---
    model = YeNet(num_classes=200).to(device)

    # --- 4. 定义优化器 & 变速箱 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 【修复重点】去掉了报错的 verbose=True
    # mode='min': 监测 Loss，只要 Loss 不降了就触发
    # factor=0.1: 触发时学习率变小 10 倍
    # patience=2: 忍耐 2 个 Epoch
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=2)

    # --- 5. 训练循环 ---
    print("\n🏁 开始训练 YeNet (Pro版)...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # 每 50 个 Batch 报一次平安
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # --- 验证阶段 (期中考试) ---
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # --- 【关键】告诉变速箱刚才考得怎么样 ---
        scheduler.step(val_loss) 
        
        # --- 【手动播报】因为删了 verbose，我们要自己打印 LR ---
        current_lr = optimizer.param_groups[0]['lr']
        print(f"🌡️ 当前学习率 (LR): {current_lr:.6f}")
        
        # 如果学习率降到了 0.0001 以下，说明进入精细调整期了
        if current_lr < LR:
            print("💡 注意：自动变速箱已介入，学习率下降，准备起飞！")

    # --- 6. 保存 ---
    print("\n💾 正在保存模型...")
    torch.save(model.state_dict(), 'ye_net_final.pth')
    print("🎉 训练结束！模型已保存！")

def validate(model, val_loader, criterion, device):
    """验证函数：返回 Loss 和 Accuracy"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # 计算 Loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    acc = 100 * correct / total
    print(f"📊 验证集 -> Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")
    
    return avg_loss, acc

if __name__ == "__main__":
    main()