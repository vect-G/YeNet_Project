import os
import datetime
import argparse  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

# 引入你的积木
from dataset import CUBDataset
from model import YeNet

def get_args():
    """ 定义命令行参数，让代码变聪明 """
    parser = argparse.ArgumentParser(description='YeNet Training Script')
    
    # 核心参数：存档路径
    # 如果不填，默认为 None (从头开始)
    # 如果填了，比如 --checkpoint ./logs/xxx.pth，就加载
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to checkpoint to resume from (default: None, fresh start)')
    
    # 其他可以灵活调整的参数 (我也顺手帮你加上了，以后改这些也不用进代码了)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate (default: 0.001)')

    return parser.parse_args()

def main():
    # 1. 获取命令行参数
    args = get_args()
    
    # --- 实验配置 (从 args 里读) ---
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    NUM_WORKERS = 8
    SAVE_FREQ = 10

    # --- 2. 搭建日志窝 ---
    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('./logs', start_time)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, 'train_log.txt')
    with open(log_file_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_Acc,Learning_Rate\n")

    print(f"📁 [日志] 本次实验保存在: {log_dir}")
    print(f"⚙️ [配置] Batch: {BATCH_SIZE} | Epochs: {EPOCHS} | LR: {LR}")

    # --- 3. 硬件配置 ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 [硬件] 4090 引擎启动: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    # --- 4. 数据增强 (V3 标准版) ---
    data_transform = transforms.Compose([
        # TODO: 后期冲刺 A+ 时，可以尝试把输入调大到 448x448 (配合 RandomResizedCrop(448))，
        #       大分辨率对细粒度分类提分非常显著！目前先用 256/224 跑通基准。
        transforms.Resize((512, 512)),
        transforms.RandomResizedCrop(448), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),      # 先放大到 512 (和训练集保持一致)
        transforms.CenterCrop(448),         # 从正中间切一块 448 (保住长宽比，不 deform)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("🔄 正在加载数据...")
    train_dataset = CUBDataset(root_dir='./data/train', transform=data_transform)
    val_dataset = CUBDataset(root_dir='./data/val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)
    print("✅ 数据集加载完毕！")

    # --- 5. 模型初始化 ---
    model = YeNet(num_classes=200).to(device)

    # 🔥【智能加载逻辑】
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"♻️ [续训] 正在加载存档: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        print("✅ 记忆已唤醒！继续战斗！")
    else:
        print("✨ [新建] 从零开始训练 (Fresh Start)")

    # --- 6. 优化器 & 变速箱 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --- 7. 训练循环 ---
    print("\n🏁 开始训练...")
    
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
            
            if (i + 1) % 20 == 0:
                 print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_loss_avg = running_loss / len(train_loader)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 变速
        scheduler.step(val_loss) 
        current_lr = optimizer.param_groups[0]['lr']

        # 记账
        log_line = f"{epoch+1},{train_loss_avg:.4f},{val_loss:.4f},{val_acc:.2f},{current_lr:.6f}\n"
        with open(log_file_path, 'a') as f:
            f.write(log_line)
        
        print(f"✨ Epoch {epoch+1} | Train Loss: {train_loss_avg:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        # 存档
        if (epoch + 1) % SAVE_FREQ == 0 or (epoch + 1) == EPOCHS:
            ckpt_name = f'checkpoint_epoch_{epoch+1}_acc_{val_acc:.2f}.pth'
            save_path = os.path.join(log_dir, ckpt_name)
            torch.save(model.state_dict(), save_path)
            print(f"💾 存了个档: {ckpt_name}")

    print(f"\n🎉 跑完啦！快去 {log_dir} 收菜！")

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    acc = 100 * correct / total
    return avg_loss, acc

if __name__ == "__main__":
    main()