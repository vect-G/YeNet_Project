import os
import datetime
import argparse  
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler # 🔥 引入 AMP 核心组件

# --- 引入你的积木 ---
from dataset import CUBDataset
from model_pro import YeResNet  # 确保 model_pro.py 是之前那个 SE 版

# --- Mixup 核心函数 ---
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_args():
    parser = argparse.ArgumentParser(description='YeResNet Training Script')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to checkpoint to resume from')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--alpha', type=float, default=1.0, help='Mixup alpha value')
    return parser.parse_args()

def main():
    args = get_args()
    
    # --- 配置 ---
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    NUM_WORKERS = 4 
    SAVE_FREQ = 10

    # --- 日志 ---
    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('./logs', start_time)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, 'train_log.txt')
    with open(log_file_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_Acc,Learning_Rate\n")

    print(f"📁 [日志] 本次实验保存在: {log_dir}")
    print(f"⚙️ [配置] Model: SE-YeResNet | AMP: ON (混合精度) | Mixup: ON")

    # --- 硬件 ---
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print(f"🔥 [硬件] 4090 引擎启动 (AMP Ready): {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        # AMP 必须要有 GPU 才能发挥作用，CPU 跑不动
        print("❌ [错误] AMP 需要 CUDA 支持，当前环境为 CPU。")
        return

    # --- 数据增强 ---
    data_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomResizedCrop(448), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("🔄 正在加载 CUB-200 数据...")
    train_dataset = CUBDataset(root_dir='./data/train', transform=data_transform)
    val_dataset = CUBDataset(root_dir='./data/val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    # --- 模型初始化 ---
    model = YeResNet(num_classes=200).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"♻️ [续训] 加载存档: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)

    # --- 优化器 & Loss ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 🔥 初始化 Scaler (AMP 的核心，负责缩放梯度)
    scaler = GradScaler()

    # --- 训练循环 ---
    print("\n🏁 开始训练 (混合精度模式)...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Mixup 数据准备 (这一步不需要 autocast，因为它只是数据搬运)
            inputs, targets_a, targets_b, lam = mixup_data(images, labels, args.alpha, use_cuda)
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
            
            optimizer.zero_grad()
            
            # 🔥 开启混合精度上下文 (AutoCast)
            # 这里面的计算会自动选择 float16 或 float32
            with autocast():
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            # 🔥 用 Scaler 处理反向传播
            # 1. scale(loss): 把 loss 放大，防止 float16 下溢出
            scaler.scale(loss).backward()
            
            # 2. step(optimizer): 先把梯度缩放回来，如果发现 NaN/Inf 就跳过更新
            scaler.step(optimizer)
            
            # 3. update(): 更新 Scaler 的缩放因子，为下一次迭代做准备
            scaler.update()

            running_loss += loss.item()
            
            if (i + 1) % 20 == 0:
                 print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f} (AMP)")

        train_loss_avg = running_loss / len(train_loader)

        # 验证 (验证也可以开 autocast 加速，虽然提升不明显，但省显存)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss) 
        current_lr = optimizer.param_groups[0]['lr']

        # 记账
        log_line = f"{epoch+1},{train_loss_avg:.4f},{val_loss:.4f},{val_acc:.2f},{current_lr:.6f}\n"
        with open(log_file_path, 'a') as f:
            f.write(log_line)
        
        print(f"✨ Epoch {epoch+1} | Train Loss: {train_loss_avg:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        if (epoch + 1) % SAVE_FREQ == 0 or val_acc > 70.0:
            ckpt_name = f'checkpoint_epoch_{epoch+1}_acc_{val_acc:.2f}.pth'
            save_path = os.path.join(log_dir, ckpt_name)
            torch.save(model.state_dict(), save_path)

    print(f"\n🎉 训练结束！最终模型在: {log_dir}")

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 验证也可以开混合精度，省显存
            with autocast():
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