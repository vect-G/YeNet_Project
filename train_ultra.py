import os
import datetime
import argparse  
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR # 🔥 换成更高级的余弦退火
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler 

# --- 引入你的积木 ---
from dataset import CUBDataset
# 既然追求最强，我们就默认用 Ultra 版的 DenseNet
from model_ultra import YeDenseNet 

# ==========================================
# 🔥 核心增强算法区 (Mixup & CutMix)
# ==========================================

def rand_bbox(size, lam):
    """CutMix 用的：随机生成一个剪切框"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 随机中心点
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    """CutMix: 剪切粘贴增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    # 生成一个随机框
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    # 也就是把图片A的这个框里的内容，换成图片B的
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # 重新计算 lambda (因为框可能切出去了，要算真实的像素比例)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mixup_data(x, y, alpha=1.0):
    """Mixup: 混合增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# 🚀 主程序
# ==========================================

def get_args():
    parser = argparse.ArgumentParser(description='YeDenseNet Ultra Training')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (Decrease if OOM)')
    parser.add_argument('--epochs', type=int, default=1000, help='Total epochs (Cosine needs more time)')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    return parser.parse_args()

def main():
    args = get_args()
    
    # --- 配置 ---
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    NUM_WORKERS = 4 
    
    # 日志初始化
    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('./logs', 'Ultra_' + start_time)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'train_dense_log_with_train_ultra_py.txt')
    with open(log_file_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_Acc_TTA,Learning_Rate\n") # 注意这里是 TTA Acc

    print(f"🔥 [ULTRA MODE] 全维作战启动！")
    print(f"⚔️  策略: Mixup(50%) + CutMix(50%) + CosineAnnealing + TTA + AMP")
    print(f"📁 日志: {log_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True # 加速

    # --- 数据增强 (保持激进) ---
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

    print("🔄 Loading Data...")
    train_dataset = CUBDataset(root_dir='./data/train', transform=data_transform)
    val_dataset = CUBDataset(root_dir='./data/val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    # --- 模型构建 ---
    print("🏗️ Building YeDenseNet (Ultra)...")
    model = YeDenseNet(num_classes=200).to(device)

    if args.checkpoint:
        print(f"♻️ Resuming from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # --- 优化器 & 调度器 ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4) # 用 AdamW 更好
    
    # 🔥 余弦退火：让学习率从 LR 慢慢降到 0，不仅优雅，而且收敛极佳
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    scaler = GradScaler() # AMP

    # --- 训练循环 ---
    print("\n🏁 Start Training...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # 🔥 随机选择增强策略
            r = np.random.rand(1)
            if r < 0.5:
                # 50% 概率用 Mixup
                inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0)
            else:
                # 50% 概率用 CutMix
                inputs, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
            
            inputs = inputs.to(device)
            targets_a, targets_b = targets_a.to(device), targets_b.to(device)

            # AMP 前向传播
            with autocast():
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            # AMP 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            
            if (i + 1) % 20 == 0:
                 print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i+1}] Loss: {loss.item():.4f} (Mix/Cut)")

        train_loss_avg = running_loss / len(train_loader)

        # 🔥 TTA 验证 (Test Time Augmentation)
        val_loss, val_acc = validate_tta(model, val_loader, criterion, device)
        
        # Scheduler Step (Cosine 不需要传 loss)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 记录
        with open(log_file_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss_avg:.4f},{val_loss:.4f},{val_acc:.2f},{current_lr:.6f}\n")
        
        print(f"✨ Epoch {epoch+1} | Train Loss: {train_loss_avg:.4f} | TTA Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            print(f"🏆 New Best Accuracy! Model Saved.")
        
        # 定期存档
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, f'epoch_{epoch+1}.pth'))

    print(f"\n🎉 训练完成！最高 TTA 准确率: {best_acc:.2f}%")

# 🔥 核心：TTA 验证函数
def validate_tta(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # TTA 变换列表：原图 + 水平翻转
    tta_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [3])
    ]

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 累加 logits
            outputs_sum = torch.zeros(images.size(0), 200).to(device)
            
            for t in tta_transforms:
                # 1. 变换图片
                aug_images = t(images)
                # 2. 预测 (开 AMP 省显存)
                with autocast():
                    outputs = model(aug_images)
                # 3. 累加
                outputs_sum += outputs
            
            # 取平均
            final_outputs = outputs_sum / len(tta_transforms)

            loss = criterion(final_outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(final_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    acc = 100 * correct / total
    return avg_loss, acc

if __name__ == "__main__":
    main()