import os
import datetime
import argparse  
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR 
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler 

# --- 引入你的积木 ---
from dataset import CUBDataset
from model_ultra import YeDenseNet 

# ==========================================
# 🔥 核心增强算法区 (Mixup & CutMix)
# ==========================================
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mixup_data(x, y, alpha=1.0):
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
    parser.add_argument('--log_dir', type=str, default=None, help='Specific log directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Total epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    return parser.parse_args()

def main():
    args = get_args()
    
    # --- 配置 ---
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    NUM_WORKERS = 4 
    
    # --- 🔥 智能日志目录逻辑 ---
    if args.log_dir:
        log_dir = args.log_dir
        print(f"♻️ [Ultra] 强制锁定日志目录: {log_dir}")
    elif args.checkpoint:
        log_dir = os.path.dirname(args.checkpoint)
        print(f"♻️ [Ultra] 自动推断日志目录: {log_dir}")
    else:
        start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join('./logs', 'Ultra_' + start_time)
        print(f"✨ [Ultra] 创建新日志目录: {log_dir}")

    os.makedirs(log_dir, exist_ok=True)
    
    # --- 🔥 智能日志文件逻辑 ---
    log_file_path = os.path.join(log_dir, 'train_dense_log_with_train_ultra_py.txt')
    
    if os.path.exists(log_file_path):
        print(f"📂 [日志] 发现旧日志，开启【无缝续写模式】...")
        log_mode = 'a' 
    else:
        print(f"✨ [日志] 创建新日志文件...")
        log_mode = 'w' 
    
    with open(log_file_path, log_mode) as f:
        if log_mode == 'w':
            f.write("Epoch,Train_Loss,Val_Loss,Val_Acc_TTA,Learning_Rate\n")

    print(f"🔥 [ULTRA MODE] 全维作战启动！")
    print(f"⚔️  策略: Mixup(50%) + CutMix(50%) + CosineAnnealing + TTA + AMP")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True 

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

    # --- 优化器定义 (先定义优化器，因为加载Checkpoint可能需要load它的状态) ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4) 

    # --- 加载权重 & 恢复断点 ---
    start_epoch = 0
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"♻️ Resuming from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # 1. 加载模型权重 (兼容两种格式：纯权重 or 字典)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            # 如果Checkpoint里保存了epoch，优先使用
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            # 如果保存了optimizer，也加载
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            # 旧格式：只保存了 state_dict
            model.load_state_dict(checkpoint)
        
        # 2. 只有当 start_epoch 还是 0 的时候，才尝试从文件名推断
        if start_epoch == 0:
            try:
                filename = os.path.basename(args.checkpoint)
                # 分割文件名，例如 "epoch_80.pth"
                parts = filename.split('_')
                for i, part in enumerate(parts):
                    # 找到 'epoch' 后面那一部分，并去掉 .pth 后缀
                    if part == 'epoch' and i + 1 < len(parts):
                        epoch_str = parts[i+1].replace('.pth', '') # 🔥 姐姐帮你修好的地方
                        start_epoch = int(epoch_str)
                        print(f"🕒 [推断] 从文件名识别出起始 Epoch: {start_epoch}")
                        break
            except Exception as e:
                print(f"⚠️ 无法从文件名推断 Epoch，将从 0 开始。Error: {e}")
    
    
    # ====================================================
    # 🚑 急救补丁：给失忆的优化器注入 initial_lr
    # ====================================================
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            print(f"⚠️ [Fix] 优化器丢失 initial_lr，正在手动恢复为: {LR}")
            param_group['initial_lr'] = LR
            
    # ====================================================
    
    # --- 调度器 ---
    # 🔥 关键修复：last_epoch 参数
    # 如果 start_epoch 是 80，说明我们想跑第 81 轮 (index 80)。
    # 告诉 Scheduler 上一轮是 79 (start_epoch - 1)，它就会算出第 80 轮该有的 LR。
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6, last_epoch=start_epoch-1)
    
    scaler = GradScaler() 

    # --- 训练循环 ---
    print(f"\n🏁 Start Training from Epoch {start_epoch + 1}...")
    
    # 如果是为了 TTA 验证保存 best accuracy，这里先设个初始值
    best_acc = 0.0

    # 🔥 循环范围：从 start_epoch 到 EPOCHS
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Mixup / CutMix
            r = np.random.rand(1)
            if r < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0)
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
            
            inputs = inputs.to(device)
            targets_a, targets_b = targets_a.to(device), targets_b.to(device)

            with autocast():
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            
            if (i + 1) % 20 == 0:
                 # 获取当前真实 LR 打印出来让你放心
                 curr_lr_disp = optimizer.param_groups[0]['lr']
                 print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i+1}] Loss: {loss.item():.4f} LR: {curr_lr_disp:.6f}")

        train_loss_avg = running_loss / len(train_loader)

        # TTA 验证
        val_loss, val_acc = validate_tta(model, val_loader, criterion, device)
        
        # 调度器更新
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 记录日志
        with open(log_file_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss_avg:.4f},{val_loss:.4f},{val_acc:.2f},{current_lr:.6f}\n")
        
        print(f"✨ Epoch {epoch+1} | Train Loss: {train_loss_avg:.4f} | TTA Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            print(f"🏆 New Best Accuracy! Model Saved.")
        
        # 定期保存 (这里我帮你改成了保存完整字典，以后就不用猜文件名了)
        if (epoch + 1) % 10 == 0:
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(checkpoint_dict, os.path.join(log_dir, f'epoch_{epoch+1}.pth'))
            print(f"💾 Checkpoint saved: epoch_{epoch+1}.pth")

    print(f"\n🎉 训练完成！最高 TTA 准确率: {best_acc:.2f}%")

def validate_tta(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    tta_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [3])
    ]

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs_sum = torch.zeros(images.size(0), 200).to(device)
            
            for t in tta_transforms:
                aug_images = t(images)
                with autocast():
                    outputs = model(aug_images)
                outputs_sum += outputs
            
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