import os
import datetime
import argparse  
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from copy import deepcopy 
from torch.optim.lr_scheduler import CosineAnnealingLR 
from torch.utils.data import DataLoader
from torchvision import transforms

# --- 引入积木 ---
from dataset import CUBDataset
# 🔥 引入你的究极融合模型 (确保 model_fusion.py 里的 bias bug 已经修好了哦)
from model_fusion import YeDenseNet 

# ==========================================
# 🍬 绝活：EMA (Exponential Moving Average)
# ==========================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = next(model.parameters()).device
        self.module.to(self.device)

    def update(self, model):
        with torch.no_grad():
            msd = self.module.state_dict()
            model_sd = model.state_dict()
            for k, v in model_sd.items():
                if k in msd:
                    msd[k].copy_(self.decay * msd[k] + (1. - self.decay) * v)

# ==========================================
# 🔥 数据增强区 (Mixup & CutMix)
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
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mixup_data(x, y, alpha=1.0):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
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
    parser = argparse.ArgumentParser(description='YeDenseNet Fusion Training')
    # 🔥 修改点1：Batch Size 降为 16 (Float32 吃显存，求稳)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size') 
    parser.add_argument('--epochs', type=int, default=200, help='Total epochs')
    # 🔥 修改点2：LR 降为 0.0005 (双线性池化容易爆炸，温柔一点)
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate')
    return parser.parse_args()

def main():
    args = get_args()
    
    # --- 配置 ---
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    WARMUP_EPOCHS = 5 
    
    # --- 日志 (自动生成新文件夹，防止跟旧的混淆) ---
    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('./logs', 'Fusion_Fix_' + start_time)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'train_fusion_log.txt')
    
    with open(log_file_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_Acc,EMA_Val_Acc,LR\n") 

    print(f"🔥 [FUSION MODE] 究极融合体启动 (Float32稳健版)！")
    print(f"⚔️  策略: Mixup + CutMix + EMA + Warmup + Bilinear + SE + TrivialAugment")
    print(f"📝 日志目录: {log_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True 

    # --- 数据增强 (究极版) ---
    data_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomResizedCrop(448), 
        transforms.RandomHorizontalFlip(),
        # 🔥 神器一：自动增强
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # 🔥 神器二：随机擦除
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
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
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=4, pin_memory=True)

    print("🏗️ Building YeDenseNet Fusion...")
    model = YeDenseNet(num_classes=200).to(device)
    
    ema_model = ModelEMA(model, decay=0.999) 

    # --- 优化器 ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4) 
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    
    # ❌ 删掉了 Scaler (Mix Precision)

    # --- 训练循环 ---
    print("\n🏁 Start Training...")
    best_acc = 0.0
    best_ema_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Warmup
        if epoch < WARMUP_EPOCHS:
            warmup_lr = LR * (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Mixup / CutMix
            r = np.random.rand(1)
            if r < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0)
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
            
            inputs, targets_a, targets_b = inputs.to(device), targets_a.to(device), targets_b.to(device)

            # 🔥 修改点3：去掉 autocast，回归纯天然 Float32
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            # 🔥 修改点4：普通反向传播
            loss.backward()
            
            # 依然保留梯度裁剪 (Bilinear 必备)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            # ❌ 删掉了 scaler.update()
            
            ema_model.update(model)

            running_loss += loss.item()
            
            if (i + 1) % 20 == 0:
                 curr_lr = optimizer.param_groups[0]['lr']
                 print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i+1}] Loss: {loss.item():.4f} LR: {curr_lr:.6f}")

        train_loss_avg = running_loss / len(train_loader)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        _, ema_val_acc = validate(ema_model.module, val_loader, criterion, device)
        
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()
            
        current_lr = optimizer.param_groups[0]['lr']

        # 记录
        with open(log_file_path, 'a') as f:
            f.write(f"{epoch+1},{train_loss_avg:.4f},{val_loss:.4f},{val_acc:.2f},{ema_val_acc:.2f},{current_lr:.6f}\n")
        
        print(f"✨ Epoch {epoch+1} | Loss: {train_loss_avg:.4f} | Acc: {val_acc:.2f}% | EMA Acc: {ema_val_acc:.2f}% | LR: {current_lr:.6f}")

        if ema_val_acc > best_ema_acc:
            best_ema_acc = ema_val_acc
            torch.save(ema_model.module.state_dict(), os.path.join(log_dir, 'best_ema_model.pth'))
            
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
        
        if (epoch + 1) % 20 == 0:
             torch.save(model.state_dict(), os.path.join(log_dir, f'epoch_{epoch+1}.pth'))

    print(f"\n🎉 训练完成！最高普通准确率: {best_acc:.2f}% | 最高 EMA 准确率: {best_ema_acc:.2f}%")

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            # 🔥 修改点5：验证也没用 autocast
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(val_loader), 100 * correct / total

if __name__ == "__main__":
    main()