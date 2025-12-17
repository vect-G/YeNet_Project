import os
import datetime
import argparse  
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np # Mixup 需要用到 numpy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

# --- 引入你的积木 (注意这里改成从 model_pro 引入 YeResNet) ---
from dataset import CUBDataset
from model_pro import YeResNet  # <--- 确保 model_pro.py 是刚才那个 SE 版

# --- Mixup 核心函数 (魔法所在) ---
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
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (Default 32 for ResNet)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    # 新增：Mixup 参数
    parser.add_argument('--alpha', type=float, default=1.0, help='Mixup alpha value (1.0 is standard)')
    return parser.parse_args()

def main():
    args = get_args()
    
    # --- 配置 ---
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    NUM_WORKERS = 4 # Windows下如果报错改成0，Linux/Mac用4或8
    SAVE_FREQ = 10

    # --- 日志 ---
    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('./logs', start_time)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file_path = os.path.join(log_dir, 'train_log.txt')
    with open(log_file_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_Acc,Learning_Rate\n")

    print(f"📁 [日志] 本次实验保存在: {log_dir}")
    print(f"⚙️ [配置] Model: SE-YeResNet | Mixup: On (alpha={args.alpha}) | LabelSmoothing: 0.1")

    # --- 硬件 ---
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print(f"🔥 [硬件] 显卡启动: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 [硬件] Mac MPS 加速启动")
        use_cuda = False # MPS 不支持部分 CUDA 操作，Mixup里要注意
    else:
        device = torch.device("cpu")
        use_cuda = False
        print("🐢 [硬件] CPU 慢速模式")

    # --- 数据增强 (保持激进的高分辨率策略) ---
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
    # 确保 dataset.py 和 data 文件夹都在
    train_dataset = CUBDataset(root_dir='./data/train', transform=data_transform)
    val_dataset = CUBDataset(root_dir='./data/val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    # --- 模型初始化 ---
    print("🏗️ 正在构建 SE-YeResNet...")
    model = YeResNet(num_classes=200).to(device)

    # 续训逻辑
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"♻️ [续训] 加载存档: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)

    # --- 损失函数 & 优化器 ---
    # 🔥 关键修改：开启 Label Smoothing (防止过拟合，让 Softmax 没那么绝对)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --- 训练循环 ---
    print("\n🏁 开始全维作战 (SE + Mixup + LabelSmoothing)...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # --- 🔥 Mixup 介入流程 ---
            # 1. 生成混合数据
            inputs, targets_a, targets_b, lam = mixup_data(images, labels, args.alpha, use_cuda)
            # 2. 放到 Variable 里 (新版 PyTorch 其实不需要这步，为了兼容性保留)
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
            
            # 3. 前向传播 (用混合后的图)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 4. 计算混合 Loss
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            # 5. 反向传播
            loss.backward()
            optimizer.step()
            # -----------------------

            running_loss += loss.item()
            
            if (i + 1) % 20 == 0:
                 print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f} (Mixup)")

        train_loss_avg = running_loss / len(train_loader)

        # 验证 (注意：验证集从来不做 Mixup，要是纯净的)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss) 
        current_lr = optimizer.param_groups[0]['lr']

        # 记录日志
        log_line = f"{epoch+1},{train_loss_avg:.4f},{val_loss:.4f},{val_acc:.2f},{current_lr:.6f}\n"
        with open(log_file_path, 'a') as f:
            f.write(log_line)
        
        print(f"✨ Epoch {epoch+1} | Train Loss: {train_loss_avg:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        # 存档
        if (epoch + 1) % SAVE_FREQ == 0 or val_acc > 70.0: # 如果准确率超过70也存一下
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
            outputs = model(images)
            loss = criterion(outputs, labels) # 验证时用普通 Loss
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    acc = 100 * correct / total
    return avg_loss, acc

if __name__ == "__main__":
    main()