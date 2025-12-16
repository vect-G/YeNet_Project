import os

# 这里对应你的 VSCode 截图路径
train_path = './data/train' 

print(f"🧐 正在检查路径: {train_path}")

# 1. 看看能不能找到 train 文件夹
if not os.path.exists(train_path):
    print("❌ 找不到 data/train！请确认你的脚本在 PRCODE 根目录下运行。")
else:
    print("✅ 成功找到 data/train 文件夹！")
    
    # 2. 看看里面有啥
    subfolders = sorted(os.listdir(train_path))
    print(f"📂 发现 {len(subfolders)} 个鸟类文件夹 (例如: {subfolders[0]})")
    
    # 3. 钻进第一个文件夹看看有没有图
    first_bird_folder = os.path.join(train_path, subfolders[0])
    files = os.listdir(first_bird_folder)
    jpg_count = len([f for f in files if f.endswith('.jpg')])
    pt_count = len([f for f in files if f.endswith('.pt')])
    
    print(f"🕊️ 在第一类鸟 ({subfolders[0]}) 里发现:")
    print(f"   - {jpg_count} 张图片 (.jpg) -> 这是做 Task 2 (CNN) 用的")
    print(f"   - {pt_count} 个特征文件 (.pt) -> 这是做 Task 1 (SVM) 用的")

    if jpg_count > 0:
        print("\n🎉 恭喜小狗！路径完全正确，可以开始写 CNN 啦！")
    else:
        print("\n😱 居然没找到 jpg 图片？快叫姐姐！")