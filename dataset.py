import os
from torch.utils.data import Dataset
from PIL import Image

class CUBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        :param root_dir: 数据集根目录 (例如 './data/train')
        :param transform: 预处理 (比如把图片变大变小、转成Tensor)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 1. 扫描所有文件夹
        # 必须排序(sorted)，保证 001号鸟永远对应 ID 0
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        # 2. 收集所有 jpg 图片路径
        for cls_name in classes:
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_folder):
                continue
            
            label = self.class_to_idx[cls_name]
            
            for file_name in os.listdir(cls_folder):
                # 重点：Task 2 只要 jpg 图片
                if file_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(cls_folder, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. 根据索引拿路径
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 2. 读取图片并转为 RGB (防止黑白图报错)
        image = Image.open(img_path).convert('RGB')
        
        # 3. 如果有预处理，就处理一下 (变成 Tensor)
        if self.transform:
            image = self.transform(image)
            
        return image, label