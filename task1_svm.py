import os
import torch
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

def load_features(root_dir, num_classes=10):
    """
    Load pre-extracted feature vectors (.pt files) for the first `num_classes` categories.

    Args:
        root_dir (str): Path to the dataset directory (e.g., './data/train').
        num_classes (int): Number of categories to load (default: 10).

    Returns:
        X (np.ndarray): Feature matrix with shape (N, D).
        y (np.ndarray): Label vector with shape (N,).
    """
    print(f"[Info] Loading data from {root_dir}...")
    
    # Sort class names to ensure consistent label encoding
    all_classes = sorted(os.listdir(root_dir))
    selected_classes = all_classes[:num_classes]
    
    features_list = []
    labels_list = []
    
    for label_idx, cls_name in enumerate(selected_classes):
        cls_folder = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_folder):
            continue
            
        for file_name in os.listdir(cls_folder):
            # Only process .pt feature files
            if file_name.endswith('.pt'):
                file_path = os.path.join(cls_folder, file_name)
                
                # Load Tensor, convert to numpy, and flatten to 1D vector
                feature = torch.load(file_path)
                feature_np = feature.numpy().flatten()
                
                features_list.append(feature_np)
                labels_list.append(label_idx)
                
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"       Loaded {len(selected_classes)} classes. Data shape: {X.shape}")
    return X, y

if __name__ == "__main__":
    # Configuration
    train_dir = './data/train'
    val_dir = './data/val'
    num_classes = 10  # As required by Task 1
    
    # 1. Load Data
    print("Step 1: Loading Datasets...")
    X_train, y_train = load_features(train_dir, num_classes=num_classes)
    X_val, y_val = load_features(val_dir, num_classes=num_classes)
    
    # 2. Initialize Model (SVM with Linear Kernel)
    print("\nStep 2: Initializing SVM Classifier (Kernel='linear')...")
    clf = svm.SVC(kernel='linear', C=1.0)
    
    # 3. Train
    print("Step 3: Training model...")
    clf.fit(X_train, y_train)
    
    # 4. Evaluate
    print("Step 4: Evaluating on validation set...")
    y_pred = clf.predict(X_val)
    
    # 5. Report Results
    acc = accuracy_score(y_val, y_pred)
    print("-" * 30)
    print(f"Task 1 Result - SVM Classification")
    print(f"Classes: {num_classes}")
    print(f"Validation Accuracy: {acc * 100:.2f}%")
    print("-" * 30)