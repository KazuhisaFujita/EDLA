#---------------------------------------
#Since : 2024/09/05
#Update: 2024/11/25
# -*- coding: utf-8 -*-
#---------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision import datasets, transforms
from sklearn.preprocessing import Binarizer
import numpy as np

train_transform = transforms.Compose([
    transforms.ToTensor(),
#     transforms.RandomAffine(
#         degrees=10,               # 回転角度（-10度から10度の範囲でランダム）
#         translate=(0.1, 0.1),     # 平行移動（x, y軸方向に±10%の範囲）
#         scale=(0.9, 1.1),         # スケール（90%から110%の範囲）
#         shear=(-10, 10, -10, 10)  # せん断（10度の範囲）
#     ),
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

class DigitsDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Digitsデータの準備
def prepare_digits_data(batch_size=64, test_size=0.2, random_state=None):
    digits = load_digits()
    X, y = digits.data, digits.target

    # データの正規化（0-1の範囲）
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # 2次元に変換 (N, 64) -> (N, 8, 8)
    X = X.astype(np.float32).reshape(-1, 8, 8)

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    y_train = torch.tensor(y_train)
    y_test  = torch.tensor(y_test)

    train_dataset = DigitsDataset(data=X_train, targets=y_train, transform=train_transform)
    test_dataset  = DigitsDataset(data=X_test,  targets=y_test,  transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# MNISTデータの準備
def prepare_mnist_data(batch_size=64):

    X_train = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    X_test = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Fashion MNISTデータの準備
def prepare_fashion_mnist_data(batch_size=64):

    X_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
    X_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# CIFAR-10データの準備
def prepare_cifar10_data(batch_size=64, one_hot=False):

    X_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    X_test  = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 任意のnビットのパリティチェックデータセットを作成（全パターン）
def generate_parity_data(n_bits, max_n=16):
    # n_bitsが大きすぎる場合に上限を設定
    if n_bits > max_n:
        raise ValueError(f"n_bitsが大きすぎます。最大 {max_n} ビットまでサポートされています。")

    # 2^n_bits通りの全てのパターンを生成
    num_samples = 2 ** n_bits
    inputs = np.array([list(np.binary_repr(i, width=n_bits)) for i in range(num_samples)], dtype=np.int32)

    # 偶数パリティを計算
    targets = np.sum(inputs, axis=1) % 2  # 偶数パリティ（ビットの和が偶数なら0、奇数なら1）

    # データをPyTorchのテンソルに変換
    inputs = torch.FloatTensor(inputs)
    targets = torch.FloatTensor(targets)

    return inputs, targets.unsqueeze(1)

# DataLoaderを作成する汎用関数
def create_dataloaders(X_train, X_test, y_train, y_test, batch_size):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# メイン関数で選択したデータセットに基づくDataLoaderを生成
def load_dataset(dataset_type="mnist", batch_size=64, n_bits=None):
    if dataset_type == "mnist":
        return prepare_mnist_data(batch_size)

    elif dataset_type == "fashion_mnist":
        return prepare_fashion_mnist_data(batch_size)

    elif dataset_type == "cifar10":
        return prepare_cifar10_data(batch_size)

    elif dataset_type == "digits":
        return prepare_digits_data(batch_size)

    elif dataset_type == "parity" and n_bits is not None:
        X, y = generate_parity_data(n_bits)
        return create_dataloaders(X, X, y, y, batch_size)

    else:
        raise ValueError("無効なデータセットの種類です")
