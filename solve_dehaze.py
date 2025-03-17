import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from tqdm import tqdm
import os
import time
from Dehaze import dehaze_main

# 数据加载
train = pd.read_excel('Training_Tag.xlsx')
test = pd.read_csv('Saint_ODIR.csv')

# 数据划分
train_df, val_df = train_test_split(train, test_size=0.2, random_state=73)


# 图像预处理
# 剪切黑色部分
def crop_image_from_gray(img, tol=7):
    # 若为灰度图
    if img.ndim == 2:
        mask = img > tol  # 标记大于阈值的像素
        return img[np.ix_(mask.any(1), mask.any(0))]  # 裁剪
    # 若为RGB
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 彩色转灰度
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:  # 分别裁剪三个通道
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


# 加强对比度
def load_ben_color(image, sigmaX=10):  # sigmaX为x方向标准差
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (300, 300))
    # 高斯模糊-用高斯核对图像进行加权平均
    # 加权叠加- img * w1 + Gaussian_img * w2 + Gamma
    return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)


# 左右眼数据集
class ODIRDataset(Dataset):
    def __init__(self, df,  is_train=True):
        self.df = df
        self.is_train = is_train
        self.labels = df[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].values
        # 数据增强
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((300, 300)),
            transforms.RandomAffine(
                degrees=10,  # 旋转10°
                translate=(0.1, 0.1),  # 平移10%
                shear=10,  # 剪切10°
                scale=(0.9, 1.1)  # 缩放0.9-1.1倍
            ),
            transforms.RandomHorizontalFlip(),  # 50%概率水平翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]) if is_train else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 加载左右眼图像
        #运用去光照预处理
        left_path = os.path.join('images_dehazed_train/', self.df.iloc[idx]['Left-Fundus'])
        right_path = os.path.join('images_dehazed_train/', self.df.iloc[idx]['Right-Fundus'])

        # 转为RGB
        left_img = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)
        left_img = crop_image_from_gray(left_img)
        right_img = crop_image_from_gray(right_img)

        # 转换为Tensor
        left_tensor = self.transform(left_img)
        right_tensor = self.transform(right_img)
        label = torch.FloatTensor(self.labels[idx])
        return (left_tensor, right_tensor), label

# 模型
class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享权重的基础网络
        self.base = models.efficientnet_b3(pretrained=True)
        self.base.classifier = nn.Identity()  # 移除最后的分类层
        # 融合网络
        self.fc = nn.Sequential(
            nn.Linear(1536 * 2, 256),  # EfficientNetB3输出1536通道
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.Sigmoid()
        )

    def forward(self, x_left, x_right):
        # 提取特征
        feat_left = self.base(x_left)
        feat_right = self.base(x_right)
        # 拼接特征
        combined = torch.cat([feat_left, feat_right], dim=1)
        return self.fc(combined)

# Focal Loss：FL(pt)=−α(1−pt)γ*log(pt)
# pt是正确类别的预测概率
# α（平衡因子）：用于平衡正负类别的权重，防止少数类的梯度消失
# γ（调节因子）：用于降低易分类样本的影响，提高模型对困难样本的关注度
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


# 训练配置
def train_model():
    # 数据加载器
    train_dataset = ODIRDataset(train_df, is_train=True)
    val_dataset = ODIRDataset(val_df,  is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNet().to(device)
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=3, min_lr=1e-6)
    criterion = FocalLoss()

    # 训练循环
    epochs = 20
    best_val_loss = float('inf')  # 初始化最佳验证损失
    best_model_path = 'best_model.pth'  # 定义最佳模型保存路径

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=True)
        for (left, right), labels in progress_bar:
            left = left.to(device)
            right = right.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(left, right)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'train_loss': loss.item()})  # 动态更新进度条信息

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc='Validating', leave=False)  # 验证阶段进度条
            for (left, right), labels in val_progress_bar:
                left = left.to(device)
                right = right.to(device)
                labels = labels.to(device)
                outputs = model(left, right)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_progress_bar.set_postfix({'val_loss': loss.item()})  # 动态更新验证损失

        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)  # 覆盖保存最佳模型
            tqdm.write(f'Best model saved at epoch {epoch + 1} with val loss: {avg_val_loss:.4f}')

        # 打印日志
        tqdm.write(f'Epoch {epoch + 1}/{epochs} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}')

# 测试预测
def predict():
    test_dataset = ODIRDataset(test, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNet().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    predictions = []
    with torch.no_grad():
        for (left, right), _ in test_loader:
            left = left.to(device)
            right = right.to(device)
            outputs = model(left, right)
            predictions.append(outputs.cpu().numpy())

    y_test = np.concatenate(predictions)
    for i, j in enumerate(['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']):
        test[j] = y_test[:, i]
    test.drop(test.columns[[1, 2]], axis=1, inplace=True)
    test.to_csv('SaintCHEN_ODIR.csv', index=False)

if __name__ == '__main__':

    predict()