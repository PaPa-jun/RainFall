from model import BinaryClassificationModel, train_model
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 忽略警告
warnings.filterwarnings('ignore')

# 加载训练数据
train_file_path = "data/playground-series-s5e3/train.csv"
test_file_path = "data/playground-series-s5e3/test.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
test_df.fillna(test_df.mean(), inplace=True)

# 检查数据
print("训练数据：")
print(train_df.head())
print(train_df.info())

print("\n测试数据：")
print(test_df.head())
print(test_df.info())

# 分离特征和目标变量
X_train = train_df.drop(columns=['id', 'day', 'rainfall']).values  # 训练特征
y_train = train_df['rainfall'].values  # 训练目标变量
X_test = test_df.drop(columns=['id', 'day']).values  # 测试特征

# 标准化特征数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 将数据转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 转换为列向量
X_test = torch.tensor(X_test, dtype=torch.float32)

# 定义模型参数
input_size = X_train.shape[1]  # 输入特征维度
hidden_size = 64  # 隐藏层大小
output_size = 1  # 输出层大小（二分类问题）

# 初始化模型、损失函数和优化器
model = BinaryClassificationModel(input_size, hidden_size, output_size)
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.002)

# 从训练集中划分验证集（用于计算准确率）
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 训练模型
train_losses, val_losses, val_accuracies = train_model(
    model, criterion, optimizer, X_train, y_train, X_val, y_val,
    num_epochs=500, batch_size=32, print_every=5
)

# 可视化训练过程
plt.figure(figsize=(12, 5))

# 绘制训练损失和验证损失
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# 绘制验证准确率
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Val Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 在验证集上计算最终准确率
with torch.no_grad():
    model.eval()  # 设置模型为评估模式
    val_outputs = model(X_val)
    val_preds = (val_outputs > 0.5).float()  # 将概率值转换为0或1
    val_accuracy = (val_preds == y_val).float().mean()
    print(f'Final Validation Accuracy: {val_accuracy.item():.4f}')

# 在测试集上进行预测
with torch.no_grad():
    model.eval()  # 设置模型为评估模式
    test_outputs = model(X_test)
    # test_preds = (test_outputs > 0.5).float()  # 将概率值转换为0或1

# 将预测结果保存到 CSV 文件
test_preds = test_outputs.numpy().flatten()  # 转换为 NumPy 数组并展平
submission_df = pd.DataFrame({'id': test_df['id'], 'rainfall': test_preds})
submission_df['rainfall'] = submission_df['rainfall'].round(1)
submission_df.to_csv('submission.csv', index=False)

print("测试集预测结果已保存到 submission.csv")