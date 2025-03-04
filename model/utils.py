import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

def train_model(model, criterion, optimizer, X_train, y_train, X_val=None, y_val=None, num_epochs=100, batch_size=32, print_every=10):
    """
    训练二分类神经网络模型的工具函数。

    参数:
    - model: PyTorch 模型实例
    - criterion: 损失函数（如 nn.BCELoss）
    - optimizer: 优化器（如 optim.Adam）
    - X_train: 训练数据（Tensor）
    - y_train: 训练标签（Tensor）
    - X_val: 验证数据（Tensor，可选）
    - y_val: 验证标签（Tensor，可选）
    - num_epochs: 训练轮数
    - batch_size: 批量大小
    - print_every: 每隔多少轮打印一次训练信息

    返回:
    - train_losses: 训练损失列表
    - val_losses: 验证损失列表（如果有验证数据）
    - val_accuracies: 验证准确率列表（如果有验证数据）
    """
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 将数据转换为 DataLoader 以便分批训练
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        epoch_loss = 0

        # 分批训练
        for batch_X, batch_y in train_loader:
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证过程（如果有验证数据）
        if X_val is not None and y_val is not None:
            model.eval()  # 设置模型为评估模式
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())

                # 计算验证准确率
                val_preds = (val_outputs > 0.5).float()
                val_accuracy = accuracy_score(y_val.numpy(), val_preds.numpy())
                val_accuracies.append(val_accuracy)

        # 打印训练信息
        if (epoch + 1) % print_every == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}', end='')
            if X_val is not None and y_val is not None:
                print(f', Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')
            else:
                print()

    return train_losses, val_losses, val_accuracies