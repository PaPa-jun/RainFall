import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个简单的二分类神经网络模型
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(BinaryClassificationModel, self).__init__()
        
        # 定义多层网络
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch Normalization
        self.dropout1 = nn.Dropout(dropout_prob)  # Dropout
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.dropout3 = nn.Dropout(dropout_prob)
        
        self.fc4 = nn.Linear(hidden_size // 4, output_size)
        
        # 残差连接
        self.residual = nn.Linear(input_size, hidden_size // 4)
        
    def forward(self, x):
        # 第一层
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out)  # 使用 LeakyReLU 激活函数
        out = self.dropout1(out)
        
        # 第二层
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.leaky_relu(out)
        out = self.dropout2(out)
        
        # 第三层
        out = self.fc3(out)
        out = self.bn3(out)
        out = F.leaky_relu(out)
        out = self.dropout3(out)
        
        # 残差连接
        residual = self.residual(x)
        out = out + residual  # 残差连接
        
        # 输出层
        out = self.fc4(out)
        out = torch.sigmoid(out)  # 使用 Sigmoid 激活函数进行二分类
        return out