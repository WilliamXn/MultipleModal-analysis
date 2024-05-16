import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader

# 定义数据集类
class ChordDataset(Dataset):
    def __init__(self, csv_files, root_dir):
        self.data = []
        self.labels = []
        # 读取每个文件
        for idx, file_name in enumerate(csv_files):
            file_path = os.path.join(root_dir, file_name)
            csv_data = pd.read_csv(file_path, header=None)
            self.data.append(csv_data.iloc[:, :78].values)
            self.labels.extend([idx] * len(csv_data))
        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        sample['data'] = torch.FloatTensor(sample['data'])
        return sample

# 多模态ViT模型定义
class MultiModalViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_layer = TransformerEncoderLayer(d_model=42, nhead=6)
        self.transformer = TransformerEncoder(self.transformer_layer, num_layers=6)
        self.fc_hand = nn.Linear(42, 768)  # 将hand_features的维度从42增加到768
        self.fc_audio = nn.Linear(36, 768)   # 音频特征适配
        self.classifier = nn.Linear(768 * 2, 10)  # 假设有10个类别

    def forward(self, hand_points, audio):
        hand_points = hand_points.view(hand_points.size(0), -1, 42)  # 确保手部坐标数据的形状是(batch_size, seq_len, 42)
        hand_features = self.transformer(hand_points).mean(dim=1)  # 通过Transformer提取特征并取均值
        hand_features = self.fc_hand(hand_features)  # 通过全连接层将hand_features的维度从42增加到768
        audio_features = self.fc_audio(audio)
        combined_features = torch.cat([hand_features, audio_features], dim=1)
        output = self.classifier(combined_features)
        return output

# 初始化数据加载器
def create_datasets(root_dir):
    csv_files = [f for f in os.listdir(root_dir) if f.endswith('.csv')]
    dataset = ChordDataset(csv_files=csv_files, root_dir=root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    return dataloader

# 模型训练
def train_model(dataloader):
    model = MultiModalViT()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 打印模型结构
    print(model)

    for epoch in range(2):  # 简单训练两个epoch
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for batch in dataloader:
            data = batch['data']
            labels = batch['label'].long()  # 转换标签为LongTensor
            optimizer.zero_grad()
            output = model(data[:, :42], data[:, 42:])
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch}, Loss: {avg_loss}, Accuracy: {accuracy}")

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')

    # 从数据集中选择一个样本
    sample = next(iter(dataloader))
    data = sample['data']
    labels = sample['label']

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        output = model(data[:, :42], data[:, 42:])
    _, predicted = torch.max(output, 1)

    # 打印预测结果
    print(f"Predicted: {predicted}, Actual: {labels}")

if __name__ == '__main__':
    root_dir = 'D:/PycharmProjects/pythonProject/MultipleModal/csv/C'
    dataloader = create_datasets(root_dir)
    train_model(dataloader)
