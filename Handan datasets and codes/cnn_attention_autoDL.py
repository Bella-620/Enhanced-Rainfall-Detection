import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from functions import cml_confusion_matrix, plot_roc_curve
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns


train_dir = '/root/autodl-tmp/handan_cwt_images/train'
test_dir = '/root/autodl-tmp/handan_cwt_images/test'

# 通过定义transform来对图片做一系列的处理，如转化为张量、设置大小、归一化、设置旋转等
transform = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
])

# 使用torchvision.datasets.ImageFolder类分别创建训练数据和测试数据的dataset
train_ds = torchvision.datasets.ImageFolder(
            train_dir,
            transform=transform)
test_ds = torchvision.datasets.ImageFolder(
            test_dir,
            transform=transform)

print(train_ds.classes)
print(train_ds.class_to_idx)
print(len(train_ds), len(test_ds))


BATCHSIZE = 16      # 需手动
train_dl = Data.DataLoader(
                            train_ds,
                            batch_size=BATCHSIZE,
                            shuffle=True)
test_dl = Data.DataLoader(
                          test_ds,
                          batch_size=BATCHSIZE)

id_to_class = dict((v, k) for k, v in train_ds.class_to_idx.items())
print(id_to_class)

imgs, labels = next(iter(train_dl))
print(imgs.shape)


class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)  # 初始size(batch,3,256,256)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.ca = ChannelAttention(64)  # 加入通道注意力模块
        self.fc1 = nn.Linear(64*14*14, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 2)
        self.drop = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积后(batch,16,254,254), pool之后(batch,16,127,127)    128尺寸卷积后(batch,16,126,126), pool之后(batch,16,63,63)
        x = self.pool(F.relu(self.conv2(x)))  # 卷积后(batch,32,125,125), pool之后(batch,32,62,62)             卷积后(batch,32,61,61), pool之后(batch,32,30,30)
        x = self.pool(F.relu(self.conv3(x)))  # 卷积后(batch,64,60,60), pool之后(batch,64,30,30)               卷积后(batch,64,28,28), pool之后(batch,64,14,14)
        x = self.ca(x)  # 应用通道注意力
        x = x.view(-1, 64*14*14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x


def extract_features(dataloader, model):
    features = []
    labels = []
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            # 获取倒数第二层的输出作为特征
            x = model.conv1(X)
            x = model.pool(F.relu(x))
            x = model.pool(F.relu(model.conv2(x)))
            x = model.pool(F.relu(model.conv3(x)))
            x = model.ca(x)
            x = x.view(-1, 64*14*14)
            x = F.relu(model.fc1(x))
            x = F.relu(model.fc2(x))
            
            features.extend(x.cpu().numpy())
            labels.extend(y.cpu().numpy())
    return np.array(features), np.array(labels)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = Model().to(device)
preds = model(imgs.to(device))

print(preds.shape)
print(torch.argmax(preds, 1))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            train_loss += loss.item()
    train_loss /= num_batches
    correct /= size
    return train_loss, correct


def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []
    y_scores = []  # 存储预测概率
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        with torch.no_grad():
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # 将预测结果和真实标签转换为numpy数组并收集
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
              
            probs = torch.softmax(pred, dim=1).cpu().numpy()
            probs = probs[:, 1]  # 取正类的概率（假设类别1是正类）
            y_scores.extend(probs)

    test_loss /= num_batches
    correct /= size
    return test_loss, correct, np.array(all_preds), np.array(all_labels), np.array(y_scores)


epochs = 200
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(epochs):
    epoch_loss, epoch_acc = train(train_dl, model, loss_fn, optimizer)
    epoch_test_loss, epoch_test_acc, all_preds, all_labels, y_scores = test(test_dl, model)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

    template = ("epoch:{:2d}, train_loss:{:.5f}, train_acc:{:.1f}%,"
                "test_loss:{:.5f}, test_acc:{:.1f}%")
    print(template.format(epoch, epoch_loss, epoch_acc*100, epoch_test_loss, epoch_test_acc*100))

print("Done!")


# 提取测试集的特征和标签
train_features, train_labels = extract_features(train_dl, model)
test_features, test_labels = extract_features(test_dl, model)

# 使用T-SNE降维
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
train_features_tsne = tsne.fit_transform(train_features)  # 训练集降维
test_features_tsne = tsne.fit_transform(test_features)    # 测试集降维

# 构建DataFrame（修正变量名）
train_tsne_df = pd.DataFrame({
    'TSNE1': train_features_tsne[:, 0],  # 使用train_features_tsne
    'TSNE2': train_features_tsne[:, 1],
    'Label': train_labels
})
test_tsne_df = pd.DataFrame({
    'TSNE1': test_features_tsne[:, 0],   # 使用test_features_tsne
    'TSNE2': test_features_tsne[:, 1],
    'Label': test_labels
})
# tsne_df['Label'] = tsne_df['Label'].map(id_to_class)  # 将数字标签映射回类别名称
train_tsne_df.to_pickle('/root/autodl-tmp/performance_handan_cnn_attention/train_tsne_data.pkl')
test_tsne_df.to_pickle('/root/autodl-tmp/performance_handan_cnn_attention/test_tsne_data.pkl')

# 保存模型和预处理器
torch.save(model.state_dict(), '/root/autodl-tmp/performance_handan_cnn_attention/model.pth')
# joblib.dump(scaler, '/root/autodl-tmp/performance_handan_cnn_attention/scaler.joblib')


# 绘制T-SNE散点图
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='TSNE1', y='TSNE2',
    hue='Label',
    palette=sns.color_palette("hls", 2),
    data=train_tsne_df,
    legend="full",
    alpha=0.7
)
plt.savefig('/root/autodl-tmp/performance_handan_cnn_attention/train_tsne.png')  # 保存图像
plt.close()


# 绘制T-SNE散点图
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='TSNE1', y='TSNE2',
    hue='Label',
    palette=sns.color_palette("hls", 2),
    data=test_tsne_df,
    legend="full",
    alpha=0.7
)
plt.savefig('/root/autodl-tmp/performance_handan_cnn_attention/test_tsne.png')  # 保存图像
plt.close()


plt.plot(range(1, epochs+1), train_loss, label='train_loss')
plt.plot(range(1, epochs+1), test_loss, label='test_loss', ls="--")
plt.xlabel('epoch')
plt.legend()
plt.savefig('/root/autodl-tmp/performance_handan_cnn_attention/loss.jpg')  # 需手动
plt.close()

plt.plot(range(1, epochs+1), train_acc, label='train_acc')
plt.plot(range(1, epochs+1), test_acc, label='test_acc', ls="--")
plt.xlabel('epoch')
plt.legend()
plt.savefig('/root/autodl-tmp/performance_handan_cnn_attention/acc.png')  # 需手动
plt.close()


cml_confusion_matrix(all_labels, all_preds)

# print(y_true)
print('y_pred:', all_preds)
np.savetxt("/root/autodl-tmp/performance_handan_cnn_attention/y_pred.txt", all_preds, fmt='%d')  # 需手动
np.savetxt("/root/autodl-tmp/performance_handan_cnn_attention/y_true.txt", all_labels, fmt='%d')  # 需手动
np.savetxt("/root/autodl-tmp/performance_handan_cnn_attention/y_score.txt", y_scores)  # 需手动

# print(y_score)


plot_roc_curve(all_labels, y_scores)


