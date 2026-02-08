import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles

# 函数传入num，代表每种类别的数据个数
def make_data(num):
    # 设定随机数生成器的种子
    # 使随机数序列，每次运行时都是确定的
    np.random.seed(0)
    
    # 红色数据使用make_blobs生成
    # 以(0,0)为中心的正态分布数据
    red, _ = make_blobs(n_samples=num,
                       centers=[[0,0]],
                       cluster_std=0.15)
    
    # 绿色数据用make_circles生成，分布在红色的周围
    green, _ = make_circles(n_samples=num,
                           noise=0.02,
                           factor=0.7)
    
    # 蓝色数据，数据分布在四个角落
    blue, _ = make_blobs(n_samples=num,
                        centers=[[-1.2,-1.2],
                                [-1.2,1.2],
                                [1.2,-1.2],
                                [1.2,1.2]],
                        cluster_std=0.2)
    
    return green, blue, red


class Network(nn.Module):
    """神经网络模型类"""
    
    def __init__(self, n_in, n_hidden, n_out):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(n_in, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_out)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        return self.layer2(x)


def generate_sample_data():
    """生成示例训练数据"""
    # green = np.array([[0.1, 0.2], [0.3, 0.4]])
    # blue = np.array([[0.5, 0.6], [0.7, 0.8]])
    # red = np.array([[0.9, 1.0], [1.1, 1.2]])
    return make_data(100)


def prepare_data(green, blue, red):
    """将numpy数据转换为PyTorch张量并生成标签"""
    green_tensor = torch.FloatTensor(green)
    blue_tensor = torch.FloatTensor(blue)
    red_tensor = torch.FloatTensor(red)
    
    data = torch.cat((green_tensor, blue_tensor, red_tensor), dim=0)
    label = torch.LongTensor([0] * len(green) + [1] * len(blue) + [2] * len(red))
    
    return data, label


def create_model(n_features, n_hidden, n_classes, learning_rate):
    """创建模型、损失函数和优化器"""
    model = Network(n_features, n_hidden, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer


def train_epoch(model, data, label, criterion, optimizer):
    """执行单次训练迭代"""
    outputs = model(data)
    loss = criterion(outputs, label)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_model(model, data, label, criterion, optimizer, n_epochs, print_interval=1000):
    """训练模型"""
    print("开始训练...")
    for epoch in range(n_epochs):
        loss = train_epoch(model, data, label, criterion, optimizer)
        
        if (epoch + 1) % print_interval == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss:.4f}')
    
    print("训练完成！")


def evaluate_model(model, test_data):
    """使用模型进行预测"""
    model.eval()
    with torch.no_grad():
        predictions = model(test_data)
        _, predicted = torch.max(predictions.data, 1)
    return predicted


def draw_decision_boundary(minx1, maxx1, minx2, maxx2, model):
    """绘制决策边界"""
    xx1, xx2 = np.meshgrid(np.arange(minx1, maxx1, 0.02),
                           np.arange(minx2, maxx2, 0.02))
    
    x1s = xx1.ravel()
    x2s = xx2.ravel()
    z = list()
    
    for x1, x2 in zip(x1s, x2s):
        test_point = torch.FloatTensor([[x1, x2]])
        output = model(test_point)
        _, predicted = torch.max(output, 1)
        z.append(predicted.item())
    
    z = np.array(z).reshape(xx1.shape)
    return xx1, xx2, z


def plot_results(model, green, blue, red):
    """绘制决策边界和数据点"""
    xx1, xx2, z = draw_decision_boundary(-4, 4, -4, 4, model)
    plt.contour(xx1, xx2, z, colors=['orange'])
    
    # 绘制原始数据点
    plt.scatter(green[:, 0], green[:, 1], c='green', label='Green')
    plt.scatter(blue[:, 0], blue[:, 1], c='blue', label='Blue')
    plt.scatter(red[:, 0], red[:, 1], c='red', label='Red')
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Multi-class Classification Decision Boundary')
    plt.show()


def main():
    """主函数"""
    # 参数设置
    n_features = 2
    n_hidden = 5
    n_classes = 3
    n_epochs = 10000
    learning_rate = 0.001
    
    # 生成和准备数据
    green, blue, red = generate_sample_data()
    data, label = prepare_data(green, blue, red)
    
    # 创建模型
    model, criterion, optimizer = create_model(n_features, n_hidden, n_classes, learning_rate)
    
    # 训练模型
    train_model(model, data, label, criterion, optimizer, n_epochs)
    
    # 预测示例
    test_data = torch.FloatTensor([[0.2, 0.3], [0.6, 0.7], [1.0, 1.1]])
    predicted = evaluate_model(model, test_data)
    print(f"预测结果: {predicted.numpy()}")
    
    # 绘制结果
    plot_results(model, green, blue, red)


if __name__ == '__main__':
    main()
