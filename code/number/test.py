import torch
import os
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from number import Network  # 假设Network类定义在network.py文件中

if __name__ == '__main__':
    # 1. 加载测试数据集
    test_data = mnist.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_data, batch_size=32)
    
    # 2. 创建模型并加载预训练权重
    model = Network()  # 创建一个network模型
    # 调用load_state_dict，读取已经训练好的模型文件network.digit
    # 使用绝对路径确保能找到模型文件
    model_path = os.path.join(os.path.dirname(__file__), 'network.digit')
    model.load_state_dict(torch.load(model_path))
    
    # 3. 模型测试
    right = 0  # 设置right变量，保存预测正确的样本数量
    all_samples = 0  # 注意：避免使用内置函数名'all'作为变量名
    
    # 遍历test_loader中的数据
    # x表示样本的特征张量，y表示样本的标签
    for (x, y) in test_loader:
        pred = model(x)  # 使用模型预测x的结果，保存在pred中
        
        # 检查pred和y是否相同
        # pred.argmax(1)获取每个样本预测的最大值索引
        # eq(y)比较预测结果与真实标签是否相等
        if pred.argmax(1).eq(y)[0] == True:
            right += 1  # 如果相同，那么right加1
        
        all_samples += 1  # 每次循环，all_samples变量加1
    
    # 4. 计算并输出准确率
    acc = right * 1.0 / all_samples
    print("test accuracy = %d / %d = %.3lf" % (right, all_samples, acc))