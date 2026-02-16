"""
MNIST决策边界可视化脚本
使用PCA降维来可视化神经网络的决策边界
"""

# ========== 标准库导入 ==========
# numpy: 数值计算库，用于数组操作、数学计算
import numpy as np

# ========== 第三方库导入 ==========
# matplotlib.pyplot: 绘图库，用于创建可视化图表
import matplotlib.pyplot as plt

# sklearn.decomposition.PCA: 主成分分析，用于降维
# 将高维数据(784维)降到2维以便于可视化
from sklearn.decomposition import PCA

# sklearn.manifold.TSNE: t-SNE降维算法，另一种非线性降维方法(本脚本中未使用但导入)
from sklearn.manifold import TSNE

# torch: PyTorch深度学习框架核心库
import torch
# torch.nn: 神经网络模块，包含层、激活函数等
from torch import nn
# torchvision.datasets.mnist: MNIST数据集加载器
from torchvision.datasets import mnist
# torchvision.transforms.ToTensor: 数据预处理，将PIL图像转换为Tensor
from torchvision.transforms import ToTensor
# torch.utils.data.DataLoader: 数据加载器，用于批量加载数据
from torch.utils.data import DataLoader
# torch.optim: 优化器模块，包含Adam、SGD等优化算法
import torch.optim as optim

# ========== Matplotlib配置 ==========
# 设置中文字体支持，确保中文标签能正确显示
# 'SimHei': 黑体字体，'DejaVu Sans': 备用字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# ========== 颜色配置 ==========
# 定义标准颜色方案，用于可视化
COLORS = {
    'red': '#FF6B6B',      # 红色 - 错误/警告
    'green': '#51CF66',    # 绿色 - 正确/成功
    'blue': '#339AF0',     # 蓝色 - 主要信息
    'yellow': '#FFD93D',   # 黄色 - 注意/提醒
    'purple': '#9C36B5',   # 紫色
    'orange': '#FF922B',   # 橙色 - 低置信度
    'cyan': '#22D3EE',     # 青色
    'pink': '#F472B6',     # 粉色
    'brown': '#A16207',    # 棕色
    'gray': '#6B7280'      # 灰色
}

# 数字类别对应的颜色列表(0-9共10个数字)
DIGIT_COLORS = ['#FF6B6B', '#51CF66', '#339AF0', '#FFD93D', '#9C36B5', 
                '#FF922B', '#22D3EE', '#F472B6', '#A16207', '#6B7280']


class Network(nn.Module):
    """
    神经网络定义 - 两层全连接网络
    输入: 28x28=784维图像像素
    隐藏层: 256个神经元
    输出: 10个类别(数字0-9)
    """
    def __init__(self):
        # super().__init__(): 调用父类nn.Module的构造函数
        # 初始化网络结构
        super().__init__()
        # nn.Linear(in_features, out_features): 全连接层
        # layer1: 输入784维(28x28像素)，输出256维(隐藏层)
        self.layer1 = nn.Linear(28 * 28, 256)
        # layer2: 输入256维(隐藏层)，输出10维(10个数字类别)
        self.layer2 = nn.Linear(256, 10)

    def forward(self, x):
        """
        前向传播函数 - 定义数据在网络中的流动
        Args:
            x: 输入张量，形状为(batch_size, channels, height, width)
        Returns:
            输出张量，形状为(batch_size, 10)
        """
        # view(-1, 28*28): 重塑张量形状
        # -1表示自动计算batch_size，将每个28x28图像展平为784维向量
        x = x.view(-1, 28 * 28)
        # 第一层线性变换: x @ W1^T + b1
        x = self.layer1(x)
        # torch.relu(x): ReLU激活函数，将负值设为0，增加非线性
        # ReLU(x) = max(0, x)
        x = torch.relu(x)
        # 第二层线性变换，返回logits(未归一化的预测分数)
        return self.layer2(x)


def train_model():
    """
    训练模型函数
    Returns:
        model: 训练好的神经网络模型
        losses: 每轮训练的平均损失列表
        accuracies: 每轮训练的准确率列表
    """
    print("正在训练模型...")
    
    # ========== 数据加载 ==========
    # mnist.MNIST(): 加载MNIST手写数字数据集
    # root='./data': 数据集保存路径
    # train=True: 加载训练集(60000张)
    # download=True: 如果本地没有则自动下载
    # transform=ToTensor(): 将PIL图像转换为PyTorch张量，像素值归一化到[0,1]
    train_data = mnist.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    
    # DataLoader: 数据加载器，用于批量读取数据
    # train_data: 数据集对象
    # shuffle=True: 每个epoch打乱数据顺序，增加随机性
    # batch_size=128: 每批加载128个样本
    train_load = DataLoader(train_data, shuffle=True, batch_size=128)
    
    # ========== 模型、优化器、损失函数初始化 ==========
    # 创建网络实例
    model = Network()
    # optim.Adam(): Adam优化器，自适应学习率优化算法
    # model.parameters(): 返回模型所有可训练参数
    # Adam结合了动量法和RMSprop的优点，收敛速度快
    optimizer = optim.Adam(model.parameters())
    # nn.CrossEntropyLoss(): 交叉熵损失函数
    # 用于多分类问题，结合了LogSoftmax和NLLLoss
    criterion = nn.CrossEntropyLoss()
    
    # 记录训练过程的损失和准确率
    losses = []
    accuracies = []
    
    # ========== 训练循环 ==========
    # 训练10个epoch(完整遍历数据集10次)
    for epoch in range(10):
        epoch_loss = 0
        correct = 0  # 正确预测的数量
        total = 0    # 总样本数
        
        # enumerate(): 获取批次索引和数据
        # batch_idx: 批次索引(0, 1, 2, ...)
        # data: 输入图像张量，形状(batch_size, 1, 28, 28)
        # label: 真实标签张量，形状(batch_size,)
        for batch_idx, (data, label) in enumerate(train_load):
            # model(data): 前向传播，获取预测输出
            # output形状: (batch_size, 10)，每个样本10个类别的分数
            output = model(data)
            # criterion(output, label): 计算预测输出与真实标签之间的损失
            loss = criterion(output, label)
            
            # ========== 计算准确率 ==========
            # output.argmax(dim=1): 获取每个样本预测概率最大的类别索引
            # dim=1表示在第1维度(类别维度)上求argmax
            pred = output.argmax(dim=1)
            # (pred == label): 比较预测和真实标签，返回布尔张量
            # .sum().item(): 统计True的数量(正确预测数)
            correct += (pred == label).sum().item()
            # label.size(0): 获取当前批次的样本数
            total += label.size(0)
            
            # ========== 反向传播和优化 ==========
            # loss.backward(): 反向传播，计算梯度
            # 使用链式法则自动计算损失对每个参数的梯度
            loss.backward()
            # optimizer.step(): 根据梯度更新模型参数
            # 执行优化算法(Adam)更新权重
            optimizer.step()
            # optimizer.zero_grad(): 清零梯度
            # 防止梯度累积，为下一次迭代做准备
            optimizer.zero_grad()
            # 累加当前批次的损失值
            epoch_loss += loss.item()
            
            # 每100个批次打印一次进度
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/10 | Batch {batch_idx}/{len(train_load)} | Loss: {loss.item():.4f}")
        
        # ========== 计算本轮训练统计 ==========
        # avg_loss: 本轮平均损失 = 总损失 / 批次数
        avg_loss = epoch_loss / len(train_load)
        # accuracy: 本轮准确率 = 正确数 / 总数
        accuracy = correct / total
        # 记录本轮统计
        losses.append(avg_loss)
        accuracies.append(accuracy)
        print(f"Epoch {epoch+1} 完成 - 平均损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
    
    # ========== 保存模型 ==========
    import os
    # os.makedirs(): 创建目录(如果不存在)
    # exist_ok=True: 如果目录已存在则不报错
    os.makedirs('../../docs', exist_ok=True)
    # torch.save(): 保存模型状态字典到文件
    # model.state_dict(): 返回包含模型所有参数和缓冲区的字典
    torch.save(model.state_dict(), '../../docs/network_visualization.digit')
    return model, losses, accuracies


def load_or_train_model():
    """
    加载已有模型或训练新模型
    优先尝试加载预训练模型，如果不存在则重新训练
    Returns:
        model: 神经网络模型
        losses: 训练损失(如果是加载则为None)
        accuracies: 训练准确率(如果是加载则为None)
    """
    # 创建新的模型实例
    model = Network()
    try:
        # torch.load(): 从文件加载保存的对象
        # model.load_state_dict(): 将加载的状态字典加载到模型
        model.load_state_dict(torch.load('../../docs/network_visualization.digit'))
        print("已加载预训练模型")
        return model, None, None
    except:
        # 如果加载失败(文件不存在或损坏)，则训练新模型
        print("未找到预训练模型，开始训练...")
        return train_model()


def visualize_training_process(losses, accuracies):
    """
    可视化训练过程 - 绘制损失和准确率曲线
    Args:
        losses: 每轮训练的平均损失列表
        accuracies: 每轮训练的准确率列表
    """
    # plt.subplots(): 创建包含多个子图的图形
    # 1, 2: 创建1行2列的子图布局
    # figsize=(14, 5): 图形尺寸(宽度14英寸，高度5英寸)
    # fig: 图形对象，ax1和ax2: 两个子图对象
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ========== 左图：损失曲线 ==========
    # ax.plot(): 绘制折线图
    # range(1, len(losses)+1): x轴数据(1到10轮)
    # losses: y轴数据(损失值)
    # 'b-o': 蓝色线条(b)带圆形标记(o)
    # linewidth=2: 线宽2像素
    # markersize=6: 标记大小6像素
    ax1.plot(range(1, len(losses)+1), losses, 'b-o', linewidth=2, markersize=6)
    # set_xlabel(): 设置x轴标签
    ax1.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax1.set_ylabel('损失值 (Loss)', fontsize=12)
    # set_title(): 设置图表标题
    # fontweight='bold': 标题加粗
    ax1.set_title('训练损失变化', fontsize=14, fontweight='bold')
    # grid(True, alpha=0.3): 显示网格线，透明度30%
    ax1.grid(True, alpha=0.3)
    # set_xticks(): 设置x轴刻度位置
    ax1.set_xticks(range(1, len(losses)+1))
    
    # ========== 右图：准确率曲线 ==========
    # [a*100 for a in accuracies]: 将准确率转换为百分比(0-100)
    # 'g-s': 绿色线条(g)带方形标记(s)
    ax2.plot(range(1, len(accuracies)+1), [a*100 for a in accuracies], 'g-s', linewidth=2, markersize=6)
    ax2.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_title('训练准确率变化', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, len(accuracies)+1))
    # set_ylim([0, 100]): 设置y轴范围0-100%
    ax2.set_ylim([0, 100])
    
    # ========== 保存图表 ==========
    # tight_layout(): 自动调整子图间距，防止标签重叠
    plt.tight_layout()
    # savefig(): 保存图形到文件
    # dpi=150: 分辨率150点每英寸
    # bbox_inches='tight': 去除白边
    # facecolor='white': 背景白色
    plt.savefig('../../docs/training_process.png', dpi=150, bbox_inches='tight', facecolor='white')
    # close(): 关闭图形，释放内存
    plt.close()
    print("已保存训练过程图: docs/training_process.png")


def visualize_pca_projection(model):
    """
    使用PCA降维可视化数据分布
    将784维图像数据降到2维，在2D平面上展示各类别分布
    Args:
        model: 训练好的神经网络模型
    Returns:
        data_2d: 降维后的2D数据
        labels_np: 真实标签
        predictions: 预测标签
        pca: 训练好的PCA模型
    """
    print("正在生成PCA投影可视化...")
    
    # ========== 加载测试数据 ==========
    # train=False: 加载测试集(10000张)
    test_data = mnist.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    # batch_size=1000: 加载1000个测试样本
    # shuffle=False: 不打乱顺序，保持数据一致性
    test_load = DataLoader(test_data, batch_size=1000, shuffle=False)
    
    # next(iter(test_load)): 获取第一个批次的数据
    # data: 图像张量(1000, 1, 28, 28)
    # labels: 标签张量(1000,)
    data, labels = next(iter(test_load))
    
    # view(-1, 28*28): 展平为(1000, 784)
    # .numpy(): 转换为numpy数组，便于sklearn处理
    data_flat = data.view(-1, 28*28).numpy()
    labels_np = labels.numpy()
    
    # ========== PCA降维 ==========
    # PCA(n_components=2): 创建PCA模型，保留2个主成分
    # n_components=2: 降到2维，方便在2D平面上可视化
    pca = PCA(n_components=2)
    # fit_transform(): 拟合数据并转换
    # 学习主成分方向，并将数据投影到这些方向上
    # data_2d形状: (1000, 2)
    data_2d = pca.fit_transform(data_flat)
    
    # ========== 获取模型预测 ==========
    # torch.no_grad(): 禁用梯度计算，节省内存，加速推理
    # 因为我们只进行前向传播，不需要反向传播
    with torch.no_grad():
        # 对测试数据进行前向传播
        outputs = model(data)
        # 获取预测类别
        predictions = outputs.argmax(dim=1).numpy()
    
    # ========== 创建可视化图形 ==========
    # 创建2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # ========== 子图1: 真实标签分布 ==========
    ax = axes[0, 0]
    # 遍历0-9每个数字类别
    for digit in range(10):
        # labels_np == digit: 创建布尔掩码，标记属于当前数字的样本
        mask = labels_np == digit
        # scatter(): 绘制散点图
        # data_2d[mask, 0]: x坐标(第一主成分)
        # data_2d[mask, 1]: y坐标(第二主成分)
        # c=DIGIT_COLORS[digit]: 使用对应颜色
        # label=f'{digit}': 图例标签
        # alpha=0.6: 透明度60%
        # s=20: 点大小20像素
        # edgecolors='none': 无边框
        ax.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                  c=DIGIT_COLORS[digit], label=f'{digit}', 
                  alpha=0.6, s=20, edgecolors='none')
    # pca.explained_variance_ratio_: 各主成分解释的方差比例
    # [0]: 第一主成分解释的方差比例(如0.12表示12%)
    ax.set_xlabel(f'主成分 1 ({pca.explained_variance_ratio_[0]:.1%} 方差)', fontsize=11)
    ax.set_ylabel(f'主成分 2 ({pca.explained_variance_ratio_[1]:.1%} 方差)', fontsize=11)
    ax.set_title('PCA投影 - 真实标签分布', fontsize=13, fontweight='bold')
    # legend(): 显示图例，title='数字'设置图例标题
    ax.legend(title='数字', loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ========== 子图2: 预测标签分布 ==========
    ax = axes[0, 1]
    for digit in range(10):
        mask = predictions == digit
        # mask.sum() > 0: 只有当有预测为该类的样本时才绘制
        if mask.sum() > 0:
            ax.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                      c=DIGIT_COLORS[digit], label=f'{digit}', 
                      alpha=0.6, s=20, edgecolors='none')
    ax.set_xlabel(f'主成分 1 ({pca.explained_variance_ratio_[0]:.1%} 方差)', fontsize=11)
    ax.set_ylabel(f'主成分 2 ({pca.explained_variance_ratio_[1]:.1%} 方差)', fontsize=11)
    ax.set_title('PCA投影 - 预测标签分布', fontsize=13, fontweight='bold')
    ax.legend(title='预测', loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # ========== 子图3: 正确与错误分类 ==========
    ax = axes[1, 0]
    # predictions == labels_np: 判断每个样本是否正确分类
    correct = predictions == labels_np
    # ~correct: 按位取反，得到错误分类的掩码
    wrong = ~correct
    
    # 绘制正确分类的点(绿色，小点)
    ax.scatter(data_2d[correct, 0], data_2d[correct, 1], 
              c=COLORS['green'], label='正确分类', alpha=0.5, s=15)
    # 绘制错误分类的点(红色，大x标记)
    # marker='x': 使用x形状标记
    # linewidth=2: 标记线宽
    ax.scatter(data_2d[wrong, 0], data_2d[wrong, 1], 
              c=COLORS['red'], label='错误分类', alpha=0.8, s=50, marker='x', linewidth=2)
    ax.set_xlabel(f'主成分 1 ({pca.explained_variance_ratio_[0]:.1%} 方差)', fontsize=11)
    ax.set_ylabel(f'主成分 2 ({pca.explained_variance_ratio_[1]:.1%} 方差)', fontsize=11)
    ax.set_title('分类结果可视化', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # ========== 子图4: 预测置信度 ==========
    ax = axes[1, 1]
    with torch.no_grad():
        # torch.softmax(): Softmax函数，将logits转换为概率
        # dim=1: 在类别维度上应用softmax
        # probs形状: (1000, 10)，每行是一个概率分布
        probs = torch.softmax(model(data), dim=1)
        # max(dim=1): 沿类别维度求最大值
        # [0]: 获取最大概率值(置信度)
        confidences = probs.max(dim=1)[0].numpy()
    
    # 绘制散点图，颜色表示置信度
    # c=confidences: 颜色映射到置信度值
    # cmap='RdYlGn': 使用红-黄-绿色图，红色=低置信度，绿色=高置信度
    scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], 
                        c=confidences, cmap='RdYlGn', 
                        alpha=0.6, s=25, vmin=0, vmax=1)
    # colorbar(): 添加颜色条，显示颜色到置信度的映射
    plt.colorbar(scatter, ax=ax, label='置信度')
    ax.set_xlabel(f'主成分 1 ({pca.explained_variance_ratio_[0]:.1%} 方差)', fontsize=11)
    ax.set_ylabel(f'主成分 2 ({pca.explained_variance_ratio_[1]:.1%} 方差)', fontsize=11)
    ax.set_title('预测置信度分布', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../docs/pca_visualization.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("已保存PCA可视化: docs/pca_visualization.png")
    
    return data_2d, labels_np, predictions, pca


def visualize_decision_regions(model, pca, data_2d, labels):
    """
    在PCA空间中绘制决策边界
    通过在整个PCA空间采样并预测，可视化神经网络的决策区域
    Args:
        model: 训练好的神经网络模型
        pca: 训练好的PCA模型
        data_2d: 降维后的2D数据
        labels: 真实标签
    """
    print("正在生成决策边界可视化...")
    
    # ========== 创建网格点 ==========
    # 确定数据点的边界，并扩展1个单位
    x_min, x_max = data_2d[:, 0].min() - 1, data_2d[:, 0].max() + 1
    y_min, y_max = data_2d[:, 1].min() - 1, data_2d[:, 1].max() + 1
    
    # np.meshgrid(): 创建网格坐标
    # np.linspace(x_min, x_max, 100): 在x范围生成100个均匀分布的点
    # np.linspace(y_min, y_max, 100): 在y范围生成100个均匀分布的点
    # xx, yy: 形状均为(100, 100)的网格坐标矩阵
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # np.c_[]: 按列拼接数组
    # xx.ravel(): 将xx展平为1维(10000,)
    # yy.ravel(): 将yy展平为1维(10000,)
    # grid_points形状: (10000, 2)，包含网格上所有点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # ========== 网格点逆变换到原始空间 ==========
    # pca.inverse_transform(): 将2D PCA坐标逆变换回784D原始空间
    # 注意：这是近似重构，因为我们只保留了2个主成分
    grid_original = pca.inverse_transform(grid_points)
    # 转换为PyTorch张量，并重塑为图像形状(batch, channel, height, width)
    grid_tensor = torch.FloatTensor(grid_original).view(-1, 1, 28, 28)
    
    # ========== 对网格点进行预测 ==========
    with torch.no_grad():
        outputs = model(grid_tensor)
        # 获取每个网格点的预测类别
        grid_preds = outputs.argmax(dim=1).numpy()
        # 获取每个网格点的预测置信度(最大概率)
        grid_probs = torch.softmax(outputs, dim=1).max(dim=1)[0].numpy()
    
    # reshape(): 将预测结果重塑回网格形状(100, 100)
    grid_preds = grid_preds.reshape(xx.shape)
    grid_probs = grid_probs.reshape(xx.shape)
    
    # ========== 创建可视化图形 ==========
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # ========== 子图1: 决策区域 ==========
    ax = axes[0]
    # ListedColormap: 创建离散颜色映射
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(DIGIT_COLORS)
    
    # contourf(): 绘制填充等高线图
    # xx, yy: 网格坐标
    # grid_preds: 每个网格点的预测类别(0-9)
    # alpha=0.3: 透明度30%，让数据点可见
    # cmap=cmap: 使用10种颜色表示10个类别
    # levels=np.arange(11)-0.5: 设置10个类别的边界
    ax.contourf(xx, yy, grid_preds, alpha=0.3, cmap=cmap, levels=np.arange(11)-0.5)
    
    # 在决策区域上绘制真实数据点
    for digit in range(10):
        mask = labels == digit
        ax.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                  c=DIGIT_COLORS[digit], label=f'{digit}', 
                  alpha=0.8, s=30, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('主成分 1', fontsize=12)
    ax.set_ylabel('主成分 2', fontsize=12)
    ax.set_title('PCA空间中的决策边界', fontsize=14, fontweight='bold')
    ax.legend(title='数字类别', loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    # set_xlim/ylim(): 设置坐标轴范围
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # ========== 子图2: 预测置信度热力图 ==========
    ax = axes[1]
    # 绘制置信度等高线填充图
    # levels=20: 使用20个等高线级别
    # cmap='RdYlGn': 红-黄-绿颜色映射
    contour = ax.contourf(xx, yy, grid_probs, levels=20, cmap='RdYlGn', alpha=0.8, vmin=0, vmax=1)
    plt.colorbar(contour, ax=ax, label='预测置信度')
    
    # 绘制数据点(灰色小点)
    ax.scatter(data_2d[:, 0], data_2d[:, 1], 
              c='black', alpha=0.3, s=10)
    
    ax.set_xlabel('主成分 1', fontsize=12)
    ax.set_ylabel('主成分 2', fontsize=12)
    ax.set_title('决策边界附近的置信度分布', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('../../docs/decision_boundaries.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("已保存决策边界图: docs/decision_boundaries.png")


def visualize_hidden_features(model):
    """
    可视化隐藏层特征
    展示神经网络第一层(ReLU后)的特征分布和激活模式
    Args:
        model: 训练好的神经网络模型
    """
    print("正在生成隐藏层特征可视化...")
    
    # 加载测试数据
    test_data = mnist.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    test_load = DataLoader(test_data, batch_size=500, shuffle=False)
    data, labels = next(iter(test_load))
    
    # ========== 提取隐藏层特征 ==========
    with torch.no_grad():
        # 展平输入数据
        x = data.view(-1, 28*28)
        # model.layer1(x): 第一层线性变换
        # torch.relu(): ReLU激活
        # hidden形状: (500, 256)，500个样本的256维隐藏特征
        hidden = torch.relu(model.layer1(x)).numpy()
    
    labels_np = labels.numpy()
    
    # ========== 对隐藏层特征进行PCA ==========
    pca_hidden = PCA(n_components=2)
    # 将256维隐藏特征降到2维
    hidden_2d = pca_hidden.fit_transform(hidden)
    
    # ========== 创建可视化图形 ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ========== 子图1: 隐藏空间中的类别分布 ==========
    ax = axes[0]
    for digit in range(10):
        mask = labels_np == digit
        ax.scatter(hidden_2d[mask, 0], hidden_2d[mask, 1], 
                  c=DIGIT_COLORS[digit], label=f'{digit}', 
                  alpha=0.7, s=30, edgecolors='none')
    
    ax.set_xlabel(f'隐藏层主成分 1 ({pca_hidden.explained_variance_ratio_[0]:.1%} 方差)', fontsize=11)
    ax.set_ylabel(f'隐藏层主成分 2 ({pca_hidden.explained_variance_ratio_[1]:.1%} 方差)', fontsize=11)
    ax.set_title('隐藏层特征空间的类别分布', fontsize=13, fontweight='bold')
    ax.legend(title='数字', loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # ========== 子图2: 隐藏层特征的平均值 ==========
    ax = axes[1]
    mean_features = []
    # 计算每个数字类别的平均隐藏特征
    for digit in range(10):
        mask = labels_np == digit
        # hidden[mask]: 提取属于当前类别的所有隐藏特征
        # .mean(axis=0): 沿样本维度求平均，得到256维的平均特征
        mean_feat = hidden[mask].mean(axis=0)
        mean_features.append(mean_feat)
    
    mean_features = np.array(mean_features)
    
    # imshow(): 显示热力图
    # aspect='auto': 自动调整宽高比
    # cmap='RdYlBu_r': 红-黄-蓝反向色图
    # interpolation='nearest': 最近邻插值，保持像素清晰
    im = ax.imshow(mean_features, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='平均激活值')
    ax.set_xlabel('隐藏层神经元索引', fontsize=11)
    ax.set_ylabel('数字类别', fontsize=11)
    ax.set_title('各类别在隐藏层的平均激活模式', fontsize=13, fontweight='bold')
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'{i}' for i in range(10)])
    
    plt.tight_layout()
    plt.savefig('../../docs/hidden_features.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("已保存隐藏层特征图: ../../docs/hidden_features.png")


def visualize_boundary_examples(model):
    """
    可视化边界附近的样本
    展示高置信度、低置信度和错误分类的样本图像
    Args:
        model: 训练好的神经网络模型
    """
    print("正在生成边界样本可视化...")
    
    # 加载测试数据
    test_data = mnist.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    test_load = DataLoader(test_data, batch_size=1000, shuffle=False)
    data, labels = next(iter(test_load))
    
    # ========== 获取预测和置信度 ==========
    with torch.no_grad():
        outputs = model(data)
        # torch.softmax(): 计算概率分布
        probs = torch.softmax(outputs, dim=1)
        # 获取预测类别
        predictions = outputs.argmax(dim=1)
        # 获取每个预测的置信度(最大概率)
        confidences = probs.max(dim=1)[0]
        
        # 创建布尔掩码，标记不同类型的样本
        # high_conf_mask: 置信度>95%的样本(远离决策边界)
        high_conf_mask = confidences > 0.95
        # low_conf_mask: 置信度<60%的样本(靠近决策边界)
        low_conf_mask = confidences < 0.6
        # wrong_mask: 错误分类的样本(跨越决策边界)
        wrong_mask = predictions != labels
    
    # ========== 创建可视化图形 ==========
    fig = plt.figure(figsize=(16, 10))
    # add_gridspec(): 创建自定义网格布局
    # 3, 10: 3行10列
    # hspace=0.4: 行间距
    # wspace=0.3: 列间距
    gs = fig.add_gridspec(3, 10, hspace=0.4, wspace=0.3)
    
    # ========== 第1行：高置信度正确分类样本 ==========
    # fig.text(): 在图形上添加文本
    # 0.5, 0.95: 文本位置(相对于图形，0-1之间)
    # ha='center': 水平居中对齐
    fig.text(0.5, 0.95, '决策边界可视化示例', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.92, '高置信度样本 (>95%) - 远离决策边界', ha='center', fontsize=12, color='green')
    
    # torch.where(): 返回满足条件的索引
    # high_conf_indices: 高置信度样本的索引
    high_conf_indices = torch.where(high_conf_mask)[0][:10]
    for i, idx in enumerate(high_conf_indices):
        # add_subplot(): 在网格中添加子图
        ax = fig.add_subplot(gs[0, i])
        # squeeze(): 移除大小为1的维度，将(1, 28, 28)转为(28, 28)
        img = data[idx].squeeze().numpy()
        # imshow(): 显示图像，cmap='gray'使用灰度颜色图
        ax.imshow(img, cmap='gray')
        # set_title(): 设置子图标题，显示真实标签、预测和置信度
        ax.set_title(f'真实:{labels[idx].item()}\n预测:{predictions[idx].item()}\n置信度:{confidences[idx].item():.2f}', 
                    fontsize=8)
        # axis('off'): 关闭坐标轴
        ax.axis('off')
    
    # ========== 第2行：低置信度样本 ==========
    fig.text(0.5, 0.62, '低置信度样本 (<60%) - 靠近决策边界', ha='center', fontsize=12, color='orange')
    
    low_conf_indices = torch.where(low_conf_mask)[0][:10]
    if len(low_conf_indices) > 0:
        for i, idx in enumerate(low_conf_indices):
            ax = fig.add_subplot(gs[1, i])
            img = data[idx].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.set_title(f'真实:{labels[idx].item()}\n预测:{predictions[idx].item()}\n置信度:{confidences[idx].item():.2f}', 
                        fontsize=8)
            ax.axis('off')
    
    # ========== 第3行：错误分类样本 ==========
    fig.text(0.5, 0.32, '错误分类样本 - 跨越决策边界', ha='center', fontsize=12, color='red')
    
    wrong_indices = torch.where(wrong_mask)[0][:10]
    for i, idx in enumerate(wrong_indices):
        ax = fig.add_subplot(gs[2, i])
        img = data[idx].squeeze().numpy()
        true_label = labels[idx].item()
        pred_label = predictions[idx].item()
        conf = confidences[idx].item()
        ax.set_title(f'真实:{true_label} 预测:{pred_label}\n置信度:{conf:.2f}', fontsize=8)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.savefig('../../docs/boundary_examples.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("已保存边界样本图: ../../docs/boundary_examples.png")


def visualize_weight_patterns(model):
    """
    可视化学习到的权重模式
    展示神经网络的权重矩阵，理解模型学习到的特征
    Args:
        model: 训练好的神经网络模型
    """
    print("正在生成权重模式可视化...")
    
    # ========== 获取权重矩阵 ==========
    # model.layer1.weight: 第一层权重矩阵，形状(256, 784)
    # .data: 获取张量数据(不含梯度信息)
    # .numpy(): 转换为numpy数组
    W1 = model.layer1.weight.data.numpy()  # (256, 784)
    W2 = model.layer2.weight.data.numpy()  # (10, 256)
    
    # ========== 创建可视化图形 ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # ========== 子图1: Layer1 权重矩阵 ==========
    ax = axes[0, 0]
    # imshow(): 显示权重矩阵热力图
    # aspect='auto': 自动调整宽高比
    # cmap='RdBu_r': 红-蓝反向色图，红色为正，蓝色为负
    im1 = ax.imshow(W1, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(im1, ax=ax, label='权重值')
    ax.set_xlabel('输入像素索引 (784)', fontsize=11)
    ax.set_ylabel('隐藏层神经元 (256)', fontsize=11)
    ax.set_title('Layer1 权重矩阵 W₁', fontsize=13, fontweight='bold')
    
    # ========== 子图2: Layer2 权重矩阵 ==========
    ax = axes[0, 1]
    im2 = ax.imshow(W2, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(im2, ax=ax, label='权重值')
    ax.set_xlabel('隐藏层神经元 (256)', fontsize=11)
    ax.set_ylabel('输出类别 (10)', fontsize=11)
    ax.set_title('Layer2 权重矩阵 W₂', fontsize=13, fontweight='bold')
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'{i}' for i in range(10)])
    
    # ========== 子图3: Layer1 权重重塑为图像 ==========
    ax = axes[1, 0]
    # 展示前16个神经元的权重模式
    n_show = 16
    # np.ceil(): 向上取整，计算网格大小(4x4=16)
    grid_size = int(np.ceil(np.sqrt(n_show)))
    
    for i in range(n_show):
        # 计算子图位置(行、列索引)
        row = i // grid_size
        col = i % grid_size
        # reshape(28, 28): 将784维权重向量重塑为28x28图像
        weight_img = W1[i].reshape(28, 28)
        
        # 归一化到[0, 1]范围用于显示
        # (weight_img - weight_img.min()): 减去最小值使最小值为0
        # / (weight_img.max() - weight_img.min()): 除以范围使最大值为1
        # + 1e-8: 防止除以0
        weight_img_norm = (weight_img - weight_img.min()) / (weight_img.max() - weight_img.min() + 1e-8)
        
        # 在网格中添加子图
        ax_sub = fig.add_subplot(grid_size, grid_size, i+1)
        ax_sub.imshow(weight_img_norm, cmap='RdBu_r')
        ax_sub.set_title(f'神经元 {i}', fontsize=8)
        ax_sub.axis('off')
    
    # 关闭原始子图的坐标轴，因为我们用add_subplot创建了新的子图
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Layer1 前16个神经元的权重模式', fontsize=13, fontweight='bold', pad=10)
    
    # ========== 子图4: Layer2 权重条形图 ==========
    ax = axes[1, 1]
    # 创建x轴位置(0到255)
    x_pos = np.arange(256)
    # 条形图宽度
    width = 0.08
    
    # 为每个数字绘制权重条形图
    for digit in range(10):
        # offset: 条形图偏移量，避免重叠
        # (digit - 4.5): 居中排列
        offset = (digit - 4.5) * width
        # ax.bar(): 绘制条形图
        # x_pos + offset: x轴位置
        # W2[digit]: 该数字对256个隐藏神经元的权重
        ax.bar(x_pos + offset, W2[digit], width, 
               label=f'数字 {digit}', alpha=0.7, color=DIGIT_COLORS[digit])
    
    ax.set_xlabel('隐藏层神经元索引', fontsize=11)
    ax.set_ylabel('权重值', fontsize=11)
    ax.set_title('Layer2 各类别对隐藏神经元的权重', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim([0, 256])
    
    plt.tight_layout()
    plt.savefig('../../docs/weight_patterns.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("已保存权重模式图: ../../docs/weight_patterns.png")


def main():
    """
    主函数 - 执行完整的可视化流程
    """
    print("="*60)
    print("MNIST 决策边界可视化")
    print("="*60)
    
    # ========== 步骤1: 加载或训练模型 ==========
    # load_or_train_model(): 尝试加载预训练模型，如果不存在则训练
    model, losses, accuracies = load_or_train_model()
    # model.eval(): 设置模型为评估模式
    # 在评估模式下，Dropout和BatchNorm层的行为会改变(本模型没有这些层，但养成好习惯)
    model.eval()
    
    # ========== 步骤2: 可视化训练过程 ==========
    # 如果有训练历史(losses不为None)，则绘制训练曲线
    if losses and accuracies:
        visualize_training_process(losses, accuracies)
    
    # ========== 步骤3: PCA投影可视化 ==========
    # visualize_pca_projection(): 在PCA降维空间中可视化数据分布
    # 返回降维后的数据、标签、预测和PCA模型
    data_2d, labels, predictions, pca = visualize_pca_projection(model)
    
    # ========== 步骤4: 决策边界可视化 ==========
    # visualize_decision_regions(): 在PCA空间中绘制决策边界
    visualize_decision_regions(model, pca, data_2d, labels)
    
    # ========== 步骤5: 隐藏层特征可视化 ==========
    # visualize_hidden_features(): 可视化神经网络隐藏层的特征分布
    visualize_hidden_features(model)
    
    # ========== 步骤6: 边界样本可视化 ==========
    # visualize_boundary_examples(): 展示不同置信度的样本图像
    visualize_boundary_examples(model)
    
    # ========== 步骤7: 权重模式可视化 ==========
    # visualize_weight_patterns(): 可视化学习到的权重矩阵
    visualize_weight_patterns(model)
    
    print("\n" + "="*60)
    print("所有可视化完成！图片保存在 docs/ 文件夹中：")
    print("  - training_process.png    (训练过程)")
    print("  - pca_visualization.png   (PCA投影)")
    print("  - decision_boundaries.png (决策边界)")
    print("  - hidden_features.png     (隐藏层特征)")
    print("  - boundary_examples.png   (边界样本)")
    print("  - weight_patterns.png     (权重模式)")
    print("="*60)


if __name__ == '__main__':
    main()
