"""
MNIST决策边界可视化脚本
使用PCA降维来可视化神经网络的决策边界
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch import nn
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义颜色方案
COLORS = {
    'red': '#FF6B6B',
    'green': '#51CF66', 
    'blue': '#339AF0',
    'yellow': '#FFD93D',
    'purple': '#9C36B5',
    'orange': '#FF922B',
    'cyan': '#22D3EE',
    'pink': '#F472B6',
    'brown': '#A16207',
    'gray': '#6B7280'
}

DIGIT_COLORS = ['#FF6B6B', '#51CF66', '#339AF0', '#FFD93D', '#9C36B5', 
                '#FF922B', '#22D3EE', '#F472B6', '#A16207', '#6B7280']


class Network(nn.Module):
    """神经网络定义（与number.py相同）"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 256)
        self.layer2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layer1(x)
        x = torch.relu(x)
        return self.layer2(x)


def train_model():
    """训练模型并返回"""
    print("正在训练模型...")
    train_data = mnist.MNIST(root='./data', train=True, download=True, transform=ToTensor())
    train_load = DataLoader(train_data, shuffle=True, batch_size=128)
    
    model = Network()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    accuracies = []
    
    for epoch in range(10):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, label) in enumerate(train_load):
            output = model(data)
            loss = criterion(output, label)
            
            # 计算准确率
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/10 | Batch {batch_idx}/{len(train_load)} | Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_load)
        accuracy = correct / total
        losses.append(avg_loss)
        accuracies.append(accuracy)
        print(f"Epoch {epoch+1} 完成 - 平均损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
    
    # 保存模型
    import os
    os.makedirs('../../docs', exist_ok=True)
    torch.save(model.state_dict(), '../../docs/network_visualization.digit')
    return model, losses, accuracies


def load_or_train_model():
    """加载已有模型或训练新模型"""
    model = Network()
    try:
        # 尝试加载预训练模型
        model.load_state_dict(torch.load('../../docs/network_visualization.digit'))
        print("已加载预训练模型")
        return model, None, None
    except:
        print("未找到预训练模型，开始训练...")
        return train_model()


def visualize_training_process(losses, accuracies):
    """可视化训练过程"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(range(1, len(losses)+1), losses, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax1.set_ylabel('损失值 (Loss)', fontsize=12)
    ax1.set_title('训练损失变化', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, len(losses)+1))
    
    # 准确率曲线
    ax2.plot(range(1, len(accuracies)+1), [a*100 for a in accuracies], 'g-s', linewidth=2, markersize=6)
    ax2.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.set_title('训练准确率变化', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, len(accuracies)+1))
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('../../docs/training_process.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("已保存训练过程图: docs/training_process.png")


def visualize_pca_projection(model):
    """使用PCA降维可视化数据分布"""
    print("正在生成PCA投影可视化...")
    
    # 加载测试数据
    test_data = mnist.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    test_load = DataLoader(test_data, batch_size=1000, shuffle=False)
    
    # 获取一批数据
    data, labels = next(iter(test_load))
    
    # 展平数据
    data_flat = data.view(-1, 28*28).numpy()
    labels_np = labels.numpy()
    
    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_flat)
    
    # 获取预测
    with torch.no_grad():
        outputs = model(data)
        predictions = outputs.argmax(dim=1).numpy()
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. 真实标签分布
    ax = axes[0, 0]
    for digit in range(10):
        mask = labels_np == digit
        ax.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                  c=DIGIT_COLORS[digit], label=f'{digit}', 
                  alpha=0.6, s=20, edgecolors='none')
    ax.set_xlabel(f'主成分 1 ({pca.explained_variance_ratio_[0]:.1%} 方差)', fontsize=11)
    ax.set_ylabel(f'主成分 2 ({pca.explained_variance_ratio_[1]:.1%} 方差)', fontsize=11)
    ax.set_title('PCA投影 - 真实标签分布', fontsize=13, fontweight='bold')
    ax.legend(title='数字', loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. 预测标签分布
    ax = axes[0, 1]
    for digit in range(10):
        mask = predictions == digit
        if mask.sum() > 0:
            ax.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                      c=DIGIT_COLORS[digit], label=f'{digit}', 
                      alpha=0.6, s=20, edgecolors='none')
    ax.set_xlabel(f'主成分 1 ({pca.explained_variance_ratio_[0]:.1%} 方差)', fontsize=11)
    ax.set_ylabel(f'主成分 2 ({pca.explained_variance_ratio_[1]:.1%} 方差)', fontsize=11)
    ax.set_title('PCA投影 - 预测标签分布', fontsize=13, fontweight='bold')
    ax.legend(title='预测', loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 3. 正确与错误分类
    ax = axes[1, 0]
    correct = predictions == labels_np
    wrong = ~correct
    
    ax.scatter(data_2d[correct, 0], data_2d[correct, 1], 
              c=COLORS['green'], label='正确分类', alpha=0.5, s=15)
    ax.scatter(data_2d[wrong, 0], data_2d[wrong, 1], 
              c=COLORS['red'], label='错误分类', alpha=0.8, s=50, marker='x', linewidth=2)
    ax.set_xlabel(f'主成分 1 ({pca.explained_variance_ratio_[0]:.1%} 方差)', fontsize=11)
    ax.set_ylabel(f'主成分 2 ({pca.explained_variance_ratio_[1]:.1%} 方差)', fontsize=11)
    ax.set_title('分类结果可视化', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. 预测置信度
    ax = axes[1, 1]
    with torch.no_grad():
        probs = torch.softmax(model(data), dim=1)
        confidences = probs.max(dim=1)[0].numpy()
    
    scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], 
                        c=confidences, cmap='RdYlGn', 
                        alpha=0.6, s=25, vmin=0, vmax=1)
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
    """在PCA空间中绘制决策边界"""
    print("正在生成决策边界可视化...")
    
    # 创建网格点
    x_min, x_max = data_2d[:, 0].min() - 1, data_2d[:, 0].max() + 1
    y_min, y_max = data_2d[:, 1].min() - 1, data_2d[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 将网格点从2D PCA空间映射回784D原始空间
    # 这需要使用PCA的逆变换，但我们只需要近似
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 使用PCA的均值和主成分重构近似原始空间点
    # 注意：这是一个近似，因为我们只保留了2个主成分
    grid_original = pca.inverse_transform(grid_points)
    grid_tensor = torch.FloatTensor(grid_original).view(-1, 1, 28, 28)
    
    # 预测网格点
    with torch.no_grad():
        outputs = model(grid_tensor)
        grid_preds = outputs.argmax(dim=1).numpy()
        grid_probs = torch.softmax(outputs, dim=1).max(dim=1)[0].numpy()
    
    grid_preds = grid_preds.reshape(xx.shape)
    grid_probs = grid_probs.reshape(xx.shape)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. 决策区域
    ax = axes[0]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(DIGIT_COLORS)
    
    ax.contourf(xx, yy, grid_preds, alpha=0.3, cmap=cmap, levels=np.arange(11)-0.5)
    
    # 绘制数据点
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
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # 2. 预测置信度热力图
    ax = axes[1]
    contour = ax.contourf(xx, yy, grid_probs, levels=20, cmap='RdYlGn', alpha=0.8, vmin=0, vmax=1)
    plt.colorbar(contour, ax=ax, label='预测置信度')
    
    # 绘制数据点（灰色）
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
    """可视化隐藏层特征"""
    print("正在生成隐藏层特征可视化...")
    
    # 加载测试数据
    test_data = mnist.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    test_load = DataLoader(test_data, batch_size=500, shuffle=False)
    data, labels = next(iter(test_load))
    
    # 提取隐藏层特征
    with torch.no_grad():
        x = data.view(-1, 28*28)
        hidden = torch.relu(model.layer1(x)).numpy()
    
    labels_np = labels.numpy()
    
    # 使用PCA降维隐藏层特征到2D
    pca_hidden = PCA(n_components=2)
    hidden_2d = pca_hidden.fit_transform(hidden)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. 隐藏空间中的类别分布
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
    
    # 2. 隐藏层特征的平均值
    ax = axes[1]
    mean_features = []
    for digit in range(10):
        mask = labels_np == digit
        mean_feat = hidden[mask].mean(axis=0)
        mean_features.append(mean_feat)
    
    mean_features = np.array(mean_features)
    
    # 显示热力图
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
    """可视化边界附近的样本"""
    print("正在生成边界样本可视化...")
    
    # 加载测试数据
    test_data = mnist.MNIST(root='./data', train=False, download=True, transform=ToTensor())
    test_load = DataLoader(test_data, batch_size=1000, shuffle=False)
    data, labels = next(iter(test_load))
    
    # 获取预测和置信度
    with torch.no_grad():
        outputs = model(data)
        probs = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)
        confidences = probs.max(dim=1)[0]
        
        # 找出高置信度和低置信度的样本
        high_conf_mask = confidences > 0.95
        low_conf_mask = confidences < 0.6
        wrong_mask = predictions != labels
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 10, hspace=0.4, wspace=0.3)
    
    # 高置信度正确分类样本
    fig.text(0.5, 0.95, '决策边界可视化示例', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.92, '高置信度样本 (>95%) - 远离决策边界', ha='center', fontsize=12, color='green')
    
    high_conf_indices = torch.where(high_conf_mask)[0][:10]
    for i, idx in enumerate(high_conf_indices):
        ax = fig.add_subplot(gs[0, i])
        img = data[idx].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'真实:{labels[idx].item()}\n预测:{predictions[idx].item()}\n置信度:{confidences[idx].item():.2f}', 
                    fontsize=8)
        ax.axis('off')
    
    # 低置信度样本（靠近决策边界）
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
    
    # 错误分类样本
    fig.text(0.5, 0.32, '错误分类样本 - 跨越决策边界', ha='center', fontsize=12, color='red')
    
    wrong_indices = torch.where(wrong_mask)[0][:10]
    for i, idx in enumerate(wrong_indices):
        ax = fig.add_subplot(gs[2, i])
        img = data[idx].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        true_label = labels[idx].item()
        pred_label = predictions[idx].item()
        conf = confidences[idx].item()
        ax.set_title(f'真实:{true_label} 预测:{pred_label}\n置信度:{conf:.2f}', fontsize=8)
        ax.axis('off')
    
    plt.savefig('../../docs/boundary_examples.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("已保存边界样本图: ../../docs/boundary_examples.png")


def visualize_weight_patterns(model):
    """可视化学习到的权重模式"""
    print("正在生成权重模式可视化...")
    
    # 获取权重
    W1 = model.layer1.weight.data.numpy()  # (256, 784)
    W2 = model.layer2.weight.data.numpy()  # (10, 256)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Layer1 权重分布
    ax = axes[0, 0]
    im1 = ax.imshow(W1, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(im1, ax=ax, label='权重值')
    ax.set_xlabel('输入像素索引 (784)', fontsize=11)
    ax.set_ylabel('隐藏层神经元 (256)', fontsize=11)
    ax.set_title('Layer1 权重矩阵 W₁', fontsize=13, fontweight='bold')
    
    # 2. Layer2 权重分布
    ax = axes[0, 1]
    im2 = ax.imshow(W2, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    plt.colorbar(im2, ax=ax, label='权重值')
    ax.set_xlabel('隐藏层神经元 (256)', fontsize=11)
    ax.set_ylabel('输出类别 (10)', fontsize=11)
    ax.set_title('Layer2 权重矩阵 W₂', fontsize=13, fontweight='bold')
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'{i}' for i in range(10)])
    
    # 3. 部分 Layer1 权重重塑为图像（展示前16个神经元）
    ax = axes[1, 0]
    n_show = 16
    grid_size = int(np.ceil(np.sqrt(n_show)))
    
    for i in range(n_show):
        row = i // grid_size
        col = i % grid_size
        weight_img = W1[i].reshape(28, 28)
        
        # 归一化到 [0, 1] 用于显示
        weight_img_norm = (weight_img - weight_img.min()) / (weight_img.max() - weight_img.min() + 1e-8)
        
        ax_sub = fig.add_subplot(grid_size, grid_size, i+1)
        ax_sub.imshow(weight_img_norm, cmap='RdBu_r')
        ax_sub.set_title(f'神经元 {i}', fontsize=8)
        ax_sub.axis('off')
    
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Layer1 前16个神经元的权重模式', fontsize=13, fontweight='bold', pad=10)
    
    # 4. Layer2 权重条形图
    ax = axes[1, 1]
    x_pos = np.arange(256)
    width = 0.08
    
    for digit in range(10):
        offset = (digit - 4.5) * width
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
    """主函数"""
    print("="*60)
    print("MNIST 决策边界可视化")
    print("="*60)
    
    # 1. 加载或训练模型
    model, losses, accuracies = load_or_train_model()
    model.eval()
    
    # 2. 可视化训练过程
    if losses and accuracies:
        visualize_training_process(losses, accuracies)
    
    # 3. PCA投影可视化
    data_2d, labels, predictions, pca = visualize_pca_projection(model)
    
    # 4. 决策边界可视化
    visualize_decision_regions(model, pca, data_2d, labels)
    
    # 5. 隐藏层特征可视化
    visualize_hidden_features(model)
    
    # 6. 边界样本可视化
    visualize_boundary_examples(model)
    
    # 7. 权重模式可视化
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
