import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def make_data(num):
    """生成三分类数据"""
    np.random.seed(0)
    
    # 红色：中心点
    red, _ = make_blobs(n_samples=num,
                       centers=[[0,0]],
                       cluster_std=0.15)
    
    # 绿色：环形分布
    green, _ = make_circles(n_samples=num,
                           noise=0.02,
                           factor=0.7)
    
    # 蓝色：四个角落
    blue, _ = make_blobs(n_samples=num,
                        centers=[[-1.2,-1.2],
                                [-1.2,1.2],
                                [1.2,-1.2],
                                [1.2,1.2]],
                        cluster_std=0.2)
    
    return green, blue, red


class Network(nn.Module):
    """神经网络模型"""
    
    def __init__(self, n_in, n_hidden, n_out):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(n_in, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_out)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        return self.layer2(x)


def prepare_data(green, blue, red):
    """准备数据"""
    green_tensor = torch.FloatTensor(green)
    blue_tensor = torch.FloatTensor(blue)
    red_tensor = torch.FloatTensor(red)
    
    data = torch.cat((green_tensor, blue_tensor, red_tensor), dim=0)
    label = torch.LongTensor([0] * len(green) + [1] * len(blue) + [2] * len(red))
    
    return data, label


def train_with_history(model, data, label, n_epochs=2000, lr=0.01, snapshot_interval=50):
    """训练模型并记录历史"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'loss': [],
        'accuracy': [],
        'weights_layer1': [],
        'weights_layer2': [],
        'bias_layer1': [],
        'bias_layer2': [],
        'gradients_layer1': [],
        'gradients_layer2': [],
        'snapshots': [],
        'epochs': []
    }
    
    print("开始训练并记录历史...")
    
    for epoch in range(n_epochs):
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, label)
        
        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == label).float().mean().item()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 记录梯度（在更新前）
        grad_l1 = model.layer1.weight.grad.clone().numpy() if model.layer1.weight.grad is not None else np.zeros_like(model.layer1.weight.data.numpy())
        grad_l2 = model.layer2.weight.grad.clone().numpy() if model.layer2.weight.grad is not None else np.zeros_like(model.layer2.weight.data.numpy())
        
        # 更新参数
        optimizer.step()
        
        # 记录历史
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy)
        history['weights_layer1'].append(model.layer1.weight.data.clone().numpy())
        history['weights_layer2'].append(model.layer2.weight.data.clone().numpy())
        history['bias_layer1'].append(model.layer1.bias.data.clone().numpy())
        history['bias_layer2'].append(model.layer2.bias.data.clone().numpy())
        history['gradients_layer1'].append(grad_l1)
        history['gradients_layer2'].append(grad_l2)
        history['epochs'].append(epoch)
        
        # 定期保存快照
        if epoch % snapshot_interval == 0 or epoch == n_epochs - 1:
            history['snapshots'].append({
                'epoch': epoch,
                'loss': loss.item(),
                'accuracy': accuracy
            })
            if epoch % (snapshot_interval * 4) == 0:
                print(f"  Epoch {epoch:4d}: Loss = {loss.item():.4f}, Accuracy = {accuracy*100:.1f}%")
    
    return history


def get_decision_boundary(model, x_range=(-2, 2), y_range=(-2, 2), resolution=100):
    """获取决策边界"""
    xx = np.linspace(x_range[0], x_range[1], resolution)
    yy = np.linspace(y_range[0], y_range[1], resolution)
    XX, YY = np.meshgrid(xx, yy)
    
    grid = np.c_[XX.ravel(), YY.ravel()]
    grid_tensor = torch.FloatTensor(grid)
    
    with torch.no_grad():
        outputs = model(grid_tensor)
        _, predicted = torch.max(outputs, 1)
    
    Z = predicted.numpy().reshape(XX.shape)
    return XX, YY, Z


def plot_training_process_animation(green, blue, red, data, label, n_hidden=6):
    """创建训练过程的动画"""
    print(f"\n创建训练过程动画 (n_hidden={n_hidden})...")
    
    # 初始化模型
    model = Network(2, n_hidden, 3)
    history = train_with_history(model, data, label, n_epochs=1500, lr=0.01, snapshot_interval=25)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 子图布局
    ax_loss = fig.add_subplot(gs[0, :])
    ax_boundary = fig.add_subplot(gs[1:, :2])
    ax_weights = fig.add_subplot(gs[1, 2])
    ax_gradients = fig.add_subplot(gs[2, 2])
    
    colors = ['lightgreen', 'lightblue', 'lightcoral']
    cmap = ListedColormap(colors)
    
    # 准备网格
    xx = np.linspace(-2, 2, 80)
    yy = np.linspace(-2, 2, 80)
    XX, YY = np.meshgrid(xx, yy)
    
    def init():
        ax_loss.clear()
        ax_boundary.clear()
        ax_weights.clear()
        ax_gradients.clear()
        return []
    
    def update(frame):
        epoch = frame * 5  # 每5个epoch一帧
        if epoch >= len(history['loss']):
            epoch = len(history['loss']) - 1
        
        # 1. 损失曲线
        ax_loss.clear()
        epochs_to_show = history['epochs'][:epoch+1]
        loss_to_show = history['loss'][:epoch+1]
        acc_to_show = [a*100 for a in history['accuracy'][:epoch+1]]
        
        ax_loss.plot(epochs_to_show, loss_to_show, 'b-', linewidth=2, label='Loss', alpha=0.8)
        ax_loss_twin = ax_loss.twinx()
        ax_loss_twin.plot(epochs_to_show, acc_to_show, 'r-', linewidth=2, label='Accuracy (%)', alpha=0.8)
        
        ax_loss.axvline(x=epoch, color='green', linestyle='--', alpha=0.5)
        ax_loss.set_xlabel('Training Epoch', fontsize=11)
        ax_loss.set_ylabel('Loss', color='blue', fontsize=11)
        ax_loss_twin.set_ylabel('Accuracy (%)', color='red', fontsize=11)
        ax_loss.set_title(f'Training Progress - Epoch {epoch}\nLoss: {history["loss"][epoch]:.4f}, Accuracy: {history["accuracy"][epoch]*100:.1f}%', 
                         fontsize=13, fontweight='bold')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_yscale('log')
        ax_loss.legend(loc='upper left')
        ax_loss_twin.legend(loc='upper right')
        
        # 2. 决策边界演化
        ax_boundary.clear()
        
        # 使用当前权重创建临时模型
        temp_model = Network(2, n_hidden, 3)
        temp_model.layer1.weight.data = torch.FloatTensor(history['weights_layer1'][epoch])
        temp_model.layer1.bias.data = torch.FloatTensor(history['bias_layer1'][epoch])
        temp_model.layer2.weight.data = torch.FloatTensor(history['weights_layer2'][epoch])
        temp_model.layer2.bias.data = torch.FloatTensor(history['bias_layer2'][epoch])
        
        grid = np.c_[XX.ravel(), YY.ravel()]
        grid_tensor = torch.FloatTensor(grid)
        with torch.no_grad():
            outputs = temp_model(grid_tensor)
            _, predicted = torch.max(outputs, 1)
        Z = predicted.numpy().reshape(XX.shape)
        
        ax_boundary.contourf(XX, YY, Z, levels=2, colors=colors, alpha=0.6)
        ax_boundary.contour(XX, YY, Z, levels=[0.5, 1.5], colors='black', linewidths=2.5)
        ax_boundary.scatter(green[:, 0], green[:, 1], c='green', s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax_boundary.scatter(blue[:, 0], blue[:, 1], c='blue', s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax_boundary.scatter(red[:, 0], red[:, 1], c='red', s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
        ax_boundary.set_title('Decision Boundary Evolution', fontsize=13, fontweight='bold')
        ax_boundary.set_xlim(-2, 2)
        ax_boundary.set_ylim(-2, 2)
        ax_boundary.set_aspect('equal')
        ax_boundary.grid(True, alpha=0.3)
        
        # 3. 权重变化
        ax_weights.clear()
        w1_history = np.array([w.flatten() for w in history['weights_layer1'][:epoch+1]])
        if w1_history.shape[0] > 0 and w1_history.shape[1] > 0:
            for i in range(min(w1_history.shape[1], 6)):
                ax_weights.plot(history['epochs'][:epoch+1], w1_history[:, i], 
                              label=f'W{i+1}', alpha=0.7, linewidth=1.5)
        ax_weights.set_xlabel('Epoch', fontsize=10)
        ax_weights.set_ylabel('Weight Value', fontsize=10)
        ax_weights.set_title('Layer 1 Weights', fontsize=11, fontweight='bold')
        ax_weights.grid(True, alpha=0.3)
        if w1_history.shape[0] > 0 and w1_history.shape[1] <= 6:
            ax_weights.legend(fontsize=8, loc='best')
        
        # 4. 梯度变化
        ax_gradients.clear()
        if epoch > 0:
            grad_l1_history = np.array([g.flatten() for g in history['gradients_layer1'][1:epoch+1]])
            grad_norms = np.linalg.norm(grad_l1_history, axis=1)
            ax_gradients.plot(history['epochs'][1:epoch+1], grad_norms, 'purple', linewidth=2)
            ax_gradients.set_xlabel('Epoch', fontsize=10)
            ax_gradients.set_ylabel('Gradient Norm', fontsize=10)
            ax_gradients.set_title('Gradient Magnitude', fontsize=11, fontweight='bold')
            ax_gradients.grid(True, alpha=0.3)
            ax_gradients.set_yscale('log')
        
        return []
    
    # 创建动画
    n_frames = len(history['loss']) // 5
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, 
                        blit=False, interval=50, repeat=True)
    
    # 保存动画
    try:
        anim.save('gradient_descent_training.gif', writer='pillow', fps=20, dpi=100)
        print("训练动画已保存: gradient_descent_training.gif")
    except Exception as e:
        print(f"保存GIF失败: {e}")
        plt.savefig('gradient_descent_final.png', dpi=150, bbox_inches='tight')
        print("已保存最终帧: gradient_descent_final.png")
    
    plt.show()
    
    return history


def plot_detailed_training_analysis(history, green, blue, red, data, label):
    """绘制详细的训练分析图"""
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 损失和准确率曲线
    ax1 = plt.subplot(3, 3, 1)
    epochs = history['epochs']
    ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Loss')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', color='blue', fontsize=11)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Training Loss Curve', fontsize=12, fontweight='bold')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, [a*100 for a in history['accuracy']], 'r-', linewidth=2, label='Accuracy')
    ax1_twin.set_ylabel('Accuracy (%)', color='red', fontsize=11)
    ax1_twin.legend(loc='center right')
    
    # 2. 学习速度（损失下降速度）
    ax2 = plt.subplot(3, 3, 2)
    if len(history['loss']) > 1:
        loss_diff = np.diff(history['loss'])
        ax2.plot(epochs[1:], -loss_diff, 'g-', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss Decrease per Epoch', fontsize=11)
        ax2.set_title('Learning Speed', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # 3. 梯度范数变化
    ax3 = plt.subplot(3, 3, 3)
    grad_norms_l1 = [np.linalg.norm(g) for g in history['gradients_layer1']]
    grad_norms_l2 = [np.linalg.norm(g) for g in history['gradients_layer2']]
    ax3.plot(epochs, grad_norms_l1, 'purple', linewidth=2, label='Layer 1', alpha=0.8)
    ax3.plot(epochs, grad_norms_l2, 'orange', linewidth=2, label='Layer 2', alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Gradient Norm', fontsize=11)
    ax3.set_yscale('log')
    ax3.set_title('Gradient Magnitude', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4-6. 不同时期的决策边界
    snapshot_epochs = [0, len(history['epochs'])//4, len(history['epochs'])//2, 
                       3*len(history['epochs'])//4, len(history['epochs'])-1]
    
    colors = ['lightgreen', 'lightblue', 'lightcoral']
    
    for idx, snap_idx in enumerate(snapshot_epochs[:6]):
        if idx >= 6:
            break
        ax = plt.subplot(3, 3, idx + 4)
        
        # 创建临时模型
        temp_model = Network(2, len(history['weights_layer1'][0]), 3)
        temp_model.layer1.weight.data = torch.FloatTensor(history['weights_layer1'][snap_idx])
        temp_model.layer1.bias.data = torch.FloatTensor(history['bias_layer1'][snap_idx])
        temp_model.layer2.weight.data = torch.FloatTensor(history['weights_layer2'][snap_idx])
        temp_model.layer2.bias.data = torch.FloatTensor(history['bias_layer2'][snap_idx])
        
        XX, YY, Z = get_decision_boundary(temp_model)
        
        ax.contourf(XX, YY, Z, levels=2, colors=colors, alpha=0.6)
        ax.contour(XX, YY, Z, levels=[0.5, 1.5], colors='black', linewidths=2)
        ax.scatter(green[:, 0], green[:, 1], c='green', s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
        ax.scatter(blue[:, 0], blue[:, 1], c='blue', s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
        ax.scatter(red[:, 0], red[:, 1], c='red', s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
        
        epoch = history['epochs'][snap_idx]
        loss = history['loss'][snap_idx]
        acc = history['accuracy'][snap_idx]
        
        ax.set_title(f'Epoch {epoch}\nLoss: {loss:.4f}, Acc: {acc*100:.1f}%', 
                    fontsize=11, fontweight='bold')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Gradient Descent Training Process Analysis\n' + 
                'Showing how decision boundaries evolve during training',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('gradient_descent_analysis.png', dpi=150, bbox_inches='tight')
    print("详细分析图已保存: gradient_descent_analysis.png")
    plt.show()


def plot_weight_evolution_heatmap(history):
    """绘制权重演化的热力图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 选择几个关键时间点
    key_epochs = [0, len(history['epochs'])//10, len(history['epochs'])//5, 
                  len(history['epochs'])//2, 3*len(history['epochs'])//4, len(history['epochs'])-1]
    
    for idx, epoch_idx in enumerate(key_epochs):
        ax = axes[idx // 3, idx % 3]
        
        weights = history['weights_layer1'][epoch_idx]
        im = ax.imshow(weights, cmap='RdBu_r', aspect='auto', vmin=-8, vmax=8)
        
        epoch = history['epochs'][epoch_idx]
        ax.set_title(f'Epoch {epoch}', fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['W1', 'W2'])
        ax.set_yticks(range(len(weights)))
        ax.set_yticklabels([f'H{i+1}' for i in range(len(weights))])
        
        # 添加数值
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                text = ax.text(j, i, f'{weights[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Layer 1 Weight Evolution During Training', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('weight_evolution_heatmap.png', dpi=150, bbox_inches='tight')
    print("权重演化热力图已保存: weight_evolution_heatmap.png")
    plt.show()


def analyze_gradient_flow(history):
    """分析梯度流"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = history['epochs']
    
    # 1. 每层梯度范数
    ax1 = axes[0, 0]
    grad_norms_l1 = [np.linalg.norm(g) for g in history['gradients_layer1']]
    grad_norms_l2 = [np.linalg.norm(g) for g in history['gradients_layer2']]
    ax1.semilogy(epochs, grad_norms_l1, 'purple', linewidth=2, label='Layer 1', alpha=0.8)
    ax1.semilogy(epochs, grad_norms_l2, 'orange', linewidth=2, label='Layer 2', alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Gradient Norm (log scale)', fontsize=11)
    ax1.set_title('Gradient Magnitude per Layer', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 梯度方差
    ax2 = axes[0, 1]
    if len(history['gradients_layer1']) > 1:
        grad_vars_l1 = [np.var(g) for g in history['gradients_layer1']]
        grad_vars_l2 = [np.var(g) for g in history['gradients_layer2']]
        ax2.semilogy(epochs, grad_vars_l1, 'purple', linewidth=2, label='Layer 1', alpha=0.8)
        ax2.semilogy(epochs, grad_vars_l2, 'orange', linewidth=2, label='Layer 2', alpha=0.8)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Gradient Variance', fontsize=11)
        ax2.set_title('Gradient Variance', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 权重更新幅度
    ax3 = axes[1, 0]
    if len(history['weights_layer1']) > 1:
        weight_changes = [np.linalg.norm(history['weights_layer1'][i] - history['weights_layer1'][i-1]) 
                         for i in range(1, len(history['weights_layer1']))]
        ax3.semilogy(epochs[1:], weight_changes, 'blue', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Weight Update Magnitude', fontsize=11)
        ax3.set_title('Weight Update Size per Epoch', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # 4. 损失下降率
    ax4 = axes[1, 1]
    if len(history['loss']) > 10:
        # 计算移动平均的损失下降
        window = 10
        loss_smooth = np.convolve(history['loss'], np.ones(window)/window, mode='valid')
        loss_derivative = np.diff(loss_smooth)
        # 修复维度匹配
        plot_epochs = epochs[window:window+len(loss_derivative)]
        ax4.plot(plot_epochs, -loss_derivative, 'green', linewidth=2, alpha=0.7)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Loss Decrease Rate', fontsize=11)
        ax4.set_title('Training Speed (Smoothed)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Gradient Flow and Training Dynamics Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('gradient_flow_analysis.png', dpi=150, bbox_inches='tight')
    print("梯度流分析图已保存: gradient_flow_analysis.png")
    plt.show()


def main():
    """主函数"""
    print("="*70)
    print("梯度下降训练过程可视化分析")
    print("="*70)
    
    # 生成数据
    print("\n1. 生成数据...")
    green, blue, red = make_data(100)
    data, label = prepare_data(green, blue, red)
    
    # 训练并记录历史
    print("\n2. 训练模型并记录梯度下降过程...")
    model = Network(2, 6, 3)
    history = train_with_history(model, data, label, n_epochs=1500, lr=0.01)
    
    # 绘制详细分析
    print("\n3. 生成详细训练分析图...")
    plot_detailed_training_analysis(history, green, blue, red, data, label)
    
    # 权重演化热力图
    print("\n4. 生成权重演化热力图...")
    plot_weight_evolution_heatmap(history)
    
    # 梯度流分析
    print("\n5. 生成梯度流分析...")
    analyze_gradient_flow(history)
    
    # 创建动画
    print("\n6. 创建训练过程动画...")
    plot_training_process_animation(green, blue, red, data, label, n_hidden=6)
    
    print("\n" + "="*70)
    print("所有可视化完成！")
    print("="*70)
    print("\n生成的文件:")
    print("  - gradient_descent_analysis.png - 训练过程详细分析")
    print("  - weight_evolution_heatmap.png - 权重演化热力图")
    print("  - gradient_flow_analysis.png - 梯度流动态分析")
    print("  - gradient_descent_training.gif - 训练过程动画")
    print("="*70)


if __name__ == '__main__':
    main()
