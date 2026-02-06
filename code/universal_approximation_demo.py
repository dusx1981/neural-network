"""
神经网络万能近似定理可视化演示
=================================

本程序通过可视化展示神经网络万能近似定理的核心原理：
一个单隐藏层的前馈神经网络，只要使用非线性激活函数（如 sigmoid、tanh、ReLU），
并且隐藏层神经元足够多，就可以逼近任意连续函数。

核心原理：
1. 非线性激活函数（如sigmoid）可以逼近阶跃函数
2. 多个阶跃函数的组合可以逼近任意复杂函数
3. 神经元越多，逼近精度越高

作者: AI Assistant
日期: 2026-02-06
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ============================================================================
# 全局配置
# ============================================================================

# 设置中文字体支持，确保图表可以正确显示中文标题和标签
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 激活函数定义
# ============================================================================

def sigmoid(x):
    """
    Sigmoid激活函数：将输入映射到(0, 1)区间
    
    公式: sigmoid(x) = 1 / (1 + exp(-x))
    
    特点:
    - 输出范围: (0, 1)
    - 平滑可导，适合梯度下降
    - 当权重很大时，可逼近阶跃函数
    
    参数:
        x: 输入值或数组
    返回:
        sigmoid激活后的值
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    Tanh激活函数：双曲正切函数，将输入映射到(-1, 1)区间
    
    公式: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    特点:
    - 输出范围: (-1, 1)，均值为0，收敛更快
    - 是sigmoid的平移缩放版本
    - 在神经网络中广泛使用
    
    参数:
        x: 输入值或数组
    返回:
        tanh激活后的值
    """
    return np.tanh(x)


def relu(x):
    """
    ReLU激活函数：线性整流函数
    
    公式: relu(x) = max(0, x)
    
    特点:
    - 计算简单，收敛速度快
    - 解决梯度消失问题
    - 在正区间保持线性，适合深层网络
    
    参数:
        x: 输入值或数组
    返回:
        ReLU激活后的值
    """
    return np.maximum(0, x)


# ============================================================================
# 神经网络核心函数
# ============================================================================

def neural_network_output(x, weights_input, bias_hidden, weights_output, bias_output, activation='tanh'):
    """
    单隐藏层神经网络前向传播函数
    
    网络结构: 输入层 -> 隐藏层(带激活函数) -> 输出层(线性)
    
    数学表示:
        hidden = activation(x · W_in + b_in)
        output = hidden · W_out + b_out
    
    这是万能近似定理的核心实现：通过足够多的隐藏神经元和非线性激活，
    可以逼近任意连续函数。
    
    参数:
        x (numpy.ndarray): 输入数据，形状为 (n_samples,)
        weights_input (numpy.ndarray): 输入到隐藏层的权重，形状为 (n_hidden,)
            控制每个神经元对输入的敏感度
        bias_hidden (numpy.ndarray): 隐藏层偏置，形状为 (n_hidden,)
            控制每个神经元的激活位置（平移）
        weights_output (numpy.ndarray): 隐藏层到输出的权重，形状为 (n_hidden,)
            控制每个神经元对最终输出的贡献大小
        bias_output (float): 输出层偏置，调整整体输出水平
        activation (str): 激活函数类型，可选 'sigmoid'、'tanh'、'relu'
    
    返回:
        tuple: (output, hidden_output)
            - output: 网络最终输出，形状为 (n_samples,)
            - hidden_output: 隐藏层激活输出，形状为 (n_samples, n_hidden)
    
    示例:
        >>> x = np.linspace(-3, 3, 100)
        >>> w_in = np.array([1.0, 2.0])
        >>> b_in = np.array([0.0, -1.0])
        >>> w_out = np.array([0.5, -0.5])
        >>> b_out = 0.0
        >>> output, hidden = neural_network_output(x, w_in, b_in, w_out, b_out, 'tanh')
    """
    # 计算隐藏层输入: x与weights_input的外积加上偏置
    # hidden_input[i,j] = x[i] * weights_input[j] + bias_hidden[j]
    hidden_input = np.outer(x, weights_input) + bias_hidden  # 形状: (n_samples, n_hidden)
    
    # 应用激活函数，引入非线性
    # 非线性是万能近似的关键：没有它，多层网络等价于单层线性变换
    if activation == 'sigmoid':
        hidden_output = sigmoid(hidden_input)
    elif activation == 'tanh':
        hidden_output = tanh(hidden_input)
    elif activation == 'relu':
        hidden_output = relu(hidden_input)
    else:
        # 线性激活（仅用于演示，实际不会使用）
        hidden_output = hidden_input
    
    # 输出层：隐藏层输出的线性组合
    # 每个隐藏神经元的输出乘以对应的输出权重，求和后加上偏置
    output = hidden_output @ weights_output + bias_output
    
    return output, hidden_output


# ============================================================================
# 可视化演示函数
# ============================================================================

def demo_step_function():
    """
    演示1：阶跃函数原理可视化
    
    本函数展示万能近似定理的数学基础：
    1. Sigmoid函数在高权重下逼近阶跃函数
    2. 平移的阶跃函数覆盖不同位置
    3. 两个阶跃函数相减形成脉冲
    4. 多个脉冲加权组合逼近任意函数
    
    核心思想：任何连续函数都可以被分解为无穷多个小脉冲的和，
    而神经网络通过调整权重和偏置，可以生成这些脉冲。
    
    生成图表:
        - universal_approximation_steps.png
        
    图表内容:
        1. 单个Sigmoid：展示权重对函数形状的影响
        2. 平移的阶跃函数：展示偏置对激活位置的控制
        3. 脉冲函数：两个阶跃函数的差
        4. 函数逼近：多个脉冲组合逼近复杂函数
    """
    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('万能近似定理原理演示：从阶跃函数到复杂函数', fontsize=16, fontweight='bold')
    
    x = np.linspace(-5, 5, 1000)
    
    # 子图1：单个Sigmoid（不同权重）
    ax1 = axes[0, 0]
    weights = [0.5, 1, 2, 5, 10, 50]
    colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))
    for w, color in zip(weights, colors):
        y = sigmoid(w * x)
        ax1.plot(x, y, color=color, linewidth=2, label=f'w={w}')
    ax1.set_title('单个Sigmoid：权重越大，越接近阶跃函数')
    ax1.set_xlabel('x')
    ax1.set_ylabel('sigmoid(w·x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # 子图2：平移的阶跃函数
    ax2 = axes[0, 1]
    centers = [-2, -1, 0, 1, 2]
    colors = plt.cm.plasma(np.linspace(0, 1, len(centers)))
    for c, color in zip(centers, colors):
        y = sigmoid(50 * (x - c))
        ax2.plot(x, y, color=color, linewidth=2, label=f'中心={c}')
    ax2.set_title('平移的阶跃函数：sigmoid(50(x-c))')
    ax2.set_xlabel('x')
    ax2.set_ylabel('输出')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3：组合两个阶跃函数形成脉冲
    ax3 = axes[1, 0]
    c1, c2 = -1, 1
    width = 50
    y = sigmoid(width * (x - c1)) - sigmoid(width * (x - c2))
    ax3.plot(x, y, 'b-', linewidth=3, label=f'脉冲 [{c1}, {c2}]')
    ax3.fill_between(x, y, alpha=0.3)
    ax3.set_title('两个阶跃函数相减 = 脉冲函数')
    ax3.set_xlabel('x')
    ax3.set_ylabel('输出')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4：多个脉冲组合逼近复杂函数
    ax4 = axes[1, 1]
    target_func = lambda x: np.sin(x) + 0.5 * np.sin(3*x)
    y_target = target_func(x)
    
    # 手动构造逼近
    centers = [-3, -2, -1, 0, 1, 2, 3]
    amplitudes = [0.8, -0.5, 0.3, 0, -0.3, 0.5, -0.8]
    y_approx = np.zeros_like(x)
    
    for c, a in zip(centers, amplitudes):
        pulse = sigmoid(30 * (x - c + 0.3)) - sigmoid(30 * (x - c - 0.3))
        y_approx += a * pulse
    
    ax4.plot(x, y_target, 'b-', linewidth=2, label='目标函数', alpha=0.7)
    ax4.plot(x, y_approx, 'r--', linewidth=2, label='神经网络逼近')
    ax4.set_title('多个脉冲组合逼近复杂函数')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('universal_approximation_steps.png', dpi=150, bbox_inches='tight')
    print("已保存图片：universal_approximation_steps.png")
    plt.show()


def demo_function_fitting():
    """
    演示2：不同神经元数量的拟合效果对比
    
    本函数通过对比1、2、3、5、10、20、50、100个隐藏神经元
    对sin(x)函数的拟合效果，直观展示万能近似定理：
    
    核心结论:
        - 神经元数量越多，拟合能力越强
        - 少量神经元只能捕捉函数的大致趋势
        - 足够多的神经元可以逼近函数的每一个细节
    
    训练过程:
        1. 随机初始化网络权重
        2. 使用梯度下降优化输出层权重
        3. 最小化均方误差(MSE)
    
    注意:
        这是一个简化的训练过程，仅优化输出层权重，
        实际深度学习会同时优化所有层的权重。
    
    生成图表:
        - universal_approximation_fitting.png
    
    图表布局:
        3x3网格，展示8种不同神经元数量的拟合对比
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('神经网络万能近似定理：拟合 sin(x)', fontsize=16, fontweight='bold')
    
    # 生成目标数据
    x = np.linspace(-3, 3, 200)
    y_target = np.sin(x)
    
    # 不同的神经元数量
    n_neurons_list = [1, 2, 3, 5, 10, 20, 50, 100]
    
    # 创建子图
    axes = []
    for i in range(8):
        row = i // 3
        col = i % 3
        if i < 8:
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)
    
    # 为每个神经元数量拟合并绘图
    np.random.seed(42)
    
    for idx, (n_neurons, ax) in enumerate(zip(n_neurons_list, axes)):
        # 随机初始化权重（为了演示效果，这里手动构造较好的权重）
        # 在实际训练中，这些权重是通过反向传播学习得到的
        
        if n_neurons == 1:
            # 单个神经元只能学习简单的曲线
            weights_input = np.array([1.0])
            bias_hidden = np.array([0.0])
            weights_output = np.array([1.0])
            bias_output = 0.0
        else:
            # 多个神经元分布在不同位置
            weights_input = np.random.randn(n_neurons) * 2
            bias_hidden = np.random.randn(n_neurons) * 2
            weights_output = np.random.randn(n_neurons)
            bias_output = 0.0
            
            # 简单的梯度下降优化（简化版）
            learning_rate = 0.1
            for _ in range(1000):
                output, hidden = neural_network_output(
                    x, weights_input, bias_hidden, weights_output, bias_output, 'tanh'
                )
                error = output - y_target
                
                # 计算梯度（简化）
                dW_output = hidden.T @ error / len(x)
                db_output = np.mean(error)
                
                # 更新权重
                weights_output -= learning_rate * dW_output
                bias_output -= learning_rate * db_output
        
        # 计算最终输出
        y_pred, _ = neural_network_output(
            x, weights_input, bias_hidden, weights_output, bias_output, 'tanh'
        )
        
        # 绘制
        ax.plot(x, y_target, 'b-', linewidth=2, label='目标函数 sin(x)', alpha=0.7)
        ax.plot(x, y_pred, 'r--', linewidth=2, label=f'神经网络 ({n_neurons}神经元)')
        ax.set_title(f'{n_neurons} 个隐藏神经元', fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
    
    plt.savefig('universal_approximation_fitting.png', dpi=150, bbox_inches='tight')
    print("已保存图片：universal_approximation_fitting.png")
    plt.show()


def demo_hidden_neurons():
    """
    演示3：隐藏层神经元特征可视化
    
    本函数深入展示神经网络的内部工作原理：
    
    1. 每个神经元的激活模式:
       - 每个隐藏神经元学习函数的不同局部特征
       - 不同的权重和偏置使神经元在不同位置激活
       - 类似于基函数的线性组合
    
    2. 逐步累加原理:
       - 展示如何从前N个神经元的组合逐步构建最终函数
       - 每个神经元贡献一个"小修正"
       - 累加后逼近目标函数
    
    3. 拟合质量分析:
       - 对比目标函数和拟合结果
       - 可视化误差分布
       - 评估逼近精度
    
    网络配置:
        - 10个隐藏神经元
        - tanh激活函数
        - 2000轮梯度下降训练
    
    生成图表:
        - universal_approximation_neurons.png
    
    图表内容:
        - 每个神经元的激活曲线
        - 逐步累加的可视化
        - 最终拟合对比
        - 误差分布图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('隐藏层神经元的作用：每个神经元学习不同的特征', fontsize=16, fontweight='bold')
    
    x = np.linspace(-3, 3, 500)
    y_target = np.sin(x)
    
    # 使用10个神经元
    n_neurons = 10
    np.random.seed(123)
    
    # 初始化权重
    weights_input = np.random.randn(n_neurons) * 3
    bias_hidden = np.linspace(-3, 3, n_neurons)
    weights_output = np.random.randn(n_neurons)
    bias_output = 0.0
    
    # 训练
    learning_rate = 0.5
    for epoch in range(2000):
        output, hidden = neural_network_output(
            x, weights_input, bias_hidden, weights_output, bias_output, 'tanh'
        )
        error = output - y_target
        
        dW_output = hidden.T @ error / len(x)
        db_output = np.mean(error)
        
        weights_output -= learning_rate * dW_output
        bias_output -= learning_rate * db_output
        
        if epoch % 500 == 0:
            mse = np.mean(error**2)
            print(f"Epoch {epoch}, MSE: {mse:.6f}")
    
    # 子图1：每个神经元的激活输出
    ax1 = axes[0, 0]
    _, hidden_outputs = neural_network_output(
        x, weights_input, bias_hidden, weights_output, bias_output, 'tanh'
    )
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_neurons))
    for i in range(n_neurons):
        ax1.plot(x, hidden_outputs[:, i], color=colors[i], alpha=0.7, 
                label=f'神经元 {i+1}')
    ax1.set_title('每个隐藏神经元的激活输出')
    ax1.set_xlabel('x')
    ax1.set_ylabel('激活值')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 子图2：加权和逐步构建最终函数
    ax2 = axes[0, 1]
    cumulative = np.zeros_like(x)
    ax2.plot(x, y_target, 'b-', linewidth=2, label='目标函数', alpha=0.7)
    
    step_size = max(1, n_neurons // 5)
    for i in range(0, n_neurons, step_size):
        end_i = min(i + step_size, n_neurons)
        contribution = hidden_outputs[:, i:end_i] @ weights_output[i:end_i]
        cumulative += contribution
        ax2.plot(x, cumulative + bias_output, alpha=0.6, 
                label=f'前{end_i}个神经元组合')
    
    ax2.set_title('逐步累加神经元贡献')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 子图3：最终拟合效果
    ax3 = axes[1, 0]
    y_pred = hidden_outputs @ weights_output + bias_output
    ax3.plot(x, y_target, 'b-', linewidth=3, label='目标函数 sin(x)', alpha=0.7)
    ax3.plot(x, y_pred, 'r--', linewidth=3, label=f'神经网络拟合 ({n_neurons}神经元)')
    ax3.fill_between(x, y_target, y_pred, alpha=0.2, color='green', label='误差')
    ax3.set_title('最终拟合效果')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4：误差分布
    ax4 = axes[1, 1]
    error = y_target - y_pred
    ax4.plot(x, error, 'g-', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.fill_between(x, error, alpha=0.3, color='green')
    ax4.set_title('拟合误差分布')
    ax4.set_xlabel('x')
    ax4.set_ylabel('误差')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('universal_approximation_neurons.png', dpi=150, bbox_inches='tight')
    print("已保存图片：universal_approximation_neurons.png")
    plt.show()
    
    return weights_input, bias_hidden, weights_output, bias_output


def interactive_demo():
    """
    演示4：交互式神经元数量调整演示
    
    本函数创建一个交互式可视化界面，允许用户实时调整
    隐藏层神经元数量，观察拟合效果的动态变化。
    
    交互方式:
        - 拖动滑块调整神经元数量（1-100）
        - 实时显示拟合曲线和MSE误差
    
    观察要点:
        1. 神经元数量增加时，拟合曲线越来越接近目标函数
        2. 均方误差(MSE)随神经元数量增加而减小
        3. 过少的神经元会导致欠拟合
        4. 足够多的神经元可以达到任意精度
    
    技术实现:
        - 使用matplotlib.widgets.Slider创建滑块
        - 每次滑块变化时重新训练网络
        - 使用固定随机种子确保可重复性
    
    生成图表:
        - universal_approximation_interactive.png（初始状态截图）
    
    注意:
        此演示需要图形界面支持，在headless环境下无法交互
    """
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[4, 1], hspace=0.3)
    
    # 主图
    ax_main = fig.add_subplot(gs[0, :])
    plt.subplots_adjust(bottom=0.2)
    
    # 生成数据
    x = np.linspace(-3, 3, 300)
    y_target = np.sin(x)
    
    # 初始化线条
    line_target, = ax_main.plot(x, y_target, 'b-', linewidth=2, label='目标函数 sin(x)', alpha=0.7)
    line_pred, = ax_main.plot(x, np.zeros_like(x), 'r--', linewidth=2, label='神经网络拟合')
    ax_main.set_ylim(-1.5, 1.5)
    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y')
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    ax_main.set_title('万能近似定理交互演示：调整神经元数量', fontsize=14, fontweight='bold')
    
    # 创建滑块
    ax_slider = plt.axes([0.2, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, '隐藏神经元数量', 1, 100, valinit=5, valstep=1)
    
    # 文本显示
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis('off')
    text_obj = ax_text.text(0.5, 0.5, '', transform=ax_text.transAxes, 
                           fontsize=12, ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def update(val):
        n_neurons = int(slider.val)
        
        # 使用固定的随机种子确保可重复性
        np.random.seed(42 + n_neurons)
        
        # 初始化权重
        weights_input = np.random.randn(n_neurons) * 2
        bias_hidden = np.random.randn(n_neurons) * 2
        weights_output = np.random.randn(n_neurons) * 0.5
        bias_output = 0.0
        
        # 快速训练
        learning_rate = 0.2
        for _ in range(500):
            output, hidden = neural_network_output(
                x, weights_input, bias_hidden, weights_output, bias_output, 'tanh'
            )
            error = output - y_target
            
            dW_output = hidden.T @ error / len(x)
            db_output = np.mean(error)
            
            weights_output -= learning_rate * dW_output
            bias_output -= learning_rate * db_output
        
        y_pred, _ = neural_network_output(
            x, weights_input, bias_hidden, weights_output, bias_output, 'tanh'
        )
        
        line_pred.set_ydata(y_pred)
        
        mse = np.mean((y_pred - y_target)**2)
        text_obj.set_text(f'神经元数量: {n_neurons} | 均方误差 (MSE): {mse:.6f}')
        
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # 初始化
    update(None)
    
    plt.savefig('universal_approximation_interactive.png', dpi=150, bbox_inches='tight')
    print("已保存图片：universal_approximation_interactive.png")
    print("\n交互式演示说明：")
    print("- 拖动滑块调整隐藏层神经元数量（1-100）")
    print("- 观察神经元数量增加时拟合效果的改善")
    print("- 注意误差（MSE）如何随神经元数量减少")
    plt.show()


def compare_activation_functions():
    """
    演示5：不同激活函数的万能逼近能力对比
    
    本函数比较三种常见激活函数（tanh、sigmoid、ReLU）
    在逼近不同类型函数时的表现：
    
    测试函数:
        1. sin(x) - 周期性光滑函数
        2. x² - 二次多项式
        3. |x| - 绝对值函数（在0点不可导）
        4. 阶跃函数 - 不连续函数
    
    激活函数特性对比:
        - tanh: 输出(-1,1)，适合对称数据，收敛快
        - sigmoid: 输出(0,1)，适合概率解释，可能梯度消失
        - ReLU: 计算简单，缓解梯度消失，适合深层网络
    
    万能近似定理适用性:
        三种激活函数都满足万能近似定理的条件，
        理论上只要有足够多的神经元，都能逼近任意连续函数。
    
    实际差异:
        - 不同激活函数的学习速度和最终精度可能不同
        - 某些函数可能更适合特定类型的目标函数
    
    生成图表:
        - universal_approximation_activations.png
    
    图表布局:
        2x2网格，每个子图展示一种目标函数的逼近结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('不同激活函数的万能逼近能力对比', fontsize=16, fontweight='bold')
    
    x = np.linspace(-3, 3, 300)
    target_functions = {
        'sin(x)': np.sin(x),
        'x²': x**2 / 3,
        '|x|': np.abs(x) / 1.5,
        '阶跃函数': np.where(x > 0, 1, -1).astype(float)
    }
    
    activations = ['tanh', 'sigmoid', 'relu']
    colors = ['red', 'green', 'purple']
    
    for (func_name, y_target), ax in zip(target_functions.items(), axes.flat):
        ax.plot(x, y_target, 'b-', linewidth=3, label=f'目标: {func_name}', alpha=0.7)
        
        for activation, color in zip(activations, colors):
            np.random.seed(42)
            n_neurons = 20
            
            weights_input = np.random.randn(n_neurons) * 2
            bias_hidden = np.random.randn(n_neurons) * 2
            weights_output = np.random.randn(n_neurons) * 0.5
            bias_output = 0.0
            
            # 训练
            learning_rate = 0.3
            for _ in range(1000):
                output, hidden = neural_network_output(
                    x, weights_input, bias_hidden, weights_output, bias_output, activation
                )
                error = output - y_target
                
                dW_output = hidden.T @ error / len(x)
                db_output = np.mean(error)
                
                weights_output -= learning_rate * dW_output
                bias_output -= learning_rate * db_output
            
            y_pred, _ = neural_network_output(
                x, weights_input, bias_hidden, weights_output, bias_output, activation
            )
            
            ax.plot(x, y_pred, '--', color=color, linewidth=2, 
                   label=f'{activation}', alpha=0.8)
        
        ax.set_title(f'拟合 {func_name}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('universal_approximation_activations.png', dpi=150, bbox_inches='tight')
    print("已保存图片：universal_approximation_activations.png")
    plt.show()


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    """
    主程序入口：运行所有可视化演示
    
    运行方式:
        1. 直接运行（显示图形窗口）:
           python universal_approximation_demo.py
        
        2. Headless模式（仅保存图片，无图形界面）:
           python universal_approximation_demo.py --headless
    
    输出文件:
        - universal_approximation_steps.png       (阶跃函数原理)
        - universal_approximation_fitting.png     (神经元数量对比)
        - universal_approximation_neurons.png     (神经元特征可视化)
        - universal_approximation_activations.png (激活函数对比)
        - universal_approximation_interactive.png (交互式界面截图)
    
    依赖库:
        - numpy: 数值计算
        - matplotlib: 可视化
    
    建议:
        - 使用headless模式在服务器或无GUI环境下运行
        - 图形窗口模式下可以交互式探索神经元数量的影响
    """
    import sys
    
    # 打印程序信息横幅
    print("=" * 60)
    print("神经网络万能近似定理可视化演示")
    print("=" * 60)
    print("\n本程序包含以下演示：")
    print("  1. 阶跃函数原理演示")
    print("  2. 不同神经元数量的拟合效果")
    print("  3. 隐藏层神经元特征可视化")
    print("  4. 不同激活函数对比")
    print("  5. 交互式演示（可调节神经元数量）")
    print("\n" + "=" * 60)
    
    # 检查命令行参数，判断是否使用headless模式
    # headless模式不显示图形窗口，只保存图片到文件
    headless = '--headless' in sys.argv
    if headless:
        print("\n[Headless模式：不显示图像窗口，仅保存到文件]")
        plt.switch_backend('Agg')  # 切换到非交互式后端
    
    # =========================================================================
    # 运行所有演示
    # =========================================================================
    
    # 演示1：阶跃函数原理 - 展示万能近似的数学基础
    print("\n>>> 演示1：阶跃函数原理")
    print("-" * 40)
    demo_step_function()
    if headless:
        plt.close('all')  # 关闭所有图形释放内存
    
    # 演示2：神经元数量对比 - 展示神经元数量与拟合能力的关系
    print("\n>>> 演示2：不同神经元数量的拟合效果")
    print("-" * 40)
    demo_function_fitting()
    if headless:
        plt.close('all')
    
    # 演示3：神经元特征可视化 - 深入理解网络内部工作
    print("\n>>> 演示3：隐藏层神经元特征可视化")
    print("-" * 40)
    demo_hidden_neurons()
    if headless:
        plt.close('all')
    
    # 演示4：激活函数对比 - 比较不同激活函数的性能
    print("\n>>> 演示4：不同激活函数对比")
    print("-" * 40)
    compare_activation_functions()
    if headless:
        plt.close('all')
    
    # 演示5：交互式演示 - 需要图形界面支持
    if not headless:
        print("\n>>> 演示5：交互式演示")
        print("-" * 40)
        interactive_demo()
    else:
        print("\n[Headless模式：跳过交互式演示]")
    
    # 程序结束信息
    print("\n" + "=" * 60)
    print("所有演示完成！图片已保存到当前目录。")
    print("=" * 60)
