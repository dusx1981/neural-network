# -*- coding: utf-8 -*-
"""
CNN Visualization Tutorial
=========================
This script visualizes Convolutional Neural Network concepts
(convolution and pooling layers) in an easy-to-understand way.

Author: AI Assistant
Date: 2026-02-16
"""

from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义颜色方案
COLORS = {
    'blue': '#339AF0',
    'green': '#51CF66',
    'red': '#FF6B6B',
    'orange': '#FF922B',
    'purple': '#9C36B5',
    'yellow': '#FFD93D',
    'gray': '#6B7280',
    'light_blue': '#E7F5FF',
    'light_green': '#D3F9D8',
    'light_red': '#FFE7E7'
}


def create_digit_7_image():
    """
    创建文档中描述的6x6数字"7"图像
    """
    # 6x6的图像矩阵，1表示白色(有笔画)，0表示黑色(背景)
    image = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    return image


def create_vertical_edge_kernel():
    """
    创建垂直边缘检测卷积核
    """
    kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    return kernel


def visualize_input_image():
    """
    可视化输入的数字"7"图像
    展示6x6像素网格
    """
    image = create_digit_7_image()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 使用灰度显示图像
    im = ax.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    
    # 添加网格线
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # 在每个格子中显示数值
    for i in range(6):
        for j in range(6):
            text = ax.text(j, i, str(int(image[i, j])),
                          ha="center", va="center", color="black" if image[i, j] == 0 else "white",
                          fontsize=16, fontweight='bold')
    
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(5.5, -0.5)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels([f'列{i}' for i in range(6)], fontsize=12)
    ax.set_yticklabels([f'行{i}' for i in range(6)], fontsize=12)
    ax.set_title('输入图像：数字"7" (6×6像素)\n1=白色(有笔画), 0=黑色(背景)', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('code/cnn_visualization/01_input_image.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Saved: 01_input_image.png")


def visualize_kernel():
    """
    可视化垂直边缘检测卷积核
    解释卷积核的设计原理
    """
    kernel = create_vertical_edge_kernel()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：卷积核矩阵
    im1 = ax1.imshow(kernel, cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest')
    ax1.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax1.grid(which='minor', color='black', linestyle='-', linewidth=2)
    
    for i in range(3):
        for j in range(3):
            text = ax1.text(j, i, kernel[i, j],
                          ha="center", va="center", color="white" if abs(kernel[i, j]) > 1 else "black",
                          fontsize=20, fontweight='bold')
    
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(2.5, -0.5)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_title('卷积核 (3×3)\n垂直边缘检测器', fontsize=14, fontweight='bold', pad=15)
    
    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('权重值', fontsize=12)
    
    # 右图：卷积核工作原理解释
    ax2.axis('off')
    explanation_text = """
    卷积核设计原理：
    
    ┌────┬────┬────┐
    │ -1 │  0 │ +1 │  左列负值：检测左侧暗区域
    ├────┼────┼────┤
    │ -1 │  0 │ +1 │  中列零值：过渡区域
    ├────┼────┼────┤
    │ -1 │  0 │ +1 │  右列正值：检测右侧亮区域
    └────┴────┴────┘
    
    工作原理：
    * 当卷积核滑到"左暗右亮"的区域时，
      负值*0 + 正值*1 = 大正值
      -> 检测到垂直边缘！
    
    * 当区域亮度均匀时，
      正负值相互抵消 ~ 0
      -> 没有边缘
    """
    
    ax2.text(0.1, 0.5, explanation_text, fontsize=13, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor=COLORS['light_blue'], alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('code/cnn_visualization/02_convolution_kernel.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Saved: 02_convolution_kernel.png")


def perform_convolution(image, kernel):
    """
    执行卷积运算
    返回特征图和每一步的计算过程
    """
    # 获取图像和卷积核的尺寸
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    # 计算输出特征图的尺寸（不使用padding）
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1
    
    # 初始化特征图
    feature_map = np.zeros((output_h, output_w))
    
    # 存储每一步的计算过程
    steps = []
    
    # 滑动卷积核
    for i in range(output_h):
        for j in range(output_w):
            # 提取当前窗口
            window = image[i:i+kernel_h, j:j+kernel_w]
            
            # 逐元素相乘并求和（点积）
            conv_result = np.sum(window * kernel)
            feature_map[i, j] = conv_result
            
            # 记录计算步骤
            steps.append({
                'position': (i, j),
                'window': window.copy(),
                'result': conv_result,
                'calculation': window * kernel
            })
    
    return feature_map, steps


def visualize_convolution_steps():
    """
    可视化卷积运算的几个关键步骤
    展示窗口滑动和点积计算过程
    """
    image = create_digit_7_image()
    kernel = create_vertical_edge_kernel()
    feature_map, steps = perform_convolution(image, kernel)
    
    # 选择几个关键步骤展示
    # 步骤1：左上角(平坦区域)
    # 步骤2：覆盖部分边缘
    # 步骤3：覆盖"7"的竖线(强响应)
    key_steps = [0, 5, 8]  # 选择第1、6、9个窗口位置
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)
    
    # 标题
    fig.suptitle('卷积运算详解：滑动窗口与点积计算', fontsize=20, fontweight='bold', y=0.98)
    
    # 展示3个关键步骤
    for idx, step_idx in enumerate(key_steps):
        step = steps[step_idx]
        row = idx
        
        # 左：输入图像及窗口位置
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        
        # 高亮当前窗口
        i, j = step['position']
        rect = Rectangle((j-0.5, i-0.5), 3, 3, linewidth=4, 
                        edgecolor=COLORS['red'], facecolor='none')
        ax1.add_patch(rect)
        
        ax1.set_title(f'步骤{idx+1}：窗口位置({i},{j})', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(6))
        ax1.set_yticks(range(6))
        
        # 中：窗口内容
        ax2 = fig.add_subplot(gs[row, 1])
        window = step['window']
        im2 = ax2.imshow(window, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        for wi in range(3):
            for wj in range(3):
                ax2.text(wj, wi, str(int(window[wi, wj])), ha="center", va="center",
                        fontsize=16, fontweight='bold',
                        color="white" if window[wi, wj] == 1 else "black")
        ax2.set_title('窗口内容', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(3))
        ax2.set_yticks(range(3))
        
        # 右：卷积核
        ax3 = fig.add_subplot(gs[row, 2])
        im3 = ax3.imshow(kernel, cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest')
        for ki in range(3):
            for kj in range(3):
                ax3.text(kj, ki, kernel[ki, kj], ha="center", va="center",
                        fontsize=16, fontweight='bold',
                        color="white" if abs(kernel[ki, kj]) > 0.5 else "black")
        ax3.set_title('卷积核', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(3))
        ax3.set_yticks(range(3))
        
        # 计算过程展示
        ax4 = fig.add_subplot(gs[row, 3])
        ax4.axis('off')
        
        # 生成计算过程文本
        calc_text = "点积计算：\n\n"
        total = 0
        for ki in range(3):
            for kj in range(3):
                val1 = int(window[ki, kj])
                val2 = int(kernel[ki, kj])
                prod = val1 * val2
                total += prod
                if prod != 0:
                    calc_text += f"{val1:2d} × {val2:+2d} = {prod:+3d}\n"
        
        calc_text += f"\n{'─' * 20}\n"
        calc_text += f"总和 = {total:3d}"
        
        # 根据结果值设置背景色
        if total > 2:
            bg_color = COLORS['light_green']
            note = "\n\n检测到\n垂直边缘！"
        elif total < -2:
            bg_color = COLORS['light_red']
            note = "\n\n反向边缘"
        else:
            bg_color = COLORS['light_blue']
            note = "\n\n无明显\n边缘"
        
        calc_text += note
        
        ax4.text(0.1, 0.5, calc_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.9))
    
    plt.savefig('code/cnn_visualization/03_convolution_steps.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Saved: 03_convolution_steps.png")


def visualize_feature_map():
    """
    可视化卷积后的特征图
    展示检测到的垂直边缘
    """
    image = create_digit_7_image()
    kernel = create_vertical_edge_kernel()
    feature_map, _ = perform_convolution(image, kernel)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 左图：原始输入
    ax1 = axes[0]
    im1 = ax1.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax1.set_title('输入图像：数字"7"', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(6))
    ax1.set_yticks(range(6))
    
    # 添加箭头
    ax1.annotate('', xy=(1.3, 0.5), xytext=(0.7, 0.5),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['blue']))
    
    # 中图：卷积核
    ax2 = axes[1]
    im2 = ax2.imshow(kernel, cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest')
    ax2.set_title('卷积核\n(垂直边缘检测)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, kernel[i, j], ha="center", va="center",
                    fontsize=14, fontweight='bold',
                    color="white" if abs(kernel[i, j]) > 0.5 else "black")
    
    # 添加箭头
    ax2.annotate('', xy=(1.3, 0.5), xytext=(0.7, 0.5),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['blue']))
    
    # 右图：特征图
    ax3 = axes[2]
    # 使用diverging colormap，0为白色，正值为绿色，负值为红色
    im3 = ax3.imshow(feature_map, cmap='RdYlGn', vmin=-3, vmax=3, interpolation='nearest')
    ax3.set_title('输出特征图\n(4×4，显示边缘强度)', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(4))
    ax3.set_yticks(range(4))
    
    # 在特征图上标注数值
    for i in range(4):
        for j in range(4):
            val = feature_map[i, j]
            color = "white" if abs(val) > 1.5 else "black"
            ax3.text(j, i, f'{val:.0f}', ha="center", va="center",
                    fontsize=14, fontweight='bold', color=color)
    
    # 添加颜色条
    cbar = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar.set_label('激活强度', fontsize=12)
    cbar.ax.text(0.5, 3.2, '强边缘', ha='center', fontsize=10, color='green')
    cbar.ax.text(0.5, -3.2, '反向边缘', ha='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('code/cnn_visualization/04_feature_map.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Saved: 04_feature_map.png")
    
    return feature_map


def visualize_max_pooling(feature_map):
    """
    可视化最大池化过程
    展示如何从4x4特征图得到2x2池化结果
    """
    # 使用文档中的示例特征图
    feature_map_example = np.array([
        [1, 0, 2, 3],
        [4, 6, 5, 1],
        [1, 2, 9, 0],
        [0, 1, 2, 1]
    ])
    
    # 执行最大池化（2x2窗口，步长2）
    pool_size = 2
    stride = 2
    output_size = feature_map_example.shape[0] // pool_size
    pooled = np.zeros((output_size, output_size))
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('最大池化(Max Pooling)详解', fontsize=20, fontweight='bold')
    
    # 上方：输入特征图（占满第一行两列）
    ax_input = fig.add_subplot(gs[0, :])
    im = ax_input.imshow(feature_map_example, cmap='YlOrRd', vmin=0, vmax=10, interpolation='nearest')
    ax_input.set_title('输入特征图 (4×4) - 卷积层输出', fontsize=16, fontweight='bold', pad=15)
    ax_input.set_xticks(range(4))
    ax_input.set_yticks(range(4))
    
    # 在格子里显示数值
    for i in range(4):
        for j in range(4):
            val = feature_map_example[i, j]
            color = "white" if val > 5 else "black"
            ax_input.text(j, i, f'{val}', ha="center", va="center",
                         fontsize=16, fontweight='bold', color=color)
    
    # 高亮4个池化区域
    colors_regions = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple']]
    for idx, (i, j) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2)]):
        rect = Rectangle((j-0.5, i-0.5), 2, 2, linewidth=5,
                        edgecolor=colors_regions[idx], facecolor='none', linestyle='--')
        ax_input.add_patch(rect)
        # 添加区域编号
        ax_input.text(j+0.5, i-0.8, f'区域{idx+1}', ha="center", fontsize=12,
                     fontweight='bold', color=colors_regions[idx])
    
    plt.colorbar(im, ax=ax_input, label='激活值', shrink=0.8)
    
    # 下方：展示每个区域的池化过程（2x2网格）
    region_names = ['左上', '右上', '左下', '右下']
    region_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for idx, ((i, j), name) in enumerate(zip(region_positions, region_names)):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[1 + row, col])
        
        # 提取当前区域
        region = feature_map_example[i*2:(i+1)*2, j*2:(j+1)*2]
        max_val = np.max(region)
        pooled[i, j] = max_val
        
        # 显示区域
        im_reg = ax.imshow(region, cmap='YlOrRd', vmin=0, vmax=10, interpolation='nearest')
        ax.set_title(f'区域{idx+1} ({name}) - 2×2窗口', fontsize=13, fontweight='bold')
        ax.set_xticks(range(2))
        ax.set_yticks(range(2))
        
        # 显示数值，最大值高亮
        for ri in range(2):
            for rj in range(2):
                val = region[ri, rj]
                is_max = (val == max_val)
                bbox_props = dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow' if is_max else 'white',
                                edgecolor='red' if is_max else 'gray',
                                linewidth=3 if is_max else 1)
                ax.text(rj, ri, f'{val}', ha="center", va="center",
                       fontsize=18, fontweight='bold',
                       bbox=bbox_props)
        
        # 显示池化结果
        ax.text(0.5, 2.3, f'最大值 = {max_val}', ha="center", fontsize=14,
               fontweight='bold', color=COLORS['red'],
               bbox=dict(boxstyle='round', facecolor=COLORS['light_green'], alpha=0.8))
    
    plt.savefig('code/cnn_visualization/05_max_pooling_process.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Saved: 05_max_pooling_process.png")
    
    return pooled


def visualize_pooled_result(pooled):
    """
    可视化池化后的结果
    对比池化前后的变化
    """
    feature_map_example = np.array([
        [1, 0, 2, 3],
        [4, 6, 5, 1],
        [1, 2, 9, 0],
        [0, 1, 2, 1]
    ])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：池化前的特征图
    ax1 = axes[0]
    im1 = ax1.imshow(feature_map_example, cmap='YlOrRd', vmin=0, vmax=10, interpolation='nearest')
    ax1.set_title('池化前：4×4 特征图', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    
    for i in range(4):
        for j in range(4):
            val = feature_map_example[i, j]
            color = "white" if val > 5 else "black"
            ax1.text(j, i, f'{val}', ha="center", va="center",
                    fontsize=14, fontweight='bold', color=color)
    
    # 添加箭头
    ax1.annotate('', xy=(1.3, 0.5), xytext=(0.7, 0.5),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['blue']))
    ax1.text(1.0, -0.5, 'Max Pooling\n(2×2, 步长2)', ha="center", fontsize=11,
            bbox=dict(boxstyle='round', facecolor=COLORS['light_blue'], alpha=0.8))
    
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='激活值')
    
    # 右图：池化后的特征图
    ax2 = axes[1]
    im2 = ax2.imshow(pooled, cmap='YlOrRd', vmin=0, vmax=10, interpolation='nearest')
    ax2.set_title('池化后：2×2 特征图\n(尺寸减半，保留最强特征)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(2))
    ax2.set_yticks(range(2))
    
    for i in range(2):
        for j in range(2):
            val = pooled[i, j]
            color = "white" if val > 5 else "black"
            ax2.text(j, i, f'{val:.0f}', ha="center", va="center",
                    fontsize=20, fontweight='bold', color=color)
    
    # 添加说明框
    explanation = """
    池化层的效果：
    
    [OK] 数据量减少：4x4 -> 2x2
      (变为原来的1/4)
    
    [OK] 保留关键信息：
      只保留每个区域的最强激活
    
    [OK] 平移不变性：
      特征位置微小变化不影响结果
    """
    
    ax2.text(2.5, 0.5, explanation, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['light_green'], alpha=0.9))
    
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='激活值')
    
    plt.tight_layout()
    plt.savefig('code/cnn_visualization/06_pooled_result.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Saved: 06_pooled_result.png")


def visualize_complete_flow():
    """
    可视化完整的卷积+池化流程
    一张图展示从输入到输出的完整过程
    """
    image = create_digit_7_image()
    kernel = create_vertical_edge_kernel()
    feature_map, _ = perform_convolution(image, kernel)
    
    # 对特征图进行池化
    pool_size = 2
    pooled = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            region = feature_map[i*2:(i+1)*2, j*2:(j+1)*2]
            pooled[i, j] = np.max(region)
    
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 5, wspace=0.4)
    
    fig.suptitle('CNN完整流程：卷积层 + 池化层', fontsize=22, fontweight='bold', y=1.02)
    
    # 1. 输入图像
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax1.set_title('1. 输入图像\n(6×6)', fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # 2. 卷积运算（箭头）
    ax_arrow1 = fig.add_subplot(gs[0, 1])
    ax_arrow1.axis('off')
    ax_arrow1.text(0.5, 0.5, '卷积\n(特征提取)', ha='center', va='center',
                  fontsize=14, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor=COLORS['light_blue'], alpha=0.9))
    ax_arrow1.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                      arrowprops=dict(arrowstyle='->', lw=4, color=COLORS['blue']))
    
    # 3. 特征图
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(feature_map, cmap='RdYlGn', vmin=-3, vmax=3, interpolation='nearest')
    ax2.set_title('2. 卷积层输出\n特征图 (4×4)', fontsize=14, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # 4. 池化运算（箭头）
    ax_arrow2 = fig.add_subplot(gs[0, 3])
    ax_arrow2.axis('off')
    ax_arrow2.text(0.5, 0.5, '最大池化\n(信息压缩)', ha='center', va='center',
                  fontsize=14, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor=COLORS['light_green'], alpha=0.9))
    ax_arrow2.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                      arrowprops=dict(arrowstyle='->', lw=4, color=COLORS['green']))
    
    # 5. 池化结果
    ax3 = fig.add_subplot(gs[0, 4])
    im3 = ax3.imshow(pooled, cmap='RdYlGn', vmin=-3, vmax=3, interpolation='nearest')
    ax3.set_title('3. 池化层输出\n压缩特征 (2×2)', fontsize=14, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # 添加总结说明
    summary_text = """
    总结：
    
    【卷积层】
    * 作用：特征提取器
    * 方法：卷积核滑动 x 点积运算
    * 输出：特征图（显示特征位置和强度）
    * 特点：自动学习检测特定模式
    
    【池化层】
    * 作用：信息压缩器
    * 方法：取窗口内最大值
    * 输出：缩小尺寸的特征图
    * 特点：保留关键信息，提高鲁棒性
    """
    
    fig.text(0.5, -0.15, summary_text, ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.savefig('code/cnn_visualization/07_complete_flow.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Saved: 07_complete_flow.png")


def main():
    """
    Main function: Execute all visualizations
    """
    print("="*60)
    print("CNN Convolution & Pooling Layer Visualization")
    print("="*60)
    
    # Create output directory
    import os
    os.makedirs('code/cnn_visualization', exist_ok=True)
    
    print("\n[1/7] Visualizing input image...")
    visualize_input_image()
    
    print("\n[2/7] Visualizing convolution kernel...")
    visualize_kernel()
    
    print("\n[3/7] Visualizing convolution steps...")
    visualize_convolution_steps()
    
    print("\n[4/7] Visualizing feature map...")
    feature_map = visualize_feature_map()
    
    print("\n[5/7] Visualizing max pooling process...")
    pooled = visualize_max_pooling(feature_map)
    
    print("\n[6/7] Visualizing pooling result...")
    visualize_pooled_result(pooled)
    
    print("\n[7/7] Visualizing complete flow...")
    visualize_complete_flow()
    
    print("\n" + "="*60)
    print("All visualizations completed!")
    print("Images saved to: code/cnn_visualization/")
    print("="*60)
    print("\nGenerated files:")
    print("  1. 01_input_image.png          - Input image (digit 7)")
    print("  2. 02_convolution_kernel.png   - Convolution kernel intro")
    print("  3. 03_convolution_steps.png    - Convolution steps detail")
    print("  4. 04_feature_map.png          - Feature map output")
    print("  5. 05_max_pooling_process.png  - Max pooling process")
    print("  6. 06_pooled_result.png        - Pooling comparison")
    print("  7. 07_complete_flow.png        - Complete flow summary")


if __name__ == '__main__':
    main()
