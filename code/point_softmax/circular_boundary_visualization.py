"""
神经网络如何画出两个圆形边界 - 高中生友好版
用生活比喻和分步动画解释复杂概念
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects
from matplotlib import transforms
import math
import time

# 设置友好的字体和颜色
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 友好的配色方案
COLORS = {
    'red': '#FF6B6B',
    'green': '#51CF66',
    'blue': '#339AF0',
    'yellow': '#FFD93D',
    'purple': '#9C36B5',
    'orange': '#FF922B',
    'light_blue': '#74C0FC',
    'light_green': '#8CE99A',
    'light_red': '#FFA8A8',
    'gray': '#ADB5BD'
}

def create_simple_data():
    """创建简单的三类别数据"""
    np.random.seed(42)
    
    # 靶心（红色）- 中心区域
    n = 30
    red_r = np.random.uniform(0, 0.2, n)
    red_angle = np.random.uniform(0, 2*np.pi, n)
    red = np.column_stack([red_r * np.cos(red_angle), 
                           red_r * np.sin(red_angle)])
    
    # 靶环（绿色）- 中间环
    green_r = np.random.uniform(0.6, 0.8, n)
    green_angle = np.random.uniform(0, 2*np.pi, n)
    green = np.column_stack([green_r * np.cos(green_angle), 
                             green_r * np.sin(green_angle)])
    
    # 外围（蓝色）- 四个角落
    blue = np.random.uniform(-1.2, 1.2, (n, 2))
    # 确保蓝色点离中心远一些
    mask = np.sqrt(blue[:,0]**2 + blue[:,1]**2) < 0.9
    blue[mask] *= 1.5
    
    return green, blue, red

def draw_target_with_explanation():
    """绘制靶子图并用比喻解释"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    green, blue, red = create_simple_data()
    
    # ========== 子图1：数据分布像靶子 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    ax1.set_facecolor('#F8F9FA')
    
    # 画背景圆环
    for r, color, alpha in [(0.2, COLORS['light_red'], 0.3), 
                           (0.7, COLORS['light_green'], 0.3),
                           (1.2, COLORS['light_blue'], 0.3)]:
        circle = Circle((0, 0), r, fill=True, color=color, 
                       alpha=alpha, linewidth=0)
        ax1.add_patch(circle)
    
    # 画数据点
    ax1.scatter(red[:, 0], red[:, 1], color=COLORS['red'], s=150, 
               edgecolors='white', linewidth=2, zorder=10, 
               label='红色点 (靶心)')
    ax1.scatter(green[:, 0], green[:, 1], color=COLORS['green'], s=150,
               edgecolors='white', linewidth=2, zorder=10,
               label='绿色点 (靶环)')
    ax1.scatter(blue[:, 0], blue[:, 1], color=COLORS['blue'], s=150,
               edgecolors='white', linewidth=2, zorder=10,
               label='蓝色点 (外围)')
    
    # 添加比喻标签
    ax1.text(0, 0, '🍎 苹果\n(红点)', ha='center', va='center', 
            fontsize=12, color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.text(0.8, 0, '🥦 西兰花\n(绿点)', ha='center', va='center',
            fontsize=12, color='darkgreen', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.text(1.3, 1.3, '💧 水滴\n(蓝点)', ha='center', va='center',
            fontsize=12, color='darkblue', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax1.set_title('🎯 数据分布：像射击靶子一样', fontsize=16, 
                 fontweight='bold', pad=20, color='#2C3E50')
    ax1.legend(loc='upper left', fontsize=11)
    
    # ========== 子图2：为什么需要两个圆？ ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # 用糖果比喻来解释
    explanation = """
🤔 脑筋急转弯：
桌上放着三种糖果：
🍭 红色棒棒糖（在中间）
🍏 绿色苹果糖（围着红色）
💎 蓝色硬糖（在四个角）

问：最少需要几个篮子来分开它们？

💡 答案：需要2个篮子！
① 小篮子：只装红色棒棒糖
② 大篮子：装绿色和蓝色糖果

翻译成数学：
小篮子 = 小圆圈 (半径≈0.35)
大篮子 = 大圆圈 (半径≈0.9)
    """
    
    # 创建糖果图案
    candy_colors = [COLORS['red'], COLORS['green'], COLORS['blue']]
    candy_positions = [(0.3, 0.85), (0.5, 0.85), (0.7, 0.85)]
    candy_labels = ['🍭', '🍏', '💎']
    
    for (x, y), color, label in zip(candy_positions, candy_colors, candy_labels):
        circle = Circle((x, y), 0.04, color=color, zorder=5)
        ax2.add_patch(circle)
        ax2.text(x, y, label, ha='center', va='center', 
                fontsize=16, zorder=6)
    
    # 创建篮子（圆环）
    basket1 = Circle((0.5, 0.6), 0.1, fill=False, 
                    color=COLORS['red'], linewidth=4, 
                    linestyle='--', alpha=0.7)
    basket2 = Circle((0.5, 0.6), 0.2, fill=False,
                    color=COLORS['blue'], linewidth=4,
                    linestyle='--', alpha=0.7)
    ax2.add_patch(basket1)
    ax2.add_patch(basket2)
    
    ax2.text(0.5, 0.6, '中心', ha='center', va='center', fontsize=10)
    ax2.text(0.5, 0.45, '小篮子', ha='center', va='center', 
            fontsize=12, color=COLORS['red'], fontweight='bold')
    ax2.text(0.5, 0.35, '大篮子', ha='center', va='center',
            fontsize=12, color=COLORS['blue'], fontweight='bold')
    
    # 添加文字说明
    ax2.text(0.5, 0.15, explanation, transform=ax2.transAxes,
            fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#FFF3CD', 
                     edgecolor='#FFC107', linewidth=3, pad=1))
    
    ax2.set_title('🧺 为什么需要两个"篮子"？', fontsize=16,
                 fontweight='bold', pad=20, color='#2C3E50')
    
    # ========== 子图3：神经网络只能画直线 ==========
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # 画一个卡通神经网络
    # 输入层
    for i in range(3):
        y = 0.8 - i * 0.2
        circle = Circle((0.2, y), 0.04, color=COLORS['light_blue'], zorder=5)
        ax3.add_patch(circle)
        ax3.text(0.2, y, f'{i+1}', ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # 隐藏层
    for i in range(4):
        y = 0.9 - i * 0.2
        circle = Circle((0.5, y), 0.04, color=COLORS['light_green'], zorder=5)
        ax3.add_patch(circle)
        ax3.text(0.5, y, f'H{i+1}', ha='center', va='center',
                fontsize=10, fontweight='bold')
    
    # 输出层
    for i in range(3):
        y = 0.8 - i * 0.2
        circle = Circle((0.8, y), 0.04, color=COLORS['light_red'], zorder=5)
        ax3.add_patch(circle)
        ax3.text(0.8, y, f'O{i+1}', ha='center', va='center',
                fontsize=10, fontweight='bold')
    
    # 连接线（只能画直线）
    for i in range(3):
        for j in range(4):
            ax3.plot([0.24, 0.46], [0.8-i*0.2, 0.9-j*0.2], 
                    'gray', linewidth=1, alpha=0.5)
    
    for i in range(4):
        for j in range(3):
            ax3.plot([0.54, 0.76], [0.9-i*0.2, 0.8-j*0.2],
                    'gray', linewidth=1, alpha=0.5)
    
    # 添加标题和说明
    ax3.set_title('🧠 神经网络：只能画直线', fontsize=16,
                 fontweight='bold', pad=20, color='#2C3E50')
    
    explanation3 = """
神经网络就像一支只能画直线的笔：
• 每个神经元画一条直线
• 多条直线组合成复杂形状
• 就像用直尺画图一样

🤖 神经网络工具箱：
📏 直尺 × 1
✏️ 铅笔 × 很多支
🎨 调色板 × 1

问题：怎么用直尺画圆圈？
    """
    
    ax3.text(0.5, 0.25, explanation3, transform=ax3.transAxes,
            fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD',
                     edgecolor=COLORS['blue'], linewidth=3, pad=1))
    
    # ========== 子图4：用直线拼出圆形 ==========
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(-1.2, 1.2)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.2)
    ax4.set_facecolor('#F8F9FA')
    
    # 画一个理想的圆形（虚线）
    circle = Circle((0, 0), 0.8, fill=False, color='gray',
                   linewidth=3, linestyle='--', alpha=0.5)
    ax4.add_patch(circle)
    ax4.text(0, 0.9, '理想的圆形', ha='center', fontsize=12, color='gray')
    
    # 用不同边数的多边形逼近
    n_sides_list = [3, 4, 5, 6, 8]
    colors = [COLORS['red'], COLORS['green'], COLORS['blue'], 
             COLORS['purple'], COLORS['orange']]
    
    for i, (n_sides, color) in enumerate(zip(n_sides_list, colors)):
        radius = 0.7
        angles = np.linspace(0, 2*math.pi, n_sides + 1)
        
        # 计算多边形顶点
        vertices = []
        for angle in angles[:-1]:
            x = radius * math.cos(angle) + (i-2) * 0.3
            y = radius * math.sin(angle)
            vertices.append((x, y))
        
        vertices.append(vertices[0])  # 闭合多边形
        vertices = np.array(vertices)
        
        # 画多边形
        ax4.plot(vertices[:, 0], vertices[:, 1], '-', color=color,
                linewidth=3, alpha=0.8, marker='o', markersize=8,
                label=f'{n_sides}边形')
        
        # 计算误差
        side_length = 2 * radius * math.sin(math.pi / n_sides)
        perimeter = n_sides * side_length
        circle_perimeter = 2 * math.pi * radius
        error = abs(perimeter - circle_perimeter) / circle_perimeter * 100
        
        # 添加标签
        ax4.text((i-2) * 0.3, -0.9, f'{n_sides}边\n差{error:.1f}%',
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    # 添加生活比喻
    ax4.text(-0.6, 1.1, '3边形 ≈ 三角形', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=COLORS['light_red']))
    ax4.text(0, 1.1, '4边形 ≈ 正方形', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=COLORS['light_green']))
    ax4.text(0.6, 1.1, '8边形 ≈ 停止标志', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor=COLORS['light_blue']))
    
    ax4.set_title('🔧 用直线拼出圆形（像折纸一样）', fontsize=16,
                 fontweight='bold', pad=20, color='#2C3E50')
    ax4.legend(loc='upper right', fontsize=10)
    
    # 添加底部总结
    plt.figtext(0.5, 0.02, 
               "💡 总结：神经网络用多条直线（多边形）来近似圆形，就像用折纸做圆形一样！",
               ha='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#D4EDDA', 
                        edgecolor='#28A745', linewidth=2))
    
    plt.suptitle('第一部分：神经网络如何"思考"画圆形？', fontsize=20,
                fontweight='bold', y=0.98, color='#1A237E')
    
    plt.tight_layout()
    plt.savefig('高中生版_神经网络画圆.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

def create_animation_demo():
    """创建动画演示：多边形如何逼近圆形"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左侧：多边形逼近动画
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    ax1.set_facecolor('#F8F9FA')
    ax1.set_title('🎬 看！多边形变成圆形', fontsize=16, fontweight='bold')
    
    # 右侧：神经网络模拟
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)
    ax2.set_facecolor('#F8F9FA')
    ax2.set_title('🤖 神经网络正在学习...', fontsize=16, fontweight='bold')
    
    # 画理想圆形
    ideal_circle = Circle((0, 0), 1.0, fill=False, color='gray',
                         linewidth=3, linestyle='--', alpha=0.5)
    ax1.add_patch(ideal_circle)
    ax1.text(0, 1.15, '目标：完美圆形', ha='center', fontsize=12, color='gray')
    
    # 初始化多边形
    polygon_lines = []
    polygon_points = []
    
    def update(frame):
        """更新动画帧"""
        # 清空当前图形
        for line in polygon_lines:
            line.remove()
        for point in polygon_points:
            point.remove()
        polygon_lines.clear()
        polygon_points.clear()
        
        # 计算当前边数（从3增加到20）
        n_sides = min(3 + frame // 5, 20)
        
        # 画多边形
        angles = np.linspace(0, 2*math.pi, n_sides + 1)
        vertices = []
        
        for angle in angles[:-1]:
            x = 1.0 * math.cos(angle)
            y = 1.0 * math.sin(angle)
            vertices.append((x, y))
        
        vertices.append(vertices[0])  # 闭合
        vertices = np.array(vertices)
        
        # 画边
        line, = ax1.plot(vertices[:, 0], vertices[:, 1], '-',
                        color=COLORS['blue'], linewidth=3, alpha=0.8)
        polygon_lines.append(line)
        
        # 画顶点
        points = ax1.scatter(vertices[:-1, 0], vertices[:-1, 1],
                           color=COLORS['red'], s=50, zorder=10)
        polygon_points.append(points)
        
        # 显示边数
        info_text = f'边数：{n_sides}\n'
        if n_sides >= 8:
            info_text += '越来越圆了！'
        elif n_sides >= 5:
            info_text += '像五角星'
        else:
            info_text += '像三角形'
        
        ax1.text(0, -1.3, info_text, ha='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # 右侧：神经网络学习过程
        ax2.clear()
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.2)
        ax2.set_facecolor('#F8F9FA')
        ax2.set_title('🤖 神经网络正在学习...', fontsize=16, fontweight='bold')
        
        # 模拟神经网络的"思考"过程
        if frame < 30:
            # 阶段1：尝试画直线
            for i in range(min(frame, 8)):
                angle = i * math.pi / 4
                x = [1.2 * math.cos(angle), -1.2 * math.cos(angle)]
                y = [1.2 * math.sin(angle), -1.2 * math.sin(angle)]
                ax2.plot(x, y, color=COLORS['gray'], alpha=0.3)
            
            ax2.text(0, 0, f'第{frame}步：\n学习画直线中...',
                    ha='center', fontsize=12, color='blue')
            
        elif frame < 60:
            # 阶段2：组合成多边形
            n = (frame - 30) // 5 + 3
            angles = np.linspace(0, 2*math.pi, n + 1)
            vertices = []
            
            for angle in angles[:-1]:
                x = 0.8 * math.cos(angle)
                y = 0.8 * math.sin(angle)
                vertices.append((x, y))
            
            vertices.append(vertices[0])
            vertices = np.array(vertices)
            
            ax2.plot(vertices[:, 0], vertices[:, 1], '-',
                    color=COLORS['green'], linewidth=3, alpha=0.8)
            ax2.scatter(vertices[:-1, 0], vertices[:-1, 1],
                       color=COLORS['red'], s=30)
            
            ax2.text(0, 0, f'发现规律：\n用{n}条直线\n拼成多边形',
                    ha='center', fontsize=12, color='green',
                    bbox=dict(boxstyle='round', facecolor='white'))
            
        else:
            # 阶段3：画出两个圆
            ax2.add_patch(Circle((0, 0), 0.35, fill=False,
                               color=COLORS['red'], linewidth=4))
            ax2.add_patch(Circle((0, 0), 0.9, fill=False,
                               color=COLORS['blue'], linewidth=4))
            
            ax2.text(0, 1.2, '成功！学会了画两个圆',
                    ha='center', fontsize=14, fontweight='bold',
                    color='purple')
            ax2.text(0.3, 0.3, '小圆', color='red', fontweight='bold')
            ax2.text(0.9, 0.9, '大圆', color='blue', fontweight='bold')
        
        return []
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=100, interval=200, blit=False)
    
    # 添加说明文字
    plt.figtext(0.5, 0.02, 
               "🎯 观察：边越多，多边形越接近圆形。神经网络也是这样学习的！",
               ha='center', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存为GIF（需要安装pillow）
    try:
        anim.save('多边形逼近动画.gif', writer='pillow', fps=5)
        print("✅ 动画已保存为 '多边形逼近动画.gif'")
    except:
        print("⚠️ 无法保存GIF，请安装pillow：pip install pillow")
    
    plt.show()

def create_interactive_demo():
    """创建交互式演示：让用户体验多边形逼近"""
    while True:
        print("\n" + "="*60)
        print("🤖 神经网络画圆交互式演示")
        print("="*60)
        print("\n在这个演示中，你可以：")
        print("1. 尝试不同边数的多边形")
        print("2. 查看误差有多大")
        print("3. 理解神经网络如何思考")
        print("4. 退出程序")
        
        choice = input("\n请输入你的选择 (1-4): ").strip()
        
        if choice == '4':
            print("👋 再见！希望你喜欢这个演示！")
            break
        elif choice == '1':
            try:
                n_sides = int(input("请输入多边形的边数 (3-20): "))
                n_sides = max(3, min(20, n_sides))
                
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.2)
                ax.set_facecolor('#F8F9FA')
                
                # 画理想圆形
                ideal_circle = Circle((0, 0), 1.0, fill=False,
                                    color='gray', linewidth=4,
                                    linestyle='--', alpha=0.5)
                ax.add_patch(ideal_circle)
                ax.text(0, 1.15, '目标圆形', ha='center',
                       fontsize=14, color='gray')
                
                # 画多边形
                angles = np.linspace(0, 2*math.pi, n_sides + 1)
                vertices = []
                
                for angle in angles[:-1]:
                    x = 1.0 * math.cos(angle)
                    y = 1.0 * math.sin(angle)
                    vertices.append((x, y))
                
                vertices.append(vertices[0])
                vertices = np.array(vertices)
                
                # 画边
                ax.plot(vertices[:, 0], vertices[:, 1], '-',
                       color=COLORS['blue'], linewidth=4, alpha=0.8)
                ax.scatter(vertices[:-1, 0], vertices[:-1, 1],
                          color=COLORS['red'], s=100, zorder=10)
                
                # 计算误差
                side_length = 2 * math.sin(math.pi / n_sides)
                perimeter = n_sides * side_length
                circle_perimeter = 2 * math.pi
                error = abs(perimeter - circle_perimeter) / circle_perimeter * 100
                
                # 显示结果
                result_text = f"你用 {n_sides} 条直线画了一个 {n_sides}边形\n"
                result_text += f"误差: {error:.2f}%\n\n"
                
                if error < 5:
                    result_text += "🎉 太棒了！几乎和圆形一样！"
                elif error < 15:
                    result_text += "👍 很不错！已经很像圆形了！"
                elif error < 30:
                    result_text += "😊 还可以，继续努力！"
                else:
                    result_text += "🤔 还需要更多边才能更像圆形！"
                
                ax.text(0, -1.3, result_text, ha='center',
                       fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow',
                                alpha=0.9, pad=1))
                
                plt.title(f"🎯 {n_sides}边形 vs 圆形", fontsize=18,
                         fontweight='bold', pad=20)
                plt.tight_layout()
                plt.show()
                
            except ValueError:
                print("⚠️ 请输入一个有效的数字！")
                
        elif choice == '2':
            print("\n📊 误差分析表：")
            print("="*40)
            print("边数 | 误差% | 像什么")
            print("-"*40)
            
            examples = [
                (3, "66.3%", "三角形"),
                (4, "36.3%", "正方形"),
                (5, "24.7%", "五边形"),
                (6, "17.0%", "六边形"),
                (8, "9.7%", "八边形"),
                (12, "4.3%", "接近圆形"),
                (16, "2.4%", "很像圆形"),
                (20, "1.5%", "几乎完美")
            ]
            
            for n, error, desc in examples:
                print(f"{n:4d} | {error:6s} | {desc}")
            
            print("\n💡 神经网络的秘密：")
            print("• 用5条直线误差约25%，已经能区分形状")
            print("• 用8条直线误差<10%，足够用于分类")
            print("• 不需要完美圆形，只需要能分开数据！")
            
        elif choice == '3':
            print("\n🧠 神经网络是如何'思考'的：")
            print("="*50)
            print("\n想象神经网络是一个小机器人：")
            print("1. 它先观察数据点（红、绿、蓝）")
            print("2. 它尝试画一条直线来分开它们")
            print("3. 发现一条直线不够，尝试多条")
            print("4. 把多条直线组合成多边形")
            print("5. 调整直线位置，让多边形更圆")
            print("6. 最终学会用两个多边形（近似圆）分开所有点")
            
            print("\n📝 数学原理（简化版）：")
            print("• 每条直线：y = ax + b")
            print("• 多个直线组合：y₁ = a₁x + b₁, y₂ = a₂x + b₂, ...")
            print("• 组合起来：形成一个多边形区域")
            print("• 两个多边形：形成两个圆形区域")
            
            print("\n🎯 关键：神经网络不需要画完美的圆")
            print("只需要画得足够好来分开数据点！")
        
        else:
            print("⚠️ 请输入1-6之间的数字！")


def visualize_two_circles_approximation():
    """
    可视化：用直线逼近两个圆的过程
    展示神经网络如何分别逼近内圆和外圆
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 生成数据
    green, blue, red = create_simple_data()
    
    # ========== 第一行：展示内圆（小圆）的逼近过程 ==========
    ax_title1 = fig.add_subplot(gs[0, :])
    ax_title1.axis('off')
    ax_title1.text(0.5, 0.5, '🔴 第一个圆：用直线逼近内圆（分隔红色和绿色）', 
                   ha='center', va='center', fontsize=18, fontweight='bold',
                   color='darkred',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='#ffe6e6', 
                            edgecolor='red', linewidth=3))
    
    # 子图1：3条直线逼近内圆
    ax1 = fig.add_subplot(gs[1, 0])
    plot_circle_approximation(ax1, red, green, blue, n_lines=3, 
                             radius=0.35, title='3条直线 → 三角形', 
                             color='#FF6B6B', subplot_type='inner')
    
    # 子图2：5条直线逼近内圆
    ax2 = fig.add_subplot(gs[1, 1])
    plot_circle_approximation(ax2, red, green, blue, n_lines=5, 
                             radius=0.35, title='5条直线 → 五边形 ✓', 
                             color='#FF6B6B', subplot_type='inner')
    
    # 子图3：8条直线逼近内圆
    ax3 = fig.add_subplot(gs[1, 2])
    plot_circle_approximation(ax3, red, green, blue, n_lines=8, 
                             radius=0.35, title='8条直线 → 更像圆', 
                             color='#FF6B6B', subplot_type='inner')
    
    # ========== 第二行：展示外圆（大圆）的逼近过程 ==========
    ax_title2 = fig.add_subplot(gs[2, :])
    ax_title2.axis('off')
    ax_title2.text(0.5, 0.5, '🔵 第二个圆：用直线逼近外圆（分隔绿色和蓝色）', 
                   ha='center', va='center', fontsize=18, fontweight='bold',
                   color='darkblue',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='#e6f3ff', 
                            edgecolor='blue', linewidth=3))
    
    # 创建新的子图来展示外圆逼近（放在第三行）
    # 注意：这里我们重新调整布局
    plt.close()  # 关闭之前的图
    
    # 重新创建更合适的布局
    fig = plt.figure(figsize=(20, 14))
    
    # 主标题
    fig.suptitle('🎨 神经网络逼近两个圆的过程\n（用直线一步步画出圆形）', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    # ========== 第一部分：内圆逼近 ==========
    gs_inner = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.25,
                       left=0.05, right=0.95, top=0.90, bottom=0.52)
    
    # 标题
    ax_title = fig.add_subplot(gs_inner[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.6, '🔴 第一个圆：逼近内圆（半径 r = 0.35）', 
                 ha='center', fontsize=16, fontweight='bold', color='darkred')
    ax_title.text(0.5, 0.2, '目标：把红色点（靶心）和绿色点（靶环）分开', 
                 ha='center', fontsize=12, style='italic', color='#666')
    
    # 展示不同数量直线的逼近效果
    n_lines_list = [3, 4, 5, 8]
    descriptions = ['三角形', '四边形', '五边形 ✓（神经网络常用）', '八边形（更精确）']
    
    for idx, (n, desc) in enumerate(zip(n_lines_list, descriptions)):
        ax = fig.add_subplot(gs_inner[1, idx])
        
        # 画背景
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        # 画目标圆（虚线）
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(0.35 * np.cos(theta), 0.35 * np.sin(theta), 'r--', 
               linewidth=2, alpha=0.5, label='目标圆')
        
        # 画逼近的多边形
        polygon_x, polygon_y = create_polygon(n, 0.35)
        ax.plot(polygon_x, polygon_y, 'o-', color=COLORS['red'], 
               linewidth=3, markersize=6, label=f'{n}条直线')
        ax.fill(polygon_x, polygon_y, alpha=0.2, color=COLORS['red'])
        
        # 画数据点
        mask_red = np.sqrt(red[:, 0]**2 + red[:, 1]**2) < 0.35
        ax.scatter(red[mask_red, 0], red[mask_red, 1], 
                  color=COLORS['red'], s=80, alpha=0.8, 
                  edgecolors='white', linewidth=1.5, zorder=10)
        ax.scatter(green[:, 0], green[:, 1], 
                  color=COLORS['green'], s=50, alpha=0.4, zorder=5)
        
        # 添加说明
        ax.set_title(f'{n}条直线\n→ {desc}', fontsize=11, fontweight='bold')
        
        # 计算并显示误差
        error = calculate_polygon_error(n, 0.35)
        ax.text(0, -1.0, f'误差: {error:.1f}%', ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # ========== 第二部分：外圆逼近 ==========
    gs_outer = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.25,
                       left=0.05, right=0.95, top=0.48, bottom=0.10)
    
    # 标题
    ax_title2 = fig.add_subplot(gs_outer[0, :])
    ax_title2.axis('off')
    ax_title2.text(0.5, 0.6, '🔵 第二个圆：逼近外圆（半径 r = 0.9）', 
                  ha='center', fontsize=16, fontweight='bold', color='darkblue')
    ax_title2.text(0.5, 0.2, '目标：把绿色点（靶环）和蓝色点（外围）分开', 
                  ha='center', fontsize=12, style='italic', color='#666')
    
    for idx, (n, desc) in enumerate(zip(n_lines_list, descriptions)):
        ax = fig.add_subplot(gs_outer[1, idx])
        
        # 画背景
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        # 画目标圆（虚线）
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(0.9 * np.cos(theta), 0.9 * np.sin(theta), 'b--', 
               linewidth=2, alpha=0.5, label='目标圆')
        
        # 画逼近的多边形
        polygon_x, polygon_y = create_polygon(n, 0.9)
        ax.plot(polygon_x, polygon_y, 'o-', color=COLORS['blue'], 
               linewidth=3, markersize=6, label=f'{n}条直线')
        ax.fill(polygon_x, polygon_y, alpha=0.15, color=COLORS['blue'])
        
        # 画数据点
        ax.scatter(green[:, 0], green[:, 1], 
                  color=COLORS['green'], s=80, alpha=0.8, 
                  edgecolors='white', linewidth=1.5, zorder=10)
        mask_blue = np.sqrt(blue[:, 0]**2 + blue[:, 1]**2) > 0.9
        ax.scatter(blue[mask_blue, 0], blue[mask_blue, 1], 
                  color=COLORS['blue'], s=50, alpha=0.4, zorder=5)
        
        # 添加说明
        ax.set_title(f'{n}条直线\n→ {desc}', fontsize=11, fontweight='bold')
        
        # 计算并显示误差
        error = calculate_polygon_error(n, 0.9)
        ax.text(0, -1.0, f'误差: {error:.1f}%', ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.savefig('two_circles_approximation.png', dpi=200, bbox_inches='tight', 
                facecolor='white')
    plt.show()
    
    print("\n✅ 两个圆的逼近过程可视化已生成！")
    print("\n💡 关键发现：")
    print("• 内圆（r=0.35）：用5条直线，误差约25%")
    print("• 外圆（r=0.9）：用5条直线，误差约25%")
    print("• 两个圆同时逼近，神经网络只需要学习一组权重！")

def create_polygon(n_sides, radius):
    """创建正多边形顶点"""
    angles = np.linspace(0, 2*np.pi, n_sides + 1)
    # 计算多边形顶点到中心的距离（内切圆半径）
    r_polygon = radius * np.cos(np.pi / n_sides)
    
    x = r_polygon * np.cos(angles)
    y = r_polygon * np.sin(angles)
    return x, y

def calculate_polygon_error(n_sides, target_radius):
    """计算多边形逼近圆形的误差百分比"""
    # 误差 = (1 - cos(π/n)) * 100%
    error = (1 - np.cos(np.pi / n_sides)) * 100
    return error

def plot_circle_approximation(ax, red, green, blue, n_lines, radius, 
                             title, color, subplot_type='inner'):
    """辅助函数：绘制单个圆逼近图"""
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # 画目标圆
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), 
           '--', color=color, linewidth=2, alpha=0.5, label='目标圆')
    
    # 画逼近多边形
    polygon_x, polygon_y = create_polygon(n_lines, radius)
    ax.plot(polygon_x, polygon_y, 'o-', color=color, linewidth=2.5, markersize=5)
    ax.fill(polygon_x, polygon_y, alpha=0.2, color=color)
    
    # 画数据点
    if subplot_type == 'inner':
        ax.scatter(red[:, 0], red[:, 1], color=COLORS['red'], s=60, alpha=0.8)
        ax.scatter(green[:, 0], green[:, 1], color=COLORS['green'], s=40, alpha=0.4)
    else:
        ax.scatter(green[:, 0], green[:, 1], color=COLORS['green'], s=60, alpha=0.8)
        ax.scatter(blue[:, 0], blue[:, 1], color=COLORS['blue'], s=40, alpha=0.4)
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    # 计算误差
    error = calculate_polygon_error(n_lines, radius)
    ax.text(0, -1.0, f'误差: {error:.1f}%', ha='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))


def visualize_neuron_comparison():
    """
    详细比较两个圆的神经元分工和数学原理
    展示内圆和外圆分别由哪些神经元负责
    """
    fig = plt.figure(figsize=(20, 16))
    
    # 生成数据
    green, blue, red = create_simple_data()
    
    # ========== 第一部分：神经元分工对比 ==========
    gs1 = GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.25,
                   left=0.05, right=0.95, top=0.93, bottom=0.52)
    
    # 主标题
    fig.suptitle('🔬 神经元分工详解：两个圆是如何分别形成的？\n', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    # 子图标题
    ax_title = fig.add_subplot(gs1[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.7, '⚡ 5个神经元 = 5条直线，但如何形成两个不同的圆？', 
                 ha='center', fontsize=16, fontweight='bold', color='#2c3e50')
    ax_title.text(0.5, 0.3, '关键：每个神经元对两个圆的贡献不同！', 
                 ha='center', fontsize=13, style='italic', color='#e74c3c')
    
    # 模拟5个神经元的权重（用于可视化）
    neuron_weights = [
        {'name': '神经元1', 'inner_contrib': 0.9, 'outer_contrib': 0.3, 'color': '#e74c3c'},
        {'name': '神经元2', 'inner_contrib': 0.8, 'outer_contrib': 0.4, 'color': '#e67e22'},
        {'name': '神经元3', 'inner_contrib': 0.6, 'outer_contrib': 0.6, 'color': '#f39c12'},
        {'name': '神经元4', 'inner_contrib': 0.4, 'outer_contrib': 0.8, 'color': '#27ae60'},
        {'name': '神经元5', 'inner_contrib': 0.3, 'outer_contrib': 0.9, 'color': '#3498db'}
    ]
    
    # 绘制每个神经元的分工
    for idx, neuron in enumerate(neuron_weights):
        ax = fig.add_subplot(gs1[1, idx])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 神经元图标
        circle = Circle((0.5, 0.75), 0.12, facecolor=neuron['color'], 
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.5, 0.75, f'N{idx+1}', ha='center', va='center', 
               fontsize=14, fontweight='bold', color='white')
        
        # 内圆贡献
        inner_height = neuron['inner_contrib'] * 0.35
        rect_inner = Rectangle((0.15, 0.25), 0.25, inner_height, 
                              facecolor='#ff6b6b', edgecolor='darkred', linewidth=2)
        ax.add_patch(rect_inner)
        ax.text(0.275, 0.15, f'内圆\n{neuron["inner_contrib"]*100:.0f}%', 
               ha='center', fontsize=10, fontweight='bold', color='darkred')
        
        # 外圆贡献
        outer_height = neuron['outer_contrib'] * 0.35
        rect_outer = Rectangle((0.60, 0.25), 0.25, outer_height,
                              facecolor='#4dabf7', edgecolor='darkblue', linewidth=2)
        ax.add_patch(rect_outer)
        ax.text(0.725, 0.15, f'外圆\n{neuron["outer_contrib"]*100:.0f}%', 
               ha='center', fontsize=10, fontweight='bold', color='darkblue')
        
        # 说明文字
        if neuron['inner_contrib'] > neuron['outer_contrib']:
            role = '主要画内圆'
        elif neuron['outer_contrib'] > neuron['inner_contrib']:
            role = '主要画外圆'
        else:
            role = '两个圆都参与'
        
        ax.text(0.5, 0.05, role, ha='center', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    # ========== 第二部分：数学原理解析 ==========
    gs2 = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3,
                   left=0.05, right=0.95, top=0.48, bottom=0.05)
    
    # 子图1：内圆的数学原理
    ax1 = fig.add_subplot(gs2[0, 0])
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.set_facecolor('#fff5f5')
    
    # 画内圆
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(0.35 * np.cos(theta), 0.35 * np.sin(theta), 'r--', 
            linewidth=3, alpha=0.6, label='内圆边界')
    
    # 画内圆的5条直线（偏重前3条）
    for i in range(5):
        angle = i * 2 * np.pi / 5
        # 内圆由前3个神经元主导
        alpha = 0.9 if i < 3 else 0.3
        linewidth = 3 if i < 3 else 1.5
        
        # 计算直线
        offset = 0.35 * np.cos(np.pi / 5)
        x_line = np.linspace(-1, 1, 100)
        if abs(np.sin(angle)) > 0.01:
            y_line = (offset - np.cos(angle) * x_line) / np.sin(angle)
            valid = (y_line >= -1.2) & (y_line <= 1.2)
            ax1.plot(x_line[valid], y_line[valid], 
                    color=neuron_weights[i]['color'], 
                    alpha=alpha, linewidth=linewidth)
    
    # 画数据
    ax1.scatter(red[:, 0], red[:, 1], c=COLORS['red'], s=100, 
               alpha=0.9, edgecolors='white', linewidth=2, zorder=10)
    ax1.scatter(green[:, 0], green[:, 1], c=COLORS['green'], s=60, 
               alpha=0.5, zorder=5)
    
    ax1.set_title('🔴 内圆的形成\n（神经元1、2、3主导）', fontsize=12, fontweight='bold')
    
    # 子图2：外圆的数学原理
    ax2 = fig.add_subplot(gs2[0, 1])
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.set_facecolor('#f0f8ff')
    
    # 画外圆
    ax2.plot(0.9 * np.cos(theta), 0.9 * np.sin(theta), 'b--',
            linewidth=3, alpha=0.6, label='外圆边界')
    
    # 画外圆的5条直线（偏重后3条）
    for i in range(5):
        angle = i * 2 * np.pi / 5
        # 外圆由后3个神经元主导
        alpha = 0.3 if i < 2 else 0.9
        linewidth = 1.5 if i < 2 else 3
        
        offset = 0.9 * np.cos(np.pi / 5)
        x_line = np.linspace(-1, 1, 100)
        if abs(np.sin(angle)) > 0.01:
            y_line = (offset - np.cos(angle) * x_line) / np.sin(angle)
            valid = (y_line >= -1.2) & (y_line <= 1.2)
            ax2.plot(x_line[valid], y_line[valid],
                    color=neuron_weights[i]['color'],
                    alpha=alpha, linewidth=linewidth)
    
    # 画数据
    ax2.scatter(green[:, 0], green[:, 1], c=COLORS['green'], s=100,
               alpha=0.9, edgecolors='white', linewidth=2, zorder=10)
    ax2.scatter(blue[:, 0], blue[:, 1], c=COLORS['blue'], s=60,
               alpha=0.5, zorder=5)
    
    ax2.set_title('🔵 外圆的形成\n（神经元3、4、5主导）', fontsize=12, fontweight='bold')
    
    # 子图3：公式对比
    ax3 = fig.add_subplot(gs2[0, 2])
    ax3.axis('off')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    formula_text = """【数学公式对比】

🔴 内圆（r = 0.35）：
┌─────────────────────────┐
│ 边界条件：              │
│ f₁(x,y) = f₂(x,y)       │
│                         │
│ 其中：                  │
│ f₁ = w₁·h₁ + w₂·h₂      │
│      + w₃·h₃            │
│ f₂ = w₄·h₄ + w₅·h₅      │
│                         │
│ 前3个神经元主导！       │
└─────────────────────────┘

🔵 外圆（r = 0.9）：
┌─────────────────────────┐
│ 边界条件：              │
│ g₁(x,y) = g₂(x,y)       │
│                         │
│ 其中：                  │
│ g₁ = w₃·h₃ + w₄·h₄      │
│      + w₅·h₅            │
│ g₂ = w₁·h₁ + w₂·h₂      │
│                         │
│ 后3个神经元主导！       │
└─────────────────────────┘

💡 关键：同一组h，不同权重组合！"""
    
    ax3.text(0.5, 0.5, formula_text, transform=ax3.transAxes,
            fontsize=9.5, verticalalignment='center', horizontalalignment='center',
            family='monospace', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow',
                     edgecolor='orange', linewidth=2.5, alpha=0.9))
    ax3.set_title('📐 数学原理', fontsize=12, fontweight='bold')
    
    # 子图4：两个圆的关系
    ax4 = fig.add_subplot(gs2[1, :])
    ax4.axis('off')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    relationship = """【核心洞察：两个圆是如何共存的？】

🧩 神经网络的巧妙设计：

1️⃣ 共享神经元：5个神经元同时参与两个圆的形成
   • 每个神经元学习一条直线边界
   • 5条直线 = 1个五边形

2️⃣ 权重分工：输出层权重决定每个神经元对两个圆的贡献
   • 内圆：主要使用前3个神经元（权重高）
   • 外圆：主要使用后3个神经元（权重高）
   • 神经元3：同时参与两个圆（共享）

3️⃣ 数学本质：
   • 内圆边界：Σᵢ₌₁³ wᵢ·hᵢ = Σᵢ₌₄⁵ wᵢ·hᵢ  （前3 = 后2）
   • 外圆边界：Σᵢ₌₃⁵ wᵢ·hᵢ = Σᵢ₌₁² wᵢ·hᵢ  （后3 = 前2）

4️⃣ 直观理解：
   • 离中心近 → 前几个神经元激活 → 形成内圆
   • 离中心远 → 后几个神经元激活 → 形成外圆
   • 中间区域 → 激活程度适中 → 绿色区域

🎯 结果：用同一组神经元，通过不同的权重组合，画出两个同心圆！"""
    
    ax4.text(0.5, 0.95, relationship, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='center',
            family='monospace', linespacing=1.6,
            bbox=dict(boxstyle='round,pad=1.5', facecolor='#e8f5e9',
                     edgecolor='#4caf50', linewidth=3, alpha=0.95))
    
    plt.savefig('neuron_comparison.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.show()
    
    print("\n✅ 神经元分工对比可视化已生成！")
    print("\n💡 核心发现：")
    print("• 神经元1、2：主要画内圆（对中心区域敏感）")
    print("• 神经元4、5：主要画外圆（对外围区域敏感）")
    print("• 神经元3：两个圆都参与（中间区域）")
    print("• 同一组神经元，不同权重 → 两个不同圆！")

def main():
    """主函数"""
    print("🎨" + "="*60)
    print("           神经网络画圆 - 高中生友好版")
    print("="*60 + "🤖")
    
    print("\n欢迎来到神经网络可视化课堂！")
    print("在这里，我们将用有趣的方式学习：")
    print("• 神经网络如何'思考'")
    print("• 为什么需要画两个圆")
    print("• 如何用直线画出圆形")
    
    while True:
        print("\n" + "="*60)
        print("📚 学习菜单：")
        print("1. 📊 观看静态讲解图")
        print("2. 🎬 观看动画演示")
        print("3. 🎮 互动体验（自己尝试）")
        print("4. 📖 查看数学原理")
        print("5. ⭕ 两个圆的逼近过程（新！）")
        print("6. 🔬 神经元分工详解（新！）")
        print("7. 🚪 退出")
        
        choice = input("\n请选择学习方式 (1-7): ").strip()
        
        if choice == '1':
            print("\n正在生成讲解图...")
            draw_target_with_explanation()
            print("✅ 讲解图已生成！")
            
        elif choice == '2':
            print("\n正在生成动画...（可能需要几秒钟）")
            create_animation_demo()
            print("✅ 动画演示完成！")
            
        elif choice == '3':
            create_interactive_demo()
            
        elif choice == '4':
            print("\n📐 高中生能懂的数学原理：")
            print("="*50)
            print("\n1. 圆形公式：")
            print("   x² + y² = r²")
            print("   • r是半径")
            print("   • 点在圆上：x² + y² = r²")
            print("   • 点在圆内：x² + y² < r²")
            print("   • 点在圆外：x² + y² > r²")
            
            print("\n2. 直线公式：")
            print("   y = ax + b")
            print("   • a是斜率（倾斜程度）")
            print("   • b是截距（与y轴交点）")
            
            print("\n3. 多边形的秘密：")
            print("   • 正n边形有n条相等的边")
            print("   • 边长 = 2R × sin(π/n)")
            print("   • 周长 = n × 边长")
            print("   • 当n→∞，周长→2πR（圆的周长）")
            
            print("\n4. 神经网络的工作：")
            print("   步骤1：学习画直线 y = a₁x + b₁")
            print("   步骤2：组合多条直线")
            print("   步骤3：调整a和b，让直线围成圆形")
            print("   步骤4：用两个'圆形'分开三类数据")
            
            print("\n💡 简单来说：")
            print("神经网络就像用很多短直尺")
            print("弯成圆形来装糖果！")
            
            input("\n按回车键继续...")
            
        elif choice == '5':
            print("\n正在生成两个圆的逼近过程可视化...")
            visualize_two_circles_approximation()
            print("✅ 可视化完成！")
            
        elif choice == '6':
            print("\n正在生成神经元分工详解...")
            visualize_neuron_comparison()
            print("✅ 神经元分工详解已生成！")
            
        elif choice == '7':
            print("\n感谢使用！再见！👋")
            break
            
        else:
            print("⚠️ 请输入1-7之间的数字！")

if __name__ == '__main__':
    main()