"""
Digit 8 CNN Visualization Demo
==============================
Visualization of CNN convolution and pooling layers using digit "8"
Similar structure to the digit 7 demo, but with digit 8
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# Setup Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Color scheme
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


def create_digit_8_image():
    """
    Create a 8x8 digit "8" image
    1 = white (stroke), 0 = black (background)
    """
    image = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0]
    ])
    return image


def create_vertical_edge_kernel():
    """Vertical edge detection kernel"""
    return np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])


def create_horizontal_edge_kernel():
    """Horizontal edge detection kernel"""
    return np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ])


def perform_convolution(image, kernel):
    """Perform convolution operation"""
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    output_h = image_h - kernel_h + 1
    output_w = image_w - kernel_w + 1
    
    feature_map = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            window = image[i:i+kernel_h, j:j+kernel_w]
            feature_map[i, j] = np.sum(window * kernel)
    
    return feature_map


def visualize_digit_8_input():
    """Visualize the input digit 8 image"""
    image = create_digit_8_image()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display with proper colors: 0=black, 1=white
    im = ax.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Add pixel values
    for i in range(8):
        for j in range(8):
            val = int(image[i, j])
            text_color = 'white' if val == 1 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                   color=text_color, fontsize=14, fontweight='bold')
    
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(7.5, -0.5)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels([f'Col {i}' for i in range(8)], fontsize=11)
    ax.set_yticklabels([f'Row {i}' for i in range(8)], fontsize=11)
    ax.set_title('Input Image: Digit "8" (8x8 pixels)\n1=White (stroke), 0=Black (background)', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('code/cnn_visualization/10_digit8_input.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Saved: 10_digit8_input.png")


def visualize_kernels():
    """Visualize both vertical and horizontal edge detection kernels"""
    v_kernel = create_vertical_edge_kernel()
    h_kernel = create_horizontal_edge_kernel()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Vertical kernel
    ax1 = axes[0]
    im1 = ax1.imshow(v_kernel, cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest')
    ax1.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax1.grid(which='minor', color='black', linestyle='-', linewidth=2)
    
    for i in range(3):
        for j in range(3):
            val = v_kernel[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax1.text(j, i, f'{val:+d}', ha='center', va='center',
                    color=text_color, fontsize=20, fontweight='bold')
    
    ax1.set_title('Vertical Edge Detector\nKernel', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Horizontal kernel
    ax2 = axes[1]
    im2 = ax2.imshow(h_kernel, cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest')
    ax2.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax2.grid(which='minor', color='black', linestyle='-', linewidth=2)
    
    for i in range(3):
        for j in range(3):
            val = h_kernel[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax2.text(j, i, f'{val:+d}', ha='center', va='center',
                    color=text_color, fontsize=20, fontweight='bold')
    
    ax2.set_title('Horizontal Edge Detector\nKernel', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('code/cnn_visualization/11_digit8_kernels.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Saved: 11_digit8_kernels.png")


def visualize_convolution_comparison():
    """Compare vertical vs horizontal edge detection on digit 8"""
    image = create_digit_8_image()
    v_kernel = create_vertical_edge_kernel()
    h_kernel = create_horizontal_edge_kernel()
    
    v_feature_map = perform_convolution(image, v_kernel)
    h_feature_map = perform_convolution(image, h_kernel)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Original and vertical detection
    ax = axes[0, 0]
    ax.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('Original Image\nDigit "8"', fontsize=13, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax = axes[0, 1]
    im = ax.imshow(v_kernel, cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest')
    ax.set_title('Vertical Edge\nKernel', fontsize=13, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    for i in range(3):
        for j in range(3):
            val = v_kernel[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:+d}', ha='center', va='center',
                   color=text_color, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    ax = axes[0, 2]
    im = ax.imshow(v_feature_map, cmap='RdYlGn', vmin=-4, vmax=4, interpolation='nearest')
    ax.set_title('Vertical Edges\nFeature Map (6x6)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    for i in range(6):
        for j in range(6):
            val = v_feature_map[i, j]
            text_color = 'white' if abs(val) > 2 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                   color=text_color, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Row 2: Original and horizontal detection
    ax = axes[1, 0]
    ax.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title('Original Image\nDigit "8"', fontsize=13, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax = axes[1, 1]
    im = ax.imshow(h_kernel, cmap='RdBu_r', vmin=-2, vmax=2, interpolation='nearest')
    ax.set_title('Horizontal Edge\nKernel', fontsize=13, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    for i in range(3):
        for j in range(3):
            val = h_kernel[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:+d}', ha='center', va='center',
                   color=text_color, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    ax = axes[1, 2]
    im = ax.imshow(h_feature_map, cmap='RdYlGn', vmin=-4, vmax=4, interpolation='nearest')
    ax.set_title('Horizontal Edges\nFeature Map (6x6)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    for i in range(6):
        for j in range(6):
            val = h_feature_map[i, j]
            text_color = 'white' if abs(val) > 2 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                   color=text_color, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('code/cnn_visualization/12_digit8_convolution_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Saved: 12_digit8_convolution_comparison.png")


def visualize_pooling_on_digit8():
    """Visualize max pooling on digit 8 feature maps"""
    image = create_digit_8_image()
    v_kernel = create_vertical_edge_kernel()
    h_kernel = create_horizontal_edge_kernel()
    
    v_feature_map = perform_convolution(image, v_kernel)
    h_feature_map = perform_convolution(image, h_kernel)
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Max Pooling on Digit "8" Feature Maps', fontsize=18, fontweight='bold')
    
    # Vertical edge feature map and pooling
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(v_feature_map, cmap='RdYlGn', vmin=-4, vmax=4, interpolation='nearest')
    ax1.set_title('Vertical Edge\nFeature Map (6x6)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(6))
    ax1.set_yticks(range(6))
    plt.colorbar(im, ax=ax1, shrink=0.8)
    
    # Divide into pooling regions
    colors = [COLORS['blue'], COLORS['green'], COLORS['orange']]
    for idx, (i, j) in enumerate([(0, 0), (0, 3), (3, 0), (3, 3)]):
        rect = Rectangle((j-0.5, i-0.5), 3, 3, linewidth=3,
                        edgecolor=colors[idx % 3], facecolor='none', linestyle='--')
        ax1.add_patch(rect)
    
    # Vertical pooled result
    v_pooled = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            region = v_feature_map[i*3:(i+1)*3, j*3:(j+1)*3]
            v_pooled[i, j] = np.max(region)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(v_pooled, cmap='RdYlGn', vmin=-4, vmax=4, interpolation='nearest')
    ax2.set_title('After Max Pooling\n(2x2)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(2))
    ax2.set_yticks(range(2))
    for i in range(2):
        for j in range(2):
            val = v_pooled[i, j]
            text_color = 'white' if abs(val) > 2 else 'black'
            ax2.text(j, i, f'{val:.0f}', ha='center', va='center',
                    color=text_color, fontsize=18, fontweight='bold')
    plt.colorbar(im, ax=ax2, shrink=0.8)
    
    # Horizontal edge feature map and pooling
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(h_feature_map, cmap='RdYlGn', vmin=-4, vmax=4, interpolation='nearest')
    ax3.set_title('Horizontal Edge\nFeature Map (6x6)', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(6))
    ax3.set_yticks(range(6))
    plt.colorbar(im, ax=ax3, shrink=0.8)
    
    for idx, (i, j) in enumerate([(0, 0), (0, 3), (3, 0), (3, 3)]):
        rect = Rectangle((j-0.5, i-0.5), 3, 3, linewidth=3,
                        edgecolor=colors[idx % 3], facecolor='none', linestyle='--')
        ax3.add_patch(rect)
    
    # Horizontal pooled result
    h_pooled = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            region = h_feature_map[i*3:(i+1)*3, j*3:(j+1)*3]
            h_pooled[i, j] = np.max(region)
    
    ax4 = fig.add_subplot(gs[0, 3])
    im = ax4.imshow(h_pooled, cmap='RdYlGn', vmin=-4, vmax=4, interpolation='nearest')
    ax4.set_title('After Max Pooling\n(2x2)', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(2))
    ax4.set_yticks(range(2))
    for i in range(2):
        for j in range(2):
            val = h_pooled[i, j]
            text_color = 'white' if abs(val) > 2 else 'black'
            ax4.text(j, i, f'{val:.0f}', ha='center', va='center',
                    color=text_color, fontsize=18, fontweight='bold')
    plt.colorbar(im, ax=ax4, shrink=0.8)
    
    # Bottom row: Summary
    ax_summary = fig.add_subplot(gs[1, :])
    ax_summary.axis('off')
    
    summary = """
    Summary for Digit "8":
    
    Vertical Edge Detection:
    - Detects left and right edges of the digit "8"
    - Strong responses at the vertical sides of both loops
    - After pooling: 6x6 -> 2x2, keeps strongest edge responses
    
    Horizontal Edge Detection:
    - Detects top and bottom edges of the digit "8"
    - Strong responses at the top and bottom of both loops
    - After pooling: 6x6 -> 2x2, keeps strongest edge responses
    
    Key Insight:
    Different kernels detect different features. Combining multiple kernels
    (vertical + horizontal) gives a complete picture of the digit's structure!
    """
    
    ax_summary.text(0.5, 0.5, summary, fontsize=11, verticalalignment='center',
                   horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.savefig('code/cnn_visualization/13_digit8_pooling.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] Saved: 13_digit8_pooling.png")


def main():
    """Main function"""
    print("="*60)
    print("Digit 8 CNN Visualization")
    print("="*60)
    
    os.makedirs('code/cnn_visualization', exist_ok=True)
    
    print("\n[1/4] Visualizing digit 8 input...")
    visualize_digit_8_input()
    
    print("\n[2/4] Visualizing edge detection kernels...")
    visualize_kernels()
    
    print("\n[3/4] Visualizing convolution comparison...")
    visualize_convolution_comparison()
    
    print("\n[4/4] Visualizing pooling on digit 8...")
    visualize_pooling_on_digit8()
    
    print("\n" + "="*60)
    print("Digit 8 visualization completed!")
    print("="*60)
    print("\nGenerated files:")
    print("  10. 10_digit8_input.png                - Digit 8 input image")
    print("  11. 11_digit8_kernels.png              - Edge detection kernels")
    print("  12. 12_digit8_convolution_comparison.png - Vertical vs Horizontal edges")
    print("  13. 13_digit8_pooling.png              - Pooling results")


if __name__ == '__main__':
    main()
