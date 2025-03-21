import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 設置中文字體支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 優先使用的中文字體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
plt.rcParams['font.family'] = 'sans-serif'  # 使用無襯線字體

# 創建數據點
x = np.linspace(-5, 5, 1000)

# 定義激活函數
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha * x, x)

# 定義導數函數
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu_derivative(x, alpha=0.1):
    return np.where(x > 0, 1, alpha)

# 計算函數值
relu_values = relu(x)
leaky_relu_values = leaky_relu(x)
relu_deriv = relu_derivative(x)
leaky_relu_deriv = leaky_relu_derivative(x)

# 創建圖表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 繪製 ReLU 函數
ax1.plot(x, relu_values, 'b-', linewidth=2.5, label='ReLU')
ax1.grid(True)
ax1.set_title('ReLU 函數', fontsize=16)
ax1.set_xlabel('x', fontsize=14)
ax1.set_ylabel('f(x) = max(0, x)', fontsize=14)
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.legend(loc='upper left')

# 繪製 Leaky ReLU 函數
ax2.plot(x, leaky_relu_values, 'r-', linewidth=2.5, label='Leaky ReLU (α=0.1)')
ax2.grid(True)
ax2.set_title('Leaky ReLU 函數', fontsize=16)
ax2.set_xlabel('x', fontsize=14)
ax2.set_ylabel('f(x) = max(αx, x), α=0.1', fontsize=14)
ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.legend(loc='upper left')

# 繪製 ReLU 導數
ax3.plot(x, relu_deriv, 'b-', linewidth=2.5, label='ReLU 導數')
ax3.grid(True)
ax3.set_title('ReLU 函數的導數', fontsize=16)
ax3.set_xlabel('x', fontsize=14)
ax3.set_ylabel('f\'(x)', fontsize=14)
ax3.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax3.set_ylim(-0.1, 1.1)
ax3.legend(loc='upper left')

# 繪製 Leaky ReLU 導數
ax4.plot(x, leaky_relu_deriv, 'r-', linewidth=2.5, label='Leaky ReLU 導數')
ax4.grid(True)
ax4.set_title('Leaky ReLU 函數的導數', fontsize=16)
ax4.set_xlabel('x', fontsize=14)
ax4.set_ylabel('f\'(x)', fontsize=14)
ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax4.set_ylim(-0.1, 1.1)
ax4.legend(loc='upper left')

# 標記差異點
ax1.fill_between([-5, 0], 0, -1, alpha=0.1, color='red')
ax1.text(-3, -0.5, '死亡區域\n梯度為0', color='red', ha='center')

ax2.fill_between([-5, 0], 0, leaky_relu(np.array([-5, 0])), alpha=0.1, color='green')
ax2.text(-3, -0.2, '漏斜率區域\n梯度為α', color='green', ha='center')

# 添加總體說明
plt.figtext(0.5, 0.01, 
           """ReLU 與 Leaky ReLU 的關鍵區別:

1. 負值區域處理:
   - ReLU: x < 0 時輸出為 0，會導致神經元死亡問題
   - Leaky ReLU: x < 0 時輸出為 αx (α 通常為 0.01~0.1)，保留小梯度避免神經元死亡

2. 導數特性:
   - ReLU: x < 0 時導數為 0，x > 0 時導數為 1，在 x = 0 處不可導
   - Leaky ReLU: x < 0 時導數為 α，x > 0 時導數為 1，也在 x = 0 處不可導

3. 優缺點比較:
   - ReLU 計算更簡單，但可能出現神經元死亡問題
   - Leaky ReLU 能緩解神經元死亡問題，但增加了模型複雜度和超參數α""", 
           ha='center', fontsize=12, 
           bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round,pad=1'))

plt.tight_layout(rect=[0, 0.15, 1, 0.95])
plt.suptitle('ReLU vs Leaky ReLU 激活函數比較', fontsize=18, y=0.98)
plt.savefig('relu_vs_leaky_relu.png', dpi=300, bbox_inches='tight')
plt.show()