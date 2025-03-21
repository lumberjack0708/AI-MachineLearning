import numpy as np
import matplotlib.pyplot as plt

# 創建數據點
x = np.linspace(-5, 5, 1000)

# 計算各種激活函數值
tanh = np.tanh(x)
sigmoid = 1 / (1 + np.exp(-x))
relu = np.maximum(0, x)

# 對於Softmax，在一維情況下將其簡化
# 注意：這只是為了視覺比較，真正的Softmax用於多類分類問題
def softmax_simplified(x):
    # 在一維情況下，Softmax會將每個輸入轉換為相對概率
    # 這裡我們簡化為一個S形曲線，大致反映其特性
    return np.exp(x) / (1 + np.exp(x))

softmax = softmax_simplified(x)

# 創建圖表
plt.figure(figsize=(12, 8))

# 繪製各函數
plt.plot(x, tanh, 'b-', linewidth=2, label='tanh')
plt.plot(x, sigmoid, 'g-', linewidth=2, label='Sigmoid')
plt.plot(x, relu, 'r-', linewidth=2, label='ReLU')
plt.plot(x, softmax, 'm-', linewidth=2, label='Softmax (簡化)')

# 添加網格和坐標軸
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# 添加水平線標記重要值
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)

# 設置圖表
plt.title('常見激活函數比較', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.legend(fontsize=12)
plt.ylim(-1.5, 5)

# 添加說明
plt.text(-4.8, 4.5, r'函數特性:', fontsize=12)
plt.text(-4.8, 4.0, r'Sigmoid: 輸出範圍(0,1)', fontsize=10)
plt.text(-4.8, 3.5, r'tanh: 輸出範圍(-1,1)', fontsize=10)
plt.text(-4.8, 3.0, r'ReLU: 輸出範圍[0,∞)', fontsize=10)
plt.text(-4.8, 2.5, r'Softmax: 用於多類分類', fontsize=10)

# 標記函數特點
plt.annotate('飽和區域', xy=(-3, sigmoid[-3]), xytext=(-3.5, 0.3),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))
plt.annotate('線性增長', xy=(3, relu[800]), xytext=(2, 4),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))

plt.tight_layout()
plt.savefig('activation_functions_comparison.png', dpi=300, bbox_inches='tight')
plt.show()