import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 設置中文字體支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 優先使用的中文字體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
plt.rcParams['font.family'] = 'sans-serif'  # 使用無襯線字體

# 創建數據點
x = np.linspace(-10, 10, 1000)
sigmoid = 1 / (1 + np.exp(-x))

# 計算導數/梯度
sigmoid_derivative = sigmoid * (1 - sigmoid)

# 創建圖表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 繪製Sigmoid函數
ax1.plot(x, sigmoid, 'b-', linewidth=2.5)
ax1.grid(True)
ax1.set_title('Sigmoid 函數', fontsize=16)
ax1.set_xlabel('x', fontsize=14)
ax1.set_ylabel('σ(x) = 1/(1+e^(-x))', fontsize=14)
ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='y=0.5')
ax1.axhline(y=1.0, color='g', linestyle='--', alpha=0.7, label='y=1.0 (漸近線)')
ax1.axhline(y=0.0, color='g', linestyle='--', alpha=0.7, label='y=0.0 (漸近線)')
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# 標記特殊點
ax1.plot(0, 0.5, 'ro', markersize=5)
ax1.annotate('(0, 0.5)', xy=(0, 0.5), xytext=(1, 0.6),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1))

# 添加水平線標記範圍
ax1.fill_between(x, 0, sigmoid, alpha=0.1, color='blue')
ax1.legend(loc='lower right')

# 繪製Sigmoid導數/梯度
ax2.plot(x, sigmoid_derivative, 'r-', linewidth=2.5)
ax2.grid(True)
ax2.set_title('Sigmoid 函數的導數', fontsize=16)
ax2.set_xlabel('x', fontsize=14)
ax2.set_ylabel('σ\'(x) = σ(x)(1-σ(x))', fontsize=14)
ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# 標記最大導數點
ax2.plot(0, 0.25, 'ro', markersize=5)
ax2.annotate('最大值: (0, 0.25)', xy=(0, 0.25), xytext=(2, 0.2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1))

# 添加Sigmoid函數特性說明
plt.figtext(0.5, 0.01, 
           """Sigmoid 函數特性:
1. 輸出範圍: (0,1) - 適合表示概率
2. 在原點處導數最大 (0.25)
3. 具有飽和效應 - 輸入絕對值較大時，輸出接近0或1，梯度接近0
4. 非零中心化 - 輸出總是正的 (>0)
5. 計算簡單，處處可導
6. 缺點: 存在梯度消失問題，計算量較大 (包含指數運算)""", 
           ha='center', fontsize=12, 
           bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round,pad=1'))

plt.tight_layout(rect=[0, 0.15, 1, 0.95])
plt.suptitle('Sigmoid 激活函數及其導數', fontsize=18, y=0.98)
plt.savefig('sigmoid_function_analysis.png', dpi=300, bbox_inches='tight')
plt.show()