import numpy as np
import matplotlib.pyplot as plt

# 創建數據點
x = np.linspace(-5, 5, 1000)
y = np.maximum(0, x)  # ReLU 函數: max(0, x)

# 繪製ReLU函數
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'r-', linewidth=2)
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# 設置圖表
plt.title('ReLU 激活函數', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('ReLU(x)', fontsize=14)

# 添加文本說明
plt.text(2, 3, r'$ReLU(x) = \max(0, x)$', fontsize=14)
plt.text(-4, 1, r'$ReLU(x) = 0, x < 0$', fontsize=12)
plt.text(1, 4, r'$ReLU(x) = x, x \geq 0$', fontsize=12)

# 標記轉折點
plt.plot(0, 0, 'bo', markersize=6)
plt.annotate('轉折點 (0,0)', xy=(0, 0), xytext=(0.5, 1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

plt.ylim(-0.5, 5)
plt.savefig('relu_function.png', dpi=300, bbox_inches='tight')
plt.show()