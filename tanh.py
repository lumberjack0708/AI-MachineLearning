import numpy as np
import matplotlib.pyplot as plt

# 創建數據點
x = np.linspace(-5, 5, 1000)
y = np.tanh(x)

# 繪製tanh函數
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# 設置圖表
plt.title('tanh 激活函數', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('tanh(x)', fontsize=14)

# 顯示範圍線
plt.plot([-5, 5], [1, 1], 'r--', alpha=0.5)
plt.plot([-5, 5], [-1, -1], 'r--', alpha=0.5)

# 添加文本說明
plt.text(3, 0.5, r'$y = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$', fontsize=14)

plt.savefig('tanh_function.png', dpi=300, bbox_inches='tight')
plt.show()