import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 設置中文字體支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # 優先使用的中文字體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
plt.rcParams['font.family'] = 'sans-serif'  # 使用無襯線字體

# 創建非中心化數據
x_noncentered = np.linspace(0, 10, 1000)  # 偏向正值的分佈
# 創建中心化數據
x_centered = np.linspace(-5, 5, 1000)     # 圍繞0的分佈

# 計算激活函數值
sigmoid_noncentered = 1 / (1 + np.exp(-x_noncentered))
sigmoid_centered = 1 / (1 + np.exp(-x_centered))

# 計算梯度（導數）
sigmoid_gradient_noncentered = sigmoid_noncentered * (1 - sigmoid_noncentered)
sigmoid_gradient_centered = sigmoid_centered * (1 - sigmoid_centered)

# 創建圖表
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 繪製輸入分佈直方圖
axs[0, 0].hist(x_noncentered, bins=30, alpha=0.7, color='red')
axs[0, 0].set_title('非中心化輸入分佈')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('頻率')
axs[0, 0].axvline(x=0, color='k', linestyle='--')

axs[0, 1].hist(x_centered, bins=30, alpha=0.7, color='blue')
axs[0, 1].set_title('中心化輸入分佈')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('頻率')
axs[0, 1].axvline(x=0, color='k', linestyle='--')

# 繪製Sigmoid激活及其梯度
axs[1, 0].plot(x_noncentered, sigmoid_noncentered, 'r-', label='Sigmoid')
axs[1, 0].plot(x_noncentered, sigmoid_gradient_noncentered, 'r--', label='梯度')
axs[1, 0].set_title('非中心化輸入的Sigmoid及其梯度')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('值')
axs[1, 0].axvline(x=0, color='k', linestyle='--')
axs[1, 0].axhline(y=0, color='k', linestyle='--')
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].plot(x_centered, sigmoid_centered, 'b-', label='Sigmoid')
axs[1, 1].plot(x_centered, sigmoid_gradient_centered, 'b--', label='梯度')
axs[1, 1].set_title('中心化輸入的Sigmoid及其梯度')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('值')
axs[1, 1].axvline(x=0, color='k', linestyle='--')
axs[1, 1].axhline(y=0, color='k', linestyle='--')
axs[1, 1].legend()
axs[1, 1].grid(True)

# 添加整體標題和說明
plt.suptitle('中心化的影響：改善梯度和激活函數行為', fontsize=16)
fig.text(0.5, 0.01, 
         '中心化數據使更多輸入落在激活函數的高梯度區域，\n' + 
         '減輕梯度消失問題並加速模型收斂', 
         ha='center', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('centered_vs_noncentered.png', dpi=300, bbox_inches='tight')
plt.show()