"""
本文件计算窄带阵列方向图及点扩散响应灵敏度
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
# 定义一个均匀线阵
# 求解其窄带阵列方向图
cp = 6260  # 介质中固体声速 m/s
fc = 5.0e6  # 中心频率 Hz
lambda_c = cp / fc  # 波长 m
N = 64  # 阵元个数
d = lambda_c / 4  # 阵元间距 m
# 不满足半波长空间采样定理会发生混叠 产生两个方向的栅瓣

dx = lambda_c / 16  # 空间分辨率 m
Lx = 30.0e-3  # 水平范围 m
Ly = 30.0e-3  # 垂直范围 m

Lx_arr = np.arange(0, Lx, dx)
Ly_arr = np.arange(0, Ly, dx)

theta = np.linspace(-np.pi / 2.0, np.pi / 2.0, 181)  # 角度范围 (L1,)
w = np.ones(N)  # 阵元权重 (N,)

# 波束形成器指向角度 theta_0 = 0
theta_0 = np.pi / 3
n = np.arange(N)  # 阵列元素
dt = n * d / cp * np.sin(theta_0)  # 阵元延时

k = 2 * np.pi / lambda_c  # 波数
u_diff = np.sin(theta) - np.sin(theta_0)  # shape (L1, )
phase = k * d * n[:, None] * u_diff[None, :]  # shape(N, L1)
B = np.sum(w[:, None] * np.exp(1j * phase), axis=0)

B_mag = np.abs(B)
B_mag = B_mag / np.max(B_mag)  # 归一化

# 极坐标显式
fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(111, projection="polar")
ax.plot(theta, B_mag)

# 直角坐标显示
theta_deg = np.rad2deg(theta)  # 弧度转角度，横轴好看
B_dB = 20 * np.log10(B_mag + 1e-12)  # 转成 dB

plt.figure(figsize=(6, 6))
plt.plot(theta_deg, B_dB, linewidth=2)
plt.grid(True, linestyle="--")
plt.xlabel("Angle θ (degrees)")
plt.ylabel("Normalized Beampattern (dB)")
plt.ylim(-40, 0)  # 只显示 -40dB 以上，看清旁瓣
plt.xlim(-90, 90)
plt.title("Array Pattern (Broadside)")
plt.show()

# %%
# 分析相控阵孔径及探测灵敏度
