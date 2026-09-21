"""
本文件实现
    理想散射体延迟叠加合成回波
    Hilbert 变换包络叠加成像
    IQ 解调相位叠加成像

变量名称
    阵元 element
    发射/接收 tx/rx
    模拟回波 simulate_rf
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from src.beamforming import das_beamform_iq_cf
from src.plotting import plot_images
from src.signal_utils import das_trace


def simulate_rf(element_pos, defect_pos, cp, n_t, fs, signal, tx_idx=None):
    """
    由散射体坐标延迟叠加生成回波信号
    要求两个输入必须是二维数组, 且最后一维必须相等
    后续应考虑拆分成两个函数，本函数只传入idx，执行计算idx之后的全部流程
    """
    if tx_idx is None:
        tx_idx = np.arange(len(element_pos))
    else:
        tx_idx = np.atleast_1d(tx_idx).astype(int)

    path = cdist(element_pos, defect_pos)
    path = path[tx_idx][:, None, :] + path[None, :, :]
    delay = path / cp
    idx = np.round(delay * fs).astype(int)  # 后续应该将这里修改为精确的插值

    j = np.arange(n_t)
    rel = j[None, None, None, :] - idx[:, :, :, None]
    valid = (rel >= 0) & (rel < len(signal))

    rf = np.where(valid, signal[np.clip(rel, 0, len(signal) - 1)], 0)
    return rf.sum(axis=-2)


def main():
    # ----- 参数 -----
    cp = 6260.0  # 声速 m/s
    fc = 5e6  # 中心频率 hz
    fs = 50e6  # 采样率 hz
    n_elements = 32  # 传感器数量
    Lx, Ly = 15.0e-3, 15.0e-3  # 成像区域 m

    # 传感器位置
    lmb = cp / fc  # 波长 m
    pitch = 0.5 * lmb  # 阵元间距 m (半波长即可)
    element_pos = np.column_stack(
        (
            np.arange(n_elements) * pitch + (Lx / 2 - (n_elements - 1) * pitch / 2),
            np.zeros(n_elements),
        )
    )

    # 缺陷位置
    defect_pos = np.array([[5.5e-3, 1.5e-3], [10.5e-3, 5.0e-3], [7.5e-3, 10.5e-3]])
    # defect_pos = np.array([[7.5e-3, 1.5e-3], [7.5e-3, 5.0e-3], [7.5e-3, 10.5e-3]])
    # defect_pos = np.array([[5.5e-3, 10.5e-3], [15.5e-3, 20.0e-3], [25.5e-3, 25.5e-3]])

    # 发射传感器
    # tx_idx = n_elements // 2
    tx_idx = np.arange(n_elements)  # FMC数据

    # 时间轴
    t_end = 2 * np.sqrt(Lx**2 + Ly**2) / cp
    t = np.arange(0, t_end, 1 / fs)

    # ----- 生成回波信号 -----
    # 正弦信号
    sine = np.sin(2 * np.pi * fc * t)

    # 高斯窗：中心在时间中点，sigma = n_cycle*T/4 使窗有效宽度约 n_cycle 个周期
    n_cycle = 2
    T = 1.0 / fc
    t_center = n_cycle * T / 2.0
    sigma = n_cycle * T / 4
    window = np.exp(-((t - t_center) ** 2) / (2 * sigma**2))

    # 加窗信号（tone burst）
    tone_burst = sine * window
    rf = simulate_rf(
        element_pos, defect_pos, cp, len(t), fs, tone_burst, tx_idx
    )  # shape(tx, rx, n_t)
    print(f"rf.shape = {rf.shape}")
    """
    # 脉冲回波
    signal = np.zeros((n_sensors, len(t)))
    pulse_len = int(1e-6 * fs)  # 1 μs 脉冲
    pulse = np.sin(2 * np.pi * fc * t[:pulse_len]) * np.hanning(
        pulse_len
    )
    """

    # ----- 相控阵空间分辨率分析 -----
    D = (n_elements - 1) * pitch  # 相控阵物理孔径
    t_HPBW = 0.886 * lmb / (n_elements * pitch)  # 主瓣宽度 rad
    t_HPBW_deg = np.degrees(t_HPBW)  # 主瓣宽度角度
    delta_x = lmb * Ly / D  # 瑞利准则下横向分辨率

    B_win = 2.0 * np.sqrt(2 * np.log(2)) / (2 * np.pi * sigma)
    delta_r = cp / (2 * B_win)  # 距离分辨率

    print(f"物理孔径 D = {D * 1e3:.4f}mm")
    print(rf"主瓣宽度 \theta_HPBW = {t_HPBW_deg:.4f}°")
    print(f"横向分辨率极限 delta_x = {delta_x * 1e3:.4f}mm ")
    print(f"径向分辨率极限 delta_r = {delta_r * 1e3:.4f}mm")

    # ----- 成像 -----
    dx = lmb / 16.0  # 最小空间分辨率
    x_arr = np.arange(0, Lx, dx)  # 空间序列 shape(L1,)
    y_arr = np.arange(0, Ly, dx)  # 空间序列 shape(L2,)

    path = das_trace(x_arr, y_arr, element_pos)
    delay = path / cp

    # 延迟合成
    tx_idx = np.atleast_1d(tx_idx)
    if tx_idx.shape == n_elements:
        delay_ = delay
    else:
        delay_ = np.concatenate((delay[tx_idx], delay), axis=0)

    # IQ 解调成像
    img = das_beamform_iq_cf(rf, delay_, t, fc, fs, t_center)

    img_norm = img / np.max(img)  # 归一化到最大值
    img_dB = 20 * np.log10(img_norm)  # 对数压缩，动态范围通常限制在 -60 ~ 0 dB
    img_dB = np.clip(img_dB, -40, 0)  # 可选：限制显示范围

    # ----- 可视化 -----
    fig3, _ = plot_images(img_norm, img_dB, defect_pos, name="IQ CF", dx=dx)

    # 可视化回波
    plt.figure()
    idx = n_elements // 4 * 3
    plt.plot(t, rf[0, idx, :], label=f"Senor{idx + 1}")
    plt.legend()
    plt.grid()
    plt.xlabel("Time / s")
    plt.show()


if __name__ == "__main__":
    main()
