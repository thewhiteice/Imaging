import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.beamforming import das_beamform_iq_cf
from src.readers import check, read_with_header
from src.signal_utils import das_trace

if __name__ == "__main__":
    # ===== 导入设置 =====
    pwd = Path(__file__).parent.parent
    script_dir = pwd / "data"
    script_dir.mkdir(parents=True, exist_ok=True)
    file_path0 = script_dir / "iso_1L_5.0Mhz_32sensor_0point_20260917_105634"
    file_path1 = script_dir / "iso_1L_5.0Mhz_32sensor_4point_20260917_110326"

    result_dir = pwd / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    assert check(file_path0, file_path1), "检查失败，程序退出"  # 检查文件夹

    # 读取参数设置
    cp = read_with_header(file_path0 / "cp.bin")  # 纵波波速
    cp = cp.item()

    fc = read_with_header(file_path0 / "f0.bin")  # 中心频率
    fc = fc.item()

    lambda_min = read_with_header(file_path0 / "lambda.bin")  # 空间最小波长
    lambda_min = lambda_min.item()

    Lx = read_with_header(file_path0 / "Lx.bin")  # 空间宽度
    Ly = read_with_header(file_path0 / "Ly.bin")  # 空间深度
    Lx, Ly = Lx.item(), Ly.item()

    element_pos = read_with_header(file_path0 / "sensor_pos.bin")  # 传感器位置

    t_axis = read_with_header(file_path0 / "t_axis.bin")  # 时间轴
    t_axis = t_axis.squeeze()

    data_o0 = read_with_header(file_path0 / "sensor_data.bin")  # 完整传感器数据
    data_o1 = read_with_header(file_path1 / "sensor_data.bin")  # 缺陷下传感器数据
    assert data_o0.shape == data_o1.shape, "传感器数据维度不相等"

    start = time.perf_counter()

    # ===== 传感器信号数据处理 =====
    lmb = cp / fc  # 波长 m
    dt = np.median(np.diff(t_axis))  # 时间间隔 s
    fs = 1.0 / dt  # 采样率 Hz
    n_cycle = 2
    T = 1.0 / fc
    t_center = n_cycle * T / 2.0  # 波包中心位置 s
    tx_idx = 0
    n_elements = element_pos.shape[0]

    # 信号背景减除
    rf = data_o1 - data_o0

    if rf.ndim == 2:
        rf = rf[None, ...]
    print(f"rf.shape={rf.shape}")

    # 零相位滤波
    low = fc - 1.0 * fc / 5
    high = fc + 1.0 * fc / 5
    nyq = fs / 2.0  # Nyq 频率
    low = low / nyq
    high = high / nyq
    sos = butter(N=4, Wn=[low, high], btype="band", output="sos")
    sensor_data_fir = sosfiltfilt(sos, rf)

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
    img = das_beamform_iq_cf(rf, delay_, t_axis, fc, fs, t_center)

    img_norm = img / np.max(img)  # 归一化到最大值
    img_dB = 20 * np.log10(img_norm)  # 对数压缩，动态范围通常限制在 -60 ~ 0 dB
    img_dB = np.clip(img_dB, -40, 0)  # 可选：限制显示范围

    # ----- 可视化 -----
    data_list = [img_norm, img_dB]
    name = "IQ CF"
    titles = [f"{name} Fig", f"{name} Fig dB"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title in zip(axes, data_list, titles):
        im = ax.imshow(data, cmap="jet", aspect="auto", origin="upper")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)

    fig.tight_layout()
    plt.show()
