"""
本文件对相控阵采集信号进行数据分析
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

from src.readers import check, read_with_header
from src.signal_utils import iq_demod


def check_pos(Lx, Ly, pos):
    if pos[:, 0].max() > Lx or pos[:, 0].min() < 0:
        raise ValueError("坐标 x 越界")

    if pos[:, 1].max() > Ly or pos[:, 1].min() < 0:
        raise ValueError("坐标 y 越界")

    if max(pos[:, 1]) >= 0.1 * Ly:
        print("这不是传感器坐标")


if __name__ == "__main__":
    # %%
    # ===== 导入设置 =====
    pwd = Path(__file__).parent.parent
    script_dir = pwd / "data"
    script_dir.mkdir(parents=True, exist_ok=True)
    file_path0 = script_dir / "iso_1L_5.0Mhz_32sensor_0point_20260917_105634"
    file_path1 = script_dir / "iso_1L_5.0Mhz_32sensor_4point_20260917_110326"

    # 检查文件夹
    assert check(file_path0, file_path1), "检查失败，程序退出"

    # 读取参数设置
    fc = read_with_header(file_path0 / "f0.bin")  # 中心频率
    fc = fc.item()

    lambda_min = read_with_header(file_path0 / "lambda.bin")  # 空间最小波长
    lambda_min = lambda_min.item()

    Lx = read_with_header(file_path0 / "Lx.bin")  # 空间宽度
    Ly = read_with_header(file_path0 / "Ly.bin")  # 空间深度
    Lx, Ly = Lx.item(), Ly.item()

    sensor_pos = read_with_header(file_path0 / "sensor_pos.bin")  # 传感器位置

    t_axis = read_with_header(file_path0 / "t_axis.bin")  # 时间轴
    t_axis = t_axis.squeeze()

    sensor_data0 = read_with_header(file_path0 / "sensor_data.bin")  # 完整传感器数据
    sensor_data1 = read_with_header(file_path1 / "sensor_data.bin")  # 缺陷下传感器数据
    assert sensor_data0.shape == sensor_data1.shape, "传感器数据维度不相等"

    # 检查传感器坐标位置是否匹配
    check_pos(Lx=Lx, Ly=Ly, pos=sensor_pos)

    # ===== 传感器信号数据处理 =====
    # 信号背景减除
    diff_signal = sensor_data1 - sensor_data0  # 差分信号

    ch = 0  # Python 索引从 0 开始，对应第一个传感器
    orig_peak = np.max(np.abs(sensor_data1[ch, :]))
    diff_peak = np.max(np.abs(diff_signal[ch, :]))

    print(f"原始信号峰值: {orig_peak:.4e}")
    print(f"差值信号峰值: {diff_peak:.4e}")
    print(f"比值: {diff_peak / orig_peak * 100:.2f}%")

    dt = np.median(np.diff(t_axis))  # 时间间隔 s
    fs = 1.0 / dt  # 采样率 Hz

    # 零相位滤波
    low = fc - 1.0 * fc / 5
    high = fc + 1.0 * fc / 5
    nyq = fs / 2.0  # Nyq 频率
    low = low / nyq
    high = high / nyq
    sos = butter(N=4, Wn=[low, high], btype="band", output="sos")

    data0_fir = sosfiltfilt(sos, sensor_data0)
    data1_fir = sosfiltfilt(sos, sensor_data1)
    sensor_data_fir = sosfiltfilt(sos, diff_signal)

    # IQ解调
    sig = iq_demod(sig=sensor_data_fir, time=t_axis, fc=fc, fs=fs)  # 缺陷信号

    print(f"sig.shape = {sig.shape}")

    # ===== 可视化 =====
    # 看一下频谱
    freqs = np.fft.rfftfreq(len(sensor_data0[0]), d=dt)
    amp0 = np.abs(np.fft.rfft(sensor_data0[-1]))
    amp = np.abs(np.fft.rfft(sensor_data1[-1]))
    amp1 = np.abs(np.fft.rfft(sensor_data1[-1] - sensor_data0[-1]))
    plt.figure()
    plt.plot(freqs / 1e6, amp0, label="Background")
    plt.plot(freqs / 1e6, amp, label="Signal")
    plt.plot(freqs / 1e6, amp1, label="Diff Signal")
    plt.xlim(0, 25)
    plt.xlabel("Frequency (MHz)")
    plt.legend()
    plt.grid()

    freqs = np.fft.rfftfreq(len(sensor_data0[0]), d=dt)
    amp0 = np.abs(np.fft.rfft(data0_fir[-1]))
    amp = np.abs(np.fft.rfft(data1_fir[-1]))
    amp1 = np.abs(np.fft.rfft(data1_fir[-1] - data0_fir[-1]))
    plt.figure()
    plt.plot(freqs / 1e6, amp0, label="Background", linestyle="--")
    plt.plot(freqs / 1e6, amp, label="Signal", linestyle="--")
    plt.plot(freqs / 1e6, amp1, label="Diff Signal", linestyle="--")
    plt.xlim(0, 25)
    plt.xlabel("Frequency (MHz)")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(t_axis * 1e3, diff_signal[1], label="Diff Data", linestyle="-")
    plt.plot(t_axis * 1e3, sensor_data1[1], label="Sensor", linestyle="--")
    plt.plot(t_axis * 1e3, sensor_data0[1], label="Background", linestyle="--")
    plt.plot(t_axis * 1e3, np.abs(sig[1]), label="IQ Demod Data", linestyle="-")
    plt.legend()
    plt.grid()
    plt.xlabel("Time (ms)")

    plt.show()
