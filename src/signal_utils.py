import numpy as np
from numba import njit, prange
from scipy import signal
from scipy.spatial.distance import cdist


def das_trace(x_array, y_array, s_pos):
    """
    das 全局声路径计算函数
    计算均一各向同性介质下点s位置到空间坐标几何距离

    输入:
        x_array x方向一维序列 (L1,)
        y_array y方向一维序列 (L2,)
        s_pos   点s位置 (L3,2)
    输出:
        dis     几何距离 (L3, L2, L1)
    """

    xx, yy = np.meshgrid(x_array, y_array)  # shape (L2, L1)
    xy_pos = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # shape (L1*L2, 2)
    return cdist(s_pos, xy_pos).reshape(
        -1, len(y_array), len(x_array)
    )  # shape (L3, L2, L1)


def sinc_interp(signal: np.ndarray, fs: float, t: np.ndarray, L=8, beta=6.0):
    """
    sinc 插值函数
    使用 Kaiser 窗截断的 sinc 插值
    超出范围设计为镜像延拓

    输入:
        signal  原始信号一维序列 调整 shape(L3, N)
        fs      signal 信号采样率
        t       插值时刻 任意形状张量 调整 (L2, L1)
        L       Kaiser 窗半长   默认8
        beta    Kaiser 窗参数   默认6.0
    输出:
        y       插值点处值
    """
    t = np.asarray(t)
    shape = t.shape
    t_flat = t.ravel()
    N = len(signal)  # N = signal.shape[-1]

    n0 = np.round(t_flat * fs).astype(int)
    offsets = np.arange(-L, L + 1)

    idx = n0[:, None] + offsets[None, :]
    idx_mirror = np.where(idx < 0, -idx, idx)
    idx_mirror = np.where(idx_mirror >= N, 2 * N - 2 - idx_mirror, idx_mirror)

    x = signal[idx_mirror]
    u = (t_flat[:, None] - idx / fs) * fs
    s = np.sinc(u)

    # Kaiser 窗
    w = np.kaiser(2 * L + 1, beta)

    y_flat = np.sum(x * s * w, axis=1)
    return y_flat.reshape(shape)


@njit(parallel=True, cache=True, boundscheck=False)
def __sinc_interp_kernel(signal, fs, t_flat, w, L, out):
    """
    sinc 插值函数 Numba高性能内核
    """
    n_rx, N = signal.shape
    _, M = t_flat.shape
    K = 2 * L + 1

    for rx_idx in prange(n_rx):
        zero = signal[rx_idx, 0] - signal[rx_idx, 0]
        for t_idx in range(M):
            tr = t_flat[rx_idx, t_idx]
            n0 = round(tr * fs)
            acc = zero
            for k in range(K):
                i = n0 - L + k
                i_m = -i if i < 0 else i
                i_m = 2 * N - 2 - i_m if i_m >= N else i_m
                u = tr * fs - i
                acc += signal[rx_idx, i_m] * np.sinc(u) * w[k]
            out[rx_idx, t_idx] = acc


def sinc_interp_v(signal: np.ndarray, fs: float, t: np.ndarray, L=8, beta=6.0):
    """
    sinc 插值函数
    使用 Kaiser 窗截断的 sinc 插值
    超出范围设计为镜像延拓

    输入:
        signal  原始信号序列 shape(rx, N)
        fs      signal 信号采样率
        t       插值时刻 shape(rx, L2, L1)
        L       Kaiser 窗半长   默认8
        beta    Kaiser 窗参数   默认6.0
    输出:
        y       插值点处值 shape(rx, L2, L1)
    """

    n_rx, _ = signal.shape
    L2, L1 = t.shape[1], t.shape[2]
    M = L2 * L1
    t_2d = np.ascontiguousarray(t.reshape(n_rx, M))  # (n_rx, M)

    # Kaiser 窗
    w = np.kaiser(2 * L + 1, beta)

    # Numba加速的插值点生成
    out_type = signal.dtype
    y = np.empty((n_rx, M), dtype=out_type)
    __sinc_interp_kernel(signal, fs, t_2d, w, L, y)

    return y.reshape(n_rx, L2, L1)


def iq_demod(sig, time, fc, fs, order=4):
    """
    IQ 解调函数
    signal 信号IQ混频后低通滤波, 还原复信号

    输入:
        sig     原始信号序列 (N, L1)
        time    sig 信号对应时间一维序列 (L1,)
        fc      sig 信号对应中心频率 Hz
        fs      sig 信号采样频率 Hz
        order   低通滤波器阶数 默认4
    输出
        signal_demod 解调后复信号 (N, L1)
    """

    assert time.ndim == 1, "时间不是一维序列"
    assert sig.shape[-1] == time.shape[0], "信号与时间形状不对应"

    It = sig * np.cos(2 * np.pi * fc * time)
    Qt = -sig * np.sin(2 * np.pi * fc * time)

    nyquist = fs / 2
    cutoff_freq = fc * 1.2
    normal_cutoff = cutoff_freq / nyquist  # 归一化频率

    # 设计巴特沃斯低通滤波器 (SOS格式，数值稳定性好)
    sos = signal.butter(order, normal_cutoff, btype="low", output="sos")

    # 零相位滤波（正向+反向），有效去除相位失真
    It = signal.sosfiltfilt(sos, It)
    Qt = signal.sosfiltfilt(sos, Qt)

    return It + Qt * (0 + 1j)


def main():

    import matplotlib.pyplot as plt

    # ===== 测试 sinc 插值 =====
    fs = 5.0e6  # 采样频率 Hz
    fc1, fc2 = 1.0e6, 0.9e6  # 信号频率 1Mhz, 900khz
    fa1, fa2 = fc1 / 5, fc2 / 3  # 包络频率 200khz, 300khz

    t = np.arange(0.0, 1000 / fc1, 1.0 / fs)
    sig1 = np.sin(2 * np.pi * fa1 * t) * np.sin(2 * np.pi * fc1 * t + np.pi / 3)
    sig2 = np.sin(2 * np.pi * fa2 * t) * np.sin(2 * np.pi * fc2 * t + np.pi / 3)
    sig = sig1 + sig2

    fs1 = 20.0e6  # 上采样频率 Hz
    t_interp = np.arange(0.0, 1000 / fc1, 1.0 / fs1)
    sig_interp = sinc_interp(signal=sig, fs=fs, t=t_interp)

    X_sig = np.fft.fft(sig)  # 复数频谱
    X_sig_interp = np.fft.fft(sig_interp)
    freq = np.fft.fftfreq(len(sig), d=1 / fs)  # 频率轴
    freq_interp = np.fft.fftfreq(len(sig_interp), d=1 / fs1)  # 频率轴

    # 单边频谱（幅度）
    X_mag = np.abs(X_sig[: len(X_sig) // 2]) / len(X_sig)
    X_mag[1:] = X_mag[1:] * 2
    X_mag_interp = np.abs(X_sig_interp[: len(X_sig_interp) // 2]) / len(X_sig_interp)
    X_mag_interp[1:] = X_mag_interp[1:] * 2
    freq_single = freq[: len(freq) // 2]
    freq_interp_signle = freq_interp[: len(freq_interp) // 2]

    # ===== 可视化信号幅值 =====
    num = 31  # Python 切片左闭右开
    q = np.round(fs1 / fs).astype(int)

    plt.figure(figsize=(7, 7))
    plt.plot(t[0 : num + 1] * 1.0e6, sig[0 : num + 1], label="Origin Signal")
    plt.plot(
        t_interp[0 : q * num + 1] * 1.0e6,
        sig_interp[0 : q * num + 1],
        label="Sinc Interp Signal",
    )
    # plt.plot(t_interp * 1.e6, abs, label="IQ Demod Signal")
    plt.xlabel(r"Time/$\mu s$")
    plt.ylabel("Amp")
    plt.legend()
    plt.grid(True)

    # ===== 可视化信号频谱 =====
    plt.figure(figsize=(7, 7))
    plt.plot(freq_single / 1.0e6, X_mag, "--", marker="o", label="Origin Signal")
    plt.plot(freq_interp_signle / 1.0e6, X_mag_interp, label="Sinc Interp Signal")
    plt.xlim(0, 2.0)
    plt.xlabel(r"Freq/$Mhz$")
    plt.ylabel("Amp")
    plt.legend()
    plt.grid(True)

    # ===== 测试 IQ 解调 =====
    f = 1000.0  # 正弦频率 (Hz)
    n_cycle = 5  # 高斯窗覆盖的周期数
    N = 20  # 总信号长度（周期数），应远大于 N
    fs = 10000  # 采样率 (Hz)
    T = 1 / f  # 周期 (秒)
    t_end = N * T  # 总时间

    # 时间轴：从0到 total_cycles * T
    t = np.linspace(0, t_end, int(t_end * fs) + 1)
    # 正弦信号
    sine = np.sin(2 * np.pi * f * t)

    # 高斯窗：中心在时间中点，sigma = n_cycle*T/4 使窗有效宽度约 n_cycle 个周期
    t_center = n_cycle * T / 2.0
    sigma = n_cycle * T / 4
    window = np.exp(-((t - t_center) ** 2) / (2 * sigma**2))

    # 加窗信号（tone burst）
    tone_burst = sine * window
    abs = iq_demod(sig=tone_burst, time=t, fc=f, fs=fs)
    abs = np.abs(abs)

    # 绘图
    plt.figure(figsize=(10, 4))
    plt.plot(t, sine, "--", label="Original sine", alpha=0.5)
    plt.plot(t, window, ":", label="Gaussian window", alpha=0.7)
    plt.plot(t, tone_burst, label="Tone burst")
    plt.plot(t, abs, label="IQ demod")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Tone Burst: {f} Hz, {N} cycles Gaussian window, total {n_cycle} cycles")
    plt.legend()
    plt.grid(True)

    # ===== 测试 DAS FMC 延迟叠加 =====
    fc = 1.0e6  # 频率 Hz
    cp = 6260  # 纵波声速 m/s
    dx = cp / (fc * 8)  # 空间分辨率 1/4波长

    Lx = 30.0e-3  # 横向空间距离
    Ly = 30.0e-3  # 纵向空间距离

    Lx_array = np.arange(0, Lx, dx)
    Ly_array = np.arange(0, Ly, dx)

    n_elements = 8
    # 传感器位置
    lmb = cp / fc  # 波长 m
    pitch = 0.5 * lmb  # 阵元间距 m (半波长即可)
    element_pos = np.column_stack(
        (
            np.arange(n_elements) * pitch + (Lx / 2 - (n_elements - 1) * pitch / 2),
            np.zeros(n_elements),
        )
    )  # shape(n, 2)

    dist = das_trace(Lx_array, Ly_array, element_pos)  # shape (n, L2, L1)

    # 取第一个阵元激发 全部阵元接收
    dist = dist / cp
    dist1, dist2 = dist[0], dist[-1]  # shape (L1*L2,)

    img = dist1 + dist2

    plt.figure(figsize=(10, 7))
    # plt.imshow(img, cmap="jet", aspect="auto", origin="upper")
    plt.contour(img, colors="black", origin="upper", linewidths=0.5)
    plt.contourf(img, cmap="coolwarm", origin="upper")
    plt.colorbar()  # 显示颜色条
    plt.xlabel("gird x")
    plt.ylabel("grid y")

    plt.show()


if __name__ == "__main__":
    main()
