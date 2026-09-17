from typing import cast

import numpy as np
from scipy.signal import hilbert

from src.signal_utils import iq_demod, sinc_interp_v


def das_beamform_env(sig, delay, fs, t_center=0.0):
    """
    使用 Hilbert 变换延时叠加包络成像
    为避免过大的中间内存分配，delay输入单一阵元与空间采样点延迟
    sig.shape = [n_tx, n_rx, n_t]
    delay.shape=[L3, L2, L1]
    按发射轮次进行for循环，接收轮次使用向量化广播叠加
    输入:
        sig     阵元实信号 shape(n_tx, n_rx, n_t)
        delay   阵元空间点延迟 shape(n_tx+n_rx, L2, L1)
        t_axis  发射信号时间轴向量 shape(n_t,)
        fc      发射信号中心频率 Hz
        fs      阵元信号采样频率 Hz
        t_center    激励信号包络峰值延迟
    输出:
        image   图像包络实信号 shape(L2, L1)
    """

    n_tx, n_rx, _ = sig.shape
    sig_env = cast(np.ndarray, hilbert(sig, axis=-1))   # shape tx, rx, n_t

    if delay.shape[0] == n_tx + n_rx:
        off = n_tx  # 拼接表：rx 从 n_tx 开始
    else:
        off = 0  # 共享表：rx 从 0 开始

    image = np.zeros(delay.shape[1:], dtype=np.complex128)  # shape L2, L1
    for i in range(n_tx):
        tr_delay = delay[i] + delay[off:] + t_center
        # val = np.interp(delay.ravel(), t_axis, env[rx]).reshape(delay.shape)
        val = sinc_interp_v(sig_env[i], fs, tr_delay)   # shape rx, L2, L1
        image += np.sum(val, axis=0)
    return np.abs(image)


def das_beamform_iq(sig, delay, t_axis, fc, fs, t_center=0.0):
    """
    使用 IQ 解调延时叠加成像
    输入:
        sig     阵元实信号 shape(n_tx, n_rx, n_t)
        delay   阵元空间点延迟 shape(n_tx+n_rx, L2, L1)
        t_axis  发射信号时间轴向量 shape(n_t,)
        fc      发射信号中心频率 Hz
        fs      阵元信号采样频率 Hz
        t_center    激励信号包络峰值延迟
    输出:
        image   图像包络实信号 shape(L2, L1)
    """
    n_tx, n_rx, _ = sig.shape
    sig_iq = iq_demod(sig, t_axis, fc, fs)  # shape tx, rx, n_t

    if delay.shape[0] == n_tx + n_rx:
        off = n_tx  # 拼接表：rx 从 n_tx 开始
    else:
        off = 0  # 共享表：rx 从 0 开始

    image = np.zeros(delay.shape[1:], dtype=np.complex128)
    for i in range(n_tx):
        tr_delay = delay[i] + delay[off:] + t_center
        val = sinc_interp_v(sig_iq[i], fs, tr_delay)
        phase = 2.0 * np.pi * fc * tr_delay
        image += np.sum(val * np.exp(1j * phase), axis=0)
    return np.abs(image)


def das_beamform_iq_cf(sig, delay, t_axis, fc, fs, t_center=0.0):
    """
    使用 IQ 解调延时叠加成像, 使用相位因子 CF 修正
    输入:
        sig     阵元实信号 shape(n_tx, n_rx, n_t)
        delay   阵元空间点延迟 shape(n_tx+n_rx, L2, L1)
        t_axis  发射信号时间轴向量 shape(n_t,)
        fc      发射信号中心频率 Hz
        fs      阵元信号采样频率 Hz
        t_center    激励信号包络峰值延迟
    输出:
        image   图像包络实信号 shape(L2, L1)
    """
    n_tx, n_rx, _ = sig.shape
    sig_iq = iq_demod(sig, t_axis, fc, fs)  # shape tx, rx, n_t

    if delay.shape[0] == n_tx + n_rx:
        off = n_tx  # 拼接表：rx 从 n_tx 开始
    else:
        off = 0  # 共享表：rx 从 0 开始

    image = np.zeros(delay.shape[1:], dtype=np.complex128)
    image_abs = np.zeros(delay.shape[1:], dtype=np.float64)
    for i in range(n_tx):
        tr_delay = delay[i] + delay[off:] + t_center
        val = sinc_interp_v(sig_iq[i], fs, tr_delay)
        phase = 2.0 * np.pi * fc * tr_delay
        image += np.sum(val * np.exp(1j * phase), axis=0)
        image_abs += np.sum(np.abs(val * np.exp(1j * phase)), axis=0)
    cf = np.abs(image) / (image_abs + 1e-12)
    return np.abs(image) * cf


def das_beamform_mvdr():
    """
    不用做了。因为本来就是理想回波信号，不需要抑制什么其他方向回波。
    只要当出现不能背景减除，需要抑制非目标方向波束时才需要使用。
    """
