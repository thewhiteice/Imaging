from typing import cast

import numpy as np
from scipy.signal import hilbert

from src.signal_utils import iq_demod, sinc_interp


def das_beamform_env(sig, delay, fs):
    """
    使用 Hilbert 变换延时叠加包络成像
    为避免过大的中间内存分配，delay输入单一阵元与空间采样点延迟
    sig.shape = [n_tx, n_rx, n_t]
    delay.shape=[L3, L2, L1]
    按发射轮次进行for循环，接收轮次使用向量化广播叠加
    """

    n_tx, n_rx, _ = sig.shape
    # sig_env = np.abs(cast(np.ndarray, hilbert(sig, axis=-1)))  # shape tx, rx, n_t
    sig_env = cast(np.ndarray, hilbert(sig, axis=-1))

    if delay.shape[0] == n_tx + n_rx:
        off = n_tx  # 拼接表：rx 从 n_tx 开始
    else:
        off = 0  # 共享表：rx 从 0 开始

    image = np.zeros(delay.shape[1:], dtype=np.complex128)
    for i in range(n_tx):
        tr_delay = delay[i] + delay[off:]
        for j in range(n_rx):
            # val = np.interp(delay.ravel(), t_axis, env[rx]).reshape(delay.shape)
            val = sinc_interp(sig_env[i][j], fs, tr_delay[j])
            image += val
    return np.abs(image)


def das_beamform_iq(sig, delay, t_axis, fc, fs):
    """
    使用 IQ 解调延时叠加成像
    """
    n_tx, n_rx, _ = sig.shape
    sig_iq = iq_demod(sig, t_axis, fc, fs)  # shape tx, rx, n_t

    if delay.shape[0] == n_tx + n_rx:
        off = n_tx  # 拼接表：rx 从 n_tx 开始
    else:
        off = 0  # 共享表：rx 从 0 开始

    image = np.zeros(delay.shape[1:], dtype=np.complex128)
    for i in range(n_tx):
        tr_delay = delay[i] + delay[off:]
        for j in range(n_rx):
            # val = np.interp(delay.ravel(), t_axis, env[rx]).reshape(delay.shape)
            val = sinc_interp(sig_iq[i][j], fs, tr_delay[j])
            phase = 2.0 * np.pi * fc * tr_delay[j]
            image += val * np.exp(1j * phase)
    return np.abs(image)


def das_beamform_iq_cf(sig, delay, t_axis, fc, fs):
    """
    使用 IQ 解调延时叠加成像
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
        tr_delay = delay[i] + delay[off:]
        for j in range(n_rx):
            # val = np.interp(delay.ravel(), t_axis, env[rx]).reshape(delay.shape)
            val = sinc_interp(sig_iq[i][j], fs, tr_delay[j])
            phase = 2.0 * np.pi * fc * tr_delay[j]
            image += val * np.exp(1j * phase)
            image_abs += np.abs(val * np.exp(1j * phase))
    cf = np.abs(image) / (image_abs + 1e-12)
    return np.abs(image) * cf
