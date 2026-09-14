import numpy as np


def simulate_rf(delay_idx, n_t: int, signal):
    """
    由散射体时间延迟索引生成理想回波信号
    输入:
        delay_idx   三维延迟序列索引 shape(n_tx, n_rx, n_defect)
        n_t         标量时间索引长度 int
        signal      一维激励信号序列 shape(n1,)
    输出:
        rf          三维回波信号张量 shape(n_tx, n_rx, n_t)
    """
    j = np.arange(n_t)
    rel = j[None, None, None, :] - delay_idx[:, :, :, None]
    valid = (rel >= 0) & (rel < len(signal))

    rf = np.where(valid, signal[np.clip(rel, 0, len(signal) - 1)], 0)
    return rf.sum(axis=-2)
