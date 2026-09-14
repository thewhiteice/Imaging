import matplotlib.pyplot as plt


def plot_images(image_norm, image_dB, defect_pos, name, dx=1.0, figsize=(14, 6)):
    """
    可视化成像图（线性 + dB）并标注缺陷位置。
    参数
        image_norm : 2D ndarray  归一化后的图像
        image_dB   : 2D ndarray  dB 图像
        defect_pos : ndarray (N, 2)  缺陷坐标（物理坐标，除以 dx 转像素索引）
        dx         : float 网格间距
        figsize    : tuple 图像尺寸

    返回
        fig : matplotlib.figure.Figure
        axes : ndarray of Axes, 形状 (2,)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    data_list = [image_norm, image_dB]
    titles = [f"{name} Fig", f"{name} Fig dB"]

    for ax, data, title in zip(axes, data_list, titles):
        im = ax.imshow(data, cmap="jet", aspect="auto", origin="upper")
        fig.colorbar(im, ax=ax)
        for i in range(defect_pos.shape[0]):
            ax.plot(
                defect_pos[i, 0] / dx,
                defect_pos[i, 1] / dx,
                "wx",
                markersize=10,
                markeredgewidth=4,
            )
        ax.set_title(title)

    fig.tight_layout()
    return fig, axes
