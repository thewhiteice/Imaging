"""
要写一个读取.bin的函数
"""

from pathlib import Path

import numpy as np


def read_with_header(filename):
    """
    读取 filename 文件夹下.bin文件
    格式:
        维度 数据
    按列优先恢复为原始形状
    """
    with open(filename, "rb") as f:
        ndims = np.fromfile(f, dtype=np.int32, count=1)[0]
        dims = np.fromfile(f, dtype=np.int32, count=ndims)
        total = np.prod(dims)
        data = np.fromfile(f, dtype=np.float64, count=total)
        # 按列优先（Fortran顺序）重塑为原始形状
        return data.reshape(dims, order="F")


def check(file1, file2):
    """
    自动检查两个文件夹下除 sensor_data.bin, fmc_data.bin 外所有同名数据文件
    防止设置出错
    """
    path1 = Path(file1)
    path2 = Path(file2)

    files1 = {f.name: f for f in path1.iterdir() if f.is_file()}
    files2 = {f.name: f for f in path2.iterdir() if f.is_file()}

    common = sorted(set(files1.keys()) & set(files2.keys()))
    mismatched = []

    for name in common:
        if name == "sensor_data.bin":
            continue

        if name == "fmc_data.bin":
            continue

        f1, f2 = files1[name], files2[name]

        if f1.stat().st_size != f2.stat().st_size:
            mismatched.append(name)
            continue

        # 分块逐字节对比
        with open(f1, "rb") as a, open(f2, "rb") as b:
            while True:
                chunk1 = a.read(65536)
                chunk2 = b.read(65536)
                if chunk1 != chunk2:
                    mismatched.append(name)
                    break
                if not chunk1:
                    break

    if mismatched:
        print("不一致的文件:", mismatched)
        return False

    # print(f"检查通过，共 {len(common) - 1} 个文件一致")
    return True


def main():
    script_dir = Path(__file__).parent.parent / "data"
    script_dir.mkdir(parents=True, exist_ok=True)
    file_path = script_dir / "iso_1L_5.0Mhz_6sensor_0point_20260831_203818"

    sensor_data_p = read_with_header(file_path / "sensor_data_p.bin")
    print(sensor_data_p.shape)

    sensor_pos = read_with_header(file_path / "sensor_pos.bin")
    print(sensor_pos.shape)

    t_axis = read_with_header(file_path / "t_axis.bin")
    print(t_axis.shape)

    print(file_path)


if __name__ == "__main__":
    main()
