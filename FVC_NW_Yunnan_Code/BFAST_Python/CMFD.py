# -*- coding: utf-8 -*-
# @Date:   2024-07-26 22:12:59
# @Last Modified time: 2024-09-01 23:41:46
# @All Rights Reserved!

import os
import rasterio
from rasterio import Affine
import bfast_main
import pandas as pd
import math  # 导入数学库，用于处理数学函数
import numpy as np  # 导入NumPy库，用于数组操作

# NDVI数据的目录路径
data_path = "data/CMFD"
# 执行的步骤（1到4）
step = 4

def read_tif(tif_file):
    """
    读取TIFF格式的影像文件，并返回像素数据、宽度、高度和元数据。

    参数：
    tif_file (str): TIFF文件的路径。

    返回:
    tuple: 包含像素数据、宽度、高度和元数据的元组。
    """
    with rasterio.open(tif_file) as src:
        data = src.read(1)  # 读取第一个波段
        width = src.width
        height = src.height
        meta = src.meta.copy()
    return (data, width, height, meta)

def get_tif_paths_by_year():
    """
    获取按年份组织的TIFF文件路径。

    返回:
    dict: 键为年份，值为对应TIFF文件的路径的字典。
    """
    file_paths = {}
    file_names = os.listdir(data_path)
    for name in file_names:
        tmp = name.split(".")
        # 检查文件是否为TIFF格式
        if len(tmp) != 2:
            continue
        if tmp[1].lower() != "tif":
            continue

        tif_path = os.path.join(data_path, name)
        # 从文件名中提取年份（文件名格式为 "CMFD_年份.tif"）
        split_name = tmp[0].split("_")
        if len(split_name) != 2:
            print(f"文件名格式不正确: {name}")
            continue
        year_str = split_name[1]
        if not year_str.isdigit():
            print(f"年份部分不是数字: {year_str} 在文件 {name}")
            continue
        year = int(year_str)
        file_paths[year] = tif_path
    return file_paths

def read_all_tifs():
    """
    读取所有年份的TIFF文件，并生成包含所有像素点时间序列的CSV文件。
    """
    file_paths = get_tif_paths_by_year()
    print("找到的文件路径:", file_paths)

    years = list(file_paths.keys())
    years = sorted(years)
    print("排序后的年份列表:", years)
    start_year = years[0]
    start_year_data, width, height, meta = read_tif(file_paths[start_year])

    # 标志位，用于调试时提前结束循环
    flag = False
    grids = []  # 存储有效的像素坐标
    datas = []  # 存储像素点的数据
    for x in range(width):
        for y in range(height):
            r = start_year_data[y, x]  # rasterio 读取的数组索引为 [行, 列] 即 [y, x]
            # 过滤无效值（小于-100、空值或NaN）
            if r < -100 or np.isnan(r):
                continue

            grids.append((x, y))
            data = [x, y, r]
            datas.append(data)

            # 如果需要，只处理第一个像素点
            # flag = True
            if flag:
                break
        if flag:
            break

    result_path = "data/all_FVC.csv"
    columns = ["x", "y", str(start_year)]
    print("列名:", columns)
    datas = pd.DataFrame(datas, columns=columns)
    # 将数据保存为CSV文件
    datas.to_csv(result_path, index=False)

    # 处理剩余年份的数据
    for year in years:
        if year == start_year:
            continue
        al_ready = pd.read_csv(result_path)
        year_data, _, _, _ = read_tif(file_paths[year])
        datas = []
        for grid in grids:
            x, y = grid
            r = year_data[y, x]  # rasterio 读取的数组索引为 [行, 列] 即 [y, x]
            # 过滤无效值
            if r < -100 or np.isnan(r):
                r = np.nan  # 使用NaN表示无效值
            datas.append([r])
        columns = [str(year)]
        datas = pd.DataFrame(datas, columns=columns)
        # 将新年份的数据与已有数据合并
        result = pd.concat([al_ready, datas], axis=1)
        result.to_csv(result_path, index=False)

def get_years():
    """
    从CSV文件中获取所有的年份列表。

    返回:
    list: 包含所有年份的列表。
    """
    year_path = "data/all_FVC.csv"
    if not os.path.exists(year_path):
        return []
    with open(year_path, "r") as f:
        line = f.readline()
        tmp = line.strip().split(",")
        # 删除前两列（x和y坐标）
        if len(tmp) <= 2:
            return []
        del tmp[0]
        del tmp[0]
        # 转换为整数年份
        try:
            years = [int(x) for x in tmp]
        except ValueError as e:
            print(f"年份转换错误: {e}")
            years = []
    return years

def ndvi_main():
    """
    对每个像素点的时间序列数据进行BFAST分析，检测趋势和突变点。
    """
    f = open("data/all_FVC.csv", "r")
    process_file = "process.txt"
    # 检查是否有进度文件，支持断点续跑
    if os.path.exists(process_file):
        with open(process_file, "r") as cur_i:
            cur_i_line = cur_i.readline()
            tmp = cur_i_line.strip()
            if tmp == "":
                skip_i = 1
            else:
                skip_i = int(tmp)
    else:
        skip_i = 1

    cur_f = open(process_file, "w")
    res_f = open("data/FVC_res.txt", "a")
    line = f.readline()  # 读取表头
    years = get_years()
    print("分析的年份:", years)

    print("跳过的行数 skip_i:", skip_i)
    i = 0
    line = f.readline()  # 从第二行开始读取数据
    while line:
        i += 1
        if i < skip_i:
            line = f.readline()
            continue

        tmp = line.strip().split(",")
        if len(tmp) < 3:
            print(f"数据行格式不正确: {line}")
            line = f.readline()
            continue
        try:
            x, y = int(tmp[0]), int(tmp[1])
            res = [float(item) if item.lower() != 'nan' else np.nan for item in tmp[2:]]
        except ValueError as e:
            print(f"数据转换错误: {e} 在行: {line}")
            line = f.readline()
            continue
        # 调用BFAST算法进行分析
        bp_cnt, bp_type, break_year = bfast_main.bfast_main(years, res)
        if break_year > 0:
            break_year = int(years[0] + break_year)
        print("bp_cnt, bp_type, break_year:", i, bp_cnt, bp_type, break_year)
        line = f.readline()

        # 将结果写入文件
        res_f.write(f"{x},{y},{bp_cnt},{bp_type},{break_year}\n")
        # 可选的中断条件（调试用）
        # if i > skip_i + 10000:
        #     break
        # if bp_cnt == 0 and bp_type == 2:
        #     break

    # 保存当前进度
    cur_f.write(str(i) + "\n")
    cur_f.close()
    f.close()
    res_f.close()


def to_tif(is_year=False):
    """
    将分析结果转换为TIFF图像。

    参数:
    is_year (bool): 如果为True，则输出突变年份图像；否则输出突变类型图像。
    """
    # 获取输入TIFF文件的元数据（假设所有文件的空间元数据相同）
    file_paths = get_tif_paths_by_year()
    if not file_paths:
        print("未找到输入的TIFF文件。")
        return

    # 选择第一个TIFF文件作为参考
    reference_tif = next(iter(file_paths.values()))
    _, width, height, meta = read_tif(reference_tif)

    # 根据是否是年份选择合适的数据类型
    if is_year:
        # 使用uint16可以存储0-65535的值，足够表示年份
        dtype = rasterio.uint16
        nodata = 0
    else:
        # 使用uint8存储突变类型
        dtype = rasterio.uint8
        nodata = 255

    # 初始化输出数组
    if is_year:
        array = np.full((height, width), nodata, dtype=np.uint16)
    else:
        array = np.full((height, width), nodata, dtype=np.uint8)

    with open("data/FVC_res.txt", "r") as res_f:
        lines = res_f.readlines()

    cnt = 0
    for line in lines:
        tmp = line.strip().split(",")
        if len(tmp) < 5:
            print(f"结果行格式不正确: {line}")
            continue
        try:
            x = int(tmp[0])
            y = int(tmp[1])
            if is_year:
                val = int(tmp[4])  # 突变年份
            else:
                val = int(tmp[3])  # 突变类型
            array[y, x] = val  # rasterio 读取的数组索引为 [行, 列] 即 [y, x]
        except ValueError as e:
            print(f"结果转换错误: {e} 在行: {line}")
            continue
        cnt += 1
        if cnt % 10000 == 0:
            print(f"已处理 {cnt} 个像素，当前值: {val}")

    # 更新元数据
    meta.update({
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": 'uint16' if is_year else 'uint8',
        "nodata": nodata
    })

    # 设置输出文件名
    output_file = 'data/FVC_year.tif' if is_year else 'data/FVC_type.tif'

    # 写入TIFF文件
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(array, 1)

    print(f"TIFF图像已保存: {output_file}")

if __name__ == '__main__':
    # 第一步：提取每个像素点的时间序列数据
    if step == 1:
        read_all_tifs()
    # 第二步：对每个像素点进行BFAST分析，计算趋势和突变点
    if step == 2:
        ndvi_main()
    # 第三步：将突变类型结果转换为TIFF图像
    if step == 3:
        to_tif()
    # 第四步：将突变年份结果转换为TIFF图像
    if step == 4:
        to_tif(True)
