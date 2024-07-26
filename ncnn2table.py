import subprocess
import os

ncnn2table_path = "E:\\download\\ncnn-20240102-windows-vs2022\\x64\\bin\\ncnn2table.exe"
param_path = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes-sim.param"
bin_path = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes-sim.bin"
image_list_path = r"E:\cityscapes-quant\calibration_list.txt"  # 使用图像列表文件路径
table_path = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes.table"

mean_values = "103.53,116.28,123.675"
norm_values = "0.003922,0.003922,0.003922"
size = "640,640,3"
num_threads = "15"

# 调整命令格式，去掉"--"前缀
cmd = [
    ncnn2table_path,
    param_path,
    bin_path,
    image_list_path,
    table_path,
    "mean=" + mean_values,
    "norm=" + norm_values,
    "shape=" + size,
    "pixel=BGR",
    "thread=" + num_threads
]
#
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

if process.returncode == 0:
    print("量化表生成成功:", table_path)
else:
    print("量化表生成失败")
    print("Error message:", stderr.decode("utf-8"))
