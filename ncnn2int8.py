import subprocess

# 指定ncnn2int8.exe的路径
ncnn2int8_path = r"E:\download\ncnn-20240102-windows-vs2022\x64\bin\ncnn2int8.exe"

# 指定输入的bin文件、param文件和table文件路径
bin_file = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes-sim.bin"
param_file = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes-sim.param"
table_file = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes.table"

# 指定输出的int8格式的bin文件和param文件路径  
output_param_file = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes-int8.param"
output_bin_file = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes-int8.bin"

# 构建ncnn2int8.exe的命令行参数
cmd_args = [
    ncnn2int8_path,
    param_file,
    bin_file,
    output_param_file, 
    output_bin_file,
    table_file
]

# 使用subprocess.run()执行ncnn2int8.exe
result = subprocess.run(cmd_args, capture_output=True, text=True)

# 打印ncnn2int8.exe的输出结果
print(result.stdout)

# 检查执行是否成功
if result.returncode == 0:
    print("转换成功！")
else:
    print(f"转换失败，错误码: {result.returncode}")