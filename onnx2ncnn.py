import subprocess

# onnx2ncnn.exe的路径
onnx2ncnn_path = r"E:\download\ncnn-20240102-windows-vs2022\x64\bin\onnx2ncnn.exe"

# 简化后的onnx模型路径
simplified_onnx_path = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes-sim.onnx"

# 转换后的ncnn模型param文件路径
ncnn_param_path = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes-sim.param"

# 转换后的ncnn模型bin文件路径
ncnn_bin_path = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes-sim.bin"

# 调用onnx2ncnn.exe
result = subprocess.run([onnx2ncnn_path, simplified_onnx_path, ncnn_param_path, ncnn_bin_path],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 打印转换结果
print(result.stdout.decode('utf-8'))
print(result.stderr.decode('utf-8'))
