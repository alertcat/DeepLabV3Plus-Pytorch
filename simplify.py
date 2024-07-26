import onnx
from onnxsim import simplify

# 指定输入的onnx模型路径
input_path = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes.onnx"
# 指定简化后输出的onnx模型路径
output_path = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes-sim.onnx"
# 加载onnx模型
onnx_model = onnx.load(input_path)

# 简化onnx模型
model_simp, check = simplify(onnx_model)

# 检查简化后的模型是否有效
assert check, "Simplified ONNX model could not be validated"

# 保存简化后的onnx模型
onnx.save(model_simp, output_path)

print("ONNX model simplified successfully!")

