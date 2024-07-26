'''
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torch.quantization import get_default_qconfig, quantize_jit
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes


def load_model(model_path):
    model = deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=21)
    checkpoint = torch.load(model_path, map_location=torch.device('cuda'))
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model


def prepare_data_loader(data_path, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    target_transform = transforms.Compose([
        transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    dataset = Cityscapes(data_path, split='train', mode='fine',
                         target_type='semantic',
                         transform=transform,
                         target_transform=target_transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader


def prepare_model_for_quantization(model):
    model.eval()
    model.qconfig = get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    return model


def calibrate_model(model, data_loader, num_batches=100):
    print("开始校准模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for i, (images, _) in enumerate(data_loader):
        images = images.to(device)
        with torch.no_grad():
            model(images)
        if i >= num_batches:
            break
    print("模型校准完成。")


def quantize_model(model):
    print("开始量化模型...")
    quantized_model = torch.quantization.convert(model.cpu(), inplace=True)
    print("模型量化完成。")
    return quantized_model


def get_model_size(model):
    torch.save(model.state_dict(), "temp_model.pth")
    size = os.path.getsize("temp_model.pth") / (1024 * 1024)  # Size in MB
    os.remove("temp_model.pth")
    return size


def main():
    model_path = r"E:\download\best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
    data_path = r"E:\cityscapes"

    # 加载原始模型
    model = load_model(model_path)
    print("原始模型加载完成")

    # 获取原始模型大小
    original_size = get_model_size(model)
    print(f"原始模型大小: {original_size:.2f} MB")

    # 准备数据加载器
    data_loader = prepare_data_loader(data_path)

    # 准备模型进行量化
    model = prepare_model_for_quantization(model)

    # 使用Cityscapes数据集校准模型
    calibrate_model(model, data_loader)

    # 完成量化
    quantized_model = quantize_model(model)

    # 获取量化后的模型大小
    quantized_size = get_model_size(quantized_model)
    print(f"量化后模型大小: {quantized_size:.2f} MB")

    # 保存量化后的模型
    torch.save(quantized_model.state_dict(), r"E:\download\static_quantized_deeplabv3plus_mobilenet.pth")
    print("量化后的模型已保存")

    print(f"大小减少: {(original_size - quantized_size) / original_size:.2%}")


if __name__ == "__main__":
    main()
'''
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torch.quantization import get_default_qconfig, quantize_jit
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes


def load_model(model_path):
    model = deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=21)
    checkpoint = torch.load(model_path, map_location=torch.device('cuda'))
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def prepare_data_loader(data_path, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    target_transform = transforms.Compose([
        transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])

    dataset = Cityscapes(data_path, split='train', mode='fine',
                         target_type='semantic',
                         transform=transform,
                         target_transform=target_transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader


def prepare_model_for_quantization(model):
    model.eval()
    model.qconfig = get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    return model


def calibrate_model(model, data_loader, num_batches=100):
    print("开始校准模型...")
    model = model.to('cpu')
    for i, (images, _) in enumerate(data_loader):
        with torch.no_grad():
            model(images)
        if i >= num_batches:
            break
    print("模型校准完成。")


def quantize_model(model):
    print("开始量化模型...")
    quantized_model = torch.quantization.convert(model, inplace=False)
    print("模型量化完成。")
    return quantized_model


def get_model_size(model):
    torch.save(model.state_dict(), "temp_model.pth")
    size = os.path.getsize("temp_model.pth") / (1024 * 1024)  # Size in MB
    os.remove("temp_model.pth")
    return size


def compare_model_structures(model1, model2):
    print("比较模型结构：")

    def get_layers(model):
        return {name: module for name, module in model.named_modules() if not list(module.children())}

    layers1 = get_layers(model1)
    layers2 = get_layers(model2)

    print(f"原始模型层数: {len(layers1)}")
    print(f"量化模型层数: {len(layers2)}")

    if len(layers1) != len(layers2):
        print("警告：层数不同！")

    for name in layers1.keys():
        if name not in layers2:
            print(f"警告：量化模型中缺少层 {name}")
        elif not isinstance(layers2[name], type(layers1[name])):
            print(f"警告：层 {name} 的类型发生了变化")
            print(f"  原始: {type(layers1[name])}")
            print(f"  量化后: {type(layers2[name])}")

    for name in layers2.keys():
        if name not in layers1:
            print(f"警告：量化模型中新增了层 {name}")


def main():
    model_path = r"E:\download\best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
    data_path = r"E:\cityscapes"

    # 加载本地模型
    model = load_model(model_path)
    print("本地模型加载完成")

    # 获取原始模型大小
    original_size = get_model_size(model)
    print(f"原始模型大小: {original_size:.2f} MB")

    # 准备数据加载器
    data_loader = prepare_data_loader(data_path)

    # 准备模型进行量化
    model = prepare_model_for_quantization(model)

    # 使用Cityscapes数据集校准模型
    calibrate_model(model, data_loader)

    # 完成量化
    quantized_model = quantize_model(model)

    # 比较模型结构
    compare_model_structures(original_model, quantized_model)

    # 获取量化后的模型大小
    quantized_size = get_model_size(quantized_model)
    print(f"量化后模型大小: {quantized_size:.2f} MB")

    # 保存量化后的模型
    torch.save(quantized_model.state_dict(), r"E:\download\static_quantized_deeplabv3plus_mobilenet.pth")
    print("量化后的模型已保存")

    print(f"大小减少: {(original_size - quantized_size) / original_size:.2%}")


if __name__ == "__main__":
    main()