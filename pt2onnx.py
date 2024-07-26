import network
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
import os
import re


dirname, filename = os.path.split(os.path.abspath(__file__))
print(dirname)

def get_pytorch_onnx_model(original_model):
    # define the directory for further converted model save
    onnx_model_path = dirname
    # define the name of further converted model
    onnx_model_name = "deeplabv3plus_mobilenet_cityscapes.onnx"

    # create directory for further converted model
    os.makedirs(onnx_model_path, exist_ok=True)

    # get full path to the converted model
    full_model_path = os.path.join(onnx_model_path, onnx_model_name)

    # generate model input
    generated_input = Variable(
        torch.randn(1, 3, 513, 513)
    )

    # model export into ONNX format
    torch.onnx.export(
        original_model,
        generated_input,
        full_model_path,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )

    return full_model_path


model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=19, output_stride=8)
checkpoint = torch.load("best_deeplabv3plus_mobilenet_cityscapes_os16.pth", map_location=torch.device('cuda'))
model.load_state_dict(checkpoint["model_state"])
full_model_path = get_pytorch_onnx_model(model)