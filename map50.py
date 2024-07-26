import ncnn
import numpy as np
import cv2
import os
import json
from collections import defaultdict

def load_model(param_path, bin_path):
    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)
    return net

def preprocess_image(image_path, target_size=(513, 513)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image file: {image_path}")
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def run_inference(net, image):
    mat_in = ncnn.Mat.from_pixels_resize(image, ncnn.Mat.PixelType.PIXEL_RGB, image.shape[1], image.shape[0], 513, 513)
    ex = net.create_extractor()
    ex.input("input", mat_in)
    ret, mat_out = ex.extract("output")
    return mat_out

def process_output(output, original_width, original_height):
    output_array = np.array(output)
    num_classes, height, width = output_array.shape

    # 获取每个像素的最高概率类别
    segmentation = np.argmax(output_array, axis=0)

    # 将分割图调整到原始图像大小
    segmentation = cv2.resize(segmentation, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    return segmentation

def load_cityscapes_annotation(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def evaluate_iou(prediction, ground_truth, num_classes=19):
    iou_per_class = []

    for class_id in range(num_classes):
        pred_mask = (prediction == class_id)
        gt_mask = (ground_truth == class_id)

        if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0:
            continue  # 跳过此类别，因为预测和地面实况中都没有此类别

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        iou = intersection / union if union > 0 else 0
        iou_per_class.append(iou)

    return np.mean(iou_per_class) if len(iou_per_class) > 0 else 0

def evaluate_model(net, image_dir, annotation_dir):
    total_iou = 0
    processed_images = 0
    class_ious = defaultdict(list)

    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            if filename.endswith('_leftImg8bit.png'):
                image_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, image_dir)
                json_filename = filename.replace('_leftImg8bit.png', '_gtFine_polygons.json')
                json_path = os.path.join(annotation_dir, relative_path, json_filename)

                if not os.path.exists(json_path):
                    print(f"Warning: Annotation file not found for {filename}")
                    continue

                try:
                    image = preprocess_image(image_path)
                except FileNotFoundError as e:
                    print(f"Warning: {e}")
                    continue

                output = run_inference(net, image)

                original_image = cv2.imread(image_path)
                height, width = original_image.shape[:2]

                segmentation = process_output(output, width, height)

                annotation_data = load_cityscapes_annotation(json_path)
                ground_truth = np.zeros((height, width), dtype=np.uint8)

                for obj in annotation_data['objects']:
                    if 'label' in obj and obj['label'] in cityscapes_labels:
                        class_id = cityscapes_labels.index(obj['label'])
                        polygon = np.array(obj['polygon'], dtype=np.int32)
                        cv2.fillPoly(ground_truth, [polygon], class_id)

                total_intersection = 0
                total_union = 0
                for class_id in range(len(cityscapes_labels)):
                    pred_mask = (segmentation == class_id)
                    gt_mask = (ground_truth == class_id)

                    intersection = np.logical_and(pred_mask, gt_mask).sum()
                    union = np.logical_or(pred_mask, gt_mask).sum()

                    total_intersection += intersection
                    total_union += union

                    if union > 0:
                        iou = intersection / union
                        class_ious[class_id].append(iou)

                image_iou = total_intersection / total_union if total_union > 0 else 0
                total_iou += image_iou
                processed_images += 1
                print(f"Processed image {processed_images}: {filename}, IoU: {image_iou:.4f}")

    mean_iou = total_iou / processed_images if processed_images > 0 else 0
    print(f"\nTotal processed images: {processed_images}")
    print(f"Mean IoU: {mean_iou:.4f}")

    print("\nClass-wise IoU:")
    valid_class_ious = []
    for class_id, ious in class_ious.items():
        if ious:
            class_mean_iou = np.mean(ious)
            valid_class_ious.append(class_mean_iou)
            print(f"Class {class_id} ({cityscapes_labels[class_id]}): {class_mean_iou:.4f}")

    overall_mean_iou = np.mean(valid_class_ious) if valid_class_ious else 0
    print(f"\nOverall Mean IoU: {overall_mean_iou:.4f}")

    return mean_iou


# 主执行部分
param_path = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes-int8.param"
bin_path = r"E:\download\DeepLabV3Plus-Pytorch-master\deeplabv3plus_mobilenet_cityscapes-int8.bin"
test_image_dir = r"E:\cityscapes\leftImg8bit\val"
test_annotation_dir = r"E:\cityscapes\gtFine\val"

cityscapes_labels = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

net = load_model(param_path, bin_path)

mean_iou = evaluate_model(net, test_image_dir, test_annotation_dir)
print(f"Final Mean IoU: {mean_iou}")