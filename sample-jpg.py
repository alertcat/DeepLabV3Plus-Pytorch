import os
import random
import shutil

def random_sample(image_dir, num_samples, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 获取所有图片文件
    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 随机选择指定数量的图片
    selected_images = random.sample(all_images, min(num_samples, len(all_images)))

    sample_paths = []
    for img in selected_images:
        src_path = os.path.join(image_dir, img)
        dst_path = os.path.join(output_dir, img)
        shutil.copy(src_path, dst_path)
        sample_paths.append(src_path)

    return sample_paths

# 设置路径和参数
image_dir = r"E:\cityscapes-yolo\images\train"
output_dir = r'E:\cityscapes-quant'
num_samples = 400

# 执行随机抽样
sampled_paths = random_sample(image_dir, num_samples, output_dir)

# 创建校准列表
calibration_list = os.path.join(output_dir, 'calibration_list.txt')
with open(calibration_list, 'w') as f:
    for path in sampled_paths:
        f.write(path + '\n')

print(f'Calibration images copied to: {output_dir}')
print(f'Calibration image list saved to: {calibration_list}')
print(f'Total images sampled: {len(sampled_paths)}')