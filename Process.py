import os
import cv2
from Dehaze import dehaze_main  # 假设 dehaze_main 是从 Dehaze 模块导入的

# 输入和输出文件夹
input_dir = "./dataset/all/"
output_dir = "./images_dehazed_train/"

# 确保输出文件夹存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历输入文件夹中的图片
for filename in os.listdir(input_dir):
    # 构建输入图片路径
    input_path = os.path.join(input_dir, filename)

    # 调用 dehaze_main 函数处理图片
    dehazed_image = dehaze_main(input_path)

    # 构建输出图片路径，保持原文件名
    output_path = os.path.join(output_dir, filename)

    # 保存处理后的图片
    cv2.imwrite(output_path, cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2RGB))
    print(f"已处理并保存: {output_path}")