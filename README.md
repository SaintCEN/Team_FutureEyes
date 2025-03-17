# 不谈用Tensorflow的！！！

创建两个子文件夹

datasets/all/

images_dehazed_train/

datasets/all存储所有数据集（包括训练集和测试集）

运行process.py将预处理图像存储到images_dehazed_train 

原始的预处理运行solve.py

去雾处理运行solve_dehaze.py

GPU：RTX4090 若显存不足需要降低batch_size

云端服务器怎么搞就不教了
