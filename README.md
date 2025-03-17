# 不谈用Tensorflow的！！！

datasets/all/

images_dehazed_train/

datasets/alls存储所有数据集（包括训练集和测试集）

运行process.py将预处理图像存储到images_dehazed_train.然后运行solve_dehazed_image

如果用原始的预处理只需运行solve.py.

注意在官网提交SaintCHEN_ODIR.csv时需要删除第二列和第三列

GPU：RTX4090 若显存不足需要降低batch_size
