import cv2
import numpy as np
from PIL import Image
'''
# 生成填充图 防止黑色/白色背景部分卷积干扰
def reflective_padding(img, background_color=[0, 0, 0]):
    h, w = img.shape[:2]
    # 图像中心坐标
    center = np.array([h // 2, w // 2])
    # 拷贝一份结果图
    result = img.copy()
    # 构造二值掩膜：True 表示对象（非背景），False 表示背景（黑色）
    binary = np.any(img != background_color, axis=2)
    # 遍历所有像素
    for y in range(h):
        for x in range(w):
            if not binary[y, x]:
                # 计算从中心到当前像素的方向向量
                direction = np.array([y, x]) - center
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue  # 当前点为中心时跳过
                direction_norm = direction / norm
                # 沿着射线从中心出发，寻找第一个背景像素（边缘）
                edge = None
                # 从 1 开始（假设中心在对象内）到当前像素距离处
                num_steps = int(np.ceil(norm))
                for step in range(1, num_steps + 1):
                    pos = center + step * direction_norm
                    pos_int = np.round(pos).astype(int)
                    # 检查是否在图像内部
                    if pos_int[0] < 0 or pos_int[0] >= h or pos_int[1] < 0 or pos_int[1] >= w:
                        break
                    # 若遇到背景（黑色）像素，则视为边缘
                    if not binary[pos_int[0], pos_int[1]]:
                        edge = pos_int
                        break
                # 如果找到了边缘，则计算反射位置
                if edge is not None:
                    mirror = 2 * edge - np.array([y, x])
                    # 检查反射点是否在图像内部
                    if 0 <= mirror[0] < h and 0 <= mirror[1] < w:
                        result[y, x] = img[mirror[0], mirror[1]]
                    else:
                        result[y, x] = background_color
    return result

def fill_black_edges(img, threshold=10, iterations=20, neighbor_size=31, k=1.0):
    # 读取图像并转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建黑色区域掩膜（阈值可根据实际情况调整）
    mask = (gray < threshold).astype(np.uint8) * 255
    # 计算邻居范围
    half_size = neighbor_size // 2
    # 迭代扩散填充
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(iterations):
        # 找到边缘像素
        edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        # 获取边缘像素的坐标
        coords = np.argwhere(edges > 0)
        # 对每个边缘像素进行颜色扩散
        for y, x in coords:
            # 获取更大范围的周围像素
            neighbors = img[max(0, y - half_size):y + half_size + 1, max(0, x - half_size):x + half_size + 1]
            # 排除黑色像素
            non_black_pixels = neighbors[(neighbors >= threshold).all(axis=2)]
            if non_black_pixels.size > 0:
                # 计算非黑色像素的平均颜色
                avg_color = np.mean(non_black_pixels, axis=0) * k
                img[y, x] = avg_color
        # 更新掩膜
        mask = cv2.erode(mask, kernel)
    return img
'''
# 自适应最小值滤波 得到暗通道图像
def adaptive_min_filter(img, k=0.03):
    h, w = img.shape[:2]  # 获取图像的高度和宽度
    short_side = min(h, w)
    radius = max(5, int(short_side * k))  # 自适应窗口尺寸半径
    kernel_size = 2 * radius + 1  # 滤波核大小
    dark_channel = cv2.erode(np.min(img, axis=2), np.ones((kernel_size, kernel_size)))  # 对 RGB 三通道的最小值滤波
    return dark_channel, radius

# 估计大气光值A 取前0.1%像素点亮度平均值 作为大气散射模型的参数
def estimate_A(dark_channel, percentile=0.1):
    flat = dark_channel.flatten() # 将暗通道展成一维 便于处理
    top_percent = np.percentile(flat, 100 - percentile) #计算百分位值
    pixels = flat[flat >= top_percent] # 取大于等于top_percent的值
    return np.mean(pixels)

# 导向滤波
def guided_filter(I, rough_t, radius, eps=1e-8):
    # 1. 计算 I 在局部窗口内的均值 mean(I)
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (radius, radius))
    # 2. 计算 rough_t 在局部窗口内的均值 mean(p)
    mean_t = cv2.boxFilter(rough_t, cv2.CV_64F, (radius, radius))
    # 3. 计算 I*I 和 I*rough_t 在局部窗口内的均值
    #    用于后续计算方差和协方差
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (radius, radius))  # mean(I^2)
    mean_It = cv2.boxFilter(I * rough_t, cv2.CV_64F, (radius, radius))  # mean(I*p)
    # 4. 计算方差 Var(I) 和协方差 Cov(I, p)
    #    Var(I) = E[I^2] - (E[I])^2
    #    Cov(I, p) = E[I*p] - E[I]*E[p]
    var_I = mean_II - mean_I * mean_I  # Var(I)
    cov_Ip = mean_It - mean_I * mean_t  # Cov(I, p)
    # 5. 根据导向滤波公式，计算线性系数 a 和 b
    #    a = Cov(I, p) / (Var(I) + eps)
    #    b = mean(p) - a * mean(I)
    a = cov_Ip / (var_I + eps)
    b = mean_t - a * mean_I
    # 6. 在局部窗口内对 a 和 b 进行平均，以保证输出结果平滑
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
    # 7. 得到最终的输出：refined_t = mean_a * I + mean_b
    #    这里与公式 q = (mean(a)) * I + (mean(b)) 相对应
    refined_t = mean_a * I + mean_b
    return refined_t

# 去雾主函数
def dehaze_retina(img):
    img_normalized = np.array(img, dtype=np.float32) / 255.0
    dark_channel, radius = adaptive_min_filter(img_normalized, k=0.03) # 计算暗通道及自适应滤波半径
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    A = estimate_A(dark_channel) # 估计大气光 A
    A = np.maximum(A, 1e-10)  # 将 A 的最小值设置为一个很小的正数，避免除以零
    #  这里先对归一化图像按通道取最小值，再进行局部最小值滤波（erode）
    dark_normalized = cv2.erode(np.min(img_normalized / A, axis=2), kernel)
    trans_coarse = 1 - 0.9 * dark_normalized  # 粗透射率

    b, g, r = cv2.split(img_normalized)
    channels = [b, g, r]
    dehazed_channels = []
    for c in channels:
        trans_refined = guided_filter(c, trans_coarse, radius=8 * radius) #导向滤波
        trans_refined = np.maximum(0.45, trans_refined) # 保证透射率不低于0.45
        J = (c - A) / trans_refined + A # 图像复原公式：J = (I - A) / T + A
        J = np.clip(J, 0, 1)
        dehazed_channels.append(J)

    # 合并三个通道得到去雾图像
    dehazed = cv2.merge(dehazed_channels)

    # 高斯滤波和 Gamma 变换进行亮度调整
    gamma = 0.7  # Gamma 变换参数
    mask = cv2.GaussianBlur((dehazed < 0.3).astype(np.float32), (31, 31), 0) #(31,31)更平滑
    brightened = dehazed ** gamma # Gamma变换
    result = dehazed * (1 - mask) + brightened * mask # 合并亮部和暗部加亮

    # 返回去雾后的图像
    return (result * 255).astype(np.uint8)

    #确定像素值阈值 调整dehazed参数 显示大部分在0.3之下
    #plt.hist(dehazed.ravel(), bins=256, range=(0, 1))
    #plt.title('Histogram of dehazed image')
    #plt.show()

def dehaze_main(img):
    img = cv2.imread(img)
    # 去雾处理
    dehazed_img = dehaze_retina(img)
    # 将图像转换为RGB格式
    dehazed_img_rgb = cv2.cvtColor(dehazed_img, cv2.COLOR_BGR2RGB)
    return dehazed_img_rgb