import cv2
import os
import time

def save_image(output_dir, img, filename):
    """
    将图像保存到指定的文件夹。
    如果文件夹不存在则创建它。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建输出文件夹
    cv2.imwrite(os.path.join(output_dir, filename), img)  # 将图像保存为文件

def convert_to_gray(image):
    """
    转换图像为灰度图像。
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 将图像从RGB转换为灰度

def apply_bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
    """
    对图像应用双边滤波来去噪。
    双边滤波可以保留边缘的同时去除噪声。
    """
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)  # 双边滤波去噪

def adaptive_threshold(image):
    """
    对图像进行自适应阈值化处理。
    将图像二值化，适合用于检测和分割图像。
    """
    return cv2.adaptiveThreshold(image, 255, cv2.THRESH_BINARY,
                                 cv2.THRESH_BINARY_INV, 15, 5)  # 二值化处理

def process_image_for_EAST(image_path, output_dir):
    """
    针对 EAST 模型的预处理流程。
    包含灰度转换、自适应阈值化处理以及双边滤波。
    """
    start_time = time.time()  # 记录开始时间
    image = cv2.imread(image_path)  # 读取输入图像

    gray = convert_to_gray(image)  # 转换为灰度图像
    adaptive_thresh = adaptive_threshold(gray)  # 应用自适应阈值化处理
    blurred = apply_bilateral_filter(adaptive_thresh)  # 使用双边滤波去噪

    save_image(output_dir, blurred, "preprocessed_image_east.png")  # 保存预处理后的图像
    total_time = time.time() - start_time  # 计算总耗时
    print(f"EAST预处理后的图片已保存到 {output_dir}")  # 输出处理时间和保存路径

def preprocess_for_paddleocr(image_path, output_dir):
    """
    针对 PaddleOCR 模型的预处理流程。
    包含灰度转换和双边滤波。
    """
    image = cv2.imread(image_path)  # 读取输入图像

    gray = convert_to_gray(image)  # 转换为灰度图像
    blurred = apply_bilateral_filter(gray)  # 使用双边滤波去噪

    save_image(output_dir, blurred, "preprocessed_image_paddleocr.png")  # 保存预处理后的图像
    print(f"PaddleOCR 预处理后的图片已保存到 {output_dir}")  # 输出处理结果

def process_image(image_path, output_dir, model_name):
    """
    根据模型名称自动选择对应的预处理方法。
    支持 EAST 和 PaddleOCR 两种模型。
    """
    if model_name == "EAST":
        process_image_for_EAST(image_path, output_dir)  # EAST 模型的预处理
    elif model_name == "PaddleOCR":
        preprocess_for_paddleocr(image_path, output_dir)  # PaddleOCR 模型的预处理
    else:
        raise ValueError("不支持的模型名称")  # 如果输入模型名称不支持，抛出错误
