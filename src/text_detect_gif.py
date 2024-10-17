import os
import cv2
import paddleocr
from PIL import Image, ImageSequence
import numpy as np
import concurrent.futures
from preprocess import save_image  # 从 preprocess 文件中导入 save_image 函数

# 初始化 PaddleOCR
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='ch', rec=True, drop_score=0.3, show_log=False)
# 初始化 PaddleOCR 模型，使用中文识别，带有角度分类，置信度过滤器设为 0.3，禁用日志输出

def convert_to_gray(image):
    """
    将图像转换为灰度图像
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 使用 OpenCV 将 RGB 图像转换为灰度图像

def apply_bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
    """
    双边滤波去噪
    """
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)  # 使用双边滤波去噪，保持图像边缘清晰

def adaptive_threshold(image):
    """
    自适应阈值化处理
    """
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)  # 对灰度图像进行自适应阈值处理，产生二值化图像

def preprocess_image(image, output_dir, filename):
    """
    对图像进行预处理，并保存预处理后的图像
    """
    gray = convert_to_gray(image)  # 转换为灰度图像
    blurred = apply_bilateral_filter(gray)  # 使用双边滤波去噪
    binary = adaptive_threshold(blurred)  # 自适应阈值化处理

    preprocessed_image_path = os.path.join(output_dir, filename)  # 生成保存路径
    save_image(output_dir, blurred, filename)  # 保存预处理后的图像

    return preprocessed_image_path  # 返回保存的图像路径

# 提取 GIF 的所有帧
def extract_frames_from_gif(gif_path):
    gif = Image.open(gif_path)  # 打开 GIF 文件
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]  # 提取 GIF 的所有帧并存储为列表
    return frames  # 返回帧列表

# 计算两帧图像的相似性
def frame_similarity(frame1, frame2):
    img1 = np.array(frame1)  # 将帧1转换为 NumPy 数组
    img2 = np.array(frame2)  # 将帧2转换为 NumPy 数组
    if img1.shape != img2.shape:  # 如果两帧的形状不同，返回无穷大作为相似度
        return float('inf')
    return np.mean((img1 - img2) ** 2)  # 计算两帧图像之间的均方误差作为相似性

# 过滤重复的帧
def filter_duplicate_frames(frames, threshold=1000):
    unique_frames = []  # 创建一个存储唯一帧的列表
    for i, frame in enumerate(frames):
        if i == 0:  # 第一帧直接添加到 unique_frames
            unique_frames.append(frame)
        else:
            similarity = frame_similarity(unique_frames[-1], frame)  # 计算与上一帧的相似度
            if similarity > threshold:  # 如果相似度超过阈值，则认为是不同帧
                unique_frames.append(frame)
    return unique_frames  # 返回唯一帧列表

def ocr_on_frame(frame, output_dir, text_output_dir, frame_index):
    """
    对单帧图片使用 PaddleOCR 进行文本检测、框选和识别，并将结果保存到单独的文本文件
    """
    try:
        # 预处理图像
        frame_np = np.array(frame)  # 将帧转换为 NumPy 数组
        preprocessed_image_path = preprocess_image(frame_np, output_dir, f"preprocessed_frame_{frame_index}.png")  # 对帧进行预处理并保存

        # 使用 OpenCV 读取预处理后的图像
        frame_cv = cv2.imread(preprocessed_image_path)  # 读取保存的预处理图像

        # 检查图像是否成功读取
        if frame_cv is None or not isinstance(frame_cv, np.ndarray):
            raise ValueError(f"无法读取预处理后的图像: {preprocessed_image_path}")  # 如果图像读取失败则抛出异常

        # 如果图像是灰度图像，转换为三通道的 BGR 格式
        if len(frame_cv.shape) == 2:
            frame_cv = cv2.cvtColor(frame_cv, cv2.COLOR_GRAY2BGR)  # 灰度转换为 BGR 格式

        # 如果图像不是三通道，抛出异常
        if frame_cv.shape[2] != 3:
            raise ValueError(f"图像通道数异常，跳过帧 {frame_index}")

        # 使用 PaddleOCR 进行文本检测与识别
        result = ocr.ocr(frame_cv, det=True, rec=True)  # 进行文本检测和识别

        # 如果OCR结果为空，直接返回空结果
        if not result or not result[0]:
            print(f"帧 {frame_index} 没有检测到文字，跳过")  # 如果未检测到文字，则跳过该帧
            return []

        # 框选并标记检测到的文本
        for line in result[0]:
            box = line[0]  # 文本框的坐标
            text = line[1][0]  # 识别的文字
            print(f"帧 {frame_index} 检测到文字: {text}")

            # 在图像上绘制文本框
            box = np.int0(box)  # 将坐标转换为整数
            cv2.polylines(frame_cv, [box], isClosed=True, color=(0, 255, 0), thickness=2)  # 在图像上绘制文本框

        # 保存带有框选的图片
        save_image(output_dir, frame_cv, f"frame_{frame_index}_with_boxes.png")  # 保存带框的图像

        # 保存该帧的 OCR 识别结果到文本文件
        recognized_text_file = os.path.join(text_output_dir, f"gif_recognized_frame{frame_index}_text.txt")
        with open(recognized_text_file, 'w', encoding='utf-8') as f:  # 打开文本文件用于写入
            recognized_text = [line[1][0] for line in result[0]]  # 提取识别到的文字
            f.write("\n".join(recognized_text))  # 写入识别到的文字

        return result  # 返回 OCR 结果

    except Exception as e:
        print(f"处理帧 {frame_index} 时跳过")  # 如果出现异常，跳过该帧
        return []

def process_gif_frames(gif_path, output_dir, text_output_dir, output_text_file, max_workers=4):
    """
    处理 GIF 文件中的帧，使用并行处理加快帧的 OCR 识别
    """
    frames = extract_frames_from_gif(gif_path)  # 提取 GIF 中的所有帧
    print(f"提取了 {len(frames)} 帧")

    # 过滤重复帧
    unique_frames = filter_duplicate_frames(frames)  # 过滤掉重复帧
    print(f"去重后剩余 {len(unique_frames)} 帧")

    recognized_texts = []  # 存储所有识别的文字结果

    # 并行处理帧的 OCR
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  # 使用多线程并行处理帧
        futures = [executor.submit(ocr_on_frame, frame, output_dir, text_output_dir, i) for i, frame in enumerate(unique_frames)]  # 提交任务给线程池

        # 将所有识别的文字写入总文件
        with open(output_text_file, 'w', encoding='utf-8') as f:  # 打开文件用于写入识别结果
            for i, future in enumerate(concurrent.futures.as_completed(futures)):  # 遍历线程执行结果
                try:
                    result = future.result()  # 获取线程执行结果
                    if result:
                        recognized_text = [line[1][0] for line in result[0]]  # 提取识别的文字
                        f.write(f"第 {i} 帧识别的文字：\n")  # 写入帧的识别信息
                        f.write("\n".join(recognized_text) + "\n\n")  # 写入识别文字并换行
                        recognized_texts.append(recognized_text)  # 保存识别结果
                except Exception as e:
                    print(f"处理帧 {i} 时跳过，错误: {e}")  # 如果出现错误，跳过该帧

    return recognized_texts  # 返回所有帧的识别结果
