import time
import cv2
import pytesseract
import numpy as np
import os
"""
EAST模型进行文本检测，Tesseract进行文本识别
"""
def load_east_model():
    # 加载预训练的 EAST 模型（frozen graph）
    net = cv2.dnn.readNet('../models/frozen_east_text_detection.pb')  # 从指定路径加载 EAST 模型
    return net  # 返回加载的模型对象

def detect_text_east(image, east_net):
    """
    使用 EAST 模型检测文本区域，并返回文本边界框。
    :param image: 输入图像
    :param east_net: 加载的 EAST 模型
    :return: 检测到的文本边界框列表
    """
    orig = image.copy()  # 复制原始图像
    (H, W) = image.shape[:2]  # 获取图像的高度和宽度

    # 设置新图像的宽高，EAST 模型要求是 32 的倍数
    newW, newH = (W // 32) * 32, (H // 32) * 32  # 将图像尺寸调整为32的倍数
    rW = W / float(newW)  # 计算宽度缩放比例
    rH = H / float(newH)  # 计算高度缩放比例

    # 调整图像大小以适应 EAST 模型的输入
    resized_image = cv2.resize(image, (newW, newH))  # 调整图像大小

    # 创建 blob 并进行前向传播，获取预测结果
    # 将图像转换为模型输入 blob
    blob = cv2.dnn.blobFromImage(resized_image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    east_net.setInput(blob)  # 设置输入
    # 进行前向传播，获取两个输出：文本得分和几何信息
    (scores, geometry) = east_net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    # 解码文本区域的边界框
    rectangles, confidences = decode_predictions(scores, geometry)

    # 应用非极大值抑制来过滤重叠的边界框
    boxes = cv2.dnn.NMSBoxes(rectangles, confidences, 0.5, 0.4)  # 过滤重叠的检测框，0.5为置信度阈值，0.4为重叠阈值

    results = []
    if len(boxes) > 0:
        for i in boxes.flatten():  # 遍历 NMS 过滤后的边界框索引
            (startX, startY, endX, endY) = rectangles[i]  # 获取边界框坐标
            # 根据缩放比例调整边界框尺寸回到原始图像大小
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            results.append((startX, startY, endX, endY))  # 添加调整后的边界框到结果列表

    return results  # 返回文本区域的边界框

def decode_predictions(scores, geometry):
    """
    将 EAST 模型的输出解码为边界框坐标和置信度分数。
    :param scores: 文本区域置信度得分
    :param geometry: 几何信息，描述文本区域的位置和尺寸
    :return: 文本区域的边界框列表和对应的置信度分数
    """
    num_rows, num_cols = scores.shape[2:4]  # 获取输出特征图的行数和列数
    rectangles = []  # 存储解码出的矩形框
    confidences = []  # 存储对应的置信度

    for y in range(num_rows):
        # 获取当前行的各项数据
        scores_data = scores[0, 0, y]  # 置信度得分
        x_data0 = geometry[0, 0, y]  # 几何信息 x_data0
        x_data1 = geometry[0, 1, y]  # 几何信息 x_data1
        x_data2 = geometry[0, 2, y]  # 几何信息 x_data2
        x_data3 = geometry[0, 3, y]  # 几何信息 x_data3
        angles_data = geometry[0, 4, y]  # 文本区域的旋转角度

        for x in range(num_cols):
            if scores_data[x] < 0.5:  # 忽略置信度小于 0.5 的区域
                continue

            # 根据几何信息计算边界框的宽和高
            offset_x = x * 4.0  # 位置偏移量 x
            offset_y = y * 4.0  # 位置偏移量 y

            angle = angles_data[x]  # 获取角度
            cos = np.cos(angle)  # 计算角度的余弦值
            sin = np.sin(angle)  # 计算角度的正弦值

            h = x_data0[x] + x_data2[x]  # 计算高度
            w = x_data1[x] + x_data3[x]  # 计算宽度

            # 计算边界框的四个角的坐标
            end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
            end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rectangles.append((start_x, start_y, end_x, end_y))  # 将边界框添加到矩形列表
            confidences.append(float(scores_data[x]))  # 将置信度添加到对应列表

    return rectangles, confidences  # 返回矩形边界框和置信度列表

def tesseract_ocr(image, box):
    """
    在指定的边界框区域使用 Tesseract 进行文本识别。
    :param image: 输入图像
    :param box: 文本区域的边界框
    :return: 识别到的文本
    """
    (startX, startY, endX, endY) = box  # 获取边界框坐标
    roi = image[startY:endY, startX:endX]  # 提取图像中的文本区域

    # 使用 Tesseract 进行文本识别
    config = '--oem 3 --psm 6 -l chi_sim+chi_tra+eng'  # 配置 Tesseract 使用中英文字库并设置识别模式
    text = pytesseract.image_to_string(roi, config=config)  # 调用 Tesseract 进行 OCR 识别
    return text  # 返回识别结果

def detect_text_and_recognize(image_path):
    """
    使用 EAST 模型进行文本检测，并使用 Tesseract 识别文本。
    :param image_path: 输入图像的路径
    """
    start_time = time.time()  # 记录开始时间
    print("EAST检测文本，Tesseract 识别文本开始")

    # 读取输入图像
    image = cv2.imread(image_path)  # 读取输入图像
    if image is None:  # 如果图像无法读取，返回
        print("无法读取图像")
        return

    # 加载 EAST 模型
    east_net = load_east_model()  # 加载 EAST 模型

    # 使用 EAST 模型检测文本区域
    boxes = detect_text_east(image, east_net)  # 检测文本区域
    print("EAST检测到的文本区域数量:", len(boxes))  # 输出检测到的文本区域数量

    detected_text = []  # 用于存储识别到的文本

    # 对每个检测到的文本区域使用 Tesseract 进行文本识别
    for idx, box in enumerate(boxes):
        text = tesseract_ocr(image, box)  # 使用 Tesseract 进行文本识别
        print(f"Tesseract识别文本 {idx + 1}: {text}")  # 输出识别到的文本
        if text:  # 如果识别到了文本
            detected_text.append(text)  # 添加到结果列表

        # 绘制边界框
        (startX, startY, endX, endY) = box  # 获取边界框坐标
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)  # 在图像上绘制边界框

    # 保存检测结果的图像
    output_image_path = '../result_images/tesseract_EAST_detected.png'  # 输出图像的保存路径
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)  # 创建保存目录（如果不存在）
    cv2.imwrite(output_image_path, image)  # 保存绘制边界框后的图像
    print(f"EAST模型检测文本边界框结果已保存到 {output_image_path}")  # 输出保存路径

    # 保存识别出的文字到文本文件
    output_text_path = '../text_output/EAST_tesseract_recognized_text.txt'  # 输出文本的保存路径
    os.makedirs(os.path.dirname(output_text_path), exist_ok=True)  # 创建保存目录（如果不存在）
    with open(output_text_path, 'w', encoding='utf-8') as f:  # 打开文件用于写入
        for line in detected_text:  # 将识别到的文本写入文件
            f.write(line + '\n')

    print(f"Tesseract识别文字结果已保存到: {output_text_path}")  # 输出保存路径
    print("EAST检测文本，Tesseract 识别文本结束")  # 输出结束信息
    end_time = time.time()  # 记录结束时间
    print(f"EAST文本检测、Tesseract识别总执行时间: {end_time - start_time:.2f} 秒")  # 输出总执行时间
