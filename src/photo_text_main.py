import cv2
import threading
from preprocess import process_image_for_EAST, preprocess_for_paddleocr, process_image
from text_detect_Paddle import detect_identify_text
from text_detect_tesseract_EAST import detect_text_and_recognize

# 定义预处理图像路径和输出文件夹
image_path = '../images/02.07.45.png'
output_dir = '../result_images'

# 调用 preprocess.py 中的函数预处理图像
process_image(image_path, output_dir, "EAST")
process_image(image_path, output_dir, "PaddleOCR")

# 定义文本检测与识别的图像路径、输出文件夹
detect_path1 = '../result_images/preprocessed_image_paddleocr.png'
detect_path2 = '../result_images/preprocessed_image_east.png'

# 使用锁机制防止线程竞争
lock = threading.Lock()

# 定义子线程的辅助函数，用于调用子线程的检测与识别功能
def run_east_thread():
    with lock:
        # 使用锁，确保子线程安全操作
        detect_text_and_recognize(detect_path2)

# 创建并启动子线程
east_thread = threading.Thread(target=run_east_thread)
east_thread.start()

# 主线程进行文本检测与识别（PaddleOCR）
# 由于使用了独立的图像副本，不会与子线程冲突
detect_image = cv2.imread(detect_path1)
detect_identify_text(detect_image, detect_path1)

# 等待子线程完成
east_thread.join()

print("所有识别任务已完成")
