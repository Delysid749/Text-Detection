"""
动态git图识别启动文件
"""
from text_detect_gif import process_gif_frames

# 设置路径和输出文件夹
gif_path = "../gifs/congratulations-2228_512.gif"
output_dir = "../gif_result_images"
text_output_dir = "../gif_text_output"
output_text_file = "../gif_text_output/recognized_texts.txt"

# 调用处理函数
process_gif_frames(gif_path, output_dir, text_output_dir, output_text_file)
