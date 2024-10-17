"""
显示git图程序启动文件
"""
from tkinter import Tk, Label
from PIL import Image, ImageTk, ImageSequence

# 创建 tkinter 主窗口
root = Tk()  # 初始化主窗口
root.title("GIF 动态显示")  # 设置窗口标题为 "GIF 动态显示"

# 确保文件路径正确
image = Image.open("../gifs/subscribe-button-5244.gif")  # 打开 GIF 文件，确保文件路径正确

# 使用 ImageSequence 迭代帧
frames = [ImageTk.PhotoImage(img) for img in ImageSequence.Iterator(image)]  # 将 GIF 的每一帧转换为 tkinter 可显示的 PhotoImage 对象
frame_count = len(frames)  # 获取总帧数

# 在 tkinter 中创建标签并显示第一个帧
label = Label(root)  # 创建一个标签用于显示 GIF 帧
label.pack()  # 将标签添加到主窗口并显示

# 更新帧的函数
def update_gif(ind):
    frame = frames[ind]  # 获取当前索引的帧
    label.configure(image=frame)  # 更新标签的图像为当前帧
    root.after(100, update_gif, (ind + 1) % frame_count)  # 100 毫秒后更新到下一帧，使用模运算来循环帧数

# 启动帧更新
root.after(0, update_gif, 0)  # 在程序启动后立即调用 update_gif，从帧索引 0 开始

# 强制刷新窗口
root.update()  # 刷新窗口，使其立即生效

# 进入主循环显示窗口
root.mainloop()  # 启动 tkinter 主循环，保持窗口打开并响应事件
