# 项目名称

文本检测与识别项目

## 项目简介

该项目实现了从静态图像和动态GIF文件中检测并识别文本的功能。使用了EAST模型进行文本区域的检测，结合Tesseract进行文本识别，并通过PaddleOCR提供额外的OCR能力。项目同时支持处理含有中英文混合文本的图像。该系统设计模块化，方便维护和扩展，适用于文本检测和识别的相关研究和应用场景。

## 项目结构

```
plaintext
├── gif_result_images/                # 用于保存处理后的GIF图像结果
├── gif_text_output/                  # 用于保存GIF检测后的文本输出
├── gifs/                             # 存放待处理的GIF文件
├── images/                           # 存放待处理的图像文件
├── models/                           # 预训练的模型文件存放目录
│   └── frozen_east_text_detection.pb # EAST文本检测模型
├── result_images/                    # 存放检测并识别后的结果图像
├── src/                              # 源代码目录
│   ├── gif_text_main.py              # GIF文件文本检测主程序
│   ├── photo_text_main.py            # 图像文件文本检测主程序
│   ├── preprocess.py                 # 图像预处理脚本
│   ├── show_gif.py                   # GIF图像显示脚本
│   ├── text_detect_gif.py            # 针对GIF的文本检测脚本
│   ├── text_detect_Paddle.py         # 使用PaddleOCR的文本检测脚本
│   └── text_detect_tesseract_EAST.py # 使用Tesseract和EAST模型的文本检测脚本
├── text_output/                      # 存放文本检测结果
└── README.md                         # 项目说明文件
```

## 项目功能

- 从图像文件（JPG, PNG等）中检测并识别文本
- 从GIF文件中检测并识别文本
- 使用EAST模型检测文本区域
- 使用Tesseract进行文本识别
- 通过PaddleOCR进行高效的文本检测与识别
- 支持对图像进行预处理（灰度、二值化等）

### 环境配置

### 依赖库

确保你已安装以下依赖：

- Python 3.x
- OpenCV
- PaddleOCR
- Tesseract OCR
- TensorFlow
- NumPy
- Pillow

你可以使用以下命令来安装依赖：

```
pip install opencv-python-headless paddleocr tensorflow numpy pillow pytesseract
```

### Tesseract 配置

请确保本地安装了Tesseract OCR，并正确配置了环境变量。你可以通过以下命令检查Tesseract是否成功安装：

```
tesseract --version
```

如果没有安装，可以通过以下方式安装：

- **macOS**: 使用Homebrew安装

  ```
  brew install tesseract
  ```

- **Windows**: 下载并安装Tesseract [下载链接](https://github.com/tesseract-ocr/tesseract/wiki)

- **Linux**: 使用包管理器安装

  ```
  sudo apt-get install tesseract-ocr
  ```

### 模型文件

EAST模型文件 `frozen_east_text_detection.pb` 已包含在 `models/` 目录中，若需要使用其他模型，请自行下载或替换。

## 使用方法

### 1. 处理图像文件的文本检测与识别

运行 `photo_text_main.py` 来检测图像文件中的文本，并将结果保存到 `result_images/` 和 `text_output/` 目录。

#### 示例：

```
python src/photo_text_main.py --input images/sample_image.jpg --output result_images/output_image.jpg
```

**输入参数**：

- `--input`：待处理的图像文件路径
- `--output`：输出图像保存路径

检测后的文本输出会存储在 `text_output/` 文件夹中。

### 2. 处理GIF文件的文本检测与识别

运行 `gif_text_main.py` 来处理GIF文件，识别其中的文本。

#### 示例：

```
python src/gif_text_main.py --input gifs/sample.gif --output gif_result_images/output.gif
```

**输入参数**：

- `--input`：待处理的GIF文件路径
- `--output`：输出GIF文件保存路径

识别到的文本将保存在 `gif_text_output/` 文件夹中。

### 3. 使用PaddleOCR进行文本检测

通过运行 `text_detect_Paddle.py`，使用PaddleOCR模型来检测文本区域。

#### 示例：

```
python src/text_detect_Paddle.py --input images/sample_image.jpg --output result_images/paddle_output_image.jpg
```

### 4. 使用Tesseract和EAST模型进行文本检测

运行 `text_detect_tesseract_EAST.py` 使用EAST模型检测文本区域，使用Tesseract进行识别。

#### 示例：

```
python src/text_detect_tesseract_EAST.py --input images/sample_image.jpg --output result_images/tesseract_east_output_image.jpg
```

## 文件说明

- **gif_text_main.py**: 负责处理GIF文件的文本检测和识别的主程序。支持从GIF文件提取每帧图像并进行文本识别。
- **photo_text_main.py**: 负责处理静态图像文件的文本检测主程序。
- **preprocess.py**: 图像预处理模块，提供了灰度化、二值化等图像增强和清理功能，提高文本识别的准确性。
- **show_gif.py**: 提供显示GIF处理过程的功能，用于检查和展示GIF处理的效果。
- **text_detect_gif.py**: 专门为GIF文件设计的文本检测模块。
- **text_detect_Paddle.py**: 使用PaddleOCR模型来进行文本检测的脚本。
- **text_detect_tesseract_EAST.py**: 结合EAST模型和Tesseract OCR进行文本检测和识别的脚本。

## 输入与输出

### 输入

- 图像格式：JPG, PNG, BMP
- 动态GIF文件格式：GIF

### 输出

- 文本检测和识别后的图像文件：存储在 `result_images/` 或 `gif_result_images/` 目录
- 文本识别结果：存储为 `.txt` 文件，分别放置在 `text_output/` 或 `gif_text_output/` 目录

## 常见问题

1. **文本识别效果不佳怎么办？**

   1. 确保输入图像的清晰度和对比度。可以使用 `preprocess.py` 模块进行预处理。

   2. 调整Tesseract的语言包，确保包含中英文支持：

      ```
      tesseract --list-langs
      ```

   3. 尝试使用PaddleOCR进行检测。

2. **如何替换EAST模型？**

   在 `models/` 文件夹中替换 `frozen_east_text_detection.pb` 文件，并确保模型与TensorFlow兼容。

3. **GIF文件检测结果不正确？**

   GIF文件通常含有多个帧，确保每一帧中的文本都可以被识别。可以调整帧率以便获取更高质量的图像帧。

## 贡献者

- 某某洋

## 许可证

