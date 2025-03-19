# Gemini Playground

个人用于调戏 `gemini-2.0-flash-exp` 的脚本

## 脚本概述

### `background_generator.py`

![image](https://github.com/user-attachments/assets/b9f1e85d-a87c-411b-8070-ffb8efdf9346)


- **目的**：根据用户提示和图像上下文为图像生成新背景。
- **功能**：
  - 简单模式，用于直接生成背景。
  - 上下文模式，根据图像描述生成背景。
  - 正常模式，生成包含指定元素的背景。

### `multi_generator.py`

![image](https://github.com/user-attachments/assets/f33596f4-b793-4c7d-b9f3-5932979ff289)


- **目的**：用相同 prompts 生成多张图像，测试 Gemini 的稳定性
- **功能**：
  - 允许指定会话数量和每个会话生成的图像数量。
  - 使用 Gemini API 进行图像生成。

## 说明

### 设置

1. **环境变量**：
   - 设置 `GEMINI_API_KEY` 为您的 Gemini API 密钥。
   - 可选地，设置 `OUTPUT_PATH` 指定输出文件的目录（默认是 `./output`）。

2. **依赖**：
   - 使用 `uv sync` 安装所需的 Python 包：

### 运行脚本

#### `background_generator.py`

- 设置 `XAI_API_KEY` 为您的 xAI API 密钥。

- **使用方法**：
  ```bash
  un run python background_generator.py
  ```
- **Gradio 界面**：
  - 上传一个或多个图像。
  - 输入背景提示（元素）。
  - 选择模式：简单、正常或上下文。
  - 设置每个图像生成的背景数量。
  - 点击“生成”开始处理。

#### `multi_generator.py`

- **使用方法**：
  ```bash
  uv run python multi_generator.py
  ```
- **Gradio 界面**：
  - 上传一个图像。//z
  - 输入背景提示。
  - 设置会话数量。
  - 设置每个会话生成的图像数量。
  - 点击“生成”开始处理。

## 环境

- **Python**：推荐使用 3.12 或更高版本。
- **API**：确保您有 Gemini权限。
- **Gradio**：用于创建用户界面。

## 注意事项

- 确保您的 API 密钥安全存储，不要提交到仓库中。
- 生成的图像会保存在指定的输出目录中，并带有时间戳以便于跟踪。
