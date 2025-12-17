# AI Singing Align & Mix

用 HuBERT + DTW 自动对齐人声，并参考原唱的音色/响度/情感包络完成智能混音。包含可选的 Demucs 降噪、对齐、混音、质量检查工具链。

## 环境与依赖
- Python 3.10+
- 必备命令行：`ffmpeg`、`rubberband`（对齐变速）、`demucs`（降噪，人声提取）
- Python 依赖：见 `pyproject.toml`，可直接 `pip install -e .`（或用 `uv`/`pip` 安装）

示例安装：
```bash
pip install -e .
# 或
uv pip install -e .
```

## 快速上手
下面用占位路径表示，可替换成你的文件。

1) 可选：降噪/提取纯人声（Demucs）
```bash
python denoise.py 原始人声.wav output_denoised.wav
```

2) 对齐你的声音到参考
```bash
python align.py 你的干声.wav 参考音频.wav output_aligned.wav
# 如需跳过预处理: --no-preprocess
# 如需强制变速（不做安全回退）: --force-warp
```

3) 智能混音（参考原唱的亮度/响度/情感包络）
```bash
python mixing.py output_aligned.wav 伴奏.wav 原唱干声.wav output_final_mix.wav
# 关闭情感包络对齐: --disable_emotion
# 调低情感跟随力度: --emotion_strength 0.6
```

混音完成后会生成：
- `<输出名>_vocal.wav`：处理后的你的人声（含 EQ/压缩/情感对齐，带母带限制）
- `<输出名>`（即 out_wav）：你的人声 + 伴奏
- `<输出名>_with_ref.wav`：你的人声 + 原唱（无伴奏，对比音色/情绪用）
- `<输出名>_plus_ref.wav`：你的人声 + 伴奏 + 原唱(-6dB)，便于 AB 对照

## 工具概览
- `denoise.py`：Demucs 提取人声，降噪/去伴奏。
- `align.py`：HuBERT 语义特征 + fast DTW + Rubberband 变速对齐。
- `mixing.py`：参考原唱的响度/亮度/情感包络，自动 EQ、压缩、配平并输出多个对照版本。
- `mix_check.py` / `benchmark.py`：辅助检查和指标（可按需运行）。 

## 常见问题
- **采样率不一致**：程序会自动重采样，检测到 ffmpeg 时优先使用 ffmpeg 提升速度。
- **Demucs 未安装**：运行 `pip install demucs`，确保 `demucs` 命令在 PATH。
- **显存不足**：HuBERT 会自动 fallback 到 CPU；可先裁剪片段做测试。 
