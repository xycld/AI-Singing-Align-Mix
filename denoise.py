import os
import subprocess
import shutil
import argparse


class AIDenoiser:
    def __init__(self):
        print("初始化 AI 降噪器 (切换为 Demucs 音乐专用模型)...")
        # Demucs 是命令行工具，不需要预加载 Python 对象

    def _ensure_demucs(self):
        if shutil.which("demucs") is None:
            raise RuntimeError("未检测到 demucs 命令，请先运行: pip install demucs")

    def process_file(self, input_path, output_path):
        """
        使用 Demucs 提取人声（相当于降噪）
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"找不到文件: {input_path}")
        
        print(f"正在使用 Demucs 进行人声提取 (降噪)...")
        print("说明: Demucs 是专为音乐设计的，它能完美区分'人声'和'噪音'，不会误删弱音。")
        
        # 使用 htdemucs 模型，开启 --two-stems=vocals 只提取人声
        # 这会自动把背景噪音归类到 "no_vocals" 里去，从而实现降噪
        cmd = [
            "demucs", 
            "-n", "htdemucs", 
            "--two-stems=vocals", 
            input_path
        ]
        
        # 运行 Demucs
        self._ensure_demucs()
        subprocess.run(cmd, check=True)
        
        # 找到生成的文件
        # 默认输出在 separated/htdemucs/<文件名>/vocals.wav
        filename = os.path.splitext(os.path.basename(input_path))[0]
        demucs_out_dir = os.path.join("separated", "htdemucs", filename)
        vocals_out = os.path.join(demucs_out_dir, "vocals.wav")
        
        if os.path.exists(vocals_out):
            # 移动/重命名到目标路径
            shutil.copy(vocals_out, output_path)
            print(f"降噪完成: {input_path} -> {output_path}")
        else:
            raise RuntimeError("Demucs 处理失败，未找到输出文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="使用 Demucs 提取人声 (降噪)，保存为指定输出"
    )
    parser.add_argument("input", help="输入音频文件（含噪人声）")
    parser.add_argument("output", help="输出纯人声路径")
    args = parser.parse_args()

    denoiser = AIDenoiser()
    denoiser.process_file(args.input, args.output)
