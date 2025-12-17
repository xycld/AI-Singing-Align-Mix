import soundfile as sf
import librosa
import numpy as np
import sys
import os
from scipy import signal

def normalize_audio(y):
    """辅助函数：将音频归一化到 Peak=1.0"""
    max_val = np.max(np.abs(y))
    if max_val > 1e-6: # 避免除以零
        return y / max_val
    return y

def get_coarse_offset(y_user, y_ref, sr):
    """计算两个音频的粗略时间差 (用于对齐开头)"""
    hop = 512
    o_user = librosa.onset.onset_strength(y=y_user, sr=sr, hop_length=hop)
    o_ref = librosa.onset.onset_strength(y=y_ref, sr=sr, hop_length=hop)
    
    correlation = signal.correlate(o_ref, o_user, mode='full')
    lags = signal.correlation_lags(len(o_ref), len(o_user), mode='full')
    lag_frames = lags[np.argmax(correlation)]
    
    return int(lag_frames * hop)

def debug_drift(user_raw_path, aligned_path, ref_path, output_path):
    """
    生成漂移诊断音频：
    左声道：原始干声 (经过粗对齐平移，起点对齐)
    右声道：AI对齐后的干声
    
    通过对比左右声道，可以直观听到 AI 到底把声音拉伸了多少，以及哪里没对上。
    """
    if not os.path.exists(user_raw_path) or not os.path.exists(aligned_path) or not os.path.exists(ref_path):
        print("错误: 找不到输入文件")
        return

    print(f"正在生成漂移诊断文件...")
    # 1. 加载
    y_raw, sr = librosa.load(user_raw_path, sr=None, mono=True)
    y_aligned, _ = librosa.load(aligned_path, sr=sr, mono=True)
    y_ref, _ = librosa.load(ref_path, sr=sr, mono=True)

    # 2. 计算原始干声相对于参考的偏移 (粗对齐)
    offset = get_coarse_offset(y_raw, y_ref, sr)
    print(f"   [粗对齐] 原始录音偏移: {offset/sr:.3f}s")

    # 3. 平移原始干声
    if offset > 0:
        y_raw_shifted = np.pad(y_raw, (offset, 0))
    elif offset < 0:
        y_raw_shifted = y_raw[-offset:]
    else:
        y_raw_shifted = y_raw

    # 4. 统一长度 (以参考音频为准)
    target_len = len(y_ref)
    
    # 裁剪或填充 Raw
    if len(y_raw_shifted) > target_len:
        y_raw_shifted = y_raw_shifted[:target_len]
    else:
        y_raw_shifted = np.pad(y_raw_shifted, (0, target_len - len(y_raw_shifted)))
        
    # 裁剪或填充 Aligned
    if len(y_aligned) > target_len:
        y_aligned = y_aligned[:target_len]
    else:
        y_aligned = np.pad(y_aligned, (0, target_len - len(y_aligned)))

    # 5. 归一化
    y_raw_shifted = normalize_audio(y_raw_shifted) * 0.8
    y_aligned = normalize_audio(y_aligned) * 0.8

    # 6. 合成立体声 (左=原始粗对齐，右=AI精对齐)
    y_out = np.vstack((y_raw_shifted, y_aligned)).T
    
    sf.write(output_path, y_out, sr)
    print(f"诊断文件已生成: {output_path}")
    print("   -> 左耳: 原始录音 (仅平移，无变速)")
    print("   -> 右耳: AI对齐后 (有变速)")
    print("   -> 听法: 如果两者逐渐错开，说明 AI 正在工作(修正速度)。如果右耳比左耳更贴合原唱节奏，说明对齐成功。")

def debug_raw_vs_ref(user_raw_path, ref_path, output_path):
    """
    生成原始录音与原唱的对比 (用于确认原始录音的问题)
    左声道：原始干声 (经过粗对齐平移)
    右声道：原唱参考
    """
    if not os.path.exists(user_raw_path) or not os.path.exists(ref_path):
        print("错误: 找不到输入文件")
        return

    print(f"正在生成原始vs原唱对比文件...")
    # 1. 加载
    y_raw, sr = librosa.load(user_raw_path, sr=None, mono=True)
    y_ref, _ = librosa.load(ref_path, sr=sr, mono=True)

    # 2. 计算原始干声相对于参考的偏移 (粗对齐)
    offset = get_coarse_offset(y_raw, y_ref, sr)
    
    # 3. 平移原始干声
    if offset > 0:
        y_raw_shifted = np.pad(y_raw, (offset, 0))
    elif offset < 0:
        y_raw_shifted = y_raw[-offset:]
    else:
        y_raw_shifted = y_raw

    # 4. 统一长度
    min_len = min(len(y_raw_shifted), len(y_ref))
    y_raw_shifted = y_raw_shifted[:min_len]
    y_ref = y_ref[:min_len]

    # 5. 归一化
    y_raw_shifted = normalize_audio(y_raw_shifted) * 0.8
    y_ref = normalize_audio(y_ref) * 0.8

    # 6. 合成立体声
    y_out = np.vstack((y_raw_shifted, y_ref)).T
    
    sf.write(output_path, y_out, sr)
    print(f"对比文件已生成: {output_path}")
    print("   -> 左耳: 原始录音 (粗对齐)")
    print("   -> 右耳: 原唱参考")
    print("   -> 听法: 这样可以直接听到原始录音和原唱的时间差。")

def mix_audio(aligned_user_path, ref_path, output_path, user_vol=1.0, ref_vol=0.15, mode="mix"):
    """
    快速混音工具：将对齐后的人声与原曲混合。
    
    参数:
    mode: "mix" (混合单声道) 或 "stereo" (左原唱 右人声)
    """
    
    if not os.path.exists(aligned_user_path) or not os.path.exists(ref_path):
        print("错误: 找不到输入文件，请检查路径。")
        return

    print(f"1. 正在加载音频...")
    y_user, sr = librosa.load(aligned_user_path, sr=None, mono=True)
    y_ref, _ = librosa.load(ref_path, sr=sr, mono=True) 

    print("   [预处理] 正在统一音频电平 (Normalize)...")
    y_user = normalize_audio(y_user)
    y_ref = normalize_audio(y_ref)

    # 裁剪到较短的长度
    min_len = min(len(y_user), len(y_ref))
    y_user = y_user[:min_len]
    y_ref = y_ref[:min_len]

    if mode == "stereo":
        print(f"2. 正在生成立体声对比 (左: 原唱 / 右: 人声)...")
        # 左声道：原唱 (ref)，右声道：人声 (user)
        # 这种模式下忽略音量参数，直接归一化输出，方便对比细节
        left = y_ref * 0.8 
        right = y_user * 0.8
        y_out = np.vstack((left, right)).T
    else:
        print(f"2. 正在混合 (User: {int(user_vol*100)}% + Ref: {int(ref_vol*100)}%)...")
        y_mix = (y_user * user_vol) + (y_ref * ref_vol)
        # 防爆音
        max_amp = np.max(np.abs(y_mix))
        if max_amp > 1.0:
            print(f"   [提示] 检测到爆音 (Max amp: {max_amp:.2f})，正在自动归一化...")
            y_mix = y_mix / max_amp
        y_out = y_mix

    print(f"3. 保存文件 -> {output_path}")
    # 使用 PCM_24 格式保存，保留更多动态细节
    sf.write(output_path, y_out, sr, subtype='PCM_24')
    print("完成！")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: uv run mix_check.py <aligned.wav> <ref.wav> [out.wav] [user_vol] [ref_vol] [mode]")
        print("示例 (混合): uv run mix_check.py test_aligned.wav original.wav check.wav 1.0 0.2 mix")
        print("示例 (分离): uv run mix_check.py test_aligned.wav original.wav check_lr.wav 1.0 0.2 stereo")
    else:
        u_path = sys.argv[1]
        r_path = sys.argv[2]
        o_path = sys.argv[3] if len(sys.argv) > 3 else "result_mix_check.wav"
        
        u_vol = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        r_vol = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
        
        # 增加模式选择
        mode = sys.argv[6] if len(sys.argv) > 6 else "mix"
        
        mix_audio(u_path, r_path, o_path, user_vol=u_vol, ref_vol=r_vol, mode=mode)