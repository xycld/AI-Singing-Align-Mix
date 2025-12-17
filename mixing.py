import os
import sys
import shutil
import tempfile
import subprocess
import numpy as np
import librosa
import soundfile as sf
from pedalboard import (
    Pedalboard,
    Compressor,
    HighpassFilter,
    Reverb,
    NoiseGate,
    Limiter,
    Gain,
    HighShelfFilter,
    PeakFilter
)
from pedalboard.io import AudioFile

class AIMixer:
    def __init__(self, sr=44100):
        self.sr = sr
        print("载入 AI Mixer (Reference Matching Mode)...")

    def _ensure_2d(self, audio):
        """确保音频是 (channels, samples) 形状"""
        if audio.ndim == 1:
            return audio[np.newaxis, :]
        return audio

    def _ensure_stereo(self, audio):
        """将单声道复制成立体声 (channel-first)"""
        audio = self._ensure_2d(audio)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        return audio

    def _to_mono(self, audio):
        """转换为单声道波形"""
        audio = self._ensure_2d(audio)
        # channel-first -> 平均左右声道
        return np.mean(audio, axis=0)

    def _smooth_curve(self, curve, window=5):
        """简单滑动平均，避免包络过于剧烈"""
        if window <= 1 or curve.size == 0:
            return curve
        window = int(window)
        if window >= curve.size:
            return np.repeat(np.mean(curve), curve.size)
        kernel = np.ones(window) / window
        return np.convolve(curve, kernel, mode="same")

    def _extract_prosody(self, audio, sr, hop_length=1024):
        """提取力度/亮度/音高等情感相关特征"""
        audio = self._ensure_2d(audio)
        mono = self._to_mono(audio)

        rms = librosa.feature.rms(
            y=mono,
            frame_length=2048,
            hop_length=hop_length,
            center=True
        )[0]
        centroid = librosa.feature.spectral_centroid(
            y=mono,
            sr=sr,
            hop_length=hop_length
        )[0]
        # 选择人声常见音高范围，让 YIN 收敛更快且减少错误检测
        pitch = librosa.yin(
            mono,
            fmin=80,
            fmax=1000,
            sr=sr,
            frame_length=2048,
            hop_length=hop_length
        )
        pitch = np.where(np.isfinite(pitch), pitch, 0.0)

        return {
            "rms": rms,
            "centroid": centroid,
            "pitch": pitch,
            "hop_length": hop_length,
        }

    def apply_emotion_alignment(self, user_audio, ref_audio, sr, strength=1.0):
        """
        根据原唱的情绪/力度曲线自动调整用户人声
        - 对齐 RMS 包络，让强弱起伏一致
        - 给出音高/颤音差异的提示，便于后续微调
        """
        print("   [情感对齐] 根据原唱的力度/情绪包络微调你的人声...")
        hop_length = 1024
        strength = float(np.clip(strength, 0.0, 1.0))

        # 仅取重叠部分做分析，避免不同长度导致的错位
        min_len = min(user_audio.shape[1], ref_audio.shape[1])
        user_seg = user_audio[:, :min_len]
        ref_seg = ref_audio[:, :min_len]

        ref_feat = self._extract_prosody(ref_seg, sr, hop_length)
        user_feat = self._extract_prosody(user_seg, sr, hop_length)

        min_frames = min(len(ref_feat["rms"]), len(user_feat["rms"]))
        if min_frames < 2:
            print("      -> 片段过短，跳过情感对齐")
            return user_audio

        ref_rms = self._smooth_curve(ref_feat["rms"][:min_frames], window=5)
        user_rms = self._smooth_curve(user_feat["rms"][:min_frames], window=5)

        ref_db = 20 * np.log10(ref_rms + 1e-6)
        user_db = 20 * np.log10(user_rms + 1e-6)
        # 只做相对动态跟随，并限幅避免泵音
        gain_db = np.clip((ref_db - user_db) * strength, -9.0, 9.0)

        frame_positions = np.arange(min_frames) * hop_length
        gain_linear = 10 ** (gain_db / 20)
        sample_positions = np.arange(user_audio.shape[1])

        if gain_linear.size == 1:
            gain_curve = np.repeat(gain_linear[0], user_audio.shape[1])
        else:
            # 插值到采样点，保证相位连续，避免突兀
            gain_curve = np.interp(
                sample_positions,
                frame_positions,
                gain_linear,
                left=gain_linear[0],
                right=gain_linear[-1],
            )

        adjusted = user_audio * gain_curve

        # 提示音高/颤音差异（不直接修音，只做参考）
        ref_pitch = ref_feat["pitch"][:min_frames]
        user_pitch = user_feat["pitch"][:min_frames]
        valid = (ref_pitch > 0) & (user_pitch > 0)

        print(f"      -> 增益包络范围: {gain_db.min():+.1f} ~ {gain_db.max():+.1f} dB")

        if np.any(valid):
            cents = 1200 * np.log2((ref_pitch[valid] + 1e-6) / (user_pitch[valid] + 1e-6))
            pitch_bias = np.median(cents)
            vibrato_delta = np.std(ref_pitch[valid]) - np.std(user_pitch[valid])
            print(f"      -> 音高中位偏差: {pitch_bias:+.0f} cents, 颤音差异 Δσ: {vibrato_delta:+.2f} Hz")
        else:
            print("      -> 未检测到稳定音高，跳过音高提示")

        return adjusted

    # -------- 重采样工具 --------
    def _has_ffmpeg(self):
        return shutil.which("ffmpeg") is not None

    def _resample_with_ffmpeg(self, input_path, target_sr):
        """
        用 ffmpeg 做整段重采样（比 librosa 快），返回 (audio, sr)
        """
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav.close()
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ar", str(target_sr),
                tmp_wav.name
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            with AudioFile(tmp_wav.name) as f:
                audio = f.read(f.frames)
                sr = f.samplerate
            return audio, sr
        finally:
            try:
                os.remove(tmp_wav.name)
            except OSError:
                pass

    def get_loudness(self, audio):
        """计算感知响度 (RMS dB)"""
        rms = np.sqrt(np.mean(audio**2))
        return 20 * np.log10(rms + 1e-9)

    def get_spectral_centroid(self, audio):
        """计算频谱质心 (判断声音是亮还是闷)"""
        # 使用 3000Hz 作为基准频率进行归一化分析
        cent = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        return np.mean(cent)

    def analyze_reference(self, ref_vocal_path, inst_path):
        """
        分析原唱和伴奏的关系，提取混音参数
        """
        print("   [AI分析] 正在“听”原唱的混音参数...")
        
        # 加载原唱干声
        y_ref, _ = librosa.load(ref_vocal_path, sr=self.sr, mono=True)
        # 加载伴奏
        y_inst, _ = librosa.load(inst_path, sr=self.sr, mono=True)
        
        # 1. 分析响度平衡 (Balance)
        ref_db = self.get_loudness(y_ref)
        inst_db = self.get_loudness(y_inst)
        target_ratio = ref_db - inst_db # 原唱比伴奏响多少dB
        
        print(f"      -> 原唱响度: {ref_db:.1f}dB, 伴奏响度: {inst_db:.1f}dB")
        print(f"      -> 目标平衡: 人声比伴奏{'高' if target_ratio>0 else '低'} {abs(target_ratio):.1f}dB")

        # 2. 分析亮度 (Brightness/Tone)
        ref_brightness = self.get_spectral_centroid(y_ref)
        print(f"      -> 原唱亮度值: {ref_brightness:.0f} Hz")
        
        return target_ratio, ref_brightness

    def build_smart_chain(self, user_audio, target_brightness, vocal_rms_db):
        """
        根据原唱特征构建处理链
        """
        # 1. 分析用户当前的亮度
        user_brightness = self.get_spectral_centroid(user_audio)
        print(f"      -> 用户亮度值: {user_brightness:.0f} Hz")
        
        # 计算 EQ 补偿量
        # 如果原唱比用户亮，就要提升高频
        brightness_diff = target_brightness - user_brightness
        # 简单的线性映射：每差 500Hz 调整 1dB，最大不超过 6dB
        high_shelf_gain = np.clip(brightness_diff / 500.0, -4.0, 6.0)
        
        print(f"      -> 自动 EQ 决策: HighShelf {high_shelf_gain:+.1f}dB")

        # 2. 动态阈值
        comp_threshold = max(-40.0, vocal_rms_db - 8.0) 
        gate_threshold = max(-60.0, vocal_rms_db - 35.0)

        board = Pedalboard([
            # 基础清理
            NoiseGate(threshold_db=gate_threshold, ratio=4, release_ms=200),
            HighpassFilter(cutoff_frequency_hz=80),

            # 智能 EQ (Tone Matching)
            # 核心：根据原唱的亮度来调整你的音色，而不是瞎调
            HighShelfFilter(cutoff_frequency_hz=8000, gain_db=high_shelf_gain),
            
            # 修正鼻音/浑浊感 (通用优化)
            PeakFilter(cutoff_frequency_hz=400, gain_db=-2.5, q=1.0),

            # 压缩 (控制动态，但不压死)
            Compressor(
                threshold_db=comp_threshold,
                ratio=2.5,       # 稍微温柔一点，保留你的声音特征
                attack_ms=10.0,  # 慢一点启动，保留更多咬字头
                release_ms=100.0,
            ),

            # 混响 (润色)
            # 默认给一点点通用的 Plate 混响，避免太干
            Reverb(room_size=0.5, wet_level=0.2, dry_level=0.8, width=0.6),
        ])
        
        return board

    def process_mix(self, user_path, inst_path, ref_path, output_path, enable_emotion=True, emotion_strength=1.0):
        if not os.path.exists(user_path) or not os.path.exists(inst_path) or not os.path.exists(ref_path):
            raise FileNotFoundError("输入文件不存在")

        # 1. 分析原唱 (学习目标)
        target_balance_db, target_brightness = self.analyze_reference(ref_path, inst_path)

        print("2. 加载与处理用户人声...")
        # 重新加载用于处理 (Pedalboard 格式)
        with AudioFile(user_path) as f:
            user_audio = f.read(f.frames)
            user_sr = f.samplerate

        user_audio = self._ensure_stereo(user_audio)

        # 2.0 情感对齐：根据原唱的力度/颤音包络微调
        if enable_emotion:
            with AudioFile(ref_path) as f:
                ref_audio = f.read(f.frames)
                ref_sr = f.samplerate

            ref_audio = self._ensure_stereo(ref_audio)
            if ref_sr != user_sr:
                if self._has_ffmpeg():
                    print(f"   [采样率] ffmpeg 重采样原唱 {ref_sr}->{user_sr}")
                    # 直接重读文件而不对数组重采样，避免大文件占内存
                    ref_audio, ref_sr = self._resample_with_ffmpeg(ref_path, user_sr)
                else:
                    ref_audio = librosa.resample(
                        ref_audio,
                        orig_sr=ref_sr,
                        target_sr=user_sr,
                        axis=1,
                        res_type="kaiser_fast"
                    )
                    ref_sr = user_sr

            user_audio = self.apply_emotion_alignment(
                user_audio,
                ref_audio,
                sr=user_sr,
                strength=emotion_strength
            )

        # 2.1 预计算用户响度
        user_mono = self._to_mono(user_audio)
        user_db = self.get_loudness(user_mono)

        # 2.2 构建并应用效果链
        board = self.build_smart_chain(user_mono, target_brightness, user_db)
        processed_user = board(user_audio, user_sr)

        # 2.3 自动电平匹配 (Auto-Leveling)
        # 处理后的用户响度
        processed_db = self.get_loudness(self._to_mono(processed_user))

        # 我们需要伴奏的响度来做基准
        with AudioFile(inst_path) as f:
            inst_audio = f.read(f.frames)
            inst_sr = f.samplerate

        inst_audio = self._ensure_2d(inst_audio)

        # [修复] 检查并统一采样率 (防止伴奏变慢/变快)
        if inst_sr != user_sr:
            if self._has_ffmpeg():
                print(f"   [采样率] ffmpeg 重采样伴奏 {inst_sr}->{user_sr}")
                inst_audio, inst_sr = self._resample_with_ffmpeg(inst_path, user_sr)
                inst_audio = self._ensure_2d(inst_audio)
            else:
                print(f"   [警告] 采样率不匹配: User={user_sr}, Inst={inst_sr}. 正在重采样伴奏...")
                inst_audio = librosa.resample(
                    inst_audio,
                    orig_sr=inst_sr,
                    target_sr=user_sr,
                    axis=0,
                    res_type="kaiser_fast"
                )
                inst_sr = user_sr

        inst_mono = self._to_mono(inst_audio)
        current_inst_db = self.get_loudness(inst_mono)

        # 目标用户响度 = 伴奏响度 + 原唱与伴奏的差值
        target_user_db = current_inst_db + target_balance_db

        # 计算需要补偿的增益
        gain_needed_db = target_user_db - processed_db
        gain_linear = 10 ** (gain_needed_db / 20)
        
        print(f"   [AI混音] 自动配平: 需要增益 {gain_needed_db:+.1f}dB 以匹配原唱平衡")
        
        # 应用最终增益
        final_user = processed_user * gain_linear

        print("3. 最终混合...")
        # 对齐长度
        inst_audio = self._ensure_stereo(inst_audio)
        min_len = min(final_user.shape[1], inst_audio.shape[1])

        final_user = final_user[:, :min_len]
        inst_audio = inst_audio[:, :min_len]

        # 母带链复用
        master_board = self.build_master_chain()

        # 3.1 输出：仅混音后的你的人声
        vocal_only = master_board(final_user, user_sr)
        vocal_only_path = os.path.splitext(output_path)[0] + "_vocal.wav"
        with AudioFile(vocal_only_path, 'w', user_sr, vocal_only.shape[0]) as f:
            f.write(vocal_only)
        print(f"   已输出人声处理版 -> {vocal_only_path}")

        # 混合 (伴奏不降音量，人声去适配伴奏)
        mix_audio = final_user + inst_audio

        final_mix = master_board(mix_audio, user_sr)

        with AudioFile(output_path, 'w', user_sr, final_mix.shape[0]) as f:
            f.write(final_mix)
        print(f"完成! 已保存 -> {output_path}")

        # --- 新增：生成含原唱的对照版 (Guide Mix) ---
        print("4. 生成含原唱对照版...")
        try:
            with AudioFile(ref_path) as f:
                ref_audio = f.read(f.frames)
                ref_sr = f.samplerate

            ref_audio = self._ensure_2d(ref_audio)
            # 重采样与声道处理
            if ref_sr != user_sr:
                if self._has_ffmpeg():
                    print(f"   [采样率] ffmpeg 重采样原唱(对照) {ref_sr}->{user_sr}")
                    ref_audio, ref_sr = self._resample_with_ffmpeg(ref_path, user_sr)
                    ref_audio = self._ensure_2d(ref_audio)
                else:
                    ref_audio = librosa.resample(
                        ref_audio,
                        orig_sr=ref_sr,
                        target_sr=user_sr,
                        axis=0,
                        res_type="kaiser_fast"
                    )
                    ref_sr = user_sr
            ref_audio = self._ensure_stereo(ref_audio)

            # 裁剪
            ref_audio = ref_audio[:, :min_len]

            # 3.2 输出：用户 + 原唱 对照版（无伴奏）
            user_plus_ref = master_board(final_user + ref_audio, user_sr)
            user_plus_ref_path = os.path.splitext(output_path)[0] + "_with_ref.wav"
            with AudioFile(user_plus_ref_path, 'w', user_sr, user_plus_ref.shape[0]) as f:
                f.write(user_plus_ref)
            print(f"已生成对照版 (你 + 原唱) -> {user_plus_ref_path}")

            # 3.3 输出：用户 + 伴奏 + 原唱(低音量 -6dB)
            guide_mix = final_user + inst_audio + (ref_audio * 0.5)

            final_guide = master_board(guide_mix, user_sr)

            guide_path = os.path.splitext(output_path)[0] + "_plus_ref.wav"
            with AudioFile(guide_path, 'w', user_sr, final_guide.shape[0]) as f:
                f.write(final_guide)
            print(f"已生成对照版 (含原唱) -> {guide_path}")

        except Exception as e:
            print(f"生成对照版失败: {e}")

    def build_master_chain(self):
        return Pedalboard([Limiter(threshold_db=-1.0)])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI 智能参考混音器")
    parser.add_argument("user_wav", help="你的干声 (修音后)")
    parser.add_argument("inst_wav", help="伴奏文件 (no_vocals)")
    parser.add_argument("ref_wav", help="原唱干声 (用于学习混音参数)")
    parser.add_argument("out_wav", help="输出文件")
    parser.add_argument("--emotion_strength", type=float, default=1.0, help="情感包络跟随强度 (0~1, 默认1)")
    parser.add_argument("--disable_emotion", action="store_true", help="关闭基于原唱的情感对齐")
    
    args = parser.parse_args()
    
    mixer = AIMixer()
    mixer.process_mix(
        args.user_wav,
        args.inst_wav,
        args.ref_wav,
        args.out_wav,
        enable_emotion=not args.disable_emotion,
        emotion_strength=args.emotion_strength
    )
