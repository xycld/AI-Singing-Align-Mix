import os
import sys
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
from scipy.spatial.distance import cosine
from scipy import signal
from fastdtw import fastdtw
from transformers import Wav2Vec2FeatureExtractor, HubertModel

class AIAligner:
    def __init__(self, sr=44100, safe_mode=True, preprocess=True):
        self.sr = sr
        self.safe_mode = safe_mode
        self.preprocess = preprocess
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 在特征域内改用 16k 采样率，减少帧数、提升 DTW 速度
        self.feature_sr = 16000
        # 两阶段 DTW：先用降采样特征 + 小窗口粗对齐，再在原始特征上做精对齐
        self.coarse_downsample = 2           # 对 HuBERT 序列按 2 帧(40ms)做平均
        self.coarse_radius = 120             # 粗对齐时的搜索半径
        self.min_refine_radius = 80          # 精对齐下限
        self.max_refine_radius = 300         # 精对齐上限
        
        print(f"载入 AI Aligner (设备: {self.device})...")
        if self.device == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            print(f"   -> 发现显卡: {props.name} (显存: {props.total_memory / 1024**3:.1f} GB)")

        print("   -> 加载 HuBERT 模型 (语义特征提取)...")
        # 使用 safetensors 避免 pickle 安全警告
        self.model_name = "facebook/hubert-base-ls960"
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        try:
            self.model = HubertModel.from_pretrained(self.model_name, use_safetensors=True)
        except Exception:
             # 如果没有 safetensors，回退到默认
             self.model = HubertModel.from_pretrained(self.model_name)
             
        self.model.eval()
        self.model.to(self.device)

    def _prep_for_alignment(self, y, sr):
        """
        尝试把“纯人声 vs 伴奏混音”变得更像，避免 DTW 路径跑偏：
        - 预加重 + 人声频段带通
        - harmonic 提取（弱化鼓点/瞬态）
        """
        if not self.preprocess:
            return y

        y = y.astype(np.float32, copy=False)
        # 预加重（更像语音输入）
        try:
            y = librosa.effects.preemphasis(y)
        except Exception:
            pass

        # 人声常用频段带通（弱化低频鼓、超高频噪声）
        try:
            low_hz = 80.0
            high_hz = min(5000.0, sr * 0.45)
            if high_hz > low_hz:
                sos = signal.butter(
                    6,
                    [low_hz / (sr * 0.5), high_hz / (sr * 0.5)],
                    btype="band",
                    output="sos",
                )
                y = signal.sosfiltfilt(sos, y).astype(np.float32, copy=False)
        except Exception:
            pass

        # harmonic: 对“参考带伴奏”时通常能明显减少鼓点干扰
        try:
            y = librosa.effects.harmonic(y, margin=6.0)
        except Exception:
            pass

        return y

    def _to_mono_float32(self, y):
        if y.ndim == 2:
            y = np.mean(y, axis=1)
        return y.astype(np.float32)

    def _load_audio(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件未找到: {path}")
        y, sr = sf.read(path, always_2d=False)
        y = self._to_mono_float32(y)
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        return y

    def get_offset(self, y_user, y_ref):
        """粗对齐：计算前奏时间差"""
        print("   [计算] 正在检测全局时间差 (Coarse Alignment)...")
        hop = 512
        o_user = librosa.onset.onset_strength(y=y_user, sr=self.sr, hop_length=hop)
        o_ref = librosa.onset.onset_strength(y=y_ref, sr=self.sr, hop_length=hop)
        
        correlation = signal.correlate(o_ref, o_user, mode='full')
        lags = signal.correlation_lags(len(o_ref), len(o_user), mode='full')
        lag_frames = lags[np.argmax(correlation)]
        
        lag_samples = int(lag_frames * hop)
        return lag_samples

    def extract_deep_features(self, y):
        """使用 HuBERT 提取深度语义特征"""
        # HuBERT 必须使用 16k 输入
        if self.sr != self.feature_sr:
            y_16k = librosa.resample(y, orig_sr=self.sr, target_sr=self.feature_sr)
        else:
            y_16k = y
        y_16k = self._prep_for_alignment(y_16k, self.feature_sr)

        inputs = self.processor(y_16k, sampling_rate=self.feature_sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
        
        return outputs.last_hidden_state[0].cpu().numpy()

    def _downsample_features(self, feats, factor):
        """将 HuBERT 特征按 factor 帧做平均池化"""
        if factor <= 1:
            return feats
        total_frames = feats.shape[0]
        trim = total_frames - (total_frames % factor)
        if trim <= 0:
            return feats
        feats_trim = feats[:trim]
        return feats_trim.reshape(-1, factor, feats.shape[1]).mean(axis=1)

    def warp_audio_rubberband(self, y, sr, path, frame_duration, target_samples):
        """
        使用 Rubberband CLI 执行无损变速
        """
        # 创建临时文件，因为 CLI 只能读文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out, \
             tempfile.NamedTemporaryFile(suffix=".map", delete=False, mode='w') as f_map:
            
            infile = f_in.name
            outfile = f_out.name
            mapfile = f_map.name
            
            sf.write(infile, y, sr)

            # 生成映射表
            # 格式: InputSample OutputSample
            map_content = ["0 0"]
            # [恢复] 步长设为 1 (20ms)，用户反馈此精度下对齐效果完美
            step = 1

            total_samples = int(len(y))
            target_samples = int(target_samples)
            if target_samples <= 0:
                target_samples = 1

            # 把 HuBERT 帧索引映射回采样点时，做一次尺度对齐，确保终点能对齐到真实音频长度
            raw_u_end = int(path[-1, 0] * frame_duration * sr) if len(path) else total_samples
            raw_r_end = int(path[-1, 1] * frame_duration * sr) if len(path) else target_samples
            u_scale = (total_samples / max(raw_u_end, 1)) if raw_u_end > 0 else 1.0
            r_scale = (target_samples / max(raw_r_end, 1)) if raw_r_end > 0 else 1.0
            
            for i in range(1, len(path), step):
                # 将 HuBERT 帧索引转换为时间，再转换为当前音频的采样点
                # Time = FrameIndex * 0.02s
                # Sample = Time * SR
                u_time = path[i, 0] * frame_duration
                r_time = path[i, 1] * frame_duration
                
                # 留 1 个采样点给强制终点，避免最后一行无法严格递增
                u_sample = int(u_time * sr * u_scale)
                r_sample = int(r_time * sr * r_scale)
                u_sample = min(max(u_sample, 0), max(total_samples - 1, 0))
                r_sample = min(max(r_sample, 0), max(target_samples - 1, 0))
                
                last_entry = map_content[-1].split()
                last_u = int(last_entry[0])
                last_r = int(last_entry[1])

                # 严格单调递增检查，防止 Rubberband 报错
                if u_sample > last_u and r_sample > last_r:
                    map_content.append(f"{u_sample} {r_sample}")
            
            # 强制终点
            last_entry = map_content[-1].split()
            last_u = int(last_entry[0])
            last_r = int(last_entry[1])
            if total_samples > last_u and target_samples > last_r:
                map_content.append(f"{total_samples} {target_samples}")
            
            f_map.write("\n".join(map_content))
            f_map.flush()

            # 计算总时长用于 -D 参数
            target_duration = target_samples / sr
            
            cmd = [
                "rubberband", 
                "-3", # R3 高精度引擎
                "-q", # 安静模式
                "-M", mapfile, # 使用映射表
                "-D", str(target_duration), # 必须指定总时长 (Rubberband 要求使用 -M 时必须指定 -D/-t/-T)
                # "--smoothing", "0.05", # [移除] R3 引擎可能不支持此参数，且默认效果已足够好
                "-F", # 共振峰保留
                infile, 
                outfile
            ]
            
            subprocess.run(cmd, check=True)
            y_out, _ = librosa.load(outfile, sr=sr)
            
            # 清理临时文件
            try:
                os.remove(infile); os.remove(outfile); os.remove(mapfile)
            except: pass
                
            return y_out

    def load_and_preprocess(self, user_audio_path, ref_audio_path):
        """步骤1: 加载音频并进行粗对齐 (前奏同步)"""
        if not os.path.exists(user_audio_path) or not os.path.exists(ref_audio_path):
            raise FileNotFoundError("文件不存在")

        print(f"1. 加载音频 (SR={self.sr})...")
        y_user, _ = librosa.load(user_audio_path, sr=self.sr, mono=True)
        y_ref, _ = librosa.load(ref_audio_path, sr=self.sr, mono=True)

        # 2. 粗对齐 (前奏处理)
        # 用“更像人声”的版本来做 offset 检测，但实际平移还是作用在原始波形上
        y_user_for_offset = self._prep_for_alignment(y_user, self.sr)
        y_ref_for_offset = self._prep_for_alignment(y_ref, self.sr)
        offset_samples = self.get_offset(y_user_for_offset, y_ref_for_offset)
        if offset_samples > 0:
            print(f"   [同步] 填充前奏: {offset_samples} samples")
            y_user = np.pad(y_user, (offset_samples, 0))
        elif offset_samples < 0:
            print(f"   [同步] 裁剪开头: {-offset_samples} samples")
            y_user = y_user[-offset_samples:]
        
        return y_user, y_ref

    def compute_path(self, y_user, y_ref):
        """步骤2: 提取特征并计算 DTW 路径 (耗时步骤)"""
        print("3. 提取 HuBERT 深度语义特征 (这可能需要显卡)...")
        # HuBERT 提取的是 768维 的向量
        feats_user = self.extract_deep_features(y_user)
        feats_ref = self.extract_deep_features(y_ref)
        
        print(f"   特征形状: User {feats_user.shape}, Ref {feats_ref.shape}")

        # 4. DTW
        print("4. 计算 DTW 路径 (两阶段加速)...")
        # 第一步：对特征做降采样，快速估计全局路径
        feats_user_ds = self._downsample_features(feats_user, self.coarse_downsample)
        feats_ref_ds = self._downsample_features(feats_ref, self.coarse_downsample)
        print(f"   [粗对齐] 降采样因子 x{self.coarse_downsample}, 特征长度 {feats_user_ds.shape[0]} / {feats_ref_ds.shape[0]}")
        _, coarse_path = fastdtw(feats_user_ds, feats_ref_ds, dist=cosine, radius=self.coarse_radius)
        coarse_path = np.array(coarse_path)
        coarse_offset = np.abs(coarse_path[:, 0] - coarse_path[:, 1]).max() if len(coarse_path) > 0 else self.coarse_radius
        # 估算需要的精细窗口（乘回原始帧率后留一点余量）
        estimated_radius = int(coarse_offset * self.coarse_downsample * 1.2) + self.coarse_downsample
        refine_radius = max(self.min_refine_radius, min(self.max_refine_radius, estimated_radius))
        print(f"   [精对齐] 自适应搜索半径 -> {refine_radius} (原始特征长度 {feats_user.shape[0]} / {feats_ref.shape[0]})")
        
        _, path = fastdtw(feats_user, feats_ref, dist=cosine, radius=refine_radius)
        path = np.array(path)
        return path

    def _alignment_score(self, y_test, y_ref, window_seconds=8.0, hop=512, max_lag_seconds=1.25):
        """
        用 onset envelope 做一个轻量级“对齐好不好”的评估：
        返回 (mean_abs_lag_seconds, [lags_seconds...])
        """
        y_test = np.asarray(y_test, dtype=np.float32)
        y_ref = np.asarray(y_ref, dtype=np.float32)
        if y_test.size == 0 or y_ref.size == 0:
            return float("inf"), []

        # 统一时长（评估用最短部分）
        min_len = min(len(y_test), len(y_ref))
        y_test = y_test[:min_len]
        y_ref = y_ref[:min_len]

        try:
            o_test = librosa.onset.onset_strength(y=y_test, sr=self.sr, hop_length=hop)
            o_ref = librosa.onset.onset_strength(y=y_ref, sr=self.sr, hop_length=hop)
        except Exception:
            return float("inf"), []

        n = min(len(o_test), len(o_ref))
        if n < 10:
            return float("inf"), []
        o_test = o_test[:n]
        o_ref = o_ref[:n]

        win_frames = int((window_seconds * self.sr) / hop)
        win_frames = max(20, min(win_frames, n))
        max_lag_frames = int((max_lag_seconds * self.sr) / hop)
        max_lag_frames = max(5, max_lag_frames)

        # 取 5 个窗口：起/1/4/1/2/3/4（尽量覆盖全曲）
        if n <= win_frames:
            starts = [0]
        else:
            starts = np.linspace(0, n - win_frames, num=5)
            starts = [int(round(x)) for x in starts]

        lags_s = []
        for s in starts:
            a = o_ref[s : s + win_frames]
            b = o_test[s : s + win_frames]
            corr = signal.correlate(a, b, mode="full")
            lags = signal.correlation_lags(len(a), len(b), mode="full")
            mask = (lags >= -max_lag_frames) & (lags <= max_lag_frames)
            lags = lags[mask]
            corr = corr[mask]
            if corr.size == 0:
                continue
            lag_frames = int(lags[int(np.argmax(corr))])
            lags_s.append(lag_frames * hop / self.sr)

        if not lags_s:
            return float("inf"), []
        mean_abs = float(np.mean(np.abs(lags_s)))
        return mean_abs, lags_s

    def apply_warping(self, y_user, y_ref, path, output_path):
        """步骤3: 应用 Rubberband 变速并保存"""
        print("5. 执行 Rubberband 变速...")
        # HuBERT base model 的下采样率是 320 (在 16k Hz 下)
        # 所以一帧的时间 = 320 / 16000 = 0.02 秒
        hubert_frame_duration = 0.02
        
        try:
            target_len = len(y_ref)
            y_aligned = self.warp_audio_rubberband(
                y_user,
                self.sr,
                path,
                hubert_frame_duration,
                target_samples=target_len,
            )
            
            # 最终裁剪
            if len(y_aligned) > target_len:
                y_aligned = y_aligned[:target_len]
            else:
                y_aligned = np.pad(y_aligned, (0, target_len - len(y_aligned)))

            # --- 安全检查：如果 AI 对齐变差，就回退到“仅粗对齐(平移)” ---
            if self.safe_mode:
                base = y_user
                if len(base) > target_len:
                    base = base[:target_len]
                else:
                    base = np.pad(base, (0, target_len - len(base)))

                base_score, base_lags = self._alignment_score(base, y_ref)
                aligned_score, aligned_lags = self._alignment_score(y_aligned, y_ref)
                print(
                    "   [评估] 平均窗口偏移(越小越好): "
                    f"Baseline={base_score:.3f}s, Aligned={aligned_score:.3f}s"
                )
                if base_lags:
                    print("   [评估] Baseline 窗口偏移:", ", ".join(f"{x:+.3f}s" for x in base_lags))
                if aligned_lags:
                    print("   [评估] Aligned  窗口偏移:", ", ".join(f"{x:+.3f}s" for x in aligned_lags))

                max_base = max((abs(x) for x in base_lags), default=float("inf"))
                max_aligned = max((abs(x) for x in aligned_lags), default=float("inf"))

                # 经验阈值：如果对齐后平均偏移显著变大，并且最大偏移也更大，则回退
                if (
                    np.isfinite(base_score)
                    and np.isfinite(aligned_score)
                    and (aligned_score > base_score + 0.12)
                    and (max_aligned > max_base + 0.15)
                ):
                    print("   [回退] 检测到 AI 对齐变差，输出将改为仅粗对齐(平移)，避免越对越偏。")
                    y_aligned = base

            sf.write(output_path, y_aligned, self.sr)
            print(f"对齐完成! 已保存 -> {output_path}")

            # --- 生成对比混合音频 ---
            print("正在生成混合验证文件 (User 100% + Ref 15%)...")
            mix_len = min(len(y_aligned), len(y_ref))
            y_mix = y_aligned[:mix_len] + y_ref[:mix_len] * 0.15
            
            # 简单的防爆音处理
            if np.max(np.abs(y_mix)) > 1.0:
                y_mix = y_mix / np.max(np.abs(y_mix))
                
            mix_path = os.path.splitext(output_path)[0] + "_mix.wav"
            sf.write(mix_path, y_mix, self.sr)
            print(f"混合验证文件已保存 -> {mix_path}")
            
        except Exception as e:
            print(f"对齐失败: {e}")
            import traceback
            traceback.print_exc()

    def process_file(self, user_audio_path, ref_audio_path, output_path):
        """一键处理 (兼容旧代码)"""
        try:
            y_user, y_ref = self.load_and_preprocess(user_audio_path, ref_audio_path)
            path = self.compute_path(y_user, y_ref)
            self.apply_warping(y_user, y_ref, path, output_path)
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="HuBERT + DTW 自动对齐人声到参考 (输出变速后的人声音频)"
    )
    parser.add_argument("user_wav", help="你的原始人声 (待对齐)")
    parser.add_argument("ref_wav", help="参考音频 (带伴奏也可)")
    parser.add_argument("out_wav", help="对齐后的人声输出路径")
    parser.add_argument("--force-warp", action="store_true", help="强制使用 AI 变速对齐 (不做安全回退)")
    parser.add_argument("--no-preprocess", action="store_true", help="关闭对齐前的预处理 (带通/HPSS)")
    
    args = parser.parse_args()
    
    aligner = AIAligner(
        safe_mode=(not args.force_warp),
        preprocess=(not args.no_preprocess),
    )
    aligner.process_file(args.user_wav, args.ref_wav, args.out_wav)
