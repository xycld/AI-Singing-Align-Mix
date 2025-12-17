import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.spatial.distance import cosine
from scipy import signal

class AlignmentBenchmark:
    def __init__(self, sr=16000, window_seconds=8.0, step_seconds=4.0, max_lag_seconds=1.0, hop=512):
        self.sr = sr
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.max_lag_seconds = max_lag_seconds
        self.hop = hop

    def _load_or_pass(self, input_data):
        """Helper to handle both file paths and numpy arrays"""
        if isinstance(input_data, str):
             if not input_data:
                 raise ValueError("Path is empty")
             y, _ = librosa.load(input_data, sr=self.sr, mono=True)
             return y
        return input_data

    def compute_chroma_similarity(self, y1, y2):
        """计算色度特征相似度 (衡量音高/和声的一致性)"""
        # Extract Chroma features (pitch class profiles)
        # 使用 CQT 色度图，对人声更准确
        c1 = librosa.feature.chroma_cqt(y=y1, sr=self.sr)
        c2 = librosa.feature.chroma_cqt(y=y2, sr=self.sr)
        
        # Align lengths
        min_len = min(c1.shape[1], c2.shape[1])
        c1 = c1[:, :min_len]
        c2 = c2[:, :min_len]
        
        # Compute frame-wise cosine similarity
        # 1 - cosine distance = cosine similarity
        sims = []
        # 避免除零错误，添加极小值
        epsilon = 1e-8
        
        for i in range(min_len):
            # Cosine distance returns [0, 2], where 0 is identical
            # We want similarity [0, 1], where 1 is identical
            v1 = c1[:, i] + epsilon
            v2 = c2[:, i] + epsilon
            dist = cosine(v1, v2)
            sims.append(1.0 - dist)
            
        return np.mean(sims), np.array(sims)

    def compute_spectral_distance(self, y1, y2):
        """计算梅尔频谱差异 (衡量整体听感/时间的一致性)"""
        # Mel Spectrogram
        s1 = librosa.feature.melspectrogram(y=y1, sr=self.sr)
        s2 = librosa.feature.melspectrogram(y=y2, sr=self.sr)
        
        # Log scale (dB)
        s1_db = librosa.power_to_db(s1, ref=np.max)
        s2_db = librosa.power_to_db(s2, ref=np.max)
        
        min_len = min(s1_db.shape[1], s2_db.shape[1])
        s1_db = s1_db[:, :min_len]
        s2_db = s2_db[:, :min_len]
        
        # Mean Absolute Difference
        diff = np.abs(s1_db - s2_db)
        return np.mean(diff), diff

    def compute_windowed_drift(self, y_test, y_ref):
        """
        使用 onset 包络的窗口化互相关，估计各时间段的相对偏移 (秒)
        返回: mean_abs, p95_abs, max_abs, drift_series(list), time_axis(list)
        """
        hop = self.hop
        win_frames = int((self.window_seconds * self.sr) / hop)
        step_frames = int((self.step_seconds * self.sr) / hop)
        max_lag_frames = int((self.max_lag_seconds * self.sr) / hop)
        max_lag_frames = max(5, max_lag_frames)

        o_test = librosa.onset.onset_strength(y=y_test, sr=self.sr, hop_length=hop)
        o_ref = librosa.onset.onset_strength(y=y_ref, sr=self.sr, hop_length=hop)

        n = min(len(o_test), len(o_ref))
        if n < win_frames or win_frames <= 0:
            return float("inf"), float("inf"), float("inf"), [], []
        o_test = o_test[:n]
        o_ref = o_ref[:n]

        drifts = []
        times = []
        for start in range(0, n - win_frames + 1, step_frames):
            end = start + win_frames
            a = o_ref[start:end]
            b = o_test[start:end]
            corr = signal.correlate(a, b, mode="full")
            lags = signal.correlation_lags(len(a), len(b), mode="full")
            mask = (lags >= -max_lag_frames) & (lags <= max_lag_frames)
            lags = lags[mask]
            corr = corr[mask]
            if corr.size == 0:
                continue
            lag_frames = int(lags[int(np.argmax(corr))])
            drifts.append(lag_frames * hop / self.sr)
            times.append(start * hop / self.sr)

        if not drifts:
            return float("inf"), float("inf"), float("inf"), [], []

        abs_arr = np.abs(drifts)
        mean_abs = float(np.mean(abs_arr))
        p95_abs = float(np.percentile(abs_arr, 95))
        max_abs = float(np.max(abs_arr))
        return mean_abs, p95_abs, max_abs, drifts, times

    def compute_onset_accuracy(self, y_user, y_ref):
        """计算瞬态(字头)对齐精度"""
        # 1. 检测瞬态 (Onsets)
        # 使用 backtrack=True 可以更精确地定位到声音开始的瞬间
        onset_env_ref = librosa.onset.onset_strength(y=y_ref, sr=self.sr)
        onsets_ref = librosa.onset.onset_detect(onset_envelope=onset_env_ref, sr=self.sr, units='time', backtrack=True)
        
        onset_env_user = librosa.onset.onset_strength(y=y_user, sr=self.sr)
        onsets_user = librosa.onset.onset_detect(onset_envelope=onset_env_user, sr=self.sr, units='time', backtrack=True)
        
        # 2. 匹配瞬态
        # 对于每一个参考瞬态，寻找最近的一个用户瞬态
        errors = []
        matched_pairs = []
        
        tolerance = 0.2 # 200ms 容差窗口
        
        for t_ref in onsets_ref:
            # 找到最近的 user onset
            dist = np.abs(onsets_user - t_ref)
            min_idx = np.argmin(dist)
            min_dist = dist[min_idx]
            
            if min_dist < tolerance:
                # 记录误差 (User - Ref)
                # 正值表示用户慢了(滞后)，负值表示用户快了(抢拍)
                diff = onsets_user[min_idx] - t_ref
                errors.append(diff)
                matched_pairs.append((t_ref, onsets_user[min_idx]))
                
        errors = np.array(errors)
        
        # 3. 计算统计指标
        if len(errors) == 0:
            return 0.0, 0.0, 0.0, np.array([])
            
        mae = np.mean(np.abs(errors)) # 平均绝对误差 (秒)
        median_error = np.median(errors) # 中位数误差 (判断整体是偏快还是偏慢)
        
        # 计算 "完美对齐率" (误差 < 50ms 的比例)
        perfect_count = np.sum(np.abs(errors) < 0.05)
        accuracy_score = perfect_count / len(onsets_ref)
        
        return mae, median_error, accuracy_score, errors

    def _coarse_shift(self, y_user, y_ref):
        """
        用 onset 包络相关计算全局偏移，并返回平移后的 user (不变速)
        """
        hop = self.hop
        o_user = librosa.onset.onset_strength(y=y_user, sr=self.sr, hop_length=hop)
        o_ref = librosa.onset.onset_strength(y=y_ref, sr=self.sr, hop_length=hop)
        corr = signal.correlate(o_ref, o_user, mode="full")
        lags = signal.correlation_lags(len(o_ref), len(o_user), mode="full")
        lag_frames = lags[np.argmax(corr)]
        lag_samples = int(lag_frames * hop)

        if lag_samples > 0:
            shifted = np.pad(y_user, (lag_samples, 0))
        elif lag_samples < 0:
            shifted = y_user[-lag_samples:]
        else:
            shifted = y_user
        # 对齐长度到参考
        if len(shifted) > len(y_ref):
            shifted = shifted[:len(y_ref)]
        else:
            shifted = np.pad(shifted, (0, len(y_ref) - len(shifted)))
        return shifted, lag_samples

    def run_benchmark(self, target, reference, label="", raw=None):
        print(f"\n--- 正在评估对齐质量 {label} ---")
        
        y_aligned = self._load_or_pass(target)
        y_ref = self._load_or_pass(reference)
        y_raw = self._load_or_pass(raw) if raw is not None else None
        
        # Trim to same length
        min_len = min(len(y_aligned), len(y_ref))
        y_aligned = y_aligned[:min_len]
        y_ref = y_ref[:min_len]
        if y_raw is not None:
            y_raw = y_raw[:min_len]
        
        # 1. Chroma Similarity (Pitch/Harmonic Alignment)
        chroma_score, chroma_curve = self.compute_chroma_similarity(y_aligned, y_ref)
        print(f"  -> 音高/和声相似度 (Chroma Similarity): {chroma_score:.4f} (越高越好, Max 1.0)")
        
        # 2. Spectral Distance (Timbre/Timing Alignment)
        spec_dist, spec_diff = self.compute_spectral_distance(y_aligned, y_ref)
        print(f"  -> 频谱差异度 (Spectral Distance): {spec_dist:.2f} dB (越低越好)")
        
        # 3. Onset Accuracy (Word-level Timing)
        mae, median_err, acc_score, errors = self.compute_onset_accuracy(y_aligned, y_ref)
        print(f"  -> 字头对齐精度 (Onset Accuracy):")
        print(f"     - 平均误差: {mae*1000:.1f} ms")
        print(f"     - 整体偏移: {median_err*1000:.1f} ms (正=滞后, 负=抢拍)")
        print(f"     - 完美对齐率 (<50ms): {acc_score*100:.1f}%")

        # 4. 窗口化漂移 (更直观看“越到后面是否漂移”)
        drift_mean, drift_p95, drift_max, drift_series, drift_times = self.compute_windowed_drift(y_aligned, y_ref)
        print("  -> 窗口漂移 (Onset window drift, 越低越好)")
        print(f"     - 平均: {drift_mean:.3f}s, P95: {drift_p95:.3f}s, 最大: {drift_max:.3f}s")

        baseline_result = None
        if y_raw is not None:
            y_base_shifted, base_offset = self._coarse_shift(y_raw, y_ref)
            base_drift_mean, base_drift_p95, base_drift_max, base_drift_series, base_drift_times = self.compute_windowed_drift(
                y_base_shifted, y_ref
            )
            improvement = base_drift_mean - drift_mean
            print("  -> 对比基线(仅平移，不变速):")
            print(f"     - 粗对齐偏移: {base_offset / self.sr:.3f}s")
            print(f"     - 漂移平均: {base_drift_mean:.3f}s, P95: {base_drift_p95:.3f}s, 最大: {base_drift_max:.3f}s")
            print(f"     - 改善(基线-当前): {improvement:+.3f}s")
            baseline_result = {
                "shifted": y_base_shifted,
                "offset": base_offset,
                "drift_mean": base_drift_mean,
                "drift_p95": base_drift_p95,
                "drift_max": base_drift_max,
                "drift_series": base_drift_series,
                "drift_times": base_drift_times,
            }
        
        return {
            "chroma_score": chroma_score,
            "chroma_curve": chroma_curve,
            "spec_dist": spec_dist,
            "spec_diff": spec_diff,
            "onset_errors": errors,
            "drift_mean": drift_mean,
            "drift_p95": drift_p95,
            "drift_max": drift_max,
            "drift_series": drift_series,
            "drift_times": drift_times,
            "baseline": baseline_result,
            "y_aligned": y_aligned,
            "y_ref": y_ref
        }

    def plot_results(self, results):
        chroma_curve = results["chroma_curve"]
        spec_diff = results["spec_diff"]
        onset_errors = results["onset_errors"]

        drift_series = results.get("drift_series", [])
        drift_times = results.get("drift_times", [])
        baseline = results.get("baseline")
        
        plt.figure(figsize=(12, 14))
        
        # Plot 1: Window drift
        plt.subplot(4, 1, 1)
        if drift_series:
            plt.plot(drift_times, drift_series, label="AI对齐漂移", color="royalblue")
        if baseline and baseline.get("drift_series"):
            plt.plot(baseline["drift_times"], baseline["drift_series"], label="仅平移漂移", color="gray", alpha=0.7)
        plt.axhline(0, color="black", linewidth=1)
        plt.axhline(results.get("drift_mean", 0), color="royalblue", linestyle="--", alpha=0.6, label="AI平均漂移")
        if baseline:
            plt.axhline(baseline.get("drift_mean", 0), color="gray", linestyle="--", alpha=0.6, label="基线平均漂移")
        plt.ylabel("漂移 (秒)")
        plt.xlabel("时间 (秒)")
        plt.title("窗口化漂移 (onset 互相关)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Chroma Similarity over time
        plt.subplot(4, 1, 2)
        # librosa default hop_length is 512
        times = np.arange(len(chroma_curve)) * 512 / self.sr 
        plt.plot(times, chroma_curve, label="Chroma Similarity", color='green', alpha=0.8)
        
        # Add a moving average line for better readability
        window_size = 20
        if len(chroma_curve) > window_size:
            moving_avg = np.convolve(chroma_curve, np.ones(window_size)/window_size, mode='valid')
            ma_times = times[window_size-1:]
            plt.plot(ma_times, moving_avg, label="Moving Avg (Smooth)", color='darkgreen', linewidth=2)

        plt.title(f"Chroma 相似度 (Avg: {results['chroma_score']:.3f})")
        plt.ylabel("Similarity (0-1)")
        plt.xlabel("Time (s)")
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        
        # Plot 3: Spectrogram Difference
        plt.subplot(4, 1, 3)
        librosa.display.specshow(spec_diff, x_axis='time', sr=self.sr, cmap='magma', hop_length=512)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Spectral Difference Spectrogram (Darker is better match, Avg: {results['spec_dist']:.1f} dB)")
        
        # Plot 4: Onset Error Histogram
        plt.subplot(4, 1, 4)
        if len(onset_errors) > 0:
            # Convert to ms
            errors_ms = onset_errors * 1000
            plt.hist(errors_ms, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            plt.axvline(0, color='red', linestyle='--', linewidth=2, label="Perfect Alignment")
            plt.axvline(np.median(errors_ms), color='orange', linestyle='-', linewidth=2, label=f"Median: {np.median(errors_ms):.1f}ms")
            plt.title(f"Word Onset Timing Error Distribution (Std Dev: {np.std(errors_ms):.1f}ms)")
            plt.xlabel("Timing Error (ms) - Negative=Early, Positive=Late")
            plt.ylabel("Count (Number of Words)")
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, "No Onsets Detected", ha='center', va='center')
        
        plt.tight_layout()
        plt.show()
