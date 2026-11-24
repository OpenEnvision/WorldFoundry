import csv
import os
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from piq import brisque


# ==========================================
# ==========================================
class MetricConfig:
    EPS = 1e-8
    SUPPORTED_FORMATS = (".mp4", ".avi", ".mov", ".mkv", ".ts", ".flv", ".webm")
    EXP_DECAY_ALPHA = -0.2
    MAX_WORKERS = 4
    ENABLE_VISUALIZATION = False

    BRIGHTNESS_LAMDA = 15

    HUE_LAMDA = 15

    NOISE_DEVICE = "auto"

    CLARITY_K = 3
    CLARITY_NOISE_THRESHOLD = 0.5

    MEMORY_ALPHA = 0.1
    MEMORY_K_VAL = 0.0001
    MEMORY_K_EXP = 1.1
    MEMORY_A_OFFSET = 1000


cfg = MetricConfig()

# ==========================================
# ==========================================
EPS = cfg.EPS
SUPPORTED_FORMATS = cfg.SUPPORTED_FORMATS


def _cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1) + EPS
    norm2 = np.linalg.norm(vec2) + EPS
    return dot_product / (norm1 * norm2)


def _modified_softmax_transform(x, lamda):
    x = np.clip(x, 0.0, 1.0)
    numerator = np.exp(lamda * x) - 1
    denominator = np.exp(lamda) - 1 + EPS
    return (
        (numerator / denominator).item() if isinstance(x, (float, np.float32, np.float64)) else numerator / denominator
    )


def _normalized_log_transform(x, k):
    x = np.clip(x, 0.0, 1.0)
    numerator = np.log(1 + k * (x + EPS))
    denominator = np.log(1 + k) + EPS
    return (
        (numerator / denominator).item() if isinstance(x, (float, np.float32, np.float64)) else numerator / denominator
    )


def _exponential_decay_weighted_sum(similarity_array, alpha=None):
    alpha = alpha if alpha is not None else cfg.EXP_DECAY_ALPHA
    if not isinstance(similarity_array, np.ndarray):
        similarity_array = np.array(similarity_array)
    N = len(similarity_array)
    if N < 2:
        return 1.0 if N == 1 else 0.0
    frame_scores = similarity_array[1:]
    frame_distances = np.arange(1, N)
    weights = np.exp(-alpha * frame_distances)
    weight_sum = np.sum(weights) + EPS
    return np.clip(np.dot(weights / weight_sum, frame_scores), 0.0, 1.0)


def _read_video_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        raise Exception(f"：{video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise Exception(f"：{video_path}")
    return frames


def _calculate_mse(img1, img2):
    return np.mean(np.square(img1.astype(np.float32) - img2.astype(np.float32)))


def _map_mse_to_score(mse_value, k_val=None, k_exp=None, a_offset=None):
    k_val = k_val if k_val is not None else cfg.MEMORY_K_VAL
    k_exp = k_exp if k_exp is not None else cfg.MEMORY_K_EXP
    a_offset = a_offset if a_offset is not None else cfg.MEMORY_A_OFFSET

    inner_val = max(mse_value - a_offset, 0)
    score = np.exp(-(k_val * (inner_val**k_exp)))
    return np.clip(score, 0.0, 1.0)


# ==========================================


# ==========================================
class ColorMetric(ABC):
    def __init__(self, name, short_name, save_base_dir):
        self.name = name
        self.short_name = short_name
        self.metric_root = os.path.join(save_base_dir, self.name)
        self.values_dir = os.path.join(self.metric_root, "values")
        self.log_dir = os.path.join(save_base_dir, "logs")
        self.log_file = os.path.join(self.log_dir, f"{self.name}.txt")

        if cfg.ENABLE_VISUALIZATION:
            os.makedirs(self.values_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def get_processed_files(self):
        if not os.path.exists(self.log_file):
            return set()
        with open(self.log_file, encoding="utf-8") as f:
            return {line.strip() for line in f}

    def mark_as_done(self, file_name):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{file_name}\n")

    def visualize(self, file_name, data):
        if not cfg.ENABLE_VISUALIZATION:
            return

        plt.figure(figsize=(10, 4))
        plt.plot(data, color="navy", linewidth=1)
        plt.title(f"{self.short_name} - {file_name}")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.values_dir, f"{os.path.splitext(file_name)[0]}.png"))
        np.save(os.path.join(self.values_dir, f"{os.path.splitext(file_name)[0]}.npy"), data)
        plt.close()

    @abstractmethod
    def calculate(self, frames, **kwargs):
        pass


# ==========================================
# ==========================================
class BrightnessImpl(ColorMetric):
    def calculate(self, frames, lamda=None):
        lamda = lamda if lamda is not None else cfg.BRIGHTNESS_LAMDA

        def get_v(f):
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            return (
                np.array(
                    [np.sum(gray <= 63), np.sum((gray >= 64) & (gray <= 191)), np.sum(gray >= 192)], dtype=np.float32
                )
                / gray.size
            )

        first_v = get_v(frames[0])
        return np.array(
            [_modified_softmax_transform(_cosine_similarity(first_v, get_v(f)), lamda) for f in frames],
            dtype=np.float32,
        )


class HueImpl(ColorMetric):
    def calculate(self, frames, lamda=None):
        lamda = lamda if lamda is not None else cfg.HUE_LAMDA

        def get_v(f):
            h = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)[:, :, 0]
            ints = [(0, 15), (16, 30), (31, 60), (61, 90), (91, 120), (121, 150), (151, 179)]
            return np.array([np.sum((h >= s) & (h <= e)) for s, e in ints], dtype=np.float32) / h.size

        v_list = [get_v(f) for f in frames]
        first_v = v_list[0]

        scores = []

        for i in range(len(v_list)):
            current_v = v_list[i]

            if i == 0:
                sim_avg = 1.0
            else:
                sim1 = _cosine_similarity(first_v, current_v)
                sim2 = _cosine_similarity(v_list[i - 1], current_v)
                sim_avg = (sim1 + sim2) / 2.0

            scores.append(_modified_softmax_transform(sim_avg, lamda))

        return np.array(scores, dtype=np.float32)


class NoiseImpl(ColorMetric):
    def calculate(self, frames, device=None):
        if device is None:
            if cfg.NOISE_DEVICE == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = cfg.NOISE_DEVICE
        scores = [0.0]
        for i in range(1, len(frames)):
            img = (
                torch.from_numpy(np.transpose(frames[i].astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                s = brisque(img, data_range=1.0)
            scores.append(np.clip(s.item() / (100.0 + EPS), 0.0, 1.0))
        return np.array(scores, dtype=np.float32)


class ClarityImpl(ColorMetric):
    def calculate(self, frames, noise_cache, k=None, threshold=None):
        k = k if k is not None else cfg.CLARITY_K
        threshold = threshold if threshold is not None else cfg.CLARITY_NOISE_THRESHOLD

        def get_v(f):
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            # return np.array([np.sum(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0))),
            #                  np.sum(np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1)))], dtype=np.float32)
            return np.array(
                [np.sum(cv2.Sobel(gray, cv2.CV_64F, 1, 0)), np.sum(cv2.Sobel(gray, cv2.CV_64F, 0, 1))], dtype=np.float32
            )

        sims = [1.0]
        noise_counter = 0
        triggered = False

        for i in range(1, len(frames)):
            if noise_cache[i] >= threshold:
                noise_counter += 1
            else:
                noise_counter = 0

            if noise_counter >= 5:
                triggered = True

            v_0, v_c = get_v(frames[0]), get_v(frames[i])
            raw = _cosine_similarity(v_0, v_c)
            # print(v_c)
            # print(raw)
            final = np.clip(1.0 - raw, 0.0, 0.2) if noise_cache[i] >= threshold else raw

            if triggered:
                final = np.clip(1.0 - raw, 0.0, 0.2)

            sims.append(_normalized_log_transform(final, k))

        return np.array(sims, dtype=np.float32)


class MemorySymmetryImpl(ColorMetric):
    def __init__(self, save_base_dir):
        super().__init__("memory_symmetry_mse", "Memory_MSE", save_base_dir)

    def calculate(self, frames, alpha=None):
        alpha = alpha if alpha is not None else cfg.MEMORY_ALPHA
        n = len(frames)
        if n < 2:
            return np.array([0.0], dtype=np.float32)

        num_pairs = n // 2
        mse_list = []

        for i in range(num_pairs):
            mse = _calculate_mse(frames[i], frames[n - 1 - i])
            mse_list.append(mse)

        pair_scores = [_map_mse_to_score(m) for m in mse_list]

        full_frame_scores = [0.0] * n
        for i in range(num_pairs):
            full_frame_scores[i] = pair_scores[i]
            full_frame_scores[n - 1 - i] = pair_scores[i]

        if n % 2 != 0:
            mid_idx = n // 2
            avg_val = (full_frame_scores[mid_idx - 1] + full_frame_scores[mid_idx + 1]) / 2.0
            full_frame_scores[mid_idx] = avg_val

        return np.array(full_frame_scores, dtype=np.float32)

    def compute_final_score(self, data_array, alpha=None):
        alpha = alpha if alpha is not None else cfg.MEMORY_ALPHA
        """
        ： data_array  calculate ，
        """
        n = len(data_array)
        num_pairs = n // 2
        pair_scores = data_array[:num_pairs]

        mid_point = (n - 1) / 2.0
        weights = []
        for i in range(num_pairs):
            dist = abs(i - mid_point)
            weights.append(np.exp(alpha * dist))

        weights = np.array(weights)

        final_score = np.dot(pair_scores, weights) / (np.sum(weights) + EPS)
        return np.clip(final_score, 0.0, 1.0)


# ==========================================
# ==========================================
class AtomicProcessor:
    def __init__(self, video_dir, save_base_dir, max_workers=None):
        self.max_workers = max_workers if max_workers is not None else cfg.MAX_WORKERS
        self.video_dir = video_dir
        self.save_base_dir = save_base_dir
        self.report_dir = os.path.join(save_base_dir, "reports")

        path_parts = os.path.normpath(video_dir).split(os.sep)
        if len(path_parts) >= 2:
            self.dataset_name = f"{path_parts[-2]}_{path_parts[-1]}"
        else:
            self.dataset_name = path_parts[-1]

        self.lock = threading.Lock()
        self.done = 0
        os.makedirs(self.report_dir, exist_ok=True)

    def run(self, metric_obj, compute_func, params_tag):
        files = [
            f for f in os.listdir(self.video_dir) if not f.startswith(".") and f.lower().endswith(SUPPORTED_FORMATS)
        ]
        total = len(files)
        self.done = 0
        start_t = time.time()
        print(f"\n>>> : {metric_obj.short_name} | : {self.dataset_name} | : {total}")
        print(f">>> : {'(PNG/NPY)' if cfg.ENABLE_VISUALIZATION else '()'}")

        def _worker(file_name):
            try:
                v_path = os.path.join(self.video_dir, file_name)
                if file_name in metric_obj.get_processed_files():
                    with self.lock:
                        self.done += 1
                    return None

                frames = _read_video_all_frames(v_path)
                data_array = compute_func(frames)
                score = _exponential_decay_weighted_sum(data_array)

                metric_obj.visualize(file_name, data_array)
                metric_obj.mark_as_done(file_name)

                with self.lock:
                    self.done += 1
                    pct = (self.done / total) * 100
                    print(
                        f"\r: [{(int(pct // 2) * '='): <50}] {pct:.1f}% | {self.done}/{total} | {time.time() - start_t:.1f}s",
                        end="",
                    )
                return (file_name, score)
            except Exception as e:
                print(f"\n[] {file_name}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            new_results = list(executor.map(_worker, files))

        new_data_dict = {res[0]: res[1] for res in new_results if res}
        self._merge_and_write_csv(metric_obj, new_data_dict, params_tag)

    def _merge_and_write_csv(self, metric_obj, new_data, params_tag):
        v_csv = os.path.join(self.report_dir, f"video_{metric_obj.short_name}_{self.dataset_name}_{params_tag}.csv")
        s_csv = os.path.join(self.report_dir, f"summary_{metric_obj.short_name}_{self.dataset_name}_{params_tag}.csv")

        all_data = {}
        if os.path.exists(v_csv):
            try:
                with open(v_csv, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        if len(row) >= 2:
                            all_data[row[0]] = float(row[1])
            except Exception as e:
                print(f"\n[]  CSV ，: {e}")

        all_data.update(new_data)

        with open(v_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Video_Name", f"{metric_obj.short_name}_Score"])
            for n, v in sorted(all_data.items()):
                w.writerow([n, round(v, 6)])

        with open(s_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([f"{metric_obj.short_name}_Avg_Score"])
            avg = np.mean(list(all_data.values())) if all_data else 0.0
            w.writerow([round(avg, 6)])

        print(f"\n[] ，：{v_csv}")


# ==========================================
# ==========================================
def calculate_brightness(video_dir, save_dir, lamda=None, max_workers=None):
    lamda = lamda if lamda is not None else cfg.BRIGHTNESS_LAMDA
    max_workers = max_workers if max_workers is not None else cfg.MAX_WORKERS

    obj = BrightnessImpl("color_brightness_cosine_similarity", "Brightness", save_dir)
    engine = AtomicProcessor(video_dir, save_dir, max_workers)
    engine.run(obj, lambda f: obj.calculate(f, lamda=lamda), f"L{lamda}")


def calculate_hue(video_dir, save_dir, lamda=None, max_workers=None):
    lamda = lamda if lamda is not None else cfg.HUE_LAMDA
    max_workers = max_workers if max_workers is not None else cfg.MAX_WORKERS

    obj = HueImpl("color_hue_cosine_similarity", "Hue", save_dir)
    engine = AtomicProcessor(video_dir, save_dir, max_workers)
    engine.run(obj, lambda f: obj.calculate(f, lamda=lamda), f"L{lamda}")


def calculate_noise(video_dir, save_dir, max_workers=None):
    max_workers = max_workers if max_workers is not None else cfg.MAX_WORKERS

    obj = NoiseImpl("color_brisque_noise", "Noise_BRISQUE", save_dir)
    engine = AtomicProcessor(video_dir, save_dir, max_workers)
    engine.run(obj, lambda f: obj.calculate(f), "BRISQUE")


def calculate_clarity(video_dir, save_dir, k=None, threshold=None, max_workers=None):
    k = k if k is not None else cfg.CLARITY_K
    threshold = threshold if threshold is not None else cfg.CLARITY_NOISE_THRESHOLD
    max_workers = max_workers if max_workers is not None else cfg.MAX_WORKERS

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.NOISE_DEVICE == "auto" else cfg.NOISE_DEVICE

    n_obj = NoiseImpl("color_brisque_noise_for_clarity", "Noise_Internal", save_dir)
    c_obj = ClarityImpl("color_tenengrad_clarity_similarity", "Clarity_Tenengrad", save_dir)

    def _logic(frames):
        n_arr = n_obj.calculate(frames, device=dev)
        return c_obj.calculate(frames, noise_cache=n_arr, k=k, threshold=threshold)

    engine = AtomicProcessor(video_dir, save_dir, max_workers)
    engine.run(c_obj, _logic, f"K{k}_T{threshold}")


def calculate_memory(video_dir, save_dir, alpha=None, max_workers=None):
    alpha = alpha if alpha is not None else cfg.MEMORY_ALPHA
    max_workers = max_workers if max_workers is not None else cfg.MAX_WORKERS

    obj = MemorySymmetryImpl(save_dir)

    def _memory_logic(frames):
        mse_array = obj.calculate(frames, alpha=alpha)

        return mse_array

    class MemoryEngine(AtomicProcessor):
        def _get_score(self, data_array):
            return obj.compute_final_score(data_array, alpha=alpha)

    mem_engine = MemoryEngine(video_dir, save_dir, max_workers)

    mem_engine.run(obj, lambda f: obj.calculate(f, alpha=alpha), f"alpha{alpha}")


def calculate_all(video_dir, save_dir, max_workers=None):
    max_workers = max_workers if max_workers is not None else cfg.MAX_WORKERS

    calculate_brightness(video_dir, save_dir, max_workers=max_workers)
    calculate_hue(video_dir, save_dir, max_workers=max_workers)
    calculate_noise(video_dir, save_dir, max_workers=max_workers)
    calculate_clarity(video_dir, save_dir, max_workers=max_workers)
