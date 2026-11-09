"""NVML VRAM sampling (device-level, includes CUDA context and allocator overhead) with phase peaks."""

import threading
import time

import pynvml


class VramSampler:
    """Samples NVML used memory of one physical GPU every `interval_s` on a background thread."""

    def __init__(self, physical_index, interval_s=0.05):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        self.interval_s = interval_s
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.total_mib = info.total / 2**20
        self.baseline_mib = info.used / 2**20
        self.peak_mib = self.baseline_mib
        self.phase = "start"
        self.phase_peaks = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            used = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2**20
            self.peak_mib = max(self.peak_mib, used)
            self.phase_peaks[self.phase] = max(self.phase_peaks.get(self.phase, 0.0), used)
            time.sleep(self.interval_s)

    def start(self):
        self._thread.start()
        return self

    def mark(self, phase):
        self.phase = phase

    def stop(self):
        self._stop.set()
        self._thread.join()

    def report(self):
        import torch

        return {
            "nvml_total_mib": round(self.total_mib, 1),
            "nvml_baseline_mib": round(self.baseline_mib, 1),
            "nvml_peak_mib": round(self.peak_mib, 1),
            "phase_peaks_mib": {k: round(v, 1) for k, v in self.phase_peaks.items()},
            "torch_max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1) if torch.cuda.is_available() else None,
            "interval_s": self.interval_s,
        }


def gpu_info(physical_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
    name = pynvml.nvmlDeviceGetName(handle)
    return {
        "gpu_name": name.decode() if isinstance(name, bytes) else name,
        "driver_version": pynvml.nvmlSystemGetDriverVersion(),
    }
