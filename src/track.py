import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import psutil


@dataclass
class ResourceUsage:
    duration_seconds: float
    peak_rss_mb: float


@contextmanager
def track_resources(poll_interval_seconds: float = 0.1):
    """
    Tracks wall-clock time and peak RSS (resident memory) for the code
    inside the `with` block. Peak RSS is sampled by a background thread,
    since the peak generally occurs mid-execution, not at the end.

    Usage:
        with track_resources() as usage:
            do_something()
        print(usage.duration_seconds, usage.peak_rss_mb)
    """
    process = psutil.Process()
    peak_rss = [process.memory_info().rss] 
    stop_event = threading.Event()

    def poll():
        while not stop_event.is_set():
            rss = process.memory_info().rss
            if rss > peak_rss[0]:
                peak_rss[0] = rss
            stop_event.wait(poll_interval_seconds)

    poller = threading.Thread(target=poll, daemon=True)
    usage = ResourceUsage(duration_seconds=0.0, peak_rss_mb=0.0)

    start = time.perf_counter()
    poller.start()
    try:
        yield usage
    finally:
        stop_event.set()
        poller.join()
        usage.duration_seconds = time.perf_counter() - start
        usage.peak_rss_mb = peak_rss[0] / (1024 ** 2)