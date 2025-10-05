import argparse
import os
import shutil
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from deepspeed.ops.op_builder import AsyncIOBuilder
from scipy.stats import t
from tqdm import tqdm


def parse_size(size_str: str) -> int:
    """Parse human-readable size (e.g., '2M' -> 2097152)."""
    size_str = size_str.upper().strip()
    if size_str.endswith('K'):
        return int(float(size_str[:-1]) * 1024)
    elif size_str.endswith('M'):
        return int(float(size_str[:-1]) * 1024 * 1024)
    elif size_str.endswith('G'):
        return int(float(size_str[:-1]) * 1024 * 1024 * 1024)
    else:
        return int(size_str)


def calculate_statistics(latencies: List[float], size_bytes: int) -> Dict:
    """Compute mean, median, std dev, 95% CI, and key percentiles for latencies and bandwidths (MB/s)."""
    if not latencies:
        raise ValueError("No latencies provided for statistics.")

    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    std_dev_lat = np.std(latencies)

    # 95% Confidence Interval for latency
    confidence_level = 0.95
    degrees_freedom = len(latencies) - 1
    sample_standard_error = std_dev_lat / np.sqrt(len(latencies))
    ci_lat = t.interval(
        confidence_level, degrees_freedom, loc=mean_lat, scale=sample_standard_error
    )

    lat_percentiles = {
        "50th": np.percentile(latencies, 50),
        "90th": np.percentile(latencies, 90),
        "99th": np.percentile(latencies, 99),
    }

    # Bandwidths in MB/s
    bandwidths = [size_bytes / lat / 1e6 for lat in latencies if lat > 0]
    if len(bandwidths) != len(latencies):
        raise ValueError("Some latencies were zero; cannot compute bandwidth.")

    mean_bw = np.mean(bandwidths)
    median_bw = np.median(bandwidths)
    std_dev_bw = np.std(bandwidths)

    # 95% Confidence Interval for bandwidth
    sample_standard_error_bw = std_dev_bw / np.sqrt(len(bandwidths))
    ci_bw = t.interval(
        confidence_level, degrees_freedom, loc=mean_bw, scale=sample_standard_error_bw
    )

    bw_percentiles = {
        "50th": np.percentile(bandwidths, 50),
        "90th": np.percentile(bandwidths, 90),
        "99th": np.percentile(bandwidths, 99),
    }

    return {
        "latency": {
            "mean": mean_lat,
            "median": median_lat,
            "std_dev": std_dev_lat,
            "confidence_interval": ci_lat,
            "percentiles": lat_percentiles,
        },
        "bandwidth_mb_s": {
            "mean": mean_bw,
            "median": median_bw,
            "std_dev": std_dev_bw,
            "confidence_interval": ci_bw,
            "percentiles": bw_percentiles,
        },
    }


def get_aio_handle(block_size: int, queue_depth: int, threads: int):
    """Initialize DeepSpeed AsyncIO handle."""
    try:
        aio_op = AsyncIOBuilder().load(verbose=True)
        return aio_op.aio_handle(block_size, queue_depth, False, True, threads)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize AIO handle: {e}")


def benchmark(aio_handle, tensor_sizes: List[int], iterations: int, nvme_path: str):
    """Run read/write benchmarks for given sizes."""
    nvme_dir = Path(nvme_path)
    nvme_dir.mkdir(parents=True, exist_ok=True)

    for size in tensor_sizes:
        print(f"Benchmarking size: {size} bytes")
        test_files = []

        # Write benchmarks
        write_latencies = []
        for i in tqdm(range(iterations), desc="Writes"):
            tensor = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
            test_file = nvme_dir / f"test_write_{size}_{i}.swap"
            test_files.append(test_file)

            start = time.perf_counter()
            aio_handle.async_pwrite(tensor, str(test_file), 0)
            aio_handle.wait()
            end = time.perf_counter()

            write_latencies.append(end - start)

        write_stats = calculate_statistics(write_latencies, size)
        print("Write Statistics:")
        print(f"  Latency Mean: {write_stats['latency']['mean']:.9f}s")
        print(f"  Latency Median: {write_stats['latency']['median']:.9f}s")
        print(f"  Latency Std Dev: {write_stats['latency']['std_dev']:.9f}s")
        print(f"  Latency 95% CI: ({write_stats['latency']['confidence_interval'][0]:.9f}, {write_stats['latency']['confidence_interval'][1]:.9f})")
        print(f"  Latency Percentiles: {write_stats['latency']['percentiles']}")
        print(f"  Bandwidth Mean: {write_stats['bandwidth_mb_s']['mean']:.3f} MB/s")
        print(f"  Bandwidth Median: {write_stats['bandwidth_mb_s']['median']:.3f} MB/s")
        print(f"  Bandwidth Std Dev: {write_stats['bandwidth_mb_s']['std_dev']:.3f} MB/s")
        print(f"  Bandwidth 95% CI: ({write_stats['bandwidth_mb_s']['confidence_interval'][0]:.3f}, {write_stats['bandwidth_mb_s']['confidence_interval'][1]:.3f})")
        print(f"  Bandwidth Percentiles: {write_stats['bandwidth_mb_s']['percentiles']}")

        # Read benchmarks (reuse write files)
        read_latencies = []
        for i in tqdm(range(iterations), desc="Reads"):
            tensor = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
            test_file = test_files[i]

            start = time.perf_counter()
            aio_handle.async_pread(tensor, str(test_file), 0)
            aio_handle.wait()
            end = time.perf_counter()

            read_latencies.append(end - start)

        read_stats = calculate_statistics(read_latencies, size)
        print("Read Statistics:")
        print(f"  Latency Mean: {read_stats['latency']['mean']:.9f}s")
        print(f"  Latency Median: {read_stats['latency']['median']:.9f}s")
        print(f"  Latency Std Dev: {read_stats['latency']['std_dev']:.9f}s")
        print(f"  Latency 95% CI: ({read_stats['latency']['confidence_interval'][0]:.9f}, {read_stats['latency']['confidence_interval'][1]:.9f})")
        print(f"  Latency Percentiles: {read_stats['latency']['percentiles']}")
        print(f"  Bandwidth Mean: {read_stats['bandwidth_mb_s']['mean']:.3f} MB/s")
        print(f"  Bandwidth Median: {read_stats['bandwidth_mb_s']['median']:.3f} MB/s")
        print(f"  Bandwidth Std Dev: {read_stats['bandwidth_mb_s']['std_dev']:.3f} MB/s")
        print(f"  Bandwidth 95% CI: ({read_stats['bandwidth_mb_s']['confidence_interval'][0]:.3f}, {read_stats['bandwidth_mb_s']['confidence_interval'][1]:.3f})")
        print(f"  Bandwidth Percentiles: {read_stats['bandwidth_mb_s']['percentiles']}")

        # Cleanup files for this size
        for file in test_files:
            file.unlink(missing_ok=True)
        print(f"Cleaned up files for size {size}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSpeed AsyncIO read/write latencies and bandwidths on NVMe."
    )
    parser.add_argument(
        "--nvme-path",
        default="/mnt/nvme/test",
        help="Path to NVMe directory (default: /mnt/nvme/test)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["2M", "8M", "32M"],
        help="Tensor sizes (e.g., 2M 8M; supports K/M/G) (default: 2M 8M 32M)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Iterations per size (default: 200)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="AIO threads (default: 16)",
    )
    parser.add_argument(
        "--queue-depth",
        type=int,
        default=64,
        help="AIO queue depth (default: 64)",
    )
    parser.add_argument(
        "--block-size",
        default="2M",
        help="AIO block size (supports K/M/G) (default: 2M)",
    )

    args = parser.parse_args()

    # Parse sizes and block_size
    try:
        tensor_sizes = [parse_size(s) for s in args.sizes]
        block_size = parse_size(args.block_size)
    except ValueError as e:
        raise ValueError(f"Invalid size format: {e}")

    nvme_path = Path(args.nvme_path).expanduser()  # Handle ~ if needed
    parent_dir = nvme_path.parent
    if not parent_dir.exists() or not os.access(parent_dir, os.W_OK):
        raise ValueError(f"Parent of NVMe path '{nvme_path}' must exist and be writable.")

    # Cleanup NVMe dir
    try:
        shutil.rmtree(nvme_path)
        print(f"Cleaned NVMe directory: {nvme_path}")
    except FileNotFoundError:
        pass
    except Exception as e:
        raise RuntimeError(f"Failed to clean NVMe directory: {e}")

    # Run benchmark
    try:
        aio_handle = get_aio_handle(block_size, args.queue_depth, args.threads)
        benchmark(aio_handle, tensor_sizes, args.iterations, str(nvme_path))
    except Exception as e:
        print(f"Benchmark failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    import sys
    main()