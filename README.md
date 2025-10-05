# DS-AIO-Benchmark

This project benchmarks the latency of asynchronous I/O operations (using DeepSpeed's AsyncIOBuilder) for read and write tasks on an NVMe storage device. It measures performance across varying tensor sizes, providing detailed statistics including mean, median, standard deviation, 95% confidence intervals, and percentiles (50th, 90th, 99th). The goal is to evaluate NVMe throughput under controlled conditions, helping identify bottlenecks in high-performance computing workflows like those in machine learning.

**Note:** This benchmark assumes a Linux environment with NVMe access.

## Prerequisites

- Ubuntu/Debian-based system (or compatible Linux distro).
- Root/sudo access for package installation.
- NVMe device mounted (e.g., at `/mnt/nvme`—configurable via script arguments).
- Python 3.10+.

## Installation

Update your package index and install required system dependencies:

```bash
sudo apt-get update && sudo apt-get install -y python3.10-venv libaio-dev python3.10-dev
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the benchmark with default parameters:

```bash
python benchmark.py
```

For customization, use command-line arguments (see `--help` for full options):

```bash
python benchmark.py --nvme-path /mnt/my-nvme --sizes 2M 8M 32M 256M --iterations 100 --threads 8
```

### Key Arguments
- `--nvme-path` (default: `/mnt/nvme`): Path to the NVMe mount point. Ensure it's empty or backed up—the script will clean it.
- `--sizes` (default: `2M 8M 32M`): Tensor sizes to test (supports K/M/G suffixes, e.g., `4K 1M`).
- `--iterations` (default: `200`): Number of iterations per size for statistical reliability.
- `--threads` (default: `16`): Number of I/O threads.
- `--queue-depth` (default: `64`): AIO queue depth.
- `--block-size` (default: `2M`): Block size for AIO operations (supports K/M/G).

### Example Output
The script prints results to stdout, like:

```
Writes: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:01<00:00, 116.80it/s]
Write Statistics:
  Latency Mean: 0.008422327s
  Latency Median: 0.009718733s
  Latency Std Dev: 0.002184281s
  Latency 95% CI: (0.008117754, 0.008726900)
  Latency Percentiles: {'50th': np.float64(0.00971873349044472), '90th': np.float64(0.00992813630728051), '99th': np.float64(0.012846746661234643)}
  Bandwidth Mean: 1083.946 MB/s
  Bandwidth Median: 863.138 MB/s
  Bandwidth Std Dev: 341.978 MB/s
  Bandwidth 95% CI: (1036.261, 1131.630)
  Bandwidth Percentiles: {'50th': np.float64(863.1380261316615), '90th': np.float64(1591.8030968247208), '99th': np.float64(1613.1072831188094)}
Reads: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:01<00:00, 129.08it/s]
Read Statistics:
  Latency Mean: 0.006736185s
  Latency Median: 0.007893178s
  Latency Std Dev: 0.002424730s
  Latency 95% CI: (0.006398085, 0.007074286)
  Latency Percentiles: {'50th': np.float64(0.007893177564255893), '90th': np.float64(0.009326752624474466), '99th': np.float64(0.009442441564751788)}
  Bandwidth Mean: 1496.787 MB/s
  Bandwidth Median: 1062.767 MB/s
  Bandwidth Std Dev: 703.571 MB/s
  Bandwidth 95% CI: (1398.682, 1594.891)
  Bandwidth Percentiles: {'50th': np.float64(1062.766920840555), '90th': np.float64(2540.1458479962716), '99th': np.float64(2570.0019490567906)}
```