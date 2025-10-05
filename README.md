# DS-AIO-Benchmark

This project benchmarks the latency of asynchronous I/O operations (using DeepSpeed's AsyncIOBuilder) for read and write tasks on an NVMe storage device. It measures performance across varying tensor sizes, providing detailed statistics including mean, median, standard deviation, 95% confidence intervals, and percentiles (50th, 90th, 99th). The goal is to evaluate NVMe throughput under controlled conditions, helping identify bottlenecks in high-performance computing workflows like those in machine learning.

**Note:** This benchmark assumes a Linux environment.

## Prerequisites

- Ubuntu/Debian-based system (or compatible Linux distro).
- Root/sudo access for package installation.
- Disk device mounted (e.g., at `/mnt/nvme`—configurable via script arguments).
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
- `--nvme-path` (default: `/mnt/nvme`): Path to the NVMe mount point (it can also be other type of disk). Ensure it's empty or backed up—the script will clean it.
- `--sizes` (default: `2M 8M 32M`): Tensor sizes to test (supports K/M/G suffixes, e.g., `4K 1M`).
- `--iterations` (default: `200`): Number of iterations per size for statistical reliability.
- `--threads` (default: `16`): Number of I/O threads.
- `--queue-depth` (default: `64`): AIO queue depth.
- `--block-size` (default: `2M`): Block size for AIO operations (supports K/M/G).

### Example Output
The script prints results to stdout, like:

```
Benchmarking size: 8388608 bytes
Writes: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 412.36it/s]
Write Statistics:
  Latency Mean: 0.002318887s
  Latency Median: 0.002228687s
  Latency Std Dev: 0.000534179s
  Latency 95% CI: (0.002244401, 0.002393372)
  Latency Percentiles: {'50th': np.float64(0.0022286869352683425), '90th': np.float64(0.0023388271103613077), '99th': np.float64(0.006486591713037342)}
  Bandwidth Mean: 3692.992 MB/s
  Bandwidth Median: 3763.924 MB/s
  Bandwidth Std Dev: 340.833 MB/s
  Bandwidth 95% CI: (3645.467, 3740.517)
  Bandwidth Percentiles: {'50th': np.float64(3763.9241826468337), '90th': np.float64(3868.211800553445), '99th': np.float64(3907.243455674987)}
Reads: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 1454.76it/s]
Read Statistics:
  Latency Mean: 0.000677568s
  Latency Median: 0.000676752s
  Latency Std Dev: 0.000004102s
  Latency 95% CI: (0.000676996, 0.000678140)
  Latency Percentiles: {'50th': np.float64(0.0006767519516870379), '90th': np.float64(0.0006814692867919802), '99th': np.float64(0.0006894060247577727)}
  Bandwidth Mean: 12380.909 MB/s
  Bandwidth Median: 12395.395 MB/s
  Bandwidth Std Dev: 73.861 MB/s
  Bandwidth 95% CI: (12370.610, 12391.208)
  Bandwidth Percentiles: {'50th': np.float64(12395.39536003464), '90th': np.float64(12444.124615030207), '99th': np.float64(12500.948521844197)}
Cleaned up files for size 8388608
```