#!/usr/bin/env python3

import subprocess
import time
import re
import statistics
import os
import shutil

# --- Configuration ---
RUNS = 10
SLEEP_TIME = 2  # RAM doesn't thermal throttle, so we can run much faster!
DB_DIR = "/dev/shm/rocksdb_ioopt_test" # Target the RAM disk

DB_BENCH_CMD = [
    "./db_bench",
    "--benchmarks=fillseq,readseq",
    "--use_existing_db=0",
    "--num=5000000",
    "--threads=4",
    "--compression_type=none",  # Force raw I/O throughput
    f"--db={DB_DIR}"
]

fill_pattern = re.compile(r"fillseq\s+:\s+([\d.]+)\s+micros/op\s+(\d+)\s+ops/sec")
read_pattern = re.compile(r"readseq\s+:\s+([\d.]+)\s+micros/op\s+(\d+)\s+ops/sec")

def drop_caches():
    subprocess.run(["sync"], check=True)
    with open("/proc/sys/vm/drop_caches", "w") as f:
        f.write("3\n")

def run_benchmark(run_number):
    print(f"\n[Run {run_number}/{RUNS}] Preparing RAM Disk environment...")
    
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        
    #drop_caches()
    time.sleep(SLEEP_TIME)
    
    print(f"[Run {run_number}/{RUNS}] Executing db_bench on /dev/shm...")
    result = subprocess.run(DB_BENCH_CMD, capture_output=True, text=True, check=True)
    
    fill_ops, fill_lat, read_ops, read_lat = 0, 0.0, 0, 0.0
    for line in result.stdout.splitlines():
        if match := fill_pattern.search(line):
            fill_lat, fill_ops = float(match.group(1)), int(match.group(2))
        elif match := read_pattern.search(line):
            read_lat, read_ops = float(match.group(1)), int(match.group(2))
            
    print(f"  -> Fill: {fill_ops} ops/sec ({fill_lat} us/op)")
    print(f"  -> Read: {read_ops} ops/sec ({read_lat} us/op)")
    return fill_ops, fill_lat, read_ops, read_lat

def main():
    print("=========================================================")
    print(f"  RocksDB RAM DISK (/dev/shm) Benchmark Suite ({RUNS} runs)")
    print("=========================================================")
    
    f_ops_list, f_lat_list, r_ops_list, r_lat_list = [], [], [], []
    
    for i in range(1, RUNS + 1):
        f_ops, f_lat, r_ops, r_lat = run_benchmark(i)
        f_ops_list.append(f_ops)
        f_lat_list.append(f_lat)
        r_ops_list.append(r_ops)
        r_lat_list.append(r_lat)
        
    print("\n=========================================================")
    print("                 FINAL AGGREGATE RESULTS")
    print("=========================================================")
    print(f"Write (fillseq) TPS | Avg: {statistics.mean(f_ops_list):8.2f} | Max: {max(f_ops_list):8.2f} ops/sec")
    print(f"Read (readseq) TPS  | Avg: {statistics.mean(r_ops_list):8.2f} | Max: {max(r_ops_list):8.2f} ops/sec")
    print("---------------------------------------------------------")
    print(f"Write Latency       | Avg: {statistics.mean(f_lat_list):8.2f} us/op")
    print(f"Read Latency        | Avg: {statistics.mean(r_lat_list):8.2f} us/op")
    print("=========================================================")
    
    # Clean up RAM disk
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

if __name__ == "__main__":
    main()
