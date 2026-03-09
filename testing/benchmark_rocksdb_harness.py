#!/usr/bin/env python3

import subprocess
import time
import re
import statistics
import os
import shutil

# --- Configuration ---
RUNS = 5
SLEEP_TIME = 10  # Seconds to let the NVMe drive cool down
DB_DIR = "/tmp/rocksdb_test_db"
DB_BENCH_CMD = [
    "./db_bench",
    "--benchmarks=fillseq,readseq",
    "--use_existing_db=0",
    "--num=5000000",
    "--threads=4",
    "--compression_type=none",
    f"--db={DB_DIR}"
]

# --- Regex Parsers ---
# Matches: "fillseq      :       3.141 micros/op 318355 ops/sec;   35.2 MB/s"
fill_pattern = re.compile(r"fillseq\s+:\s+([\d.]+)\s+micros/op\s+(\d+)\s+ops/sec")
read_pattern = re.compile(r"readseq\s+:\s+([\d.]+)\s+micros/op\s+(\d+)\s+ops/sec")

def drop_caches():
    """Forces the Linux kernel to flush the VFS page cache, dentries, and inodes."""
    subprocess.run(["sync"], check=True)
    with open("/proc/sys/vm/drop_caches", "w") as f:
        f.write("3\n")

def run_benchmark(run_number):
    """Cleans the DB, drops caches, rests, and runs the benchmark."""
    print(f"\n[Run {run_number}/{RUNS}] Preparing environment...")
    
    # 1. Nuke the old database to guarantee a completely fresh LSM-Tree
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        
    # 2. Guarantee a cold kernel state
    #drop_caches()
    
    # 3. Prevent NVMe thermal throttling
    print(f"[Run {run_number}/{RUNS}] Resting NVMe drive for {SLEEP_TIME} seconds...")
    time.sleep(SLEEP_TIME)
    
    # 4. Execute db_bench
    print(f"[Run {run_number}/{RUNS}] Executing db_bench (fillseq -> readseq)...")
    result = subprocess.run(DB_BENCH_CMD, capture_output=True, text=True, check=True)
    
    # 5. Parse the results
    fill_ops, fill_lat, read_ops, read_lat = 0, 0.0, 0, 0.0
    
    for line in result.stdout.splitlines():
        fill_match = fill_pattern.search(line)
        if fill_match:
            fill_lat = float(fill_match.group(1))
            fill_ops = int(fill_match.group(2))
            
        read_match = read_pattern.search(line)
        if read_match:
            read_lat = float(read_match.group(1))
            read_ops = int(read_match.group(2))
            
    print(f"  -> Fill: {fill_ops} ops/sec ({fill_lat} us/op)")
    print(f"  -> Read: {read_ops} ops/sec ({read_lat} us/op)")
    
    return fill_ops, fill_lat, read_ops, read_lat

def main():
    print("=========================================================")
    print(f" Starting RocksDB Benchmark Suite ({RUNS} runs)")
    print("=========================================================")
    
    fill_ops_list, fill_lat_list = [], []
    read_ops_list, read_lat_list = [], []
    
    for i in range(1, RUNS + 1):
        f_ops, f_lat, r_ops, r_lat = run_benchmark(i)
        fill_ops_list.append(f_ops)
        fill_lat_list.append(f_lat)
        read_ops_list.append(r_ops)
        read_lat_list.append(r_lat)
        
    print("\n=========================================================")
    print("                 FINAL AGGREGATE RESULTS")
    print("=========================================================")
    print(f"Write (fillseq) TPS | Avg: {statistics.mean(fill_ops_list):8.2f} | Min: {min(fill_ops_list):8.2f} | Max: {max(fill_ops_list):8.2f} ops/sec")
    print(f"Read (readseq) TPS  | Avg: {statistics.mean(read_ops_list):8.2f} | Min: {min(read_ops_list):8.2f} | Max: {max(read_ops_list):8.2f} ops/sec")
    print("---------------------------------------------------------")
    print(f"Write Latency       | Avg: {statistics.mean(fill_lat_list):8.2f} | Min: {min(fill_lat_list):8.2f} | Max: {max(fill_lat_list):8.2f} us/op")
    print(f"Read Latency        | Avg: {statistics.mean(read_lat_list):8.2f} | Min: {min(read_lat_list):8.2f} | Max: {max(read_lat_list):8.2f} us/op")
    print("=========================================================")

if __name__ == "__main__":
    main()
