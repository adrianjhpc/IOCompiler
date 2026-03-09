import subprocess
import re
import argparse
import time
import shutil
import os
from statistics import mean

def run_cmd(cmd, check=True, capture=False):
    result = subprocess.run(cmd, shell=True, text=True, capture_output=capture, check=check)
    return result.stdout if capture else None

def parse_sysbench_output(output):
    """Extracts TPS and Average Latency from sysbench output."""
    # Sysbench format: transactions:  12345 (1234.56 per sec.)
    tps_match = re.search(r"transactions:\s+\d+\s+\(([\d\.]+)\s+per sec\.\)", output)
    # Sysbench format: avg:     1.23
    lat_match = re.search(r"avg:\s+([\d\.]+)", output)
    
    tps = float(tps_match.group(1)) if tps_match else 0.0
    lat = float(lat_match.group(1)) if lat_match else 0.0
    return tps, lat

def setup_mysql(bin_dir, base_dir):
    """Initializes and starts MySQL, then prepares the sysbench data."""
    print("  [Setup] Initializing and starting MySQL...")
    data_dir = f"{base_dir}/mysql_data"
    sock_file = f"{base_dir}/mysql.sock"
    
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        
    # 1. Initialize data directory (Insecure for fast local testing)
    run_cmd(f"{bin_dir}/mysqld --initialize-insecure --datadir={data_dir}", capture=True)
    
    # 2. Start MySQL in the background
    mysql_proc = subprocess.Popen(
        f"{bin_dir}/mysqld --datadir={data_dir} --socket={sock_file} --user=$USER", 
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    
    # 3. Wait for the socket to accept connections
    ready = False
    for _ in range(30):
        try:
            run_cmd(f"{bin_dir}/mysqladmin ping -S {sock_file} --silent", check=True)
            ready = True
            break
        except subprocess.CalledProcessError:
            time.sleep(1)
            
    if not ready:
        raise Exception("MySQL failed to start.")

    # 4. Create database and load Sysbench data
    run_cmd(f"{bin_dir}/mysql -S {sock_file} -u root -e 'CREATE DATABASE sbtest;'", capture=True)
    
    # Sysbench Prepare: Creates 10 tables with 100,000 rows each
    sysbench_cmd = f"sysbench oltp_read_write --db-driver=mysql --mysql-socket={sock_file} --mysql-user=root --mysql-db=sbtest --tables=10 --table-size=100000"
    run_cmd(f"{sysbench_cmd} prepare", capture=True)
    
    return mysql_proc, sock_file, sysbench_cmd

def teardown_mysql(bin_dir, base_dir, mysql_proc, sock_file):
    """Stops MySQL and wipes the data directory."""
    print("  [Teardown] Stopping and cleaning MySQL...")
    run_cmd(f"{bin_dir}/mysqladmin -S {sock_file} -u root shutdown", check=False, capture=True)
    mysql_proc.wait() # Ensure process is fully dead
    
    data_dir = f"{base_dir}/mysql_data"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs", type=int, default=3)
    parser.add_argument("-b", "--bin-dir", type=str, default="/usr/local/mysql/bin", help="Path to mysql bin dir")
    args = parser.parse_args()

    base_dir = os.getcwd()
    metrics = {'write_tps': [], 'write_lat': [], 'read_tps': [], 'read_lat': []}

    print(f"=== Starting MySQL Sysbench Harness ({args.runs} Runs) ===")

    for i in range(args.runs):
        print(f"\n--- Run {i+1} of {args.runs} ---")
        
        # --- WRITE TEST (Mixed OLTP) ---
        db_proc, sock, sb_base = setup_mysql(args.bin_dir, base_dir)
        print("  Running Write Test (oltp_read_write)...")
        write_out = run_cmd(f"{sb_base} --threads=4 --time=60 run", capture=True)
        w_tps, w_lat = parse_sysbench_output(write_out)
        metrics['write_tps'].append(w_tps)
        metrics['write_lat'].append(w_lat)
        print(f"    Result -> TPS: {w_tps:.2f}, Latency: {w_lat:.3f} ms")
        teardown_mysql(args.bin_dir, base_dir, db_proc, sock)

        # --- READ TEST (Select Only) ---
        db_proc, sock, sb_base = setup_mysql(args.bin_dir, base_dir)
        print("  Running Read Test (oltp_read_only)...")
        read_out = run_cmd(f"sysbench oltp_read_only --db-driver=mysql --mysql-socket={sock} --mysql-user=root --mysql-db=sbtest --tables=10 --table-size=100000 --threads=8 --time=60 run", capture=True)
        r_tps, r_lat = parse_sysbench_output(read_out)
        metrics['read_tps'].append(r_tps)
        metrics['read_lat'].append(r_lat)
        print(f"    Result -> TPS: {r_tps:.2f}, Latency: {r_lat:.3f} ms")
        teardown_mysql(args.bin_dir, base_dir, db_proc, sock)

    # --- REPORTING ---
    print("\n=========================================================")
    print("             FINAL MYSQL AGGREGATE RESULTS               ")
    print("=========================================================")
    def print_stats(name, data, unit):
        print(f"{name:15} | Avg: {mean(data):9.2f} | Min: {min(data):9.2f} | Max: {max(data):9.2f} {unit}")

    print_stats("Write TPS", metrics['write_tps'], "TPS")
    print_stats("Read TPS", metrics['read_tps'], "TPS")
    print("-" * 57)
    print_stats("Write Latency", metrics['write_lat'], "ms")
    print_stats("Read Latency", metrics['read_lat'], "ms")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
