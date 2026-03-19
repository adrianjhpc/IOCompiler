import subprocess
import re
import argparse
import time
import shutil
import os
from statistics import mean

# --- CUSTOM BENCHMARK SQL ---
SETUP_SQL = """
DROP TABLE IF EXISTS wal_stress_test;
DROP TABLE IF EXISTS data_stress_test;

CREATE TABLE wal_stress_test (
    id SERIAL PRIMARY KEY,
    data TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE UNLOGGED TABLE data_stress_test (
    id SERIAL PRIMARY KEY,
    data TEXT
);

CREATE OR REPLACE FUNCTION bench_unlogged_io(iterations INTEGER) RETURNS void AS $$
BEGIN
    FOR i IN 1..iterations LOOP
        INSERT INTO data_stress_test (data) VALUES (repeat('y', 100));
    END LOOP;
END;
$$ LANGUAGE plpgsql;
"""

def prepare_sql_files():
    """Writes the benchmark SQL commands to disk so pgbench can use them."""
    with open("setup_schema.sql", "w") as f:
        f.write(SETUP_SQL)
    with open("insert_stress.sql", "w") as f:
        f.write("INSERT INTO wal_stress_test (data) VALUES (repeat('x', 100));\n")
    with open("run_unlogged.sql", "w") as f:
        f.write("SELECT bench_unlogged_io(10000);\n")

def run_cmd(cmd, check=True, capture=False, env=None):
    """Executes a shell command and optionally captures output, passing specific ENV vars."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
        
    result = subprocess.run(cmd, shell=True, text=True,
                            capture_output=capture, check=check, env=full_env)
    return result.stdout if capture else None

def parse_pgbench_output(output):
    """Extracts TPS and Latency from pgbench output."""
    tps_match = re.search(r"tps = ([\d\.]+)", output)
    lat_match = re.search(r"latency average = ([\d\.]+)", output)

    tps = float(tps_match.group(1)) if tps_match else 0.0
    lat = float(lat_match.group(1)) if lat_match else 0.0
    return tps, lat

def setup_db(bin_dir, db_dir, inject_custom=False):
    """Initializes and starts a fresh database, then runs pgbench init."""
    print(f"  [Setup] Initializing and starting DB in {db_dir}...")
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    run_cmd(f"{bin_dir}/initdb -D {db_dir}", capture=True)
    # Put the logfile inside the db_dir to keep the working directory clean
    run_cmd(f"{bin_dir}/pg_ctl -D {db_dir} -l {db_dir}/logfile -w start", capture=True)
    run_cmd(f"{bin_dir}/createdb test_perf", capture=True)
    run_cmd(f"{bin_dir}/pgbench -i -s 10 test_perf", capture=True)
    
    if inject_custom:
        print("  [Setup] Injecting Custom Small I/O Schema...")
        run_cmd(f"{bin_dir}/psql -d test_perf -f setup_schema.sql", capture=True)

def teardown_db(bin_dir, db_dir):
    """Stops and destroys the database."""
    print(f"  [Teardown] Stopping and cleaning DB in {db_dir}...")
    # Use -m fast to forcefully disconnect pgbench if it hangs
    run_cmd(f"{bin_dir}/pg_ctl -D {db_dir} -m fast stop", check=False, capture=True)
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

def stop_db(bin_dir, db_dir):
    """Stops the database."""
    print(f"  [Stopping] Stopping the database...")
    # Use -m fast to forcefully disconnect pgbench if it hangs
    run_cmd(f"{bin_dir}/pg_ctl -D {db_dir} -m fast stop", check=False, capture=True)

def start_db(bin_dir, db_dir):
    """Starts the database."""
    print(f"  [Startup] Starting the database...")
    run_cmd(f"{bin_dir}/pg_ctl -D {db_dir} -l {db_dir}/logfile -w start", capture=True)

def drop_os_caches():
    """Drops the OS pagecache to ensure read benchmarks hit the disk."""
    run_cmd("sync", capture=True)
    run_cmd("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", check=False, capture=True)

def main():
    parser = argparse.ArgumentParser(description="PostgreSQL Full Benchmark Harness")
    parser.add_argument("-r", "--runs", type=int, default=3, help="Number of benchmark iterations to run")
    parser.add_argument("-b", "--bin-dir", type=str, default="./bin", help="Path to postgres bin directory")
    parser.add_argument("-d", "--db-dir", type=str, default="./test_db", help="Path to the database data directory")
    parser.add_argument("-t", "--time", type=int, default=60, help="Duration to run each pgbench test in seconds")
    args = parser.parse_args()

    # Define the compiler environments we want to compare
    configs = [
        {"name": "Baseline (Vanilla)", "env": {}}
    ]

    # Initialize nested results dictionary
    metrics = {cfg["name"]: {'std_write': [], 'std_read': [], 'wal_stress': [], 'data_stress': [], 
                             'std_write_lat': [], 'std_read_lat': [], 'wal_stress_lat': [], 'data_stress_lat': []} 
               for cfg in configs}

    # Setup sudo cache once at the beginning
    run_cmd("sudo bash -c exit")
    prepare_sql_files()

    print(f"=== Starting Full Benchmark Harness ({args.runs} Runs per Config) ===")

    for config in configs:
        cfg_name = config["name"]
        env_vars = config["env"]
        print(f"\n{'='*60}\nEvaluating Configuration: {cfg_name}\n{'='*60}")

        for i in range(args.runs):
            print(f"\n--- Run {i+1} of {args.runs} ---")

            # 1. Standard Write Test (TPC-B)
            setup_db(args.bin_dir, args.db_dir, inject_custom=False)
            print("  Running Standard Write Test (TPC-B)...")
            out = run_cmd(f"{args.bin_dir}/pgbench -c 10 -j 2 -T {args.time} test_perf", capture=True, env=env_vars)
            tps, lat = parse_pgbench_output(out)
            metrics[cfg_name]['std_write'].append(tps); metrics[cfg_name]['std_write_lat'].append(lat)
            print(f"    Result -> TPS: {tps:.2f}, Latency: {lat:.3f} ms")
            teardown_db(args.bin_dir, args.db_dir)

            # 2. Standard Read Test
            setup_db(args.bin_dir, args.db_dir, inject_custom=False)
            stop_db(args.bin_dir, args.db_dir)
            drop_os_caches()
            start_db(args.bin_dir, args.db_dir)
            print("  Running Standard Read Test (Select Only)...")
            out = run_cmd(f"{args.bin_dir}/pgbench -S -c 20 -j 4 -T {args.time} test_perf", capture=True, env=env_vars)
            tps, lat = parse_pgbench_output(out)
            metrics[cfg_name]['std_read'].append(tps); metrics[cfg_name]['std_read_lat'].append(lat)
            print(f"    Result -> TPS: {tps:.2f}, Latency: {lat:.3f} ms")
            teardown_db(args.bin_dir, args.db_dir)

            # 3. Custom WAL Stress Test (Small I/O)
            setup_db(args.bin_dir, args.db_dir, inject_custom=True)
            print("  Running Custom WAL Stress Test (100-byte Inserts)...")
            out = run_cmd(f"{args.bin_dir}/pgbench -c 10 -j 2 -T {args.time} -f insert_stress.sql test_perf", capture=True, env=env_vars)
            tps, lat = parse_pgbench_output(out)
            metrics[cfg_name]['wal_stress'].append(tps); metrics[cfg_name]['wal_stress_lat'].append(lat)
            print(f"    Result -> TPS: {tps:.2f}, Latency: {lat:.3f} ms")
            teardown_db(args.bin_dir, args.db_dir)

            # 4. Custom Data File Stress (Unlogged Table)
            setup_db(args.bin_dir, args.db_dir, inject_custom=True)
            drop_os_caches()
            print("  Running Custom Data Stress Test (Unlogged Function)...")
            # Note: -t 100 runs the function exactly 100 times per client, generating 1,000,000 internal inserts total
            out = run_cmd(f"{args.bin_dir}/pgbench -c 1 -t 100 -f run_unlogged.sql test_perf", capture=True, env=env_vars)
            tps, lat = parse_pgbench_output(out)
            metrics[cfg_name]['data_stress'].append(tps); metrics[cfg_name]['data_stress_lat'].append(lat)
            print(f"    Result -> TPS: {tps:.2f}, Latency: {lat:.3f} ms")
            teardown_db(args.bin_dir, args.db_dir)

    # --- PRINT AGGREGATE RESULTS ---
    print("\n" + "="*80)
    print(" " * 26 + "FINAL AGGREGATE RESULTS")
    print("="*80)

    for config in configs:
        cfg_name = config["name"]
        print(f"\nConfiguration: {cfg_name}")
        print("-" * 80)
        print(f"{'Test':<16} | {'Avg TPS':>10} | {'Min TPS':>10} | {'Max TPS':>10} | {'Avg Lat (ms)':>12}")
        print("-" * 80)
        
        tests = [
            ("Std Write", 'std_write', 'std_write_lat'),
            ("Std Read", 'std_read', 'std_read_lat'),
            ("WAL Stress", 'wal_stress', 'wal_stress_lat'),
            ("Data Stress", 'data_stress', 'data_stress_lat')
        ]
        
        for display_name, tps_key, lat_key in tests:
            tps_data = metrics[cfg_name][tps_key]
            lat_data = metrics[cfg_name][lat_key]
            
            avg_tps = mean(tps_data)
            min_tps = min(tps_data)
            max_tps = max(tps_data)
            avg_lat = mean(lat_data)
            
            print(f"{display_name:<16} | {avg_tps:10.2f} | {min_tps:10.2f} | {max_tps:10.2f} | {avg_lat:12.3f}")
            
    print("="*80 + "\n")

    # Cleanup temporary sql files
    for f in ["setup_schema.sql", "insert_stress.sql", "run_unlogged.sql"]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()
