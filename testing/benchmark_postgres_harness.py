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

def setup_db(bin_dir, db_dir, inject_custom=False, scale_factor=10):
    """Initializes and starts a fresh database, then runs pgbench init."""
    print(f"  [Setup] Initializing DB in {db_dir} (Scale Factor: {scale_factor}). This may take a while...")
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    run_cmd(f"{bin_dir}/initdb -D {db_dir}", capture=True)
    # Put the logfile inside the db_dir to keep the working directory clean
    run_cmd(f"{bin_dir}/pg_ctl -D {db_dir} -l {db_dir}/logfile -w start", capture=True)
    run_cmd(f"{bin_dir}/createdb test_perf", capture=True)
    
    # Initialize pgbench with the requested scale factor.
    # Added -q (quiet) to prevent massive terminal spam on huge datasets.
    # Added --foreign-keys to ensure realistic DB constraints.
    run_cmd(f"{bin_dir}/pgbench -i -s {scale_factor} -q --foreign-keys test_perf", capture=True)
    
    if inject_custom:
        print("  [Setup] Injecting Custom Small I/O Schema...")
        run_cmd(f"{bin_dir}/psql -d test_perf -f setup_schema.sql", capture=True)

def teardown_db(bin_dir, db_dir):
    """Stops and destroys the database."""
    print(f"  [Teardown] Stopping and cleaning DB in {db_dir}...")
    run_cmd(f"{bin_dir}/pg_ctl -D {db_dir} -m fast stop", check=False, capture=True)
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

def stop_db(bin_dir, db_dir):
    """Stops the database."""
    print(f"  [Stopping] Stopping the database...")
    run_cmd(f"{bin_dir}/pg_ctl -D {db_dir} -m fast stop", check=False, capture=True)

def start_db(bin_dir, db_dir):
    """Starts the database."""
    print(f"  [Startup] Starting the database...")
    run_cmd(f"{bin_dir}/pg_ctl -D {db_dir} -l {db_dir}/logfile -w start", capture=True)

def drop_os_caches():
    """Drops the OS pagecache to ensure read benchmarks hit the disk."""
    print("  [Cache] Dropping OS page caches...")
    run_cmd("sync", capture=True)
    run_cmd("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", check=False, capture=True)

def main():
    parser = argparse.ArgumentParser(description="PostgreSQL Full Benchmark Harness")
    parser.add_argument("-r", "--runs", type=int, default=3, help="Number of benchmark iterations to run")
    parser.add_argument("-b", "--bin-dir", type=str, default="./bin", help="Path to postgres bin directory")
    parser.add_argument("-d", "--db-dir", type=str, default="./test_db", help="Path to the database data directory")
    parser.add_argument("-t", "--time", type=int, default=60, help="Duration to run each pgbench test in seconds")
    args = parser.parse_args()

    configs = [
        {"name": "Baseline (Vanilla)", "env": {}}
    ]

    metrics = {cfg["name"]: {'std_write': [], 'std_read': [], 'wal_stress': [], 'data_stress': [], 
                             'std_write_lat': [], 'std_read_lat': [], 'wal_stress_lat': [], 'data_stress_lat': []} 
               for cfg in configs}

    # Setup sudo cache once at the beginning
    run_cmd("sudo bash -c exit")
    prepare_sql_files()

    # The Scale Factor for the massive tests (~160GB dataset)
    MASSIVE_SF = 10000

    print(f"=== Starting Full Benchmark Harness ({args.runs} Runs per Config) ===")

    for config in configs:
        cfg_name = config["name"]
        env_vars = config["env"]
        print(f"\n{'='*60}\nEvaluating Configuration: {cfg_name}\n{'='*60}")

        # Initialize the massive DB once for both Read and Write tests
        setup_db(args.bin_dir, args.db_dir, inject_custom=False, scale_factor=MASSIVE_SF)

        for i in range(args.runs):
            print(f"\n--- Run {i+1} of {args.runs} ---")

            # -----------------------------------------------------------------
            # PHASE 1: MASSIVE DATASET TESTS (> 77GB RAM)
            # -----------------------------------------------------------------

            # 1. Standard Read Test (Run FIRST to get pristine data)
            stop_db(args.bin_dir, args.db_dir)
            drop_os_caches()
            start_db(args.bin_dir, args.db_dir)
            
            print(f"  Running Standard Read Test (Massive {MASSIVE_SF} SF)...")
            out = run_cmd(f"{args.bin_dir}/pgbench -S -c 32 -j 8 -T {args.time} test_perf", capture=True, env=env_vars)
            tps, lat = parse_pgbench_output(out)
            metrics[cfg_name]['std_read'].append(tps); metrics[cfg_name]['std_read_lat'].append(lat)
            print(f"    Result -> TPS: {tps:.2f}, Latency: {lat:.3f} ms")

            # 2. Standard Write Test (Run SECOND on the existing massive DB)
            print(f"  Running Standard Write Test (Massive {MASSIVE_SF} SF)...")
            # Lowering clients slightly for TPC-B write on a massive DB to prevent immediate locking bottlenecks
            out = run_cmd(f"{args.bin_dir}/pgbench -c 16 -j 4 -T {args.time} test_perf", capture=True, env=env_vars)
            tps, lat = parse_pgbench_output(out)
            metrics[cfg_name]['std_write'].append(tps); metrics[cfg_name]['std_write_lat'].append(lat)
            print(f"    Result -> TPS: {tps:.2f}, Latency: {lat:.3f} ms")
            
        # Now we can safely tear down the 160GB monster
        teardown_db(args.bin_dir, args.db_dir)

        for i in range(args.runs):
            print(f"\n--- Run {i+1} of {args.runs} ---")

            # -----------------------------------------------------------------
            # PHASE 2: CUSTOM STRESS TESTS (Small Data / High Protocol Overhead)
            # -----------------------------------------------------------------
            # 3. Custom WAL Stress Test
            setup_db(args.bin_dir, args.db_dir, inject_custom=True, scale_factor=10)
            print("  Running Custom WAL Stress Test (100-byte Inserts)...")
            out = run_cmd(f"{args.bin_dir}/pgbench -c 10 -j 2 -T {args.time} -f insert_stress.sql test_perf", capture=True, env=env_vars)
            tps, lat = parse_pgbench_output(out)
            metrics[cfg_name]['wal_stress'].append(tps); metrics[cfg_name]['wal_stress_lat'].append(lat)
            print(f"    Result -> TPS: {tps:.2f}, Latency: {lat:.3f} ms")
            teardown_db(args.bin_dir, args.db_dir)

            # 4. Custom Data File Stress (Unlogged Table)
            setup_db(args.bin_dir, args.db_dir, inject_custom=True, scale_factor=10)
            drop_os_caches()
            print("  Running Custom Data Stress Test (Unlogged Function)...")
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

    for f in ["setup_schema.sql", "insert_stress.sql", "run_unlogged.sql"]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()
