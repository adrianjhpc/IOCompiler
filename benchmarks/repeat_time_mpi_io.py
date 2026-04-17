import subprocess
import argparse
import hashlib
import os
import time
import statistics

def get_sha256(filepath):
    """Calculates the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()
    except Exception as e:
        return None

def benchmark_mpi_execution(binary_path, num_procs, out_file, iterations):
    """Runs an MPI executable multiple times and returns the mean and std dev."""
    print(f"--- Benchmarking {binary_path} on {num_procs} processes ({iterations} iterations) ---")
    times = []
    
    cmd = ["mpiexec", "-n", str(num_procs), binary_path]

    for i in range(iterations):
        if out_file and os.path.exists(out_file):
            os.remove(out_file)
        start_time = time.perf_counter()
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        except subprocess.CalledProcessError as e:
            print(f"Error running MPI command for {binary_path} on iteration {i+1}: {e}")
            return None, None
        except FileNotFoundError:
            print("Executable or MPI launcher not found. Is MPI installed and in your PATH?")
            return None, None

    mean_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if iterations > 1 else 0.0
    
    return mean_time, std_dev

def main():
    parser = argparse.ArgumentParser(description="Compare MPI execution times with statistical analysis and verify integrity.")
    parser.add_argument("standard_exe", help="Path to the unoptimized executable")
    parser.add_argument("fast_exe", help="Path to the optimized executable")
    parser.add_argument("-np", "--num-procs", type=int, default=4, help="Number of MPI processes to use (default: 4)")
    parser.add_argument("--out-file", dest="out_file", help="The name of the log file to verify", default=None)
    parser.add_argument("-i", "--iterations", type=int, default=10, help="Number of times to run each benchmark (default: 10)")
    args = parser.parse_args()

    # --- Run Standard MPI Benchmark ---
    if args.out_file and os.path.exists(args.out_file):
        os.remove(args.out_file) 
        
    std_mean, std_dev = benchmark_mpi_execution(args.standard_exe, args.num_procs, args.out_file, args.iterations)
    std_hash = None
    
    if args.out_file and os.path.exists(args.out_file):
        os.rename(args.out_file, "standard.out")
        std_hash = get_sha256("standard.out")

    # --- Run Optimized MPI Benchmark ---
    if args.out_file and os.path.exists(args.out_file):
        os.remove(args.out_file)
        
    fast_mean, fast_dev = benchmark_mpi_execution(args.fast_exe, args.num_procs, args.out_file, args.iterations)
    fast_hash = None
    
    if args.out_file and os.path.exists(args.out_file):
        os.rename(args.out_file, "fast.out")
        fast_hash = get_sha256("fast.out")

    # --- Print Timing Results ---
    if std_mean is not None and fast_mean is not None:
        # Calculate Coefficient of Variation (CV) as a percentage
        std_cv = (std_dev / std_mean) * 100 if std_mean > 0 else 0.0
        fast_cv = (fast_dev / fast_mean) * 100 if fast_mean > 0 else 0.0

        # Format: Mean ± StdDev (CV%)
        std_display = f"{std_mean:.4f} ± {std_dev:.4f} ({std_cv:.1f}%)"
        fast_display = f"{fast_mean:.4f} ± {fast_dev:.4f} ({fast_cv:.1f}%)"

        # Expanded table width to accommodate the longer strings
        print("\n" + "="*70)
        print(f"{'Metric':<15} | {'Standard':<22} | {'Optimized':<22}")
        print("-" * 70)
        print(f"{'Time (s)':<15} | {std_display:<22} | {fast_display:<22}")
        print("="*70)

        if fast_mean < std_mean:
            reduction = ((std_mean - fast_mean) / std_mean) * 100
            speedup = std_mean / fast_mean
            print(f"SUCCESS: Average execution time reduced by {reduction:.2f}%")
            print(f"SPEEDUP: Optimized version is {speedup:.2f}x faster on average")
        else:
            print("FAILURE: No speedup detected (or optimized version was slower).")
        print("="*70)

    # --- Print Integrity Results ---
    if args.out_file:
        print("\n" + "="*70)
        print("DATA INTEGRITY CHECK (From final iteration):")
        print(f"Standard Hash : {std_hash or 'FILE NOT FOUND'}")
        print(f"Optimized Hash: {fast_hash or 'FILE NOT FOUND'}")
        print("-" * 70)
        if std_hash and fast_hash and std_hash == fast_hash:
            print("RESULT: EXACT MATCH (Mathematically Sound!)")
        else:
            print("RESULT: MISMATCH! (Data corruption detected)")
        print("="*70)

if __name__ == "__main__":
    main()
