#!/bin/bash

# Configuration
RUNS=5
SLEEP_TIME=10
FIO_BIN="./fio" # Point this to your IOOpt-compiled FIO, or /usr/bin/fio for baseline

echo "========================================================="
echo " Starting FIO Benchmark Suite ($RUNS runs per config)"
echo " Binary: $FIO_BIN"
echo "========================================================="

run_benchmark() {
    local config_file=$1
    local engine_name=$2

    echo -e "\n--- Testing Engine: $engine_name ---"

    for i in $(seq 1 $RUNS); do
        # Guarantee a cold kernel state
        sync
        echo 3 > /proc/sys/vm/drop_caches

        # Let an storage controllers flush caches and cool down
        sleep $SLEEP_TIME

        # Run the benchmark and extract the IOPS line
        echo -n "Run $i: "
        $FIO_BIN $config_file | grep -A 1 "read:" | grep "IOPS" | awk -F',' '{print $1 ", " $2}'
    done
}

# Run the Vectored Control (vsync)
run_benchmark "native_vsync_benchmark.fio" "Native Vectored (vsync)"

# Run the Scalar Target (psync)
run_benchmark "ioopt_benchmark.fio" "Scalar (psync)"

echo -e "\n========================================================="
echo " Benchmark Suite Complete"
echo "========================================================="

