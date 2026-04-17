# IOOpt: Transparent I/O Coalescing via LLVM LTO

[![LLVM: 20.0](https://img.shields.io/badge/LLVM-20.0-blue.svg)](https://llvm.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests: 23/23](https://img.shields.io/badge/Tests-23%20Passed-brightgreen.svg)]()

**IOOpt** is a custom LLVM compiler pass that acts as a transparent systems-level OS adapter. It bridges the semantic gap between fragmented user-space applications (like relational databases) and the Linux Virtual File System (VFS) by automatically translating scalar POSIX I/O into hardware-optimized scatter-gather arrays (`readv`, `writev`, `preadv`, `pwritev`).


By leveraging Link-Time Optimization (LTO), Alias Analysis, and Scalar Evolution (SCEV), IOOpt safely hoists, classifies, and coalesces I/O operations across translation unit boundaries, completely eliminating the developer burden of manual vectorization.

## 🚀 Performance

Tested on a range of example mini-apps (in the benchmarks directory) we see 2-3x performance improvement using this functionality.

---

## 🧠 Architecture & Features

IOOpt is built on a decoupled **Classifier and Router** architecture, enabling surgical transformation of intermediate representation (IR) based on dynamic memory layouts.

* **Explicit-Offset Contiguity Tracking:** Fully supports explicit offset I/O (`pread` / `pwrite`) heavily used by database storage engines (like InnoDB). IOOpt utilizes LLVM's `ScalarEvolution` (SCEV) to mathematically prove offset contiguity algebraically, safely synthesizing `preadv` and `pwritev` arrays even when offsets are dynamically calculated at runtime.
* **Strict Memory Hazard Protection:** Uses LLVM's `AAManager` (Alias Analysis) to detect buffer mutations between sequential I/O calls. If a hazard, file descriptor reassignment, or an opaque barrier is detected, the pass safely flushes the batch to guarantee strict ACID semantics.
* **Loop-Exit (Lazy) Flushing:** Utilizes `LoopInfo` and `SCEVExpander` to mathematically calculate loop trip counts and stride lengths. It hoists I/O instructions out of rigid loops into the Loop-Closed SSA (LCSSA) exit block, turning $O(N)$ system calls into $O(1)$.
* **High-Water Mark Protection:** Actively monitors batch byte-weights. To prevent overwhelming the Linux VFS allocator, it forces a pipeline flush the exact moment the batch crosses the 64KB OS Page Cache boundary.
* **The I/O Pattern Classifier:** Automatically routes memory access patterns to the optimal silicon/OS primitive (Contiguous, Shadow Buffered, Vectored, or Strided SIMD Gather).

---

## 🛡️ Bypassing IOOpt (Manual Opt-Out)

In certain systems programming scenarios—such as writing to an `eventfd`, a network socket, or an IPC pipe to signal another process—developers rely on precise, immediate OS wake-ups. If IOOpt batches a signaling write, the receiving process may deadlock.

Because IOOpt respects strict memory semantics and standard LLVM attributes, you can easily force the pass to ignore specific I/O calls using one of the following methods. 



### Method 1: The Opaque Wrapper (Targeted Opt-Out)
Hide the system call inside a function that the LLVM pass is forbidden from inlining or analyzing. 

```c
#include <unistd.h>

// __attribute__((noinline)) prevents Clang from unpacking this
// __attribute__((optnone)) hides it from the LTO pass entirely
__attribute__((noinline, optnone))
static ssize_t write_signal(int fd, const void *buf, size_t count) {
    return write(fd, buf, count); // IOOpt will ignore this!
}
```

### Method 2: The Inline Assembly Barrier (Block-Level Opt-Out)
If you want to manually chop up a massive sequential block of I/O without changing your function calls, drop a zero-cost compiler memory barrier into your code. IOOpt treats this as an opaque memory hazard and will instantly flush any pending I/O batches.

```c
#define IO_OPT_BARRIER() __asm__ volatile("" ::: "memory")

write(fd, data1, 10); // IOOpt batches this...
write(fd, data2, 10); // ...with this.

IO_OPT_BARRIER();     // <--- IOOpt hits this and flushes the batch!

write(ipc_fd, sig, 1); // This signaling write executes immediately and alone.

IO_OPT_BARRIER();     // <--- Another barrier protects it from the bottom.

write(fd, data3, 10); // IOOpt starts a new batch here.
```

---

## 🛠️ Building and Installation

### Prerequisites
* LLVM / Clang 20.0+
* CMake 3.10+
* C++17 Compiler

If you want to run the tests you will also require:
* lit

### Compilation
```bash
git clone https://github.com/adrianjhpc/IOOptCompilerPass.git
cd IOOpt
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Running the Test Suite
IOOpt includes a `lit` suite verifying the maths, hazards, offset contiguity, and LCSSA dominance rules.
```bash
make test
```

---

## 💻 Usage

To compile your application with IOOpt, you must inject it into the Clang LTO linker pipeline.

**For a standard Makefile/C project:**
```bash
export CFLAGS="-O3 -flto"
export LDFLAGS="-flto -Wl,--load-pass-plugin=/path/to/libIOOpt.so"
make
```

**For CMake projects (e.g., MySQL):**
```bash
cmake . \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_FLAGS="-O3 -flto" \
  -DCMAKE_CXX_FLAGS="-O3 -flto" \
  -DCMAKE_EXE_LINKER_FLAGS="-flto -Wl,--load-pass-plugin=/path/to/libIOOpt.so"
```

### Tunable CLI Parameters
IOOpt behavior can be tuned by passing LLVM standard arguments during the linking phase:
* `-io-batch-threshold=<int>`: Minimum scattered calls required to trigger `writev` (Default: 4).
* `-io-shadow-buffer-max=<int>`: Maximum bytes to safely pack on the stack (Default: 4096).
* `-io-high-water-mark=<int>`: Maximum cumulative bytes before forcing a VFS flush (Default: 65536).

---

## Authors
Adrian Jackson 

---

## 📜 License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
