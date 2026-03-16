import os

# Paths injected by CMake
config.shlibdir = "/home/adrianj/IOOptCompilerPass/build/llvm-src"
config.shlibext = ".so"
config.io_source_root = "/home/adrianj/IOOptCompilerPass"

# Tool paths discovered by CMake
config.clang_bin = "/data/llvm-install/bin/clang"
config.clangpp_bin = "clang++"
config.opt_bin = "opt"
config.filecheck_bin = "/usr/bin/FileCheck-20"
config.llvm_link_bin = "llvm-link"

# Now load the actual tool logic from the main lit.cfg.py
lit_config.load_config(config, "/home/adrianj/IOOptCompilerPass/llvm-tests/lit.cfg.py")
