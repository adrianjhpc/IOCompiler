cmake_minimum_required(VERSION 3.16)
project(IOOptimiationPass)

# Find the LLVM package installed on your system
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_subdirectory(src)
