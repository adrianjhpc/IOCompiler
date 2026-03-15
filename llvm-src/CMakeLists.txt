find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM) # This gives us access to add_llvm_pass_plugin

# Add LLVM includes and definitions
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Create the Pass Plugin using LLVM's official macro
add_llvm_pass_plugin(IOOpt 
    IOOptimisationPass.cpp
)
