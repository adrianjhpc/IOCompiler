include_directories(${MLIR_INCLUDE_DIRS})

# Run TableGen to generate Dialect C++ code from IODialect.td
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/IODialect.td)

mlir_tablegen(IODialect.h.inc -gen-op-decls)
mlir_tablegen(IODialect.cpp.inc -gen-op-defs)
mlir_tablegen(IODialectDialect.h.inc -gen-dialect-decls)
mlir_tablegen(IODialectDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(IODialectIncGen)

# Build the MLIR Passes library
add_library(MLIRIOOptimiser
  IODialect.cpp 
  IOLoopBatching.cpp
  IOLoweringToLLVM.cpp)

# Ensure TableGen runs BEFORE trying to compile the C++ files
add_dependencies(MLIRIOOptimiser IODialectIncGen)

# Link against all required MLIR Dialects and Core Libraries
target_link_libraries(MLIRIOOptimiser PRIVATE
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRRewrite
  MLIRSCFDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
)
