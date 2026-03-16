#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "IODialect.h"

// Forward declare our custom pass registration functions
namespace mlir {
namespace io {
  void registerRecognizeIOPass();
  void registerIOPasses();
  void registerConvertIOToLLVMPass();
}
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  
  // Register EXACTLY the dialects our tool actually uses!
  registry.insert<mlir::func::FuncDialect,
                  mlir::scf::SCFDialect,
                  mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::io::IODialect>();

  // Register our custom passes
  mlir::io::registerRecognizeIOPass();
  mlir::io::registerIOPasses();
  mlir::io::registerConvertIOToLLVMPass();

  // Start the MLIR command-line tool
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "IO Optimiser tool\n", registry));
}
