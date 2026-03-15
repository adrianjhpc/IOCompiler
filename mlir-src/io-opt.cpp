#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Include the specific core MLIR dialects we interact with
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROps.h"

#include "IODialect.h"
#include "IOPasses.h"

struct WriteToBatchWritePattern : public OpConversionPattern<cir::CallOp> {
  using OpConversionPattern<cir::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cir::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    // Check if the function being called is "write"
    if (op.getCallee() != "write")
      return rewriter.notifyMatchFailure(op, "not a write call");

    // Extract arguments: fd, buffer, and size
    // Standard write(int fd, const void *buf, size_t count)
    auto operands = adaptor.getOperands();
    if (operands.size() < 3)
      return failure();

    Value fd = operands[0];
    Value buf = operands[1];
    Value size = operands[2];

    // Replace the ClangIR call with our custom IO dialect op
    // Note: You might need to add type casting if ClangIR types differ
    rewriter.replaceOpWithNewOp<mlir:io::BatchWriteOp>(
        op, op.getResultTypes(), fd, buf, size);

    return success();
  }
};

struct CirToIoPass : public PassWrapper<CirToIoPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CirToIoPass)

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    // Define the Conversion Target
    // We tell the system that 'cir.call' is now illegal if it calls "write",
    // and our 'io' dialect operations are now legal.
    ConversionTarget target(*context);
    target.addLegalDialect<mlir::io::IoDialect>();
    target.addLegalDialect<func::FuncDialect>();
    
    // We mark cir.call as dynamically illegal
    target.addDynamicallyLegalOp<cir::CallOp>([](cir::CallOp op) {
        return op.getCallee() != "write";
    });

    // Collect the Patterns
    RewritePatternSet patterns(context);
    patterns.add<WriteToBatchWritePattern>(context);

    // Apply the Conversion
    // This looks through the IR and replaces the illegal ops using your pattern
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};


int main(int argc, char **argv) {
    mlir::DialectRegistry registry;

    // Register Dialects
    registry.insert<mlir::func::FuncDialect,
                    mlir::scf::SCFDialect,
                    mlir::arith::ArithDialect,
                    mlir::memref::MemRefDialect,
                    mlir::LLVM::LLVMDialect,
                    mlir::cir::CIRDialect,   // Standard ClangIR naming
                    mlir::io::IODialect>();   // Your custom dialect

    // Register Pass manually if it's defined in this file
    mlir::PassRegistration<CirToIoPass>();

    // Register your generated passes (from tablegen)
    mlir::io::registerIOPasses();
    mlir::io::registerConvertIOToLLVMPass();

    // Run the main entry point
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "IO Optimiser Driver\n", registry));
}
