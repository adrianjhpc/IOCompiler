//===- IOLoweringToLLVM.cpp - Translates io.batch_write to LLVM IR --------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "IODialect.h"

using namespace mlir;

namespace {

struct BatchWriteLowering : public ConvertOpToLLVMPattern<io::BatchWriteOp> {
  using ConvertOpToLLVMPattern<io::BatchWriteOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(io::BatchWriteOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    
    auto module = op->getParentOfType<ModuleOp>();
    
    // Ensure standard POSIX `write(int fd, void* buf, size_t count)` exists in the LLVM module
    auto writeFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("write");
    if (!writeFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto writeType = LLVM::LLVMFunctionType::get(
          rewriter.getI64Type(), 
          {rewriter.getI32Type(), LLVM::LLVMPointerType::get(rewriter.getContext()), rewriter.getI64Type()}
      );
      writeFunc = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), "write", writeType);
    }

    // Cast the MLIR types down to raw LLVM types
    Value fdI32 = rewriter.create<LLVM::TruncOp>(op.getLoc(), rewriter.getI32Type(), adaptor.getFd());
    
    // Extract the raw pointer from the MLIR MemRef descriptor
    auto memrefType = op.getBuffer().getType().cast<MemRefType>();
    Value rawPtr = getStridedElementPtr(op.getLoc(), memrefType, adaptor.getBuffer(), {}, rewriter);

    // Emit the actual LLVM IR Call instruction!
    auto llvmCall = rewriter.create<LLVM::CallOp>(
        op.getLoc(), 
        writeFunc, 
        ValueRange{fdI32, rawPtr, adaptor.getTotalSize()}
    );

    rewriter.replaceOp(op, llvmCall.getResult());
    return success();
  }
};

} // end namespace
