#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Interfaces/CallInterfaces.h" // NEEDED for CallOpInterface
#include "mlir/Pass/Pass.h"

#include "IODialect.h"
#include "TargetUtils.h"

using namespace mlir;

namespace {

// Pattern to lift `write(fd, buf, count)` across ANY dialect (func or cir)
struct LiftWritePattern : public OpInterfaceRewritePattern<CallOpInterface> {
  using OpInterfaceRewritePattern<CallOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CallOpInterface callOp, PatternRewriter &rewriter) const override {
    auto calleeAttr = dyn_cast_or_null<SymbolRefAttr>(callOp.getCallableForCallee());
    if (!calleeAttr) return failure();
    
    StringRef callee = calleeAttr.getRootReference().getValue();
    if (callee != "write" && callee != "write32" && callee != "write64") return failure();

    if (callOp.getArgOperands().size() != 3) return failure();
    if (callOp->getNumResults() != 1) return failure(); 

    Value fd = callOp.getArgOperands()[0];
    Value buf = callOp.getArgOperands()[1];
    Value count = callOp.getArgOperands()[2];

    // The cleanest, most modern way to replace an operation in MLIR
    rewriter.replaceOpWithNewOp<io::WriteOp>(
        callOp,
        callOp->getResult(0).getType(), 
        fd, buf, count
    );

    return success();
  }
};

// Pattern to lift `read(fd, buf, count)` across ANY dialect (func or cir)
struct LiftReadPattern : public OpInterfaceRewritePattern<CallOpInterface> {
  using OpInterfaceRewritePattern<CallOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CallOpInterface callOp, PatternRewriter &rewriter) const override {
    auto calleeAttr = dyn_cast_or_null<SymbolRefAttr>(callOp.getCallableForCallee());
    if (!calleeAttr) return failure();
    
    StringRef callee = calleeAttr.getRootReference().getValue();
    if (callee != "read" && callee != "read32" && callee != "read64") return failure();

    if (callOp.getArgOperands().size() != 3) return failure();
    if (callOp->getNumResults() != 1) return failure(); 

    Value fd = callOp.getArgOperands()[0];
    Value buf = callOp.getArgOperands()[1];
    Value count = callOp.getArgOperands()[2];

    rewriter.replaceOpWithNewOp<io::ReadOp>(
        callOp,
        callOp->getResult(0).getType(), 
        fd, buf, count
    );

    return success();
  }
};

struct RecogniseIOPass : public PassWrapper<RecogniseIOPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RecogniseIOPass)

  llvm::StringRef getArgument() const final { return "recognise-io"; }
  llvm::StringRef getDescription() const final { return "Lifts standard C library I/O calls into the custom IO dialect."; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<io::IODialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    mlir::io::bootstrapTargetInfo(module);

    RewritePatternSet patterns(context);
    patterns.add<LiftWritePattern>(context);
    patterns.add<LiftReadPattern>(context);
    
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

// Expose it to the pass manager
namespace mlir {
namespace io {
  std::unique_ptr<mlir::Pass> createRecogniseIOPass() {
    return std::make_unique<RecogniseIOPass>();
  }
  
  void registerRecogniseIOPass() {
    PassRegistration<RecogniseIOPass>();
  }
} // namespace io
} // namespace mlir
