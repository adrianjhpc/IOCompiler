#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

#include "IODialect.h"

using namespace mlir;

namespace {

// Helper to extract a constant integer from an MLIR Value
static std::optional<int64_t> getConstantIntValue(Value val) {
    if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>())
        return constOp.value();
    if (auto constIntOp = val.getDefiningOp<arith::ConstantIntOp>())
        return constIntOp.value();
    return std::nullopt;
}

// Proves that: (Offset_Multiplier * Loop_Step) == Write_Size
static bool verifySCEVOffset(Value dynamicOffset, scf::ForOp loop, Value writeSize) {
    Value iv = loop.getInductionVar();
   
    if (auto castOp = dynamicOffset.getDefiningOp<arith::IndexCastOp>()) {
        dynamicOffset = castOp.getIn();
    }
 
    // Attempt to resolve the writeSize and step to compile-time constants
    auto optWriteSize = getConstantIntValue(writeSize);
    auto optStep = getConstantIntValue(loop.getStep());
    
    // If we can't statically prove the step and size, it is too dangerous to 
    // batch contiguously. Conservatively abort.
    if (!optWriteSize || !optStep) return false;
    
    int64_t targetAdvance = *optWriteSize;
    int64_t step = *optStep;

    // Case 1: The offset is the induction variable (Multiplier = 1)
    // Example: ptr = base + iv. 
    // This is only contiguous if the loop step exactly matches the write size.
    if (dynamicOffset == iv) {
        return step == targetAdvance;
    }

    // Case 2: The offset is explicitly calculated: offset = iv * multiplier
    if (auto mulOp = dynamicOffset.getDefiningOp<arith::MulIOp>()) {
        Value lhs = mulOp.getLhs();
        Value rhs = mulOp.getRhs();

        if (lhs == iv || rhs == iv) {
            Value multiplierVal = (lhs == iv) ? rhs : lhs;
            auto optMultiplier = getConstantIntValue(multiplierVal);
            
            if (optMultiplier) {
                return (*optMultiplier * step) == targetAdvance;
            }
        }
    }

    return false;
}

static bool isContiguousMemoryAccess(Value buffer, scf::ForOp loop, Value writeSize, Value &outBasePointer) {
    // If the buffer doesn't change during the loop, it's writing to the exact 
    // same memory address every iteration. This is not contiguous batchable.
    if (loop.isDefinedOutsideOfLoop(buffer)) return false;

    Operation *defOp = buffer.getDefiningOp();
    if (!defOp) return false;

    // ------------------------------------------------------------------
    // PATTERN 1: mlir::memref::SubViewOp
    // Example: %sub = memref.subview %base[%iv] [%size] [%stride]
    // ------------------------------------------------------------------
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
        outBasePointer = subviewOp.getSource();
        
        // Base pointer must live outside the loop
        if (!loop.isDefinedOutsideOfLoop(outBasePointer)) return false;

        auto mixedOffsets = subviewOp.getMixedOffsets();
        if (mixedOffsets.empty()) return false; 

        // Get the first dimension's offset
        OpFoldResult firstOffset = mixedOffsets.front();
         
        // Modern LLVM uses global casting templates instead of member functions.
        // dyn_cast safely checks if it's a Value and extracts it in one step.
        if (auto dynVal = dyn_cast<Value>(firstOffset)) {
            return verifySCEVOffset(dynVal, loop, writeSize);
        }
        
        // If it's not a Value, it's a static constant Attribute. 
        // A static offset means it writes to the exact same memory address 
        // on every single loop iteration. This is not contiguous.
        return false;
    }

    // ------------------------------------------------------------------
    // PATTERN 2: mlir::LLVM::GEPOp (GetElementPtr)
    // Example: %ptr = llvm.getelementptr %base[%iv]
    // ------------------------------------------------------------------
    if (auto gepOp = dyn_cast<LLVM::GEPOp>(defOp)) {
        outBasePointer = gepOp.getBase();
        
        if (!loop.isDefinedOutsideOfLoop(outBasePointer)) return false;

        // GEPs separate static and dynamic indices.
        auto dynamicIndices = gepOp.getDynamicIndices();
        if (dynamicIndices.empty()) return false;

        // Check the primary dynamic index advancing the pointer
        return verifySCEVOffset(dynamicIndices.back(), loop, writeSize);
    }

    // ------------------------------------------------------------------
    // PATTERN 3: Raw arith.addi Pointer Math
    // Example: %ptr = arith.addi %base, %offset
    // ------------------------------------------------------------------
    if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
        Value lhs = addOp.getLhs();
        Value rhs = addOp.getRhs();

        Value dynamicOffset;
        if (loop.isDefinedOutsideOfLoop(lhs)) {
            outBasePointer = lhs;
            dynamicOffset = rhs;
        } else if (loop.isDefinedOutsideOfLoop(rhs)) {
            outBasePointer = rhs;
            dynamicOffset = lhs;
        } else {
            return false; // Neither side is a stable base pointer
        }

        return verifySCEVOffset(dynamicOffset, loop, writeSize);
    }

    // Unrecognized memory access pattern
    return false;
}

struct CirLoopBatchingPattern : public OpInterfaceRewritePattern<cir::LoopOpInterface> {
  using OpInterfaceRewritePattern<cir::LoopOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(cir::LoopOpInterface loopInterface, PatternRewriter &rewriter) const override {
    // Safely extract the underlying ClangIR operation
    Operation *op = loopInterface.getOperation();
    
    Region *bodyRegion = nullptr;
    Region *condRegion = nullptr;

    // Cast to the concrete loop types to bypass the barebones interface
    if (auto forOp = dyn_cast<cir::ForOp>(op)) {
      bodyRegion = &forOp.getBody();
      condRegion = &forOp.getCond();
    } else if (auto whileOp = dyn_cast<cir::WhileOp>(op)) {
      bodyRegion = &whileOp.getBody();
      condRegion = &whileOp.getCond();
    } else {
      return failure(); // Unrecognized CIR loop type
    }

    if (!bodyRegion || bodyRegion->empty()) return failure();

    cir::CallOp writeCall = nullptr;
    // Walk the body to find a 'write' call
    bodyRegion->walk([&](cir::CallOp call) {
      auto callee = call.getCallee();
      if (callee && callee->starts_with("write")) {
        writeCall = call;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!writeCall) return failure();

    // Extract Trip Count from the Condition Region
    int64_t tripCount = -1;
    if (condRegion && !condRegion->empty()) {
      condRegion->walk([&](cir::CmpOp cmp) {
        if (auto constOp = cmp.getRhs().getDefiningOp<cir::ConstantOp>()) {
          if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
            tripCount = intAttr.getInt();
        }
      });
    }

    if (tripCount <= 1) return failure();

    Location loc = op->getLoc();
    auto ctx = rewriter.getContext();

    // Use the Operation* (op) for the insertion point
    rewriter.setInsertionPoint(op); 

    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getI64Type();
    auto ptrTy = cir::PointerType::get(ctx, rewriter.getI8Type());
    auto alignAttr = rewriter.getI64IntegerAttr(16); // Strict ClangIR alignment

    // Allocate space for the pointers and sizes
    auto ptrArrayTy = cir::ArrayType::get(ctx, ptrTy, tripCount);
    auto sizeArrayTy = cir::ArrayType::get(ctx, i64Ty, tripCount);

    Value iovPtrs = rewriter.create<cir::AllocaOp>(
        loc, cir::PointerType::get(ctx, ptrArrayTy), ptrArrayTy, "iov_ptrs", alignAttr);
    Value iovSizes = rewriter.create<cir::AllocaOp>(
        loc, cir::PointerType::get(ctx, sizeArrayTy), sizeArrayTy, "iov_sizes", alignAttr);

    // Track index for the array inside the loop
    Value idxAlloca = rewriter.create<cir::AllocaOp>(
        loc, cir::PointerType::get(ctx, i32Ty), i32Ty, "iov_idx", alignAttr);
    Value zero = rewriter.create<cir::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(0));
    rewriter.create<cir::StoreOp>(loc, zero, idxAlloca);

    rewriter.setInsertionPoint(writeCall);

    Value currentIdx = rewriter.create<cir::LoadOp>(loc, i32Ty, idxAlloca);

    Value fdArg = writeCall.getOperand(0);
    Value bufArg = writeCall.getOperand(1);
    Value lenArg = writeCall.getOperand(2);

    // Store buffer pointer and length into our pre-allocated arrays
    Value targetPtrSlot = rewriter.create<cir::GetArrayElementOp>(
        loc, cir::PointerType::get(ctx, ptrTy), iovPtrs, currentIdx);
    Value targetSizeSlot = rewriter.create<cir::GetArrayElementOp>(
        loc, cir::PointerType::get(ctx, i64Ty), iovSizes, currentIdx);

    rewriter.create<cir::StoreOp>(loc, bufArg, targetPtrSlot);
    rewriter.create<cir::StoreOp>(loc, lenArg, targetSizeSlot);
    
    // Increment index: idx++
    Value one = rewriter.create<cir::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value nextIdx = rewriter.create<cir::BinOp>(loc, i32Ty, cir::BinOpKind::Add, currentIdx, one);
    rewriter.create<cir::StoreOp>(loc, nextIdx, idxAlloca);

    // Erase the original write call safely
    rewriter.eraseOp(writeCall);

    // Use the Operation* (op) for the insertion point
    rewriter.setInsertionPointAfter(op);

    // Inject the high-level BatchWriteVOp (with the i64 return type!)
    Value tripCountVal = rewriter.create<arith::ConstantIndexOp>(loc, tripCount);
    rewriter.create<io::BatchWriteVOp>(loc, rewriter.getI64Type(), fdArg, iovPtrs, iovSizes, tripCountVal);

    return success();
  }
};

struct HoistWriteLoopPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(cir::LoopOpInterface loopInterface, PatternRewriter &rewriter) const override {
    Operation *op = loopInterface.getOperation();
    
    Region *bodyRegion = nullptr;
    Region *condRegion = nullptr;

    // Safely cast to the concrete loop types to access their specific regions
    if (auto forOp = dyn_cast<cir::ForOp>(op)) {
      bodyRegion = &forOp.getBody();
      condRegion = &forOp.getCond();
    } else if (auto whileOp = dyn_cast<cir::WhileOp>(op)) {
      bodyRegion = &whileOp.getBody();
      condRegion = &whileOp.getCond();
    } else {
      return failure(); // Unrecognized CIR loop type
    }

    if (!bodyRegion || bodyRegion->empty()) return failure();

    cir::CallOp writeCall = nullptr;
    // Walk the body to find a 'write' call
    bodyRegion->walk([&](cir::CallOp call) {
      auto callee = call.getCallee();
      if (callee && callee->starts_with("write")) {
        writeCall = call;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!writeCall) return failure();

    // Extract Trip Count from the Condition Region
    int64_t tripCount = -1;
    condRegion->walk([&](cir::CmpOp cmp) {
      if (auto constOp = cmp.getRhs().getDefiningOp<cir::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
          tripCount = intAttr.getInt();
      }
    });

    if (tripCount <= 1) return failure();

    Location loc = op->getLoc();
    auto ctx = rewriter.getContext();

    // Make sure we pass the Operation* to setInsertionPoint
    rewriter.setInsertionPoint(op);

    // Contiguous vs Vector Routing
    Value basePointer;
    if (isContiguousMemoryAccess(writeOp.getBuffer(), loop, writeOp.getSize(), basePointer)) {
        // Contigious writes
        Value totalSize = rewriter.create<arith::MulIOp>(loc, tripCountI64, writeOp.getSize());
        rewriter.create<io::BatchWriteOp>(loc, rewriter.getI64Type(), writeOp.getFd(), basePointer, totalSize);
    } else {
        // Fallback to scattered writes (writev) 
        auto ptrArrayType = MemRefType::get({ShapedType::kDynamic}, writeOp.getBuffer().getType());
        auto sizeArrayType = MemRefType::get({ShapedType::kDynamic}, rewriter.getI64Type());
        
        Value ptrsMemref = rewriter.create<memref::AllocaOp>(loc, ptrArrayType, tripCount);
        Value sizesMemref = rewriter.create<memref::AllocaOp>(loc, sizeArrayType, tripCount);

        // Build a new loop just to calculate the addresses
        auto calcLoop = rewriter.create<scf::ForOp>(loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep());
        rewriter.setInsertionPointToStart(calcLoop.getBody());

        Value currentIV = calcLoop.getInductionVar();
        Value ivOffset = rewriter.create<arith::SubIOp>(loc, currentIV, loop.getLowerBound());
        Value arrayIdx = rewriter.create<arith::DivSIOp>(loc, ivOffset, loop.getStep());
        Value arrayIdxCast = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), arrayIdx);

        // Clone the address calculations
        IRMapping mapping;
        mapping.map(loop.getInductionVar(), currentIV);
        for (Operation &op : *body) {
            if (isa<io::WriteOp, scf::YieldOp>(op)) continue;
            rewriter.clone(op, mapping);
        }

        Value mappedBuffer = mapping.lookupOrDefault(writeOp.getBuffer());
        Value mappedSize = mapping.lookupOrDefault(writeOp.getSize());
        
        rewriter.create<memref::StoreOp>(loc, mappedBuffer, ptrsMemref, arrayIdxCast);
        rewriter.create<memref::StoreOp>(loc, mappedSize, sizesMemref, arrayIdxCast);

        // Emit the batched scatter operation after the calculation loop
        rewriter.setInsertionPointAfter(calcLoop);
        rewriter.create<io::BatchWriteVOp>(loc, rewriter.getI64Type(), writeOp.getFd(), ptrsMemref, sizesMemref, tripCount);
    }

    // Completely erase the original write loop
    rewriter.eraseOp(loop);

    return success();
  }
};

struct HoistReadLoopPattern : public OpRewritePattern<scf::ForOp> {
  AliasAnalysis &aliasAnalysis;

  // Constructor takes the AliasAnalysis engine from the Pass Manager
  HoistReadLoopPattern(MLIRContext *context, AliasAnalysis &aa)
      : OpRewritePattern<scf::ForOp>(context), aliasAnalysis(aa) {}

  LogicalResult matchAndRewrite(scf::ForOp loop, PatternRewriter &rewriter) const override {
    Block *body = loop.getBody();
    if (!body || body->empty()) return failure();

    io::ReadOp readOp = nullptr;

    // Find the ReadOp
    for (Operation &op : *body) {
      if (isa<scf::YieldOp>(op)) continue;
      if (auto ioRead = dyn_cast<io::ReadOp>(op)) {
        if (readOp) return failure(); 
        readOp = ioRead;
      }
    }
    if (!readOp || !loop.isDefinedOutsideOfLoop(readOp.getFd())) return failure();

    // Hazard checking
    for (Operation &op : *body) {
      if (isa<scf::YieldOp, io::ReadOp>(op)) continue;

      // Does this operation write to memory?
      auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
      if (memEffect && memEffect.hasEffect<MemoryEffects::Write>()) {
        SmallVector<MemoryEffects::EffectInstance, 4> effects;
        memEffect.getEffects<MemoryEffects::Write>(effects);
        
        for (auto &effect : effects) {
          Value writtenVal = effect.getValue();
          if (writtenVal) {
            // Ask MLIR's Alias Analysis if the written memory overlaps with our read buffer
            AliasResult aliasResult = aliasAnalysis.alias(readOp.getBuffer(), writtenVal);
            if (!aliasResult.isNo()) {
              // It's a MayAlias or MustAlias. We cannot safely hoist
              return failure(); 
            }
          } else {
            // Opaque memory write (e.g., an external function call). Unsafe to hoist.
            return failure();
          }
        }
      } else if (!isMemoryEffectFree(&op)) {
         // Catch-all for operations that lack specific memory interfaces but aren't pure
         return failure();
      }
    }

    Location loc = loop.getLoc();
    Value diff = rewriter.create<arith::SubIOp>(loc, loop.getUpperBound(), loop.getLowerBound());
    Value tripCount = rewriter.create<arith::DivSIOp>(loc, diff, loop.getStep());
    Value tripCountI64 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), tripCount);

    rewriter.setInsertionPoint(loop);

    // Contiguous vs Vector Routing
    Value basePointer;
    if (isContiguousMemoryAccess(readOp.getBuffer(), loop, readOp.getSize(), basePointer)) {
        // Fast contigious read
        Value totalSize = rewriter.create<arith::MulIOp>(loc, tripCountI64, readOp.getSize());
        rewriter.create<io::BatchReadOp>(loc, rewriter.getI64Type(), readOp.getFd(), basePointer, totalSize);
    } else {
        // Gather (readv)
        auto ptrArrayType = MemRefType::get({ShapedType::kDynamic}, readOp.getBuffer().getType());
        auto sizeArrayType = MemRefType::get({ShapedType::kDynamic}, rewriter.getI64Type());

        Value ptrsMemref = rewriter.create<memref::AllocaOp>(loc, ptrArrayType, tripCount);
        Value sizesMemref = rewriter.create<memref::AllocaOp>(loc, sizeArrayType, tripCount);


        // Build an address-calculation loop before the main loop
        auto calcLoop = rewriter.create<scf::ForOp>(loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep());
        rewriter.setInsertionPointToStart(calcLoop.getBody());

        Value currentIV = calcLoop.getInductionVar();
        Value ivOffset = rewriter.create<arith::SubIOp>(loc, currentIV, loop.getLowerBound());
        Value arrayIdx = rewriter.create<arith::DivSIOp>(loc, ivOffset, loop.getStep());
        Value arrayIdxCast = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), arrayIdx);

        // Clone only the pure address/math calculations
        IRMapping mapping;
        mapping.map(loop.getInductionVar(), currentIV);
        for (Operation &op : *body) {
            if (isa<io::ReadOp, scf::YieldOp>(op)) continue;
            if (isMemoryEffectFree(&op)) {
                rewriter.clone(op, mapping);
            }
        }

        Value mappedBuffer = mapping.lookupOrDefault(readOp.getBuffer());
        Value mappedSize = mapping.lookupOrDefault(readOp.getSize());
        
        rewriter.create<memref::StoreOp>(loc, mappedBuffer, ptrsMemref, arrayIdxCast);
        rewriter.create<memref::StoreOp>(loc, mappedSize, sizesMemref, arrayIdxCast);

        // Emit the batched gather operation immediately after our calculation loop
        rewriter.setInsertionPoint(loop); 
        rewriter.create<io::BatchReadVOp>(loc, rewriter.getI64Type(), readOp.getFd(), ptrsMemref, sizesMemref, tripCount);
    }

    // Clean up the original loop
    // Replace the slow I/O calls inside the loop with just the expected byte count,
    // leaving the processing logic completely intact to run against the now-populated memory
    rewriter.setInsertionPoint(readOp);
    rewriter.replaceOp(readOp, readOp.getSize());

    return success();
  }
};

/// The MLIR Pass Wrapper
struct IOLoopBatchingPass : public PassWrapper<IOLoopBatchingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IOLoopBatchingPass)

  llvm::StringRef getArgument() const final { return "io-loop-batching"; }
  llvm::StringRef getDescription() const final { return "Hoists and batches I/O operations from scf.for loops."; }

  // The Pass Manager runs this sequentially before spinning up the threads
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<io::IODialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<cir::CIRDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();

    RewritePatternSet patterns(context);
    patterns.add<HoistWriteLoopPattern>(context);
    patterns.add<HoistReadLoopPattern>(context, aliasAnalysis);  
    patterns.add<CirLoopBatchingPattern>(context);
    
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

namespace mlir {
namespace io {

// Expose the constructor
std::unique_ptr<mlir::Pass> createIOLoopBatchingPass() {
  return std::make_unique<IOLoopBatchingPass>(); 
}

// Register the pass so `io-opt` knows it exists
void registerIOPasses() {
  mlir::PassRegistration<IOLoopBatchingPass>();
}

} // namespace io
} // namespace mlir
