#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Analysis/AliasAnalysis.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"

#include "IODialect.h"

using namespace mlir;

namespace {

static StringRef getCalleeName(CallOpInterface call) {
    if (auto sym = dyn_cast_or_null<SymbolRefAttr>(call.getCallableForCallee()))
        return sym.getRootReference().getValue();
    if (auto attr = call->getAttrOfType<FlatSymbolRefAttr>("callee"))
        return attr.getValue();
    return "";
}

struct ZeroCopyPromotionPass : public PassWrapper<ZeroCopyPromotionPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZeroCopyPromotionPass)
    StringRef getArgument() const final { return "io-zero-copy-promotion"; }
    StringRef getDescription() const final { return "Promotes read/write pairs to zero-copy io.sendfile ops"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<mlir::io::IODialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        IRRewriter rewriter(&getContext());

        // Find all read calls first (to avoid modifying the tree while walking it)
        SmallVector<CallOpInterface> readCalls;
        module.walk([&](CallOpInterface readCall) {
            auto calleeAttr = dyn_cast_or_null<SymbolRefAttr>(readCall.getCallableForCallee());
            if (!calleeAttr) return;
            
            StringRef callee = getCalleeName(readCall); 
            if ((callee == "read" || callee == "read64" || callee == "read32") && 
                 readCall->getNumOperands() == 3) {
                readCalls.push_back(readCall);
            }
        });

        // Process them sequentially
        for (CallOpInterface readCall : readCalls) {
            llvm::errs() << "[IOOpt-Telemetry] Found 'read' call. Scanning block...\n";
            
            Value fdIn = readCall->getOperand(0);
            Value readSize = readCall->getOperand(2);

            auto getRootAllocation = [](Value v) {
                while (Operation *def = v.getDefiningOp()) {
                    StringRef name = def->getName().getStringRef();
                    if (name == "cir.cast" || name == "cir.load" || 
                        name == "cir.ptr_stride" || name == "cir.get_element") {
                        v = def->getOperand(0);
                    } else break;
                }
                return v;
            };        

            Value readBufferRoot = getRootAllocation(readCall->getOperand(1));
            CallOpInterface writeCall = nullptr; 
            bool abortSearch = false;
            
            // Scan forward in the block for the matching write
            for (Operation *nextOp = readCall->getNextNode(); nextOp != nullptr; nextOp = nextOp->getNextNode()) {
                if (auto maybeWrite = dyn_cast<CallOpInterface>(nextOp)) {
                    auto nextCalleeAttr = dyn_cast_or_null<SymbolRefAttr>(maybeWrite.getCallableForCallee());
                    if (nextCalleeAttr) {
                        StringRef nextCallee = nextCalleeAttr.getRootReference().getValue();
                        if (nextCallee == "write" || nextCallee == "write64" || nextCallee == "write32") {
                            if (maybeWrite->getNumOperands() == 3) {
                                Value writeBufferRoot = getRootAllocation(maybeWrite->getOperand(1));
                                if (readBufferRoot == writeBufferRoot) {
                                    writeCall = maybeWrite;
                                    break; 
                                } else {
                                    llvm::errs() << "[IOOpt-Telemetry] Abort: Found 'write', but buffer roots mismatch!\n";
                                }
                            }
                        }
                    }
                }

                StringRef opName = nextOp->getName().getStringRef();
                if (opName == "cir.store") {
                    Value storePtr = nextOp->getOperand(1);
                    if (getRootAllocation(storePtr) == readBufferRoot) {
                        llvm::errs() << "[IOOpt-Telemetry] Abort: cir.store mutated the buffer root!\n";
                        abortSearch = true; break; 
                    }
                    continue; 
                }
                
                if (isa<CallOpInterface>(nextOp)) {
                    llvm::errs() << "[IOOpt-Telemetry] Abort: Intervening CallOp found: " << opName << "\n";
                    abortSearch = true; break; 
                }
                
                if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(nextOp)) {
                    if (memInterface.hasEffect<MemoryEffects::Write>()) {
                        llvm::errs() << "[IOOpt-Telemetry] Abort: Intervening memory write found: " << opName << "\n";
                        abortSearch = true; break;
                    }
                }
            }

            if (abortSearch || !writeCall) {
                if (!writeCall && !abortSearch)
                    llvm::errs() << "[IOOpt-Telemetry] Abort: Reached end of block. No matching 'write' found.\n";
                continue;
            }

            Value fdOut = writeCall->getOperand(0);
            Value writeSizeOut = writeCall->getOperand(2);

            if (getRootAllocation(readSize) != getRootAllocation(writeSizeOut)) {
                llvm::errs() << "[IOOpt-Telemetry] Abort: Size variables do not match.\n";
                continue;
            }

            // ==========================================
            // The Rewrite
            // ==========================================
            rewriter.setInsertionPoint(writeCall);
            Value nullOffset = readCall->getOperand(1); 
            Location loc = writeCall->getLoc();

            auto i32Ty = rewriter.getI32Type();
            auto i64Ty = rewriter.getI64Type();

            Value stdFdOut = fdOut;
            if (stdFdOut.getType() != i32Ty) 
                stdFdOut = mlir::io::IOCastOp::create(rewriter, loc, i32Ty, stdFdOut);
                
            Value stdFdIn = fdIn;
            if (stdFdIn.getType() != i32Ty) 
                stdFdIn = mlir::io::IOCastOp::create(rewriter, loc, i32Ty, stdFdIn);
                
            Value stdSizeOut = writeSizeOut;
            if (stdSizeOut.getType() != i64Ty) 
                stdSizeOut = mlir::io::IOCastOp::create(rewriter, loc, i64Ty, stdSizeOut);

            auto sendfileOp = mlir::io::SendfileOp::create(
                rewriter, loc, i64Ty, stdFdOut, stdFdIn, nullOffset, stdSizeOut
            );

            Value replacement = sendfileOp.getBytesWritten();
            Type expectedReturnType = readCall->getResult(0).getType();

            if (replacement.getType() != expectedReturnType) {
                replacement = mlir::io::IOCastOp::create(rewriter, loc, expectedReturnType, replacement);
            }

            rewriter.replaceOp(writeCall, replacement);
            rewriter.replaceOp(readCall, replacement);

            llvm::errs() << "[IOOpt-Telemetry] SUCCESS: Replaced read/write with sendfile!\n";
        }
    }
};

// ============================================================================
// 2. Straight-Line Serialization -> Basic Block Vectored I/O (Deterministic)
// ============================================================================
struct BlockVectoredIOPass : public PassWrapper<BlockVectoredIOPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BlockVectoredIOPass)

    StringRef getArgument() const final { return "io-block-vectored"; }
    StringRef getDescription() const final { return "Batches straight-line sequential writes into writev"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<func::FuncDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();

        // --------------------------------------------------------------------
        // Step 1: PRE-PASS
        // --------------------------------------------------------------------
        CallOpInterface firstWrite = nullptr;
        module.walk([&](CallOpInterface call) {
            auto calleeAttr = dyn_cast_or_null<SymbolRefAttr>(call.getCallableForCallee());
            if (calleeAttr) {
                StringRef callee = getCalleeName(call);
                if ((callee == "write" || callee == "write64" || callee == "write32") && call->getNumOperands() == 3) {
                    firstWrite = call;
                    return WalkResult::interrupt();
                }
            }
            return WalkResult::advance();
        });

        if (!firstWrite) return;

        OpBuilder builder(module.getBodyRegion());
        Type fdType = firstWrite->getOperand(0).getType();
        Type ptrType = firstWrite->getOperand(1).getType(); 
        Type sizeType = firstWrite->getOperand(2).getType();
        Type retType = firstWrite->getResult(0).getType();
        
        for (int i = 2; i <= 4; ++i) {
            std::string funcName = "ioopt_writev_" + std::to_string(i);
            if (!module.lookupSymbol(funcName)) {
                SmallVector<Type> argTypes;
                argTypes.push_back(fdType);
                for(int j = 0; j < i; j++) {
                    argTypes.push_back(ptrType);
                    argTypes.push_back(sizeType);
                }
                auto funcType = builder.getFunctionType(argTypes, {retType});
                func::FuncOp::create(builder, module.getLoc(), funcName, funcType).setPrivate();
            }
        }

        // --------------------------------------------------------------------
        // Step 2: DETERMINISTIC TOP-DOWN WALK
        // --------------------------------------------------------------------
        SmallVector<SmallVector<Operation*, 4>> allBatches;

        module.walk([&](Block *block) {
            SmallVector<Operation*, 4> currentBatch;
            Value currentFdRoot = nullptr;

            auto getRootAllocation = [](Value v) {
                while (Operation *def = v.getDefiningOp()) {
                    StringRef name = def->getName().getStringRef();
                    if (name == "cir.load" || name == "cir.cast" || 
                        name == "cir.get_element" || name == "cir.ptr_stride") {
                        v = def->getOperand(0);
                    } else {
                        break;
                    }
                }
                return v;
            };

            for (Operation &opRef : *block) {
                Operation *op = &opRef;
                
                if (auto call = dyn_cast<CallOpInterface>(op)) {
                    auto calleeAttr = dyn_cast_or_null<SymbolRefAttr>(call.getCallableForCallee());
                    if (calleeAttr) {
                       StringRef callee = getCalleeName(call); 
                       if ((callee == "write" || callee == "write64" || callee == "write32") && call->getNumOperands() == 3) {
                            Value fdRoot = getRootAllocation(call->getOperand(0));
                            
                            if (currentBatch.empty()) {
                                currentBatch.push_back(op);
                                currentFdRoot = fdRoot;
                            } else if (fdRoot == currentFdRoot) {
                                currentBatch.push_back(op);
                                if (currentBatch.size() == 4) { 
                                    allBatches.push_back(currentBatch);
                                    currentBatch.clear();
                                    currentFdRoot = nullptr;
                                }
                            } else {
                                if (currentBatch.size() >= 2) allBatches.push_back(currentBatch);
                                currentBatch.clear();
                                currentBatch.push_back(op);
                                currentFdRoot = fdRoot;
                            }
                            continue;
                        }
                    }
                }

                StringRef opName = op->getName().getStringRef();
                
                if (opName == "cir.load" || opName == "cir.cast" || 
                    opName == "cir.get_element" || opName == "cir.ptr_stride" ||
                    opName == "cir.const") {
                    continue;
                }

                if (currentBatch.size() >= 2) allBatches.push_back(currentBatch);
                currentBatch.clear();
                currentFdRoot = nullptr;
            }
            
            if (currentBatch.size() >= 2) allBatches.push_back(currentBatch);
        });

        // --------------------------------------------------------------------
        // Step 3: APPLY THE REPLACEMENTS
        // --------------------------------------------------------------------
        for (auto &batch : allBatches) {
            OpBuilder replaceBuilder(batch.back()); 
            std::string funcName = "ioopt_writev_" + std::to_string(batch.size());
            
            SmallVector<Value, 9> newArgs;
            auto firstWriteInBatch = cast<CallOpInterface>(batch[0]);
            
            newArgs.push_back(firstWriteInBatch->getOperand(0)); // Shared FD
            
            for (Operation *w : batch) {
                auto wCall = cast<CallOpInterface>(w);
                newArgs.push_back(wCall->getOperand(1)); // Buffer
                newArgs.push_back(wCall->getOperand(2)); // Size
            }

            auto writevCall = func::CallOp::create(
                replaceBuilder,
                batch.back()->getLoc(),
                funcName,
                firstWriteInBatch->getResultTypes(),
                newArgs
            );

            for (Operation *w : batch) {
                w->replaceAllUsesWith(writevCall.getResults());
                w->erase();
            }
            
            llvm::errs() << "[IOOpt-Vectored] SUCCESS: Merged " << batch.size() << " writes into " << funcName << "!\n";
        }
    } 
};

// ============================================================================
// 3. Compute-Bound Block -> Auto-Asynchrony (io_uring / aio)
// ============================================================================
struct PromoteToAsyncIOPass : public PassWrapper<PromoteToAsyncIOPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteToAsyncIOPass)
    StringRef getArgument() const final { return "io-async-promotion"; }
    StringRef getDescription() const final { return "Software pipelines blocking I/O with independent compute"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<mlir::io::IODialect, cir::CIRDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        IRRewriter rewriter(&getContext());

        SmallVector<CallOpInterface> readCandidates;

        module.walk([&](CallOpInterface callOp) {
            StringRef callee = getCalleeName(callOp);
            if (callee == "read" || callee == "read64" || callee == "read32") {
                readCandidates.push_back(callOp);
            }
        });

        for (CallOpInterface readOp : readCandidates) {
            if (readOp->getNumOperands() < 3) continue;
            
            Value fd = readOp->getOperand(0);
            Value buffer = readOp->getOperand(1);
            Value size = readOp->getOperand(2);
            Value bytesReadResult = readOp->getResult(0);

            Block *block = readOp->getBlock();
            auto it = Block::iterator(readOp.getOperation());
            ++it; 

            Operation *waitInsertionPoint = nullptr;
            int independentComputeCount = 0;

            // Safe backward tracer for ClangIR memory tracking
            auto getRoot = [&](Value v) {
                while (auto def = v.getDefiningOp()) {
                    StringRef n = def->getName().getStringRef();
                    if (n == "cir.cast" || n == "cir.load" || n == "cir.ptr_stride" || n == "cir.get_element") {
                        v = def->getOperand(0);
                    } else break;
                }
                return v;
            };
            Value rootBuf = getRoot(buffer);

            while (it != block->end()) {
                Operation &currentOp = *it;
                bool isDependent = false;

                // Safely check if the instruction touches the buffer or the read result
                for (Value operand : currentOp.getOperands()) {
                    if (operand == bytesReadResult || getRoot(operand) == rootBuf) {
                        isDependent = true;
                        break;
                    }
                }

                if (!isDependent) {
                    if (currentOp.hasTrait<OpTrait::IsTerminator>()) isDependent = true;
                    else if (isa<CallOpInterface>(&currentOp)) {
                        auto memEffects = dyn_cast<MemoryEffectOpInterface>(&currentOp);
                        if (!memEffects || !memEffects.hasNoEffect()) isDependent = true;
                    }
                }

                if (isDependent) {
                    waitInsertionPoint = &currentOp;
                    break;
                }

                StringRef opName = currentOp.getName().getStringRef();
                if (opName.starts_with("cir.binop") || opName.starts_with("arith.")) {
                    independentComputeCount++;
                }
                
                ++it;
            }

            if (independentComputeCount > 0 && waitInsertionPoint) {
                rewriter.setInsertionPoint(readOp);
                Location loc = readOp->getLoc();

                auto i32Ty = rewriter.getI32Type();
                auto i64Ty = rewriter.getI64Type();

                Value stdFd = fd;
                if (stdFd.getType() != i32Ty) stdFd = mlir::io::IOCastOp::create(rewriter, loc, i32Ty, stdFd);
                Value stdSize = size;
                if (stdSize.getType() != i64Ty) stdSize = mlir::io::IOCastOp::create(rewriter, loc, i64Ty, stdSize);

                auto submitOp = mlir::io::SubmitOp::create(rewriter, loc, i32Ty, stdFd, buffer, stdSize);

                rewriter.setInsertionPoint(waitInsertionPoint);
                auto waitOp = mlir::io::WaitOp::create(rewriter, loc, i64Ty, submitOp.getTicket());

                Value waitResult = waitOp.getBytesRead();
                Type expectedReturnType = readOp->getResult(0).getType();
                if (waitResult.getType() != expectedReturnType) {
                    waitResult = mlir::io::IOCastOp::create(rewriter, loc, expectedReturnType, waitResult);
                }

                readOp->replaceAllUsesWith(ValueRange{waitResult});
                rewriter.eraseOp(readOp);
                
                llvm::errs() << "[IOOpt-Telemetry] SUCCESS: Promoted read to async submit/wait!\n";
            }
        }
    }
};

// ============================================================================
// 4. Bulk Random Access -> Auto-mmap Promotion
// ============================================================================
struct MmapPromotionPass : public PassWrapper<MmapPromotionPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MmapPromotionPass)
    StringRef getArgument() const final { return "io-mmap-promotion"; }
    StringRef getDescription() const final { return "Promotes bulk file reads into memory mapped (mmap) buffers"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<mlir::io::IODialect, mlir::arith::ArithDialect, cir::CIRDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        IRRewriter rewriter(&getContext());
        SmallVector<Operation*, 4> opsToErase; 

        auto safeCast = [&](Value v, Type targetTy, Location loc) -> Value {
            if (v.getType() == targetTy) return v;
            if (v.getType().isIntOrIndex() && targetTy.isIntOrIndex()) {
                unsigned inBits = v.getType().getIntOrFloatBitWidth();
                unsigned outBits = targetTy.getIntOrFloatBitWidth();
                if (inBits < outBits) return arith::ExtUIOp::create(rewriter, loc, targetTy, v);
                if (inBits > outBits) return arith::TruncIOp::create(rewriter, loc, targetTy, v);
            }
            return mlir::io::IOCastOp::create(rewriter, loc, targetTy, v);
        };

        module.walk([&](CallOpInterface allocCall) {
            StringRef allocName = getCalleeName(allocCall);
            if (allocName != "malloc" && allocName != "malloc32") return;

            Value allocSize = allocCall->getOperand(0);
            Value allocResult = allocCall->getResult(0);

            CallOpInterface targetRead = nullptr;
            for (Operation *op = allocCall->getNextNode(); op != nullptr; op = op->getNextNode()) {
                if (auto readCall = dyn_cast<CallOpInterface>(op)) {
                    StringRef readName = getCalleeName(readCall);
                    if (readName == "read" || readName == "read64" || readName == "read32") {
                        targetRead = readCall;
                        break;
                    }
                }
            }

            if (!targetRead) return;

            auto getRootAllocation = [](Value v) {
                while (Operation *def = v.getDefiningOp()) {
                    StringRef name = def->getName().getStringRef();
                    if (name == "cir.cast" || name == "cir.load" || name == "cir.ptr_stride" || name == "cir.get_element") {
                        v = def->getOperand(0);
                    } else break;
                }
                return v;
            };

            bool bufferMatches = false;
            Value readRoot = getRootAllocation(targetRead->getOperand(1));
            
            if (readRoot == allocResult) {
                bufferMatches = true; // Standard MLIR direct connection
            } else {
                // ClangIR connection: Check if malloc was stored into the read's alloca
                for (Operation *op = allocCall->getNextNode(); op != targetRead; op = op->getNextNode()) {
                    StringRef opName = op->getName().getStringRef();
                    if (opName == "cir.store" || opName == "memref.store") {
                        if (getRootAllocation(op->getOperand(0)) == allocResult && op->getOperand(1) == readRoot) {
                            bufferMatches = true;
                            break;
                        }
                    }
                }
            }

            if (bufferMatches && getRootAllocation(allocSize) == getRootAllocation(targetRead->getOperand(2))) {
                rewriter.setInsertionPoint(allocCall);
                Location loc = allocCall->getLoc();

                auto i32Ty = rewriter.getI32Type();
                auto i64Ty = rewriter.getI64Type();
                
                Value zeroOffset = arith::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(0));
                Value stdSize = safeCast(allocSize, i64Ty, loc);

                Value stdFd = targetRead->getOperand(0);
                if (auto loadOp = stdFd.getDefiningOp()) {
                    if (loadOp->getName().getStringRef() == "cir.load" || loadOp->getName().getStringRef() == "memref.load") {
                        stdFd = rewriter.clone(*loadOp)->getResult(0);
                    }
                }
                stdFd = safeCast(stdFd, i32Ty, loc);

                auto mmapOp = mlir::io::MmapOp::create(rewriter, loc, allocResult.getType(), stdFd, stdSize, zeroOffset);
                rewriter.replaceOp(allocCall, mmapOp.getBuffer());

                rewriter.setInsertionPoint(targetRead);
                Value replacementResult = safeCast(allocSize, targetRead->getResult(0).getType(), targetRead->getLoc());

                targetRead->replaceAllUsesWith(ValueRange{replacementResult});
                opsToErase.push_back(targetRead);

                llvm::errs() << "[IOOpt-Telemetry] SUCCESS: Promoted malloc+read to io.mmap!\n";
            }
        });

        for (auto op : opsToErase) op->erase();
    }
};

// ============================================================================
// 5. Compute-Heavy Loops -> Prefetch Injection
// ============================================================================
struct PrefetchInjectionPass : public PassWrapper<PrefetchInjectionPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrefetchInjectionPass)
    StringRef getArgument() const final { return "io-prefetch-injection"; }
    StringRef getDescription() const final { return "Injects kernel read-ahead hints into compute-heavy sequential I/O loops"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<mlir::io::IODialect, mlir::arith::ArithDialect, cir::CIRDialect, scf::SCFDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        IRRewriter rewriter(&getContext());

        auto safeCast = [&](Value v, Type targetTy, Location loc) -> Value {
            if (v.getType() == targetTy) return v;
            if (v.getType().isIntOrIndex() && targetTy.isIntOrIndex()) {
                unsigned inBits = v.getType().getIntOrFloatBitWidth();
                unsigned outBits = targetTy.getIntOrFloatBitWidth();
                if (inBits < outBits) return arith::ExtUIOp::create(rewriter, loc, targetTy, v);
                if (inBits > outBits) return arith::TruncIOp::create(rewriter, loc, targetTy, v);
            }
            return mlir::io::IOCastOp::create(rewriter, loc, targetTy, v);
        };

        auto injectPrefetch = [&](Operation *loopOp) {
            Value fdToPrefetch;
            Value sizeToPrefetch;
            Operation *readInsertionPoint = nullptr;
            
            bool alreadyPrefetched = false;
            int computeInstructionCount = 0;

            loopOp->walk([&](Operation *op) {
                if (isa<mlir::io::PrefetchOp>(op)) {
                    alreadyPrefetched = true;
                } 
                else if (auto callOp = dyn_cast<CallOpInterface>(op)) {
                    StringRef callee = getCalleeName(callOp); 
                    if (callee == "read" || callee == "read64" || callee == "read32") {
                        if (!readInsertionPoint && callOp->getNumOperands() == 3) {
                            fdToPrefetch = callOp->getOperand(0);
                            sizeToPrefetch = callOp->getOperand(2);
                            readInsertionPoint = op;
                        }
                    }
                } 
                else {
                    StringRef opName = op->getName().getStringRef();
                    if (opName.starts_with("cir.binop") || opName.starts_with("arith.")) {
                        computeInstructionCount++;
                    }
                }
            });

            if (!readInsertionPoint || alreadyPrefetched || computeInstructionCount < 5) return; 

            rewriter.setInsertionPoint(loopOp);
            Value four = arith::ConstantOp::create(rewriter, loopOp->getLoc(), rewriter.getI64IntegerAttr(4));

            // Move insertion point back inside the loop for the rest of the operations
            rewriter.setInsertionPoint(readInsertionPoint);
            Location loc = readInsertionPoint->getLoc();

            Value stdFd = safeCast(fdToPrefetch, rewriter.getI32Type(), loc);
            Value sizeI64 = safeCast(sizeToPrefetch, rewriter.getI64Type(), loc);
            Value lookaheadSize = arith::MulIOp::create(rewriter, loc, sizeI64, four);

            mlir::io::PrefetchOp::create(rewriter, loc, stdFd, lookaheadSize);
            llvm::errs() << "[IOOpt-Telemetry] SUCCESS: Injected io.prefetch ahead of loop read!\n";
        };

        module.walk([&](cir::ForOp op) { injectPrefetch(op); });
        module.walk([&](scf::ForOp op) { injectPrefetch(op); });
    }
};

} // end anonymous namespace

// ============================================================================
// Registration Hooks
// ============================================================================
namespace mlir {
namespace io {
    void registerAdvancedIOPasses() {
        PassRegistration<ZeroCopyPromotionPass>();
        PassRegistration<BlockVectoredIOPass>();
        PassRegistration<PromoteToAsyncIOPass>();
        PassRegistration<MmapPromotionPass>();
        PassRegistration<PrefetchInjectionPass>();
    }
} 
}
