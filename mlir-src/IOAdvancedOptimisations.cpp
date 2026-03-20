#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Analysis/AliasAnalysis.h"

using namespace mlir;

namespace {

struct PromoteToZeroCopyPattern : public RewritePattern {
    PromoteToZeroCopyPattern(MLIRContext *context)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/2, context) {}

   LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        auto readCall = dyn_cast<CallOpInterface>(op);
        if (!readCall) return failure();

        auto calleeAttr = dyn_cast<SymbolRefAttr>(readCall.getCallableForCallee());
        if (!calleeAttr || calleeAttr.getRootReference() != "read") return failure();
        
        if (readCall.getArgOperands().size() != 3) return failure();
        
        llvm::errs() << "[IOOpt-Telemetry] Found 'read' call. Scanning block...\n";
        
        Value fdIn = readCall.getArgOperands()[0];
        Value readSize = readCall.getArgOperands()[2];

        // Hardened to trace through array indexing and pointer math
        auto getRootAllocation = [](Value v) {
            while (Operation *def = v.getDefiningOp()) {
                StringRef name = def->getName().getStringRef();
                if (name == "cir.cast" || name == "cir.load" || 
                    name == "cir.ptr_stride" || name == "cir.get_element") {
                    v = def->getOperand(0);
                } else {
                    break;
                }
            }
            return v;
        };        

        Value readBufferRoot = getRootAllocation(readCall.getArgOperands()[1]);

        CallOpInterface writeCall = nullptr;
        
        for (Operation *nextOp = op->getNextNode(); nextOp != nullptr; nextOp = nextOp->getNextNode()) {
            
            if (auto maybeWrite = dyn_cast<CallOpInterface>(nextOp)) {
                auto nextCallee = dyn_cast<SymbolRefAttr>(maybeWrite.getCallableForCallee());
                if (nextCallee && nextCallee.getRootReference() == "write") {
                    if (maybeWrite.getArgOperands().size() == 3) {
                        Value writeBufferRoot = getRootAllocation(maybeWrite.getArgOperands()[1]);
                        if (readBufferRoot == writeBufferRoot) {
                            writeCall = maybeWrite;
                            break; 
                        } else {
                            llvm::errs() << "[IOOpt-Telemetry] Abort: Found 'write', but buffer roots mismatch!\n";
                        }
                    }
                }
            }

            StringRef opName = nextOp->getName().getStringRef();
            
            if (opName == "cir.store") {
                Value storePtr = nextOp->getOperand(1);
                if (getRootAllocation(storePtr) == readBufferRoot) {
                    llvm::errs() << "[IOOpt-Telemetry] Abort: cir.store mutated the buffer root!\n";
                    return failure(); 
                }
                continue; 
            }
            
            if (isa<CallOpInterface>(nextOp)) {
                llvm::errs() << "[IOOpt-Telemetry] Abort: Intervening CallOp found: " << opName << "\n";
                return failure(); 
            }
            
            if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(nextOp)) {
                if (memInterface.hasEffect<MemoryEffects::Write>()) {
                    llvm::errs() << "[IOOpt-Telemetry] Abort: Intervening memory write found: " << opName << "\n";
                    return failure();
                }
            }
        }

        if (!writeCall) {
            llvm::errs() << "[IOOpt-Telemetry] Abort: Reached end of block. No matching 'write' found.\n";
            return failure();
        }

        Value fdOut = writeCall.getArgOperands()[0];
        Value writeSize = writeCall.getArgOperands()[2];

        if (getRootAllocation(readSize) != getRootAllocation(writeSize)) {
            llvm::errs() << "[IOOpt-Telemetry] Abort: Size variables do not match.\n";
            return failure();
        }

        // The Rewrite
        rewriter.setInsertionPoint(writeCall);
        Value nullOffset = readCall.getArgOperands()[1]; 

        auto sendfileCall = rewriter.create<func::CallOp>(
            writeCall.getLoc(),
            "sendfile",
            readCall->getResultTypes(),
            ValueRange{fdOut, fdIn, nullOffset, writeSize}
        );

        rewriter.replaceOp(writeCall, sendfileCall.getResults());
        rewriter.replaceOp(readCall, sendfileCall.getResults());

        llvm::errs() << "[IOOpt-Telemetry] SUCCESS: Replaced read/write with sendfile!\n";
        return success();
    }

};

struct ZeroCopyPromotionPass : public PassWrapper<ZeroCopyPromotionPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZeroCopyPromotionPass)
    StringRef getArgument() const final { return "io-zero-copy-promotion"; }
    StringRef getDescription() const final { return "Promotes read/write pairs to zero-copy sendfile syscalls"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<func::FuncDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *context = &getContext();

        // PRE-PASS: Walk the module sequentially to find a 'read' call.
        // We will steal its exact ClangIR types so our sendfile signature matches perfectly!
        CallOpInterface firstRead = nullptr;
        module.walk([&](CallOpInterface call) {
            auto callee = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
            if (callee && callee.getRootReference() == "read" && call.getArgOperands().size() == 3) {
                firstRead = call;
                return WalkResult::interrupt(); // Stop walking, we got what we need!
            }
            return WalkResult::advance();
        });

        // Safely declare 'sendfile' using the stolen types
        auto sendfileSym = StringAttr::get(context, "sendfile");
        if (firstRead && !module.lookupSymbol(sendfileSym)) {
            OpBuilder builder(module.getBodyRegion());
            
            // Steal the precise ClangIR types dynamically!
            Type fdType = firstRead.getArgOperands()[0].getType();
            Type ptrType = firstRead.getArgOperands()[1].getType(); 
            Type sizeType = firstRead.getArgOperands()[2].getType();
            Type retType = firstRead->getResultTypes()[0];
            
            auto sendfileType = builder.getFunctionType({fdType, fdType, ptrType, sizeType}, {retType});
            builder.create<func::FuncOp>(module.getLoc(), "sendfile", sendfileType).setPrivate();
        }

        // Now run the multithreaded greedy pattern matcher
        RewritePatternSet patterns(context);
        patterns.add<PromoteToZeroCopyPattern>(context);
        
        if (failed(applyPatternsGreedily(module, std::move(patterns))))
            signalPassFailure();
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
        // Step 1: PRE-PASS (Steal types and inject ioopt_writev_* intrinsics)
        // --------------------------------------------------------------------
        CallOpInterface firstWrite = nullptr;
        module.walk([&](CallOpInterface call) {
            auto callee = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
            if (callee && callee.getRootReference() == "write" && call.getArgOperands().size() == 3) {
                firstWrite = call;
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });

        // If there are no writes in this module, just exit early!
        if (!firstWrite) return;

        OpBuilder builder(module.getBodyRegion());
        Type fdType = firstWrite.getArgOperands()[0].getType();
        Type ptrType = firstWrite.getArgOperands()[1].getType(); 
        Type sizeType = firstWrite.getArgOperands()[2].getType();
        Type retType = firstWrite->getResultTypes()[0];
        
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
                builder.create<func::FuncOp>(module.getLoc(), funcName, funcType).setPrivate();
            }
        }

        // --------------------------------------------------------------------
        // Step 2: DETERMINISTIC TOP-DOWN WALK
        // --------------------------------------------------------------------
        // We bypass the greedy pattern rewriter entirely to guarantee order.
        SmallVector<SmallVector<Operation*, 4>> allBatches;

        module.walk([&](Block *block) {
            SmallVector<Operation*, 4> currentBatch;
            Value currentFdRoot = nullptr;

            // Our trusty SSA Root Tracer
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
                
                // If we find a write, check if we can add it to the current batch
                if (auto call = dyn_cast<CallOpInterface>(op)) {
                    auto callee = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
                    if (callee && callee.getRootReference() == "write" && call.getArgOperands().size() == 3) {
                        Value fdRoot = getRootAllocation(call.getArgOperands()[0]);
                        
                        if (currentBatch.empty()) {
                            currentBatch.push_back(op);
                            currentFdRoot = fdRoot;
                        } else if (fdRoot == currentFdRoot) {
                            currentBatch.push_back(op);
                            if (currentBatch.size() == 4) { // Cap at 4 for intrinsic chunking
                                allBatches.push_back(currentBatch);
                                currentBatch.clear();
                                currentFdRoot = nullptr;
                            }
                        } else {
                            // Hit a write to a DIFFERENT file. Save old batch, start new one.
                            if (currentBatch.size() >= 2) allBatches.push_back(currentBatch);
                            currentBatch.clear();
                            currentBatch.push_back(op);
                            currentFdRoot = fdRoot;
                        }
                        continue;
                    }
                }

                StringRef opName = op->getName().getStringRef();
                
                // Safe memory preparation instructions (ignore them)
                if (opName == "cir.load" || opName == "cir.cast" || 
                    opName == "cir.get_element" || opName == "cir.ptr_stride" ||
                    opName == "cir.const") {
                    continue;
                }

                // HAZARD DETECTED: Bank the current batch and reset!
                if (currentBatch.size() >= 2) allBatches.push_back(currentBatch);
                currentBatch.clear();
                currentFdRoot = nullptr;
            }
            
            // End of the block: Bank any remaining writes
            if (currentBatch.size() >= 2) allBatches.push_back(currentBatch);
        });

        // --------------------------------------------------------------------
        // Step 3: APPLY THE REPLACEMENTS
        // --------------------------------------------------------------------
        // We do this AFTER the walk to avoid crashing the block iterator!
        for (auto &batch : allBatches) {
            // Insert exactly where the last write was
            OpBuilder replaceBuilder(batch.back()); 
            std::string funcName = "ioopt_writev_" + std::to_string(batch.size());
            
            SmallVector<Value, 9> newArgs;
            auto firstWriteInBatch = cast<CallOpInterface>(batch[0]);
            
            newArgs.push_back(firstWriteInBatch.getArgOperands()[0]); // Shared FD
            
            for (Operation *w : batch) {
                auto wCall = cast<CallOpInterface>(w);
                newArgs.push_back(wCall.getArgOperands()[1]); // Buffer
                newArgs.push_back(wCall.getArgOperands()[2]); // Size
            }

            auto writevCall = replaceBuilder.create<func::CallOp>(
                batch.back()->getLoc(),
                funcName,
                firstWriteInBatch->getResultTypes(),
                newArgs
            );

            // Replace all original writes with the new writev call
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

namespace {

struct PromoteToAsyncIOPass : public PassWrapper<PromoteToAsyncIOPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteToAsyncIOPass)
    StringRef getArgument() const final { return "io-async-promotion"; }
    StringRef getDescription() const final { return "Software pipelines blocking I/O with independent compute"; }

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        IRRewriter rewriter(&getContext());
        
        // Request MLIR's Alias Analysis for the current function
        AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();

        SmallVector<func::CallOp> readCandidates;

        // 1. Walk the function to find synchronous 'read' calls
        func.walk([&](func::CallOp callOp) {
            StringRef callee = callOp.getCallee();
            if (callee == "read" || callee == "read64") {
                readCandidates.push_back(callOp);
            }
        });

        for (func::CallOp readOp : readCandidates) {
            if (readOp.getNumOperands() < 3) continue;
            
            Value fd = readOp.getOperand(0);
            Value buffer = readOp.getOperand(1);
            Value size = readOp.getOperand(2);
            Value bytesReadResult = readOp.getResult(0);

            Block *block = readOp->getBlock();
            auto it = Block::iterator(readOp);
            ++it; // Start scanning immediately *after* the read

            Operation *waitInsertionPoint = nullptr;
            int independentComputeCount = 0;

            // 2. The Dependency Scanner (Now with Alias Analysis!)
            while (it != block->end()) {
                Operation &currentOp = *it;
                bool isDependent = false;

                // Rule A: Does this operation use the bytes_read integer result directly?
                for (Value operand : currentOp.getOperands()) {
                    if (operand == bytesReadResult) {
                        isDependent = true;
                        break;
                    }
                }

                if (!isDependent) {
                    // Rule B: Ask AliasAnalysis if this instruction touches our buffer!
                    // In Async I/O, the Kernel owns the buffer between submit and wait.
                    // If the CPU tries to Read (Ref) OR Write (Mod) to it, we MUST wait!
                    ModRefResult modRef = aliasAnalysis.getModRef(&currentOp, buffer);
                    
                    if (modRef.isMod() || modRef.isRef()) {
                        isDependent = true;
                    }
                }

                // Rule C: Safely handle control flow boundaries and black boxes
                if (!isDependent) {
                    // If it's a terminator (branch, return), we must stop and wait.
                    // We cannot safely float an async wait into another basic block 
                    // without advanced control-flow dominance analysis.
                    if (currentOp.hasTrait<OpTrait::IsTerminator>()) {
                        isDependent = true;
                    }
                    // If it's an opaque function call that might do hidden I/O or state mutation
                    else if (auto call = dyn_cast<CallOpInterface>(&currentOp)) {
                        auto memEffects = dyn_cast<MemoryEffectOpInterface>(&currentOp);
                        // If it doesn't explicitly declare itself free of memory effects, assume it's a hazard.
                        if (!memEffects || !memEffects.hasNoEffect()) {
                            isDependent = true;
                        }
                    }
                }

                // We found the hazard! This is where the wait() must go.
                if (isDependent) {
                    waitInsertionPoint = &currentOp;
                    break;
                }

                independentComputeCount++;
                ++it;
            }

            // 3. Evaluate Profitability
            // We only split the I/O if we actually managed to jump over independent instructions.
            // (e.g., if independentComputeCount is 0, a synchronous read is faster anyway).
            if (independentComputeCount > 0 && waitInsertionPoint) {
                rewriter.setInsertionPoint(readOp);

                // Create the ASYNC SUBMIT call
                auto submitCall = rewriter.create<func::CallOp>(
                    readOp.getLoc(),
                    rewriter.getI32Type(), // Returns an async ticket/token
                    "io_submit",
                    ArrayRef<Value>{fd, buffer, size}
                );
                Value asyncTicket = submitCall.getResult(0);

                // Move the rewriter down to right before the hazard
                rewriter.setInsertionPoint(waitInsertionPoint);

                // Create the ASYNC WAIT call
                auto waitCall = rewriter.create<func::CallOp>(
                    readOp.getLoc(),
                    readOp.getResult(0).getType(), // Returns actual bytes read (ssize_t)
                    "io_wait",
                    ArrayRef<Value>{asyncTicket}
                );

                // Swap the synchronous return value for the async wait return value
                readOp.replaceAllUsesWith(waitCall.getResults());
                
                // Erase the old synchronous read
                rewriter.eraseOp(readOp);
            }
        }
    }
};

} // end anonymous namespace

// ============================================================================
// 4. Bulk Random Access -> Auto-mmap Promotion
// ============================================================================
struct PromoteToMmapPattern : public RewritePattern {
    PromoteToMmapPattern(MLIRContext *context)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/2, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        // TODO:
        // 1. Match an allocation (malloc/alloca) whose size exactly matches a file's size (fstat).
        // 2. Match a single massive 'read' that fills this buffer.
        // 3. Replace the allocation and read with an 'io.mmap' operation.
        return failure();
    }
};

struct MmapPromotionPass : public PassWrapper<MmapPromotionPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MmapPromotionPass)
    StringRef getArgument() const final { return "io-mmap-promotion"; }
    StringRef getDescription() const final { return "Promotes bulk file reads into memory mapped (mmap) buffers"; }

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<PromoteToMmapPattern>(&getContext());
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

// ============================================================================
// 5. Predictable Scan -> Automated Prefetch Injection
// ============================================================================
struct InjectPrefetchPattern : public OpRewritePattern<scf::ForOp> {
    InjectPrefetchPattern(MLIRContext *context)
        : OpRewritePattern<scf::ForOp>(context, /*benefit=*/1) {}

    LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter &rewriter) const override {
        // TODO:
        // 1. Check if the loop contains a sequential 'read' pattern (pointer advances by fixed size).
        // 2. Check if the loop also contains heavy compute (high instruction count).
        // 3. Calculate a safe lookahead distance.
        // 4. Inject 'posix_fadvise' (or a custom io.prefetch op) into the loop to trigger kernel read-ahead.
        return failure();
    }
};

struct PrefetchInjectionPass : public PassWrapper<PrefetchInjectionPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrefetchInjectionPass)
    StringRef getArgument() const final { return "io-prefetch-injection"; }
    StringRef getDescription() const final { return "Injects kernel read-ahead hints into compute-heavy sequential I/O loops"; }

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<InjectPrefetchPattern>(&getContext());
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
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
} // namespace io
} // namespace mlir
