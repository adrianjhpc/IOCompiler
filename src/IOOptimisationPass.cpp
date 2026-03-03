#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include <vector>

using namespace llvm;

namespace {

  struct IOArgs {
    Value *Target; 
    Value *Buffer; 
    Value *Length; 
    enum { NONE, C_FWRITE, C_FREAD, POSIX_WRITE, POSIX_READ, CXX_WRITE, CXX_READ } Type;
  };

  IOArgs getIOArguments(CallInst *Call) {
    Function *F = Call->getCalledFunction();
    if (!F || !F->hasName() || !F->isDeclaration()) return {nullptr, nullptr, nullptr, IOArgs::NONE};

    std::string Demangled = llvm::demangle(F->getName().str());
    
    if (Demangled == "fwrite") return {Call->getArgOperand(3), Call->getArgOperand(0), Call->getArgOperand(2), IOArgs::C_FWRITE};
    if (Demangled == "fread")  return {Call->getArgOperand(3), Call->getArgOperand(0), Call->getArgOperand(2), IOArgs::C_FREAD};
    if (Demangled == "write")  return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_WRITE};
    if (Demangled == "read")   return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_READ};

    if ((Demangled.find("std::basic_ostream") != std::string::npos || 
         Demangled.find("std::ostream") != std::string::npos) && Demangled.find("::write") != std::string::npos) {
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::CXX_WRITE};
    }
    
    return {nullptr, nullptr, nullptr, IOArgs::NONE};
  }

  bool checkAdjacency(Value *Buf1, Value *Size1, Value *Buf2, const DataLayout &DL) {
    if (auto *GEP = dyn_cast<GEPOperator>(Buf2)) {
      if (GEP->getPointerOperand() == Buf1) {
        APInt Offset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
        if (GEP->accumulateConstantOffset(DL, Offset)) {
          if (auto *ConstSize1 = dyn_cast<ConstantInt>(Size1)) {
            if (Offset == ConstSize1->getValue()) return true; 
          }
        }
      }
    }
    return false;
  }

bool isSafeToAmalgamate(CallInst *Call1, CallInst *Call2, AAResults &AA, const DataLayout &DL) {
    if (Call1->getCalledFunction() != Call2->getCalledFunction()) return false;
    
    IOArgs Args1 = getIOArguments(Call1);
    IOArgs Args2 = getIOArguments(Call2);
    if (!Args1.Target || !Args2.Target) return false;

    // --- SSA HARDENED TARGET CHECK ---
    bool TargetsMatch = (Args1.Target == Args2.Target);
    if (!TargetsMatch) {
        // Under -O3, the same 'fd' might be loaded into two different SSA variables.
        // Check if both targets are loads from the exact same memory allocation.
        auto *Load1 = dyn_cast<LoadInst>(Args1.Target);
        auto *Load2 = dyn_cast<LoadInst>(Args2.Target);
        if (Load1 && Load2 && Load1->getPointerOperand() == Load2->getPointerOperand()) {
            TargetsMatch = true;
        }
    }
    if (!TargetsMatch) return false;
    // ---------------------------------

    auto getPreciseLoc = [&](Value *Buf, Value *Len) {
        if (auto *C = dyn_cast<ConstantInt>(Len)) {
            return MemoryLocation(Buf, LocationSize::precise(C->getZExtValue()));
        }
        return MemoryLocation::getBeforeOrAfter(Buf);
    };

    MemoryLocation Loc1 = getPreciseLoc(Args1.Buffer, Args1.Length);
    MemoryLocation Loc2 = getPreciseLoc(Args2.Buffer, Args2.Length);

    for (Instruction *I = Call1->getNextNode(); I != Call2; I = I->getNextNode()) {
        if (!I) return false; 
        
        // Skip purely mathematical operations (add, mul, etc.) injected by -O3
        if (!I->mayReadOrWriteMemory()) continue;
        
        if (isModOrRefSet(AA.getModRefInfo(I, Loc1)) || isModOrRefSet(AA.getModRefInfo(I, Loc2))) {
            return false;
        }
    }
    
    return true; 
}
  
    
  void mergeWrites(CallInst *Call1, CallInst *Call2) {
    IRBuilder<> Builder(Call2);
    IOArgs Args1 = getIOArguments(Call1);
    IOArgs Args2 = getIOArguments(Call2);
    Value *NewLen = Builder.CreateAdd(Args1.Length, Args2.Length, "merged.io.len");
    std::vector<Value *> NewArgs;
    if (Args1.Type == IOArgs::C_FWRITE) {
      NewArgs = {Args1.Buffer, Call1->getArgOperand(1), NewLen, Args1.Target};
    } else {
      NewArgs = {Args1.Target, Args1.Buffer, NewLen};
    }
    Builder.CreateCall(Call1->getCalledFunction(), NewArgs);
    Call1->eraseFromParent();
    Call2->eraseFromParent();
    errs() << "[IOOpt] SUCCESS: Merged contiguous writes!\n";
  }

  // ADDED: The writev logic as a standalone extension
  void mergeWritesWithVectoredIO(CallInst *Call1, CallInst *Call2, Module *M) {
    IRBuilder<> Builder(Call2);
    IOArgs Args1 = getIOArguments(Call1);
    IOArgs Args2 = getIOArguments(Call2);
    const DataLayout &DL = M->getDataLayout();

    Type *Int32Ty = Builder.getInt32Ty();
    Type *PtrTy = PointerType::getUnqual(M->getContext());
    Type *SizeTy = DL.getIntPtrType(M->getContext());
    
    FunctionType *WritevTy = FunctionType::get(SizeTy, {Int32Ty, PtrTy, Int32Ty}, false);
    FunctionCallee WritevFunc = M->getOrInsertFunction("writev", WritevTy);
    StructType *IovecTy = StructType::get(M->getContext(), {PtrTy, SizeTy});
    ArrayType *IovArrayTy = ArrayType::get(IovecTy, 2);

    AllocaInst *IovArray = Builder.CreateAlloca(IovArrayTy, nullptr, "iovec.array");

    auto FillIov = [&](int Index, Value *Buf, Value *Len) {
      Value *IovPtr = Builder.CreateInBoundsGEP(IovArrayTy, IovArray, {Builder.getInt32(0), Builder.getInt32(Index)});
      Builder.CreateStore(Buf, Builder.CreateStructGEP(IovecTy, IovPtr, 0));
      Builder.CreateStore(Builder.CreateIntCast(Len, SizeTy, false), Builder.CreateStructGEP(IovecTy, IovPtr, 1));
    };

    FillIov(0, Args1.Buffer, Args1.Length);
    FillIov(1, Args2.Buffer, Args2.Length);

    Value *Fd = Builder.CreateIntCast(Args1.Target, Int32Ty, false);
    Builder.CreateCall(WritevFunc, {Fd, IovArray, Builder.getInt32(2)});
    Call1->eraseFromParent();
    Call2->eraseFromParent();
    errs() << "[IOOpt] SUCCESS: Converted separate writes to writev!\n";
  }

  // REVERTED: Original hoistRead logic
  bool hoistRead(CallInst *ReadCall, AAResults &AA, const DataLayout &DL) {
    IOArgs Args = getIOArguments(ReadCall);
    if (!Args.Buffer) return false;
    MemoryLocation DestLoc(Args.Buffer, LocationSize::beforeOrAfterPointer());
    Instruction *InsertPoint = ReadCall;
    Instruction *CurrentInst = ReadCall->getPrevNode();
    BasicBlock *CurrentBB = ReadCall->getParent();
    
    while (true) {
      if (CurrentInst) {
        if (!CurrentInst->isTerminator() && !isa<PHINode>(CurrentInst)) {
          bool DependsOnPrev = false;
          for (Value *Op : ReadCall->operands()) if (Op == CurrentInst) { DependsOnPrev = true; break; }
          if (DependsOnPrev) break;
          if (CurrentInst->mayReadOrWriteMemory()) {
            if (isModOrRefSet(AA.getModRefInfo(CurrentInst, DestLoc))) break;
            MemoryLocation TargetLoc(Args.Target, LocationSize::beforeOrAfterPointer());
            if (isModSet(AA.getModRefInfo(CurrentInst, TargetLoc))) break;
          }
          InsertPoint = CurrentInst;
        }
        CurrentInst = CurrentInst->getPrevNode();
      } else {
        BasicBlock *PredBB = CurrentBB->getSinglePredecessor();
        if (!PredBB || PredBB->getTerminator()->getNumSuccessors() > 1) break;
        CurrentBB = PredBB;
        CurrentInst = CurrentBB->getTerminator();
      }
    }
    if (InsertPoint != ReadCall) {
      ReadCall->moveBefore(InsertPoint->getIterator());
      return true;
    }
    return false;
  }

  // REVERTED: Original loop optimization
  bool optimiseLoopIO(Loop *L, ScalarEvolution &SE, const DataLayout &DL) {
    BasicBlock *Preheader = L->getLoopPreheader();
    if (!Preheader) return false;
    unsigned TripCount = SE.getSmallConstantTripCount(L);
    if (TripCount == 0) return false;
    bool Changed = false;
    SCEVExpander Expander(SE, DL, "io.expander");
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : llvm::make_early_inc_range(*BB)) {
        if (auto *Call = dyn_cast<CallInst>(&I)) {
          IOArgs Args = getIOArguments(Call);
          if (Args.Type != IOArgs::POSIX_WRITE && Args.Type != IOArgs::C_FWRITE) continue;
          if (!L->isLoopInvariant(Args.Target)) continue;
          auto *ConstLen = dyn_cast<ConstantInt>(Args.Length);
          if (!ConstLen) continue;
          const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(Args.Buffer));
          if (!AddRec || AddRec->getLoop() != L) continue;
          const SCEVConstant *Step = dyn_cast<SCEVConstant>(AddRec->getStepRecurrence(SE));
          if (Step && Step->getValue()->getValue() == ConstLen->getValue()) {
            Value *BasePtr = Expander.expandCodeFor(AddRec->getStart(), Args.Buffer->getType(), Preheader->getTerminator());
            IRBuilder<> Builder(Preheader->getTerminator());
            Value *NewLen = Builder.getIntN(Args.Length->getType()->getIntegerBitWidth(), TripCount * ConstLen->getZExtValue());
            std::vector<Value *> NewArgs;
            if (Args.Type == IOArgs::C_FWRITE) NewArgs = {BasePtr, Call->getArgOperand(1), NewLen, Args.Target};
            else NewArgs = {Args.Target, BasePtr, NewLen};
            Builder.CreateCall(Call->getCalledFunction(), NewArgs);
            Call->eraseFromParent();
            Changed = true;
          }
        }
      }
    }
    return Changed;
  }

  struct IOOptimisationPass : public PassInfoMixin<IOOptimisationPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
      errs() << "[IOOpt] Analyzing function: " << F.getName() << "\n";
      bool Changed = false;
      AAResults &AA = FAM.getResult<AAManager>(F);
      const DataLayout &DL = F.getParent()->getDataLayout();
      LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
      ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);

      for (Loop *L : LI) if (optimiseLoopIO(L, SE, DL)) Changed = true;

      for (BasicBlock &BB : F) {
        CallInst *LastWrite = nullptr;
        for (Instruction &I : llvm::make_early_inc_range(BB)) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
            IOArgs CArgs = getIOArguments(Call);
            if (CArgs.Type == IOArgs::POSIX_WRITE || CArgs.Type == IOArgs::C_FWRITE || CArgs.Type == IOArgs::CXX_WRITE) {
              if (LastWrite && isSafeToAmalgamate(LastWrite, Call, AA, DL)) {
                IOArgs LArgs = getIOArguments(LastWrite);
                if (checkAdjacency(LArgs.Buffer, LArgs.Length, CArgs.Buffer, DL)) mergeWrites(LastWrite, Call);
                else mergeWritesWithVectoredIO(LastWrite, Call, F.getParent());
                Changed = true;
                LastWrite = nullptr;
              } else {
		LastWrite = Call;
		errs() << "[IOOpt] FAILED: Unsafe to merge or intervening memory hazard.\n";
	      }
            } else if (CArgs.Type == IOArgs::POSIX_READ || CArgs.Type == IOArgs::C_FREAD) {
              if (hoistRead(Call, AA, DL)) Changed = true;
              LastWrite = nullptr;
            } else LastWrite = nullptr;
          }
        }
      }
      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
}

// -----------------------------------------------------------------------------
// Pass plugin registration
// -----------------------------------------------------------------------------
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "IOOpt", LLVM_VERSION_STRING, [](PassBuilder &PB) {
    
    // 1. Used by 'opt' in your lit tests
    PB.registerPipelineParsingCallback([](StringRef Name, FunctionPassManager &FPM, ...) {
      if (Name == "io-opt") { FPM.addPass(IOOptimisationPass()); return true; }
      return false;
    });

    // 2. Used by 'clang++ -O3' when building bench_fast
    // We must include 'ThinOrFullLTOPhase Phase' to satisfy the LLVM 20 type checker
    PB.registerPipelineEarlySimplificationEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase Phase) {
          if (Level != OptimizationLevel::O0) {
            FunctionPassManager FPM;
            FPM.addPass(IOOptimisationPass());
            MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
          }
        });
  }};
}
