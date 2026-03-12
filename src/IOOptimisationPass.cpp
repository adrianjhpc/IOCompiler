#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/CFG.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Transforms/Utils/LoopSimplify.h" 
#include "llvm/Transforms/Utils/LCSSA.h"        
#include "llvm/Demangle/Demangle.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <vector>
#include <cstdlib>
#include <string>

using namespace llvm;

static unsigned getEnvOrDefault(const char* Name, unsigned Default) {
  if (const char* Val = std::getenv(Name)) return std::stoi(Val);
  return Default;
}

static unsigned IOBatchThreshold = 4;
static unsigned IOShadowBufferSize = 4096;
static unsigned IOHighWaterMark = 65536;

namespace {

  struct IOArgs {
    Value *Target; 
    Value *Buffer; 
    Value *Length; 
    enum { 
      NONE, C_FWRITE, C_FREAD, POSIX_WRITE, POSIX_READ, POSIX_PWRITE, POSIX_PREAD, 
      CXX_WRITE, CXX_READ, MPI_WRITE_AT, MPI_READ_AT, 
      SPLICE, SENDFILE, POSIX_PWRITEV, POSIX_PREADV, IO_SUBMIT, AIO_WRITE 
    } Type;
  };

  IOArgs getIOArguments(CallInst *Call) {
    auto getCStreamBytes = [](CallInst *CI) -> Value* {
      Value *Size = CI->getArgOperand(1);
      Value *Count = CI->getArgOperand(2);
      if (auto *CSize = dyn_cast<ConstantInt>(Size)) {
        if (CSize->getZExtValue() == 1) return Count;
        if (auto *CCount = dyn_cast<ConstantInt>(Count)) {
          return ConstantInt::get(Count->getType(), CSize->getZExtValue() * CCount->getZExtValue());
        }
      }
      if (auto *CCount = dyn_cast<ConstantInt>(Count)) {
        if (CCount->getZExtValue() == 1) return Size;
      }
      return nullptr; 
    };

    Function *F = Call->getCalledFunction();
    if (!F || !F->hasName() || !F->isDeclaration()) return {nullptr, nullptr, nullptr, IOArgs::NONE};

    std::string Demangled = llvm::demangle(F->getName().str());
    StringRef Name = F->getName();
    
    // Standard POSIX
    if (Demangled == "pread" || Demangled == "pread64") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PREAD};
    if (Demangled == "pwrite" || Demangled == "pwrite64") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PWRITE};
    if (Demangled == "write" || Demangled == "write64") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_WRITE};
    if (Demangled == "read" || Demangled == "read64")   return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_READ};
    if (Demangled == "fwrite") return {Call->getArgOperand(3), Call->getArgOperand(0), getCStreamBytes(Call), IOArgs::C_FWRITE};
    if (Demangled == "fread")  return {Call->getArgOperand(3), Call->getArgOperand(0), getCStreamBytes(Call), IOArgs::C_FREAD};
    
    // High-Performance & Zero-Copy POSIX
    if (Demangled == "preadv" || Demangled == "preadv2") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PREADV};
    if (Demangled == "pwritev" || Demangled == "pwritev2") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PWRITEV};
    if (Demangled == "splice") return {Call->getArgOperand(2), Call->getArgOperand(0), Call->getArgOperand(4), IOArgs::SPLICE}; // out_fd, in_fd, len
    if (Demangled == "sendfile" || Demangled == "sendfile64") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(3), IOArgs::SENDFILE}; // out, in, count
    
    // Async Linux I/O
    if (Demangled == "io_submit") return {Call->getArgOperand(0), Call->getArgOperand(2), Call->getArgOperand(1), IOArgs::IO_SUBMIT}; // ctx, iocb**, nr
    if (Demangled == "aio_write" || Demangled == "aio_write64") return {Call->getArgOperand(0), nullptr, nullptr, IOArgs::AIO_WRITE}; 

    // MPI 
    if (Demangled == "MPI_File_write_at" || Demangled == "PMPI_File_write_at") return {Call->getArgOperand(0), Call->getArgOperand(2), Call->getArgOperand(3), IOArgs::MPI_WRITE_AT};
    if (Demangled == "MPI_File_read_at" || Demangled == "PMPI_File_read_at")  return {Call->getArgOperand(0), Call->getArgOperand(2), Call->getArgOperand(3), IOArgs::MPI_READ_AT};
    
    // C++ Standard Streams (std::basic_ostream / std::basic_istream)
    if ((Demangled.find("std::basic_ostream") != std::string::npos || Demangled.find("std::ostream") != std::string::npos) && 
         Demangled.find("::write") != std::string::npos) {
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::CXX_WRITE};
    }
    if ((Demangled.find("std::basic_istream") != std::string::npos || Demangled.find("std::istream") != std::string::npos) && 
         Demangled.find("::read") != std::string::npos) {
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::CXX_READ};
    }
    
    return {nullptr, nullptr, nullptr, IOArgs::NONE};
  }

  bool checkAdjacency(Value *Buf1, Value *Len1, Value *Buf2, const DataLayout &DL, ScalarEvolution *SE, bool AllowGaps = false) {
    if (SE) {
      const SCEV *Ptr1 = SE->getSCEV(Buf1);
      const SCEV *Ptr2 = SE->getSCEV(Buf2);
      const SCEV *Size1 = SE->getSCEV(Len1);

      if (!isa<SCEVCouldNotCompute>(Ptr1) && !isa<SCEVCouldNotCompute>(Ptr2) && !isa<SCEVCouldNotCompute>(Size1)) {
        const SCEV *ExtendedSize = SE->getTruncateOrZeroExtend(Size1, Ptr1->getType());
        const SCEV *ExpectedNext = SE->getAddExpr(Ptr1, ExtendedSize);

        if (ExpectedNext == Ptr2) return true; 

        if (AllowGaps) {
          const SCEV *Distance = SE->getMinusSCEV(Ptr2, ExpectedNext);
          if (auto *ConstDist = dyn_cast<SCEVConstant>(Distance)) {
            int64_t Gap = ConstDist->getValue()->getSExtValue();
            if (Gap >= 0 && Gap < 1024) return true;
          }
        }
      }
    }

    APInt Off1(DL.getIndexTypeSizeInBits(Buf1->getType()), 0);
    const Value *Base1 = Buf1->stripAndAccumulateConstantOffsets(DL, Off1, true);
    APInt Off2(DL.getIndexTypeSizeInBits(Buf2->getType()), 0);
    const Value *Base2 = Buf2->stripAndAccumulateConstantOffsets(DL, Off2, true);

    if (Base1 && Base1 == Base2) {
      if (auto *CLen = dyn_cast<ConstantInt>(Len1)) {
        uint64_t End1 = Off1.getZExtValue() + CLen->getZExtValue();
        uint64_t Start2 = Off2.getZExtValue();
        if (End1 == Start2) return true;
        if (AllowGaps && Start2 > End1 && (Start2 - End1) < 1024) return true;
      }
    }

    return false;
  }

  bool dependsOn(Value *V, Value *Target, int Depth = 0) {
    if (V == Target) return true;
    if (Depth > 4) return false; 
    if (auto *Inst = dyn_cast<Instruction>(V)) {
        for (Value *Op : Inst->operands()) {
            if (dependsOn(Op, Target, Depth + 1)) return true;
        }
    }
    return false;
  }

  bool isSafeToAddToBatch(const std::vector<CallInst*> &Batch, CallInst *NewCall, AAResults &AA, const DataLayout &DL, ScalarEvolution &SE, DominatorTree &DT, PostDominatorTree &PDT, bool &ForceShadowBuffer) {
    if (Batch.empty()) return true;

    CallInst *LastCall = Batch.back();
    IOArgs FirstArgs = getIOArguments(Batch.front());
    IOArgs LastArgs = getIOArguments(LastCall);
    IOArgs NewArgs = getIOArguments(NewCall);
    
    // --- NEW: Block structural modifications for Async / Struct-based APIs ---
    if (NewArgs.Type == IOArgs::IO_SUBMIT || NewArgs.Type == IOArgs::AIO_WRITE || 
        NewArgs.Type == IOArgs::POSIX_PREADV || NewArgs.Type == IOArgs::POSIX_PWRITEV) {
        errs() << "[IOOpt-Debug] Batch Break: Async/Vectored I/O merging requires dynamic struct array rewriting.\n";
        return false;
    }

    bool isReadBatch = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD || 
                        FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::MPI_READ_AT ||
                        FirstArgs.Type == IOArgs::CXX_READ);

    if (!DT.dominates(LastCall, NewCall)) {
      errs() << "[IOOpt-Debug] Batch Break: CFG Dominance violation.\n";
      return false;
    }

    if (isReadBatch) {
        for (Value *Op : NewCall->operands()) {
            if (auto *Inst = dyn_cast<Instruction>(Op)) {
                if (!DT.dominates(Inst, Batch.front())) {
                    errs() << "[IOOpt-Debug] Batch Break: SSA Use-Before-Def hazard for Read Hoisting.\n";
                    return false;
                }
            }
        }
    }

    if (LastCall->getCalledFunction() != NewCall->getCalledFunction()) {
      errs() << "[IOOpt-Debug] Batch Break: Function mismatch.\n";
      return false;
    }

    bool TargetIsSame = false;
    if (FirstArgs.Target == NewArgs.Target) {
      TargetIsSame = true;
    } else {
      const Value *Base1 = getUnderlyingObject(FirstArgs.Target);
      const Value *Base2 = getUnderlyingObject(NewArgs.Target);
      if (Base1 && Base2 && Base1 == Base2) {
        TargetIsSame = true;
      } else {
        auto *L1 = dyn_cast<LoadInst>(FirstArgs.Target);
        auto *L2 = dyn_cast<LoadInst>(NewArgs.Target);
        if (L1 && L2 && L1->getPointerOperand() == L2->getPointerOperand()) {
          TargetIsSame = true;
        }
      }
    }
    if (!TargetIsSame) {
      errs() << "[IOOpt-Debug] Batch Break: Target streams do not match.\n";
      return false;
    }

    // Zero-Copy Validation (splice/sendfile)
    if (NewArgs.Type == IOArgs::SPLICE || NewArgs.Type == IOArgs::SENDFILE) {
        if (FirstArgs.Buffer != NewArgs.Buffer) {
            errs() << "[IOOpt-Debug] Batch Break: Splice/Sendfile Input FDs do not match.\n";
            return false;
        }
    }

    if (NewArgs.Type == IOArgs::POSIX_PREAD || NewArgs.Type == IOArgs::POSIX_PWRITE) {
      Value *LastOffset = LastCall->getArgOperand(3);
      Value *NewOffset = NewCall->getArgOperand(3);
      Value *LastLen = LastArgs.Length;
      bool isContiguous = false;
    
      if (SE.isSCEVable(LastOffset->getType()) && SE.isSCEVable(NewOffset->getType()) && SE.isSCEVable(LastLen->getType())) {
        const SCEV *SLast = SE.getSCEV(LastOffset);
        const SCEV *SNew = SE.getSCEV(NewOffset);
        const SCEV *SLen = SE.getSCEV(LastLen);
        if (!isa<SCEVCouldNotCompute>(SLast) && !isa<SCEVCouldNotCompute>(SNew) && !isa<SCEVCouldNotCompute>(SLen)) {
          const SCEV *ExpectedNext = SE.getAddExpr(SLast, SE.getTruncateOrZeroExtend(SLen, SLast->getType()));
          if (ExpectedNext == SNew) isContiguous = true;
        }
      }
      if (!isContiguous) {
        if (auto *CLastOff = dyn_cast<ConstantInt>(LastOffset)) {
          if (auto *CNewOff = dyn_cast<ConstantInt>(NewOffset)) {
            if (auto *CLen = dyn_cast<ConstantInt>(LastLen)) {
              if (CLastOff->getZExtValue() + CLen->getZExtValue() == CNewOff->getZExtValue()) isContiguous = true;
            }
          }
        }
      }
      if (!isContiguous) {
        errs() << "[IOOpt-Debug] Batch Break: Explicit offsets are not contiguous.\n";
        return false;
      }

    } else if (NewArgs.Type == IOArgs::MPI_READ_AT || NewArgs.Type == IOArgs::MPI_WRITE_AT) {
      if (LastCall->getArgOperand(5) != NewCall->getArgOperand(5)) return false;
      if (LastCall->getArgOperand(4) != NewCall->getArgOperand(4)) return false; 
      Value *LastOffset = LastCall->getArgOperand(1);
      Value *NewOffset = NewCall->getArgOperand(1);
      Value *LastCount = LastArgs.Length;
      bool isContiguous = false;

      if (SE.isSCEVable(LastOffset->getType()) && SE.isSCEVable(NewOffset->getType())) {
        const SCEV *SLast = SE.getSCEV(LastOffset);
        const SCEV *SNew = SE.getSCEV(NewOffset);
        const SCEV *SCount = SE.getTruncateOrZeroExtend(SE.getSCEV(LastCount), SLast->getType());
        if (!isa<SCEVCouldNotCompute>(SLast) && !isa<SCEVCouldNotCompute>(SNew)) {
          const SCEV *ExpectedNext = SE.getAddExpr(SLast, SCount);
          if (ExpectedNext == SNew) isContiguous = true;
        }
      }
      if (!isContiguous) {
        errs() << "[IOOpt-Debug] Batch Break: MPI offsets are not contiguous.\n";
        return false;
      }
    } 

    auto getPreciseLoc = [&](Value *Buf, Value *Len) {
      if (auto *C = dyn_cast<ConstantInt>(Len)) return MemoryLocation(Buf, LocationSize::precise(C->getZExtValue()));
      return MemoryLocation(Buf, LocationSize::beforeOrAfterPointer());
    };

    MemoryLocation NewLoc = getPreciseLoc(NewArgs.Buffer, NewArgs.Length);

    BasicBlock *BB1 = LastCall->getParent();
    BasicBlock *BB2 = NewCall->getParent();

    if (BB1 != BB2) {
      if (isReadBatch) {
          errs() << "[IOOpt-Debug] Batch Break: Cross-block batching forbidden for reads (EOF safety).\n";
          return false;
      }
      
      if (!PDT.dominates(BB2, BB1)) {
          auto *Term = BB1->getTerminator();
          if (auto *Br = dyn_cast<BranchInst>(Term)) {
              if (!Br->isConditional() || !dependsOn(Br->getCondition(), LastCall)) {
                  errs() << "[IOOpt-Debug] Batch Break: CFG diverges for non-I/O reasons.\n";
                  return false;
              }
          } else {
              errs() << "[IOOpt-Debug] Batch Break: Non-branching terminator in divergent block.\n";
              return false;
          }
      }
    }
    
    bool isExplicitOffset = (FirstArgs.Type == IOArgs::POSIX_PWRITE || FirstArgs.Type == IOArgs::POSIX_PREAD ||
                             FirstArgs.Type == IOArgs::MPI_WRITE_AT || FirstArgs.Type == IOArgs::MPI_READ_AT);

    LoadInst *Load1 = dyn_cast<LoadInst>(FirstArgs.Target);

    auto checkHazard = [&](Instruction *Inst) -> bool {
      bool isOpaqueCall = false;

      if (auto *CI = dyn_cast<CallInst>(Inst)) {
        if (getIOArguments(CI).Type != IOArgs::NONE) {
            errs() << "[IOOpt-Debug] Batch Break: Intervening I/O call found.\n";
            return true;
        }
        if (!CI->onlyReadsMemory() && !CI->doesNotAccessMemory()) {
            if (!isExplicitOffset && (FirstArgs.Type == IOArgs::POSIX_WRITE || FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::CXX_WRITE)) {
                errs() << "[IOOpt-Debug] Batch Break: Opaque function mutation on implicit offset.\n";
                return true;
            }
            if (isReadBatch) {
                errs() << "[IOOpt-Debug] Batch Break: Opaque function mutation in READ batch.\n";
                return true; 
            }
            isOpaqueCall = true; 
            ForceShadowBuffer = true;
        }
      }

      // Bypass buffer checks for SPLICE/SENDFILE because Buffer is an FD, not memory.
      if (FirstArgs.Type == IOArgs::SPLICE || FirstArgs.Type == IOArgs::SENDFILE) return false;

      if (Inst->mayReadOrWriteMemory()) {
        if (isModSet(AA.getModRefInfo(Inst, NewLoc))) {
            bool isFalseAlarm = (isOpaqueCall && isa<AllocaInst>(NewLoc.Ptr) && !PointerMayBeCaptured(NewLoc.Ptr, true, true));
            if (!isFalseAlarm) {
                if (isReadBatch) {
                    errs() << "[IOOpt-Debug] Batch Break: Target buffer overwritten.\n";
                    return true;
                }
                errs() << "[IOOpt-Debug] Hazard Detected: Forcing Eager Shadow Buffer snapshot!\n";
                ForceShadowBuffer = true;
            }
        }

        for (CallInst *BatchedCall : Batch) {
          IOArgs BArgs = getIOArguments(BatchedCall);
          MemoryLocation BLoc = getPreciseLoc(BArgs.Buffer, BArgs.Length);
          
          if (isModSet(AA.getModRefInfo(Inst, BLoc))) {
             bool isFalseAlarm = (isOpaqueCall && isa<AllocaInst>(BLoc.Ptr) && !PointerMayBeCaptured(BLoc.Ptr, true, true));
             if (!isFalseAlarm) {
                 if (isReadBatch) {
                     errs() << "[IOOpt-Debug] Batch Break: Previously batched buffer overwritten.\n";
                     return true;
                 }
                 errs() << "[IOOpt-Debug] Hazard Detected: Forcing Eager Shadow Buffer snapshot!\n";
                 ForceShadowBuffer = true; 
             }
          }
        }

        if (!isOpaqueCall) {
          if (FirstArgs.Target->getType()->isPointerTy()) {
            MemoryLocation TargetLoc(FirstArgs.Target, LocationSize::beforeOrAfterPointer());
            if (isModSet(AA.getModRefInfo(Inst, TargetLoc))) {
                errs() << "[IOOpt-Debug] Batch Break: File stream pointer mutated.\n";
                return true;
            }
          }
          if (Load1) {
            MemoryLocation FdLoc(Load1->getPointerOperand(), LocationSize::beforeOrAfterPointer());
            if (isModSet(AA.getModRefInfo(Inst, FdLoc))) {
                errs() << "[IOOpt-Debug] Batch Break: File descriptor mutated.\n";
                return true;
            }
          }
        }
      }
      return false; 
    };

    for (Instruction *I = LastCall->getNextNode(); I != nullptr; I = I->getNextNode()) {
      if (I == NewCall) return true; 
      if (checkHazard(I)) return false;
    }

    SmallPtrSet<BasicBlock*, 8> Visited;
    std::vector<BasicBlock*> Worklist;
    for (BasicBlock *Succ : successors(BB1)) {
      if (Succ != BB2) {
        Worklist.push_back(Succ);
        Visited.insert(Succ);
      }
    }

    while (!Worklist.empty()) {
      BasicBlock *CurrBB = Worklist.back();
      Worklist.pop_back();
      for (Instruction &I : *CurrBB) {
        if (checkHazard(&I)) return false;
      }
      for (BasicBlock *Succ : successors(CurrBB)) {
        if (!DT.dominates(BB1, Succ)) {
            errs() << "[IOOpt-Debug] Batch Break: CFG Bleed (Block escapes dominance).\n";
            return false; 
        }
        if (Succ != BB2 && Visited.insert(Succ).second) {
          Worklist.push_back(Succ);
        }
      }
    }

    for (Instruction &I : *BB2) {
      if (&I == NewCall) break; 
      if (checkHazard(&I)) return false;
    }

    return true; 
  }

  enum class IOPattern { Contiguous, Strided, ShadowBuffer, DynamicShadowBuffer, Vectored, Unprofitable };

  bool isStridedPattern(const std::vector<CallInst*> &Batch, const DataLayout &DL, ScalarEvolution *SE) {
    if (!SE || Batch.size() < 2) return false;
    const SCEV *Ptr0 = SE->getSCEV(getIOArguments(Batch[0]).Buffer);
    const SCEV *Ptr1 = SE->getSCEV(getIOArguments(Batch[1]).Buffer);
    const SCEV *Stride = SE->getMinusSCEV(Ptr1, Ptr0);

    if (isa<SCEVCouldNotCompute>(Stride) || Stride->isZero()) return false;

    for (size_t i = 1; i < Batch.size() - 1; ++i) {
      const SCEV *CurrentPtr = SE->getSCEV(getIOArguments(Batch[i]).Buffer);
      const SCEV *NextPtr = SE->getSCEV(getIOArguments(Batch[i+1]).Buffer);
      if (SE->getMinusSCEV(NextPtr, CurrentPtr) != Stride) return false;
    }
    return true;
  }

  IOPattern classifyBatch(const std::vector<CallInst*> &Batch, const DataLayout &DL, 
                          uint64_t &OutTotalRange, ScalarEvolution *SE, bool ForceShadowBuffer) {
    if (Batch.size() < 2) return IOPattern::Unprofitable;

    IOArgs FirstArgs = getIOArguments(Batch.front());
    bool isRead = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD || FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::CXX_READ);
    
    // Zero-Copy classification 
    if (FirstArgs.Type == IOArgs::SPLICE || FirstArgs.Type == IOArgs::SENDFILE) {
        return IOPattern::Contiguous; 
    }
    
    size_t ExpectedThreshold = 2;

    if (!ForceShadowBuffer) {
        bool StrictPhysical = true;
        for (size_t i = 0; i < Batch.size() - 1; ++i) {
          if (!checkAdjacency(getIOArguments(Batch[i]).Buffer, getIOArguments(Batch[i]).Length, 
                              getIOArguments(Batch[i+1]).Buffer, DL, SE, false)) {
            StrictPhysical = false;
            break;
          }
        }
        if (StrictPhysical) return IOPattern::Contiguous;

        if (!isRead && isStridedPattern(Batch, DL, SE)) {
          if (auto *ConstLen = dyn_cast<ConstantInt>(getIOArguments(Batch.front()).Length)) {
            uint64_t ElementBytes = ConstLen->getZExtValue();
            if (ElementBytes == 1 || ElementBytes == 2 || ElementBytes == 4 || ElementBytes == 8) {
              OutTotalRange = ElementBytes;
              return IOPattern::Strided;
            }
          }
        }

        if (Batch.size() >= ExpectedThreshold) {
          if (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::POSIX_WRITE || 
              FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::POSIX_PWRITE) {
            return IOPattern::Vectored;
          }
        }
    }

    if (FirstArgs.Type == IOArgs::POSIX_WRITE || FirstArgs.Type == IOArgs::POSIX_PWRITE || 
        FirstArgs.Type == IOArgs::MPI_WRITE_AT || FirstArgs.Type == IOArgs::C_FWRITE || FirstArgs.Type == IOArgs::CXX_WRITE) { 
        
      uint64_t TotalConstSize = 0;
      bool AllSizesConstant = true;
        
      for (CallInst *C : Batch) {
        if (auto *CI = dyn_cast<ConstantInt>(getIOArguments(C).Length)) {
          TotalConstSize += CI->getZExtValue();
        } else {
          AllSizesConstant = false;
          break;
        }
      }
        
      if (AllSizesConstant && TotalConstSize > 0 && TotalConstSize <= IOShadowBufferSize) {
        OutTotalRange = TotalConstSize;
        return IOPattern::ShadowBuffer;
      }
      
      ExpectedThreshold = 2; 
      if (Batch.size() >= ExpectedThreshold) {
          if (!ForceShadowBuffer) {
              return IOPattern::DynamicShadowBuffer;
          } else {
              errs() << "[IOOpt-Debug] Batch Unprofitable: Hazard prevents dynamic heap buffering.\n";
              return IOPattern::Unprofitable;
          }
      }
    }

    return IOPattern::Unprofitable;
  }

  bool flushBatch(std::vector<CallInst*> &Batch, Module *M, ScalarEvolution &SE, bool ForceShadowBuffer) {
    if (Batch.empty()) return false;

    const DataLayout &DL = M->getDataLayout();
    uint64_t TotalConstSize = 0;
    
    IOPattern Pattern = classifyBatch(Batch, DL, TotalConstSize, &SE, ForceShadowBuffer);

    if (Pattern == IOPattern::Unprofitable) {
      Batch.clear();
      return false; 
    }

    IOArgs FirstArgs = getIOArguments(Batch.front());
    bool isRead = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD || FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::MPI_READ_AT || FirstArgs.Type == IOArgs::CXX_READ);
    bool isExplicit = (FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::POSIX_PWRITE);

    Instruction *InsertPt = isRead ? Batch.front() : Batch.back();
    IRBuilder<> InsertBuilder(InsertPt);

    Value *TotalDynLen = InsertBuilder.getIntN(FirstArgs.Length->getType()->getIntegerBitWidth(), 0);
    for (CallInst *C : Batch) {
        Value *L = getIOArguments(C).Length;
        if (L->getType() != TotalDynLen->getType()) L = InsertBuilder.CreateZExtOrTrunc(L, TotalDynLen->getType());
        TotalDynLen = InsertBuilder.CreateAdd(TotalDynLen, L);
    }

    CallInst *MergedCall = nullptr;

    switch (Pattern) {
    case IOPattern::Contiguous: {
      std::vector<Value *> NewArgs;
      if (FirstArgs.Type == IOArgs::MPI_WRITE_AT || FirstArgs.Type == IOArgs::MPI_READ_AT) {
        NewArgs = { Batch[0]->getArgOperand(0), Batch[0]->getArgOperand(1), FirstArgs.Buffer, TotalDynLen, Batch[0]->getArgOperand(4), Batch[0]->getArgOperand(5) };
      } else if (FirstArgs.Type == IOArgs::C_FWRITE || FirstArgs.Type == IOArgs::C_FREAD) {
        Value *SizeOne = InsertBuilder.getIntN(TotalDynLen->getType()->getIntegerBitWidth(), 1);
        NewArgs = {FirstArgs.Buffer, SizeOne, TotalDynLen, FirstArgs.Target};
      } else if (FirstArgs.Type == IOArgs::SPLICE || FirstArgs.Type == IOArgs::SENDFILE) {
        // Zero-copy concatenation
        NewArgs = {FirstArgs.Target, FirstArgs.Buffer, Batch[0]->getArgOperand(2), TotalDynLen};
        if (FirstArgs.Type == IOArgs::SPLICE) NewArgs.push_back(Batch[0]->getArgOperand(5)); // Add flags back for splice
      } else if (isExplicit) {
        NewArgs = {FirstArgs.Target, FirstArgs.Buffer, TotalDynLen, Batch[0]->getArgOperand(3)};
      } else {
        NewArgs = {FirstArgs.Target, FirstArgs.Buffer, TotalDynLen};
      }
      MergedCall = InsertBuilder.CreateCall(Batch[0]->getCalledFunction(), NewArgs);
      errs() << "[IOOpt] SUCCESS: N-Way merged " << Batch.size() << " contiguous " << (isRead ? "reads" : "writes") << "!\n";
      break;
    }

    case IOPattern::Strided: {
      unsigned ElementBytes = TotalConstSize; 
      unsigned NumElements = Batch.size();
      Type *ElementTy = InsertBuilder.getIntNTy(ElementBytes * 8); 
      auto *VecTy = FixedVectorType::get(ElementTy, NumElements);
    
      Value *GatherVec = PoisonValue::get(VecTy);
      for (unsigned i = 0; i < NumElements; ++i) {
        IOArgs Args = getIOArguments(Batch[i]);
        LoadInst *LoadedVal = InsertBuilder.CreateLoad(ElementTy, Args.Buffer, "strided.load");
        GatherVec = InsertBuilder.CreateInsertElement(GatherVec, LoadedVal, InsertBuilder.getInt32(i), "gather.insert");
      }
    
      Function *F = Batch.back()->getFunction();
      IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());
      AllocaInst *ContiguousBuf = EntryBuilder.CreateAlloca(VecTy, nullptr, "simd.shadow.buf");
      ContiguousBuf->setAlignment(Align(16));
      InsertBuilder.CreateStore(GatherVec, ContiguousBuf);
    
      Value *BufCast = InsertBuilder.CreatePointerCast(ContiguousBuf, InsertBuilder.getPtrTy());
      std::vector<Value *> NewArgs;
      if (FirstArgs.Type == IOArgs::C_FWRITE) {
        Value *SizeOne = InsertBuilder.getIntN(TotalDynLen->getType()->getIntegerBitWidth(), 1);
        NewArgs = {BufCast, SizeOne, TotalDynLen, FirstArgs.Target};
      } else if (isExplicit) {
        NewArgs = {FirstArgs.Target, BufCast, TotalDynLen, Batch[0]->getArgOperand(3)};
      } else {
        NewArgs = {FirstArgs.Target, BufCast, TotalDynLen};
      }
      MergedCall = InsertBuilder.CreateCall(Batch[0]->getCalledFunction(), NewArgs);
      errs() << "[IOOpt] SUCCESS: SIMD Gathered " << NumElements << " strided writes into 1!\n";
      break;
    }

    case IOPattern::ShadowBuffer: {
      Function *F = Batch.back()->getFunction();
      IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());

      Type *Int8Ty = InsertBuilder.getInt8Ty();
      ArrayType *ShadowArrTy = ArrayType::get(Int8Ty, TotalConstSize);
      AllocaInst *ShadowBuf = EntryBuilder.CreateAlloca(ShadowArrTy, nullptr, "shadow.buf");
      ShadowBuf->setAlignment(Align(16)); 

      uint64_t CurrentOffset = 0;
      for (size_t i = 0; i < Batch.size(); ++i) {
        CallInst *C = Batch[i];
        IOArgs Args = getIOArguments(C);

        IRBuilder<> CallBuilder(C);
        Value *DestPtr = CallBuilder.CreateInBoundsGEP(
                       ShadowArrTy, ShadowBuf,
                       {CallBuilder.getInt32(0), CallBuilder.getInt32(CurrentOffset)},
                       "shadow.ptr"
                       );

        CallBuilder.CreateMemCpy(DestPtr, Align(1), Args.Buffer, Align(1), Args.Length);
        if (auto *ConstLen = dyn_cast<ConstantInt>(Args.Length)) {
          CurrentOffset += ConstLen->getZExtValue();
        }
      }

      Value *BufPtr = InsertBuilder.CreatePointerCast(ShadowBuf, InsertBuilder.getPtrTy());

      std::vector<Value *> NewArgs;
      if (FirstArgs.Type == IOArgs::MPI_WRITE_AT) {
        NewArgs = { Batch[0]->getArgOperand(0), Batch[0]->getArgOperand(1), BufPtr, TotalDynLen, Batch[0]->getArgOperand(4), Batch[0]->getArgOperand(5) };
      } else if (FirstArgs.Type == IOArgs::C_FWRITE) {
        Value *SizeOne = InsertBuilder.getIntN(TotalDynLen->getType()->getIntegerBitWidth(), 1);
        NewArgs = {BufPtr, SizeOne, TotalDynLen, FirstArgs.Target};
      } else if (isExplicit) {
        NewArgs = {FirstArgs.Target, BufPtr, TotalDynLen, Batch[0]->getArgOperand(3)};
      } else {
        NewArgs = {FirstArgs.Target, BufPtr, TotalDynLen};
      }

      MergedCall = InsertBuilder.CreateCall(Batch[0]->getCalledFunction(), NewArgs);
      errs() << "[IOOpt] SUCCESS: Shadow Buffered " << Batch.size() << " writes into 1 (" << TotalConstSize << " bytes)!\n";
      break;
    }

    case IOPattern::DynamicShadowBuffer: {
      Type *SizeTy = DL.getIntPtrType(M->getContext());
      Type *Int8Ty = InsertBuilder.getInt8Ty();
      Type *PtrTy = InsertBuilder.getPtrTy();
      
      FunctionCallee MallocFunc = M->getOrInsertFunction("malloc", PtrTy, SizeTy);
      FunctionCallee FreeFunc = M->getOrInsertFunction("free", InsertBuilder.getVoidTy(), PtrTy);
      
      Value *MallocSize = InsertBuilder.CreateZExtOrTrunc(TotalDynLen, SizeTy);
      Value *HeapBuf = InsertBuilder.CreateCall(MallocFunc, {MallocSize}, "dyn.shadow.buf");
      
      Value *CurrentOffset = InsertBuilder.getIntN(SizeTy->getIntegerBitWidth(), 0);
      for (size_t i = 0; i < Batch.size(); ++i) {
          CallInst *C = Batch[i];
          IOArgs Args = getIOArguments(C);
          Value *Len = InsertBuilder.CreateZExtOrTrunc(Args.Length, SizeTy);
          Value *DestPtr = InsertBuilder.CreateInBoundsGEP(Int8Ty, HeapBuf, CurrentOffset, "dyn.dest");
          InsertBuilder.CreateMemCpy(DestPtr, Align(1), Args.Buffer, Align(1), Len);
          CurrentOffset = InsertBuilder.CreateAdd(CurrentOffset, Len, "dyn.offset");
      }
      
      std::vector<Value *> NewArgs;
      if (FirstArgs.Type == IOArgs::MPI_WRITE_AT) {
        NewArgs = { Batch[0]->getArgOperand(0), Batch[0]->getArgOperand(1), HeapBuf, TotalDynLen, Batch[0]->getArgOperand(4), Batch[0]->getArgOperand(5) };
      } else if (FirstArgs.Type == IOArgs::C_FWRITE) {
        Value *SizeOne = InsertBuilder.getIntN(TotalDynLen->getType()->getIntegerBitWidth(), 1);
        NewArgs = {HeapBuf, SizeOne, TotalDynLen, FirstArgs.Target};
      } else if (isExplicit) {
        NewArgs = {FirstArgs.Target, HeapBuf, TotalDynLen, Batch[0]->getArgOperand(3)};
      } else {
        NewArgs = {FirstArgs.Target, HeapBuf, TotalDynLen};
      }
      
      MergedCall = InsertBuilder.CreateCall(Batch[0]->getCalledFunction(), NewArgs);
      InsertBuilder.CreateCall(FreeFunc, {HeapBuf});
      errs() << "[IOOpt] SUCCESS: Dynamic Heap Buffered " << Batch.size() << " writes!\n";
      break;
    }

    case IOPattern::Vectored: {
      Type *Int32Ty = InsertBuilder.getInt32Ty();
      Type *PtrTy = InsertBuilder.getPtrTy();
      Type *SizeTy = DL.getIntPtrType(M->getContext());
            
      StringRef FuncName = isRead ? (isExplicit ? "preadv" : "readv") : (isExplicit ? "pwritev" : "writev");
      FunctionType *VecTy = isExplicit ? 
        FunctionType::get(SizeTy, {Int32Ty, PtrTy, Int32Ty, Batch[0]->getArgOperand(3)->getType()}, false) :
        FunctionType::get(SizeTy, {Int32Ty, PtrTy, Int32Ty}, false);
            
      FunctionCallee VecFunc = M->getOrInsertFunction(FuncName, VecTy);
      StructType *IovecTy = StructType::get(M->getContext(), {PtrTy, SizeTy});
      ArrayType *IovArrayTy = ArrayType::get(IovecTy, Batch.size());
            
      Function *F = Batch.back()->getFunction();
      IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());
      AllocaInst *IovArray = EntryBuilder.CreateAlloca(IovArrayTy, nullptr, "iovec.array.N");
      IovArray->setAlignment(Align(8));
      
      for (size_t i = 0; i < Batch.size(); ++i) {
        IOArgs Args = getIOArguments(Batch[i]);
        Value *IovPtr = InsertBuilder.CreateInBoundsGEP(IovArrayTy, IovArray, {InsertBuilder.getInt32(0), InsertBuilder.getInt32(i)});
        InsertBuilder.CreateStore(Args.Buffer, InsertBuilder.CreateStructGEP(IovecTy, IovPtr, 0));
        InsertBuilder.CreateStore(InsertBuilder.CreateIntCast(Args.Length, SizeTy, false), InsertBuilder.CreateStructGEP(IovecTy, IovPtr, 1));
      }
            
      Value *IovBasePtr = InsertBuilder.CreateInBoundsGEP(IovArrayTy, IovArray, {InsertBuilder.getInt32(0), InsertBuilder.getInt32(0)}, "iovec.base.ptr");
      Value *Fd = InsertBuilder.CreateIntCast(FirstArgs.Target, Int32Ty, false);
      if (isExplicit) {
        MergedCall = InsertBuilder.CreateCall(VecFunc, {Fd, IovBasePtr, InsertBuilder.getInt32(Batch.size()), Batch[0]->getArgOperand(3)});
      } else {
        MergedCall = InsertBuilder.CreateCall(VecFunc, {Fd, IovBasePtr, InsertBuilder.getInt32(Batch.size())});
      }
      errs() << "[IOOpt] SUCCESS: N-Way converted " << Batch.size() << " " << (isRead ? "reads" : "writes") << " to " << FuncName << "!\n";
      break;
    }
    default: break;
    }

    IRBuilder<> RetBuilder(MergedCall->getNextNode());
    
    for (size_t i = 0; i < Batch.size(); ++i) {
      CallInst *C = Batch[i];
      if (C->use_empty()) {
          C->eraseFromParent();
          continue;
      }

      IOArgs CArgs = getIOArguments(C);
      Value *Rep = nullptr;

      if (CArgs.Type == IOArgs::CXX_WRITE) {
          Rep = C->getArgOperand(0); 
      } else if (CArgs.Type == IOArgs::MPI_WRITE_AT || CArgs.Type == IOArgs::MPI_READ_AT) {
          Rep = RetBuilder.getInt32(0); 
      } else {
          Value *ExpectedLen = CArgs.Length;
          if (CArgs.Type == IOArgs::C_FWRITE || CArgs.Type == IOArgs::C_FREAD) {
              ExpectedLen = C->getArgOperand(2);
          }

          if (!isRead && i != Batch.size() - 1) {
              Rep = ExpectedLen; 
          } else {
              Value *RealRet = MergedCall;
              if (RealRet->getType() != ExpectedLen->getType()) {
                  RealRet = RetBuilder.CreateIntCast(RealRet, ExpectedLen->getType(), true);
              }
              
              if (FirstArgs.Type == IOArgs::POSIX_WRITE || FirstArgs.Type == IOArgs::POSIX_READ || 
                  FirstArgs.Type == IOArgs::POSIX_PWRITE || FirstArgs.Type == IOArgs::POSIX_PREAD ||
                  FirstArgs.Type == IOArgs::SPLICE || FirstArgs.Type == IOArgs::SENDFILE) {
                  Value *Zero = RetBuilder.getIntN(RealRet->getType()->getIntegerBitWidth(), 0);
                  Value *IsErr = RetBuilder.CreateICmpSLT(RealRet, Zero);
                  Rep = RetBuilder.CreateSelect(IsErr, RealRet, ExpectedLen, "spoofed.posix.ret");
              } else {
                  Value *TotalDynCast = RetBuilder.CreateIntCast(TotalDynLen, RealRet->getType(), false);
                  Value *IsPerfect = RetBuilder.CreateICmpEQ(RealRet, TotalDynCast);
                  Value *Zero = RetBuilder.getIntN(ExpectedLen->getType()->getIntegerBitWidth(), 0);
                  Rep = RetBuilder.CreateSelect(IsPerfect, ExpectedLen, Zero, "spoofed.cstream.ret");
              }
          }

          if (C->getType() != Rep->getType()) {
              Rep = RetBuilder.CreateIntCast(Rep, C->getType(), false);
          }
      }
      
      C->replaceAllUsesWith(Rep);
      C->eraseFromParent();
    }
    
    Batch.clear();
    return true;
  }

  struct IOOptimisationPass : public PassInfoMixin<IOOptimisationPass> {
    IOOptimisationPass() {
      IOBatchThreshold = getEnvOrDefault("IO_BATCH_THRESHOLD", 4);
      IOShadowBufferSize = getEnvOrDefault("IO_SHADOW_BUFFER_MAX", 4096);
      IOHighWaterMark = getEnvOrDefault("IO_HIGH_WATER_MARK", 65536);
    }

    bool optimiseLoopIO(Loop *L, ScalarEvolution &SE, const DataLayout &DL) {
      
      BasicBlock *Preheader = L->getLoopPreheader();
      BasicBlock *ExitBB = L->getExitBlock();
      if (!Preheader || !ExitBB) {
          errs() << "[IOOpt-Debug] Loop Hoist Blocked: Loop has multiple exits or missing preheader.\n";
          return false;
      }

      const SCEV *BackedgeCount = SE.getBackedgeTakenCount(L);
      if (isa<SCEVCouldNotCompute>(BackedgeCount)) {
          errs() << "[IOOpt-Debug] Loop Hoist Blocked: Uncomputable loop bounds.\n";
          return false;
      }
      
      Type *IntPtrTy = DL.getIntPtrType(Preheader->getContext());
      const SCEV *TripCountSCEV = SE.getAddExpr(SE.getTruncateOrZeroExtend(BackedgeCount, IntPtrTy), SE.getOne(IntPtrTy));

      bool LoopChanged = false;
      Loop *HoistLoop = L;
      
      BasicBlock *HoistPreheader = HoistLoop->getLoopPreheader();
      BasicBlock *HoistExitBB = HoistLoop->getExitBlock();

      SCEVExpander Expander(SE, DL, "io.dyn.expander");

      for (BasicBlock *BB : L->blocks()) {
        for (Instruction &I : llvm::make_early_inc_range(*BB)) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
            IOArgs Args = getIOArguments(Call);
            
            bool isHoistableIO = (Args.Type == IOArgs::POSIX_WRITE || Args.Type == IOArgs::POSIX_READ ||
                                  Args.Type == IOArgs::C_FWRITE || Args.Type == IOArgs::C_FREAD ||
                                  Args.Type == IOArgs::CXX_WRITE || Args.Type == IOArgs::CXX_READ);

            if (isHoistableIO) {
                
              if (!HoistLoop->isLoopInvariant(Args.Length)) continue;

              Value *ExtraArg = nullptr;
              if (Args.Type == IOArgs::C_FWRITE || Args.Type == IOArgs::C_FREAD) {
                ExtraArg = Call->getArgOperand(1);
                if (!HoistLoop->isLoopInvariant(ExtraArg)) continue; 
              }

              if (!HoistLoop->isLoopInvariant(Args.Target)) continue;

              const SCEV *ElementSizeSCEV = SE.getTruncateOrZeroExtend(SE.getSCEV(Args.Length), IntPtrTy);
              const SCEV *TotalBytesSCEV = SE.getMulExpr(ElementSizeSCEV, TripCountSCEV);

              const SCEV *PtrSCEV = SE.getSCEV(Args.Buffer);
              Value *BasePointer = nullptr;

              if (auto *AddRec = dyn_cast<SCEVAddRecExpr>(PtrSCEV)) {
                if (AddRec->getLoop() != L) continue;

                const SCEV *StepSCEV = AddRec->getStepRecurrence(SE);
                StepSCEV = SE.getTruncateOrZeroExtend(StepSCEV, IntPtrTy);

                if (StepSCEV != ElementSizeSCEV) continue;

                if (!SE.isLoopInvariant(AddRec->getStart(), HoistLoop)) continue;

                BasePointer = Expander.expandCodeFor(AddRec->getStart(), Args.Buffer->getType(), HoistPreheader->getTerminator());
              }

              if (!BasePointer) continue;

              IRBuilder<> ExitBuilder(&*HoistExitBB->getFirstInsertionPt());
              Value *TotalLenVal = Expander.expandCodeFor(TotalBytesSCEV, Args.Length->getType(), &*ExitBuilder.GetInsertPoint());
              
              std::vector<Value *> NewArgs;
              if (Args.Type == IOArgs::C_FWRITE || Args.Type == IOArgs::C_FREAD) {
                NewArgs = {BasePointer, ExtraArg, TotalLenVal, Args.Target};
              } else {
                NewArgs = {Args.Target, BasePointer, TotalLenVal};
              }
              ExitBuilder.CreateCall(Call->getCalledFunction(), NewArgs);

              errs() << "[IOOpt] SUCCESS: Hoisted DYNAMIC Loop I/O! Deployed SCEV math in IR.\n";

              Call->eraseFromParent();
              LoopChanged = true;
            }
          }
        }
      }
      return LoopChanged;
    }

    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
      bool Changed = false;
      std::vector<Instruction*> Lifetimes; 
      
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto *CI = dyn_cast<CallInst>(&I)) {
            if (CI->getIntrinsicID() == Intrinsic::lifetime_end) {
              Lifetimes.push_back(CI);
            }
          }
        }
      }
      
      for (Instruction *I : Lifetimes) {
        I->eraseFromParent();
      }
      if (!Lifetimes.empty()) Changed = true;
            
      AAResults &AA = FAM.getResult<AAManager>(F);
      const DataLayout &DL = F.getParent()->getDataLayout();
      LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
      ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
      DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
      PostDominatorTree &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);
      
      for (Loop *L : LI.getLoopsInPreorder()) {
        if (optimiseLoopIO(L, SE, DL)) Changed = true;
      }
      
      std::vector<CallInst*> IOBatch;
      uint64_t CurrentBatchBytes = 0;
      bool ForceShadowBuffer = false;

      for (BasicBlock &BB : F) {
        for (Instruction &I : llvm::make_early_inc_range(BB)) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
              
            if (Function *CalleeF = Call->getCalledFunction()) {
              StringRef FuncName = CalleeF->getName();
              if (FuncName == "fsync" || FuncName == "fdatasync" || 
                  FuncName == "sync_file_range" || FuncName == "posix_fadvise" || 
                  FuncName == "posix_fadvise64" || FuncName == "madvise") {
                    
                if (flushBatch(IOBatch, F.getParent(), SE, ForceShadowBuffer)) Changed = true;
                CurrentBatchBytes = 0;
                ForceShadowBuffer = false;
                continue; 
              }
            }

            IOArgs CArgs = getIOArguments(Call);

            bool isWrite = (CArgs.Type == IOArgs::POSIX_WRITE || CArgs.Type == IOArgs::C_FWRITE || 
                            CArgs.Type == IOArgs::CXX_WRITE || CArgs.Type == IOArgs::POSIX_PWRITE || 
                            CArgs.Type == IOArgs::MPI_WRITE_AT || CArgs.Type == IOArgs::SPLICE ||
                            CArgs.Type == IOArgs::SENDFILE);
            
            bool isRead = (CArgs.Type == IOArgs::POSIX_READ || CArgs.Type == IOArgs::C_FREAD || 
                           CArgs.Type == IOArgs::POSIX_PREAD || CArgs.Type == IOArgs::MPI_READ_AT ||
                           CArgs.Type == IOArgs::CXX_READ);

            if (isWrite || isRead) {
              uint64_t CallBytes = 4096; 
              if (auto *ConstLen = dyn_cast<ConstantInt>(CArgs.Length)) {
                CallBytes = ConstLen->getZExtValue();
              }

              if (!IOBatch.empty()) {
                IOArgs BatchArgs = getIOArguments(IOBatch.front());
                bool BatchIsRead = (BatchArgs.Type == IOArgs::POSIX_READ || BatchArgs.Type == IOArgs::C_FREAD || 
                                    BatchArgs.Type == IOArgs::POSIX_PREAD || BatchArgs.Type == IOArgs::MPI_READ_AT ||
                                    BatchArgs.Type == IOArgs::CXX_READ);
                if (BatchIsRead != isRead) {
                  if (flushBatch(IOBatch, F.getParent(), SE, ForceShadowBuffer)) Changed = true;
                  CurrentBatchBytes = 0;
                  ForceShadowBuffer = false; 
                }
              }

              if (isSafeToAddToBatch(IOBatch, Call, AA, DL, SE, DT, PDT, ForceShadowBuffer)) {
                IOBatch.push_back(Call);
                CurrentBatchBytes += CallBytes;

                if (CurrentBatchBytes >= IOHighWaterMark) {
                  if (flushBatch(IOBatch, F.getParent(), SE, ForceShadowBuffer)) Changed = true;
                  CurrentBatchBytes = 0;
                  ForceShadowBuffer = false; 
                }
              } else {
                if (flushBatch(IOBatch, F.getParent(), SE, ForceShadowBuffer)) Changed = true;
                IOBatch.push_back(Call);
                CurrentBatchBytes = CallBytes; 
                ForceShadowBuffer = false; 
              }
            }
          }
        }
      }
      
      if (flushBatch(IOBatch, F.getParent(), SE, ForceShadowBuffer)) Changed = true;

      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    } 
     
  };
}

struct IOWrapperInlinePass : public PassInfoMixin<IOWrapperInlinePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    bool Changed = false;
        
    for (Function &F : M) {
      if (F.isDeclaration()) continue;

      unsigned InstCount = 0;
      bool HasIO = false;

      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          InstCount++;
          if (auto *Call = dyn_cast<CallInst>(&I)) {
            if (Function *Callee = Call->getCalledFunction()) {
              StringRef Name = Callee->getName();
              if (Name == "write" || Name == "writev" || Name == "write64" ||
                  Name == "read" || Name == "readv" || Name == "read64" ||
                  Name == "fwrite" || Name == "fread" ||
                  Name == "pwrite" || Name == "pread" || 
                  Name == "pwrite64" || Name == "pread64" || 
                  Name == "splice" || Name == "sendfile" || Name == "sendfile64" ||
                  Name == "preadv" || Name == "pwritev" || Name == "io_submit" ||
                  Name == "MPI_File_write_at" || Name == "PMPI_File_write_at" ||
                  Name == "MPI_File_read_at" || Name == "PMPI_File_read_at") {
                HasIO = true;
              }
            }
          }
        }
      }

      if (HasIO && InstCount < 10 && !F.hasFnAttribute(Attribute::NoInline)) {
        F.addFnAttr(Attribute::AlwaysInline);
        Changed = true;
      }
    }
        
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "IOOpt", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      
      PB.registerPipelineParsingCallback(
                     [](StringRef Name, FunctionPassManager &FPM,
                        ArrayRef<PassBuilder::PipelineElement>) {
                       if (Name == "io-opt") {
                         FPM.addPass(IOOptimisationPass()); 
                         return true;
                       }
                       return false;
                     });
      
      PB.registerPipelineStartEPCallback(
                     [](ModulePassManager &MPM, OptimizationLevel Level) {
                       MPM.addPass(IOWrapperInlinePass());
                     });
      
      PB.registerOptimizerLastEPCallback(
                     [](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase Phase) {
                       MPM.addPass(createModuleToFunctionPassAdaptor(IOOptimisationPass())); 
                     });
      
      PB.registerFullLinkTimeOptimizationLastEPCallback(
                            [](ModulePassManager &MPM, OptimizationLevel Level) {
                               MPM.addPass(createModuleToFunctionPassAdaptor(IOOptimisationPass())); 
                            });
    }};
}
