#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"

using namespace llvm;

namespace {

// -----------------------------------------------------------------------------
// Helper: Alias Analysis & Interference Logic
// -----------------------------------------------------------------------------
bool isSafeToAmalgamate(CallInst *Call1, CallInst *Call2, AAResults &AA, const DataLayout &DL) {
    if (Call1->getCalledFunction() != Call2->getCalledFunction()) return false;
    
    Value *FilePtr1 = Call1->getArgOperand(3);
    Value *FilePtr2 = Call2->getArgOperand(3);
    if (FilePtr1 != FilePtr2) return false;

    Value *Buf1 = Call1->getArgOperand(0);
    Value *Buf2 = Call2->getArgOperand(0);
    
    LocationSize Size1 = LocationSize::precise(DL.getTypeStoreSize(Buf1->getType()));
    
    // TODO: Add adjacency check (Buf2 == Buf1 + Size1) here

    MemoryLocation Loc1(Buf1, Size1);
    MemoryLocation Loc2(Buf2, LocationSize::precise(DL.getTypeStoreSize(Buf2->getType())));

    for (Instruction *I = Call1->getNextNode(); I != Call2; I = I->getNextNode()) {
        if (!I) return false; 
        if (!I->mayReadOrWriteMemory()) continue;

        ModRefInfo MR1 = AA.getModRefInfo(I, Loc1);
        if (isModOrRefSet(MR1)) return false;

        ModRefInfo MR2 = AA.getModRefInfo(I, Loc2);
        if (isModOrRefSet(MR2)) return false;
    }

    return true; 
}

// -----------------------------------------------------------------------------
// The Main Pass
// -----------------------------------------------------------------------------
struct IOOptimizationPass : public PassInfoMixin<IOOptimizationPass> {
    
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
        bool Changed = false;
        
        // Extract Analysis Results
        AAResults &AA = FAM.getResult<AAManager>(F);
        const DataLayout &DL = F.getParent()->getDataLayout();

        // Basic block and instruction iteration goes here...
        // (In a real implementation, you would store found I/O calls in a list 
        // and then compare them using isSafeToAmalgamate)

        errs() << "Running IOOptimizationPass on: " << F.getName() << "\n";

        return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
};

} // end anonymous namespace

// -----------------------------------------------------------------------------
// Pass Plugin Registration Boilerplate
// -----------------------------------------------------------------------------
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "IOOptimizationPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "io-opt") {
                        FPM.addPass(IOOptimizationPass());
                        return true;
                    }
                    return false;
                });
        }};
}
