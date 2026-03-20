#include "IODialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::io;

#include "IODialectDialect.cpp.inc"

#define GET_OP_CLASSES
#include "IOOps.cpp.inc"

void mlir::io::IODialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IOOps.cpp.inc"
      >();
}
