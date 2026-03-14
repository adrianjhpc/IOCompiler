#include "IODialect.h"

using namespace mlir;
using namespace mlir::io; // Replace 'io' with whatever namespace is in your .td file

// 1. Include the generated Dialect definitions
#include "IODialectDialect.cpp.inc"

// 2. Initialize the dialect (Registering all your custom ops)
void IODialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IODialect.cpp.inc"
      >();
}

// 3. Include the generated Operation definitions
#define GET_OP_CLASSES
#include "IODialect.cpp.inc"
