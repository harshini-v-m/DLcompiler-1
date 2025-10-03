#include "Compiler/Dialect/nova/NovaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "Compiler/Dialect/nova/TypeInference.h"
#include "mlir/IR/Diagnostics.h" 
using namespace mlir;
using namespace mlir::nova;

#define GET_OP_CLASSES
#include "Compiler/Dialect/nova/NovaOps.cpp.inc"


//----------------------add op infer return type---------------------
LogicalResult AddOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {

  // Call the general helper and provide `computeBroadcastShape` as the logic
    auto result = TypeInference::inferBinaryOp(operands[0].getType(), operands[1].getType());
    if (!result.success) {
    if (loc) {
      mlir::emitError(*loc) << result.errorMessage;
    }
    return failure();
  }
  
  inferredReturnTypes.push_back(result.inferredType);
  return success();
}


//----------------------sub op infer return type---------------------
LogicalResult SubOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    llvm::SmallVectorImpl<Type> &inferredReturnTypes) {

   // Call the general helper and provide `computeBroadcastShape` as the logic
    auto result = TypeInference::inferBinaryOp(operands[0].getType(), operands[1].getType());
    if (!result.success) {
    if (loc) {
      mlir::emitError(*loc) << result.errorMessage;
    }
    return failure();
  }
  
  inferredReturnTypes.push_back(result.inferredType);
  return success();
}
//---------------------negation-----------------------
LogicalResult NegOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> loc,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  
  if (operands.empty()) 
    return failure();
  
  // Still more efficient than original, but uses inferUnaryOp
  if (auto result = TypeInference::inferUnaryOp(operands.front().getType());
      result.success) {
    inferredReturnTypes.push_back(result.inferredType);
    return success();
  }
  
  return failure();
}
