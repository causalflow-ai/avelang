#pragma once

#include "AST/ast_nodes_expr.h"
#include "generator_context.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <utility>

namespace causalflow::avelang::ir {

/// Layout algebra helper functions
class LayoutOperation {
  public:
    using StaticLayout =
        std::pair<llvm::SmallVector<int64_t>, llvm::SmallVector<int64_t>>;

    static long extractIndexValue(mlir::Value value);

    static bool flattenTupleValues(mlir::Value tupleValue,
                                   llvm::SmallVector<int64_t> &values);

    static int64_t computeShapeProduct(llvm::ArrayRef<int64_t> shape);

    static int64_t computeCoSize(llvm::ArrayRef<int64_t> shape,
                                 llvm::ArrayRef<int64_t> strides);

    static StaticLayout coalesceStaticLayout(
        const llvm::SmallVector<int64_t> &shape,
        const llvm::SmallVector<int64_t> &strides);

    static StaticLayout computeCompositionLayout(
        const llvm::SmallVector<int64_t> &shapeA,
        const llvm::SmallVector<int64_t> &stridesA,
        const llvm::SmallVector<int64_t> &shapeB,
        const llvm::SmallVector<int64_t> &stridesB);

    static StaticLayout computeComplementLayout(
        const llvm::SmallVector<int64_t> &shape,
        const llvm::SmallVector<int64_t> &strides, int64_t size);

    static StaticLayout concatenateStaticLayouts(
        const llvm::SmallVector<int64_t> &shapeA,
        const llvm::SmallVector<int64_t> &stridesA,
        const llvm::SmallVector<int64_t> &shapeB,
        const llvm::SmallVector<int64_t> &stridesB);

    static bool ExpandIndicesForNestedLayout(
        mlir::OpBuilder &builder, mlir::Value memrefValue,
        llvm::SmallVector<mlir::Value> &indices);

    static mlir::Value
    createViewFunction(ast::Call *callExpr, GeneratorContext *ctx,
                       llvm::ArrayRef<mlir::Value> resolvedArgs);

    static mlir::Value
    createMakeLayoutFunction(ast::Call *callExpr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolvedArgs);

  private:
    static mlir::Value CastToIndex(mlir::OpBuilder &builder, mlir::Location loc,
                                   mlir::Value value);

    static void CollectTupleLeaves(mlir::Value value,
                                   llvm::SmallVector<mlir::Value> &out);

    static void ExpandLinearIndex(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value linearIndex,
                                  llvm::ArrayRef<mlir::Value> dims,
                                  llvm::SmallVector<mlir::Value> &out);
};

} // namespace causalflow::avelang::ir
