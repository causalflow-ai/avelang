#pragma once

#include "AST/ast_nodes_expr.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Value.h>
#pragma clang diagnostic pop

#include <optional>

namespace causalflow::avelang::ir {

struct TypeInfo {
    std::optional<bool> is_unsigned_integer;
};

bool IsIntegerValueOrVectorOfIntegers(mlir::Type type);
bool IsMemRefWithIntegerElements(mlir::Type type);

TypeInfo GetTypeInfo(mlir::Value value);
TypeInfo GetTypeInfo(ast::Expr *expr);
TypeInfo GetFunctionResultTypeInfo(mlir::func::FuncOp function,
                                   unsigned resultIndex);
void SetTypeInfo(mlir::Value value, const TypeInfo &typeInfo);
void SetFunctionResultTypeInfo(mlir::func::FuncOp function,
                               unsigned resultIndex, const TypeInfo &typeInfo);

} // namespace causalflow::avelang::ir
