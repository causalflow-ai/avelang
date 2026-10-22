#pragma once

#include <string>

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Value.h>

namespace causalflow::avelang::ast {
class FunctionDef;
} // namespace causalflow::avelang::ast

namespace causalflow::avelang::ir {

std::string MangleFunctionName(ast::FunctionDef *func,
                               llvm::ArrayRef<std::string> scope = {},
                               llvm::ArrayRef<mlir::Value> arguments = {});

} // namespace causalflow::avelang::ir
