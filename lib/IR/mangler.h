#pragma once

#include <string>
#include <utility>

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>

namespace causalflow::avelang::ast {
class FunctionDef;
} // namespace causalflow::avelang::ast

namespace causalflow::avelang::ir {

std::string MangleFunctionName(
    ast::FunctionDef *func, llvm::ArrayRef<std::string> scope = {},
    llvm::ArrayRef<std::pair<std::string, mlir::Attribute>> address_spaces = {},
    llvm::ArrayRef<std::pair<std::string, mlir::Value>> constexpr_values = {});

} // namespace causalflow::avelang::ir
