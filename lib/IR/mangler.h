#pragma once

#include <string>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>

namespace causalflow::avelang::ir {

std::string MangleFunctionName(llvm::ArrayRef<std::string> scope,
                               llvm::StringRef name);
std::string MangleFunctionName(llvm::ArrayRef<std::string> scope,
                               llvm::StringRef name,
                               llvm::ArrayRef<std::string> address_space_tags);
std::string MangleFunctionName(llvm::ArrayRef<std::string> scope,
                               llvm::StringRef name,
                               llvm::ArrayRef<std::string> address_space_tags,
                               llvm::ArrayRef<std::string> constexpr_tags);

} // namespace causalflow::avelang::ir
