#include "mangler.h"

namespace causalflow::avelang::ir {

std::string MangleFunctionName(llvm::ArrayRef<std::string> scope,
                               llvm::StringRef name,
                               llvm::ArrayRef<std::string> address_space_tags) {
    return MangleFunctionName(scope, name, address_space_tags, {});
}

std::string MangleFunctionName(llvm::ArrayRef<std::string> scope,
                               llvm::StringRef name,
                               llvm::ArrayRef<std::string> address_space_tags,
                               llvm::ArrayRef<std::string> constexpr_tags) {
    if (name.empty()) {
        return {};
    }
    if (scope.empty()) {
        std::string base = name.str();
        if (address_space_tags.empty() && constexpr_tags.empty()) {
            return base;
        }
        if (!address_space_tags.empty()) {
            base.append("__as");
            for (const auto &tag : address_space_tags) {
                base.push_back('_');
                base.append(tag);
            }
        }
        if (!constexpr_tags.empty()) {
            base.append("__ce");
            for (const auto &tag : constexpr_tags) {
                base.push_back('_');
                base.append(tag);
            }
        }
        return base;
    }
    size_t total_size = name.size();
    for (const auto &part : scope) {
        total_size += part.size() + 1;
    }
    if (!address_space_tags.empty()) {
        total_size += 4; // "__as"
        for (const auto &tag : address_space_tags) {
            total_size += tag.size() + 1;
        }
    }
    if (!constexpr_tags.empty()) {
        total_size += 4; // "__ce"
        for (const auto &tag : constexpr_tags) {
            total_size += tag.size() + 1;
        }
    }
    std::string mangled;
    mangled.reserve(total_size);
    for (const auto &part : scope) {
        if (!mangled.empty()) {
            mangled.push_back('_');
        }
        mangled.append(part);
    }
    if (!mangled.empty()) {
        mangled.push_back('_');
    }
    mangled.append(name.data(), name.size());
    if (!address_space_tags.empty()) {
        mangled.append("__as");
        for (const auto &tag : address_space_tags) {
            mangled.push_back('_');
            mangled.append(tag);
        }
    }
    if (!constexpr_tags.empty()) {
        mangled.append("__ce");
        for (const auto &tag : constexpr_tags) {
            mangled.push_back('_');
            mangled.append(tag);
        }
    }
    return mangled;
}

std::string MangleFunctionName(llvm::ArrayRef<std::string> scope,
                               llvm::StringRef name) {
    return MangleFunctionName(scope, name, {});
}

} // namespace causalflow::avelang::ir
