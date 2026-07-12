#include "mangler.h"

#include "AST/ast_nodes_expr.h"
#include "AST/ast_nodes_stmt.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "constant_folder.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#pragma clang diagnostic pop

#include <cctype>
#include <optional>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

namespace causalflow::avelang::ir {

namespace {

namespace cf = causalflow::avelang::dialect;

std::string SanitizeManglePart(llvm::StringRef value) {
    std::string result;
    result.reserve(value.size());
    for (char c : value) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc)) {
            result.push_back(c);
        } else if (c == '\'') {
            result.append("_sq_");
        } else if (c == '-') {
            result.push_back('m');
        } else {
            result.push_back('_');
        }
    }
    return result;
}

std::string MangleAddressSpace(mlir::Attribute memorySpace) {
    if (!memorySpace) {
        return "default";
    }
    if (auto gpuSpace =
            mlir::dyn_cast<mlir::gpu::AddressSpaceAttr>(memorySpace)) {
        switch (gpuSpace.getValue()) {
        case mlir::gpu::AddressSpace::Global:
            return "global";
        case mlir::gpu::AddressSpace::Workgroup:
            return "workgroup";
        case mlir::gpu::AddressSpace::Private:
            return "private";
        default:
            break;
        }
        return "as" + std::to_string(static_cast<int>(gpuSpace.getValue()));
    }
    if (auto intSpace = mlir::dyn_cast<mlir::IntegerAttr>(memorySpace)) {
        return "as" + std::to_string(intSpace.getInt());
    }
    return "unknown";
}

std::string MangleType(mlir::Type type) {
    if (!type) {
        return "unknown";
    }
    if (type.isIndex()) {
        return "index";
    }
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
        return "i" + std::to_string(intType.getWidth());
    }
    if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
        return "f" + std::to_string(floatType.getWidth());
    }
    if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(type)) {
        std::string result = "vec";
        for (auto dim : vectorType.getShape()) {
            if (result.size() > 3) {
                result.push_back('x');
            }
            result.append(std::to_string(dim));
        }
        result.push_back('x');
        result.append(MangleType(vectorType.getElementType()));
        return result;
    }
    if (auto memrefType = mlir::dyn_cast<cf::MemRefType>(type)) {
        std::string result = "memref";
        result.append(std::to_string(memrefType.getRank()));
        result.append("d_");
        result.append(MangleType(memrefType.getElementType()));
        return result;
    }
    if (auto functionType = mlir::dyn_cast<mlir::FunctionType>(type)) {
        std::string result = "func";
        for (auto input : functionType.getInputs()) {
            result.push_back('_');
            result.append(MangleType(input));
        }
        return result;
    }

    std::string storage;
    llvm::raw_string_ostream os(storage);
    type.print(os);
    return SanitizeManglePart(os.str());
}

std::optional<std::string> MangleConstexprValueTag(mlir::Value value) {
    if (!value) {
        return std::nullopt;
    }

    std::string type = MangleType(value.getType());
    if (value.getType().isInteger(1)) {
        if (auto boolValue = ConstantFolder::FoldBoolValue(value)) {
            return type + "_" + (*boolValue ? "1" : "0");
        }
    }
    if (value.getType().isIntOrIndex()) {
        if (auto intValue = ConstantFolder::FoldIntValue(value)) {
            return type + "_" + SanitizeManglePart(std::to_string(*intValue));
        }
    }
    if (auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto floatAttr =
                mlir::dyn_cast<mlir::FloatAttr>(constOp.getValue())) {
            std::string storage;
            llvm::raw_string_ostream os(storage);
            floatAttr.print(os);
            return type + "_" + SanitizeManglePart(os.str());
        }
    }

    std::string storage;
    llvm::raw_string_ostream os(storage);
    if (auto *op = value.getDefiningOp()) {
        op->print(os);
    } else {
        value.print(os);
    }
    return type + "_expr_" + SanitizeManglePart(os.str());
}

template <typename T>
std::optional<T>
FindBinding(llvm::ArrayRef<std::pair<std::string, T>> bindings,
            llvm::StringRef name) {
    for (const auto &[bindingName, value] : bindings) {
        if (bindingName == name) {
            return value;
        }
    }
    return std::nullopt;
}

bool IsConstexprArg(ast::Arg *arg) {
    auto *attrExpr =
        llvm::dyn_cast_or_null<ast::AttributeExpr>(arg->GetAnnotation());
    return attrExpr && attrExpr->GetAttr() == "constexpr";
}

std::string BuildMangledName(llvm::ArrayRef<std::string> scope,
                             llvm::StringRef name,
                             llvm::ArrayRef<std::string> addressSpaceTags,
                             llvm::ArrayRef<std::string> constexprTags) {
    if (name.empty()) {
        return {};
    }

    size_t totalSize = name.size();
    for (const auto &part : scope) {
        totalSize += part.size() + 1;
    }
    if (!addressSpaceTags.empty()) {
        totalSize += 4; // "__as"
        for (const auto &tag : addressSpaceTags) {
            totalSize += tag.size() + 1;
        }
    }
    if (!constexprTags.empty()) {
        totalSize += 4; // "__ce"
        for (const auto &tag : constexprTags) {
            totalSize += tag.size() + 1;
        }
    }

    std::string mangled;
    mangled.reserve(totalSize);
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
    if (!addressSpaceTags.empty()) {
        mangled.append("__as");
        for (const auto &tag : addressSpaceTags) {
            mangled.push_back('_');
            mangled.append(tag);
        }
    }
    if (!constexprTags.empty()) {
        mangled.append("__ce");
        for (const auto &tag : constexprTags) {
            mangled.push_back('_');
            mangled.append(tag);
        }
    }
    return mangled;
}

} // namespace

std::string MangleFunctionName(
    ast::FunctionDef *func, llvm::ArrayRef<std::string> scope,
    llvm::ArrayRef<std::pair<std::string, mlir::Attribute>> addressSpaces,
    llvm::ArrayRef<std::pair<std::string, mlir::Value>> constexprValues) {
    if (!func) {
        return {};
    }

    llvm::SmallVector<std::string, 4> addressSpaceTags;
    llvm::SmallVector<std::string, 4> constexprTags;
    if (auto *args = func->GetArguments()) {
        for (auto *arg : args->GetArgs()) {
            if (!arg) {
                continue;
            }
            const auto &argName = arg->GetArgName();
            if (IsConstexprArg(arg)) {
                auto value = FindBinding(constexprValues, argName);
                if (!value) {
                    continue;
                }
                auto tag = MangleConstexprValueTag(*value);
                if (tag) {
                    constexprTags.push_back(SanitizeManglePart(argName) + "_" +
                                            *tag);
                }
                continue;
            }

            auto addressSpace = FindBinding(addressSpaces, argName);
            if (addressSpace) {
                addressSpaceTags.push_back(MangleAddressSpace(*addressSpace));
            }
        }
    }

    return BuildMangledName(scope, func->GetName(), addressSpaceTags,
                            constexprTags);
}

} // namespace causalflow::avelang::ir
