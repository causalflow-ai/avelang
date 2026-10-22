#include "mangler.h"

#include "AST/ast_nodes_expr.h"
#include "AST/ast_nodes_stmt.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "type_system.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#pragma clang diagnostic pop

#include <cctype>
#include <optional>
#include <sstream>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
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

    auto *definingOp = value.getDefiningOp();
    if (!definingOp || !definingOp->hasTrait<mlir::OpTrait::ConstantLike>()) {
        return std::nullopt;
    }
    auto attribute = definingOp->getAttrOfType<mlir::TypedAttr>("value");
    if (!attribute) {
        return std::nullopt;
    }

    std::string type = MangleType(value.getType());
    if (auto intValue = mlir::dyn_cast<mlir::IntegerAttr>(attribute)) {
        if (value.getType().isInteger(1)) {
            return type + "_" + (intValue.getValue().isZero() ? "0" : "1");
        }
        llvm::SmallString<32> storage;
        bool isUnsigned =
            GetTypeInfo(value).is_unsigned_integer.value_or(false);
        intValue.getValue().toString(storage, 10, !isUnsigned);
        return type + "_" + (isUnsigned ? "u" : "s") + "_" +
               SanitizeManglePart(storage);
    }
    if (auto floatValue = mlir::dyn_cast<mlir::FloatAttr>(attribute)) {
        std::string storage;
        llvm::raw_string_ostream os(storage);
        floatValue.print(os);
        return type + "_" + SanitizeManglePart(os.str());
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

    std::stringstream mangled;
    bool hasPart = false;
    for (const auto &part : scope) {
        if (hasPart) {
            mangled << '_';
        }
        mangled << part;
        hasPart = true;
    }
    if (hasPart) {
        mangled << '_';
    }
    mangled << name.str();
    if (!addressSpaceTags.empty()) {
        mangled << "__as";
        for (const auto &tag : addressSpaceTags) {
            mangled << '_' << tag;
        }
    }
    if (!constexprTags.empty()) {
        mangled << "__ce";
        for (const auto &tag : constexprTags) {
            mangled << '_' << tag;
        }
    }
    return mangled.str();
}

} // namespace

std::string MangleFunctionName(ast::FunctionDef *func,
                               llvm::ArrayRef<std::string> scope,
                               llvm::ArrayRef<mlir::Value> arguments) {
    if (!func) {
        return {};
    }

    llvm::SmallVector<std::string, 4> addressSpaceTags;
    llvm::SmallVector<std::string, 4> constexprTags;
    if (auto *args = func->GetArguments()) {
        const auto &parameters = args->GetArgs();
        if (parameters.size() != arguments.size()) {
            return {};
        }
        for (auto [index, arg] : llvm::enumerate(parameters)) {
            if (!arg) {
                continue;
            }
            auto value = arguments[index];
            const auto &argName = arg->GetArgName();
            if (IsConstexprArg(arg)) {
                if (!value) {
                    return {};
                }
                auto tag = MangleConstexprValueTag(value);
                if (!tag) {
                    return {};
                }
                constexprTags.push_back(SanitizeManglePart(argName) + "_" +
                                        *tag);
                continue;
            }

            if (value) {
                if (auto memrefType =
                        mlir::dyn_cast<cf::MemRefType>(value.getType())) {
                    addressSpaceTags.push_back(
                        MangleAddressSpace(memrefType.getMemorySpace()));
                }
            }
        }
    }

    return BuildMangledName(scope, func->GetName(), addressSpaceTags,
                            constexprTags);
}

} // namespace causalflow::avelang::ir
