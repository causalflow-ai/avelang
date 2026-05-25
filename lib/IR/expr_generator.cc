#include "Utils/assert.h"
#include "constant_folder.h"
#include "layout_operation.h"
#include "mlir_generator_impl.h"
#include "parsing_utils.h"
#include "type_promotion.h"
#include "type_system.h"

#include "Dialect/AveLang/IR/AveLangOps.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#pragma clang diagnostic pop

#include <cctype>
#include <functional>
#include <optional>
#include <string_view>
#include <unordered_map>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>

namespace causalflow::avelang::ir {

namespace cf = causalflow::avelang::dialect;

static cf::MemRefType
createAveLangMemRefType(mlir::MLIRContext *context,
                        llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                        mlir::Attribute memorySpace = {},
                        llvm::ArrayRef<int64_t> strides = {});

namespace {

static clang::DiagnosticBuilder
Report(GeneratorContext *ctx, basic::DiagnosticCode code,
       clang::SourceLocation loc = clang::SourceLocation()) {
    return ctx->diagnostic_manager->Report(code, loc);
}

static clang::DiagnosticBuilder
Report(ExprGenerator *gen, basic::DiagnosticCode code,
       clang::SourceLocation loc = clang::SourceLocation()) {
    return Report(gen->GetParent()->GetContext(), code, loc);
}

static std::optional<int64_t>
GetStaticElementCount(const cf::MemRefType &memrefType) {
    int64_t elementCount = 1;
    for (int64_t dim : memrefType.getShape()) {
        if (dim == mlir::ShapedType::kDynamic) {
            return std::nullopt;
        }
        elementCount *= dim;
    }
    return elementCount;
}

static std::optional<int64_t> GetStaticByteSize(mlir::Type type) {
    if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(type)) {
        auto elementBytes = GetStaticByteSize(vectorType.getElementType());
        if (!elementBytes) {
            return std::nullopt;
        }
        return *elementBytes * vectorType.getNumElements();
    }

    if (type.isIndex()) {
        return 8;
    }

    if (!type.isIntOrFloat()) {
        return std::nullopt;
    }

    int64_t bitWidth = type.getIntOrFloatBitWidth();
    int64_t elementBytes = (bitWidth + 7) / 8;
    if (elementBytes <= 0) {
        return std::nullopt;
    }
    return elementBytes;
}

static std::optional<int64_t>
GetStaticByteSize(const cf::MemRefType &memrefType) {
    auto elementCount = GetStaticElementCount(memrefType);
    auto elementBytes = GetStaticByteSize(memrefType.getElementType());
    if (!elementCount || !elementBytes) {
        return std::nullopt;
    }
    return *elementCount * *elementBytes;
}

static std::optional<int64_t> GetConstantIndexLikeValue(mlir::Value value) {
    if (!value) {
        return std::nullopt;
    }

    if (auto constIndex = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        return constIndex.value();
    }

    if (auto indexCast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
        return GetConstantIndexLikeValue(indexCast.getIn());
    }

    if (auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr =
                mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            return intAttr.getValue().getSExtValue();
        }
    }

    return std::nullopt;
}

static mlir::Value LoadScalarFromMemref(mlir::OpBuilder &builder,
                                        GeneratorContext *ctx, ast::Call *call,
                                        mlir::Value memrefVal,
                                        ast::Expr *arg_expr,
                                        llvm::StringRef contextName) {
    auto memrefType = mlir::dyn_cast<cf::MemRefType>(memrefVal.getType());
    if (!memrefType) {
        return memrefVal;
    }

    auto *diagnostics = ctx->diagnostic_manager.get();
    mlir::Location loc = ctx->GetMLIRLocation(builder.getContext(), arg_expr);

    if (mlir::isa<mlir::VectorType>(memrefType.getElementType())) {
        if (diagnostics) {
            auto source_loc = call->GetSourceRange().getBegin();
            if (arg_expr) {
                source_loc = arg_expr->GetSourceRange().getBegin();
            }
            Report(ctx, basic::DiagnosticCode::kUnimplemented, source_loc)
                << contextName << " argument must have a scalar element type";
        }
        return nullptr;
    }

    auto elementCountOpt = GetStaticElementCount(memrefType);
    if (!elementCountOpt) {
        if (diagnostics) {
            auto source_loc = call->GetSourceRange().getBegin();
            if (arg_expr) {
                source_loc = arg_expr->GetSourceRange().getBegin();
            }
            Report(ctx, basic::DiagnosticCode::kUnimplemented, source_loc)
                << contextName << " argument must have static memref shape";
        }
        return nullptr;
    }

    if (*elementCountOpt != 1) {
        if (diagnostics) {
            auto source_loc = call->GetSourceRange().getBegin();
            if (arg_expr) {
                source_loc = arg_expr->GetSourceRange().getBegin();
            }
            Report(ctx, basic::DiagnosticCode::kUnimplemented, source_loc)
                << contextName << " argument must have a single element";
        }
        return nullptr;
    }

    llvm::SmallVector<mlir::Value> indices;
    indices.reserve(memrefType.getRank());
    for (int64_t i = 0; i < memrefType.getRank(); ++i) {
        indices.push_back(
            mlir::arith::ConstantIndexOp::create(builder, loc, 0));
    }

    auto elemType = memrefType.getElementType();
    auto load = cf::AveLangMemRefLoadOp::create(builder, loc, elemType,
                                                memrefVal, indices);
    SetTypeInfo(load.getResult(), GetTypeInfo(memrefVal));
    return load.getResult();
}

static mlir::Value LoadVectorFromMemref(mlir::OpBuilder &builder,
                                        GeneratorContext *ctx, ast::Call *call,
                                        mlir::Value memrefVal,
                                        ast::Expr *arg_expr,
                                        llvm::StringRef contextName) {
    auto memrefType = mlir::dyn_cast<cf::MemRefType>(memrefVal.getType());
    if (!memrefType) {
        return memrefVal;
    }

    auto *diagnostics = ctx->diagnostic_manager.get();
    mlir::Location loc = ctx->GetMLIRLocation(builder.getContext(), arg_expr);

    auto elementCountOpt = GetStaticElementCount(memrefType);
    if (!elementCountOpt) {
        if (diagnostics) {
            auto source_loc = call->GetSourceRange().getBegin();
            if (arg_expr) {
                source_loc = arg_expr->GetSourceRange().getBegin();
            }
            Report(ctx, basic::DiagnosticCode::kUnimplemented, source_loc)
                << contextName << " arguments must have static memref shapes";
        }
        return nullptr;
    }
    int64_t elementCount = *elementCountOpt;

    llvm::SmallVector<mlir::Value> indices;
    indices.reserve(memrefType.getRank());
    for (int64_t i = 0; i < memrefType.getRank(); ++i) {
        indices.push_back(
            mlir::arith::ConstantIndexOp::create(builder, loc, 0));
    }

    auto elemType = memrefType.getElementType();
    if (auto vecElemType = mlir::dyn_cast<mlir::VectorType>(elemType)) {
        auto load = cf::AveLangMemRefLoadOp::create(builder, loc, vecElemType,
                                                    memrefVal, indices);
        SetTypeInfo(load.getResult(), GetTypeInfo(memrefVal));
        return load.getResult();
    }

    auto loadVecType = mlir::VectorType::get(memrefType.getShape(), elemType);
    auto load = cf::AveLangMemRefLoadVecOp::create(builder, loc, loadVecType,
                                                   memrefVal, indices);
    if (loadVecType.getRank() == 1) {
        SetTypeInfo(load.getResult(), GetTypeInfo(memrefVal));
        return load.getResult();
    }

    auto flatType = mlir::VectorType::get(elementCount, elemType);
    auto flat = mlir::vector::ShapeCastOp::create(builder, loc, flatType,
                                                  load.getResult());
    SetTypeInfo(flat.getResult(), GetTypeInfo(load.getResult()));
    return flat.getResult();
}

static std::optional<int> ParseIntAfterMarker(const std::string &name,
                                              std::string_view marker) {
    auto pos = name.find(marker);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos += marker.size();
    if (pos >= name.size() ||
        !std::isdigit(static_cast<unsigned char>(name[pos]))) {
        return std::nullopt;
    }
    int value = 0;
    while (pos < name.size() &&
           std::isdigit(static_cast<unsigned char>(name[pos]))) {
        value = value * 10 + (name[pos] - '0');
        ++pos;
    }
    return value;
}

bool IsGpuMmaIntrinsic(const SymbolScope::Function &function) {
    if (!function.IsValid()) {
        return false;
    }
    const std::string &module_name = function.module_name;
    if (module_name != "nvvm" && module_name != "amdgpu") {
        return false;
    }
    const std::string &symbol_name = function.symbol_name;
    return symbol_name.rfind("mma_", 0) == 0 ||
           symbol_name.rfind("mfma_", 0) == 0;
}

mlir::Value LoadVectorFromMemrefForMma(mlir::OpBuilder &builder,
                                       GeneratorContext *ctx, ast::Call *call,
                                       mlir::Value memrefVal,
                                       ast::Expr *arg_expr) {
    auto memrefType = mlir::dyn_cast<cf::MemRefType>(memrefVal.getType());
    if (!memrefType) {
        return memrefVal;
    }

    auto *diagnostics = ctx->diagnostic_manager.get();
    mlir::Location loc = ctx->GetMLIRLocation(builder.getContext(), arg_expr);

    auto reportDynamicShape = [&]() {
        if (diagnostics) {
            auto source_loc = call->GetSourceRange().getBegin();
            if (arg_expr) {
                source_loc = arg_expr->GetSourceRange().getBegin();
            }
            Report(ctx, basic::DiagnosticCode::kUnimplemented, source_loc)
                << "mma/mfma arguments must have static memref shapes";
        }
    };

    auto elementCountOpt = GetStaticElementCount(memrefType);
    if (!elementCountOpt) {
        reportDynamicShape();
        return nullptr;
    }
    int64_t elementCount = *elementCountOpt;

    llvm::SmallVector<mlir::Value> indices;
    indices.reserve(memrefType.getRank());
    for (int64_t i = 0; i < memrefType.getRank(); ++i) {
        indices.push_back(
            mlir::arith::ConstantIndexOp::create(builder, loc, 0));
    }

    auto elemType = memrefType.getElementType();
    if (auto vecElemType = mlir::dyn_cast<mlir::VectorType>(elemType)) {
        auto load = cf::AveLangMemRefLoadOp::create(builder, loc, vecElemType,
                                                    memrefVal, indices);
        return load.getResult();
    }

    auto loadVecType = mlir::VectorType::get(memrefType.getShape(), elemType);
    auto load = cf::AveLangMemRefLoadVecOp::create(builder, loc, loadVecType,
                                                   memrefVal, indices);
    if (loadVecType.getRank() == 1) {
        return load.getResult();
    }

    auto flatType = mlir::VectorType::get(elementCount, elemType);
    return mlir::vector::ShapeCastOp::create(builder, loc, flatType,
                                             load.getResult())
        .getResult();
}

} // namespace

static std::string GetTypeName(mlir::Type type) {
    std::string storage;
    llvm::raw_string_ostream os(storage);
    type.print(os);
    return os.str();
}

static cf::MemRefType createAveLangMemRefType(mlir::MLIRContext *context,
                                              llvm::ArrayRef<int64_t> shape,
                                              mlir::Type elementType,
                                              mlir::Attribute memorySpace,
                                              llvm::ArrayRef<int64_t> strides) {
    auto layoutType = cf::LayoutType::get(context, shape, strides);
    return cf::MemRefType::get(context, layoutType, elementType, memorySpace);
}

// Binary operation factory functions
using BinaryOpFactory = std::function<mlir::Value(
    mlir::OpBuilder &, mlir::Location, mlir::Value, mlir::Value)>;

// Macro to simplify binary operation definition
#define DEFINE_BINARY_OP(name, op_class)                                       \
    {                                                                          \
        name, [](mlir::OpBuilder &builder, mlir::Location loc,                 \
                 mlir::Value lhs, mlir::Value rhs) {                           \
            return op_class::create(builder, loc, lhs, rhs);                   \
        }                                                                      \
    }

// Create maps for float operations

static const std::unordered_map<std::string, BinaryOpFactory> kFloatOps = {
    DEFINE_BINARY_OP("Add", mlir::arith::AddFOp),
    DEFINE_BINARY_OP("Sub", mlir::arith::SubFOp),
    DEFINE_BINARY_OP("Mult", mlir::arith::MulFOp),
    DEFINE_BINARY_OP("Div", mlir::arith::DivFOp),
    DEFINE_BINARY_OP("Mod", mlir::arith::RemFOp)};

// Bitwise operations (integer only)
static const std::unordered_map<std::string, BinaryOpFactory> kBitwiseOps = {
    DEFINE_BINARY_OP("BitAnd", mlir::arith::AndIOp),
    DEFINE_BINARY_OP("BitOr", mlir::arith::OrIOp),
    DEFINE_BINARY_OP("BitXor", mlir::arith::XOrIOp),
    DEFINE_BINARY_OP("LShift", mlir::arith::ShLIOp),
    DEFINE_BINARY_OP("RShift", mlir::arith::ShRSIOp)};

// Enhanced integer operations with alternative floor division implementation
static const std::unordered_map<std::string, BinaryOpFactory> kIntOps = {
    DEFINE_BINARY_OP("Add", mlir::arith::AddIOp),
    DEFINE_BINARY_OP("Sub", mlir::arith::SubIOp),
    DEFINE_BINARY_OP("Mult", mlir::arith::MulIOp),
    DEFINE_BINARY_OP("Div", mlir::arith::DivSIOp),
    DEFINE_BINARY_OP("Mod", mlir::arith::RemSIOp),
    DEFINE_BINARY_OP("UDiv", mlir::arith::DivUIOp),
    DEFINE_BINARY_OP("URShift", mlir::arith::ShRUIOp)};

// Comparison predicates for integer operations
static const std::unordered_map<std::string, mlir::arith::CmpIPredicate>
    kIntCompareOps = {{"Gt", mlir::arith::CmpIPredicate::sgt},
                      {"Lt", mlir::arith::CmpIPredicate::slt},
                      {"GtE", mlir::arith::CmpIPredicate::sge},
                      {"LtE", mlir::arith::CmpIPredicate::sle},
                      {"Eq", mlir::arith::CmpIPredicate::eq},
                      {"NotEq", mlir::arith::CmpIPredicate::ne}};

// Comparison predicates for float operations
static const std::unordered_map<std::string, mlir::arith::CmpFPredicate>
    kFloatCompareOps = {{"Gt", mlir::arith::CmpFPredicate::OGT},
                        {"Lt", mlir::arith::CmpFPredicate::OLT},
                        {"GtE", mlir::arith::CmpFPredicate::OGE},
                        {"LtE", mlir::arith::CmpFPredicate::OLE},
                        {"Eq", mlir::arith::CmpFPredicate::OEQ},
                        {"NotEq", mlir::arith::CmpFPredicate::ONE}};

ExprGenerator::ExprGenerator(FunctionGenerator *parent) : parent_(parent) {
    SS_ASSERT(parent_ && parent_->GetContext());
}

mlir::Location ExprGenerator::GetMLIRLocation(const ast::ASTNode *node) const {
    return parent_->GetContext()->GetMLIRLocation(
        parent_->GetBuilder().getContext(), node);
}

mlir::Location ExprGenerator::GetMLIRLocation(clang::SourceLocation loc) const {
    return parent_->GetContext()->GetMLIRLocation(
        parent_->GetBuilder().getContext(), loc);
}

mlir::Value ExprGenerator::EnsureCompatibleTypes(mlir::Value value,
                                                 mlir::Type source_type,
                                                 mlir::Type target_type,
                                                 clang::SourceLocation loc) {
    auto *ctx = parent_->GetContext();
    auto &builder = parent_->GetBuilder();
    auto *diagnostics = ctx->diagnostic_manager.get();

    // If types are already the same, no conversion needed
    if (source_type == target_type) {
        return value;
    }

    auto source_memref = mlir::dyn_cast<cf::MemRefType>(source_type);
    auto target_memref = mlir::dyn_cast<cf::MemRefType>(target_type);
    if (source_memref && target_memref) {
        auto sourceBytes = GetStaticByteSize(source_memref);
        auto targetBytes = GetStaticByteSize(target_memref);
        if (sourceBytes && targetBytes && *sourceBytes == *targetBytes &&
            source_memref.getMemorySpace() == target_memref.getMemorySpace()) {
            mlir::Location mlirLoc = GetMLIRLocation(loc);
            return cf::AveLangMemRefCastOp::create(builder, mlirLoc, value,
                                                   target_memref)
                .getResult();
        }
    }

    // Special handling for vector types
    auto source_vector = mlir::dyn_cast<mlir::VectorType>(source_type);
    auto target_vector = mlir::dyn_cast<mlir::VectorType>(target_type);

    if (source_vector && target_vector) {
        // Both are vectors - check compatibility
        if (source_vector.getShape() == target_vector.getShape()) {
            // Same shape - check if element types are compatible
            auto source_element = source_vector.getElementType();
            auto target_element = target_vector.getElementType();

            if (source_element == target_element) {
                // Exact match - no conversion needed
                return value;
            }

            // Try to convert element types if they're different but compatible
            auto converted_element = CreateTypeConversion(
                mlir::vector::ExtractOp::create(builder, GetMLIRLocation(loc),
                                                value, 0),
                source_element, target_element, GetMLIRLocation(loc), builder);

            if (converted_element) {
                // Element types are convertible - this could work but would
                // need element-wise conversion For now, allow it through and
                // let MLIR handle the details
                return value;
            }
        }

        // Incompatible vector types
        if (diagnostics) {
            std::string source_str, target_str;
            llvm::raw_string_ostream source_os(source_str),
                target_os(target_str);
            source_type.print(source_os);
            target_type.print(target_os);
            std::string error_msg =
                "Incompatible vector types: cannot convert from " +
                source_os.str() + " to " + target_os.str();
            diagnostics->Report(basic::DiagnosticCode::kUnimplemented, loc)
                << error_msg;
        }
        return mlir::Value();
    }

    // If one is vector and the other is not, they're incompatible for direct
    // assignment
    if (source_vector || target_vector) {
        if (diagnostics) {
            std::string source_str, target_str;
            llvm::raw_string_ostream source_os(source_str),
                target_os(target_str);
            source_type.print(source_os);
            target_type.print(target_os);
            diagnostics->Report(basic::DiagnosticCode::kUnimplemented, loc)
                << "Cannot convert between vector and scalar types: "
                << source_os.str() << " to " << target_os.str();
        }
        return mlir::Value();
    }

    // For non-vector types, use the standard type conversion
    auto result = CreateTypeConversion(value, source_type, target_type,
                                       GetMLIRLocation(loc), builder);
    if (!result) {
        if (diagnostics) {
            std::string source_str, target_str;
            llvm::raw_string_ostream source_os(source_str),
                target_os(target_str);
            source_type.print(source_os);
            target_type.print(target_os);
            std::string error_msg = "Incompatible types: cannot convert from " +
                                    source_os.str() + " to " + target_os.str();
            diagnostics->Report(basic::DiagnosticCode::kUnimplemented, loc)
                << error_msg;
        }
        return mlir::Value();
    }
    return result;
}

mlir::Value ExprGenerator::CreateVoidValue() { return mlir::Value(); }

mlir::Value ExprGenerator::VisitBinOp(ast::BinOp *binop) {
    if (!binop)
        return nullptr;

    const std::string &op = binop->GetOp();
    auto &builder = parent_->GetBuilder();
    auto location = GetMLIRLocation(binop);

    // Check if this is a supported operation in any category
    auto enhancedIntOpIt = kIntOps.find(op);
    auto floatOpIt = kFloatOps.find(op);
    auto bitwiseOpIt = kBitwiseOps.find(op);

    // Special case for FloorDiv since we handle it custom
    bool isFloorDiv = (op == "FloorDiv");

    if (enhancedIntOpIt == kIntOps.end() && floatOpIt == kFloatOps.end() &&
        bitwiseOpIt == kBitwiseOps.end() && !isFloorDiv) {
        // Unsupported binary operation
        Report(this, basic::DiagnosticCode::kUnimplemented,
               binop->GetSourceRange().getBegin())
            << "Unsupported binary operation: " << op;
        return nullptr;
    }

    // Generate operands
    auto lhs = Dispatch(binop->GetLeft());
    auto rhs = Dispatch(binop->GetRight());

    if (!lhs || !rhs) {
        return nullptr;
    }

    // For bitwise operations, we need integer types and may need different type
    // conversion
    if (bitwiseOpIt != kBitwiseOps.end()) {
        // Bitwise operations require integer types
        auto lhs_type = lhs.getType();
        auto rhs_type = rhs.getType();
        auto lhsTypeInfo = GetTypeInfo(lhs);

        if (!lhs_type.isIntOrIndex() || !rhs_type.isIntOrIndex()) {
            Report(this, basic::DiagnosticCode::kUnimplemented,
                   binop->GetSourceRange().getBegin())
                << "Bitwise operations require integer types";
            return nullptr;
        }

        // Convert types for bitwise operations (similar to arithmetic but no
        // int-to-float conversion)
        auto [converted_lhs, converted_rhs] =
            ConvertTypesForBitwise(lhs, rhs, builder, location);
        if (!converted_lhs || !converted_rhs) {
            Report(this, basic::DiagnosticCode::kUnimplemented,
                   binop->GetSourceRange().getBegin())
                << "Cannot perform bitwise operations with incompatible types";
            return nullptr;
        }

        if ((op == "LShift" || op == "RShift") &&
            lhsTypeInfo.is_unsigned_integer &&
            GetTypeInfo(converted_lhs).is_unsigned_integer !=
                lhsTypeInfo.is_unsigned_integer) {
            auto recast = CreateTypeConversion(
                converted_lhs, converted_lhs.getType(), converted_lhs.getType(),
                location, builder,
                /*allow_demotion=*/false,
                GetTypeInfo(converted_lhs).is_unsigned_integer,
                lhsTypeInfo.is_unsigned_integer);
            if (recast) {
                converted_lhs = recast;
            }
        }

        bool isUnsigned =
            GetTypeInfo(converted_lhs).is_unsigned_integer.value_or(false);
        mlir::Value result;
        if (op == "RShift" && isUnsigned) {
            result = mlir::arith::ShRUIOp::create(builder, location,
                                                  converted_lhs, converted_rhs);
        } else {
            result = bitwiseOpIt->second(builder, location, converted_lhs,
                                         converted_rhs);
        }

        if (result) {
            SetTypeInfo(result, TypeInfo{isUnsigned});
        }
        return result;
    }

    // For arithmetic operations, use the existing type conversion
    auto [converted_lhs, converted_rhs] =
        ConvertTypesForArithmetic(lhs, rhs, builder, location);
    if (!converted_lhs || !converted_rhs) {
        Report(this, basic::DiagnosticCode::kUnimplemented,
               binop->GetSourceRange().getBegin())
            << "Cannot perform arithmetic with incompatible types";
        return nullptr;
    }

    // Create the appropriate operation based on operator and type
    auto result_type = converted_lhs.getType();
    if (result_type.isIntOrIndex()) {
        bool isUnsigned =
            GetTypeInfo(converted_lhs).is_unsigned_integer.value_or(false);
        // Special handling for floor division
        if (op == "FloorDiv") {
            if (isUnsigned) {
                auto result = mlir::arith::DivUIOp::create(
                    builder, location, converted_lhs, converted_rhs);
                SetTypeInfo(result.getResult(), TypeInfo{true});
                return result;
            }
            // Implement floor division using: floor(a / b) for
            // positive/negative integers Formula: floor(a/b) = (a >= 0) ? a/b :
            // ((a - (b - 1)) / b) when b > 0 More robust: floor(a/b) = (a - ((a
            // % b + b) % b)) / b

            auto zero = mlir::arith::ConstantIntOp::create(builder, location,
                                                           result_type, 0);

            // Compute a % b
            auto remainder = mlir::arith::RemSIOp::create(
                builder, location, converted_lhs, converted_rhs);

            // Check if remainder is zero (no adjustment needed)
            auto remainder_is_zero = mlir::arith::CmpIOp::create(
                builder, location, mlir::arith::CmpIPredicate::eq, remainder,
                zero);

            // Compute adjustment: (remainder + b) % b
            // This gives us the positive remainder even when a is negative
            auto adjustment_add = mlir::arith::AddIOp::create(
                builder, location, remainder, converted_rhs);
            auto adjustment_mod = mlir::arith::RemSIOp::create(
                builder, location, adjustment_add, converted_rhs);

            // Compute adjusted dividend: a - adjustment
            auto adjusted_dividend = mlir::arith::SubIOp::create(
                builder, location, converted_lhs, adjustment_mod);

            // Compute floor division result: adjusted_dividend / b
            auto floor_div_result = mlir::arith::DivSIOp::create(
                builder, location, adjusted_dividend, converted_rhs);

            // Use regular division if remainder is zero, otherwise use floor
            // division
            auto regular_div = mlir::arith::DivSIOp::create(
                builder, location, converted_lhs, converted_rhs);

            auto result = mlir::arith::SelectOp::create(
                builder, location, remainder_is_zero, regular_div,
                floor_div_result);
            SetTypeInfo(result.getResult(), TypeInfo{false});
            return result;
        }

        if (enhancedIntOpIt != kIntOps.end()) {
            mlir::Value result;
            if (op == "Div" && isUnsigned) {
                result = mlir::arith::DivUIOp::create(
                    builder, location, converted_lhs, converted_rhs);
            } else if (op == "Mod" && isUnsigned) {
                result = mlir::arith::RemUIOp::create(
                    builder, location, converted_lhs, converted_rhs);
            } else {
                result = enhancedIntOpIt->second(builder, location,
                                                 converted_lhs, converted_rhs);
            }
            SetTypeInfo(result, TypeInfo{isUnsigned});
            return result;
        }
    } else if (mlir::isa<mlir::FloatType>(result_type)) {
        if (floatOpIt != kFloatOps.end()) {
            return floatOpIt->second(builder, location, converted_lhs,
                                     converted_rhs);
        }
    }

    // Unsupported type for this operation
    Report(this, basic::DiagnosticCode::kUnimplemented,
           binop->GetSourceRange().getBegin())
        << "Unsupported type for " << op << " operation";
    return nullptr;
}

mlir::Value ExprGenerator::VisitUnaryOp(ast::UnaryOp *unaryop) {
    if (!unaryop)
        return nullptr;

    const std::string &op = unaryop->GetOp();
    auto &builder = parent_->GetBuilder();
    auto location = GetMLIRLocation(unaryop);

    // Generate operand
    auto operand = Dispatch(unaryop->GetOperand());
    if (!operand) {
        return nullptr;
    }

    // Handle 'Not' operator (boolean negation)
    if (op == "Not") {
        // Convert operand to i1 (boolean) if needed
        if (!operand.getType().isInteger(1)) {
            // Create comparison with zero to convert to boolean
            auto zero = mlir::arith::ConstantOp::create(
                builder, location, operand.getType(),
                builder.getIntegerAttr(operand.getType(), 0));
            operand = mlir::arith::CmpIOp::create(
                builder, location, mlir::arith::CmpIPredicate::eq, operand,
                zero);
        }

        // Logical NOT
        auto one = mlir::arith::ConstantOp::create(
            builder, location, mlir::IntegerType::get(builder.getContext(), 1),
            builder.getIntegerAttr(
                mlir::IntegerType::get(builder.getContext(), 1), 1));
        return mlir::arith::XOrIOp::create(builder, location, operand, one);
    }

    if (op == "UAdd") {
        return operand;
    }

    if (op == "USub") {
        auto operandType = operand.getType();

        if (mlir::isa<mlir::FloatType>(operandType)) {
            return mlir::arith::NegFOp::create(builder, location, operand);
        }

        if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(operandType)) {
            if (mlir::isa<mlir::FloatType>(vectorType.getElementType())) {
                return mlir::arith::NegFOp::create(builder, location, operand);
            }

            if (vectorType.getElementType().isInteger()) {
                auto zeroAttr =
                    builder.getIntegerAttr(vectorType.getElementType(), 0);
                auto zero = mlir::arith::ConstantOp::create(
                    builder, location, vectorType,
                    mlir::DenseElementsAttr::get(vectorType, zeroAttr));
                auto result = mlir::arith::SubIOp::create(builder, location,
                                                          zero, operand);
                SetTypeInfo(result, GetTypeInfo(operand));
                return result;
            }
        }

        if (operandType.isIndex()) {
            auto zero =
                mlir::arith::ConstantIndexOp::create(builder, location, 0);
            return mlir::arith::SubIOp::create(builder, location, zero,
                                               operand);
        }

        if (operandType.isInteger()) {
            auto zero = mlir::arith::ConstantOp::create(
                builder, location, operandType,
                builder.getIntegerAttr(operandType, 0));
            auto result =
                mlir::arith::SubIOp::create(builder, location, zero, operand);
            SetTypeInfo(result, GetTypeInfo(operand));
            return result;
        }
    }

    Report(this, basic::DiagnosticCode::kUnimplemented,
           unaryop->GetSourceRange().getBegin())
        << ("Unsupported unary operation: " + op);
    return nullptr;
}

mlir::Value ExprGenerator::VisitBoolOp(ast::BoolOp *boolop) {
    if (!boolop)
        return nullptr;

    const std::string &op = boolop->GetOp();
    auto &builder = parent_->GetBuilder();
    auto location = GetMLIRLocation(boolop);

    const auto &values = boolop->GetValues();
    if (values.size() < 2) {
        Report(this, basic::DiagnosticCode::kUnimplemented,
               boolop->GetSourceRange().getBegin())
            << "BoolOp requires at least 2 operands";
        return nullptr;
    }

    // Generate the first operand
    mlir::Value result = Dispatch(values[0]);
    if (!result) {
        return nullptr;
    }

    // Convert operands to i1 (boolean) if needed
    auto convertToBool = [&](mlir::Value value) -> mlir::Value {
        if (value.getType().isInteger(1)) {
            return value;
        }
        // Create comparison with zero to convert to boolean
        auto zero = mlir::arith::ConstantOp::create(
            builder, location, value.getType(),
            builder.getIntegerAttr(value.getType(), 0));
        return mlir::arith::CmpIOp::create(
            builder, location, mlir::arith::CmpIPredicate::ne, value, zero);
    };

    result = convertToBool(result);

    // Process remaining operands
    for (size_t i = 1; i < values.size(); ++i) {
        auto operand = Dispatch(values[i]);
        if (!operand) {
            return nullptr;
        }

        operand = convertToBool(operand);

        if (op == "And") {
            // Logical AND
            result =
                mlir::arith::AndIOp::create(builder, location, result, operand);
        } else if (op == "Or") {
            // Logical OR
            result =
                mlir::arith::OrIOp::create(builder, location, result, operand);
        } else {
            Report(this, basic::DiagnosticCode::kUnimplemented,
                   boolop->GetSourceRange().getBegin())
                << "Unsupported boolean operation: " << op;
            return nullptr;
        }
    }

    return result;
}

mlir::Value ExprGenerator::VisitName(ast::Name *name) {
    if (!name)
        return nullptr;

    // Use ResolveRefExpr to look up the symbol
    auto value = parent_->GetContext()->syms->ResolveRefExpr(name);
    if (!value)
        return nullptr;

    // Check if this is an immutable (constexpr) constant that needs to be
    // cloned
    auto symbol = parent_->GetContext()->syms->ResolveSymbol(
        name, ir::SymbolScope::SymbolKind::kValue, false);
    if (symbol && symbol->immutable && value.getDefiningOp()) {
        // This is a constexpr value - clone it into the current insertion point
        if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(
                value.getDefiningOp())) {
            auto &builder = parent_->GetBuilder();
            value = mlir::arith::ConstantOp::create(builder, constOp.getLoc(),
                                                    constOp.getValue());
        }
    }

    // If the symbol is a scalar memref (created by variable assignment), load
    // from it. Keep scalar memref block arguments as memrefs so call sites can
    // pass tensors of shape (1,).
    if (auto memref_type = mlir::dyn_cast<cf::MemRefType>(value.getType())) {
        // Check if it's a scalar memref (rank 1, size 1)
        if (memref_type.getRank() == 1 && memref_type.getShape()[0] == 1 &&
            !mlir::isa<mlir::BlockArgument>(value)) {
            auto &builder = parent_->GetBuilder();
            auto zero_index = mlir::arith::ConstantIndexOp::create(
                builder, GetMLIRLocation(name), 0);
            auto load = cf::AveLangMemRefLoadOp::create(
                builder, GetMLIRLocation(name), memref_type.getElementType(),
                value, mlir::ValueRange{zero_index});
            SetTypeInfo(load.getResult(), GetTypeInfo(value));
            return load.getResult();
        }
    }

    // For non-scalar memrefs or other types, return as-is
    return value;
}

mlir::Value ExprGenerator::VisitSubscript(ast::Subscript *subscript) {
    if (!subscript)
        return nullptr;

    auto &builder = parent_->GetBuilder();
    auto location = GetMLIRLocation(subscript);

    // Get the value being indexed
    auto value = Dispatch(subscript->GetValue());
    if (!value)
        return nullptr;

    auto value_type = value.getType();

    // Handle the slice expression (which might be a tuple for multi-dimensional
    // access)
    auto *slice_expr = subscript->GetSlice();
    llvm::SmallVector<mlir::Value> indices;

    if (auto *tuple_expr = llvm::dyn_cast<ast::Tuple>(slice_expr)) {
        // Multi-dimensional indexing: a[0, 1, 2]
        for (auto *element : tuple_expr->GetElts()) {
            auto index = Dispatch(element);
            if (!index)
                return nullptr;
            // Convert to index type if needed
            if (!index.getType().isIndex()) {
                index = mlir::arith::IndexCastOp::create(
                    builder, GetMLIRLocation(element), builder.getIndexType(),
                    index);
            }
            indices.push_back(index);
        }
    } else {
        // Single-dimensional indexing: a[0]
        auto index = Dispatch(slice_expr);
        if (!index)
            return nullptr;
        // Convert to index type if needed
        if (!index.getType().isIndex()) {
            index = mlir::arith::IndexCastOp::create(
                builder, GetMLIRLocation(slice_expr), builder.getIndexType(),
                index);
        }
        indices.push_back(index);
    }

    // Check if we're accessing a vector type directly
    if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(value_type)) {
        // Vector element extraction
        if (static_cast<int64_t>(indices.size()) > vector_type.getRank()) {
            Report(this, basic::DiagnosticCode::kUnimplemented,
                   subscript->GetSourceRange().getBegin())
                << "Too many indices for vector type access";
            return nullptr;
        }

        // Use vector.extract for vector element access
        // For dynamic indices, we need to build ArrayRef<OpFoldResult>
        llvm::SmallVector<mlir::OpFoldResult> position_indices;
        position_indices.reserve(indices.size());
        for (auto idx : indices) {
            if (auto constantIndex = GetConstantIndexLikeValue(idx)) {
                position_indices.push_back(
                    builder.getIndexAttr(*constantIndex));
            } else {
                position_indices.push_back(idx);
            }
        }
        auto extract = mlir::vector::ExtractOp::create(builder, location, value,
                                                       position_indices);
        auto result = extract.getResult();
        SetTypeInfo(result, GetTypeInfo(value));
        if (auto result_vec_type =
                mlir::dyn_cast<mlir::VectorType>(result.getType())) {
            int64_t element_count = 1;
            int non_unit_dims = 0;
            for (auto dim : result_vec_type.getShape()) {
                element_count *= dim;
                if (dim != 1) {
                    ++non_unit_dims;
                }
            }
            if (result_vec_type.getRank() > 1 && non_unit_dims <= 1) {
                auto flat_type = mlir::VectorType::get(
                    element_count, result_vec_type.getElementType());
                auto flat = mlir::vector::ShapeCastOp::create(
                    builder, location, flat_type, result);
                SetTypeInfo(flat.getResult(), GetTypeInfo(result));
                result = flat.getResult();
            }
        }
        return result;
    } else if (auto memref_type = mlir::dyn_cast<cf::MemRefType>(value_type)) {
        size_t rank = memref_type.getRank();
        LayoutOperation::ExpandIndicesForNestedLayout(builder, value, indices);
        if (indices.size() > rank) {
            Report(this, basic::DiagnosticCode::kUnimplemented,
                   subscript->GetSourceRange().getBegin())
                << "Too many indices for memref type access";
            return nullptr;
        }
        if (indices.size() < rank) {
            // Return a sub-memref or a vector view of a 1-D slice.
            llvm::SmallVector<int64_t> sub_shape(
                memref_type.getShape().begin() + indices.size(),
                memref_type.getShape().end());
            llvm::SmallVector<int64_t> sub_strides;
            auto base_strides = memref_type.getStrides();
            if (base_strides.size() ==
                static_cast<size_t>(memref_type.getRank())) {
                sub_strides.assign(base_strides.begin() + indices.size(),
                                   base_strides.end());
            }
            auto sub_memref_type = createAveLangMemRefType(
                builder.getContext(), sub_shape, memref_type.getElementType(),
                memref_type.getMemorySpace(), sub_strides);
            auto one =
                mlir::arith::ConstantIndexOp::create(builder, location, 1);
            llvm::SmallVector<mlir::Value> sizes(indices.size(), one);
            llvm::SmallVector<mlir::Value> strides(indices.size(), one);
            auto subview = cf::AveLangMemRefSubViewOp::create(
                               builder, location, sub_memref_type, value,
                               indices, sizes, strides)
                               .getResult();
            SetTypeInfo(subview, GetTypeInfo(value));

            bool has_dynamic = false;
            int64_t element_count = 1;
            int non_unit_dims = 0;
            for (auto dim : sub_shape) {
                if (dim == mlir::ShapedType::kDynamic) {
                    has_dynamic = true;
                    break;
                }
                element_count *= dim;
                if (dim != 1) {
                    ++non_unit_dims;
                }
            }

            bool element_is_vector =
                mlir::isa<mlir::VectorType>(memref_type.getElementType());
            if (!element_is_vector && !has_dynamic && non_unit_dims <= 1) {
                llvm::SmallVector<mlir::Value> vec_indices;
                vec_indices.reserve(sub_shape.size());
                for (size_t i = 0; i < sub_shape.size(); ++i) {
                    vec_indices.push_back(mlir::arith::ConstantIndexOp::create(
                        builder, location, 0));
                }

                auto load_vec_type = mlir::VectorType::get(
                    sub_shape, memref_type.getElementType());
                auto load = cf::AveLangMemRefLoadVecOp::create(
                    builder, location, load_vec_type, subview, vec_indices);
                if (load_vec_type.getRank() == 1) {
                    SetTypeInfo(load.getResult(), GetTypeInfo(subview));
                    return load.getResult();
                }

                auto flat_type = mlir::VectorType::get(
                    element_count, memref_type.getElementType());
                auto flat = mlir::vector::ShapeCastOp::create(
                    builder, location, flat_type, load.getResult());
                SetTypeInfo(flat.getResult(), GetTypeInfo(load.getResult()));
                return flat.getResult();
            }

            return subview;
        } else {
            // Full element access
            if (auto element_vector_type = mlir::dyn_cast<mlir::VectorType>(
                    memref_type.getElementType())) {
                auto load = cf::AveLangMemRefLoadVecOp::create(
                    builder, location, element_vector_type, value, indices);
                SetTypeInfo(load.getResult(), GetTypeInfo(value));
                return load.getResult();
            } else {
                // Regular memref element access
                auto load = cf::AveLangMemRefLoadOp::create(
                    builder, location, memref_type.getElementType(), value,
                    mlir::ValueRange(indices));
                SetTypeInfo(load.getResult(), GetTypeInfo(value));
                return load.getResult();
            }
        }
    }

    // If we reach here, the type is not supported for subscripting
    Report(this, basic::DiagnosticCode::kUnimplemented,
           subscript->GetSourceRange().getBegin())
        << "Subscript operation not supported for this type";
    return nullptr;
}

mlir::Value ExprGenerator::VisitConstant(ast::Constant *constant) {
    if (!constant)
        return nullptr;

    auto &builder = parent_->GetBuilder();
    const std::string &value = constant->GetValue();
    auto location = GetMLIRLocation(constant);

    // Try to parse as integer first
    if (auto intValue = ParseConstantInteger(value)) {
        // For integer constants, we'll use a default i32 type unless context
        // suggests otherwise This can be extended later to infer type from
        // usage context
        auto intType = builder.getI32Type();

        // Create integer constant using ConstantOp instead of deprecated
        // ConstantIndexOp for portability
        auto attrValue = builder.getIntegerAttr(intType, *intValue);
        auto constantOp =
            mlir::arith::ConstantOp::create(builder, location, attrValue);
        SetTypeInfo(constantOp.getResult(), TypeInfo{false});
        return constantOp.getResult();
    }

    // Try to parse as float
    if (auto floatValue = ParseConstantFloat(value)) {
        // For float constants, determine type based on the literal format or
        // default to f64
        mlir::Type floatType = builder.getF64Type();

        // Check if it looks like a single-precision float (ends with 'f' or
        // 'F')
        if (!value.empty() && (value.back() == 'f' || value.back() == 'F')) {
            floatType = builder.getF32Type();
        }

        auto attrValue = builder.getFloatAttr(floatType, *floatValue);
        return mlir::arith::ConstantOp::create(builder, location, attrValue);
    }

    // Handle boolean constants
    if (value == "True" || value == "true") {
        auto attrValue = builder.getBoolAttr(true);
        return mlir::arith::ConstantOp::create(builder, location, attrValue);
    }
    if (value == "False" || value == "false") {
        auto attrValue = builder.getBoolAttr(false);
        return mlir::arith::ConstantOp::create(builder, location, attrValue);
    }

    if (value != "None") {
        auto stringAttr = builder.getStringAttr(value);
        auto castOp = mlir::UnrealizedConversionCastOp::create(
            builder, location, mlir::TypeRange{builder.getNoneType()},
            mlir::ValueRange{});

        castOp->setAttr("ave.string_literal", stringAttr);
        castOp->setAttr("is_string_literal", builder.getBoolAttr(true));
        return castOp.getResult(0);
    }

    // Leave other constants unresolved so specialized handlers can process
    // them.
    return nullptr;
}

mlir::Value ExprGenerator::VisitTuple(ast::Tuple *tuple) {
    if (!tuple)
        return nullptr;

    auto &builder = parent_->GetBuilder();
    auto location = GetMLIRLocation(tuple);
    llvm::SmallVector<mlir::Value> elements;
    elements.reserve(tuple->GetElts().size());

    for (auto *element : tuple->GetElts()) {
        auto value = Dispatch(element);
        if (!value) {
            return nullptr;
        }
        elements.push_back(value);
    }

    if (elements.empty()) {
        Report(this, basic::DiagnosticCode::kUnimplemented,
               tuple->GetSourceRange().getBegin())
            << "Empty tuple expressions are not supported";
        return nullptr;
    }

    auto tuple_op = cf::MakeIntTupleOp::create(builder, location, elements);
    return tuple_op.getResult();
}

mlir::Value ExprGenerator::CastTensorVector(mlir::Value value,
                                            mlir::Type target_type,
                                            clang::SourceLocation loc) {
    if (!value || !target_type || value.getType() == target_type)
        return value;

    auto &builder = parent_->GetBuilder();
    auto mlir_loc = GetMLIRLocation(loc);

    auto createZeroIndices =
        [&](int64_t rank) -> llvm::SmallVector<mlir::Value> {
        llvm::SmallVector<mlir::Value> indices;
        indices.reserve(rank);
        for (int64_t i = 0; i < rank; ++i) {
            indices.push_back(
                mlir::arith::ConstantIndexOp::create(builder, mlir_loc, 0));
        }
        return indices;
    };

    if (auto target_vector = mlir::dyn_cast<mlir::VectorType>(target_type)) {
        if (auto source_memref_type =
                mlir::dyn_cast<cf::MemRefType>(value.getType())) {
            if (source_memref_type.getRank() != target_vector.getRank())
                return value;

            bool compatible = true;
            for (int64_t i = 0; i < source_memref_type.getRank(); ++i) {
                int64_t vec_dim = target_vector.getShape()[i];
                int64_t mem_dim = source_memref_type.getShape()[i];
                if (mem_dim != vec_dim &&
                    mem_dim != mlir::ShapedType::kDynamic) {
                    compatible = false;
                    break;
                }
            }

            if (compatible && source_memref_type.getElementType() ==
                                  target_vector.getElementType()) {
                auto indices = createZeroIndices(source_memref_type.getRank());
                auto load = cf::AveLangMemRefLoadVecOp::create(
                    builder, mlir_loc, target_vector, value, indices);
                return load.getResult();
            }
        }
    } else if (auto target_memref =
                   mlir::dyn_cast<cf::MemRefType>(target_type)) {
        if (auto source_vector =
                mlir::dyn_cast<mlir::VectorType>(value.getType())) {
            if (source_vector.getRank() != target_memref.getRank())
                return value;

            bool compatible = true;
            for (int64_t i = 0; i < target_memref.getRank(); ++i) {
                int64_t vec_dim = source_vector.getShape()[i];
                int64_t mem_dim = target_memref.getShape()[i];
                if (mem_dim != vec_dim) {
                    compatible = false;
                    break;
                }
            }

            if (compatible && source_vector.getElementType() ==
                                  target_memref.getElementType()) {
                // Only support static memref conversions for now.
                if (llvm::any_of(target_memref.getShape(), [](int64_t dim) {
                        return dim == mlir::ShapedType::kDynamic;
                    })) {
                    return value;
                }

                auto addressSpaceAttr = mlir::gpu::AddressSpaceAttr::get(
                    builder.getContext(), mlir::gpu::AddressSpace::Private);
                cf::MemRefType target_memref_type;
                if (auto target_avelang_memref =
                        mlir::dyn_cast<cf::MemRefType>(target_type)) {
                    target_memref_type = cf::MemRefType::get(
                        builder.getContext(), target_avelang_memref.getLayout(),
                        target_avelang_memref.getElementType(),
                        addressSpaceAttr);
                } else {
                    target_memref_type = createAveLangMemRefType(
                        builder.getContext(), target_memref.getShape(),
                        target_memref.getElementType(), addressSpaceAttr);
                }

                auto alloca = cf::AveLangMemRefAllocaOp::create(
                    builder, mlir_loc, target_memref_type, mlir::ValueRange{});
                auto indices = createZeroIndices(target_memref_type.getRank());
                cf::AveLangMemRefStoreOp::create(builder, mlir_loc, value,
                                                 alloca, indices);
                return alloca.getResult();
            }
        }
    }

    return value;
}

mlir::Value
ExprGenerator::GenerateFuncCall(ast::Call *call, mlir::func::FuncOp func_op,
                                llvm::ArrayRef<mlir::Value> resolved_args) {
    return GenerateFuncCall(call, func_op, resolved_args, call->GetArgs());
}

mlir::Value
ExprGenerator::GenerateFuncCall(ast::Call *call, mlir::func::FuncOp func_op,
                                llvm::ArrayRef<mlir::Value> resolved_args,
                                llvm::ArrayRef<ast::Expr *> source_arg_exprs) {
    if (!call || !func_op) {
        return nullptr;
    }

    auto func_type = func_op.getFunctionType();
    auto input_types = func_type.getInputs();
    if (input_types.size() != source_arg_exprs.size()) {
        Report(this, basic::DiagnosticCode::kTypeMismatch,
               call->GetSourceRange().getBegin())
            << "Function '" << func_op.getSymName() << "' expects "
            << input_types.size() << " arguments but got "
            << source_arg_exprs.size();
        return nullptr;
    }

    if (resolved_args.size() < input_types.size()) {
        Report(this, basic::DiagnosticCode::kUnimplemented,
               call->GetSourceRange().getBegin())
            << "Insufficient arguments evaluated for call to '"
            << func_op.getSymName() << "'";
        return nullptr;
    }

    llvm::SmallVector<mlir::Value> call_args;
    call_args.reserve(input_types.size());
    for (size_t i = 0; i < input_types.size(); ++i) {
        auto *arg_expr = source_arg_exprs[i];
        auto expected_type = input_types[i];
        mlir::Value arg_value;

        if (mlir::isa<cf::MemRefType>(expected_type)) {
            arg_value = parent_->ResolveMemrefValue(arg_expr);
        }
        if (!arg_value) {
            arg_value = resolved_args[i];
        }
        if (!arg_value) {
            Report(this, basic::DiagnosticCode::kUnimplemented,
                   arg_expr->GetSourceRange().getBegin())
                << "Failed to generate call argument " << i;
            return nullptr;
        }

        auto source_loc = arg_expr->GetSourceRange().getBegin();

        auto cast_value =
            CastTensorVector(arg_value, expected_type, source_loc);

        if (cast_value.getType() != expected_type) {
            cast_value = EnsureCompatibleTypes(cast_value, cast_value.getType(),
                                               expected_type, source_loc);
        }

        if (!cast_value || cast_value.getType() != expected_type) {
            std::string error_msg = "Cannot convert argument " +
                                    std::to_string(i) + " from type '" +
                                    GetTypeName(arg_value.getType()) +
                                    "' to '" + GetTypeName(expected_type) + "'";
            Report(this, basic::DiagnosticCode::kTypeMismatch, source_loc)
                << error_msg;
            return nullptr;
        }

        call_args.push_back(cast_value);
    }

    auto result_types = func_type.getResults();

    auto &builder = parent_->GetBuilder();
    mlir::Location loc = GetMLIRLocation(call);

    auto call_op = mlir::func::CallOp::create(
        builder, loc, func_op.getSymName(), result_types, call_args);
    if (result_types.empty()) {
        return CreateVoidValue();
    }
    if (result_types.size() == 1) {
        return call_op.getResult(0);
    }

    for (auto type : result_types) {
        if (!type.isIntOrIndex() && !mlir::isa<mlir::NoneType>(type)) {
            Report(this, basic::DiagnosticCode::kUnimplemented,
                   call->GetSourceRange().getBegin())
                << "Multiple return values must be integers, indices, or "
                   "tuples";
            return nullptr;
        }
    }

    auto tuple_op =
        cf::MakeIntTupleOp::create(builder, loc, call_op.getResults());
    return tuple_op.getResult();
}

mlir::Value ExprGenerator::GenerateJitFunctionCall(
    ast::Call *call, ast::FunctionDef *func,
    llvm::ArrayRef<mlir::Value> resolved_args) {
    if (!call || !func) {
        return mlir::Value();
    }

    auto *ctx = parent_->GetContext();
    auto fold_constexpr_argument = [&](ast::Expr *expr,
                                       mlir::Value value) -> mlir::Attribute {
        ConstantFolder folder(ctx);
        if (auto evaluated = folder.Evaluate(expr)) {
            return parent_->GetBuilder().getI64IntegerAttr(*evaluated);
        }
        if (!value) {
            return {};
        }

        auto &builder = parent_->GetBuilder();
        if (auto const_op = value.getDefiningOp<mlir::arith::ConstantOp>()) {
            auto attr = const_op.getValue();
            if (mlir::isa<mlir::IntegerAttr, mlir::FloatAttr>(attr)) {
                return attr;
            }
        }
        if (auto folded = ConstantFolder::FoldIntValue(value)) {
            auto type = value.getType();
            if (type.isIndex()) {
                return builder.getIndexAttr(*folded);
            }
            if (type.isInteger()) {
                return builder.getIntegerAttr(type, *folded);
            }
            return builder.getI64IntegerAttr(*folded);
        }
        if (auto folded = ConstantFolder::FoldBoolValue(value)) {
            return builder.getBoolAttr(*folded);
        }
        return {};
    };

    auto collect_call_specialization =
        [&](ast::FunctionDef *callee, llvm::ArrayRef<mlir::Value> all_call_args,
            llvm::SmallVectorImpl<mlir::Value> &runtime_call_args,
            llvm::SmallVectorImpl<ast::Expr *> &runtime_arg_exprs,
            ConstexprBindingMap &constexpr_bindings) -> ArgAddressSpaceMap {
        ArgAddressSpaceMap address_spaces;
        if (!callee) {
            return address_spaces;
        }
        auto *func_args = callee->GetArguments();
        if (!func_args) {
            return address_spaces;
        }
        const auto &source_args = call->GetArgs();
        if (source_args.size() != func_args->GetArgs().size()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kTypeMismatch,
                call->GetSourceRange().getBegin())
                << "JIT function '" << callee->GetName() << "' expects "
                << func_args->GetArgs().size() << " arguments but got "
                << source_args.size();
            return address_spaces;
        }
        size_t call_index = 0;
        for (auto *arg : func_args->GetArgs()) {
            if (!arg) {
                continue;
            }
            if (auto *attr_expr = llvm::dyn_cast_or_null<ast::AttributeExpr>(
                    arg->GetAnnotation())) {
                if (attr_expr->GetAttr() == "constexpr") {
                    if (call_index >= all_call_args.size()) {
                        break;
                    }
                    auto attr = fold_constexpr_argument(
                        source_args[call_index], all_call_args[call_index]);
                    if (!attr) {
                        ctx->diagnostic_manager->Report(
                            basic::DiagnosticCode::kUnimplemented,
                            source_args[call_index]
                                ->GetSourceRange()
                                .getBegin())
                            << "JIT helper constexpr parameter '"
                            << arg->GetArgName()
                            << "' is not a specialization-time constant";
                    } else {
                        constexpr_bindings[arg->GetArgName()] = attr;
                    }
                    ++call_index;
                    continue;
                }
            }
            if (call_index >= all_call_args.size()) {
                break;
            }
            auto value = all_call_args[call_index];
            runtime_call_args.push_back(value);
            runtime_arg_exprs.push_back(source_args[call_index]);
            ++call_index;
            if (!value) {
                continue;
            }
            if (auto memref_type =
                    mlir::dyn_cast<cf::MemRefType>(value.getType())) {
                address_spaces.emplace(arg->GetArgName(),
                                       memref_type.getMemorySpace());
                continue;
            }

            auto target_type =
                parent_->GetContext()->syms->ResolveType(arg->GetAnnotation());
            if (!target_type) {
                continue;
            }

            if (mlir::isa<cf::MemRefType>(target_type)) {
                auto addressSpaceAttr = mlir::gpu::AddressSpaceAttr::get(
                    parent_->GetBuilder().getContext(),
                    mlir::gpu::AddressSpace::Private);
                address_spaces.emplace(arg->GetArgName(), addressSpaceAttr);
            }
        }
        return address_spaces;
    };

    auto &impl = parent_->GetParent();
    const auto &name = func->GetName();
    if (name.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call->GetSourceRange().getBegin())
            << "JIT function has no name";
        return mlir::Value();
    }
    auto arg_address_spaces = ArgAddressSpaceMap();
    llvm::SmallVector<mlir::Value> runtime_args;
    llvm::SmallVector<ast::Expr *> runtime_arg_exprs;
    ConstexprBindingMap constexpr_bindings;
    arg_address_spaces =
        collect_call_specialization(func, resolved_args, runtime_args,
                                    runtime_arg_exprs, constexpr_bindings);
    auto scope_prefix = parent_->GetQualifiedScopePrefix();
    auto mangled_name = impl.GetMangledFunctionName(
        func, &arg_address_spaces, scope_prefix, &constexpr_bindings);
    if (mangled_name.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call->GetSourceRange().getBegin())
            << "Failed to resolve JIT function name for '" << name << "'";
        return mlir::Value();
    }

    mlir::func::FuncOp func_op;
    auto func_it = impl.jit_function_ops_.find(mangled_name);
    if (func_it != impl.jit_function_ops_.end()) {
        func_op = func_it->second;
    }
    auto module = parent_->GetModule();
    if (!func_op && module) {
        func_op = module.lookupSymbol<mlir::func::FuncOp>(mangled_name);
        if (func_op) {
            impl.jit_function_ops_[mangled_name] = func_op;
        }
    }
    if (!func_op) {
        GeneratorContext::SymbolTableGuard sym_guard;
        if (ctx) {
            auto dep_it = impl.jit_function_deps_.find(name);
            if (dep_it != impl.jit_function_deps_.end() &&
                dep_it->second == func) {
                sym_guard =
                    ctx->GetSymbolTableGuard(impl.module_syms_->Clone());
            }
        }
        mlir::OpBuilder::InsertionGuard guard(parent_->GetBuilder());
        FunctionGenerator function_generator(
            impl, MLIRGenerator::FunctionType::kPrivateFunction,
            std::move(arg_address_spaces), scope_prefix,
            std::move(constexpr_bindings));
        function_generator.Generate(func);
        module = parent_->GetModule();
        if (module) {
            func_op = module.lookupSymbol<mlir::func::FuncOp>(mangled_name);
            if (func_op) {
                impl.jit_function_ops_[mangled_name] = func_op;
            }
        }
    }

    if (!func_op) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call->GetSourceRange().getBegin())
            << "Failed to generate JIT function '" << name << "'";
        return mlir::Value();
    }

    return GenerateFuncCall(call, func_op, runtime_args, runtime_arg_exprs);
}

mlir::Value ExprGenerator::VisitCall(ast::Call *call) {
    if (!call)
        return nullptr;

    // Resolve positional arguments once at the call site
    llvm::SmallVector<mlir::Value> resolved_args;
    resolved_args.reserve(call->GetArgs().size());
    for (auto *arg_expr : call->GetArgs()) {
        // Skip dispatch for type-like expressions (attribute access such as
        // avelang.f32 or factory calls like avelang.Tensor(...))
        if (llvm::isa<ast::AttributeExpr>(arg_expr)) {
            resolved_args.emplace_back();
            continue;
        }

        if (auto *call_arg = llvm::dyn_cast<ast::Call>(arg_expr)) {
            if (auto *attr_func =
                    llvm::dyn_cast<ast::AttributeExpr>(call_arg->GetFunc())) {
                auto attr_name = attr_func->GetAttr();
                if (attr_name == "Tensor" || attr_name == "Pointer" ||
                    attr_name == "Layout") {
                    resolved_args.emplace_back();
                    continue;
                }
            }
        }

        resolved_args.push_back(Dispatch(arg_expr));
    }

    // Try to resolve the function through the symbol table
    auto function =
        parent_->GetContext()->syms->ResolveFunction(call->GetFunc());
    if (function.IsValid() && IsGpuMmaIntrinsic(function)) {
        const auto &args = call->GetArgs();
        for (size_t i = 0; i < resolved_args.size() && i < args.size(); ++i) {
            if (!resolved_args[i]) {
                continue;
            }
            if (mlir::isa<cf::MemRefType>(resolved_args[i].getType())) {
                auto loaded = LoadVectorFromMemrefForMma(
                    parent_->GetBuilder(), parent_->GetContext(), call,
                    resolved_args[i], args[i]);
                if (!loaded) {
                    return nullptr;
                }
                resolved_args[i] = loaded;
            }
        }
    }

    if (function.IsValid()) {
        const auto &args = call->GetArgs();
        const auto &module_name = function.module_name;
        const auto &symbol_name = function.symbol_name;

        if (module_name == "amdgpu") {
            if (symbol_name.rfind("raw_buffer_load_x", 0) == 0) {
                if (resolved_args.size() > 0 && resolved_args[0] &&
                    mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
                    auto loaded = LoadVectorFromMemref(
                        parent_->GetBuilder(), parent_->GetContext(), call,
                        resolved_args[0], args[0], "raw_buffer_load");
                    if (!loaded) {
                        return nullptr;
                    }
                    resolved_args[0] = loaded;
                }
            } else if (symbol_name.rfind("raw_buffer_store_x", 0) == 0) {
                auto widthOpt = ParseIntAfterMarker(symbol_name, "_x");
                int width = widthOpt.value_or(0);
                if (resolved_args.size() > 0 && resolved_args[0] &&
                    mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
                    mlir::Value loaded;
                    if (width == 1) {
                        loaded = LoadScalarFromMemref(
                            parent_->GetBuilder(), parent_->GetContext(), call,
                            resolved_args[0], args[0], "raw_buffer_store");
                    } else {
                        loaded = LoadVectorFromMemref(
                            parent_->GetBuilder(), parent_->GetContext(), call,
                            resolved_args[0], args[0], "raw_buffer_store");
                    }
                    if (!loaded) {
                        return nullptr;
                    }
                    resolved_args[0] = loaded;
                }
                if (resolved_args.size() > 1 && resolved_args[1] &&
                    mlir::isa<cf::MemRefType>(resolved_args[1].getType())) {
                    auto loaded = LoadVectorFromMemref(
                        parent_->GetBuilder(), parent_->GetContext(), call,
                        resolved_args[1], args[1], "raw_buffer_store");
                    if (!loaded) {
                        return nullptr;
                    }
                    resolved_args[1] = loaded;
                }
            }
        } else if (module_name == "nvvm") {
            if (symbol_name.rfind("stmatrix_", 0) == 0) {
                auto numOpt = ParseIntAfterMarker(symbol_name, "_x");
                int num = numOpt.value_or(0);
                if (resolved_args.size() > 1 && resolved_args[1] &&
                    mlir::isa<cf::MemRefType>(resolved_args[1].getType())) {
                    mlir::Value loaded;
                    auto memref_type =
                        mlir::cast<cf::MemRefType>(resolved_args[1].getType());
                    bool elem_is_vector = mlir::isa<mlir::VectorType>(
                        memref_type.getElementType());
                    if (num == 1 && !elem_is_vector) {
                        loaded = LoadScalarFromMemref(
                            parent_->GetBuilder(), parent_->GetContext(), call,
                            resolved_args[1], args[1], "stmatrix");
                    } else {
                        loaded = LoadVectorFromMemref(
                            parent_->GetBuilder(), parent_->GetContext(), call,
                            resolved_args[1], args[1], "stmatrix");
                    }
                    if (!loaded) {
                        return nullptr;
                    }
                    resolved_args[1] = loaded;
                }
            }
        }
    }

    if (function.IsValid()) {
        if (!function.RunChecker(call, parent_->GetContext(), resolved_args)) {
            return nullptr;
        }
        // Invoke the registered function to generate the appropriate MLIR op
        return function.Invoke(call, parent_->GetContext(), resolved_args);
    }

    auto *name_expr = llvm::dyn_cast<ast::Name>(call->GetFunc());
    if (!name_expr) {
        Report(this, basic::DiagnosticCode::kUnimplemented,
               call->GetSourceRange().getBegin())
            << "Unsupported function call target";
        return nullptr;
    }

    // Look up the callee in the generated module
    auto symbol_name = name_expr->GetId();
    auto func_op =
        parent_->GetModule().lookupSymbol<mlir::func::FuncOp>(symbol_name);
    if (!func_op) {
        Report(this, basic::DiagnosticCode::kSymbolNotFound,
               call->GetSourceRange().getBegin())
            << symbol_name;
        return nullptr;
    }

    return GenerateFuncCall(call, func_op, resolved_args);
}

mlir::Value ExprGenerator::VisitCompare(ast::Compare *compare) {
    if (!compare)
        return nullptr;

    auto &builder = parent_->GetBuilder();

    // For now, only handle simple binary comparisons (left op right)
    // Python allows chained comparisons like a < b < c, but we'll implement
    // those later
    if (compare->GetOps().size() != 1 ||
        compare->GetComparators().size() != 1) {
        Report(this, basic::DiagnosticCode::kUnimplemented,
               compare->GetSourceRange().getBegin())
            << "Chained comparisons not yet supported";
        return nullptr;
    }

    // Generate operands
    auto lhs = Dispatch(compare->GetLeft());
    auto rhs = Dispatch(compare->GetComparators()[0]);

    if (!lhs || !rhs) {
        return nullptr;
    }

    const std::string &op = compare->GetOps()[0];
    auto location = GetMLIRLocation(compare);

    // Convert operands to compatible types for comparison
    auto [converted_lhs, converted_rhs] =
        ConvertTypesForArithmetic(lhs, rhs, builder, location);
    if (!converted_lhs || !converted_rhs) {
        Report(this, basic::DiagnosticCode::kUnimplemented,
               compare->GetSourceRange().getBegin())
            << "Cannot perform comparison with incompatible types";
        return nullptr;
    }

    auto result_type = converted_lhs.getType();

    // Generate the appropriate comparison operation based on operand type
    if (result_type.isIntOrIndex()) {
        // Integer comparisons
        auto predicate_it = kIntCompareOps.find(op);
        if (predicate_it == kIntCompareOps.end()) {
            Report(this, basic::DiagnosticCode::kUnimplemented,
                   compare->GetSourceRange().getBegin())
                << "Unsupported integer comparison operator: " << op;
            return nullptr;
        }

        auto predicate = predicate_it->second;
        if (GetTypeInfo(converted_lhs).is_unsigned_integer.value_or(false)) {
            if (op == "Gt") {
                predicate = mlir::arith::CmpIPredicate::ugt;
            } else if (op == "Lt") {
                predicate = mlir::arith::CmpIPredicate::ult;
            } else if (op == "GtE") {
                predicate = mlir::arith::CmpIPredicate::uge;
            } else if (op == "LtE") {
                predicate = mlir::arith::CmpIPredicate::ule;
            }
        }

        return mlir::arith::CmpIOp::create(builder, location, predicate,
                                           converted_lhs, converted_rhs);
    } else if (mlir::isa<mlir::FloatType>(result_type)) {
        // Float comparisons
        auto predicate_it = kFloatCompareOps.find(op);
        if (predicate_it == kFloatCompareOps.end()) {
            Report(this, basic::DiagnosticCode::kUnimplemented,
                   compare->GetSourceRange().getBegin())
                << "Unsupported float comparison operator: " << op;
            return nullptr;
        }

        return mlir::arith::CmpFOp::create(builder, location,
                                           predicate_it->second, converted_lhs,
                                           converted_rhs);
    }

    // Unsupported type for comparison
    Report(this, basic::DiagnosticCode::kUnimplemented,
           compare->GetSourceRange().getBegin())
        << "Unsupported type for comparison operation";
    return nullptr;
}

mlir::Value ExprGenerator::VisitIfExp(ast::IfExp *if_exp) {
    if (!if_exp)
        return nullptr;

    auto &builder = parent_->GetBuilder();
    auto location = GetMLIRLocation(if_exp);

    // Generate the condition expression
    auto condition = Dispatch(if_exp->GetTest());
    if (!condition) {
        Report(this, basic::DiagnosticCode::kUnimplemented,
               if_exp->GetSourceRange().getBegin())
            << "Failed to generate condition expression for conditional "
               "expression";
        return nullptr;
    }

    // Generate the true value expression
    auto true_value = Dispatch(if_exp->GetBody());
    if (!true_value) {
        Report(this, basic::DiagnosticCode::kUnimplemented,
               if_exp->GetSourceRange().getBegin())
            << "Failed to generate true value expression for conditional "
               "expression";
        return nullptr;
    }

    // Generate the false value expression
    auto false_value = Dispatch(if_exp->GetOrelse());
    if (!false_value) {
        Report(this, basic::DiagnosticCode::kUnimplemented,
               if_exp->GetSourceRange().getBegin())
            << "Failed to generate false value expression for conditional "
               "expression";
        return nullptr;
    }

    // Convert condition to i1 if needed
    if (!condition.getType().isInteger(1)) {
        // Convert to boolean - non-zero values are true
        if (condition.getType().isInteger()) {
            auto zeroAttr = builder.getIntegerAttr(condition.getType(), 0);
            auto zero =
                mlir::arith::ConstantOp::create(builder, location, zeroAttr);
            condition = mlir::arith::CmpIOp::create(
                builder, location, mlir::arith::CmpIPredicate::ne, condition,
                zero);
        } else {
            Report(this, basic::DiagnosticCode::kUnimplemented,
                   if_exp->GetSourceRange().getBegin())
                << "Unsupported condition type for conditional expression";
            return nullptr;
        }
    }

    // Check if true and false values can be converted to compatible types
    auto true_type = true_value.getType();
    auto false_type = false_value.getType();

    // Try to convert true and false values to compatible types
    auto [converted_true, converted_false] =
        ConvertTypesForArithmetic(true_value, false_value, builder, location);
    if (!converted_true || !converted_false) {
        // Provide a more specific error message for conditional expressions
        std::string true_type_str, false_type_str;
        llvm::raw_string_ostream true_stream(true_type_str),
            false_stream(false_type_str);
        true_type.print(true_stream);
        false_type.print(false_stream);
        true_stream.flush();
        false_stream.flush();

        Report(this, basic::DiagnosticCode::kUnimplemented,
               if_exp->GetSourceRange().getBegin())
            << "Conditional expression branches have incompatible types: true "
               "branch has type '"
            << true_type_str << "', false branch has type '" << false_type_str
            << "' - these types cannot converge to a common type";
        return nullptr;
    }

    // Create the select operation: arith.select condition, true_value,
    // false_value
    auto select = mlir::arith::SelectOp::create(
        builder, location, condition, converted_true, converted_false);
    SetTypeInfo(select.getResult(), GetTypeInfo(converted_true));
    return select;
}

// Clean up macro to avoid polluting global namespace
#undef DEFINE_BINARY_OP

// Stub methods for import statements (these should never be called for
// expressions)
mlir::Value ExprGenerator::VisitImport(ast::Import *) {
    SS_ASSERT(false &&
              "Import statements should not be evaluated as expressions");
    return nullptr;
}

mlir::Value ExprGenerator::VisitImportFrom(ast::ImportFrom *) {
    SS_ASSERT(false &&
              "ImportFrom statements should not be evaluated as expressions");
    return nullptr;
}

} // namespace causalflow::avelang::ir
