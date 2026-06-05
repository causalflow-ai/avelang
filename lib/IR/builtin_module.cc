#include "builtin_module.h"
#include "AST/ast_nodes_expr.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "Utils/assert.h"
#include "constant_folder.h"
#include "generator_context.h"
#include "layout_operation.h"
#include "mlir_generator_impl.h"
#include "parsing_utils.h"
#include "type_promotion.h"
#include "type_system.h"

#include <algorithm>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/Ptr/IR/PtrAttrs.h>
#include <mlir/Dialect/Ptr/IR/PtrTypes.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <tuple>

namespace causalflow::avelang::ir {

using namespace mlir;
namespace cf = causalflow::avelang::dialect;

// Helper function to generate row-major stride tuples matching nested structure
static void
generateRowMajorStrides(const llvm::SmallVector<int64_t> &shapeValues,
                        llvm::SmallVector<mlir::Value> &strideValues,
                        mlir::OpBuilder &builder, mlir::Location location) {

    int64_t currentStride = 1;
    for (auto it = shapeValues.rbegin(); it != shapeValues.rend(); ++it) {
        strideValues.insert(strideValues.begin(),
                            mlir::arith::ConstantIndexOp::create(
                                builder, location, currentStride));

        if (*it != mlir::ShapedType::kDynamic) {
            currentStride *= *it;
        } else {
            currentStride = mlir::ShapedType::kDynamic;
        }
    }
}

static std::optional<llvm::SmallVector<int64_t>>
foldStaticShapeTuple(ast::Expr *shapeExpr, GeneratorContext *ctx) {
    auto *tupleExpr = llvm::dyn_cast_if_present<ast::Tuple>(shapeExpr);
    if (!tupleExpr) {
        return std::nullopt;
    }

    ConstantFolder folder(ctx);
    llvm::SmallVector<int64_t> shape;
    shape.reserve(tupleExpr->GetElts().size());
    for (auto *dimExpr : tupleExpr->GetElts()) {
        auto dimValue = folder.Evaluate(dimExpr);
        if (!dimValue) {
            return std::nullopt;
        }
        shape.push_back(*dimValue);
    }
    return shape;
}

static mlir::Value createStaticIntTuple(llvm::ArrayRef<int64_t> values,
                                        mlir::OpBuilder &builder,
                                        mlir::Location location) {
    llvm::SmallVector<mlir::Value> tupleValues;
    tupleValues.reserve(values.size());
    for (int64_t value : values) {
        tupleValues.push_back(
            mlir::arith::ConstantIndexOp::create(builder, location, value));
    }
    return cf::MakeIntTupleOp::create(builder, location, tupleValues)
        .getResult();
}

// Helper to resolve and validate element type for tensor operations
static mlir::Type resolveAndValidateTensorElementType(
    ast::Expr *dtype_expr, GeneratorContext *ctx,
    clang::SourceLocation location, const std::string &context_name) {
    mlir::Type elementType = ctx->syms->ResolveBuiltinType(dtype_expr);
    return elementType;
}

// Helper to create ave memref type with optional memory space.
static cf::MemRefType
createTensorMemRefType(llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                       mlir::Attribute memorySpaceAttr = {},
                       llvm::ArrayRef<int64_t> strides = {}) {
    auto layoutType =
        cf::LayoutType::get(elementType.getContext(), shape, strides);
    return cf::MemRefType::get(elementType.getContext(), layoutType,
                               elementType, memorySpaceAttr);
}

static llvm::SmallVector<int64_t>
computeRowMajorStrides(llvm::ArrayRef<int64_t> shape) {
    llvm::SmallVector<int64_t> strides;
    strides.reserve(shape.size());
    int64_t current = 1;
    for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
        strides.insert(strides.begin(), current);
        if (*it == mlir::ShapedType::kDynamic ||
            current == mlir::ShapedType::kDynamic) {
            current = mlir::ShapedType::kDynamic;
        } else {
            current *= *it;
        }
    }
    return strides;
}

static mlir::Location GetCallLocation(GeneratorContext *ctx,
                                      const ast::ASTNode *node) {
    auto *func_gen = ctx ? ctx->GetCurrentFunctionGenerator() : nullptr;
    auto *builder = func_gen ? &func_gen->GetBuilder() : nullptr;
    SS_ASSERT(ctx && builder);
    return ctx->GetMLIRLocation(builder->getContext(), node);
}

static std::optional<int64_t> getElementByteSize(mlir::Type elementType) {
    if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(elementType)) {
        auto scalarType = vectorType.getElementType();
        if (!scalarType.isIntOrFloat()) {
            return std::nullopt;
        }
        int64_t bitWidth = scalarType.getIntOrFloatBitWidth();
        if (bitWidth % 8 != 0) {
            return std::nullopt;
        }
        return (bitWidth / 8) * vectorType.getNumElements();
    }

    if (elementType.isIndex()) {
        return 8;
    }

    if (!elementType.isIntOrFloat()) {
        return std::nullopt;
    }

    int64_t bitWidth = elementType.getIntOrFloatBitWidth();
    if (bitWidth % 8 != 0) {
        return std::nullopt;
    }
    return bitWidth / 8;
}

static std::optional<int64_t>
getStaticTotalByteSize(llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
    auto elementBytes = getElementByteSize(elementType);
    if (!elementBytes) {
        return std::nullopt;
    }

    int64_t elementCount = 1;
    for (auto dim : shape) {
        if (mlir::ShapedType::isDynamic(dim) || dim < 0) {
            return std::nullopt;
        }
        elementCount *= dim;
    }
    return elementCount * (*elementBytes);
}

template <typename MathOp>
static mlir::Value
CreateUnaryFloatMathFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args,
                             llvm::StringRef functionName) {
    if (call_expr->GetArgs().size() != 1 || resolved_args.size() != 1 ||
        !resolved_args[0]) {
        std::string message =
            ("avelang." + functionName + "() expects exactly 1 argument").str();
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
        return nullptr;
    }

    auto value = resolved_args[0];
    if (!mlir::isa<mlir::FloatType>(value.getType())) {
        std::string message =
            ("avelang." + functionName + "() expects a floating-point argument")
                .str();
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    return MathOp::create(builder, GetCallLocation(ctx, call_expr), value)
        .getResult();
}

// Helper function for GPU functions that require dimension argument processing
template <typename OpType>
static mlir::Value
CreateGPUDimensionFunction(ast::Call *call_expr, GeneratorContext *ctx,
                           llvm::ArrayRef<mlir::Value> resolved_args,
                           const std::string &function_name) {

    auto location = GetCallLocation(ctx, call_expr);
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();

    const auto &args = call_expr->GetArgs();

    // Validate argument count
    if (args.size() != 1) {
        std::string error_msg =
            function_name + "() expects exactly 1 argument (dimension)";
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << error_msg;
        return nullptr;
    }

    if (resolved_args.size() != 1 || !resolved_args[0]) {
        std::string error_msg =
            "Failed to generate dimension argument for " + function_name + "()";
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << error_msg;
        return nullptr;
    }

    auto dim_arg = resolved_args[0];

    // Extract dimension value - it should be a constant
    int64_t dimension_value = -1;
    if (auto const_op = dim_arg.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr =
                mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            dimension_value = int_attr.getInt();
        }
    }

    // Convert to GPU dimension enum with range check
    if (dimension_value < 0 || dimension_value > 2) {
        std::string error_msg =
            function_name +
            "() dimension must be a constant integer 0, 1, or 2";
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << error_msg;
        return nullptr;
    }

    gpu::Dimension dimension = static_cast<gpu::Dimension>(dimension_value);
    return OpType::create(builder, location, builder.getIndexType(), dimension);
}

// Helper function for creating syncthreads operation
static mlir::Value CreateSyncthreadsFunction(ast::Call *call_expr,
                                             GeneratorContext *ctx) {
    auto location = GetCallLocation(ctx, call_expr);
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();

    const auto &args = call_expr->GetArgs();

    // Validate argument count - syncthreads() should take no arguments
    if (args.size() != 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "syncthreads() expects no arguments";
        return nullptr;
    }

    // Create gpu.barrier operation
    gpu::BarrierOp::create(builder, location);

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

static mlir::Value
CreatePrintfFunction(ast::Call *call_expr, GeneratorContext *ctx,
                     llvm::ArrayRef<mlir::Value> resolved_args) {
    auto location = GetCallLocation(ctx, call_expr);
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();

    const auto &args = call_expr->GetArgs();

    // Validate argument count - printf requires at least format string
    if (args.size() < 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "printf() requires at least a format string argument";
        return nullptr;
    }

    if (resolved_args.size() < args.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve printf() arguments";
        return nullptr;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve printf() format string";
        return nullptr;
    }

    auto format_value = resolved_args[0];
    auto cast_op =
        format_value.getDefiningOp<mlir::UnrealizedConversionCastOp>();
    if (!cast_op || !cast_op->hasAttr("ave.string_literal")) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "printf() format argument must be a string literal";
        return nullptr;
    }

    auto format_attr =
        cast_op->getAttrOfType<mlir::StringAttr>("ave.string_literal");
    if (!format_attr || format_attr.getValue().empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "printf() format string cannot be empty";
        return nullptr;
    }

    // Generate values for the remaining arguments
    llvm::SmallVector<mlir::Value> printf_args;
    for (size_t i = 1; i < args.size(); ++i) {
        auto arg_value = resolved_args[i];
        if (!arg_value) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate printf() argument " << i;
            return nullptr;
        }
        printf_args.push_back(arg_value);
    }

    // Create gpu.printf operation
    gpu::PrintfOp::create(builder, location, format_attr, printf_args);

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

template <typename SignedOpType, typename UnsignedOpType>
static mlir::Value
CreateIntegerExtremaFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args,
                             llvm::StringRef function_name) {
    if (call_expr->GetArgs().size() != 2 || resolved_args.size() != 2 ||
        !resolved_args[0] || !resolved_args[1]) {
        std::string message = ("avelang." + function_name +
                               "() expects exactly 2 integer arguments")
                                  .str();
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
        return nullptr;
    }

    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto [lhs, rhs] = causalflow::avelang::ir::ConvertTypesForBitwise(
        resolved_args[0], resolved_args[1], builder, location);
    if (!lhs || !rhs) {
        std::string message = ("avelang." + function_name +
                               "() requires compatible integer operands")
                                  .str();
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
        return nullptr;
    }

    if (!IsIntegerValueOrVectorOfIntegers(lhs.getType()) &&
        !lhs.getType().isIndex()) {
        std::string message = ("avelang." + function_name +
                               "() only supports integer or index operands")
                                  .str();
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << message;
        return nullptr;
    }

    bool isUnsigned = GetTypeInfo(lhs).is_unsigned_integer.value_or(false);
    if (lhs.getType().isIndex()) {
        auto predicate = function_name == "min"
                             ? mlir::arith::CmpIPredicate::slt
                             : mlir::arith::CmpIPredicate::sgt;
        auto cmp =
            mlir::arith::CmpIOp::create(builder, location, predicate, lhs, rhs);
        return mlir::arith::SelectOp::create(builder, location, cmp, lhs, rhs);
    }

    mlir::Value result;
    if (isUnsigned) {
        result = UnsignedOpType::create(builder, location, lhs, rhs);
    } else {
        result = SignedOpType::create(builder, location, lhs, rhs);
    }
    SetTypeInfo(result, TypeInfo{isUnsigned});
    return result;
}

// AveLangModule Implementation
AveLangModule::AveLangModule(IRContext *ir_context)
    : NamedModule("avelang"), ir_context_(ir_context) {}

void AveLangModule::Initialize() {
    SS_ASSERT(ir_context_ && "IRContext must be set for AveLangModule");
    OpBuilder builder(ir_context_->GetMLIRContext());

    // Scalar types
    AddType("i8", builder.getI8Type());
    AddType("i16", builder.getI16Type());
    AddType("i32", builder.getI32Type());
    AddType("i64", builder.getI64Type());

    // MLIR uses integer types for both signed and unsigned integers. Therefore
    // we tag the type at the definition of values
    AddType("u8", builder.getI8Type());
    AddType("u16", builder.getI16Type());
    AddType("u32", builder.getI32Type());
    AddType("u64", builder.getI64Type());

    AddType("f16", builder.getF16Type());
    AddType("f32", builder.getF32Type());
    AddType("f64", builder.getF64Type());
    AddType("bf16", builder.getBF16Type());
    AddType("f8e4m3fn", Float8E4M3FNType::get(builder.getContext()));
    AddType("f8e4m3fnuz", Float8E4M3FNUZType::get(builder.getContext()));

    // FIXME: Constexpr type (placeholder, currently i32)
    AddType("constexpr", builder.getI32Type());

    AddSymbol("jit", [&builder]() -> mlir::Value {
        // special marker for @avelang.jit
        auto jitOp = UnrealizedConversionCastOp::create(
            builder, builder.getUnknownLoc(), TypeRange{builder.getNoneType()},
            ValueRange{});

        jitOp->setAttr("marker_purpose", builder.getStringAttr("jit"));
        jitOp->setAttr("is_marker", builder.getBoolAttr(true));
        jitOp->setAttr("tags", builder.getArrayAttr({}));
        return jitOp.getResult(0);
    }());

    // Add Tensor type factory function
    AddTypeFactory(
        "Tensor",
        [](NamedModule *module, ast::Call *call_expr,
           GeneratorContext *gen_ctx) -> mlir::Type {
            auto *avelang_module = static_cast<AveLangModule *>(module);
            return avelang_module->CreateTensorType(call_expr, gen_ctx);
        });

    // Add Pointer type factory function
    AddTypeFactory(
        "Pointer",
        [](NamedModule *module, ast::Call *call_expr,
           GeneratorContext *gen_ctx) -> mlir::Type {
            auto *avelang_module = static_cast<AveLangModule *>(module);
            return avelang_module->CreatePointerType(call_expr, gen_ctx);
        });

    // Add Layout type factory function
    AddTypeFactory(
        "Layout",
        [](NamedModule *module, ast::Call *call_expr,
           GeneratorContext *gen_ctx) -> mlir::Type {
            auto *avelang_module = static_cast<AveLangModule *>(module);
            return avelang_module->CreateLayoutType(call_expr, gen_ctx);
        });

    // Add function wrappers for convert and bitcast
    AddFunction(
        "convert",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateConvertFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "bitcast",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateBitcastFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "abs",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateAbsFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "exp",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateExpFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "exp2",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateExp2Function(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "tanh",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateTanhFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "log",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateLogFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "log2",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateLog2Function(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "erf",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateErfFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "sqrt",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateSqrtFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "range",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRangeFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "shuffle",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateShuffleFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "shuffle_up",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateShuffleUpFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "shuffle_down",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateShuffleDownFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "shuffle_xor",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateShuffleXorFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction("min",
                [](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                    return CreateIntegerExtremaFunction<mlir::arith::MinSIOp,
                                                        mlir::arith::MinUIOp>(
                        call_expr, gen_ctx, resolved_args, "min");
                });

    AddFunction("max",
                [](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                    return CreateIntegerExtremaFunction<mlir::arith::MaxSIOp,
                                                        mlir::arith::MaxUIOp>(
                        call_expr, gen_ctx, resolved_args, "max");
                });

    AddFunction("block_id",
                [](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                    return CreateGPUDimensionFunction<gpu::BlockIdOp>(
                        call_expr, gen_ctx, resolved_args, "block_id");
                });

    AddFunction("thread_id",
                [](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                    return CreateGPUDimensionFunction<gpu::ThreadIdOp>(
                        call_expr, gen_ctx, resolved_args, "thread_id");
                });

    AddFunction("block_dim",
                [](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                    return CreateGPUDimensionFunction<gpu::BlockDimOp>(
                        call_expr, gen_ctx, resolved_args, "block_dim");
                });

    AddFunction("grid_dim",
                [](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                    return CreateGPUDimensionFunction<gpu::GridDimOp>(
                        call_expr, gen_ctx, resolved_args, "grid_dim");
                });

    AddFunction("syncthreads",
                [](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value>) -> mlir::Value {
                    return CreateSyncthreadsFunction(call_expr, gen_ctx);
                });

    AddFunction("printf",
                [](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                    return CreatePrintfFunction(call_expr, gen_ctx,
                                                resolved_args);
                });

    AddFunction(
        "make_tensor",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMakeTensorFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "make_local",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMakeLocalFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "make_shared",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMakeSharedFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "subview",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateSubviewFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction("make_layout",
                [](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                    return LayoutOperation::createMakeLayoutFunction(
                        call_expr, gen_ctx, resolved_args);
                });

    AddFunction(
        "full",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateFullFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "view",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateViewFunction(call_expr, gen_ctx, resolved_args);
        });

    // Mirror Python package structure: expose DSL constructs under
    // avelang.language in addition to the root module.
    AddModule("language", this);

    // Register NVVM intrinsic module as a submodule
    nvvm_module_ = CreateNVVMIntrinsicModule();
    nvvm_module_->Initialize();
    AddModule("nvvm", nvvm_module_.get());

    // Register AMDGPU intrinsic module as a submodule
    amdgpu_module_ = CreateAMDGPUIntrinsicModule();
    amdgpu_module_->Initialize();
    AddModule("amdgpu", amdgpu_module_.get());
}

void AveLangModule::DeclareModules(mlir::ModuleOp module) {
    if (!module)
        return;

    if (nvvm_module_) {
        nvvm_module_->DeclareModules(module);
    }
    if (amdgpu_module_) {
        amdgpu_module_->DeclareModules(module);
    }
}

mlir::Type AveLangModule::CreateTensorType(ast::Call *call_expr,
                                           GeneratorContext *ctx) const {
    if (!call_expr)
        return mlir::Type();

    // Expect avelang.Tensor((M,N), dtype) format
    const auto &args = call_expr->GetArgs();
    if (args.size() != 2) {
        return mlir::Type(); // Invalid number of arguments
    }

    auto *shape_expr = args[0];
    auto *dtype_expr = args[1];

    mlir::Type element_type = resolveAndValidateTensorElementType(
        dtype_expr, ctx, call_expr->GetSourceRange().getBegin(), "Tensor");
    if (!element_type) {
        return mlir::Type();
    }

    // Parse shape from first argument
    llvm::SmallVector<int64_t> shape;
    if (auto *tuple_expr = llvm::dyn_cast<ast::Tuple>(shape_expr)) {
        ConstantFolder folder(ctx);
        for (auto *dim_expr : tuple_expr->GetElts()) {
            if (auto dim_value = folder.Evaluate(dim_expr)) {
                shape.push_back(*dim_value);
            } else {
                shape.push_back(mlir::ShapedType::kDynamic);
            }
        }
    } else {
        return mlir::Type(); // Shape must be a tuple
    }

    // Create memref type with the parsed shape and element type.
    auto strides = computeRowMajorStrides(shape);
    return createTensorMemRefType(shape, element_type, {}, strides);
}

mlir::Type AveLangModule::CreatePointerType(ast::Call *call_expr,
                                            GeneratorContext *ctx) const {
    if (!call_expr)
        return mlir::Type();

    const auto &args = call_expr->GetArgs();
    if (args.size() != 1) {
        return mlir::Type();
    }

    auto *dtype_expr = args[0];
    mlir::Type element_type = resolveAndValidateTensorElementType(
        dtype_expr, ctx, call_expr->GetSourceRange().getBegin(), "Pointer");
    if (!element_type) {
        return mlir::Type();
    }

    auto *mlirContext = element_type.getContext();
    return mlir::ptr::PtrType::get(mlirContext, nullptr);
}

mlir::Type AveLangModule::CreateLayoutType(ast::Call *call_expr,
                                           GeneratorContext *ctx) const {
    if (!call_expr)
        return mlir::Type();

    // Expect avelang.Layout((Dims), (Stride)) format
    const auto &args = call_expr->GetArgs();
    if (args.size() != 2) {
        return mlir::Type(); // Invalid number of arguments
    }

    auto *dims_expr = args[0];
    auto *stride_expr = args[1];

    // Validate that both arguments are tuples
    if (!llvm::isa<ast::Tuple>(dims_expr) ||
        !llvm::isa<ast::Tuple>(stride_expr)) {
        return mlir::Type(); // Both must be tuples
    }

    auto *dims_tuple = llvm::dyn_cast<ast::Tuple>(dims_expr);
    auto *stride_tuple = llvm::dyn_cast<ast::Tuple>(stride_expr);

    // Validate that both tuples have the same number of elements
    if (dims_tuple->GetElts().size() != stride_tuple->GetElts().size()) {
        return mlir::Type(); // Must have same dimensions
    }

    if (dims_tuple->GetElts().empty()) {
        return mlir::Type(); // Cannot be empty
    }

    // For Layout type, we return an opaque type that represents the layout
    // We use a custom MLIR type (OpaqueType) to store layout information
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    return mlir::OpaqueType::get(builder.getStringAttr("ave"), "layout");
}

mlir::Value AveLangModule::CreateConvertFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    const auto &args = call_expr->GetArgs();

    // avelang.convert(value, target_type)
    if (args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "avelang.convert requires exactly 2 arguments: value and "
               "target_type";
        return nullptr;
    }

    if (resolved_args.size() < 2 || !resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve value for convert";
        return nullptr;
    }

    auto value = resolved_args[0];

    // Resolve the target type from the second argument
    auto target_type = ctx->syms->ResolveType(args[1]);
    if (!target_type) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve target type for convert";
        return nullptr;
    }

    auto source_type = value.getType();
    auto location = GetCallLocation(ctx, call_expr);

    // Explicit type conversion (allows demotion since user explicitly requested
    // it)
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    std::optional<bool> targetUnsigned;
    if (auto typeInfo = GetTypeInfo(args[1]); typeInfo.is_unsigned_integer) {
        targetUnsigned = typeInfo.is_unsigned_integer;
    }

    auto result = causalflow::avelang::ir::CreateTypeConversion(
        value, source_type, target_type, location, builder, true,
        GetTypeInfo(value).is_unsigned_integer, targetUnsigned);
    if (!result) {
        std::string source_str, target_str;
        llvm::raw_string_ostream source_os(source_str), target_os(target_str);
        source_type.print(source_os);
        target_type.print(target_os);
        std::string error_msg = "Failed to convert from type " +
                                source_os.str() + " to type " + target_os.str();
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << error_msg;
        return nullptr;
    }
    return result;
}

mlir::Value AveLangModule::CreateBitcastFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    const auto &args = call_expr->GetArgs();

    // avelang.bitcast(value, target_type)
    if (args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "avelang.bitcast requires exactly 2 arguments: value and "
               "target_type";
        return nullptr;
    }

    if (resolved_args.size() < 2 || !resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve value for bitcast";
        return nullptr;
    }

    auto value = resolved_args[0];

    // Resolve the target type from the second argument
    auto target_type = ctx->syms->ResolveType(args[1]);
    if (!target_type) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve target type for bitcast";
        return nullptr;
    }

    auto location = GetCallLocation(ctx, call_expr);

    // Bitcast - reinterpret bits without conversion
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto bitcast =
        mlir::arith::BitcastOp::create(builder, location, target_type, value);
    if (auto targetTypeInfo = GetTypeInfo(args[1]);
        targetTypeInfo.is_unsigned_integer) {
        SetTypeInfo(bitcast.getResult(), targetTypeInfo);
    }
    return bitcast.getResult();
}

mlir::Value AveLangModule::CreateAbsFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateUnaryFloatMathFunction<mlir::math::AbsFOp>(
        call_expr, ctx, resolved_args, "abs");
}

mlir::Value AveLangModule::CreateExpFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateUnaryFloatMathFunction<mlir::math::ExpOp>(
        call_expr, ctx, resolved_args, "exp");
}

mlir::Value AveLangModule::CreateExp2Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateUnaryFloatMathFunction<mlir::math::Exp2Op>(
        call_expr, ctx, resolved_args, "exp2");
}

mlir::Value AveLangModule::CreateTanhFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateUnaryFloatMathFunction<mlir::math::TanhOp>(
        call_expr, ctx, resolved_args, "tanh");
}

mlir::Value AveLangModule::CreateLogFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateUnaryFloatMathFunction<mlir::math::LogOp>(
        call_expr, ctx, resolved_args, "log");
}

mlir::Value AveLangModule::CreateLog2Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateUnaryFloatMathFunction<mlir::math::Log2Op>(
        call_expr, ctx, resolved_args, "log2");
}

mlir::Value AveLangModule::CreateErfFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateUnaryFloatMathFunction<mlir::math::ErfOp>(
        call_expr, ctx, resolved_args, "erf");
}

mlir::Value AveLangModule::CreateSqrtFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateUnaryFloatMathFunction<mlir::math::SqrtOp>(
        call_expr, ctx, resolved_args, "sqrt");
}

mlir::Value AveLangModule::CreateRangeFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto location = GetCallLocation(ctx, call_expr);
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();

    const auto &args = call_expr->GetArgs();

    // Validate argument count
    if (args.size() < 1 || args.size() > 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "range() expects 1, 2, or 3 arguments";
        return nullptr;
    }

    if (resolved_args.size() < args.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve range arguments";
        return nullptr;
    }

    mlir::SmallVector<mlir::Value> arg_values;
    for (size_t i = 0; i < args.size(); ++i) {
        auto value = resolved_args[i];
        if (!value) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate range argument";
            return nullptr;
        }
        arg_values.push_back(value);
    }

    // Parse arguments into lower_bound, upper_bound, step based on the three
    // variants
    mlir::Value lower_bound, upper_bound, step;

    if (args.size() == 1) {
        // range(stop)
        lower_bound =
            mlir::arith::ConstantIndexOp::create(builder, location, 0);
        upper_bound = arg_values[0];
        step = mlir::arith::ConstantIndexOp::create(builder, location, 1);
    } else if (args.size() == 2) {
        // range(start, stop)
        lower_bound = arg_values[0];
        upper_bound = arg_values[1];
        step = mlir::arith::ConstantIndexOp::create(builder, location, 1);
    } else { // args.size() == 3
        // range(start, stop, step)
        lower_bound = arg_values[0];
        upper_bound = arg_values[1];
        step = arg_values[2];
    }

    // Convert to index type if needed
    if (!lower_bound.getType().isIndex()) {
        lower_bound = mlir::arith::IndexCastOp::create(
            builder, location, builder.getIndexType(), lower_bound);
    }
    if (!upper_bound.getType().isIndex()) {
        upper_bound = mlir::arith::IndexCastOp::create(
            builder, location, builder.getIndexType(), upper_bound);
    }
    if (!step.getType().isIndex()) {
        step = mlir::arith::IndexCastOp::create(builder, location,
                                                builder.getIndexType(), step);
    }

    // Create the range object with normalized lower_bound, upper_bound, step
    mlir::SmallVector<mlir::Value> range_values = {lower_bound, upper_bound,
                                                   step};
    auto range_op = mlir::UnrealizedConversionCastOp::create(
        builder, location, mlir::TypeRange{builder.getNoneType()},
        mlir::ValueRange(range_values));

    range_op->setAttr("avelang_range", builder.getBoolAttr(true));

    return range_op.getResult(0);
}

static mlir::Value
CreateShuffleFunctionImpl(ast::Call *call_expr, GeneratorContext *ctx,
                          llvm::ArrayRef<mlir::Value> resolved_args,
                          llvm::StringRef functionName,
                          mlir::gpu::ShuffleMode mode) {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    const auto &args = call_expr->GetArgs();

    if (args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << functionName << "() requires 3 arguments: value, offset, width";
        return nullptr;
    }

    if (resolved_args.size() != args.size() || !resolved_args[0] ||
        !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for " << functionName << "()";
        return nullptr;
    }

    auto value = resolved_args[0];
    auto valueType = value.getType();
    auto vectorType = mlir::dyn_cast<mlir::VectorType>(valueType);
    bool validValueType = valueType.isIntOrFloat() ||
                          (vectorType && vectorType.getRank() == 1 &&
                           vectorType.getElementType().isIntOrFloat());
    if (!validValueType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << functionName
            << "() expects value to be an integer, float, or 1D vector";
        return nullptr;
    }

    auto normalizeToI32 = [&](mlir::Value input,
                              llvm::StringRef operandName) -> mlir::Value {
        auto inputType = input.getType();
        if (inputType.isIndex()) {
            return mlir::arith::IndexCastOp::create(
                builder, location, builder.getI32Type(), input);
        }

        auto intType = mlir::dyn_cast<mlir::IntegerType>(inputType);
        if (!intType) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << functionName << "() expects " << operandName
                << " to be an integer or index value";
            return nullptr;
        }

        if (intType.getWidth() < 32) {
            return mlir::arith::ExtUIOp::create(builder, location,
                                                builder.getI32Type(), input);
        }
        if (intType.getWidth() > 32) {
            return mlir::arith::TruncIOp::create(builder, location,
                                                 builder.getI32Type(), input);
        }
        if (inputType != builder.getI32Type()) {
            return mlir::arith::BitcastOp::create(builder, location,
                                                  builder.getI32Type(), input);
        }
        return input;
    };

    auto offset = normalizeToI32(resolved_args[1], "offset");
    auto width = normalizeToI32(resolved_args[2], "width");
    if (!offset || !width) {
        return nullptr;
    }

    auto shuffle = mlir::gpu::ShuffleOp::create(builder, location, value,
                                                offset, width, mode);
    return shuffle.getShuffleResult();
}

mlir::Value AveLangModule::CreateShuffleFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateShuffleFunctionImpl(call_expr, ctx, resolved_args, "shuffle",
                                     mlir::gpu::ShuffleMode::IDX);
}

mlir::Value AveLangModule::CreateShuffleUpFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateShuffleFunctionImpl(call_expr, ctx, resolved_args,
                                     "shuffle_up", mlir::gpu::ShuffleMode::UP);
}

mlir::Value AveLangModule::CreateShuffleDownFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateShuffleFunctionImpl(call_expr, ctx, resolved_args,
                                     "shuffle_down",
                                     mlir::gpu::ShuffleMode::DOWN);
}

mlir::Value AveLangModule::CreateShuffleXorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateShuffleFunctionImpl(call_expr, ctx, resolved_args,
                                     "shuffle_xor",
                                     mlir::gpu::ShuffleMode::XOR);
}

mlir::Value AveLangModule::CreateMakeTensorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto location = GetCallLocation(ctx, call_expr);
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();

    const auto &args = call_expr->GetArgs();
    if (args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tensor() expects (pointer, dtype, layout)";
        return nullptr;
    }

    mlir::Value ptrValue =
        resolved_args.size() > 0 ? resolved_args[0] : mlir::Value();
    mlir::Value layoutValue =
        resolved_args.size() > 2 ? resolved_args[2] : mlir::Value();
    if (!ptrValue || !layoutValue) {
        auto *expr_generator =
            ctx->GetCurrentFunctionGenerator()->GetExprGenerator();
        if (!ptrValue) {
            ptrValue = expr_generator->Dispatch(args[0]);
        }
        if (!layoutValue) {
            layoutValue = expr_generator->Dispatch(args[2]);
        }
    }
    if (!ptrValue || !layoutValue) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve make_tensor() arguments";
        return nullptr;
    }

    auto elementType = resolveAndValidateTensorElementType(
        args[1], ctx, call_expr->GetSourceRange().getBegin(), "make_tensor");
    if (!elementType) {
        return nullptr;
    }

    auto ptrMemRefType = mlir::dyn_cast<cf::MemRefType>(ptrValue.getType());
    auto ptrPtrType = mlir::dyn_cast<mlir::ptr::PtrType>(ptrValue.getType());
    if (ptrMemRefType) {
        if (!ptrMemRefType.getElementType().isInteger(8)) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "make_tensor() expects an i8 memref pointer argument";
            return nullptr;
        }
    } else if (!ptrPtrType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tensor() expects a pointer argument of ptr type or i8 "
               "memref type";
        return nullptr;
    }

    auto layoutOp = layoutValue.getDefiningOp<cf::MakeLayoutOp>();
    if (!layoutOp) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        args[2]->GetSourceRange().getBegin())
            << "make_tensor() expects a layout from make_layout()";
        return nullptr;
    }

    llvm::SmallVector<int64_t> shapeValues;
    llvm::SmallVector<int64_t> strideValues;
    if (!LayoutOperation::flattenTupleValues(layoutOp.getDims(), shapeValues) ||
        !LayoutOperation::flattenTupleValues(layoutOp.getStride(),
                                             strideValues)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        args[2]->GetSourceRange().getBegin())
            << "Failed to extract layout values for make_tensor()";
        return nullptr;
    }

    mlir::Attribute memorySpaceAttr;
    if (ptrMemRefType) {
        memorySpaceAttr = ptrMemRefType.getMemorySpace();
    }
    auto resultType = createTensorMemRefType(shapeValues, elementType,
                                             memorySpaceAttr, strideValues);

    auto castOp = cf::AveLangMemRefCastOp::create(builder, location, ptrValue,
                                                  layoutValue, resultType);
    return castOp.getResult();
}

// Helper function to implement make_tensor with a specific memory space
static mlir::Value
CreateMakeTensorWithMemorySpace(ast::Call *call_expr, GeneratorContext *ctx,
                                llvm::ArrayRef<mlir::Value> resolved_args,
                                mlir::gpu::AddressSpace addressSpace,
                                const std::string &functionName) {
    auto location = GetCallLocation(ctx, call_expr);
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();

    const auto &args = call_expr->GetArgs();

    if (args.size() != 2 && args.size() != 3) {
        std::string error_msg = functionName +
                                "() only supports (shape, dtype) or "
                                "(shape, dtype, alignment) with static "
                                "shape and no strides";
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << error_msg;
        return nullptr;
    }

    if (resolved_args.size() < args.size() || !resolved_args[0] ||
        (args.size() == 3 && !resolved_args[2])) {
        std::string error_msg =
            "Failed to resolve " + functionName + "() arguments";
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << error_msg;
        return nullptr;
    }

    // Parse shape and strides
    mlir::Value shapeTupleValue;
    mlir::Value strideTupleValue;
    mlir::Attribute memorySpaceAttr =
        mlir::gpu::AddressSpaceAttr::get(builder.getContext(), addressSpace);

    int dtypeArgIndex = 1;
    mlir::Type elementType = resolveAndValidateTensorElementType(
        args[dtypeArgIndex], ctx, call_expr->GetSourceRange().getBegin(),
        functionName);
    if (!elementType) {
        return nullptr;
    }

    mlir::IntegerAttr alignmentAttr;
    if (args.size() == 3) {
        auto alignment = ConstantFolder::FoldIntValue(resolved_args[2]);
        if (!alignment) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                args[2]->GetSourceRange().getBegin())
                << functionName
                << "() alignment must be an integer constant byte width";
            return nullptr;
        }
        if (*alignment <= 0 ||
            (*alignment & (*alignment - static_cast<int64_t>(1))) != 0) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                args[2]->GetSourceRange().getBegin())
                << functionName
                << "() alignment must be a positive power-of-two byte width";
            return nullptr;
        }
        alignmentAttr = builder.getI64IntegerAttr(*alignment);
    }

    if (resolved_args[0].getDefiningOp<cf::MakeLayoutOp>()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        args[0]->GetSourceRange().getBegin())
            << functionName
            << "() requires a static shape tuple and does not accept layouts";
        return nullptr;
    }

    auto shapeTupleOp = resolved_args[0].getDefiningOp<cf::MakeIntTupleOp>();
    if (!shapeTupleOp) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        args[0]->GetSourceRange().getBegin())
            << "First argument to " << functionName
            << "() must be a tuple representing shape";
        return nullptr;
    }
    shapeTupleValue = shapeTupleOp.getResult();

    // Only shape provided, generate row-major strides by default.
    // for shape tuple s = (d_1, d_2, ..., d_m), its stride tuple should
    // be (d_{m-1}*d_{m-2}*...*d_1, ..., d_2*d_1, 1). For example,
    // nested shape ((2, 2), (2, 2)) should have strides ((4, 1), (2, 1)).
    llvm::SmallVector<int64_t> shapeValues;
    if (auto foldedShape = foldStaticShapeTuple(args[0], ctx)) {
        shapeValues = std::move(*foldedShape);
        shapeTupleValue = createStaticIntTuple(shapeValues, builder, location);
    } else {
        if (!LayoutOperation::flattenTupleValues(resolved_args[0],
                                                 shapeValues)) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                args[0]->GetSourceRange().getBegin())
                << "Failed to extract shape values for " << functionName
                << "()";
            return nullptr;
        }
    }

    llvm::SmallVector<mlir::Value> strideValues;
    generateRowMajorStrides(shapeValues, strideValues, builder, location);
    auto tupleOp = cf::MakeIntTupleOp::create(builder, location, strideValues);
    strideTupleValue = tupleOp.getResult();

    // Create the resulting memref type
    llvm::SmallVector<int64_t> staticShape;
    LayoutOperation::flattenTupleValues(shapeTupleValue, staticShape);
    for (auto dim : staticShape) {
        if (dim == mlir::ShapedType::kDynamic) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << (addressSpace == mlir::gpu::AddressSpace::Workgroup
                        ? functionName +
                              "() does not support dynamic shared memory; "
                              "shape dimensions must be constant"
                        : functionName +
                              "() requires a static shape with constant "
                              "dimensions");
            return nullptr;
        }
    }

    llvm::SmallVector<int64_t> staticStrides;
    if (strideTupleValue) {
        if (!LayoutOperation::flattenTupleValues(strideTupleValue,
                                                 staticStrides)) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to extract stride values for " << functionName
                << "()";
            return nullptr;
        }
    }

    // Create memref type with optional memory space
    auto resultType = createTensorMemRefType(staticShape, elementType,
                                             memorySpaceAttr, staticStrides);

    auto totalBytes = getStaticTotalByteSize(staticShape, elementType);
    if (!totalBytes) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << functionName
            << "() requires static shape and element types "
               "with known byte size";
        return nullptr;
    }

    auto i8Type = builder.getIntegerType(8);
    llvm::SmallVector<int64_t> byteShape = {static_cast<int64_t>(*totalBytes)};
    auto baseType = createTensorMemRefType(byteShape, i8Type, memorySpaceAttr);

    mlir::Value alloca;
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        if (auto *func_gen = ctx->GetCurrentFunctionGenerator()) {
            if (auto *entry_block = func_gen->GetEntryBlock()) {
                builder.setInsertionPointToStart(entry_block);
            }
        }
        auto allocaOp = cf::AveLangMemRefAllocaOp::create(
            builder, location, baseType, mlir::ValueRange{});
        if (alignmentAttr) {
            allocaOp.setAlignmentAttr(alignmentAttr);
        }
        alloca = allocaOp.getResult();
    }

    mlir::Value layoutValue;
    if (auto layoutOp = resolved_args[0].getDefiningOp<cf::MakeLayoutOp>()) {
        layoutValue = layoutOp.getResult();
    } else {
        auto makeLayoutOp = cf::MakeLayoutOp::create(
            builder, location, shapeTupleValue, strideTupleValue);
        layoutValue = makeLayoutOp.getResult();
    }

    auto castOp = cf::AveLangMemRefCastOp::create(builder, location, alloca,
                                                  layoutValue, resultType);
    if (auto typeInfo = GetTypeInfo(args[1]); typeInfo.is_unsigned_integer) {
        SetTypeInfo(alloca, typeInfo);
        SetTypeInfo(castOp.getResult(), typeInfo);
    }
    return castOp.getResult();
}

mlir::Value AveLangModule::CreateMakeLocalFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateMakeTensorWithMemorySpace(call_expr, ctx, resolved_args,
                                           mlir::gpu::AddressSpace::Private,
                                           "make_local");
}

mlir::Value AveLangModule::CreateMakeSharedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateMakeTensorWithMemorySpace(call_expr, ctx, resolved_args,
                                           mlir::gpu::AddressSpace::Workgroup,
                                           "make_shared");
}

mlir::Value AveLangModule::CreateSubviewFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto location = GetCallLocation(ctx, call_expr);
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();

    const auto &args = call_expr->GetArgs();

    if (args.size() != 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "subview() expects exactly 4 argument (base memref, "
               "offsets, "
               "sizes, strides)";
        return nullptr;
    }

    if (resolved_args.size() < 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve arguments for subview()";
        return nullptr;
    }

    // Get the memref to cast
    auto base_memref_value = resolved_args[0];
    if (!base_memref_value) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate memref argument for subview()";
        return nullptr;
    }

    // Verify it's a memref type
    auto memref_type =
        mlir::dyn_cast<cf::MemRefType>(base_memref_value.getType());
    if (!memref_type) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "First argument to subview() must be a memref";
        return nullptr;
    }

    // Parse offsets, sizes, strides tuple
    auto *offsets_tuple = llvm::dyn_cast<ast::Tuple>(args[1]);
    auto *sizes_tuple = llvm::dyn_cast<ast::Tuple>(args[2]);
    auto *strides_tuple = llvm::dyn_cast<ast::Tuple>(args[3]);

    if (!offsets_tuple || !sizes_tuple || !strides_tuple) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "offsets, sizes and strides must be tuples";
        return nullptr;
    }

    if (offsets_tuple->GetElts().size() != sizes_tuple->GetElts().size() ||
        sizes_tuple->GetElts().size() != strides_tuple->GetElts().size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "offsets, sizes and strides must have the same number of "
               "dimensions in subview() function";
        return nullptr;
    }

    if (offsets_tuple->GetElts().size() == 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "offsets, sizes and strides cannot be empty in subview() "
               "function";
        return nullptr;
    }

    size_t rank = static_cast<size_t>(memref_type.getRank());

    if (offsets_tuple->GetElts().size() != rank) {
        std::string error_msg =
            "offsets, sizes and strides tuple sizes must match memref rank (" +
            std::to_string(rank) + ") in subview() function";
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << error_msg;
        return nullptr;
    }

    // Helper function to extract OpFoldResult from AST tuple
    auto extract_from_ast_tuple =
        [&](ast::Tuple *tuple) -> llvm::SmallVector<mlir::OpFoldResult> {
        llvm::SmallVector<mlir::OpFoldResult> result;
        ConstantFolder folder(ctx);

        for (auto *elem : tuple->GetElts()) {
            if (auto dim_value = folder.Evaluate(elem)) {
                result.push_back(builder.getIndexAttr(*dim_value));
                continue;
            }

            // Generate runtime expression
            auto dynamic_value = ctx->GetCurrentFunctionGenerator()
                                     ->GetExprGenerator()
                                     ->Dispatch(elem);
            if (!dynamic_value) {
                ctx->diagnostic_manager->Report(
                    basic::DiagnosticCode::kUnimplemented,
                    elem->GetSourceRange().getBegin())
                    << "Failed to generate expression for tuple element";
                return {};
            }

            // Convert to index type if needed
            if (!dynamic_value.getType().isIndex()) {
                dynamic_value = mlir::arith::IndexCastOp::create(
                    builder, location, builder.getIndexType(), dynamic_value);
            }

            if (auto folded = ConstantFolder::FoldIntValue(dynamic_value)) {
                result.push_back(builder.getIndexAttr(*folded));
                continue;
            }

            result.push_back(dynamic_value);
        }
        return result;
    };

    auto offsets_folded = extract_from_ast_tuple(offsets_tuple);
    auto sizes_folded = extract_from_ast_tuple(sizes_tuple);
    auto strides_folded = extract_from_ast_tuple(strides_tuple);

    if (offsets_folded.empty() || sizes_folded.empty() ||
        strides_folded.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to extract values from tuples";
        return nullptr;
    }

    // Compute result shape from subview sizes (rank-reduce size-1 dims).
    llvm::SmallVector<int64_t> target_shape;
    target_shape.reserve(sizes_folded.size());
    for (auto size : sizes_folded) {
        if (auto attr = size.dyn_cast<mlir::Attribute>()) {
            if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
                auto dim = intAttr.getInt();
                if (dim == 1) {
                    continue;
                }
                target_shape.push_back(dim);
                continue;
            }
        }
        target_shape.push_back(mlir::ShapedType::kDynamic);
    }

    llvm::SmallVector<int64_t> baseStrides(memref_type.getStrides().begin(),
                                           memref_type.getStrides().end());
    if (baseStrides.empty()) {
        baseStrides.resize(memref_type.getRank());
        int64_t running = 1;
        for (int64_t idx = memref_type.getRank() - 1; idx >= 0; --idx) {
            baseStrides[idx] = running;
            auto dim = memref_type.getDimSize(idx);
            if (dim == mlir::ShapedType::kDynamic ||
                running == mlir::ShapedType::kDynamic) {
                running = mlir::ShapedType::kDynamic;
            } else {
                running *= dim;
            }
        }
    }

    auto getStaticFolded =
        [&](mlir::OpFoldResult value) -> std::optional<int64_t> {
        if (auto attr = value.dyn_cast<mlir::Attribute>()) {
            if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
                return intAttr.getInt();
            }
        }
        return std::nullopt;
    };

    llvm::SmallVector<int64_t> resultStrides;
    resultStrides.reserve(baseStrides.size());
    for (size_t i = 0; i < baseStrides.size(); ++i) {
        bool drop_dim = false;
        if (auto sizeVal = getStaticFolded(sizes_folded[i])) {
            drop_dim = (*sizeVal == 1);
        }
        if (drop_dim) {
            continue;
        }

        int64_t stride = mlir::ShapedType::kDynamic;
        auto subStride = getStaticFolded(strides_folded[i]);
        if (subStride && baseStrides[i] != mlir::ShapedType::kDynamic) {
            stride = baseStrides[i] * (*subStride);
        }
        resultStrides.push_back(stride);
    }

    auto resultType =
        createTensorMemRefType(target_shape, memref_type.getElementType(),
                               memref_type.getMemorySpace(), resultStrides);

    auto materializeIndex = [&](mlir::OpFoldResult value) -> mlir::Value {
        return mlir::getValueOrCreateConstantIndexOp(builder, location, value);
    };

    llvm::SmallVector<mlir::Value> offsets_values;
    llvm::SmallVector<mlir::Value> sizes_values;
    llvm::SmallVector<mlir::Value> strides_values;
    offsets_values.reserve(offsets_folded.size());
    sizes_values.reserve(sizes_folded.size());
    strides_values.reserve(strides_folded.size());
    for (auto value : offsets_folded) {
        offsets_values.push_back(materializeIndex(value));
    }
    for (auto value : sizes_folded) {
        sizes_values.push_back(materializeIndex(value));
    }
    for (auto value : strides_folded) {
        strides_values.push_back(materializeIndex(value));
    }

    auto subview_op = cf::AveLangMemRefSubViewOp::create(
        builder, location, resultType, base_memref_value, offsets_values,
        sizes_values, strides_values);
    SetTypeInfo(subview_op.getResult(), GetTypeInfo(base_memref_value));
    return subview_op.getResult();
}

mlir::Value AveLangModule::CreateFullFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto location = GetCallLocation(ctx, call_expr);
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();

    const auto &args = call_expr->GetArgs();

    // Validate argument count: avelang.full(shapes, value, dtype)
    if (args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "avelang.full() expects exactly 3 arguments: (shapes, value, "
               "dtype)";
        return nullptr;
    }

    if (resolved_args.size() < 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve arguments for avelang.full()";
        return nullptr;
    }

    auto shape_tuple_value = resolved_args[0];
    auto fill_value = resolved_args[1];

    if (!shape_tuple_value || !fill_value) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate shape or fill value for avelang.full()";
        return nullptr;
    }

    // Resolve the dtype from the third argument
    mlir::Type element_type = ctx->syms->ResolveType(args[2]);
    if (!element_type) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "avelang.full() requires a valid dtype as the third argument";
        return nullptr;
    }

    if (!fill_value.getType().isIntOrIndexOrFloat()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "avelang.full() requires the fill value to be scalar type";
        return nullptr;
    }

    auto target_scalar_type = element_type;
    if (auto element_vector_type =
            mlir::dyn_cast<mlir::VectorType>(element_type)) {
        target_scalar_type = element_vector_type.getElementType();
    }

    std::optional<bool> targetUnsigned;
    if (auto typeInfo = GetTypeInfo(args[2]); typeInfo.is_unsigned_integer) {
        targetUnsigned = typeInfo.is_unsigned_integer;
    }

    if (fill_value.getType() != target_scalar_type) {
        // Convert if needed
        auto converted_value = causalflow::avelang::ir::CreateTypeConversion(
            fill_value, fill_value.getType(), target_scalar_type, location,
            builder, true, GetTypeInfo(fill_value).is_unsigned_integer,
            targetUnsigned);
        if (!converted_value) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to convert fill value to target dtype in "
                   "avelang.full()";
            return nullptr;
        }
        fill_value = converted_value;
    }

    // Parse the shape tuple to get the result type
    llvm::SmallVector<int64_t> shape;
    if (auto *shape_tuple_ast = llvm::dyn_cast<ast::Tuple>(args[0])) {
        for (auto *dim_expr : shape_tuple_ast->GetElts()) {
            if (auto *constant = llvm::dyn_cast<ast::Constant>(dim_expr)) {
                if (auto dim_value =
                        ParseConstantInteger(constant->GetValue())) {
                    shape.push_back(*dim_value);
                } else {
                    ctx->diagnostic_manager->Report(
                        basic::DiagnosticCode::kUnimplemented,
                        dim_expr->GetSourceRange().getBegin())
                        << "avelang.full() only supports static shape "
                           "dimensions";
                    return nullptr;
                }
            } else {
                ctx->diagnostic_manager->Report(
                    basic::DiagnosticCode::kUnimplemented,
                    dim_expr->GetSourceRange().getBegin())
                    << "avelang.full() only supports static shape dimensions";
                return nullptr;
            }
        }
    } else {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "First argument to avelang.full() must be a tuple "
               "representing shape";
        return nullptr;
    }

    if (shape.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Shape cannot be empty in avelang.full()";
        return nullptr;
    }

    // Create the result memref type with private memory space
    auto addressSpaceAttr = mlir::gpu::AddressSpaceAttr::get(
        builder.getContext(), mlir::gpu::AddressSpace::Private);
    auto resultType =
        createTensorMemRefType(shape, element_type, addressSpaceAttr);

    // Create the FullOp
    auto fullOp = cf::FullOp::create(builder, location, shape_tuple_value,
                                     fill_value, resultType);
    if (targetUnsigned) {
        SetTypeInfo(fullOp.getResult(), TypeInfo{*targetUnsigned});
    }

    return fullOp.getResult();
}

mlir::Value AveLangModule::CreateViewFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto location = GetCallLocation(ctx, call_expr);
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();

    const auto &args = call_expr->GetArgs();

    // view(memref, TensorType) OR view(memref, dtype, layout)
    if (args.size() == 3) {
        return LayoutOperation::createViewFunction(call_expr, ctx,
                                                   resolved_args);
    }

    if (args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "view() expects either 2 arguments (memref, TensorType) or 3 "
               "arguments (memref, dtype, layout)";
        return nullptr;
    }

    if (resolved_args.size() < 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve arguments for view()";
        return nullptr;
    }

    // Get the memref to cast
    auto memref_value = resolved_args[0];
    if (!memref_value) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate memref argument for view()";
        return nullptr;
    }

    auto value_type = memref_value.getType();

    // Check if the value is a constant - raise error in this case
    if (auto const_op = memref_value.getDefiningOp<mlir::arith::ConstantOp>()) {
        (void)const_op; // Suppress unused variable warning
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "First argument to view() cannot be a constant value";
        return nullptr;
    } else if (auto const_index_op =
                   memref_value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        (void)const_index_op; // Suppress unused variable warning
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "First argument to view() cannot be a constant value";
        return nullptr;
    }

    // Resolve the target tensor type from the second argument
    mlir::Type target_tensor_type = ctx->syms->ResolveType(args[1]);
    if (!target_tensor_type) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve target TensorType for view()";
        return nullptr;
    }

    // Verify the target type is also a memref
    auto target_memref_type =
        mlir::dyn_cast<cf::MemRefType>(target_tensor_type);
    if (!target_memref_type) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Second argument to view() must be a Tensor type (memref)";
        return nullptr;
    }

    if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(value_type)) {
        auto src_elem_type = vector_type.getElementType();
        auto target_elem_type = target_memref_type.getElementType();
        if (!src_elem_type.isIntOrFloat() || !target_elem_type.isIntOrFloat()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "view() only supports vector element types that are "
                   "int/float";
            return nullptr;
        }

        auto target_shape = target_memref_type.getShape();
        int64_t target_elements = 1;
        for (auto dim : target_shape) {
            if (dim == mlir::ShapedType::kDynamic) {
                ctx->diagnostic_manager->Report(
                    basic::DiagnosticCode::kUnimplemented,
                    call_expr->GetSourceRange().getBegin())
                    << "view() on vectors requires a static target shape";
                return nullptr;
            }
            target_elements *= dim;
        }

        int64_t source_elements = vector_type.getNumElements();
        int64_t source_bits = src_elem_type.getIntOrFloatBitWidth();
        int64_t target_bits = target_elem_type.getIntOrFloatBitWidth();
        if (source_elements * source_bits != target_elements * target_bits) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "view() requires source and target to have the same total "
                   "bitwidth";
            return nullptr;
        }

        mlir::Value flattened = memref_value;
        if (vector_type.getRank() != 1) {
            auto flat_type =
                mlir::VectorType::get(source_elements, src_elem_type);
            flattened = mlir::vector::ShapeCastOp::create(builder, location,
                                                          flat_type, flattened);
            SetTypeInfo(flattened, GetTypeInfo(memref_value));
        }

        auto bitcast_type =
            mlir::VectorType::get(target_elements, target_elem_type);
        mlir::Value bitcasted = mlir::vector::BitCastOp::create(
            builder, location, bitcast_type, flattened);
        if (auto typeInfo = GetTypeInfo(args[1]);
            typeInfo.is_unsigned_integer) {
            SetTypeInfo(bitcasted, typeInfo);
        } else {
            SetTypeInfo(bitcasted, GetTypeInfo(flattened));
        }

        if (target_shape.size() != 1) {
            auto target_vector_type =
                mlir::VectorType::get(target_shape, target_elem_type);
            bitcasted = mlir::vector::ShapeCastOp::create(
                builder, location, target_vector_type, bitcasted);
            SetTypeInfo(bitcasted,
                        GetTypeInfo(bitcasted.getDefiningOp()->getOperand(0)));
        }

        return bitcasted;
    }

    // Check if it's already a memref type
    mlir::Value actual_memref_value;
    if (mlir::isa<cf::MemRefType>(value_type)) {
        // Already a memref, use it directly
        actual_memref_value = memref_value;
    } else {
        // Check if it's a scalar type
        bool is_scalar = value_type.isIntOrIndexOrFloat();

        if (is_scalar) {
            mlir::Value base_memref;

            // Create a temporary memref to hold this value
            mlir::Attribute address_space_attr;

            mlir::Type element_type = value_type;
            llvm::SmallVector<int64_t> shape = {1};

            auto temp_memref_type =
                createTensorMemRefType(shape, element_type, address_space_attr);

            auto temp_memref = cf::AveLangMemRefAllocaOp::create(
                builder, location, temp_memref_type, mlir::ValueRange{});
            SetTypeInfo(temp_memref.getResult(), GetTypeInfo(memref_value));

            // Store the value into the temporary memref
            auto zero_index =
                mlir::arith::ConstantIndexOp::create(builder, location, 0);

            cf::AveLangMemRefStoreOp::create(builder, location, memref_value,
                                             temp_memref,
                                             mlir::ValueRange{zero_index});

            base_memref = temp_memref;

            actual_memref_value = base_memref;
        } else {
            // Not a memref, scalar, or vector type
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "First argument to view() must be a memref, scalar, "
                   "or vector type";
            return nullptr;
        }
    }

    // Verify we have a valid memref to cast
    auto actual_memref_type =
        mlir::dyn_cast<cf::MemRefType>(actual_memref_value.getType());
    if (!actual_memref_type) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to determine memref to cast in view()";
        return nullptr;
    }

    auto address_space_attr = actual_memref_type.getMemorySpace();
    cf::MemRefType target_with_memspace = target_memref_type;
    if (address_space_attr) {
        target_with_memspace = cf::MemRefType::get(
            builder.getContext(), target_memref_type.getLayout(),
            target_memref_type.getElementType(), address_space_attr);
    }

    // Create the memref.cast op
    auto castOp = cf::AveLangMemRefCastOp::create(
        builder, location, actual_memref_value, target_with_memspace);
    if (auto typeInfo = GetTypeInfo(args[1]); typeInfo.is_unsigned_integer) {
        SetTypeInfo(castOp.getResult(), typeInfo);
    } else {
        SetTypeInfo(castOp.getResult(), GetTypeInfo(actual_memref_value));
    }

    return castOp.getResult();
}

} // namespace causalflow::avelang::ir
