#include "AST/ast_nodes_expr.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "IR/builtin_module.h"
#include "IR/generator_context.h"
#include "IR/mlir_generator_impl.h"
#include "IR/named_module.h"
#include "Utils/assert.h"
#include "Utils/embedded_filesystem_view.h"
#include "intrinsic_support.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/NVGPU/IR/NVGPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/NVVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <string>
#include <string_view>
#include <utility>

extern "C" const unsigned char _binary_nvvm_intrinsics_mlirbc_start[];
extern "C" const unsigned char _binary_nvvm_intrinsics_mlirbc_end[];

namespace causalflow::avelang::ir {

using namespace mlir;
using namespace mlir::NVVM;
using namespace causalflow::avelang::dialect;
namespace cf = causalflow::avelang::dialect;

namespace {

static llvm::StringRef GetNvvmIntrinsicLibrary() {
    auto *start =
        reinterpret_cast<const char *>(_binary_nvvm_intrinsics_mlirbc_start);
    auto *end =
        reinterpret_cast<const char *>(_binary_nvvm_intrinsics_mlirbc_end);
    return {start, static_cast<size_t>(end - start)};
}

constexpr llvm::StringRef kNvvmIntrinsicLibraryName = "nvvm_intrinsics.mlirbc";
constexpr llvm::StringRef kNvvmIntrinsicLibraryTag =
    "embedded:nvvm_intrinsics.mlirbc";


static std::optional<int64_t> getConstantIntValue(mlir::Value value) {
    if (!value) {
        return std::nullopt;
    }
    if (auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr =
                mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            return intAttr.getInt();
        }
    }
    if (auto constOp = value.getDefiningOp<mlir::LLVM::ConstantOp>()) {
        if (auto intAttr =
                mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            return intAttr.getInt();
        }
    }
    return std::nullopt;
}

static bool extractConstantTupleValues(
    mlir::Value tupleValue, llvm::SmallVectorImpl<int64_t> &values) {
    if (auto tupleOp = tupleValue.getDefiningOp<cf::MakeIntTupleOp>()) {
        for (auto elem : tupleOp.getElements()) {
            if (!extractConstantTupleValues(elem, values)) {
                return false;
            }
        }
        return true;
    }

    auto value = getConstantIntValue(tupleValue);
    if (!value) {
        return false;
    }
    values.push_back(*value);
    return true;
}

} // namespace

// NVVM Intrinsics Module
class NVVMIntrinsic : public NamedModule {
  public:
    explicit NVVMIntrinsic();

    void Initialize() override;
    void DeclareModules(mlir::ModuleOp module) override;

    struct MMAConfig {
        int m, n, k;
        MMATypes type_a, type_b;
        MMALayout layout_a, layout_b;
        int fragments_a_count, fragments_b_count, fragments_c_count;
        int fragment_size;
    };

    mlir::Value CreateMma16x8x16F16F16Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateMma16x8x8F16F32Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMakeTMADescriptorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

  private:
    void AddLdMatrixFactory(const std::string &name, const std::string &shape,
                            int num, int bit_width, bool transpose);

    void AddStMatrixFactory(const std::string &name, const std::string &shape,
                            int num, int bit_width, bool transpose);

    mlir::Value
    CreateGenericMMAFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args,
                             const MMAConfig &config) const;

    mlir::Value
    CreateLdMatrixWithShape(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args,
                            const std::string &shape, int num, int bit_width,
                            bool transpose = false) const;

    mlir::Value
    CreateStMatrixWithShape(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args,
                            const std::string &shape, int num, int bit_width,
                            bool transpose = false) const;

    bool
    CheckGenericMMAFunction(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckLdMatrixWithShape(ast::Call *call_expr, GeneratorContext *ctx,
                                llvm::ArrayRef<mlir::Value> resolved_args,
                                const std::string &shape, int num,
                                int bit_width, bool transpose) const;

    bool CheckStMatrixWithShape(ast::Call *call_expr, GeneratorContext *ctx,
                                llvm::ArrayRef<mlir::Value> resolved_args,
                                const std::string &shape, int num,
                                int bit_width, bool transpose) const;
    bool CheckMakeTMADescriptorFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;

};

NVVMIntrinsic::NVVMIntrinsic() : NamedModule("nvvm") {}

void NVVMIntrinsic::Initialize() {
    // Add MMA sync intrinsic function helpers with constraints encoded in
    // function names
    AddFunction(
        "mma_16x8x16_f16_f16",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMma16x8x16F16F16Function(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericMMAFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "mma_16x8x8_f16_f32",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMma16x8x8F16F32Function(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericMMAFunction(call_expr, gen_ctx, resolved_args);
        });

    // LLVM 22 models ldmatrix/stmatrix shape and element type explicitly.
    // The implemented wrappers cover the standard m8n8/b16 family; the old
    // b8 and m16n16 registrations do not match the stricter NVVM verifier.
    for (int num : {1, 2, 4}) {
        std::string base_name =
            "ldmatrix_m8n8_x" + std::to_string(num) + "_b16";
        AddLdMatrixFactory(base_name, "m8n8", num, 16, false);
        AddLdMatrixFactory(base_name + "_trans", "m8n8", num, 16, true);
    }

    for (int num : {1, 2, 4}) {
        std::string base_name =
            "stmatrix_m8n8_x" + std::to_string(num) + "_b16";
        AddStMatrixFactory(base_name, "m8n8", num, 16, false);
        AddStMatrixFactory(base_name + "_trans", "m8n8", num, 16, true);
    }
    AddFunction(
        "make_tma_descriptor",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMakeTMADescriptorFunction(call_expr, gen_ctx,
                                                   resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMakeTMADescriptorFunction(call_expr, gen_ctx,
                                                  resolved_args);
        });

}

void NVVMIntrinsic::DeclareModules(mlir::ModuleOp module) {
    if (!module)
        return;

    // Register the bytecode in the intrinsic registry
    auto libraryBytes = GetNvvmIntrinsicLibrary();
    auto &registry = utils::EmbeddedFilesystemView::getInstance();
    registry.registerFile(std::string(kNvvmIntrinsicLibraryName), libraryBytes);

    intrinsics::GetOrCreateImplementationContainer(module, "nvvm",
                                                   kNvvmIntrinsicLibraryTag);

    auto loadDialects = [](mlir::MLIRContext *ctx) {
        ctx->loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
                         mlir::memref::MemRefDialect,
                         mlir::vector::VectorDialect, cf::AveLangDialect>();
    };

    if (failed(intrinsics::EnsureIntrinsicDeclarations(
            module, kNvvmIntrinsicLibraryName, libraryBytes, loadDialects))) {
        module.emitError() << "failed to declare NVVM intrinsics";
    }
}

void NVVMIntrinsic::AddLdMatrixFactory(const std::string &name,
                                       const std::string &shape, int num,
                                       int bit_width, bool transpose) {
    AddFunction(
        name,
        [this, shape, num, bit_width,
         transpose](ast::Call *call_expr, GeneratorContext *gen_ctx,
                    llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateLdMatrixWithShape(call_expr, gen_ctx, resolved_args,
                                           shape, num, bit_width, transpose);
        },
        [this, shape, num, bit_width,
         transpose](ast::Call *call_expr, GeneratorContext *gen_ctx,
                    llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckLdMatrixWithShape(call_expr, gen_ctx, resolved_args,
                                          shape, num, bit_width, transpose);
        });
}

void NVVMIntrinsic::AddStMatrixFactory(const std::string &name,
                                       const std::string &shape, int num,
                                       int bit_width, bool transpose) {
    AddFunction(
        name,
        [this, shape, num, bit_width,
         transpose](ast::Call *call_expr, GeneratorContext *gen_ctx,
                    llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateStMatrixWithShape(call_expr, gen_ctx, resolved_args,
                                           shape, num, bit_width, transpose);
        },
        [this, shape, num, bit_width,
         transpose](ast::Call *call_expr, GeneratorContext *gen_ctx,
                    llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckStMatrixWithShape(call_expr, gen_ctx, resolved_args,
                                          shape, num, bit_width, transpose);
        });
}

mlir::Value NVVMIntrinsic::CreateGenericMMAFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const MMAConfig &config) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    auto a = resolved_args[0];
    auto b = resolved_args[1];
    auto c = resolved_args[2];

    auto c_vector = mlir::dyn_cast<mlir::VectorType>(c.getType());
    if (!c_vector) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "C operand for MMA operation must be a vector type";
        return nullptr;
    }

    // Determine result type based on configuration
    mlir::Type result_type;
    if (c_vector.getElementType().isF16()) {
        result_type = c_vector;
    } else if (c_vector.getElementType().isF32()) {
        result_type = c_vector;
    } else {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Unsupported result type for MMA operation";
        return nullptr;
    }

    // Create GPUOp NVVM MMA operation
    auto mma_op =
        cf::NVVMMMAOp::create(builder, location, result_type, a, b, c);

    return mma_op.getResult();
}

mlir::Value NVVMIntrinsic::CreateMma16x8x16F16F16Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    MMAConfig config = {.m = 16,
                        .n = 8,
                        .k = 16,
                        .type_a = MMATypes::f16,
                        .type_b = MMATypes::f16,
                        .layout_a = MMALayout::row,
                        .layout_b = MMALayout::col,
                        .fragments_a_count = 4,
                        .fragments_b_count = 2,
                        .fragments_c_count = 2,
                        .fragment_size = 2};
    return CreateGenericMMAFunction(call_expr, ctx, resolved_args, config);
}

mlir::Value NVVMIntrinsic::CreateMma16x8x8F16F32Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    MMAConfig config = {.m = 16,
                        .n = 8,
                        .k = 8,
                        .type_a = MMATypes::f16,
                        .type_b = MMATypes::f32,
                        .layout_a = MMALayout::row,
                        .layout_b = MMALayout::col,
                        .fragments_a_count = 2,
                        .fragments_b_count = 1,
                        .fragments_c_count = 4,
                        .fragment_size = 2};
    return CreateGenericMMAFunction(call_expr, ctx, resolved_args, config);
}

mlir::Value NVVMIntrinsic::CreateLdMatrixWithShape(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const std::string &shape,
    int num, int bit_width, bool transpose) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    auto memref_ptr = resolved_args[0];

    auto memref_type = mlir::dyn_cast<cf::MemRefType>(memref_ptr.getType());
    if (!memref_type) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "ldmatrix expects a memref argument";
        return nullptr;
    }

    auto is_compatible_memref = [&](const std::string &shape_tag) {
        auto gpu_space = mlir::gpu::AddressSpaceAttr::get(
            builder.getContext(), mlir::gpu::AddressSpace::Workgroup);

        if (memref_type.getMemorySpace() != gpu_space) {
            return false;
        }

        auto elem_type = memref_type.getElementType();

        if (bit_width == 16) {
            bool valid_type = elem_type.isF16() || elem_type.isBF16() ||
                              elem_type.isInteger(16);
            if (!valid_type) {
                return false;
            }
        } else if (bit_width == 8) {
            bool valid_type = elem_type.isInteger(8) ||
                              isa<Float8E4M3FNType>(elem_type) ||
                              isa<Float8E4M3FNUZType>(elem_type);
            if (!valid_type) {
                return false;
            }
        }

        auto shape_vec = memref_type.getShape();
        if (shape_tag == "m8n8") {
            return shape_vec.size() == 2 && shape_vec[0] == 8 &&
                   shape_vec[1] == 8;
        }
        if (shape_tag == "m16n16") {
            return shape_vec.size() == 2 && shape_vec[0] == 16 &&
                   shape_vec[1] == 16;
        }
        return false;
    };

    auto type_to_string = [](mlir::Type type) {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        type.print(os);
        return buffer;
    };

    if (!is_compatible_memref(shape)) {
        std::string expected_shape = shape == "m8n8" ? "8x8" : "16x16";
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "ldmatrix_" << shape << " expects memref<" << expected_shape
            << "xb" << bit_width
            << ", #gpu.address_space<workgroup>> (with optional "
               "strides/offset) but found "
            << type_to_string(memref_ptr.getType());
        return nullptr;
    }

    mlir::Type result_type;
    if (num == 1) {
        result_type = builder.getI32Type();
    } else {
        result_type = mlir::VectorType::get({num}, builder.getI32Type());
    }

    auto ld_matrix_op = cf::NVVMLdMatrixOp::create(
        builder, location, result_type, memref_ptr,
        mlir::StringAttr::get(builder.getContext(), shape),
        mlir::IntegerAttr::get(builder.getI32Type(), num),
        mlir::IntegerAttr::get(builder.getI32Type(), bit_width));

    return ld_matrix_op.getResult();
}

mlir::Value NVVMIntrinsic::CreateStMatrixWithShape(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const std::string &shape,
    int num, int bit_width, bool transpose) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    auto memref_ptr = resolved_args[0];

    auto memref_type = mlir::dyn_cast<cf::MemRefType>(memref_ptr.getType());
    if (!memref_type) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "stmatrix expects a memref argument";
        return nullptr;
    }

    auto is_compatible_memref = [&](const std::string &shape_tag) {
        auto gpu_space = mlir::gpu::AddressSpaceAttr::get(
            builder.getContext(), mlir::gpu::AddressSpace::Workgroup);

        if (memref_type.getMemorySpace() != gpu_space) {
            return false;
        }

        auto elem_type = memref_type.getElementType();

        if (bit_width == 16) {
            bool valid_type = elem_type.isF16() || elem_type.isBF16() ||
                              elem_type.isInteger(16);
            if (!valid_type) {
                return false;
            }
        } else if (bit_width == 8) {
            bool valid_type = elem_type.isInteger(8) ||
                              isa<Float8E4M3FNType>(elem_type) ||
                              isa<Float8E4M3FNUZType>(elem_type);
            if (!valid_type) {
                return false;
            }
        }

        auto shape_vec = memref_type.getShape();
        if (shape_tag == "m8n8") {
            return shape_vec.size() == 2 && shape_vec[0] == 8 &&
                   shape_vec[1] == 8;
        }
        if (shape_tag == "m16n16") {
            return shape_vec.size() == 2 && shape_vec[0] == 16 &&
                   shape_vec[1] == 16;
        }
        return false;
    };

    auto type_to_string = [](mlir::Type type) {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        type.print(os);
        return buffer;
    };

    if (!is_compatible_memref(shape)) {
        std::string expected_shape = shape == "m8n8" ? "8x8" : "16x16";
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "stmatrix_" << shape << " expects memref<" << expected_shape
            << "xb" << bit_width
            << ", #gpu.address_space<workgroup>> (with optional "
               "strides/offset) but found "
            << type_to_string(memref_ptr.getType());
        return nullptr;
    }

    auto source_arg = resolved_args[1];
    if (!source_arg) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate source operand for stmatrix_" << shape;
        return nullptr;
    }

    auto source_type = source_arg.getType();

    if (num == 1) {
        if (!source_type.isInteger(32) && source_type.isInteger()) {
            source_arg = mlir::arith::ExtSIOp::create(
                builder, location, builder.getI32Type(), source_arg);
            source_type = source_arg.getType();
        }
    } else {
        auto vector_type = mlir::cast<mlir::VectorType>(source_type);
        (void)vector_type;
    }

    cf::NVVMStMatrixOp::create(
        builder, location, source_arg, memref_ptr,
        mlir::StringAttr::get(builder.getContext(), shape),
        mlir::IntegerAttr::get(builder.getI32Type(), num),
        mlir::IntegerAttr::get(builder.getI32Type(), bit_width));

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool NVVMIntrinsic::CheckGenericMMAFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mma operation requires exactly 3 arguments: a, b, c";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mma operation";
        return false;
    }

    if (!mlir::dyn_cast<mlir::VectorType>(resolved_args[0].getType()) ||
        !mlir::dyn_cast<mlir::VectorType>(resolved_args[1].getType()) ||
        !mlir::dyn_cast<mlir::VectorType>(resolved_args[2].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mma operands must be vector types";
        return false;
    }

    return true;
}

bool NVVMIntrinsic::CheckLdMatrixWithShape(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const std::string &shape,
    int num, int bit_width, bool transpose) const {
    if (resolved_args.size() != 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "ldmatrix_" << shape << "_x" << num << "_b" << bit_width
            << (transpose ? "_trans" : "")
            << " requires exactly 1 argument: ptr";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate pointer operand for ldmatrix_" << shape;
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "ldmatrix_" << shape << " expects memref pointer operand";
        return false;
    }

    return true;
}

bool NVVMIntrinsic::CheckStMatrixWithShape(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, const std::string &shape,
    int num, int bit_width, bool transpose) const {
    size_t expected_args = 2;
    if (resolved_args.size() != expected_args) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "stmatrix_" << shape << "_x" << num << "_b" << bit_width
            << (transpose ? "_trans" : "") << " requires exactly "
            << expected_args << " arguments: ptr + vector source values";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for stmatrix_" << shape;
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "stmatrix_" << shape << " expects memref pointer operand";
        return false;
    }

    auto source_type = resolved_args[1].getType();
    if (num == 1) {
        if (!source_type.isInteger(32)) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "stmatrix_" << shape
                << "_x1 source operand must be integer type";
            return false;
        }
    } else {
        auto vector_type = mlir::dyn_cast<mlir::VectorType>(source_type);
        if (!vector_type || vector_type.getNumElements() != num ||
            !vector_type.getElementType().isInteger(32)) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "stmatrix_" << shape << "_x" << num
                << " source operand must be i32x" << num << " vector type";
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMakeTMADescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMakeTMADescriptorFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto tensorType = mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    if (!tensorType) {
        return nullptr;
    }

    auto layoutOp = resolved_args[1].getDefiningOp<cf::MakeLayoutOp>();
    llvm::SmallVector<int64_t> smemDims;
    llvm::SmallVector<int64_t> smemStrides;
    if (!layoutOp ||
        !extractConstantTupleValues(layoutOp.getDims(), smemDims) ||
        !extractConstantTupleValues(layoutOp.getStride(), smemStrides) ||
        smemDims.size() != smemStrides.size()) {
        return nullptr;
    }

    auto workgroupMemorySpace = mlir::IntegerAttr::get(
        mlir::IntegerType::get(builder.getContext(), 64), 3);
    auto descriptorTensorType = mlir::MemRefType::get(
        smemDims, tensorType.getElementType(),
        mlir::StridedLayoutAttr::get(builder.getContext(), 0, smemStrides),
        workgroupMemorySpace);

    auto resultType = mlir::nvgpu::TensorMapDescriptorType::get(
        builder.getContext(), descriptorTensorType,
        mlir::nvgpu::TensorMapSwizzleKind::SWIZZLE_NONE,
        mlir::nvgpu::TensorMapL2PromoKind::L2PROMO_NONE,
        mlir::nvgpu::TensorMapOOBKind::OOB_ZERO,
        mlir::nvgpu::TensorMapInterleaveKind::INTERLEAVE_NONE);

    auto descriptor = cf::NVVMTMADescriptorOp::create(builder, location, resultType, resolved_args[0], resolved_args[1]);
    return descriptor.getResult();
}

bool NVVMIntrinsic::CheckMakeTMADescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor requires exactly 2 arguments: tensor, smem_layout";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate tensor operand for make_tma_descriptor";
        return false;
    }

    if (!resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate smem_layout operand for make_tma_descriptor";
        return false;
    }

    auto tensorType = mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    if (!tensorType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor expects a memref tensor operand";
        return false;
    }

    auto layoutOp = resolved_args[1].getDefiningOp<cf::MakeLayoutOp>();
    if (!layoutOp) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor expects smem_layout from make_layout()";
        return false;
    }

    auto dimsTuple = layoutOp.getDims().getDefiningOp<cf::MakeIntTupleOp>();
    if (!dimsTuple || dimsTuple.getNumElements() == 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor requires a non-empty smem_layout";
        return false;
    }

    llvm::SmallVector<int64_t> smemDims;
    llvm::SmallVector<int64_t> smemStrides;
    if (!extractConstantTupleValues(layoutOp.getDims(), smemDims) ||
        !extractConstantTupleValues(layoutOp.getStride(), smemStrides) ||
        smemDims.size() != smemStrides.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_tma_descriptor requires a static smem_layout";
        return false;
    }

    return true;
}

// Factory function to create NVVM intrinsic module
std::unique_ptr<NamedModule> CreateNVVMIntrinsicModule() {
    return std::make_unique<NVVMIntrinsic>();
}

} // namespace causalflow::avelang::ir
