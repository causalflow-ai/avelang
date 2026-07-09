#include "AST/ast_nodes_expr.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "IR/Intrinsics/amdgpu_mfma_signatures.h"
#include "IR/builtin_module.h"
#include "IR/constant_folder.h"
#include "IR/generator_context.h"
#include "IR/mlir_generator_impl.h"
#include "IR/named_module.h"
#include "Utils/assert.h"
#include "Utils/embedded_filesystem_view.h"
#include "intrinsic_support.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

extern "C" const unsigned char _binary_amdgpu_intrinsics_mlirbc_start[];
extern "C" const unsigned char _binary_amdgpu_intrinsics_mlirbc_end[];

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

namespace causalflow::avelang::ir {

using namespace mlir;
using namespace mlir::ROCDL;
using namespace causalflow::avelang::dialect;
namespace cf = causalflow::avelang::dialect;
namespace amdgpu_mfma = causalflow::avelang::amdgpu::mfma;

namespace {

static llvm::StringRef GetAmdgpuIntrinsicLibrary() {
    auto *start =
        reinterpret_cast<const char *>(_binary_amdgpu_intrinsics_mlirbc_start);
    auto *end =
        reinterpret_cast<const char *>(_binary_amdgpu_intrinsics_mlirbc_end);
    return {start, static_cast<size_t>(end - start)};
}

constexpr llvm::StringRef kAmdgpuIntrinsicLibraryName =
    "amdgpu_intrinsics.mlirbc";
constexpr llvm::StringRef kAmdgpuIntrinsicLibraryTag =
    "embedded:amdgpu_intrinsics.mlirbc";
static constexpr unsigned kDataFormatU32Config = 4u << 15;

static mlir::Location GetCallLocation(GeneratorContext *ctx,
                                      const ast::ASTNode *node) {
    auto *func_gen = ctx ? ctx->GetCurrentFunctionGenerator() : nullptr;
    auto *builder = func_gen ? &func_gen->GetBuilder() : nullptr;
    SS_ASSERT(ctx && builder);
    return ctx->GetMLIRLocation(builder->getContext(), node);
}

} // namespace

// AMDGPU Intrinsics Module
class AMDGPUIntrinsic : public NamedModule {
  public:
    explicit AMDGPUIntrinsic();

    void Initialize() override;
    void DeclareModules(mlir::ModuleOp module) override;

    mlir::Value CreateMfmaFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                   llvm::ArrayRef<mlir::Value> resolved_args,
                                   const amdgpu_mfma::MFMAConfig &config) const;
    mlir::Value CreateRawBufferLoadX1Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateRawBufferLoadX2Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateRawBufferLoadX4Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateRawBufferLoadX1LdsFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateRawBufferStoreX1Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateRawBufferStoreX2Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateRawBufferStoreX4Function(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateMakeRsrcFunction(ast::Call *call_expr, GeneratorContext *ctx,
                           llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreatePermFunction(ast::Call *call_expr, GeneratorContext *ctx,
                       llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateRcpFunction(ast::Call *call_expr, GeneratorContext *ctx,
                      llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateSWaitcntFunction(ast::Call *call_expr, GeneratorContext *ctx,
                           llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateReadFirstLaneFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateSchedGroupBarrierFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateSchedBarrierFunction(ast::Call *call_expr, GeneratorContext *ctx,
                               llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateSetPrioFunction(ast::Call *call_expr, GeneratorContext *ctx,
                          llvm::ArrayRef<mlir::Value> resolved_args) const;

  private:
    mlir::Value
    CreateGenericMFMAFunction(ast::Call *call_expr, GeneratorContext *ctx,
                              llvm::ArrayRef<mlir::Value> resolved_args,
                              const amdgpu_mfma::MFMAConfig &config) const;

    mlir::Value CreateGenericRawBufferLoadFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, int width) const;

    mlir::Value CreateGenericRawBufferStoreFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, int width) const;

    bool
    CheckGenericMFMAFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckGenericRawBufferLoadFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, int width) const;
    bool CheckRawBufferLoadX1LdsFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckGenericRawBufferStoreFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args, int width) const;

    bool CheckMakeRsrcFunction(ast::Call *call_expr, GeneratorContext *ctx,
                               llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckPermFunction(ast::Call *call_expr, GeneratorContext *ctx,
                           llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckRcpFunction(ast::Call *call_expr, GeneratorContext *ctx,
                          llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckSWaitcntFunction(ast::Call *call_expr, GeneratorContext *ctx,
                               llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckReadFirstLaneFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckSchedGroupBarrierFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckSchedBarrierFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckSetPrioFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
};

AMDGPUIntrinsic::AMDGPUIntrinsic() : NamedModule("amdgpu") {}

void AMDGPUIntrinsic::Initialize() {
    for (const auto &config : amdgpu_mfma::MFMAConfig::GetConfigs()) {
        AddFunction(
            config.name.str(),
            [this,
             config](ast::Call *call_expr, GeneratorContext *gen_ctx,
                     llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                return CreateMfmaFunction(call_expr, gen_ctx, resolved_args,
                                          config);
            },
            [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
                return CheckGenericMFMAFunction(call_expr, gen_ctx,
                                                resolved_args);
            });
    }

    AddFunction(
        "make_rsrc",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMakeRsrcFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMakeRsrcFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "perm",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreatePermFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckPermFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "rcp",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRcpFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckRcpFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "s_waitcnt",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateSWaitcntFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckSWaitcntFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "readfirstlane",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateReadFirstLaneFunction(call_expr, gen_ctx,
                                               resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckReadFirstLaneFunction(call_expr, gen_ctx,
                                              resolved_args);
        });

    AddFunction(
        "raw_buffer_load_x1",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferLoadX1Function(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericRawBufferLoadFunction(call_expr, gen_ctx,
                                                     resolved_args, 1);
        });

    AddFunction(
        "raw_buffer_load_x2",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferLoadX2Function(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericRawBufferLoadFunction(call_expr, gen_ctx,
                                                     resolved_args, 2);
        });

    AddFunction(
        "raw_buffer_load_x4",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferLoadX4Function(call_expr, gen_ctx,
                                                 resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericRawBufferLoadFunction(call_expr, gen_ctx,
                                                     resolved_args, 4);
        });

    AddFunction(
        "raw_buffer_load_x1_lds",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferLoadX1LdsFunction(call_expr, gen_ctx,
                                                    resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckRawBufferLoadX1LdsFunction(call_expr, gen_ctx,
                                                   resolved_args);
        });

    AddFunction(
        "raw_buffer_store_x1",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferStoreX1Function(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericRawBufferStoreFunction(call_expr, gen_ctx,
                                                      resolved_args, 1);
        });

    AddFunction(
        "raw_buffer_store_x2",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferStoreX2Function(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericRawBufferStoreFunction(call_expr, gen_ctx,
                                                      resolved_args, 2);
        });

    AddFunction(
        "raw_buffer_store_x4",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateRawBufferStoreX4Function(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckGenericRawBufferStoreFunction(call_expr, gen_ctx,
                                                      resolved_args, 4);
        });

    AddFunction(
        "sched_group_barrier",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateSchedGroupBarrierFunction(call_expr, gen_ctx,
                                                   resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckSchedGroupBarrierFunction(call_expr, gen_ctx,
                                                  resolved_args);
        });

    AddFunction(
        "sched_barrier",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateSchedBarrierFunction(call_expr, gen_ctx,
                                              resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckSchedBarrierFunction(call_expr, gen_ctx, resolved_args);
        });

    AddFunction(
        "s_setprio",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateSetPrioFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckSetPrioFunction(call_expr, gen_ctx, resolved_args);
        });

}

void AMDGPUIntrinsic::DeclareModules(mlir::ModuleOp module) {
    if (!module)
        return;

    // Register the bytecode in the intrinsic registry
    auto libraryBytes = GetAmdgpuIntrinsicLibrary();
    auto &registry = utils::EmbeddedFilesystemView::getInstance();
    registry.registerFile(std::string(kAmdgpuIntrinsicLibraryName),
                          libraryBytes);

    intrinsics::GetOrCreateImplementationContainer(module, "amdgpu",
                                                   kAmdgpuIntrinsicLibraryTag);

    auto loadDialects = [](mlir::MLIRContext *ctx) {
        ctx->loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::vector::VectorDialect, mlir::LLVM::LLVMDialect,
                         mlir::ROCDL::ROCDLDialect, cf::AveLangDialect>();
    };

    if (failed(intrinsics::EnsureIntrinsicDeclarations(
            module, kAmdgpuIntrinsicLibraryName, libraryBytes, loadDialects))) {
        module.emitError() << "failed to declare AMDGPU intrinsics";
    }
}

namespace {
mlir::Type GetMfmaElemType(amdgpu_mfma::VectorElemKind kind,
                           mlir::OpBuilder &builder) {
    switch (kind) {
    case amdgpu_mfma::VectorElemKind::I32:
        return builder.getI32Type();
    case amdgpu_mfma::VectorElemKind::F16:
        return builder.getF16Type();
    case amdgpu_mfma::VectorElemKind::F32:
        return builder.getF32Type();
    case amdgpu_mfma::VectorElemKind::BF16:
        return builder.getBF16Type();
    }
    llvm_unreachable("Unsupported MFMA element kind");
}

mlir::Value ConvertToI32(mlir::OpBuilder &builder, mlir::Location location,
                         mlir::Value value) {
    if (value.getType().isIndex()) {
        return mlir::arith::IndexCastOp::create(builder, location,
                                                builder.getI32Type(), value);
    }

    auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
    if (!intType || intType.getWidth() == 32) {
        return value;
    }
    if (intType.getWidth() < 32) {
        return mlir::arith::ExtUIOp::create(builder, location,
                                            builder.getI32Type(), value);
    }
    return mlir::arith::TruncIOp::create(builder, location,
                                         builder.getI32Type(), value);
}

} // namespace

mlir::Value AMDGPUIntrinsic::CreateGenericMFMAFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args,
    const amdgpu_mfma::MFMAConfig &config) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);

    auto a = resolved_args[0];
    auto b = resolved_args[1];
    auto c = resolved_args[2];

    auto type_a = GetMfmaElemType(config.aElem, builder);
    auto type_c = GetMfmaElemType(config.cElem, builder);

    int64_t c_elements = config.GetCElementCount();
    auto result_vector_type = mlir::VectorType::get({c_elements}, type_c);

    // Convert MLIR types to string representations for attributes
    std::string type_a_str, type_c_str;
    llvm::raw_string_ostream type_a_stream(type_a_str);
    llvm::raw_string_ostream type_c_stream(type_c_str);

    if (type_a.isF16()) {
        type_a_stream << "f16";
    } else if (type_a.isBF16()) {
        type_a_stream << "bf16";
    } else if (type_a.isF32()) {
        type_a_stream << "f32";
    } else if (type_a.isInteger(32)) {
        type_a_stream << "i32";
    } else if (type_a.isInteger(8)) {
        type_a_stream << "i8";
    } else {
        type_a_stream << type_a;
    }

    if (type_c.isF32()) {
        type_c_stream << "f32";
    } else if (type_c.isF16()) {
        type_c_stream << "f16";
    } else if (type_c.isInteger(32)) {
        type_c_stream << "i32";
    } else {
        type_c_stream << type_c;
    }

    // Create GPUOp AMDGPU MFMA operation with config attributes
    auto mfma_op = cf::AMDGPUMfmaOp::create(
        builder, location, result_vector_type, a, b, c,
        mlir::IntegerAttr::get(builder.getI32Type(), config.m),
        mlir::IntegerAttr::get(builder.getI32Type(), config.n),
        mlir::IntegerAttr::get(builder.getI32Type(), config.k),
        mlir::StringAttr::get(builder.getContext(), type_a_str),
        mlir::StringAttr::get(builder.getContext(), type_c_str));

    return mfma_op.getResult();
}

mlir::Value AMDGPUIntrinsic::CreateMfmaFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args,
    const amdgpu_mfma::MFMAConfig &config) const {
    return CreateGenericMFMAFunction(call_expr, ctx, resolved_args, config);
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferLoadX1Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateGenericRawBufferLoadFunction(call_expr, ctx, resolved_args, 1);
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferLoadX2Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateGenericRawBufferLoadFunction(call_expr, ctx, resolved_args, 2);
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferLoadX4Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateGenericRawBufferLoadFunction(call_expr, ctx, resolved_args, 4);
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferLoadX1LdsFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();
    auto size = ConstantFolder::FoldIntValue(resolved_args[2]);
    auto aux = ConstantFolder::FoldIntValue(resolved_args[6]);

    if (!size || *size != 4 || !aux || *aux != 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x1_lds currently requires compile-time "
               "size=4 and aux=0";
        return nullptr;
    }

    auto ldsPtr = cf::AveLangMemRefExtractAlignedPointerAsIndexOp::create(
        builder, location, builder.getIndexType(), resolved_args[1]);

    auto funcName = intrinsics::MakeIntrinsicFuncName(
        "amdgpu", "llvm_amdgcn_raw_buffer_load_lds_u32");
    mlir::func::CallOp::create(
        builder, location, funcName, mlir::TypeRange{},
        mlir::ValueRange{resolved_args[0], ldsPtr,
                         ConvertToI32(builder, location, resolved_args[5]),
                         ConvertToI32(builder, location, resolved_args[3]),
                         ConvertToI32(builder, location, resolved_args[4])});

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferStoreX1Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateGenericRawBufferStoreFunction(call_expr, ctx, resolved_args,
                                               1);
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferStoreX2Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateGenericRawBufferStoreFunction(call_expr, ctx, resolved_args,
                                               2);
}

mlir::Value AMDGPUIntrinsic::CreateRawBufferStoreX4Function(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    return CreateGenericRawBufferStoreFunction(call_expr, ctx, resolved_args,
                                               4);
}

mlir::Value AMDGPUIntrinsic::CreateMakeRsrcFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);

    auto memref = resolved_args[0];
    auto rangeValue = resolved_args[1];

    auto ptrIndex = cf::AveLangMemRefExtractAlignedPointerAsIndexOp::create(
        builder, location, builder.getIndexType(), memref);
    auto ptrI64 = mlir::arith::IndexCastOp::create(
        builder, location, builder.getI64Type(), ptrIndex);

    auto shiftAmount =
        mlir::arith::ConstantIntOp::create(builder, location, 32, 64);
    auto ptrHighI64 =
        mlir::arith::ShRUIOp::create(builder, location, ptrI64, shiftAmount);
    auto ptrLowI32 = mlir::arith::TruncIOp::create(
        builder, location, builder.getI32Type(), ptrI64);
    auto ptrHighI32 = mlir::arith::TruncIOp::create(
        builder, location, builder.getI32Type(), ptrHighI64);
    mlir::Value rangeI32 = rangeValue;
    if (rangeI32.getType().isIndex()) {
        rangeI32 = mlir::arith::IndexCastOp::create(
            builder, location, builder.getI32Type(), rangeI32);
    } else if (auto intType =
                   mlir::dyn_cast<mlir::IntegerType>(rangeI32.getType())) {
        if (intType.getWidth() < 32) {
            rangeI32 = mlir::arith::ExtUIOp::create(
                builder, location, builder.getI32Type(), rangeI32);
        } else if (intType.getWidth() > 32) {
            rangeI32 = mlir::arith::TruncIOp::create(
                builder, location, builder.getI32Type(), rangeI32);
        }
    }
    auto configI32 = mlir::arith::ConstantIntOp::create(
        builder, location, kDataFormatU32Config, 32);

    auto rsrcType = mlir::VectorType::get({4}, builder.getI32Type());
    auto zeroAttr = builder.getIntegerAttr(builder.getI32Type(), 0);
    auto zeroRsrcAttr = mlir::DenseElementsAttr::get(rsrcType, zeroAttr);
    mlir::Value rsrc = mlir::arith::ConstantOp::create(builder, location,
                                                       rsrcType, zeroRsrcAttr);

    rsrc = mlir::vector::InsertOp::create(
        builder, location, ptrLowI32, rsrc,
        llvm::SmallVector<mlir::OpFoldResult>{builder.getI64IntegerAttr(0)});
    rsrc = mlir::vector::InsertOp::create(
        builder, location, ptrHighI32, rsrc,
        llvm::SmallVector<mlir::OpFoldResult>{builder.getI64IntegerAttr(1)});
    rsrc = mlir::vector::InsertOp::create(
        builder, location, rangeI32, rsrc,
        llvm::SmallVector<mlir::OpFoldResult>{builder.getI64IntegerAttr(2)});
    rsrc = mlir::vector::InsertOp::create(
        builder, location, configI32, rsrc,
        llvm::SmallVector<mlir::OpFoldResult>{builder.getI64IntegerAttr(3)});

    return rsrc;
}

mlir::Value AMDGPUIntrinsic::CreateRcpFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto funcName =
        intrinsics::MakeIntrinsicFuncName("amdgpu", "llvm_amdgcn_rcp_f32");
    auto callOp = mlir::func::CallOp::create(
        builder, location, funcName, builder.getF32Type(), resolved_args[0]);
    return callOp.getResult(0);
}

mlir::Value AMDGPUIntrinsic::CreateSWaitcntFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto vmcnt = ConstantFolder::FoldIntValue(resolved_args[0]);
    auto expcnt = ConstantFolder::FoldIntValue(resolved_args[1]);
    auto lgkmcnt = ConstantFolder::FoldIntValue(resolved_args[2]);

    auto is_valid = [](std::optional<int64_t> value, int64_t max) {
        return value && *value >= 0 && *value <= max;
    };

    if (!is_valid(vmcnt, 63) || !is_valid(expcnt, 7) ||
        !is_valid(lgkmcnt, 15)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "s_waitcnt(vmcnt, expcnt, lgkmcnt) requires compile-time "
               "integer arguments in ranges vmcnt=[0,63], expcnt=[0,7], "
               "lgkmcnt=[0,15]";
        return nullptr;
    }

    auto vmcntU = static_cast<uint32_t>(*vmcnt);
    auto waitcntValue = ((vmcntU & 48u) << 10) |
                        ((static_cast<uint32_t>(*lgkmcnt) & 15u) << 8) |
                        ((static_cast<uint32_t>(*expcnt) & 7u) << 4) |
                        (vmcntU & 15u);
    auto waitcntAttr =
        mlir::IntegerAttr::get(builder.getI32Type(), waitcntValue);
    mlir::ROCDL::SWaitcntOp::create(builder, location, waitcntAttr);
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value AMDGPUIntrinsic::CreatePermFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto funcName =
        intrinsics::MakeIntrinsicFuncName("amdgpu", "llvm_amdgcn_perm");
    auto callOp = mlir::func::CallOp::create(
        builder, location, funcName, builder.getI32Type(), resolved_args);
    return callOp.getResult(0);
}

mlir::Value AMDGPUIntrinsic::CreateReadFirstLaneFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto value = ConvertToI32(builder, location, resolved_args[0]);
    auto op = mlir::ROCDL::ReadfirstlaneOp::create(builder, location,
                                                   value.getType(), value);
    return op.getResult();
}

mlir::Value AMDGPUIntrinsic::CreateSchedGroupBarrierFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);

    auto mask = ConstantFolder::FoldIntValue(resolved_args[0]);
    auto size = ConstantFolder::FoldIntValue(resolved_args[1]);
    auto group_id = ConstantFolder::FoldIntValue(resolved_args[2]);

    auto is_valid_u32 = [](std::optional<int64_t> value) {
        return value && *value >= 0 &&
               static_cast<uint64_t>(*value) <=
                   static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    };

    if (!is_valid_u32(mask) || !is_valid_u32(size) || !is_valid_u32(group_id)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "sched_group_barrier(mask, size, group_id) requires compile-"
               "time non-negative integer arguments <= 2^32-1";
        return nullptr;
    }

    mlir::ROCDL::SchedGroupBarrier::create(
        builder, location, static_cast<uint32_t>(*mask),
        static_cast<uint32_t>(*size), static_cast<uint32_t>(*group_id));

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value AMDGPUIntrinsic::CreateSchedBarrierFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto mask = ConstantFolder::FoldIntValue(resolved_args[0]);

    if (!mask || *mask < 0 ||
        static_cast<uint64_t>(*mask) >
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "sched_barrier(mask) requires a compile-time non-negative "
               "integer argument <= 2^32-1";
        return nullptr;
    }

    mlir::ROCDL::SchedBarrier::create(builder, location,
                                      static_cast<uint32_t>(*mask));
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value AMDGPUIntrinsic::CreateSetPrioFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);
    auto priority = ConstantFolder::FoldIntValue(resolved_args[0]);

    if (!priority || *priority < 0 ||
        static_cast<uint64_t>(*priority) >
            static_cast<uint64_t>(std::numeric_limits<uint16_t>::max())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "s_setprio(priority) requires a compile-time non-negative "
               "integer argument <= 2^16-1";
        return nullptr;
    }

    mlir::ROCDL::SetPrioOp::create(builder, location,
                                   static_cast<uint16_t>(*priority));
    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

mlir::Value AMDGPUIntrinsic::CreateGenericRawBufferLoadFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int width) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);

    auto rsrc = resolved_args[0];
    auto vindex = resolved_args[1];
    auto soffset = resolved_args[2];
    auto aux = resolved_args[3];

    mlir::Type result_type;
    if (width == 1) {
        result_type = builder.getI32Type();
    } else if (width == 2) {
        result_type = mlir::VectorType::get({2}, builder.getI32Type());
    } else if (width == 4) {
        result_type = mlir::VectorType::get({4}, builder.getI32Type());
    } else {
        llvm_unreachable("Unsupported raw_buffer_load width");
    }

    // Create GPUOp AMDGPU Raw Buffer Load operation
    auto load_op = cf::AMDGPURawBufferLoadOp::create(
        builder, location, result_type, rsrc, vindex, soffset, aux);

    return load_op.getResult();
}

mlir::Value AMDGPUIntrinsic::CreateGenericRawBufferStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int width) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = GetCallLocation(ctx, call_expr);

    auto vdata = resolved_args[0];
    auto rsrc = resolved_args[1];
    auto vindex = resolved_args[2];
    auto soffset = resolved_args[3];
    auto aux = resolved_args[4];

    auto vdata_type = vdata.getType();

    if (width > 1) {
        auto vector_type = mlir::cast<mlir::VectorType>(vdata_type);
        (void)vector_type;
    }

    // Create GPUOp AMDGPU Raw Buffer Store operation
    cf::AMDGPURawBufferStoreOp::create(builder, location, vdata, rsrc, vindex,
                                       soffset, aux);

    return ctx->GetCurrentFunctionGenerator()
        ->GetExprGenerator()
        ->CreateVoidValue();
}

bool AMDGPUIntrinsic::CheckGenericMFMAFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mfma operation requires exactly 3 arguments: a, b, c";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for mfma operation";
        return false;
    }

    if (!mlir::dyn_cast<mlir::VectorType>(resolved_args[0].getType()) ||
        !mlir::dyn_cast<mlir::VectorType>(resolved_args[1].getType()) ||
        !mlir::dyn_cast<mlir::VectorType>(resolved_args[2].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "MFMA operands must be vector types";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckGenericRawBufferLoadFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int width) const {
    if (resolved_args.size() != 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x" << width
            << " requires exactly 4 arguments: rsrc, vindex, soffset, aux";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2] ||
        !resolved_args[3]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for raw_buffer_load_x" << width;
        return false;
    }

    if (width != 1 && width != 2 && width != 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Unsupported width for raw_buffer_load_x" << width
            << " (supported: 1, 2, 4)";
        return false;
    }

    auto rsrc_type = resolved_args[0].getType();
    auto rsrc_vector = mlir::dyn_cast<mlir::VectorType>(rsrc_type);
    if (!rsrc_vector || rsrc_vector.getNumElements() != 4 ||
        !rsrc_vector.getElementType().isInteger(32)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x" << width
            << " expects rsrc to be vector<4xi32>";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckRawBufferLoadX1LdsFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 7) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x1_lds requires exactly 7 arguments: "
               "rsrc, lds_ptr, size, vindex, soffset, offset, aux";
        return false;
    }

    for (auto value : resolved_args) {
        if (!value) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate operands for raw_buffer_load_x1_lds";
            return false;
        }
    }

    auto rsrcType = resolved_args[0].getType();
    auto rsrcVector = mlir::dyn_cast<mlir::VectorType>(rsrcType);
    if (!rsrcVector || rsrcVector.getNumElements() != 4 ||
        !rsrcVector.getElementType().isInteger(32)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x1_lds expects rsrc to be vector<4xi32>";
        return false;
    }

    auto memrefType =
        mlir::dyn_cast<cf::MemRefType>(resolved_args[1].getType());
    if (!memrefType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x1_lds expects lds_ptr to be a memref";
        return false;
    }

    for (auto operand : {resolved_args[2], resolved_args[3], resolved_args[4],
                         resolved_args[5], resolved_args[6]}) {
        if (!operand.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "raw_buffer_load_x1_lds size/vindex/soffset/offset/aux "
                   "operands must be integer or index types";
            return false;
        }
    }

    auto size = ConstantFolder::FoldIntValue(resolved_args[2]);
    auto aux = ConstantFolder::FoldIntValue(resolved_args[6]);
    if (!size || *size != 4 || !aux || *aux != 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_load_x1_lds currently requires compile-time "
               "size=4 and aux=0";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckGenericRawBufferStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args, int width) const {
    if (resolved_args.size() != 5) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_store_x" << width
            << " requires exactly 5 arguments: vdata, rsrc, vindex, soffset, "
               "aux";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2] ||
        !resolved_args[3] || !resolved_args[4]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for raw_buffer_store_x" << width;
        return false;
    }

    if (width != 1 && width != 2 && width != 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Unsupported width for raw_buffer_store_x" << width
            << " (supported: 1, 2, 4)";
        return false;
    }

    auto rsrc_type = resolved_args[1].getType();
    auto rsrc_vector = mlir::dyn_cast<mlir::VectorType>(rsrc_type);
    if (!rsrc_vector || rsrc_vector.getNumElements() != 4 ||
        !rsrc_vector.getElementType().isInteger(32)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "raw_buffer_store_x" << width
            << " expects rsrc to be vector<4xi32>";
        return false;
    }

    auto vdata_type = resolved_args[0].getType();
    if (width == 1) {
        if (!mlir::isa<mlir::IntegerType>(vdata_type)) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "raw_buffer_store_x1 expects i32 data type";
            return false;
        }
    } else {
        auto vector_type = mlir::dyn_cast<mlir::VectorType>(vdata_type);
        if (!vector_type || vector_type.getNumElements() != width ||
            !mlir::isa<mlir::IntegerType>(vector_type.getElementType())) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "raw_buffer_store_x" << width << " expects vector<" << width
                << "xi32> data type";
            return false;
        }
    }

    return true;
}

bool AMDGPUIntrinsic::CheckMakeRsrcFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (call_expr->GetArgs().size() != 2 || resolved_args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_rsrc() requires exactly 2 arguments: tensor, range";
        return false;
    }

    auto tensor = resolved_args[0];
    auto range = resolved_args[1];
    if (!tensor) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve tensor argument for make_rsrc()";
        return false;
    }
    if (!range) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to resolve range argument for make_rsrc()";
        return false;
    }

    auto memrefType = mlir::dyn_cast<cf::MemRefType>(tensor.getType());
    if (!memrefType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_rsrc() expects a tensor argument";
        return false;
    }

    if (!range.getType().isIndex() &&
        !mlir::isa<mlir::IntegerType>(range.getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_rsrc() expects range to be an integer or index value";
        return false;
    }

    auto rangeValue = ConstantFolder::FoldIntValue(range);
    if (rangeValue &&
        (*rangeValue < 0 ||
         static_cast<uint64_t>(*rangeValue) >
             static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_rsrc() requires range to be in [0, 2^32-1]";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckRcpFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (call_expr->GetArgs().size() != 1 || resolved_args.size() != 1 ||
        !resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "rcp() requires exactly 1 argument";
        return false;
    }

    if (!resolved_args[0].getType().isF32()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "rcp() expects an f32 argument";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckSWaitcntFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "s_waitcnt requires exactly 3 arguments: vmcnt, expcnt, "
               "lgkmcnt";
        return false;
    }

    for (auto value : resolved_args) {
        if (!value) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "Failed to generate operands for s_waitcnt";
            return false;
        }
        if (!value.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "s_waitcnt operands must be integer or index types";
            return false;
        }
    }

    auto vmcnt = ConstantFolder::FoldIntValue(resolved_args[0]);
    auto expcnt = ConstantFolder::FoldIntValue(resolved_args[1]);
    auto lgkmcnt = ConstantFolder::FoldIntValue(resolved_args[2]);
    auto is_valid = [](std::optional<int64_t> value, int64_t max) {
        return value && *value >= 0 && *value <= max;
    };
    if (!is_valid(vmcnt, 63) || !is_valid(expcnt, 7) ||
        !is_valid(lgkmcnt, 15)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "s_waitcnt(vmcnt, expcnt, lgkmcnt) requires compile-time "
               "integer arguments in ranges vmcnt=[0,63], expcnt=[0,7], "
               "lgkmcnt=[0,15]";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckPermFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (call_expr->GetArgs().size() != 3 || resolved_args.size() != 3 ||
        !resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "perm() requires exactly 3 arguments";
        return false;
    }

    for (auto arg : resolved_args) {
        auto intType = mlir::dyn_cast<mlir::IntegerType>(arg.getType());
        if (!intType || intType.getWidth() != 32) {
            ctx->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                call_expr->GetSourceRange().getBegin())
                << "perm() expects 32-bit integer arguments";
            return false;
        }
    }

    return true;
}

bool AMDGPUIntrinsic::CheckReadFirstLaneFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (call_expr->GetArgs().size() != 1 || resolved_args.size() != 1 ||
        !resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "readfirstlane() requires exactly 1 argument";
        return false;
    }

    if (!resolved_args[0].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "readfirstlane() expects an integer or index argument";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckSchedGroupBarrierFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (call_expr->GetArgs().size() != 3 || resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "sched_group_barrier() requires exactly 3 arguments: "
               "mask, size, group_id";
        return false;
    }

    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operands for sched_group_barrier()";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckSchedBarrierFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (call_expr->GetArgs().size() != 1 || resolved_args.size() != 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "sched_barrier() requires exactly 1 argument: mask";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operand for sched_barrier()";
        return false;
    }

    return true;
}

bool AMDGPUIntrinsic::CheckSetPrioFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (call_expr->GetArgs().size() != 1 || resolved_args.size() != 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "s_setprio() requires exactly 1 argument: priority";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate operand for s_setprio()";
        return false;
    }

    return true;
}

// Factory function to create AMDGPU intrinsic module
std::unique_ptr<NamedModule> CreateAMDGPUIntrinsicModule() {
    return std::make_unique<AMDGPUIntrinsic>();
}

} // namespace causalflow::avelang::ir
