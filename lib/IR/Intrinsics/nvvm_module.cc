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
#include <mlir/Dialect/Arith/Utils/Utils.h>
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

static void emitInlinePtxVoid(mlir::OpBuilder &builder, mlir::Location loc,
                              llvm::StringRef asmString) {
    mlir::LLVM::InlineAsmOp::create(
        builder, loc, mlir::TypeRange{}, mlir::ValueRange{}, asmString, "",
        /*hasSideEffects=*/true, /*isAlignStack=*/false,
        mlir::LLVM::tailcallkind::TailCallKind::None,
        mlir::LLVM::AsmDialectAttr{}, mlir::ArrayAttr{});
}

static mlir::MemRefType getBuiltinTensorMapMemRefType(mlir::Type type) {
    if (auto builtinType = mlir::dyn_cast<mlir::MemRefType>(type)) {
        return builtinType;
    }

    auto substrateType = mlir::dyn_cast<cf::MemRefType>(type);
    if (!substrateType) {
        return {};
    }

    mlir::MemRefLayoutAttrInterface layout;
    auto strides = substrateType.getStrides();
    if (!strides.empty()) {
        layout = mlir::StridedLayoutAttr::get(type.getContext(),
                                              /*offset=*/0, strides);
    }

    return mlir::MemRefType::get(substrateType.getShape(),
                                 substrateType.getElementType(), layout,
                                 substrateType.getMemorySpace());
}

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
    mlir::Value CreateWgmmaFenceAlignedFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateWgmmaGroupSyncAlignedFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateWgmmaWaitGroupSyncFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateMakeWGMMADescriptorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateMakeTMADescriptorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateTMAFenceFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateTMALoadFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateTMAStoreFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateWgmmaAsyncFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateWgmmaInitAccumulatorFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateWgmmaStoreFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;

    mlir::Value CreateMBarrierCreateFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateMBarrierInitFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateMBarrierTryWaitParityFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateMBarrierArriveFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateMBarrierTestWaitFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateMBarrierArriveExpectTxFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateCpAsyncCaSharedGlobalFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateCpAsyncCommitGroupFunction(
        ast::Call *call_expr, GeneratorContext *ctx,
        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value CreateCpAsyncWaitGroupFunction(
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

    bool CheckWgmmaFenceAlignedFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                        llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaGroupSyncAlignedFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                            llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaWaitGroupSyncFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                         llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaAsyncFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                         llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaInitAccumulatorFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                         llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckWgmmaStoreFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                         llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMBarrierCreateFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckMBarrierInitFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckMBarrierTryWaitParityFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckMBarrierArriveFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckMBarrierTestWaitFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckMBarrierArriveExpectTxFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckCpAsyncCaSharedGlobalFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckCpAsyncCommitGroupFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckCpAsyncWaitGroupFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;

    bool CheckMakeWGMMADescriptorFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckMakeTMADescriptorFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckTMAFenceFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckTMALoadFunction(ast::Call *call_expr, GeneratorContext *ctx,
                                     llvm::ArrayRef<mlir::Value> resolved_args) const;
    bool CheckTMAStoreFunction(ast::Call *call_expr, GeneratorContext *ctx,
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

    // wgmma operations
    AddFunction(
        "wgmma_fence_aligned",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaFenceAlignedFunction(call_expr, gen_ctx,
                                                    resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaFenceAlignedFunction(call_expr, gen_ctx,
                                                   resolved_args);
        });

    AddFunction(
        "wgmma_group_sync_aligned",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaGroupSyncAlignedFunction(call_expr, gen_ctx,
                                                        resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaGroupSyncAlignedFunction(call_expr, gen_ctx,
                                                      resolved_args);
        });

    AddFunction(
        "wgmma_wait_group_sync",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaWaitGroupSyncFunction(call_expr, gen_ctx,
                                                     resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaWaitGroupSyncFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });

    AddFunction(
        "make_wgmma_descriptor",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMakeWGMMADescriptorFunction(call_expr, gen_ctx,
                                                     resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMakeWGMMADescriptorFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });
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
    AddFunction(
        "tma_fence",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateTMAFenceFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckTMAFenceFunction(call_expr, gen_ctx, resolved_args);
        });
    AddFunction(
        "tma_load",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateTMALoadFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckTMALoadFunction(call_expr, gen_ctx, resolved_args);
        });
    AddFunction(
        "tma_store",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateTMAStoreFunction(call_expr, gen_ctx, resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckTMAStoreFunction(call_expr, gen_ctx, resolved_args);
        });
    AddFunction(
        "wgmma_async",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaAsyncFunction(call_expr, gen_ctx,
                                                     resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaAsyncFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });
    AddFunction(
        "wgmma_init_accumulator",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaInitAccumulatorFunction(call_expr, gen_ctx,
                                                     resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaInitAccumulatorFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });
    AddFunction(
        "wgmma_store",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateWgmmaStoreFunction(call_expr, gen_ctx,
                                                     resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckWgmmaStoreFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });

    // mbarrier functions
    AddFunction(
        "mbarrier_create",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierCreateFunction(call_expr, gen_ctx,
                                                     resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierCreateFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });

    AddFunction(
        "mbarrier_init",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierInitFunction(call_expr, gen_ctx,
                                                     resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierInitFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });

    AddFunction(
        "mbarrier_try_wait_parity",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierTryWaitParityFunction(call_expr, gen_ctx,
                                                     resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierTryWaitParityFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });

    AddFunction(
        "mbarrier_arrive",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierArriveFunction(call_expr, gen_ctx,
                                                     resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierArriveFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });

    AddFunction(
        "mbarrier_test_wait",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierTestWaitFunction(call_expr, gen_ctx,
                                                     resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierTestWaitFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });

    AddFunction(
        "mbarrier_arrive_expect_tx",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateMBarrierArriveExpectTxFunction(call_expr, gen_ctx,
                                                     resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckMBarrierArriveExpectTxFunction(call_expr, gen_ctx,
                                                    resolved_args);
        });

    AddFunction(
        "cp_async_ca_shared_global",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateCpAsyncCaSharedGlobalFunction(call_expr, gen_ctx,
                                                       resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckCpAsyncCaSharedGlobalFunction(call_expr, gen_ctx,
                                                      resolved_args);
        });

    AddFunction(
        "cp_async_commit_group",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateCpAsyncCommitGroupFunction(call_expr, gen_ctx,
                                                    resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckCpAsyncCommitGroupFunction(call_expr, gen_ctx,
                                                   resolved_args);
        });

    AddFunction(
        "cp_async_wait_group",
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
            return CreateCpAsyncWaitGroupFunction(call_expr, gen_ctx,
                                                  resolved_args);
        },
        [this](ast::Call *call_expr, GeneratorContext *gen_ctx,
               llvm::ArrayRef<mlir::Value> resolved_args) -> bool {
            return CheckCpAsyncWaitGroupFunction(call_expr, gen_ctx,
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

mlir::Value NVVMIntrinsic::CreateWgmmaFenceAlignedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    emitInlinePtxVoid(builder, location, "wgmma.fence.sync.aligned;");

    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

bool NVVMIntrinsic::CheckWgmmaFenceAlignedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_fence_aligned requires no arguments";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaGroupSyncAlignedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    emitInlinePtxVoid(builder, location, "wgmma.commit_group.sync.aligned;");

    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

bool NVVMIntrinsic::CheckWgmmaGroupSyncAlignedFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_group_sync_aligned requires no arguments";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaWaitGroupSyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (resolved_args.size() != 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_wait_group_sync requires exactly 1 argument: group";
        return nullptr;
    }

    auto group = resolved_args[0];
    if (!group) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate group operand for wgmma_wait_group_sync";
        return nullptr;
    }

    // The group parameter needs to be a constant value to use as attribute
    int64_t group_value = 0;
    bool found_constant = false;

    // Try to get constant value from defining operation
    if (auto const_op = group.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            group_value = int_attr.getInt();
            found_constant = true;
        }
    }

    if (!found_constant) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_wait_group_sync requires a constant integer value for group";
        return nullptr;
    }

    std::string asmString =
        "wgmma.wait_group.sync.aligned " + std::to_string(group_value) + ";";
    emitInlinePtxVoid(builder, location, asmString);

    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

bool NVVMIntrinsic::CheckWgmmaWaitGroupSyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_wait_group_sync requires exactly 1 argument: group";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate group operand for wgmma_wait_group_sync";
        return false;
    }

    auto group_type = resolved_args[0].getType();
    if (!group_type.isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_wait_group_sync group operand must be integer type";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierCreateFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierCreateFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    return mlir::nvgpu::MBarrierCreateOp::create(
        builder, location,
        nvgpu::MBarrierGroupType::get(builder.getContext(),
        mlir::gpu::AddressSpaceAttr::get(
            builder.getContext(), mlir::gpu::AddressSpace::Workgroup)));
}

bool NVVMIntrinsic::CheckMBarrierCreateFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!resolved_args.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_create does not take any arguments";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierInitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierInitFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    const auto &keywords = call_expr->GetKeywords();
    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    mlir::Value mbarId =
        positionalCount > 1 ? resolved_args[1] : mlir::Value();
    mlir::Value count =
        positionalCount > 2
            ? resolved_args[2]
            : arith::ConstantIndexOp::create(builder, location, 0).getResult();
    mlir::Value predicate =
        positionalCount > 3
            ? resolved_args[3]
            : mlir::Value();

    for (size_t i = 0; i < keywordCount; ++i) {
        mlir::Value value = resolved_args[positionalCount + i];
        llvm::StringRef name = keywords[i];
        if (name == "mbar_id") {
            mbarId = value;
        } else if (name == "count") {
            count = value;
        } else if (name == "predicate") {
            predicate = value;
        }
    }

    auto mbarIdValue = getConstantIntValue(mbarId);
    if (!mbarIdValue) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_init requires a constant integer value for mbarId";
        return nullptr;
    }

    if (!count.getType().isIndex()) {
        count = arith::IndexCastOp::create(builder, location, builder.getIndexType(), count);
    }
    if (predicate) {
        if (predicate.getType().isIndex()) {
            predicate = arith::IndexCastOp::create(builder, location, builder.getI1Type(), predicate);
        } else if (auto intType =
                       mlir::dyn_cast<mlir::IntegerType>(predicate.getType())) {
            if (intType.getWidth() != 1) {
                predicate = arith::TruncIOp::create(builder, location, builder.getI1Type(), predicate);
            }
        }
    }

    mlir::nvgpu::MBarrierInitOp::create(builder, location, resolved_args[0], count,
                                       arith::ConstantIndexOp::create(builder, location, *mbarIdValue),
                                       predicate);
    mlir::gpu::BarrierOp::create(builder, location);

    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

bool NVVMIntrinsic::CheckMBarrierInitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    const auto &keywords = call_expr->GetKeywords();
    if (resolved_args.size() < keywords.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_init internal argument mismatch";
        return false;
    }

    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;
    if (positionalCount < 1 || positionalCount > 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_init requires barrier, mbarId and optional count, predicate";
        return false;
    }

    for (auto name : keywords) {
        if (name != "mbar_id" && name != "count" && name != "predicate") {
            ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                            call_expr->GetSourceRange().getBegin())
                << "mbarrier_init got unsupported keyword argument '" << name << "'";
            return false;
        }
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate barrier operand for mbarrier_init";
        return false;
    }

    auto keywordValue = [&](llvm::StringRef wanted) -> mlir::Value {
        for (size_t i = 0; i < keywordCount; ++i) {
            if (keywords[i] == wanted) {
                return resolved_args[positionalCount + i];
            }
        }
        return {};
    };

    mlir::Value mbarId = positionalCount > 1 ? resolved_args[1]
                                             : keywordValue("mbar_id");
    if (!mbarId) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate mbarId operand for mbarrier_init";
        return false;
    }

    auto barrier_type = resolved_args[0].getType();
    if (!mlir::isa<nvgpu::MBarrierGroupType>(barrier_type)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_init operand must be of type mbarrier_group_t";
        return false;
    }

    auto mbarId_type = mbarId.getType();
    if (!mbarId_type.isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_init mbarId operand must be an integer type";
        return false;
    }

    auto checkIntArg = [&](mlir::Value value, llvm::StringRef name) -> bool {
        if (value && !value.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                            call_expr->GetSourceRange().getBegin())
                << "mbarrier_init " << name
                << " operand must be an integer type";
            return false;
        }
        return true;
    };

    if (positionalCount > 2 && !checkIntArg(resolved_args[2], "count")) {
        return false;
    }
    if (positionalCount > 3 && !checkIntArg(resolved_args[3], "predicate")) {
        return false;
    }
    for (size_t i = 0; i < keywordCount; ++i) {
        if (!checkIntArg(resolved_args[positionalCount + i], keywords[i])) {
            return false;
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierTryWaitParityFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierTryWaitParityFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    int64_t parity_value = 0, ticks_value = 0, mbarId_value = 0;
    bool found_constant = false;

    if (auto const_op = resolved_args[1].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            parity_value = int_attr.getInt();
            found_constant = true;
        }
    }

    if (!found_constant) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_try_wait_parity requires a constant integer value for phaseParity";
        return nullptr;
    }

    found_constant = false;
    if (auto const_op = resolved_args[2].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            ticks_value = int_attr.getInt();
            found_constant = true;
        }
    }

    if (!found_constant) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_try_wait_parity requires a constant integer value for ticks";
        return nullptr;
    }

    found_constant = false;
    if (auto const_op = resolved_args[3].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            mbarId_value = int_attr.getInt();
            found_constant = true;
        }
    }

    if (!found_constant) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_try_wait_parity requires a constant integer value for mbarId";
        return nullptr;
    }

    mlir::nvgpu::MBarrierTryWaitParityOp::create(builder, location, resolved_args[0], mlir::LLVM::ConstantOp::create(builder, location, builder.getI1Type(), parity_value), arith::ConstantIndexOp::create(builder, location, ticks_value), arith::ConstantIndexOp::create(builder, location, mbarId_value));

    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

bool NVVMIntrinsic::CheckMBarrierTryWaitParityFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_try_wait_parity requires exactly 4 arguments: barrier, phaseParity, ticks, mbarId";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate barrier operand for mbarrier_try_wait_parity";
        return false;
    }

    if (!resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate phaseParity operand for mbarrier_try_wait_parity";
        return false;
    }

    if (!resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate ticks operand for mbarrier_try_wait_parity";
        return false;
    }

    if (!resolved_args[3]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate mbarId operand for mbarrier_try_wait_parity";
        return false;
    }

    auto barrier_type = resolved_args[0].getType();
    if (!mlir::isa<nvgpu::MBarrierGroupType>(barrier_type)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_try_wait_parity operand must be of type mbarrier_group_t";
        return false;
    }

    if (!resolved_args[1].getType().isIntOrIndex() || !resolved_args[2].getType().isIntOrIndex() || !resolved_args[3].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_try_wait_parity phaseParity, ticks, and mbarId operands must be integer types";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierArriveFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierArriveFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    int64_t mbarId_value = 0;
    bool found_constant = false;

    if (auto const_op = resolved_args[1].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            mbarId_value = int_attr.getInt();
            found_constant = true;
        }
    }

    if (!found_constant) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_try_wait_parity requires a constant integer value for mbarId";
        return nullptr;
    }

    return mlir::nvgpu::MBarrierArriveOp::create(builder, location, resolved_args[0], arith::ConstantIndexOp::create(builder, location, mbarId_value));
}

bool NVVMIntrinsic::CheckMBarrierArriveFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive requires exactly 2 argument: barrier, mbarId";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate barrier operand for mbarrier_arrive";
        return false;
    }

    if (!resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate mbarId operand for mbarrier_arrive";
        return false;
    }

    auto barrier_type = resolved_args[0].getType();
    if (!mlir::isa<nvgpu::MBarrierGroupType>(barrier_type)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive operand must be of type mbarrier_group_t";
        return false;
    }

    auto mbarId_type = resolved_args[1].getType();
    if (!mbarId_type.isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive mbarId operand must be an integer type";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierArriveExpectTxFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierArriveExpectTxFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    int64_t mbarId_value = 0, txcount_value = 0, predicate_value = 0;
    bool found_constant = false;

        if (auto const_op = resolved_args[1].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            txcount_value = int_attr.getInt();
            found_constant = true;
        }
    }

    if (!found_constant) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive_expect_tx requires a constant integer value for txcount";
        return nullptr;
    }

    found_constant = false;
    if (auto const_op = resolved_args[2].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            mbarId_value = int_attr.getInt();
            found_constant = true;
        }
    }

    if (!found_constant) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive_expect_tx requires a constant integer value for mbarId";
        return nullptr;
    }

    found_constant = false;
    if (auto const_op = resolved_args[3].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            predicate_value = int_attr.getInt();
            found_constant = true;
        }
    }

    if (!found_constant) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive_expect_tx requires a constant integer value for predicate";
        return nullptr;
    }

    mlir::nvgpu::MBarrierArriveExpectTxOp::create(builder, location, resolved_args[0], arith::ConstantIndexOp::create(builder, location, txcount_value), arith::ConstantIndexOp::create(builder, location, mbarId_value), LLVM::ConstantOp::create(builder, location, builder.getI1Type(), predicate_value));

    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

bool NVVMIntrinsic::CheckMBarrierArriveExpectTxFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive_expect_tx requires exactly 4 arguments: barrier, txcount, mbarId, predicate";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate barrier operand for mbarrier_arrive_expect_tx";
        return false;
    }

    if (!resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate txcount operand for mbarrier_arrive_expect_tx";
        return false;
    }

    if (!resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate mbarId operand for mbarrier_arrive_expect_tx";
        return false;
    }

    if (!resolved_args[3]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate predicate operand for mbarrier_arrive_expect_tx";
        return false;
    }

    auto barrier_type = resolved_args[0].getType();
    if (!mlir::isa<nvgpu::MBarrierGroupType>(barrier_type)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive_expect_tx operand must be of type mbarrier_group_t";
        return false;
    }

    if (!resolved_args[1].getType().isIntOrIndex() || !resolved_args[2].getType().isIntOrIndex() || !resolved_args[3].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_arrive_expect_tx txcount, mbarId, and predicate operands must be integer types";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateCpAsyncCaSharedGlobalFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckCpAsyncCaSharedGlobalFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto toIndex = [&](mlir::Value v) -> mlir::Value {
        if (v.getType().isIndex()) {
            return v;
        }
        return mlir::arith::IndexCastOp::create(builder, location,
                                                builder.getIndexType(),
                                                v);
    };

    auto dstBase = cf::AveLangMemRefExtractAlignedPointerAsIndexOp::create(builder, location, builder.getIndexType(), resolved_args[0]);
    auto srcBase = cf::AveLangMemRefExtractAlignedPointerAsIndexOp::create(builder, location, builder.getIndexType(), resolved_args[1]);

    auto dstOffset = toIndex(resolved_args[2]);
    auto srcOffset = toIndex(resolved_args[3]);
    auto dstAddr = mlir::arith::AddIOp::create(builder, location, dstBase, dstOffset);
    auto srcAddr = mlir::arith::AddIOp::create(builder, location, srcBase, srcOffset);

    auto dstAddrI64 = mlir::arith::IndexCastOp::create(builder, location, builder.getI64Type(), dstAddr);
    auto srcAddrI64 = mlir::arith::IndexCastOp::create(builder, location, builder.getI64Type(), srcAddr);

    auto dstPtrType = mlir::LLVM::LLVMPointerType::get(
        builder.getContext(),
        static_cast<unsigned>(mlir::NVVM::NVVMMemorySpace::Shared));
    auto srcPtrType = mlir::LLVM::LLVMPointerType::get(
        builder.getContext(),
        static_cast<unsigned>(mlir::NVVM::NVVMMemorySpace::Global));

    auto dstPtr = mlir::LLVM::IntToPtrOp::create(builder, location, dstPtrType,
                                                  dstAddrI64.getResult());
    auto srcPtr = mlir::LLVM::IntToPtrOp::create(builder, location, srcPtrType,
                                                          srcAddrI64.getResult());

    auto sizeConstOp = resolved_args[4].getDefiningOp<mlir::arith::ConstantOp>();
    auto sizeAttr =
        sizeConstOp ? mlir::dyn_cast<mlir::IntegerAttr>(sizeConstOp.getValue())
                    : mlir::IntegerAttr();
    if (!sizeAttr) {
        return nullptr;
    }
    auto sizeBytes = static_cast<int64_t>(sizeAttr.getInt());

    mlir::NVVM::CpAsyncOp::create(builder, location, dstPtr, srcPtr, builder.getI32IntegerAttr(sizeBytes),
        mlir::NVVM::LoadCacheModifierKindAttr::get(
            builder.getContext(), mlir::NVVM::LoadCacheModifierKind::CA),
        mlir::Value());

    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

bool NVVMIntrinsic::CheckCpAsyncCaSharedGlobalFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 5) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global requires 5 arguments: dst, src, dst_offset_bytes, src_offset_bytes, size_bytes";
        return false;
    }
    if (!resolved_args[0] || !resolved_args[1] || !resolved_args[2] ||
        !resolved_args[3] || !resolved_args[4]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "failed to generate one or more operands for cp_async_ca_shared_global";
        return false;
    }

    auto dstType = mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    auto srcType = mlir::dyn_cast<cf::MemRefType>(resolved_args[1].getType());
    if (!dstType || !srcType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global dst/src operands must be memrefs";
        return false;
    }

    auto gpuSpace = mlir::gpu::AddressSpaceAttr::get(
        ctx->GetCurrentFunctionGenerator()->GetBuilder().getContext(),
        mlir::gpu::AddressSpace::Workgroup);
    if (dstType.getMemorySpace() != gpuSpace) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global dst operand must be in workgroup memory";
        return false;
    }

    if (!resolved_args[2].getType().isIntOrIndex() ||
        !resolved_args[3].getType().isIntOrIndex() ||
        !resolved_args[4].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global offset/size operands must be integer/index";
        return false;
    }

    auto sizeConstOp = resolved_args[4].getDefiningOp<mlir::arith::ConstantOp>();
    auto sizeAttr =
        sizeConstOp ? mlir::dyn_cast<mlir::IntegerAttr>(sizeConstOp.getValue())
                    : mlir::IntegerAttr();
    if (!sizeAttr) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global size_bytes must be a compile-time constant";
        return false;
    }

    auto sizeBytes = sizeAttr.getInt();
    if (sizeBytes != 4 && sizeBytes != 8 && sizeBytes != 16) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_ca_shared_global size_bytes must be one of 4, 8, or 16";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateCpAsyncCommitGroupFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckCpAsyncCommitGroupFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    mlir::NVVM::CpAsyncCommitGroupOp::create(builder, location);
    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

bool NVVMIntrinsic::CheckCpAsyncCommitGroupFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (!resolved_args.empty()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_commit_group takes no arguments";
        return false;
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateCpAsyncWaitGroupFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckCpAsyncWaitGroupFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto nConstOp = resolved_args[0].getDefiningOp<mlir::arith::ConstantOp>();
    auto nAttr = nConstOp ? mlir::dyn_cast<mlir::IntegerAttr>(nConstOp.getValue())
                          : mlir::IntegerAttr();
    if (!nAttr) {
        return nullptr;
    }
    mlir::NVVM::CpAsyncWaitGroupOp::create(builder, location, builder.getI32IntegerAttr(static_cast<int32_t>(nAttr.getInt())));

    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

bool NVVMIntrinsic::CheckCpAsyncWaitGroupFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 1 || !resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_wait_group requires 1 argument: n";
        return false;
    }
    if (!resolved_args[0].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_wait_group n operand must be integer/index";
        return false;
    }
    auto nConstOp = resolved_args[0].getDefiningOp<mlir::arith::ConstantOp>();
    auto nAttr = nConstOp ? mlir::dyn_cast<mlir::IntegerAttr>(nConstOp.getValue())
                          : mlir::IntegerAttr();
    if (!nAttr) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_wait_group n must be a compile-time constant";
        return false;
    }
    auto n = nAttr.getInt();
    if (n < 0 || n > 8) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "cp_async_wait_group n must be in [0, 8]";
        return false;
    }
    return true;
}

mlir::Value NVVMIntrinsic::CreateMBarrierTestWaitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMBarrierTestWaitFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    int64_t mbarId_value = 0;
    bool found_constant = false;

    if (auto const_op = resolved_args[2].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            mbarId_value = int_attr.getInt();
            found_constant = true;
        }
    }

    if (!found_constant) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_test_wait requires a constant integer value for mbarId";
        return nullptr;
    }

    return mlir::nvgpu::MBarrierTestWaitOp::create(builder, location, resolved_args[0], resolved_args[1], mlir::arith::ConstantIndexOp::create(builder, location, mbarId_value));
}

bool NVVMIntrinsic::CheckMBarrierTestWaitFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_test_wait requires exactly 2 arguments: barrier, token, mbarId";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate barrier operand for mbarrier_test_wait";
        return false;
    }

    if (!resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate token operand for mbarrier_test_wait";
        return false;
    }

    if (!resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate mbarId operand for mbarrier_test_wait";
        return false;
    }

    auto barrier_type = resolved_args[0].getType();
    if (!mlir::isa<nvgpu::MBarrierGroupType>(barrier_type)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_test_wait operand must be of type mbarrier_group_t";
        return false;
    }

    // Token must be a nvgpu.mbarrier.token, the result of mbarrier_arrive
    auto token_type = resolved_args[1].getType();
    if (!mlir::isa<nvgpu::MBarrierTokenType>(token_type)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_test_wait token operand must be of type mbarrier_token_t";
        return false;
    }

    if (!resolved_args[2].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "mbarrier_test_wait mbarId operand must be an integer type";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateMakeWGMMADescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckMakeWGMMADescriptorFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto substrate_memref_type =
        mlir::dyn_cast<cf::MemRefType>(resolved_args[0].getType());
    auto workgroupMemorySpace = mlir::IntegerAttr::get(
        mlir::IntegerType::get(builder.getContext(), 64), 3);
    mlir::MemRefType memref_type = mlir::MemRefType::getChecked(
        [&]() { return mlir::emitError(location); },
        substrate_memref_type.getShape(), substrate_memref_type.getElementType(),
        mlir::MemRefLayoutAttrInterface(), workgroupMemorySpace);
    if (!memref_type) {
        return nullptr;
    }

    auto result_type = mlir::nvgpu::WarpgroupMatrixDescriptorType::get(builder.getContext(), memref_type);
    auto descriptor = cf::NVVMWGMMADescriptorOp::create(
        builder, location, result_type, resolved_args[0],
       mlir::IntegerAttr::get(builder.getI32Type(), mlir::dyn_cast<mlir::IntegerAttr>(mlir::dyn_cast<mlir::arith::ConstantOp>(resolved_args[1].getDefiningOp()).getValue()).getInt()),
        mlir::IntegerAttr::get(builder.getI32Type(), mlir::dyn_cast<mlir::IntegerAttr>(mlir::dyn_cast<mlir::arith::ConstantOp>(resolved_args[2].getDefiningOp()).getValue()).getInt()),
        mlir::IntegerAttr::get(builder.getI32Type(), mlir::dyn_cast<mlir::IntegerAttr>(mlir::dyn_cast<mlir::arith::ConstantOp>(resolved_args[3].getDefiningOp()).getValue()).getInt()),
        mlir::IntegerAttr::get(builder.getI32Type(), mlir::dyn_cast<mlir::IntegerAttr>(mlir::dyn_cast<mlir::arith::ConstantOp>(resolved_args[4].getDefiningOp()).getValue()).getInt()));
    return descriptor.getResult();
}

bool NVVMIntrinsic::CheckMakeWGMMADescriptorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 5) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor requires exactly 5 arguments: tensor, swizzle_kind, l2promo_kind, oob_kind, interleave_kind";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate tensor operand for make_wgmma_descriptor";
        return false;
    }

    if (!resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate swizzle_kind operand for make_wgmma_descriptor";
        return false;
    }

    if (!resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate l2promo_kind operand for make_wgmma_descriptor";
        return false;
    }

    if (!resolved_args[3]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate oob_kind operand for make_wgmma_descriptor";
        return false;
    }

    if (!resolved_args[4]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate interleave_kind operand for make_wgmma_descriptor";
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor expects memref pointer operand for tensor";
        return false;
    }

    // Check swizzle_kind
    // Tensor map swizzling mode of shared memory banks
    // enum class TensorMapSwizzleKind : uint32_t {
    // SWIZZLE_NONE = 0,
    // SWIZZLE_32B = 1,
    // SWIZZLE_64B = 2,
    // SWIZZLE_128B = 3,
    // };
    auto swizzle_kind_type = resolved_args[1].getType();
    auto swizzle_kind_value = 0;
    bool found_constant = false;
    if (!swizzle_kind_type.isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor swizzle_kind operand must be an integer type";
        return false;
    }
    if (auto const_op = resolved_args[1].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            swizzle_kind_value = int_attr.getInt();
            found_constant = true;
        }
    }
    if (!found_constant) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor requires a constant integer value for swizzle_kind";
        return false;
    }
    if (swizzle_kind_value < 0 || swizzle_kind_value > 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor swizzle_kind operand has invalid value " << swizzle_kind_value << ", must be 0, 1, 2, or 3";
        return false;
    }

    // Check l2promo_kind
    // Tensor map L2 promotion type
    // enum class TensorMapL2PromoKind : uint32_t {
    // L2PROMO_NONE = 0,
    // L2PROMO_64B = 1,
    // L2PROMO_128B = 2,
    // L2PROMO_256B = 3,
    // };
    auto l2promo_kind_type = resolved_args[2].getType();
    auto l2promo_kind_value = 0;
    found_constant = false;
    if (!l2promo_kind_type.isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor l2promo_kind operand must be an integer type";
        return false;
    }
    if (auto const_op = resolved_args[2].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            l2promo_kind_value = int_attr.getInt();
            found_constant = true;
        }
    }
    if (!found_constant) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor requires a constant integer value for l2promo_kind";
        return false;
    }
    if (l2promo_kind_value < 0 || l2promo_kind_value > 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor l2promo_kind operand has invalid value " << l2promo_kind_value << ", must be 0, 1, 2, or 3";
        return false;
    }

    // Check oob_kind
    // Tensor map out-of-bounds fill type
    // enum class TensorMapOOBKind : uint32_t {
    // OOB_ZERO = 0,
    // OOB_NAN = 1,
    // };
    auto oob_kind_type = resolved_args[3].getType();
    auto oob_kind_value = 0;
    found_constant = false;
    if (!oob_kind_type.isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor oob_kind operand must be an integer type";
        return false;
    }
    if (auto const_op = resolved_args[3].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            oob_kind_value = int_attr.getInt();
            found_constant = true;
        }
    }
    if (!found_constant) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor requires a constant integer value for oob_kind";
        return false;
    }
    if (oob_kind_value < 0 || oob_kind_value > 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor oob_kind operand has invalid value " << oob_kind_value << ", must be 0 or 1";
        return false;
    }

    // Check interleave_kind
    // Tensor map interleave layout type
    // enum class TensorMapInterleaveKind : uint32_t {
    // INTERLEAVE_NONE = 0,
    // INTERLEAVE_16B = 1,
    // INTERLEAVE_32B = 2,
    // };
    auto interleave_kind_type = resolved_args[4].getType();
    auto interleave_kind_value = 0;
    found_constant = false;
    if (!interleave_kind_type.isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor interleave_kind operand must be an integer type";
        return false;
    }
    if (auto const_op = resolved_args[4].getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            interleave_kind_value = int_attr.getInt();
            found_constant = true;
        }
    }
    if (!found_constant) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor requires a constant integer value for interleave_kind";
        return false;
    }
    if (interleave_kind_value < 0 || interleave_kind_value > 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "make_wgmma_descriptor interleave_kind operand has invalid value " << interleave_kind_value << ", must be 0, 1, or 2";
        return false;
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

    auto tensorType = getBuiltinTensorMapMemRefType(resolved_args[0].getType());
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

    auto tensorType = getBuiltinTensorMapMemRefType(resolved_args[0].getType());
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

mlir::Value NVVMIntrinsic::CreateTMAFenceFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckTMAFenceFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    cf::NVVMTMAFenceOp::create(builder, location, mlir::ValueRange{resolved_args[0]}, mlir::ArrayRef<mlir::NamedAttribute>{});
    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

bool NVVMIntrinsic::CheckTMAFenceFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 1) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_fence requires exactly 1 argument: desc";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate desc operand for tma_fence";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::TensorMapDescriptorType>(
            resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_fence desc operand must be a TMA descriptor";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateTMALoadFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckTMALoadFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    const auto &keywords = call_expr->GetKeywords();
    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    mlir::Value mbarId =
        positionalCount > 4
            ? resolved_args[4]
            : mlir::arith::ConstantIndexOp::create(builder, location, 0).getResult();
    mlir::Value predicate =
        positionalCount > 5
            ? resolved_args[5]
            : mlir::arith::ConstantIntOp::create(builder, location, 1, 1).getResult();
    mlir::Value multicastMask =
        positionalCount > 6
            ? resolved_args[6]
            : mlir::arith::ConstantIntOp::create(builder, location, -1, 32).getResult();

    for (size_t i = 0; i < keywordCount; ++i) {
        mlir::Value value = resolved_args[positionalCount + i];
        llvm::StringRef name = keywords[i];
        if (name == "mbar_id") {
            mbarId = value;
        } else if (name == "predicate") {
            predicate = value;
        } else if (name == "multicast_mask") {
            if (value) {
                multicastMask = value;
            }
        }
    }

    cf::NVVMTMALoadOp::create(builder, location, mlir::ValueRange{resolved_args[0], resolved_args[1], resolved_args[2],
                         resolved_args[3], mbarId, predicate, multicastMask},
        mlir::ArrayRef<mlir::NamedAttribute>{});
    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

bool NVVMIntrinsic::CheckTMALoadFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    const auto &keywords = call_expr->GetKeywords();
    if (resolved_args.size() < keywords.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load internal argument mismatch";
        return false;
    }

    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;
    if (positionalCount < 4 || positionalCount > 7) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load requires dst, desc, coords, barrier and optional "
               "mbar_id, predicate, multicast_mask";
        return false;
    }

    for (auto name : keywords) {
        if (name != "mbar_id" && name != "predicate" &&
            name != "multicast_mask") {
            ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                            call_expr->GetSourceRange().getBegin())
                << "tma_load got unsupported keyword argument '" << name << "'";
            return false;
        }
    }

    auto requireArg = [&](size_t index, llvm::StringRef name) -> bool {
        if (index >= resolved_args.size() || !resolved_args[index]) {
            ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                            call_expr->GetSourceRange().getBegin())
                << "Failed to generate " << name << " operand for tma_load";
            return false;
        }
        return true;
    };

    if (!requireArg(0, "dst") || !requireArg(1, "desc") ||
        !requireArg(2, "coords") || !requireArg(3, "barrier")) {
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load dst operand must be a memref";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::TensorMapDescriptorType>(
            resolved_args[1].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load desc operand must be a TMA descriptor";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::MBarrierGroupType>(
            resolved_args[3].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_load barrier operand must be of type mbarrier_group_t";
        return false;
    }

    auto checkIntArg = [&](mlir::Value value, llvm::StringRef name) -> bool {
        if (value && !value.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                            call_expr->GetSourceRange().getBegin())
                << "tma_load " << name << " operand must be an integer type";
            return false;
        }
        return true;
    };

    for (size_t i = 4; i < positionalCount; ++i) {
        if (!checkIntArg(resolved_args[i], i == 4 ? "mbar_id" :
                                           i == 5 ? "predicate" :
                                                    "multicast_mask")) {
            return false;
        }
    }
    for (size_t i = 0; i < keywordCount; ++i) {
        auto name = llvm::StringRef(keywords[i]);
        auto value = resolved_args[positionalCount + i];
        if (name == "mbar_id" || name == "predicate" ||
            (name == "multicast_mask" && value)) {
            if (!checkIntArg(value, name)) {
                return false;
            }
        }
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateTMAStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckTMAStoreFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    const auto &keywords = call_expr->GetKeywords();
    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;

    mlir::Value predicate =
        positionalCount > 3
            ? resolved_args[3]
            : mlir::arith::ConstantIntOp::create(builder, location, 1, 1).getResult();

    for (size_t i = 0; i < keywordCount; ++i) {
        mlir::Value value = resolved_args[positionalCount + i];
        llvm::StringRef name = keywords[i];
        if (name == "predicate") {
            predicate = value;
        }
    }

    cf::NVVMTMAStoreOp::create(builder, location, mlir::ValueRange{resolved_args[0], resolved_args[1], resolved_args[2], predicate}, mlir::ArrayRef<mlir::NamedAttribute>{});
    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

bool NVVMIntrinsic::CheckTMAStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    const auto &keywords = call_expr->GetKeywords();
    if (resolved_args.size() < keywords.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_store internal argument mismatch";
        return false;
    }

    size_t keywordCount = keywords.size();
    size_t positionalCount = resolved_args.size() - keywordCount;
    if (positionalCount < 3 || positionalCount > 4) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_store requires src, desc, coords and optional predicate";
        return false;
    }

    for (auto name : keywords) {
        if (name != "predicate") {
            ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                            call_expr->GetSourceRange().getBegin())
                << "tma_store got unsupported keyword argument '" << name << "'";
            return false;
        }
    }

    auto requireArg = [&](size_t index, llvm::StringRef name) -> bool {
        if (index >= resolved_args.size() || !resolved_args[index]) {
            ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                            call_expr->GetSourceRange().getBegin())
                << "Failed to generate " << name << " operand for tma_store";
            return false;
        }
        return true;
    };

    if (!requireArg(0, "src") || !requireArg(1, "desc") ||
        !requireArg(2, "coords")) {
        return false;
    }

    if (!mlir::isa<cf::MemRefType>(resolved_args[0].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_store src operand must be a memref";
        return false;
    }

    if (!mlir::isa<mlir::nvgpu::TensorMapDescriptorType>(
            resolved_args[1].getType())) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "tma_store desc operand must be a TMA descriptor";
        return false;
    }

    auto checkIntArg = [&](mlir::Value value, llvm::StringRef name) -> bool {
        if (value && !value.getType().isIntOrIndex()) {
            ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                            call_expr->GetSourceRange().getBegin())
                << "tma_store " << name << " operand must be an integer type";
            return false;
        }
        return true;
    };

    if (positionalCount > 3 &&
        !checkIntArg(resolved_args[3], "predicate")) {
        return false;
    }
    for (size_t i = 0; i < keywordCount; ++i) {
        if (!checkIntArg(resolved_args[positionalCount + i], keywords[i])) {
            return false;
        }
    }

    return true;
}


mlir::Value NVVMIntrinsic::CreateWgmmaAsyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckWgmmaAsyncFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto async = cf::NVVMWGMMAAsyncOp::create(builder, location, resolved_args[2].getType(), resolved_args[0], resolved_args[1], resolved_args[2]);
    return async.getResult();
}

bool NVVMIntrinsic::CheckWgmmaAsyncFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    // wgmma_async(desc_a, desc_b, acc)
    if (resolved_args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_async requires exactly 3 arguments: desc_a, desc_b, acc";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate desc_a operand for wgmma_async";
        return false;
    }

    if (!resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate desc_b operand for wgmma_async";
        return false;
    }

    if (!resolved_args[2]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate acc operand for wgmma_async";
        return false;
    }

    // desc_a and desc_b must be wgmma_descriptors
    auto desc_a_type = resolved_args[0].getType();
    if (!mlir::isa<mlir::nvgpu::WarpgroupMatrixDescriptorType>(desc_a_type)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_async desc_a operand must be of type wgmma_descriptor";
        return false;
    }
    auto desc_b_type = resolved_args[1].getType();
    if (!mlir::isa<mlir::nvgpu::WarpgroupMatrixDescriptorType>(desc_b_type)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_async desc_b operand must be of type wgmma_descriptor";
        return false;
    }

    // acc must be a warpgroup accumulator
    auto acc_type = resolved_args[2].getType();
    if (!mlir::isa<mlir::nvgpu::WarpgroupAccumulatorType>(acc_type)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_async acc operand must be of type warpgroup_accumulator";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaInitAccumulatorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckWgmmaInitAccumulatorFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    auto m_size = mlir::cast<mlir::IntegerAttr>(mlir::dyn_cast<mlir::arith::ConstantOp>(resolved_args[0].getDefiningOp()).getValue()).getInt();
    auto n_size = mlir::cast<mlir::IntegerAttr>(mlir::dyn_cast<mlir::arith::ConstantOp>(resolved_args[1].getDefiningOp()).getValue()).getInt();

    auto f32_ty = builder.getF32Type();
    auto vec_ty = mlir::VectorType::get({m_size, n_size}, f32_ty);
    auto acc_ty = mlir::nvgpu::WarpgroupAccumulatorType::get(builder.getContext(),
                                                             vec_ty);

    auto acc =
        mlir::nvgpu::WarpgroupMmaInitAccumulatorOp::create(builder, location, acc_ty);
    return acc.getResult();
}

bool NVVMIntrinsic::CheckWgmmaInitAccumulatorFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_init_accumulator requires exactly 2 arguments: m, n";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate m operand for wgmma_init_accumulator";
        return false;
    }

    if (!resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate n operand for wgmma_init_accumulator";
        return false;
    }

    if (!resolved_args[0].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_init_accumulator m operand must be an integer type";
        return false;
    }

    if (!resolved_args[1].getType().isIntOrIndex()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_init_accumulator n operand must be an integer type";
        return false;
    }

    return true;
}

mlir::Value NVVMIntrinsic::CreateWgmmaStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();
    auto location = builder.getUnknownLoc();

    if (!CheckWgmmaStoreFunction(call_expr, ctx, resolved_args)) {
        return nullptr;
    }

    cf::NVVMWGMMAStoreOp::create(builder, location, resolved_args[0], resolved_args[1]);
    return ctx->GetCurrentFunctionGenerator()->GetExprGenerator()->CreateVoidValue();
}

bool NVVMIntrinsic::CheckWgmmaStoreFunction(
    ast::Call *call_expr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolved_args) const {
    if (resolved_args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_store requires exactly 2 arguments: acc, desc";
        return false;
    }

    if (!resolved_args[0]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate acc operand for wgmma_store";
        return false;
    }

    if (!resolved_args[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "Failed to generate desc operand for wgmma_store";
        return false;
    }

    auto acc_type = resolved_args[0].getType();
    if (!mlir::isa<mlir::nvgpu::WarpgroupAccumulatorType>(acc_type)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_store acc operand must be of type warpgroup_accumulator";
        return false;
    }

    auto desc_type = resolved_args[1].getType();
    if (!mlir::isa<cf::MemRefType>(desc_type)) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        call_expr->GetSourceRange().getBegin())
            << "wgmma_store desc operand must be of memref type";
        return false;
    }

    return true;
}

// Factory function to create NVVM intrinsic module
std::unique_ptr<NamedModule> CreateNVVMIntrinsicModule() {
    return std::make_unique<NVVMIntrinsic>();
}

} // namespace causalflow::avelang::ir
