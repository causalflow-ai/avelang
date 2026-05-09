#include "IR/mlir_generator.h"
#include "gpu_passes.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#pragma clang diagnostic pop

#include <llvm/ADT/SmallVector.h>

namespace causalflow::avelang::target::gpu {

class GpuOutliningPass
    : public mlir::PassWrapper<GpuOutliningPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuOutliningPass)

    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();
        mlir::MLIRContext *context = &getContext();
        mlir::OpBuilder builder(context);

        // Collect functions with ave.gpu_func == kPrivateFunction or
        // kGlobalKernel
        llvm::SmallVector<mlir::func::FuncOp> gpuFunctions;

        module.walk([&](mlir::func::FuncOp funcOp) {
            auto gpuFuncAttr =
                funcOp->getAttrOfType<mlir::IntegerAttr>("ave.gpu_func");
            if (gpuFuncAttr) {
                gpuFunctions.push_back(funcOp);
            }
        });

        if (gpuFunctions.empty()) {
            return;
        }

        // Create GPU module
        builder.setInsertionPointToEnd(module.getBody());
        auto gpuModule = mlir::gpu::GPUModuleOp::create(
            builder, builder.getUnknownLoc(), "ave_gpu_module");

        // Set insertion point to the GPU module body
        auto &gpuModuleBody = gpuModule.getBodyRegion().front();
        builder.setInsertionPointToStart(&gpuModuleBody);

        // Collect all functions referenced by GPU functions
        llvm::SmallVector<mlir::func::FuncOp> referencedFunctions;
        llvm::DenseSet<llvm::StringRef> referencedNames;

        for (auto funcOp : gpuFunctions) {
            funcOp.walk([&](mlir::func::CallOp callOp) {
                auto callee = callOp.getCallee();
                if (referencedNames.insert(callee).second) {
                    if (auto referencedFunc =
                            module.lookupSymbol<mlir::func::FuncOp>(callee)) {
                        referencedFunctions.push_back(referencedFunc);
                    }
                }
            });
        }

        // Copy referenced functions (like intrinsics) into the GPU module
        for (auto referencedFunc : referencedFunctions) {
            // Check if this is a GPU function (has ave.gpu_func) or an
            // intrinsic
            auto gpuFuncAttr = referencedFunc->getAttrOfType<mlir::IntegerAttr>(
                "ave.gpu_func");
            bool isIntrinsic =
                !gpuFuncAttr && llvm::isa<mlir::func::FuncOp>(referencedFunc);

            if (isIntrinsic) {
                auto clonedFunc = referencedFunc.clone();
                builder.insert(clonedFunc);

                // Intrinsics are cloned into the GPU module only; they are
                // not part of the gpuFunctions list.
            }
        }

        // Move and convert functions to gpu.func (for kernels) or func.func
        // (for JIT functions)
        llvm::SmallVector<mlir::func::FuncOp> kernelsToErase;

        for (auto funcOp : gpuFunctions) {
            auto gpuFuncAttr =
                funcOp->getAttrOfType<mlir::IntegerAttr>("ave.gpu_func");
            bool isKernel =
                gpuFuncAttr.getInt() ==
                static_cast<int>(causalflow::avelang::ir::MLIRGenerator::
                                     FunctionType::kGlobalKernel);

            if (isKernel) {
                // Kernels are converted to gpu.func
                auto gpuFunc = mlir::gpu::GPUFuncOp::create(
                    builder, funcOp.getLoc(), funcOp.getName(),
                    funcOp.getFunctionType(),
                    mlir::TypeRange{}, // workgroup attributions
                    mlir::TypeRange{}  // private attributions
                );
                gpuFunc->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                                 builder.getUnitAttr());

                // Copy function body
                mlir::IRMapping mapping;
                auto &gpuBody = gpuFunc.getBody();
                for (auto &block : gpuBody.getBlocks()) {
                    block.clear();
                }
                gpuBody.getBlocks().clear();
                funcOp.getBody().cloneInto(&gpuBody, mapping);

                auto isWorkgroupMemref = [](mlir::MemRefType memrefType) {
                    if (auto addressSpace = memrefType.getMemorySpace()) {
                        if (auto addrSpaceAttr =
                                mlir::dyn_cast<mlir::gpu::AddressSpaceAttr>(
                                    addressSpace)) {
                            return addrSpaceAttr.getValue() ==
                                   mlir::gpu::AddressSpace::Workgroup;
                        }
                        if (auto intSpace =
                                mlir::dyn_cast<mlir::IntegerAttr>(addressSpace)) {
                            return intSpace.getInt() == 3;
                        }
                    }
                    return false;
                };

                llvm::SmallVector<mlir::memref::AllocaOp> workgroupAllocaOps;
                gpuFunc.walk([&](mlir::memref::AllocaOp allocaOp) {
                    if (isWorkgroupMemref(allocaOp.getType())) {
                        workgroupAllocaOps.push_back(allocaOp);
                    }
                });

                for (auto allocaOp : workgroupAllocaOps) {
                    auto memrefType = allocaOp.getType();
                    unsigned attributionIndex =
                        gpuFunc.getNumWorkgroupAttributions();
                    auto workgroupArg = gpuFunc.addWorkgroupAttribution(
                        memrefType, allocaOp.getLoc());
                    gpuFunc.setWorkgroupAttributionAttr(
                        attributionIndex,
                        mlir::LLVM::LLVMDialect::getAlignAttrName(),
                        builder.getI64IntegerAttr(128));
                    allocaOp.getResult().replaceAllUsesWith(workgroupArg);
                    allocaOp.erase();
                }

                // Convert func.return to gpu.return
                gpuFunc.walk([&](mlir::func::ReturnOp returnOp) {
                    mlir::OpBuilder::InsertionGuard guard(builder);
                    builder.setInsertionPoint(returnOp);
                    mlir::gpu::ReturnOp::create(builder, returnOp.getLoc(),
                                                returnOp.getOperands());
                    returnOp.erase();
                });

                // Mark kernel for erasure
                kernelsToErase.push_back(funcOp);
            } else {
                // Device functions remain as func.func but are moved to GPU
                // module Move the function operation to the GPU module
                funcOp->moveBefore(&gpuModuleBody, gpuModuleBody.end());
                // Remove the ave.gpu_func attribute
                funcOp->removeAttr("ave.gpu_func");
            }
        }

        // Remove the original kernel functions
        for (auto funcOp : kernelsToErase) {
            funcOp.erase();
        }
    }

    llvm::StringRef getArgument() const final { return "gpu-outlining"; }

    llvm::StringRef getDescription() const final {
        return "Outline functions with ave.gpu_func==kPrivateFunction or "
               "kGlobalKernel to GPU module";
    }
};

std::unique_ptr<mlir::Pass> createGpuOutliningPass() {
    return std::make_unique<GpuOutliningPass>();
}

} // namespace causalflow::avelang::target::gpu
