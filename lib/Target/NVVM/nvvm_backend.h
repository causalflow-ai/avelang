#pragma once

#include "Target/GPU/gpu_backend.h"
#include <memory>

namespace causalflow::avelang::target::nvvm {

class NVVMBackend
    : public causalflow::avelang::target::gpu::GPUBackendInterface {
  public:
    void buildLoweringPipeline(
        mlir::OpPassManager &pm,
        const causalflow::avelang::target::gpu::GPUCompilationOptions &options)
        override;

    bool supportsTriple(llvm::StringRef triple) const override;

    std::string getName() const override;

    llvm::Expected<std::string>
    generateBinary(llvm::Module &module,
                   const causalflow::avelang::target::gpu::GPUCompilationOptions
                       &options) override;

    llvm::Expected<std::string> generateAssembly(
        llvm::Module &module,
        const causalflow::avelang::target::gpu::GPUCompilationOptions &options)
        override;

    llvm::json::Object getKernelMetadata(mlir::ModuleOp module) const override;

    void EnsureInitialized() override;
};

std::unique_ptr<causalflow::avelang::target::gpu::GPUBackendInterface>
CreateNVVMBackend();

} // namespace causalflow::avelang::target::nvvm
