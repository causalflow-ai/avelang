#pragma once

#include <functional>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>
#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <string>
#include <vector>

namespace llvm {
class Module;
}

namespace causalflow::avelang::target::gpu {

struct GPUCompilationOptions;

class GPUBackendInterface {
  public:
    virtual ~GPUBackendInterface() = default;

    virtual void
    buildLoweringPipeline(mlir::OpPassManager &pm,
                          const GPUCompilationOptions &options) = 0;

    virtual bool supportsTriple(llvm::StringRef triple) const = 0;

    virtual std::string getName() const = 0;

    virtual llvm::Expected<std::string>
    generateBinary(llvm::Module &module,
                   const GPUCompilationOptions &options) = 0;

    virtual llvm::Expected<std::string>
    generateAssembly(llvm::Module &module,
                     const GPUCompilationOptions &options) = 0;

    virtual llvm::json::Object getKernelMetadata(mlir::ModuleOp module) const {
        return {};
    }

    virtual void EnsureInitialized() {}
};

using BackendFactory = std::function<std::unique_ptr<GPUBackendInterface>()>;

class GPUBackendRegistry {
  public:
    static GPUBackendRegistry &getInstance();

    void registerBackend(BackendFactory factory);

    std::unique_ptr<GPUBackendInterface>
    createBackendForTriple(llvm::StringRef triple);

  private:
    GPUBackendRegistry();

    std::vector<BackendFactory> backends_;
};

} // namespace causalflow::avelang::target::gpu
