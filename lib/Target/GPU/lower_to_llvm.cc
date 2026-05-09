#include "lower_to_llvm.h"
#include "Dialect/AveLang/IR/AveLangDialect.h"
#include "Dialect/AveLang/Transforms/allocation_op_interface_impl.h"
#include "Dialect/AveLang/Transforms/hoist_alloca_pass.h"
#include "Dialect/AveLang/Transforms/lower_ave_lang_to_memref_pass.h"
#include "Dialect/AveLang/Transforms/lower_gpuop_to_intrinsics_pass.h"
#include "IR/builtin_module.h"
#include "IR/ir_context.h"
#include "avelang/config.h"
#include "gpu_backend.h"
#include "gpu_passes.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/UBToLLVM/UBToLLVM.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Transforms/VectorTransforms.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

namespace mlir::func {
class FuncDialect;
}

namespace causalflow::avelang::target::gpu {

using namespace mlir;

class LowerToLLVM::Impl {
  public:
    explicit Impl(causalflow::avelang::ir::IRContext *ir_context)
        : ir_context_(ir_context) {}

    std::unique_ptr<::llvm::Module>
    compile(mlir::ModuleOp module, ::llvm::LLVMContext &llvmContext,
            const GPUCompilationOptions &options) {
        // Set up target machine to get correct data layout first
        std::string targetTriple = options.triple.str();
        llvm::Triple triple(targetTriple);

        std::string error;
        const llvm::Target *target =
            llvm::TargetRegistry::lookupTarget(triple, error);
        if (!target) {
            llvm::errs() << "Failed to lookup target for triple "
                         << targetTriple << ": " << error << "\n";
            return nullptr;
        }

        llvm::TargetOptions targetOptions;
        auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
            target->createTargetMachine(triple, options.chipset.str(), "",
                                        targetOptions, llvm::Reloc::PIC_));
        if (!targetMachine) {
            llvm::errs() << "Failed to create target machine for triple "
                         << targetTriple << "\n";
            return nullptr;
        }

        // Set the data layout on the MLIR module
        std::string dataLayoutStr;

        auto dataLayout = targetMachine->createDataLayout();
        dataLayoutStr = dataLayout.getStringRepresentation();

        module->setAttr(
            "llvm.data_layout",
            ::mlir::StringAttr::get(module.getContext(), dataLayoutStr));
        module->setAttr(
            "llvm.target_triple",
            ::mlir::StringAttr::get(module.getContext(), targetTriple));

        {
            ::mlir::DialectRegistry registry;
            ::mlir::func::registerAllExtensions(registry);
            ::mlir::arith::registerConvertArithToLLVMInterface(registry);
            ::mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
            ::mlir::registerConvertComplexToLLVMInterface(registry);
            ::mlir::registerConvertFuncToLLVMInterface(registry);
            ::mlir::index::registerConvertIndexToLLVMInterface(registry);
            ::mlir::LLVM::registerInlinerInterface(registry);
            ::mlir::NVVM::registerInlinerInterface(registry);
            ::mlir::registerConvertMathToLLVMInterface(registry);
            ::mlir::registerConvertMemRefToLLVMInterface(registry);
            ::mlir::registerConvertNVVMToLLVMInterface(registry);
            ::mlir::ub::registerConvertUBToLLVMInterface(registry);
            ::mlir::vector::registerConvertVectorToLLVMInterface(registry);
            causalflow::avelang::dialect::
                registerAllocationOpInterfaceExternalModels(registry);
            ::mlir::memref::registerAllocationOpInterfaceExternalModels(
                registry);
            auto *context = ir_context_->GetMLIRContext();
            context->appendDialectRegistry(registry);
            context->loadDialect<::mlir::affine::AffineDialect>();
            context->loadDialect<::mlir::func::FuncDialect>();
            context->loadDialect<::mlir::vector::VectorDialect>();
            context->loadDialect<::mlir::gpu::GPUDialect>();
            context->loadDialect<::mlir::arith::ArithDialect>();
            context->loadDialect<::mlir::math::MathDialect>();
            context->loadDialect<::mlir::bufferization::BufferizationDialect>();
            context->loadDialect<::mlir::scf::SCFDialect>();
            context->loadDialect<::mlir::ub::UBDialect>();
            context->loadDialect<::mlir::LLVM::LLVMDialect>();
            context->loadDialect<::mlir::memref::MemRefDialect>();
            context
                ->loadDialect<causalflow::avelang::dialect::AveLangDialect>();
        }

        // Declare intrinsic modules to make intrinsic functions available
        // This is crucial for the GPUOp lowering pass to find intrinsic
        // functions
        if (targetTriple.find("nvptx") != std::string::npos) {
            // NVVM target
            auto nvvmModule =
                causalflow::avelang::ir::CreateNVVMIntrinsicModule();
            nvvmModule->DeclareModules(module);
        } else if (targetTriple.find("amdgcn") != std::string::npos) {
            // AMDGPU target
            auto amdgpuModule =
                causalflow::avelang::ir::CreateAMDGPUIntrinsicModule();
            amdgpuModule->DeclareModules(module);
        }

        PassManager pm(ir_context_->GetMLIRContext());
        pm.enableVerifier(false);

        pm.addPass(causalflow::avelang::dialect::
                       createLowerAveLangGPUToIntrinsicsPass());

        pm.addPass(::mlir::createInlinerPass());
        pm.addPass(::mlir::createCanonicalizerPass());
        pm.addPass(::mlir::createCSEPass());
        pm.addNestedPass<mlir::func::FuncOp>(
            causalflow::avelang::dialect::createHoistAllocaPass());
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::bufferization::createBufferHoistingPass());
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::bufferization::createBufferLoopHoistingPass());

        // Lower ave memref types and dialect ops to memref dialect
        pm.addPass(
            causalflow::avelang::dialect::createLowerAveLangToMemRefPass());
        pm.addNestedPass<mlir::func::FuncOp>(
            causalflow::avelang::dialect::createHoistAllocaPass());
        pm.addPass(createLinkIntrinsicImplementationPass());
        pm.addPass(::mlir::createCanonicalizerPass());
        pm.addPass(::mlir::createCSEPass());

        // Add GPU outlining pass first to move kGlobalKernel functions to GPU
        // modules
        pm.addPass(createGpuOutliningPass());
        pm.addNestedPass<mlir::gpu::GPUModuleOp>(
            ::mlir::createCanonicalizerPass());
        pm.addNestedPass<mlir::gpu::GPUModuleOp>(::mlir::createCSEPass());
        pm.addPass(::mlir::createSymbolDCEPass());

        // Propagate data layout to GPU modules
        auto propagateDataLayoutPass = [&](::mlir::Operation *op) {
            if (auto gpuModule = dyn_cast<::mlir::gpu::GPUModuleOp>(op)) {
                gpuModule->setAttr("llvm.data_layout",
                                   ::mlir::StringAttr::get(module.getContext(),
                                                           dataLayoutStr));
                gpuModule->setAttr(
                    "llvm.target_triple",
                    ::mlir::StringAttr::get(module.getContext(), targetTriple));
            }
        };
        module.walk(propagateDataLayoutPass);

        // Use backend registry to find appropriate backend for triple
        auto &registry = GPUBackendRegistry::getInstance();
        auto backend = registry.createBackendForTriple(options.triple);
        if (!backend) {
            llvm::errs() << "No GPU backend registered for triple "
                         << targetTriple << "\n";
            return nullptr;
        }

        backend->buildLoweringPipeline(pm, options);

        pm.addPass(::mlir::createFinalizeMemRefToLLVMConversionPass());

        pm.addPass(::mlir::createCanonicalizerPass());
        pm.addPass(::mlir::createCSEPass());

        pm.addPass(::mlir::createReconcileUnrealizedCastsPass());

        pm.addPass(::mlir::createCanonicalizerPass());
        pm.addPass(::mlir::createCSEPass());

        if (failed(pm.run(module))) {
            llvm::errs() << "Pass manager failed for triple " << targetTriple
                         << "\n";
            return nullptr;
        }

        SmallVector<::mlir::gpu::GPUModuleOp> gpuModules;
        module.walk([&](::mlir::gpu::GPUModuleOp gpuModule) {
            gpuModules.push_back(gpuModule);
        });

        if (gpuModules.size() != 1) {
            llvm::errs() << "Expected exactly one GPU module after lowering, "
                         << "found " << gpuModules.size() << "\n";
            return nullptr;
        }

        // NVGPU TMA descriptor lowering materializes this helper as a top-level
        // llvm.func on the builtin.module, but we translate the nested
        // gpu.module in isolation. Ensure the declaration exists in the
        // translated symbol scope.
        constexpr llvm::StringLiteral kTensorMapEncodeFn =
            "mgpuTensorMapEncodeTiledMemref";
        if (auto topLevelFn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(
                kTensorMapEncodeFn)) {
            auto gpuModule = gpuModules[0];
            if (!gpuModule.lookupSymbol<mlir::LLVM::LLVMFuncOp>(
                    kTensorMapEncodeFn)) {
                mlir::OpBuilder builder(gpuModule.getContext());
                builder.setInsertionPointToEnd(gpuModule.getBody());
                builder.clone(*topLevelFn.getOperation());
            }
        }

        // Translate MLIR to LLVM IR
        auto llvmModule = translateModuleToLLVMIR(gpuModules[0], llvmContext);
        if (!llvmModule) {
            return nullptr;
        }

        // Configure the LLVM module with correct target triple and data layout
        llvmModule->setTargetTriple(llvm::Triple(targetTriple));
        llvmModule->setDataLayout(targetMachine->createDataLayout());

        // Only global kernels should have external linkage.
        for (auto &func : llvmModule->functions()) {
            if (func.isDeclaration())
                continue;
            if (func.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL ||
                func.getCallingConv() == llvm::CallingConv::PTX_Kernel) {
                func.setLinkage(llvm::GlobalValue::ExternalLinkage);
                continue;
            }
            func.setLinkage(llvm::GlobalValue::InternalLinkage);
        }

        if (options.optimization_level > 0) {
            llvm::OptimizationLevel optLevel;
            switch (options.optimization_level) {
            case 1:
                optLevel = llvm::OptimizationLevel::O1;
                break;
            case 2:
                optLevel = llvm::OptimizationLevel::O2;
                break;
            case 3:
                optLevel = llvm::OptimizationLevel::O3;
                break;
            default:
                optLevel = llvm::OptimizationLevel::O0;
                break;
            }

            llvm::PipelineTuningOptions tuningOptions;
            tuningOptions.LoopUnrolling = true;
            tuningOptions.LoopInterleaving = true;
            tuningOptions.LoopVectorization = true;
            tuningOptions.SLPVectorization = true;

            llvm::PassBuilder pb(targetMachine.get(), tuningOptions,
                                 std::nullopt, nullptr);

            llvm::LoopAnalysisManager lam;
            llvm::FunctionAnalysisManager fam;
            llvm::CGSCCAnalysisManager cgam;
            llvm::ModuleAnalysisManager mam;

            pb.registerModuleAnalyses(mam);
            pb.registerCGSCCAnalyses(cgam);
            pb.registerFunctionAnalyses(fam);
            pb.registerLoopAnalyses(lam);
            pb.crossRegisterProxies(lam, fam, cgam, mam);

            llvm::ModulePassManager mpm =
                pb.buildPerModuleDefaultPipeline(optLevel);

            mpm.run(*llvmModule, mam);
        }

        return llvmModule;
    }

  public:
    causalflow::avelang::ir::IRContext *ir_context_;
};

LowerToLLVM::LowerToLLVM(causalflow::avelang::ir::IRContext *ir_context)
    : impl_(std::make_unique<Impl>(ir_context)) {}

LowerToLLVM::~LowerToLLVM() = default;

std::unique_ptr<::llvm::Module>
LowerToLLVM::compile(mlir::ModuleOp module, ::llvm::LLVMContext &llvmContext,
                     const GPUCompilationOptions &options) {
    return impl_->compile(module, llvmContext, options);
}

mlir::MLIRContext *LowerToLLVM::getContext() {
    return impl_->ir_context_->GetMLIRContext();
}

} // namespace causalflow::avelang::target::gpu
