#include "gpu_to_nvvm_pipeline.h"
#include "Dialect/AveLang/Transforms/normalize_ave_lang_return_pass.h"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h>
#include <mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/NVVMDialect.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

namespace causalflow::avelang::target::nvvm {

using namespace mlir;

static const int kIndexBitwidth = 64;

static void buildCommonPassPipeline(OpPassManager &pm,
                                    const NVVMToLLVMPipelineOptions &options) {
    bufferization::OneShotBufferizePassOptions bufferizationOptions;
    bufferizationOptions.bufferizeFunctionBoundaries = true;
    pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOptions));
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(
        causalflow::avelang::dialect::createNormalizeAveLangReturnPass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(affine::createAffineExpandIndexOpsPass());
    pm.addPass(createLowerAffinePass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // Reconcile unrealized casts at the end to resolve any remaining type
    // conversion issues
    pm.addPass(createReconcileUnrealizedCastsPass());

    // Add final canonicalization to clean up after conversion
    pm.addPass(createCanonicalizerPass());
}

/// Build the GPU pass pipeline for GPU module-specific transformations.
static void buildGpuPassPipeline(OpPassManager &pm,
                                 const NVVMToLLVMPipelineOptions &options) {
    GpuNVVMAttachTargetOptions nvvmOptions;
    nvvmOptions.chip = options.chipset;
    nvvmOptions.triple = options.triple;
    nvvmOptions.optLevel = options.optimization_level;
    pm.addPass(createGpuNVVMAttachTarget(nvvmOptions));

    ConvertGpuOpsToNVVMOpsOptions convertOptions;
    convertOptions.indexBitwidth = kIndexBitwidth;
    convertOptions.useBarePtrCallConv = options.use_bare_ptr_memref_call_conv;
    pm.addNestedPass<gpu::GPUModuleOp>(
        createConvertGpuOpsToNVVMOps(convertOptions));
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertNVGPUToNVVMPass());
    pm.addPass(createConvertNVVMToLLVMPass());

    // Add vector-to-LLVM pass to lower vector operations from intrinsics
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertVectorToLLVMPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertVectorToLLVMPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createArithToLLVMConversionPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertIndexToLLVMPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createUBToLLVMConversionPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());

    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createReconcileUnrealizedCastsPass());

    // Add final canonicalization to clean up after GPU conversion
    pm.addPass(createCanonicalizerPass());
}

void BuildLowerToNVVMPassPipeline(OpPassManager &pm,
                                  const NVVMToLLVMPipelineOptions &options) {
    buildCommonPassPipeline(pm, options);
    buildGpuPassPipeline(pm, options);
}

} // namespace causalflow::avelang::target::nvvm
