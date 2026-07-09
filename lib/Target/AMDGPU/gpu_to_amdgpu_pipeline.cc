#include "gpu_to_amdgpu_pipeline.h"
#include "Dialect/AveLang/Transforms/normalize_ave_lang_return_pass.h"
#include "legalize_gpu_shuffle_to_idx_pass.h"
#include "lower_math_to_amdgpu_pass.h"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h>
#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/AMDGPU/Utils/Chipset.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#define GEN_PASS_DECL_EXPANDSTRIDEDMETADATAPASS
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#undef GEN_PASS_DECL_EXPANDSTRIDEDMETADATAPASS
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <map>
#include <sstream>

namespace causalflow::avelang::target::amdgpu {

using namespace mlir;

namespace {

class OverrideRocdlMaxFlatWorkgroupSizePass
    : public PassWrapper<OverrideRocdlMaxFlatWorkgroupSizePass,
                         OperationPass<gpu::GPUModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        OverrideRocdlMaxFlatWorkgroupSizePass)

    OverrideRocdlMaxFlatWorkgroupSizePass(int maxFlatWorkgroupSize)
        : maxFlatWorkgroupSize(maxFlatWorkgroupSize) {}

    void runOnOperation() override {
        if (maxFlatWorkgroupSize <= 0) {
            return;
        }

        gpu::GPUModuleOp gpuModule = getOperation();
        MLIRContext *context = gpuModule.getContext();
        auto *rocdlDialect = context->getOrLoadDialect<ROCDL::ROCDLDialect>();
        auto maxFlatWorkgroupSizeAttr =
            rocdlDialect->getMaxFlatWorkGroupSizeAttrHelper();
        auto wavesPerEuAttr = rocdlDialect->getWavesPerEuAttrHelper();
        Builder builder(context);
        IntegerAttr attr = builder.getI32IntegerAttr(maxFlatWorkgroupSize);
        IntegerAttr wavesAttr = builder.getI32IntegerAttr(1);

        gpuModule.walk([&](gpu::GPUFuncOp func) {
            if (func.isKernel()) {
                maxFlatWorkgroupSizeAttr.setAttr(func, attr);
                wavesPerEuAttr.setAttr(func, wavesAttr);
            }
        });
    }

  private:
    int maxFlatWorkgroupSize;
};

static std::unique_ptr<Pass>
createOverrideRocdlMaxFlatWorkgroupSizePass(int maxFlatWorkgroupSize) {
    return std::make_unique<OverrideRocdlMaxFlatWorkgroupSizePass>(
        maxFlatWorkgroupSize);
}

} // namespace

static void
buildCommonPassPipeline(OpPassManager &pm,
                        const AMDGPUToLLVMPipelineOptions &options) {
    pm.addPass(bufferization::createOneShotBufferizePass());
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
    pm.addPass(createCanonicalizerPass());
}

/// Build the GPU pass pipeline for GPU module-specific transformations.
static void buildGpuPassPipeline(OpPassManager &pm,
                                 const AMDGPUToLLVMPipelineOptions &options) {
    // Parse target features to configure GPU passes
    auto parseFeatures =
        [](const std::string &featuresStr) -> std::map<std::string, bool> {
        std::map<std::string, bool> features;
        if (featuresStr.empty())
            return features;

        std::stringstream ss(featuresStr);
        std::string feature;
        while (std::getline(ss, feature, ',')) {
            feature.erase(0, feature.find_first_not_of(" \t"));
            feature.erase(feature.find_last_not_of(" \t") + 1);
            if (feature.empty())
                continue;

            if (feature[0] == '+') {
                features[feature.substr(1)] = true;
            } else if (feature[0] == '-') {
                features[feature.substr(1)] = false;
            }
        }
        return features;
    };

    auto features = parseFeatures(options.target_features);

    // Set wave64 flag based on target features, defaulting to true for backward
    // compatibility
    bool wave64Flag = true; // Default
    if (features.count("wavefrontsize64")) {
        wave64Flag = features["wavefrontsize64"];
    }

    GpuROCDLAttachTargetOptions rocdlOptions;
    rocdlOptions.chip = options.chipset;
    rocdlOptions.triple = options.triple;
    rocdlOptions.optLevel = options.optimization_level;
    rocdlOptions.wave64Flag = wave64Flag;
    pm.addPass(createGpuROCDLAttachTarget(rocdlOptions));
    if (options.num_warps > 0) {
        const int waveSize = wave64Flag ? 64 : 32;
        pm.addNestedPass<gpu::GPUModuleOp>(
            createOverrideRocdlMaxFlatWorkgroupSizePass(options.num_warps *
                                                        waveSize));
    }
    if (options.optimization_level > 0) {
        pm.addNestedPass<gpu::GPUModuleOp>(createLowerMathToAMDGPUPass());
    }
    pm.addNestedPass<gpu::GPUModuleOp>(createLegalizeGPUShuffleToIDXPass());

    ConvertGpuOpsToROCDLOpsOptions gpuToRocdlOptions;
    gpuToRocdlOptions.chipset = options.chipset;
    gpuToRocdlOptions.indexBitwidth = kDeriveIndexBitwidthFromDataLayout;
    gpuToRocdlOptions.useBarePtrCallConv =
        options.use_bare_ptr_memref_call_conv;
    gpuToRocdlOptions.runtime = gpu::amd::Runtime::HIP;
    pm.addNestedPass<gpu::GPUModuleOp>(
        createConvertGpuOpsToROCDLOps(std::move(gpuToRocdlOptions)));

    pm.addNestedPass<gpu::GPUModuleOp>(createConvertAMDGPUToROCDLPass());

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

void BuildLowerToAMDGPUPassPipeline(
    OpPassManager &pm, const AMDGPUToLLVMPipelineOptions &options) {
    buildCommonPassPipeline(pm, options);
    buildGpuPassPipeline(pm, options);
}

} // namespace causalflow::avelang::target::amdgpu
