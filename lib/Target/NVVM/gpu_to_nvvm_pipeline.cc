#include "gpu_to_nvvm_pipeline.h"
#include "Dialect/AveLang/Transforms/normalize_ave_lang_return_pass.h"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h>
#include <mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Affine/Transforms/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/NVGPU/IR/NVGPUDialect.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

namespace causalflow::avelang::target::nvvm {

using namespace mlir;

static const int kIndexBitwidth = 32;

namespace {

struct ConvertGPUAndNVGPUToNVVMPass
    : public PassWrapper<ConvertGPUAndNVGPUToNVVMPass,
                         OperationPass<gpu::GPUModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertGPUAndNVGPUToNVVMPass)

    ConvertGPUAndNVGPUToNVVMPass() = default;
    ConvertGPUAndNVGPUToNVVMPass(const ConvertGPUAndNVGPUToNVVMPass &) =
        default;
    ConvertGPUAndNVGPUToNVVMPass(unsigned indexBitwidth,
                                 bool useBarePtrCallConv)
        : indexBitwidth(indexBitwidth), useBarePtrCallConv(useBarePtrCallConv) {
    }

    void runOnOperation() override {
        LowerToLLVMOptions options(&getContext());
        options.overrideIndexBitwidth(indexBitwidth);
        options.useBarePtrCallConv = useBarePtrCallConv;

        LLVMTypeConverter converter(&getContext(), options);
        configureGpuToNVVMTypeConverter(converter);
        converter.addConversion([&](nvgpu::DeviceAsyncTokenType type) -> Type {
            return converter.convertType(
                IntegerType::get(type.getContext(), 32));
        });
        converter.addConversion(
            [&](nvgpu::WarpgroupAccumulatorType type) -> Type {
                auto fragmented = type.getFragmented();
                unsigned numMembers =
                    fragmented.getElementType().isF32() ||
                            fragmented.getElementType().isInteger(32)
                        ? fragmented.getDimSize(1) / 2
                        : fragmented.getDimSize(1) / 4;
                auto rowType = LLVM::LLVMStructType::getLiteral(
                    &getContext(),
                    SmallVector<Type>(numMembers, fragmented.getElementType()));
                auto resultType = LLVM::LLVMStructType::getLiteral(
                    &getContext(),
                    SmallVector<Type>(fragmented.getDimSize(0) / kWgmmaSizeM,
                                      rowType));
                return converter.convertType(resultType);
            });
        converter.addConversion([&](nvgpu::MBarrierTokenType type) -> Type {
            return converter.convertType(
                IntegerType::get(type.getContext(), 64));
        });
        converter.addConversion(
            [&](nvgpu::WarpgroupMatrixDescriptorType type) -> Type {
                return converter.convertType(
                    IntegerType::get(type.getContext(), 64));
            });
        converter.addConversion([&](nvgpu::MBarrierGroupType type) -> Type {
            return converter.convertType(
                nvgpu::getMBarrierMemrefType(&getContext(), type));
        });
        converter.addConversion(
            [&](nvgpu::TensorMapDescriptorType type) -> Type {
                return LLVM::LLVMPointerType::get(type.getContext());
            });

        RewritePatternSet patterns(&getContext());
        populateGpuToNVVMConversionPatterns(converter, patterns,
                                            /*benefit=*/10);
        populateGpuWMMAToNVVMConversionPatterns(converter, patterns);
        populateNVGPUToNVVMConversionPatterns(converter, patterns);
        cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
        populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);

        LLVMConversionTarget target(getContext());
        configureGpuToNVVMConversionLegality(target);
        target.addIllegalDialect<nvgpu::NVGPUDialect>();
        target.addIllegalDialect<cf::ControlFlowDialect>();
        target.addIllegalDialect<memref::MemRefDialect>();
        target.addLegalDialect<arith::ArithDialect, vector::VectorDialect>();
        scf::populateSCFStructuralTypeConversionsAndLegality(converter,
                                                             patterns, target);
        if (failed(applyPartialConversion(getOperation(), target,
                                          std::move(patterns)))) {
            signalPassFailure();
        }
    }

    unsigned indexBitwidth = 32;
    bool useBarePtrCallConv = true;
};

} // namespace

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

    pm.addNestedPass<gpu::GPUModuleOp>(
        std::make_unique<ConvertGPUAndNVGPUToNVVMPass>(
            kIndexBitwidth, options.use_bare_ptr_memref_call_conv));
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
