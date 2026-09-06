#include "nvvm_backend.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "IR/mlir_generator.h"
#include "IR/type_system.h"
#include "Target/GPU/lower_to_llvm.h"
#include "gpu_to_nvvm_pipeline.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mutex>
#include <optional>
#include <sstream>

namespace causalflow::avelang::target::nvvm {

namespace {

namespace ave = causalflow::avelang::dialect;

std::optional<int64_t> getConstantIntValue(mlir::Value value) {
    if (auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr =
                mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            return intAttr.getInt();
        }
    }
    return std::nullopt;
}

bool extractConstantTupleValues(mlir::Value tupleValue,
                                llvm::SmallVectorImpl<int64_t> &values) {
    auto tupleOp = tupleValue.getDefiningOp<ave::MakeIntTupleOp>();
    if (!tupleOp) {
        return false;
    }
    for (auto element : tupleOp.getElements()) {
        auto value = getConstantIntValue(element);
        if (!value) {
            return false;
        }
        values.push_back(*value);
    }
    return true;
}

std::optional<uint64_t> getTMAElementSize(mlir::Type type) {
    if (type.isInteger()) {
        return type.getIntOrFloatBitWidth() / 8;
    }
    if (type.isF16() || type.isBF16()) {
        return 2;
    }
    if (type.isF32()) {
        return 4;
    }
    if (type.isF64()) {
        return 8;
    }
    return std::nullopt;
}

std::optional<std::string> getTMADataType(mlir::Type type,
                                          ir::TypeInfo typeInfo) {
    if (type.isInteger()) {
        unsigned width = type.getIntOrFloatBitWidth();
        bool isUnsigned = typeInfo.is_unsigned_integer.value_or(false);
        // CUDA tensor maps describe the storage width for 8- and 16-bit
        // integers and only expose unsigned enum names for those widths. TMA
        // copies the raw bits, so signed and unsigned AveLang tensors use the
        // same CUDA tensor-map data type.
        if (width == 8) {
            return "CU_TENSOR_MAP_DATA_TYPE_UINT8";
        }
        if (width == 16) {
            return "CU_TENSOR_MAP_DATA_TYPE_UINT16";
        }
        if (isUnsigned && width == 32) {
            return "CU_TENSOR_MAP_DATA_TYPE_UINT32";
        }
        if (!isUnsigned && width == 32) {
            return "CU_TENSOR_MAP_DATA_TYPE_INT32";
        }
        if (isUnsigned && width == 64) {
            return "CU_TENSOR_MAP_DATA_TYPE_UINT64";
        }
        if (!isUnsigned && width == 64) {
            return "CU_TENSOR_MAP_DATA_TYPE_INT64";
        }
        return std::nullopt;
    }
    if (type.isF16()) {
        return "CU_TENSOR_MAP_DATA_TYPE_FLOAT16";
    }
    if (type.isF32()) {
        return "CU_TENSOR_MAP_DATA_TYPE_FLOAT32";
    }
    if (type.isF64()) {
        return "CU_TENSOR_MAP_DATA_TYPE_FLOAT64";
    }
    if (type.isBF16()) {
        return "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16";
    }
    return std::nullopt;
}

mlir::MemRefType getBuiltinMemRefType(mlir::Type type) {
    if (auto builtinType = mlir::dyn_cast<mlir::MemRefType>(type)) {
        return builtinType;
    }
    if (auto aveType = mlir::dyn_cast<ave::MemRefType>(type)) {
        mlir::MemRefLayoutAttrInterface layout;
        auto strides = aveType.getStrides();
        if (!strides.empty()) {
            layout = mlir::StridedLayoutAttr::get(type.getContext(),
                                                  /*offset=*/0, strides);
        }
        return mlir::MemRefType::get(aveType.getShape(),
                                     aveType.getElementType(), layout,
                                     aveType.getMemorySpace());
    }
    return {};
}

llvm::json::Array toJSONArray(llvm::ArrayRef<int64_t> values) {
    llvm::json::Array result;
    for (int64_t value : values) {
        result.emplace_back(value);
    }
    return result;
}

} // namespace

void NVVMBackend::buildLoweringPipeline(
    mlir::OpPassManager &pm,
    const causalflow::avelang::target::gpu::GPUCompilationOptions &options) {
    NVVMToLLVMPipelineOptions nvvmOptions;
    nvvmOptions.chipset = options.chipset.str();
    nvvmOptions.triple = options.triple.str();
    nvvmOptions.optimization_level = options.optimization_level;
    nvvmOptions.num_warps = options.num_warps;
    nvvmOptions.use_bare_ptr_memref_call_conv =
        options.use_bare_ptr_memref_call_conv;

    BuildLowerToNVVMPassPipeline(pm, nvvmOptions);
}

bool NVVMBackend::supportsTriple(llvm::StringRef triple) const {
    return triple.contains("nvptx") || triple.contains("nvidia");
}

std::string NVVMBackend::getName() const { return "NVVM"; }

llvm::json::Object NVVMBackend::getKernelMetadata(mlir::ModuleOp module) const {
    llvm::json::Array specs;
    module.walk([&](ave::NVVMTMADescriptorOp op) {
        mlir::Value tensor = op.getMemref();
        auto tensorType = getBuiltinMemRefType(tensor.getType());
        if (!tensorType || !tensorType.hasStaticShape()) {
            return;
        }

        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(tensor);
        if (!blockArg) {
            return;
        }
        auto funcOp = blockArg.getOwner()
                          ? mlir::dyn_cast_or_null<mlir::func::FuncOp>(
                                blockArg.getOwner()->getParentOp())
                          : mlir::func::FuncOp{};
        if (!funcOp) {
            return;
        }
        auto gpuFuncAttr =
            funcOp->getAttrOfType<mlir::IntegerAttr>("ave.gpu_func");
        if (!gpuFuncAttr ||
            gpuFuncAttr.getInt() !=
                static_cast<int>(
                    ir::MLIRGenerator::FunctionType::kGlobalKernel)) {
            return;
        }

        auto argName = funcOp.getArgAttrOfType<mlir::StringAttr>(
            blockArg.getArgNumber(), "llvm.name");
        auto layoutOp = op.getLayout().getDefiningOp<ave::MakeLayoutOp>();
        if (!argName || !layoutOp) {
            return;
        }

        llvm::SmallVector<int64_t> boxDims;
        if (!extractConstantTupleValues(layoutOp.getDims(), boxDims) ||
            boxDims.empty()) {
            return;
        }

        auto elementSize = getTMAElementSize(tensorType.getElementType());
        auto dataType = getTMADataType(tensorType.getElementType(),
                                       ir::GetTypeInfo(tensor));
        auto swizzleKind = getConstantIntValue(op.getSwizzle());
        if (!elementSize || !dataType || !swizzleKind) {
            return;
        }

        llvm::SmallVector<int64_t> globalDims;
        for (int64_t dim : llvm::reverse(tensorType.getShape())) {
            globalDims.push_back(dim);
        }

        llvm::SmallVector<int64_t> defaultStrides(tensorType.getRank(), 1);
        for (int64_t i = tensorType.getRank() - 2; i >= 0; --i) {
            defaultStrides[i] =
                defaultStrides[i + 1] * tensorType.getDimSize(i + 1);
        }

        llvm::SmallVector<int64_t> globalStrides;
        for (int64_t i = tensorType.getRank() - 2; i >= 0; --i) {
            globalStrides.push_back(defaultStrides[i] * *elementSize);
        }

        llvm::SmallVector<int64_t> reversedBoxDims;
        for (int64_t dim : llvm::reverse(boxDims)) {
            reversedBoxDims.push_back(dim);
        }

        llvm::json::Object spec;
        spec["arg_name"] = argName.getValue();
        spec["rank"] = tensorType.getRank();
        spec["global_dims"] = toJSONArray(globalDims);
        spec["global_strides"] = toJSONArray(globalStrides);
        spec["box_dims"] = toJSONArray(reversedBoxDims);
        spec["dtype"] = *dataType;
        spec["swizzle_kind"] = *swizzleKind;
        specs.emplace_back(std::move(spec));
    });

    llvm::json::Object metadata;
    metadata["tma_descriptor_specs"] = std::move(specs);
    return metadata;
}

void NVVMBackend::EnsureInitialized() {
    static std::once_flag initFlag;
    std::call_once(initFlag, []() {
        LLVMInitializeNVPTXTarget();
        LLVMInitializeNVPTXTargetInfo();
        LLVMInitializeNVPTXTargetMC();
        LLVMInitializeNVPTXAsmPrinter();
    });
}

llvm::Expected<std::string> NVVMBackend::generateBinary(
    llvm::Module &module,
    const causalflow::avelang::target::gpu::GPUCompilationOptions &options) {
    EnsureInitialized();

    // Get the target triple
    std::string targetTripleStr = options.triple.str();
    if (targetTripleStr.empty()) {
        targetTripleStr = "nvptx64-nvidia-cuda";
    }
    llvm::Triple targetTriple(targetTripleStr);

    // Set up the target machine
    std::string error;
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Failed to lookup NVPTX target: " +
                                           error);
    }

    llvm::TargetOptions targetOptions;
    auto targetMachine =
        std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
            targetTriple, options.chipset.str(), "", targetOptions,
            llvm::Reloc::PIC_, std::nullopt,
            options.optimization_level == 3 ? llvm::CodeGenOptLevel::Aggressive
            : options.optimization_level == 2 ? llvm::CodeGenOptLevel::Default
            : options.optimization_level == 1 ? llvm::CodeGenOptLevel::Less
                                              : llvm::CodeGenOptLevel::None,
            false));
    if (!targetMachine) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Failed to create NVPTX target machine");
    }

    // Set up the module with the target data layout
    module.setTargetTriple(targetTriple);
    module.setDataLayout(targetMachine->createDataLayout());

    // Generate PTX assembly using raw_svector_ostream
    llvm::SmallVector<char, 1024> ptxBuffer;
    llvm::raw_svector_ostream ptxStream(ptxBuffer);

    llvm::legacy::PassManager passManager;
    if (targetMachine->addPassesToEmitFile(
            passManager, ptxStream, nullptr,
            llvm::CodeGenFileType::AssemblyFile)) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "Target machine cannot emit PTX assembly");
    }

    passManager.run(module);
    std::string ptxAssembly = ptxStream.str().str();

    if (ptxAssembly.empty()) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Generated PTX assembly is empty");
    }

    return ptxAssembly;
}

llvm::Expected<std::string> NVVMBackend::generateAssembly(
    llvm::Module &module,
    const causalflow::avelang::target::gpu::GPUCompilationOptions &options) {
    // For NVVM, assembly is the same as binary (PTX)
    return generateBinary(module, options);
}

std::unique_ptr<causalflow::avelang::target::gpu::GPUBackendInterface>
CreateNVVMBackend() {
    return std::make_unique<NVVMBackend>();
}

} // namespace causalflow::avelang::target::nvvm
