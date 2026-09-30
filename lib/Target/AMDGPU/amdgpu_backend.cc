#include "amdgpu_backend.h"
#include "Target/GPU/lower_to_llvm.h"
#include "amdgpu_toolchain.h"
#include "gpu_to_amdgpu_pipeline.h"
#include "rocm_installation_detector.h"
#include <algorithm>
#include <clang/Driver/OffloadBundler.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/AMDGPUTargetParser.h>
#include <llvm/TargetParser/Triple.h>
#include <map>
#include <mutex>
#include <sstream>

namespace causalflow::avelang::target::amdgpu {

void AMDGPUBackend::buildLoweringPipeline(
    mlir::OpPassManager &pm,
    const causalflow::avelang::target::gpu::GPUCompilationOptions &options) {
    AMDGPUToLLVMPipelineOptions amdgpuOptions;
    amdgpuOptions.chipset = options.chipset.str();
    amdgpuOptions.triple = options.triple.str();
    amdgpuOptions.optimization_level = options.optimization_level;
    amdgpuOptions.num_warps = options.num_warps;
    amdgpuOptions.use_bare_ptr_memref_call_conv =
        options.use_bare_ptr_memref_call_conv;
    amdgpuOptions.target_features = options.target_features;

    BuildLowerToAMDGPUPassPipeline(pm, amdgpuOptions);
}

bool AMDGPUBackend::supportsTriple(llvm::StringRef triple) const {
    return triple.contains("amdgcn") || triple.contains("amd");
}

std::string AMDGPUBackend::getName() const { return "AMDGPU"; }

void AMDGPUBackend::EnsureInitialized() {
    static std::once_flag initFlag;
    std::call_once(initFlag, []() {
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUAsmPrinter();
    });
}

namespace {
llvm::CodeGenOptLevel toCodeGenOptLevel(unsigned optLevel) {
    switch (optLevel) {
    case 3:
        return llvm::CodeGenOptLevel::Aggressive;
    case 2:
        return llvm::CodeGenOptLevel::Default;
    case 1:
        return llvm::CodeGenOptLevel::Less;
    default:
        return llvm::CodeGenOptLevel::None;
    }
}

// Parse target features string into a feature map
// Input: "+wavefrontsize64,-sramecc,+xnack"
// Output: {wavefrontsize64: true, sramecc: false, xnack: true}
std::map<std::string, bool>
parseTargetFeatures(const std::string &featuresStr) {
    std::map<std::string, bool> features;

    if (featuresStr.empty()) {
        return features;
    }

    // Split by comma
    std::stringstream ss(featuresStr);
    std::string feature;

    while (std::getline(ss, feature, ',')) {
        // Trim whitespace
        feature.erase(0, feature.find_first_not_of(" \t"));
        feature.erase(feature.find_last_not_of(" \t") + 1);

        if (feature.empty())
            continue;

        if (feature[0] == '+') {
            features[feature.substr(1)] = true;
        } else if (feature[0] == '-') {
            features[feature.substr(1)] = false;
        }
        // Ignore features without +/- prefix
    }

    return features;
}

// Get default feature values based on GPU architecture
std::map<std::string, bool> getDefaultFeatures(const std::string &chipset) {
    std::map<std::string, bool> defaults;

    // Parse GPU kind
    auto gpuKind = llvm::AMDGPU::parseArchAMDGCN(chipset);

    // Set default wavefront size based on architecture
    const unsigned archAttr = llvm::AMDGPU::getArchAttrAMDGCN(gpuKind);
    bool hasWave32 = (archAttr & llvm::AMDGPU::FEATURE_WAVE32);
    defaults["wavefrontsize64"] =
        !hasWave32; // Default to wave64 for older GPUs

    // Set other defaults (conservative choices)
    defaults["daz"] = false; // Denormals are zero: off (preserve precision)
    defaults["unsafe-math"] = false; // Unsafe math: off (preserve accuracy)
    defaults["finite-only"] = false; // Finite only: off (handle inf/nan)
    defaults["correctly-rounded-sqrt"] =
        true; // Correctly rounded sqrt: on (precision)

    // Target ID features defaults based on GPU capabilities
    if (archAttr & llvm::AMDGPU::FEATURE_SRAMECC) {
        defaults["sramecc"] = true; // Enable if supported
    }
    if (archAttr & llvm::AMDGPU::FEATURE_XNACK) {
        defaults["xnack"] = false; // Usually disabled by default
    }

    return defaults;
}

// Merge user features with defaults
std::map<std::string, bool> resolveFeatures(const std::string &featuresStr,
                                            const std::string &chipset) {
    auto defaults = getDefaultFeatures(chipset);
    auto userFeatures = parseTargetFeatures(featuresStr);

    // Override defaults with user-specified features
    for (const auto &[feature, enabled] : userFeatures) {
        defaults[feature] = enabled;
    }

    return defaults;
}
} // namespace

llvm::Expected<std::string> AMDGPUBackend::generateBinary(
    llvm::Module &module,
    const causalflow::avelang::target::gpu::GPUCompilationOptions &options) {
    std::string targetTriple = options.triple.str();
    if (targetTriple.empty()) {
        targetTriple = "amdgcn-amd-amdhsa";
    }
    llvm::Triple triple(targetTriple);

    std::string error;
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Failed to lookup AMDGPU target: " +
                                           error);
    }

    llvm::TargetOptions targetOptions;
    auto targetMachine =
        std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
            triple, options.chipset.str(), "", targetOptions, llvm::Reloc::PIC_,
            std::nullopt, toCodeGenOptLevel(options.optimization_level),
            false));
    if (!targetMachine) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "Failed to create AMDGPU target machine");
    }

    // Set up the module with the target data layout
    module.setTargetTriple(llvm::Triple(targetTriple));
    module.setDataLayout(targetMachine->createDataLayout());

    auto emitBitcode = [&]() -> llvm::Expected<std::string> {
        llvm::SmallVector<char, 1024> bitcodeBuffer;
        llvm::raw_svector_ostream bitcodeStream(bitcodeBuffer);
        llvm::WriteBitcodeToFile(module, bitcodeStream);
        std::string bitcode = bitcodeStream.str().str();
        if (bitcode.empty()) {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "Generated LLVM bitcode is empty");
        }
        return bitcode;
    };

    // Link LLVM bitcode with ROCm device libraries so OCML functions can be
    // internalized and inlined into the kernel before final codegen.
    auto linkBitcodeWithClang =
        [&](const std::string &bitcode) -> llvm::Expected<std::string> {
        auto inputTempFile =
            llvm::sys::fs::TempFile::create("/tmp/kernel-%%%%%%.bc");
        if (!inputTempFile) {
            return joinErrors(llvm::createStringError(
                                  llvm::inconvertibleErrorCode(),
                                  "Failed to create temporary bitcode file"),
                              inputTempFile.takeError());
        }

        auto outTempFile =
            llvm::sys::fs::TempFile::create("/tmp/kernel-%%%%%%.out");
        if (!outTempFile) {
            llvm::consumeError(inputTempFile->discard());
            return joinErrors(llvm::createStringError(
                                  llvm::inconvertibleErrorCode(),
                                  "Failed to create temporary output file"),
                              outTempFile.takeError());
        }

        {
            llvm::raw_fd_ostream inputOS(inputTempFile->FD,
                                         /*shouldClose=*/false);
            inputOS.write(bitcode.data(), bitcode.size());
            inputOS.flush();
        }

        std::string inputFilePath = inputTempFile->TmpName;
        std::string outFilePath = outTempFile->TmpName;

        // Prepare ld.lld command using ROCm device library discovery similar to
        // clang
        std::string chipset = options.chipset.str();
        if (chipset.empty()) {
            chipset = "gfx90a";
        }

        // Create toolchain to manage tools and their discovery
        AMDGPUToolChain toolchain{llvm::Triple(targetTriple)};
        auto &linker = toolchain.getLinker();

        std::string linkerPath = linker.getLinkerPath();

        // Use RocmInstallationDetector to find and select device libraries
        // based on features
        auto findDeviceLibraries = [&]() -> std::vector<std::string> {
            static RocmInstallationDetector rocmDetector;

            if (!rocmDetector.hasDeviceLibrary()) {
                return {};
            }

            // Parse target features and resolve with defaults
            auto features = resolveFeatures(options.target_features, chipset);

            // Get GPU architecture in canonical form
            auto gpuKind = llvm::AMDGPU::parseArchAMDGCN(chipset);
            const llvm::StringRef canonArch =
                llvm::AMDGPU::getArchNameAMDGCN(gpuKind);

            // Get device library file for this arch
            llvm::StringRef libDeviceFile =
                rocmDetector.getLibDeviceFile(canonArch);

            if (libDeviceFile.empty()) {
                return {};
            }

            // Create ABI version (use v5 by default)
            auto abiVer = DeviceLibABIVersion::fromCodeObjectVersion(5);

            // Check if we have the required libraries
            if (!rocmDetector.checkCommonBitcodeLibs(canonArch, libDeviceFile,
                                                     abiVer)) {
                return {};
            }

            // Use feature-based selection
            bool wave64 = features["wavefrontsize64"];
            bool daz = features["daz"];
            bool finiteOnly = features["finite-only"];
            bool unsafeMathOpt = features["unsafe-math"];
            bool fastRelaxedMath =
                false; // Not supported in our feature set yet
            bool correctSqrt = features["correctly-rounded-sqrt"];

            // Get common bitcode libraries using the detector
            return rocmDetector.getCommonBitcodeLibs(
                libDeviceFile, wave64, daz, finiteOnly, unsafeMathOpt,
                fastRelaxedMath, correctSqrt, abiVer);
        };

        // Get device libraries and construct linker arguments
        auto deviceLibs = findDeviceLibraries();
        auto cmdArgs = linker.constructLinkerArgs(chipset, deviceLibs,
                                                  inputFilePath, outFilePath);

        // Execute the linker command safely using LLVM's ExecuteAndWait
        // This avoids potential command injection vulnerabilities from
        // std::system
        std::vector<llvm::StringRef> args;
        // First argument must be the program name (argv[0])
        args.push_back(linkerPath);
        for (const auto &arg : cmdArgs) {
            args.push_back(arg);
        }

        std::string errMsg;
        int result =
            llvm::sys::ExecuteAndWait(linkerPath, args, /*Env=*/std::nullopt,
                                      /*Redirects=*/{}, /*SecondsToWait=*/0,
                                      /*MemoryLimit=*/0, &errMsg);

        if (result != 0) {
            // Clean up temporary files on failure
            llvm::consumeError(inputTempFile->discard());
            llvm::consumeError(outTempFile->discard());
            std::string errorDetail =
                errMsg.empty()
                    ? "AMDGPU device linking failed with exit code " +
                          std::to_string(result)
                    : "AMDGPU device linking failed: " + errMsg +
                          " (exit code " + std::to_string(result) + ")";
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           errorDetail);
        }

        // Read the linked output
        auto bufferOrErr = llvm::MemoryBuffer::getFile(outFilePath);
        if (!bufferOrErr) {
            // Clean up temporary files on failure
            llvm::consumeError(inputTempFile->discard());
            llvm::consumeError(outTempFile->discard());
            return llvm::createStringError(bufferOrErr.getError(),
                                           "Failed to read linked output file");
        }

        std::string result_data = (*bufferOrErr)->getBuffer().str();

        // Explicitly discard the temporary files to ensure proper cleanup
        if (auto err = inputTempFile->discard()) {
            // Log the error but don't fail since we have our result
            llvm::consumeError(std::move(err));
        }
        if (auto err = outTempFile->discard()) {
            // Log the error but don't fail since we have our result
            llvm::consumeError(std::move(err));
        }

        return result_data;
    };

    auto bitcodeOrErr = emitBitcode();
    if (!bitcodeOrErr) {
        return bitcodeOrErr.takeError();
    }

    auto linkResult = linkBitcodeWithClang(*bitcodeOrErr);
    if (linkResult) {
        return *linkResult;
    }
    return linkResult.takeError();
}

llvm::Expected<std::string> AMDGPUBackend::generateAssembly(
    llvm::Module &module,
    const causalflow::avelang::target::gpu::GPUCompilationOptions &options) {
    std::string targetTriple = options.triple.str();
    if (targetTriple.empty()) {
        targetTriple = "amdgcn-amd-amdhsa";
    }
    llvm::Triple triple(targetTriple);

    std::string error;
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Failed to lookup AMDGPU target: " +
                                           error);
    }

    llvm::TargetOptions targetOptions;
    auto targetMachine =
        std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
            triple, options.chipset.str(), "", targetOptions, llvm::Reloc::PIC_,
            std::nullopt, toCodeGenOptLevel(options.optimization_level),
            false));
    if (!targetMachine) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "Failed to create AMDGPU target machine");
    }

    // Set up the module with the target data layout
    module.setTargetTriple(llvm::Triple(targetTriple));
    module.setDataLayout(targetMachine->createDataLayout());

    // Generate AMDGPU assembly using raw_svector_ostream
    llvm::SmallVector<char, 1024> asmBuffer;
    llvm::raw_svector_ostream asmStream(asmBuffer);

    llvm::legacy::PassManager passManager;
    if (targetMachine->addPassesToEmitFile(
            passManager, asmStream, nullptr,
            llvm::CodeGenFileType::AssemblyFile)) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "Target machine cannot emit AMDGPU assembly");
    }

    passManager.run(module);
    std::string amdgpuAssembly = asmStream.str().str();

    if (amdgpuAssembly.empty()) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Generated AMDGPU assembly is empty");
    }

    return amdgpuAssembly;
}

std::unique_ptr<causalflow::avelang::target::gpu::GPUBackendInterface>
CreateAMDGPUBackend() {
    return std::make_unique<AMDGPUBackend>();
}

} // namespace causalflow::avelang::target::amdgpu
