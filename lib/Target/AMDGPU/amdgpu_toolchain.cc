#include "amdgpu_toolchain.h"
#include "rocm_installation_detector.h"

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Process.h>
#include <sstream>

namespace causalflow::avelang::target::amdgpu {

//===----------------------------------------------------------------------===//
// Linker Implementation
//===----------------------------------------------------------------------===//

Linker::Linker() : Tool("amdgpu::Linker", "ld.lld") {}

std::string Linker::findLinkerExecutable() const {
    // For AMDGPU device linking, we should use clang instead of ld.lld directly
    // because clang knows how to properly link LLVM bitcode files
    auto toolPaths = RocmPaths::getToolPaths();
    std::vector<std::string> candidates;

    // Try to find clang/hipcc first
    for (const auto &toolPath : toolPaths) {
        candidates.push_back(toolPath + "/clang");
    }

    for (const auto &candidate : candidates) {
        if (llvm::sys::fs::can_execute(candidate)) {
            return candidate;
        }
    }

    // Fallback: try to find clang in PATH
    auto clangPath = llvm::sys::findProgramByName("clang");
    if (clangPath) {
        return *clangPath;
    }

    // Last resort: fall back to ld.lld (this won't work for bitcode linking)
    auto ldlldPath = llvm::sys::findProgramByName("ld.lld");
    if (ldlldPath) {
        return *ldlldPath;
    }

    // Default fallback
    return "clang";
}

std::string Linker::getLinkerPath() const {
    if (CachedLinkerPath.empty()) {
        CachedLinkerPath = findLinkerExecutable();
    }
    return CachedLinkerPath;
}

std::vector<std::string> Linker::constructLinkerArgs(
    const std::string &chipset, const std::vector<std::string> &deviceLibs,
    const std::string &inputFile, const std::string &outputFile) const {

    std::vector<std::string> args;

    // Use clang-style arguments for device linking
    args.push_back("-target");
    args.push_back("amdgcn-amd-amdhsa");
    args.push_back("-flto");
    args.push_back("-mcpu=" + chipset);
    args.push_back("-nostdlib");
    args.push_back("-Xlinker");
    args.push_back("--no-undefined");
    args.push_back("-Xlinker");
    args.push_back("-shared");

    args.push_back("-mllvm");
    args.push_back("-amdgpu-internalize-symbols");
    args.push_back("-mllvm");
    args.push_back("-amdgpu-early-inline-all=true");
    args.push_back("-mllvm");
    args.push_back("-amdgpu-function-calls=false");
    args.push_back("-Xlinker");
    args.push_back("-plugin-opt=-amdgpu-mfma-vgpr-form=true");

    // Add input object file
    args.push_back(inputFile);

    // Add device libraries as bitcode files
    for (const auto &lib : deviceLibs) {
        args.push_back(lib);
    }

    args.push_back("-o");
    args.push_back(outputFile);

    return args;
}

//===----------------------------------------------------------------------===//
// AMDGPUToolChain Implementation
//===----------------------------------------------------------------------===//

AMDGPUToolChain::AMDGPUToolChain(const llvm::Triple &triple) : Triple(triple) {
    initializeToolSearchPaths();
}

AMDGPUToolChain::~AMDGPUToolChain() = default;

void AMDGPUToolChain::initializeToolSearchPaths() {
    ToolSearchPaths = RocmPaths::getToolPaths();
}

Linker &AMDGPUToolChain::getLinker() const {
    if (!LinkerTool) {
        LinkerTool = std::make_unique<Linker>();
    }
    return *LinkerTool;
}

std::string AMDGPUToolChain::getProgramPath(const std::string &name) const {
    // First, try the tool search paths
    for (const auto &path : ToolSearchPaths) {
        llvm::SmallString<128> fullPath(path);
        llvm::sys::path::append(fullPath, name);

        if (llvm::sys::fs::can_execute(fullPath)) {
            return fullPath.str().str();
        }
    }

    // Fallback: try to find in PATH
    auto progPath = llvm::sys::findProgramByName(name);
    if (progPath) {
        return *progPath;
    }

    // Last resort: return the name as-is (let system handle it)
    return name;
}

std::vector<std::string> AMDGPUToolChain::getToolSearchPaths() const {
    return ToolSearchPaths;
}

} // namespace causalflow::avelang::target::amdgpu
