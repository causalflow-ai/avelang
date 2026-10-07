#pragma once

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/VersionTuple.h>
#include <llvm/TargetParser/Triple.h>
#include <map>
#include <string>
#include <vector>

namespace causalflow::avelang::target::amdgpu {

/// ABI version of device library.
struct DeviceLibABIVersion {
    unsigned ABIVersion = 0;
    DeviceLibABIVersion(unsigned V) : ABIVersion(V) {}

    static DeviceLibABIVersion
    fromCodeObjectVersion(unsigned CodeObjectVersion) {
        if (CodeObjectVersion < 4)
            CodeObjectVersion = 4;
        return DeviceLibABIVersion(CodeObjectVersion * 100);
    }

    /// Whether ABI version bc file is requested.
    bool requiresLibrary() const { return ABIVersion >= 500; }

    std::string toString() const { return std::to_string(ABIVersion / 100); }
};

/// A simplified ROCm installation detector for avelang
class RocmInstallationDetector {
  private:
    struct ConditionalLibrary {
        llvm::SmallString<0> On;
        llvm::SmallString<0> Off;

        bool isValid() const { return !On.empty() && !Off.empty(); }

        llvm::StringRef get(bool Enabled) const { return Enabled ? On : Off; }
    };

    bool HasDeviceLibrary = false;

    // Paths
    llvm::SmallString<0> LibDevicePath;
    llvm::StringMap<std::string> LibDeviceMap;

    // Libraries that are always linked.
    llvm::SmallString<0> OCML;
    llvm::SmallString<0> OCKL;

    // Libraries swapped based on compile flags.
    ConditionalLibrary WavefrontSize64;
    ConditionalLibrary FiniteOnly;
    ConditionalLibrary UnsafeMath;
    ConditionalLibrary DenormalsAreZero;
    ConditionalLibrary CorrectlyRoundedSqrt;

    // Maps ABI version to library path.
    std::map<unsigned, std::string> ABIVersionMap;

    bool allGenericLibsValid() const {
        // Newer ROCm releases removed some conditional control libraries.
        // Require only OCML, OCKL and wavefrontsize64; link others when present.
        return !OCML.empty() && !OCKL.empty() && WavefrontSize64.isValid();
    }

    void scanLibDevicePath(llvm::StringRef Path);
    std::vector<std::string> getInstallationPathCandidates();

  public:
    RocmInstallationDetector();

    /// Get file paths of default bitcode libraries common to AMDGPU based
    /// toolchains.
    std::vector<std::string>
    getCommonBitcodeLibs(llvm::StringRef LibDeviceFile, bool Wave64, bool DAZ,
                         bool FiniteOnly, bool UnsafeMathOpt,
                         bool FastRelaxedMath, bool CorrectSqrt,
                         DeviceLibABIVersion ABIVer) const;

    /// Check file paths of default bitcode libraries.
    bool checkCommonBitcodeLibs(llvm::StringRef GPUArch,
                                llvm::StringRef LibDeviceFile,
                                DeviceLibABIVersion ABIVer) const;

    /// Check whether we detected a valid ROCm device library.
    bool hasDeviceLibrary() const { return HasDeviceLibrary; }

    /// Get the detected ROCm device library path.
    llvm::StringRef getLibDevicePath() const { return LibDevicePath; }

    llvm::StringRef getOCMLPath() const { return OCML; }
    llvm::StringRef getOCKLPath() const { return OCKL; }

    llvm::StringRef getWavefrontSize64Path(bool Enabled) const {
        return WavefrontSize64.get(Enabled);
    }
    llvm::StringRef getFiniteOnlyPath(bool Enabled) const {
        return FiniteOnly.get(Enabled);
    }
    llvm::StringRef getUnsafeMathPath(bool Enabled) const {
        return UnsafeMath.get(Enabled);
    }
    llvm::StringRef getDenormalsAreZeroPath(bool Enabled) const {
        return DenormalsAreZero.get(Enabled);
    }
    llvm::StringRef getCorrectlyRoundedSqrtPath(bool Enabled) const {
        return CorrectlyRoundedSqrt.get(Enabled);
    }

    llvm::StringRef getABIVersionPath(DeviceLibABIVersion ABIVer) const {
        auto Loc = ABIVersionMap.find(ABIVer.ABIVersion);
        if (Loc == ABIVersionMap.end())
            return llvm::StringRef();
        return Loc->second;
    }

    /// Get libdevice file for given architecture
    llvm::StringRef getLibDeviceFile(llvm::StringRef Gpu) const {
        auto Loc = LibDeviceMap.find(Gpu);
        if (Loc == LibDeviceMap.end())
            return "";
        return Loc->second;
    }

    void detectDeviceLibrary();
};

/// Utility functions for ROCm installation paths (config-based)
namespace RocmPaths {
/// Get device library search paths (for bitcode files)
std::vector<std::string> getDeviceLibraryPaths();

/// Get tool search paths (for binaries)
std::vector<std::string> getToolPaths();
} // namespace RocmPaths

} // namespace causalflow::avelang::target::amdgpu