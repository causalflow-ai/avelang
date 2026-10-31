#pragma once

#include <cstdint>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinTypes.h>

namespace causalflow::avelang::amdgpu::mfma {

enum class VectorElemKind {
    I32,
    FP8,
    F16,
    F32,
    BF16,
};

struct MFMAConfig {
    llvm::StringRef name;
    int64_t m;
    int64_t n;
    int64_t k;
    llvm::StringRef typeA;
    llvm::StringRef typeC;
    llvm::StringRef intrinsic;
    VectorElemKind aElem;
    VectorElemKind cElem;

    static constexpr unsigned kWarpSize = 64;

    unsigned GetAElementCount() const { return (m * k) / kWarpSize; }

    unsigned GetAStorageElementCount() const {
        return GetAElementCount() * GetElementBitWidth(aElem) / 32;
    }

    unsigned GetCElementCount() const { return (m * n) / kWarpSize; }

    bool MatchesAType(mlir::Type type) const {
        return MatchesVectorType(type, VectorElemKind::I32,
                                 GetAStorageElementCount());
    }

    bool MatchesBType(mlir::Type type) const {
        return MatchesVectorType(type, VectorElemKind::I32,
                                 GetAStorageElementCount());
    }

    bool MatchesCType(mlir::Type type) const {
        return MatchesVectorType(type, cElem, GetCElementCount());
    }

    static llvm::ArrayRef<MFMAConfig> GetConfigs() {
        static const MFMAConfig kConfigs[] = {
            {
                "mfma_16x16x16_f16_f32",
                16,
                16,
                16,
                "f16",
                "f32",
                "rocdl_mfma_f32_16x16x16_f16",
                VectorElemKind::F16,
                VectorElemKind::F32,
            },
            {
                "mfma_16x16x16_bf16_f32",
                16,
                16,
                16,
                "bf16",
                "f32",
                "rocdl_mfma_f32_16x16x16bf16_1k",
                VectorElemKind::BF16,
                VectorElemKind::F32,
            },
            {
                "mfma_f32_16x16x16_bf16",
                16,
                16,
                16,
                "bf16",
                "f32",
                "rocdl_mfma_f32_16x16x16bf16_1k",
                VectorElemKind::BF16,
                VectorElemKind::F32,
            },
            {
                "mfma_32x32x8_bf16_f32",
                32,
                32,
                8,
                "bf16",
                "f32",
                "rocdl_mfma_f32_32x32x8bf16_1k",
                VectorElemKind::BF16,
                VectorElemKind::F32,
            },
            {
                "mfma_f32_32x32x8_bf16",
                32,
                32,
                8,
                "bf16",
                "f32",
                "rocdl_mfma_f32_32x32x8bf16_1k",
                VectorElemKind::BF16,
                VectorElemKind::F32,
            },
            {
                "mfma_16x16x32_fp8_fp8",
                16,
                16,
                32,
                "fp8",
                "f32",
                "rocdl_mfma_f32_16x16x32_fp8_fp8",
                VectorElemKind::FP8,
                VectorElemKind::F32,
            },
            {
                "mfma_f32_16x16x32_fp8_fp8",
                16,
                16,
                32,
                "fp8",
                "f32",
                "rocdl_mfma_f32_16x16x32_fp8_fp8",
                VectorElemKind::FP8,
                VectorElemKind::F32,
            },
        };

        return llvm::ArrayRef(kConfigs);
    }

    static const MFMAConfig *Find(int64_t m, int64_t n, int64_t k,
                                  llvm::StringRef typeA,
                                  llvm::StringRef typeC) {
        for (const auto &cfg : GetConfigs()) {
            if (cfg.m == m && cfg.n == n && cfg.k == k && cfg.typeA == typeA &&
                cfg.typeC == typeC) {
                return &cfg;
            }
        }
        return nullptr;
    }

  private:
    static unsigned GetElementBitWidth(VectorElemKind elem) {
        switch (elem) {
        case VectorElemKind::I32:
        case VectorElemKind::F32:
            return 32;
        case VectorElemKind::FP8:
            return 8;
        case VectorElemKind::F16:
        case VectorElemKind::BF16:
            return 16;
        }
        return 0;
    }

    static bool MatchesVectorType(mlir::Type type, VectorElemKind elem,
                                  int64_t elements) {
        auto vec = mlir::dyn_cast<mlir::VectorType>(type);
        if (!vec || vec.getRank() != 1 || vec.getNumElements() != elements) {
            return false;
        }

        auto elemType = vec.getElementType();
        switch (elem) {
        case VectorElemKind::I32:
            return elemType.isInteger(32);
        case VectorElemKind::FP8:
            return elemType.isInteger(8);
        case VectorElemKind::F16:
            return elemType.isF16();
        case VectorElemKind::F32:
            return elemType.isF32();
        case VectorElemKind::BF16:
            return elemType.isBF16();
        }
        return false;
    }
};

// F8/F6/F4 MFMA instructions with packed i32 inputs and E8M0 scales.
struct MFMAScaleConfig {
    llvm::StringRef name;
    int64_t m;
    int64_t n;
    int64_t k;
    llvm::StringRef rocdlOp;

    static constexpr unsigned kWarpSize = 64;

    unsigned GetCElementCount() const { return (m * n) / kWarpSize; }

    bool MatchesCType(mlir::Type type) const {
        auto vec = mlir::dyn_cast<mlir::VectorType>(type);
        return vec && vec.getRank() == 1 &&
               vec.getNumElements() == GetCElementCount() &&
               vec.getElementType().isF32();
    }

    static bool IsValidFormat(int64_t format) {
        return format >= 0 && format <= 4;
    }

    static bool IsValidScaleSelector(int64_t selector) {
        return selector >= 0 && selector <= 3;
    }

    static bool MatchesInputType(mlir::Type type, int64_t format) {
        auto vec = mlir::dyn_cast<mlir::VectorType>(type);
        if (!IsValidFormat(format) || !vec || vec.getRank() != 1 ||
            !vec.getElementType().isInteger(32)) {
            return false;
        }
        // LLVM permits excess registers beyond the selected format's minimum.
        constexpr int64_t minElements[] = {8, 8, 6, 6, 4};
        auto count = vec.getNumElements();
        return (count == 4 || count == 6 || count == 8) &&
               count >= minElements[format];
    }

    static llvm::ArrayRef<MFMAScaleConfig> GetConfigs() {
        static const MFMAScaleConfig kConfigs[] = {
            {"mfma_scale_f32_16x16x128_f8f6f4", 16, 16, 128,
             "rocdl.mfma.scale.f32.16x16x128.f8f6f4"},
            {"mfma_scale_f32_32x32x64_f8f6f4", 32, 32, 64,
             "rocdl.mfma.scale.f32.32x32x64.f8f6f4"},
        };
        return llvm::ArrayRef(kConfigs);
    }

    static const MFMAScaleConfig *Find(int64_t m, int64_t n, int64_t k) {
        for (const auto &config : GetConfigs()) {
            if (config.m == m && config.n == n && config.k == k) {
                return &config;
            }
        }
        return nullptr;
    }
};

} // namespace causalflow::avelang::amdgpu::mfma
