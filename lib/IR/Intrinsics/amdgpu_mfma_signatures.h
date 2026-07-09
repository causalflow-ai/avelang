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

} // namespace causalflow::avelang::amdgpu::mfma
