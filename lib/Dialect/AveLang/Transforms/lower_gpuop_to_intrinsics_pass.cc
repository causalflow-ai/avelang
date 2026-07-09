#include "lower_gpuop_to_intrinsics_pass.h"
#include "AveLangOps.h"
#include "IR/Intrinsics/amdgpu_mfma_signatures.h"
#include "IR/Intrinsics/intrinsic_support.h"
#include "Utils/assert.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/NVVMDialect.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace causalflow::avelang::dialect {
namespace amdgpu_mfma = causalflow::avelang::amdgpu::mfma;

namespace {

/// Helper to convert a memref to an aligned pointer index with optional bounds
/// checking. Returns a null Value on failure and emits a diagnostic.
mlir::Value convertMemrefToPointerIndex(mlir::PatternRewriter &rewriter,
                                        mlir::Location loc, mlir::Value memref,
                                        llvm::StringRef opName) {
    auto getStaticStrides =
        [&](MemRefType type, llvm::SmallVectorImpl<int64_t> &strides) -> bool {
        auto layoutType = type.getLayoutType();
        auto layoutStrides = layoutType.getStrides();
        if (!layoutStrides.empty()) {
            for (auto stride : layoutStrides) {
                if (stride == mlir::ShapedType::kDynamic) {
                    return false;
                }
                strides.push_back(stride);
            }
            return true;
        }

        auto shape = layoutType.getDims();
        strides.assign(shape.size(), 1);
        int64_t running = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            if (shape[i] == mlir::ShapedType::kDynamic) {
                return false;
            }
            strides[i] = running;
            running *= shape[i];
        }
        return true;
    };

    auto computePointerIndex = [&](mlir::Value value,
                                   auto &&self) -> mlir::Value {
        if (auto subviewOp = value.getDefiningOp<AveLangMemRefSubViewOp>()) {
            auto base = subviewOp.getSource();
            auto basePtrIndex = self(base, self);
            if (!basePtrIndex) {
                return nullptr;
            }

            auto baseAveLangType = mlir::dyn_cast<MemRefType>(base.getType());
            if (!baseAveLangType) {
                mlir::emitError(loc)
                    << opName << ": Expected memref type for subview";
                return nullptr;
            }

            llvm::SmallVector<mlir::Value> strideValues;
            llvm::SmallVector<int64_t> staticStrides;
            if (!getStaticStrides(baseAveLangType, staticStrides)) {
                mlir::emitError(loc)
                    << opName
                    << ": Cannot compute static strides for subview base";
                return nullptr;
            }
            for (auto stride : staticStrides) {
                strideValues.push_back(mlir::arith::ConstantIndexOp::create(
                    rewriter, loc, stride));
            }
            auto offsets = subviewOp.getOffsets();
            if (offsets.size() > strideValues.size()) {
                mlir::emitError(loc)
                    << opName << ": Subview offsets exceed base memref rank";
                return nullptr;
            }

            mlir::Value totalOffset =
                mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
            for (size_t i = 0; i < offsets.size(); ++i) {
                mlir::Value offsetVal = offsets[i];
                if (!offsetVal.getType().isIndex()) {
                    offsetVal =
                        mlir::arith::IndexCastOp::create(
                            rewriter, loc, rewriter.getIndexType(), offsetVal)
                            .getResult();
                }
                mlir::Value strideVal = strideValues[i];
                if (!strideVal.getType().isIndex()) {
                    strideVal =
                        mlir::arith::IndexCastOp::create(
                            rewriter, loc, rewriter.getIndexType(), strideVal)
                            .getResult();
                }
                auto prod = mlir::arith::MulIOp::create(rewriter, loc,
                                                        offsetVal, strideVal);
                totalOffset = mlir::arith::AddIOp::create(
                                  rewriter, loc, totalOffset, prod.getResult())
                                  .getResult();
            }

            auto baseElemType = baseAveLangType.getElementType();
            int64_t elemBitwidth = baseElemType.getIntOrFloatBitWidth();
            if (elemBitwidth == 0) {
                mlir::emitError(loc)
                    << opName << ": Unsupported element type for pointer math";
                return nullptr;
            }
            auto byteNum = mlir::arith::ConstantIndexOp::create(
                rewriter, loc, (elemBitwidth >> 3));
            auto offsetBytes = mlir::arith::MulIOp::create(
                rewriter, loc, totalOffset, byteNum);

            return mlir::arith::AddIOp::create(rewriter, loc, basePtrIndex,
                                               offsetBytes.getResult())
                .getResult();
        }

        if (mlir::isa<MemRefType>(value.getType())) {
            return AveLangMemRefExtractAlignedPointerAsIndexOp::create(
                       rewriter, loc, rewriter.getIndexType(), value)
                .getResult();
        }

        mlir::emitError(loc) << opName << ": Expected memref type";
        return nullptr;
    };

    return computePointerIndex(memref, computePointerIndex);
}

std::string buildIntrinsicFuncName(
    llvm::StringRef dialect,
    llvm::function_ref<void(llvm::raw_svector_ostream &)> nameBuilder) {
    llvm::SmallString<32> intrinsicName;
    llvm::raw_svector_ostream nameStream(intrinsicName);
    nameBuilder(nameStream);
    return ir::intrinsics::MakeIntrinsicFuncName(
        dialect, std::string_view(intrinsicName.data(), intrinsicName.size()));
}

enum class VectorElemKind {
    I32,
    F16,
    F32,
    BF16,
};

struct VectorTypePattern {
    VectorElemKind elem;
    int64_t elements;
};

static bool matchesVectorType(mlir::Type type,
                              const VectorTypePattern &pattern) {
    auto vec = mlir::dyn_cast<mlir::VectorType>(type);
    if (!vec || vec.getRank() != 1 ||
        vec.getNumElements() != pattern.elements) {
        return false;
    }

    auto elemType = vec.getElementType();
    switch (pattern.elem) {
    case VectorElemKind::I32:
        return elemType.isInteger(32);
    case VectorElemKind::F16:
        return elemType.isF16();
    case VectorElemKind::F32:
        return elemType.isF32();
    case VectorElemKind::BF16:
        return elemType.isBF16();
    }
    return false;
}

static bool matchesAnyVectorType(mlir::Type type,
                                 llvm::ArrayRef<VectorTypePattern> patterns) {
    for (const auto &pattern : patterns) {
        if (matchesVectorType(type, pattern)) {
            return true;
        }
    }
    return false;
}

static mlir::Type getElemType(VectorElemKind kind, mlir::OpBuilder &builder) {
    switch (kind) {
    case VectorElemKind::I32:
        return builder.getI32Type();
    case VectorElemKind::F16:
        return builder.getF16Type();
    case VectorElemKind::F32:
        return builder.getF32Type();
    case VectorElemKind::BF16:
        return builder.getBF16Type();
    }
    return mlir::Type();
}

static mlir::Type getMfmaElemType(amdgpu_mfma::VectorElemKind kind,
                                  mlir::OpBuilder &builder) {
    switch (kind) {
    case amdgpu_mfma::VectorElemKind::I32:
        return builder.getI32Type();
    case amdgpu_mfma::VectorElemKind::FP8:
        return builder.getI8Type();
    case amdgpu_mfma::VectorElemKind::F16:
        return builder.getF16Type();
    case amdgpu_mfma::VectorElemKind::F32:
        return builder.getF32Type();
    case amdgpu_mfma::VectorElemKind::BF16:
        return builder.getBF16Type();
    }
    return mlir::Type();
}

struct NvvMmaSignature {
    llvm::StringRef intrinsic;
    llvm::ArrayRef<VectorTypePattern> aTypes;
    llvm::ArrayRef<VectorTypePattern> bTypes;
    llvm::ArrayRef<VectorTypePattern> cTypes;
    VectorElemKind aElem;
    int64_t aCount;
    VectorElemKind bElem;
    int64_t bCount;
    VectorElemKind cElem;
    int64_t cCount;
};

static const VectorTypePattern kNvvmMma16x8x16A[] = {
    {VectorElemKind::I32, 4},
    {VectorElemKind::F16, 8},
};
static const VectorTypePattern kNvvmMma16x8x16B[] = {
    {VectorElemKind::I32, 2},
    {VectorElemKind::F16, 4},
};
static const VectorTypePattern kNvvmMma16x8x16C[] = {
    {VectorElemKind::F16, 4},
};

static const VectorTypePattern kNvvmMma16x8x8A[] = {
    {VectorElemKind::I32, 2},
    {VectorElemKind::F16, 4},
};
static const VectorTypePattern kNvvmMma16x8x8B[] = {
    {VectorElemKind::I32, 1},
    {VectorElemKind::F16, 2},
};
static const VectorTypePattern kNvvmMma16x8x8C[] = {
    {VectorElemKind::F32, 4},
};

static const NvvMmaSignature kNvvmMmaSignatures[] = {
    {
        "mma_16x8x16_f16_f16",
        llvm::ArrayRef(kNvvmMma16x8x16A),
        llvm::ArrayRef(kNvvmMma16x8x16B),
        llvm::ArrayRef(kNvvmMma16x8x16C),
        VectorElemKind::F16,
        8,
        VectorElemKind::F16,
        4,
        VectorElemKind::F16,
        4,
    },
    {
        "mma_16x8x8_f16_f32",
        llvm::ArrayRef(kNvvmMma16x8x8A),
        llvm::ArrayRef(kNvvmMma16x8x8B),
        llvm::ArrayRef(kNvvmMma16x8x8C),
        VectorElemKind::F16,
        4,
        VectorElemKind::F16,
        2,
        VectorElemKind::F32,
        4,
    },
};

static const NvvMmaSignature *findNvvmMmaSignature(mlir::Value a, mlir::Value b,
                                                   mlir::Value c) {
    for (const auto &sig : kNvvmMmaSignatures) {
        if (matchesAnyVectorType(a.getType(), sig.aTypes) &&
            matchesAnyVectorType(b.getType(), sig.bTypes) &&
            matchesAnyVectorType(c.getType(), sig.cTypes)) {
            return &sig;
        }
    }
    return nullptr;
}

class NVVMMmaLowering : public mlir::OpRewritePattern<NVVMMMAOp> {
  public:
    using mlir::OpRewritePattern<NVVMMMAOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(NVVMMMAOp op,
                    mlir::PatternRewriter &rewriter) const override {
        auto a = op.getA();
        auto b = op.getB();
        auto c = op.getC();
        auto resultType = op.getResult().getType();

        auto *signature = findNvvmMmaSignature(a, b, c);
        if (!signature) {
            return mlir::failure();
        }

        if (!mlir::dyn_cast<mlir::VectorType>(a.getType()) ||
            !mlir::dyn_cast<mlir::VectorType>(b.getType()) ||
            !mlir::dyn_cast<mlir::VectorType>(c.getType())) {
            return mlir::failure();
        }

        auto prepareArg = [&](mlir::Value vecVal, mlir::Type desiredElem,
                              int64_t desiredCount) -> mlir::Value {
            auto expectedVecType =
                mlir::VectorType::get(desiredCount, desiredElem);

            if (vecVal.getType() == expectedVecType) {
                return vecVal;
            }

            return mlir::vector::BitCastOp::create(rewriter, op.getLoc(),
                                                   expectedVecType, vecVal);
        };

        auto aElemType = getElemType(signature->aElem, rewriter);
        auto bElemType = getElemType(signature->bElem, rewriter);
        auto cElemType = getElemType(signature->cElem, rewriter);

        auto aVec = prepareArg(a, aElemType, signature->aCount);
        auto bVec = prepareArg(b, bElemType, signature->bCount);
        auto cVec = prepareArg(c, cElemType, signature->cCount);

        auto funcName =
            ir::intrinsics::MakeIntrinsicFuncName("nvvm", signature->intrinsic);

        auto callOp = mlir::func::CallOp::create(
            rewriter, op.getLoc(), funcName, mlir::TypeRange{resultType},
            mlir::ValueRange{aVec, bVec, cVec});

        rewriter.replaceOp(op, callOp.getResult(0));
        return mlir::success();
    }
};

class NVVMLdMatrixLowering : public mlir::OpRewritePattern<NVVMLdMatrixOp> {
  public:
    using mlir::OpRewritePattern<NVVMLdMatrixOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(NVVMLdMatrixOp op,
                    mlir::PatternRewriter &rewriter) const override {
        auto memref = op.getMemref();
        auto ptrIndex = convertMemrefToPointerIndex(rewriter, op.getLoc(),
                                                    memref, "nvvm_ldmatrix");
        if (!ptrIndex) {
            return mlir::failure();
        }

        llvm::StringRef shape = op.getMatrixShapeAttr().getValue();
        auto num = op.getMatrixNum();
        auto bitWidth = op.getMatrixBitWidth();

        auto funcName = buildIntrinsicFuncName(
            "nvvm", [&](llvm::raw_svector_ostream &nameStream) {
                nameStream << "ldmatrix_" << shape << "_x" << num << "_b"
                           << bitWidth;
            });

        mlir::Type resultType;
        if (num == 1) {
            resultType = rewriter.getI32Type();
        } else {
            llvm::SmallVector<mlir::Type> structElements(num,
                                                         rewriter.getI32Type());
            resultType = mlir::LLVM::LLVMStructType::getLiteral(
                rewriter.getContext(), structElements);
        }

        auto callOp = mlir::func::CallOp::create(
            rewriter, op.getLoc(), funcName, mlir::TypeRange{resultType},
            mlir::ValueRange{ptrIndex});

        auto ldmatrixResult = callOp.getResult(0);

        if (num == 1) {
            rewriter.replaceOp(op, ldmatrixResult);
            return mlir::success();
        }

        llvm::SmallVector<mlir::Value> results;
        for (unsigned i = 0; i < num; ++i) {
            mlir::Value extractedValue = mlir::LLVM::ExtractValueOp::create(
                rewriter, op.getLoc(), rewriter.getI32Type(), ldmatrixResult,
                llvm::ArrayRef<int64_t>{static_cast<int64_t>(i)});
            results.push_back(extractedValue);
        }

        auto vecType = mlir::VectorType::get(num, rewriter.getI32Type());
        mlir::Value vec = mlir::vector::FromElementsOp::create(
            rewriter, op.getLoc(), vecType, results);

        rewriter.replaceOp(op, vec);
        return mlir::success();
    }
};

class NVVMStMatrixLowering : public mlir::OpRewritePattern<NVVMStMatrixOp> {
  public:
    using mlir::OpRewritePattern<NVVMStMatrixOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(NVVMStMatrixOp op,
                    mlir::PatternRewriter &rewriter) const override {
        auto value = op.getValue();
        auto memref = op.getMemref();

        auto ptrIndex = convertMemrefToPointerIndex(rewriter, op.getLoc(),
                                                    memref, "nvvm_stmatrix");
        if (!ptrIndex) {
            return mlir::failure();
        }

        llvm::StringRef shape = op.getMatrixShapeAttr().getValue();
        auto num = op.getMatrixNum();
        auto bitWidth = op.getMatrixBitWidth();

        auto funcName = buildIntrinsicFuncName(
            "nvvm", [&](llvm::raw_svector_ostream &nameStream) {
                nameStream << "stmatrix_" << shape << "_x" << num << "_b"
                           << bitWidth;
            });

        mlir::func::CallOp::create(rewriter, op.getLoc(), funcName,
                                   mlir::TypeRange(),
                                   mlir::ValueRange{value, ptrIndex});
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

class AMDGPUMfmaLowering : public mlir::OpRewritePattern<AMDGPUMfmaOp> {
  public:
    using mlir::OpRewritePattern<AMDGPUMfmaOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AMDGPUMfmaOp op,
                    mlir::PatternRewriter &rewriter) const override {
        auto a = op.getA();
        auto b = op.getB();
        auto c = op.getC();
        auto resultType = op.getResult().getType();

        int m = op.getMAttr().getInt();
        int n = op.getNAttr().getInt();
        int k = op.getKAttr().getInt();
        std::string typeA = op.getTypeAAttr().str();
        std::string typeC = op.getTypeCAttr().str();

        auto *config = amdgpu_mfma::MFMAConfig::Find(m, n, k, typeA, typeC);
        if (!config) {
            return mlir::failure();
        }

        if (!mlir::dyn_cast<mlir::VectorType>(a.getType()) ||
            !mlir::dyn_cast<mlir::VectorType>(b.getType()) ||
            !mlir::dyn_cast<mlir::VectorType>(c.getType())) {
            return mlir::failure();
        }

        if (!config->MatchesAType(a.getType()) ||
            !config->MatchesBType(b.getType()) ||
            !config->MatchesCType(c.getType())) {
            return mlir::failure();
        }

        auto desiredAElem = rewriter.getI32Type();
        auto desiredBElem = desiredAElem;
        auto desiredCElem = getMfmaElemType(config->cElem, rewriter);
        int64_t desiredACount = config->GetAStorageElementCount();
        int64_t desiredBCount = desiredACount;
        int64_t desiredCCount = config->GetCElementCount();

        auto prepareArg = [&](mlir::Value vecVal, mlir::Type desiredElem,
                              int64_t desiredCount) -> mlir::Value {
            auto expectedVecType =
                mlir::VectorType::get(desiredCount, desiredElem);

            if (vecVal.getType() == expectedVecType) {
                return vecVal;
            }

            return mlir::vector::BitCastOp::create(rewriter, op.getLoc(),
                                                   expectedVecType, vecVal);
        };

        auto aVec = prepareArg(a, desiredAElem, desiredACount);
        auto bVec = prepareArg(b, desiredBElem, desiredBCount);
        auto cVec = prepareArg(c, desiredCElem, desiredCCount);

        auto funcName =
            ir::intrinsics::MakeIntrinsicFuncName("amdgpu", config->intrinsic);

        auto callOp = mlir::func::CallOp::create(
            rewriter, op.getLoc(), funcName, mlir::TypeRange{resultType},
            mlir::ValueRange{aVec, bVec, cVec});

        rewriter.replaceOp(op, callOp.getResult(0));
        return mlir::success();
    }
};

class AMDGPURawBufferLoadLowering
    : public mlir::OpRewritePattern<AMDGPURawBufferLoadOp> {
  public:
    using mlir::OpRewritePattern<AMDGPURawBufferLoadOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AMDGPURawBufferLoadOp op,
                    mlir::PatternRewriter &rewriter) const override {
        auto rsrc = op.getRsrc();
        auto vindex = op.getVindex();
        auto soffset = op.getSoffset();
        auto aux = op.getAux();
        auto resultType = op.getResult().getType();

        auto loadOp = mlir::ROCDL::RawBufferLoadOp::create(
            rewriter, op.getLoc(), resultType, rsrc, vindex, soffset, aux);
        rewriter.replaceOp(op, loadOp.getResult());
        return mlir::success();
    }
};

class AMDGPURawBufferStoreLowering
    : public mlir::OpRewritePattern<AMDGPURawBufferStoreOp> {
  public:
    using mlir::OpRewritePattern<AMDGPURawBufferStoreOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AMDGPURawBufferStoreOp op,
                    mlir::PatternRewriter &rewriter) const override {
        auto data = op.getData();
        auto rsrc = op.getRsrc();
        auto vindex = op.getVindex();
        auto soffset = op.getSoffset();
        auto aux = op.getAux();

        mlir::ROCDL::RawBufferStoreOp::create(rewriter, op.getLoc(), data, rsrc,
                                              vindex, soffset, aux);
        rewriter.eraseOp(op);
        return mlir::success();
    }
};

/// Pass that lowers GPUOp dialect operations to intrinsic function calls.
class LowerAveLangGPUToIntrinsicsPass
    : public mlir::PassWrapper<LowerAveLangGPUToIntrinsicsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        LowerAveLangGPUToIntrinsicsPass)

    llvm::StringRef getArgument() const final {
        return "lower-gpuop-to-intrinsics";
    }

    llvm::StringRef getDescription() const final {
        return "Lower GPUOp dialect operations to intrinsic function calls";
    }

    void runOnOperation() override {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<NVVMMmaLowering, NVVMLdMatrixLowering,
                     NVVMStMatrixLowering, AMDGPUMfmaLowering,
                     AMDGPURawBufferLoadLowering, AMDGPURawBufferStoreLowering>(
            &getContext());

        if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                     std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerAveLangGPUToIntrinsicsPass() {
    return std::make_unique<LowerAveLangGPUToIntrinsicsPass>();
}

} // namespace causalflow::avelang::dialect
