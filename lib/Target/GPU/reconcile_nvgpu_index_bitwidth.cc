#include "reconcile_nvgpu_index_bitwidth.h"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <optional>

namespace causalflow::avelang::target::gpu {

using namespace mlir;

namespace {

static Value convertLLVMIntegerPlain(Value value, IntegerType targetType,
                                     Location loc, PatternRewriter &rewriter) {
    if (value.getType() == targetType) {
        return value;
    }

    auto sourceType = dyn_cast<IntegerType>(value.getType());
    if (!sourceType) {
        return {};
    }

    unsigned sourceWidth = sourceType.getWidth();
    unsigned targetWidth = targetType.getWidth();
    if (sourceWidth == targetWidth) {
        return value;
    }
    if (sourceWidth > targetWidth) {
        return LLVM::TruncOp::create(rewriter, loc, targetType, value);
    }
    return LLVM::ZExtOp::create(rewriter, loc, targetType, value);
}

static bool isIntegerType(Type type, unsigned width) {
    auto intType = dyn_cast<IntegerType>(type);
    return intType && intType.getWidth() == width;
}

static std::optional<Value>
widenPointerArithmeticToI64(Value value, Location loc,
                            PatternRewriter &rewriter) {
    auto i64Type = rewriter.getI64Type();
    if (auto ptrToInt = value.getDefiningOp<LLVM::PtrToIntOp>()) {
        return LLVM::PtrToIntOp::create(rewriter, loc, i64Type,
                                        ptrToInt.getArg());
    }

    auto add = value.getDefiningOp<LLVM::AddOp>();
    if (!add || !isIntegerType(add.getResult().getType(), 32)) {
        return std::nullopt;
    }

    auto lhsPointer = widenPointerArithmeticToI64(add.getLhs(), loc, rewriter);
    auto rhsPointer = widenPointerArithmeticToI64(add.getRhs(), loc, rewriter);
    if (!lhsPointer && !rhsPointer) {
        return std::nullopt;
    }

    Value lhs = lhsPointer ? *lhsPointer
                           : convertLLVMIntegerPlain(add.getLhs(), i64Type, loc,
                                                     rewriter);
    Value rhs = rhsPointer ? *rhsPointer
                           : convertLLVMIntegerPlain(add.getRhs(), i64Type, loc,
                                                     rewriter);
    if (!lhs || !rhs) {
        return std::nullopt;
    }
    return LLVM::AddOp::create(rewriter, loc, i64Type, lhs, rhs);
}

static Value convertLLVMInteger(Value value, IntegerType targetType,
                                Location loc, PatternRewriter &rewriter) {
    auto sourceType = dyn_cast<IntegerType>(value.getType());
    if (sourceType && sourceType.getWidth() == 32 &&
        targetType.getWidth() == 64) {
        if (auto widenedPointer =
                widenPointerArithmeticToI64(value, loc, rewriter)) {
            return *widenedPointer;
        }
    }
    return convertLLVMIntegerPlain(value, targetType, loc, rewriter);
}

static bool isLLVMArrayOfInteger(Type type, unsigned width,
                                 int64_t *numElements = nullptr) {
    auto arrayType = dyn_cast<LLVM::LLVMArrayType>(type);
    if (!arrayType || !isIntegerType(arrayType.getElementType(), width)) {
        return false;
    }
    if (numElements) {
        *numElements = arrayType.getNumElements();
    }
    return true;
}

static std::optional<Value>
widenMemRefDescriptorMetadata(Value source, LLVM::LLVMStructType targetType,
                              Location loc, PatternRewriter &rewriter) {
    auto sourceType = dyn_cast<LLVM::LLVMStructType>(source.getType());
    if (!sourceType) {
        return std::nullopt;
    }

    auto sourceBody = sourceType.getBody();
    auto targetBody = targetType.getBody();
    if (sourceBody.size() != 5 || targetBody.size() != 5 ||
        sourceBody[0] != targetBody[0] || sourceBody[1] != targetBody[1] ||
        !isIntegerType(sourceBody[2], 32) ||
        !isIntegerType(targetBody[2], 64)) {
        return std::nullopt;
    }

    int64_t sourceRank = 0;
    int64_t targetRank = 0;
    if (!isLLVMArrayOfInteger(sourceBody[3], 32, &sourceRank) ||
        !isLLVMArrayOfInteger(sourceBody[4], 32) ||
        !isLLVMArrayOfInteger(targetBody[3], 64, &targetRank) ||
        !isLLVMArrayOfInteger(targetBody[4], 64) || sourceRank != targetRank) {
        return std::nullopt;
    }

    Value result = LLVM::PoisonOp::create(rewriter, loc, targetType);
    for (int64_t field : {0, 1}) {
        Value ptr = LLVM::ExtractValueOp::create(
            rewriter, loc, targetBody[field], source, ArrayRef<int64_t>{field});
        result = LLVM::InsertValueOp::create(rewriter, loc, targetType, result,
                                             ptr, ArrayRef<int64_t>{field});
    }

    Value offset = LLVM::ExtractValueOp::create(rewriter, loc, sourceBody[2],
                                                source, ArrayRef<int64_t>{2});
    offset = convertLLVMInteger(offset, cast<IntegerType>(targetBody[2]), loc,
                                rewriter);
    result = LLVM::InsertValueOp::create(rewriter, loc, targetType, result,
                                         offset, ArrayRef<int64_t>{2});

    auto targetElementType =
        cast<LLVM::LLVMArrayType>(targetBody[3]).getElementType();
    for (int64_t field : {3, 4}) {
        for (int64_t index = 0; index < sourceRank; ++index) {
            Value metadata = LLVM::ExtractValueOp::create(
                rewriter, loc,
                cast<LLVM::LLVMArrayType>(sourceBody[field]).getElementType(),
                source, ArrayRef<int64_t>{field, index});
            metadata = convertLLVMInteger(
                metadata, cast<IntegerType>(targetElementType), loc, rewriter);
            result = LLVM::InsertValueOp::create(
                rewriter, loc, targetType, result, metadata,
                ArrayRef<int64_t>{field, index});
        }
    }

    return result;
}

class ReconcileNVGPUIndex32CastsPattern
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  public:
    using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                  PatternRewriter &rewriter) const override {
        if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) {
            return failure();
        }

        Value input = op.getInputs()[0];
        Type outputType = op.getOutputs()[0].getType();

        if (auto targetInteger = dyn_cast<IntegerType>(outputType)) {
            auto indexCast = input.getDefiningOp<UnrealizedConversionCastOp>();
            if (!indexCast || indexCast.getInputs().size() != 1 ||
                indexCast.getOutputs().size() != 1 ||
                !indexCast.getOutputs()[0].getType().isIndex()) {
                return failure();
            }

            Value integerInput = indexCast.getInputs()[0];
            if (!isa<IntegerType>(integerInput.getType())) {
                return failure();
            }

            Value converted = convertLLVMInteger(integerInput, targetInteger,
                                                 op.getLoc(), rewriter);
            if (!converted) {
                return failure();
            }
            rewriter.replaceOp(op, converted);
            return success();
        }

        auto targetStruct = dyn_cast<LLVM::LLVMStructType>(outputType);
        if (!targetStruct) {
            return failure();
        }

        auto memrefCast = input.getDefiningOp<UnrealizedConversionCastOp>();
        if (!memrefCast || memrefCast.getInputs().size() != 1 ||
            memrefCast.getOutputs().size() != 1 ||
            !isa<MemRefType>(memrefCast.getOutputs()[0].getType())) {
            return failure();
        }

        auto widened = widenMemRefDescriptorMetadata(
            memrefCast.getInputs()[0], targetStruct, op.getLoc(), rewriter);
        if (!widened) {
            return failure();
        }
        rewriter.replaceOp(op, *widened);
        return success();
    }
};

template <typename ExtOp>
class WidenPointerArithmeticExtPattern : public OpRewritePattern<ExtOp> {
  public:
    using OpRewritePattern<ExtOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ExtOp op,
                                  PatternRewriter &rewriter) const override {
        if (!isIntegerType(op.getResult().getType(), 64) ||
            !isIntegerType(op.getArg().getType(), 32)) {
            return failure();
        }

        auto widened =
            widenPointerArithmeticToI64(op.getArg(), op.getLoc(), rewriter);
        if (!widened) {
            return failure();
        }
        rewriter.replaceOp(op, *widened);
        return success();
    }
};

class ReconcileNVGPUIndexBitwidthPass
    : public PassWrapper<ReconcileNVGPUIndexBitwidthPass,
                         OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        ReconcileNVGPUIndexBitwidthPass)

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<ReconcileNVGPUIndex32CastsPattern,
                     WidenPointerArithmeticExtPattern<LLVM::SExtOp>,
                     WidenPointerArithmeticExtPattern<LLVM::ZExtOp>>(
            &getContext());
        if (failed(
                applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }

    StringRef getArgument() const final {
        return "reconcile-nvgpu-index-bitwidth";
    }

    StringRef getDescription() const final {
        return "Bridge NVGPU i64 metadata expectations after 32-bit index "
               "lowering";
    }
};

} // namespace

std::unique_ptr<Pass> createReconcileNVGPUIndexBitwidthPass() {
    return std::make_unique<ReconcileNVGPUIndexBitwidthPass>();
}

} // namespace causalflow::avelang::target::gpu
