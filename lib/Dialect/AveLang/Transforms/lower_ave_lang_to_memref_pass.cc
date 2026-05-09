#include "lower_ave_lang_to_memref_pass.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "IR/constant_folder.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/MathExtras.h>
#include <numeric>
#include <optional>

namespace causalflow::avelang::dialect {

namespace {

bool isStaticValue(mlir::Value value) {
    return ir::ConstantFolder::FoldIntValue(value).has_value();
}

std::optional<int64_t> getStaticValue(mlir::Value value) {
    return ir::ConstantFolder::FoldIntValue(value);
}

mlir::Attribute normalizeBuiltinMemorySpace(mlir::MLIRContext *context,
                                            mlir::Attribute memorySpace) {
    if (auto gpuSpace =
            mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(memorySpace)) {
        switch (gpuSpace.getValue()) {
        case mlir::gpu::AddressSpace::Private:
            return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                          5);
        case mlir::gpu::AddressSpace::Global:
            return memorySpace;
        case mlir::gpu::AddressSpace::Workgroup:
            // Keep workgroup memory space in integer form to satisfy NVVM
            // memref-to-LLVM conversions and avoid mixed-address-space views.
            return mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                          3);
        }
    }
    return memorySpace;
}

mlir::Attribute resolveMemorySpace(mlir::MLIRContext *context,
                                   mlir::Attribute preferred,
                                   mlir::Attribute fallback = {}) {
    auto normalizedPreferred = normalizeBuiltinMemorySpace(context, preferred);
    if (normalizedPreferred) {
        return normalizedPreferred;
    }
    return normalizeBuiltinMemorySpace(context, fallback);
}

bool isWorkgroupMemorySpaceAttr(mlir::Attribute memorySpace) {
    auto intSpace = mlir::dyn_cast_or_null<mlir::IntegerAttr>(memorySpace);
    return intSpace && intSpace.getInt() == 3;
}

mlir::MemRefType withResolvedMemorySpace(mlir::MLIRContext *context,
                                         mlir::MemRefType type,
                                         mlir::Attribute fallback = {}) {
    return mlir::MemRefType::get(
        type.getShape(), type.getElementType(), type.getLayout(),
        resolveMemorySpace(context, type.getMemorySpace(), fallback));
}

mlir::MemRefType getBuiltinMemRefType(mlir::Type type) {
    if (auto builtinType = mlir::dyn_cast<mlir::MemRefType>(type)) {
        return withResolvedMemorySpace(type.getContext(), builtinType);
    }

    auto substrateType = mlir::dyn_cast<MemRefType>(type);
    if (!substrateType) {
        return {};
    }

    mlir::MemRefLayoutAttrInterface layout;
    auto layoutType = mlir::cast<LayoutType>(substrateType.getLayout());
    auto strides = layoutType.getStrides();
    if (!strides.empty()) {
        layout = mlir::StridedLayoutAttr::get(type.getContext(),
                                              /*offset=*/0, strides);
    }

    return mlir::MemRefType::get(
        layoutType.getDims(), substrateType.getElementType(), layout,
        normalizeBuiltinMemorySpace(type.getContext(),
                                    substrateType.getMemorySpace()));
}

mlir::Value castToBuiltinMemRef(mlir::Location loc, mlir::Value value,
                                mlir::PatternRewriter &rewriter) {
    auto builtinType = getBuiltinMemRefType(value.getType());
    if (!builtinType) {
        return {};
    }
    if (value.getType() == builtinType) {
        return value;
    }
    return rewriter
        .create<AveLangMemRefCastOp>(loc, value, builtinType)
        .getResult();
}

std::optional<int64_t> getElementByteSize(mlir::Type elementType) {
    if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(elementType)) {
        auto scalarType = vectorType.getElementType();
        if (scalarType.isIndex()) {
            return 8 * vectorType.getNumElements();
        }
        if (!scalarType.isIntOrFloat()) {
            return std::nullopt;
        }
        int64_t bitWidth = scalarType.getIntOrFloatBitWidth();
        int64_t elementBytes = (bitWidth + 7) / 8;
        if (elementBytes <= 0) {
            return std::nullopt;
        }
        return elementBytes * vectorType.getNumElements();
    }

    if (elementType.isIndex()) {
        return 8;
    }

    if (!elementType.isIntOrFloat()) {
        return std::nullopt;
    }
    int64_t bitWidth = elementType.getIntOrFloatBitWidth();
    int64_t elementBytes = (bitWidth + 7) / 8;
    if (elementBytes <= 0) {
        return std::nullopt;
    }
    return elementBytes;
}

std::optional<int64_t> getPreferredAlignment(mlir::Type elementType) {
    auto elementBytes = getElementByteSize(elementType);
    if (!elementBytes || *elementBytes <= 0) {
        return std::nullopt;
    }

    return static_cast<int64_t>(
        llvm::PowerOf2Ceil(static_cast<uint64_t>(*elementBytes)));
}

std::optional<int64_t> getPreferredAllocaAlignment(mlir::MemRefType type) {
    auto preferred = getPreferredAlignment(type.getElementType());
    if (!isWorkgroupMemorySpaceAttr(type.getMemorySpace())) {
        return preferred;
    }

    // WGMMA shared-memory descriptors are sensitive to base alignment.
    int64_t minWorkgroupAlignment = 128;
    if (!preferred || *preferred < minWorkgroupAlignment) {
        return minWorkgroupAlignment;
    }
    return preferred;
}

std::optional<int64_t> getStaticElementCount(mlir::MemRefType type) {
    int64_t count = 1;
    for (auto dim : type.getShape()) {
        if (mlir::ShapedType::isDynamic(dim) || dim < 0) {
            return std::nullopt;
        }
        count *= dim;
    }
    return count;
}

std::optional<int64_t> getStaticByteSize(mlir::MemRefType type) {
    auto elemBytes = getElementByteSize(type.getElementType());
    auto elemCount = getStaticElementCount(type);
    if (!elemBytes || !elemCount) {
        return std::nullopt;
    }
    return (*elemBytes) * (*elemCount);
}

bool extractTupleValues(mlir::Value tupleValue,
                        llvm::SmallVector<mlir::Value> &values) {
    if (auto tupleOp = tupleValue.getDefiningOp<MakeIntTupleOp>()) {
        for (auto elem : tupleOp.getElements()) {
            if (!extractTupleValues(elem, values)) {
                return false;
            }
        }
        return true;
    }

    values.push_back(tupleValue);
    return true;
}

mlir::AffineMap
buildSemiAffineMap(const llvm::SmallVector<mlir::Value> &shapeValues,
                   const llvm::SmallVector<mlir::Value> &strideValues,
                   mlir::PatternRewriter &rewriter) {
    if (shapeValues.size() != strideValues.size()) {
        return mlir::AffineMap();
    }

    unsigned numDims = shapeValues.size();

    llvm::SmallVector<mlir::Value> symbols;
    llvm::DenseMap<mlir::Value, unsigned> symbolMap;
    for (auto stride : strideValues) {
        if (!isStaticValue(stride) && !symbolMap.count(stride)) {
            symbolMap[stride] = symbols.size();
            symbols.push_back(stride);
        }
    }

    unsigned numSymbols = symbols.size();
    llvm::SmallVector<mlir::AffineExpr> dimExprs;
    for (unsigned i = 0; i < numDims; ++i) {
        dimExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
    }

    mlir::AffineExpr indexExpr =
        mlir::getAffineConstantExpr(0, rewriter.getContext());
    for (unsigned i = 0; i < numDims; ++i) {
        mlir::AffineExpr strideExpr;
        if (auto staticStride = getStaticValue(strideValues[i])) {
            strideExpr = mlir::getAffineConstantExpr(*staticStride,
                                                     rewriter.getContext());
        } else {
            unsigned symbolIdx = symbolMap[strideValues[i]];
            strideExpr =
                mlir::getAffineSymbolExpr(symbolIdx, rewriter.getContext());
        }
        indexExpr = indexExpr + dimExprs[i] * strideExpr;
    }

    return mlir::AffineMap::get(numDims, numSymbols, indexExpr,
                                rewriter.getContext());
}

void addDebugAttributes(mlir::Operation *op,
                        const llvm::SmallVector<mlir::Value> &shapeValues,
                        const llvm::SmallVector<mlir::Value> &strideValues,
                        mlir::PatternRewriter &rewriter) {
    llvm::SmallVector<mlir::Attribute> shapeAttrs;
    for (auto shape : shapeValues) {
        if (auto staticVal = getStaticValue(shape)) {
            shapeAttrs.push_back(rewriter.getI64IntegerAttr(*staticVal));
        } else {
            shapeAttrs.push_back(rewriter.getStringAttr("?"));
        }
    }
    op->setAttr("shape_pattern", rewriter.getArrayAttr(shapeAttrs));

    llvm::SmallVector<mlir::Attribute> strideAttrs;
    for (auto stride : strideValues) {
        if (auto staticVal = getStaticValue(stride)) {
            strideAttrs.push_back(rewriter.getI64IntegerAttr(*staticVal));
        } else {
            strideAttrs.push_back(rewriter.getStringAttr("?"));
        }
    }
    op->setAttr("stride_pattern", rewriter.getArrayAttr(strideAttrs));
}

mlir::Value materializeIndexValue(mlir::OpBuilder &rewriter, mlir::Location loc,
                                  mlir::OpFoldResult value) {
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(value)) {
        auto intAttr = mlir::cast<mlir::IntegerAttr>(attr);
        return mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                    intAttr.getInt());
    }
    return mlir::cast<mlir::Value>(value);
}

struct BaseMemRefInfo {
    mlir::Value base;
    mlir::Value byteShift;
};

std::optional<BaseMemRefInfo> findBaseI8Memref(mlir::Value value,
                                               mlir::OpBuilder &rewriter) {
    mlir::Location loc = value.getLoc();
    auto makeZero = [&]() {
        return mlir::arith::ConstantIndexOp::create(rewriter, loc, 0)
            .getResult();
    };

    auto accumulateSubviewByteShift =
        [&](mlir::Value source, llvm::ArrayRef<mlir::OpFoldResult> mixedOffsets)
        -> std::optional<mlir::Value> {
        auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
        if (!sourceType) {
            return std::nullopt;
        }

        auto elemBytes = getElementByteSize(sourceType.getElementType());
        if (!elemBytes) {
            return std::nullopt;
        }

        int64_t sourceOffset = 0;
        llvm::SmallVector<int64_t> sourceStrides;
        if (failed(
                sourceType.getStridesAndOffset(sourceStrides, sourceOffset))) {
            return std::nullopt;
        }

        if (mixedOffsets.size() > sourceStrides.size()) {
            return std::nullopt;
        }

        auto shift =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, 0).getResult();
        for (auto [dim, stride] : llvm::enumerate(sourceStrides)) {
            if (mlir::ShapedType::isDynamic(stride)) {
                return std::nullopt;
            }

            mlir::OpFoldResult offset = rewriter.getIndexAttr(0);
            if (dim < mixedOffsets.size()) {
                offset = mixedOffsets[dim];
            }

            auto offsetValue = materializeIndexValue(rewriter, loc, offset);
            if (stride != 1) {
                auto strideValue =
                    mlir::arith::ConstantIndexOp::create(rewriter, loc, stride);
                offsetValue = mlir::arith::MulIOp::create(
                    rewriter, loc, offsetValue, strideValue);
            }
            if (*elemBytes != 1) {
                auto byteValue = mlir::arith::ConstantIndexOp::create(
                    rewriter, loc, *elemBytes);
                offsetValue = mlir::arith::MulIOp::create(
                    rewriter, loc, offsetValue, byteValue);
            }
            shift =
                mlir::arith::AddIOp::create(rewriter, loc, shift, offsetValue);
        }
        return shift;
    };

    mlir::Value current = value;
    while (true) {
        auto memrefType = mlir::dyn_cast<mlir::MemRefType>(current.getType());
        if (!memrefType) {
            return std::nullopt;
        }

        if (memrefType.getElementType().isInteger(8)) {
            return BaseMemRefInfo{current, makeZero()};
        }

        if (auto viewOp = current.getDefiningOp<mlir::memref::ViewOp>()) {
            auto source = viewOp.getSource();
            auto sourceType =
                mlir::dyn_cast<mlir::MemRefType>(source.getType());
            if (sourceType && sourceType.getElementType().isInteger(8)) {
                return BaseMemRefInfo{source, viewOp.getByteShift()};
            }
            current = source;
            continue;
        }

        if (auto reinterpretOp =
                current.getDefiningOp<mlir::memref::ReinterpretCastOp>()) {
            auto source = reinterpretOp.getSource();
            auto sourceType =
                mlir::dyn_cast<mlir::MemRefType>(source.getType());
            if (sourceType && sourceType.getElementType().isInteger(8)) {
                auto offsets = reinterpretOp.getMixedOffsets();
                if (offsets.empty()) {
                    return std::nullopt;
                }
                auto offset = offsets.front();
                return BaseMemRefInfo{
                    source, materializeIndexValue(rewriter, loc, offset)};
            }
            current = source;
            continue;
        }

        if (auto subviewOp = current.getDefiningOp<mlir::memref::SubViewOp>()) {
            auto sourceBase = findBaseI8Memref(subviewOp.getSource(), rewriter);
            if (!sourceBase) {
                return std::nullopt;
            }
            auto localShift = accumulateSubviewByteShift(
                subviewOp.getSource(), subviewOp.getMixedOffsets());
            if (!localShift) {
                return std::nullopt;
            }
            auto totalShift = mlir::arith::AddIOp::create(
                rewriter, loc, sourceBase->byteShift, *localShift);
            return BaseMemRefInfo{sourceBase->base, totalShift};
        }

        if (auto subviewOp = current.getDefiningOp<AveLangMemRefSubViewOp>()) {
            llvm::SmallVector<mlir::OpFoldResult> mixedOffsets;
            mixedOffsets.reserve(subviewOp.getOffsets().size());
            for (auto offset : subviewOp.getOffsets()) {
                if (auto constVal = mlir::getConstantIntValue(offset)) {
                    mixedOffsets.push_back(rewriter.getIndexAttr(*constVal));
                } else {
                    mixedOffsets.push_back(offset);
                }
            }

            auto sourceBase = findBaseI8Memref(subviewOp.getSource(), rewriter);
            if (!sourceBase) {
                return std::nullopt;
            }
            auto localShift =
                accumulateSubviewByteShift(subviewOp.getSource(), mixedOffsets);
            if (!localShift) {
                return std::nullopt;
            }
            auto totalShift = mlir::arith::AddIOp::create(
                rewriter, loc, sourceBase->byteShift, *localShift);
            return BaseMemRefInfo{sourceBase->base, totalShift};
        }

        if (auto castOp = current.getDefiningOp<mlir::memref::CastOp>()) {
            current = castOp.getSource();
            continue;
        }

        if (auto castOp = current.getDefiningOp<AveLangMemRefCastOp>()) {
            current = castOp.getSource();
            continue;
        }

        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(current)) {
            if (memrefType.getElementType().isInteger(8)) {
                return BaseMemRefInfo{blockArg, makeZero()};
            }
        }

        return std::nullopt;
    }
}

mlir::LogicalResult lowerFunctionArgsToI8(mlir::ModuleOp module) {
    auto *context = module.getContext();
    auto i8Type = mlir::IntegerType::get(context, 8);

    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        auto funcType = func.getFunctionType();
        llvm::SmallVector<mlir::Type> newInputs(funcType.getInputs().begin(),
                                                funcType.getInputs().end());
        llvm::SmallVector<mlir::MemRefType> originalTypes;
        llvm::SmallVector<unsigned> memrefIndices;

        for (unsigned i = 0; i < newInputs.size(); ++i) {
            auto memrefType = mlir::dyn_cast<mlir::MemRefType>(newInputs[i]);
            if (!memrefType) {
                continue;
            }

            auto totalBytes = getStaticByteSize(memrefType);
            if (!totalBytes) {
                func.emitError(
                    "cannot lower function argument with dynamic shape or "
                    "unsupported element size");
                return mlir::failure();
            }

            auto byteMemRefType = mlir::MemRefType::get(
                {static_cast<int64_t>(*totalBytes)}, i8Type,
                mlir::MemRefLayoutAttrInterface(),
                normalizeBuiltinMemorySpace(context,
                                            memrefType.getMemorySpace()));
            newInputs[i] = byteMemRefType;
            memrefIndices.push_back(i);
            originalTypes.push_back(memrefType);
        }

        if (memrefIndices.empty()) {
            continue;
        }

        auto newFuncType =
            mlir::FunctionType::get(context, newInputs, funcType.getResults());
        func.setType(newFuncType);

        if (!func.isExternal()) {
            auto &entryBlock = func.getBody().front();
            for (unsigned i = 0; i < newInputs.size(); ++i) {
                entryBlock.getArgument(i).setType(newInputs[i]);
            }

            mlir::OpBuilder builder(&entryBlock, entryBlock.begin());
            auto loc = func.getLoc();

            for (size_t idx = 0; idx < memrefIndices.size(); ++idx) {
                auto argIndex = memrefIndices[idx];
                auto originalType = originalTypes[idx];
                auto arg = entryBlock.getArgument(argIndex);

                auto zero =
                    mlir::arith::ConstantIndexOp::create(builder, loc, 0);
                auto contiguousType = mlir::MemRefType::get(
                    originalType.getShape(), originalType.getElementType(),
                    mlir::MemRefLayoutAttrInterface(),
                    normalizeBuiltinMemorySpace(context,
                                                originalType.getMemorySpace()));
                llvm::SmallVector<mlir::Value> dynamicSizes;
                auto view = mlir::memref::ViewOp::create(
                    builder, loc, contiguousType, arg, zero, dynamicSizes);

                mlir::Value castResult = view.getResult();
                if (!originalType.getLayout().isIdentity()) {
                    auto stridedLayout =
                        llvm::dyn_cast<mlir::StridedLayoutAttr>(
                            originalType.getLayout());
                    if (!stridedLayout) {
                        func.emitError("unsupported memref layout for "
                                       "function argument");
                        return mlir::failure();
                    }

                    llvm::SmallVector<int64_t> staticStrides;
                    llvm::SmallVector<mlir::OpFoldResult> strideOfrs;
                    for (auto stride : stridedLayout.getStrides()) {
                        if (mlir::ShapedType::isDynamic(stride)) {
                            func.emitError("dynamic strides are not supported "
                                           "for function arguments");
                            return mlir::failure();
                        }
                        staticStrides.push_back(stride);
                        strideOfrs.push_back(builder.getIndexAttr(stride));
                    }

                    llvm::SmallVector<mlir::OpFoldResult> sizeOfrs;
                    for (auto dim : originalType.getShape()) {
                        if (mlir::ShapedType::isDynamic(dim) || dim < 0) {
                            func.emitError("cannot lower function argument "
                                           "with dynamic shape");
                            return mlir::failure();
                        }
                        sizeOfrs.push_back(builder.getIndexAttr(dim));
                    }

                    auto reinterpretOp =
                        mlir::memref::ReinterpretCastOp::create(
                            builder, loc, originalType, castResult,
                            builder.getIndexAttr(0), sizeOfrs, strideOfrs);
                    castResult = reinterpretOp.getResult();
                }

                arg.replaceAllUsesExcept(castResult, view);
            }
        }

        auto symbolUses = mlir::SymbolTable::getSymbolUses(func, module);
        if (!symbolUses) {
            continue;
        }

        for (auto use : *symbolUses) {
            auto call = mlir::dyn_cast<mlir::CallOpInterface>(use.getUser());
            if (!call) {
                continue;
            }
            for (size_t idx = 0; idx < memrefIndices.size(); ++idx) {
                auto argIndex = memrefIndices[idx];
                auto operand = call.getArgOperands()[argIndex];
                auto memrefType =
                    mlir::dyn_cast<mlir::MemRefType>(operand.getType());
                if (!memrefType) {
                    call.emitError("expected memref operand for function call");
                    return mlir::failure();
                }
                auto totalBytes = getStaticByteSize(memrefType);
                if (!totalBytes) {
                    call.emitError(
                        "unable to compute byte size for call operand");
                    return mlir::failure();
                }

                mlir::OpBuilder builder(call.getOperation());
                auto byteMemRefType = mlir::MemRefType::get(
                    {static_cast<int64_t>(*totalBytes)}, i8Type,
                    mlir::MemRefLayoutAttrInterface(),
                    normalizeBuiltinMemorySpace(context,
                                                memrefType.getMemorySpace()));

                mlir::Value base;
                if (auto baseInfo = findBaseI8Memref(operand, builder)) {
                    base = mlir::memref::ViewOp::create(
                        builder, call.getLoc(), byteMemRefType, baseInfo->base,
                        baseInfo->byteShift, mlir::ValueRange{});
                } else {
                    auto castOp = AveLangMemRefCastOp::create(
                        builder, call.getLoc(), operand, byteMemRefType);
                    base = castOp.getResult();
                }
                auto argOperands = call.getArgOperandsMutable();
                argOperands[argIndex].set(base);
            }
        }
    }

    return mlir::success();
}

bool isConvertibleShape(llvm::ArrayRef<int64_t> shape) {
    for (auto dim : shape) {
        if (dim < 0 && dim != mlir::ShapedType::kDynamic) {
            return false;
        }
    }
    return true;
}

bool isConvertibleStrides(llvm::ArrayRef<int64_t> strides) {
    for (auto stride : strides) {
        if (stride < 0 && stride != mlir::ShapedType::kDynamic) {
            return false;
        }
    }
    return true;
}

bool isConvertibleLayoutType(LayoutType layoutType) {
    auto dims = layoutType.getDims();
    auto strides = layoutType.getStrides();
    if (!strides.empty() && dims.size() != strides.size()) {
        return false;
    }
    if (!isConvertibleShape(dims)) {
        return false;
    }
    if (!strides.empty() && !isConvertibleStrides(strides)) {
        return false;
    }
    return true;
}

bool validateMemRefType(MemRefType type, mlir::Location loc) {
    if (!mlir::MemRefType::isValidElementType(type.getElementType())) {
        mlir::emitError(loc) << "cannot lower !ave.memref with invalid "
                                "element type";
        return false;
    }

    auto layoutType = mlir::dyn_cast<LayoutType>(type.getLayout());
    if (!layoutType) {
        mlir::emitError(loc)
            << "cannot lower !ave.memref without ave.layout type";
        return false;
    }

    if (!isConvertibleShape(layoutType.getDims())) {
        mlir::emitError(loc) << "cannot lower !ave.memref with negative "
                                "shape dimensions";
        return false;
    }

    if (!isConvertibleLayoutType(layoutType)) {
        mlir::emitError(loc)
            << "cannot lower !ave.memref with invalid layout strides";
        return false;
    }

    return true;
}

mlir::MemRefType convertMemRefType(MemRefType type) {
    mlir::MemRefLayoutAttrInterface layout;
    auto layoutType = mlir::cast<LayoutType>(type.getLayout());
    auto strides = layoutType.getStrides();
    if (!strides.empty()) {
        layout = mlir::StridedLayoutAttr::get(type.getContext(),
                                              /*offset=*/0, strides);
    }
    auto memorySpace =
        normalizeBuiltinMemorySpace(type.getContext(), type.getMemorySpace());
    return mlir::MemRefType::get(layoutType.getDims(), type.getElementType(),
                                 layout, memorySpace);
}

class AnyOpTypeConversionPattern : public mlir::ConversionPattern {
  public:
    AnyOpTypeConversionPattern(const mlir::TypeConverter &converter,
                               mlir::MLIRContext *context)
        : mlir::ConversionPattern(converter, mlir::Pattern::MatchAnyOpTypeTag(),
                                  1, context) {}

    mlir::LogicalResult
    matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter &rewriter) const override {
        auto *converter = getTypeConverter();
        if (!converter) {
            return mlir::failure();
        }

        auto newOp =
            mlir::convertOpResultTypes(op, operands, *converter, rewriter);
        if (mlir::failed(newOp)) {
            return mlir::failure();
        }
        rewriter.replaceOp(op, (*newOp)->getResults());
        return mlir::success();
    }
};

class ReinterpretCastLayoutPattern
    : public mlir::OpRewritePattern<mlir::memref::ReinterpretCastOp> {
  public:
    using mlir::OpRewritePattern<
        mlir::memref::ReinterpretCastOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::memref::ReinterpretCastOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

class FlattenShapeCastExtractPattern
    : public mlir::OpRewritePattern<mlir::vector::ExtractOp> {
  public:
    using mlir::OpRewritePattern<mlir::vector::ExtractOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(mlir::vector::ExtractOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

class EraseDeadMakeTuplePattern
    : public mlir::OpRewritePattern<MakeIntTupleOp> {
  public:
    using mlir::OpRewritePattern<MakeIntTupleOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(MakeIntTupleOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

class EraseDeadMakeLayoutPattern : public mlir::OpRewritePattern<MakeLayoutOp> {
  public:
    using mlir::OpRewritePattern<MakeLayoutOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(MakeLayoutOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

class FullLoweringPattern : public mlir::OpRewritePattern<FullOp> {
  public:
    using mlir::OpRewritePattern<FullOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(FullOp op, mlir::PatternRewriter &rewriter) const override;
};

class AveLangMemRefAllocaLoweringPattern
    : public mlir::OpRewritePattern<AveLangMemRefAllocaOp> {
  public:
    using mlir::OpRewritePattern<AveLangMemRefAllocaOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AveLangMemRefAllocaOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

class AveLangMemRefLoadLoweringPattern
    : public mlir::OpRewritePattern<AveLangMemRefLoadOp> {
  public:
    using mlir::OpRewritePattern<AveLangMemRefLoadOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AveLangMemRefLoadOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

class AveLangMemRefLoadVecLoweringPattern
    : public mlir::OpRewritePattern<AveLangMemRefLoadVecOp> {
  public:
    using mlir::OpRewritePattern<AveLangMemRefLoadVecOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AveLangMemRefLoadVecOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

class AveLangMemRefStoreLoweringPattern
    : public mlir::OpRewritePattern<AveLangMemRefStoreOp> {
  public:
    using mlir::OpRewritePattern<AveLangMemRefStoreOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AveLangMemRefStoreOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

class AveLangMemRefViewLoweringPattern
    : public mlir::OpRewritePattern<AveLangMemRefViewOp> {
  public:
    using mlir::OpRewritePattern<AveLangMemRefViewOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AveLangMemRefViewOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

class AveLangMemRefCastLoweringPattern
    : public mlir::OpRewritePattern<AveLangMemRefCastOp> {
  public:
    using mlir::OpRewritePattern<AveLangMemRefCastOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AveLangMemRefCastOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

class AveLangMemRefExtractAlignedPointerLoweringPattern
    : public mlir::OpRewritePattern<
          AveLangMemRefExtractAlignedPointerAsIndexOp> {
  public:
    using mlir::OpRewritePattern<
        AveLangMemRefExtractAlignedPointerAsIndexOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AveLangMemRefExtractAlignedPointerAsIndexOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

class AveLangMemRefSubViewLoweringPattern
    : public mlir::OpRewritePattern<AveLangMemRefSubViewOp> {
  public:
    using mlir::OpRewritePattern<AveLangMemRefSubViewOp>::OpRewritePattern;

    mlir::LogicalResult
    matchAndRewrite(AveLangMemRefSubViewOp op,
                    mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult ReinterpretCastLayoutPattern::matchAndRewrite(
    mlir::memref::ReinterpretCastOp op, mlir::PatternRewriter &rewriter) const {
    if (op->hasAttr("layout_map")) {
        return mlir::failure();
    }

    auto resultType = op.getType();
    auto layoutAttr = resultType.getLayout();
    if (!layoutAttr) {
        return mlir::failure();
    }

    auto stridedLayout = llvm::dyn_cast<mlir::StridedLayoutAttr>(layoutAttr);
    if (!stridedLayout) {
        return mlir::failure();
    }

    auto strides = stridedLayout.getStrides();
    auto offset = stridedLayout.getOffset();
    auto shape = resultType.getShape();
    unsigned numDims = shape.size();

    unsigned numSymbols = 0;
    for (auto stride : strides) {
        if (mlir::ShapedType::isDynamic(stride)) {
            numSymbols++;
        }
    }
    bool hasDynamicOffset = mlir::ShapedType::isDynamic(offset);
    if (hasDynamicOffset) {
        numSymbols++;
    }

    llvm::SmallVector<mlir::AffineExpr> dimExprs;
    llvm::SmallVector<mlir::AffineExpr> symbolExprs;
    for (unsigned i = 0; i < numDims; ++i) {
        dimExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
    }
    for (unsigned i = 0; i < numSymbols; ++i) {
        symbolExprs.push_back(
            mlir::getAffineSymbolExpr(i, rewriter.getContext()));
    }

    mlir::AffineExpr indexExpr;
    unsigned symbolIdx = 0;
    if (hasDynamicOffset) {
        indexExpr = symbolExprs[symbolIdx++];
    } else {
        indexExpr = mlir::getAffineConstantExpr(offset, rewriter.getContext());
    }

    for (unsigned i = 0; i < numDims; ++i) {
        mlir::AffineExpr strideExpr;
        if (mlir::ShapedType::isDynamic(strides[i])) {
            strideExpr = symbolExprs[symbolIdx++];
        } else {
            strideExpr =
                mlir::getAffineConstantExpr(strides[i], rewriter.getContext());
        }
        indexExpr = indexExpr + dimExprs[i] * strideExpr;
    }

    auto layoutMap = mlir::AffineMap::get(numDims, numSymbols, indexExpr,
                                          rewriter.getContext());
    llvm::SmallVector<mlir::Attribute> strideAttrs;
    for (auto stride : strides) {
        if (mlir::ShapedType::isDynamic(stride)) {
            strideAttrs.push_back(rewriter.getStringAttr("?"));
        } else {
            strideAttrs.push_back(rewriter.getI64IntegerAttr(stride));
        }
    }

    op->setAttr("layout_map", mlir::AffineMapAttr::get(layoutMap));
    op->setAttr("stride_pattern", rewriter.getArrayAttr(strideAttrs));

    return mlir::success();
}

mlir::LogicalResult FlattenShapeCastExtractPattern::matchAndRewrite(
    mlir::vector::ExtractOp op, mlir::PatternRewriter &rewriter) const {
    if (mlir::isa<mlir::VectorType>(op.getResult().getType())) {
        return mlir::failure();
    }

    auto shapeCast = op.getSource().getDefiningOp<mlir::vector::ShapeCastOp>();
    if (!shapeCast) {
        return mlir::failure();
    }

    auto sourceType = shapeCast.getSourceVectorType();
    auto resultType = shapeCast.getResultVectorType();
    if (sourceType.getRank() != 1 || !sourceType.hasStaticShape() ||
        !resultType.hasStaticShape()) {
        return mlir::failure();
    }

    auto shape = resultType.getShape();
    if (shape.size() != op.getNumIndices() ||
        sourceType.getNumElements() != resultType.getNumElements()) {
        return mlir::failure();
    }

    auto loc = op.getLoc();
    auto mixedPosition = op.getMixedPosition();
    mlir::Value linearIndex =
        mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);

    int64_t stride = resultType.getNumElements();
    for (auto [dim, position] : llvm::zip(shape, mixedPosition)) {
        if (dim <= 0) {
            return mlir::failure();
        }

        stride /= dim;
        mlir::Value index =
            mlir::getValueOrCreateConstantIndexOp(rewriter, loc, position);
        if (stride != 1) {
            auto strideValue =
                mlir::arith::ConstantIndexOp::create(rewriter, loc, stride);
            index =
                mlir::arith::MulIOp::create(rewriter, loc, index, strideValue);
        }
        linearIndex =
            mlir::arith::AddIOp::create(rewriter, loc, linearIndex, index);
    }

    rewriter.replaceOpWithNewOp<mlir::vector::ExtractOp>(
        op, shapeCast.getSource(), linearIndex);
    return mlir::success();
}

mlir::LogicalResult EraseDeadMakeTuplePattern::matchAndRewrite(
    MakeIntTupleOp op, mlir::PatternRewriter &rewriter) const {
    if (!op.use_empty()) {
        return mlir::failure();
    }
    rewriter.eraseOp(op);
    return mlir::success();
}

mlir::LogicalResult EraseDeadMakeLayoutPattern::matchAndRewrite(
    MakeLayoutOp op, mlir::PatternRewriter &rewriter) const {
    if (!op.use_empty()) {
        return mlir::failure();
    }
    rewriter.eraseOp(op);
    return mlir::success();
}

mlir::LogicalResult
FullLoweringPattern::matchAndRewrite(FullOp op,
                                     mlir::PatternRewriter &rewriter) const {
    // Get the operands
    auto shapeValue = op.getShape();
    auto fillValue = op.getValue();

    // Validate that we have the operands we expect
    if (!shapeValue || !fillValue) {
        return mlir::failure();
    }

    // Get the result type
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!resultType) {
        return mlir::failure();
    }

    auto static_shape = resultType.getShape();
    if (static_shape.empty()) {
        return mlir::failure();
    }

    // Extract shape values from the shape tuple
    llvm::SmallVector<mlir::Value> dimValues;
    auto shapeTupleOp = shapeValue.getDefiningOp<MakeIntTupleOp>();
    if (!shapeTupleOp) {
        return mlir::failure();
    }

    for (auto elem : shapeTupleOp.getElements()) {
        if (!extractTupleValues(elem, dimValues)) {
            return mlir::failure();
        }
    }

    // Verify all dimensions are static
    for (auto dim : static_shape) {
        if (dim == mlir::ShapedType::kDynamic) {
            return mlir::failure(); // Don't support dynamic shapes for now
        }
    }

    auto elementType = resultType.getElementType();
    auto location = op.getLoc();

    // Check if element type is a vector type
    auto vectorElementType = mlir::dyn_cast<mlir::VectorType>(elementType);
    bool isVectorElement = vectorElementType != nullptr;

    // Step 1: Calculate total size (flattened)
    int64_t totalSize = 1;
    for (auto dim : static_shape) {
        totalSize *= dim;
    }

    // Step 2: Create fill vector - broadcast to element type if vector
    mlir::Value fillVector;
    if (isVectorElement) {
        // Broadcast the scalar fill value to result memref's element vector
        // type
        fillVector = mlir::vector::BroadcastOp::create(
            rewriter, location, vectorElementType, fillValue);
    } else {
        // Create a 1D vector type of size totalSize and broadcast
        auto vectorType = mlir::VectorType::get({totalSize}, elementType);
        fillVector = mlir::vector::BroadcastOp::create(rewriter, location,
                                                       vectorType, fillValue);
    }

    // Step 3: Create a memref with target shape and private memory space
    auto addressSpaceAttr = normalizeBuiltinMemorySpace(
        rewriter.getContext(),
        mlir::gpu::AddressSpaceAttr::get(rewriter.getContext(),
                                         mlir::gpu::AddressSpace::Private));
    auto targetMemRefType = mlir::MemRefType::get(
        static_shape, elementType, mlir::MemRefLayoutAttrInterface(),
        addressSpaceAttr);

    mlir::IntegerAttr alignmentAttr;
    if (auto alignment = getPreferredAlignment(elementType)) {
        alignmentAttr = rewriter.getI64IntegerAttr(*alignment);
    }

    auto targetMemref =
        mlir::memref::AllocaOp::create(rewriter, location, targetMemRefType,
                                       mlir::ValueRange{}, alignmentAttr);

    // Step 4: Create a 1D view of the memref for initialization
    llvm::SmallVector<int64_t> flattenedShape = {totalSize};
    auto flattenedMemRefType = mlir::MemRefType::get(
        flattenedShape, elementType, mlir::MemRefLayoutAttrInterface(),
        addressSpaceAttr);

    // For the reinterpret_cast, sizes and strides should match the RESULT type
    // Since we're creating a 1D view, we need 1 size and 1 stride
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    sizes.push_back(rewriter.getIndexAttr(totalSize));

    llvm::SmallVector<mlir::OpFoldResult> strides;
    strides.push_back(rewriter.getIndexAttr(1)); // Unit stride for 1D

    // Collapse the memref to 1D
    auto collapsedMemref = mlir::memref::ReinterpretCastOp::create(
        rewriter, location, flattenedMemRefType, targetMemref,
        rewriter.getIndexAttr(0), sizes, strides);

    // Step 5: Write the fill vector to the 1D memref
    if (isVectorElement) {
        // Use a for-loop to write the vector element to each position
        auto lowerBound =
            mlir::arith::ConstantIndexOp::create(rewriter, location, 0);
        auto upperBound =
            mlir::arith::ConstantIndexOp::create(rewriter, location, totalSize);
        auto step = mlir::arith::ConstantIndexOp::create(rewriter, location, 1);

        auto loop = mlir::scf::ForOp::create(rewriter, location, lowerBound,
                                             upperBound, step);

        rewriter.setInsertionPointToStart(loop.getBody());
        auto iv = loop.getInductionVar();

        // Store the vector directly at this position in the flattened memref
        mlir::vector::StoreOp::create(rewriter, location, fillVector,
                                      collapsedMemref.getResult(),
                                      mlir::ValueRange{iv});

        rewriter.setInsertionPointAfter(loop);
    } else {
        // Step 6: Write the scalar vector to the 1D memref using
        // vector::StoreOp
        auto zeroIndex =
            mlir::arith::ConstantIndexOp::create(rewriter, location, 0);
        mlir::vector::StoreOp::create(rewriter, location, fillVector,
                                      collapsedMemref.getResult(),
                                      mlir::ValueRange{zeroIndex});
    }

    rewriter.replaceOp(op, targetMemref.getResult());
    return mlir::success();
}

mlir::LogicalResult AveLangMemRefAllocaLoweringPattern::matchAndRewrite(
    AveLangMemRefAllocaOp op, mlir::PatternRewriter &rewriter) const {
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!resultType) {
        return mlir::failure();
    }

    if (!op.getDynamicSizes().empty() || !resultType.hasStaticShape()) {
        op.emitError("ave.memref.alloca only supports static sizes");
        return mlir::failure();
    }

    auto totalBytes = getStaticByteSize(resultType);
    if (!totalBytes) {
        op.emitError("unable to compute static allocation size for memref");
        return mlir::failure();
    }

    auto normalizedMemorySpace = normalizeBuiltinMemorySpace(
        rewriter.getContext(), resultType.getMemorySpace());
    auto contiguousType = mlir::MemRefType::get(
        resultType.getShape(), resultType.getElementType(),
        mlir::MemRefLayoutAttrInterface(), normalizedMemorySpace);

    mlir::IntegerAttr alignmentAttr;
    if (auto alignment = getPreferredAllocaAlignment(resultType)) {
        alignmentAttr = rewriter.getI64IntegerAttr(*alignment);
    }

    auto buildResultWithTypedAlloca = [&]() -> mlir::Value {
        auto typedAlloca = mlir::memref::AllocaOp::create(
            rewriter, op.getLoc(), contiguousType, mlir::ValueRange{},
            alignmentAttr);

        if (resultType.getLayout().isIdentity()) {
            return typedAlloca.getResult();
        }

        auto stridedLayout =
            llvm::dyn_cast<mlir::StridedLayoutAttr>(resultType.getLayout());
        if (!stridedLayout) {
            op.emitError("unsupported memref layout for alloca");
            return {};
        }

        llvm::SmallVector<int64_t> staticStrides;
        llvm::SmallVector<mlir::OpFoldResult> strideOfrs;
        for (auto stride : stridedLayout.getStrides()) {
            if (mlir::ShapedType::isDynamic(stride)) {
                op.emitError("dynamic strides are not supported for alloca");
                return {};
            }
            staticStrides.push_back(stride);
            strideOfrs.push_back(rewriter.getIndexAttr(stride));
        }

        llvm::SmallVector<mlir::OpFoldResult> sizeOfrs;
        for (auto dim : resultType.getShape()) {
            sizeOfrs.push_back(rewriter.getIndexAttr(dim));
        }

        auto reinterpretOp = mlir::memref::ReinterpretCastOp::create(
            rewriter, op.getLoc(), resultType, typedAlloca.getResult(),
            rewriter.getIndexAttr(0), sizeOfrs, strideOfrs);
        return reinterpretOp.getResult();
    };

    // Preserve single-bit scalar memrefs as typed allocas so later scalar
    // promotion passes can still recognize them as booleans instead of raw
    // byte buffers.
    if (resultType.getElementType().isInteger(1)) {
        auto typedResult = buildResultWithTypedAlloca();
        if (!typedResult) {
            return mlir::failure();
        }
        rewriter.replaceOp(op, typedResult);
        return mlir::success();
    }

    // Keep workgroup allocations typed so outlined shared-memory globals retain
    // natural element typing and requested alignment.
    if (isWorkgroupMemorySpaceAttr(normalizedMemorySpace)) {
        auto typedResult = buildResultWithTypedAlloca();
        if (!typedResult) {
            return mlir::failure();
        }
        rewriter.replaceOp(op, typedResult);
        return mlir::success();
    }

    auto i8Type = rewriter.getIntegerType(8);
    auto byteMemRefType = mlir::MemRefType::get(
        {static_cast<int64_t>(*totalBytes)}, i8Type,
        mlir::MemRefLayoutAttrInterface(), normalizedMemorySpace);

    auto base =
        mlir::memref::AllocaOp::create(rewriter, op.getLoc(), byteMemRefType,
                                       mlir::ValueRange{}, alignmentAttr);

    auto zero = mlir::arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    llvm::SmallVector<mlir::Value> dynamicSizes;
    auto view = mlir::memref::ViewOp::create(
        rewriter, op.getLoc(), contiguousType, base, zero, dynamicSizes);

    mlir::Value castResult = view.getResult();
    if (!resultType.getLayout().isIdentity()) {
        auto stridedLayout =
            llvm::dyn_cast<mlir::StridedLayoutAttr>(resultType.getLayout());
        if (!stridedLayout) {
            op.emitError("unsupported memref layout for alloca");
            return mlir::failure();
        }

        llvm::SmallVector<int64_t> staticStrides;
        llvm::SmallVector<mlir::OpFoldResult> strideOfrs;
        for (auto stride : stridedLayout.getStrides()) {
            if (mlir::ShapedType::isDynamic(stride)) {
                op.emitError("dynamic strides are not supported for alloca");
                return mlir::failure();
            }
            staticStrides.push_back(stride);
            strideOfrs.push_back(rewriter.getIndexAttr(stride));
        }

        llvm::SmallVector<mlir::OpFoldResult> sizeOfrs;
        for (auto dim : resultType.getShape()) {
            sizeOfrs.push_back(rewriter.getIndexAttr(dim));
        }

        auto reinterpretOp = mlir::memref::ReinterpretCastOp::create(
            rewriter, op.getLoc(), resultType, castResult,
            rewriter.getIndexAttr(0), sizeOfrs, strideOfrs);
        castResult = reinterpretOp.getResult();
    }

    rewriter.replaceOp(op, castResult);
    return mlir::success();
}

mlir::LogicalResult AveLangMemRefLoadLoweringPattern::matchAndRewrite(
    AveLangMemRefLoadOp op, mlir::PatternRewriter &rewriter) const {
    auto memrefType = getBuiltinMemRefType(op.getMemref().getType());
    if (!memrefType) {
        return mlir::failure();
    }

    auto elementType = memrefType.getElementType();
    auto resultType = op.getResult().getType();
    auto indices = op.getIndices();

    if (resultType == elementType) {
        auto memref = castToBuiltinMemRef(op.getLoc(), op.getMemref(),
                                          rewriter);
        if (!memref) {
            return mlir::failure();
        }
        rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, memref, indices);
        return mlir::success();
    }

    return mlir::failure();
}

mlir::LogicalResult AveLangMemRefLoadVecLoweringPattern::matchAndRewrite(
    AveLangMemRefLoadVecOp op, mlir::PatternRewriter &rewriter) const {
    auto memrefType = getBuiltinMemRefType(op.getMemref().getType());
    if (!memrefType) {
        return mlir::failure();
    }

    auto resultType =
        mlir::dyn_cast<mlir::VectorType>(op.getResult().getType());
    if (!resultType) {
        return mlir::failure();
    }

    auto memref = castToBuiltinMemRef(op.getLoc(), op.getMemref(), rewriter);
    if (!memref) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::vector::LoadOp>(op, resultType, memref,
                                                      op.getIndices());
    return mlir::success();
}

mlir::LogicalResult AveLangMemRefStoreLoweringPattern::matchAndRewrite(
    AveLangMemRefStoreOp op, mlir::PatternRewriter &rewriter) const {
    auto memrefType = getBuiltinMemRefType(op.getMemref().getType());
    if (!memrefType) {
        return mlir::failure();
    }

    auto elementType = memrefType.getElementType();
    auto valueType = op.getValue().getType();
    auto indices = op.getIndices();

    if (valueType == elementType) {
        auto memref = castToBuiltinMemRef(op.getLoc(), op.getMemref(),
                                          rewriter);
        if (!memref) {
            return mlir::failure();
        }
        rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(
            op, op.getValue(), memref, indices);
        return mlir::success();
    }

    if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(valueType)) {
        if (elementType == vectorType.getElementType()) {
            auto memref =
                castToBuiltinMemRef(op.getLoc(), op.getMemref(), rewriter);
            if (!memref) {
                return mlir::failure();
            }
            rewriter.replaceOpWithNewOp<mlir::vector::StoreOp>(
                op, op.getValue(), memref, indices);
            return mlir::success();
        }
    }

    return mlir::failure();
}

mlir::LogicalResult AveLangMemRefViewLoweringPattern::matchAndRewrite(
    AveLangMemRefViewOp op, mlir::PatternRewriter &rewriter) const {
    auto resultType = getBuiltinMemRefType(op.getResult().getType());
    if (!resultType) {
        return mlir::failure();
    }
    auto sourceType = getBuiltinMemRefType(op.getSource().getType());
    auto source = castToBuiltinMemRef(op.getLoc(), op.getSource(), rewriter);
    if (!source) {
        return mlir::failure();
    }
    auto viewType = withResolvedMemorySpace(
        rewriter.getContext(), resultType,
        sourceType ? sourceType.getMemorySpace() : mlir::Attribute{});
    rewriter.replaceOpWithNewOp<mlir::memref::ViewOp>(
        op, viewType, source, op.getByteShift(), op.getSizes());
    return mlir::success();
}

mlir::LogicalResult AveLangMemRefCastLoweringPattern::matchAndRewrite(
    AveLangMemRefCastOp op, mlir::PatternRewriter &rewriter) const {
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!resultType) {
        return mlir::failure();
    }

    auto sourceType = mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    if (sourceType && sourceType.getShape() == resultType.getShape() &&
        sourceType.getElementType() == resultType.getElementType() &&
        sourceType.getLayout() == resultType.getLayout() &&
        sourceType.getMemorySpace() != resultType.getMemorySpace()) {
        rewriter.replaceOpWithNewOp<mlir::memref::MemorySpaceCastOp>(
            op, resultType, op.getSource());
        return mlir::success();
    }

    auto loc = op.getLoc();
    std::optional<BaseMemRefInfo> baseInfo;
    if (mlir::isa<mlir::ptr::PtrType>(op.getSource().getType())) {
        auto i8Type = rewriter.getIntegerType(8);
        auto baseType = mlir::MemRefType::get(
            {mlir::ShapedType::kDynamic}, i8Type,
            mlir::MemRefLayoutAttrInterface(),
            normalizeBuiltinMemorySpace(rewriter.getContext(),
                                        resultType.getMemorySpace()));
        auto base = mlir::UnrealizedConversionCastOp::create(
            rewriter, loc, mlir::TypeRange{baseType},
            mlir::ValueRange{op.getSource()});
        auto zero =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, 0).getResult();
        baseInfo = BaseMemRefInfo{base.getResult(0), zero};
    } else {
        baseInfo = findBaseI8Memref(op.getSource(), rewriter);
    }
    if (!baseInfo) {
        return mlir::failure();
    }

    llvm::SmallVector<mlir::Value> dimValues;
    llvm::SmallVector<mlir::Value> strideValues;
    bool hasLayout = false;
    if (auto layoutValue = op.getLayout()) {
        auto layoutOp = layoutValue.getDefiningOp<MakeLayoutOp>();
        if (!layoutOp) {
            return mlir::failure();
        }
        if (!extractTupleValues(layoutOp.getDims(), dimValues) ||
            !extractTupleValues(layoutOp.getStride(), strideValues)) {
            return mlir::failure();
        }
        hasLayout = true;
    }

    llvm::SmallVector<mlir::Value> sizeValues;
    llvm::SmallVector<mlir::OpFoldResult> sizeOfrs;
    auto shape = resultType.getShape();
    llvm::SmallVector<int64_t> foldedShape(shape.begin(), shape.end());
    llvm::SmallVector<mlir::Value> runtimeDimValues(shape.size());
    if (hasLayout) {
        if (dimValues.size() != shape.size()) {
            return mlir::failure();
        }
        for (size_t i = 0; i < dimValues.size(); ++i) {
            auto dimValue = dimValues[i];
            if (auto staticVal = getStaticValue(dimValue)) {
                foldedShape[i] = *staticVal;
            } else {
                if (!dimValue.getType().isIndex()) {
                    dimValue = mlir::arith::IndexCastOp::create(
                        rewriter, loc, rewriter.getIndexType(), dimValue);
                }
                foldedShape[i] = shape[i];
                runtimeDimValues[i] = dimValue;
            }
        }
    } else {
        for (size_t i = 0; i < shape.size(); ++i) {
            auto dim = shape[i];
            if (mlir::ShapedType::isDynamic(dim)) {
                return mlir::failure();
            }
            foldedShape[i] = dim;
        }
    }

    for (size_t i = 0; i < foldedShape.size(); ++i) {
        if (mlir::ShapedType::isDynamic(foldedShape[i])) {
            if (!runtimeDimValues[i]) {
                return mlir::failure();
            }
            sizeOfrs.push_back(runtimeDimValues[i]);
        } else {
            sizeOfrs.push_back(rewriter.getIndexAttr(foldedShape[i]));
        }
    }

    for (size_t i = 0; i < foldedShape.size(); ++i) {
        if (mlir::ShapedType::isDynamic(foldedShape[i])) {
            sizeValues.push_back(
                materializeIndexValue(rewriter, loc, sizeOfrs[i]));
        }
    }

    bool useDirectReinterpret = false;
    mlir::Value reinterpretSource = op.getSource();
    mlir::OpFoldResult reinterpretOffset = rewriter.getIndexAttr(0);
    bool sourceIsSubviewLike =
        op.getSource().getDefiningOp<mlir::memref::SubViewOp>() ||
        op.getSource().getDefiningOp<AveLangMemRefSubViewOp>();
    if (sourceType &&
        sourceType.getElementType() == resultType.getElementType() &&
        !sourceIsSubviewLike) {
        int64_t sourceOffset = 0;
        llvm::SmallVector<int64_t> sourceStrides;
        if (mlir::succeeded(
                sourceType.getStridesAndOffset(sourceStrides, sourceOffset)) &&
            !mlir::ShapedType::isDynamic(sourceOffset)) {
            useDirectReinterpret = true;
            reinterpretOffset = rewriter.getIndexAttr(sourceOffset);
        }
    }

    bool needReinterpret = false;
    llvm::SmallVector<mlir::OpFoldResult> strideOfrs;
    llvm::SmallVector<int64_t> staticStrides;

    if (hasLayout) {
        needReinterpret = true;
        if (strideValues.size() != dimValues.size()) {
            return mlir::failure();
        }

        for (auto strideValue : strideValues) {
            if (auto staticVal = getStaticValue(strideValue)) {
                staticStrides.push_back(*staticVal);
                strideOfrs.push_back(rewriter.getIndexAttr(*staticVal));
            } else {
                if (!strideValue.getType().isIndex()) {
                    strideValue = mlir::arith::IndexCastOp::create(
                        rewriter, loc, rewriter.getIndexType(), strideValue);
                }
                staticStrides.push_back(mlir::ShapedType::kDynamic);
                strideOfrs.push_back(strideValue);
            }
        }
    } else if (auto stridedLayout = llvm::dyn_cast<mlir::StridedLayoutAttr>(
                   resultType.getLayout())) {
        auto strides = stridedLayout.getStrides();
        if (!strides.empty()) {
            needReinterpret = true;
            staticStrides.reserve(strides.size());
            for (auto stride : strides) {
                if (mlir::ShapedType::isDynamic(stride)) {
                    return mlir::failure();
                }
                staticStrides.push_back(stride);
                strideOfrs.push_back(rewriter.getIndexAttr(stride));
            }
        }
    }

    if (useDirectReinterpret && needReinterpret) {
        auto stridedLayout = mlir::StridedLayoutAttr::get(rewriter.getContext(),
                                                          0, staticStrides);
        auto stridedType = mlir::MemRefType::get(
            foldedShape, resultType.getElementType(), stridedLayout,
            resolveMemorySpace(rewriter.getContext(),
                               resultType.getMemorySpace(),
                               sourceType.getMemorySpace()));

        auto reinterpretOp = mlir::memref::ReinterpretCastOp::create(
            rewriter, loc, stridedType, reinterpretSource, reinterpretOffset,
            sizeOfrs, strideOfrs);

        if (hasLayout) {
            if (auto layoutMap =
                    buildSemiAffineMap(dimValues, strideValues, rewriter);
                layoutMap) {
                reinterpretOp->setAttr("layout_map",
                                       mlir::AffineMapAttr::get(layoutMap));
                addDebugAttributes(reinterpretOp, dimValues, strideValues,
                                   rewriter);
            }
        }

        rewriter.replaceOp(op, reinterpretOp.getResult());
        return mlir::success();
    }

    auto baseType = getBuiltinMemRefType(baseInfo->base.getType());
    if (!baseType) {
        return mlir::failure();
    }

    // First, create a contiguous view from the base i8 buffer to the target
    // element type.
    auto contiguousType = mlir::MemRefType::get(
        foldedShape, resultType.getElementType(),
        mlir::MemRefLayoutAttrInterface(),
        resolveMemorySpace(rewriter.getContext(), resultType.getMemorySpace(),
                           baseType.getMemorySpace()));
    auto view = rewriter.create<mlir::memref::ViewOp>(
        loc, contiguousType, baseInfo->base, baseInfo->byteShift, sizeValues);

    if (!needReinterpret) {
        rewriter.replaceOp(op, view.getResult());
        return mlir::success();
    }

    auto stridedLayout =
        mlir::StridedLayoutAttr::get(rewriter.getContext(), 0, staticStrides);
    auto stridedType = mlir::MemRefType::get(
        foldedShape, resultType.getElementType(), stridedLayout,
        resolveMemorySpace(rewriter.getContext(), resultType.getMemorySpace(),
                           baseType.getMemorySpace()));

    auto reinterpretOp = mlir::memref::ReinterpretCastOp::create(
        rewriter, loc, stridedType, view.getResult(), rewriter.getIndexAttr(0),
        sizeOfrs, strideOfrs);

    if (hasLayout) {
        if (auto layoutMap =
                buildSemiAffineMap(dimValues, strideValues, rewriter);
            layoutMap) {
            reinterpretOp->setAttr("layout_map",
                                   mlir::AffineMapAttr::get(layoutMap));
            addDebugAttributes(reinterpretOp, dimValues, strideValues,
                               rewriter);
        }
    }

    rewriter.replaceOp(op, reinterpretOp.getResult());
    return mlir::success();
}

mlir::LogicalResult
AveLangMemRefExtractAlignedPointerLoweringPattern::matchAndRewrite(
    AveLangMemRefExtractAlignedPointerAsIndexOp op,
    mlir::PatternRewriter &rewriter) const {
    auto memrefType = getBuiltinMemRefType(op.getSource().getType());
    if (!memrefType) {
        return mlir::failure();
    }

    auto loc = op.getLoc();
    if (auto baseInfo = findBaseI8Memref(op.getSource(), rewriter)) {
        auto basePtr = mlir::memref::ExtractAlignedPointerAsIndexOp::create(
            rewriter, loc, baseInfo->base);
        auto byteShift = baseInfo->byteShift;
        if (auto shiftConst = mlir::getConstantIntValue(byteShift);
            shiftConst && *shiftConst == 0) {
            rewriter.replaceOp(op, basePtr.getResult());
            return mlir::success();
        }
        auto shiftedPtr = mlir::arith::AddIOp::create(
            rewriter, loc, basePtr.getResult(), byteShift);
        rewriter.replaceOp(op, shiftedPtr.getResult());
        return mlir::success();
    }

    auto source = castToBuiltinMemRef(loc, op.getSource(), rewriter);
    if (!source) {
        return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<mlir::memref::ExtractAlignedPointerAsIndexOp>(
        op, source);
    return mlir::success();
}

mlir::LogicalResult AveLangMemRefSubViewLoweringPattern::matchAndRewrite(
    AveLangMemRefSubViewOp op, mlir::PatternRewriter &rewriter) const {
    auto sourceType = getBuiltinMemRefType(op.getSource().getType());
    if (!sourceType) {
        return mlir::failure();
    }
    auto source = castToBuiltinMemRef(op.getLoc(), op.getSource(), rewriter);
    if (!source) {
        return mlir::failure();
    }

    auto resultType = getBuiltinMemRefType(op.getResult().getType());
    if (!resultType) {
        return mlir::failure();
    }

    auto offsets = op.getOffsets();
    auto sizes = op.getSizes();
    auto strides = op.getStrides();
    auto rank = static_cast<size_t>(sourceType.getRank());

    if (offsets.size() > rank || sizes.size() > rank || strides.size() > rank) {
        return mlir::failure();
    }

    llvm::SmallVector<mlir::OpFoldResult> offsetsFold;
    llvm::SmallVector<mlir::OpFoldResult> sizesFold;
    llvm::SmallVector<mlir::OpFoldResult> stridesFold;
    offsetsFold.reserve(rank);
    sizesFold.reserve(rank);
    stridesFold.reserve(rank);

    for (size_t i = 0; i < rank; ++i) {
        auto foldValue = [&](mlir::Value value) -> mlir::OpFoldResult {
            if (auto constVal = mlir::getConstantIntValue(value)) {
                return rewriter.getIndexAttr(*constVal);
            }
            return value;
        };

        if (i < offsets.size()) {
            offsetsFold.push_back(foldValue(offsets[i]));
        } else {
            offsetsFold.push_back(rewriter.getIndexAttr(0));
        }

        if (i < sizes.size()) {
            sizesFold.push_back(foldValue(sizes[i]));
        } else if (!sourceType.isDynamicDim(i)) {
            sizesFold.push_back(
                rewriter.getIndexAttr(sourceType.getDimSize(i)));
        } else {
            auto dimOp = rewriter.create<mlir::memref::DimOp>(
                op.getLoc(), source, static_cast<int64_t>(i));
            sizesFold.push_back(dimOp.getResult());
        }

        if (i < strides.size()) {
            stridesFold.push_back(foldValue(strides[i]));
        } else {
            stridesFold.push_back(rewriter.getIndexAttr(1));
        }
    }

    auto inferredType = getBuiltinMemRefType(
        mlir::memref::SubViewOp::inferRankReducedResultType(
            resultType.getShape(), sourceType, offsetsFold, sizesFold,
            stridesFold));
    if (!inferredType) {
        return mlir::failure();
    }

    auto subview = rewriter.create<mlir::memref::SubViewOp>(
        op.getLoc(), inferredType, source, offsetsFold, sizesFold,
        stridesFold);

    rewriter.replaceOp(op, subview.getResult());
    return mlir::success();
}

/// Pass that lowers AveLang dialect operations to memref operations and adds
/// layout metadata using rewrite patterns.
class LowerAveLangToMemRefPass
    : public mlir::PassWrapper<LowerAveLangToMemRefPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAveLangToMemRefPass)

    llvm::StringRef getArgument() const final {
        return "lower-ave-lang-to-memref";
    }

    llvm::StringRef getDescription() const final {
        return "Lower AveLang dialect operations to memref ops with layout "
               "metadata";
    }

    void runOnOperation() override;
};

} // namespace

void LowerAveLangToMemRefPass::runOnOperation() {
    bool valid = true;
    getOperation()->walk([&](mlir::Operation *op) {
        auto checkType = [&](mlir::Type type) {
            if (auto memrefType = mlir::dyn_cast<MemRefType>(type)) {
                if (!validateMemRefType(memrefType, op->getLoc())) {
                    valid = false;
                }
            }
        };

        for (auto type : op->getOperandTypes()) {
            checkType(type);
        }
        for (auto type : op->getResultTypes()) {
            checkType(type);
        }

        if (auto func = mlir::dyn_cast<mlir::FunctionOpInterface>(op)) {
            for (auto type : func.getArgumentTypes()) {
                checkType(type);
            }
            for (auto type : func.getResultTypes()) {
                checkType(type);
            }
        }
    });

    if (!valid) {
        signalPassFailure();
        return;
    }

    mlir::TypeConverter converter;
    converter.addConversion([](mlir::Type type) { return type; });
    converter.addConversion(
        [](MemRefType type) -> mlir::Type { return convertMemRefType(type); });
    converter.addConversion([](mlir::ptr::PtrType type) -> mlir::Type {
        auto i8Type = mlir::IntegerType::get(type.getContext(), 8);
        return mlir::MemRefType::get(
            {1}, i8Type, mlir::MemRefLayoutAttrInterface(), mlir::Attribute());
    });

    mlir::RewritePatternSet typePatterns(&getContext());
    mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(typePatterns,
                                                              converter);
    typePatterns.add<AnyOpTypeConversionPattern>(converter, &getContext());

    mlir::ConversionTarget typeTarget(getContext());
    typeTarget.markUnknownOpDynamicallyLegal(
        [&](mlir::Operation *op) -> std::optional<bool> {
            if (auto func = mlir::dyn_cast<mlir::FunctionOpInterface>(op)) {
                auto funcType =
                    mlir::dyn_cast<mlir::FunctionType>(func.getFunctionType());
                if (!funcType) {
                    return std::nullopt;
                }
                return converter.isSignatureLegal(funcType) &&
                       converter.isLegal(op);
            }
            return converter.isLegal(op);
        });

    if (mlir::failed(mlir::applyFullConversion(getOperation(), typeTarget,
                                               std::move(typePatterns)))) {
        signalPassFailure();
        return;
    }

    if (mlir::failed(lowerFunctionArgsToI8(
            mlir::cast<mlir::ModuleOp>(getOperation())))) {
        signalPassFailure();
        return;
    }

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<
        ReinterpretCastLayoutPattern, FlattenShapeCastExtractPattern,
        FullLoweringPattern, AveLangMemRefAllocaLoweringPattern,
        AveLangMemRefLoadLoweringPattern, AveLangMemRefLoadVecLoweringPattern,
        AveLangMemRefStoreLoweringPattern, AveLangMemRefViewLoweringPattern,
        AveLangMemRefCastLoweringPattern,
        AveLangMemRefExtractAlignedPointerLoweringPattern,
        AveLangMemRefSubViewLoweringPattern, EraseDeadMakeTuplePattern,
        EraseDeadMakeLayoutPattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> createLowerAveLangToMemRefPass() {
    return std::make_unique<LowerAveLangToMemRefPass>();
}

} // namespace causalflow::avelang::dialect
