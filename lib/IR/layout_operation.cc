#include "layout_operation.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "Utils/assert.h"
#include "constant_folder.h"
#include "generator_context.h"
#include "mlir_generator_impl.h"
#include "parsing_utils.h"
#include "type_system.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace causalflow::avelang::ir {

using namespace mlir;
namespace cf = causalflow::avelang::dialect;

static cf::MemRefType
createAveLangMemRefType(mlir::MLIRContext *context,
                        llvm::ArrayRef<int64_t> shape,
                        llvm::ArrayRef<int64_t> strides, mlir::Type elementType,
                        mlir::Attribute memorySpace = {}) {
    auto layoutType = cf::LayoutType::get(context, shape, strides);
    return cf::MemRefType::get(context, layoutType, elementType, memorySpace);
}

static int64_t ceilDivInt64(int64_t numerator, int64_t denominator) {
    if (denominator <= 0) {
        return 0;
    }
    return (numerator + denominator - 1) / denominator;
}

static LayoutOperation::StaticLayout
squeezeStaticLayout(const llvm::SmallVector<int64_t> &shape,
                    const llvm::SmallVector<int64_t> &strides) {
    llvm::SmallVector<int64_t> squeezedShape;
    llvm::SmallVector<int64_t> squeezedStrides;
    for (size_t i = 0; i < shape.size() && i < strides.size(); ++i) {
        if (shape[i] != 1) {
            squeezedShape.push_back(shape[i]);
            squeezedStrides.push_back(strides[i]);
        }
    }
    return {squeezedShape, squeezedStrides};
}

long LayoutOperation::extractIndexValue(mlir::Value value) {
    if (auto folded = ConstantFolder::FoldIntValue(value)) {
        return *folded;
    }
    return mlir::ShapedType::kDynamic;
}

int64_t LayoutOperation::computeShapeProduct(llvm::ArrayRef<int64_t> shape) {
    int64_t product = 1;
    for (int64_t dim : shape) {
        if (dim == mlir::ShapedType::kDynamic) {
            return mlir::ShapedType::kDynamic;
        }
        product *= dim;
    }
    return product;
}

int64_t LayoutOperation::computeCoSize(llvm::ArrayRef<int64_t> shape,
                                       llvm::ArrayRef<int64_t> strides) {
    if (shape.empty() || shape.size() != strides.size()) {
        return mlir::ShapedType::kDynamic;
    }

    int64_t cosize = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == mlir::ShapedType::kDynamic ||
            strides[i] == mlir::ShapedType::kDynamic) {
            return mlir::ShapedType::kDynamic;
        }
        cosize += (shape[i] - 1) * std::abs(strides[i]);
    }
    return cosize;
}

// Extract values from potentially nested tuples (flattens the tuple)
bool LayoutOperation::flattenTupleValues(mlir::Value tupleValue,
                                         llvm::SmallVector<int64_t> &values) {
    if (auto makeTupleOp = tupleValue.getDefiningOp<cf::MakeIntTupleOp>()) {
        for (auto elem : makeTupleOp.getElements()) {
            // Recursively handle nested tuples
            if (elem.getDefiningOp<cf::MakeIntTupleOp>()) {
                if (!flattenTupleValues(elem, values)) {
                    return false;
                }
            } else {
                values.push_back(extractIndexValue(elem));
            }
        }
        return true;
    }

    // If it's not a MakeIntTupleOp, treat it as a single value
    values.push_back(extractIndexValue(tupleValue));
    return true;
}

LayoutOperation::StaticLayout LayoutOperation::coalesceStaticLayout(
    const llvm::SmallVector<int64_t> &shape,
    const llvm::SmallVector<int64_t> &strides) {
    if (shape.empty() || shape.size() != strides.size()) {
        return {shape, strides};
    }

    llvm::SmallVector<int64_t> coalescedShape;
    llvm::SmallVector<int64_t> coalescedStrides;
    int64_t currentShapeProduct = shape[0];
    int64_t currentStride = strides[0];

    for (size_t i = 1; i < shape.size(); ++i) {
        if (currentShapeProduct * currentStride == strides[i]) {
            currentShapeProduct *= shape[i];
        } else {
            coalescedShape.push_back(currentShapeProduct);
            coalescedStrides.push_back(currentStride);
            currentShapeProduct = shape[i];
            currentStride = strides[i];
        }
    }

    coalescedShape.push_back(currentShapeProduct);
    coalescedStrides.push_back(currentStride);
    return {coalescedShape, coalescedStrides};
}

LayoutOperation::StaticLayout LayoutOperation::computeComplementLayout(
    const llvm::SmallVector<int64_t> &shape,
    const llvm::SmallVector<int64_t> &strides, int64_t size) {
    if (shape.empty() || shape.size() != strides.size() || size <= 0) {
        return {{}, {}};
    }

    llvm::SmallVector<int64_t> remainingShape(shape.begin(), shape.end());
    llvm::SmallVector<int64_t> remainingStrides(strides.begin(), strides.end());
    llvm::SmallVector<int64_t> resultShape;
    llvm::SmallVector<int64_t> resultStrides = {1};

    while (remainingShape.size() > 1) {
        auto minStrideIt = std::min_element(remainingStrides.begin(),
                                            remainingStrides.end());
        size_t minIndex = std::distance(remainingStrides.begin(), minStrideIt);
        int64_t minStride = *minStrideIt;
        int64_t previousStride = resultStrides.back();
        if (minStride <= 0 || minStride % previousStride != 0) {
            return {{}, {}};
        }

        int64_t newShape = minStride / previousStride;
        int64_t newStride = minStride * remainingShape[minIndex];
        if (newShape <= 0) {
            return {{}, {}};
        }

        resultShape.push_back(newShape);
        resultStrides.push_back(newStride);
        remainingShape.erase(remainingShape.begin() + minIndex);
        remainingStrides.erase(remainingStrides.begin() + minIndex);
    }

    int64_t previousStride = resultStrides.back();
    if (remainingStrides.front() <= 0 ||
        remainingStrides.front() % previousStride != 0) {
        return {{}, {}};
    }
    int64_t lastShape = remainingStrides.front() / previousStride;
    if (lastShape <= 0) {
        return {{}, {}};
    }
    resultShape.push_back(lastShape);

    int64_t newStride = remainingStrides.front() * remainingShape.front();
    llvm::SmallVector<int64_t> flatShape(resultShape.begin(), resultShape.end());
    llvm::SmallVector<int64_t> flatStrides(resultStrides.begin(),
                                           resultStrides.end());
    int64_t restShape = ceilDivInt64(size, newStride);
    if (restShape > 0) {
        flatShape.push_back(restShape);
        flatStrides.push_back(newStride);
    }

    auto [squeezedShape, squeezedStrides] =
        squeezeStaticLayout(flatShape, flatStrides);
    return coalesceStaticLayout(squeezedShape, squeezedStrides);
}

LayoutOperation::StaticLayout LayoutOperation::concatenateStaticLayouts(
    const llvm::SmallVector<int64_t> &shapeA,
    const llvm::SmallVector<int64_t> &stridesA,
    const llvm::SmallVector<int64_t> &shapeB,
    const llvm::SmallVector<int64_t> &stridesB) {
    llvm::SmallVector<int64_t> resultShape(shapeA.begin(), shapeA.end());
    llvm::SmallVector<int64_t> resultStrides(stridesA.begin(), stridesA.end());
    resultShape.append(shapeB.begin(), shapeB.end());
    resultStrides.append(stridesB.begin(), stridesB.end());
    return {resultShape, resultStrides};
}

LayoutOperation::StaticLayout LayoutOperation::computeCompositionLayout(
    const llvm::SmallVector<int64_t> &shapeA,
    const llvm::SmallVector<int64_t> &stridesA,
    const llvm::SmallVector<int64_t> &shapeB,
    const llvm::SmallVector<int64_t> &stridesB) {
    if (shapeA.empty() || shapeB.empty() || shapeA.size() != stridesA.size() ||
        shapeB.size() != stridesB.size()) {
        return {{}, {}};
    }

    auto [coalescedShapeA, coalescedStridesA] =
        coalesceStaticLayout(shapeA, stridesA);

    llvm::SmallVector<int64_t> resultShape;
    llvm::SmallVector<int64_t> resultStrides;
    for (size_t i = 0; i < shapeB.size(); ++i) {
        int64_t rhsShape = shapeB[i];
        int64_t rhsStride = stridesB[i];
        if (rhsStride == 0) {
            resultShape.push_back(rhsShape);
            resultStrides.push_back(rhsStride);
            continue;
        }

        llvm::SmallVector<int64_t> subShape;
        llvm::SmallVector<int64_t> subStrides;
        if (coalescedShapeA.size() == 1) {
            subShape.push_back(rhsShape);
            subStrides.push_back(rhsStride * coalescedStridesA[0]);
        } else {
            int64_t restShape = rhsShape;
            int64_t restStride = rhsStride;
            for (size_t j = 0; j + 1 < coalescedShapeA.size(); ++j) {
                int64_t currShape = coalescedShapeA[j];
                int64_t currStride = coalescedStridesA[j];
                int64_t nextShape = ceilDivInt64(currShape, std::abs(restStride));
                int64_t nextStride =
                    ceilDivInt64(std::abs(restStride), currShape);
                if (restStride < 0) {
                    nextStride = -nextStride;
                }

                if (nextShape == 1 || restShape == 1) {
                    restStride = nextStride;
                    continue;
                }

                int64_t newShape = std::min(nextShape, restShape);
                if (newShape <= 0 || restShape % newShape != 0) {
                    return {{}, {}};
                }

                subShape.push_back(newShape);
                subStrides.push_back(restStride * currStride);
                restShape /= newShape;
                restStride = nextStride;
            }

            if (subShape.empty()) {
                subShape.push_back(restShape);
                subStrides.push_back(restStride * coalescedStridesA.back());
            } else if (restShape != 1) {
                subShape.push_back(restShape);
                subStrides.push_back(restStride * coalescedStridesA.back());
            }
        }

        resultShape.append(subShape.begin(), subShape.end());
        resultStrides.append(subStrides.begin(), subStrides.end());
    }

    auto [squeezedShape, squeezedStrides] =
        squeezeStaticLayout(resultShape, resultStrides);
    return {squeezedShape, squeezedStrides};
}

mlir::Value LayoutOperation::CastToIndex(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::Value value) {
    if (value.getType().isIndex()) {
        return value;
    }
    if (value.getType().isIntOrIndexOrFloat()) {
        return mlir::arith::IndexCastOp::create(builder, loc,
                                                builder.getIndexType(), value);
    }
    return value;
}

void LayoutOperation::CollectTupleLeaves(mlir::Value value,
                                         llvm::SmallVector<mlir::Value> &out) {
    if (auto tupleOp = value.getDefiningOp<cf::MakeIntTupleOp>()) {
        for (auto elem : tupleOp.getElements()) {
            CollectTupleLeaves(elem, out);
        }
        return;
    }
    out.push_back(value);
}

void LayoutOperation::ExpandLinearIndex(mlir::OpBuilder &builder,
                                        mlir::Location loc,
                                        mlir::Value linearIndex,
                                        llvm::ArrayRef<mlir::Value> dims,
                                        llvm::SmallVector<mlir::Value> &out) {
    if (dims.empty()) {
        return;
    }

    auto linear = CastToIndex(builder, loc, linearIndex);
    llvm::SmallVector<mlir::Value> basis;
    if (dims.size() > 1) {
        basis.reserve(dims.size() - 1);
        for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
            if (it == dims.rbegin()) {
                continue;
            }
            basis.push_back(CastToIndex(builder, loc, *it));
        }
    }

    auto delinearize = mlir::affine::AffineDelinearizeIndexOp::create(
        builder, loc, linear, basis, /*hasOuterBound=*/false);
    auto results = delinearize.getResults();
    for (auto index = results.size(); index > 0; --index) {
        out.push_back(results[index - 1]);
    }
}

bool LayoutOperation::ExpandIndicesForNestedLayout(
    mlir::OpBuilder &builder, mlir::Value memrefValue,
    llvm::SmallVector<mlir::Value> &indices) {
    auto memrefType = mlir::dyn_cast<cf::MemRefType>(memrefValue.getType());
    if (!memrefType || indices.empty()) {
        return false;
    }

    auto castOp = memrefValue.getDefiningOp<cf::AveLangMemRefCastOp>();
    if (!castOp || !castOp.getLayout()) {
        return false;
    }

    auto layoutOp = castOp.getLayout().getDefiningOp<cf::MakeLayoutOp>();
    if (!layoutOp) {
        return false;
    }

    auto dimsTuple = layoutOp.getDims().getDefiningOp<cf::MakeIntTupleOp>();
    if (!dimsTuple) {
        return false;
    }

    bool hasNested = false;
    for (auto elem : dimsTuple.getElements()) {
        if (elem.getDefiningOp<cf::MakeIntTupleOp>()) {
            hasNested = true;
            break;
        }
    }
    if (!hasNested) {
        return false;
    }

    size_t logicalRank = dimsTuple.getNumElements();
    if (indices.size() > logicalRank) {
        return false;
    }

    if (logicalRank == static_cast<size_t>(memrefType.getRank())) {
        return false;
    }

    mlir::Location loc = builder.getUnknownLoc();
    llvm::SmallVector<mlir::Value> expanded;
    expanded.reserve(indices.size());

    size_t idxPos = 0;
    for (auto elem : dimsTuple.getElements()) {
        if (idxPos >= indices.size()) {
            break;
        }
        auto idxVal = indices[idxPos++];
        if (elem.getDefiningOp<cf::MakeIntTupleOp>()) {
            llvm::SmallVector<mlir::Value> nestedDims;
            CollectTupleLeaves(elem, nestedDims);
            ExpandLinearIndex(builder, loc, idxVal, nestedDims, expanded);
        } else {
            expanded.push_back(idxVal);
        }
    }

    if (expanded.size() == indices.size()) {
        return false;
    }

    indices.swap(expanded);
    return true;
}

mlir::Value
LayoutOperation::createViewFunction(ast::Call *callExpr, GeneratorContext *ctx,
                                    llvm::ArrayRef<mlir::Value> resolvedArgs) {
    auto location =
        ctx->GetCurrentFunctionGenerator()->GetBuilder().getUnknownLoc();
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();

    const auto &args = callExpr->GetArgs();

    // Validate argument count - view(memref, dtype, layout)
    if (args.size() != 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "view() expects exactly 3 arguments: memref, dtype, layout";
        return nullptr;
    }

    if (resolvedArgs.size() < 3) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Failed to resolve arguments for view()";
        return nullptr;
    }

    // Get the memref to cast
    auto memrefValue = resolvedArgs[0];
    if (!memrefValue) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Failed to generate memref argument for view()";
        return nullptr;
    }

    // Verify it's a memref type
    auto memrefType = mlir::dyn_cast<cf::MemRefType>(memrefValue.getType());
    if (!memrefType) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "First argument to view() must be a memref";
        return nullptr;
    }

    // Resolve the element type from the second argument
    auto elementType = ctx->syms->ResolveBuiltinType(args[1]);
    if (!elementType || !elementType.isSignlessIntOrFloat()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Failed to resolve element type for view()";
        return nullptr;
    }

    // Get the layout from the third argument
    auto layoutValue = resolvedArgs[2];
    if (!layoutValue) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Failed to resolve layout argument for view()";
        return nullptr;
    }

    // Check if layout is created by make_layout.
    auto makeLayoutOp = layoutValue.getDefiningOp<cf::MakeLayoutOp>();
    if (!makeLayoutOp) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Third argument to view() must be a layout created by "
               "make_layout";
        return nullptr;
    }

    // Get dims and stride from the layout
    auto dimsValue = makeLayoutOp.getDims();
    auto strideValue = makeLayoutOp.getStride();

    auto dimsTupleOp = dimsValue.getDefiningOp<cf::MakeIntTupleOp>();
    auto strideTupleOp = strideValue.getDefiningOp<cf::MakeIntTupleOp>();

    if (!dimsTupleOp || !strideTupleOp) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Internal error: layout must contain valid dims and stride "
               "tuples";
        return nullptr;
    }

    // Validate that dims and stride have the same dimensions
    if (dimsTupleOp.getNumElements() != strideTupleOp.getNumElements()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Layout dims and stride must have the same number of dimensions";
        return nullptr;
    }

    if (dimsTupleOp.getNumElements() == 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Layout dims and stride cannot be empty";
        return nullptr;
    }

    // Process shapes to build static shape array
    llvm::SmallVector<int64_t> staticShape;
    llvm::SmallVector<int64_t> staticStrides;
    flattenTupleValues(dimsValue, staticShape); // flatten tuple
    flattenTupleValues(strideValue, staticStrides);
    if (staticShape.size() != staticStrides.size()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Layout dims and stride must have the same number of dimensions";
        return nullptr;
    }

    // Create the new memref type with the specified shape and element type
    auto newMemRefType = createAveLangMemRefType(
        builder.getContext(), staticShape, staticStrides, elementType,
        memrefType.getMemorySpace());

    // Create the new memref.cast op from the AveLang dialect
    auto castOp = cf::AveLangMemRefCastOp::create(
        builder, location, memrefValue, layoutValue, newMemRefType);
    if (auto typeInfo = GetTypeInfo(args[1]); typeInfo.is_unsigned_integer) {
        SetTypeInfo(castOp.getResult(), typeInfo);
    } else {
        SetTypeInfo(castOp.getResult(), GetTypeInfo(memrefValue));
    }

    return castOp.getResult();
}

mlir::Value LayoutOperation::createMakeLayoutFunction(
    ast::Call *callExpr, GeneratorContext *ctx,
    llvm::ArrayRef<mlir::Value> resolvedArgs) {
    auto location =
        ctx->GetCurrentFunctionGenerator()->GetBuilder().getUnknownLoc();
    auto &builder = ctx->GetCurrentFunctionGenerator()->GetBuilder();

    const auto &args = callExpr->GetArgs();

    // Validate argument count - make_layout(dims, stride)
    if (args.size() != 2) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "make_layout() expects exactly 2 arguments: dims and stride";
        return nullptr;
    }

    if (resolvedArgs.size() < 2 || !resolvedArgs[0] || !resolvedArgs[1]) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Failed to resolve arguments for make_layout()";
        return nullptr;
    }

    auto dimsValue = resolvedArgs[0];
    auto strideValue = resolvedArgs[1];

    // Get the dims and stride as tuples
    auto dimsTupleOp = dimsValue.getDefiningOp<cf::MakeIntTupleOp>();
    auto strideTupleOp = strideValue.getDefiningOp<cf::MakeIntTupleOp>();

    if (!dimsTupleOp || !strideTupleOp) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Dims and stride arguments to make_layout() must be integer "
               "tuples";
        return nullptr;
    }

    // Validate that dims and stride have the same number of elements
    if (dimsTupleOp.getNumElements() != strideTupleOp.getNumElements()) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Dims and stride tuples must have the same number of elements";
        return nullptr;
    }

    if (dimsTupleOp.getNumElements() == 0) {
        ctx->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                        callExpr->GetSourceRange().getBegin())
            << "Dims and stride tuples cannot be empty";
        return nullptr;
    }

    auto makeLayoutOp =
        cf::MakeLayoutOp::create(builder, location, dimsTupleOp, strideTupleOp);

    return makeLayoutOp.getResult();
}

} // namespace causalflow::avelang::ir
