#include "AveLangOps.h"
#include "AveLangDialect.h"
#include "IR/Intrinsics/amdgpu_mfma_signatures.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Ptr/IR/PtrTypes.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <optional>
#include <string>

namespace causalflow::avelang::dialect {
namespace amdgpu_mfma = causalflow::avelang::amdgpu::mfma;

static bool isAveLangMemRefType(mlir::Type type) {
    return mlir::isa<MemRefType>(type);
}

namespace {

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

mlir::LogicalResult EnsureRank1Vector(mlir::Operation *op, mlir::VectorType vec,
                                      llvm::StringRef label) {
    if (vec.getRank() != 1) {
        op->emitOpError() << label << " must be a 1-D vector";
        return mlir::failure();
    }
    return mlir::success();
}

bool MatchesVectorType(mlir::VectorType vec, const VectorTypePattern &pattern) {
    if (vec.getRank() != 1 || vec.getNumElements() != pattern.elements) {
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

bool MatchesAnyVectorType(mlir::VectorType vec,
                          llvm::ArrayRef<VectorTypePattern> patterns) {
    for (const auto &pattern : patterns) {
        if (MatchesVectorType(vec, pattern)) {
            return true;
        }
    }
    return false;
}

struct NvvMmaSignature {
    llvm::StringRef name;
    llvm::ArrayRef<VectorTypePattern> aTypes;
    llvm::ArrayRef<VectorTypePattern> bTypes;
    llvm::ArrayRef<VectorTypePattern> cTypes;
    llvm::ArrayRef<VectorTypePattern> rTypes;
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
static const VectorTypePattern kNvvmMma16x8x16R[] = {
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
static const VectorTypePattern kNvvmMma16x8x8R[] = {
    {VectorElemKind::F32, 4},
};

static const NvvMmaSignature kNvvmMmaSignatures[] = {
    {
        "mma_16x8x16_f16_f16",
        llvm::ArrayRef(kNvvmMma16x8x16A),
        llvm::ArrayRef(kNvvmMma16x8x16B),
        llvm::ArrayRef(kNvvmMma16x8x16C),
        llvm::ArrayRef(kNvvmMma16x8x16R),
    },
    {
        "mma_16x8x8_f16_f32",
        llvm::ArrayRef(kNvvmMma16x8x8A),
        llvm::ArrayRef(kNvvmMma16x8x8B),
        llvm::ArrayRef(kNvvmMma16x8x8C),
        llvm::ArrayRef(kNvvmMma16x8x8R),
    },
};

static std::string BuildAmdgpuMfmaSignatureList() {
    std::string list;
    for (const auto &cfg : amdgpu_mfma::MFMAConfig::GetConfigs()) {
        if (!list.empty()) {
            list += ", ";
        }
        list += cfg.name.str();
    }
    return list;
}

static std::optional<int64_t> getElementByteSize(mlir::Type elementType) {
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

static std::optional<int64_t> getStaticTotalByteSize(MemRefType memrefType) {
    auto elementBytes = getElementByteSize(memrefType.getElementType());
    if (!elementBytes) {
        return std::nullopt;
    }

    int64_t elementCount = 1;
    for (auto dim : memrefType.getShape()) {
        if (mlir::ShapedType::isDynamic(dim) || dim < 0) {
            return std::nullopt;
        }
        elementCount *= dim;
    }

    return elementCount * (*elementBytes);
}

static int64_t countTupleElements(mlir::Value value) {
    if (auto tupleOp = value.getDefiningOp<MakeIntTupleOp>()) {
        int64_t count = 0;
        for (auto elem : tupleOp.getElements()) {
            count += countTupleElements(elem);
        }
        return count;
    }
    return 1;
}

} // namespace

// Custom build method
void MakeIntTupleOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state,
                           mlir::ValueRange elements) {
    // Set the result type to NoneType (as a placeholder for tuple type)
    state.addTypes(builder.getNoneType());

    // Add all elements as operands
    state.addOperands(elements);

    // Add an attribute to mark this as an ave-lang tuple
    state.addAttribute("is_tuple", builder.getBoolAttr(true));
}

// Custom verify method
mlir::LogicalResult MakeIntTupleOp::verify() {
    // This operation also carries Python tuple values for multi-result calls.
    // Shape/layout consumers validate their integer-only requirements.
    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AveLangMemRefLoadOp
//===----------------------------------------------------------------------===//

void AveLangMemRefLoadOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
    effects.emplace_back(mlir::MemoryEffects::Read::get(),
                         mlir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// AveLangMemRefLoadVecOp
//===----------------------------------------------------------------------===//

void AveLangMemRefLoadVecOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
    effects.emplace_back(mlir::MemoryEffects::Read::get(),
                         mlir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// AveLangMemRefStoreOp
//===----------------------------------------------------------------------===//

void AveLangMemRefStoreOp::getEffects(
    llvm::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
    effects.emplace_back(mlir::MemoryEffects::Write::get(),
                         mlir::SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// MakeLayoutOp
//===----------------------------------------------------------------------===//

void MakeLayoutOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Value dims, mlir::Value stride) {
    state.addOperands({dims, stride});
    // FIXME: Use OpaqueType to represent the layout type
    auto layoutType =
        mlir::OpaqueType::get(builder.getStringAttr("ave"), "layout");
    state.addTypes(layoutType);
}

mlir::LogicalResult MakeLayoutOp::verify() {
    // Verify that dims and stride are tuples (MakeIntTupleOp)
    if (!getDims().getDefiningOp<MakeIntTupleOp>()) {
        return emitOpError("dims must be a tuple created by make_int_tuple");
    }
    if (!getStride().getDefiningOp<MakeIntTupleOp>()) {
        return emitOpError("stride must be a tuple created by make_int_tuple");
    }

    auto dimsTuple = getDims().getDefiningOp<MakeIntTupleOp>();
    auto strideTuple = getStride().getDefiningOp<MakeIntTupleOp>();

    // Verify that dims and stride have the same number of elements
    if (dimsTuple.getNumElements() != strideTuple.getNumElements()) {
        return emitOpError(
            "dims and stride must have the same number of elements");
    }

    // Verify that dims and stride are not empty
    if (dimsTuple.getNumElements() == 0) {
        return emitOpError("dims and stride cannot be empty");
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AveLangMemRefCastOp
//===----------------------------------------------------------------------===//

void AveLangMemRefCastOp::build(mlir::OpBuilder &builder,
                                mlir::OperationState &state, mlir::Value source,
                                mlir::Type resultType) {
    state.addOperands({source});
    state.addTypes(resultType);
}

void AveLangMemRefCastOp::build(mlir::OpBuilder &builder,
                                mlir::OperationState &state, mlir::Value source,
                                mlir::Value layout, mlir::Type resultType) {
    state.addOperands({source, layout});
    state.addTypes(resultType);
}

mlir::LogicalResult AveLangMemRefCastOp::verify() {
    auto sourceType = getSource().getType();
    auto sourceMemref = mlir::dyn_cast<MemRefType>(sourceType);
    auto sourceBuiltinMemref = mlir::dyn_cast<mlir::MemRefType>(sourceType);
    auto sourcePtr = mlir::dyn_cast<mlir::ptr::PtrType>(sourceType);
    auto resultType = getResult().getType();
    auto resultMemref = mlir::dyn_cast<MemRefType>(resultType);
    auto resultBuiltinMemref = mlir::dyn_cast<mlir::MemRefType>(resultType);
    if ((!sourceMemref && !sourceBuiltinMemref && !sourcePtr) ||
        (!resultMemref && !resultBuiltinMemref)) {
        return emitOpError("source must be a memref or ptr type; result must "
                           "be a memref type");
    }

    auto resultRank =
        resultMemref ? resultMemref.getRank() : resultBuiltinMemref.getRank();

    bool hasLayout = false;
    if (auto layoutValue = getLayout()) {
        hasLayout = true;
        if (!layoutValue.getDefiningOp<MakeLayoutOp>()) {
            return emitOpError("layout must be a value created by make_layout");
        }

        auto layoutOp = layoutValue.getDefiningOp<MakeLayoutOp>();
        auto dimsValue = layoutOp.getDims();
        auto strideValue = layoutOp.getStride();

        auto dimsTuple = dimsValue.getDefiningOp<MakeIntTupleOp>();
        auto strideTuple = strideValue.getDefiningOp<MakeIntTupleOp>();

        if (!dimsTuple || !strideTuple) {
            return emitOpError(
                "layout must contain valid dims and stride tuples");
        }

        int64_t dimsCount = countTupleElements(dimsValue);
        int64_t strideCount = countTupleElements(strideValue);
        if (dimsCount != strideCount) {
            return emitOpError(
                "layout dims and stride must have the same number of elements");
        }

        if (dimsCount != resultRank) {
            return emitOpError("layout dims must match the result memref rank");
        }
    }

    if (!hasLayout && sourceMemref && resultMemref) {
        auto sourceBytes = getStaticTotalByteSize(sourceMemref);
        auto resultBytes = getStaticTotalByteSize(resultMemref);
        if (!sourceBytes || !resultBytes) {
            return emitOpError(
                "layout is required when casting dynamic shapes or unsupported "
                "element types");
        }
        if (*sourceBytes != *resultBytes) {
            return emitOpError("The source memref and target type must have "
                               "the same total byte size");
        }
    }

    if (!hasLayout && sourcePtr) {
        if (resultMemref) {
            auto resultBytes = getStaticTotalByteSize(resultMemref);
            if (!resultBytes) {
                return emitOpError(
                    "layout is required when casting from ptr with dynamic "
                    "shapes or unsupported element types");
            }
        }
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// GPU Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// NVVMMMAOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult NVVMMMAOp::verify() {
    auto aVector = mlir::dyn_cast<mlir::VectorType>(getA().getType());
    auto bVector = mlir::dyn_cast<mlir::VectorType>(getB().getType());
    auto cVector = mlir::dyn_cast<mlir::VectorType>(getC().getType());
    auto resultVector = mlir::dyn_cast<mlir::VectorType>(getResult().getType());

    if (!aVector || !bVector || !cVector) {
        return emitOpError("all operands must be vector types");
    }
    if (!resultVector) {
        return emitOpError("result must be a vector type");
    }

    if (mlir::failed(EnsureRank1Vector(getOperation(), aVector, "A")) ||
        mlir::failed(EnsureRank1Vector(getOperation(), bVector, "B")) ||
        mlir::failed(EnsureRank1Vector(getOperation(), cVector, "C")) ||
        mlir::failed(
            EnsureRank1Vector(getOperation(), resultVector, "result"))) {
        return mlir::failure();
    }

    for (const auto &sig : kNvvmMmaSignatures) {
        if (MatchesAnyVectorType(aVector, sig.aTypes) &&
            MatchesAnyVectorType(bVector, sig.bTypes) &&
            MatchesAnyVectorType(cVector, sig.cTypes) &&
            MatchesAnyVectorType(resultVector, sig.rTypes)) {
            return mlir::success();
        }
    }

    return emitOpError(
        "operand types do not match any supported NVVM MMA signature; "
        "supported: mma_16x8x16_f16_f16, mma_16x8x8_f16_f32");
}

//===----------------------------------------------------------------------===//
// NVVMLdMatrixOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult NVVMLdMatrixOp::verify() {
    // Verify memref operand
    if (!isAveLangMemRefType(getMemref().getType())) {
        return emitOpError("memref operand must be an ave-lang memref type");
    }

    // Verify matrix shape is valid
    auto shape = getMatrixShape();
    if (shape != "m8n8" && shape != "m16n16") {
        return emitOpError("unsupported matrix shape: " + shape);
    }

    // Verify matrix num is valid (1, 2, 4)
    auto num = getMatrixNum();
    if (num != 1 && num != 2 && num != 4) {
        return emitOpError("matrix num must be 1, 2, or 4");
    }

    // Verify bit width (8, 16)
    auto bitWidth = getMatrixBitWidth();
    if (bitWidth != 8 && bitWidth != 16) {
        return emitOpError("matrix bit width must be 8 or 16");
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// NVVMStMatrixOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult NVVMStMatrixOp::verify() {
    // Verify memref operand
    if (!isAveLangMemRefType(getMemref().getType())) {
        return emitOpError("memref operand must be an ave-lang memref type");
    }

    // Verify matrix shape is valid
    auto shape = getMatrixShape();
    if (shape != "m8n8" && shape != "m16n16") {
        return emitOpError("unsupported matrix shape: " + shape);
    }

    // Verify matrix num is valid (1, 2, 4)
    auto num = getMatrixNum();
    if (num != 1 && num != 2 && num != 4) {
        return emitOpError("matrix num must be 1, 2, or 4");
    }

    // Verify bit width (8, 16)
    auto bitWidth = getMatrixBitWidth();
    if (bitWidth != 8 && bitWidth != 16) {
        return emitOpError("matrix bit width must be 8 or 16");
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPUMfmaOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AMDGPUMfmaOp::verify() {
    auto aVector = mlir::dyn_cast<mlir::VectorType>(getA().getType());
    auto bVector = mlir::dyn_cast<mlir::VectorType>(getB().getType());
    auto cVector = mlir::dyn_cast<mlir::VectorType>(getC().getType());
    auto resultVector = mlir::dyn_cast<mlir::VectorType>(getResult().getType());

    if (!aVector || !bVector || !cVector) {
        return emitOpError("all operands must be vector types");
    }
    if (!resultVector) {
        return emitOpError("result must be a vector type");
    }

    if (mlir::failed(EnsureRank1Vector(getOperation(), aVector, "A")) ||
        mlir::failed(EnsureRank1Vector(getOperation(), bVector, "B")) ||
        mlir::failed(EnsureRank1Vector(getOperation(), cVector, "C")) ||
        mlir::failed(
            EnsureRank1Vector(getOperation(), resultVector, "result"))) {
        return mlir::failure();
    }

    auto typeAName = getTypeAAttr().getValue();
    auto typeCName = getTypeCAttr().getValue();

    const amdgpu_mfma::MFMAConfig *config = amdgpu_mfma::MFMAConfig::Find(
        getMAttr().getInt(), getNAttr().getInt(), getKAttr().getInt(),
        typeAName, typeCName);

    if (!config) {
        return emitOpError("unsupported MFMA configuration; supported: " +
                           BuildAmdgpuMfmaSignatureList());
    }

    if (!config->MatchesAType(aVector) || !config->MatchesBType(bVector) ||
        !config->MatchesCType(cVector) || !config->MatchesCType(resultVector)) {
        return emitOpError("operand types do not match MFMA signature")
               << " " << config->name;
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPURawBufferLoadOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AMDGPURawBufferLoadOp::verify() {
    // Verify result type (can be scalar or vector)
    auto resultType = getResult().getType();
    if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(resultType)) {
        if (vectorType.getRank() != 1) {
            return emitOpError("result vector must be 1-dimensional");
        }
        auto numElements = vectorType.getNumElements();
        if (numElements != 1 && numElements != 2 && numElements != 4) {
            return emitOpError("result vector must have 1, 2 or 4 elements");
        }
    }

    // All operands should be integer types for buffer addressing
    for (auto operand : {getVindex(), getSoffset(), getAux()}) {
        if (!operand.getType().isIntOrIndex()) {
            return emitOpError(
                "buffer operands must be integer or index types");
        }
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// AMDGPURawBufferStoreOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult AMDGPURawBufferStoreOp::verify() {
    // Verify data type (can be scalar or vector)
    auto dataType = getData().getType();
    if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(dataType)) {
        if (vectorType.getRank() != 1) {
            return emitOpError("data vector must be 1-dimensional");
        }
        auto numElements = vectorType.getNumElements();
        if (numElements != 1 && numElements != 2 && numElements != 4) {
            return emitOpError("data vector must have 1, 2 or 4 elements");
        }
    }

    // All other operands should be integer types for buffer addressing
    for (auto operand : {getVindex(), getSoffset(), getAux()}) {
        if (!operand.getType().isIntOrIndex()) {
            return emitOpError(
                "buffer operands must be integer or index types");
        }
    }

    return mlir::success();
}

} // namespace causalflow::avelang::dialect

// Include the generated definitions
#define GET_OP_CLASSES
#include "AveLangOps.cpp.inc"

// Manual implementations for missing generic build methods
namespace causalflow::avelang::dialect {

// NVVMMMAOp build method
void NVVMMMAOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      mlir::ValueRange operands, mlir::TypeRange resultTypes,
                      mlir::ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() == 3u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    assert(resultTypes.size() == 1u && "mismatched number of return types");
    state.addTypes(resultTypes);
}

// NVVMLdMatrixOp build method - This is already generated, but let's ensure
// it's correct (This method already exists in the generated code)

// NVVMStMatrixOp build method - This is already generated, but let's ensure
// it's correct (This method already exists in the generated code)

// AMDGPUMfmaOp build method
void AMDGPUMfmaOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::ValueRange operands, mlir::TypeRange resultTypes,
                         mlir::ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() == 3u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    assert(resultTypes.size() == 1u && "mismatched number of return types");
    state.addTypes(resultTypes);
}

// NVVMLdMatrixOp build method
void NVVMLdMatrixOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state,
                           mlir::ValueRange operands,
                           mlir::TypeRange resultTypes,
                           mlir::ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() == 1u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    assert(resultTypes.size() == 1u && "mismatched number of return types");
    state.addTypes(resultTypes);
}

// NVVMStMatrixOp build method
void NVVMStMatrixOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state,
                           mlir::ValueRange operands,
                           mlir::ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() == 2u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    // No result types to add (void operation)
}


void NVVMTMADescriptorOp::build(mlir::OpBuilder &builder,
                                mlir::OperationState &state,
                                mlir::ValueRange operands,
                                mlir::TypeRange resultTypes,
                                mlir::ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() == 2u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    assert(resultTypes.size() == 1u && "mismatched number of return types");
    state.addTypes(resultTypes);
}
// AMDGPURawBufferLoadOp build method
void AMDGPURawBufferLoadOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state,
    mlir::ValueRange operands, mlir::TypeRange resultTypes,
    mlir::ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() == 4u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    assert(resultTypes.size() == 1u && "mismatched number of return types");
    state.addTypes(resultTypes);
}

// AMDGPURawBufferStoreOp build method
void AMDGPURawBufferStoreOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state,
    mlir::ValueRange operands,
    mlir::ArrayRef<mlir::NamedAttribute> attributes) {
    assert(operands.size() == 5u && "mismatched number of parameters");
    state.addOperands(operands);
    state.addAttributes(attributes);
    // No result types to add (void operation)
}

//===----------------------------------------------------------------------===//
// FullOp
//===----------------------------------------------------------------------===//

// FullOp build method
void FullOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value shape, mlir::Value value,
                   mlir::Type resultType) {
    state.addOperands({shape, value});
    state.addTypes(resultType);
}

// FullOp verify method
mlir::LogicalResult FullOp::verify() {
    // Verify that the result type is a memref
    auto resultInfo = mlir::dyn_cast<MemRefType>(getResult().getType());
    if (!resultInfo) {
        return emitOpError("result must be an ave-lang memref type");
    }

    // Verify that the value type matches the memref element type
    if (auto result_element_type =
            mlir::dyn_cast<mlir::VectorType>(resultInfo.getElementType())) {
        auto valueType = getValue().getType();
        if (valueType != result_element_type &&
            valueType != result_element_type.getElementType()) {
            return emitOpError(
                "fill value type must match memref element type");
        }
    } else if (getValue().getType() != resultInfo.getElementType()) {
        return emitOpError("fill value type must match memref element type");
    }

    return mlir::success();
}

} // namespace causalflow::avelang::dialect
