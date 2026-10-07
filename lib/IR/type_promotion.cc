#include "type_promotion.h"

#include "type_system.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinOps.h>
#pragma clang diagnostic pop

namespace causalflow::avelang::ir {

namespace {

mlir::Value CreateTypeInfoCast(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value value, const TypeInfo &typeInfo) {
    auto cast = mlir::UnrealizedConversionCastOp::create(
        builder, loc, mlir::TypeRange{value.getType()},
        mlir::ValueRange{value});
    SetTypeInfo(cast.getResult(0), typeInfo);
    return cast.getResult(0);
}

} // namespace

mlir::Value CreateTypeConversion(mlir::Value value, mlir::Type source_type,
                                 mlir::Type target_type,
                                 mlir::Location location,
                                 mlir::OpBuilder &builder, bool allow_demotion,
                                 std::optional<bool> source_unsigned,
                                 std::optional<bool> target_unsigned) {
    if (!source_unsigned) {
        source_unsigned = GetTypeInfo(value).is_unsigned_integer;
    }

    // If types are the same, only signedness may need retagging.
    if (source_type == target_type) {
        if (target_unsigned && source_unsigned &&
            *target_unsigned != *source_unsigned &&
            IsIntegerValueOrVectorOfIntegers(target_type)) {
            return CreateTypeInfoCast(builder, location, value,
                                      TypeInfo{*target_unsigned});
        }
        return value;
    }

    auto source_vector = mlir::dyn_cast<mlir::VectorType>(source_type);
    auto target_vector = mlir::dyn_cast<mlir::VectorType>(target_type);
    if (source_vector || target_vector) {
        if (!source_vector || !target_vector ||
            source_vector.getShape() != target_vector.getShape() ||
            source_vector.getScalableDims() !=
                target_vector.getScalableDims()) {
            return nullptr;
        }
    }
    auto source_element_type =
        source_vector ? source_vector.getElementType() : source_type;
    auto target_element_type =
        target_vector ? target_vector.getElementType() : target_type;

    // Integer to integer conversions
    if (source_element_type.isInteger() && target_element_type.isInteger()) {
        auto source_width = source_element_type.getIntOrFloatBitWidth();
        auto target_width = target_element_type.getIntOrFloatBitWidth();

        if (source_width < target_width) {
            // Extend by source signedness, but tag the result by target dtype.
            auto isUnsigned = source_unsigned.value_or(false);
            if (isUnsigned) {
                auto result = mlir::arith::ExtUIOp::create(builder, location,
                                                           target_type, value);
                SetTypeInfo(result.getResult(),
                            TypeInfo{target_unsigned.value_or(true)});
                return result;
            } else {
                auto result = mlir::arith::ExtSIOp::create(builder, location,
                                                           target_type, value);
                SetTypeInfo(result.getResult(),
                            TypeInfo{target_unsigned.value_or(false)});
                return result;
            }
        } else if (source_width > target_width) {
            if (allow_demotion) {
                auto result = mlir::arith::TruncIOp::create(builder, location,
                                                            target_type, value);
                SetTypeInfo(result.getResult(),
                            TypeInfo{target_unsigned.value_or(
                                source_unsigned.value_or(false))});
                return result;
            } else {
                return value;
            }
        } else if (target_unsigned && source_unsigned &&
                   *target_unsigned != *source_unsigned) {
            return CreateTypeInfoCast(builder, location, value,
                                      TypeInfo{*target_unsigned});
        }
    }

    // Integer to float conversions
    if (source_element_type.isInteger() &&
        mlir::isa<mlir::FloatType>(target_element_type)) {
        if (source_unsigned.value_or(false)) {
            return mlir::arith::UIToFPOp::create(builder, location, target_type,
                                                 value);
        } else {
            return mlir::arith::SIToFPOp::create(builder, location, target_type,
                                                 value);
        }
    }

    // Float to integer conversions
    if (mlir::isa<mlir::FloatType>(source_element_type) &&
        target_element_type.isInteger()) {
        if (allow_demotion) {
            auto isUnsigned = target_unsigned.value_or(false);
            if (isUnsigned) {
                auto result = mlir::arith::FPToUIOp::create(builder, location,
                                                            target_type, value);
                SetTypeInfo(result.getResult(), TypeInfo{true});
                return result;
            } else {
                auto result = mlir::arith::FPToSIOp::create(builder, location,
                                                            target_type, value);
                SetTypeInfo(result.getResult(), TypeInfo{false});
                return result;
            }
        } else {
            return value;
        }
    }

    // Float to float conversions
    if (mlir::isa<mlir::FloatType>(source_element_type) &&
        mlir::isa<mlir::FloatType>(target_element_type)) {
        auto source_width = source_element_type.getIntOrFloatBitWidth();
        auto target_width = target_element_type.getIntOrFloatBitWidth();

        if (source_width < target_width) {
            return mlir::arith::ExtFOp::create(builder, location, target_type,
                                               value);
        } else if (source_width > target_width) {
            if (allow_demotion) {
                return mlir::arith::TruncFOp::create(builder, location,
                                                     target_type, value);
            } else {
                return value;
            }
        } else {
            // Same-width format changes (e.g. f16 <-> bf16) require ConvertFOp.
            return mlir::arith::ConvertFOp::create(
                builder, location, target_type, value, nullptr, nullptr);
        }
    }

    // Index type conversions
    if (source_element_type.isIndex() && target_element_type.isInteger()) {
        auto result = mlir::arith::IndexCastOp::create(builder, location,
                                                       target_type, value);
        SetTypeInfo(result.getResult(),
                    TypeInfo{target_unsigned.value_or(false)});
        return result;
    }
    if (source_element_type.isInteger() && target_element_type.isIndex()) {
        return mlir::arith::IndexCastOp::create(builder, location, target_type,
                                                value);
    }

    // Unsupported conversion - return nullptr to indicate failure
    return nullptr;
}

std::pair<mlir::Value, mlir::Value>
ConvertTypesForArithmetic(mlir::Value lhs, mlir::Value rhs,
                          mlir::OpBuilder &builder, mlir::Location location) {
    auto lhs_type = lhs.getType();
    auto rhs_type = rhs.getType();
    auto lhs_unsigned = GetTypeInfo(lhs).is_unsigned_integer;
    auto rhs_unsigned = GetTypeInfo(rhs).is_unsigned_integer;
    auto common_unsigned =
        lhs_unsigned.value_or(false) || rhs_unsigned.value_or(false);

    if (lhs_type == rhs_type) {
        if (lhs_type.isInteger() && lhs_unsigned != rhs_unsigned) {
            if (lhs_unsigned.value_or(false) != common_unsigned) {
                lhs = CreateTypeInfoCast(builder, location, lhs,
                                         TypeInfo{common_unsigned});
            }
            if (rhs_unsigned.value_or(false) != common_unsigned) {
                rhs = CreateTypeInfoCast(builder, location, rhs,
                                         TypeInfo{common_unsigned});
            }
        }
        return {lhs, rhs};
    }

    if (lhs_type.isIndex() && rhs_type.isInteger()) {
        lhs =
            mlir::arith::IndexCastOp::create(builder, location, rhs_type, lhs);
        SetTypeInfo(lhs, TypeInfo{common_unsigned});
        return {lhs, rhs};
    } else if (rhs_type.isIndex() && lhs_type.isInteger()) {
        rhs =
            mlir::arith::IndexCastOp::create(builder, location, lhs_type, rhs);
        SetTypeInfo(rhs, TypeInfo{common_unsigned});
        return {lhs, rhs};
    }

    if (lhs_type.isInteger() && rhs_type.isInteger()) {
        auto lhs_width = lhs_type.getIntOrFloatBitWidth();
        auto rhs_width = rhs_type.getIntOrFloatBitWidth();
        auto common_type = lhs_width < rhs_width ? rhs_type : lhs_type;

        if (lhs_width < rhs_width) {
            lhs =
                CreateTypeConversion(lhs, lhs_type, rhs_type, location, builder,
                                     false, lhs_unsigned, common_unsigned);
            return {lhs, rhs};
        } else if (rhs_width < lhs_width) {
            rhs =
                CreateTypeConversion(rhs, rhs_type, lhs_type, location, builder,
                                     false, rhs_unsigned, common_unsigned);
            return {lhs, rhs};
        }

        if (lhs_unsigned.value_or(false) != common_unsigned) {
            lhs = CreateTypeConversion(lhs, lhs_type, common_type, location,
                                       builder, false, lhs_unsigned,
                                       common_unsigned);
        }
        if (rhs_unsigned.value_or(false) != common_unsigned) {
            rhs = CreateTypeConversion(rhs, rhs_type, common_type, location,
                                       builder, false, rhs_unsigned,
                                       common_unsigned);
            return {lhs, rhs};
        }
    }

    if (mlir::isa<mlir::FloatType>(lhs_type) &&
        mlir::isa<mlir::FloatType>(rhs_type)) {
        auto lhs_width = lhs_type.getIntOrFloatBitWidth();
        auto rhs_width = rhs_type.getIntOrFloatBitWidth();

        if (lhs_width < rhs_width) {
            lhs = mlir::arith::ExtFOp::create(builder, location, rhs_type, lhs);
            return {lhs, rhs};
        } else if (rhs_width < lhs_width) {
            rhs = mlir::arith::ExtFOp::create(builder, location, lhs_type, rhs);
            return {lhs, rhs};
        }
    }

    if (lhs_type.isInteger() && mlir::isa<mlir::FloatType>(rhs_type)) {
        lhs = CreateTypeConversion(lhs, lhs_type, rhs_type, location, builder,
                                   false, lhs_unsigned);
        return {lhs, rhs};
    } else if (mlir::isa<mlir::FloatType>(lhs_type) && rhs_type.isInteger()) {
        rhs = CreateTypeConversion(rhs, rhs_type, lhs_type, location, builder,
                                   false, rhs_unsigned);
        return {lhs, rhs};
    }

    return {nullptr, nullptr};
}

std::pair<mlir::Value, mlir::Value>
ConvertTypesForBitwise(mlir::Value lhs, mlir::Value rhs,
                       mlir::OpBuilder &builder, mlir::Location location) {
    auto lhs_type = lhs.getType();
    auto rhs_type = rhs.getType();
    auto lhs_unsigned = GetTypeInfo(lhs).is_unsigned_integer;
    auto rhs_unsigned = GetTypeInfo(rhs).is_unsigned_integer;
    bool common_unsigned =
        lhs_unsigned.value_or(false) || rhs_unsigned.value_or(false);

    if (lhs_type == rhs_type) {
        if (lhs_type.isInteger() && lhs_unsigned != rhs_unsigned) {
            if (lhs_unsigned.value_or(false) != common_unsigned) {
                lhs = CreateTypeInfoCast(builder, location, lhs,
                                         TypeInfo{common_unsigned});
            }
            if (rhs_unsigned.value_or(false) != common_unsigned) {
                rhs = CreateTypeInfoCast(builder, location, rhs,
                                         TypeInfo{common_unsigned});
            }
        }
        return {lhs, rhs};
    }

    if (lhs_type.isIndex() && rhs_type.isInteger()) {
        lhs =
            mlir::arith::IndexCastOp::create(builder, location, rhs_type, lhs);
        SetTypeInfo(lhs, TypeInfo{common_unsigned});
        return {lhs, rhs};
    } else if (rhs_type.isIndex() && lhs_type.isInteger()) {
        rhs =
            mlir::arith::IndexCastOp::create(builder, location, lhs_type, rhs);
        SetTypeInfo(rhs, TypeInfo{common_unsigned});
        return {lhs, rhs};
    }

    if (lhs_type.isInteger() && rhs_type.isInteger()) {
        auto lhs_width = lhs_type.getIntOrFloatBitWidth();
        auto rhs_width = rhs_type.getIntOrFloatBitWidth();

        if (lhs_width < rhs_width) {
            lhs =
                mlir::arith::ExtUIOp::create(builder, location, rhs_type, lhs);
            SetTypeInfo(lhs, TypeInfo{common_unsigned});
            return {lhs, rhs};
        } else if (rhs_width < lhs_width) {
            rhs =
                mlir::arith::ExtUIOp::create(builder, location, lhs_type, rhs);
            SetTypeInfo(rhs, TypeInfo{common_unsigned});
            return {lhs, rhs};
        }

        if (lhs_unsigned.value_or(false) != common_unsigned) {
            lhs = CreateTypeInfoCast(builder, location, lhs,
                                     TypeInfo{common_unsigned});
        }
        if (rhs_unsigned.value_or(false) != common_unsigned) {
            rhs = CreateTypeInfoCast(builder, location, rhs,
                                     TypeInfo{common_unsigned});
            return {lhs, rhs};
        }
    }

    return {nullptr, nullptr};
}

} // namespace causalflow::avelang::ir
