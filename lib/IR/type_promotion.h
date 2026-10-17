#pragma once

#include <optional>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#pragma clang diagnostic pop

#include <utility>

namespace causalflow::avelang::ir {

/// Scalar type conversion function with configurable demotion policy.
/// This function can perform both promotions and demotions based on the
/// allow_demotion flag.
///
/// @param value The input value to convert
/// @param source_type The current type of the value
/// @param target_type The desired output type
/// @param location The source location for error reporting
/// @param builder The MLIR builder to use for creating conversion operations
/// @param allow_demotion If true, allows demoting conversions (truncation,
/// float->int).
///                       If false, only performs safe promotions to prevent
///                       data loss.
/// @return The converted value, or the original value if conversion would cause
/// demotion
///         when allow_demotion=false, or nullptr if conversion is not supported
mlir::Value
CreateTypeConversion(mlir::Value value, mlir::Type source_type,
                     mlir::Type target_type, mlir::Location location,
                     mlir::OpBuilder &builder, bool allow_demotion = false,
                     std::optional<bool> source_unsigned = std::nullopt,
                     std::optional<bool> target_unsigned = std::nullopt);

/// Explicit numeric conversion for scalars and same-shaped vectors.
mlir::Value CreateExplicitTypeConversion(
    mlir::Value value, mlir::Type source_type, mlir::Type result_type,
    mlir::Location location, mlir::OpBuilder &builder,
    std::optional<bool> source_unsigned = std::nullopt,
    std::optional<bool> target_unsigned = std::nullopt);

/// Converts two values to compatible types for arithmetic operations.
/// This function promotes both operands to a common type following these rules:
/// - Integer types are promoted to the wider integer type
/// - Float types are promoted to the wider float type
/// - Integer operands are promoted to float when mixed with float operands
/// - Index types are converted to integer types
///
/// @param lhs The left-hand side operand
/// @param rhs The right-hand side operand
/// @param builder The MLIR builder to use for creating conversion operations
/// @return A pair of converted values, or {nullptr, nullptr} if conversion
/// fails
std::pair<mlir::Value, mlir::Value>
ConvertTypesForArithmetic(mlir::Value lhs, mlir::Value rhs,
                          mlir::OpBuilder &builder, mlir::Location location);

/// Converts two values to compatible types for bitwise operations.
/// This function is similar to ConvertTypesForArithmetic but:
/// - Does not allow float types (bitwise ops are integer-only)
/// - Uses zero extension for integer widening to preserve bit patterns
/// - Converts index types to integers
///
/// @param lhs The left-hand side operand
/// @param rhs The right-hand side operand
/// @param builder The MLIR builder to use for creating conversion operations
/// @return A pair of converted values, or {nullptr, nullptr} if conversion
/// fails
std::pair<mlir::Value, mlir::Value>
ConvertTypesForBitwise(mlir::Value lhs, mlir::Value rhs,
                       mlir::OpBuilder &builder, mlir::Location location);

} // namespace causalflow::avelang::ir
