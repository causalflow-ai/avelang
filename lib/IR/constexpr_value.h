#pragma once

#include "type_system.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>

namespace causalflow::avelang::ir {

/// Canonical, context-owned representation of a compile-time value.
///
/// Constexpr symbols store attributes rather than SSA values.  An SSA value is
/// created only when a function that uses the constexpr is generated.
struct ConstexprValue {
    mlir::TypedAttr attribute;
    TypeInfo type_info;

    explicit operator bool() const { return static_cast<bool>(attribute); }

    mlir::Type GetType() const {
        return attribute ? attribute.getType() : mlir::Type();
    }
};

} // namespace causalflow::avelang::ir
