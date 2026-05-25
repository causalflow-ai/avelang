#include "constant_folder.h"

#include "AST/ast_nodes_expr.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "generator_context.h"
#include "mlir_generator_impl.h"
#include "parsing_utils.h"
#include "symbol_table.h"
#include "type_system.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <llvm/Support/Casting.h>

namespace causalflow::avelang::ir {

namespace {

static std::optional<int64_t> GetStaticElementCount(mlir::Type type) {
    auto getCount = [](auto memrefType) -> std::optional<int64_t> {
        int64_t count = 1;
        for (int64_t dim : memrefType.getShape()) {
            if (dim == mlir::ShapedType::kDynamic) {
                return std::nullopt;
            }
            count *= dim;
        }
        return count;
    };

    if (auto avelangType =
            mlir::dyn_cast<causalflow::avelang::dialect::MemRefType>(type)) {
        return getCount(avelangType);
    }
    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
        return getCount(memrefType);
    }
    return std::nullopt;
}

static bool AreZeroIndices(mlir::ValueRange indices) {
    return llvm::all_of(indices, [](mlir::Value index) {
        if (auto constOp = index.getDefiningOp<mlir::arith::ConstantOp>()) {
            if (auto intAttr =
                    mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
                return intAttr.getInt() == 0;
            }
        }
        if (auto constIndex =
                index.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
            return constIndex.value() == 0;
        }
        return false;
    });
}

static std::optional<int64_t> FoldScalarMemrefValue(mlir::Value memref) {
    auto elementCount = GetStaticElementCount(memref.getType());
    if (!elementCount || *elementCount != 1) {
        return std::nullopt;
    }

    std::optional<int64_t> foldedValue;
    for (mlir::Operation *user : memref.getUsers()) {
        if (auto store = mlir::dyn_cast<
                causalflow::avelang::dialect::AveLangMemRefStoreOp>(user)) {
            if (!AreZeroIndices(store.getIndices())) {
                return std::nullopt;
            }
            auto value = ConstantFolder::FoldIntValue(store.getValue());
            if (!value || foldedValue) {
                return std::nullopt;
            }
            foldedValue = value;
            continue;
        }

        if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user)) {
            if (!AreZeroIndices(store.getIndices())) {
                return std::nullopt;
            }
            auto value = ConstantFolder::FoldIntValue(store.getValue());
            if (!value || foldedValue) {
                return std::nullopt;
            }
            foldedValue = value;
            continue;
        }

        if (auto load = mlir::dyn_cast<
                causalflow::avelang::dialect::AveLangMemRefLoadOp>(user)) {
            if (!AreZeroIndices(load.getIndices())) {
                return std::nullopt;
            }
            continue;
        }

        if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(user)) {
            if (!AreZeroIndices(load.getIndices())) {
                return std::nullopt;
            }
            continue;
        }

        return std::nullopt;
    }

    return foldedValue;
}

template <typename LoadOpTy>
static std::optional<int64_t> FoldScalarMemrefLoadImpl(LoadOpTy load) {
    auto memref = load.getMemref();
    auto elementCount = GetStaticElementCount(memref.getType());
    if (!elementCount || *elementCount != 1 ||
        !AreZeroIndices(load.getIndices())) {
        return std::nullopt;
    }

    std::optional<int64_t> foldedValue;
    for (mlir::Operation *user : memref.getUsers()) {
        if (user == load.getOperation()) {
            continue;
        }

        if (auto store = mlir::dyn_cast<
                causalflow::avelang::dialect::AveLangMemRefStoreOp>(user)) {
            if (!AreZeroIndices(store.getIndices())) {
                return std::nullopt;
            }
            auto value = ConstantFolder::FoldIntValue(store.getValue());
            if (!value || foldedValue) {
                return std::nullopt;
            }
            foldedValue = value;
            continue;
        }

        if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user)) {
            if (!AreZeroIndices(store.getIndices())) {
                return std::nullopt;
            }
            auto value = ConstantFolder::FoldIntValue(store.getValue());
            if (!value || foldedValue) {
                return std::nullopt;
            }
            foldedValue = value;
            continue;
        }

        return std::nullopt;
    }

    return foldedValue;
}

struct IntegerTypeInfo {
    unsigned bitWidth;
    bool isUnsigned;
};

static std::optional<IntegerTypeInfo>
GetBuiltinIntegerTypeInfo(ast::Expr *expr) {
    auto typeInfo = GetTypeInfo(expr);
    if (!typeInfo.is_unsigned_integer) {
        return std::nullopt;
    }

    auto getBitWidth = [](llvm::StringRef typeName) -> std::optional<unsigned> {
        if (typeName.ends_with("8")) {
            return 8;
        }
        if (typeName.ends_with("16")) {
            return 16;
        }
        if (typeName.ends_with("32")) {
            return 32;
        }
        if (typeName.ends_with("64")) {
            return 64;
        }
        return std::nullopt;
    };

    if (auto *name = llvm::dyn_cast<ast::Name>(expr)) {
        if (auto width = getBitWidth(name->GetId())) {
            return IntegerTypeInfo{*width, *typeInfo.is_unsigned_integer};
        }
    }
    if (auto *attr = llvm::dyn_cast<ast::AttributeExpr>(expr)) {
        if (auto width = getBitWidth(attr->GetAttr())) {
            return IntegerTypeInfo{*width, *typeInfo.is_unsigned_integer};
        }
    }
    return std::nullopt;
}

static bool IsBitcastCall(const ast::Call *call) {
    if (!call) {
        return false;
    }
    auto *func = call->GetFunc();
    if (auto *name = llvm::dyn_cast<ast::Name>(func)) {
        return name->GetId() == "bitcast";
    }
    if (auto *attr = llvm::dyn_cast<ast::AttributeExpr>(func)) {
        return attr->GetAttr() == "bitcast";
    }
    return false;
}

static bool IsConvertCall(const ast::Call *call) {
    if (!call) {
        return false;
    }
    auto *func = call->GetFunc();
    if (auto *name = llvm::dyn_cast<ast::Name>(func)) {
        return name->GetId() == "convert";
    }
    if (auto *attr = llvm::dyn_cast<ast::AttributeExpr>(func)) {
        return attr->GetAttr() == "convert";
    }
    return false;
}

static bool IsUnsignedIntegerExpr(ast::Expr *expr) {
    if (auto *call = llvm::dyn_cast<ast::Call>(expr)) {
        if (call->GetArgs().size() != 2) {
            return false;
        }
        if (IsBitcastCall(call) || IsConvertCall(call)) {
            auto typeInfo = GetTypeInfo(call->GetArgs()[1]);
            return typeInfo.is_unsigned_integer.value_or(false);
        }
        return false;
    }

    if (auto *binop = llvm::dyn_cast<ast::BinOp>(expr)) {
        const auto &op = binop->GetOp();
        if (op == "BitAnd" || op == "BitOr" || op == "BitXor" ||
            op == "LShift" || op == "RShift" || op == "Add" || op == "Sub" ||
            op == "Mult" || op == "Div" || op == "Mod" || op == "FloorDiv") {
            if (IsUnsignedIntegerExpr(binop->GetLeft())) {
                return true;
            }
            return IsUnsignedIntegerExpr(binop->GetRight());
        }
    }

    return false;
}

static int64_t NormalizeBitcastIntegerValue(int64_t value,
                                            const IntegerTypeInfo &typeInfo) {
    uint64_t bits = static_cast<uint64_t>(value);
    if (typeInfo.bitWidth < 64) {
        uint64_t mask = (uint64_t{1} << typeInfo.bitWidth) - 1;
        bits &= mask;
    }

    if (typeInfo.isUnsigned || typeInfo.bitWidth == 64) {
        return static_cast<int64_t>(bits);
    }

    uint64_t signBit = uint64_t{1} << (typeInfo.bitWidth - 1);
    if (bits & signBit) {
        uint64_t extendMask = ~((uint64_t{1} << typeInfo.bitWidth) - 1);
        bits |= extendMask;
    }
    return static_cast<int64_t>(bits);
}

} // namespace

std::optional<int64_t> ConstantFolder::GetConstantIntValue(mlir::Value value) {
    if (!value) {
        return std::nullopt;
    }

    if (auto const_op = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr =
                mlir::dyn_cast<mlir::IntegerAttr>(const_op.getValue())) {
            return int_attr.getInt();
        }
    }

    return std::nullopt;
}

std::optional<bool> ConstantFolder::FoldBoolValue(mlir::Value value) {
    if (!value) {
        return std::nullopt;
    }

    if (auto constant = GetConstantIntValue(value)) {
        return *constant != 0;
    }

    if (auto cmp = value.getDefiningOp<mlir::arith::CmpIOp>()) {
        auto lhs = FoldIntValue(cmp.getLhs());
        auto rhs = FoldIntValue(cmp.getRhs());
        if (!lhs || !rhs) {
            return std::nullopt;
        }
        switch (cmp.getPredicate()) {
        case mlir::arith::CmpIPredicate::eq:
            return *lhs == *rhs;
        case mlir::arith::CmpIPredicate::ne:
            return *lhs != *rhs;
        case mlir::arith::CmpIPredicate::slt:
            return *lhs < *rhs;
        case mlir::arith::CmpIPredicate::sle:
            return *lhs <= *rhs;
        case mlir::arith::CmpIPredicate::sgt:
            return *lhs > *rhs;
        case mlir::arith::CmpIPredicate::sge:
            return *lhs >= *rhs;
        case mlir::arith::CmpIPredicate::ult:
            return static_cast<uint64_t>(*lhs) < static_cast<uint64_t>(*rhs);
        case mlir::arith::CmpIPredicate::ule:
            return static_cast<uint64_t>(*lhs) <= static_cast<uint64_t>(*rhs);
        case mlir::arith::CmpIPredicate::ugt:
            return static_cast<uint64_t>(*lhs) > static_cast<uint64_t>(*rhs);
        case mlir::arith::CmpIPredicate::uge:
            return static_cast<uint64_t>(*lhs) >= static_cast<uint64_t>(*rhs);
        }
    }

    return std::nullopt;
}

std::optional<int64_t> ConstantFolder::FoldIntValue(mlir::Value value) {
    if (!value) {
        return std::nullopt;
    }

    if (auto constant = GetConstantIntValue(value)) {
        return constant;
    }

    if (auto cast_op = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
        return FoldIntValue(cast_op.getIn());
    }

    if (auto load = value.getDefiningOp<
                    causalflow::avelang::dialect::AveLangMemRefLoadOp>()) {
        if (auto folded = FoldScalarMemrefLoadImpl(load)) {
            return folded;
        }
    }

    if (auto load = value.getDefiningOp<mlir::memref::LoadOp>()) {
        if (auto folded = FoldScalarMemrefLoadImpl(load)) {
            return folded;
        }
    }

    if (auto add = value.getDefiningOp<mlir::arith::AddIOp>()) {
        auto lhs = FoldIntValue(add.getLhs());
        auto rhs = FoldIntValue(add.getRhs());
        if (lhs && rhs) {
            return *lhs + *rhs;
        }
        return std::nullopt;
    }

    if (auto sub = value.getDefiningOp<mlir::arith::SubIOp>()) {
        auto lhs = FoldIntValue(sub.getLhs());
        auto rhs = FoldIntValue(sub.getRhs());
        if (lhs && rhs) {
            return *lhs - *rhs;
        }
        return std::nullopt;
    }

    if (auto mul = value.getDefiningOp<mlir::arith::MulIOp>()) {
        auto lhs = FoldIntValue(mul.getLhs());
        auto rhs = FoldIntValue(mul.getRhs());
        if (lhs && rhs) {
            return *lhs * *rhs;
        }
        return std::nullopt;
    }

    if (auto div = value.getDefiningOp<mlir::arith::DivSIOp>()) {
        auto lhs = FoldIntValue(div.getLhs());
        auto rhs = FoldIntValue(div.getRhs());
        if (lhs && rhs && *rhs != 0) {
            return *lhs / *rhs;
        }
        return std::nullopt;
    }

    if (auto div = value.getDefiningOp<mlir::arith::DivUIOp>()) {
        auto lhs = FoldIntValue(div.getLhs());
        auto rhs = FoldIntValue(div.getRhs());
        if (lhs && rhs && *rhs != 0) {
            return static_cast<uint64_t>(*lhs) / static_cast<uint64_t>(*rhs);
        }
        return std::nullopt;
    }

    if (auto rem = value.getDefiningOp<mlir::arith::RemSIOp>()) {
        auto lhs = FoldIntValue(rem.getLhs());
        auto rhs = FoldIntValue(rem.getRhs());
        if (lhs && rhs && *rhs != 0) {
            return *lhs % *rhs;
        }
        return std::nullopt;
    }

    if (auto rem = value.getDefiningOp<mlir::arith::RemUIOp>()) {
        auto lhs = FoldIntValue(rem.getLhs());
        auto rhs = FoldIntValue(rem.getRhs());
        if (lhs && rhs && *rhs != 0) {
            return static_cast<uint64_t>(*lhs) % static_cast<uint64_t>(*rhs);
        }
        return std::nullopt;
    }

    if (auto shl = value.getDefiningOp<mlir::arith::ShLIOp>()) {
        auto lhs = FoldIntValue(shl.getLhs());
        auto rhs = FoldIntValue(shl.getRhs());
        if (lhs && rhs) {
            return *lhs << *rhs;
        }
        return std::nullopt;
    }

    if (auto shr = value.getDefiningOp<mlir::arith::ShRSIOp>()) {
        auto lhs = FoldIntValue(shr.getLhs());
        auto rhs = FoldIntValue(shr.getRhs());
        if (lhs && rhs) {
            return *lhs >> *rhs;
        }
        return std::nullopt;
    }

    if (auto shr = value.getDefiningOp<mlir::arith::ShRUIOp>()) {
        auto lhs = FoldIntValue(shr.getLhs());
        auto rhs = FoldIntValue(shr.getRhs());
        if (lhs && rhs) {
            return static_cast<uint64_t>(*lhs) >> static_cast<uint64_t>(*rhs);
        }
        return std::nullopt;
    }

    if (auto select = value.getDefiningOp<mlir::arith::SelectOp>()) {
        auto cond = FoldBoolValue(select.getCondition());
        if (!cond) {
            return std::nullopt;
        }
        return *cond ? FoldIntValue(select.getTrueValue())
                     : FoldIntValue(select.getFalseValue());
    }

    return std::nullopt;
}

std::optional<int64_t>
ConstantFolder::ResolveConstantReference(ast::Expr *expr) const {
    if (!ctx_ || !ctx_->syms || !expr) {
        return std::nullopt;
    }

    if (auto *name = llvm::dyn_cast<ast::Name>(expr)) {
        if (auto *func_gen = ctx_->GetCurrentFunctionGenerator()) {
            if (auto value = func_gen->LookupConstexprInteger(name->GetId())) {
                return value;
            }
        }
        auto symbol = ctx_->syms->LookupSymbol(name->GetId());
        if (!symbol || !symbol->isa(SymbolTable::SymbolKind::kValue)) {
            return std::nullopt;
        }
        if (auto value = FoldIntValue(symbol->value)) {
            return value;
        }
        return FoldScalarMemrefValue(symbol->value);
    }

    if (auto *attr = llvm::dyn_cast<ast::AttributeExpr>(expr)) {
        auto symbol = ctx_->syms->ResolveSymbol(attr, std::nullopt,
                                                /*report_not_found=*/false);
        if (!symbol || !symbol->isa(SymbolTable::SymbolKind::kValue)) {
            return std::nullopt;
        }
        if (auto value = FoldIntValue(symbol->value)) {
            return value;
        }
        return FoldScalarMemrefValue(symbol->value);
    }

    return std::nullopt;
}

std::optional<int64_t> ConstantFolder::Evaluate(ast::Expr *expr) const {
    if (!expr) {
        return std::nullopt;
    }

    if (auto *constant = llvm::dyn_cast<ast::Constant>(expr)) {
        return ParseConstantInteger(constant->GetValue());
    }

    if (auto value = ResolveConstantReference(expr)) {
        return value;
    }

    if (auto *call = llvm::dyn_cast<ast::Call>(expr)) {
        if ((IsBitcastCall(call) || IsConvertCall(call)) &&
            call->GetArgs().size() == 2) {
            auto value = Evaluate(call->GetArgs()[0]);
            auto typeInfo = GetBuiltinIntegerTypeInfo(call->GetArgs()[1]);
            if (!value || !typeInfo) {
                return std::nullopt;
            }
            return NormalizeBitcastIntegerValue(*value, *typeInfo);
        }
        return std::nullopt;
    }

    if (auto *unary = llvm::dyn_cast<ast::UnaryOp>(expr)) {
        auto operand = Evaluate(unary->GetOperand());
        if (!operand) {
            return std::nullopt;
        }
        const auto &op = unary->GetOp();
        if (op == "UAdd") {
            return *operand;
        }
        if (op == "USub") {
            return -*operand;
        }
        return std::nullopt;
    }

    if (auto *binop = llvm::dyn_cast<ast::BinOp>(expr)) {
        auto lhs = Evaluate(binop->GetLeft());
        auto rhs = Evaluate(binop->GetRight());
        if (!lhs || !rhs) {
            return std::nullopt;
        }

        const auto &op = binop->GetOp();
        if (op == "Add") {
            return *lhs + *rhs;
        }
        if (op == "Sub") {
            return *lhs - *rhs;
        }
        if (op == "Mult") {
            return *lhs * *rhs;
        }
        if (op == "FloorDiv") {
            if (*rhs == 0) {
                return std::nullopt;
            }
            if (IsUnsignedIntegerExpr(binop->GetLeft())) {
                return static_cast<uint64_t>(*lhs) /
                       static_cast<uint64_t>(*rhs);
            }
            int64_t quotient = *lhs / *rhs;
            int64_t remainder = *lhs % *rhs;
            if (remainder != 0 && ((remainder > 0) != (*rhs > 0))) {
                --quotient;
            }
            return quotient;
        }
        if (op == "Div") {
            if (*rhs == 0) {
                return std::nullopt;
            }
            if (IsUnsignedIntegerExpr(binop->GetLeft())) {
                if ((static_cast<uint64_t>(*lhs) %
                     static_cast<uint64_t>(*rhs)) != 0) {
                    return std::nullopt;
                }
                return static_cast<uint64_t>(*lhs) /
                       static_cast<uint64_t>(*rhs);
            }
            if ((*lhs % *rhs) != 0) {
                return std::nullopt;
            }
            return *lhs / *rhs;
        }
        if (op == "Mod") {
            if (*rhs == 0) {
                return std::nullopt;
            }
            if (IsUnsignedIntegerExpr(binop->GetLeft())) {
                return static_cast<uint64_t>(*lhs) %
                       static_cast<uint64_t>(*rhs);
            }
            return *lhs % *rhs;
        }
        if (op == "LShift") {
            return *lhs << *rhs;
        }
        if (op == "RShift") {
            if (IsUnsignedIntegerExpr(binop->GetLeft())) {
                return static_cast<int64_t>(static_cast<uint64_t>(*lhs) >>
                                            static_cast<uint64_t>(*rhs));
            }
            return *lhs >> *rhs;
        }
        if (op == "BitAnd") {
            return *lhs & *rhs;
        }
        if (op == "BitOr") {
            return *lhs | *rhs;
        }
        if (op == "BitXor") {
            return *lhs ^ *rhs;
        }
        return std::nullopt;
    }

    return std::nullopt;
}

} // namespace causalflow::avelang::ir
