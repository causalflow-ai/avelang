#pragma once

#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include <llvm/ADT/ArrayRef.h>

#include "AST/ast_nodes_expr.h"
#include "constexpr_value.h"

namespace causalflow::avelang::ir {

struct GeneratorContext;
class NamedModule;

/// Symbol scope that can store different types of symbols in a unified way
class SymbolScope {
  public:
    enum SymbolKind {
        kModule,
        kTypeFactory,
        kFunction,
        kType,
        kValue,
        kConstexpr,
    };

    using TypeFactoryFunction = std::function<mlir::Type(
        NamedModule *, ast::Call *, GeneratorContext *)>;
    struct Function {
        using Implementation = std::function<mlir::Value(
            ast::Call *, GeneratorContext *, llvm::ArrayRef<mlir::Value>)>;
        using Checker = std::function<bool(ast::Call *, GeneratorContext *,
                                           llvm::ArrayRef<mlir::Value>)>;

        Function() = default;
        Function(Implementation impl, Checker check = nullptr,
                 std::string module = {}, std::string symbol = {})
            : implementation(std::move(impl)), checker(std::move(check)),
              module_name(std::move(module)), symbol_name(std::move(symbol)) {}

        bool IsValid() const { return static_cast<bool>(implementation); }

        bool RunChecker(ast::Call *call_expr, GeneratorContext *ctx,
                        llvm::ArrayRef<mlir::Value> resolved_args) const {
            if (!checker)
                return true;
            return checker(call_expr, ctx, resolved_args);
        }

        mlir::Value Invoke(ast::Call *call_expr, GeneratorContext *ctx,
                           llvm::ArrayRef<mlir::Value> resolved_args) const {
            if (!implementation)
                return mlir::Value();
            return implementation(call_expr, ctx, resolved_args);
        }

        Implementation implementation;
        Checker checker;
        std::string module_name;
        std::string symbol_name;
    };

    struct Symbol {
        SymbolKind kind = kValue;

        // Symbols are deliberately regular value types.  The former manual
        // union made copying symbol-table snapshots unsafe as new symbol kinds
        // were added.
        NamedModule *module = nullptr;
        TypeFactoryFunction type_factory;
        Function function;
        mlir::Type type;
        mlir::Value value;
        ConstexprValue constexpr_value;

        Symbol();
        Symbol(NamedModule *m);
        Symbol(TypeFactoryFunction tf);
        Symbol(Function ff);
        Symbol(mlir::Type t);
        Symbol(mlir::Value v);
        Symbol(ConstexprValue v);

        bool isa(SymbolKind k) const;
    };

    // Symbol storage and access
    void AddSymbol(const std::string &name, const Symbol &symbol);
    std::optional<Symbol> LookupSymbol(const std::string &name) const;

    // Convenience methods for different symbol types
    void AddValue(const std::string &name, mlir::Value value);
    void AddConstexpr(const std::string &name, ConstexprValue value);
    void AddType(const std::string &name, mlir::Type type);
    void AddModule(const std::string &name, NamedModule *module);
    void AddTypeFactory(const std::string &name, TypeFactoryFunction factory);
    void AddFunction(const std::string &name,
                     Function::Implementation implementation,
                     Function::Checker checker = nullptr);
    void AddFunction(const std::string &name, Function function);

    // Legacy lookup methods for compatibility
    mlir::Value LookupValue(const std::string &name) const;
    std::optional<ConstexprValue>
    LookupConstexpr(const std::string &name) const;
    mlir::Type LookupType(const std::string &name) const;
    NamedModule *LookupModule(const std::string &name) const;
    TypeFactoryFunction LookupTypeFactory(const std::string &name) const;
    Function LookupFunction(const std::string &name) const;

  private:
    std::map<std::string, Symbol> symbols_;
};

} // namespace causalflow::avelang::ir
