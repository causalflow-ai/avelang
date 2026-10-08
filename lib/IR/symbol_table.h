#pragma once

#include "AST/ast.h"
#include "AST/ast_nodes_expr.h"
#include "AST/ast_nodes_stmt.h"
#include "named_module.h"
#include "symbol_scope.h"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace causalflow::avelang::ir {

class NamedModule;
class NamedModuleRegistry;
struct GeneratorContext;

/// Enhanced symbol table with module support
class SymbolTable {
  public:
    explicit SymbolTable(GeneratorContext *context,
                         NamedModuleRegistry *registry);

    // Use SymbolScope types
    using SymbolKind = SymbolScope::SymbolKind;
    using Symbol = SymbolScope::Symbol;

    std::optional<Symbol>
    ResolveSymbol(ast::Expr *expr,
                  std::optional<SymbolKind> expected_kind = SymbolKind::kValue,
                  bool report_not_found = true);

    // Type resolution
    mlir::Type ResolveType(ast::Expr *annotation);
    // Resolve a primitive type expression which does not support the Call
    // expression.
    mlir::Type ResolveBuiltinType(ast::Expr *annotation);
    // Resolve a reference expression to a value. return a null value if failed.
    mlir::Value ResolveRefExpr(ast::Expr *expr);
    std::optional<ConstexprValue>
    ResolveConstexpr(ast::Expr *expr, bool report_not_found = true);
    // Resolve a registered function for calling functions
    NamedModule::Function ResolveFunction(ast::Expr *expr);

    // Simple string-based symbol lookup (does not create AST nodes)
    std::optional<Symbol> LookupSymbol(const std::string &name) const;

    bool ImportModule(const std::string &module_name,
                      const std::string &alias = "");
    bool ImportSymbolFrom(const std::string &module_name,
                          const std::string &symbol_name,
                          const std::string &alias = "");
    void DefineSymbol(const std::string &name, mlir::Value value);
    void DefineConstexpr(const std::string &name, ConstexprValue value);
    void DeclareModules(mlir::ModuleOp module);

    // Use SymbolScope instead of Frame
    using Frame = SymbolScope;

    void Initialize();
    std::unique_ptr<SymbolTable> Clone() const;
    void PushFrame();
    void PopFrame();
    Frame &GetCurrentFrame();

    class FrameGuard {
      public:
        explicit FrameGuard(SymbolTable *syms) : syms_(syms) {
            syms_->PushFrame();
        }
        ~FrameGuard() { syms_->PopFrame(); }

      private:
        SymbolTable *syms_;
    };

  private:
    std::optional<Symbol> ResolveSymbolImpl(ast::Expr *expr,
                                            bool report_not_found = true);

    GeneratorContext *parent_;
    NamedModuleRegistry *registry_;
    std::vector<SymbolScope> frames_;
};

} // namespace causalflow::avelang::ir
