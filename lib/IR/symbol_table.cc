#include "symbol_table.h"
#include "Utils/assert.h"
#include "generator_context.h"
#include "named_module.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/IR/BuiltinTypes.h>
#pragma clang diagnostic pop

#include <llvm/Support/Error.h>
#include <ranges>

namespace causalflow::avelang::ir {

SymbolTable::SymbolTable(GeneratorContext *context,
                         NamedModuleRegistry *registry)
    : parent_(context), registry_(registry) {
    SS_ASSERT(registry_ && "Named module registry not initialized");
}

void SymbolTable::Initialize() {
    SymbolScope scope;
    frames_.emplace_back(scope);
}

bool SymbolTable::ImportModule(const std::string &module_name,
                               const std::string &alias) {
    auto m = registry_->GetModule(module_name);
    if (!m) {
        parent_->diagnostic_manager->Report(
            basic::DiagnosticCode::kSymbolNotFound, parent_->current_source_loc)
            << ("Unknown module '" + module_name + "' in import statement");
        return false;
    }
    const std::string &import_name = alias.empty() ? module_name : alias;
    auto &frame = GetCurrentFrame();
    frame.AddModule(import_name, m);
    return true;
}

bool SymbolTable::ImportSymbolFrom(const std::string &module_name,
                                   const std::string &symbol_name,
                                   const std::string &alias) {
    auto *module = registry_->GetModule(module_name);
    if (!module) {
        parent_->diagnostic_manager->Report(
            basic::DiagnosticCode::kSymbolNotFound, parent_->current_source_loc)
            << ("Unknown module '" + module_name +
                "' in from-import statement");
        return false;
    }

    auto symbol = module->LookupSymbol(symbol_name);
    if (!symbol) {
        parent_->diagnostic_manager->Report(
            basic::DiagnosticCode::kSymbolNotFound, parent_->current_source_loc)
            << ("Symbol '" + symbol_name + "' not found in module '" +
                module_name + "'");
        return false;
    }

    const std::string &import_name = alias.empty() ? symbol_name : alias;
    auto &frame = GetCurrentFrame();
    frame.AddSymbol(import_name, *symbol);
    return true;
}

void SymbolTable::DefineSymbol(const std::string &name, mlir::Value value) {
    auto &frame = GetCurrentFrame();
    frame.AddValue(name, value);
}

void SymbolTable::DefineConstexpr(const std::string &name,
                                  ConstexprValue value) {
    auto &frame = GetCurrentFrame();
    frame.AddConstexpr(name, std::move(value));
}

std::optional<SymbolTable::Symbol>
SymbolTable::LookupSymbol(const std::string &name) const {
    // Walk frames from innermost to outermost
    for (auto &frame : frames_ | std::views::reverse) {
        auto symbol = frame.LookupSymbol(name);
        if (symbol) {
            return symbol;
        }
    }
    return std::nullopt;
}

void SymbolTable::DeclareModules(mlir::ModuleOp module) {
    registry_->DeclareModules(module);
}

std::unique_ptr<SymbolTable> SymbolTable::Clone() const {
    auto clone = std::make_unique<SymbolTable>(parent_, registry_);
    clone->frames_ = frames_;
    return clone;
}

void SymbolTable::PushFrame() { frames_.emplace_back(); }

void SymbolTable::PopFrame() {
    SS_ASSERT(frames_.size() > 1 && "Cannot pop the global frame");
    frames_.pop_back();
}

SymbolTable::Frame &SymbolTable::GetCurrentFrame() {
    SS_ASSERT(!frames_.empty());
    return frames_.back();
}

std::optional<SymbolTable::Symbol>
SymbolTable::ResolveSymbol(ast::Expr *expr,
                           std::optional<SymbolKind> expected_kind,
                           bool report_not_found) {
    parent_->current_source_loc = expr->GetSourceRange().getBegin();
    auto symbol = ResolveSymbolImpl(expr, report_not_found);
    if (!symbol) {
        return std::nullopt;
    }
    if (expected_kind && !symbol->isa(*expected_kind)) {
        parent_->diagnostic_manager->Report(
            basic::DiagnosticCode::kTypeMismatch, parent_->current_source_loc)
            << "Symbol has wrong type";
        return std::nullopt;
    }
    return symbol;
}

std::optional<SymbolTable::Symbol>
SymbolTable::ResolveSymbolImpl(ast::Expr *expr, bool report_not_found) {
    if (!expr) {
        return std::nullopt;
    }

    // Build resolution path by following the expression chain
    std::vector<std::string> resolution_path;
    ast::Expr *current = expr;

    // First, build the path from the expression
    while (current) {
        if (auto *name = llvm::dyn_cast<ast::Name>(current)) {
            resolution_path.insert(resolution_path.begin(), name->GetId());
            break;
        } else if (auto *call_expr = llvm::dyn_cast<ast::Call>(current)) {
            current = call_expr->GetFunc();
        } else if (auto *attr_expr =
                       llvm::dyn_cast<ast::AttributeExpr>(current)) {
            resolution_path.insert(resolution_path.begin(),
                                   attr_expr->GetAttr());
            current = attr_expr->GetValue();
        } else {
            parent_->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                parent_->current_source_loc)
                << "Unsupported expression type for symbol resolution";
            return std::nullopt;
        }
    }

    if (resolution_path.empty()) {
        return std::nullopt;
    }

    // Now resolve iteratively along the path
    std::optional<Symbol> current_symbol;

    // Look up the root symbol (first element in path)
    const std::string &root_name = resolution_path[0];
    for (auto &frame : frames_ | std::views::reverse) {
        current_symbol = frame.LookupSymbol(root_name);
        if (current_symbol) {
            break;
        }
    }

    if (!current_symbol) {
        if (report_not_found) {
            parent_->diagnostic_manager->Report(
                basic::DiagnosticCode::kSymbolNotFound,
                parent_->current_source_loc)
                << root_name;
        }
        return std::nullopt;
    }

    // Resolve remaining path elements
    for (size_t i = 1; i < resolution_path.size(); ++i) {
        const std::string &attr_name = resolution_path[i];

        if (!current_symbol->isa(SymbolKind::kModule)) {
            std::string path_so_far;
            for (size_t j = 0; j < i; ++j) {
                if (j > 0)
                    path_so_far += ".";
                path_so_far += resolution_path[j];
            }
            parent_->diagnostic_manager->Report(
                basic::DiagnosticCode::kTypeMismatch,
                parent_->current_source_loc)
                << "Symbol '" << path_so_far
                << "' is not a module, cannot access attribute '" << attr_name
                << "'";
            return std::nullopt;
        }

        auto next_symbol = current_symbol->module->LookupSymbol(attr_name);
        if (!next_symbol) {
            std::string path_so_far;
            for (size_t j = 0; j <= i; ++j) {
                if (j > 0)
                    path_so_far += ".";
                path_so_far += resolution_path[j];
            }
            if (report_not_found) {
                parent_->diagnostic_manager->Report(
                    basic::DiagnosticCode::kSymbolNotFound,
                    parent_->current_source_loc)
                    << path_so_far;
            }
            return std::nullopt;
        }

        current_symbol = next_symbol;
    }

    return current_symbol;
}

mlir::Value SymbolTable::ResolveRefExpr(ast::Expr *expr) {
    if (!llvm::isa<ast::Name, ast::AttributeExpr, ast::Call>(expr)) {
        return mlir::Value();
    }

    auto symbol =
        ResolveSymbol(expr, SymbolKind::kValue, /*report_not_found=*/false);
    return symbol ? symbol->value : mlir::Value();
}

std::optional<ConstexprValue>
SymbolTable::ResolveConstexpr(ast::Expr *expr, bool report_not_found) {
    auto symbol =
        ResolveSymbol(expr, SymbolKind::kConstexpr, report_not_found);
    return symbol ? std::optional<ConstexprValue>(symbol->constexpr_value)
                  : std::nullopt;
}

mlir::Type SymbolTable::ResolveBuiltinType(ast::Expr *expr) {
    auto symbol = ResolveSymbol(expr, SymbolKind::kType);
    if (!symbol) {
        return mlir::Type();
    }

    auto type = symbol->type;

    // Validate that this is a builtin type (builtin scalar / vector types)
    if (!type.isIntOrFloat() && !type.isIndex() &&
        !mlir::isa<mlir::VectorType>(type)) {
        parent_->diagnostic_manager->Report(
            basic::DiagnosticCode::kTypeMismatch, parent_->current_source_loc)
            << "Expected primitive type, got complex type";
        return mlir::Type();
    }

    return type;
}

mlir::Type SymbolTable::ResolveType(ast::Expr *expr) {
    if (auto *call_expr = llvm::dyn_cast<ast::Call>(expr)) {
        auto symbol =
            ResolveSymbol(call_expr->GetFunc(), SymbolKind::kTypeFactory);
        if (symbol) {
            return symbol->type_factory(symbol->module, call_expr, parent_);
        }
    }

    // Handle Attribute expressions (e.g., S.f32 as an attribute access)
    // These are like S.f32 without the call parentheses
    if (llvm::isa<ast::AttributeExpr>(expr)) {
        // Try to resolve it as a type first
        auto symbol = ResolveSymbol(expr, SymbolKind::kType);
        if (symbol && symbol->type) {
            return symbol->type;
        }
        // If not a type directly, try as a type factory with empty args
        auto factory_symbol = ResolveSymbol(expr, SymbolKind::kTypeFactory);
        if (factory_symbol) {
            // Create a dummy call with no arguments for the type factory
            // This handles cases like S.f32 where f32 is a type factory
            return factory_symbol->type;
        }
    }

    // Handle direct Name symbols
    auto symbol = ResolveSymbol(expr, SymbolKind::kType);
    return symbol ? symbol->type : mlir::Type();
}

NamedModule::Function SymbolTable::ResolveFunction(ast::Expr *expr) {
    auto symbol = ResolveSymbol(expr, SymbolKind::kFunction);
    return symbol ? symbol->function : NamedModule::Function();
}

} // namespace causalflow::avelang::ir
