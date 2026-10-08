#include "symbol_scope.h"
#include "named_module.h"

#include <utility>

namespace causalflow::avelang::ir {

SymbolScope::Symbol::Symbol() = default;

SymbolScope::Symbol::Symbol(NamedModule *m) : kind(kModule), module(m) {}

SymbolScope::Symbol::Symbol(TypeFactoryFunction tf)
    : kind(kTypeFactory), type_factory(tf) {}

SymbolScope::Symbol::Symbol(Function ff)
    : kind(kFunction), function(std::move(ff)) {}

SymbolScope::Symbol::Symbol(mlir::Type t) : kind(kType), type(t) {}

SymbolScope::Symbol::Symbol(mlir::Value v) : kind(kValue), value(v) {}

SymbolScope::Symbol::Symbol(ConstexprValue v)
    : kind(kConstexpr), constexpr_value(std::move(v)) {}

bool SymbolScope::Symbol::isa(SymbolKind k) const { return this->kind == k; }

void SymbolScope::AddSymbol(const std::string &name, const Symbol &symbol) {
    symbols_[name] = symbol;
}

std::optional<SymbolScope::Symbol>
SymbolScope::LookupSymbol(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void SymbolScope::AddValue(const std::string &name, mlir::Value value) {
    symbols_[name] = Symbol(value);
}

void SymbolScope::AddConstexpr(const std::string &name, ConstexprValue value) {
    symbols_[name] = Symbol(std::move(value));
}

void SymbolScope::AddType(const std::string &name, mlir::Type type) {
    symbols_[name] = Symbol(type);
}

void SymbolScope::AddModule(const std::string &name, NamedModule *module) {
    symbols_[name] = Symbol(module);
}

void SymbolScope::AddTypeFactory(const std::string &name,
                                 TypeFactoryFunction factory) {
    symbols_[name] = Symbol(factory);
}

void SymbolScope::AddFunction(const std::string &name,
                              Function::Implementation implementation,
                              Function::Checker checker) {
    symbols_[name] = Symbol(
        Function(std::move(implementation), std::move(checker), {}, name));
}

void SymbolScope::AddFunction(const std::string &name, Function function) {
    symbols_[name] = Symbol(std::move(function));
}

mlir::Value SymbolScope::LookupValue(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kValue)) {
        return it->second.value;
    }
    return mlir::Value();
}

std::optional<ConstexprValue>
SymbolScope::LookupConstexpr(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kConstexpr)) {
        return it->second.constexpr_value;
    }
    return std::nullopt;
}

mlir::Type SymbolScope::LookupType(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kType)) {
        return it->second.type;
    }
    return mlir::Type();
}

NamedModule *SymbolScope::LookupModule(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kModule)) {
        return it->second.module;
    }
    return nullptr;
}

SymbolScope::TypeFactoryFunction
SymbolScope::LookupTypeFactory(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kTypeFactory)) {
        return it->second.type_factory;
    }
    return nullptr;
}

SymbolScope::Function
SymbolScope::LookupFunction(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kFunction)) {
        return it->second.function;
    }
    return Function();
}

} // namespace causalflow::avelang::ir
