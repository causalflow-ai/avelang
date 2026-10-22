#include "symbol_scope.h"
#include "named_module.h"

#include <utility>

namespace causalflow::avelang::ir {

SymbolScope::Symbol::Symbol(NamedModule *m) : payload_(m) {}

SymbolScope::Symbol::Symbol(TypeFactoryFunction tf, NamedModule *module)
    : payload_(TypeFactory{module, std::move(tf)}) {}

SymbolScope::Symbol::Symbol(Function ff) : payload_(std::move(ff)) {}

SymbolScope::Symbol::Symbol(mlir::Type t) : payload_(t) {}

SymbolScope::Symbol::Symbol(mlir::Value v, bool immutable)
    : payload_(v), immutable_(immutable) {}

bool SymbolScope::Symbol::isa(SymbolKind k) const {
    switch (k) {
    case kModule:
        return std::holds_alternative<NamedModule *>(payload_);
    case kTypeFactory:
        return std::holds_alternative<TypeFactory>(payload_);
    case kFunction:
        return std::holds_alternative<Function>(payload_);
    case kType:
        return std::holds_alternative<mlir::Type>(payload_);
    case kValue:
        return std::holds_alternative<mlir::Value>(payload_);
    }
    return false;
}

NamedModule *SymbolScope::Symbol::GetModule() const {
    if (auto *module = std::get_if<NamedModule *>(&payload_)) {
        return *module;
    }
    if (auto *factory = std::get_if<TypeFactory>(&payload_)) {
        return factory->module;
    }
    return nullptr;
}

SymbolScope::TypeFactoryFunction SymbolScope::Symbol::GetTypeFactory() const {
    auto *factory = std::get_if<TypeFactory>(&payload_);
    return factory ? factory->implementation : TypeFactoryFunction();
}

SymbolScope::Function SymbolScope::Symbol::GetFunction() const {
    auto *function = std::get_if<Function>(&payload_);
    return function ? *function : Function();
}

mlir::Type SymbolScope::Symbol::GetType() const {
    auto *type = std::get_if<mlir::Type>(&payload_);
    return type ? *type : mlir::Type();
}

mlir::Value SymbolScope::Symbol::GetValue() const {
    auto *value = std::get_if<mlir::Value>(&payload_);
    return value ? *value : mlir::Value();
}

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

void SymbolScope::AddValue(const std::string &name, mlir::Value value,
                           bool immutable) {
    symbols_[name] = Symbol(value, immutable);
}

void SymbolScope::AddType(const std::string &name, mlir::Type type) {
    symbols_[name] = Symbol(type);
}

void SymbolScope::AddModule(const std::string &name, NamedModule *module) {
    symbols_[name] = Symbol(module);
}

void SymbolScope::AddTypeFactory(const std::string &name,
                                 TypeFactoryFunction factory,
                                 NamedModule *module) {
    symbols_[name] = Symbol(std::move(factory), module);
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
        return it->second.GetValue();
    }
    return mlir::Value();
}

mlir::Type SymbolScope::LookupType(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kType)) {
        return it->second.GetType();
    }
    return mlir::Type();
}

NamedModule *SymbolScope::LookupModule(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kModule)) {
        return it->second.GetModule();
    }
    return nullptr;
}

SymbolScope::TypeFactoryFunction
SymbolScope::LookupTypeFactory(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kTypeFactory)) {
        return it->second.GetTypeFactory();
    }
    return nullptr;
}

SymbolScope::Function
SymbolScope::LookupFunction(const std::string &name) const {
    auto it = symbols_.find(name);
    if (it != symbols_.end() && it->second.isa(kFunction)) {
        return it->second.GetFunction();
    }
    return Function();
}

} // namespace causalflow::avelang::ir
