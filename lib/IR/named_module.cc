#include "named_module.h"
#include "builtin_module.h"
#include "ir_context.h"

#include <utility>

#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

namespace causalflow::avelang::ir {

using namespace mlir;

// NamedModule Implementation
NamedModule::NamedModule(const std::string &name) : name_(name) {}

std::optional<SymbolScope::Symbol>
NamedModule::LookupSymbol(const std::string &name) const {
    return symbols_.LookupSymbol(name);
}

void NamedModule::AddSymbol(const std::string &name, mlir::Value value) {
    symbols_.AddValue(name, value);
}

void NamedModule::AddType(const std::string &name, mlir::Type type) {
    symbols_.AddType(name, type);
}

void NamedModule::AddModule(const std::string &name, NamedModule *module) {
    symbols_.AddModule(name, module);
}

void NamedModule::AddTypeFactory(const std::string &name,
                                 TypeFactoryFunction factory) {
    symbols_.AddTypeFactory(name, std::move(factory), this);
}

void NamedModule::AddFunction(const std::string &name,
                              Function::Implementation implementation,
                              Function::Checker checker) {
    symbols_.AddFunction(name, Function(std::move(implementation),
                                        std::move(checker), name_, name));
}

NamedModuleRegistry::NamedModuleRegistry(IRContext *ir_context)
    : ir_context_(ir_context) {}

void NamedModuleRegistry::Initialize() {
    std::unique_ptr<AveLangModule> m =
        std::make_unique<AveLangModule>(ir_context_);
    m->Initialize();
    builtin_modules_["avelang"] = std::move(m);
}

void NamedModuleRegistry::DeclareModules(mlir::ModuleOp module) {
    for (auto &entry : builtin_modules_) {
        if (entry.second) {
            entry.second->DeclareModules(module);
        }
    }
}

NamedModule *NamedModuleRegistry::GetModule(const std::string &name) {
    if (name.empty()) {
        return nullptr;
    }

    auto extract_segment = [](const std::string &str, size_t begin,
                              size_t end) -> std::string {
        if (end == std::string::npos) {
            return str.substr(begin);
        }
        return str.substr(begin, end - begin);
    };

    size_t segment_end = name.find('.');
    std::string root = extract_segment(name, 0, segment_end);
    auto it = builtin_modules_.find(root);
    if (it == builtin_modules_.end()) {
        return nullptr;
    }
    NamedModule *module = it->second.get();

    while (segment_end != std::string::npos) {
        size_t next_begin = segment_end + 1;
        if (next_begin >= name.size()) {
            return nullptr;
        }
        size_t next_end = name.find('.', next_begin);
        std::string segment = extract_segment(name, next_begin, next_end);
        if (segment.empty()) {
            return nullptr;
        }

        auto symbol = module->LookupSymbol(segment);
        if (!symbol || !symbol->isa(SymbolScope::kModule) ||
            !symbol->GetModule()) {
            return nullptr;
        }
        module = symbol->GetModule();
        segment_end = next_end;
    }

    return module;
}

} // namespace causalflow::avelang::ir
