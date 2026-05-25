#include "mlir_generator.h"
#include "AST/ast_nodes_expr.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "Utils/assert.h"
#include "generator_context.h"
#include "mangler.h"
#include "mlir_generator_impl.h"
#include "symbol_table.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#pragma clang diagnostic pop

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>

#include <cctype>
#include <limits>
#include <memory>
#include <string>

namespace causalflow::avelang::ir {

using namespace mlir;
using namespace llvm;

GeneratorContext::GeneratorContext(
    IRContext *ir_context,
    llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diagnostic_manager)
    : ir_context(ir_context), diagnostic_manager(diagnostic_manager) {}

ExprGenerator *GeneratorContext::GetExprGenerator() const {
    auto *generator = GetCurrentFunctionGenerator();
    return generator ? generator->GetExprGenerator() : nullptr;
}

MLIRGenerator::MLIRGenerator(
    IRContext *ir_context,
    llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diagnostic_manager)
    : ctx_(std::make_unique<GeneratorContext>(ir_context, diagnostic_manager)),
      impl_(std::make_unique<MLIRGeneratorImpl>(ctx_.get())) {
    impl_->InitializeSymbolTable();
}

MLIRGenerator::~MLIRGenerator() = default;

mlir::ModuleOp MLIRGenerator::CreateModule() { return impl_->CreateModule(); }

mlir::ModuleOp MLIRGenerator::Generate(ast::ASTNode *root) {
    return impl_->Generate(root);
}

mlir::ModuleOp MLIRGenerator::GetModule() { return impl_->CreateModule(); }

static llvm::Error InjectConstexprsIntoModule(MLIRGenerator &Generator,
                                              IRContext *IRContext,
                                              llvm::StringRef ConstexprsJSON) {
    if (ConstexprsJSON.empty() || ConstexprsJSON == "[]")
        return llvm::Error::success();

    auto ParseResult = llvm::json::parse(ConstexprsJSON);
    if (!ParseResult)
        return ParseResult.takeError();

    auto *ConstexprsArray = ParseResult->getAsArray();
    if (!ConstexprsArray)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Expected JSON array for constexprs");

    mlir::OpBuilder builder(IRContext->GetMLIRContext());
    auto module = Generator.CreateModule();
    builder.setInsertionPointToStart(module.getBody());

    auto *symbolTable = Generator.GetSymbolTable();
    auto &globalFrame = symbolTable->GetCurrentFrame();

    for (size_t index = 0; index < ConstexprsArray->size(); ++index) {
        const auto &item = (*ConstexprsArray)[index];
        auto *obj = item.getAsObject();
        if (!obj) {
            return llvm::createStringError(
                llvm::inconvertibleErrorCode(),
                "constexprs entry must be a JSON object");
        }

        auto name = obj->getString("name");
        auto type = obj->getString("type");
        auto value = obj->get("value");

        if (!name || name->empty()) {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "constexprs entry missing 'name'");
        }
        if (!type || type->empty()) {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "constexprs entry missing 'type'");
        }
        if (!value) {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "constexprs entry missing 'value'");
        }

        mlir::Value constValue;

        if (*type == "i32") {
            auto intVal = value->getAsInteger();
            if (!intVal) {
                return llvm::createStringError(
                    llvm::inconvertibleErrorCode(),
                    "constexpr i32 value must be integer");
            }
            if (*intVal < std::numeric_limits<int32_t>::min() ||
                *intVal > std::numeric_limits<int32_t>::max()) {
                return llvm::createStringError(
                    llvm::inconvertibleErrorCode(),
                    "constexpr i32 value out of range");
            }
            auto attr =
                builder.getI32IntegerAttr(static_cast<int32_t>(*intVal));
            constValue = mlir::arith::ConstantOp::create(
                builder, builder.getUnknownLoc(), attr);
        } else if (*type == "i64") {
            auto intVal = value->getAsInteger();
            if (!intVal) {
                return llvm::createStringError(
                    llvm::inconvertibleErrorCode(),
                    "constexpr i64 value must be integer");
            }
            auto attr = builder.getI64IntegerAttr(*intVal);
            constValue = mlir::arith::ConstantOp::create(
                builder, builder.getUnknownLoc(), attr);
        } else if (*type == "f64") {
            auto floatVal = value->getAsNumber();
            if (!floatVal) {
                return llvm::createStringError(
                    llvm::inconvertibleErrorCode(),
                    "constexpr f64 value must be number");
            }
            auto attr = builder.getF64FloatAttr(*floatVal);
            constValue = mlir::arith::ConstantOp::create(
                builder, builder.getUnknownLoc(), attr);
        } else if (*type == "i1") {
            auto boolVal = value->getAsBoolean();
            if (!boolVal) {
                return llvm::createStringError(
                    llvm::inconvertibleErrorCode(),
                    "constexpr i1 value must be boolean");
            }
            auto attr = builder.getBoolAttr(*boolVal);
            constValue = mlir::arith::ConstantOp::create(
                builder, builder.getUnknownLoc(), attr);
        } else {
            return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "Unsupported constexpr type");
        }

        globalFrame.AddValue(name->str(), constValue, /*immutable=*/true);
    }

    return llvm::Error::success();
}

llvm::Error MLIRGenerator::InjectConstexprs(llvm::StringRef constexprs_json) {
    return InjectConstexprsIntoModule(*this, ctx_->ir_context, constexprs_json);
}

llvm::Error MLIRGenerator::VisitFunctionDef(ast::FunctionDef *func,
                                            llvm::StringRef constexprs_json) {
    return VisitFunctionDefWithType(func, FunctionType::kGlobalKernel,
                                    constexprs_json);
}

llvm::Error
MLIRGenerator::VisitFunctionDefWithType(ast::FunctionDef *func,
                                        FunctionType func_type,
                                        llvm::StringRef constexprs_json) {
    if (!func)
        return llvm::Error::success();

    CreateModule();

    if (auto E = InjectConstexprs(constexprs_json))
        return E;
    impl_->SnapshotModuleSymbolTable();

    FunctionGenerator function_generator(*impl_, func_type);
    function_generator.Generate(func);
    return llvm::Error::success();
}

void MLIRGenerator::RegisterJitDependency(ast::FunctionDef *func) {
    if (!func) {
        return;
    }
    impl_->RegisterJitDependency(func);
}

MLIRGeneratorImpl::MLIRGeneratorImpl(GeneratorContext *context)
    : ctx_(context),
      named_module_registry_(context ? context->ir_context : nullptr) {
    named_module_registry_.Initialize();
}

static std::string AddressSpaceTag(mlir::Attribute memorySpace) {
    if (!memorySpace) {
        return "default";
    }
    if (auto gpuSpace =
            mlir::dyn_cast<mlir::gpu::AddressSpaceAttr>(memorySpace)) {
        switch (gpuSpace.getValue()) {
        case mlir::gpu::AddressSpace::Global:
            return "global";
        case mlir::gpu::AddressSpace::Workgroup:
            return "workgroup";
        case mlir::gpu::AddressSpace::Private:
            return "private";
        default:
            break;
        }
        return "as" + std::to_string(static_cast<int>(gpuSpace.getValue()));
    }
    if (auto intSpace = mlir::dyn_cast<mlir::IntegerAttr>(memorySpace)) {
        return "as" + std::to_string(intSpace.getInt());
    }
    return "unknown";
}

static std::string ConstexprTag(llvm::StringRef name, mlir::Attribute value) {
    std::string tag = "c";
    tag.append(name.begin(), name.end());
    tag.push_back('_');
    if (auto int_attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(value)) {
        auto v = int_attr.getInt();
        if (v < 0) {
            tag.append("neg");
            tag.append(std::to_string(-v));
        } else {
            tag.append(std::to_string(v));
        }
        return tag;
    }
    if (auto float_attr = mlir::dyn_cast_or_null<mlir::FloatAttr>(value)) {
        std::string buffer;
        llvm::raw_string_ostream os(buffer);
        float_attr.getValue().print(os);
        os.flush();
        for (char &ch : buffer) {
            if (!std::isalnum(static_cast<unsigned char>(ch))) {
                ch = '_';
            }
        }
        tag.append(buffer);
        return tag;
    }
    tag.append("unknown");
    return tag;
}

llvm::SmallVector<std::string, 4>
MLIRGeneratorImpl::GetFunctionAddressSpaceTags(
    ast::FunctionDef *func,
    const ArgAddressSpaceMap *arg_address_spaces) const {
    llvm::SmallVector<std::string, 4> tags;
    if (!func || !arg_address_spaces || arg_address_spaces->empty()) {
        return tags;
    }
    auto *args = func->GetArguments();
    if (!args) {
        return tags;
    }
    for (auto *arg : args->GetArgs()) {
        if (!arg) {
            continue;
        }
        if (auto *attr_expr = llvm::dyn_cast_or_null<ast::AttributeExpr>(
                arg->GetAnnotation())) {
            if (attr_expr->GetAttr() == "constexpr") {
                continue;
            }
        }
        auto it = arg_address_spaces->find(arg->GetArgName());
        if (it == arg_address_spaces->end()) {
            continue;
        }
        tags.push_back(AddressSpaceTag(it->second));
    }
    return tags;
}

llvm::SmallVector<std::string, 4> MLIRGeneratorImpl::GetFunctionConstexprTags(
    ast::FunctionDef *func,
    const ConstexprBindingMap *constexpr_bindings) const {
    llvm::SmallVector<std::string, 4> tags;
    if (!func || !constexpr_bindings || constexpr_bindings->empty()) {
        return tags;
    }
    auto *args = func->GetArguments();
    if (!args) {
        return tags;
    }
    for (auto *arg : args->GetArgs()) {
        if (!arg) {
            continue;
        }
        auto *attr_expr =
            llvm::dyn_cast_or_null<ast::AttributeExpr>(arg->GetAnnotation());
        if (!attr_expr || attr_expr->GetAttr() != "constexpr") {
            continue;
        }
        auto it = constexpr_bindings->find(arg->GetArgName());
        if (it == constexpr_bindings->end()) {
            continue;
        }
        tags.push_back(ConstexprTag(arg->GetArgName(), it->second));
    }
    return tags;
}

std::string MLIRGeneratorImpl::GetFunctionScopeName(
    ast::FunctionDef *func, const ArgAddressSpaceMap *arg_address_spaces,
    const ConstexprBindingMap *constexpr_bindings) {
    if (!func) {
        return {};
    }
    const auto &name = func->GetName();
    if (name.empty()) {
        return {};
    }
    auto tags = GetFunctionAddressSpaceTags(func, arg_address_spaces);
    auto constexpr_tags = GetFunctionConstexprTags(func, constexpr_bindings);
    tags.append(constexpr_tags.begin(), constexpr_tags.end());
    return MangleFunctionName({}, name, tags);
}

std::string MLIRGeneratorImpl::GetMangledFunctionName(
    ast::FunctionDef *func, const ArgAddressSpaceMap *arg_address_spaces,
    llvm::StringRef name_prefix,
    const ConstexprBindingMap *constexpr_bindings) {
    if (!func) {
        return {};
    }
    const auto &name = func->GetName();
    if (name.empty()) {
        return {};
    }
    llvm::SmallVector<std::string, 4> scope;
    if (!name_prefix.empty()) {
        scope.push_back(name_prefix.str());
    }
    auto tags = GetFunctionAddressSpaceTags(func, arg_address_spaces);
    auto constexpr_tags = GetFunctionConstexprTags(func, constexpr_bindings);
    tags.append(constexpr_tags.begin(), constexpr_tags.end());
    return MangleFunctionName(scope, name, tags);
}

mlir::ModuleOp MLIRGeneratorImpl::CreateModule() {
    if (module_) {
        return module_; // Module already created
    }
    mlir::OpBuilder builder(ctx_->ir_context->GetMLIRContext());
    module_ = mlir::ModuleOp::create(builder.getUnknownLoc());
    ctx_->syms->DeclareModules(module_);
    return module_;
}

void MLIRGeneratorImpl::RegisterJitDependency(ast::FunctionDef *func) {
    if (!func) {
        return;
    }
    const auto &name = func->GetName();
    if (name.empty()) {
        return;
    }
    if (!ctx_ || !ctx_->GetCurrentFunctionGenerator()) {
        jit_function_deps_[name] = func;
    }
    ctx_->syms->GetCurrentFrame().AddFunction(
        name,
        SymbolScope::Function(
            [func](ast::Call *call_expr, GeneratorContext *gen_ctx,
                   llvm::ArrayRef<mlir::Value> resolved_args) -> mlir::Value {
                if (!gen_ctx || !gen_ctx->GetCurrentFunctionGenerator()) {
                    return mlir::Value();
                }
                auto *expr_generator = gen_ctx->GetExprGenerator();
                return expr_generator ? expr_generator->GenerateJitFunctionCall(
                                            call_expr, func, resolved_args)
                                      : mlir::Value();
            }));
}

void MLIRGeneratorImpl::InitializeSymbolTable() {
    if (!ctx_) {
        return;
    }
    ctx_->syms = std::make_unique<SymbolTable>(ctx_, &named_module_registry_);
    ctx_->syms->Initialize();
}

void MLIRGeneratorImpl::SnapshotModuleSymbolTable() {
    if (!ctx_ || !ctx_->syms) {
        return;
    }
    module_syms_ = ctx_->syms->Clone();
}

void MLIRGeneratorImpl::HandleImport(ast::Import *import_stmt) {
    if (!import_stmt) {
        return;
    }

    ctx_->current_source_loc = import_stmt->GetSourceRange().getBegin();

    for (const auto &alias : import_stmt->GetNames()) {
        const std::string &module_name = alias.name;
        const std::string &as_name = alias.asname;
        ctx_->syms->ImportModule(module_name, as_name);
    }
}

void MLIRGeneratorImpl::HandleImportFrom(ast::ImportFrom *import_from_stmt) {
    if (!import_from_stmt) {
        return;
    }

    ctx_->current_source_loc = import_from_stmt->GetSourceRange().getBegin();

    const std::string &module_name = import_from_stmt->GetModule();
    for (const auto &alias : import_from_stmt->GetNames()) {
        const std::string &symbol_name = alias.name;
        const std::string &as_name = alias.asname;
        ctx_->syms->ImportSymbolFrom(module_name, symbol_name, as_name);
    }
}

std::optional<MLIRGenerator::FunctionType>
MLIRGeneratorImpl::EnsureMarker(ast::Expr *decorator) {
    if (!decorator) {
        return std::nullopt;
    }

    ctx_->current_source_loc = decorator->GetSourceRange().getBegin();

    auto symbol = ctx_->syms->ResolveSymbol(decorator, std::nullopt);
    if (!symbol) {
        return std::nullopt;
    }
    if (!symbol->isa(SymbolTable::SymbolKind::kValue)) {
        return std::nullopt;
    }

    auto marker_value = symbol->value;
    if (!marker_value) {
        return std::nullopt;
    }
    auto *marker_op = marker_value.getDefiningOp();
    if (!marker_op || !marker_op->hasAttr("is_marker")) {
        return std::nullopt;
    }

    auto purpose_attr =
        marker_op->getAttrOfType<mlir::StringAttr>("marker_purpose");
    if (!purpose_attr) {
        return std::nullopt;
    }

    auto purpose = purpose_attr.getValue();
    if (purpose == "jit") {
        return MLIRGenerator::FunctionType::kGlobalKernel;
    }

    return std::nullopt;
}

std::optional<MLIRGenerator::FunctionType>
MLIRGeneratorImpl::ResolveFunctionType(ast::FunctionDef *func) {
    if (!func) {
        return std::nullopt;
    }

    const auto &name = func->GetName();
    if (!name.empty() &&
        jit_function_deps_.find(name) != jit_function_deps_.end()) {
        return MLIRGenerator::FunctionType::kPrivateFunction;
    }

    for (auto *decorator : func->GetDecorators()) {
        if (auto marker_type = EnsureMarker(decorator)) {
            return marker_type;
        }
    }

    return MLIRGenerator::FunctionType::kHostFunction;
}

mlir::ModuleOp MLIRGeneratorImpl::Generate(ast::ASTNode *root) {
    CreateModule();

    if (!root) {
        return module_;
    }

    auto *module = llvm::dyn_cast<ast::Module>(root);
    if (!module) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         root->GetSourceRange().getBegin())
            << "Expected AST module root for MLIR generation";
        return module_;
    }

    for (auto *stmt : module->GetBody()) {
        if (auto *import_stmt = llvm::dyn_cast<ast::Import>(stmt)) {
            HandleImport(import_stmt);
            continue;
        }
        if (auto *import_from_stmt = llvm::dyn_cast<ast::ImportFrom>(stmt)) {
            HandleImportFrom(import_from_stmt);
            continue;
        }
    }

    for (auto *stmt : module->GetBody()) {
        if (auto *func = llvm::dyn_cast<ast::FunctionDef>(stmt)) {
            auto func_type = ResolveFunctionType(func);
            if (!func_type ||
                *func_type == MLIRGenerator::FunctionType::kHostFunction) {
                continue;
            }
            SnapshotModuleSymbolTable();
            FunctionGenerator function_generator(*this, *func_type);
            function_generator.Generate(func);
            continue;
        }

        if (llvm::isa<ast::Import>(stmt) || llvm::isa<ast::ImportFrom>(stmt)) {
            continue;
        }

        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         stmt->GetSourceRange().getBegin())
            << "Only function definitions are supported at module scope";
        break;
    }

    return module_;
}

} // namespace causalflow::avelang::ir
