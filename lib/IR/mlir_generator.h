#pragma once

#include "AST/ast.h"
#include "Basic/diagnostic.h"
#include "generator_context.h"

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <mlir/IR/BuiltinOps.h>

namespace causalflow::avelang::ir {

class MLIRGeneratorImpl;
class SymbolTable;
class MLIRGenerator {
  public:
    // Function type classification for GPU emission and linkage.
    enum class FunctionType {
        kHostFunction,
        kPrivateFunction,
        kGlobalKernel,
    };
    explicit MLIRGenerator(
        IRContext *ir_context,
        llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diagnostic_manager);
    ~MLIRGenerator();

    // Create the module and return it for constexpr initialization
    mlir::ModuleOp CreateModule();

    // Generate MLIR from AST into the created module
    mlir::ModuleOp Generate(ast::ASTNode *root);

    // Visit a single function definition. Optionally inject constexpr values.
    llvm::Error VisitFunctionDef(ast::FunctionDef *func,
                                 llvm::StringRef constexprs_json = "");

    // Visit a function definition with an explicit function type.
    llvm::Error VisitFunctionDefWithType(ast::FunctionDef *func,
                                         FunctionType func_type,
                                         llvm::StringRef constexprs_json = "");

    // Register a JIT dependency for lazy generation.
    void RegisterJitDependency(ast::FunctionDef *func);

    // Register an alias for a JIT dependency, used when a constexpr parameter
    // is specialized with a JIT function.
    llvm::Error RegisterJitDependencyAlias(llvm::StringRef alias,
                                           llvm::StringRef dependency_name);

    // Inject constexpr values into the current module.
    llvm::Error InjectConstexprs(llvm::StringRef constexprs_json);

    // Return the current module (created on demand).
    mlir::ModuleOp GetModule();

    // Provide access to symbol table for constexpr initialization
    SymbolTable *GetSymbolTable() { return ctx_->syms.get(); }

  private:
    std::unique_ptr<GeneratorContext> ctx_;
    std::unique_ptr<MLIRGeneratorImpl> impl_;
};

} // namespace causalflow::avelang::ir
