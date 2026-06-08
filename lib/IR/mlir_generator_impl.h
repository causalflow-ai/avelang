#pragma once

#include "AST/ast.h"
#include "AST/ast_nodes_expr.h"
#include "AST/ast_nodes_stmt.h"
#include "AST/visitor.h"
#include "Basic/diagnostic.h"
#include "generator_context.h"
#include "mlir_generator.h"
#include "named_module.h"
#include "symbol_table.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#pragma clang diagnostic pop

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

namespace mlir::func {
class FuncOp;
}

namespace causalflow::avelang::ir {

class MLIRGeneratorImpl;
class FunctionGenerator;
using ArgAddressSpaceMap = std::unordered_map<std::string, mlir::Attribute>;

class ExprGenerator : public ast::ASTVisitor<ExprGenerator, mlir::Value> {
  public:
    ExprGenerator(FunctionGenerator *parent);

    mlir::Value VisitBinOp(ast::BinOp *binop);
    mlir::Value VisitUnaryOp(ast::UnaryOp *unaryop);
    mlir::Value VisitBoolOp(ast::BoolOp *boolop);
    mlir::Value VisitName(ast::Name *name);
    mlir::Value VisitSubscript(ast::Subscript *subscript);
    mlir::Value VisitConstant(ast::Constant *constant);
    mlir::Value VisitTuple(ast::Tuple *tuple);
    mlir::Value VisitCall(ast::Call *call);
    mlir::Value VisitCompare(ast::Compare *compare);
    mlir::Value VisitIfExp(ast::IfExp *if_exp);
    mlir::Value VisitImport(ast::Import *import_stmt);
    mlir::Value VisitImportFrom(ast::ImportFrom *import_from_stmt);

    mlir::Value GenerateFuncCall(ast::Call *call, mlir::func::FuncOp func_op,
                                 llvm::ArrayRef<mlir::Value> resolved_args);
    mlir::Value
    GenerateJitFunctionCall(ast::Call *call, ast::FunctionDef *func,
                            llvm::ArrayRef<mlir::Value> resolved_args);

    FunctionGenerator *GetParent() const { return parent_; }

    mlir::Value CastTensorVector(mlir::Value value, mlir::Type target_type,
                                 clang::SourceLocation loc);
    mlir::Value CreateVoidValue();
    mlir::Location GetMLIRLocation(const ast::ASTNode *node) const;
    mlir::Location GetMLIRLocation(clang::SourceLocation loc) const;

  private:
    mlir::Value EnsureCompatibleTypes(mlir::Value value, mlir::Type source_type,
                                      mlir::Type target_type,
                                      clang::SourceLocation loc);

    FunctionGenerator *parent_;
};

class FunctionGenerator : public ast::ASTVisitor<FunctionGenerator> {
  public:
    explicit FunctionGenerator(MLIRGeneratorImpl &parent,
                               MLIRGenerator::FunctionType function_type,
                               ArgAddressSpaceMap argument_address_spaces = {},
                               std::string name_prefix = {});

    MLIRGeneratorImpl &GetParent() { return parent_; }
    GeneratorContext *GetContext() { return ctx_; }
    mlir::OpBuilder &GetBuilder() { return builder_; }
    mlir::ModuleOp GetModule() const;
    ExprGenerator *GetExprGenerator() { return &expr_generator_; }
    mlir::Block *GetEntryBlock() const { return entry_block_; }
    mlir::Value ResolveMemrefValue(ast::Expr *expr);
    mlir::Location GetMLIRLocation(const ast::ASTNode *node) const;
    mlir::Location GetMLIRLocation(clang::SourceLocation loc) const;

    void Generate(ast::FunctionDef *func);
    void VisitFunctionDef(ast::FunctionDef *func);
    void VisitReturn(ast::Return *ret);
    void VisitAssign(ast::Assign *assign);
    void VisitAugAssign(ast::AugAssign *aug_assign);
    void VisitIf(ast::If *if_stmt);
    void VisitFor(ast::For *for_stmt);
    void VisitWhile(ast::While *while_stmt);
    void VisitExprStmt(ast::ExprStmt *expr_stmt);
    void VisitImport(ast::Import *import_stmt);
    void VisitImportFrom(ast::ImportFrom *import_from_stmt);

    const std::string &GetQualifiedScopePrefix() const {
        return qualified_scope_prefix_;
    }
    const ast::FunctionDef *GetCurrentFunctionDef() const {
        return current_func_;
    }

  private:
    mlir::Value GenerateExpr(ast::Expr *expr);
    bool AssignValueToTarget(ast::Expr *target, mlir::Value value,
                             clang::SourceLocation source_loc);
    bool ResolveNameAssignmentTarget(ast::Name *name_target, mlir::Value value,
                                     clang::SourceLocation source_loc,
                                     mlir::Value &memref,
                                     llvm::SmallVector<mlir::Value> &indices,
                                     bool &assignment_complete);
    bool ResolveSubscriptAssignmentTarget(
        ast::Subscript *subscript_target, mlir::Value &value,
        clang::SourceLocation source_loc, mlir::Value &memref,
        llvm::SmallVector<mlir::Value> &indices, bool &assignment_complete);
    bool BuildSubscriptIndices(ast::Expr *slice_expr,
                               clang::SourceLocation source_loc,
                               llvm::SmallVector<mlir::Value> &indices);
    bool MaterializeAssignableValue(mlir::Value &value,
                                    clang::SourceLocation source_loc,
                                    llvm::StringRef memref_assignment_error);
    void MaybeCreateSubviewForVectorAssignment(
        mlir::Value &memref, llvm::SmallVector<mlir::Value> &indices,
        mlir::Value &value);
    bool ValidateMemrefAssignmentTarget(mlir::Value memref,
                                        llvm::ArrayRef<mlir::Value> indices,
                                        clang::SourceLocation source_loc);
    void AppendZeroIndices(mlir::Location loc, int64_t rank,
                           llvm::SmallVectorImpl<mlir::Value> &indices);
    mlir::Value GenerateIndex(ast::Expr *expr, clang::SourceLocation loc);

    MLIRGeneratorImpl &parent_;
    GeneratorContext *ctx_;
    mlir::OpBuilder builder_;
    ExprGenerator expr_generator_;
    MLIRGenerator::FunctionType function_type_;
    std::string local_symbol_scope_name_;
    std::string name_prefix_;
    std::string qualified_scope_prefix_;
    const ast::FunctionDef *current_func_ = nullptr;
    ArgAddressSpaceMap argument_address_spaces_;
    mlir::Block *entry_block_ = nullptr;
};

class MLIRGeneratorImpl {
  public:
    explicit MLIRGeneratorImpl(GeneratorContext *context);

    mlir::ModuleOp CreateModule();
    mlir::ModuleOp Generate(ast::ASTNode *root);
    void RegisterJitDependency(ast::FunctionDef *func);
    llvm::Error RegisterJitDependencyAlias(llvm::StringRef alias,
                                           llvm::StringRef dependency_name);
    void InitializeSymbolTable();
    void SnapshotModuleSymbolTable();

  private:
    friend class ExprGenerator;
    friend class FunctionGenerator;
    std::string GetMangledFunctionName(
        ast::FunctionDef *func,
        const ArgAddressSpaceMap *arg_address_spaces = nullptr,
        llvm::StringRef name_prefix = {});
    std::string GetFunctionScopeName(
        ast::FunctionDef *func,
        const ArgAddressSpaceMap *arg_address_spaces = nullptr);
    llvm::SmallVector<std::string, 4> GetFunctionAddressSpaceTags(
        ast::FunctionDef *func,
        const ArgAddressSpaceMap *arg_address_spaces) const;
    std::string getArgName(ast::ASTNode *arg);
    void HandleImport(ast::Import *import_stmt);
    void HandleImportFrom(ast::ImportFrom *import_from_stmt);
    std::optional<MLIRGenerator::FunctionType>
    EnsureMarker(ast::Expr *decorator);
    std::optional<MLIRGenerator::FunctionType>
    ResolveFunctionType(ast::FunctionDef *func);

    GeneratorContext *ctx_;

    mlir::ModuleOp module_;
    std::unordered_map<std::string, ast::FunctionDef *> jit_function_deps_;
    std::unordered_map<std::string, mlir::func::FuncOp> jit_function_ops_;
    std::unique_ptr<SymbolTable> module_syms_;
    NamedModuleRegistry named_module_registry_;

    // Separate components for context management and type resolution
    // syms_ is now in the GeneratorContext (ctx_->syms)
};

} // namespace causalflow::avelang::ir
