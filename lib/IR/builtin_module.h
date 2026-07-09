#pragma once

#include "named_module.h"

#include <llvm/ADT/ArrayRef.h>
#include <utility>

namespace causalflow::avelang::ir {

class IRContext;
struct GeneratorContext;

// AveLang-specific module implementation
class AveLangModule : public NamedModule {
  public:
    explicit AveLangModule(IRContext *ir_context);
    void Initialize() override;
    void DeclareModules(mlir::ModuleOp module) override;
    mlir::Type CreateTensorType(ast::Call *call_expr,
                                GeneratorContext *ctx) const;
    mlir::Type CreatePointerType(ast::Call *call_expr,
                                 GeneratorContext *ctx) const;
    mlir::Type CreateLayoutType(ast::Call *call_expr,
                                GeneratorContext *ctx) const;
    mlir::Value
    CreateConvertFunction(ast::Call *call_expr, GeneratorContext *ctx,
                          llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateBitcastFunction(ast::Call *call_expr, GeneratorContext *ctx,
                          llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateAbsFunction(ast::Call *call_expr, GeneratorContext *ctx,
                      llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateExpFunction(ast::Call *call_expr, GeneratorContext *ctx,
                      llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateExp2Function(ast::Call *call_expr, GeneratorContext *ctx,
                       llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateFmaFunction(ast::Call *call_expr, GeneratorContext *ctx,
                      llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateFmaxFunction(ast::Call *call_expr, GeneratorContext *ctx,
                       llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateTanhFunction(ast::Call *call_expr, GeneratorContext *ctx,
                       llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateLogFunction(ast::Call *call_expr, GeneratorContext *ctx,
                      llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateLog2Function(ast::Call *call_expr, GeneratorContext *ctx,
                       llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateErfFunction(ast::Call *call_expr, GeneratorContext *ctx,
                      llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateSqrtFunction(ast::Call *call_expr, GeneratorContext *ctx,
                       llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateRangeFunction(ast::Call *call_expr, GeneratorContext *ctx,
                        llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateShuffleFunction(ast::Call *call_expr, GeneratorContext *ctx,
                          llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateShuffleUpFunction(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateShuffleDownFunction(ast::Call *call_expr, GeneratorContext *ctx,
                              llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateShuffleXorFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateMakeTensorFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateMakeLocalFunction(ast::Call *call_expr, GeneratorContext *ctx,
                            llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateMakeSharedFunction(ast::Call *call_expr, GeneratorContext *ctx,
                             llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateSubviewFunction(ast::Call *call_expr, GeneratorContext *ctx,
                          llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateFullFunction(ast::Call *call_expr, GeneratorContext *ctx,
                       llvm::ArrayRef<mlir::Value> resolved_args) const;
    mlir::Value
    CreateViewFunction(ast::Call *call_expr, GeneratorContext *ctx,
                       llvm::ArrayRef<mlir::Value> resolved_args) const;

  private:
    IRContext *ir_context_;
    std::unique_ptr<NamedModule> nvvm_module_;
    std::unique_ptr<NamedModule> amdgpu_module_;
};

// Forward declaration for NVVM intrinsic module
std::unique_ptr<NamedModule> CreateNVVMIntrinsicModule();

// Forward declaration for AMDGPU intrinsic module
std::unique_ptr<NamedModule> CreateAMDGPUIntrinsicModule();

} // namespace causalflow::avelang::ir
