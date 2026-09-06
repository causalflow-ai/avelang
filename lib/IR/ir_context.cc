#include "ir_context.h"
#include "../Dialect/AveLang/IR/AveLangDialect.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/NVGPU/IR/NVGPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/LLVMIR/NVVMDialect.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Ptr/IR/PtrDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h>

namespace causalflow::avelang::ir {

IRContext::IRContext() {
    mlir::DialectRegistry registry;

    // Register core dialects
    registry.insert<mlir::affine::AffineDialect, mlir::func::FuncDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                    mlir::memref::MemRefDialect, mlir::cf::ControlFlowDialect,
                    mlir::index::IndexDialect, mlir::scf::SCFDialect,
                    mlir::gpu::GPUDialect, mlir::vector::VectorDialect,
                    mlir::nvgpu::NVGPUDialect,
                    mlir::ptr::PtrDialect,
                    causalflow::avelang::dialect::AveLangDialect>();

    registry.insert<mlir::ROCDL::ROCDLDialect, mlir::NVVM::NVVMDialect>();

    // Register LLVM dialect translations
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerROCDLDialectTranslation(registry);
    mlir::registerGPUDialectTranslation(registry);
    mlir::registerNVVMDialectTranslation(registry);

    // Register bufferization interface implementations
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::
        registerBufferizableOpInterfaceExternalModels(registry);

    mlir_context_ = std::make_unique<mlir::MLIRContext>(registry);
    mlir_context_->loadDialect<mlir::affine::AffineDialect>();
    mlir_context_->loadDialect<mlir::func::FuncDialect>();
    mlir_context_->loadDialect<mlir::arith::ArithDialect>();
    mlir_context_->loadDialect<mlir::tensor::TensorDialect>();
    mlir_context_->loadDialect<mlir::cf::ControlFlowDialect>();
    mlir_context_->loadDialect<mlir::scf::SCFDialect>();
    mlir_context_->loadDialect<mlir::gpu::GPUDialect>();
    mlir_context_->loadDialect<mlir::vector::VectorDialect>();
    mlir_context_->loadDialect<mlir::nvgpu::NVGPUDialect>();
    mlir_context_->loadDialect<mlir::ptr::PtrDialect>();
    mlir_context_->loadDialect<mlir::ROCDL::ROCDLDialect>();
    mlir_context_->loadDialect<mlir::NVVM::NVVMDialect>();
    mlir_context_->loadDialect<causalflow::avelang::dialect::AveLangDialect>();
}

std::unique_ptr<IRContext> IRContext::Create() {
    return std::unique_ptr<IRContext>(new IRContext());
}

mlir::MLIRContext *IRContext::GetMLIRContext() const {
    return mlir_context_.get();
}

} // namespace causalflow::avelang::ir
