add_library(ave-lang-target-nvvm
  gpu_to_nvvm_pipeline.cc
  nvvm_backend.cc
)

target_link_libraries(ave-lang-target-nvvm
  PRIVATE
  MLIRArithToLLVM
  MLIRNVVMToLLVM
  MLIRVectorToLLVMPass
  MLIRAffineToStandard
  MLIRAffineTransforms
  MLIRIndexToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRTransforms
  MLIRGPUTransforms
  MLIRBufferizationTransforms
  MLIRGPUDialect
  MLIRNVGPUDialect
  MLIRSCFToControlFlow
  MLIRNVVMDialect
  MLIRNVGPUToNVVM
  MLIRGPUToNVVMTransforms
  MLIRMemRefTransforms
  MLIRUBToLLVM
  ave-lang-mlir
  ave-lang-dialect-transforms
  LLVMNVPTXCodeGen
  LLVMNVPTXInfo
  LLVMNVPTXDesc
  LLVMAsmPrinter
  LLVMCodeGen
  LLVMTarget
  LLVMCore
  LLVMSupport
)

target_compile_definitions(ave-lang-target-nvvm PRIVATE ${LLVM_DEFINITIONS})

# Add NVVM codegen test
add_executable(nvvm_codegen_test nvvm_codegen_test.cc)
target_link_libraries(nvvm_codegen_test
  ave-lang-target-nvvm
  ave-lang-driver
  ave-lang-basic
  ave-lang-ast
  GTest::gtest_main
  LLVMCore
  LLVMSupport
  LLVMIRReader
  CUDA::cuda_driver
)

add_test(nvvm_codegen_test nvvm_codegen_test)
