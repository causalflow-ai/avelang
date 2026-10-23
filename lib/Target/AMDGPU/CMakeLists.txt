add_library(ave-lang-target-amdgpu
  gpu_to_amdgpu_pipeline.cc
  legalize_gpu_shuffle_to_idx_pass.cc
  lower_math_to_amdgpu_pass.cc
  amdgpu_backend.cc
  rocm_installation_detector.cc
  amdgpu_toolchain.cc
)

target_link_libraries(ave-lang-target-amdgpu
  PRIVATE
  MLIRArithToAMDGPU
  MLIRArithToLLVM
  MLIRArithDialect
  MLIRVectorToLLVMPass
  MLIRAffineToStandard
  MLIRAffineTransforms
  MLIRIndexToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRTransforms
  MLIRGPUTransforms
  MLIRBufferizationTransforms
  MLIRGPUDialect
  MLIRAMDGPUDialect
  MLIRAMDGPUToROCDL
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRSCFToControlFlow
  MLIRUBToLLVM
  MLIRROCDLDialect
  MLIRGPUToROCDLTransforms
  MLIRMemRefTransforms
  ave-lang-mlir
  ave-lang-dialect-transforms
  LLVMAMDGPUCodeGen
  LLVMAMDGPUInfo
  LLVMAMDGPUDesc
  LLVMAsmPrinter
  LLVMBitWriter
  LLVMCodeGen
  LLVMTarget
  LLVMCore
  LLVMSupport
  clangDriver
)

target_compile_definitions(ave-lang-target-amdgpu PRIVATE ${LLVM_DEFINITIONS})

# Add AMDGPU codegen test
add_executable(amdgpu_codegen_test amdgpu_codegen_test.cc)
target_link_libraries(amdgpu_codegen_test
  ave-lang-target-amdgpu
  ave-lang-driver
  ave-lang-basic
  ave-lang-ast
  hip::host
  GTest::gtest_main
)

add_test(amdgpu_codegen_test amdgpu_codegen_test)
