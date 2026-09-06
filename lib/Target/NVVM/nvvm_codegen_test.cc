#include "IR/ir_context.h"
#include "Target/GPU/lower_to_llvm.h"
#include "nvvm_backend.h"
#include "test/utils/cuda_helper.h"

#include <gtest/gtest.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>

#include <memory>
#include <random>
#include <vector>

using namespace llvm;
using namespace causalflow::avelang::target::nvvm;
using namespace causalflow::avelang::target::gpu;

static const std::string kAxpyMLIRCode = R"(
module {
  func.func @axpy(%arg0: i32 {llvm.name = "a"}, %arg1: i32 {llvm.name = "b"}, %arg2: memref<64xi32> {llvm.name = "x"}, %arg3: memref<64xi32> {llvm.name = "y"}) attributes {ave.gpu_func = 2 : i32} {
    %c0_i32 = arith.constant 0 : i32
    %block_id_x = gpu.block_id  x
    %c0_i32_0 = arith.constant 0 : i32
    %block_dim_x = gpu.block_dim  x
    %0 = arith.muli %block_id_x, %block_dim_x : index
    %c0_i32_1 = arith.constant 0 : i32
    %thread_id_x = gpu.thread_id  x
    %1 = arith.addi %0, %thread_id_x : index
    %2 = memref.load %arg2[%1] : memref<64xi32>
    %3 = arith.muli %arg0, %2 : i32
    %4 = arith.addi %3, %arg1 : i32
    memref.store %4, %arg3[%1] : memref<64xi32>
    return
  }
})";

static const std::string kTMADescriptorMLIRCode = R"(
module {
  func.func @tma_descriptor_test(%arg0: !ave.memref<!ave.layout<dims = [16, 16], strides = [16, 1]>, f32> {llvm.name = "global_mem"}) attributes {ave.gpu_func = 2 : i32} {
    %c16 = arith.constant 16 : i32
    %dims = ave.make_int_tuple %c16, %c16 {is_tuple = true} : i32, i32 -> none
    %c1 = arith.constant 1 : i32
    %strides = ave.make_int_tuple %c16, %c1 {is_tuple = true} : i32, i32 -> none
    %layout = ave.make_layout %dims, %strides : (none, none) -> !ave.layout
    %swizzle = arith.constant 0 : index
    %desc = ave.gpu.nvvm_tma_descriptor %arg0, %layout, %swizzle : !ave.memref<!ave.layout<dims = [16, 16], strides = [16, 1]>, f32>, !ave.layout, index -> !nvgpu.tensormap.descriptor<tensor = memref<16x16xf32, strided<[16, 1]>, 3>, swizzle = none, l2promo = none, oob = zero, interleave = none>
    return
  }
})";

static std::string getTMADescriptorMLIRCode(llvm::StringRef elementType) {
    std::string result = kTMADescriptorMLIRCode;
    std::string replacement = elementType.str();
    size_t offset = 0;
    while ((offset = result.find("f32", offset)) != std::string::npos) {
        result.replace(offset, 3, replacement);
        offset += replacement.size();
    }
    return result;
}

class NVVMCodegenTest : public ::testing::Test {
  protected:
    void SetUp() override {
        backend = std::make_unique<NVVMBackend>();
        llvmContext = std::make_unique<LLVMContext>();
        ir_context = causalflow::avelang::ir::IRContext::Create();

        static std::once_flag initNVPTXFlag;
        std::call_once(initNVPTXFlag, []() {
            LLVMInitializeNVPTXTarget();
            LLVMInitializeNVPTXTargetInfo();
            LLVMInitializeNVPTXTargetMC();
            LLVMInitializeNVPTXAsmPrinter();
        });

        // Initialize compilation options once
        options.triple = "nvptx64-nvidia-cuda";
        options.chipset = "sm_80";
        options.optimization_level = 2;
    }

    void ParseMLIRString(const std::string &mlirCode) {
        LowerToLLVM compiler(ir_context.get());
        module = mlir::parseSourceString<mlir::ModuleOp>(mlirCode,
                                                         compiler.getContext());
        ASSERT_NE(module.get(), nullptr);
        ASSERT_TRUE(mlir::succeeded(mlir::verify(*module)));
    }

    void CompileToLLVMIR() {
        LowerToLLVM compiler(ir_context.get());
        llvmModule = compiler.compile(*module, *llvmContext, options);

        ASSERT_NE(llvmModule, nullptr);
        ASSERT_FALSE(llvmModule->empty());
    }

    // Compile LLVM IR to PTX using the NVVM backend
    void CompileToPTX() {
        auto ptxResult = backend->generateBinary(*llvmModule, options);
        ASSERT_TRUE((bool)ptxResult);

        ptx_code_ = *ptxResult;
        ASSERT_FALSE(ptx_code_.empty());
    }

    std::unique_ptr<NVVMBackend> backend;
    std::unique_ptr<LLVMContext> llvmContext;
    std::unique_ptr<causalflow::avelang::ir::IRContext> ir_context;
    GPUCompilationOptions options;
    mlir::OwningOpRef<mlir::ModuleOp> module;
    std::unique_ptr<Module> llvmModule;
    std::string ptx_code_;
};

TEST_F(NVVMCodegenTest, CollectsTMADescriptorKernelMetadata) {
    ParseMLIRString(kTMADescriptorMLIRCode);

    auto metadata = backend->getKernelMetadata(*module);
    auto *specs = metadata.getArray("tma_descriptor_specs");
    ASSERT_NE(specs, nullptr);
    ASSERT_EQ(specs->size(), 1u);

    auto *spec = (*specs)[0].getAsObject();
    ASSERT_NE(spec, nullptr);
    EXPECT_EQ(spec->getString("arg_name"), "global_mem");
    EXPECT_EQ(spec->getInteger("rank"), 2);
    EXPECT_EQ(spec->getString("dtype"), "CU_TENSOR_MAP_DATA_TYPE_FLOAT32");
    EXPECT_EQ(spec->getInteger("swizzle_kind"), 0);

    auto *globalDims = spec->getArray("global_dims");
    auto *globalStrides = spec->getArray("global_strides");
    auto *boxDims = spec->getArray("box_dims");
    ASSERT_NE(globalDims, nullptr);
    ASSERT_NE(globalStrides, nullptr);
    ASSERT_NE(boxDims, nullptr);
    ASSERT_EQ(globalDims->size(), 2u);
    ASSERT_EQ(globalStrides->size(), 1u);
    ASSERT_EQ(boxDims->size(), 2u);
    EXPECT_EQ((*globalDims)[0].getAsInteger(), 16);
    EXPECT_EQ((*globalDims)[1].getAsInteger(), 16);
    EXPECT_EQ((*globalStrides)[0].getAsInteger(), 64);
    EXPECT_EQ((*boxDims)[0].getAsInteger(), 16);
    EXPECT_EQ((*boxDims)[1].getAsInteger(), 16);
}

TEST_F(NVVMCodegenTest, CollectsSignedI8AndI16TMADescriptorMetadata) {
    auto checkMetadata = [&](llvm::StringRef elementType,
                             llvm::StringRef expectedDataType,
                             int64_t expectedGlobalStride) {
        ParseMLIRString(getTMADescriptorMLIRCode(elementType));

        auto metadata = backend->getKernelMetadata(*module);
        auto *specs = metadata.getArray("tma_descriptor_specs");
        ASSERT_NE(specs, nullptr);
        ASSERT_EQ(specs->size(), 1u);

        auto *spec = (*specs)[0].getAsObject();
        ASSERT_NE(spec, nullptr);
        EXPECT_EQ(spec->getString("dtype"), expectedDataType);
        auto *globalStrides = spec->getArray("global_strides");
        ASSERT_NE(globalStrides, nullptr);
        ASSERT_EQ(globalStrides->size(), 1u);
        EXPECT_EQ((*globalStrides)[0].getAsInteger(), expectedGlobalStride);
    };

    checkMetadata("i8", "CU_TENSOR_MAP_DATA_TYPE_UINT8", 16);
    checkMetadata("i16", "CU_TENSOR_MAP_DATA_TYPE_UINT16", 32);
}

TEST_F(NVVMCodegenTest, ExecuteAxpyOnGPU) {
    static const int kN = 64;
    ParseMLIRString(kAxpyMLIRCode);
    CompileToLLVMIR();
    CompileToPTX();

    CUmodule cuModule;
    CUfunction axpyFunction;
    CUcontext cuContext;

    CUresult initResult = cuInit(0);
    if (initResult != CUDA_SUCCESS) {
        const char *error_string = nullptr;
        cuGetErrorString(initResult, &error_string);
        GTEST_SKIP() << "CUDA initialization failed: "
                     << (error_string ? error_string : "unknown error");
        return;
    }

    int deviceCount = 0;
    CUresult deviceCountResult = cuDeviceGetCount(&deviceCount);
    if (deviceCountResult != CUDA_SUCCESS || deviceCount == 0) {
        const char *error_string = nullptr;
        cuGetErrorString(deviceCountResult, &error_string);
        GTEST_SKIP() << "CUDA device unavailable: "
                     << (error_string ? error_string : "no devices");
        return;
    }
    CUdevice device;
    CheckCuError(cuDeviceGet(&device, 0));

#if CUDA_VERSION >= 13000
    // CUDA version >= 13.0, use new context creation API
    CUctxCreateParams ctxParams = {0};
    CheckCuError(cuCtxCreate(&cuContext, &ctxParams, 0, device));
#else
    // CUDA version < 13.0, use legacy context creation API
    CheckCuError(cuCtxCreate(&cuContext, 0, device));
#endif
    CheckCuError(cuModuleLoadData(&cuModule, ptx_code_.c_str()));
    CheckCuError(cuModuleGetFunction(&axpyFunction, cuModule, "axpy"));

    CUdeviceptr d_x, d_y;
    CheckCuError(cuMemAlloc(&d_x, kN * sizeof(int)));
    CheckCuError(cuMemAlloc(&d_y, kN * sizeof(int)));

    std::vector<int> h_x(kN), h_y(kN);
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 100);
    for (int i = 0; i < kN; i++) {
        h_x[i] = dis(gen);
    }
    CheckCuError(cuMemcpyHtoD(d_x, h_x.data(), kN * sizeof(int)));
    CheckCuError(cuMemsetD32(d_y, 0, kN));

    struct {
        int a;
        int b;
        void *x;
        void *y;
    } params = {
        .a = 2,
        .b = 3,
        .x = reinterpret_cast<void *>(d_x),
        .y = reinterpret_cast<void *>(d_y),
    };
    static constexpr size_t kParamSize = sizeof(params);

    void *config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, &params,
                      CU_LAUNCH_PARAM_BUFFER_SIZE,
                      const_cast<size_t *>(&kParamSize), CU_LAUNCH_PARAM_END};

    CheckCuError(cuLaunchKernel(axpyFunction, 1, 1, 1, kN, 1, 1, 0, nullptr,
                                nullptr, config));
    CheckCuError(cuMemcpyDtoH(h_y.data(), d_y, kN * sizeof(int)));
    CheckCuError(cuCtxSynchronize());

    for (int i = 0; i < kN; i++) {
        ASSERT_EQ(h_y[i], 2 * h_x[i] + 3);
    }

    CheckCuError(cuMemFree(d_x));
    CheckCuError(cuMemFree(d_y));
    CheckCuError(cuModuleUnload(cuModule));
    CheckCuError(cuCtxDestroy(cuContext));
}
