#include "gpu_passes.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#pragma clang diagnostic pop

#include <gtest/gtest.h>

namespace causalflow::avelang::target::gpu {

class GpuOutliningTest : public ::testing::Test {
  protected:
    void SetUp() override {
        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::gpu::GPUDialect>();
        context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    }

    mlir::MLIRContext context;
};

TEST_F(GpuOutliningTest, OutlineGpuKernelFunction) {
    // Create a module with functions marked as JIT functions
    // (ave.gpu_func = kPrivateFunction or kGlobalKernel)
    const char *mlirCode = R"(
        module {
            func.func @host_func() {
                return
            }
            func.func @gpu_kernel() attributes {ave.gpu_func = 2 : i32} {
                return
            }
            func.func @jit_func() attributes {ave.gpu_func = 1 : i32} {
                return
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirCode, &context);
    ASSERT_TRUE(module);

    // Verify initial state - should have 3 functions
    int funcCount = 0;
    module->walk([&](mlir::func::FuncOp) { funcCount++; });
    EXPECT_EQ(funcCount, 3);

    // Verify no GPU module initially
    int gpuModuleCount = 0;
    module->walk([&](mlir::gpu::GPUModuleOp) { gpuModuleCount++; });
    EXPECT_EQ(gpuModuleCount, 0);

    // Run the GPU outlining pass
    mlir::PassManager pm(&context);
    pm.addPass(createGpuOutliningPass());

    auto result = pm.run(module.get());
    ASSERT_TRUE(mlir::succeeded(result));

    // After outlining:
    // - Should have 2 func.func (host_func and jit_func inside GPU module)
    // - Should have 1 GPU module with 1 gpu.func (gpu_kernel) and 1 func.func
    // (jit_func)
    funcCount = 0;
    module->walk([&](mlir::func::FuncOp funcOp) {
        funcCount++;
        // Remaining functions should not have ave.gpu_func attribute
        auto gpuFuncAttr =
            funcOp->getAttrOfType<mlir::IntegerAttr>("ave.gpu_func");
        EXPECT_FALSE(gpuFuncAttr);
    });
    EXPECT_EQ(funcCount, 2);

    // Should have exactly one GPU module
    gpuModuleCount = 0;
    module->walk([&](mlir::gpu::GPUModuleOp gpuModule) {
        gpuModuleCount++;

        // GPU module should contain exactly one gpu.func (gpu_kernel)
        int gpuFuncCount = 0;
        gpuModule.walk([&](mlir::gpu::GPUFuncOp gpuFunc) {
            gpuFuncCount++;
            // Should be gpu_kernel
            EXPECT_TRUE(gpuFunc.getName() == "gpu_kernel");
            // Should not have ave.gpu_func attribute
            EXPECT_FALSE(gpuFunc->hasAttr("ave.gpu_func"));
        });
        EXPECT_EQ(gpuFuncCount, 1);

        // GPU module should also contain one func.func (jit_func)
        int jitFuncCount = 0;
        gpuModule.walk([&](mlir::func::FuncOp funcOp) {
            jitFuncCount++;
            // Should be jit_func
            EXPECT_TRUE(funcOp.getName() == "jit_func");
            // Should not have ave.gpu_func attribute
            EXPECT_FALSE(funcOp->hasAttr("ave.gpu_func"));
        });
        EXPECT_EQ(jitFuncCount, 1);
    });
    EXPECT_EQ(gpuModuleCount, 1);
}

TEST_F(GpuOutliningTest, NoGpuJitFunctions) {
    // Create a module with no GPU JIT functions (only host functions)
    const char *mlirCode = R"(
        module {
            func.func @host_func() {
                return
            }
            func.func @another_host_func() {
                return
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirCode, &context);
    ASSERT_TRUE(module);

    // Run the GPU outlining pass
    mlir::PassManager pm(&context);
    pm.addPass(createGpuOutliningPass());

    auto result = pm.run(module.get());
    ASSERT_TRUE(mlir::succeeded(result));

    // Should still have 2 func.func and no GPU modules
    int funcCount = 0;
    module->walk([&](mlir::func::FuncOp) { funcCount++; });
    EXPECT_EQ(funcCount, 2);

    int gpuModuleCount = 0;
    module->walk([&](mlir::gpu::GPUModuleOp) { gpuModuleCount++; });
    EXPECT_EQ(gpuModuleCount, 0);
}

TEST_F(GpuOutliningTest, PreservesWorkgroupAllocationAlignment) {
    const char *mlirCode = R"(
        module {
            func.func @gpu_kernel() attributes {ave.gpu_func = 2 : i32} {
                %buffer = memref.alloca() {alignment = 128 : i64} : memref<32xi8, #gpu.address_space<workgroup>>
                return
            }
        }
    )";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirCode, &context);
    ASSERT_TRUE(module);

    mlir::PassManager pm(&context);
    pm.addPass(createGpuOutliningPass());
    ASSERT_TRUE(mlir::succeeded(pm.run(module.get())));

    mlir::gpu::GPUFuncOp kernel;
    module->walk([&](mlir::gpu::GPUFuncOp func) { kernel = func; });
    ASSERT_TRUE(kernel);
    ASSERT_EQ(kernel.getNumWorkgroupAttributions(), 1u);
    auto alignment = kernel.getWorkgroupAttributionAttr(0, "llvm.align");
    ASSERT_TRUE(alignment);
    EXPECT_EQ(mlir::cast<mlir::IntegerAttr>(alignment).getInt(), 128);
}

} // namespace causalflow::avelang::target::gpu
