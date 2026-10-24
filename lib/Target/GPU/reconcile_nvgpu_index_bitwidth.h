#pragma once

#include <memory>

namespace mlir {
class Pass;
}

namespace causalflow::avelang::target::gpu {

std::unique_ptr<mlir::Pass> createReconcileNVGPUIndexBitwidthPass();

} // namespace causalflow::avelang::target::gpu
