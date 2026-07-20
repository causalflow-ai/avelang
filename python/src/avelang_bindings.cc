#include "AST/ast.h"
#include "AST/ast_context.h"
#include "AST/ast_nodes_stmt.h"
#include "Basic/diagnostic.h"
#include "Dialect/AveLang/IR/AveLangOps.h"
#include "Driver/driver.h"
#include "Frontend/avelang_parser.h"
#include "IR/ir_context.h"
#include "IR/mlir_generator.h"
#include "IR/type_system.h"
#include "Target/GPU/gpu_backend.h"
#include "Target/GPU/lower_to_llvm.h"

#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <set>
#include <vector>

namespace py = pybind11;
using namespace llvm;
using namespace causalflow::avelang;
namespace cf = causalflow::avelang::dialect;

static ir::MLIRGenerator::FunctionType
parseFunctionType(const std::string &function_type) {
    if (function_type == "kernel" || function_type == "global_kernel") {
        return ir::MLIRGenerator::FunctionType::kGlobalKernel;
    }
    if (function_type == "jit") {
        return ir::MLIRGenerator::FunctionType::kPrivateFunction;
    }
    if (function_type == "host") {
        return ir::MLIRGenerator::FunctionType::kHostFunction;
    }
    throw std::runtime_error("Unknown function_type: " + function_type);
}

static std::string NormalizeTargetTriple(llvm::StringRef triple) {
    if (triple.empty()) {
        return triple.str();
    }
    std::string lower = triple.lower();
    if (lower == "amdgpu" || lower == "amdgcn" || lower == "rocm" ||
        lower == "hip") {
        return "amdgcn-amd-amdhsa";
    }
    return triple.str();
}

static std::string FormatMLIRLocation(mlir::Location loc) {
    std::string location;
    llvm::raw_string_ostream os(location);
    loc.print(os);
    os.flush();
    return location;
}

static bool HasMeaningfulMLIRLocation(llvm::StringRef location) {
    return !location.empty() && location != "loc(unknown)";
}

static std::string
DescribeRemainingTranslationBlockers(mlir::Operation *operation) {
    if (!operation) {
        return {};
    }

    std::set<std::string> blockers;
    operation->walk([&](mlir::UnrealizedConversionCastOp cast) {
        std::string location = FormatMLIRLocation(cast.getLoc());
        if (!HasMeaningfulMLIRLocation(location)) {
            return;
        }
        blockers.insert(location + ": builtin.unrealized_conversion_cast");
    });

    if (blockers.empty()) {
        return {};
    }

    std::string message = "remaining translation blockers:\n";
    for (const auto &blocker : blockers) {
        message += "  " + blocker + "\n";
    }
    return message;
}

static std::optional<int64_t> GetConstantIntValue(mlir::Value value) {
    if (!value) {
        return std::nullopt;
    }
    if (auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto intAttr =
                mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
            return intAttr.getInt();
        }
    }
    return std::nullopt;
}

static bool ExtractConstantTupleValues(mlir::Value tupleValue,
                                       llvm::SmallVectorImpl<int64_t> &values) {
    if (auto tupleOp = tupleValue.getDefiningOp<cf::MakeIntTupleOp>()) {
        for (auto elem : tupleOp.getElements()) {
            auto value = GetConstantIntValue(elem);
            if (!value) {
                return false;
            }
            values.push_back(*value);
        }
        return true;
    }
    return false;
}

static std::optional<uint64_t> GetTMAElementSize(mlir::Type type) {
    if (type.isInteger()) {
        return type.getIntOrFloatBitWidth() / 8;
    }
    if (type.isF16() || type.isBF16()) {
        return 2;
    }
    if (type.isF32()) {
        return 4;
    }
    if (type.isF64()) {
        return 8;
    }
    return std::nullopt;
}

static std::optional<std::string> GetTMADataType(mlir::Type type,
                                                 ir::TypeInfo typeInfo) {
    if (type.isInteger()) {
        unsigned width = type.getIntOrFloatBitWidth();
        bool isUnsigned = typeInfo.is_unsigned_integer.value_or(false);
        if (isUnsigned && width == 8) {
            return "CU_TENSOR_MAP_DATA_TYPE_UINT8";
        }
        if (isUnsigned && width == 16) {
            return "CU_TENSOR_MAP_DATA_TYPE_UINT16";
        }
        if (isUnsigned && width == 32) {
            return "CU_TENSOR_MAP_DATA_TYPE_UINT32";
        }
        if (!isUnsigned && width == 32) {
            return "CU_TENSOR_MAP_DATA_TYPE_INT32";
        }
        if (isUnsigned && width == 64) {
            return "CU_TENSOR_MAP_DATA_TYPE_UINT64";
        }
        if (!isUnsigned && width == 64) {
            return "CU_TENSOR_MAP_DATA_TYPE_INT64";
        }
        return std::nullopt;
    }
    if (type.isF16()) {
        return "CU_TENSOR_MAP_DATA_TYPE_FLOAT16";
    }
    if (type.isF32()) {
        return "CU_TENSOR_MAP_DATA_TYPE_FLOAT32";
    }
    if (type.isF64()) {
        return "CU_TENSOR_MAP_DATA_TYPE_FLOAT64";
    }
    if (type.isBF16()) {
        return "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16";
    }
    return std::nullopt;
}

static mlir::MemRefType GetBuiltinMemRefType(mlir::Type type) {
    if (auto builtinType = mlir::dyn_cast<mlir::MemRefType>(type)) {
        return builtinType;
    }
    if (auto aveType = mlir::dyn_cast<cf::MemRefType>(type)) {
        mlir::MemRefLayoutAttrInterface layout;
        auto strides = aveType.getStrides();
        if (!strides.empty()) {
            layout = mlir::StridedLayoutAttr::get(type.getContext(),
                                                  /*offset=*/0, strides);
        }
        return mlir::MemRefType::get(aveType.getShape(),
                                     aveType.getElementType(), layout,
                                     aveType.getMemorySpace());
    }
    return {};
}

static py::list ToPythonList(llvm::ArrayRef<int64_t> values) {
    py::list list;
    for (int64_t value : values) {
        list.append(value);
    }
    return list;
}

class AveLangCompiler {
  public:
    AveLangCompiler()
        : diag_opts_(std::make_unique<clang::DiagnosticOptions>()),
          diag_consumer_(errs(), *diag_opts_),
          diag_manager_(new basic::DiagnosticManager(&diag_consumer_)),
          driver_(diag_manager_, llvm_context_) {}

    std::string
    compileToBinary(const std::string &source_code,
                    const std::string &target_triple = "amdgcn-amd-amdhsa",
                    const std::string &target_chipset = "gfx90a",
                    unsigned opt_level = 2, int num_warps = -1,
                    const std::string &constexprs_json = "[]") {

        // Create compilation options
        causalflow::avelang::driver::CompilationOptions options;
        options.TargetTriple = target_triple;
        options.TargetChipset = target_chipset;
        options.OptLevel = opt_level;
        options.NumWarps = num_warps;
        options.Stage = causalflow::avelang::driver::OutputStage::Binary;

        // Create memory buffer from source
        auto buffer =
            llvm::MemoryBuffer::getMemBuffer(source_code, "<jit_input>");

        // Compile to binary with constexprs
        std::string output;
        llvm::raw_string_ostream os(output);

        auto error = driver_.compileFromBufferWithConstexprs(
            *buffer, os, options, "<jit_input>", constexprs_json);
        if (error) {
            throw std::runtime_error("Compilation failed: " +
                                     llvm::toString(std::move(error)));
        }

        return output;
    }

    py::bytes compileToBinaryBytes(const std::string &source_code,
                                   const std::string &target_triple,
                                   const std::string &target_chipset,
                                   unsigned opt_level = 2, int num_warps = -1,
                                   const std::string &constexprs_json = "[]") {
        std::string binary =
            compileToBinary(source_code, target_triple, target_chipset,
                            opt_level, num_warps, constexprs_json);
        return py::bytes(binary);
    }

  private:
    LLVMContext llvm_context_;
    std::unique_ptr<clang::DiagnosticOptions> diag_opts_;
    clang::TextDiagnosticPrinter diag_consumer_;
    llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diag_manager_;
    driver::Driver driver_;
};

class AveLangMLIRGenerator {
  public:
    AveLangMLIRGenerator();
    void CreateModule();
    void Generate(ast::ASTNode *root);
    void InjectConstexprs(const std::string &constexprs_json);
    void VisitFunctionDef(const py::object &py_function_def,
                          const std::string &constexprs_json = "[]",
                          const std::string &function_type = "kernel");
    void GenerateFromPythonAST(const py::object &py_ast_node);
    void AddJitDependency(const py::object &py_ast_node);
    py::list GetTMADescriptorSpecs();
    std::string GetMLIR();
    std::string GetLLVMIR(const std::string &target_triple,
                          const std::string &target_chipset, unsigned opt_level,
                          int num_warps = -1);
    std::string GetAssembly(const std::string &target_triple,
                            const std::string &target_chipset,
                            unsigned opt_level, int num_warps = -1);
    py::bytes CompileToBinaryBytes(const std::string &target_triple,
                                   const std::string &target_chipset,
                                   unsigned opt_level, int num_warps = -1);

  private:
    void ClearMLIRDiagnostics();
    std::string FormatLoweringFailure(mlir::Operation *operation) const;
    void ThrowIfDiagnosticsFailed(const std::string &context);

    LLVMContext llvm_context_;
    std::unique_ptr<clang::DiagnosticOptions> diag_opts_;
    clang::TextDiagnosticPrinter diag_consumer_;
    llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diag_manager_;
    std::unique_ptr<ir::IRContext> ir_context_;
    llvm::SmallVector<std::string, 4> mlir_errors_;
    std::vector<llvm::IntrusiveRefCntPtr<ast::ASTContext>> dependency_contexts_;
    std::vector<std::unique_ptr<llvm::MemoryBuffer>> owned_buffers_;
    ir::MLIRGenerator generator_;
};

AveLangMLIRGenerator::AveLangMLIRGenerator()
    : diag_opts_(std::make_unique<clang::DiagnosticOptions>()),
      diag_consumer_(errs(), *diag_opts_),
      diag_manager_(new basic::DiagnosticManager(&diag_consumer_)),
      ir_context_(ir::IRContext::Create()),
      generator_(ir_context_.get(), diag_manager_) {
    ir_context_->GetMLIRContext()->getDiagEngine().registerHandler(
        [this](mlir::Diagnostic &diag) {
            if (diag.getSeverity() != mlir::DiagnosticSeverity::Error) {
                return mlir::success();
            }

            std::string message = diag.str();
            std::string location = FormatMLIRLocation(diag.getLocation());
            if (HasMeaningfulMLIRLocation(location)) {
                message = location + ": " + message;
            }
            mlir_errors_.push_back(std::move(message));
            return mlir::success();
        });
}

void AveLangMLIRGenerator::ClearMLIRDiagnostics() { mlir_errors_.clear(); }

std::string
AveLangMLIRGenerator::FormatLoweringFailure(mlir::Operation *operation) const {
    std::string message;
    for (const auto &error : mlir_errors_) {
        message += error + "\n";
    }

    std::string blockers = DescribeRemainingTranslationBlockers(operation);
    if (!blockers.empty()) {
        if (!message.empty()) {
            message += "\n";
        }
        message += blockers;
    }

    if (!message.empty() && message.back() == '\n') {
        message.pop_back();
    }
    return message;
}

void AveLangMLIRGenerator::ThrowIfDiagnosticsFailed(
    const std::string &context) {
    if (diag_manager_ && diag_manager_->GetEngine() &&
        diag_manager_->GetEngine()->hasErrorOccurred()) {
        throw std::runtime_error(context);
    }
}

void AveLangMLIRGenerator::CreateModule() { generator_.CreateModule(); }

void AveLangMLIRGenerator::Generate(ast::ASTNode *root) {
    if (!root) {
        throw std::runtime_error("AST root is null");
    }
    auto module = generator_.Generate(root);
    ThrowIfDiagnosticsFailed("Failed to generate MLIR from AST");
    if (!module) {
        throw std::runtime_error("Failed to generate MLIR from AST");
    }
}

void AveLangMLIRGenerator::InjectConstexprs(
    const std::string &constexprs_json) {
    auto error = generator_.InjectConstexprs(constexprs_json);
    if (error) {
        throw std::runtime_error("Constexpr injection failed: " +
                                 llvm::toString(std::move(error)));
    }
}

void AveLangMLIRGenerator::VisitFunctionDef(const py::object &py_function_def,
                                            const std::string &constexprs_json,
                                            const std::string &function_type) {
    llvm::IntrusiveRefCntPtr<ast::ASTContext> ast_context(
        new ast::ASTContext());
    frontend::AveLangParser parser(ast_context, diag_manager_);
    ast::FunctionDef *func =
        parser.ParsePythonFunctionDef(py_function_def, "<jit_input>");
    if (!func) {
        throw std::runtime_error("Failed to parse Python FunctionDef");
    }
    if (auto buffer = parser.ReleaseOwnedBuffer()) {
        owned_buffers_.push_back(std::move(buffer));
    }
    ThrowIfDiagnosticsFailed("Failed to parse Python FunctionDef");
    auto error = generator_.VisitFunctionDefWithType(
        func, parseFunctionType(function_type), constexprs_json);
    if (error) {
        throw std::runtime_error("VisitFunctionDef failed: " +
                                 llvm::toString(std::move(error)));
    }
    ThrowIfDiagnosticsFailed(
        "VisitFunctionDef failed due to compiler diagnostics");
}

void AveLangMLIRGenerator::GenerateFromPythonAST(
    const py::object &py_ast_node) {
    llvm::IntrusiveRefCntPtr<ast::ASTContext> ast_context(
        new ast::ASTContext());
    frontend::AveLangParser parser(ast_context, diag_manager_);
    ast::ASTNode *root = parser.ParsePythonAST(py_ast_node, "<jit_input>");
    if (!root) {
        throw std::runtime_error("Failed to parse Python AST module");
    }
    if (auto buffer = parser.ReleaseOwnedBuffer()) {
        owned_buffers_.push_back(std::move(buffer));
    }
    auto module = generator_.Generate(root);
    ThrowIfDiagnosticsFailed("Failed to generate MLIR from Python AST");
    if (!module) {
        throw std::runtime_error("Failed to generate MLIR from AST");
    }
}

void AveLangMLIRGenerator::AddJitDependency(const py::object &py_ast_node) {
    llvm::IntrusiveRefCntPtr<ast::ASTContext> ast_context(
        new ast::ASTContext());
    frontend::AveLangParser parser(ast_context, diag_manager_);
    ast::ASTNode *root = parser.ParsePythonAST(py_ast_node, "<jit_input>");
    if (!root) {
        throw std::runtime_error("Failed to parse Python AST module");
    }
    if (auto buffer = parser.ReleaseOwnedBuffer()) {
        owned_buffers_.push_back(std::move(buffer));
    }
    ThrowIfDiagnosticsFailed("Failed to parse Python AST for JIT dependency");
    auto *module = llvm::dyn_cast<ast::Module>(root);
    if (!module) {
        throw std::runtime_error("Expected AST Module for JIT dependency");
    }
    for (auto *stmt : module->GetBody()) {
        if (auto *func = llvm::dyn_cast<ast::FunctionDef>(stmt)) {
            generator_.RegisterJitDependency(func);
        }
    }
    dependency_contexts_.push_back(std::move(ast_context));
}

py::list AveLangMLIRGenerator::GetTMADescriptorSpecs() {
    py::list specs;
    auto module = generator_.GetModule();
    if (!module) {
        return specs;
    }

    module.walk([&](cf::NVVMTMADescriptorOp op) {
        mlir::Value tensor = op.getMemref();
        auto tensorType = GetBuiltinMemRefType(tensor.getType());
        if (!tensorType || !tensorType.hasStaticShape()) {
            return;
        }

        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(tensor);
        if (!blockArg) {
            return;
        }

        auto funcOp = blockArg.getOwner()
                          ? mlir::dyn_cast_or_null<mlir::func::FuncOp>(
                                blockArg.getOwner()->getParentOp())
                          : mlir::func::FuncOp{};
        if (!funcOp) {
            return;
        }
        auto gpuFuncAttr =
            funcOp->getAttrOfType<mlir::IntegerAttr>("ave.gpu_func");
        if (!gpuFuncAttr ||
            gpuFuncAttr.getInt() !=
                static_cast<int>(
                    ir::MLIRGenerator::FunctionType::kGlobalKernel)) {
            return;
        }

        auto argName = funcOp.getArgAttrOfType<mlir::StringAttr>(
            blockArg.getArgNumber(), "llvm.name");
        if (!argName) {
            return;
        }

        auto layoutOp = op.getLayout().getDefiningOp<cf::MakeLayoutOp>();
        if (!layoutOp) {
            return;
        }

        llvm::SmallVector<int64_t> boxDims;
        if (!ExtractConstantTupleValues(layoutOp.getDims(), boxDims) ||
            boxDims.empty()) {
            return;
        }

        auto elementSize = GetTMAElementSize(tensorType.getElementType());
        auto dtype = GetTMADataType(tensorType.getElementType(),
                                    ir::GetTypeInfo(tensor));
        if (!elementSize || !dtype) {
            return;
        }

        llvm::SmallVector<int64_t> globalDims;
        for (int64_t dim : llvm::reverse(tensorType.getShape())) {
            globalDims.push_back(dim);
        }

        llvm::SmallVector<int64_t> defaultStrides(tensorType.getRank(), 1);
        for (int64_t i = tensorType.getRank() - 2; i >= 0; --i) {
            defaultStrides[i] =
                defaultStrides[i + 1] * tensorType.getDimSize(i + 1);
        }

        llvm::SmallVector<int64_t> globalStrides;
        for (int64_t i = tensorType.getRank() - 2; i >= 0; --i) {
            globalStrides.push_back(defaultStrides[i] * *elementSize);
        }

        llvm::SmallVector<int64_t> reversedBoxDims;
        for (int64_t dim : llvm::reverse(boxDims)) {
            reversedBoxDims.push_back(dim);
        }

        py::dict spec;
        spec["arg_name"] = argName.getValue().str();
        spec["rank"] = tensorType.getRank();
        spec["global_dims"] = ToPythonList(globalDims);
        spec["global_strides"] = ToPythonList(globalStrides);
        spec["box_dims"] = ToPythonList(reversedBoxDims);
        spec["dtype"] = *dtype;
        specs.append(std::move(spec));
    });

    return specs;
}

std::string AveLangMLIRGenerator::GetMLIR() {
    auto module = generator_.GetModule();
    std::string output;
    llvm::raw_string_ostream os(output);
    module.print(os);
    return os.str();
}

std::string AveLangMLIRGenerator::GetLLVMIR(const std::string &target_triple,
                                            const std::string &target_chipset,
                                            unsigned opt_level, int num_warps) {
    auto module = generator_.GetModule();
    ThrowIfDiagnosticsFailed(
        "Failed to lower MLIR to LLVM IR due to compiler diagnostics");
    if (!module) {
        throw std::runtime_error("MLIR module is null");
    }

    std::string normalized_triple = NormalizeTargetTriple(target_triple);
    target::gpu::GPUCompilationOptions options;
    options.triple = normalized_triple;
    options.chipset = target_chipset;
    options.optimization_level = opt_level;
    options.num_warps = num_warps;

    auto backend =
        target::gpu::GPUBackendRegistry::getInstance().createBackendForTriple(
            options.triple);
    if (!backend) {
        throw std::runtime_error("No backend available for target: " +
                                 target_triple);
    }
    backend->EnsureInitialized();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();

    target::gpu::LowerToLLVM lowerer(ir_context_.get());
    ClearMLIRDiagnostics();
    auto llvm_module = lowerer.compile(module, llvm_context_, options);
    if (!llvm_module) {
        std::string detail = FormatLoweringFailure(module);
        if (!detail.empty()) {
            throw std::runtime_error("Failed to lower MLIR to LLVM IR:\n" +
                                     detail);
        }
        throw std::runtime_error("Failed to lower MLIR to LLVM IR");
    }

    std::string output;
    llvm::raw_string_ostream os(output);
    llvm_module->print(os, nullptr);
    return os.str();
}

std::string AveLangMLIRGenerator::GetAssembly(const std::string &target_triple,
                                              const std::string &target_chipset,
                                              unsigned opt_level,
                                              int num_warps) {
    auto module = generator_.GetModule();
    ThrowIfDiagnosticsFailed(
        "Failed to lower MLIR to LLVM IR due to compiler diagnostics");
    if (!module) {
        throw std::runtime_error("MLIR module is null");
    }

    std::string normalized_triple = NormalizeTargetTriple(target_triple);
    target::gpu::GPUCompilationOptions options;
    options.triple = normalized_triple;
    options.chipset = target_chipset;
    options.optimization_level = opt_level;
    options.num_warps = num_warps;

    auto backend =
        target::gpu::GPUBackendRegistry::getInstance().createBackendForTriple(
            options.triple);
    if (!backend) {
        throw std::runtime_error("No backend available for target: " +
                                 target_triple);
    }
    backend->EnsureInitialized();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();

    target::gpu::LowerToLLVM lowerer(ir_context_.get());
    ClearMLIRDiagnostics();
    auto llvm_module = lowerer.compile(module, llvm_context_, options);
    if (!llvm_module) {
        std::string detail = FormatLoweringFailure(module);
        if (!detail.empty()) {
            throw std::runtime_error("Failed to lower MLIR to LLVM IR:\n" +
                                     detail);
        }
        throw std::runtime_error("Failed to lower MLIR to LLVM IR");
    }

    auto assembly_or_error = backend->generateAssembly(*llvm_module, options);
    if (!assembly_or_error) {
        throw std::runtime_error("Failed to generate assembly: " +
                                 llvm::toString(assembly_or_error.takeError()));
    }

    return *assembly_or_error;
}

py::bytes
AveLangMLIRGenerator::CompileToBinaryBytes(const std::string &target_triple,
                                           const std::string &target_chipset,
                                           unsigned opt_level, int num_warps) {
    auto module = generator_.GetModule();
    ThrowIfDiagnosticsFailed(
        "Failed to lower MLIR to LLVM IR due to compiler diagnostics");
    if (!module) {
        throw std::runtime_error("MLIR module is null");
    }

    std::string normalized_triple = NormalizeTargetTriple(target_triple);
    target::gpu::GPUCompilationOptions options;
    options.triple = normalized_triple;
    options.chipset = target_chipset;
    options.optimization_level = opt_level;
    options.num_warps = num_warps;

    auto backend =
        target::gpu::GPUBackendRegistry::getInstance().createBackendForTriple(
            options.triple);
    if (!backend) {
        throw std::runtime_error("No backend available for target: " +
                                 target_triple);
    }
    backend->EnsureInitialized();

    target::gpu::LowerToLLVM lowerer(ir_context_.get());
    ClearMLIRDiagnostics();
    auto llvm_module = lowerer.compile(module, llvm_context_, options);
    if (!llvm_module) {
        std::string detail = FormatLoweringFailure(module);
        if (!detail.empty()) {
            throw std::runtime_error("Failed to lower MLIR to LLVM IR:\n" +
                                     detail);
        }
        throw std::runtime_error("Failed to lower MLIR to LLVM IR");
    }

    auto binary_or_error = backend->generateBinary(*llvm_module, options);
    if (!binary_or_error) {
        throw std::runtime_error("Failed to generate binary: " +
                                 llvm::toString(binary_or_error.takeError()));
    }
    return py::bytes(*binary_or_error);
}

PYBIND11_MODULE(_avelang_bindings, m) {
    m.doc() = "ave-lang JIT compiler Python bindings";

    py::class_<AveLangCompiler>(m, "AveLangCompiler")
        .def(py::init<>())
        .def("compile_to_binary_bytes", &AveLangCompiler::compileToBinaryBytes,
             "Compile avelang source to binary bytes", py::arg("source_code"),
             py::arg("target_triple"), py::arg("target_chipset"),
             py::arg("opt_level") = 2, py::arg("num_warps") = -1,
             py::arg("constexprs_json") = "[]");

    py::class_<AveLangMLIRGenerator>(m, "MLIRGenerator")
        .def(py::init<>())
        .def("create_module", &AveLangMLIRGenerator::CreateModule)
        .def("generate", &AveLangMLIRGenerator::Generate, py::arg("root"))
        .def("inject_constexprs", &AveLangMLIRGenerator::InjectConstexprs,
             py::arg("constexprs_json") = "[]")
        .def("visit_function_def", &AveLangMLIRGenerator::VisitFunctionDef,
             py::arg("py_function_def"), py::arg("constexprs_json") = "[]",
             py::arg("function_type") = "kernel")
        .def("generate_from_python_ast",
             &AveLangMLIRGenerator::GenerateFromPythonAST,
             py::arg("py_ast_node"))
        .def("add_jit_dependency", &AveLangMLIRGenerator::AddJitDependency,
             py::arg("py_ast_node"))
        .def("get_tma_descriptor_specs",
             &AveLangMLIRGenerator::GetTMADescriptorSpecs)
        .def("get_mlir", &AveLangMLIRGenerator::GetMLIR)
        .def("get_llvm_ir", &AveLangMLIRGenerator::GetLLVMIR,
             py::arg("target_triple"), py::arg("target_chipset"),
             py::arg("opt_level") = 2, py::arg("num_warps") = -1)
        .def("get_assembly", &AveLangMLIRGenerator::GetAssembly,
             py::arg("target_triple"), py::arg("target_chipset"),
             py::arg("opt_level") = 2, py::arg("num_warps") = -1)
        .def("compile_to_binary_bytes",
             &AveLangMLIRGenerator::CompileToBinaryBytes,
             py::arg("target_triple"), py::arg("target_chipset"),
             py::arg("opt_level"), py::arg("num_warps") = -1);
}
