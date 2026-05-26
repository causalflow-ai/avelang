#include "Utils/assert.h"
#include "layout_operation.h"
#include "mlir_generator_impl.h"
#include "type_promotion.h"
#include "type_system.h"

#include "Dialect/AveLang/IR/AveLangOps.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Ptr/IR/PtrTypes.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#pragma clang diagnostic pop

#include <string>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

namespace causalflow::avelang::ir {

namespace cf = causalflow::avelang::dialect;

using namespace mlir;

static cf::MemRefType
createAveLangMemRefType(mlir::MLIRContext *context,
                        llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                        mlir::Attribute memorySpace = {},
                        llvm::ArrayRef<int64_t> strides = {}) {
    auto layoutType = cf::LayoutType::get(context, shape, strides);
    return cf::MemRefType::get(context, layoutType, elementType, memorySpace);
}

static bool ExtractTupleElements(mlir::Value value,
                                 llvm::SmallVectorImpl<mlir::Value> &elements) {
    if (!value) {
        return false;
    }
    if (auto tupleOp = value.getDefiningOp<cf::MakeIntTupleOp>()) {
        for (auto elem : tupleOp.getElements()) {
            elements.push_back(elem);
        }
        return true;
    }
    return false;
}

static bool
CanImplicitlyDemoteConstantWithoutPrecisionLoss(mlir::Value value,
                                                mlir::Type targetType) {
    auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constOp) {
        return false;
    }

    auto sourceFloatType = mlir::dyn_cast<mlir::FloatType>(value.getType());
    auto targetFloatType = mlir::dyn_cast<mlir::FloatType>(targetType);
    if (!sourceFloatType || !targetFloatType) {
        return false;
    }

    auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(constOp.getValue());
    if (!floatAttr) {
        return false;
    }

    llvm::APFloat rounded = floatAttr.getValue();
    bool losesInfo = false;
    auto status =
        rounded.convert(targetFloatType.getFloatSemantics(),
                        llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    if (losesInfo || status == llvm::APFloat::opInvalidOp) {
        return false;
    }

    llvm::APFloat roundTrip = rounded;
    bool roundTripLosesInfo = false;
    status = roundTrip.convert(sourceFloatType.getFloatSemantics(),
                               llvm::APFloat::rmNearestTiesToEven,
                               &roundTripLosesInfo);
    if (status == llvm::APFloat::opInvalidOp || roundTripLosesInfo) {
        return false;
    }

    return roundTrip.bitwiseIsEqual(floatAttr.getValue());
}

FunctionGenerator::FunctionGenerator(MLIRGeneratorImpl &parent,
                                     MLIRGenerator::FunctionType function_type,
                                     ArgAddressSpaceMap argument_address_spaces,
                                     std::string name_prefix,
                                     ConstexprBindingMap constexpr_bindings)
    : parent_(parent), ctx_(parent.ctx_),
      builder_(parent.ctx_->ir_context->GetMLIRContext()),
      expr_generator_(this), function_type_(function_type),
      name_prefix_(std::move(name_prefix)),
      argument_address_spaces_(std::move(argument_address_spaces)),
      constexpr_bindings_(std::move(constexpr_bindings)) {
    SS_ASSERT(ctx_);
}

mlir::Location
FunctionGenerator::GetMLIRLocation(const ast::ASTNode *node) const {
    return ctx_->GetMLIRLocation(builder_.getContext(), node);
}

mlir::Location
FunctionGenerator::GetMLIRLocation(clang::SourceLocation loc) const {
    return ctx_->GetMLIRLocation(builder_.getContext(), loc);
}

mlir::ModuleOp FunctionGenerator::GetModule() const { return parent_.module_; }

std::optional<int64_t>
FunctionGenerator::LookupConstexprInteger(llvm::StringRef name) const {
    auto it = constexpr_bindings_.find(name.str());
    if (it == constexpr_bindings_.end() || !it->second) {
        return std::nullopt;
    }
    if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(it->second)) {
        return int_attr.getInt();
    }
    if (auto bool_attr = mlir::dyn_cast<mlir::BoolAttr>(it->second)) {
        return bool_attr.getValue() ? 1 : 0;
    }
    return std::nullopt;
}

mlir::Value FunctionGenerator::GenerateExpr(ast::Expr *expr) {
    if (!expr) {
        return mlir::Value();
    }
    return expr_generator_.Dispatch(expr);
}

void FunctionGenerator::Generate(ast::FunctionDef *func) {
    if (!func) {
        return;
    }

    if (function_type_ == MLIRGenerator::FunctionType::kHostFunction) {
        return;
    }

    mlir::OpBuilder::InsertionGuard insertion_guard(builder_);
    auto generator_guard = ctx_->GetFunctionGeneratorGuard(this);

    current_func_ = func;
    local_symbol_scope_name_ = parent_.GetFunctionScopeName(
        func, &argument_address_spaces_, &constexpr_bindings_);
    qualified_scope_prefix_ = name_prefix_;
    if (!local_symbol_scope_name_.empty()) {
        if (!qualified_scope_prefix_.empty()) {
            qualified_scope_prefix_.push_back('_');
        }
        qualified_scope_prefix_.append(local_symbol_scope_name_);
    }

    auto args = func->GetArguments();
    if (args &&
        (!args->GetPosOnlyArgs().empty() || !args->GetKwOnlyArgs().empty() ||
         args->GetVarArg() || args->GetKwArg())) {
        std::string error_msg =
            "positional / kw / var args are not supported yet: " +
            func->GetName();
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         func->GetSourceRange().getBegin())
            << error_msg;
        return;
    }

    // Create function type
    SmallVector<mlir::Type> argTypes;
    SmallVector<mlir::DictionaryAttr> argAttrs;
    SmallVector<TypeInfo> argTypeInfos;
    SmallVector<std::string> argNames;
    SmallVector<mlir::Location> argLocs;
    if (args) {
        for (auto *arg : args->GetArgs()) {
            auto name = arg->GetArgName();
            auto annotation = arg->GetAnnotation();

            // Check if annotation is "constexpr" - skip these parameters
            // as they're not passed by the launcher.
            if (auto *attr_expr =
                    llvm::dyn_cast<ast::AttributeExpr>(annotation)) {
                if (attr_expr->GetAttr() == "constexpr") {
                    continue;
                }
            }
            auto source_range = arg->GetSourceRange();
            mlir::Location mlir_loc =
                ctx_->GetMLIRLocation(builder_.getContext(), arg);
            if (!annotation) {
                ctx_->diagnostic_manager->Report(
                    basic::DiagnosticCode::kUnimplemented,
                    source_range.getBegin())
                    << "type annotation is required for argument: " << name;
                return;
            }
            auto type = ctx_->syms->ResolveType(annotation);
            if (!type) {
                ctx_->diagnostic_manager->Report(
                    basic::DiagnosticCode::kUnimplemented,
                    source_range.getBegin())
                    << "unsupported type for argument: " + name;
                return;
            }
            if (auto memref_type = mlir::dyn_cast<cf::MemRefType>(type)) {
                auto it = argument_address_spaces_.find(name);
                if (it != argument_address_spaces_.end() && it->second) {
                    type = cf::MemRefType::get(
                        builder_.getContext(), memref_type.getLayout(),
                        memref_type.getElementType(), it->second);
                }
            }

            mlir::SmallVector<mlir::NamedAttribute> attrs;
            if (!mlir::isa<mlir::ptr::PtrType>(type)) {
                auto nameAttr =
                    mlir::StringAttr::get(builder_.getContext(), name);
                attrs.push_back(builder_.getNamedAttr("llvm.name", nameAttr));
            }
            argAttrs.push_back(
                mlir::DictionaryAttr::get(builder_.getContext(), attrs));
            argNames.push_back(name);
            argTypes.push_back(type);
            argTypeInfos.push_back(GetTypeInfo(annotation));
            argLocs.push_back(mlir_loc);
        }
    }

    // Resolve return type if present
    SmallVector<mlir::Type> returnTypes;
    if (auto *returns_expr = func->GetReturns()) {
        auto materialize_value_return_type =
            [&](mlir::Type type) -> mlir::Type {
            if (function_type_ !=
                MLIRGenerator::FunctionType::kPrivateFunction) {
                return type;
            }
            auto memref_type = mlir::dyn_cast<cf::MemRefType>(type);
            if (!memref_type) {
                return type;
            }
            if (!memref_type.hasStaticShape()) {
                ctx_->diagnostic_manager->Report(
                    basic::DiagnosticCode::kUnimplemented,
                    returns_expr->GetSourceRange().getBegin())
                    << "Tensor return values must have a static shape";
                return mlir::Type();
            }
            return mlir::VectorType::get(memref_type.getShape(),
                                         memref_type.getElementType());
        };

        auto resolve_tuple_types = [&](auto *tuple_expr) -> bool {
            for (auto *elem : tuple_expr->GetElts()) {
                auto elem_type = ctx_->syms->ResolveType(elem);
                if (!elem_type) {
                    ctx_->diagnostic_manager->Report(
                        basic::DiagnosticCode::kUnimplemented,
                        returns_expr->GetSourceRange().getBegin())
                        << "Failed to resolve return type for function: " +
                               func->GetName();
                    return false;
                }
                elem_type = materialize_value_return_type(elem_type);
                if (!elem_type) {
                    return false;
                }
                returnTypes.push_back(elem_type);
            }
            if (returnTypes.empty()) {
                ctx_->diagnostic_manager->Report(
                    basic::DiagnosticCode::kUnimplemented,
                    returns_expr->GetSourceRange().getBegin())
                    << "Empty tuple return types are not supported";
                return false;
            }
            return true;
        };

        if (auto *tuple_expr = llvm::dyn_cast<ast::Tuple>(returns_expr)) {
            if (!resolve_tuple_types(tuple_expr)) {
                return;
            }
        } else if (auto *list_expr = llvm::dyn_cast<ast::List>(returns_expr)) {
            if (!resolve_tuple_types(list_expr)) {
                return;
            }
        } else {
            auto return_type = ctx_->syms->ResolveType(returns_expr);
            if (!return_type) {
                ctx_->diagnostic_manager->Report(
                    basic::DiagnosticCode::kUnimplemented,
                    returns_expr->GetSourceRange().getBegin())
                    << "Failed to resolve return type for function: " +
                           func->GetName();
                return;
            }
            return_type = materialize_value_return_type(return_type);
            if (!return_type) {
                return;
            }
            returnTypes.push_back(return_type);
        }
    }

    // GPU kernels cannot return values
    if (function_type_ == MLIRGenerator::FunctionType::kGlobalKernel &&
        !returnTypes.empty()) {
        ctx_->diagnostic_manager->Report(
            basic::DiagnosticCode::kUnimplemented,
            func->GetReturns()->GetSourceRange().getBegin())
            << "GPU kernel functions cannot explicitly return values. "
            << "Kernel function '" + func->GetName() + "' has return type.";
        return;
    }

    std::string mangled_name;
    if (function_type_ == MLIRGenerator::FunctionType::kPrivateFunction) {
        mangled_name =
            parent_.GetMangledFunctionName(func, &argument_address_spaces_,
                                           name_prefix_, &constexpr_bindings_);
        if (mangled_name.empty()) {
            ctx_->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                func->GetSourceRange().getBegin())
                << "Private function has no name";
            return;
        }
    }

    auto FT = builder_.getFunctionType(argTypes, returnTypes);
    builder_.setInsertionPointToEnd(parent_.module_.getBody());
    const auto &func_name =
        function_type_ == MLIRGenerator::FunctionType::kPrivateFunction
            ? mangled_name
            : func->GetName();
    auto func_op = func::FuncOp::create(
        builder_, ctx_->GetMLIRLocation(builder_.getContext(), func), func_name,
        FT);
    func_op->setAttr("ave.gpu_func",
                     builder_.getI32IntegerAttr((int)function_type_));
    func_op.setAllArgAttrs(argAttrs);

    if (function_type_ == MLIRGenerator::FunctionType::kPrivateFunction) {
        func_op.setPrivate();
        parent_.jit_function_ops_[mangled_name] = func_op;
    }

    // Set up function body: entry block + body block.
    auto &entry_block = func_op.getFunctionBody().emplaceBlock();
    entry_block_ = &entry_block;
    entry_block.addArguments(argTypes, argLocs);
    for (size_t i = 0; i < argTypeInfos.size(); ++i) {
        SetTypeInfo(entry_block.getArgument(i), argTypeInfos[i]);
    }
    auto &body_block = func_op.getFunctionBody().emplaceBlock();

    SymbolTable::FrameGuard guard(ctx_->syms.get());

    builder_.setInsertionPointToStart(&entry_block);
    mlir::cf::BranchOp::create(
        builder_, ctx_->GetMLIRLocation(builder_.getContext(), func),
        &body_block);
    builder_.setInsertionPointToStart(&body_block);

    // Add function arguments to symbol table.
    // Note: constexpr parameters are already filtered out above, so all
    // arguments here are runtime parameters that need to be defined.
    for (size_t i = 0; i < argNames.size(); ++i) {
        ctx_->syms->DefineSymbol(argNames[i], entry_block.getArgument(i));
    }

    for (const auto &[name, attr] : constexpr_bindings_) {
        if (!attr) {
            continue;
        }
        auto typed_attr = mlir::dyn_cast<mlir::TypedAttr>(attr);
        if (!typed_attr) {
            continue;
        }
        auto const_value = mlir::arith::ConstantOp::create(
            builder_, ctx_->GetMLIRLocation(builder_.getContext(), func),
            typed_attr);
        ctx_->syms->GetCurrentFrame().AddValue(name, const_value,
                                               /*immutable=*/true);
    }

    for (auto *stmt : func->GetBody()) {
        DispatchStmt(stmt);
    }

    // Add empty return to make function valid
    if (!builder_.getInsertionBlock()->mightHaveTerminator()) {
        func::ReturnOp::create(
            builder_, ctx_->GetMLIRLocation(builder_.getContext(), func));
    }

    // Ensure subsequent functions are inserted at the module level
    builder_.setInsertionPointToEnd(parent_.module_.getBody());
}

void FunctionGenerator::VisitFunctionDef(ast::FunctionDef *func) {
    if (!func) {
        return;
    }
    parent_.RegisterJitDependency(func);
}

void FunctionGenerator::VisitReturn(ast::Return *ret) {
    if (!ret) {
        return;
    }

    auto return_loc = ctx_->GetMLIRLocation(builder_.getContext(), ret);

    auto emitPlaceholderReturn = [&]() -> bool {
        auto *block = builder_.getInsertionBlock();
        if (!block) {
            return false;
        }
        return llvm::isa<mlir::func::FuncOp>(block->getParentOp());
    };

    // Get the return value if present
    auto *value_expr = ret->GetValue();
    if (value_expr) {
        // Generate the return value expression
        mlir::Value return_value = GenerateExpr(value_expr);
        if (!return_value) {
            ctx_->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                value_expr->GetSourceRange().getBegin())
                << "Failed to generate return value expression";
            return;
        }

        // Check if the enclosing function has a return type annotation
        // Get the parent function operation
        auto *block = builder_.getInsertionBlock();
        if (block) {
            auto parent_op = block->getParentOp();
            if (auto func_op = llvm::dyn_cast<mlir::func::FuncOp>(parent_op)) {
                auto function_type = func_op.getFunctionType();
                auto num_results = function_type.getNumResults();
                if (num_results == 0) {
                    // Function has no return type but we're returning a value
                    ctx_->diagnostic_manager->Report(
                        basic::DiagnosticCode::kUnimplemented,
                        ret->GetSourceRange().getBegin())
                        << "Function '" + func_op.getName().str() +
                               "' has a return value but no return type "
                               "annotation. "
                               "Add '-> <type>' to the function signature.";
                    return;
                }

                auto prepare_return_value =
                    [&](mlir::Value value, unsigned result_index) {
                        auto expected_type =
                            func_op.getFunctionType().getResult(result_index);
                        auto cast_value = expr_generator_.CastTensorVector(
                            value, expected_type,
                            value_expr->GetSourceRange().getBegin());
                        if (cast_value &&
                            cast_value.getType() != expected_type) {
                            cast_value = CreateTypeConversion(
                                cast_value, cast_value.getType(), expected_type,
                                return_loc, builder_);
                        }
                        return cast_value && cast_value.getType() == expected_type
                                   ? cast_value
                                   : mlir::Value();
                    };

                if (num_results == 1) {
                    if (return_value.getDefiningOp<cf::MakeIntTupleOp>()) {
                        ctx_->diagnostic_manager->Report(
                            basic::DiagnosticCode::kTypeMismatch,
                            ret->GetSourceRange().getBegin())
                            << "Cannot return a tuple from a single-value "
                               "function";
                        return;
                    }
                    auto cast_value = prepare_return_value(return_value, 0);
                    if (!cast_value) {
                        ctx_->diagnostic_manager->Report(
                            basic::DiagnosticCode::kTypeMismatch,
                            ret->GetSourceRange().getBegin())
                            << "Failed to convert return value";
                        return;
                    }
                    cf::ReturnOp::create(builder_, return_loc, cast_value);
                    if (emitPlaceholderReturn()) {
                        func::ReturnOp::create(builder_, return_loc,
                                               cast_value);
                    }
                    return;
                }

                llvm::SmallVector<mlir::Value> tuple_values;
                if (!ExtractTupleElements(return_value, tuple_values)) {
                    ctx_->diagnostic_manager->Report(
                        basic::DiagnosticCode::kTypeMismatch,
                        ret->GetSourceRange().getBegin())
                        << "Tuple return requires a tuple value";
                    return;
                }

                if (tuple_values.size() != num_results) {
                    ctx_->diagnostic_manager->Report(
                        basic::DiagnosticCode::kTypeMismatch,
                        ret->GetSourceRange().getBegin())
                        << "Tuple return expects " +
                               std::to_string(num_results) +
                               " values but got " +
                               std::to_string(tuple_values.size());
                    return;
                }

                llvm::SmallVector<mlir::Value> cast_values;
                cast_values.reserve(tuple_values.size());
                for (size_t i = 0; i < tuple_values.size(); ++i) {
                    auto cast_value =
                        prepare_return_value(tuple_values[i], i);
                    if (!cast_value) {
                        ctx_->diagnostic_manager->Report(
                            basic::DiagnosticCode::kTypeMismatch,
                            ret->GetSourceRange().getBegin())
                            << "Failed to convert tuple return value";
                        return;
                    }
                    cast_values.push_back(cast_value);
                }

                cf::ReturnOp::create(builder_, return_loc, cast_values);
                if (emitPlaceholderReturn()) {
                    func::ReturnOp::create(builder_, return_loc, cast_values);
                }
                return;
            }
        }

        // Create return operation with the value
        cf::ReturnOp::create(builder_, return_loc, return_value);
        if (emitPlaceholderReturn()) {
            func::ReturnOp::create(builder_, return_loc, return_value);
        }
    } else {
        // Create return operation without value (void return)
        cf::ReturnOp::create(builder_, return_loc, mlir::ValueRange{});
        if (emitPlaceholderReturn()) {
            func::ReturnOp::create(builder_, return_loc);
        }
    }
}

bool AssignToMemref(mlir::OpBuilder &builder, mlir::Value memref,
                    llvm::ArrayRef<mlir::Value> indices, mlir::Value value,
                    mlir::Location loc) {
    // The memref element type can be a scalar or a vector type. And the value
    // assigned to it can be a scalar or a vector type as well. If the memref
    // element type is a scalar, it should match the element type of the value
    // to store; If the memref type is a vector, it should match the type of the
    // value to store.
    //
    // Case 1: memref <- scalar value:
    // Directly store the scalar value to memref[indices]. The memref can be
    // scalar-type or vector-type. If necessary, cast it to a compatible type
    // with memref's element type. If memref is vector type, we need to
    // broadcast the scalar value to vector first.
    //
    // Case 2: memref <- vector value:
    // Use vector::StoreOp to store the vector value to memref[indices]. The
    // memref can be scalar-type or vector-type. If necessary, use
    // vector::BitcastOp to cast the value to a compatible vector type with
    // vector-type memref's element type. When assigning a vector value to a
    // scalar-type memref, it may raise OOB writes which is allowed in
    // vector::StoreOp, but not all targets may support out-of-bounds vector
    // stores.
    // See https://mlir.llvm.org/docs/Dialects/Vector/#vectorstore-vectorstoreop
    // for more details.
    //
    // Other cases are not supported.
    auto memref_type = mlir::dyn_cast<cf::MemRefType>(memref.getType());
    if (!memref_type) {
        return false;
    }
    auto memref_element_type = memref_type.getElementType();
    auto value_type = value.getType();

    // Check if the value being stored is a vector type
    bool is_vector_value = mlir::isa<mlir::VectorType>(value_type);
    // Check if the memref's element type is a vector type
    bool is_vector_memref = mlir::isa<mlir::VectorType>(memref_element_type);

    if (is_vector_value) {
        // Case 2: memref <- vector value
        auto value_vector_type = mlir::cast<mlir::VectorType>(value_type);

        if (is_vector_memref) {
            // memref<vector> <- vector
            // Use vector::BitcastOp if element types don't match
            auto memref_vector_type =
                mlir::cast<mlir::VectorType>(memref_element_type);
            if (value_vector_type.getElementType() !=
                memref_vector_type.getElementType()) {
                // Need to bitcast to match the memref's vector element type
                value = mlir::vector::BitCastOp::create(
                    builder, loc, memref_vector_type, value);
            }
            if (value.getType() != memref_vector_type) {
                return false;
            }
            // Now store the vector
            cf::AveLangMemRefStoreOp::create(builder, loc, value, memref,
                                             indices);
        } else {
            // memref<scalar> <- vector
            // This may cause OOB writes but is allowed by vector::StoreOp
            cf::AveLangMemRefStoreOp::create(builder, loc, value, memref,
                                             indices);
        }
    } else {
        // Case 1: memref <- scalar value
        if (is_vector_memref) {
            // memref<vector> <- scalar
            // Broadcast the scalar to vector first
            auto memref_vector_type =
                mlir::cast<mlir::VectorType>(memref_element_type);

            // Cast scalar to match the vector's element type if needed
            if (value_type != memref_vector_type.getElementType()) {
                // Use arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, or
                // arith::BitcastOp depending on the types. For simplicity, use
                // a type conversion function
                bool allowDemotion =
                    CanImplicitlyDemoteConstantWithoutPrecisionLoss(
                        value, memref_vector_type.getElementType());
                auto converted_value = CreateTypeConversion(
                    value, value_type, memref_vector_type.getElementType(), loc,
                    builder, allowDemotion);
                if (!converted_value ||
                    converted_value.getType() !=
                        memref_vector_type.getElementType()) {
                    // Conversion failed - types are incompatible
                    return false;
                }
                value = converted_value;
            }

            // Create a vector by broadcasting the scalar
            mlir::Value vector_value = mlir::vector::BroadcastOp::create(
                builder, loc, memref_vector_type, value);

            // Store the vector
            cf::AveLangMemRefStoreOp::create(builder, loc, vector_value, memref,
                                             indices);
        } else {
            // memref<scalar> <- scalar
            // Cast scalar to match memref element type if needed
            if (value_type != memref_element_type) {
                bool allowDemotion =
                    CanImplicitlyDemoteConstantWithoutPrecisionLoss(
                        value, memref_element_type);
                auto converted_value =
                    CreateTypeConversion(value, value_type, memref_element_type,
                                         loc, builder, allowDemotion);
                if (!converted_value ||
                    converted_value.getType() != memref_element_type) {
                    // Conversion failed - types are incompatible
                    return false;
                }
                value = converted_value;
            }

            cf::AveLangMemRefStoreOp::create(builder, loc, value, memref,
                                             mlir::ValueRange(indices));
        }
    }

    return true;
}

mlir::Value FunctionGenerator::ResolveMemrefValue(ast::Expr *expr) {
    if (!expr) {
        return mlir::Value();
    }

    if (auto *subscript = llvm::dyn_cast<ast::Subscript>(expr)) {
        auto base_value = ResolveMemrefValue(subscript->GetValue());
        if (!base_value) {
            base_value = GenerateExpr(subscript->GetValue());
        }
        if (!base_value || !mlir::isa<cf::MemRefType>(base_value.getType())) {
            return mlir::Value();
        }

        auto memref_type = mlir::cast<cf::MemRefType>(base_value.getType());

        llvm::SmallVector<mlir::Value> indices;
        auto *slice_expr = subscript->GetSlice();
        auto source_loc = subscript->GetSourceRange().getBegin();

        if (auto *tuple_expr = llvm::dyn_cast<ast::Tuple>(slice_expr)) {
            for (auto *element : tuple_expr->GetElts()) {
                auto index = GenerateIndex(element, source_loc);
                if (!index) {
                    return mlir::Value();
                }
                indices.push_back(index);
            }
        } else {
            auto index = GenerateIndex(slice_expr, source_loc);
            if (!index) {
                return mlir::Value();
            }
            indices.push_back(index);
        }

        LayoutOperation::ExpandIndicesForNestedLayout(builder_, base_value,
                                                      indices);

        if (indices.size() > static_cast<size_t>(memref_type.getRank())) {
            return mlir::Value();
        }

        if (indices.size() < static_cast<size_t>(memref_type.getRank())) {
            auto subscript_loc = GetMLIRLocation(subscript);
            llvm::SmallVector<int64_t> sub_shape(
                memref_type.getShape().begin() + indices.size(),
                memref_type.getShape().end());
            llvm::SmallVector<int64_t> sub_strides;
            auto base_strides = memref_type.getStrides();
            if (base_strides.size() ==
                static_cast<size_t>(memref_type.getRank())) {
                sub_strides.assign(base_strides.begin() + indices.size(),
                                   base_strides.end());
            }
            auto sub_memref_type = createAveLangMemRefType(
                builder_.getContext(), sub_shape, memref_type.getElementType(),
                memref_type.getMemorySpace(), sub_strides);
            auto one = mlir::arith::ConstantIndexOp::create(builder_,
                                                            subscript_loc, 1);
            llvm::SmallVector<mlir::Value> sizes(indices.size(), one);
            llvm::SmallVector<mlir::Value> strides(indices.size(), one);
            return cf::AveLangMemRefSubViewOp::create(
                       builder_, subscript_loc, sub_memref_type, base_value,
                       indices, sizes, strides)
                .getResult();
        }

        return mlir::Value();
    }

    // Generate the expression to get its MLIR value
    auto value = GenerateExpr(expr);
    if (!value) {
        return mlir::Value();
    }

    // Return the value if it's already a memref, otherwise return null
    if (mlir::isa<cf::MemRefType>(value.getType())) {
        return value;
    }

    return mlir::Value();
}

mlir::Value FunctionGenerator::GenerateIndex(ast::Expr *expr,
                                             clang::SourceLocation loc) {
    auto index = GenerateExpr(expr);
    if (!index) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         loc)
            << "Failed to generate index for subscript assignment";
        return mlir::Value();
    }

    // Convert to index type if needed
    if (!index.getType().isIndex()) {
        auto index_loc = expr ? GetMLIRLocation(expr) : GetMLIRLocation(loc);
        index =
            CreateTypeConversion(index, index.getType(),
                                 builder_.getIndexType(), index_loc, builder_);
        if (!index) {
            ctx_->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented, loc)
                << "Failed to convert index to index type";
            return mlir::Value();
        }
    }

    return index;
}

void FunctionGenerator::VisitAssign(ast::Assign *assign) {
    if (!assign)
        return;

    // Generate the value expression (right-hand side)
    auto value = GenerateExpr(assign->GetValue());
    if (!value) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         assign->GetSourceRange().getBegin())
            << "Failed to generate expression for assignment";
        return;
    }

    // Handle assignment targets
    auto targets = assign->GetTargets();
    if (targets.size() != 1) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         assign->GetSourceRange().getBegin())
            << "Multiple assignment targets not supported";
        return;
    }

    auto *target = targets[0];
    auto source_loc = target->GetSourceRange().getBegin();
    if (!source_loc.isValid()) {
        source_loc = assign->GetSourceRange().getBegin();
    }

    auto handle_tuple_target = [&](auto *tuple_target) -> bool {
        llvm::SmallVector<mlir::Value> tuple_values;
        if (!ExtractTupleElements(value, tuple_values)) {
            ctx_->diagnostic_manager->Report(
                basic::DiagnosticCode::kTypeMismatch, source_loc)
                << "Tuple assignment requires a tuple value";
            return false;
        }
        if (tuple_values.size() != tuple_target->GetElts().size()) {
            ctx_->diagnostic_manager->Report(
                basic::DiagnosticCode::kTypeMismatch, source_loc)
                << "Tuple assignment expects " +
                       std::to_string(tuple_target->GetElts().size()) +
                       " values but got " + std::to_string(tuple_values.size());
            return false;
        }
        for (size_t i = 0; i < tuple_values.size(); ++i) {
            auto *elem_target = tuple_target->GetElts()[i];
            if (!AssignValueToTarget(
                    elem_target, tuple_values[i],
                    elem_target->GetSourceRange().getBegin())) {
                return false;
            }
        }
        return true;
    };

    if (auto *tuple_target = llvm::dyn_cast<ast::Tuple>(target)) {
        handle_tuple_target(tuple_target);
        return;
    }
    if (auto *list_target = llvm::dyn_cast<ast::List>(target)) {
        handle_tuple_target(list_target);
        return;
    }

    AssignValueToTarget(target, value, source_loc);
}

void FunctionGenerator::VisitAugAssign(ast::AugAssign *aug_assign) {
    if (!aug_assign) {
        return;
    }

    auto *target = aug_assign->GetTarget();
    auto source_loc = aug_assign->GetSourceRange().getBegin();
    if (!target) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         source_loc)
            << "Missing target for augmented assignment";
        return;
    }

    auto target_value = GenerateExpr(target);
    if (!target_value) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         source_loc)
            << "Failed to generate target for augmented assignment";
        return;
    }

    // Lower `target op= value` through the existing binary-op path, then
    // assign the computed result back to the original target.
    ast::BinOp binop(target, aug_assign->GetValue(), aug_assign->GetOp());
    binop.SetSourceRange(aug_assign->GetSourceRange());

    auto value = expr_generator_.VisitBinOp(&binop);
    if (!value) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         source_loc)
            << "Failed to generate expression for augmented assignment";
        return;
    }

    auto target_type = target_value.getType();
    auto value_type = value.getType();
    auto target_type_info = GetTypeInfo(target_value);
    if ((value_type != target_type ||
         GetTypeInfo(value).is_unsigned_integer !=
             target_type_info.is_unsigned_integer) &&
        (target_type.isIntOrIndexOrFloat() ||
         mlir::isa<mlir::VectorType>(target_type))) {
        auto converted = CreateTypeConversion(
            value, value_type, target_type, GetMLIRLocation(aug_assign),
            builder_, /*allow_demotion=*/false,
            GetTypeInfo(value).is_unsigned_integer,
            target_type_info.is_unsigned_integer);
        if (!converted) {
            ctx_->diagnostic_manager->Report(
                basic::DiagnosticCode::kTypeMismatch, source_loc)
                << "Implicit demotion in augmented assignment is not allowed; "
                   "use an explicit convert";
            return;
        }
        value = converted;
    }

    AssignValueToTarget(target, value, source_loc);
}

bool FunctionGenerator::AssignValueToTarget(ast::Expr *target,
                                            mlir::Value value,
                                            clang::SourceLocation source_loc) {
    if (!target) {
        return false;
    }

    mlir::Value memref;
    llvm::SmallVector<mlir::Value> indices;
    bool assignment_complete = false;

    if (auto *name_target = llvm::dyn_cast<ast::Name>(target)) {
        if (!ResolveNameAssignmentTarget(name_target, value, source_loc, memref,
                                         indices, assignment_complete)) {
            return false;
        }
    } else if (auto *subscript_target =
                   llvm::dyn_cast<ast::Subscript>(target)) {
        if (!ResolveSubscriptAssignmentTarget(subscript_target, value,
                                              source_loc, memref, indices,
                                              assignment_complete)) {
            return false;
        }
    } else {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         source_loc)
            << "Unsupported assignment target";
        return false;
    }

    if (assignment_complete) {
        return true;
    }

    if (!MaterializeAssignableValue(value, source_loc,
                                    "Cannot assign a memref into a memref")) {
        return false;
    }

    MaybeCreateSubviewForVectorAssignment(memref, indices, value);

    if (!ValidateMemrefAssignmentTarget(memref, indices, source_loc)) {
        return false;
    }

    if (!AssignToMemref(builder_, memref, indices, value,
                        ctx_->GetMLIRLocation(builder_.getContext(), target))) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kTypeMismatch,
                                         source_loc)
            << "Assignment would require implicit demotion; use an explicit "
               "convert";
        return false;
    }
    return true;
}

bool FunctionGenerator::ResolveNameAssignmentTarget(
    ast::Name *name_target, mlir::Value value, clang::SourceLocation source_loc,
    mlir::Value &memref, llvm::SmallVector<mlir::Value> &indices,
    bool &assignment_complete) {
    assignment_complete = false;

    const std::string &target_name = name_target->GetId();
    auto existing_symbol = ctx_->syms->LookupSymbol(target_name);
    if (existing_symbol &&
        !existing_symbol->isa(SymbolTable::SymbolKind::kValue)) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kTypeMismatch,
                                         source_loc)
            << "Symbol has wrong type";
        return false;
    }
    if (existing_symbol && existing_symbol->immutable) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         source_loc)
            << "Cannot assign to immutable (constexpr) variable '" +
                   target_name + "'";
        return false;
    }

    mlir::Value existing_value =
        existing_symbol ? existing_symbol->value : mlir::Value();
    if (existing_value && mlir::isa<cf::MemRefType>(existing_value.getType())) {
        auto memref_type = mlir::cast<cf::MemRefType>(existing_value.getType());
        memref = existing_value;
        AppendZeroIndices(GetMLIRLocation(name_target), memref_type.getRank(),
                          indices);
        return true;
    }

    auto value_type = value.getType();
    if (mlir::isa<cf::MemRefType>(value_type) ||
        mlir::isa<mlir::VectorType>(value_type) ||
        mlir::isa<mlir::OpaqueType>(value_type) ||
        value.getDefiningOp<cf::MakeIntTupleOp>()) {
        ctx_->syms->DefineSymbol(target_name, value);
        assignment_complete = true;
        return true;
    }

    auto address_space = mlir::gpu::AddressSpaceAttr::get(
        builder_.getContext(), mlir::gpu::AddressSpace::Private);
    auto memref_type = createAveLangMemRefType(builder_.getContext(), {1},
                                               value_type, address_space);

    auto alloc_op = cf::AveLangMemRefAllocaOp::create(
        builder_, ctx_->GetMLIRLocation(builder_.getContext(), name_target),
        memref_type, mlir::ValueRange{});
    SetTypeInfo(alloc_op.getResult(), GetTypeInfo(value));
    memref = alloc_op;
    AppendZeroIndices(GetMLIRLocation(name_target), memref_type.getRank(),
                      indices);
    ctx_->syms->DefineSymbol(target_name, alloc_op);
    return true;
}

bool FunctionGenerator::ResolveSubscriptAssignmentTarget(
    ast::Subscript *subscript_target, mlir::Value &value,
    clang::SourceLocation source_loc, mlir::Value &memref,
    llvm::SmallVector<mlir::Value> &indices, bool &assignment_complete) {
    assignment_complete = false;

    auto target_value = ResolveMemrefValue(subscript_target->GetValue());
    if (!target_value) {
        target_value = ctx_->syms->ResolveRefExpr(subscript_target->GetValue());
    }
    if (!mlir::isa<cf::MemRefType>(target_value.getType()) &&
        !mlir::isa<mlir::VectorType>(target_value.getType())) {
        target_value = GenerateExpr(subscript_target->GetValue());
    }
    if (!target_value) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         source_loc)
            << "Failed to resolve target for subscript assignment";
        return false;
    }

    mlir::Value vector_target_value;
    if (mlir::isa<mlir::VectorType>(target_value.getType())) {
        vector_target_value = target_value;
    } else if (mlir::isa<cf::MemRefType>(target_value.getType())) {
        memref = target_value;
    } else {
        memref = ResolveMemrefValue(subscript_target->GetValue());
        if (!memref) {
            ctx_->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented, source_loc)
                << "Failed to resolve memref for subscript assignment";
            return false;
        }
    }

    if (!BuildSubscriptIndices(subscript_target->GetSlice(), source_loc,
                               indices)) {
        return false;
    }

    if (memref) {
        LayoutOperation::ExpandIndicesForNestedLayout(builder_, memref,
                                                      indices);
    }

    if (!vector_target_value) {
        return true;
    }

    if (!MaterializeAssignableValue(value, source_loc,
                                    "Cannot assign a memref into a vector")) {
        return false;
    }

    auto *base_name = llvm::dyn_cast<ast::Name>(subscript_target->GetValue());
    if (!base_name) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         source_loc)
            << "Subscript assignment to vector requires a named variable";
        return false;
    }

    llvm::SmallVector<mlir::OpFoldResult> position_indices;
    position_indices.reserve(indices.size());
    for (auto idx : indices) {
        position_indices.push_back(idx);
    }

    auto inserted = mlir::vector::InsertOp::create(
        builder_, GetMLIRLocation(subscript_target), value, vector_target_value,
        position_indices);
    auto targetInfo = GetTypeInfo(vector_target_value);
    if (targetInfo.is_unsigned_integer) {
        SetTypeInfo(inserted.getResult(), targetInfo);
    } else {
        SetTypeInfo(inserted.getResult(), GetTypeInfo(value));
    }
    ctx_->syms->DefineSymbol(base_name->GetId(), inserted.getResult());
    assignment_complete = true;
    return true;
}

bool FunctionGenerator::BuildSubscriptIndices(
    ast::Expr *slice_expr, clang::SourceLocation source_loc,
    llvm::SmallVector<mlir::Value> &indices) {
    if (auto *tuple_expr = llvm::dyn_cast<ast::Tuple>(slice_expr)) {
        for (auto *element : tuple_expr->GetElts()) {
            auto index = GenerateIndex(element, source_loc);
            if (!index) {
                return false;
            }
            indices.push_back(index);
        }
        return true;
    }

    auto index = GenerateIndex(slice_expr, source_loc);
    if (!index) {
        return false;
    }
    indices.push_back(index);
    return true;
}

bool FunctionGenerator::MaterializeAssignableValue(
    mlir::Value &value, clang::SourceLocation source_loc,
    llvm::StringRef memref_assignment_error) {
    if (!mlir::isa<cf::MemRefType>(value.getType())) {
        return true;
    }

    auto value_memref_type = mlir::cast<cf::MemRefType>(value.getType());
    auto elem_type = value_memref_type.getElementType();

    bool has_dynamic = false;
    int64_t element_count = 1;
    int non_unit_dims = 0;
    for (auto dim : value_memref_type.getShape()) {
        if (dim == mlir::ShapedType::kDynamic) {
            has_dynamic = true;
            break;
        }
        element_count *= dim;
        if (dim != 1) {
            ++non_unit_dims;
        }
    }

    if (!has_dynamic) {
        llvm::SmallVector<mlir::Value> load_indices;
        load_indices.reserve(value_memref_type.getRank());
        auto value_loc = GetMLIRLocation(source_loc);
        AppendZeroIndices(value_loc, value_memref_type.getRank(), load_indices);

        if (element_count == 1 && !mlir::isa<mlir::VectorType>(elem_type)) {
            value = cf::AveLangMemRefLoadOp::create(
                builder_, value_loc, elem_type, value, load_indices);
            SetTypeInfo(value,
                        GetTypeInfo(value.getDefiningOp()->getOperand(0)));
        } else if (auto elem_vec_type =
                       mlir::dyn_cast<mlir::VectorType>(elem_type)) {
            if (element_count == 1) {
                value = cf::AveLangMemRefLoadOp::create(
                    builder_, value_loc, elem_vec_type, value, load_indices);
                SetTypeInfo(value,
                            GetTypeInfo(value.getDefiningOp()->getOperand(0)));
            }
        } else if (non_unit_dims <= 1) {
            auto load_vec_type =
                mlir::VectorType::get(value_memref_type.getShape(), elem_type);
            auto loaded = cf::AveLangMemRefLoadVecOp::create(
                builder_, value_loc, load_vec_type, value, load_indices);
            if (load_vec_type.getRank() == 1) {
                value = loaded.getResult();
                SetTypeInfo(value, GetTypeInfo(loaded.getMemref()));
            } else {
                auto flat_type =
                    mlir::VectorType::get(element_count, elem_type);
                value = mlir::vector::ShapeCastOp::create(
                    builder_, value_loc, flat_type, loaded.getResult());
                SetTypeInfo(value, GetTypeInfo(loaded.getResult()));
            }
        }
    }

    if (mlir::isa<cf::MemRefType>(value.getType())) {
        ctx_->diagnostic_manager->Report(
            basic::DiagnosticCode::kIndexDimensionMismatch, source_loc)
            << memref_assignment_error;
        return false;
    }

    return true;
}

void FunctionGenerator::MaybeCreateSubviewForVectorAssignment(
    mlir::Value &memref, llvm::SmallVector<mlir::Value> &indices,
    mlir::Value &value) {
    auto memref_type = mlir::dyn_cast<cf::MemRefType>(memref.getType());
    if (!memref_type ||
        indices.size() >= static_cast<size_t>(memref_type.getRank()) ||
        !mlir::isa<mlir::VectorType>(value.getType())) {
        return;
    }

    llvm::SmallVector<int64_t> sub_shape(memref_type.getShape().begin() +
                                             indices.size(),
                                         memref_type.getShape().end());
    llvm::SmallVector<int64_t> sub_strides;
    auto base_strides = memref_type.getStrides();
    if (base_strides.size() == static_cast<size_t>(memref_type.getRank())) {
        sub_strides.assign(base_strides.begin() + indices.size(),
                           base_strides.end());
    }

    bool has_dynamic = false;
    int non_unit_dims = 0;
    for (auto dim : sub_shape) {
        if (dim == mlir::ShapedType::kDynamic) {
            has_dynamic = true;
            break;
        }
        if (dim != 1) {
            ++non_unit_dims;
        }
    }

    if (has_dynamic || non_unit_dims > 1) {
        return;
    }

    auto sub_memref_type = createAveLangMemRefType(
        builder_.getContext(), sub_shape, memref_type.getElementType(),
        memref_type.getMemorySpace(), sub_strides);
    auto loc = value.getLoc();
    auto one = mlir::arith::ConstantIndexOp::create(builder_, loc, 1);
    llvm::SmallVector<mlir::Value> sizes(indices.size(), one);
    llvm::SmallVector<mlir::Value> strides(indices.size(), one);

    memref = cf::AveLangMemRefSubViewOp::create(builder_, loc, sub_memref_type,
                                                memref, indices, sizes, strides)
                 .getResult();
    SetTypeInfo(memref, GetTypeInfo(memref.getDefiningOp()->getOperand(0)));
    memref_type = mlir::dyn_cast<cf::MemRefType>(memref.getType());

    if (sub_shape.size() != 1) {
        auto value_vec_type = mlir::cast<mlir::VectorType>(value.getType());
        auto target_vec_type =
            mlir::VectorType::get(sub_shape, value_vec_type.getElementType());
        value = mlir::vector::ShapeCastOp::create(builder_, loc,
                                                  target_vec_type, value);
    }

    indices.clear();
    indices.reserve(memref_type.getRank());
    AppendZeroIndices(loc, memref_type.getRank(), indices);
}

bool FunctionGenerator::ValidateMemrefAssignmentTarget(
    mlir::Value memref, llvm::ArrayRef<mlir::Value> indices,
    clang::SourceLocation source_loc) {
    auto memref_type = mlir::dyn_cast<cf::MemRefType>(memref.getType());
    if (!memref_type ||
        indices.size() != static_cast<size_t>(memref_type.getRank())) {
        ctx_->diagnostic_manager->Report(
            basic::DiagnosticCode::kIndexDimensionMismatch, source_loc)
            << "Index dimension mismatch in assignment's target";
        return false;
    }
    return true;
}

void FunctionGenerator::AppendZeroIndices(
    mlir::Location loc, int64_t rank,
    llvm::SmallVectorImpl<mlir::Value> &indices) {
    for (int64_t i = 0; i < rank; ++i) {
        indices.push_back(
            mlir::arith::ConstantIndexOp::create(builder_, loc, 0));
    }
}

void FunctionGenerator::VisitIf(ast::If *if_stmt) {
    if (!if_stmt)
        return;

    auto condition = GenerateExpr(if_stmt->GetTest());
    if (!condition) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         if_stmt->GetSourceRange().getBegin())
            << "Failed to generate condition expression for if statement";
        return;
    }

    if (!condition.getType().isInteger(1)) {
        auto condition_loc = GetMLIRLocation(if_stmt->GetTest());
        if (condition.getType().isInteger()) {
            auto zeroAttr = builder_.getIntegerAttr(condition.getType(), 0);
            auto zero = mlir::arith::ConstantOp::create(builder_, condition_loc,
                                                        zeroAttr);
            condition = mlir::arith::CmpIOp::create(
                builder_, condition_loc, mlir::arith::CmpIPredicate::ne,
                condition, zero);
        } else {
            ctx_->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                if_stmt->GetSourceRange().getBegin())
                << "Unsupported condition type for if statement";
            return;
        }
    }

    // Create scf.if operation
    bool has_else = !if_stmt->GetOrelse().empty();
    auto if_op =
        mlir::scf::IfOp::create(builder_, GetMLIRLocation(if_stmt), condition,
                                /*withElseRegion=*/has_else);

    // Generate the then block with isolated scope
    {
        SymbolTable::FrameGuard guard(ctx_->syms.get());
        auto &then_region = if_op.getThenRegion();
        SS_ASSERT(then_region.hasOneBlock());
        auto &then_block = then_region.front();
        builder_.setInsertionPointToStart(&then_block);

        for (auto *stmt : if_stmt->GetBody()) {
            DispatchStmt(stmt);
        }
    }

    // Generate the else block if present with isolated scope
    if (has_else) {
        SymbolTable::FrameGuard guard(ctx_->syms.get());
        auto &else_region = if_op.getElseRegion();
        SS_ASSERT(else_region.hasOneBlock());
        auto &else_block = else_region.front();
        builder_.setInsertionPointToStart(&else_block);

        for (auto *stmt : if_stmt->GetOrelse()) {
            DispatchStmt(stmt);
        }
    }

    // Restore insertion point to after the if operation
    builder_.setInsertionPointAfter(if_op);
}

void FunctionGenerator::VisitFor(ast::For *for_stmt) {
    if (!for_stmt)
        return;

    // Generate the iterator expression using the normal expression resolution
    auto iter = GenerateExpr(for_stmt->GetIter());
    if (!iter) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         for_stmt->GetSourceRange().getBegin())
            << "Failed to generate iterator expression for for loop";
        return;
    }

    // Check if this is a range object created by avelang.range()
    auto iter_op = iter.getDefiningOp();
    if (!iter_op) {
        ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                         for_stmt->GetSourceRange().getBegin())
            << "Iterator must be a range() call";
        return;
    }

    // Check if this is our special range marker operation
    if (auto cast_op =
            llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(iter_op)) {
        if (cast_op->hasAttr("avelang_range")) {
            // Extract the pre-computed range values from the operation
            auto operands = cast_op->getOperands();
            if (operands.size() != 3) {
                ctx_->diagnostic_manager->Report(
                    basic::DiagnosticCode::kUnimplemented,
                    for_stmt->GetSourceRange().getBegin())
                    << "Invalid range object: expected 3 operands "
                       "(lower_bound, upper_bound, step)";
                return;
            }

            // Range object always contains: [lower_bound, upper_bound, step]
            // All values are already converted to index type in
            // CreateRangeFunction
            mlir::Value lower_bound = operands[0];
            mlir::Value upper_bound = operands[1];
            mlir::Value step = operands[2];

            // Create scf.for operation
            auto for_op =
                mlir::scf::ForOp::create(builder_, GetMLIRLocation(for_stmt),
                                         lower_bound, upper_bound, step);

            // Generate the loop body with isolated scope
            {
                SymbolTable::FrameGuard guard(ctx_->syms.get());
                auto &loop_region = for_op.getRegion();
                assert(loop_region.hasOneBlock());
                auto &loop_block = loop_region.front();
                builder_.setInsertionPointToStart(&loop_block);

                // Bind the loop variable (induction variable) to the target
                auto *target = for_stmt->GetTarget();
                if (auto *name_target = llvm::dyn_cast<ast::Name>(target)) {
                    ctx_->syms->DefineSymbol(name_target->GetId(),
                                             for_op.getInductionVar());
                } else {
                    ctx_->diagnostic_manager->Report(
                        basic::DiagnosticCode::kUnimplemented,
                        for_stmt->GetSourceRange().getBegin())
                        << "Complex loop targets not supported";
                    return;
                }

                // Generate loop body statements
                for (auto *stmt : for_stmt->GetBody()) {
                    DispatchStmt(stmt);
                }
            }

            // Restore insertion point to after the for operation
            builder_.setInsertionPointAfter(for_op);
            // Remove the phantom unrealized_conversion_cast operation
            cast_op->erase();
            return;
        }
    }

    // If we reach here, the iterator is not a supported range() call
    ctx_->diagnostic_manager->Report(basic::DiagnosticCode::kUnimplemented,
                                     for_stmt->GetSourceRange().getBegin())
        << "Only range() iterators are supported in for loops";
}

void FunctionGenerator::VisitWhile(ast::While *while_stmt) {
    if (!while_stmt)
        return;

    // Generate the condition expression
    auto condition = GenerateExpr(while_stmt->GetTest());
    if (!condition) {
        ctx_->diagnostic_manager->Report(
            basic::DiagnosticCode::kUnimplemented,
            while_stmt->GetSourceRange().getBegin())
            << "Failed to generate condition expression for while loop";
        return;
    }

    // Convert condition to i1 if needed
    if (!condition.getType().isInteger(1)) {
        auto condition_loc = GetMLIRLocation(while_stmt->GetTest());
        // Convert to boolean - non-zero values are true
        if (condition.getType().isInteger()) {
            auto zeroAttr = builder_.getIntegerAttr(condition.getType(), 0);
            auto zero = mlir::arith::ConstantOp::create(builder_, condition_loc,
                                                        zeroAttr);
            condition = mlir::arith::CmpIOp::create(
                builder_, condition_loc, mlir::arith::CmpIPredicate::ne,
                condition, zero);
        } else {
            ctx_->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                while_stmt->GetSourceRange().getBegin())
                << "Unsupported condition type for while loop";
            return;
        }
    }

    // Create scf.while operation
    auto while_op =
        mlir::scf::WhileOp::create(builder_, GetMLIRLocation(while_stmt),
                                   mlir::TypeRange{}, mlir::ValueRange{});

    // Generate the "before" region (condition check)
    {
        auto &before_region = while_op.getBefore();
        auto &before_block = before_region.emplaceBlock();
        builder_.setInsertionPointToStart(&before_block);

        // Re-evaluate the condition in the before block
        // Note: We need to regenerate the condition since it might depend on
        // variables that change in the loop body
        auto loop_condition = GenerateExpr(while_stmt->GetTest());
        if (!loop_condition) {
            ctx_->diagnostic_manager->Report(
                basic::DiagnosticCode::kUnimplemented,
                while_stmt->GetSourceRange().getBegin())
                << "Failed to regenerate condition in while loop";
            return;
        }

        // Convert condition to i1 if needed
        if (!loop_condition.getType().isInteger(1)) {
            auto loop_condition_loc = GetMLIRLocation(while_stmt->GetTest());
            if (loop_condition.getType().isInteger()) {
                auto zeroAttr =
                    builder_.getIntegerAttr(loop_condition.getType(), 0);
                auto zero = mlir::arith::ConstantOp::create(
                    builder_, loop_condition_loc, zeroAttr);
                loop_condition = mlir::arith::CmpIOp::create(
                    builder_, loop_condition_loc,
                    mlir::arith::CmpIPredicate::ne, loop_condition, zero);
            }
        }

        // Create condition operation that yields the condition value
        mlir::scf::ConditionOp::create(builder_,
                                       GetMLIRLocation(while_stmt->GetTest()),
                                       loop_condition, mlir::ValueRange{});
    }

    // Generate the "after" region (loop body)
    {
        SymbolTable::FrameGuard guard(ctx_->syms.get());
        auto &after_region = while_op.getAfter();
        auto &after_block = after_region.emplaceBlock();
        builder_.setInsertionPointToStart(&after_block);

        // Generate loop body statements
        for (auto *stmt : while_stmt->GetBody()) {
            DispatchStmt(stmt);
        }

        // Create yield operation to continue the loop
        mlir::scf::YieldOp::create(builder_, GetMLIRLocation(while_stmt),
                                   mlir::ValueRange{});
    }

    // Restore insertion point to after the while operation
    builder_.setInsertionPointAfter(while_op);
}

void FunctionGenerator::VisitExprStmt(ast::ExprStmt *expr_stmt) {
    if (!expr_stmt)
        return;

    // Generate the expression (which may have side effects)
    GenerateExpr(expr_stmt->GetValue());
    // The result value is discarded for expression statements
    // but the expression may have side effects (like function calls)
}

void FunctionGenerator::VisitImport(ast::Import *import_stmt) {
    parent_.HandleImport(import_stmt);
}

void FunctionGenerator::VisitImportFrom(ast::ImportFrom *import_from_stmt) {
    parent_.HandleImportFrom(import_from_stmt);
}

} // namespace causalflow::avelang::ir
