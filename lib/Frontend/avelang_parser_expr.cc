#include "avelang_parser.h"
#include "avelang_parser_utils.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include <clang/Basic/FileSystemOptions.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace causalflow::avelang::frontend {

using ASTNode = ast::ASTNode;

ast::AttributeExpr *
AveLangParser::ParseAttributeExpr(const py::object &py_attribute) {
    // Parse the value (left side of the attribute access)
    ast::Expr *value = nullptr;
    if (py::hasattr(py_attribute, "value")) {
        py::object value_obj = py_attribute.attr("value");
        value = ParseExpr(value_obj);
    }

    // Extract attribute name
    std::string attr_name;
    if (py::hasattr(py_attribute, "attr")) {
        attr_name = py_attribute.attr("attr").cast<std::string>();
    }

    // Create AttributeExpr with constructor
    ast::AttributeExpr *attribute_expr =
        context_->Allocate<ast::AttributeExpr>();
    new (attribute_expr) ast::AttributeExpr(value, std::move(attr_name));
    return attribute_expr;
}

ast::Arg *AveLangParser::ParseArg(const py::object &py_arg) {
    // Extract argument name
    std::string arg_name;
    if (py::hasattr(py_arg, "arg")) {
        arg_name = py_arg.attr("arg").cast<std::string>();
    }

    // Parse type annotation if present
    ast::Expr *annotation = nullptr;
    if (py::hasattr(py_arg, "annotation") &&
        !py_arg.attr("annotation").is_none()) {
        py::object annotation_obj = py_arg.attr("annotation");
        annotation = ParseExpr(annotation_obj);
    }

    // Create Arg with constructor
    ast::Arg *arg = context_->Allocate<ast::Arg>();
    new (arg) ast::Arg(std::move(arg_name), annotation);
    return arg;
}

ast::Name *AveLangParser::ParseName(const py::object &py_name) {
    // Extract identifier name
    std::string id;
    if (py::hasattr(py_name, "id")) {
        id = py_name.attr("id").cast<std::string>();
    }

    // Parse context if present (Load, Store, Del)
    ast::ExprContext *ctx = nullptr;
    if (py::hasattr(py_name, "ctx")) {
        py::object ctx_obj = py_name.attr("ctx");
        // For now, we skip parsing context as it's not implemented yet
        // ctx = ParseExprContext(ctx_obj);
    }

    // Create Name with constructor
    ast::Name *name = context_->Allocate<ast::Name>();
    new (name) ast::Name(std::move(id), ctx);
    return name;
}

ast::BinOp *AveLangParser::ParseBinOp(const py::object &py_binop) {
    // Parse left operand
    ast::Expr *left = nullptr;
    if (py::hasattr(py_binop, "left")) {
        left = ParseExpr(py_binop.attr("left"));
    }

    // Parse right operand
    ast::Expr *right = nullptr;
    if (py::hasattr(py_binop, "right")) {
        right = ParseExpr(py_binop.attr("right"));
    }

    // Extract operator
    std::string op = "Unknown";
    if (py::hasattr(py_binop, "op")) {
        py::object op_obj = py_binop.attr("op");
        op = op_obj.attr("__class__").attr("__name__").cast<std::string>();
    }

    ast::BinOp *binop = context_->Allocate<ast::BinOp>();
    new (binop) ast::BinOp(left, right, std::move(op));
    return binop;
}

ast::UnaryOp *AveLangParser::ParseUnaryOp(const py::object &py_unaryop) {
    // Parse operand
    ast::Expr *operand = nullptr;
    if (py::hasattr(py_unaryop, "operand")) {
        operand = ParseExpr(py_unaryop.attr("operand"));
    }

    // Extract operator
    std::string op = "Unknown";
    if (py::hasattr(py_unaryop, "op")) {
        py::object op_obj = py_unaryop.attr("op");
        op = op_obj.attr("__class__").attr("__name__").cast<std::string>();
    }

    ast::UnaryOp *unaryop = context_->Allocate<ast::UnaryOp>();
    new (unaryop) ast::UnaryOp(operand, std::move(op));
    return unaryop;
}

ast::BoolOp *AveLangParser::ParseBoolOp(const py::object &py_boolop) {
    // Parse values
    llvm::SmallVector<ast::Expr *, 4> values;
    for (auto value : MaybePyList(py_boolop, "values")) {
        ast::Expr *parsed_value = ParseExpr(value.cast<py::object>());
        if (parsed_value) {
            values.push_back(parsed_value);
        }
    }

    // Extract operator
    std::string op = "Unknown";
    if (py::hasattr(py_boolop, "op")) {
        py::object op_obj = py_boolop.attr("op");
        op = op_obj.attr("__class__").attr("__name__").cast<std::string>();
    }

    ast::BoolOp *boolop = context_->Allocate<ast::BoolOp>();
    new (boolop) ast::BoolOp(std::move(values), std::move(op));
    return boolop;
}

ast::Lambda *AveLangParser::ParseLambda(const py::object &py_lambda) {
    // Parse arguments
    ast::Arguments *args = nullptr;
    if (py::hasattr(py_lambda, "args")) {
        args = ParseArguments(py_lambda.attr("args"));
    }

    // Parse body
    ast::Expr *body = nullptr;
    if (py::hasattr(py_lambda, "body")) {
        body = ParseExpr(py_lambda.attr("body"));
    }

    ast::Lambda *lambda = context_->Allocate<ast::Lambda>();
    new (lambda) ast::Lambda(args, body);
    return lambda;
}

ast::IfExp *AveLangParser::ParseIfExp(const py::object &py_ifexp) {
    // Parse test condition
    ast::Expr *test = nullptr;
    if (py::hasattr(py_ifexp, "test")) {
        test = ParseExpr(py_ifexp.attr("test"));
    }

    // Parse body (if true)
    ast::Expr *body = nullptr;
    if (py::hasattr(py_ifexp, "body")) {
        body = ParseExpr(py_ifexp.attr("body"));
    }

    // Parse orelse (if false)
    ast::Expr *orelse = nullptr;
    if (py::hasattr(py_ifexp, "orelse")) {
        orelse = ParseExpr(py_ifexp.attr("orelse"));
    }

    ast::IfExp *ifexp = context_->Allocate<ast::IfExp>();
    new (ifexp) ast::IfExp(test, body, orelse);
    return ifexp;
}

ast::Dict *AveLangParser::ParseDict(const py::object &py_dict) {
    llvm::SmallVector<ast::Expr *, 4> keys;
    llvm::SmallVector<ast::Expr *, 4> values;

    // Parse keys
    for (auto key : MaybePyList(py_dict, "keys")) {
        ast::Expr *parsed_key = ParseExpr(key.cast<py::object>());
        if (parsed_key) {
            keys.push_back(parsed_key);
        }
    }

    // Parse values
    for (auto value : MaybePyList(py_dict, "values")) {
        ast::Expr *parsed_value = ParseExpr(value.cast<py::object>());
        if (parsed_value) {
            values.push_back(parsed_value);
        }
    }

    ast::Dict *dict = context_->Allocate<ast::Dict>();
    new (dict) ast::Dict(std::move(keys), std::move(values));
    return dict;
}

ast::Compare *AveLangParser::ParseCompare(const py::object &py_compare) {
    // Parse left operand
    ast::Expr *left = nullptr;
    if (py::hasattr(py_compare, "left")) {
        left = ParseExpr(py_compare.attr("left"));
    }

    // Parse operators
    llvm::SmallVector<std::string, 4> ops;
    for (auto op : MaybePyList(py_compare, "ops")) {
        py::object op_obj = op.cast<py::object>();
        std::string op_name =
            op_obj.attr("__class__").attr("__name__").cast<std::string>();
        ops.push_back(std::move(op_name));
    }

    // Parse comparators
    llvm::SmallVector<ast::Expr *, 4> comparators;
    for (auto comp : MaybePyList(py_compare, "comparators")) {
        ast::Expr *parsed_comp = ParseExpr(comp.cast<py::object>());
        if (parsed_comp) {
            comparators.push_back(parsed_comp);
        }
    }

    ast::Compare *compare = context_->Allocate<ast::Compare>();
    new (compare) ast::Compare(left, std::move(ops), std::move(comparators));
    return compare;
}

ast::Call *AveLangParser::ParseCall(const py::object &py_call) {
    // Parse function
    ast::Expr *func = nullptr;
    if (py::hasattr(py_call, "func")) {
        func = ParseExpr(py_call.attr("func"));
    }

    // Parse arguments
    llvm::SmallVector<ast::Expr *, 4> args;
    for (auto arg : MaybePyList(py_call, "args")) {
        ast::Expr *parsed_arg = ParseExpr(arg.cast<py::object>());
        if (parsed_arg) {
            args.push_back(parsed_arg);
        }
    }

    // Parse keywords after positional arguments and retain the keyword names.
    // Intrinsic builders can use the names to apply defaults/reordering.
    llvm::SmallVector<std::string, 4> keywords;
    for (auto keyword : MaybePyList(py_call, "keywords")) {
        py::object kw_obj = keyword.cast<py::object>();
        if (py::hasattr(kw_obj, "arg")) {
            std::string kw_name = kw_obj.attr("arg").cast<std::string>();
            if (py::hasattr(kw_obj, "value")) {
                ast::Expr *parsed_arg = ParseExpr(kw_obj.attr("value"));
                if (parsed_arg) {
                    args.push_back(parsed_arg);
                }
            }
            keywords.push_back(std::move(kw_name));
        }
    }

    ast::Call *call = context_->Allocate<ast::Call>();
    new (call) ast::Call(func, std::move(args), std::move(keywords));
    return call;
}

ast::Constant *AveLangParser::ParseConstant(const py::object &py_constant) {
    std::string value = "None";
    if (py::hasattr(py_constant, "value")) {
        py::object val_obj = py_constant.attr("value");
        try {
            value = py::str(val_obj).cast<std::string>();
        } catch (const py::cast_error &) {
            value = "UnparsableValue";
        }
    }

    ast::Constant *constant = context_->Allocate<ast::Constant>();
    new (constant) ast::Constant(std::move(value));
    return constant;
}

ast::Subscript *AveLangParser::ParseSubscript(const py::object &py_subscript) {
    // Parse value
    ast::Expr *value = nullptr;
    if (py::hasattr(py_subscript, "value")) {
        value = ParseExpr(py_subscript.attr("value"));
    }

    // Parse slice
    ast::Expr *slice = nullptr;
    if (py::hasattr(py_subscript, "slice")) {
        slice = ParseExpr(py_subscript.attr("slice"));
    }

    // Parse context (simplified)
    ast::ExprContext *ctx = nullptr;

    ast::Subscript *subscript = context_->Allocate<ast::Subscript>();
    new (subscript) ast::Subscript(value, slice, ctx);
    return subscript;
}

ast::List *AveLangParser::ParseList(const py::object &py_list) {
    llvm::SmallVector<ast::Expr *, 4> elts;

    // Parse elements
    for (auto elt : MaybePyList(py_list, "elts")) {
        ast::Expr *parsed_elt = ParseExpr(elt.cast<py::object>());
        if (parsed_elt) {
            elts.push_back(parsed_elt);
        }
    }

    // Parse context (simplified)
    ast::ExprContext *ctx = nullptr;

    ast::List *list = context_->Allocate<ast::List>();
    new (list) ast::List(std::move(elts), ctx);
    return list;
}

ast::Tuple *AveLangParser::ParseTuple(const py::object &py_tuple) {
    llvm::SmallVector<ast::Expr *, 4> elts;

    // Parse elements
    for (auto elt : MaybePyList(py_tuple, "elts")) {
        ast::Expr *parsed_elt = ParseExpr(elt.cast<py::object>());
        if (parsed_elt) {
            elts.push_back(parsed_elt);
        }
    }

    // Parse context (simplified)
    ast::ExprContext *ctx = nullptr;

    ast::Tuple *tuple = context_->Allocate<ast::Tuple>();
    new (tuple) ast::Tuple(std::move(elts), ctx);
    return tuple;
}

ast::Slice *AveLangParser::ParseSlice(const py::object &py_slice) {
    // Parse lower bound (optional)
    ast::Expr *lower = nullptr;
    if (py::hasattr(py_slice, "lower") && !py_slice.attr("lower").is_none()) {
        lower = ParseExpr(py_slice.attr("lower"));
    }

    // Parse upper bound (optional)
    ast::Expr *upper = nullptr;
    if (py::hasattr(py_slice, "upper") && !py_slice.attr("upper").is_none()) {
        upper = ParseExpr(py_slice.attr("upper"));
    }

    // Parse step (optional)
    ast::Expr *step = nullptr;
    if (py::hasattr(py_slice, "step") && !py_slice.attr("step").is_none()) {
        step = ParseExpr(py_slice.attr("step"));
    }

    ast::Slice *slice = context_->Allocate<ast::Slice>();
    new (slice) ast::Slice(lower, upper, step);
    return slice;
}

ast::Expr *AveLangParser::ParseExpr(const py::object &py_expr) {
    if (!py_expr || py_expr.is_none()) {
        return nullptr;
    }

    clang::SourceRange range = ExtractSourceRange(py_expr);

    // Use RAII guard to automatically manage source range stack
    ParsingContextGuard context_guard(this, range);

    // Expression dispatch map
    static const std::unordered_map<
        std::string,
        std::function<ast::Expr *(AveLangParser *, const py::object &)>>
        kExprParser = {
            {"Attribute", &AveLangParser::ParseAttributeExpr},
            {"Name", &AveLangParser::ParseName},
            {"BinOp", &AveLangParser::ParseBinOp},
            {"UnaryOp", &AveLangParser::ParseUnaryOp},
            {"BoolOp", &AveLangParser::ParseBoolOp},
            {"Lambda", &AveLangParser::ParseLambda},
            {"IfExp", &AveLangParser::ParseIfExp},
            {"Dict", &AveLangParser::ParseDict},
            {"Compare", &AveLangParser::ParseCompare},
            {"Call", &AveLangParser::ParseCall},
            {"Constant", &AveLangParser::ParseConstant},
            {"Subscript", &AveLangParser::ParseSubscript},
            {"List", &AveLangParser::ParseList},
            {"Tuple", &AveLangParser::ParseTuple},
            {"Slice", &AveLangParser::ParseSlice},
        };

    // Get the class name of the Python AST node
    py::object py_class = py_expr.attr("__class__");
    std::string class_name = py_class.attr("__name__").cast<std::string>();

    // Dispatch to appropriate parsing function based on class name
    auto it = kExprParser.find(class_name);
    ast::Expr *expr = nullptr;
    if (it != kExprParser.end()) {
        expr = it->second(this, py_expr);
    } else {
        Report(basic::DiagnosticCode::kUnknownASTNode) << class_name;
    }

    if (expr) {
        expr->SetSourceRange(range);
    }

    return expr;
}

} // namespace causalflow::avelang::frontend
