import ast
import json
import types

import _avelang_bindings as _C


class _LocalNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names = set()

    def _add_target(self, target):
        if isinstance(target, ast.Name):
            self.names.add(target.id)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._add_target(elt)
            return
        if isinstance(target, ast.Starred):
            self._add_target(target.value)

    def visit_FunctionDef(self, node):
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node):
        self.names.add(node.name)

    def visit_ClassDef(self, node):
        self.names.add(node.name)

    def visit_Assign(self, node):
        for target in node.targets:
            self._add_target(target)
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node):
        self._add_target(node.target)
        if node.value is not None:
            self.generic_visit(node.value)

    def visit_AugAssign(self, node):
        self._add_target(node.target)
        self.generic_visit(node.value)

    def visit_For(self, node):
        self._add_target(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self._add_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars is not None:
                self._add_target(item.optional_vars)
        self.generic_visit(node)

    def visit_AsyncWith(self, node):
        for item in node.items:
            if item.optional_vars is not None:
                self._add_target(item.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.names.add(alias.asname or alias.name)

    def visit_NamedExpr(self, node):
        self._add_target(node.target)
        self.generic_visit(node.value)


def _serialize_constexprs(src) -> str:
    constants = getattr(src, "constants", None) or {}
    global_constants = getattr(src, "global_constants", None) or {}
    if not constants and not global_constants:
        return "[]"
    constexprs_list = []
    for k, info in constants.items():
        param_name = src.fn.arg_names[k[0]]
        constexprs_list.append(
            {
                "name": param_name,
                "type": info["type"],
                "value": info["value"],
            }
        )
    for name, info in global_constants.items():
        constexprs_list.append(
            {
                "name": name,
                "type": info["type"],
                "value": info["value"],
            }
        )
    return json.dumps(constexprs_list)


def _serialize_global_constexprs(global_constants) -> str:
    if not global_constants:
        return "[]"
    constexprs_list = []
    for name, info in global_constants.items():
        constexprs_list.append(
            {
                "name": name,
                "type": info["type"],
                "value": info["value"],
            }
        )
    return json.dumps(constexprs_list)


def _get_function_def(py_module: ast.AST) -> ast.FunctionDef:
    for node in getattr(py_module, "body", []):
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError("FunctionDef not found in parsed AST")


class DependenciesFinder(ast.NodeVisitor):
    def __init__(self, jit_callable):
        self.jit_callable = jit_callable
        self.capture = jit_callable.get_capture_scope()
        self.dependencies = []
        self._locals_stack = []

    def _record_dependency(self, target):
        from avelang.runtime.jit import JITCallable

        if isinstance(target, JITCallable) and target is not self.jit_callable:
            self.dependencies.append(target)

    def _collect_local_names(self, node):
        names = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs:
                names.add(arg.arg)
            if args.vararg is not None:
                names.add(args.vararg.arg)
            if args.kwarg is not None:
                names.add(args.kwarg.arg)

        collector = _LocalNameCollector()
        if isinstance(node, ast.Lambda):
            collector.visit(node.body)
        else:
            for stmt in node.body:
                collector.visit(stmt)
        names.update(collector.names)
        return names

    def _push_local_names(self, node):
        self._locals_stack.append(self._collect_local_names(node))

    def _pop_local_names(self):
        self._locals_stack.pop()

    def _is_local_name(self, name):
        return any(name in scope for scope in self._locals_stack)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if self._is_local_name(node.func.id):
                self.generic_visit(node)
                return
            target = self.capture.get(node.func.id)
            if target is not None:
                self._record_dependency(target)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if self._is_local_name(node.func.value.id):
                    self.generic_visit(node)
                    return
                base = self.capture.get(node.func.value.id)
                if base is not None and hasattr(base, node.func.attr):
                    self._record_dependency(getattr(base, node.func.attr))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._push_local_names(node)
        self.generic_visit(node)
        self._pop_local_names()

    def visit_AsyncFunctionDef(self, node):
        self._push_local_names(node)
        self.generic_visit(node)
        self._pop_local_names()

    def visit_Lambda(self, node):
        self._push_local_names(node)
        self.generic_visit(node)
        self._pop_local_names()


def _collect_jit_dependencies(jit_fn) -> list:
    from avelang.runtime.jit import JITCallable

    seen = set()
    ordered = []

    def visit(current):
        finder = DependenciesFinder(current)
        finder.visit(current.parse())
        for dep in finder.dependencies:
            if dep in seen:
                continue
            seen.add(dep)
            visit(dep)
            ordered.append(dep)

    if not isinstance(jit_fn, JITCallable):
        return []

    visit(jit_fn)
    return ordered


def _has_constexpr_params(jit_fn) -> bool:
    return any(getattr(param, "is_constexpr", False) for param in getattr(jit_fn, "params", []))


def _build_import_module(jit_fns: list) -> ast.Module:
    import_nodes = []
    seen_imports = set()

    for fn in jit_fns:
        py_module = fn.parse()
        used_names = {node.id for node in ast.walk(py_module) if isinstance(node, ast.Name)}
        capture = fn.get_capture_scope()
        for name in sorted(used_names):
            value = capture.get(name)
            if isinstance(value, types.ModuleType):
                module_name = value.__name__
                key = (module_name, name if name != module_name else None)
                if key in seen_imports:
                    continue
                seen_imports.add(key)
                asname = None if name == module_name else name
                import_nodes.append(ast.Import(names=[ast.alias(name=module_name, asname=asname)]))

    module = ast.Module(body=import_nodes, type_ignores=[])
    return ast.fix_missing_locations(module)


def compile_to_binary(src, target, opt_level: int = 2, options=None):
    constexprs_json = _serialize_constexprs(src)

    jit_deps = _collect_jit_dependencies(src.fn)
    import_module = _build_import_module([src.fn, *jit_deps])

    generator = _C.MLIRGenerator()
    generator.generate_from_python_ast(import_module)

    for dep in jit_deps:
        generator.add_jit_dependency(dep.parse())

    for dep in jit_deps:
        dep_globals = {}
        collect_globals = getattr(dep, "_collect_global_constexprs", None)
        if callable(collect_globals):
            dep_globals = collect_globals()
        if _has_constexpr_params(dep):
            generator.inject_constexprs(_serialize_global_constexprs(dep_globals))
            continue
        dep_func = _get_function_def(dep.parse())
        generator.visit_function_def(dep_func, _serialize_global_constexprs(dep_globals), "jit")

    kernel_func = _get_function_def(src.fn.parse())
    generator.visit_function_def(kernel_func, constexprs_json, "kernel")

    num_warps = getattr(options, "num_warps", -1)
    return generator.compile_to_binary_bytes(target.tuple, target.chip, opt_level, num_warps)
