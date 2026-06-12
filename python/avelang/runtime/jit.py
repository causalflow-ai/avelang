# """
# ave-lang JIT compiler implementation.
# Provides @avelang.kernel decorator for JIT compilation of GPU kernels.
# """

import inspect
import ast
import textwrap
import re
import hashlib
import copy
import itertools
import threading
from collections import defaultdict
from functools import cached_property
from types import ModuleType
from typing import Any, Dict, Callable, Tuple, Generic, TypeVar

from .driver import driver
from .._utils import get_iterable_path, find_paths_if, type_canonicalisation_dict, is_namedtuple

AVE_LANG_MODULE = "avelang.language"

T = TypeVar("T")

# -----------------------------------------------------------------------------
# Dependencies Finder
# -----------------------------------------------------------------------------


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


class DependenciesFinder(ast.NodeVisitor):
    """
    This AST visitor is used to find dependencies of a JITFunction. This can
    be used to invalidate a JITFunction's hash when its source code -- or
    that of its dependencies -- changes.

    This visitor also keeps track of the global variables touched by the
    JITFunction.  When we launch the kernel, we check that these have the same
    values as they did when we ran this visitor.  If not, we raise an error (or
    otherwise we could recompile).
    """

    def __init__(self, name, globals, nonlocals, src) -> None:
        super().__init__()
        self.name = name
        self.hasher = hashlib.sha256(src.encode("utf-8"))

        # This function's __globals__ dict.
        self.globals = globals
        self.nonlocals = nonlocals

        # Python builtins that can be accessed from ave-lang kernels.
        self.supported_python_builtins = {
            "float",
            "getattr",
            "int",
            "isinstance",
            "len",
            "list",
            "max",
            "min",
            "print",
            "range",
        }
        self.supported_modules = {
            AVE_LANG_MODULE,
            "copy",
            "math",
        }

        # used_global_vals tells us which global variables are used by this
        # function and all those it transitively calls, plus the values of those
        # variables when each function was initially run.  (That is, if A calls
        # C, and B calls C, then the values for C in used_global_vals will be
        # from the first time C was run, either by A or B.)
        #
        # Each function may have a different __globals__ dict, so the global
        # variable `foo` may actually have a different value in the different
        # functions.  Thus this map is actually
        #  (var_name, id(__globals__)) -> (var_value, __globals__).
        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}

        self.visiting_arg_default_value = False
        self._locals_stack = []

    @property
    def ret(self):
        return self.hasher.hexdigest()

    @staticmethod
    def _values_equal(v1, v2):
        try:
            eq = v1 == v2
        except Exception:
            return False
        if eq is NotImplemented:
            return False
        if isinstance(eq, bool):
            return eq
        try:
            return bool(eq)
        except Exception:
            pass
        try:
            return bool(eq.all())
        except Exception:
            return False

    def _update_hash(self, func):
        assert isinstance(func, JITCallable)
        # Merge our used_global_vals with those of the called function,
        # after checking that all overlapping values are consistent.
        for k in self.used_global_vals.keys() & func.used_global_vals.keys():
            var_name, _ = k
            v1, _ = self.used_global_vals[k]
            v2, _ = func.used_global_vals[k]
            if not self._values_equal(v1, v2):
                raise RuntimeError(
                    f"Global variable {var_name} has value {v1} when compiling {self.name}, but inner kernel {func.__name__} has conflicting value {v2} from when it was first compiled.  This is not allowed."
                )
        self.used_global_vals.update(func.used_global_vals)
        # update hash
        func_key = func.cache_key
        func_key += str(getattr(func, "noinline", False))
        self.hasher.update(func_key.encode("utf-8"))

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

    def record_reference(self, val, var_dict=None, name=None):
        from ..language.core import constexpr

        # Only keep track of "interesting" global variables, that non-evil users
        # might change.  Don't consider functions, modules, builtins, etc.  This
        # helps keep the list of vars we have to check small.
        if val is None or type(val) is ModuleType:
            return

        # if getattr(val, "__avelang_aggregate__", False):
        #     for attr in val.hash_attrs:
        #         self.record_reference(attr)
        #     return

        # if getattr(val, "__avelang_builtin__", False):
        #     return

        # # Stubs that aren't real functions
        # if getattr(val, "__module__", "") == "avelang.language.extra.libdevice":
        #     return

        if isinstance(val, JITCallable):
            self._update_hash(val)
            return

        if callable(val) and not isinstance(val, type) and not isinstance(val, constexpr):
            raise RuntimeError(
                f"Unsupported function referenced: {val}. Functions called from @avelang.jit "
                "must be decorated with @avelang.jit."
            )

        # Python default arguments are resolved only once, when the
        # function is defined.  So if you do `foo(a=A)` and the value of
        # A changes, foo will still use the old value of A.
        # It would be pretty evil if someone did `import x` and then
        # `x = blah`.
        if self.visiting_arg_default_value:
            return

        if var_dict is not None:
            try:
                stored_val = copy.deepcopy(val)
            except Exception:
                stored_val = val
            self.used_global_vals[(name, id(var_dict))] = (stored_val, var_dict)
        return

    def visit_Name(self, node):
        if type(node.ctx) is ast.Store:
            return node.id

        if self._is_local_name(node.id):
            # The global name is hidden by the local name.
            return None

        def name_lookup(name):
            val = self.globals.get(name, None)
            if val is not None:
                return val, self.globals
            val = self.nonlocals.get(name, None)
            if val is not None:
                return val, self.nonlocals
            return None, None

        val, var_dict = name_lookup(node.id)
        if node.id in self.supported_python_builtins:
            return val

        self.record_reference(val, var_dict, node.id)
        return val

    def visit_Tuple(self, node):
        # We need to explicitly return the tuple values so that visit_Assign can
        # access them in the case of `a, b = ...`.
        return [self.visit(elt) for elt in node.elts]

    def visit_Attribute(self, node):
        lhs = self.visit(node.value)
        while isinstance(lhs, ast.Attribute):
            lhs = self.visit(lhs.value)
        lhs_name = getattr(lhs, "__name__", "")
        if lhs is None or lhs_name in self.supported_modules:
            return None
        ret = getattr(lhs, node.attr)
        self.record_reference(ret)
        return ret

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

    def visit_arguments(self, node):
        # The purpose of this function is to visit everything in `arguments`
        # just like `generic_visit`, except when we're visiting default values
        # (i.e. the `foo` part of `def fn(x = foo)`), we set
        # self.visiting_arg_default_value = True.  This allows visit_Name to be
        # aware that we're inside function default values, which have special
        # semantics.

        # According to the AST docs, the arguments node has the following structure.
        #
        # arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
        #              expr* kw_defaults, arg? kwarg, expr* defaults)
        def visit_defaults(defaults):
            try:
                assert not self.visiting_arg_default_value
                self.visiting_arg_default_value = True
                for expr in defaults:
                    if expr is not None:
                        self.visit(expr)
            finally:
                self.visiting_arg_default_value = False

        for arg in itertools.chain(node.posonlyargs, node.args, [node.vararg] if node.vararg else [], node.kwonlyargs):
            self.visit(arg)

        visit_defaults(node.kw_defaults)

        if node.kwarg is not None:
            self.visit(node.kwarg)

        visit_defaults(node.defaults)

    def visitAssnTarget(self, node):
        # Target is either a single string, or a list of strings (if the assn
        # target is a tuple).
        target = self.visit(node)
        if not self._locals_stack:
            return
        local_names = self._locals_stack[-1]
        if isinstance(target, list):
            local_names.update(target)
        else:
            local_names.add(target)

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            # TODO(jlebar): I don't actually know how to hit this.  You don't
            # get it from `a, b = ...` -- in that case, node.targets is a single
            # Tuple, and in fact we *do* need to handle that case if we want
            # existing code to work.
            raise TypeError("Simultaneous multiple assignment is not supported.")

        self.visitAssnTarget(node.targets[0])

        # This will re-visit the target, but that's OK.
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.visitAssnTarget(node.target)

        # This will re-visit the target, but that's OK.
        self.generic_visit(node)

    def visit_For(self, node):
        self.visitAssnTarget(node.target)

        # This will re-visit the target, but that's fine.
        self.generic_visit(node)


def create_function_from_signature(sig, kparams, backend):
    """
    Equivalent to sig.bind followed by apply_defaults. This generates a
    native Python function (using exec) which can be memoized on a per-kernel
    basis to avoid having to run these expensive functions -- which constitute
    much of the kernel launch overhead -- every time we run the kernel.
    """
    assert len(sig.parameters) == len(kparams)
    # Create the function argument list and the dict entries for the return statement
    specialization = []
    # signature
    for name, kp in zip(sig.parameters.keys(), kparams):
        if kp.is_constexpr:
            specialization.append(f'("constexpr", {name})')
        else:
            is_const = "True" if kp.is_const else "False"
            specialize = "False" if kp.do_not_specialize else "True"
            align = "False" if kp.do_not_specialize_on_alignment else "True"
            ret = f"specialize_impl(backend, {name}, {is_const}, {specialize}, {align})"
            if kp.annotation_type:
                if isinstance(kp.annotation_type, str):
                    if kp.annotation_type == "u1" or kp.annotation_type[:2] in ["fp", "bf"]:
                        # we do not specialize non-constexpr floats and bools:
                        specialize = False
                if specialize:
                    specialization.append(f'("{kp.annotation_type}",) + {ret}[1:]')
                else:
                    # skip runtime specialization:
                    specialization.append(f'("{kp.annotation_type}", None)')
            else:
                specialization.append(f"{ret}")

    # compute argument string for a given parameter
    def arg(x):
        return x[0] if x[1].default is inspect.Parameter.empty else f"{x[0]}=default_{x[0]}"

    func_body = f"""
def dynamic_func({", ".join(list(map(arg, sig.parameters.items())) + ["**options"])}):
    params = {{{", ".join([f"'{name}': {name}" for name in sig.parameters.keys()])}}}
    specialization = [{",".join(specialization)}]
    return params, specialization, options
"""

    # Prepare defaults to be inserted into function namespace
    func_namespace = {
        f"default_{name}": param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }

    # TODO: Use native_specialize_impl to improve performance
    from .specialize import reference_specialize_impl

    specialize_impl = reference_specialize_impl
    func_namespace["specialize_impl"] = specialize_impl
    func_namespace["backend"] = backend
    func_namespace["JITCallable"] = JITCallable

    # Execute the function string in func_namespace to create the function
    exec(func_body, func_namespace)

    # Extract the newly created function from the namespace
    return func_namespace["dynamic_func"]


def get_full_name(fn):
    return f"{fn.__module__}.{fn.__qualname__}"


def get_jit_fn_file_line(fn):
    base_fn = fn
    while not isinstance(base_fn, JITCallable):
        base_fn = base_fn.fn
    file_name = base_fn.fn.__code__.co_filename
    begin_line = base_fn.starting_line_number
    # Match the following pattern:
    # @avelang.autotune(...) <- foo.__code__.co_firstlineno
    # @avelang.heuristics(...)
    # @avelang.jit
    # def foo(...): <- this line is the first line
    for idx, line in enumerate(base_fn.raw_src):
        if line.strip().startswith("def "):
            begin_line += idx
            break
    return file_name, begin_line


def _attach_ast_source_metadata(tree, file_name: str, source_code: str):
    tree._avelang_file_name = file_name
    tree._avelang_source_code = source_code
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
        tree.body[0]._avelang_file_name = file_name
        tree.body[0]._avelang_source_code = source_code
    return tree


class KernelInterface(Generic[T]):
    run: T

    def __getitem__(self, dims) -> T:
        """
        A JIT function is launched with: fn[grid](*args, **kwargs).
        Hence JITFunction.__getitem__ returns a callable proxy that
        memorizes the grid.
        """
        return lambda *args, **kwargs: self.run(dims, *args, **kwargs)


class JITCallable:
    def __init__(self, fn: Callable):
        self.fn = fn
        self.signature = inspect.signature(fn)
        try:
            self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)
        except OSError as e:
            raise ValueError("@jit functions should be defined in a Python file") from e
        self._fn_name = get_full_name(fn)
        self._hash_lock = threading.RLock()

        src = textwrap.dedent("".join(self.raw_src))
        src = src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start() :]
        self._src = src
        self.hash = None

        # Reuse docs of wrapped function
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__qualname__ = fn.__qualname__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__

        # Map of global variables used by the function and any functions it
        # transitively calls, plus their values.  The values are collected when
        # the function is first compiled.  Then every time we run the function,
        # we check that the values of the globals match what's expected,
        # otherwise we raise an error.
        #
        # Different functions can have different __globals__ maps, so the map
        # key is actually (var name, id(__globals__)), and the map value is
        # (value, __globals__).
        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}

    def get_capture_scope(self):
        fn = self.fn
        if fn.__closure__ is None:
            return self.__globals__
        nonlocals = {name: cell.cell_contents for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}
        return self.__globals__ | nonlocals

    @property
    def cache_key(self) -> str:
        # TODO : hash should be attribute of `self`
        with self._hash_lock:
            if self.hash is not None:
                return self.hash
            # Set a placeholder hash to break recursion in case the function
            # transitively calls itself. The full hash is set after.
            self.hash = f"recursion:{self._fn_name}"
            nonlocals = inspect.getclosurevars(self.fn).nonlocals
            dependencies_finder = DependenciesFinder(
                name=self._fn_name, globals=self.__globals__, nonlocals=nonlocals, src=self.src
            )
            dependencies_finder.visit(self.parse())
            self.hash = dependencies_finder.ret + str(self.starting_line_number)
            self.used_global_vals = dict(sorted(dependencies_finder.used_global_vals.items()))

            from ..language.core import constexpr

            self.hash += str(
                [(name, val) for (name, _), (val, _) in self.used_global_vals.items() if isinstance(val, constexpr)]
            )
            self.hash = hashlib.sha256(self.hash.encode("utf-8")).hexdigest()
        return self.hash

    def __hash__(self):
        return hash(self.cache_key)

    def parse(self):
        file_name, begin_line = get_jit_fn_file_line(self)
        padded_source = ("\n" * max(begin_line - 1, 0)) + self._src
        tree = ast.parse(self._src, filename=file_name)
        if begin_line > 1:
            tree = ast.increment_lineno(tree, begin_line - 1)
        tree = _attach_ast_source_metadata(tree, file_name, padded_source)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    def _set_src(self, new_src: str):
        raise AttributeError(
            "Cannot set attribute 'src' directly. "
            "Use '_unsafe_update_src()' and manually clear `.hash` of all callers"
            "instead."
        )

    def _get_src(self):
        return self._src

    def _unsafe_update_src(self, new_src: str):
        self._src = new_src
        self.hash = None
        self.used_global_vals = {}

    src = property(fget=_get_src, fset=_set_src)


def _normalize_ty(ty) -> str:
    from ..language import core

    if isinstance(ty, str):
        ty = ty.strip()
        if ty.startswith("const "):
            ty = ty.removeprefix("const")
            ty = _normalize_ty(ty)
            assert ty.startswith("*")
            return "*k" + ty[1:]
        if ty.endswith("*"):
            return "*" + _normalize_ty(ty[:-1])
        if ty.startswith("*"):
            return "*" + _normalize_ty(ty[1:])
        if ty.startswith("S."):
            return _normalize_ty(ty.removeprefix("S."))
    elif isinstance(ty, core.pointer_type):
        return f"*{_normalize_ty(ty.element_ty)}"
    elif isinstance(ty, core.dtype):
        ty = ty.name
    elif isinstance(ty, type):
        ty = ty.__name__
    else:
        ty = str(ty)
    return type_canonicalisation_dict.get(ty.replace("_t", ""), ty)


class KernelParam:
    def __init__(self, num: int, param: inspect.Parameter):
        self.num = num
        self._param = param
        self.do_not_specialize = False
        self.do_not_specialize_on_alignment = False

    @cached_property
    def name(self):
        return self._param.name

    @cached_property
    def annotation(self) -> str:
        if not self._param.annotation or self._param.annotation == inspect.Parameter.empty:
            return ""
        return _normalize_ty(self._param.annotation)

    @cached_property
    def annotation_type(self) -> str:
        a = self.annotation
        if a.startswith("*k"):
            a = a[2:]
        elif a.startswith("*"):
            a = a[1:]
        if a in set(type_canonicalisation_dict.values()):
            return self.annotation
        return ""

    @cached_property
    def is_constexpr(self):
        return "constexpr" in self.annotation

    @cached_property
    def is_const(self):
        if self.is_constexpr:
            return False
        return "const" in self.annotation or self.annotation.startswith("*k")


def compute_cache_key(kernel_key_cache, specialization, options):
    key = (tuple(specialization), str(options))
    cache_key = kernel_key_cache.get(key, None)
    if cache_key is not None:
        return cache_key

    # Replace JITCallable objects with their hash, so the cache key will change if the src is updated
    def replace_callables(obj):
        if isinstance(obj, list):
            return [replace_callables(arg) for arg in obj]
        elif is_namedtuple(obj):
            results = [replace_callables(arg) for arg in obj]
            return obj.__class__(*results)
        elif isinstance(obj, tuple):
            return tuple(replace_callables(arg) for arg in obj)
        elif isinstance(obj, JITCallable):
            return obj.cache_key
        return obj

    cache_key = str(replace_callables(specialization)) + str(options)
    kernel_key_cache[key] = cache_key
    return cache_key


# We probably should call it AOT function as we require all parameters are statically typed
class JITFunction(JITCallable, KernelInterface[T]):
    def __init__(self, fn: Callable):
        super().__init__(fn)
        self.params = []
        for i, param in enumerate(self.signature.parameters.values()):
            self.params.append(KernelParam(i, param))

        # cache of just-in-time compiled kernels
        self.device_caches = defaultdict(self.create_binder)

    @property
    def arg_names(self):
        return [p.name for p in self.params]

    def _infer_constexpr_type(self, value):
        """Infer MLIR type string from Python value."""
        if isinstance(value, JITCallable):
            return "jit"
        elif isinstance(value, bool):
            return "i1"
        elif isinstance(value, int):
            if -2147483648 <= value <= 2147483647:
                return "i32"
            else:
                return "i64"
        elif isinstance(value, float):
            return "f64"
        else:
            raise ValueError(f"Unsupported constexpr type: {type(value)}")

    def _make_constexpr_info(self, value):
        if hasattr(value, "value"):
            value = value.value
        constexpr_type = self._infer_constexpr_type(value)
        if constexpr_type == "jit":
            return {
                "type": constexpr_type,
                "value": value.__name__,
                "_jit_callable": value,
            }
        return {"type": constexpr_type, "value": value}

    def _collect_global_constexprs(self, skip_names=None):
        _ = self.cache_key
        constexprs = {}
        skip = set(skip_names or [])
        for (name, _), (value, _) in self.used_global_vals.items():
            if name in skip:
                continue
            if hasattr(value, "value"):
                value = value.value
            try:
                info = self._make_constexpr_info(value)
            except ValueError:
                continue
            constexprs[name] = info
        return constexprs

    def create_binder(self):
        """
        Precompute as much as possible.
        """
        from ..compiler import compile, ASTSource, make_backend

        target = driver.active.get_current_target()
        backend = make_backend(target)
        self.compile = compile
        self.ASTSource = ASTSource
        binder = create_function_from_signature(self.signature, self.params, backend)
        return {}, {}, target, backend, binder

    def _pack_args(self, backend, kwargs, bound_args, specialization, options):
        # options
        options = backend.parse_options(options)
        # signature
        sigkeys = [x.name for x in self.params]
        sigvals = [x[0] for x in specialization]
        signature = {k: v for (k, v) in zip(sigkeys, sigvals)}
        # check arguments
        assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
        assert "device" not in kwargs, "device option is deprecated; current device will be used"
        assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"
        for k in kwargs:
            if k not in options.__dict__ and k not in sigkeys:
                raise KeyError("Keyword argument %s was specified but unrecognised" % k)
        # constexprs - now with name, type, and value
        constexpr_paths = find_paths_if(sigvals, lambda _, val: val == "constexpr")
        constexprs = {}
        for path in constexpr_paths:
            param_idx = path[0]
            param_name = self.arg_names[param_idx]
            value = get_iterable_path(list(bound_args.values()), path)
            constexprs[param_name] = self._make_constexpr_info(value)
        global_constexprs = self._collect_global_constexprs(constexprs.keys())
        # attributes
        attrs = None

        return options, signature, constexprs, global_constexprs, attrs

    def run(self, dims, *args, **kwargs):
        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        # specialization is list[tuple[str, Any]], where first element of tuple is
        # the type and the second parameter is the 'specialization' value.
        bound_args, specialization, options = binder(*args, **kwargs)

        grid, block = dims()

        options, signature, constexprs, global_constexprs, attrs = self._pack_args(
            backend, kwargs, bound_args, specialization, options
        )

        key = compute_cache_key(kernel_key_cache, specialization, options)
        kernel = kernel_cache.get(key, None)

        # Kernel is not cached; we have to compile.
        if kernel is None:
            options, signature, constexprs, global_constexprs, attrs = self._pack_args(
                backend, kwargs, bound_args, specialization, options
            )

            kernel = self._do_compile(key, signature, device, constexprs, global_constexprs, options, attrs)
            if kernel is None:
                raise RuntimeError(f"Kernel compilation failed for {self.__name__}")

        # TODO: support bound args so that args / kwargs work
        kernel.run(
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            stream,
            kernel.function,
            *bound_args.values(),
        )

    def _do_compile(self, key, signature, device, constexprs, global_constexprs, options, attrs):
        kernel_cache, _, target, _, _ = self.device_caches[device]

        src = self.ASTSource(self, signature, constexprs, attrs, global_constexprs)
        kernel = self.compile(src, target=target, options=options)
        kernel_cache[key] = kernel
        return kernel

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Cannot call @avelang.jit'd outside of the scope of a kernel")


def jit(fn: Callable = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        assert callable(func), "jit decorator must be used on a callable"
        return JITFunction(func)

    if fn is None:
        return decorator
    else:
        return decorator(fn)
