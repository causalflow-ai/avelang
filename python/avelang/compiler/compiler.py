import hashlib

from ..runtime.driver import driver
from ..backends import backends
from ..backends.compiler import BaseBackend, GPUTarget


class ASTSource:
    def __init__(self, fn, signature, constexprs=None, attrs=None, global_constexprs=None):
        self.fn = fn
        self.signature = signature
        self.constants = dict()
        if constexprs is not None:
            for k, v in constexprs.items():
                k = (fn.arg_names.index(k),) if isinstance(k, str) else k
                assert isinstance(k, tuple)
                self.constants[k] = v
        self.global_constants = dict()
        if global_constexprs is not None:
            for name, info in global_constexprs.items():
                self.global_constants[name] = info
        self.attrs = attrs or dict()
        for k in self.signature.keys():
            if not isinstance(k, str):
                raise TypeError("Signature keys must be string")

    def hash(self):
        sorted_sig = [v for k, v in sorted(self.signature.items())]

        def get_key(x):
            if isinstance(x, dict) and "_jit_callable" in x:
                return x["_jit_callable"].cache_key
            return x.cache_key if hasattr(x, "cache_key") else str(x)

        constants_key = "-".join([get_key(v) for k, v in sorted(self.constants.items())])
        global_key = "-".join([f"{k}:{get_key(v)}" for k, v in sorted(self.global_constants.items())])
        key = f"{self.fn.cache_key}-{str(self.attrs)}-{sorted_sig}-{constants_key}-{global_key}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CompiledKernel:
    def __init__(self, src: ASTSource):
        self.src = src
        self.module = None
        self.function = None
        self.kernel = None
        # TODO: Get name from metadata to ensure mangling is correct
        self.name = src.fn.fn.__name__
        self._run = None

    def _init_handles(self):
        # create launcher
        self._run = driver.active.launcher_cls(self.src)
        device = driver.active.get_current_device()
        self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = driver.active.utils.load_binary(
            self.name, self.kernel, 0, device
        )

    @property
    def run(self):
        if self._run is None:
            self._init_handles()
        return self._run


def make_backend(target: GPUTarget) -> BaseBackend:
    actives = [x.compiler for x in backends.values() if x.compiler.supports_target(target)]
    if len(actives) != 1:
        raise RuntimeError(
            f"{len(actives)} compatible backends for target ({target.backend}) ({actives}). There should only be one."
        )
    return actives[0](target)


def compile(src: ASTSource, target=None, options=None):
    if target is None:
        target = driver.active.get_current_target()
    backend = make_backend(target)
    data = backend.compile(src, target, options)
    k = CompiledKernel(src)
    k.kernel = data
    return k
