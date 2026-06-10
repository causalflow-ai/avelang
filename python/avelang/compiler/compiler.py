import hashlib
import json
import re
from collections import namedtuple
from pathlib import Path

from avelang import __version__
from ..runtime.driver import driver
from ..backends import backends
from ..backends.compiler import BaseBackend, GPUTarget
from .cache import get_cache_key, get_cache_manager, is_cache_disabled, stable_json


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
            return x.cache_key if hasattr(x, "cache_key") else str(x)

        constants_key = "-".join([get_key(v) for k, v in sorted(self.constants.items())])
        global_key = "-".join([f"{k}:{get_key(v)}" for k, v in sorted(self.global_constants.items())])
        key = f"{self.fn.cache_key}-{str(self.attrs)}-{sorted_sig}-{constants_key}-{global_key}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CompiledKernel:
    def __init__(self, src: ASTSource, metadata_group=None, hash=None, *, kernel=None, metadata=None):
        self.src = src
        self.hash = hash
        self.metadata_group = metadata_group or {}
        if metadata_group is not None:
            metadata_path = next((Path(path) for name, path in metadata_group.items() if name.endswith(".json")), None)
            if metadata_path is None:
                raise RuntimeError("Compiled kernel cache entry is missing metadata.")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            binary_name = metadata.get("binary")
            binary_path = metadata_group.get(binary_name)
            if binary_path is None:
                binary_path = next(
                    (path for name, path in metadata_group.items() if not name.endswith(".json")),
                    None,
                )
            if binary_path is None:
                raise RuntimeError("Compiled kernel cache entry is missing binary data.")
            kernel = Path(binary_path).read_bytes()

        metadata = dict(metadata or {})
        if "target" in metadata and isinstance(metadata["target"], dict):
            metadata["target"] = GPUTarget(**metadata["target"])
        KernelMetadata = namedtuple("KernelMetadata", sorted(metadata.keys())) if metadata else None
        self.metadata = KernelMetadata(**metadata) if KernelMetadata is not None else None
        self.kernel = kernel
        self.name = metadata.get("name", src.fn.fn.__name__)
        self._run = None
        self.module = None
        self.function = None

    def _init_handles(self):
        if self.module is not None:
            return
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


def _safe_file_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:150]
    return stem or "kernel"


def compile(src: ASTSource, target=None, options=None):
    if target is None:
        target = driver.active.get_current_target()
    backend = make_backend(target)
    options = backend.parse_options({} if options is None else options)
    hash = get_cache_key(src, backend, target, options, __version__)

    file_name = _safe_file_stem(src.fn.fn.__name__)
    metadata_filename = f"{file_name}.json"
    binary_filename = f"{file_name}.bin"

    if not is_cache_disabled():
        cache_manager = get_cache_manager(hash)
        metadata_group = cache_manager.get_group(metadata_filename)
        if metadata_group is not None and metadata_group.get(metadata_filename) and metadata_group.get(binary_filename):
            try:
                return CompiledKernel(src, metadata_group, hash)
            except Exception:
                pass

    data = backend.compile(src, target, options)
    metadata = {
        "hash": hash,
        "name": src.fn.fn.__name__,
        "target": target,
        "options": options,
        "binary": binary_filename,
        "avelang_version": __version__,
    }

    if is_cache_disabled():
        return CompiledKernel(src, hash=hash, kernel=data, metadata=json.loads(stable_json(metadata)))

    metadata_group = {}
    metadata_group[binary_filename] = cache_manager.put(data, binary_filename)
    metadata_group[metadata_filename] = cache_manager.put(stable_json(metadata), metadata_filename, binary=False)
    cache_manager.put_group(metadata_filename, metadata_group)
    return CompiledKernel(src, metadata_group, hash)
