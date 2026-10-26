import base64
import dataclasses
import functools
import hashlib
import importlib
import json
import os
import pkgutil
import shutil
import sysconfig
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

from .. import knobs


CACHE_DIR_ENV = knobs.CACHE_DIR_ENV
DISABLE_CACHE_ENV = knobs.DISABLE_CACHE_ENV


class CacheManager(ABC):
    def __init__(self, key):
        pass

    @abstractmethod
    def get_file(self, filename) -> Optional[str]:
        pass

    @abstractmethod
    def put(self, data, filename, binary=True) -> str:
        pass

    @abstractmethod
    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        pass

    @abstractmethod
    def put_group(self, filename: str, group: Dict[str, str]):
        pass


class FileCacheManager(CacheManager):
    def __init__(self, key):
        self.key = key
        self.cache_dir = os.path.join(default_cache_dir(), self.key)
        self.lock_path = os.path.join(self.cache_dir, "lock")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _make_path(self, filename) -> str:
        return os.path.join(self.cache_dir, filename)

    def has_file(self, filename) -> bool:
        return os.path.exists(self._make_path(filename))

    def get_file(self, filename) -> Optional[str]:
        if self.has_file(filename):
            return self._make_path(filename)
        return None

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        group_filename = f"__grp__{filename}"
        group_path = self.get_file(group_filename)
        if group_path is None:
            return None
        try:
            with open(group_path, encoding="utf-8") as f:
                group_data = json.load(f)
        except Exception:
            return None
        child_paths = group_data.get("child_paths")
        if not isinstance(child_paths, dict):
            return None
        result = {}
        for name, path in child_paths.items():
            if not isinstance(name, str) or not isinstance(path, str):
                return None
            if not os.path.exists(path):
                return None
            result[name] = path
        return result

    def put_group(self, filename: str, group: Dict[str, str]) -> str:
        group_filename = f"__grp__{filename}"
        group_contents = json.dumps({"child_paths": group}, sort_keys=True)
        return self.put(group_contents, group_filename, binary=False)

    def put(self, data, filename, binary=True) -> str:
        binary = isinstance(data, bytes)
        if not binary:
            data = str(data)

        filepath = self._make_path(filename)
        temp_dir = os.path.join(self.cache_dir, f"tmp.pid_{os.getpid()}_{uuid.uuid4()}")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        try:
            mode = "wb" if binary else "w"
            kwargs = {} if binary else {"encoding": "utf-8"}
            with open(temp_path, mode, **kwargs) as f:
                f.write(data)
            os.replace(temp_path, filepath)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return filepath


def _base32(key: str) -> str:
    return base64.b32encode(bytes.fromhex(key)).decode("utf-8").rstrip("=")


def default_cache_dir() -> str:
    return os.path.abspath(os.path.expanduser(knobs.cache.dir))


def is_cache_disabled() -> bool:
    return knobs.cache.disabled


def get_cache_manager(key: str) -> CacheManager:
    return FileCacheManager(_base32(key))


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(_jsonable(k)): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "cache_key"):
        return value.cache_key
    if hasattr(value, "__dict__"):
        return {k: _jsonable(v) for k, v in sorted(vars(value).items()) if not k.startswith("_")}
    return repr(value)


def stable_json(data) -> str:
    return json.dumps(_jsonable(data), sort_keys=True, separators=(",", ":"))


def get_environment_hashes() -> dict[str, str]:
    return {
        key: hashlib.sha256(value.encode("utf-8", errors="surrogateescape")).hexdigest()
        for key, value in sorted(os.environ.items())
    }


@functools.lru_cache(maxsize=None)
def _module_fingerprint(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return "unavailable"

    path = getattr(module, "__file__", None)
    if not path:
        return "builtin"
    path = path[:-1] if path.endswith((".pyc", ".pyo")) else path
    try:
        with open(path, "rb") as f:
            contents = f.read()
    except OSError:
        return "unreadable"

    py_version = sysconfig.get_config_var("py_version_short") or ""
    digest = hashlib.sha256()
    digest.update(py_version.encode("utf-8"))
    digest.update(b"\0")
    digest.update(os.path.abspath(path).encode("utf-8"))
    digest.update(b"\0")
    digest.update(contents)
    return digest.hexdigest()


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=None)
def _avelang_key(version: str) -> str:
    package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_root = os.path.dirname(package_path)
    contents = [_hash_file(__file__)]

    path_prefixes = [
        (os.path.join(package_path, "compiler"), "avelang.compiler."),
        (os.path.join(package_path, "backends"), "avelang.backends."),
        (os.path.join(package_path, "language"), "avelang.language."),
        (os.path.join(package_path, "runtime"), "avelang.runtime."),
    ]
    for path, prefix in path_prefixes:
        if not os.path.isdir(path):
            continue
        for lib in pkgutil.walk_packages([path], prefix=prefix):
            spec = lib.module_finder.find_spec(lib.name)
            origin = getattr(spec, "origin", None)
            if origin and os.path.isfile(origin):
                contents.append(_hash_file(origin))

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    binding_path = os.path.join(python_root, f"_avelang_bindings{ext_suffix}")
    if os.path.exists(binding_path):
        contents.append(_hash_file(binding_path))

    return f"{version}-{'-'.join(contents)}"


def _backend_hash(backend, target) -> str:
    if hasattr(backend, "hash"):
        return backend.hash()
    data = {
        "backend": f"{backend.__class__.__module__}.{backend.__class__.__qualname__}",
        "target": target,
        "module_fingerprint": _module_fingerprint(backend.__class__.__module__),
    }
    return hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()


def _options_hash(options) -> str:
    if hasattr(options, "hash"):
        return options.hash()
    return hashlib.sha256(stable_json(options).encode("utf-8")).hexdigest()


def get_cache_key(src, backend, target, options, version: str) -> str:
    env_vars = get_environment_hashes()
    key = (
        f"{_avelang_key(version)}-"
        f"{src.hash()}-"
        f"{_backend_hash(backend, target)}-"
        f"{_options_hash(options)}-"
        f"{str(sorted(env_vars.items()))}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
