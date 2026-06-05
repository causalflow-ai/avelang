from __future__ import annotations

import functools
import importlib.util
import os
import shutil
import subprocess
import sysconfig
import tempfile

from types import ModuleType


def _build(
    name: str,
    src: str,
    srcdir: str,
    library_dirs: list[str],
    include_dirs: list[str],
    libraries: list[str],
    ccflags: list[str],
) -> str:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so = os.path.join(srcdir, "{name}{suffix}".format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    is_cxx = os.path.splitext(src)[1] in (".cc", ".cpp", ".cxx")
    cc = os.environ.get("CXX" if is_cxx else "CC")
    if cc is None:
        clang = shutil.which("clang++" if is_cxx else "clang")
        gcc = shutil.which("g++" if is_cxx else "gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            compiler = "C++" if is_cxx else "C"
            env_var = "CXX" if is_cxx else "CC"
            raise RuntimeError(f"Failed to find {compiler} compiler. Please specify via the {env_var} environment variable.")
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()  # type: ignore
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This keeps system-wide Python installs working.
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    include_dirs = include_dirs + [srcdir, py_include_dir]
    # for -Wno-psabi, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111047
    cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", so]
    cc_cmd += [f"-l{lib}" for lib in libraries]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    cc_cmd.extend(ccflags)
    subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
    return so


@functools.lru_cache
def platform_key() -> str:
    from platform import machine, system, architecture

    return ",".join([machine(), system(), *architecture()])


def _load_module_from_path(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load newly compiled {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compile_module_from_src(
    src: str,
    name: str,
    library_dirs: list[str] | None = None,
    include_dirs: list[str] | None = None,
    libraries: list[str] | None = None,
    ccflags: list[str] | None = None,
    src_extension: str = ".c",
) -> ModuleType:
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, name + src_extension)
        with open(src_path, "w") as f:
            f.write(src)
        so = _build(name, src_path, tmpdir, library_dirs or [], include_dirs or [], libraries or [], ccflags or [])
        return _load_module_from_path(name, so)
