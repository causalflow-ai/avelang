#!/usr/bin/env python3
import dataclasses
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from avelang import knobs
from avelang.backends.compiler import GPUTarget
from avelang.compiler.compiler import ASTSource
from avelang.compiler import compiler as compiler_mod


class _KernelFunction:
    __name__ = "cached_kernel"


class _FakeJITFunction:
    arg_names = []
    fn = _KernelFunction()

    def __init__(self, cache_key):
        self.cache_key = cache_key


@dataclasses.dataclass(frozen=True)
class _DummyOptions:
    dummy_option: int = -1


class _DummyBackend:
    def __init__(self, target):
        self.target = target
        self.compile_calls = 0

    def parse_options(self, options):
        if isinstance(options, _DummyOptions):
            return options
        return _DummyOptions(dummy_option=options.get("dummy_option", -1))

    def compile(self, src, target, options=None):
        self.compile_calls += 1
        return f"compiled-{self.compile_calls}".encode("utf-8")


class TestCompilerCache(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.target = GPUTarget("dummy-gpu", "dummy-chip")

    def _src(self, source_key="source"):
        return ASTSource(_FakeJITFunction(source_key), {}, {}, {})

    def _compile_with_backend(self, backend, src, options):
        with mock.patch.object(compiler_mod, "make_backend", return_value=backend):
            return compiler_mod.compile(src, target=self.target, options=options)

    def test_reuses_cached_binary_for_same_source_target_and_options(self):
        first_backend = _DummyBackend(self.target)
        second_backend = _DummyBackend(self.target)
        env = {knobs.CACHE_DIR_ENV: self.tmpdir.name}

        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop(knobs.DISABLE_CACHE_ENV, None)
            first = self._compile_with_backend(first_backend, self._src(), {"dummy_option": 4})
            second = self._compile_with_backend(second_backend, self._src(), {"dummy_option": 4})

        self.assertEqual(first.kernel, b"compiled-1")
        self.assertEqual(second.kernel, b"compiled-1")
        self.assertEqual(first_backend.compile_calls, 1)
        self.assertEqual(second_backend.compile_calls, 0)
        self.assertEqual(first.hash, second.hash)
        cached_binaries = list(Path(self.tmpdir.name).rglob("cached_kernel.bin"))
        self.assertTrue(cached_binaries)

    def test_cache_key_includes_backend_options(self):
        backend = _DummyBackend(self.target)
        env = {knobs.CACHE_DIR_ENV: self.tmpdir.name}

        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop(knobs.DISABLE_CACHE_ENV, None)
            first = self._compile_with_backend(backend, self._src(), {"dummy_option": 4})
            second = self._compile_with_backend(backend, self._src(), {"dummy_option": 8})

        self.assertEqual(first.kernel, b"compiled-1")
        self.assertEqual(second.kernel, b"compiled-2")
        self.assertNotEqual(first.hash, second.hash)

    def test_cache_key_includes_source_hash(self):
        backend = _DummyBackend(self.target)
        env = {knobs.CACHE_DIR_ENV: self.tmpdir.name}

        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop(knobs.DISABLE_CACHE_ENV, None)
            first = self._compile_with_backend(backend, self._src("source-a"), {"dummy_option": 4})
            second = self._compile_with_backend(backend, self._src("source-b"), {"dummy_option": 4})

        self.assertEqual(first.kernel, b"compiled-1")
        self.assertEqual(second.kernel, b"compiled-2")
        self.assertNotEqual(first.hash, second.hash)

    def test_cache_key_includes_environment_hashes(self):
        backend = _DummyBackend(self.target)
        env = {knobs.CACHE_DIR_ENV: self.tmpdir.name}

        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop(knobs.DISABLE_CACHE_ENV, None)
            os.environ["AVELANG_TEST_CACHE_ENV"] = "first"
            first = self._compile_with_backend(backend, self._src(), {"dummy_option": 4})
            os.environ["AVELANG_TEST_CACHE_ENV"] = "second"
            second = self._compile_with_backend(backend, self._src(), {"dummy_option": 4})

        self.assertEqual(first.kernel, b"compiled-1")
        self.assertEqual(second.kernel, b"compiled-2")
        self.assertNotEqual(first.hash, second.hash)

    def test_disable_cache_env_bypasses_reads_and_writes(self):
        backend = _DummyBackend(self.target)
        env = {knobs.CACHE_DIR_ENV: self.tmpdir.name, knobs.DISABLE_CACHE_ENV: "1"}

        with mock.patch.dict(os.environ, env, clear=False):
            first = self._compile_with_backend(backend, self._src(), {"dummy_option": 4})
            second = self._compile_with_backend(backend, self._src(), {"dummy_option": 4})

        self.assertEqual(first.kernel, b"compiled-1")
        self.assertEqual(second.kernel, b"compiled-2")
        self.assertEqual(backend.compile_calls, 2)
        self.assertFalse(Path(self.tmpdir.name).exists() and any(Path(self.tmpdir.name).iterdir()))


if __name__ == "__main__":
    unittest.main()
