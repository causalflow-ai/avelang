from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Generic, Optional, Type, TypeVar, Union, cast


class Env:
    pass


env = Env()
propagate_env = True


CC_ENV = "CC"
CXX_ENV = "CXX"
LD_LIBRARY_PATH_ENV = "LD_LIBRARY_PATH"


def getenv(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


def getenv_bool(key: str, default: bool = False) -> bool:
    value = getenv(key)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def setenv(key: str, value: Optional[str]) -> None:
    if not propagate_env:
        return
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def toenv(value: Any) -> Optional[tuple[Optional[str]]]:
    if value is None:
        return (None,)
    if type(value) is bool:
        return ("1" if value else "0",)
    if type(value) is str:
        return (value,)
    if type(value) is int:
        return (str(value),)
    return None


SetType = TypeVar("SetType")
GetType = TypeVar("GetType")
_NOTHING = object()


class env_base(Generic[SetType, GetType]):
    def __init__(self, key: str) -> None:
        self.key = key

    def __set_name__(self, objclass: Type[object], name: str) -> None:
        self.name = name

    def __get__(self, obj: Optional[object], objclass: Optional[Type[object]]) -> GetType:
        if obj is None:
            return self  # type: ignore[return-value]
        py_value = obj.__dict__.get(self.name, _NOTHING)
        if py_value is _NOTHING:
            return self.get()
        return self.transform(py_value)

    def __set__(self, obj: object, value: Union[SetType, Env]) -> None:
        if isinstance(value, Env):
            obj.__dict__.pop(self.name, None)
            return
        obj.__dict__[self.name] = value
        env_value = toenv(value)
        if env_value is not None:
            setenv(self.key, env_value[0])

    def __delete__(self, obj: object) -> None:
        obj.__dict__.pop(self.name, None)

    def get(self) -> GetType:
        raise NotImplementedError

    def transform(self, value: SetType) -> GetType:
        return cast(GetType, value)


class env_str(env_base[str, str]):
    def __init__(self, key: str, default: str) -> None:
        super().__init__(key)
        self.default = default

    def get(self) -> str:
        return cast(str, getenv(self.key, self.default))


class env_str_callable_default(env_base[str, str]):
    def __init__(self, key: str, default_factory: Callable[[], str]) -> None:
        super().__init__(key)
        self.default_factory = default_factory

    def get(self) -> str:
        value = getenv(self.key)
        if value is None:
            return self.default_factory()
        return value


class env_bool(env_base[bool, bool]):
    def __init__(self, key: str, default: bool = False) -> None:
        super().__init__(key)
        self.default = default

    def get(self) -> bool:
        return getenv_bool(self.key, self.default)


class env_opt_str(env_base[Optional[str], Optional[str]]):
    def get(self) -> Optional[str]:
        return getenv(self.key)


knobs_type = TypeVar("knobs_type", bound="base_knobs")


class base_knobs:
    @property
    def knob_descriptors(self) -> dict[str, env_base]:
        return {name: value for name, value in type(self).__dict__.items() if isinstance(value, env_base)}

    @property
    def knobs(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.knob_descriptors}

    def copy(self: knobs_type) -> knobs_type:
        result = type(self)()
        result.__dict__.update(self.__dict__)
        return result

    def reset(self: knobs_type) -> knobs_type:
        for knob in self.knob_descriptors:
            delattr(self, knob)
        return self

    @contextmanager
    def scope(self) -> Generator[None, None, None]:
        initial_env = {knob.key: getenv(knob.key) for knob in self.knob_descriptors.values()}
        original = dict(self.__dict__)
        try:
            yield
        finally:
            self.__dict__.clear()
            self.__dict__.update(original)
            for key, value in initial_env.items():
                setenv(key, value)


class build_knobs(base_knobs):
    cc: env_opt_str = env_opt_str(CC_ENV)
    cxx: env_opt_str = env_opt_str(CXX_ENV)
    ld_library_path: env_opt_str = env_opt_str(LD_LIBRARY_PATH_ENV)


build = build_knobs()
