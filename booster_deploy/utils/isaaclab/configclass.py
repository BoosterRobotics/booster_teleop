





import inspect
import types
from collections.abc import Callable
from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, field, replace
from typing import Any, ClassVar

from .dict import class_to_dict, update_class_from_dict

_CONFIGCLASS_METHODS = ["to_dict", "from_dict", "replace", "copy", "validate"]



def __dataclass_transform__():
    return lambda a: a


@__dataclass_transform__()
def configclass(cls, **kwargs):

    _add_annotation_types(cls)

    _process_mutable_types(cls)


    if hasattr(cls, "__post_init__"):
        setattr(cls, "__post_init__", _combined_function(cls.__post_init__, _custom_post_init))
    else:
        setattr(cls, "__post_init__", _custom_post_init)

    setattr(cls, "to_dict", _class_to_dict)
    setattr(cls, "from_dict", _update_class_from_dict)
    setattr(cls, "replace", _replace_class_with_kwargs)
    setattr(cls, "copy", _copy_class)
    setattr(cls, "validate", _validate)

    cls = dataclass(cls, **kwargs)

    return cls




def _class_to_dict(obj: object) -> dict[str, Any]:
    return class_to_dict(obj)


def _update_class_from_dict(obj, data: dict[str, Any]) -> None:
    update_class_from_dict(obj, data, _ns="")


def _replace_class_with_kwargs(obj: object, **kwargs) -> object:
    return replace(obj, **kwargs)


def _copy_class(obj: object) -> object:
    return replace(obj)




def _add_annotation_types(cls):

    hints = {}


    for base in reversed(cls.__mro__):

        if base is object:
            continue

        ann = base.__dict__.get("__annotations__", {})

        hints.update(ann)



        for key in base.__dict__:

            value = getattr(base, key)

            if _skippable_class_member(key, value, hints):
                continue


            if not isinstance(value, type):
                if key not in hints:


                    if value is MISSING:
                        raise TypeError(
                            f"Missing type annotation for '{key}' in class '{cls.__name__}'."
                            " Please add a type annotation or set a default value."
                        )

                    hints[key] = type(value)
            elif key != value.__name__:



                hints[key] = f"type[{value.__name__}]"



    cls.__annotations__ = cls.__dict__.get("__annotations__", {})
    cls.__annotations__ = hints


def _validate(obj: object, prefix: str = "") -> list[str]:
    missing_fields = []

    if type(obj) is type(MISSING):
        missing_fields.append(prefix)
        return missing_fields
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            current_path = f"{prefix}[{index}]"
            missing_fields.extend(_validate(item, prefix=current_path))
        return missing_fields
    elif isinstance(obj, dict):
        obj_dict = obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return missing_fields

    for key, value in obj_dict.items():

        if key.startswith("__"):
            continue
        current_path = f"{prefix}.{key}" if prefix else key
        missing_fields.extend(_validate(value, prefix=current_path))


    if prefix == "" and missing_fields:
        formatted_message = "\n".join(f"  - {field}" for field in missing_fields)
        raise TypeError(
            f"Missing values detected in object {obj.__class__.__name__} for the following"
            f" fields:\n{formatted_message}\n"
        )
    return missing_fields


def _process_mutable_types(cls):


    ann = cls.__dict__.get("__annotations__", {})


    class_members = {}
    for base in reversed(cls.__mro__):

        if base is object:
            continue

        for key in base.__dict__:

            f = getattr(base, key)

            if _skippable_class_member(key, f):
                continue

            if not isinstance(f, type) or key in ann:
                class_members[key] = f



        for key, f in base.__dict__.get("__dataclass_fields__", {}).items():

            if not isinstance(f, type):
                class_members[key] = f



    if len(class_members) != len(ann):
        raise ValueError(
            f"In class '{cls.__name__}', number of annotations ({len(ann)}) does not match number of class members"
            f" ({len(class_members)}). Please check that all class members have type annotations and/or a default"
            " value. If you don't want to specify a default value, please use the literal `dataclasses.MISSING`."
        )

    for key in ann:

        value = class_members.get(key, MISSING)


        origin = getattr(ann[key], "__origin__", None)
        if origin is ClassVar:
            continue







        if isinstance(value, Field):
            setattr(cls, key, value)
        elif not isinstance(value, type):

            value = field(default_factory=_return_f(value))
            setattr(cls, key, value)


def _custom_post_init(obj):
    for key in dir(obj):

        if key.startswith("__"):
            continue

        value = getattr(obj, key)

        ann = obj.__class__.__dict__.get(key)

        if not callable(value) and not isinstance(ann, property):
            setattr(obj, key, deepcopy(value))


def _combined_function(f1: Callable, f2: Callable) -> Callable:

    def _combined(*args, **kwargs):


        f1(*args, **kwargs)
        f2(*args, **kwargs)

    return _combined




def _skippable_class_member(key: str, value: Any, hints: dict | None = None) -> bool:

    if key.startswith("__"):
        return True

    if key in _CONFIGCLASS_METHODS:
        return True

    if hints is not None and key in hints:
        return True

    if callable(value):


        if isinstance(value, types.MethodType):
            return True

        signature = inspect.signature(value)
        if "self" in signature.parameters or "cls" in signature.parameters:
            return True

    if isinstance(value, property):
        return True

    return False


def _return_f(f: Any) -> Callable[[], Any]:

    def _wrap():

        if isinstance(f, Field):
            if f.default_factory is MISSING:
                return deepcopy(f.default)
            else:
                return f.default_factory
        else:
            return deepcopy(f)

    return _wrap
