





import collections.abc
import hashlib
import json
import torch
from collections.abc import Iterable, Mapping, Sized
from typing import Any

from .array import TENSOR_TYPE_CONVERSIONS, TENSOR_TYPES
from .string import callable_to_string, string_to_callable, string_to_slice



def class_to_dict(obj: object) -> dict[str, Any]:

    if not hasattr(obj, "__class__"):
        raise ValueError(f"Expected a class instance. Received: {type(obj)}.")

    if isinstance(obj, dict):
        obj_dict = obj
    elif isinstance(obj, torch.Tensor):



        return obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return obj


    data = dict()
    for key, value in obj_dict.items():

        if key.startswith("__"):
            continue

        if callable(value):
            data[key] = callable_to_string(value)

        elif hasattr(value, "__dict__") or isinstance(value, dict):
            data[key] = class_to_dict(value)

        elif isinstance(value, (list, tuple)):
            data[key] = type(value)([class_to_dict(v) for v in value])
        else:
            data[key] = value
    return data


def update_class_from_dict(obj, data: dict[str, Any], _ns: str = "") -> None:
    for key, value in data.items():

        key_ns = _ns + "/" + key


        if hasattr(obj, key) or (isinstance(obj, dict) and key in obj):
            obj_mem = obj[key] if isinstance(obj, dict) else getattr(obj, key)


            if isinstance(value, Mapping):

                update_class_from_dict(obj_mem, value, _ns=key_ns)
                continue


            if isinstance(value, Iterable) and not isinstance(value, str):


                if all(not isinstance(el, Mapping) for el in value):
                    out_val = tuple(value) if isinstance(obj_mem, tuple) else value
                    if isinstance(obj, dict):
                        obj[key] = out_val
                    else:
                        setattr(obj, key, out_val)
                    continue


                if obj_mem is None:
                    raise ValueError(
                        f"[Config]: Cannot merge list under namespace: {key_ns} because the existing value is None."
                    )


                if isinstance(obj_mem, Sized) and isinstance(value, Sized) and len(obj_mem) != len(value):
                    raise ValueError(
                        f"[Config]: Incorrect length under namespace: {key_ns}."
                        f" Expected: {len(obj_mem)}, Received: {len(value)}."
                    )


                if isinstance(obj_mem, tuple):
                    value = tuple(value)
                else:
                    set_obj = True

                    for i in range(len(obj_mem)):
                        if isinstance(value[i], Mapping):
                            update_class_from_dict(obj_mem[i], value[i], _ns=key_ns)
                            set_obj = False

                    if not set_obj:
                        continue


            elif callable(obj_mem):

                value = string_to_callable(value)


            elif value is None or isinstance(value, type(obj_mem)):
                pass


            else:
                raise ValueError(
                    f"[Config]: Incorrect type under namespace: {key_ns}."
                    f" Expected: {type(obj_mem)}, Received: {type(value)}."
                )


            if isinstance(obj, dict):
                obj[key] = value
            else:
                setattr(obj, key, value)


        else:
            raise KeyError(f"[Config]: Key not found under namespace: {key_ns}.")




def dict_to_md5_hash(data: object) -> str:

    if isinstance(data, dict):
        encoded_buffer = json.dumps(data, sort_keys=True).encode()
    else:
        encoded_buffer = json.dumps(class_to_dict(data), sort_keys=True).encode()

    data_hash = hashlib.md5()
    data_hash.update(encoded_buffer)

    return data_hash.hexdigest()




def convert_dict_to_backend(
    data: dict, backend: str = "numpy", array_types: Iterable[str] = ("numpy", "torch", "warp")
) -> dict:


    if backend not in TENSOR_TYPE_CONVERSIONS:
        raise ValueError(f"Unknown backend '{backend}'. Supported backends are 'numpy', 'torch', and 'warp'.")

    tensor_type_conversions = TENSOR_TYPE_CONVERSIONS[backend]


    parsed_types = list()
    for t in array_types:

        if t not in TENSOR_TYPES:
            raise ValueError(f"Unknown array type: '{t}'. Supported array types are 'numpy', 'torch', and 'warp'.")

        if t == backend:
            continue

        parsed_types.append(TENSOR_TYPES[t])


    output_dict = dict()
    for key, value in data.items():

        data_type = type(value)

        if data_type in parsed_types:

            if data_type not in tensor_type_conversions:
                raise ValueError(f"No registered conversion for data type: {data_type} to {backend}!")

            output_dict[key] = tensor_type_conversions[data_type](value)

        elif isinstance(data[key], dict):
            output_dict[key] = convert_dict_to_backend(value)

        else:
            output_dict[key] = value

    return output_dict


def update_dict(orig_dict: dict, new_dict: collections.abc.Mapping) -> dict:
    for keyname, value in new_dict.items():
        if isinstance(value, collections.abc.Mapping):
            orig_dict[keyname] = update_dict(orig_dict.get(keyname, {}), value)
        else:
            orig_dict[keyname] = value
    return orig_dict


def replace_slices_with_strings(data: dict) -> dict:
    if isinstance(data, dict):
        return {k: replace_slices_with_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_slices_with_strings(v) for v in data]
    elif isinstance(data, slice):
        return f"slice({data.start},{data.stop},{data.step})"
    else:
        return data


def replace_strings_with_slices(data: dict) -> dict:
    if isinstance(data, dict):
        return {k: replace_strings_with_slices(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_strings_with_slices(v) for v in data]
    elif isinstance(data, str) and data.startswith("slice("):
        return string_to_slice(data)
    else:
        return data


def print_dict(val, nesting: int = -4, start: bool = True):
    if isinstance(val, dict):
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:

        if callable(val):
            print(callable_to_string(val))
        else:
            print(val)
