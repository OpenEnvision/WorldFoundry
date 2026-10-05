"""Recursive HDF5 save/load and structural equality checks for Pydantic models.

Eval artifacts and camera / action dumps are sometimes stored as HDF5
groups rather than JSON so arrays stay binary. :func:`hdf5_save` /
:func:`hdf5_load` walk a Pydantic model (or a plain dict), writing
scalars as attributes and arrays as datasets. :func:`hdf5_equal`
compares two groups structurally for golden-file tests.

None fields are dropped on save (``exclude_none=True``). This is not
a general HDF5 ORM and does not stream video frames.
"""

import h5py
import numpy as np
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────────
# Nested Pydantic / dict ↔ HDF5 — None dropped; not a video streamer
# ──────────────────────────────────────────────────────────────────────────


def hdf5_save(data: BaseModel | dict, group: h5py.Group) -> None:
    """Recursively save a Pydantic model or dict into an HDF5 *group*.

    Arrays become datasets; nested models/dicts become subgroups;
    scalars are stored as length-1 datasets. Unsupported types raise
    :class:`ValueError`.
    """
    if isinstance(data, BaseModel):
        # Convert to dict and exclude None values
        data_dict = data.model_dump(mode="python", exclude_none=True)
    else:
        data_dict = data

    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        elif isinstance(value, (BaseModel, dict)):
            subgroup = group.create_group(key)
            hdf5_save(value, subgroup)
        else:
            # For primitive types, convert to numpy array
            try:
                group.create_dataset(key, data=np.array(value))
            except TypeError:
                raise ValueError(f"Unsupported type: {type(value)} for key: {key}")


def hdf5_load(group: h5py.Group) -> dict:
    """Recursively load an HDF5 *group* into a nested dict of arrays/scalars."""
    data_dict = {}
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            data_dict[key] = value[()]
        elif isinstance(value, h5py.Group):
            data_dict[key] = hdf5_load(value)
    return data_dict


def hdf5_is_subset(this: h5py.Group, other: h5py.Group, verbose: bool = False) -> bool:
    """Return True when every dataset/group in *this* exists and matches *other*."""
    for key, value in this.items():
        if key not in other:
            if verbose:
                print(f"Key {key} not in other")
            return False
        elif isinstance(value, h5py.Group):
            if not isinstance(other[key], h5py.Group):
                if verbose:
                    print(f"Key {key} is not a group in other")
                return False
            if not hdf5_is_subset(value, other[key], verbose):
                if verbose:
                    print(f"Key {key} is not a subset of other")
                return False
        elif isinstance(value, h5py.Dataset):
            if not isinstance(other[key], h5py.Dataset):
                if verbose:
                    print(f"Key {key} is not a dataset in other")
                return False
            if not np.array_equal(value, other[key]):
                if verbose:
                    print(f"Key {key} is not equal in other")
                return False
        elif isinstance(value, h5py.Datatype):
            if not isinstance(other[key], h5py.Datatype):
                if verbose:
                    print(f"Key {key} is not a datatype in other")
                return False
            if value != other[key]:
                if verbose:
                    print(f"Key {key} is not equal in other")
                return False
        else:
            # try to compare
            if value != other[key]:
                if verbose:
                    print(f"Key {key} is not equal in other")
                return False
    return True


def hdf5_is_equal(this: h5py.Group, other: h5py.Group, verbose: bool = False) -> bool:
    """Check if this HDF5 group is equal to another HDF5 group."""
    return hdf5_is_subset(this, other, verbose) and hdf5_is_subset(other, this, verbose)
