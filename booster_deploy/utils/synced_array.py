
from __future__ import annotations

import atexit
from multiprocessing import shared_memory
from typing import Tuple, Union
import fcntl
import os
import tempfile

import numpy as np


class SyncedArray:

    def __init__(
        self,
        name: str,
        shape: Union[Tuple[int, ...], int],
        dtype: Union[str, np.dtype] = "float32",
        create: bool = True,
        owner_pid: int | None = None,
    ) -> None:

        self.name = name
        if isinstance(shape, int):
            self.shape = (shape,)
        else:
            self.shape = tuple(shape)

        self.dtype = np.dtype(dtype) if isinstance(dtype, str) else dtype

        self._numel = int(np.prod(self.shape))
        self._size = int(self._numel * self.dtype.itemsize)


        if owner_pid is None:
            owner_pid = os.getpid() if create else os.getppid()
        self._owner_pid = int(owner_pid)


        self._shm_name = f"{self.name}_{self._owner_pid}_shm"
        try:
            if create:

                self.shm = shared_memory.SharedMemory(
                    name=self._shm_name, create=True, size=self._size
                )
            else:

                self.shm = shared_memory.SharedMemory(name=self._shm_name)
        except FileExistsError:


            if create:
                self.shm = shared_memory.SharedMemory(name=self._shm_name)
            else:
                raise



        lock_dir = tempfile.gettempdir()


        self._lock_path = os.path.join(
            lock_dir, f"synced_array_{self.name}_{self._owner_pid}.lock"
        )


        self._lock_fd = open(self._lock_path, "a+")

        self._closed = False


        atexit.register(self.cleanup)


    def write(self, arr: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("SyncedArray is closed")
        data = np.asarray(arr, dtype=self.dtype)
        if data.size != self._numel:
            raise ValueError(
                f"Expected {self._numel} elements, got {data.size}"
            )
        if data.shape != self.shape:
            data = data.reshape(self.shape)


        try:
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX)
            dst = np.frombuffer(
                self.shm.buf, dtype=self.dtype, count=self._numel
            ).reshape(self.shape)
            np.copyto(dst, data)
        finally:
            try:
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

    def read(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("SyncedArray is closed")

        try:
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_SH)
            src = np.frombuffer(
                self.shm.buf, dtype=self.dtype, count=self._numel
            ).reshape(self.shape)
            return src.copy()
        finally:
            try:
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

    def modify_in_place(self, func):
        if self._closed:
            raise RuntimeError("SyncedArray is closed")
        try:
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX)
            dst = np.frombuffer(
                self.shm.buf, dtype=self.dtype, count=self._numel
            ).reshape(self.shape)

            func(dst)
        finally:
            try:
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.shm.close()
        except Exception:
            pass
        try:


            try:
                self.shm.unlink()
            except Exception:

                pass
        finally:

            try:
                if hasattr(self, "_lock_fd") and self._lock_fd:
                    try:
                        self._lock_fd.close()
                    except Exception:
                        pass
            finally:
                self._lock_fd = None

    @staticmethod
    def attach(
        name: str,
        shape: Union[Tuple[int, ...], int],
        dtype: Union[str, np.dtype] = "float32",
    ) -> "SyncedArray":

        last_exc: Exception | None = None
        for pid_candidate in (os.getpid(), os.getppid()):
            try:
                return SyncedArray(
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    create=False,
                    owner_pid=pid_candidate,
                )
            except FileNotFoundError as e:
                last_exc = e

                continue

        if last_exc:
            raise last_exc

        return SyncedArray(name=name, shape=shape, dtype=dtype, create=False)


__all__ = ["SyncedArray"]
