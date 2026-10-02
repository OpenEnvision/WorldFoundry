"""Thread-pool and async helpers for parallel I/O and batch execution.

Responsibility
    Overlap decode / H2D with CPU work via thread or process pools, plus
    fire-and-forget image writes.

Boundaries
    Not CUDA Graph — capture is stream-affine and cannot run in a thread
    pool. ``parallel_execution`` uses threads (GIL-bound for CPU work);
    use :func:`parallel_processes` for CPU-bound maps. ``async_return=True``
    must close the pool via :class:`AsyncThreadMap`.

Public surface
    :func:`parallel_execution`, :class:`AsyncThreadMap`, :func:`parallel_threads`,
    :func:`parallel_processes`, :func:`save_image` / :func:`save_image_async`.
"""

import asyncio
import os
from functools import wraps
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from threading import Thread
from typing import Any, Callable, Dict, List

from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────
# Async / fire-and-forget — I/O only; do not capture CUDA Graphs here
# ──────────────────────────────────────────────────────────────────────────


def async_call_func(func):
    """Run a sync function in the default executor so callers can ``await`` it."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        """Submit ``func`` to the loop executor; never call CUDA capture here."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    return wrapper


def slice_func(chunk_index, chunk_dim, chunk_size):
    """Build an index that selects one chunk along ``chunk_dim``."""
    return [slice(None)] * chunk_dim + [slice(chunk_index, chunk_index + chunk_size)]


def async_call(fn):
    """Start ``fn`` on a daemon-less thread and return immediately (errors are lost)."""

    def wrapper(*args, **kwargs):
        """Spawn a thread; the caller cannot join or observe exceptions."""
        Thread(target=fn, args=args, kwargs=kwargs).start()

    return wrapper


def _save_image_impl(save_img, save_path):
    """Create parent dirs then ``imageio.imwrite``; imported lazily."""
    import imageio

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, save_img)


@async_call
def save_image_async(save_img, save_path):
    """Fire-and-forget write; do not use when the caller must see I/O errors."""
    _save_image_impl(save_img, save_path)


def save_image(save_img, save_path):
    """Synchronous write so I/O errors propagate to the caller."""
    _save_image_impl(save_img, save_path)


class AsyncThreadMap:
    """Own a live thread pool and the async results scheduled on it.

    Use as a context manager or call :meth:`get`, which always closes and
    joins the pool. Dropping an unfinished handle terminates the workers as a
    final safety net, preventing ``async_return=True`` from leaking a pool.
    """

    def __init__(self, pool: Any, results: list[Any]) -> None:
        """Adopt an already-started pool and its ``ApplyResult`` list."""
        self._pool = pool
        self._results = results
        self._closed = False

    def get(self) -> list[Any]:
        """Wait for every result, then close+join; terminate the pool on error."""
        try:
            return [result.get() for result in self._results]
        except BaseException:
            self.terminate()
            raise
        finally:
            if not self._closed:
                self.close()
            self.join()

    def close(self) -> None:
        """Refuse new work; idempotent so ``get`` / ``__exit__`` can both call it."""
        if not self._closed:
            self._pool.close()
            self._closed = True

    def terminate(self) -> None:
        """Abort workers immediately; used when an action raised."""
        if not self._closed:
            self._pool.terminate()
            self._closed = True

    def join(self) -> None:
        """Close if needed, then join so threads cannot outlive the handle."""
        if not self._closed:
            self.close()
        self._pool.join()

    def __enter__(self) -> "AsyncThreadMap":
        """Return self so ``with parallel_execution(..., async_return=True)`` works."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close on success, terminate on exception, then always join."""
        del exc_value, traceback
        if exc_type is None:
            self.close()
        else:
            self.terminate()
        self.join()

    def __del__(self) -> None:
        """Last-resort terminate if the caller dropped the handle without ``get``."""
        if not getattr(self, "_closed", True):
            try:
                self.terminate()
                self.join()
            except Exception:
                pass


def parallel_execution(
    *args,
    action: Callable,
    num_processes=32,
    num_workers=None,
    print_progress=False,
    sequential=False,
    async_return=False,
    desc=None,
    **kwargs,
):
    """Map *action* over zipped *args* and *kwargs* using a THREAD pool.

    ``num_workers`` is the canonical worker-count argument; the historical
    ``num_processes`` name remains compatible. Workers are threads, so
    CPU-bound actions do not run in parallel under the GIL; use
    ``parallel_processes`` for those. ``async_return=True`` returns an
    :class:`AsyncThreadMap`; use it as a context manager or call ``get()``.
    """
    args = list(args)

    def get_length(args: List, kwargs: Dict):
        """First list-valued arg/kwarg length; failure: :exc:`NotImplementedError`."""
        for arg in args:
            if isinstance(arg, list):
                return len(arg)
        for value in kwargs.values():
            if isinstance(value, list):
                return len(value)
        raise NotImplementedError

    def get_action_args(length: int, args: List, kwargs: Dict, i: int):
        """Index list args of ``length``; broadcast scalars to every item."""
        action_args = [(arg[i] if isinstance(arg, list) and len(arg) == length else arg) for arg in args]
        action_kwargs = {
            key: (kwargs[key][i] if isinstance(kwargs[key], list) and len(kwargs[key]) == length else kwargs[key])
            for key in kwargs
        }
        return action_args, action_kwargs

    if not sequential:
        worker_count = num_processes if num_workers is None else num_workers
        pool = ThreadPool(processes=worker_count)
        results = []
        try:
            asyncs = []
            length = get_length(args, kwargs)
            for i in range(length):
                action_args, action_kwargs = get_action_args(length, args, kwargs, i)
                asyncs.append(pool.apply_async(action, action_args, action_kwargs))

            if async_return:
                return AsyncThreadMap(pool, asyncs)

            for async_result in tqdm(asyncs, desc=desc, disable=not print_progress):
                results.append(async_result.get())
        except BaseException:
            # Historical behavior leaked the pool when an action raised.
            pool.terminate()
            pool.join()
            raise
        pool.close()
        pool.join()
        return results

    results = []
    length = get_length(args, kwargs)
    for i in tqdm(range(length), desc=desc, disable=not print_progress):
        action_args, action_kwargs = get_action_args(length, args, kwargs, i)
        results.append(action(*action_args, **action_kwargs))
    return results


# ──────────────────────────────────────────────────────────────────────────
# imap-style maps — first ``front_num`` items run serially to fail fast
# ──────────────────────────────────────────────────────────────────────────


def parallel_threads(
    function,
    args,
    workers=0,
    star_args=False,
    kw_args=False,
    front_num=1,
    Pool=ThreadPool,
    **tqdm_kw,
):
    """Map ``function`` over ``args``; ``workers<=0`` means ``cpu_count + workers``."""
    while workers <= 0:
        workers += cpu_count()
    if workers == 1:
        front_num = float("inf")

    try:
        n_args_parallel = len(args) - front_num
    except TypeError:
        n_args_parallel = None
    args = iter(args)

    front = []
    while len(front) < front_num:
        try:
            arg = next(args)
        except StopIteration:
            return front
        front.append(function(*arg) if star_args else function(**arg) if kw_args else function(arg))

    out = []
    with Pool(workers) as pool:
        if star_args:
            futures = pool.imap(starcall, [(function, arg) for arg in args])
        elif kw_args:
            futures = pool.imap(starstarcall, [(function, arg) for arg in args])
        else:
            futures = pool.imap(function, args)
        for result in tqdm(futures, total=n_args_parallel, **tqdm_kw):
            out.append(result)
    return front + out


def parallel_processes(*args, **kwargs):
    """Same as :func:`parallel_threads` but with ``multiprocessing.Pool`` (no GIL)."""

    kwargs["Pool"] = mp.Pool
    return parallel_threads(*args, **kwargs)


def starcall(args):
    """Picklable ``f(*args)`` wrapper for ``Pool.imap``."""
    function, function_args = args
    return function(*function_args)


def starstarcall(args):
    """Picklable ``f(**kwargs)`` wrapper for ``Pool.imap``."""
    function, function_args = args
    return function(**function_args)
