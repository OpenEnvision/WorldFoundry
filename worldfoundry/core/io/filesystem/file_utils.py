"""Cross-platform filesystem helpers: paths, copies, archives, and pickles.

``f_*`` helpers expand ``~`` and create parents. Prefer ``storage.py``
for remote URIs; this module is local-path only.
"""

import ast
import glob
import hashlib
import os
import pickle
import shutil
import sys
from typing import Callable, Union

from worldfoundry.core.utils.python.functional_utils import is_sequence

__all__ = [
    "create_tar",
    "dump_pickle",
    "dump_text",
    "dump_text_lines",
    "extract_tar",
    "file_sha256",
    "f_add_ext",
    "f_append_before_ext",
    "f_copy",
    "f_copytree",
    "f_exists",
    "f_expand",
    "f_ext",
    "f_glob",
    "f_has_ext",
    "f_join",
    "f_listdir",
    "f_mkdir",
    "f_mkdir_in_path",
    "f_move",
    "f_not_empty",
    "f_remove",
    "f_size",
    "f_split_path",
    "f_time",
    "get_dir",
    "get_file_lock",
    "get_package_root",
    "get_parent_dir",
    "get_script_dir",
    "get_script_file_name",
    "get_script_self_path",
    "host_id",
    "host_name",
    "insert_before_ext",
    "is_abs_path",
    "is_dir",
    "is_file",
    "is_relative_path",
    "last_part_in_path",
    "load_pickle",
    "load_text",
    "load_text_lines",
    "materialize_file",
    "md5_checksum",
    "move_with_backup",
    "next_available_file_name",
    "owner_name",
    "pickle_dump",
    "pickle_load",
    "read_text",
    "read_text_lines",
    "text_dump",
    "text_load",
    "timestamp_file_name",
    "utf_open",
    "write_text",
    "write_text_lines",
]


def file_sha256(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file's SHA-256 digest without loading the whole artifact into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


f_ext = os.path.splitext

f_size = os.path.getsize

is_file = os.path.isfile

is_dir = os.path.isdir

get_dir = os.path.dirname


# ──────────────────────────────────────────────────────────────────────────
# Host identity — unix owner / hostname for lock and log prefixes
# ──────────────────────────────────────────────────────────────────────────


def owner_name(filepath):
    """
    Returns: file owner name, unix only
    """
    import pwd

    return pwd.getpwuid(os.stat(filepath).st_uid).pw_name


def host_name():
    "Get host name, alias with ``socket.gethostname()``"
    from socket import gethostname

    return gethostname()


def host_id():
    """
    Returns: first part of hostname up to '.'
    """
    return host_name().split(".")[0]


# ──────────────────────────────────────────────────────────────────────────
# Path helpers — expand ~ / $VARS; parents created on write, not on join
# ──────────────────────────────────────────────────────────────────────────


def utf_open(fname, mode):
    """
    Wrapper for codecs.open
    """
    import codecs

    return codecs.open(fname, mode=mode, encoding="utf-8")


def f_not_empty(*fpaths):
    """
    Returns:
        True if and only if the file exists and file size > 0
          if fpath is a dir, if and only if dir exists and has at least 1 file
    """
    fpath = f_join(*fpaths)
    if not os.path.exists(fpath):
        return False

    if os.path.isdir(fpath):
        return len(os.listdir(fpath)) > 0
    else:
        return os.path.getsize(fpath) > 0


def f_expand(fpath):
    """Expand ``~`` and ``$VARS`` so callers never mix raw and resolved paths."""

    return os.path.expandvars(os.path.expanduser(fpath))


def f_exists(*fpaths):
    """True when the joined, expanded path exists (file or directory)."""

    return os.path.exists(f_join(*fpaths))


def f_join(*fpaths):
    """
    Join file paths and expand special symbols like `~` for home dir
    """

    def pack_varargs(args):
        """
        Pack *args or a single list arg as list

        def f(*args):
            arg_list = pack_varargs(args)
            # arg_list is now packed as a list
        """
        assert isinstance(args, tuple), "please input the tuple `args` as in *args"
        if len(args) == 1 and is_sequence(args[0]):
            return args[0]
        else:
            return args

    fpaths = pack_varargs(fpaths)
    fpath = f_expand(os.path.join(*fpaths))
    if isinstance(fpath, str):
        fpath = fpath.strip()
    return fpath


def f_listdir(
    *fpaths,
    filter_ext=None,
    filter=None,
    sort=True,
    full_path=False,
    nonexist_ok=True,
    recursive=False,
):
    """
    Args:
        full_path: True to return full paths to the dir contents
        filter: function that takes in file name and returns True to include
        nonexist_ok: True to return [] if the dir is non-existent, False to raise
        sort: sort the file names by alphabetical
        recursive: True to use os.walk to recursively list files. Note that `filter`
            will be applied to the relative path string to the root dir.
            e.g. filter will take "a/data1.txt" and "a/b/data3.txt" as input, instead of
            just the base file names "data1.txt" and "data3.txt".
            if False, will simply call os.listdir()
    """
    assert not (filter_ext and filter), "filter_ext and filter are mutually exclusive"
    dir_path = f_join(*fpaths)
    if not os.path.exists(dir_path) and nonexist_ok:
        return []
    if recursive:
        files = [
            os.path.join(os.path.relpath(root, dir_path), file)
            for root, _, files in os.walk(dir_path)
            for file in files
        ]
    else:
        files = os.listdir(dir_path)
    if filter is not None:
        files = [f for f in files if filter(f)]
    elif filter_ext is not None:
        files = [f for f in files if f.endswith(filter_ext)]
    if sort:
        files.sort()
    if full_path:
        return [os.path.join(dir_path, f) for f in files]
    else:
        return files


def f_mkdir(*fpaths):
    """
    Recursively creates all the subdirs
    If exist, do nothing.
    """
    fpath = f_join(*fpaths)
    os.makedirs(fpath, exist_ok=True)
    return fpath


def f_mkdir_in_path(*fpaths):
    """
    fpath is a file,
    recursively creates all the parent dirs that lead to the file
    If exist, do nothing.
    """
    os.makedirs(get_dir(f_join(*fpaths)), exist_ok=True)


def last_part_in_path(fpath):
    """
    https://stackoverflow.com/questions/3925096/how-to-get-only-the-last-part-of-a-path-in-python
    """
    return os.path.basename(os.path.normpath(f_expand(fpath)))


def is_abs_path(*fpath):
    """True after join/expand — ``~/x`` is absolute once ``~`` is resolved."""

    return os.path.isabs(f_join(*fpath))


def is_relative_path(*fpath):
    """Negation of :func:`is_abs_path` after the same expand rules."""

    return not is_abs_path(f_join(*fpath))


def f_time(*fpath):
    "File modification time"
    return str(os.path.getctime(f_join(*fpath)))


def f_append_before_ext(fpath, suffix):
    """
    Append a suffix to file name and retain its extension
    """
    name, ext = f_ext(fpath)
    return name + suffix + ext


def f_add_ext(fpath, ext):
    """
    Append an extension if not already there
    Args:
      ext: will add a preceding `.` if doesn't exist
    """
    if not ext.startswith("."):
        ext = "." + ext
    if fpath.endswith(ext):
        return fpath
    else:
        return fpath + ext


def f_has_ext(fpath, ext):
    "Test if file path is a text file"
    _, actual_ext = f_ext(fpath)
    return actual_ext == "." + ext.lstrip(".")


def f_glob(*fpath):
    """Expand ``**`` globs after join; empty match is ``[]``, not an error."""

    return glob.glob(f_join(*fpath), recursive=True)


def f_remove(*fpath, verbose=False, plan_only=False):
    """
    If exist, remove. Supports both dir and file. Supports glob wildcard.
    """
    import errno

    assert isinstance(verbose, bool)
    fpath = f_join(fpath)
    if plan_only:
        print("Plan only, delete:", fpath)
        return
    for f in glob.glob(fpath):
        try:
            shutil.rmtree(f)
        except OSError as e:
            if e.errno == errno.ENOTDIR:
                try:
                    os.remove(f)
                except Exception as remove_error:  # final resort safeguard
                    import logging

                    logging.getLogger(__name__).warning('f_remove failed to delete "%s": %s', f, remove_error)
    if verbose:
        print(f'Deleted "{fpath}"')


def f_copy(fsrc, fdst, ignore=None, include=None, exists_ok=True, verbose=False):
    """
    Supports both dir and file. Supports glob wildcard.
    """
    import errno

    fsrc, fdst = f_expand(fsrc), f_expand(fdst)
    for f in glob.glob(fsrc):
        try:
            f_copytree(f, fdst, ignore=ignore, include=include, exist_ok=exists_ok)
        except OSError as e:
            if e.errno == errno.ENOTDIR:
                shutil.copy(f, fdst)
            else:
                raise
    if verbose:
        print(f'Copied "{fsrc}" to "{fdst}"')


def materialize_file(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    writable: bool = True,
) -> str:
    """Materialize one local file using the cheapest correct filesystem path.

    A copy-on-write reflink is always preferred. For immutable consumers such
    as benchmark inputs, ``writable=False`` additionally permits a hard link;
    callers that may modify the destination retain independent-copy semantics.

    Returns one of ``existing``, ``reflink``, ``hardlink``, or ``copy``.
    """
    source_path = os.path.realpath(os.path.abspath(os.path.expanduser(os.fspath(source))))
    destination_path = os.path.abspath(os.path.expanduser(os.fspath(destination)))
    if source_path == destination_path:
        return "existing"
    if not os.path.isfile(source_path):
        raise FileNotFoundError(source_path)
    if os.path.lexists(destination_path):
        try:
            if os.path.samefile(source_path, destination_path):
                return "existing"
        except OSError:
            pass

    parent = os.path.dirname(destination_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.lexists(destination_path):
        os.unlink(destination_path)

    try:
        import fcntl

        with open(source_path, "rb") as source_file, open(destination_path, "xb") as destination_file:
            # FICLONE: copy-on-write reflink when the filesystem supports it.
            fcntl.ioctl(destination_file.fileno(), 0x40049409, source_file.fileno())
        shutil.copystat(source_path, destination_path)
        return "reflink"
    except (ImportError, OSError):
        if os.path.lexists(destination_path):
            os.unlink(destination_path)

    if not writable:
        try:
            os.link(source_path, destination_path)
            return "hardlink"
        except OSError:
            pass

    shutil.copy2(source_path, destination_path)
    return "copy"


def _f_copytree(
    src,
    dst,
    symlinks=False,
    ignore=None,
    exist_ok=True,
    copy_function=shutil.copy2,
    ignore_dangling_symlinks=False,
):
    """Copied from python standard lib shutil.copytree
    except that we allow exist_ok
    Use f_copytree as entry
    """
    names = os.listdir(src)
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()

    os.makedirs(dst, exist_ok=exist_ok)
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.islink(srcname):
                linkto = os.readlink(srcname)
                if symlinks:
                    # We can't just leave it to `copy_function` because legacy
                    # code with a custom `copy_function` may rely on copytree
                    # doing the right thing.
                    os.symlink(linkto, dstname)
                    shutil.copystat(srcname, dstname, follow_symlinks=not symlinks)
                else:
                    # ignore dangling symlink if the flag is on
                    if not os.path.exists(linkto) and ignore_dangling_symlinks:
                        continue
                    # otherwise let the copy occurs. copy2 will raise an error
                    if os.path.isdir(srcname):
                        _f_copytree(srcname, dstname, symlinks, ignore, exist_ok, copy_function)
                    else:
                        copy_function(srcname, dstname)
            elif os.path.isdir(srcname):
                _f_copytree(srcname, dstname, symlinks, ignore, exist_ok, copy_function)
            else:
                # Will raise a SpecialFileError for unsupported file types
                copy_function(srcname, dstname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except shutil.Error as err:
            errors.extend(err.args[0])
        except OSError as why:
            errors.append((srcname, dstname, str(why)))
    try:
        shutil.copystat(src, dst)
    except OSError as why:
        # Copying file access times may fail on Windows
        if getattr(why, "winerror", None) is None:
            errors.append((src, dst, str(why)))
    if errors:
        raise shutil.Error(errors)
    return dst


def _include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """

    def _ignore_patterns(path, names):
        """Keep names matching *patterns*; directories are never ignored (walk continues)."""

        import fnmatch

        keep = set(name for pattern in patterns for name in fnmatch.filter(names, pattern))
        ignore = set(name for name in names if name not in keep and not os.path.isdir(os.path.join(path, name)))
        return ignore

    return _ignore_patterns


def f_copytree(fsrc, fdst, symlinks=False, ignore=None, include=None, exist_ok=True):
    """Copy a directory tree; *ignore* and *include* are mutually exclusive.

    *include* is implemented as an ignore-callback that still descends into
    directories so nested matches are not skipped.
    """

    fsrc, fdst = f_expand(fsrc), f_expand(fdst)
    assert (ignore is None) or (include is None), "ignore= and include= are mutually exclusive"
    if ignore:
        ignore = shutil.ignore_patterns(*ignore)
    elif include:
        ignore = _include_patterns(*include)
    _f_copytree(fsrc, fdst, ignore=ignore, symlinks=symlinks, exist_ok=exist_ok)


def f_move(fsrc, fdst):
    """Move each glob match of *fsrc* onto *fdst* (``shutil.move`` semantics)."""

    fsrc, fdst = f_expand(fsrc), f_expand(fdst)
    for f in glob.glob(fsrc):
        shutil.move(f, fdst)


def f_split_path(fpath, normpath=True):
    """
    Splits path into a list of its component folders

    Args:
        normpath: call os.path.normpath to remove redundant '/' and
            up-level references like ".."
    """
    if normpath:
        fpath = os.path.normpath(fpath)
    allparts = []
    while 1:
        parts = os.path.split(fpath)
        if parts[0] == fpath:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == fpath:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            fpath = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def get_script_dir():
    """
    Returns: the dir of current script
    """
    return os.path.dirname(os.path.realpath(sys.argv[0]))


def get_script_file_name():
    """
    Returns: the dir of current script
    """
    return os.path.basename(sys.argv[0])


def get_script_self_path():
    """
    Returns: the dir of current script
    """
    return os.path.realpath(sys.argv[0])


def get_parent_dir(location, abspath=False):
    """
    Args:
      location: current directory or file

    Returns:
        parent directory absolute or relative path
    """
    _path = os.path.abspath if abspath else os.path.relpath
    return _path(f_join(location, os.pardir))


def md5_checksum(*fpath):
    """
    File md5 signature
    """
    import hashlib

    hash_md5 = hashlib.md5()
    with open(f_join(*fpath), "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# ──────────────────────────────────────────────────────────────────────────
# Archives and serialization — local only; remote URIs belong in storage.py
# ──────────────────────────────────────────────────────────────────────────


def create_tar(fsrc, output_tarball, include=None, ignore=None, compress_mode="gz"):
    """
    Args:
        fsrc: source file or folder
        output_tarball: output tar file name
        compress_mode: ``gz``, ``bz2``, ``xz`` or ``''`` (empty for uncompressed write)
        include: include pattern, will trigger copy to temp directory
        ignore: ignore pattern, will trigger copy to temp directory
    """
    import tarfile
    import tempfile

    fsrc, output_tarball = f_expand(fsrc), f_expand(output_tarball)
    assert compress_mode in ["gz", "bz2", "xz", ""]
    src_base = os.path.basename(fsrc)

    tempdir = None
    if include or ignore:
        tempdir = tempfile.mkdtemp()
        tempdest = f_join(tempdir, src_base)
        f_copy(fsrc, tempdest, include=include, ignore=ignore)
        fsrc = tempdest

    with tarfile.open(output_tarball, "w:" + compress_mode) as tar:
        tar.add(fsrc, arcname=src_base)

    if tempdir:
        f_remove(tempdir)


def extract_tar(source_tarball, output_dir=".", members=None):
    """
    Args:
        source_tarball: extract members from archive
        output_dir: default to current working dir
        members: must be a subset of the list returned by getmembers()

    Extraction uses the PEP 706 ``"data"`` filter, which rejects archive
    members that would escape ``output_dir`` (``../`` or absolute paths,
    CVE-2007-4559) and strips dangerous metadata. On interpreters without
    filter support the member names are validated manually before extraction.
    """
    import tarfile

    source_tarball, output_dir = f_expand(source_tarball), f_expand(output_dir)
    with tarfile.open(source_tarball, "r:*") as tar:
        try:
            tar.extractall(output_dir, members=members, filter="data")
        except TypeError:  # Python < 3.10.12 / 3.11.4 without PEP 706
            from worldfoundry.core.io.filesystem.integrity import safe_relative_path

            for member in members if members is not None else tar.getmembers():
                safe_relative_path(member.name, field_name="tar member")
                if member.islnk() or member.issym():
                    raise ValueError(
                        f"tar member {member.name!r} is a link; refusing to extract "
                        "without PEP 706 tarfile filters"
                    )
            tar.extractall(output_dir, members=members)


def move_with_backup(*fpath, suffix=".bak"):
    """
    Ensures that a path is not occupied. If there is a file, rename it by
    adding @suffix. Resursively backs up everything.

    Args:
        fpath: file path to clear
        suffix: Add to backed up files (default: {'.bak'})
    """
    fpath = str(f_join(*fpath))
    if os.path.exists(fpath):
        move_with_backup(fpath + suffix)
        shutil.move(fpath, fpath + suffix)


def insert_before_ext(name, insert):
    """
    log.txt -> log.ep50.txt
    """
    name, ext = os.path.splitext(name)
    return name + insert + ext


def timestamp_file_name(fname):
    """Insert a UTC ``_YYYYMMDD-HHMMSS`` stamp before the suffix to avoid clobber."""

    from datetime import datetime, timezone

    timestr = datetime.now(timezone.utc).strftime("_%Y%m%d-%H%M%S")
    return insert_before_ext(fname, timestr)


def next_available_file_name(
    *fpath,
    suffix_template: Union[str, Callable[[int], str]] = "_v{i+1}",
    before_ext: bool = True,
):
    """
    Args:
        suffix_template: a format string using "i" variable or
            lambda int -> str
        before_ext: True to insert suffix before the extension
    """

    def fstring(fmt_str, **kwargs):
        """Format ``{i}`` / ``{i+1}`` suffix templates without ``eval``."""
        import re

        def _eval_node(node: ast.AST, env: dict) -> object:
            """Evaluate ``{i}`` / ``{i+1}`` AST fragments; reject names outside *env*."""

            if isinstance(node, ast.Name):
                if node.id not in env:
                    raise NameError(node.id)
                return env[node.id]
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = _eval_node(node.operand, env)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
                left = _eval_node(node.left, env)
                right = _eval_node(node.right, env)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                return left * right
            raise ValueError(f"unsupported format expression: {ast.dump(node)}")

        def substitute(match):
            """Replace one ``{...}`` group with its restricted arithmetic result."""

            tree = ast.parse(match.group(1), mode="eval")
            return str(_eval_node(tree.body, kwargs))

        return re.sub(r"\{([^{}]+)\}", substitute, fmt_str)

    orig_file_path = f_join(*fpath)
    i = 0
    fpath = orig_file_path
    while os.path.exists(fpath):
        if isinstance(suffix_template, str):
            suffix = fstring(suffix_template, i=i)
        elif callable(suffix_template):
            suffix = suffix_template(i)
            assert isinstance(suffix, str)
        else:
            raise NotImplementedError(f"Unsupported suffix template {suffix_template}")
        if before_ext:
            fpath = insert_before_ext(orig_file_path, suffix)
        else:
            fpath = orig_file_path + suffix
        i += 1
    return fpath


def get_file_lock(*fpath, timeout: int = 15, logging_level="critical"):
    """
    NFS-safe filesystem-backed lock. `pip install flufl.lock`
    https://flufllock.readthedocs.io/en/stable/apiref.html

    Args:
        fpath: should be a path on NFS so that every process can see it
        timeout: seconds
    """
    import logging

    from flufl.lock import Lock

    logging.getLogger("flufl.lock").setLevel(logging_level.upper())
    return Lock(f_join(*fpath), lifetime=timeout)


def load_pickle(*fpaths, allow_pickle=False):
    """Load a trusted pickle after an explicit unsafe-format opt-in."""
    if not allow_pickle:
        raise ValueError(
            "Refusing to unpickle without allow_pickle=True; only opt in for a trusted source"
        )
    with open(f_join(*fpaths), "rb") as fp:
        return pickle.load(fp)


def dump_pickle(data, *fpaths):
    """Write a pickle; callers that later load must pass ``allow_pickle=True``."""

    with open(f_join(*fpaths), "wb") as fp:
        pickle.dump(data, fp)


def load_text(*fpaths, by_lines=False):
    """Read a local text file; ``by_lines=True`` keeps trailing newlines."""

    with open(f_join(*fpaths), "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()


def load_text_lines(*fpaths):
    """Read the file as ``readlines()`` (each item still has its newline)."""

    return load_text(*fpaths, by_lines=True)


def dump_text(s, *fpaths):
    """Overwrite a local text file; parents are *not* created."""

    with open(f_join(*fpaths), "w") as fp:
        fp.write(s)


def dump_text_lines(lines: list[str], *fpaths, add_newline=True):
    """Write one line per item; default adds ``\\n`` even if the item already has one."""

    with open(f_join(*fpaths), "w") as fp:
        for line in lines:
            print(line, file=fp, end="\n" if add_newline else "")


def get_package_root() -> str:
    """Resolve the caller's top-level package directory from ``__package__``.

    Raises :class:`ImportError` when the caller has no package context
    (``__main__`` scripts, REPL) or the spec has no origin.
    """

    import importlib.util
    import inspect

    # Get the current frame
    current_frame = inspect.currentframe()
    if current_frame is None:
        raise ImportError("Cannot determine the package name from __package__")

    # Get the caller module
    caller_module = inspect.getmodule(current_frame.f_back)
    if caller_module is None:
        raise ImportError("Cannot determine the package name from __package__")

    # Get the package name
    package_name = caller_module.__package__
    if not package_name:
        raise ImportError("Cannot determine the package name from __package__")

    # Get the top-level package name
    top_package_name = package_name.split(".")[0]

    # Find the package specification
    spec = importlib.util.find_spec(top_package_name)

    if spec and spec.origin:
        # Get the directory containing the package's __init__.py file
        package_dir = os.path.dirname(spec.origin)
        return package_dir
    else:
        raise ImportError(f"Cannot find the package {top_package_name}")


# aliases to be consistent with other load_* and dump_*
pickle_load = load_pickle
pickle_dump = dump_pickle
text_load = load_text
read_text = load_text
read_text_lines = load_text_lines
write_text = dump_text
write_text_lines = dump_text_lines
text_dump = dump_text
