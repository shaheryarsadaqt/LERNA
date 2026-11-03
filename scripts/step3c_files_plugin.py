"""pytest plugin used by step3c_make_bundle.py. It changes no test behaviour.

When the environment variable STEP3C_FILES_OUT names a file, this plugin records which files the
test session used: every module in sys.modules at the end of the session, and every file opened
for reading (through a Python audit hook). It writes them, with the session's exit status, to
that file as JSON. When STEP3C_FILES_OUT is unset it does nothing.

Usage: python -m pytest -p scripts.step3c_files_plugin <tests>
"""
import json
import os
import sys

_OUT = os.environ.get("STEP3C_FILES_OUT")
_OPENED = set()
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT


def _audit(event, args):
    if event != "open" or not args:
        return
    path = args[0]
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else 0
    if isinstance(path, int):
        return
    try:
        p = os.fsdecode(os.fspath(path))
    except TypeError:
        return
    if isinstance(mode, str):
        if any(c in mode for c in "wax+"):
            return
    elif isinstance(flags, int) and flags & _WRITE_FLAGS:
        return
    _OPENED.add(os.path.realpath(os.path.abspath(p)))


if _OUT:
    sys.addaudithook(_audit)


def pytest_sessionfinish(session, exitstatus):
    if not _OUT:
        return
    modules = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if isinstance(f, str):
            modules[name] = os.path.realpath(f)
    data = {"exitstatus": int(exitstatus), "modules": modules,
            "opened_for_reading": sorted(p for p in _OPENED if os.path.isfile(p))}
    with open(_OUT, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
