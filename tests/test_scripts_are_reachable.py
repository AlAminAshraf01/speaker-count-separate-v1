"""Structural tests that would have caught the predecessor's silent, months-long failure.

`scripts/03_count_leak_probe.py` in the sibling repo was dead code and nobody noticed. A
`def probe_flag(...)` was inserted into the middle of `main()`, which truncated `main` at the
line above it and swallowed the rest of its body -- every probe, the naive baseline, the
speaker-disjointness audit and the report write -- into `probe_flag`, *after* an unconditional
`return`. `main()` then fell off its end returning `None`, and `sys.exit(None)` is exit code 0.
The script printed its banner, printed two status lines, wrote nothing, and reported success.
Two headline numbers were quoted from it for months that it had never produced.

No unit test would have caught that, because the functions it tested all still worked. What
catches it is a *structural* invariant over the source, and there is a precise one:

    a script's `main()` must end with an explicit `return`.

The broken `main` ended with `chance = 100.0 / len(set(raw_y.tolist()))` -- an assignment. A
function whose last statement is not a return, in a file whose entry point is
`sys.exit(main())`, is either falling off its end by accident or telling the reader it returns
None while its caller passes the value to `sys.exit`. Either way it is worth a failing test.

These checks parse the source with `ast`. They import nothing, need no store, and run in
milliseconds, so there is no excuse for skipping them.
"""

from __future__ import annotations

import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import run_checks  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")


def _script_paths() -> list[str]:
    if not os.path.isdir(SCRIPTS_DIR):
        return []
    return [os.path.join(SCRIPTS_DIR, f) for f in sorted(os.listdir(SCRIPTS_DIR))
            if f.endswith(".py") and not f.startswith("_")]


def _parse(path: str) -> ast.Module:
    with open(path, "r", encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


def _top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_every_script_has_a_main() -> None:
    """Each script exposes a top-level ``main``, so the entry point is unambiguous."""
    for path in _script_paths():
        funcs = _top_level_functions(_parse(path))
        assert "main" in funcs, f"{os.path.basename(path)} has no top-level main()"


def test_main_ends_with_an_explicit_return() -> None:
    """THE regression test for the predecessor's bug.

    A ``main`` that falls off its end returns None. When the entry point is
    ``sys.exit(main())`` that becomes exit code 0 -- success -- no matter how much of the
    body was never reached.
    """
    for path in _script_paths():
        name = os.path.basename(path)
        main = _top_level_functions(_parse(path))["main"]
        last = main.body[-1]
        assert isinstance(last, (ast.Return, ast.Raise)), (
            f"{name}: main() ends with {type(last).__name__} at line {last.lineno}, not a "
            f"return. A main that falls off its end returns None, and sys.exit(None) is 0 -- "
            f"which is exactly how the predecessor reported success while doing nothing.")


def test_no_function_is_defined_inside_main() -> None:
    """A nested ``def`` inside ``main`` is how the predecessor's body got swallowed.

    Small closures are legitimate in general, but in a *script's* ``main`` they are the exact
    shape of the accident, and the alternative -- a module-level helper -- costs nothing.
    """
    for path in _script_paths():
        name = os.path.basename(path)
        main = _top_level_functions(_parse(path))["main"]
        nested = [n.name for n in ast.walk(main)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not main]
        assert not nested, (
            f"{name}: main() defines {nested} inline. Move them to module level: a def "
            f"landing inside main is how scripts/03 in the sibling repo lost 110 lines of "
            f"its body to an unreachable branch.")


def _unreachable_after_terminator(node: ast.AST) -> list[tuple[str, int, int]]:
    """Find statement lists where a terminator is followed by more statements.

    If a block contains ``return``/``raise``/``break``/``continue`` anywhere but its last
    position, every statement after it in that block is dead. Reported as
    ``(block_owner, terminator_line, first_dead_line)``.
    """
    found: list[tuple[str, int, int]] = []
    for owner in ast.walk(node):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(owner, field, None)
            if not isinstance(block, list) or len(block) < 2:
                continue
            for i, stmt in enumerate(block[:-1]):
                if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                    label = getattr(owner, "name", type(owner).__name__)
                    found.append((str(label), stmt.lineno, block[i + 1].lineno))
                    break
    return found


def test_no_unreachable_code_after_a_return() -> None:
    """THE precise fingerprint of the predecessor's failure.

    In `scripts/03_count_leak_probe.py`, `probe_flag`'s body was::

        [docstring, if, assign, if, if, Return(line 170),
         FunctionDef(run), ..., Return(line 263)]

    A `return` at index 5 of a 40-element block means the 34 statements after it -- the entire
    remainder of what had been `main` -- can never execute. Nothing else in that repo's test
    suite could see it, because every function it tested still worked in isolation.

    Note this catches what the more obvious "does a top-level def start inside main()" check
    cannot: Python's parser had *already* truncated `main` at line 152, so `probe_flag` at 155
    does not overlap it in the AST at all. The damage is only visible from the inside.
    """
    for path in _script_paths():
        name = os.path.basename(path)
        dead = _unreachable_after_terminator(_parse(path))
        assert not dead, "\n".join(
            [f"{name}: unreachable code found --"] +
            [f"    {owner}(): line {first_dead} can never run; "
             f"line {term} always returns first." for owner, term, first_dead in dead])


def test_audit_script_verifies_its_own_output() -> None:
    """The audit must check that its report reached disk before claiming success.

    Writing the file is not the same as the file existing: an exception between the write and
    the exit, a read-only output directory, or a truncated atomic replace all produce a script
    that ran and a report that is not there.
    """
    path = os.path.join(SCRIPTS_DIR, "02_audit_and_baselines.py")
    assert os.path.exists(path), "the audit script is missing"
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()
    for needle in ("os.path.exists(path)", "os.path.getsize(path)", "return 2"):
        assert needle in source, (
            f"02_audit_and_baselines.py no longer checks {needle!r} before exiting. That check "
            f"is the whole reason this project trusts the audit's exit code.")


def test_scripts_exit_with_main() -> None:
    """Every script's entry point passes main()'s value to sys.exit, so returns matter."""
    for path in _script_paths():
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
        assert "sys.exit(main())" in source, (
            f"{os.path.basename(path)}: expected 'sys.exit(main())' so a non-zero return "
            f"actually fails the notebook cell.")


def test_every_countsep_import_resolves() -> None:
    """Catch a module rename that left an importer behind.

    ``preflight.py`` imported ``countsep.model`` for its precision gate. v1 renamed that
    file to ``counter.py``, so the import raised, the handler turned it into a WARN, and
    the check THE PROJECT WAS REBUILT AROUND silently never ran -- printing
    ``WARN precision cannot check (No module named 'countsep.model')`` on every training
    run while eight other rows said OK.

    That is v0's leak audit again in a different costume: a check that reports success at
    doing nothing. A rename is found by the thing that imports, not by the thing renamed,
    so this walks every import statement instead of trusting anybody to remember.
    """
    import ast

    package = os.path.join(REPO_ROOT, "src", "countsep")
    modules = {f[:-3] for f in os.listdir(package) if f.endswith(".py")}
    stale = []
    for folder in ("scripts", "tests", "tools"):
        directory = os.path.join(REPO_ROOT, folder)
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(directory, name), "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("countsep."):
                    targets.append(node.module)
                elif isinstance(node, ast.Import):
                    targets += [a.name for a in node.names if a.name.startswith("countsep.")]
                for target in targets:
                    if target.split(".")[1] not in modules:
                        stale.append(f"{folder}/{name}:{node.lineno} -> {target}")

    assert not stale, "these import countsep modules that do not exist: " + "; ".join(stale)


def test_stateful_modules_are_moved_to_the_device() -> None:
    """A module with buffers that is built but never ``.to(device)``'d fails only on a GPU.

    ``05_train_separator.py`` moved the model and not the loss. ``RectangularPITLoss`` holds
    ten non-persistent buffers -- the permutation tables the rectangular PIT assignment
    gathers with -- so on CUDA the gather got a cuda ``sub_loss`` and a cpu ``index`` and
    died on the first batch, after preflight had passed 9/9 and the dataset had loaded.

    No CPU test can reproduce that: with one device everything trivially agrees, and the
    ``meta`` device does not raise on a mismatched gather (checked). The counter's trainer
    hid the same class of bug by luck -- ``nn.CrossEntropyLoss`` has no buffers.

    So this is structural. It finds every statement that constructs a ``countsep`` module and
    requires the same statement to place it, by ``.to(...)`` or a ``device=`` argument. The
    set of module factories is discovered by import rather than hand-listed, because a
    hand-list rots at the first rename -- which this project has already been bitten by.
    """
    import ast
    import importlib
    import inspect
    import pkgutil

    import torch

    import countsep

    modules, factories = set(), set()
    for info in pkgutil.iter_modules(countsep.__path__):
        namespace = vars(importlib.import_module(f"countsep.{info.name}"))
        for attr, obj in namespace.items():
            if attr.startswith("_"):
                continue
            if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
                modules.add(attr)
    for info in pkgutil.iter_modules(countsep.__path__):
        namespace = vars(importlib.import_module(f"countsep.{info.name}"))
        for attr, obj in namespace.items():
            if inspect.isfunction(obj) and attr.startswith("build_"):
                # Only a factory ANNOTATED as returning an nn.Module counts. build_loader
                # returns a DataLoader and must not be flagged.
                returns = inspect.signature(obj).return_annotation
                if isinstance(returns, str) and returns in modules:
                    factories.add(attr)
    builders = modules | factories
    assert "RectangularPITLoss" in builders and "build_separator" in builders,         "the factory scan found nothing; the check would pass vacuously"
    assert "build_loader" not in builders, "build_loader returns a DataLoader, not a module"

    offenders = []
    scripts = os.path.join(REPO_ROOT, "scripts")
    for name in sorted(os.listdir(scripts)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(scripts, name), "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            built = None
            placed = False
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Call):
                    if isinstance(sub.func, ast.Name) and sub.func.id in builders:
                        built = sub.func.id
                    if any(kw.arg == "device" for kw in sub.keywords):
                        placed = True
                if isinstance(sub, ast.Attribute) and sub.attr == "to":
                    placed = True
            if built and not placed:
                offenders.append(f"scripts/{name}:{node.lineno} builds {built} without .to(device)")

    assert not offenders, ("built on the CPU and used on the GPU: "
                           + "; ".join(offenders))


if __name__ == "__main__":
    sys.exit(run_checks({
        "every script has a main": test_every_script_has_a_main,
        "main ends with an explicit return": test_main_ends_with_an_explicit_return,
        "no function defined inside main": test_no_function_is_defined_inside_main,
        "no unreachable code after a return": test_no_unreachable_code_after_a_return,
        "audit script verifies its own output": test_audit_script_verifies_its_own_output,
        "every countsep import resolves": test_every_countsep_import_resolves,
        "stateful modules are moved to the device": test_stateful_modules_are_moved_to_the_device,
        "scripts exit with main()": test_scripts_exit_with_main,
    }))
