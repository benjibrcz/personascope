"""Example 1: List every probe factory by category.

Discovery script — no API calls. Walks `personascope.probes.*` and prints every
`make_*` factory found, grouped by category. Useful for orienting yourself
in the panel.

Run:
    python examples/01_list_probes.py

Equivalent to `personascope list-probes` from the CLI.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import personascope.probes as probes_pkg


def _walk_probe_modules():
    """Yield (qualified_name, module) for every importable submodule of personascope.probes."""
    probes_root = Path(probes_pkg.__file__).parent
    for module_info in pkgutil.walk_packages([str(probes_root)], prefix="personascope.probes."):
        if module_info.ispkg:
            continue
        try:
            mod = importlib.import_module(module_info.name)
            yield module_info.name, mod
        except ImportError as e:
            print(f"  (skip {module_info.name}: {e})")


def main() -> None:
    by_category: dict[str, list[tuple[str, list[str]]]] = {}

    for qualname, mod in _walk_probe_modules():
        factories = [n for n in getattr(mod, "__all__", []) if n.startswith("make_")]
        if not factories:
            continue
        # Category = the first segment after "personascope.probes."
        # e.g. "personascope.probes.identity.external.identification_icl" -> "identity"
        category = qualname.split(".", 3)[2]
        short = qualname.rsplit(".", 1)[-1]
        is_external = ".external." in qualname
        bucket = f"{category} (external)" if is_external else category
        by_category.setdefault(bucket, []).append((short, factories))

    print("Personascope probe factories — discovered via pkgutil.walk_packages\n")
    for bucket, modules in sorted(by_category.items()):
        print(f"=== {bucket} ===")
        for short, factories in sorted(modules):
            for f in factories:
                print(f"  {short}.{f}")
        print()


if __name__ == "__main__":
    main()
