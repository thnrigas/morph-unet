"""Print the registered method names for a tier, space-separated (used by the run_tier*.sh
drivers). Usage: `python3 list_methods.py <tier>` (0 or missing -> all methods)."""

import sys

from methods import REGISTRY


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    tier = int(arg) if arg.isdigit() else 0
    if tier:
        names = [n for n, cls in REGISTRY.items() if getattr(cls, "tier", 1) == tier]
    else:
        names = list(REGISTRY)
    print(" ".join(names))


if __name__ == "__main__":
    main()
