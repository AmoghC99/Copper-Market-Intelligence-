"""Small fixer to clean up imports/line-lengths in scripts for ruff.

Run this if you want to automatically apply safe reformatting for scripts.
"""
from pathlib import Path


def main():
    for p in Path("scripts").glob("*.py"):
        txt = p.read_text()
        # simple fix: ensure a blank line after imports block start
        txt = txt.replace(
            "from pathlib import Path\nimport pandas as pd",
            "from pathlib import Path\n\nimport pandas as pd",
        )
        p.write_text(txt)


if __name__ == '__main__':
    main()
