from __future__ import annotations

import sys

sys.path.insert(0, 'src')

import pytest


def main():
    return pytest.main(['-q'])


if __name__ == '__main__':
    raise SystemExit(main())
